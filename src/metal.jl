"""
    Metal GPU backend for the interactive viewfinder.

This file implements a pure-Metal compute kernel that runs the same fixed-step
RK4 preview integrator as `render_preview`, but on the Apple GPU.  The kernel is
written in Julia and compiled to Metal Shading Language by Metal.jl.

The design intentionally mirrors the CPU preview renderer so that the two paths
are easy to keep in sync.
"""

"""
    MetalPreviewContext

GPU-resident state for the preview renderer.  Holds the background texture/array
on the GPU, a reusable output buffer, and reusable parameter buffers for the
camera and spacetime constants.
"""
struct MetalPreviewContext{B<:MtlArray{Float32,3}, O<:MtlArray{Float32,3},
                           C<:MtlVector{Float32}, S<:MtlVector{Float32},
                           D<:MtlVector{Float32}, L<:MtlArray{Float32,2},
                           V<:MtlArray{Float32,3}}
    bg_gpu::B
    out_gpu::O
    cam_params::C
    spacetime_params::S
    disc_params::D
    bb_lut::L
    vol_gpu::V                   # (nr, nphi, nz) volumetric disc density
    vol_params::MtlVector{Float32}
    # Procedural starfield (see `starfield_mtl`); [1] = 0 disables it, and the
    # cost when off is one buffer read and a branch, so it needs no Val.
    star_params::MtlVector{Float32}
    # Star colour has its OWN blackbody LUT, deliberately not the disc's.
    # Sharing one made star colour a function of the disc's white balance:
    # `wb_temperature` decides which temperature renders neutral, so grading
    # the disc re-tinted the whole sky (a 10000 K white point put every star
    # below it, and the field came out uniformly gold). Stars should look like
    # stars whatever the disc is doing.
    star_lut::L
    width::Int
    height::Int
    dt::Float32
    nmax::Int
    r_escape_factor::Float32
    has_volume::Bool
    vol_on::Base.RefValue{Bool}  # runtime volumetric toggle
    # Host mirror of "does the sky texture contribute at all": stars off, or
    # stars on with a non-zero `texture_weight`. The composite gathers the 4k
    # equirectangular map per native pixel, so when the weight is zero — the
    # shipped stars-only default — that gather is a cache-hostile read of a
    # value about to be multiplied by zero. Kept on the host because the
    # decision is frame-uniform and so belongs in a `Val`, not a branch.
    sky_tex::Base.RefValue{Bool}
    # Compiled kernel per volume mode (Val-specialised, so the no-volume
    # variant keeps the lean kernel's register budget), built lazily.
    kernel::Base.RefValue{Any}
    # Host views that ALIAS the shared-storage parameter buffers above. Apple
    # silicon has unified memory, so the per-frame `copyto!` was staging bytes
    # the GPU could already see -- ~7.5 kB of allocation per frame to move 228.
    # Filling these in place instead costs nothing and needs no copy.
    cam_host::Vector{Float32}
    st_host::Vector{Float32}
end

"""
White-balance temperature for star colour, held apart from the disc's.

10000 K, which is roughly the A0 dwarf convention (Vega, the historical zero of
the colour-index system, sits near 9600 K) and is the point the shipped star
look was tuned at. Stars hotter than this render blue-white, cooler ones
orange. Changing the *disc's* `wb_temperature` must not move this — see
[`MetalPreviewContext`](@ref)'s `star_lut`.
"""
const STAR_WB_TEMPERATURE = 10000.0

"""Star colour table: a white-balanced blackbody LUT, independent of any disc.

`saturation` scales each entry's chroma about its Rec.709 luminance. Raw
blackbody chroma is far more saturated than any star ever looks — measured
star chromaticities (Charity's dataset) top out at pale blue-white even for
O stars, and point sources barely drive colour vision at all, which is why
production sky renderers mix star colours most of the way toward white.
1.0 keeps the raw LUT; ~0.35 matches the measured/production look.
"""
function _star_lut_cpu(wb_temperature::Real; table_size::Int=1024,
                       saturation::Real=1.0)
    bb = Blackbody(; wb_temperature=wb_temperature, table_size=table_size)
    lut = Array{Float32,2}(undef, 3, bb.table_size)
    for k in 1:bb.table_size, c in 1:3
        lut[c, k] = Float32(bb.table[k][c])
    end
    if saturation != 1.0
        s = Float32(saturation)
        for k in 1:bb.table_size
            L = 0.2126f0 * lut[1, k] + 0.7152f0 * lut[2, k] +
                0.0722f0 * lut[3, k]
            for c in 1:3
                lut[c, k] = L + s * (lut[c, k] - L)
            end
        end
    end
    return lut
end

"""
    _upload_background(bg)

Upload a CPU background image to the GPU as a `(3, W, H)` `Float32` array.
"""
function _upload_background(bg)
    W, H = size(bg)
    bg_cpu = Array{Float32,3}(undef, 3, W, H)
    for j in 1:H, i in 1:W
        c = bg[i, j]
        bg_cpu[1, i, j] = Float32(red(c))
        bg_cpu[2, i, j] = Float32(green(c))
        bg_cpu[3, i, j] = Float32(blue(c))
    end
    return MtlArray(bg_cpu)
end

"""
    MetalPreviewContext(background, width, height)

Create a context for rendering previews of size `width × height` using the GPU.
The background image is uploaded once and reused across frames.
"""
function MetalPreviewContext(background, width::Int, height::Int;
                              dt::Real=0.1f0, nmax::Int=1000,
                              r_escape_factor::Real=2.0f0,
                              disc::Union{AccretionDisc,Nothing}=nothing,
                              volume::Union{DiscVolume,Nothing}=nothing)
    bg_gpu = _upload_background(background)
    out_gpu = MtlArray{Float32,3}(undef, 3, width, height)
    cam_params = MtlArray{Float32,1,Metal.SharedStorage}(undef, CAM_PARAMS_N)
    spacetime_params = MtlArray{Float32,1,Metal.SharedStorage}(undef, 8)
    cam_host = unsafe_wrap(Array, cam_params); fill!(cam_host, 0.0f0)
    st_host = unsafe_wrap(Array, spacetime_params); fill!(st_host, 0.0f0)
    disc_params = MtlVector{Float32}(undef, 6)
    if isnothing(disc)
        copyto!(disc_params, zeros(Float32, 6))
        bb_lut = MtlArray(zeros(Float32, 3, 1))
    else
        bb = disc.blackbody
        copyto!(disc_params, Float32[disc.inner_radius, disc.outer_radius,
                                     disc.density_falloff, bb.table_min,
                                     bb.table_max, bb.table_size])
        lut_cpu = Array{Float32,2}(undef, 3, bb.table_size)
        for k in 1:bb.table_size, c in 1:3
            lut_cpu[c, k] = Float32(bb.table[k][c])
        end
        bb_lut = MtlArray(lut_cpu)
    end
    vol_params = MtlVector{Float32}(undef, 9)
    if isnothing(volume)
        vol_gpu = MtlArray(zeros(Float32, 1, 1, 1))
        copyto!(vol_params, zeros(Float32, 9))
    else
        vol_gpu = MtlArray(volume.density)
        nr, nphi, nz = size(volume.density)
        copyto!(vol_params, Float32[volume.log_s_in, volume.log_s_out,
                                    volume.z_max, nr, nphi, nz,
                                    volume.emission_scale,
                                    volume.opacity_scale, 2.0f0])  # march stride
    end
    star_params = MtlVector{Float32}(undef, 16)
    copyto!(star_params, zeros(Float32, 16))     # off until set_starfield!
    # The star LUT is independent of `disc`, so a context with no disc can
    # still render stars. `set_starfield!` rebakes it if the white point moves.
    star_lut = MtlArray(_star_lut_cpu(STAR_WB_TEMPERATURE))
    return MetalPreviewContext(bg_gpu, out_gpu, cam_params, spacetime_params,
                               disc_params, bb_lut, vol_gpu, vol_params,
                               star_params, star_lut,
                               width, height, Float32(dt),
                               nmax, Float32(r_escape_factor),
                               !isnothing(volume),
                               Base.RefValue{Bool}(!isnothing(volume)),
                               Base.RefValue{Bool}(true),   # stars off until set_starfield!
                               Base.RefValue{Any}(Dict{Any,Any}()),
                               cam_host, st_host)
end

"""
    set_volume_enabled!(ctx::MetalPreviewContext, enabled::Bool)

Toggle the volumetric disc in an existing context at runtime (a 9-float
buffer update; the next frame picks it up). With the volume off the kernel
falls back to the thin-plane disc. A no-op for contexts built without a
volume.
"""
function set_volume_enabled!(ctx::MetalPreviewContext, enabled::Bool)
    ctx.has_volume || return nothing
    ctx.vol_on[] = enabled
    return nothing
end

"""
    set_disc_enabled!(ctx, disc::AccretionDisc, enabled)

Enable/disable the thin-plane accretion disc at runtime by rewriting the
6-float disc parameter buffer (`inner = 0` disables it in the kernel).
Contexts that share `disc_params` (resolution variants) all
follow. The volumetric disc has its own switch: [`set_volume_enabled!`](@ref).
"""
function set_disc_enabled!(ctx::MetalPreviewContext, disc::AccretionDisc,
                           enabled::Bool)
    bb = disc.blackbody
    copyto!(ctx.disc_params,
            Float32[enabled ? disc.inner_radius : 0.0, disc.outer_radius,
                    disc.density_falloff, bb.table_min, bb.table_max,
                    bb.table_size])
    return nothing
end

"""
    set_starfield!(ctx; strength=1.0, texture_weight=0.0, height=nothing,
                   fov_factor=0.55, density=768, fill=0.18, flux=0.004,
                   psf_pixels=1.0, galactic=(0,0,1), concentration=3.0,
                   temp_min=3000, temp_max=9000, seed=12345)

Enable the procedural starfield (see [`starfield_mtl`](@ref)). `strength = 0`
turns it off, which is the default.

`texture_weight` scales the sky texture the stars are drawn over: 1.0 keeps it
at full strength and adds stars on top, 0.0 replaces it entirely. The useful
middle is a low weight — the 4k equirectangular map is *good* at the diffuse
Milky Way glow and nebulosity, which are genuinely low-frequency and lose
nothing to magnification, and *bad* at point sources, which is what this
replaces.

`psf_pixels` sets the Gaussian width as a multiple of the pixel's angular
footprint, computed from `height` and `fov_factor`. Below about 1 the field
starts to alias into flicker under camera motion; the drizzle literature puts
the sweet spot near 0.8 pixels. Pass the **render** height, not the preview
height, or stars will be sized for the wrong frame.

`wb_temperature` is the star field's **own** white point (default
[`STAR_WB_TEMPERATURE`](@ref)), baked into a LUT the disc never touches. This
is deliberate: white balance decides which temperature renders neutral, so
while stars shared the disc's LUT, regrading the disc re-tinted the whole sky —
and a 10000 K disc white point put every star below it, turning the field
uniformly gold. Stars should look like stars whatever the disc is doing.
`temp_min`/`temp_max` should straddle this value, or the same tinting returns.

`flux` is the faintest star's linear brightness; the distribution runs up from
there as ξ^(−2/3) over roughly a 460× range. The default is calibrated against
`starmap_g4k.jpg` at the hero framing, where the brightest star in a clear-sky
crop reaches ≈2.3 linear — match that and the two skies carry comparable
weight, so `texture_weight` becomes a pure look dial rather than an exposure
correction.

`saturation` scales the colour LUT's chroma about luminance (see
[`_star_lut_cpu`](@ref)): 1.0 is raw blackbody colour, ~0.35 matches measured
star chromaticities, where even O stars are only pale blue-white.
"""
function set_starfield!(ctx::MetalPreviewContext; strength::Real=1.0,
                        texture_weight::Real=0.0,
                        height::Union{Int,Nothing}=nothing,
                        fov_factor::Real=0.55, density::Real=384,
                        fill::Real=0.5, flux::Real=0.011,
                        psf_pixels::Real=0.5,
                        galactic::NTuple{3,Real}=(0.0, 0.0, 1.0),
                        concentration::Real=3.0, temp_min::Real=3000,
                        temp_max::Real=16000, seed::Integer=12345,
                        wb_temperature::Real=STAR_WB_TEMPERATURE,
                        saturation::Real=1.0)
    H = something(height, ctx.height)
    gx, gy, gz = galactic
    gn = sqrt(gx^2 + gy^2 + gz^2)
    gn > 0 || throw(ArgumentError("galactic normal must be non-zero"))
    # The pixel's angular footprint: the vertical field is 2·fov_factor across
    # `H` rows. Sizing the PSF from this rather than from a fixed angle is what
    # keeps stars ~1 px at every resolution.
    σ = psf_pixels * 2 * fov_factor / H
    # Star colour comes from the context's OWN LUT, never the disc's, so
    # regrading the disc leaves the sky alone. Rebake only if the caller moves
    # the star white point or the chroma off the default.
    nlut = size(ctx.star_lut, 2)
    if wb_temperature != STAR_WB_TEMPERATURE || saturation != 1.0
        copyto!(ctx.star_lut, _star_lut_cpu(wb_temperature; table_size=nlut,
                                            saturation=saturation))
    end
    bb_ref = Blackbody(; wb_temperature=wb_temperature, table_size=nlut)
    lut = (bb_ref.table_min, bb_ref.table_max, Float64(nlut))
    copyto!(ctx.star_params,
            Float32[strength, density, fill, σ, flux,
                    gx / gn, gy / gn, gz / gn, concentration,
                    temp_min, temp_max - temp_min,
                    lut[1], lut[2], lut[3], seed, texture_weight])
    # With stars on and `texture_weight` zero the sky texture contributes
    # nothing, and the composite can skip the gather entirely.
    ctx.sky_tex[] = strength <= 0 || texture_weight > 0
    return nothing
end

"""
    set_march_stride!(ctx, s)

Set the volumetric march stride (gas sampled every `s` integration steps
with `s`× path weight; default 2). Coarser strides trade gas detail for
speed when the camera is inside the slab; geodesics are unaffected.
"""
function set_march_stride!(ctx::MetalPreviewContext, s::Int)
    ctx.has_volume || return nothing
    Metal.@allowscalar ctx.vol_params[9] = Float32(s)
    return nothing
end

# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

"""
    ks_rhs_mtl(x, y, z, px, py, pz, p_t, M)

Float32 geodesic RHS in **Cartesian Kerr–Schild coordinates**. The metric is
`g = η + f l⊗l` with `f = 2M/r` and `l = (1, x/r, y/r, z/r)`, giving the
Hamiltonian `H = ½(−p_t² + |p|² − f ℓ²)` with `ℓ = −p_t + (x·p)/r`. Unlike
the spherical chart this is regular at the poles **and** at the horizon —
no `1/sin²θ`, no `1/(r−2M)` — and contains no trigonometry. `p_t` is
conserved. Returns `(dx, dy, dz, dpx, dpy, dpz)`.
"""
function ks_rhs_mtl(x, y, z, px, py, pz, p_t, M)
    r2 = x * x + y * y + z * z
    inv_r = 1.0f0 / sqrt(r2)
    f = 2.0f0 * M * inv_r
    κ = (x * px + y * py + z * pz) * inv_r
    ℓ = -p_t + κ
    c1 = f * ℓ * inv_r                                  # fℓ/r
    c2 = f * ℓ * (0.5f0 * ℓ + κ) * inv_r * inv_r        # f(ℓ²/2 + ℓκ)/r²

    dx = px - c1 * x
    dy = py - c1 * y
    dz = pz - c1 * z
    dpx = c1 * px - c2 * x
    dpy = c1 * py - c2 * y
    dpz = c1 * pz - c2 * z
    return dx, dy, dz, dpx, dpy, dpz
end

"""
Pack the spacetime's spin and (padded) squared horizon for the kernel's
`spacetime_params[7:8]`. The pad keeps rays from grazing the coordinate
horizon and escaping as phantom sky.
"""
function _spin_horizon(spacetime)
    a = Float32(spin(spacetime))
    # Kill radius is the prograde photon orbit, not the horizon. Testing only
    # the horizon lets near-critical rays bounce off the coordinate ridge and
    # escape as phantom sky — visible as the lensed background reappearing
    # inside the shadow. Nothing that reaches infinity ever dips below
    # `photon_orbit_min`, so this needs no inward-motion test; the 0.5% margin
    # is slack for Float32 error right at the boundary.
    rk = Float32(0.995 * photon_orbit_min(spacetime))
    return a, rk * rk
end

@inline _rhs_mtl(::Val{false}, x, y, z, px, py, pz, p_t, M, a) =
    ks_rhs_mtl(x, y, z, px, py, pz, p_t, M)
@inline _rhs_mtl(::Val{true}, x, y, z, px, py, pz, p_t, M, a) =
    kerr_rhs_mtl(x, y, z, px, py, pz, p_t, M, a)

# Six-component tuple arithmetic for the Tsit5 stages: phase-space state
# and RHS slopes travel as (x, y, z, px, py, pz).
@inline _axpy6(s, c, k) = (s[1] + c * k[1], s[2] + c * k[2], s[3] + c * k[3],
                           s[4] + c * k[4], s[5] + c * k[5], s[6] + c * k[6])
@inline _scale6(k, c) = (c * k[1], c * k[2], c * k[3],
                         c * k[4], c * k[5], c * k[6])

"""
    kerr_rhs_mtl(x, y, z, px, py, pz, p_t, M, a)

Float32 null-geodesic RHS for **Kerr** in Cartesian Kerr–Schild coordinates,
the same chart and Hamiltonian convention as [`ks_rhs_mtl`](@ref), which it
reduces to exactly at `a = 0`.

`g = η + f l⊗l` with

    r²  = ½[(ρ² − a²) + √((ρ² − a²)² + 4a²z²)]      ρ² = x²+y²+z²
    Σ   = r⁴ + a²z²
    f   = 2Mr³/Σ
    l_μ = (1, (rx + ay)/(r²+a²), (ry − ax)/(r²+a²), z/r)

so `H = ½(−p_t² + |p|² − f ℓ²)` with `ℓ = l^μ p_μ = −p_t + L⃗·p⃗`, giving

    dxⁱ/dλ = pᵢ − f ℓ Lᵢ
    dpᵢ/dλ = ½ (∂ᵢf) ℓ² + f ℓ (∂ᵢℓ)

`r` is an implicit function of position, so the position derivatives go through
it: differentiating the quartic `r⁴ − (ρ²−a²)r² − a²z² = 0` gives

    ∂r/∂x = x r³/Σ,   ∂r/∂y = y r³/Σ,   ∂r/∂z = z r (r²+a²)/Σ

Routing both `f` and `ℓ` through `∂r/∂xⁱ` plus their explicit position
dependence costs four derivative expressions rather than the nine partials a
direct `∂ᵢLⱼ` expansion would need. `p_t` is conserved; the spacetime is
stationary and axisymmetric, so `p_φ` is too, but nothing here needs it.
"""
@inline function kerr_rhs_mtl(x, y, z, px, py, pz, p_t, M, a)
    a2 = a * a
    z2 = z * z
    w = x * x + y * y + z2 - a2
    r2 = 0.5f0 * (w + sqrt(w * w + 4.0f0 * a2 * z2))
    r2 = max(r2, 1.0f-12)
    r = sqrt(r2)
    r3 = r2 * r
    invΣ = 1.0f0 / (r2 * r2 + a2 * z2)
    R2A = r2 + a2
    iRA = 1.0f0 / R2A
    inv_r = 1.0f0 / r

    Lx = (r * x + a * y) * iRA
    Ly = (r * y - a * x) * iRA
    Lz = z * inv_r
    ℓ = -p_t + Lx * px + Ly * py + Lz * pz
    f = 2.0f0 * M * r3 * invΣ
    fl = f * ℓ

    # ∂r/∂xⁱ from the implicit quartic.
    drx = x * r3 * invΣ
    dry = y * r3 * invΣ
    drz = z * r * R2A * invΣ

    # f depends on position through r and (explicitly) through z.
    dfdr = 2.0f0 * M * r2 * (3.0f0 * a2 * z2 - r2 * r2) * invΣ * invΣ
    dfz0 = -4.0f0 * M * r3 * a2 * z * invΣ * invΣ

    # ℓ likewise: ∂ℓ/∂xⁱ at fixed r, plus ∂ℓ/∂r.
    dlx0 = (r * px - a * py) * iRA
    dly0 = (a * px + r * py) * iRA
    dlz0 = pz * inv_r
    dldr = (px * (x * R2A - 2.0f0 * r * (r * x + a * y)) +
            py * (y * R2A - 2.0f0 * r * (r * y - a * x))) * iRA * iRA -
           z * pz * inv_r * inv_r

    hl2 = 0.5f0 * ℓ * ℓ
    dpx = hl2 * (dfdr * drx) + fl * (dldr * drx + dlx0)
    dpy = hl2 * (dfdr * dry) + fl * (dldr * dry + dly0)
    dpz = hl2 * (dfdr * drz + dfz0) + fl * (dldr * drz + dlz0)
    return (px - fl * Lx, py - fl * Ly, pz - fl * Lz, dpx, dpy, dpz)
end

"""
    kerr_horizon(M, a)

Outer horizon radius `r₊ = M + √(M² − a²)` in Kerr–Schild `r`.
"""
kerr_horizon(M, a) = M + sqrt(max(M * M - a * a, 0.0))

"""
    sample_background_mtl(bg, theta, phi, W, H)

Bilinear sample of the GPU background array at spherical coordinates `(theta, phi)`.
`bg` has shape `(3, W, H)` with 1-based indexing.
"""
function sample_background_mtl(bg, theta, phi, W, H)
    two_pi = 2.0f0 * Float32(pi)
    v = clamp(theta / Float32(pi), 0.0f0, 1.0f0)
    u = clamp(mod(phi, two_pi) / two_pi, 0.0f0, 1.0f0)

    xf = u * (W - 1)
    yf = v * (H - 1)

    # unsafe_trunc + clamp instead of a checked conversion: a checked
    # Int32(NaN) traps the GPU kernel, and clamping keeps any garbage index
    # in bounds.
    x0 = clamp(unsafe_trunc(Int32, floor(xf)), Int32(0), Int32(W - 1))
    x1 = min(x0 + Int32(1), Int32(W - 1))
    y0 = clamp(unsafe_trunc(Int32, floor(yf)), Int32(0), Int32(H - 1))
    y1 = min(y0 + Int32(1), Int32(H - 1))

    fx = xf - x0
    fy = yf - y0

    # Convert to 1-based indices for MtlArray access.
    x0_1 = x0 + 1
    x1_1 = x1 + 1
    y0_1 = y0 + 1
    y1_1 = y1 + 1

    w00 = (1.0f0 - fx) * (1.0f0 - fy)
    w10 = fx * (1.0f0 - fy)
    w01 = (1.0f0 - fx) * fy
    w11 = fx * fy

    r = bg[1, x0_1, y0_1] * w00 + bg[1, x1_1, y0_1] * w10 +
        bg[1, x0_1, y1_1] * w01 + bg[1, x1_1, y1_1] * w11
    g = bg[2, x0_1, y0_1] * w00 + bg[2, x1_1, y0_1] * w10 +
        bg[2, x0_1, y1_1] * w01 + bg[2, x1_1, y1_1] * w11
    b = bg[3, x0_1, y0_1] * w00 + bg[3, x1_1, y0_1] * w10 +
        bg[3, x0_1, y1_1] * w01 + bg[3, x1_1, y1_1] * w11

    return r, g, b
end

"""
    sample_volume_mtl(vol, vol_params, s, phi, z)

Trilinear disc-volume density lookup on the GPU (Float32 twin of
`sample_disc_volume`). Returns 0 outside the grid.
"""
function sample_volume_mtl(vol, vp, s, phi, z)
    zmax = vp[3]
    if s <= 0.0f0 || abs(z) >= zmax
        return 0.0f0
    end
    ls = log(s)
    ls_in = vp[1]
    ls_out = vp[2]
    if ls <= ls_in || ls >= ls_out
        return 0.0f0
    end
    nr = unsafe_trunc(Int32, vp[4])
    nphi = unsafe_trunc(Int32, vp[5])
    nz = unsafe_trunc(Int32, vp[6])
    two_pi = 2.0f0 * Float32(pi)

    fr = (ls - ls_in) / (ls_out - ls_in) * Float32(nr - Int32(1))
    fp = mod(phi, two_pi) / two_pi * Float32(nphi)
    fz = (z + zmax) / (2.0f0 * zmax) * Float32(nz - Int32(1))

    i0 = clamp(unsafe_trunc(Int32, floor(fr)), Int32(0), nr - Int32(2))
    tr = fr - Float32(i0)
    j0 = unsafe_trunc(Int32, floor(fp))
    tp = fp - Float32(j0)
    k0 = clamp(unsafe_trunc(Int32, floor(fz)), Int32(0), nz - Int32(2))
    tz = fz - Float32(k0)

    j0a = mod(j0, nphi) + Int32(1)
    j1a = mod(j0 + Int32(1), nphi) + Int32(1)
    i0a = i0 + Int32(1)
    i1a = i0 + Int32(2)
    k0a = k0 + Int32(1)
    k1a = k0 + Int32(2)

    c00 = vol[i0a, j0a, k0a] + tr * (vol[i1a, j0a, k0a] - vol[i0a, j0a, k0a])
    c10 = vol[i0a, j1a, k0a] + tr * (vol[i1a, j1a, k0a] - vol[i0a, j1a, k0a])
    c01 = vol[i0a, j0a, k1a] + tr * (vol[i1a, j0a, k1a] - vol[i0a, j0a, k1a])
    c11 = vol[i0a, j1a, k1a] + tr * (vol[i1a, j1a, k1a] - vol[i0a, j1a, k1a])
    c0 = c00 + tp * (c10 - c00)
    c1 = c01 + tp * (c11 - c01)
    return c0 + tz * (c1 - c0)
end

# ---------------------------------------------------------------------------
# Metal kernel
# ---------------------------------------------------------------------------

"""
    starfield_mtl(dx, dy, dz, sp, star_lut) -> (r, g, b)

Procedural point stars in the escape direction `(dx, dy, dz)`, evaluated at
output resolution instead of read from a texture.

**Why this exists.** `assets/starmap_g4k.jpg` is 4096×2048 equirectangular, so
0.088° per texel. A 4K frame through a 33 mm lens has pixels of about 0.026°,
which magnifies the sky texture more than three times: stars stop being points
and become soft blobs, and no amount of render resolution recovers them
because the source has run out. A procedural field has no such limit.

**Placement.** Stars live on a uniform grid in `(u, v) = (φ/2π, (cos θ + 1)/2)`.
That parametrisation is *equal-area* — `d(cos θ) dφ` is the solid-angle element
— so a uniform grid gives uniform sky density with no pole pile-up, and the
3×3 neighbourhood search needs no latitude correction. Each cell's contents
come from a hash of its index, so the sky is deterministic, seamless and free.

**Brightness.** Star counts grow as 10^(0.6m) with limiting magnitude and flux
falls as 10^(−0.4m), which composes to a flux drawn as `ξ^(−2/3)` for uniform
ξ — no logarithms needed. Colour is the blackbody LUT the disc already uses,
at a sampled stellar temperature.

**The point-source problem.** A star is a delta function; point-sampling one
either hits or misses, which flickers under motion. Celestia hit this exactly
and needed pixel-level PSF integration. Here the PSF is a Gaussian of width
`sp[4]`, which the host sets to the pixel's angular footprint, and the
renderer's existing jittered supersampling integrates it by Monte Carlo.

The width is deliberately set in the **source sky**, not in screen pixels, so
gravitational magnification stretches and brightens star images near the
critical curve on its own — which is both correct and the thing worth seeing.
Celestia's finding that FOV-relative sizing must take over below ~0.03°/pixel
is what makes the footprint, rather than a fixed angular size, the right
choice: a 4K frame here sits at 0.026°/pixel, already inside that regime.
"""
@inline function starfield_mtl(dx, dy, dz, sp, star_lut)
    N = sp[2]
    fill = sp[3]
    σ = sp[4]
    flux0 = sp[5]
    gnx = sp[6]; gny = sp[7]; gnz = sp[8]
    gconc = sp[9]
    tmin = sp[10]; tspan = sp[11]
    lut_tmin = sp[12]; lut_tmax = sp[13]; lut_size = sp[14]
    seed = unsafe_trunc(Int32, sp[15])

    two_pi = 2.0f0 * Float32(pi)
    u = atan(dy, dx) / two_pi + 0.5f0
    v = (dz + 1.0f0) * 0.5f0
    Ni = unsafe_trunc(Int32, N)
    i0 = unsafe_trunc(Int32, floor(u * N))
    j0 = unsafe_trunc(Int32, floor(v * N))

    inv2σ2 = 1.0f0 / (2.0f0 * σ * σ)
    cut = 25.0f0 * σ * σ            # 5σ; beyond it the Gaussian is negligible
    acc_r = 0.0f0; acc_g = 0.0f0; acc_b = 0.0f0

    for dj in Int32(-1):Int32(1)
        jj = j0 + dj
        (jj < Int32(0) || jj >= Ni) && continue
        for di in Int32(-1):Int32(1)
            ii = mod(i0 + di, Ni)

            su = _sim_hash(ii, jj, seed + Int32(1))
            sv = _sim_hash(ii, jj, seed + Int32(2))
            # Star direction from its own cell coordinates.
            φs = ((Float32(ii) + su) / N - 0.5f0) * two_pi
            sz = 2.0f0 * (Float32(jj) + sv) / N - 1.0f0
            sr = sqrt(max(1.0f0 - sz * sz, 0.0f0))
            sx = sr * cos(φs)
            sy = sr * sin(φs)

            # Chord² ≈ angle² at these scales (σ is ~1e-4 rad).
            ex = dx - sx; ey = dy - sy; ez = dz - sz
            d2 = ex * ex + ey * ey + ez * ez
            d2 > cut && continue

            # Occupancy, thinned away from the galactic plane so the field has
            # a Milky Way concentration rather than being uniform noise.
            occ = fill
            if gconc > 0.0f0
                sb = abs(sx * gnx + sy * gny + sz * gnz)
                occ *= exp(-sb * gconc)
            end
            _sim_hash(ii, jj, seed) < occ || continue

            ξ = max(_sim_hash(ii, jj, seed + Int32(3)), 1.0f-4)
            flux = flux0 * exp(-0.6666667f0 * log(ξ))     # ξ^(−2/3)
            w = flux * exp(-d2 * inv2σ2)

            ht = _sim_hash(ii, jj, seed + Int32(4))
            # Uniform across the range, and the range must *straddle* the LUT's
            # white-balance temperature: the LUT is the disc's, white-balanced
            # at `wb_temperature`, so every star below that point renders warm.
            # A cool-biased draw inside a range that tops out below the white
            # point makes the whole sky gold.
            T = tmin + tspan * ht
            frac = (clamp(T, lut_tmin, lut_tmax) - lut_tmin) /
                   max(lut_tmax - lut_tmin, 1.0f-6)
            li = clamp(unsafe_trunc(Int32, frac * (lut_size - 1.0f0) + 0.5f0) +
                       Int32(1), Int32(1), unsafe_trunc(Int32, lut_size))
            acc_r += w * star_lut[1, li]
            acc_g += w * star_lut[2, li]
            acc_b += w * star_lut[3, li]
        end
    end
    return (acc_r, acc_g, acc_b)
end

"""
    trace_kernel_mtl!(out, bg, bb_lut, star_lut, vol, vol_params, cam_params,
                      spacetime_params, disc_params, width, height, nmax, dt,
                      tol, jitter_u, jitter_v, weight, row0, rows, ::Val{VOL})

Metal compute kernel: one thread per pixel of the current row tile, tracing
geodesics in **Cartesian Kerr–Schild coordinates** (see `ks_rhs_mtl`) — free
of the polar and horizon coordinate singularities of the spherical chart, so
flight never hits pole artifacts. `cam_params` is a Float32 vector of
[`CAM_PARAMS_N`](@ref) entries — pose, projection and motion-blur block, laid
out there. `spacetime_params`
is `[M; r_horizon; r_escape]`. `disc_params` is `[inner_radius; outer_radius;
density_falloff; table_min; table_max; table_size]`; when the radii describe
a valid annulus the kernel composites a semi-transparent Doppler-shaded disc,
colouring crossings from the white-balanced blackbody LUT `bb_lut` of shape
`(3, table_size)`. When `VOL` the volumetric disc grid replaces the thin
plane (see `sample_volume_mtl`).

`jitter_u`/`jitter_v` are the subpixel sample offsets in `[0, 1)`; the sample
is **accumulated** into `out` scaled by `weight`, so the host must zero `out`
before the first pass and the weights of all passes should sum to 1. `row0`
and `rows` select a horizontal tile so large frames can be split across
several short dispatches.
"""
function trace_kernel_mtl!(out, bg, bb_lut, star_lut, vol, vol_params,
                           star_params, cam_params,
                           spacetime_params, disc_params, fan, sky_params,
                           stat,
                           width, height, nmax, dt, tol, jitter_u, jitter_v,
                           jitter_w, jitter_seed,
                           weight, row0, rows, col0, cols,
                           substride, subx, suby,
                           ring_rmin,
                           ::Val{VOL}, ::Val{NB},
                           ::Val{LAYER}, ::Val{ORD},
                           ::Val{KERR}, ::Val{BAKE},
                           ::Val{STAT}) where {VOL, NB, LAYER, ORD, KERR,
                                               BAKE, STAT}
    idx = thread_position_in_grid().x
    total = cols * rows
    if idx > total
        return
    end
    # Dispatch covers the sub-rectangle [col0, col0+cols) x [row0, row0+rows);
    # `col0 = 0, cols = width` is the whole frame. The ring pass uses a real
    # sub-rectangle (see `_ring_screen_box`): it owns a filled disc around the
    # hole, so dispatching the whole frame and culling per pixel made a region
    # covering a few percent of the image cost a near-full-resolution trace —
    # every culled thread still paid the tetrad read, the ray construction, an
    # `acos` and a fan gather before it could return.
    j = (idx - 1) ÷ cols + 1 + row0
    i = (idx - 1) % cols + 1 + col0
    if j > height || i > width
        return
    end

    # Sub-grid pass (LAYER temporal accumulation): this dispatch traces every
    # `substride`-th pixel of a `substride`× larger persistent layer, offset
    # by (subx, suby); the other pixels keep their reprojected history. The
    # sensor plane is that of the full-size layer.
    fw = width
    fh = height
    if LAYER && substride > 1
        fw = width * substride
        fh = height * substride
        i = (i - 1) * substride + 1 + subx
        j = (j - 1) * substride + 1 + suby
    end

    M = spacetime_params[1]
    r_escape = spacetime_params[3]
    # Kerr spin, and the squared capture radius in Kerr-Schild r (the prograde
    # photon orbit — see `_spin_horizon`). Both are inert when !KERR.
    spin_a = spacetime_params[7]
    rkill2 = spacetime_params[8]

    # Unpack camera position, field of view and the orthonormal tetrad
    # (forward/right/up axes + observer 4-velocity, all contravariant,
    # precomputed on the CPU by `ks_camera_tetrad`).
    cx = cam_params[1];  cy = cam_params[2];  cz = cam_params[3]
    fov = cam_params[4]
    ef0 = cam_params[5];  ef1 = cam_params[6];  ef2 = cam_params[7];  ef3 = cam_params[8]
    er0 = cam_params[9];  er1 = cam_params[10]; er2 = cam_params[11]; er3 = cam_params[12]
    eu0 = cam_params[13]; eu1 = cam_params[14]; eu2 = cam_params[15]; eu3 = cam_params[16]
    ut0 = cam_params[17]; ut1 = cam_params[18]; ut2 = cam_params[19]; ut3 = cam_params[20]

    # Optional per-pixel shutter time (`per_pixel_shutter`; off by default).
    # Without it every pixel in the frame is sampled at the same instant inside
    # the pass's shutter stratum, so a moving frame is built from `samples²`
    # frame-wide copies of itself. That is the same defect the stratified-lens
    # block below fixes for the aperture, and this is the same fix: hash a point
    # per pixel inside the stratum and interpolate the pose between its
    # endpoints, cam_params[1:20] and [30:49].
    #
    # It is off by default because measurement did not support turning it on.
    # A/B against a converged reference (1280x720 lensed starfield, 31 px of
    # sweep across the shutter — 7.7 px between adjacent poses at samples=2,
    # far more than any real shot here):
    #
    #   samples   RMS vs reference        error autocorrelation at 1 px
    #             one pose   per pixel    one pose   per pixel
    #     2       0.00875    0.01058       +0.044     +0.008
    #     3       0.00494    0.00554       -0.060     -0.039
    #     4       0.00395    0.00409       -0.043     -0.035
    #
    # So it does what it claims — the error becomes ~5x less spatially
    # structured, i.e. noise rather than ghosts — but total error rises by up to
    # 20%. That is the standard trade: one pose per stratum is midpoint
    # quadrature and converges faster on a smooth integrand, while randomising
    # buys incoherence at the cost of variance. The ghosting it removes turned
    # out to be weak (+0.044 correlation in a deliberately extreme case), so on
    # these numbers there is nothing here worth 20% more noise. Kept behind the
    # flag because the trade may reverse on sharper content or lower sample
    # counts, and it costs nothing to leave available.
    if cam_params[29] > 0.0f0
        τ = _sim_hash(Int32(i), Int32(j), unsafe_trunc(Int32, cam_params[29]))
        cx += τ * (cam_params[30] - cx)
        cy += τ * (cam_params[31] - cy)
        cz += τ * (cam_params[32] - cz)
        ef0 += τ * (cam_params[34] - ef0); ef1 += τ * (cam_params[35] - ef1)
        ef2 += τ * (cam_params[36] - ef2); ef3 += τ * (cam_params[37] - ef3)
        er0 += τ * (cam_params[38] - er0); er1 += τ * (cam_params[39] - er1)
        er2 += τ * (cam_params[40] - er2); er3 += τ * (cam_params[41] - er3)
        eu0 += τ * (cam_params[42] - eu0); eu1 += τ * (cam_params[43] - eu1)
        eu2 += τ * (cam_params[44] - eu2); eu3 += τ * (cam_params[45] - eu3)
        ut0 += τ * (cam_params[46] - ut0); ut1 += τ * (cam_params[47] - ut1)
        ut2 += τ * (cam_params[48] - ut2); ut3 += τ * (cam_params[49] - ut3)
    end

    # Sensor coordinate with subpixel jitter (fw/fh: full sensor size, which
    # differs from the dispatch size only in a LAYER sub-grid pass).
    # `jitter_w > 0` decorrelates the jitter per pixel: (jitter_u, jitter_v)
    # is then this pass's stratum ORIGIN and each pixel hashes its own point
    # inside the stratum — same construction as the lens strata below. With a
    # pass-wide sample point, every pixel shares one sub-pixel phase, and
    # image content finer than a pixel (the wound-image stacks of a spinning
    # hole) aliases into coherent moiré rings instead of averaging to noise.
    ju_px = jitter_u
    jv_px = jitter_v
    if jitter_w > 0.0f0
        sdj = unsafe_trunc(Int32, jitter_seed)
        ju_px += _sim_hash(Int32(i), Int32(j), sdj) * jitter_w
        jv_px += _sim_hash(Int32(i), Int32(j), sdj + Int32(271)) * jitter_w
    end
    half_h = Float32(fh) / 2.0f0
    u = (Float32(i) - 1.0f0 + ju_px - Float32(fw) / 2.0f0) / half_h
    v = (Float32(j) - 1.0f0 + jv_px - Float32(fh) / 2.0f0) / half_h

    # Thin-lens aperture offset, per-pixel stratified: cam_params[26] is the
    # lens stratum width (0 = pinhole), [23:24] this pass's stratum origin in
    # [0,1)², [27] the world-space aperture radius, [28] the pass seed. Each
    # pixel hashes its own point inside the pass's stratum and maps it to the
    # aperture disk with the Shirley–Chiu concentric map — without this, all
    # pixels share one lens point per pass and stars render as `passes`
    # stacked copies instead of smooth bokeh.
    offr = 0.0f0
    offu = 0.0f0
    if cam_params[26] > 0.0f0
        sd = unsafe_trunc(Int32, cam_params[28])
        u01 = cam_params[23] + _sim_hash(Int32(i), Int32(j), sd) * cam_params[26]
        v01 = cam_params[24] + _sim_hash(Int32(i), Int32(j), sd + Int32(7919)) *
              cam_params[26]
        ox = 2.0f0 * u01 - 1.0f0
        oy = 2.0f0 * v01 - 1.0f0
        if ox != 0.0f0 || oy != 0.0f0
            rr = 0.0f0
            θc = 0.0f0
            if abs(ox) > abs(oy)
                rr = ox
                θc = 0.785398f0 * (oy / ox)
            else
                rr = oy
                θc = 1.570796f0 - 0.785398f0 * (ox / oy)
            end
            offr = cam_params[27] * rr * cos(θc)
            offu = cam_params[27] * rr * sin(θc)
        end
    end

    # Pixel direction as unit coefficients on the camera tetrad axes:
    # rectilinear pinhole, or equidistant fisheye (angle ∝ pixel radius).
    cr = 0.0f0
    cu = 0.0f0
    cf = 1.0f0
    if cam_params[21] > 1.5f0
        # Equirectangular over the whole sphere, used to BAKE a warp map (see
        # `bake_warp_map`). The map must be indexed by a direction that means
        # the same thing at bake time and at lookup time, so the angles are
        # measured against the camera's own axes: θ from forward, φ around it
        # from right toward up. Baking with a world-aligned camera makes that
        # index a world direction, which is what the sampler assumes.
        #
        # u spans [-W/H, W/H] and v spans [-1, 1], so a 2:1 map covers
        # φ ∈ [-π, π] and θ ∈ [0, π] exactly.
        φe = u * 1.5707963f0
        θe = (v + 1.0f0) * 1.5707963f0
        sθe = sin(θe)
        cf = cos(θe)
        cr = sθe * cos(φe)
        cu = sθe * sin(φe)
    elseif cam_params[21] > 0.5f0
        ρ = sqrt(u * u + v * v)
        θp = ρ * cam_params[22]
        sθ = sin(θp)
        inv_ρ = ρ > 1.0f-8 ? 1.0f0 / ρ : 0.0f0
        cr = sθ * u * inv_ρ
        cu = sθ * v * inv_ρ
        cf = cos(θp)
    else
        dx_local = u * fov
        dy_local = v * fov
        ν = sqrt(dx_local * dx_local + dy_local * dy_local + 1.0f0)
        # Thin lens: the ray leaves the offset origin aimed at the pinhole
        # ray's focal-plane point, matching the CPU `get_ray(::ThinLensCamera)`.
        if offr != 0.0f0 || offu != 0.0f0
            k = ν / cam_params[25]   # 1 / (focus distance along the ray)
            dx_local -= offr * k
            dy_local -= offu * k
            ν = sqrt(dx_local * dx_local + dy_local * dy_local + 1.0f0)
        end
        cr = dx_local / ν
        cu = dy_local / ν
        cf = 1.0f0 / ν
    end

    # Received photon p = ω(u + n); trace q = n − u, i.e. backward in time
    # (regular through the horizon in both directions). Lower the index:
    # p_μ = η_μν q^ν + f l_μ (l_ν q^ν), with the spacetime's own f and
    # l_μ = (1, l⃗) — radial for Schwarzschild, the spinning KS congruence for
    # Kerr (same convention as `kerr_rhs_mtl`; the CPU-side tetrad in
    # `ks_camera_tetrad` is orthonormalised under the matching metric).
    # The origin carries the aperture offset along the tetrad right/up axes
    # (zero for pinhole).
    x = cx + offr * er1 + offu * eu1
    y = cy + offr * er2 + offu * eu2
    z = cz + offr * er3 + offu * eu3
    r = sqrt(x * x + y * y + z * z)
    qt = cf * ef0 + cr * er0 + cu * eu0 - ut0
    qx = cf * ef1 + cr * er1 + cu * eu1 - ut1
    qy = cf * ef2 + cr * er2 + cu * eu2 - ut2
    qz = cf * ef3 + cr * er3 + cu * eu3 - ut3
    p_t = 0.0f0
    px = 0.0f0
    py = 0.0f0
    pz = 0.0f0
    if KERR
        a2l = spin_a * spin_a
        wl = r * r - a2l
        rk2l = 0.5f0 * (wl + sqrt(wl * wl + 4.0f0 * a2l * z * z))
        rkl = sqrt(max(rk2l, 1.0f-12))
        iRAl = 1.0f0 / (rk2l + a2l)
        lxl = (rkl * x + spin_a * y) * iRAl
        lyl = (rkl * y - spin_a * x) * iRAl
        lzl = z / rkl
        fl = 2.0f0 * M * rk2l * rkl / (rk2l * rk2l + a2l * z * z)
        lq = qt + lxl * qx + lyl * qy + lzl * qz
        flq = fl * lq
        p_t = -qt + flq
        px = qx + flq * lxl
        py = qy + flq * lyl
        pz = qz + flq * lzl
    else
        f = 2.0f0 * M / r
        lq = qt + (x * qx + y * qy + z * qz) / r
        p_t = -qt + f * lq
        flr = f * lq / r
        px = qx + flr * x
        py = qy + flr * y
        pz = qz + flr * z
    end

    # Optional relativistic shading: each ray is normalised to unit frequency
    # in the camera tetrad, and the conserved p_t is the frequency at
    # infinity, so the camera/infinity shift factor is simply g = 1/|p_t|.
    # 1 when the option is off.
    scam = spacetime_params[4] > 0.5f0 ?
           1.0f0 / clamp(abs(p_t), 0.05f0, 20.0f0) : 1.0f0

    # Kerr disc shading: the ray's impact parameter λ = L_z/E, conserved along
    # the geodesic (both are Killing charges), drives the exact circular-orbit
    # Doppler factor 1/g = u^t (1 − Ω λ) in the disc/gas blocks below —
    # replacing the Schwarzschild static-observer split (local boost ×
    # gravitational redshift), which has no spin dependence. Invariant under
    # the backward-ray sign flip (E and L_z negate together).
    lam_ray = 0.0f0
    if KERR
        Ek0 = -p_t
        if abs(Ek0) > 1.0f-6
            lam_ray = (x * py - y * px) / Ek0
        end
    end

    # Bardeen launch-time capture test (Kerr only). The runtime kill radius is
    # a single sphere (the prograde photon orbit), but the unstable photon
    # orbits of a spinning hole fill a SHELL — prograde equatorial through
    # polar to retrograde, r ∈ [photon_orbit_min, ~4M] — and near-extremal
    # spin makes rays grazing that shell wind more orbits than Float32 can
    # track: they emerge with quasi-random momenta and print speckled phantom
    # sky inside the shadow (verified against a Float64 Vern9 reference, which
    # shows clean black where a=0.998 speckles). Whether a ray is captured is
    # decided analytically at launch instead: E = −p_t and L_z = x·p_y − y·p_x
    # are Killing charges, Carter's Q completes the set, and with λ = L_z/E,
    # η = Q/E² the Boyer–Lindquist radial potential is the quartic
    #     R(r)/E² = r⁴ + (a² − λ² − η) r² + 2M(η + (λ−a)²) r − a²η ,
    # with (Σ dr/dλ)² = R. An ingoing ray is captured iff R has no root in
    # (r₊, r_cam) — no turning point before the horizon. R(r₊) ≥ 0 and
    # R(r_cam) ≥ 0 always, so it suffices to check R at the interior roots of
    # R′ (a depressed cubic — R has no r³ term). The verdict only forces the
    # ray's BACKGROUND to shadow; the geodesic still integrates, so gas and
    # disc emission in front of the shadow are kept.
    captured0 = false
    # `graze0` marks rays whose radial potential comes CLOSE to a turning
    # point without having one — near-critical rays that wind deeply and
    # escape. Their image is violently compressed and chaotic, so the pixels
    # they land in are where adaptive refinement spends its extra rays.
    graze0 = false
    if KERR && !LAYER && spin_a != 0.0f0
        E0 = -p_t
        if abs(E0) > 1.0f-6
            a2b = spin_a * spin_a
            awb = r * r - a2b
            rk2b = 0.5f0 * (awb + sqrt(awb * awb + 4.0f0 * a2b * z * z))
            rkb = sqrt(max(rk2b, 1.0f-12))
            cthb = z / rkb
            s2b = max(1.0f0 - cthb * cthb, 1.0f-8)
            sthb = sqrt(s2b)
            Lz0 = x * py - y * px
            pthb = (cthb / sthb) * (x * px + y * py) - rkb * sthb * pz
            Q0 = pthb * pthb + cthb * cthb * (Lz0 * Lz0 / s2b - a2b * E0 * E0)
            lam = Lz0 / E0
            eta = Q0 / (E0 * E0)
            c2b = a2b - lam * lam - eta
            c1b = 2.0f0 * M * (eta + (lam - spin_a) * (lam - spin_a))
            c0b = -a2b * eta
            rplus = M + sqrt(max(M * M - a2b, 0.0f0))
            # Ingoing in KS r? dr/dλ ∝ r²(x·ẋ + y·ẏ) + (r²+a²)·z·ż.
            vx0, vy0, vz0, _dq1, _dq2, _dq3 =
                kerr_rhs_mtl(x, y, z, px, py, pz, p_t, M, spin_a)
            rdot = rk2b * (x * vx0 + y * vy0) + (rk2b + a2b) * z * vz0
            if rdot < 0.0f0
                # Interior critical points of R: roots of r³ + pb·r + qb.
                pb = 0.5f0 * c2b
                qb = 0.25f0 * c1b
                Db = 0.25f0 * qb * qb + pb * pb * pb / 27.0f0
                cap = true
                if Db >= 0.0f0
                    sD = sqrt(Db)
                    u1 = -0.5f0 * qb + sD
                    u2 = -0.5f0 * qb - sD
                    rc = sign(u1) * exp(log(max(abs(u1), 1.0f-30)) / 3.0f0) +
                         sign(u2) * exp(log(max(abs(u2), 1.0f-30)) / 3.0f0)
                    if rc > rplus && rc < rkb
                        Rv = ((rc * rc + c2b) * rc + c1b) * rc + c0b
                        cap = Rv > 0.0f0
                        if abs(Rv) < 0.08f0 * max(rc * rc * rc * rc, 1.0f0)
                            graze0 = true
                        end
                    end
                else
                    mb = 2.0f0 * sqrt(-pb / 3.0f0)
                    ac = clamp(3.0f0 * qb / (pb * mb), -1.0f0, 1.0f0)
                    θb = acos(ac) / 3.0f0
                    for kk in 0:2
                        rc = mb * cos(θb - 2.0943951f0 * Float32(kk))
                        if rc > rplus && rc < rkb
                            Rv = ((rc * rc + c2b) * rc + c1b) * rc + c0b
                            if Rv <= 0.0f0
                                cap = false
                            end
                            if abs(Rv) < 0.08f0 * max(rc * rc * rc * rc, 1.0f0)
                                graze0 = true
                            end
                        end
                    end
                end
                captured0 = cap
            end
        end
    end

    # Layered mode: this pass renders only the disc/gas layer (premultiplied
    # RGB + transmittance in a 4-channel `out`; the sky is composited later
    # from the exact deflection fan). A pixel whose geodesic provably stays
    # outside the gas gate radius — periapsis from the fan, by the local
    # angle ψ to the radial tetrad axis ê_r in `sky_params[1:4]` — writes
    # pure transparency and exits without integrating a single step.
    gate = 0.0f0
    if LAYER
        gate = sky_params[7]
        # `ring_rmin > 0` marks the **ring pass**: a finer-resolution dispatch
        # that owns only the strongly-wound rays. The photon ring is disc light
        # that has looped the hole, so it lives in this layer and is the
        # highest-frequency thing in the frame — upsampling it from a coarse
        # rung is what makes it stair-step. Periapsis is the honest selector:
        # rays dipping near the photon sphere are exactly the ones whose disc
        # crossings pile into the ring, and the fan already carries it.
        if r > gate || ring_rmin > 0.0f0
            cψ = clamp(p_t * sky_params[1] + px * sky_params[2] +
                       py * sky_params[3] + pz * sky_params[4],
                       -1.0f0, 1.0f0)
            Nf = Float32(size(fan, 2))
            tf = acos(cψ) * (Nf - 1.0f0) / Float32(pi)
            k0 = clamp(unsafe_trunc(Int32, tf), Int32(0), unsafe_trunc(Int32, Nf) - Int32(2))
            frac_f = tf - Float32(k0)
            esc = fan[1, k0 + 1] + frac_f * (fan[1, k0 + 2] - fan[1, k0 + 1])
            rmin = fan[4, k0 + 1] + frac_f * (fan[4, k0 + 2] - fan[4, k0 + 1])
            # Ring pass keeps only the annulus; the bulk pass keeps everything
            # the gate cannot prove empty. Each writes transparency elsewhere,
            # so the composite can blend the two without holes.
            cull = ring_rmin > 0.0f0 ? rmin >= ring_rmin :
                                       (esc > 0.999f0 && rmin > gate)
            if cull
                out[1, i, j] = 0.0f0
                out[2, i, j] = 0.0f0
                out[3, i, j] = 0.0f0
                out[4, i, j] = 1.0f0
                return nothing
            end
        end
    end

    disc_inner = disc_params[1]
    disc_outer = disc_params[2]
    disc_falloff = disc_params[3]
    lut_tmin = disc_params[4]
    lut_tmax = disc_params[5]
    lut_size = disc_params[6]
    disc_enabled = disc_inner > 0.0f0 && disc_outer > disc_inner

    # Volumetric disc parameters; when compiled in (VOL) it replaces the
    # thin plane. The dummy buffer keeps these reads valid when !VOL.
    vol_zmax = vol_params[3]
    vol_emis = vol_params[7]
    vol_opac = vol_params[8]
    # Volume march stride: the gas is sampled every `vol_mstep * dt` of arc
    # length (default 2), on its own schedule — NOT every Nth integration
    # step, so the sampling density survives an adaptive integrator taking
    # whatever steps the ODE error controller licenses. Coarser strides are
    # the quality/speed knob for cameras inside the slab; geodesics stay
    # exact either way.
    vol_mstep = clamp(unsafe_trunc(Int32, vol_params[9]), Int32(1), Int32(16))
    ds_gas = Float32(vol_mstep) * dt
    vol_s_out = exp(vol_params[2])
    vol_rb2 = vol_s_out * vol_s_out + vol_zmax * vol_zmax
    disc_plane = disc_enabled && !VOL

    # Accumulated disc colour and remaining transmittance (alpha
    # compositing). Rays are NOT terminated at disc crossings so lensed
    # secondary images composite correctly.
    acc_r = 0.0f0
    acc_g = 0.0f0
    acc_b = 0.0f0
    alpha = 1.0f0

    # Depth bucketing (NB > 0): emission is written directly into `out`'s
    # 3·NB channels, binned by integrated path length `ell` — log-spaced bins
    # over [1.5M, 120M], last bucket reserved for the escaped background.
    # Enables physically-based depth of field as a post operation.
    ell = 0.0f0
    bscale = NB > 2 ? Float32(NB - 2) * 0.22820f0 : 1.0f0   # 1/(ln120 − ln1.5)

    # Fixed-step RK4 integration. Backward-traced rays exit the horizon
    # freely (camera inside: r strictly increases) but can never legally
    # enter it — a ray reaching the band just above r = 2M while moving
    # inward is infinitely redshifted horizon-skimming light (the shadow),
    # and numerically an unstable ridge, so it is killed as black.
    hit_horizon = false
    # Camera outside the horizon: no legal ray is ever below 2M (a dip is
    # horizon-ridge overshoot). Camera inside: rays exit through the band,
    # floor is the near-singularity safety net.
    r_floor = r > 2.05f0 * M ? 2.0f0 * M : 0.3f0 * M
    r_prev = -1.0f0
    r_layer_prev = -1.0f0
    # Adaptive-integrator state, all per-thread: the step size carried between
    # iterations, the previous step's error for the PI controller, and the
    # FSAL slope — Tsit5's 7th stage is evaluated at the accepted step's
    # endpoint, i.e. it IS the next step's k1, so it is carried instead of
    # recomputed.
    h_carry = -1.0f0
    err_prev = 1.0f-4
    k1_carry = (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    k1_valid = false
    # Arc length travelled since the last gas sample (the volumetric march is
    # decoupled from the integrator's step schedule — see below).
    march_acc = 0.0f0
    for stepi in 1:nmax
        r2 = x * x + y * y + z * z
        r = sqrt(r2)
        if LAYER
            # Outbound beyond the gas gate: a null geodesic has at most one
            # radial turning point, so once r exceeds the gate and grows the
            # ray can never re-enter — and the sky is not this pass's job.
            if r > gate && r_layer_prev > 0.0f0 && r > r_layer_prev
                break
            end
            r_layer_prev = r
        end
        if KERR
            # rho != r in Kerr-Schild, so the capture test has to be written
            # in KS r. The threshold is the prograde photon orbit rather than
            # the horizon: no ray reaching infinity dips below it, and testing
            # only the horizon let near-critical rays bounce off the
            # coordinate ridge and escape as phantom sky.
            aw = r2 - spin_a * spin_a
            rk2 = 0.5f0 * (aw + sqrt(aw * aw +
                           4.0f0 * spin_a * spin_a * z * z))
            if rk2 < rkill2
                hit_horizon = true
                break
            end
        elseif r < 3.2f0 * M   # strong-field zone: kill checks live only here
            # Exact criterion: an escaping null geodesic never has a turning
            # point below the photon sphere (periapsis > 3M requires
            # b > b_crit), so a ray moving inward below ~2.95M can never
            # legally return — it is sub-critical, horizon-bound light: the
            # shadow. This also stops rays numerically bouncing off the
            # horizon ridge and escaping as phantom sky.
            if r < r_floor ||
               (r < 2.95f0 * M && r_prev > 0.0f0 && r < r_prev - 1.0f-4 * M)
                hit_horizon = true
                break
            end
            r_prev = r
        end
        if r > r_escape
            break   # ray escaped: background will be sampled below
        end

        # Radius-adaptive affine step: curvature ~ M/r³, so scaling h with r
        # keeps the per-step bending error uniform while collapsing the
        # nearly-flat travel legs. Capped at 2× inside the gas volume: the
        # bounding sphere contains the strong field, where the shadow-kill
        # criteria need small steps (a larger cap makes near-critical rays
        # wander to the step limit — slower AND wrong).
        hcap = (VOL && r2 < vol_rb2) ? 2.0f0 : spacetime_params[6]
        h = dt * min(max(spacetime_params[5] * r / M, 1.0f0), hcap)
        if KERR
            # The capture radius and the horizon converge as a -> M (1.074M
            # against 1.063M at a = 0.998), and a step floored at `dt` cannot
            # resolve that gap — rays cross the band in one step, miss the
            # test, and escape as phantom sky. Shrink the step as the ray
            # closes on capture; far away this is a no-op.
            fr = clamp((rk2 - rkill2) / max(rkill2, 1.0f-6), 0.0f0, 1.0f0)
            h *= 0.12f0 + 0.88f0 * fr
        end

        xp = x; yp = y; zp = z
        pxp = px; pyp = py; pzp = pz

        k1 = ((ORD == 45 || ORD == 46) && k1_valid) ? k1_carry :
             _rhs_mtl(Val(KERR), x, y, z, px, py, pz, p_t, M, spin_a)

        # `ORD` picks the integrator at compile time: 4 = classical RK4 (4 RHS
        # evaluations per step), 2 = explicit midpoint (2), 1 = Euler (1),
        # 45/46 = adaptive Tsit5 (6 new evaluations per attempt; k1 is the
        # carried FSAL stage). `k1` is already in hand above — the volumetric
        # sampler needs the photon's coordinate velocity — so Euler adds no
        # evaluation at all and midpoint adds exactly one. `h_used` is the
        # step actually taken: the radius-adaptive h for the fixed-order
        # paths, whatever the error controller settled on for 45/46 — the gas
        # march below needs it.
        h_used = h
        if ORD == 45 || ORD == 46
            # Adaptive Tsitouras 5(4) — the tableau that replaced
            # Dormand-Prince as the modern default (and here replaces the
            # Cash-Karp pair the Rust shader still carries): seven stages, an
            # embedded fourth-order solution for the error estimate, FSAL (the
            # 7th stage sits at the step's endpoint, so it is next step's k1),
            # per-thread step control — SIMD lanes diverge only where
            # neighbouring rays genuinely differ, near the photon ring.
            #
            # ORD 45 caps the step at the radius-adaptive h. That cap is not
            # about accuracy — it is what the once-per-step shadow-kill tests
            # need: a long step near the photon sphere jumps the band the
            # tests look at and a captured ray escapes as phantom sky. ORD 46
            # lets the step run free (up to 50·dt) and relies on the error
            # controller tightening near the hole — the arc-length gas march
            # below stays correct under big steps, which is what makes this
            # mode usable at all. The thin-plane disc keeps an approach guard:
            # its crossing is located by linear interpolation across ONE step,
            # so that one piece still depends on the step schedule.
            hmax = h
            if ORD == 46
                hmax = 50.0f0 * dt
                if disc_plane
                    hmax = min(hmax, max(h, 0.5f0 * (r - disc_outer)))
                end
            end
            hh = min(h_carry > 0.0f0 ? h_carry : h, hmax)
            s0 = (x, y, z, px, py, pz)
            sn = s0
            accepted = false
            att = Int32(0)
            while att < Int32(4) && !accepted
                att += Int32(1)
                s2 = _axpy6(s0, hh * 0.161f0, k1)
                k2 = _rhs_mtl(Val(KERR), s2[1], s2[2], s2[3], s2[4], s2[5],
                              s2[6], p_t, M, spin_a)
                s3 = _axpy6(_axpy6(s0, -hh * 0.0084806555f0, k1),
                            hh * 0.33548066f0, k2)
                k3 = _rhs_mtl(Val(KERR), s3[1], s3[2], s3[3], s3[4], s3[5],
                              s3[6], p_t, M, spin_a)
                s4 = _axpy6(_axpy6(_axpy6(s0, hh * 2.8971531f0, k1),
                            -hh * 6.3594485f0, k2), hh * 4.3622954f0, k3)
                k4 = _rhs_mtl(Val(KERR), s4[1], s4[2], s4[3], s4[4], s4[5],
                              s4[6], p_t, M, spin_a)
                s5 = _axpy6(_axpy6(_axpy6(_axpy6(s0,
                            hh * 5.3258648f0, k1), -hh * 11.748884f0, k2),
                            hh * 7.4955393f0, k3), -hh * 0.092495066f0, k4)
                k5 = _rhs_mtl(Val(KERR), s5[1], s5[2], s5[3], s5[4], s5[5],
                              s5[6], p_t, M, spin_a)
                s6 = _axpy6(_axpy6(_axpy6(_axpy6(_axpy6(s0,
                            hh * 5.8614554f0, k1), -hh * 12.920969f0, k2),
                            hh * 8.1593679f0, k3), -hh * 0.071584973f0, k4),
                            -hh * 0.028269050f0, k5)
                k6 = _rhs_mtl(Val(KERR), s6[1], s6[2], s6[3], s6[4], s6[5],
                              s6[6], p_t, M, spin_a)
                # The seventh stage sits at the fifth-order solution itself
                # (c7 = 1, a7j = bj), so `s7` IS the proposed new state and
                # `k7` is the slope there — the FSAL carry.
                s7 = _axpy6(_axpy6(_axpy6(_axpy6(_axpy6(_axpy6(s0,
                            hh * 0.096460767f0, k1), hh * 0.01f0, k2),
                            hh * 0.47988965f0, k3), hh * 1.3790086f0, k4),
                            -hh * 3.2900695f0, k5), hh * 2.3247105f0, k6)
                k7 = _rhs_mtl(Val(KERR), s7[1], s7[2], s7[3], s7[4], s7[5],
                              s7[6], p_t, M, spin_a)
                sn = s7
                h_used = hh
                k1_carry = k7
                k1_valid = true

                # Embedded error: b − b̂ weights over all seven stages.
                et = _axpy6(_axpy6(_axpy6(_axpy6(_axpy6(_axpy6(
                     _scale6(k1, -0.0017800111f0), -0.00081643446f0, k2),
                     0.007880878f0, k3), -0.14471101f0, k4),
                     0.58235717f0, k5), -0.45808211f0, k6),
                     0.015151515f0, k7)
                ex = hh * sqrt(et[1] * et[1] + et[2] * et[2] + et[3] * et[3])
                ep = hh * sqrt(et[4] * et[4] + et[5] * et[5] + et[6] * et[6])
                # Mixed absolute/relative scaling, so a ray far from the hole
                # is not held to the same absolute error as one at periapsis.
                pn = sqrt(px * px + py * py + pz * pz)
                sc = tol * (1.0f0 + max(r, pn))
                err = max(ex, ep) / max(sc, 1.0f-30)

                if err <= 1.0f0 || hh <= 1.0f-4 * dt
                    accepted = true
                    # PI step control (Gustafsson): the growth also looks at
                    # the PREVIOUS accepted error, which damps the
                    # grow-to-the-clamp-then-reject oscillation the plain
                    # I-controller falls into at loose tolerances (measured:
                    # tol 1e-3 ran SLOWER than 1e-4 under I-control).
                    g = err > 1.0f-12 ?
                        0.9f0 * exp(-0.14f0 * log(err) +
                                    0.08f0 * log(err_prev)) : 4.0f0
                    h_carry = min(hh * clamp(g, 0.2f0, 4.0f0), hmax)
                    err_prev = max(err, 1.0f-4)
                else
                    hh = max(hh * max(0.9f0 * exp(-0.2f0 * log(err)), 0.2f0),
                             1.0f-4 * dt)
                end
                # A rejected attempt is work the GPU did and threw away; the
                # `att` bound is what keeps that bounded per step.
            end
            x = sn[1]; y = sn[2]; z = sn[3]
            px = sn[4]; py = sn[5]; pz = sn[6]
        elseif ORD == 4
            k2 = _rhs_mtl(Val(KERR), 
                x + 0.5f0 * h * k1[1], y + 0.5f0 * h * k1[2], z + 0.5f0 * h * k1[3],
                px + 0.5f0 * h * k1[4], py + 0.5f0 * h * k1[5], pz + 0.5f0 * h * k1[6],
                p_t, M, spin_a)
            k3 = _rhs_mtl(Val(KERR), 
                x + 0.5f0 * h * k2[1], y + 0.5f0 * h * k2[2], z + 0.5f0 * h * k2[3],
                px + 0.5f0 * h * k2[4], py + 0.5f0 * h * k2[5], pz + 0.5f0 * h * k2[6],
                p_t, M, spin_a)
            k4 = _rhs_mtl(Val(KERR), 
                x + h * k3[1], y + h * k3[2], z + h * k3[3],
                px + h * k3[4], py + h * k3[5], pz + h * k3[6],
                p_t, M, spin_a)

            x  += (h / 6.0f0) * (k1[1] + 2.0f0 * k2[1] + 2.0f0 * k3[1] + k4[1])
            y  += (h / 6.0f0) * (k1[2] + 2.0f0 * k2[2] + 2.0f0 * k3[2] + k4[2])
            z  += (h / 6.0f0) * (k1[3] + 2.0f0 * k2[3] + 2.0f0 * k3[3] + k4[3])
            px += (h / 6.0f0) * (k1[4] + 2.0f0 * k2[4] + 2.0f0 * k3[4] + k4[4])
            py += (h / 6.0f0) * (k1[5] + 2.0f0 * k2[5] + 2.0f0 * k3[5] + k4[5])
            pz += (h / 6.0f0) * (k1[6] + 2.0f0 * k2[6] + 2.0f0 * k3[6] + k4[6])
        elseif ORD == 2
            k2 = _rhs_mtl(Val(KERR), 
                x + 0.5f0 * h * k1[1], y + 0.5f0 * h * k1[2], z + 0.5f0 * h * k1[3],
                px + 0.5f0 * h * k1[4], py + 0.5f0 * h * k1[5], pz + 0.5f0 * h * k1[6],
                p_t, M, spin_a)
            x  += h * k2[1];  y  += h * k2[2];  z  += h * k2[3]
            px += h * k2[4];  py += h * k2[5];  pz += h * k2[6]
        else
            x  += h * k1[1];  y  += h * k1[2];  z  += h * k1[3]
            px += h * k1[4];  py += h * k1[5];  pz += h * k1[6]
        end

        # A non-finite ray can never satisfy the exit tests and would reach
        # the background sampler as NaN, trapping the kernel. Paint it black
        # and stop. (KS coordinates make this far rarer than the spherical
        # chart ever did.)
        if !(x == x) || !(z == z) || !(px == px)
            hit_horizon = true
            break
        end

        vlen = max(sqrt(k1[1] * k1[1] + k1[2] * k1[2] + k1[3] * k1[3]),
                   1.0f-20)
        if NB > 0
            ell += h_used * vlen
        end

        # Volumetric gas, marched on its OWN arc-length stride: one sample per
        # `ds_gas` of path, at positions interpolated along the accepted step.
        # Tying the march to the integrator's step schedule instead (every Nth
        # step, N× weight) breaks under an adaptive integrator — the error
        # controller bounds the ODE's truncation error and has no idea the gas
        # exists, so adaptive steps resample it at wildly uneven arc lengths
        # (0.389 RMS against RK4 in the Rust port). Decoupled, gas quality is
        # independent of how the geodesic was stepped. Shading uses the
        # start-of-step velocity `k1`; within a step the bending is small.
        # Gated on the SEGMENT's nearer endpoint, not the pre-step point: a
        # free-running ORD-46 leg can enter the bounding sphere mid-step.
        if VOL && alpha > 0.003f0 &&
           min(r2, x * x + y * y + z * z) < vol_rb2
            seg = h_used * vlen
            t_s = ds_gas - march_acc   # arc distance to the next sample
            if t_s > seg
                march_acc += seg
            else
                while t_s <= seg && alpha > 0.003f0
                    fseg = t_s / seg
                    sx = xp + fseg * (x - xp)
                    sy = yp + fseg * (y - yp)
                    sz = zp + fseg * (z - zp)
                    if abs(sz) < vol_zmax
                        s_cyl = sqrt(sx * sx + sy * sy)
                        if s_cyl > 1.0f-6
                            φv = atan(sy, sx)
                            ρ = sample_volume_mtl(vol, vol_params, s_cyl, φv, sz)
                            if ρ > 1.0f-4
                                R = s_cyl / (2.0f0 * M)
                                T_emit = exp(10.034259f0 -
                                             0.375f0 * log(max(R * R, 1.0f-6)))
                                opz = 0.1f0
                                if KERR
                                    # Exact prograde circular-orbit shift:
                                    # 1/g = u^t (1 − Ω λ), spin-aware — see
                                    # the lam_ray block at ray setup.
                                    sqM = sqrt(M)
                                    s32 = s_cyl * sqrt(s_cyl)
                                    Ωk = sqM / (s32 + spin_a * sqM)
                                    den = s32 - 3.0f0 * M * sqrt(s_cyl) +
                                          2.0f0 * spin_a * sqM
                                    ut = (s32 + spin_a * sqM) /
                                         (sqrt(s32) * sqrt(max(den, 1.0f-2)))
                                    opz = clamp(ut * (1.0f0 - Ωk * lam_ray),
                                                0.1f0, 20.0f0)
                                else
                                    v_mag = clamp(0.70710678f0 /
                                                  sqrt(max(R - 1.0f0, 0.1f0)),
                                                  0.0f0, 0.999f0)
                                    # Keplerian flow ϕ̂ = (−y, x, 0)/s against
                                    # the photon coordinate velocity k1[1:3].
                                    vdotn = v_mag *
                                            (-sy * k1[1] + sx * k1[2]) /
                                            (s_cyl * vlen)
                                    gam = 1.0f0 / sqrt(1.0f0 -
                                              clamp(v_mag * v_mag,
                                                    0.0f0, 0.99f0))
                                    Rs = sqrt(sx * sx + sy * sy + sz * sz) /
                                         (2.0f0 * M)
                                    opzg = 1.0f0 / sqrt(max(1.0f0 -
                                               1.0f0 / max(Rs, 1.0f0), 0.01f0))
                                    opz = max(gam * (1.0f0 + vdotn) * opzg,
                                              0.1f0)
                                end
                                T_obs = T_emit * scam / opz
                                inten = 100.0f0 /
                                        (exp(29622.4f0 / max(T_obs, 1.0f0)) - 1.0f0)

                                frac = (clamp(T_obs, lut_tmin, lut_tmax) - lut_tmin) /
                                       (lut_tmax - lut_tmin)
                                li = clamp(unsafe_trunc(Int32,
                                               frac * (lut_size - 1.0f0) + 0.5f0) +
                                           Int32(1), Int32(1),
                                           unsafe_trunc(Int32, lut_size))
                                col_r = bb_lut[1, li]
                                col_g = bb_lut[2, li]
                                col_b = bb_lut[3, li]

                                tau = vol_opac * ρ * ds_gas
                                a = 1.0f0 - exp(-tau)
                                w = alpha * a * inten * vol_emis
                                if NB > 0
                                    bi = ell <= 1.5f0 ? Int32(1) :
                                         clamp(unsafe_trunc(Int32,
                                                   (log(ell) - 0.405465f0) * bscale) +
                                               Int32(1), Int32(1), Int32(NB - 1))
                                    out[3 * (bi - 1) + 1, i, j] += weight * w * col_r
                                    out[3 * (bi - 1) + 2, i, j] += weight * w * col_g
                                    out[3 * (bi - 1) + 3, i, j] += weight * w * col_b
                                else
                                    acc_r += w * col_r
                                    acc_g += w * col_g
                                    acc_b += w * col_b
                                end
                                alpha *= (1.0f0 - a)
                            end
                        end
                    end
                    t_s += ds_gas
                end
                march_acc = seg - (t_s - ds_gas)
                if alpha < 0.003f0
                    break   # transmittance exhausted
                end
            end
        end

        # Thin-plane disc: equatorial (z = 0) crossing, located by linear
        # interpolation across the step.
        if disc_plane && alpha > 0.001f0 && zp * z < 0.0f0
            cf = zp / (zp - z)
            xh = xp + cf * (x - xp)
            yh = yp + cf * (y - yp)
            s = sqrt(xh * xh + yh * yh)
            if disc_inner < s && s < disc_outer
                pxh = pxp + cf * (px - pxp)
                pyh = pyp + cf * (py - pyp)
                pzh = pzp + cf * (pz - pzp)
                # Photon coordinate velocity at the crossing (z = 0 ⇒ r = s).
                fh = 2.0f0 * M / s
                κh = (xh * pxh + yh * pyh) / s
                ℓh = -p_t + κh
                c1h = fh * ℓh / s
                vx = pxh - c1h * xh
                vy = pyh - c1h * yh
                vz = pzh
                plen = max(sqrt(vx * vx + vy * vy + vz * vz), 1.0f-20)

                R = s / (2.0f0 * M)
                T_emit = exp(10.034259f0 - 0.375f0 * log(R * R))
                opz = 0.1f0
                if KERR
                    # Exact prograde circular-orbit shift, spin-aware — see
                    # the lam_ray block at ray setup.
                    sqM = sqrt(M)
                    s32 = s * sqrt(s)
                    Ωk = sqM / (s32 + spin_a * sqM)
                    den = s32 - 3.0f0 * M * sqrt(s) + 2.0f0 * spin_a * sqM
                    ut = (s32 + spin_a * sqM) /
                         (sqrt(s32) * sqrt(max(den, 1.0f-2)))
                    opz = clamp(ut * (1.0f0 - Ωk * lam_ray), 0.1f0, 20.0f0)
                else
                    v_mag = clamp(0.70710678f0 / sqrt(max(R - 1.0f0, 0.1f0)),
                                  0.0f0, 0.999f0)
                    vdotn = v_mag * (-yh * vx + xh * vy) / (s * plen)
                    gam = 1.0f0 / sqrt(1.0f0 -
                                       clamp(v_mag * v_mag, 0.0f0, 0.99f0))
                    opzg = 1.0f0 / sqrt(max(1.0f0 - 1.0f0 / max(R, 1.0f0),
                                            0.01f0))
                    opz = max(gam * (1.0f0 + vdotn) * opzg, 0.1f0)
                end
                T_obs = T_emit * scam / opz
                inten = 100.0f0 / (exp(29622.4f0 / max(T_obs, 1.0f0)) - 1.0f0)

                frac = (clamp(T_obs, lut_tmin, lut_tmax) - lut_tmin) /
                       (lut_tmax - lut_tmin)
                li = clamp(unsafe_trunc(Int32, frac * (lut_size - 1.0f0) + 0.5f0) +
                           Int32(1), Int32(1), unsafe_trunc(Int32, lut_size))
                col_r = bb_lut[1, li]
                col_g = bb_lut[2, li]
                col_b = bb_lut[3, li]

                R_in = disc_inner / (2.0f0 * M)
                R_out = disc_outer / (2.0f0 * M)
                iscotaper = clamp((R * R - R_in * R_in) * 0.3f0, 0.0f0, 1.0f0)
                outertaper = clamp(T_emit / 1000.0f0, 0.0f0, 1.0f0)
                density = clamp((R_out - R) / (R_out - R_in), 0.0f0, 1.0f0)
                dpow = density <= 1.0f-6 ?
                    (disc_falloff > 0.0f0 ? 0.0f0 : 1.0f0) :
                    exp(disc_falloff * log(density))
                opacity = iscotaper * outertaper * dpow

                w = alpha * opacity * inten
                if NB > 0
                    bi = ell <= 1.5f0 ? Int32(1) :
                         clamp(unsafe_trunc(Int32,
                                   (log(ell) - 0.405465f0) * bscale) +
                               Int32(1), Int32(1), Int32(NB - 1))
                    out[3 * (bi - 1) + 1, i, j] += weight * w * col_r
                    out[3 * (bi - 1) + 2, i, j] += weight * w * col_g
                    out[3 * (bi - 1) + 3, i, j] += weight * w * col_b
                else
                    acc_r += w * col_r
                    acc_g += w * col_g
                    acc_b += w * col_b
                end
                alpha *= (1.0f0 - opacity)
            end
        end
    end

    W = size(bg, 2)
    H = size(bg, 3)

    if BAKE
        # Warp-map output: everything about this geodesic EXCEPT the sky it
        # landed on. Baking the final colour instead caps output sharpness at
        # the map's angular resolution — 1024 px over 360 degrees is five times
        # coarser than a 1080p frame across a 10 mm field, so stars came out as
        # mush and the map size became what limited image quality.
        #
        # Storing the escape DIRECTION moves the sharpness back to render time.
        # The deflection field is smooth and survives a coarse table; the
        # starfield is then evaluated per pixel against the interpolated
        # direction, so output stays sharp at any resolution and a 512x256 map
        # is enough.
        #
        # Eight channels: direction (1:3), premultiplied disc emission (4:6),
        # transmittance to the sky (7), escape flag (8).
        rfb = max(sqrt(x * x + y * y + z * z), 1.0f-6)
        escf = 0.0f0
        if hit_horizon || captured0 || rfb < 4.0f0 * M
            out[1, i, j] = 0.0f0; out[2, i, j] = 0.0f0; out[3, i, j] = 0.0f0
        else
            escf = 1.0f0
            vbx, vby, vbz, _, _, _ = _rhs_mtl(Val(KERR), x, y, z, px, py, pz,
                                              p_t, M, spin_a)
            vbl = max(sqrt(vbx * vbx + vby * vby + vbz * vbz), 1.0f-20)
            out[1, i, j] = vbx / vbl
            out[2, i, j] = vby / vbl
            out[3, i, j] = vbz / vbl
        end
        out[4, i, j] = acc_r; out[5, i, j] = acc_g; out[6, i, j] = acc_b
        out[7, i, j] = alpha
        # Escape flag as its OWN channel rather than inferred from the
        # direction's length. Interpolating two unit vectors that point
        # different ways also shortens the result, so length would read a
        # rapidly-turning deflection field as shadow and paint a dark rim
        # exactly where the deflection is most interesting.
        out[8, i, j] = escf
        return nothing
    end

    if LAYER
        # Disc/gas layer output: premultiplied emission + transmittance to
        # the sky, written by assignment — every launched thread owns its
        # pixel outright (temporal accumulation keeps the rest). Capture
        # blackness is not written here — the sky pass owns the shadow (at
        # native resolution, from the exact fan).
        out[1, i, j] = acc_r
        out[2, i, j] = acc_g
        out[3, i, j] = acc_b
        out[4, i, j] = alpha
        return nothing
    end

    # Rays that ran out of steps while still deep in the strong field are
    # (near-)critical or horizon-hugging: treat them as black too, as are
    # rays the Bardeen launch test proved captured regardless of where the
    # Float32 integration wandered.
    rf = max(sqrt(x * x + y * y + z * z), 1.0f-6)
    if hit_horizon || captured0 || rf < 4.0f0 * M
        if NB == 0   # bucketed emission was already written during the march
            out[1, i, j] += weight * acc_r
            out[2, i, j] += weight * acc_g
            out[3, i, j] += weight * acc_b
            if STAT
                # Per-pass luminance moments + graze fraction, for adaptive
                # refinement: channel 1 Σw·L, channel 2 Σw·L², channel 3
                # Σw·graze. Weights sum to 1 over the base passes, so
                # (ch2 − ch1²) estimates the per-RAY variance.
                Ls = 0.2126f0 * acc_r + 0.7152f0 * acc_g + 0.0722f0 * acc_b
                stat[1, i, j] += weight * Ls
                stat[2, i, j] += weight * Ls * Ls
                graze0 && (stat[3, i, j] += weight)
            end
        end
    else
        # Escaped rays show the background sky attenuated by any disc gas
        # along the way. Sample by the asymptotic momentum direction, not the
        # escape position: position sampling parallax-shifts stars by up to
        # ~b/r_escape radians for disc-grazing rays.
        # Asymptotic direction is the photon's coordinate velocity, i.e. the
        # integrator's own dx/dλ — correct for Kerr as well as Schwarzschild,
        # where the previous hand-inlined form was not.
        vfx, vfy, vfz, _, _, _ = _rhs_mtl(Val(KERR), x, y, z, px, py, pz,
                                          p_t, M, spin_a)
        vx = vfx
        vy = vfy
        vz = vfz
        vl = max(sqrt(vx * vx + vy * vy + vz * vz), 1.0f-20)
        θbg = acos(clamp(vz / vl, -1.0f0, 1.0f0))
        φbg = atan(vy, vx)
        r_col, g_col, b_col = sample_background_mtl(bg, θbg, φbg, W, H)
        # Procedural point stars, evaluated in the SOURCE sky so lensing
        # magnifies them as it does everything else. `star_params[16]` dims the
        # texture, so the two can be crossfaded rather than only swapped.
        if star_params[1] > 0.0f0
            tw = star_params[16]
            r_col *= tw; g_col *= tw; b_col *= tw
            sr, sg, sb = starfield_mtl(vx / vl, vy / vl, vz / vl,
                                       star_params, star_lut)
            r_col += star_params[1] * sr
            g_col += star_params[1] * sg
            b_col += star_params[1] * sb
        end
        if scam != 1.0f0
            # Relativistic sky: a ~5800 K star observed at T = g·5800 K.
            # Per-channel Planck ratios at 610/550/465 nm; brightness boost
            # (→ g⁴ bolometric) emerges from the same formula.
            r_col *= 57.4f0 / (exp(4.067f0 / scam) - 1.0f0)
            g_col *= 90.2f0 / (exp(4.513f0 / scam) - 1.0f0)
            b_col *= 206.5f0 / (exp(5.335f0 / scam) - 1.0f0)
        end
        if NB > 0   # background lives in the last (infinity) bucket
            out[3 * (NB - 1) + 1, i, j] += weight * alpha * r_col
            out[3 * (NB - 1) + 2, i, j] += weight * alpha * g_col
            out[3 * (NB - 1) + 3, i, j] += weight * alpha * b_col
        else
            out[1, i, j] += weight * (acc_r + alpha * r_col)
            out[2, i, j] += weight * (acc_g + alpha * g_col)
            out[3, i, j] += weight * (acc_b + alpha * b_col)
            if STAT
                Ls = 0.2126f0 * (acc_r + alpha * r_col) +
                     0.7152f0 * (acc_g + alpha * g_col) +
                     0.0722f0 * (acc_b + alpha * b_col)
                stat[1, i, j] += weight * Ls
                stat[2, i, j] += weight * Ls * Ls
                graze0 && (stat[3, i, j] += weight)
            end
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
Length of the camera parameter buffer.

`1:20` is the pose — position, fov factor, and the orthonormal tetrad
(forward / right / up / observer 4-velocity). `21:22` is the fisheye flag and
half-angle, `23:28` the thin-lens stratum, aperture radius and pass seed.

`29:49` is the motion-blur block: `29` is the per-pixel shutter hash seed (0
disables it) and `30:49` is a second copy of the `1:20` pose, at the far end of
this pass's shutter stratum. When it is on, each pixel draws its own time
inside the stratum and interpolates between the two poses, so the frame does
not resolve into `samples²` sharp ghosts of itself. See the kernel's
per-pixel-shutter block.
"""
const CAM_PARAMS_N = 49

# The pose block, 1:20 and mirrored at 30:49.
const CAM_POSE_N = 20

"""
Camera parameter block: position, fov, the KS tetrad, the projection, and the
motion-blur end pose. `fisheye_deg > 0` selects an equidistant fisheye with that
vertical half-angle at the image's top edge (pixel radius ∝ view angle, so
fields wider than 180° render cleanly — a rectilinear pinhole caps below
180° at any focal length). See [`CAM_PARAMS_N`](@ref) for the layout.
"""
function _ks_cam_params!(dest::Vector{Float32}, cam::Camera, M::Float64;
                         fisheye_deg::Real=0.0, focus_dist::Real=1.0,
                         equirect::Bool=false, a::Float64=0.0)
    u4, Ef, Er, Eu = ks_camera_tetrad(cam.pos, cam.fwd, cam.right,
                                      cam.up_local, M; beta=camera_beta(cam),
                                      a=a)
    fill!(dest, 0.0f0)
    @inbounds begin
        dest[1] = cam.pos[1]; dest[2] = cam.pos[2]; dest[3] = cam.pos[3]
        dest[4] = cam.fov_factor
        for k in 1:4
            dest[4 + k]  = Ef[k]; dest[8 + k]  = Er[k]
            dest[12 + k] = Eu[k]; dest[16 + k] = u4[k]
        end
        dest[21] = equirect ? 2.0f0 : (fisheye_deg > 0 ? 1.0f0 : 0.0f0)
        dest[22] = deg2rad(max(fisheye_deg, 0.0))
        dest[25] = focus_dist
    end
    return dest
end

function _ks_cam_params(cam::Camera, M::Float64; fisheye_deg::Real=0.0,
                        focus_dist::Real=1.0, equirect::Bool=false,
                        a::Float64=0.0)
    u4, Ef, Er, Eu = ks_camera_tetrad(cam.pos, cam.fwd, cam.right,
                                      cam.up_local, M; beta=camera_beta(cam),
                                      a=a)
    p = Float32[cam.pos[1], cam.pos[2], cam.pos[3], cam.fov_factor,
                Ef..., Er..., Eu..., u4...,
                equirect ? 2.0 : (fisheye_deg > 0 ? 1.0 : 0.0),
                deg2rad(max(fisheye_deg, 0.0)),
                0.0, 0.0, focus_dist,
                0.0, 0.0, 0.0]   # lens stratum origin/width, radius, seed
    # Motion-blur block off: seed 0, end pose unused.
    return vcat(p, zeros(Float32, CAM_PARAMS_N - length(p)))
end

"""
    render_preview_mtl(ctx::MetalPreviewContext, cam::Camera,
                       spacetime::AbstractSpacetime)

Render one preview frame on the GPU using `ctx`.  Returns a `width × height`
`Matrix{RGBf}` suitable for display.

    render_preview_mtl!(img, host, ctx, cam, spacetime)

In-place variant for render loops: writes into a caller-owned `img`
(`Matrix{RGBf}(undef, width, height)`) via the caller-owned staging buffer
`host` (`Array{Float32,3}(undef, 3, width, height)`), so a flight loop
allocates nothing per frame.
"""
function render_preview_mtl(ctx::MetalPreviewContext, cam::Camera,
                            spacetime::AbstractSpacetime; fisheye_deg::Real=0.0,
                            relativistic::Bool=false,
                            order::Int=4, tol::Real=1.0f-4)
    img = Matrix{RGBf}(undef, ctx.width, ctx.height)
    host = Array{Float32,3}(undef, 3, ctx.width, ctx.height)
    return render_preview_mtl!(img, host, ctx, cam, spacetime;
                               fisheye_deg=fisheye_deg,
                               relativistic=relativistic,
                               order=order, tol=tol)
end

function render_preview_mtl!(img::Matrix{RGBf}, host::Array{Float32,3},
                             ctx::MetalPreviewContext, cam::Camera,
                             spacetime::AbstractSpacetime; fisheye_deg::Real=0.0,
                             relativistic::Bool=false,
                             band_rows::Int=0,
                             on_band::Union{Nothing,Function}=nothing,
                             order::Int=4, tol::Real=1.0f-4)
    _trace_preview_gpu!(ctx, cam, spacetime; fisheye_deg=fisheye_deg,
                        relativistic=relativistic,
                        band_rows=band_rows, on_band=on_band,
                        order=order, tol=tol)
    copyto!(host, ctx.out_gpu)
    @inbounds for j in 1:ctx.height, i in 1:ctx.width
        img[i, j] = RGBf(host[1, i, j], host[2, i, j], host[3, i, j])
    end
    return img
end

"""
GPU-only preview trace: run the kernel into `ctx.out_gpu` without downloading
to the host. The native shell presents `out_gpu` straight to a CAMetalLayer;
`render_preview_mtl!` adds the host download for CPU consumers.
"""
function _trace_preview_gpu!(ctx::MetalPreviewContext, cam::Camera,
                             spacetime::AbstractSpacetime; fisheye_deg::Real=0.0,
                             relativistic::Bool=false,
                             band_rows::Int=0,
                             on_band::Union{Nothing,Function}=nothing,
                             order::Int=4, tol::Real=1.0f-4)
    M = Float32(spacetime.M)
    r_band = Float32(2.05 * spacetime.M)
    r_escape = Float32(ctx.r_escape_factor * max(norm(cam.pos),
                                                 15.0 * spacetime.M))
    dt = ctx.dt
    # Dynamic step count. With radius-adaptive steps the travel legs are
    # logarithmic in r_escape; the constant is the strong-field winding
    # budget (a few photon-sphere orbits). The cap keeps a single dispatch
    # under the macOS GPU watchdog even when the camera is very far away.
    nmax = min(max(ctx.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt)),
               20_000)

    # Update reusable GPU parameter buffers with a single host-to-device copy.
    _ks_cam_params!(ctx.cam_host, cam, spacetime.M; fisheye_deg=fisheye_deg,
                    a=spin(spacetime))
    spin_a, rkill2 = _spin_horizon(spacetime)
    is_kerr = spin_a != 0
    sh = ctx.st_host
    @inbounds begin
        sh[1] = M; sh[2] = r_band; sh[3] = r_escape
        sh[4] = relativistic ? 1.0f0 : 0.0f0
        sh[5] = HSTEP_COEF[]; sh[6] = HSTEP_CAP[]
        sh[7] = spin_a; sh[8] = rkill2
    end

    fill!(ctx.out_gpu, 0.0f0)
    if band_rows <= 0 || on_band === nothing
        _launch_trace!(ctx, ctx.out_gpu, ctx.cam_params, ctx.spacetime_params,
                       ctx.width, ctx.height, nmax, dt,
                       0.5f0, 0.5f0, 1.0f0, 0, ctx.height; kerr=is_kerr,
                       order=order, tol=tol)
    else
        # Banded dispatch: split the frame into row bands and call `on_band`
        # after each one, so a flight loop can slot cheap reprojection
        # frames between bands — the display keeps responding while a slow
        # full trace assembles (a monolithic dispatch would occupy the GPU
        # for its whole duration; Apple GPUs don't preempt mid-dispatch).
        row0 = 0
        while row0 < ctx.height
            rows = min(band_rows, ctx.height - row0)
            _launch_trace!(ctx, ctx.out_gpu, ctx.cam_params,
                           ctx.spacetime_params, ctx.width, ctx.height,
                           nmax, dt, 0.5f0, 0.5f0, 1.0f0, row0, rows;
                           kerr=is_kerr, order=order, tol=tol)
            row0 += rows
            row0 < ctx.height && on_band()
        end
    end
    return nothing
end

"""
Compile-once launch of `trace_kernel_mtl!`. `cam_params`/`spacetime_params`
are passed explicitly so a draft render can use its own buffers and run
concurrently with preview frames that update the context's buffers.
"""
function _launch_trace!(ctx::MetalPreviewContext, out, cam_params,
                        spacetime_params, width::Int, height::Int,
                        nmax::Int, dt::Float32, ju::Float32, jv::Float32,
                        weight::Float32, row0::Int, rows::Int; nb::Int=0,
                        fan=nothing, sky_params=nothing, layer::Bool=false,
                        substride::Int=1, subx::Int=0, suby::Int=0,
                        ring_rmin::Real=0.0, col0::Int=0, cols::Int=-1,
                        order::Int=4, tol::Real=1.0f-4,
                        kerr::Bool=false, bake::Bool=false,
                        stat=nothing, jw::Real=0.0, jseed::Integer=0)
    cols < 0 && (cols = width)
    von = ctx.vol_on[]
    fan_b = fan === nothing ? _dummy_fan() : fan
    skyp_b = sky_params === nothing ? _dummy_skyp() : sky_params
    stat_on = stat !== nothing
    stat_b = stat_on ? stat : _dummy_stat()
    kernels = ctx.kernel[]::Dict{Any,Any}
    key = (von, nb, layer, order, kerr, bake, stat_on)
    if !haskey(kernels, key)
        kernels[key] = @metal launch=false trace_kernel_mtl!(
            out, ctx.bg_gpu, ctx.bb_lut, ctx.star_lut, ctx.vol_gpu,
            ctx.vol_params,
            ctx.star_params, cam_params, spacetime_params, ctx.disc_params,
            fan_b, skyp_b, stat_b,
            width, height, nmax, dt, Float32(tol), ju, jv,
            Float32(jw), Float32(jseed), weight, row0, rows,
            col0, cols, substride, subx, suby, Float32(ring_rmin),
            Val(von), Val(nb), Val(layer), Val(order), Val(kerr), Val(bake),
            Val(stat_on))
    end
    kernel = kernels[key]
    n = cols * rows
    threads = min(kernel.pipeline.maxTotalThreadsPerThreadgroup, n)
    groups = cld(n, threads)
    kernel(out, ctx.bg_gpu, ctx.bb_lut, ctx.star_lut, ctx.vol_gpu,
           ctx.vol_params,
           ctx.star_params, cam_params, spacetime_params, ctx.disc_params,
           fan_b, skyp_b, stat_b,
           width, height, nmax, dt, Float32(tol), ju, jv,
           Float32(jw), Float32(jseed), weight, row0, rows,
           col0, cols, substride, subx, suby, Float32(ring_rmin),
           Val(von), Val(nb), Val(layer), Val(order), Val(kerr), Val(bake),
           Val(stat_on);
           threads=threads, groups=groups)
    return nothing
end

"""
    render_depth_mtl(ctx, cam, spacetime; width, height, samples=2, dt=0.02,
                     nbuckets=10, fisheye_deg=0.0, relativistic=false)

Pinhole draft render with emission separated into `nbuckets` path-length
buckets (log-spaced over 1.5M–120M; the last bucket holds the escaped
background at infinity). Returns a `(3*nbuckets, width, height)`
`Array{Float32,3}` — feed to [`lens_post`](@ref) for depth-of-field as a post
operation at any aperture/focus, without re-rendering.
"""
function render_depth_mtl(ctx::MetalPreviewContext, cam::Camera,
                          spacetime::AbstractSpacetime;
                          width::Int=1920, height::Int=1080,
                          samples::Int=2, dt::Real=0.02,
                          nbuckets::Int=10, fisheye_deg::Real=0.0,
                          rng::Random.AbstractRNG=Random.default_rng(),
                          relativistic::Bool=false)
    nbuckets >= 3 || throw(ArgumentError("nbuckets must be ≥ 3"))
    dt32 = Float32(dt)
    M = Float32(spacetime.M)
    r_band = Float32(2.05 * spacetime.M)
    r_escape = Float32(ctx.r_escape_factor * max(norm(cam.pos),
                                                 15.0 * spacetime.M))
    nmax = min(max(ctx.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt32)),
               40_000)
    cam_params = MtlArray{Float32,1,Metal.SharedStorage}(undef, CAM_PARAMS_N)
    spacetime_params = MtlArray{Float32,1,Metal.SharedStorage}(undef, 8)
    cam_host = unsafe_wrap(Array, cam_params); fill!(cam_host, 0.0f0)
    st_host = unsafe_wrap(Array, spacetime_params); fill!(st_host, 0.0f0)
    copyto!(cam_params, _ks_cam_params(cam, spacetime.M;
                                       fisheye_deg=fisheye_deg,
                                       a=spin(spacetime)))
    spin_a, rkill2 = _spin_horizon(spacetime)
    is_kerr = spin_a != 0
    copyto!(spacetime_params,
            Float32[M, r_band, r_escape, relativistic ? 1.0 : 0.0,
                    HSTEP_COEF[], HSTEP_CAP[], spin_a, rkill2])
    out = MtlArray{Float32,3}(undef, 3 * nbuckets, width, height)
    fill!(out, 0.0f0)
    rows_per_tile = clamp(ceil(Int, 2.0e9 / (width * nmax)), 16, height)
    ntiles = cld(height, rows_per_tile)
    weight = Float32(1.0 / samples^2)
    offsets = jittered_grid(samples; rng=rng)
    for off in offsets, tile in 0:(ntiles - 1)
        row0 = tile * rows_per_tile
        rows = min(rows_per_tile, height - row0)
        _launch_trace!(ctx, out, cam_params, spacetime_params, width, height,
                       nmax, dt32, Float32(off[1]), Float32(off[2]), weight,
                       row0, rows; nb=nbuckets, kerr=is_kerr)
    end
    Metal.synchronize()
    return Array(out)
end

"""Copy a `(3, W, H)` GPU buffer back as a `Matrix{RGBf}`."""
function _download_rgb(out_gpu, width::Int, height::Int)
    out_cpu = Array(out_gpu)
    img = Matrix{RGBf}(undef, width, height)
    for j in 1:height, i in 1:width
        img[i, j] = RGBf(out_cpu[1, i, j], out_cpu[2, i, j], out_cpu[3, i, j])
    end
    return img
end

"""
    render_draft_mtl(ctx::MetalPreviewContext, cam, spacetime;
                     width=1920, height=1080, samples=2, dt=0.02,
                     rng=Random.default_rng(), progress=nothing)

High-quality GPU draft render: full output resolution on the Metal kernel with
a tightened integration step (default `dt=0.02` vs the preview's 0.1),
`samples²` stratified-jittered rays per pixel, and interpolated disc
crossings. Float32 and fixed-step, so the photon ring and shadow edge are a
touch softer than the CPU `render` — think "90% of the final look in a
fraction of the time" for iterating on composition and exposure.

The frame is split into row tiles so each GPU dispatch stays short; `progress`
(if given) receives the completed fraction after every dispatch. Reuses the
context's background/LUT/parameter buffers, so call it with the same `ctx` as
the live preview. Thin-lens cameras fall back to pinhole optics (no DoF).

Motion blur: pass `camera_at`, a function of the shutter fraction `s ∈ [0, 1)`
returning a `Camera`. Each of the `samples²` supersampling passes then renders
from its own stratified shutter time — the passes double as the temporal
samples, exactly as they double as the aperture samples for DoF, so the blur
costs nothing extra. The positional `cam` still sets the escape radius and is
the nominal (shutter-centre) pose.

`camera_at` has the same signature here as in [`render_motion`](@ref), so one
motion path drives either renderer. It used to return `(Camera, beta)` on the
GPU and a bare `Camera` on the CPU; velocity now lives on the camera itself.

Adaptive refinement: `refine=n` (n > 1) renders the base `samples²` passes
while accumulating per-pixel luminance variance and the fraction of rays that
graze the Kerr photon shell (from the Bardeen launch quantities), then
re-renders the flagged region's bounding box `n−1` more times with fresh
strata — flagged pixels get `n×` the rays. Aimed at the wound-image bands of
a spinning hole, whose spatial frequency is unbounded and starves uniform
sampling. `refine_var` is the relative per-ray σ threshold, `refine_graze`
the graze-fraction threshold, `refine_margin` the box margin in pixels.
Sub-pixel jitter is decorrelated per pixel by default (`pixel_jitter=true`),
turning the moiré such content aliases into during uniform passes into noise
that averages away; `false` restores the legacy pass-wide jitter point.
"""
function render_draft_mtl(ctx::MetalPreviewContext, cam::Camera,
                          spacetime::AbstractSpacetime;
                          width::Int=1920, height::Int=1080,
                          samples::Int=2, dt::Real=0.02,
                          rng::Random.AbstractRNG=Random.default_rng(),
                          progress::Union{Function,Nothing}=nothing,
                          fisheye_deg::Real=0.0,
                          aperture_world::Real=0.0, focus_dist::Real=1.0,
                          relativistic::Bool=false,
                          camera_at::Union{Function,Nothing}=nothing,
                          per_pixel_shutter::Bool=false,
                          refine::Int=1, refine_var::Real=0.35,
                          refine_graze::Real=0.05, refine_margin::Int=8,
                          pixel_jitter::Bool=true,
                          order::Int=4, tol::Real=1.0f-4)
    dt32 = Float32(dt)
    M = Float32(spacetime.M)
    r_band = Float32(2.05 * spacetime.M)
    r_escape = Float32(ctx.r_escape_factor * max(norm(cam.pos),
                                                 15.0 * spacetime.M))
    nmax = min(max(ctx.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt32)),
               40_000)

    # Own parameter buffers: preview frames may update ctx's buffers while the
    # draft's tiles are still dispatching.
    cam_params = MtlArray{Float32,1,Metal.SharedStorage}(undef, CAM_PARAMS_N)
    spacetime_params = MtlArray{Float32,1,Metal.SharedStorage}(undef, 8)
    cam_host = unsafe_wrap(Array, cam_params); fill!(cam_host, 0.0f0)
    st_host = unsafe_wrap(Array, spacetime_params); fill!(st_host, 0.0f0)
    base_params = _ks_cam_params(cam, spacetime.M; fisheye_deg=fisheye_deg,
                                 focus_dist=focus_dist, a=spin(spacetime))
    copyto!(cam_params, base_params)
    spin_a, rkill2 = _spin_horizon(spacetime)
    is_kerr = spin_a != 0
    copyto!(spacetime_params,
            Float32[M, r_band, r_escape, relativistic ? 1.0 : 0.0,
                    HSTEP_COEF[], HSTEP_CAP[], spin_a, rkill2])
    # Depth of field: each supersampling pass gets its own aperture sample,
    # so `samples²` passes double as the bokeh samples. Fisheye stays pinhole.
    use_dof = aperture_world > 0.0 && fisheye_deg <= 0.0

    out = MtlArray{Float32,3}(undef, 3, width, height)
    fill!(out, 0.0f0)

    # Row-tile size chosen so a single dispatch stays well under a second on
    # an Apple-silicon GPU (~3e9 ray-steps/s), keeping the UI and the watchdog
    # happy during multi-second drafts.
    rows_per_tile = clamp(ceil(Int, 2.0e9 / (width * nmax)), 16, height)
    ntiles = cld(height, rows_per_tile)

    weight = Float32(1.0 / samples^2)
    # Sub-pixel strata. With `pixel_jitter` each pass carries its stratum
    # ORIGIN and width to the kernel and every pixel hashes its own point
    # inside it — phase-decorrelated across pixels, so content finer than a
    # pixel (wound-image stacks near a spinning hole) averages to noise
    # instead of aliasing into moiré rings. `pixel_jitter=false` restores the
    # legacy pass-wide jittered point.
    jw = pixel_jitter ? Float32(1.0 / samples) : 0.0f0
    offsets = pixel_jitter ?
        Random.shuffle!(rng, vec([(Float32(a / samples), Float32(b / samples))
                                  for a in 0:(samples - 1), b in 0:(samples - 1)])) :
        jittered_grid(samples; rng=rng)
    ndispatch = length(offsets) * ntiles * max(refine, 1)
    done = 0
    # Adaptive refinement (refine > 1): the base passes also accumulate
    # per-pixel luminance moments and the Bardeen graze fraction into `stat`;
    # afterwards the flagged region re-renders with (refine−1) more blocks of
    # samples² passes and the sums renormalise by 1/refine. No interpolation
    # anywhere — flagged pixels just get refine× the rays.
    stat = nothing
    if refine > 1
        stat = MtlArray{Float32,3}(undef, 3, width, height)
        fill!(stat, 0.0f0)
    end
    # Depth of field: each pass owns one aperture stratum (shuffled so lens
    # strata pair randomly with the pixel-jitter strata) and every pixel
    # hashes its own point inside it — see the kernel's stratified-lens block.
    lens_perm = Random.randperm(rng, samples^2)
    # Motion blur: each pass owns one shutter *stratum*, permuted independently
    # so time strata pair randomly with the others. By default the pass renders
    # from the stratum's midpoint — midpoint quadrature, which converges fastest
    # on smooth motion. With `per_pixel_shutter` both ends go to the GPU and
    # each pixel picks its own instant between them; see the measured trade in
    # the kernel's per-pixel-shutter block.
    time_perm = Random.randperm(rng, samples^2)
    # One block = samples² passes over a row/col rectangle. The base render is
    # one full-frame block; each refinement block re-runs the flagged
    # rectangle with freshly drawn pixel/lens/time strata. `pass` stays
    # globally unique so every per-pixel hash seed differs across blocks.
    pass_base = Ref(0)
    run_block! = function (row_a::Int, nrows_blk::Int, col_a::Int,
                           ncols_blk::Int, stat_blk)
        rpt = clamp(ceil(Int, 2.0e9 / (max(ncols_blk, 1) * nmax)), 16,
                    max(nrows_blk, 16))
        ntiles_blk = cld(nrows_blk, rpt)
        for (p, (du, dv)) in enumerate(offsets)
            pass = pass_base[] + p
            if camera_at !== nothing
                m = time_perm[p] - 1
                if per_pixel_shutter
                    base_params = _ks_cam_params(camera_at(m / samples^2), spacetime.M;
                                                 fisheye_deg=fisheye_deg,
                                                 focus_dist=focus_dist,
                                                 a=spin(spacetime))
                    endp = _ks_cam_params(camera_at((m + 1) / samples^2), spacetime.M;
                                          fisheye_deg=fisheye_deg,
                                          focus_dist=focus_dist,
                                          a=spin(spacetime))
                    # Seed kept clear of the lens hashes (which use `pass` and
                    # `pass + 7919`), so time and aperture decorrelate per pixel.
                    base_params[29] = Float32(104729 + pass)
                    base_params[30:(29 + CAM_POSE_N)] .= @view endp[1:CAM_POSE_N]
                else
                    base_params = _ks_cam_params(camera_at((m + 0.5) / samples^2),
                                                 spacetime.M;
                                                 fisheye_deg=fisheye_deg,
                                                 focus_dist=focus_dist,
                                                 a=spin(spacetime))
                end
            end
            if use_dof
                m = lens_perm[p] - 1
                base_params[23] = Float32((m ÷ samples) / samples)
                base_params[24] = Float32((m % samples) / samples)
                base_params[26] = Float32(1.0 / samples)
                base_params[27] = Float32(aperture_world / 2.0)
                base_params[28] = Float32(pass)
            end
            (use_dof || camera_at !== nothing) && copyto!(cam_params, base_params)
            for t in 0:(ntiles_blk - 1)
                row0 = row_a + t * rpt
                rows = min(rpt, row_a + nrows_blk - row0)
                _launch_trace!(ctx, out, cam_params, spacetime_params,
                               width, height, nmax, dt32,
                               Float32(du), Float32(dv), weight, row0, rows;
                               kerr=is_kerr, col0=col_a, cols=ncols_blk,
                               order=order, tol=tol,
                               stat=stat_blk, jw=jw,
                               jseed=15485863 + pass)
                Metal.synchronize()
                done += 1
                isnothing(progress) || progress(min(done / ndispatch, 1.0))
            end
        end
        pass_base[] += length(offsets)
        return nothing
    end

    run_block!(0, height, 0, width, stat)

    # Refinement: flag pixels by per-ray luminance variance (relative to a
    # luminance floor) or by the Bardeen graze fraction, take the flagged
    # bounding box with a margin, and re-render it (refine−1) more times.
    # Every pixel in the box then holds refine× the weight and renormalises
    # after download — more honest rays, no interpolation, no reweighting of
    # neighbours.
    box = nothing
    if refine > 1 && stat !== nothing
        s_h = Array(stat)
        imin, imax, jmin, jmax = width + 1, 0, height + 1, 0
        @inbounds for jj in 1:height, ii in 1:width
            m1 = s_h[1, ii, jj]
            σ2 = s_h[2, ii, jj] - m1 * m1
            fl = (σ2 > 0.0f0 &&
                  sqrt(σ2) > refine_var * (m1 + 0.01f0)) ||
                 s_h[3, ii, jj] > refine_graze
            if fl
                imin = min(imin, ii); imax = max(imax, ii)
                jmin = min(jmin, jj); jmax = max(jmax, jj)
            end
        end
        if imax > 0
            c_a = max(imin - refine_margin, 1) - 1
            ncols_r = min(imax + refine_margin, width) - c_a
            r_a = max(jmin - refine_margin, 1) - 1
            nrows_r = min(jmax + refine_margin, height) - r_a
            for _ in 2:refine
                offsets = pixel_jitter ?
                    Random.shuffle!(rng,
                        vec([(Float32(a / samples), Float32(b / samples))
                             for a in 0:(samples - 1), b in 0:(samples - 1)])) :
                    jittered_grid(samples; rng=rng)
                lens_perm = Random.randperm(rng, samples^2)
                time_perm = Random.randperm(rng, samples^2)
                run_block!(r_a, nrows_r, c_a, ncols_r, nothing)
            end
            box = (c_a, ncols_r, r_a, nrows_r)
        end
    end

    img = _download_rgb(out, width, height)
    if box !== nothing
        c_a, ncols_r, r_a, nrows_r = box
        s = 1.0f0 / Float32(refine)
        @inbounds for jj in (r_a + 1):(r_a + nrows_r),
                      ii in (c_a + 1):(c_a + ncols_r)
            c = img[ii, jj]
            img[ii, jj] = RGBf(c.r * s, c.g * s, c.b * s)
        end
    end
    return img
end

function render_draft_mtl(ctx::MetalPreviewContext, cam::ThinLensCamera,
                          spacetime::AbstractSpacetime; kwargs...)
    fov = (cam.sensor_width / 2.0) / cam.focal_length
    pinhole = Camera(cam.pos, cam.pos + cam.fwd, cam.up_local, fov)
    # Same world-space aperture as the CPU get_ray: diameter = focus/f_number.
    ap = cam.aperture * cam.focus_distance / cam.focal_length
    return render_draft_mtl(ctx, pinhole, spacetime; kwargs...,
                            aperture_world=ap, focus_dist=cam.focus_distance)
end

function render_preview_mtl(ctx::MetalPreviewContext, cam::ThinLensCamera,
                            spacetime::AbstractSpacetime)
    fov = (cam.sensor_width / 2.0) / cam.focal_length
    pinhole = Camera(cam.pos, cam.pos + cam.fwd, cam.up_local, fov)
    return render_preview_mtl(ctx, pinhole, spacetime)
end

function render_preview_mtl!(img::Matrix{RGBf}, host::Array{Float32,3},
                             ctx::MetalPreviewContext, cam::ThinLensCamera,
                             spacetime::AbstractSpacetime; kwargs...)
    fov = (cam.sensor_width / 2.0) / cam.focal_length
    pinhole = Camera(cam.pos, cam.pos + cam.fwd, cam.up_local, fov)
    return render_preview_mtl!(img, host, ctx, pinhole, spacetime; kwargs...)
end

# ---------------------------------------------------------------------------
# Asynchronous reprojection ("timewarp")
# ---------------------------------------------------------------------------

# One compiled warp pipeline serves every context (argument types are fixed).
const _WARP_KERNEL = Ref{Any}(nothing)

"""
    warp_kernel_mtl!(out, prev, wp, width, height)

Rotation-only reprojection of a previously traced frame: each output pixel's
view direction (in the *new* camera basis) is expressed in the *previous*
camera basis and mapped back through the lens to a source pixel, which is
bilinearly sampled. Exact for pure rotation — turning the camera re-aims
rays without creating information — and a one-frame approximation under
translation, corrected by the next full trace.

`wp` (22 floats): new fwd/right/up (9), prev fwd/right/up (9), fov,
projection mode (>0.5 = fisheye), θ_edge (rad), pad.
"""
function warp_kernel_mtl!(out, prev, wp, width, height, ::Val{C}) where {C}
    idx = thread_position_in_grid().x
    idx > width * height && return
    j = (idx - 1) ÷ width + 1
    i = (idx - 1) % width + 1

    half_h = Float32(height) / 2.0f0
    u = (Float32(i) - 0.5f0 - Float32(width) / 2.0f0) / half_h
    v = (Float32(j) - 0.5f0 - Float32(height) / 2.0f0) / half_h

    fov = wp[19]
    fe = wp[20] > 0.5f0
    θe = wp[21]

    # Pixel direction in the new camera basis.
    cr = 0.0f0; cu = 0.0f0; cf = 1.0f0
    if fe
        ρ = sqrt(u * u + v * v)
        θp = ρ * θe
        sθ = sin(θp)
        inv_ρ = ρ > 1.0f-8 ? 1.0f0 / ρ : 0.0f0
        cr = sθ * u * inv_ρ
        cu = sθ * v * inv_ρ
        cf = cos(θp)
    else
        dx = u * fov; dy = v * fov
        ν = sqrt(dx * dx + dy * dy + 1.0f0)
        cr = dx / ν; cu = dy / ν; cf = 1.0f0 / ν
    end
    dx_w = cf * wp[1] + cr * wp[4] + cu * wp[7]
    dy_w = cf * wp[2] + cr * wp[5] + cu * wp[8]
    dz_w = cf * wp[3] + cr * wp[6] + cu * wp[9]

    # Same direction in the previous camera basis.
    a = dx_w * wp[10] + dy_w * wp[11] + dz_w * wp[12]   # · fwd_prev
    b = dx_w * wp[13] + dy_w * wp[14] + dz_w * wp[15]   # · right_prev
    c = dx_w * wp[16] + dy_w * wp[17] + dz_w * wp[18]   # · up_prev

    uo = 0.0f0; vo = 0.0f0; valid = true
    if fe
        θ = acos(clamp(a, -1.0f0, 1.0f0))
        s = sqrt(b * b + c * c)
        if s < 1.0f-6
            uo = 0.0f0; vo = 0.0f0
            valid = θ < θe
        else
            ρo = θ / θe
            uo = ρo * b / s
            vo = ρo * c / s
        end
    else
        if a < 0.02f0
            valid = false
        else
            uo = (b / a) / fov
            vo = (c / a) / fov
        end
    end

    x = uo * half_h + Float32(width) / 2.0f0 + 0.5f0
    y = vo * half_h + Float32(height) / 2.0f0 + 0.5f0
    if !valid || x < 1.0f0 || x > Float32(width) || y < 1.0f0 || y > Float32(height)
        out[1, i, j] = 0.0f0; out[2, i, j] = 0.0f0; out[3, i, j] = 0.0f0
        # 4-channel (gas layer): revealed pixels are transparent, not black —
        # the sky pass owns whatever is behind them.
        C == 4 && (out[4, i, j] = 1.0f0)
        return
    end
    x0 = clamp(unsafe_trunc(Int32, floor(x)), Int32(1), Int32(width - 1))
    y0 = clamp(unsafe_trunc(Int32, floor(y)), Int32(1), Int32(height - 1))
    tx = x - Float32(x0); ty = y - Float32(y0)
    x1 = x0 + Int32(1); y1 = y0 + Int32(1)
    w00 = (1.0f0 - tx) * (1.0f0 - ty); w10 = tx * (1.0f0 - ty)
    w01 = (1.0f0 - tx) * ty;           w11 = tx * ty
    for c in 1:C
        out[c, i, j] = prev[c, x0, y0] * w00 + prev[c, x1, y0] * w10 +
                       prev[c, x0, y1] * w01 + prev[c, x1, y1] * w11
    end
    return
end

"""
    warp_preview_mtl!(img, host, ctx, warp_out, prev, warp_params,
                      cam, prev_cam; fisheye_deg=0.0)

Emit one reprojected frame: warps `prev` (a retained copy of the last traced
`out_gpu`) from `prev_cam`'s orientation to `cam`'s, downloading into the
caller-owned `img`/`host` like `render_preview_mtl!`. `warp_out` is a
`(3, width, height)` MtlArray and `warp_params` a 22-float MtlVector, both
caller-retained.
"""
function warp_preview_mtl!(img::Matrix{RGBf}, host::Array{Float32,3},
                           ctx::MetalPreviewContext, warp_out, prev,
                           warp_params, cam::Camera, prev_cam::Camera;
                           fisheye_deg::Real=0.0)
    _warp_gpu!(ctx, warp_out, prev, warp_params, cam, prev_cam;
               fisheye_deg=fisheye_deg)
    copyto!(host, warp_out)
    @inbounds for j in 1:ctx.height, i in 1:ctx.width
        img[i, j] = RGBf(host[1, i, j], host[2, i, j], host[3, i, j])
    end
    return img
end

# ---------------------------------------------------------------------------
# Layered real-time engine: exact sky fan + disc/gas layer + composite
# ---------------------------------------------------------------------------
#
# By spherical symmetry, the entire lensed sky seen by the static observer at
# radius r is a one-dimensional function of the local angle ψ between the ray
# and the outward radial axis. Each frame we integrate an exact fan of
# geodesics over ψ ∈ [0, π] (a few thousand rays — one column's worth of
# work) and every display pixel becomes a table lookup plus one starmap
# sample: native-resolution, per-frame-exact lensing, shadow included. Only
# the disc/gas — which breaks the symmetry — still pays for per-pixel
# integration, rendered as a separate premultiplied layer (usually at lower
# resolution) and composited over the sky.
#
# The fan is indexed by the *local* angle measured in the static tetrad
# (cos ψ = p·ê_r for unit-frequency rays), which matches the pixel rays as
# long as the camera tetrad is the unboosted static observer — the current
# viewport convention.

"""
    sky_fan_kernel!(fan, cam_params, sky_params, nmax, dt)

One thread per fan entry k: integrate the exact null geodesic launched from
the camera radius at local angle `ψ_k = π(k−1)/(N−1)` from the outward radial
axis, in the z = 0 plane from position `(r, 0, 0)` with lateral direction
`+ŷ`. Writes `fan[:, k] = (escaped, cos θf, sin θf, r_min)` where `θf` is the
asymptotic escape direction's in-plane angle from the radial axis and `r_min`
the closest approach. `cam_params` is the 28-float block for the fan camera
(`fwd = x̂`, `right = ŷ`); `sky_params[5:6] = (M, r_escape)`.
"""
function sky_fan_kernel!(fan, cam_params, sky_params, nmax, dt, band)
    k = thread_position_in_grid().x
    N = size(fan, 2)
    k > N && return
    M = sky_params[5]
    r_escape = sky_params[6]

    # band > 0: refinement fan across the critical-angle band found by
    # `fan_band_kernel!` (sky_params[9:10]) — the escape/capture transition
    # where dθf/dψ diverges and the coarse fan under-resolves the sky.
    ψ = band > Int32(0) ?
        sky_params[9] + (sky_params[10] - sky_params[9]) *
                        (Float32(k) - 1.0f0) / (Float32(N) - 1.0f0) :
        Float32(pi) * (Float32(k) - 1.0f0) / (Float32(N) - 1.0f0)
    cf = cos(ψ)
    cr = sin(ψ)

    ef0 = cam_params[5];  ef1 = cam_params[6];  ef2 = cam_params[7];  ef3 = cam_params[8]
    er0 = cam_params[9];  er1 = cam_params[10]; er2 = cam_params[11]; er3 = cam_params[12]
    ut0 = cam_params[17]; ut1 = cam_params[18]; ut2 = cam_params[19]; ut3 = cam_params[20]

    x = cam_params[1]; y = cam_params[2]; z = cam_params[3]
    r = sqrt(x * x + y * y + z * z)
    f = 2.0f0 * M / r
    qt = cf * ef0 + cr * er0 - ut0
    qx = cf * ef1 + cr * er1 - ut1
    qy = cf * ef2 + cr * er2 - ut2
    qz = cf * ef3 + cr * er3 - ut3
    lq = qt + (x * qx + y * qy + z * qz) / r
    p_t = -qt + f * lq
    flr = f * lq / r
    px = qx + flr * x
    py = qy + flr * y
    pz = qz + flr * z

    r_min = r
    hit_horizon = false
    r_floor = r > 2.05f0 * M ? 2.0f0 * M : 0.3f0 * M
    r_prev = -1.0f0
    for _ in 1:nmax
        r = sqrt(x * x + y * y + z * z)
        r < r_min && (r_min = r)
        if r < 3.2f0 * M
            if r < r_floor ||
               (r < 2.95f0 * M && r_prev > 0.0f0 && r < r_prev - 1.0f-4 * M)
                hit_horizon = true
                break
            end
            r_prev = r
        end
        r > r_escape && break
        h = dt * min(max(0.16f0 * r / M, 1.0f0), 8.0f0)
        k1 = ks_rhs_mtl(x, y, z, px, py, pz, p_t, M)
        k2 = ks_rhs_mtl(
            x + 0.5f0 * h * k1[1], y + 0.5f0 * h * k1[2], z + 0.5f0 * h * k1[3],
            px + 0.5f0 * h * k1[4], py + 0.5f0 * h * k1[5], pz + 0.5f0 * h * k1[6],
            p_t, M)
        k3 = ks_rhs_mtl(
            x + 0.5f0 * h * k2[1], y + 0.5f0 * h * k2[2], z + 0.5f0 * h * k2[3],
            px + 0.5f0 * h * k2[4], py + 0.5f0 * h * k2[5], pz + 0.5f0 * h * k2[6],
            p_t, M)
        k4 = ks_rhs_mtl(
            x + h * k3[1], y + h * k3[2], z + h * k3[3],
            px + h * k3[4], py + h * k3[5], pz + h * k3[6],
            p_t, M)
        x  += (h / 6.0f0) * (k1[1] + 2.0f0 * k2[1] + 2.0f0 * k3[1] + k4[1])
        y  += (h / 6.0f0) * (k1[2] + 2.0f0 * k2[2] + 2.0f0 * k3[2] + k4[2])
        z  += (h / 6.0f0) * (k1[3] + 2.0f0 * k2[3] + 2.0f0 * k3[3] + k4[3])
        px += (h / 6.0f0) * (k1[4] + 2.0f0 * k2[4] + 2.0f0 * k3[4] + k4[4])
        py += (h / 6.0f0) * (k1[5] + 2.0f0 * k2[5] + 2.0f0 * k3[5] + k4[5])
        pz += (h / 6.0f0) * (k1[6] + 2.0f0 * k2[6] + 2.0f0 * k3[6] + k4[6])
        if !(x == x) || !(px == px)
            hit_horizon = true
            break
        end
    end

    rf = max(sqrt(x * x + y * y + z * z), 1.0f-6)
    if hit_horizon || rf < 4.0f0 * M
        fan[1, k] = 0.0f0
        fan[2, k] = 1.0f0
        fan[3, k] = 0.0f0
    else
        inv_rf = 1.0f0 / rf
        ff = 2.0f0 * M * inv_rf
        κf = (x * px + y * py + z * pz) * inv_rf
        c1f = ff * (-p_t + κf) * inv_rf
        vx = px - c1f * x
        vy = py - c1f * y
        vl = max(sqrt(vx * vx + vy * vy), 1.0f-20)
        fan[1, k] = 1.0f0
        fan[2, k] = vx / vl
        fan[3, k] = vy / vl
    end
    fan[4, k] = r_min
    return nothing
end

"""
Fraction of `ring_rmin` over which the ring layer is at full weight before it
feathers back into the bulk layer. The two layers are traced at different
resolutions, so they never agree exactly; switching between them abruptly
would replace a stair-stepped ring with a seam ring.
"""
# Affine step rule `h = dt * clamp(COEF*r/M, 1, CAP)`, exposed so the shape of
# the heuristic can be swept and measured rather than guessed at.
const HSTEP_COEF = Ref(0.16f0)
const HSTEP_CAP  = Ref(8.0f0)

const RING_FEATHER = 0.8f0

"""
Safety factor on the impact-parameter bound used to size the ring dispatch.
The box only has to *contain* the wound region; clipping it would eat the
photon ring, so the bound is deliberately loose — the cost of 6% extra
radius is a few percent of a box that is itself a small fraction of the frame.
"""
const RING_B_MARGIN = 1.06

"""
    _wound_impact_parameter(M, ring_rmin)

Impact parameter below which a null geodesic's periapsis falls under
`ring_rmin` — the ring pass's selector, expressed as a conserved quantity.

Periapsis `r_p` and impact parameter `b` satisfy `b = r_p / sqrt(1 - 2M/r_p)`,
whose right-hand side is *increasing* for `r_p > 3M` and has its minimum
`b_c = 3√3 M` at the photon sphere. So for `ring_rmin > 3M` the condition
`r_p < ring_rmin` is exactly `b < b(ring_rmin)`: captured rays (`b < b_c`,
no turning point at all) sit below the bound too, which is what we want —
they carry the shadow and its edge.
"""
function _wound_impact_parameter(M::Real, ring_rmin::Real)
    R = Float64(ring_rmin)
    R > 2 * M || return Inf
    return R / sqrt(1 - 2 * M / R)
end

"""
    _ring_screen_box(cam, M, ring_rmin, fisheye_deg, rw, rh; probes, margin)

Screen-space bounding box, in the ring buffer's own `rw × rh` pixel grid, of
the rays the ring pass keeps. Returns `(col0, cols, row0, rows)`, or `nothing`
meaning "no useful bound — dispatch the whole frame".

The wound set is `b < b_R` ([`_wound_impact_parameter`](@ref)), a *filled disc*
on screen around the hole rather than a thin annulus, because captured rays
are wound too. Its bound is found by evaluating the real ray construction —
the same tetrad, the same rectilinear/fisheye mapping, the same index-lowering
as `trace_kernel_mtl!` — on a coarse grid of probe pixels, then taking the box
of the probes that pass. Going through the actual construction rather than a
closed form in `r` is what makes this correct under camera roll, fisheye, and
relativistic aberration: all three live in the tetrad and the mapping, and none
of them survive a static-observer formula.

`b = L/E` with `E = -p_t` and `L = |x × p|`, both conserved in a spherically
symmetric spacetime, so a single evaluation at the camera decides the whole
geodesic without integrating it.
"""
function _ring_screen_box(cam::Camera, M::Real, ring_rmin::Real,
                          fisheye_deg::Real, rw::Int, rh::Int;
                          probes::Int=81, margin::Int=3)
    bR = _wound_impact_parameter(M, ring_rmin) * RING_B_MARGIN
    isfinite(bR) || return nothing
    Mf = Float64(M)
    u4, Ef, Er, Eu = ks_camera_tetrad(cam.pos, cam.fwd, cam.right,
                                      cam.up_local, Mf; beta=camera_beta(cam))
    fov = Float64(cam.fov_factor)
    fish = fisheye_deg > 0
    θmax = deg2rad(max(Float64(fisheye_deg), 0.0))
    x, y, z = cam.pos
    r = sqrt(x * x + y * y + z * z)
    r > 2 * Mf || return nothing
    f = 2 * Mf / r

    # Probe grid over the ring buffer, matching the kernel's pixel-centre
    # convention (`ju = jv = 0.5`).
    nv = max(9, probes)
    nu = max(9, round(Int, probes * rw / rh))
    half_h = rh / 2
    imin, imax = typemax(Int), typemin(Int)
    jmin, jmax = typemax(Int), typemin(Int)
    bbest, ibest, jbest = Inf, 1, 1
    for pj in 1:nv, pi in 1:nu
        # Probe pixel centre, mapped to the kernel's (u, v).
        ipx = 1 + (pi - 1) * (rw - 1) / (nu - 1)
        jpx = 1 + (pj - 1) * (rh - 1) / (nv - 1)
        u = (ipx - 1 + 0.5 - rw / 2) / half_h
        v = (jpx - 1 + 0.5 - rh / 2) / half_h
        if fish
            ρ = sqrt(u * u + v * v)
            θp = ρ * θmax
            sθ = sin(θp)
            invρ = ρ > 1e-8 ? 1 / ρ : 0.0
            cr, cu, cf = sθ * u * invρ, sθ * v * invρ, cos(θp)
        else
            dxl, dyl = u * fov, v * fov
            ν = sqrt(dxl * dxl + dyl * dyl + 1)
            cr, cu, cf = dxl / ν, dyl / ν, 1 / ν
        end
        qt = cf * Ef[1] + cr * Er[1] + cu * Eu[1] - u4[1]
        qx = cf * Ef[2] + cr * Er[2] + cu * Eu[2] - u4[2]
        qy = cf * Ef[3] + cr * Er[3] + cu * Eu[3] - u4[3]
        qz = cf * Ef[4] + cr * Er[4] + cu * Eu[4] - u4[4]
        lq = qt + (x * qx + y * qy + z * qz) / r
        p_t = -qt + f * lq
        flr = f * lq / r
        px = qx + flr * x
        py = qy + flr * y
        pz = qz + flr * z
        E = abs(p_t)
        E > 1e-12 || continue
        Lx = y * pz - z * py
        Ly = z * px - x * pz
        Lz = x * py - y * px
        b = sqrt(Lx * Lx + Ly * Ly + Lz * Lz) / E
        if b < bbest
            bbest, ibest, jbest = b, round(Int, ipx), round(Int, jpx)
        end
        if b < bR
            ip, jp = round(Int, ipx), round(Int, jpx)
            imin = min(imin, ip); imax = max(imax, ip)
            jmin = min(jmin, jp); jmax = max(jmax, jp)
        end
    end
    # A wound region smaller than the probe spacing lands between probes. The
    # nearest-approach probe is then within one spacing of it, so seeding the
    # box there and padding by two spacings still contains it.
    if imin > imax
        imin = imax = ibest
        jmin = jmax = jbest
    end
    du = 2 * ceil(Int, (rw - 1) / (nu - 1)) + margin
    dv = 2 * ceil(Int, (rh - 1) / (nv - 1)) + margin
    c0 = clamp(imin - du, 1, rw) - 1
    c1 = clamp(imax + du, 1, rw)
    r0 = clamp(jmin - dv, 1, rh) - 1
    r1 = clamp(jmax + dv, 1, rh)
    cols, rows = c1 - c0, r1 - r0
    # Below roughly r = 6M the wound cone swallows the sky and the box is the
    # frame; skip the bookkeeping and let the pass run whole.
    (cols * rows) > 0.55 * rw * rh && return nothing
    return (c0, cols, r0, rows)
end


"""Bilinear tap of a premultiplied `(4, lw, lh)` layer at display pixel `i, j`."""
@inline function _layer_bilinear(layer, i, j, width, height, lw, lh)
    fx = (Float32(i) - 0.5f0) * Float32(lw) / Float32(width) + 0.5f0
    fy = (Float32(j) - 0.5f0) * Float32(lh) / Float32(height) + 0.5f0
    x0 = clamp(unsafe_trunc(Int32, floor(fx)), Int32(1), Int32(lw - 1))
    y0 = clamp(unsafe_trunc(Int32, floor(fy)), Int32(1), Int32(lh - 1))
    tx = clamp(fx - Float32(x0), 0.0f0, 1.0f0)
    ty = clamp(fy - Float32(y0), 0.0f0, 1.0f0)
    x1 = x0 + Int32(1); y1 = y0 + Int32(1)
    w00 = (1.0f0 - tx) * (1.0f0 - ty); w10 = tx * (1.0f0 - ty)
    w01 = (1.0f0 - tx) * ty;           w11 = tx * ty
    @inbounds begin
        a = layer[1, x0, y0] * w00 + layer[1, x1, y0] * w10 +
            layer[1, x0, y1] * w01 + layer[1, x1, y1] * w11
        b = layer[2, x0, y0] * w00 + layer[2, x1, y0] * w10 +
            layer[2, x0, y1] * w01 + layer[2, x1, y1] * w11
        c = layer[3, x0, y0] * w00 + layer[3, x1, y0] * w10 +
            layer[3, x0, y1] * w01 + layer[3, x1, y1] * w11
        d = layer[4, x0, y0] * w00 + layer[4, x1, y0] * w10 +
            layer[4, x0, y1] * w01 + layer[4, x1, y1] * w11
    end
    return (a, b, c, d)
end

"""
    sky_composite_kernel!(out, bg, fan, fine, layer, cam_params,
                          spacetime_params, sky_params, star_params, star_lut,
                          ring, width, height, lw, lh, rw, rh, ring_rmin,
                          ju, jv, accumulate, row0, rows)

Per display pixel: build the pixel ray exactly like the trace kernel, find
its local angle ψ to the radial axis (`ê_r` in `sky_params[1:4]`), look up
the exact deflection in `fan` and sample the starmap along the asymptotic
direction (black when captured — the shadow at native resolution), then
composite the premultiplied disc/gas `layer` (bilinear, `lw × lh`) over it.
Near-critical fan entries (neighbours disagreeing in escape or direction)
fall back to the nearest entry — a sub-pixel zone at the photon ring.
"""
function sky_composite_kernel!(out, bg, fan, fine, layer, cam_params,
                               spacetime_params, sky_params, star_params,
                               star_lut, ring,
                               width, height, lw, lh, rw, rh, ring_rmin,
                               ju, jv, accumulate,
                               row0, rows, ::Val{SKYTEX}) where {SKYTEX}
    idx = thread_position_in_grid().x
    idx > width * rows && return
    j = (idx - 1) ÷ width + 1 + row0
    i = (idx - 1) % width + 1
    j > height && return

    M = spacetime_params[1]
    cx = cam_params[1];  cy = cam_params[2];  cz = cam_params[3]
    fov = cam_params[4]
    ef0 = cam_params[5];  ef1 = cam_params[6];  ef2 = cam_params[7];  ef3 = cam_params[8]
    er0 = cam_params[9];  er1 = cam_params[10]; er2 = cam_params[11]; er3 = cam_params[12]
    eu0 = cam_params[13]; eu1 = cam_params[14]; eu2 = cam_params[15]; eu3 = cam_params[16]
    ut0 = cam_params[17]; ut1 = cam_params[18]; ut2 = cam_params[19]; ut3 = cam_params[20]

    half_h = Float32(height) / 2.0f0
    u = (Float32(i) - 1.0f0 + ju - Float32(width) / 2.0f0) / half_h
    v = (Float32(j) - 1.0f0 + jv - Float32(height) / 2.0f0) / half_h

    cr = 0.0f0
    cu = 0.0f0
    cf = 1.0f0
    if cam_params[21] > 0.5f0
        ρ = sqrt(u * u + v * v)
        θp = ρ * cam_params[22]
        sθ = sin(θp)
        inv_ρ = ρ > 1.0f-8 ? 1.0f0 / ρ : 0.0f0
        cr = sθ * u * inv_ρ
        cu = sθ * v * inv_ρ
        cf = cos(θp)
    else
        dxl = u * fov
        dyl = v * fov
        ν = sqrt(dxl * dxl + dyl * dyl + 1.0f0)
        cr = dxl / ν
        cu = dyl / ν
        cf = 1.0f0 / ν
    end

    x = cx; y = cy; z = cz
    r = sqrt(x * x + y * y + z * z)
    f = 2.0f0 * M / r
    qt = cf * ef0 + cr * er0 + cu * eu0 - ut0
    qx = cf * ef1 + cr * er1 + cu * eu1 - ut1
    qy = cf * ef2 + cr * er2 + cu * eu2 - ut2
    qz = cf * ef3 + cr * er3 + cu * eu3 - ut3
    lq = qt + (x * qx + y * qy + z * qz) / r
    p_t = -qt + f * lq
    flr = f * lq / r
    px = qx + flr * x
    py = qy + flr * y
    pz = qz + flr * z

    scam = spacetime_params[4] > 0.5f0 ?
           1.0f0 / clamp(abs(p_t), 0.05f0, 20.0f0) : 1.0f0

    # Local angle to the radial axis, then the exact deflection (fine fan
    # inside the critical band, coarse elsewhere).
    cψ = clamp(p_t * sky_params[1] + px * sky_params[2] +
               py * sky_params[3] + pz * sky_params[4], -1.0f0, 1.0f0)
    ψ = acos(cψ)
    Nfan = Float32(size(fan, 2))
    tψ = ψ * (Nfan - 1.0f0) / Float32(pi)
    esc, cθ, sθ = _fan_dir(fan, fine, sky_params[9], sky_params[10], ψ)

    sky_r = 0.0f0
    sky_g = 0.0f0
    sky_b = 0.0f0
    if esc > 0.5f0
        # Rebuild the asymptotic direction in this ray's own geodesic plane:
        # e1 = outward radial, e2 = the unit lateral part of the coordinate
        # velocity at the camera.
        inv_r = 1.0f0 / r
        κ0 = (x * px + y * py + z * pz) * inv_r
        c10 = f * (-p_t + κ0) * inv_r
        vx = px - c10 * x
        vy = py - c10 * y
        vz = pz - c10 * z
        e1x = x * inv_r; e1y = y * inv_r; e1z = z * inv_r
        vr = vx * e1x + vy * e1y + vz * e1z
        lx = vx - vr * e1x
        ly = vy - vr * e1y
        lz = vz - vr * e1z
        ll = sqrt(lx * lx + ly * ly + lz * lz)
        if ll < 1.0f-7
            # (Anti)radial ray: no deflection plane; the direction is ±e1.
            lx = -e1y; ly = e1x; lz = 0.0f0
            lm = max(sqrt(lx * lx + ly * ly), 1.0f-6)
            lx /= lm; ly /= lm
        else
            lx /= ll; ly /= ll; lz /= ll
        end
        dx = cθ * e1x + sθ * lx
        dy = cθ * e1y + sθ * ly
        dz = cθ * e1z + sθ * lz
        # `SKYTEX` is false exactly when the texture's contribution is
        # multiplied by zero downstream (stars on, `texture_weight` 0 — the
        # shipped default). The gather is a random access into the 4k
        # equirectangular map for every native pixel, so specialising it out
        # is worth a `Val`; a runtime branch on the same frame-uniform value
        # measured *slower*, costing more in scheduling than the gather saved.
        if SKYTEX
            θbg = acos(clamp(dz / max(sqrt(dx * dx + dy * dy + dz * dz), 1.0f-9),
                             -1.0f0, 1.0f0))
            φbg = atan(dy, dx)
            sky_r, sky_g, sky_b = sample_background_mtl(bg, θbg, φbg,
                                                        size(bg, 2), size(bg, 3))
        end
        # Procedural stars, as in the trace kernel: evaluated along the
        # asymptotic direction, so the fan's exact deflection lenses them the
        # same way it lenses the texture. `star_params[16]` dims the texture,
        # so the two crossfade rather than only swapping.
        if star_params[1] > 0.0f0
            dl = max(sqrt(dx * dx + dy * dy + dz * dz), 1.0f-20)
            if SKYTEX
                tw = star_params[16]
                sky_r *= tw; sky_g *= tw; sky_b *= tw
            end
            sr, sg, sb = starfield_mtl(dx / dl, dy / dl, dz / dl,
                                       star_params, star_lut)
            sky_r += star_params[1] * sr
            sky_g += star_params[1] * sg
            sky_b += star_params[1] * sb
        end
        if scam != 1.0f0
            sky_r *= 57.4f0 / (exp(4.067f0 / scam) - 1.0f0)
            sky_g *= 90.2f0 / (exp(4.513f0 / scam) - 1.0f0)
            sky_b *= 206.5f0 / (exp(5.335f0 / scam) - 1.0f0)
        end
    end

    # Composite the (lower-resolution) premultiplied disc layer over the sky.
    lr, lg, lb, la = _layer_bilinear(layer, i, j, width, height, lw, lh)

    # Foveated ring: where the ray winds near the photon sphere, prefer the
    # finer ring layer. Blending on periapsis rather than switching matters —
    # the two layers are sampled at different rates and will not agree, so a
    # hard boundary would trade a stair-stepped ring for a seam ring.
    if ring_rmin > 0.0f0
        rk0 = clamp(unsafe_trunc(Int32, tψ), Int32(0),
                    unsafe_trunc(Int32, Nfan) - Int32(2))
        rfr = tψ - Float32(rk0)
        rmin = fan[4, rk0 + 1] + rfr * (fan[4, rk0 + 2] - fan[4, rk0 + 1])
        t = clamp((rmin - RING_FEATHER * ring_rmin) /
                  max((1.0f0 - RING_FEATHER) * ring_rmin, 1.0f-6),
                  0.0f0, 1.0f0)
        wr = 1.0f0 - t * t * (3.0f0 - 2.0f0 * t)
        if wr > 0.0f0
            rr, rg, rb, ra = _layer_bilinear(ring, i, j, width, height, rw, rh)
            lr += wr * (rr - lr); lg += wr * (rg - lg)
            lb += wr * (rb - lb); la += wr * (ra - la)
        end
    end

    if accumulate > 0.5f0
        # Progressive refinement: sum jittered passes (the presenter divides
        # by the pass count via its exposure factor).
        out[1, i, j] += lr + la * sky_r
        out[2, i, j] += lg + la * sky_g
        out[3, i, j] += lb + la * sky_b
    else
        out[1, i, j] = lr + la * sky_r
        out[2, i, j] = lg + la * sky_g
        out[3, i, j] = lb + la * sky_b
    end
    return nothing
end

"""
    fan_band_kernel!(sky_params, fan, pad)

Locate the escape/capture transition in the coarse fan and write the ψ band
`[transition − pad, transition + pad]` (in coarse spacings) into
`sky_params[9:10]`, where the refinement fan and the composite lookup read
it. The escape flag is monotone in ψ, so at most one pair differs.
"""
function fan_band_kernel!(sky_params, fan, pad)
    k = thread_position_in_grid().x
    N = size(fan, 2)
    k > N - 1 && return
    if fan[1, k] != fan[1, k + 1]
        dψ = Float32(pi) / (Float32(N) - 1.0f0)
        sky_params[9] = max(Float32(k - 1) - Float32(pad), 0.0f0) * dψ
        sky_params[10] = min(Float32(k) + Float32(pad), Float32(N) - 1.0f0) * dψ
    end
    return nothing
end

"""
Fan lookup shared by the composite: returns `(esc, cosθf, sinθf)` for local
angle ψ, using the fine critical-band fan where it applies and falling back
to the nearest entry where neighbours wind or disagree.
"""
@inline function _fan_dir(fan, fine, ψlo, ψhi, ψ)
    use_fine = ψhi > ψlo && ψlo <= ψ && ψ <= ψhi
    tf = 0.0f0
    if use_fine
        Nf = Float32(size(fine, 2))
        tf = (ψ - ψlo) / (ψhi - ψlo) * (Nf - 1.0f0)
    else
        Nc = Float32(size(fan, 2))
        tf = ψ * (Nc - 1.0f0) / Float32(pi)
    end
    nmax_i = use_fine ? Int32(size(fine, 2)) : Int32(size(fan, 2))
    k0 = clamp(unsafe_trunc(Int32, tf), Int32(0), nmax_i - Int32(2))
    frac = tf - Float32(k0)
    e_a = use_fine ? fine[1, k0 + 1] : fan[1, k0 + 1]
    e_b = use_fine ? fine[1, k0 + 2] : fan[1, k0 + 2]
    c_a = use_fine ? fine[2, k0 + 1] : fan[2, k0 + 1]
    c_b = use_fine ? fine[2, k0 + 2] : fan[2, k0 + 2]
    s_a = use_fine ? fine[3, k0 + 1] : fan[3, k0 + 1]
    s_b = use_fine ? fine[3, k0 + 2] : fan[3, k0 + 2]
    if e_a == e_b && c_a * c_b + s_a * s_b > 0.9987f0
        cθ = c_a + frac * (c_b - c_a)
        sθ = s_a + frac * (s_b - s_a)
        nl = max(sqrt(cθ * cθ + sθ * sθ), 1.0f-6)
        return e_a, cθ / nl, sθ / nl
    end
    kn = frac < 0.5f0 ? k0 + 1 : k0 + 2
    if use_fine
        return fine[1, kn], fine[2, kn], fine[3, kn]
    end
    return fan[1, kn], fan[2, kn], fan[3, kn]
end

# Compiled-once pipelines and dummy buffers for the layered engine.
const _SKY_FAN_KERNEL = Ref{Any}(nothing)
const _FAN_BAND_KERNEL = Ref{Any}(nothing)
const _COMPOSITE_KERNEL = Dict{Any,Any}()
const _DUMMY_FAN = Ref{Any}(nothing)
const _DUMMY_SKYP = Ref{Any}(nothing)
const _DUMMY_STAT = Ref{Any}(nothing)
_dummy_fan() = _DUMMY_FAN[] === nothing ?
    (_DUMMY_FAN[] = MtlArray(zeros(Float32, 4, 2))) : _DUMMY_FAN[]
_dummy_skyp() = _DUMMY_SKYP[] === nothing ?
    (_DUMMY_SKYP[] = MtlArray(zeros(Float32, 8))) : _DUMMY_SKYP[]
_dummy_stat() = _DUMMY_STAT[] === nothing ?
    (_DUMMY_STAT[] = MtlArray(zeros(Float32, 3, 1, 1))) : _DUMMY_STAT[]

"""
    SkyFanState(pos, M; n=4096)

Host-side state for the layered engine's deflection fan: the fan table, its
parameter block, and the fan camera's tetrad block.
"""
struct SkyFanState
    fan::MtlArray{Float32,2}     # coarse: uniform in ψ over [0, π]
    fine::MtlArray{Float32,2}    # refinement across the critical-angle band
    sky_params::MtlVector{Float32}
    fan_cam::MtlVector{Float32}
end

SkyFanState(; n::Int=4096, n_fine::Int=1024) = SkyFanState(
    MtlArray{Float32,2}(undef, 4, n),
    MtlArray{Float32,2}(undef, 4, n_fine),
    MtlVector{Float32}(undef, 12),
    MtlVector{Float32}(undef, CAM_PARAMS_N))

"""
    update_sky_fan!(sky::SkyFanState, ctx, pos, spacetime; gate, dt=0.02)

Rebuild the deflection fan for a camera at `pos`: integrate `n` exact
geodesics from radius `‖pos‖` (a per-frame cost of roughly one image
column). Also refreshes `sky_params`: the radial tetrad axis ê_r at `pos`,
`(M, r_escape)`, and the gas gate radius.
"""
function update_sky_fan!(sky::SkyFanState, ctx::MetalPreviewContext,
                         pos::SVector{3,Float64}, spacetime::AbstractSpacetime;
                         gate::Real, dt::Real=Float64(ctx.dt))
    M = spacetime.M
    r = norm(pos)
    x̂ = pos / r
    # Fan camera at (r, 0, 0): fwd = outward radial, right = +ŷ (the fan's
    # lateral axis), matching the kernel's θf convention.
    fpos = SVector(r, 0.0, 0.0)
    tet = ks_camera_tetrad(fpos, SVector(1.0, 0.0, 0.0),
                           SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0), M)
    copyto!(sky.fan_cam,
            Float32[fpos..., 1.0, tet[2]..., tet[3]..., tet[4]..., tet[1]...,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # ê_r at the actual camera position (any completion of the basis).
    a = abs(x̂[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)
    b1 = normalize(cross(a, x̂))
    b2 = cross(x̂, b1)
    êr = ks_camera_tetrad(pos, x̂, b1, b2, M)[2]
    r_escape = ctx.r_escape_factor * max(r, 15.0 * M)
    copyto!(sky.sky_params,
            Float32[êr..., M, r_escape, gate, 0.0, 0.0, 0.0, 0.0, 0.0])
    nmax = min(max(ctx.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt)),
               20_000)
    n = size(sky.fan, 2)
    if _SKY_FAN_KERNEL[] === nothing
        _SKY_FAN_KERNEL[] = @metal launch=false sky_fan_kernel!(
            sky.fan, sky.fan_cam, sky.sky_params, nmax, Float32(dt), Int32(0))
    end
    kern = _SKY_FAN_KERNEL[]
    threads = min(kern.pipeline.maxTotalThreadsPerThreadgroup, n)
    kern(sky.fan, sky.fan_cam, sky.sky_params, nmax, Float32(dt), Int32(0);
         threads=threads, groups=cld(n, threads))
    # Locate the escape/capture transition, then refine across it: the
    # near-critical zone is where the coarse fan's stairs would show at the
    # shadow edge. All GPU-side — no host round trip.
    if _FAN_BAND_KERNEL[] === nothing
        _FAN_BAND_KERNEL[] = @metal launch=false fan_band_kernel!(
            sky.sky_params, sky.fan, Int32(3))
    end
    bk = sky.sky_params
    kb = _FAN_BAND_KERNEL[]
    tb = min(kb.pipeline.maxTotalThreadsPerThreadgroup, n - 1)
    kb(bk, sky.fan, Int32(3); threads=tb, groups=cld(n - 1, tb))
    nf = size(sky.fine, 2)
    tf = min(kern.pipeline.maxTotalThreadsPerThreadgroup, nf)
    kern(sky.fine, sky.fan_cam, sky.sky_params, nmax, Float32(dt), Int32(1);
         threads=tf, groups=cld(nf, tf))
    return nothing
end

"""
    render_layered_gpu!(comp_out, layer_out, ctx, sky, cam, spacetime;
                        fisheye_deg=0.0, relativistic=false, dt=ctx.dt)

One frame of the layered engine, entirely on the GPU: trace the disc/gas
layer into `layer_out` (`(4, lw, lh)`, premultiplied RGB + transmittance,
gate-culled by the fan), then composite it over the exact fan-driven sky
into `comp_out` (`(3, width, height)`). Call [`update_sky_fan!`](@ref) for
the current camera position first.
"""
function render_layered_gpu!(comp_out, layer_out, ctx::MetalPreviewContext,
                             sky::SkyFanState, cam::Camera,
                             spacetime::AbstractSpacetime;
                             fisheye_deg::Real=0.0, relativistic::Bool=false,
                             dt::Real=Float64(ctx.dt), trace_layer::Bool=true,
                             substride::Int=1, subx::Int=0, suby::Int=0,
                             ju::Real=0.5, jv::Real=0.5,
                             accumulate::Bool=false,
                             row0::Int=0, rows::Int=-1,
                             ring_out=nothing, ring_rmin::Real=0.0,
                             ring_box::Bool=true, order::Int=4,
                             tol::Real=1.0f-4)
    M = Float32(spacetime.M)
    r_band = Float32(2.05 * spacetime.M)
    r_escape = Float32(ctx.r_escape_factor * max(norm(cam.pos),
                                                 15.0 * spacetime.M))
    nmax = min(max(ctx.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt)),
               20_000)
    _ks_cam_params!(ctx.cam_host, cam, spacetime.M; fisheye_deg=fisheye_deg,
                    a=spin(spacetime))
    spin_a, rkill2 = _spin_horizon(spacetime)
    is_kerr = spin_a != 0
    sh = ctx.st_host
    @inbounds begin
        sh[1] = M; sh[2] = r_band; sh[3] = r_escape
        sh[4] = relativistic ? 1.0f0 : 0.0f0
        sh[5] = HSTEP_COEF[]; sh[6] = HSTEP_CAP[]
        sh[7] = spin_a; sh[8] = rkill2
    end

    lw, lh = size(layer_out, 2), size(layer_out, 3)
    width, height = size(comp_out, 2), size(comp_out, 3)
    # Optional row band (time-sliced refinement): applies to both passes.
    # Banded use requires a display-sized layer so the rows line up.
    rows < 0 && (rows = height)
    banded = !(row0 == 0 && rows == height)
    banded && ((lw, lh) != (width, height) || substride != 1) &&
        throw(ArgumentError("row-banded layered render needs a display-sized layer and substride 1"))
    if trace_layer
        # LAYER passes write every dispatched pixel by assignment, so no
        # clear is needed — and a sub-grid pass (substride > 1) must NOT
        # clear: the undispatched pixels carry reprojected history. The
        # dispatch covers layer_out / substride pixels of it.
        sw, sh = cld(lw, substride), cld(lh, substride)
        lrow0 = banded ? row0 : 0
        lrows = banded ? rows : sh
        _launch_trace!(ctx, layer_out, ctx.cam_params, ctx.spacetime_params,
                       sw, sh, nmax, Float32(dt), Float32(ju), Float32(jv),
                       1.0f0, lrow0, lrows;
                       fan=sky.fan, sky_params=sky.sky_params, layer=true,
                       substride=substride, subx=subx, suby=suby, order=order,
                       tol=tol, kerr=is_kerr)
    end
    # With trace_layer=false the caller keeps `layer_out` pre-filled with
    # α = 1 (fully transparent): the frame is the fan-driven sky alone.

    # Foveated ring pass: a second, finer trace restricted to the wound rays.
    # It is a thin annulus (~1-5% of pixels) and spatially coherent, so whole
    # SIMD groups fall outside it and exit on the fan lookup — but those rays
    # are also the most expensive in the frame, so the saving is in resolution,
    # not in ray count.
    use_ring = ring_out !== nothing && ring_rmin > 0 && trace_layer && !banded
    rw, rh = use_ring ? (size(ring_out, 2), size(ring_out, 3)) : (lw, lh)
    ring_b = use_ring ? ring_out : layer_out
    if use_ring
        # Dispatch only the box that can contain wound rays. Everything outside
        # it would have been culled per-pixel anyway, so the pass is unchanged
        # where it matters and simply absent where it was writing transparency.
        box = ring_box ?
              _ring_screen_box(cam, spacetime.M, ring_rmin, fisheye_deg,
                               rw, rh) : nothing
        rc0, rcn, rr0, rrn = box === nothing ? (0, rw, 0, rh) : box
        _launch_trace!(ctx, ring_out, ctx.cam_params, ctx.spacetime_params,
                       rw, rh, nmax, Float32(dt), Float32(ju), Float32(jv),
                       1.0f0, rr0, rrn;
                       fan=sky.fan, sky_params=sky.sky_params, layer=true,
                       ring_rmin=ring_rmin, col0=rc0, cols=rcn, order=order,
                       tol=tol, kerr=is_kerr)
    end

    acc = accumulate ? 1.0f0 : 0.0f0
    skytex = ctx.sky_tex[]
    if !haskey(_COMPOSITE_KERNEL, skytex)
        _COMPOSITE_KERNEL[skytex] = @metal launch=false sky_composite_kernel!(
            comp_out, ctx.bg_gpu, sky.fan, sky.fine, layer_out,
            ctx.cam_params, ctx.spacetime_params, sky.sky_params,
            ctx.star_params, ctx.star_lut, ring_b,
            width, height, lw, lh, rw, rh,
            use_ring ? Float32(ring_rmin) : 0.0f0,
            Float32(ju), Float32(jv), acc, row0, rows, Val(skytex))
    end
    kern = _COMPOSITE_KERNEL[skytex]
    n = width * rows
    threads = min(kern.pipeline.maxTotalThreadsPerThreadgroup, n)
    kern(comp_out, ctx.bg_gpu, sky.fan, sky.fine, layer_out, ctx.cam_params,
         ctx.spacetime_params, sky.sky_params, ctx.star_params, ctx.star_lut,
         ring_b, width, height, lw, lh, rw, rh,
         use_ring ? Float32(ring_rmin) : 0.0f0,
         Float32(ju), Float32(jv), acc, row0, rows, Val(skytex);
         threads=threads, groups=cld(n, threads))
    return nothing
end

"""
GPU-only reprojection: warp `prev` into `warp_out` without downloading.
`channels=4` warps a premultiplied gas layer (revealed pixels transparent);
`width`/`height` override the warp dimensions (default: the context's).
"""
function _warp_gpu!(ctx::MetalPreviewContext, warp_out, prev, warp_params,
                    cam::Camera, prev_cam::Camera; fisheye_deg::Real=0.0,
                    channels::Int=3, width::Int=ctx.width,
                    height::Int=ctx.height)
    copyto!(warp_params,
            Float32[cam.fwd..., cam.right..., cam.up_local...,
                    prev_cam.fwd..., prev_cam.right..., prev_cam.up_local...,
                    cam.fov_factor,
                    fisheye_deg > 0 ? 1.0 : 0.0,
                    deg2rad(max(fisheye_deg, 0.0)), 0.0])
    kernels = _WARP_KERNEL[] === nothing ?
        (_WARP_KERNEL[] = Dict{Int,Any}()) : _WARP_KERNEL[]::Dict{Int,Any}
    if !haskey(kernels, channels)
        kernels[channels] = @metal launch=false warp_kernel_mtl!(
            warp_out, prev, warp_params, width, height, Val(channels))
    end
    kern = kernels[channels]
    n = width * height
    threads = min(kern.pipeline.maxTotalThreadsPerThreadgroup, n)
    kern(warp_out, prev, warp_params, width, height, Val(channels);
         threads=threads, groups=cld(n, threads))
    return nothing
end

# ---------------------------------------------------------------------------
# Baked warp maps
# ---------------------------------------------------------------------------
#
# Live Kerr tracing costs ~170 ns/pixel: every pixel is a full RK4 geodesic with
# a hundred-odd steps. A warp-map lookup is a couple of bilinear fetches. That
# ratio is not a speedup so much as a change of category — you stop paying for
# physics per pixel and start paying for bandwidth per pixel.
#
# What makes it work is that lensing maps a WORLD direction to a world
# direction. It has no dependence on where the camera is pointing, only on where
# it is. So one map per position serves every orientation: free look, arbitrary
# roll, any field of view, all from the same table. A racing game can exploit
# that harder than a free-flight simulator can, because the track constrains the
# camera to a curve rather than a volume.
#
# The maps store the escape DIRECTION, not the final colour. That is what keeps
# output sharp from a small table — see the kernel's `BAKE` block.
#
# One limit worth stating: the disc is baked as colour, so it does not animate
# between bakes. Baking the disc HIT COORDINATES instead (r_hit, φ_hit, and the
# redshift g) would let it rotate and shimmer at runtime with the geodesics
# still frozen. Not done here.

"""
    bake_warp_map(ctx, spacetime, pos, vel; mapw, maph)

Trace one equirectangular warp map at `pos`, for a ship moving at coordinate
3-velocity `vel`. Returns `(map_gpu, basis)`, where `basis` is the world-aligned
camera frame the map is indexed against.

The map is baked with a **world-aligned** camera rather than the ship's, which
is what makes the index a world direction and therefore orientation-independent.
`vel` still matters: it boosts the observer tetrad, so aberration is baked in as
the ship at that point would see it.
"""
function bake_warp_map(ctx::MetalPreviewContext, spacetime::AbstractSpacetime,
                       pos::SVector{3,Float64}, vel::SVector{3,Float64};
                       mapw::Int=512, maph::Int=256)
    # Bake in a frame rotated to the ship's own azimuth, so the map is stored in
    # CANONICAL form (as if the ship were at φ = 0). Kerr is axisymmetric, so
    # two maps at different azimuths are the same map rotated — and near
    # periapsis almost all the apparent change between neighbouring samples IS
    # that rotation. Factoring it out here is what lets interpolation work at
    # the one place on the track where it matters.
    #
    # The resulting basis is R_z(φ)·(x̂, −ŷ, ẑ), so the sampler only has to undo
    # a rotation about z rather than carry a general frame.
    φc = atan(pos[2], pos[1])
    cam = Camera(pos, pos + SVector(cos(φc), sin(φc), 0.0),
                 SVector(0.0, 0.0, 1.0), 1.0; velocity = vel)
    M = Float32(spacetime.M)
    r_escape = Float32(ctx.r_escape_factor * max(norm(pos), 15.0 * spacetime.M))
    dt = ctx.dt
    nmax = min(max(ctx.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt)), 20_000)
    cp = MtlArray(_ks_cam_params(cam, spacetime.M; equirect = true,
                                 a=spin(spacetime)))
    spin_a, rkill2 = _spin_horizon(spacetime)
    sp = MtlArray(Float32[M, Float32(2.05 * spacetime.M), r_escape, 0.0f0,
                          HSTEP_COEF[], HSTEP_CAP[], spin_a, rkill2])
    out = MtlArray{Float32}(undef, 8, mapw, maph)
    fill!(out, 0.0f0)
    _launch_trace!(ctx, out, cp, sp, mapw, maph, nmax, dt,
                   0.5f0, 0.5f0, 1.0f0, 0, maph;
                   kerr = (spin_a != 0), bake = true)
    return out, φc
end

"""
Sample two baked maps, cross-fade, and composite the sky at FULL output
resolution. The maps supply the deflection and the disc; the starfield is
evaluated per pixel against the interpolated direction, so stars stay sharp at
any output resolution.
"""
@inline function _warp_lookup(m, dx, dy, dz, cϕ, sϕ)
    # Rotate the view direction back by this map's own azimuth, so it indexes
    # the canonical (φ = 0) map.
    rx =  cϕ * dx + sϕ * dy
    ry = -sϕ * dx + cϕ * dy
    # Bake basis is R_z(φ)·(x̂, −ŷ, ẑ); having undone R_z(φ), what is left is
    # (x̂, −ŷ, ẑ).
    θ = acos(clamp(rx, -1.0f0, 1.0f0))
    φ = atan(dz, -ry)

    W = size(m, 2)
    H = size(m, 3)
    fH = Float32(H)
    fu = φ * fH / 3.1415927f0 + Float32(W) / 2.0f0
    fv = θ * fH / 3.1415927f0

    i0 = unsafe_trunc(Int32, floor(fu - 0.5f0))
    j0 = unsafe_trunc(Int32, floor(fv - 0.5f0))
    tu = fu - 0.5f0 - Float32(i0)
    tv = fv - 0.5f0 - Float32(j0)
    # φ wraps, θ clamps: the poles are single points, not a seam. Branches
    # rather than `mod` — integer modulo is multi-cycle and `fu` is already
    # within one period of range.
    iW = Int32(W)
    ia = i0 < Int32(0) ? i0 + iW : (i0 >= iW ? i0 - iW : i0)
    i1 = i0 + Int32(1)
    ib = i1 < Int32(0) ? i1 + iW : (i1 >= iW ? i1 - iW : i1)
    ia += Int32(1); ib += Int32(1)
    ja = clamp(j0, Int32(0), Int32(H - 1)) + Int32(1)
    jb = clamp(j0 + Int32(1), Int32(0), Int32(H - 1)) + Int32(1)

    a00 = (1.0f0 - tu) * (1.0f0 - tv)
    a10 = tu * (1.0f0 - tv)
    a01 = (1.0f0 - tu) * tv
    a11 = tu * tv

    @inbounds begin
        v1 = m[1,ia,ja]*a00 + m[1,ib,ja]*a10 + m[1,ia,jb]*a01 + m[1,ib,jb]*a11
        v2 = m[2,ia,ja]*a00 + m[2,ib,ja]*a10 + m[2,ia,jb]*a01 + m[2,ib,jb]*a11
        v3 = m[3,ia,ja]*a00 + m[3,ib,ja]*a10 + m[3,ia,jb]*a01 + m[3,ib,jb]*a11
        v4 = m[4,ia,ja]*a00 + m[4,ib,ja]*a10 + m[4,ia,jb]*a01 + m[4,ib,jb]*a11
        v5 = m[5,ia,ja]*a00 + m[5,ib,ja]*a10 + m[5,ia,jb]*a01 + m[5,ib,jb]*a11
        v6 = m[6,ia,ja]*a00 + m[6,ib,ja]*a10 + m[6,ia,jb]*a01 + m[6,ib,jb]*a11
        v7 = m[7,ia,ja]*a00 + m[7,ib,ja]*a10 + m[7,ia,jb]*a01 + m[7,ib,jb]*a11
        v8 = m[8,ia,ja]*a00 + m[8,ib,ja]*a10 + m[8,ia,jb]*a01 + m[8,ib,jb]*a11
        # The stored escape direction is in world coordinates for THIS map's
        # configuration; rotate it to canonical form too, so the two maps are
        # blended in a common frame rather than across a large rotation.
        e1 =  cϕ * v1 + sϕ * v2
        e2 = -sϕ * v1 + cϕ * v2
        return (e1, e2, v3, v4, v5, v6, v7, v8)
    end
end

"""
Sample two baked maps, cross-fade, and composite the sky at FULL output
resolution. The maps supply the deflection and the disc; the starfield is
evaluated per pixel against the interpolated direction, so stars stay sharp at
any output resolution and map size stops being what limits image quality.

Both maps are stored canonically (ship at azimuth zero) and are un-rotated into
a common frame before blending, so the interpolation only ever has to span the
change in radius — not the ship's sweep around the hole, which near periapsis is
most of the apparent change and which axisymmetry makes free.
"""
function warp_sample_kernel!(out, m0, m1, bg, star_lut, star_params, p,
                             width, height)
    idx = thread_position_in_grid().x
    if idx > width * height
        return
    end
    j = (idx - 1) ÷ width + 1
    i = (idx - 1) % width + 1

    fov = p[1]
    half_h = Float32(height) / 2.0f0
    uu = (Float32(i) - 0.5f0 - Float32(width) / 2.0f0) / half_h
    vv = (Float32(j) - 0.5f0 - Float32(height) / 2.0f0) / half_h
    dlx = uu * fov
    dly = vv * fov
    ν = sqrt(dlx * dlx + dly * dly + 1.0f0)
    cr = dlx / ν
    cu = dly / ν
    cf = 1.0f0 / ν

    # Pixel direction in world coordinates via the runtime camera basis.
    dx = cf * p[2] + cr * p[5] + cu * p[8]
    dy = cf * p[3] + cr * p[6] + cu * p[9]
    dz = cf * p[4] + cr * p[7] + cu * p[10]

    A = _warp_lookup(m0, dx, dy, dz, p[11], p[12])
    B = _warp_lookup(m1, dx, dy, dz, p[13], p[14])

    w = p[20]
    ω = 1.0f0 - w
    vx = ω*A[1] + w*B[1]
    vy = ω*A[2] + w*B[2]
    vz = ω*A[3] + w*B[3]
    dr = ω*A[4] + w*B[4]
    dg = ω*A[5] + w*B[5]
    db = ω*A[6] + w*B[6]
    al = ω*A[7] + w*B[7]
    es = clamp(ω*A[8] + w*B[8], 0.0f0, 1.0f0)

    # Back out of canonical form into the ship's current azimuth, so the sky is
    # sampled in world coordinates and the starfield stays fixed to the sky
    # rather than spinning with the ship.
    cn = p[15]; sn = p[16]
    wx = cn * vx - sn * vy
    wy = sn * vx + cn * vy

    sr = 0.0f0; sg = 0.0f0; sb = 0.0f0
    if es > 1.0f-3
        vl = max(sqrt(wx * wx + wy * wy + vz * vz), 1.0f-20)
        nx = wx / vl; ny = wy / vl; nz = vz / vl
        θb = acos(clamp(nz, -1.0f0, 1.0f0))
        φb = atan(ny, nx)
        sr, sg, sb = sample_background_mtl(bg, θb, φb, size(bg, 2), size(bg, 3))
        if star_params[1] > 0.0f0
            tw = star_params[16]
            sr *= tw; sg *= tw; sb *= tw
            pr, pg, pb = starfield_mtl(nx, ny, nz, star_params, star_lut)
            sr += star_params[1] * pr
            sg += star_params[1] * pg
            sb += star_params[1] * pb
        end
        sr *= es; sg *= es; sb *= es
    end

    ex = p[21]
    @inbounds begin
        out[1, i, j] = (dr + al * sr) * ex
        out[2, i, j] = (dg + al * sg) * ex
        out[3, i, j] = (db + al * sb) * ex
    end
    return nothing
end

"""
    BakedTrack

Warp maps baked along a track, plus the persistent buffers the sampler needs and
the sky sources it composites against.

The parameter buffer is shared storage with a host view that ALIASES it. Apple
silicon has unified memory, so staging 20 floats through a host vector and
calling `copyto!` was copying a buffer to itself — ~1 kB of allocation per frame
to move 80 bytes the GPU could already see.
"""
struct BakedTrack{M,B,L,S,K}
    maps::Vector{M}
    τ::Vector{Float64}
    # Azimuth each map was baked at. The maps are stored canonically, so this
    # is what the sampler rotates by to get back to world coordinates.
    φ::Vector{Float64}
    # Per-map exposure, baked alongside the geodesics. A single flyby spans
    # ~400x in scene luminance — the 99th percentile runs from 0.025 far out to
    # 11.0 on the approach, where a third of the frame clipped to the top
    # palette tone and the picture became a cream wall. No fixed exposure serves
    # that range.
    #
    # A live renderer would need auto-exposure with a time constant, which lags
    # and pumps. Here the track is known in advance, so the curve is
    # precomputed and interpolated: correct on the first frame, no lag, no
    # hunting. The same trick as the lensing, applied to the histogram.
    exposure::Vector{Float32}
    bg::B
    star_lut::L
    star_params::S
    params::MtlVector{Float32,Metal.SharedStorage}
    host::Vector{Float32}
    kernel::K
end

"""
    _baked_track(ctx, maps, τs, basis)

Wrap baked maps with the persistent sampler state. The kernel is compiled
eagerly so it can be stored concretely rather than fetched from a `Ref{Any}` on
every frame, and `maps` must already be a concretely-typed vector — as
`Vector{Any}` it boxed a map handle per frame for no reason.
"""
function _baked_track(ctx::MetalPreviewContext, maps::Vector,
                      τs::Vector{Float64}, φs::Vector{Float64},
                      exposure::Vector{Float32}=ones(Float32, length(maps)))
    pbuf = MtlArray{Float32,1,Metal.SharedStorage}(undef, 24)
    phost = unsafe_wrap(Array, pbuf)
    fill!(phost, 0.0f0)
    phost[21] = 1.0f0
    dummy = MtlArray{Float32}(undef, 3, 1, 1)
    kern = @metal launch=false warp_sample_kernel!(dummy, maps[1], maps[1],
                     ctx.bg_gpu, ctx.star_lut, ctx.star_params, pbuf, 1, 1)
    return BakedTrack(maps, τs, φs, exposure, ctx.bg_gpu, ctx.star_lut,
                      ctx.star_params, pbuf, phost, kern)
end

"""
    render_baked!(out, bt, τ, cam)

Composite one frame from the two maps bracketing proper time `τ`. No geodesic is
integrated: the maps supply the deflection and the disc, and the sky is
evaluated per pixel against the interpolated direction.

Blending two maps is a linear interpolation of a smooth deflection field, which
is right everywhere except across the shadow edge, where rays either escape or
are captured and there is no in-between. The escape flag carries that edge, so
it comes out antialiased rather than ghosted.
"""
function render_baked!(out, bt::BakedTrack, τ::Real, cam::Camera)
    n = length(bt.τ)
    k = clamp(searchsortedfirst(bt.τ, τ), 2, n)
    w = n == 1 ? 0.0 :
        clamp((τ - bt.τ[k-1]) / (bt.τ[k] - bt.τ[k-1]), 0.0, 1.0)
    h = bt.host
    @inbounds begin
        h[1] = cam.fov_factor
        h[2] = cam.fwd[1];      h[3] = cam.fwd[2];      h[4] = cam.fwd[3]
        h[5] = cam.right[1];    h[6] = cam.right[2];    h[7] = cam.right[3]
        h[8] = cam.up_local[1]; h[9] = cam.up_local[2]; h[10] = cam.up_local[3]
        h[11] = cos(bt.φ[k-1]); h[12] = sin(bt.φ[k-1])
        h[13] = cos(bt.φ[k]);   h[14] = sin(bt.φ[k])
        # Blend the azimuth as an ANGLE, shortest way round, so a pass that
        # crosses the branch cut of atan does not snap the sky through 2π.
        dφ = bt.φ[k] - bt.φ[k-1]
        dφ = dφ - 2π * round(dφ / 2π)
        φn = bt.φ[k-1] + w * dφ
        h[15] = cos(φn); h[16] = sin(φn)
        h[20] = Float32(w)
        h[21] = (1.0f0 - Float32(w)) * bt.exposure[max(k-1, 1)] +
                Float32(w) * bt.exposure[k]
    end
    width = size(out, 2); height = size(out, 3)
    m0 = bt.maps[max(k-1, 1)]; m1 = bt.maps[k]
    N = width * height
    threads = min(bt.kernel.pipeline.maxTotalThreadsPerThreadgroup, N)
    bt.kernel(out, m0, m1, bt.bg, bt.star_lut, bt.star_params, bt.params,
              width, height; threads=threads, groups=cld(N, threads))
    return nothing
end

"""
    _adaptive_taus(ctx, spacetime, track, n; probe_n, probe_res, floor_frac)

Choose where along the track to bake, by measuring rather than assuming.

Spacing maps uniformly in proper time is wrong, and visibly so: the ship covers
a huge amount of *field* per unit time near periapsis and almost none far out,
so uniform spacing under-samples exactly the part of the track the whole level
is built around. The result is a jolt on the approach — adjacent maps differ so
much that interpolating between them cannot hide the step.

So bake a cheap low-resolution probe pass first, measure how much the map
actually changes between neighbours, and redistribute the real samples to
equalise that change. `floor_frac` keeps a fraction of the budget uniform, so a
long quiet stretch still gets some samples instead of none.
"""
function _adaptive_taus(ctx::MetalPreviewContext, spacetime::AbstractSpacetime,
                        track, n::Int; probe_n::Int=33, probe_res::Int=128,
                        floor_frac::Float64=0.15)
    τp = collect(range(track.τ[1], track.τ[end]; length=probe_n))
    d = zeros(Float64, probe_n - 1)
    prev = nothing
    for k in 1:probe_n
        pos, _, _, vel, _ = track_sample(track, τp[k])
        m, _ = bake_warp_map(ctx, spacetime, pos, vel;
                             mapw=probe_res, maph=probe_res ÷ 2)
        a = Array(m)
        prev !== nothing && (d[k-1] = sqrt(sum(abs2, a .- prev) / length(a)))
        prev = a
    end
    tot = sum(d)
    tot <= 0 && return collect(range(track.τ[1], track.τ[end]; length=n))
    w = (1 - floor_frac) .* (d ./ tot) .+ floor_frac / length(d)
    c = vcat(0.0, cumsum(w)); c ./= c[end]
    # Invert the cumulative curve: equal steps in "field change" -> unequal τ.
    τs = Vector{Float64}(undef, n)
    for (i, u) in enumerate(range(0.0, 1.0; length=n))
        j = clamp(searchsortedfirst(c, u), 2, probe_n)
        span = c[j] - c[j-1]
        f = span > 0 ? (u - c[j-1]) / span : 0.0
        τs[i] = τp[j-1] + f * (τp[j] - τp[j-1])
    end
    τs[1] = track.τ[1]; τs[end] = track.τ[end]
    return τs
end

"""
    bake_track_maps(ctx, spacetime, track; n, mapw, maph)

Bake `n` warp maps evenly spaced in proper time along `track`, as a
[`BakedTrack`](@ref).

Bake cost is set by total texel count, not by how it is split: a few tens of
millions of geodesics take seconds on an M3, which is a loading screen rather
than a build step. That is what makes mass, spin and disc geometry free
per-level parameters — the table does not have to ship.
"""
function bake_track_maps(ctx::MetalPreviewContext, spacetime::AbstractSpacetime,
                         track; n::Int=48, mapw::Int=512, maph::Int=256,
                         exposure_target::Real=1.0, adaptive::Bool=true,
                         verbose::Bool=true)
    t0 = time()
    τs = adaptive ? _adaptive_taus(ctx, spacetime, track, n) :
                    collect(range(track.τ[1], track.τ[end]; length=n))
    # Bake the first map to learn its concrete type, then allocate for it.
    p1, _, _, v1, _ = track_sample(track, τs[1])
    m1, φ1 = bake_warp_map(ctx, spacetime, p1, v1; mapw=mapw, maph=maph)
    maps = Vector{typeof(m1)}(undef, n)
    φs = Vector{Float64}(undef, n)
    maps[1] = m1; φs[1] = φ1
    # Unwrap the azimuth so the sampler can interpolate it as a continuous
    # angle: a flyby can wind through many turns near periapsis, and a wrapped
    # atan would make every crossing look like a jump cut.
    for k in 2:n
        pos, _, _, vel, _ = track_sample(track, τs[k])
        maps[k], φk = bake_warp_map(ctx, spacetime, pos, vel;
                                    mapw=mapw, maph=maph)
        φs[k] = φs[k-1] + rem(φk - φs[k-1], 2π, RoundNearest)
    end
    Metal.synchronize()
    # Exposure per map: put the 99.5th percentile of scene luminance at
    # `exposure_target`. A percentile rather than the max, so a handful of
    # near-critical rays piling onto the photon ring cannot drag the whole
    # frame dark.
    expo = Vector{Float32}(undef, n)
    for k in 1:n
        a = Array(maps[k])
        lum = vec(0.2126f0 .* a[4, :, :] .+ 0.7152f0 .* a[5, :, :] .+
                  0.0722f0 .* a[6, :, :])
        sort!(lum)
        pk = lum[max(1, round(Int, 0.995 * length(lum)))]
        expo[k] = Float32(exposure_target / max(pk, 1.0f-4))
    end
    if verbose
        rays = n * mapw * maph
        @printf("baked %d maps (%dx%d, %.1fM geodesics) in %.2f s — %.0f MB\n",
                n, mapw, maph, rays / 1e6, time() - t0,
                n * 8 * mapw * maph * 4 / 1e6)
    end
    return _baked_track(ctx, maps, τs, φs, expo)
end

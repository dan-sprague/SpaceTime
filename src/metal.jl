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
    width::Int
    height::Int
    dt::Float32
    nmax::Int
    r_escape_factor::Float32
    has_volume::Bool
    vol_on::Base.RefValue{Bool}  # runtime volumetric toggle
    # Compiled kernel per volume mode (Val-specialised, so the no-volume
    # variant keeps the lean kernel's register budget), built lazily.
    kernel::Base.RefValue{Any}
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
    cam_params = MtlVector{Float32}(undef, 28)
    spacetime_params = MtlVector{Float32}(undef, 4)
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
    return MetalPreviewContext(bg_gpu, out_gpu, cam_params, spacetime_params,
                               disc_params, bb_lut, vol_gpu, vol_params,
                               star_params,
                               width, height, Float32(dt),
                               nmax, Float32(r_escape_factor),
                               !isnothing(volume),
                               Base.RefValue{Bool}(!isnothing(volume)),
                               Base.RefValue{Any}(Dict{Any,Any}()))
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
Contexts that share `disc_params` (flythrough resolution variants) all
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

`flux` is the faintest star's linear brightness; the distribution runs up from
there as ξ^(−2/3) over roughly a 460× range. The default is calibrated against
`starmap_g4k.jpg` at the hero framing, where the brightest star in a clear-sky
crop reaches ≈2.3 linear — match that and the two skies carry comparable
weight, so `texture_weight` becomes a pure look dial rather than an exposure
correction.
"""
function set_starfield!(ctx::MetalPreviewContext; strength::Real=1.0,
                        texture_weight::Real=0.0,
                        height::Union{Int,Nothing}=nothing,
                        fov_factor::Real=0.55, density::Real=1024,
                        fill::Real=0.5, flux::Real=0.022,
                        psf_pixels::Real=1.0,
                        galactic::NTuple{3,Real}=(0.0, 0.0, 1.0),
                        concentration::Real=3.0, temp_min::Real=3000,
                        temp_max::Real=9000, seed::Integer=12345)
    H = something(height, ctx.height)
    gx, gy, gz = galactic
    gn = sqrt(gx^2 + gy^2 + gz^2)
    gn > 0 || throw(ArgumentError("galactic normal must be non-zero"))
    # The pixel's angular footprint: the vertical field is 2·fov_factor across
    # `H` rows. Sizing the PSF from this rather than from a fixed angle is what
    # keeps stars ~1 px at every resolution.
    σ = psf_pixels * 2 * fov_factor / H
    # Star colour reuses the disc's blackbody LUT; its bounds live in
    # disc_params[4:6]. A context built without a disc has none, so fall back
    # to a plain table rather than indexing an empty one.
    lut = Metal.@allowscalar (ctx.disc_params[4], ctx.disc_params[5],
                              ctx.disc_params[6])
    if lut[3] < 1
        throw(ArgumentError("""the starfield needs a blackbody LUT for star \
            colour; build the context with `disc=` or `blackbody=`."""))
    end
    copyto!(ctx.star_params,
            Float32[strength, density, fill, σ, flux,
                    gx / gn, gy / gn, gz / gn, concentration,
                    temp_min, temp_max - temp_min,
                    lut[1], lut[2], lut[3], seed, texture_weight])
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
    starfield_mtl(dx, dy, dz, sp, bb_lut) -> (r, g, b)

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
@inline function starfield_mtl(dx, dy, dz, sp, bb_lut)
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
            T = tmin + tspan * ht * ht                    # biased cool
            frac = (clamp(T, lut_tmin, lut_tmax) - lut_tmin) /
                   max(lut_tmax - lut_tmin, 1.0f-6)
            li = clamp(unsafe_trunc(Int32, frac * (lut_size - 1.0f0) + 0.5f0) +
                       Int32(1), Int32(1), unsafe_trunc(Int32, lut_size))
            acc_r += w * bb_lut[1, li]
            acc_g += w * bb_lut[2, li]
            acc_b += w * bb_lut[3, li]
        end
    end
    return (acc_r, acc_g, acc_b)
end

"""
    trace_kernel_mtl!(out, bg, bb_lut, vol, vol_params, cam_params,
                      spacetime_params, disc_params, width, height, nmax, dt,
                      jitter_u, jitter_v, weight, row0, rows, ::Val{VOL})

Metal compute kernel: one thread per pixel of the current row tile, tracing
geodesics in **Cartesian Kerr–Schild coordinates** (see `ks_rhs_mtl`) — free
of the polar and horizon coordinate singularities of the spherical chart, so
flythroughs never hit pole artifacts. `cam_params` is a 13-element Float32
vector `[pos(3); fwd(3); right(3); up(3); fov_factor]`. `spacetime_params`
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
function trace_kernel_mtl!(out, bg, bb_lut, vol, vol_params, star_params,
                           cam_params,
                           spacetime_params, disc_params, fan, sky_params,
                           width, height, nmax, dt, jitter_u, jitter_v,
                           weight, row0, rows, substride, subx, suby,
                           ::Val{VOL}, ::Val{NB},
                           ::Val{LAYER}) where {VOL, NB, LAYER}
    idx = thread_position_in_grid().x
    total = width * rows
    if idx > total
        return
    end
    j = (idx - 1) ÷ width + 1 + row0
    i = (idx - 1) % width + 1
    if j > height
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

    # Unpack camera position, field of view and the orthonormal tetrad
    # (forward/right/up axes + observer 4-velocity, all contravariant,
    # precomputed on the CPU by `ks_camera_tetrad`).
    cx = cam_params[1];  cy = cam_params[2];  cz = cam_params[3]
    fov = cam_params[4]
    ef0 = cam_params[5];  ef1 = cam_params[6];  ef2 = cam_params[7];  ef3 = cam_params[8]
    er0 = cam_params[9];  er1 = cam_params[10]; er2 = cam_params[11]; er3 = cam_params[12]
    eu0 = cam_params[13]; eu1 = cam_params[14]; eu2 = cam_params[15]; eu3 = cam_params[16]
    ut0 = cam_params[17]; ut1 = cam_params[18]; ut2 = cam_params[19]; ut3 = cam_params[20]

    # Sensor coordinate with subpixel jitter (fw/fh: full sensor size, which
    # differs from the dispatch size only in a LAYER sub-grid pass).
    half_h = Float32(fh) / 2.0f0
    u = (Float32(i) - 1.0f0 + jitter_u - Float32(fw) / 2.0f0) / half_h
    v = (Float32(j) - 1.0f0 + jitter_v - Float32(fh) / 2.0f0) / half_h

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
    if cam_params[21] > 0.5f0
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
    # p_μ = η_μν q^ν + f l_μ (l_ν q^ν) with l_μ = (1, x̂).
    # The origin carries the aperture offset along the tetrad right/up axes
    # (zero for pinhole).
    x = cx + offr * er1 + offu * eu1
    y = cy + offr * er2 + offu * eu2
    z = cz + offr * er3 + offu * eu3
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

    # Optional relativistic shading: each ray is normalised to unit frequency
    # in the camera tetrad, and the conserved p_t is the frequency at
    # infinity, so the camera/infinity shift factor is simply g = 1/|p_t|.
    # 1 when the option is off.
    scam = spacetime_params[4] > 0.5f0 ?
           1.0f0 / clamp(abs(p_t), 0.05f0, 20.0f0) : 1.0f0

    # Layered mode: this pass renders only the disc/gas layer (premultiplied
    # RGB + transmittance in a 4-channel `out`; the sky is composited later
    # from the exact deflection fan). A pixel whose geodesic provably stays
    # outside the gas gate radius — periapsis from the fan, by the local
    # angle ψ to the radial tetrad axis ê_r in `sky_params[1:4]` — writes
    # pure transparency and exits without integrating a single step.
    gate = 0.0f0
    if LAYER
        gate = sky_params[7]
        if r > gate
            cψ = clamp(p_t * sky_params[1] + px * sky_params[2] +
                       py * sky_params[3] + pz * sky_params[4],
                       -1.0f0, 1.0f0)
            Nf = Float32(size(fan, 2))
            tf = acos(cψ) * (Nf - 1.0f0) / Float32(pi)
            k0 = clamp(unsafe_trunc(Int32, tf), Int32(0), unsafe_trunc(Int32, Nf) - Int32(2))
            frac_f = tf - Float32(k0)
            esc = fan[1, k0 + 1] + frac_f * (fan[1, k0 + 2] - fan[1, k0 + 1])
            rmin = fan[4, k0 + 1] + frac_f * (fan[4, k0 + 2] - fan[4, k0 + 1])
            if esc > 0.999f0 && rmin > gate
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
    # Volume march stride: sample the gas every Nth integration step with
    # N× path weight (default 2). Coarser strides are the quality/speed
    # knob for cameras inside the slab — geodesics stay exact.
    vol_mstep = clamp(unsafe_trunc(Int32, vol_params[9]), Int32(1), Int32(16))
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
        if r < 3.2f0 * M   # strong-field zone: kill checks live only here
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
        hcap = (VOL && r2 < vol_rb2) ? 2.0f0 : 8.0f0
        h = dt * min(max(0.16f0 * r / M, 1.0f0), hcap)

        xp = x; yp = y; zp = z
        pxp = px; pyp = py; pzp = pz

        k1 = ks_rhs_mtl(x, y, z, px, py, pz, p_t, M)
        if NB > 0
            ell += h * sqrt(k1[1] * k1[1] + k1[2] * k1[2] + k1[3] * k1[3])
        end

        # Volumetric disc: sample the density grid and accumulate
        # Doppler-shaded emission/absorption. Sampled every 2nd step (with
        # doubled path weight) — gas structure is much coarser than the
        # integration step.
        if VOL && alpha > 0.003f0 && stepi % vol_mstep == 0 && r2 < vol_rb2
            if abs(z) < vol_zmax
                s_cyl = sqrt(x * x + y * y)
                if s_cyl > 1.0f-6
                    φv = atan(y, x)
                    ρ = sample_volume_mtl(vol, vol_params, s_cyl, φv, z)
                    if ρ > 1.0f-4
                        vlen = max(sqrt(k1[1] * k1[1] + k1[2] * k1[2] +
                                        k1[3] * k1[3]), 1.0f-20)
                        ds = Float32(vol_mstep) * h * vlen

                        R = s_cyl / (2.0f0 * M)
                        T_emit = exp(10.034259f0 - 0.375f0 * log(max(R * R, 1.0f-6)))
                        v_mag = clamp(0.70710678f0 / sqrt(max(R - 1.0f0, 0.1f0)),
                                      0.0f0, 0.999f0)
                        # Keplerian flow ϕ̂ = (−y, x, 0)/s against the photon
                        # coordinate velocity k1[1:3].
                        vdotn = v_mag * (-y * k1[1] + x * k1[2]) / (s_cyl * vlen)
                        gam = 1.0f0 / sqrt(1.0f0 - clamp(v_mag * v_mag, 0.0f0, 0.99f0))
                        Rs = r / (2.0f0 * M)
                        opzg = 1.0f0 / sqrt(max(1.0f0 - 1.0f0 / max(Rs, 1.0f0), 0.01f0))
                        opz = max(gam * (1.0f0 + vdotn) * opzg, 0.1f0)
                        T_obs = T_emit * scam / opz
                        inten = 100.0f0 / (exp(29622.4f0 / max(T_obs, 1.0f0)) - 1.0f0)

                        frac = (clamp(T_obs, lut_tmin, lut_tmax) - lut_tmin) /
                               (lut_tmax - lut_tmin)
                        li = clamp(unsafe_trunc(Int32, frac * (lut_size - 1.0f0) + 0.5f0) +
                                   Int32(1), Int32(1), unsafe_trunc(Int32, lut_size))
                        col_r = bb_lut[1, li]
                        col_g = bb_lut[2, li]
                        col_b = bb_lut[3, li]

                        tau = vol_opac * ρ * ds
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
                        if alpha < 0.003f0
                            break   # transmittance exhausted
                        end
                    end
                end
            end
        end

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

        # A non-finite ray can never satisfy the exit tests and would reach
        # the background sampler as NaN, trapping the kernel. Paint it black
        # and stop. (KS coordinates make this far rarer than the spherical
        # chart ever did.)
        if !(x == x) || !(z == z) || !(px == px)
            hit_horizon = true
            break
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
                v_mag = clamp(0.70710678f0 / sqrt(max(R - 1.0f0, 0.1f0)),
                              0.0f0, 0.999f0)
                vdotn = v_mag * (-yh * vx + xh * vy) / (s * plen)
                gam = 1.0f0 / sqrt(1.0f0 - clamp(v_mag * v_mag, 0.0f0, 0.99f0))
                opzg = 1.0f0 / sqrt(max(1.0f0 - 1.0f0 / max(R, 1.0f0), 0.01f0))
                opz = max(gam * (1.0f0 + vdotn) * opzg, 0.1f0)
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
    # (near-)critical or horizon-hugging: treat them as black too.
    rf = max(sqrt(x * x + y * y + z * z), 1.0f-6)
    if hit_horizon || rf < 4.0f0 * M
        if NB == 0   # bucketed emission was already written during the march
            out[1, i, j] += weight * acc_r
            out[2, i, j] += weight * acc_g
            out[3, i, j] += weight * acc_b
        end
    else
        # Escaped rays show the background sky attenuated by any disc gas
        # along the way. Sample by the asymptotic momentum direction, not the
        # escape position: position sampling parallax-shifts stars by up to
        # ~b/r_escape radians for disc-grazing rays.
        inv_rf = 1.0f0 / rf
        ff = 2.0f0 * M * inv_rf
        κf = (x * px + y * py + z * pz) * inv_rf
        c1f = ff * (-p_t + κf) * inv_rf
        vx = px - c1f * x
        vy = py - c1f * y
        vz = pz - c1f * z
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
                                       star_params, bb_lut)
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
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
28-float camera parameter block: position, fov, the KS tetrad, and the
projection. `fisheye_deg > 0` selects an equidistant fisheye with that
vertical half-angle at the image's top edge (pixel radius ∝ view angle, so
fields wider than 180° render cleanly — a rectilinear pinhole caps below
180° at any focal length).
"""
function _ks_cam_params(cam::Camera, M::Float64; fisheye_deg::Real=0.0,
                        focus_dist::Real=1.0,
                        beta::SVector{3,Float64}=SVector(0.0, 0.0, 0.0))
    u4, Ef, Er, Eu = ks_camera_tetrad(cam.pos, cam.fwd, cam.right,
                                      cam.up_local, M; beta=beta)
    return Float32[cam.pos[1], cam.pos[2], cam.pos[3], cam.fov_factor,
                   Ef..., Er..., Eu..., u4...,
                   fisheye_deg > 0 ? 1.0 : 0.0, deg2rad(max(fisheye_deg, 0.0)),
                   0.0, 0.0, focus_dist,
                   0.0, 0.0, 0.0]   # lens stratum origin/width, radius, seed
end

"""
    render_preview_mtl(ctx::MetalPreviewContext, cam::Camera,
                       spacetime::Schwarzschild)

Render one preview frame on the GPU using `ctx`.  Returns a `width × height`
`Matrix{RGBf}` suitable for display.

    render_preview_mtl!(img, host, ctx, cam, spacetime)

In-place variant for render loops: writes into a caller-owned `img`
(`Matrix{RGBf}(undef, width, height)`) via the caller-owned staging buffer
`host` (`Array{Float32,3}(undef, 3, width, height)`), so a flight loop
allocates nothing per frame.
"""
function render_preview_mtl(ctx::MetalPreviewContext, cam::Camera,
                            spacetime::Schwarzschild; fisheye_deg::Real=0.0,
                            relativistic::Bool=false,
                            beta::SVector{3,Float64}=SVector(0.0, 0.0, 0.0))
    img = Matrix{RGBf}(undef, ctx.width, ctx.height)
    host = Array{Float32,3}(undef, 3, ctx.width, ctx.height)
    return render_preview_mtl!(img, host, ctx, cam, spacetime;
                               fisheye_deg=fisheye_deg,
                               relativistic=relativistic, beta=beta)
end

function render_preview_mtl!(img::Matrix{RGBf}, host::Array{Float32,3},
                             ctx::MetalPreviewContext, cam::Camera,
                             spacetime::Schwarzschild; fisheye_deg::Real=0.0,
                             relativistic::Bool=false,
                             beta::SVector{3,Float64}=SVector(0.0, 0.0, 0.0),
                             band_rows::Int=0,
                             on_band::Union{Nothing,Function}=nothing)
    _trace_preview_gpu!(ctx, cam, spacetime; fisheye_deg=fisheye_deg,
                        relativistic=relativistic, beta=beta,
                        band_rows=band_rows, on_band=on_band)
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
                             spacetime::Schwarzschild; fisheye_deg::Real=0.0,
                             relativistic::Bool=false,
                             beta::SVector{3,Float64}=SVector(0.0, 0.0, 0.0),
                             band_rows::Int=0,
                             on_band::Union{Nothing,Function}=nothing)
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
    copyto!(ctx.cam_params, _ks_cam_params(cam, spacetime.M;
                                           fisheye_deg=fisheye_deg, beta=beta))
    copyto!(ctx.spacetime_params,
            Float32[M, r_band, r_escape, relativistic ? 1.0 : 0.0])

    fill!(ctx.out_gpu, 0.0f0)
    if band_rows <= 0 || on_band === nothing
        _launch_trace!(ctx, ctx.out_gpu, ctx.cam_params, ctx.spacetime_params,
                       ctx.width, ctx.height, nmax, dt,
                       0.5f0, 0.5f0, 1.0f0, 0, ctx.height)
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
                           nmax, dt, 0.5f0, 0.5f0, 1.0f0, row0, rows)
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
                        substride::Int=1, subx::Int=0, suby::Int=0)
    von = ctx.vol_on[]
    fan_b = fan === nothing ? _dummy_fan() : fan
    skyp_b = sky_params === nothing ? _dummy_skyp() : sky_params
    kernels = ctx.kernel[]::Dict{Any,Any}
    key = (von, nb, layer)
    if !haskey(kernels, key)
        kernels[key] = @metal launch=false trace_kernel_mtl!(
            out, ctx.bg_gpu, ctx.bb_lut, ctx.vol_gpu, ctx.vol_params,
            ctx.star_params, cam_params, spacetime_params, ctx.disc_params,
            fan_b, skyp_b,
            width, height, nmax, dt, ju, jv, weight, row0, rows,
            substride, subx, suby, Val(von), Val(nb), Val(layer))
    end
    kernel = kernels[key]
    n = width * rows
    threads = min(kernel.pipeline.maxTotalThreadsPerThreadgroup, n)
    groups = cld(n, threads)
    kernel(out, ctx.bg_gpu, ctx.bb_lut, ctx.vol_gpu, ctx.vol_params,
           ctx.star_params, cam_params, spacetime_params, ctx.disc_params,
           fan_b, skyp_b,
           width, height, nmax, dt, ju, jv, weight, row0, rows,
           substride, subx, suby, Val(von), Val(nb), Val(layer);
           threads=threads, groups=groups)
    return nothing
end

"""
    render_depth_mtl(ctx, cam, spacetime; width, height, samples=2, dt=0.02,
                     nbuckets=10, fisheye_deg=0.0, relativistic=false, beta=0)

Pinhole draft render with emission separated into `nbuckets` path-length
buckets (log-spaced over 1.5M–120M; the last bucket holds the escaped
background at infinity). Returns a `(3*nbuckets, width, height)`
`Array{Float32,3}` — feed to [`lens_post`](@ref) for depth-of-field as a post
operation at any aperture/focus, without re-rendering.
"""
function render_depth_mtl(ctx::MetalPreviewContext, cam::Camera,
                          spacetime::Schwarzschild;
                          width::Int=1920, height::Int=1080,
                          samples::Int=2, dt::Real=0.02,
                          nbuckets::Int=10, fisheye_deg::Real=0.0,
                          rng::Random.AbstractRNG=Random.default_rng(),
                          relativistic::Bool=false,
                          beta::SVector{3,Float64}=SVector(0.0, 0.0, 0.0))
    nbuckets >= 3 || throw(ArgumentError("nbuckets must be ≥ 3"))
    dt32 = Float32(dt)
    M = Float32(spacetime.M)
    r_band = Float32(2.05 * spacetime.M)
    r_escape = Float32(ctx.r_escape_factor * max(norm(cam.pos),
                                                 15.0 * spacetime.M))
    nmax = min(max(ctx.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt32)),
               40_000)
    cam_params = MtlVector{Float32}(undef, 28)
    spacetime_params = MtlVector{Float32}(undef, 4)
    copyto!(cam_params, _ks_cam_params(cam, spacetime.M;
                                       fisheye_deg=fisheye_deg, beta=beta))
    copyto!(spacetime_params,
            Float32[M, r_band, r_escape, relativistic ? 1.0 : 0.0])
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
                       row0, rows; nb=nbuckets)
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
returning `(cam::Camera, beta::SVector{3,Float64})`. Each of the `samples²`
supersampling passes then renders from its own stratified shutter time — the
passes double as the temporal samples, exactly as they double as the aperture
samples for DoF, so the blur costs nothing extra. The positional `cam`/`beta`
still set the escape radius and are the nominal (shutter-centre) pose.
"""
function render_draft_mtl(ctx::MetalPreviewContext, cam::Camera,
                          spacetime::Schwarzschild;
                          width::Int=1920, height::Int=1080,
                          samples::Int=2, dt::Real=0.02,
                          rng::Random.AbstractRNG=Random.default_rng(),
                          progress::Union{Function,Nothing}=nothing,
                          fisheye_deg::Real=0.0,
                          aperture_world::Real=0.0, focus_dist::Real=1.0,
                          relativistic::Bool=false,
                          beta::SVector{3,Float64}=SVector(0.0, 0.0, 0.0),
                          camera_at::Union{Function,Nothing}=nothing)
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
    cam_params = MtlVector{Float32}(undef, 28)
    spacetime_params = MtlVector{Float32}(undef, 4)
    base_params = _ks_cam_params(cam, spacetime.M; fisheye_deg=fisheye_deg,
                                 focus_dist=focus_dist, beta=beta)
    copyto!(cam_params, base_params)
    copyto!(spacetime_params,
            Float32[M, r_band, r_escape, relativistic ? 1.0 : 0.0])
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
    offsets = jittered_grid(samples; rng=rng)
    ndispatch = length(offsets) * ntiles
    done = 0
    # Depth of field: each pass owns one aperture stratum (shuffled so lens
    # strata pair randomly with the pixel-jitter strata) and every pixel
    # hashes its own point inside it — see the kernel's stratified-lens block.
    lens_perm = Random.randperm(rng, samples^2)
    # Motion blur: each pass likewise owns one stratified shutter time,
    # permuted independently so time strata pair randomly with the others.
    time_perm = Random.randperm(rng, samples^2)
    for (pass, (du, dv)) in enumerate(offsets)
        if camera_at !== nothing
            s = (time_perm[pass] - 1 + rand(rng)) / samples^2
            cam_s, beta_s = camera_at(s)
            base_params = _ks_cam_params(cam_s, spacetime.M;
                                         fisheye_deg=fisheye_deg,
                                         focus_dist=focus_dist, beta=beta_s)
        end
        if use_dof
            m = lens_perm[pass] - 1
            base_params[23] = Float32((m ÷ samples) / samples)
            base_params[24] = Float32((m % samples) / samples)
            base_params[26] = Float32(1.0 / samples)
            base_params[27] = Float32(aperture_world / 2.0)
            base_params[28] = Float32(pass)
        end
        (use_dof || camera_at !== nothing) && copyto!(cam_params, base_params)
        for t in 0:(ntiles - 1)
            row0 = t * rows_per_tile
            rows = min(rows_per_tile, height - row0)
            _launch_trace!(ctx, out, cam_params, spacetime_params,
                           width, height, nmax, dt32,
                           Float32(du), Float32(dv), weight, row0, rows)
            Metal.synchronize()
            done += 1
            isnothing(progress) || progress(done / ndispatch)
        end
    end

    return _download_rgb(out, width, height)
end

function render_draft_mtl(ctx::MetalPreviewContext, cam::ThinLensCamera,
                          spacetime::Schwarzschild; kwargs...)
    fov = (cam.sensor_width / 2.0) / cam.focal_length
    pinhole = Camera(cam.pos, cam.pos + cam.fwd, cam.up_local, fov)
    # Same world-space aperture as the CPU get_ray: diameter = focus/f_number.
    ap = cam.aperture * cam.focus_distance / cam.focal_length
    return render_draft_mtl(ctx, pinhole, spacetime; kwargs...,
                            aperture_world=ap, focus_dist=cam.focus_distance)
end

function render_preview_mtl(ctx::MetalPreviewContext, cam::ThinLensCamera,
                            spacetime::Schwarzschild)
    fov = (cam.sensor_width / 2.0) / cam.focal_length
    pinhole = Camera(cam.pos, cam.pos + cam.fwd, cam.up_local, fov)
    return render_preview_mtl(ctx, pinhole, spacetime)
end

function render_preview_mtl!(img::Matrix{RGBf}, host::Array{Float32,3},
                             ctx::MetalPreviewContext, cam::ThinLensCamera,
                             spacetime::Schwarzschild; kwargs...)
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
    sky_composite_kernel!(out, bg, fan, layer, cam_params, spacetime_params,
                          sky_params, width, height, lw, lh)

Per display pixel: build the pixel ray exactly like the trace kernel, find
its local angle ψ to the radial axis (`ê_r` in `sky_params[1:4]`), look up
the exact deflection in `fan` and sample the starmap along the asymptotic
direction (black when captured — the shadow at native resolution), then
composite the premultiplied disc/gas `layer` (bilinear, `lw × lh`) over it.
Near-critical fan entries (neighbours disagreeing in escape or direction)
fall back to the nearest entry — a sub-pixel zone at the photon ring.
"""
function sky_composite_kernel!(out, bg, fan, fine, layer, cam_params,
                               spacetime_params, sky_params,
                               width, height, lw, lh, ju, jv, accumulate,
                               row0, rows)
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
    esc, cθ, sθ = _fan_dir(fan, fine, sky_params[9], sky_params[10], acos(cψ))

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
        θbg = acos(clamp(dz / max(sqrt(dx * dx + dy * dy + dz * dz), 1.0f-9),
                         -1.0f0, 1.0f0))
        φbg = atan(dy, dx)
        sky_r, sky_g, sky_b = sample_background_mtl(bg, θbg, φbg,
                                                    size(bg, 2), size(bg, 3))
        if scam != 1.0f0
            sky_r *= 57.4f0 / (exp(4.067f0 / scam) - 1.0f0)
            sky_g *= 90.2f0 / (exp(4.513f0 / scam) - 1.0f0)
            sky_b *= 206.5f0 / (exp(5.335f0 / scam) - 1.0f0)
        end
    end

    # Composite the (lower-resolution) premultiplied disc layer over the sky.
    fx = (Float32(i) - 0.5f0) * Float32(lw) / Float32(width) + 0.5f0
    fy = (Float32(j) - 0.5f0) * Float32(lh) / Float32(height) + 0.5f0
    x0 = clamp(unsafe_trunc(Int32, floor(fx)), Int32(1), Int32(lw - 1))
    y0 = clamp(unsafe_trunc(Int32, floor(fy)), Int32(1), Int32(lh - 1))
    tx = clamp(fx - Float32(x0), 0.0f0, 1.0f0)
    ty = clamp(fy - Float32(y0), 0.0f0, 1.0f0)
    x1 = x0 + Int32(1); y1 = y0 + Int32(1)
    w00 = (1.0f0 - tx) * (1.0f0 - ty); w10 = tx * (1.0f0 - ty)
    w01 = (1.0f0 - tx) * ty;           w11 = tx * ty
    lr = layer[1, x0, y0] * w00 + layer[1, x1, y0] * w10 +
         layer[1, x0, y1] * w01 + layer[1, x1, y1] * w11
    lg = layer[2, x0, y0] * w00 + layer[2, x1, y0] * w10 +
         layer[2, x0, y1] * w01 + layer[2, x1, y1] * w11
    lb = layer[3, x0, y0] * w00 + layer[3, x1, y0] * w10 +
         layer[3, x0, y1] * w01 + layer[3, x1, y1] * w11
    la = layer[4, x0, y0] * w00 + layer[4, x1, y0] * w10 +
         layer[4, x0, y1] * w01 + layer[4, x1, y1] * w11

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
const _COMPOSITE_KERNEL = Ref{Any}(nothing)
const _DUMMY_FAN = Ref{Any}(nothing)
const _DUMMY_SKYP = Ref{Any}(nothing)
_dummy_fan() = _DUMMY_FAN[] === nothing ?
    (_DUMMY_FAN[] = MtlArray(zeros(Float32, 4, 2))) : _DUMMY_FAN[]
_dummy_skyp() = _DUMMY_SKYP[] === nothing ?
    (_DUMMY_SKYP[] = MtlArray(zeros(Float32, 8))) : _DUMMY_SKYP[]

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
    MtlVector{Float32}(undef, 28))

"""
    update_sky_fan!(sky::SkyFanState, ctx, pos, spacetime; gate, dt=0.02)

Rebuild the deflection fan for a camera at `pos`: integrate `n` exact
geodesics from radius `‖pos‖` (a per-frame cost of roughly one image
column). Also refreshes `sky_params`: the radial tetrad axis ê_r at `pos`,
`(M, r_escape)`, and the gas gate radius.
"""
function update_sky_fan!(sky::SkyFanState, ctx::MetalPreviewContext,
                         pos::SVector{3,Float64}, spacetime::Schwarzschild;
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
                             spacetime::Schwarzschild;
                             fisheye_deg::Real=0.0, relativistic::Bool=false,
                             dt::Real=Float64(ctx.dt), trace_layer::Bool=true,
                             substride::Int=1, subx::Int=0, suby::Int=0,
                             ju::Real=0.5, jv::Real=0.5,
                             accumulate::Bool=false,
                             row0::Int=0, rows::Int=-1)
    M = Float32(spacetime.M)
    r_band = Float32(2.05 * spacetime.M)
    r_escape = Float32(ctx.r_escape_factor * max(norm(cam.pos),
                                                 15.0 * spacetime.M))
    nmax = min(max(ctx.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt)),
               20_000)
    copyto!(ctx.cam_params, _ks_cam_params(cam, spacetime.M;
                                           fisheye_deg=fisheye_deg))
    copyto!(ctx.spacetime_params,
            Float32[M, r_band, r_escape, relativistic ? 1.0 : 0.0])

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
                       substride=substride, subx=subx, suby=suby)
    end
    # With trace_layer=false the caller keeps `layer_out` pre-filled with
    # α = 1 (fully transparent): the frame is the fan-driven sky alone.

    acc = accumulate ? 1.0f0 : 0.0f0
    if _COMPOSITE_KERNEL[] === nothing
        _COMPOSITE_KERNEL[] = @metal launch=false sky_composite_kernel!(
            comp_out, ctx.bg_gpu, sky.fan, sky.fine, layer_out,
            ctx.cam_params, ctx.spacetime_params, sky.sky_params,
            width, height, lw, lh, Float32(ju), Float32(jv), acc, row0, rows)
    end
    kern = _COMPOSITE_KERNEL[]
    n = width * rows
    threads = min(kern.pipeline.maxTotalThreadsPerThreadgroup, n)
    kern(comp_out, ctx.bg_gpu, sky.fan, sky.fine, layer_out, ctx.cam_params,
         ctx.spacetime_params, sky.sky_params, width, height, lw, lh,
         Float32(ju), Float32(jv), acc, row0, rows;
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

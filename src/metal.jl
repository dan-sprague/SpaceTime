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
    cam_params = MtlVector{Float32}(undef, 20)
    spacetime_params = MtlVector{Float32}(undef, 3)
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
                                    volume.opacity_scale, 1.0f0])
    end
    return MetalPreviewContext(bg_gpu, out_gpu, cam_params, spacetime_params,
                               disc_params, bb_lut, vol_gpu, vol_params,
                               width, height, Float32(dt),
                               nmax, Float32(r_escape_factor),
                               !isnothing(volume),
                               Base.RefValue{Bool}(!isnothing(volume)),
                               Base.RefValue{Any}(Dict{Bool,Any}()))
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
function trace_kernel_mtl!(out, bg, bb_lut, vol, vol_params, cam_params,
                           spacetime_params, disc_params, width, height,
                           nmax, dt, jitter_u, jitter_v, weight, row0, rows,
                           ::Val{VOL}) where {VOL}
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

    # Sensor coordinate with subpixel jitter.
    half_h = Float32(height) / 2.0f0
    u = (Float32(i) - 1.0f0 + jitter_u - Float32(width) / 2.0f0) / half_h
    v = (Float32(j) - 1.0f0 + jitter_v - Float32(height) / 2.0f0) / half_h

    # Pixel direction as unit coefficients on the camera tetrad axes.
    dx_local = u * fov
    dy_local = v * fov
    ν = sqrt(dx_local * dx_local + dy_local * dy_local + 1.0f0)
    cr = dx_local / ν
    cu = dy_local / ν
    cf = 1.0f0 / ν

    # Received photon p = ω(u + n); trace q = n − u, i.e. backward in time
    # (regular through the horizon in both directions). Lower the index:
    # p_μ = η_μν q^ν + f l_μ (l_ν q^ν) with l_μ = (1, x̂).
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
    for stepi in 1:nmax
        r2 = x * x + y * y + z * z
        r = sqrt(r2)
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
        # nearly-flat travel legs. Capped at 2× inside the gas volume so
        # turbulence stays sampled at feature scale, 8× otherwise.
        hcap = (VOL && r2 < vol_rb2) ? 2.0f0 : 8.0f0
        h = dt * min(max(0.16f0 * r / M, 1.0f0), hcap)

        xp = x; yp = y; zp = z
        pxp = px; pyp = py; pzp = pz

        k1 = ks_rhs_mtl(x, y, z, px, py, pz, p_t, M)

        # Volumetric disc: sample the density grid and accumulate
        # Doppler-shaded emission/absorption. Sampled every 2nd step (with
        # doubled path weight) — gas structure is much coarser than the
        # integration step.
        if VOL && alpha > 0.003f0 && stepi % 2 == 0 && r2 < vol_rb2
            if abs(z) < vol_zmax
                s_cyl = sqrt(x * x + y * y)
                if s_cyl > 1.0f-6
                    φv = atan(y, x)
                    ρ = sample_volume_mtl(vol, vol_params, s_cyl, φv, z)
                    if ρ > 1.0f-4
                        vlen = max(sqrt(k1[1] * k1[1] + k1[2] * k1[2] +
                                        k1[3] * k1[3]), 1.0f-20)
                        ds = 2.0f0 * h * vlen

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
                        T_obs = T_emit / opz
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
                        acc_r += w * col_r
                        acc_g += w * col_g
                        acc_b += w * col_b
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
                T_obs = T_emit / opz
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
                acc_r += w * col_r
                acc_g += w * col_g
                acc_b += w * col_b
                alpha *= (1.0f0 - opacity)
            end
        end
    end

    W = size(bg, 2)
    H = size(bg, 3)

    # Rays that ran out of steps while still deep in the strong field are
    # (near-)critical or horizon-hugging: treat them as black too.
    rf = max(sqrt(x * x + y * y + z * z), 1.0f-6)
    if hit_horizon || rf < 4.0f0 * M
        out[1, i, j] += weight * acc_r
        out[2, i, j] += weight * acc_g
        out[3, i, j] += weight * acc_b
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
        out[1, i, j] += weight * (acc_r + alpha * r_col)
        out[2, i, j] += weight * (acc_g + alpha * g_col)
        out[3, i, j] += weight * (acc_b + alpha * b_col)
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""20-float camera parameter block: position, fov, and the KS tetrad."""
function _ks_cam_params(cam::Camera, M::Float64)
    u4, Ef, Er, Eu = ks_camera_tetrad(cam.pos, cam.fwd, cam.right,
                                      cam.up_local, M)
    return Float32[cam.pos[1], cam.pos[2], cam.pos[3], cam.fov_factor,
                   Ef..., Er..., Eu..., u4...]
end

"""
    render_preview_mtl(ctx::MetalPreviewContext, cam::Camera,
                       spacetime::Schwarzschild)

Render one preview frame on the GPU using `ctx`.  Returns a `width × height`
`Matrix{RGBf}` suitable for display.
"""
function render_preview_mtl(ctx::MetalPreviewContext, cam::Camera,
                            spacetime::Schwarzschild)
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
    copyto!(ctx.cam_params, _ks_cam_params(cam, spacetime.M))
    copyto!(ctx.spacetime_params, Float32[M, r_band, r_escape])

    fill!(ctx.out_gpu, 0.0f0)
    _launch_trace!(ctx, ctx.out_gpu, ctx.cam_params, ctx.spacetime_params,
                   ctx.width, ctx.height, nmax, dt,
                   0.5f0, 0.5f0, 1.0f0, 0, ctx.height)
    return _download_rgb(ctx.out_gpu, ctx.width, ctx.height)
end

"""
Compile-once launch of `trace_kernel_mtl!`. `cam_params`/`spacetime_params`
are passed explicitly so a draft render can use its own buffers and run
concurrently with preview frames that update the context's buffers.
"""
function _launch_trace!(ctx::MetalPreviewContext, out, cam_params,
                        spacetime_params, width::Int, height::Int,
                        nmax::Int, dt::Float32, ju::Float32, jv::Float32,
                        weight::Float32, row0::Int, rows::Int)
    von = ctx.vol_on[]
    kernels = ctx.kernel[]::Dict{Bool,Any}
    if !haskey(kernels, von)
        kernels[von] = @metal launch=false trace_kernel_mtl!(
            out, ctx.bg_gpu, ctx.bb_lut, ctx.vol_gpu, ctx.vol_params,
            cam_params, spacetime_params, ctx.disc_params, width, height,
            nmax, dt, ju, jv, weight, row0, rows, Val(von))
    end
    kernel = kernels[von]
    n = width * rows
    threads = min(kernel.pipeline.maxTotalThreadsPerThreadgroup, n)
    groups = cld(n, threads)
    kernel(out, ctx.bg_gpu, ctx.bb_lut, ctx.vol_gpu, ctx.vol_params,
           cam_params, spacetime_params, ctx.disc_params, width, height,
           nmax, dt, ju, jv, weight, row0, rows, Val(von);
           threads=threads, groups=groups)
    return nothing
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
"""
function render_draft_mtl(ctx::MetalPreviewContext, cam::Camera,
                          spacetime::Schwarzschild;
                          width::Int=1920, height::Int=1080,
                          samples::Int=2, dt::Real=0.02,
                          rng::Random.AbstractRNG=Random.default_rng(),
                          progress::Union{Function,Nothing}=nothing)
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
    cam_params = MtlVector{Float32}(undef, 20)
    spacetime_params = MtlVector{Float32}(undef, 3)
    copyto!(cam_params, _ks_cam_params(cam, spacetime.M))
    copyto!(spacetime_params, Float32[M, r_band, r_escape])

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
    for (du, dv) in offsets
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
    return render_draft_mtl(ctx, pinhole, spacetime; kwargs...)
end

function render_preview_mtl(ctx::MetalPreviewContext, cam::ThinLensCamera,
                            spacetime::Schwarzschild)
    fov = (cam.sensor_width / 2.0) / cam.focal_length
    pinhole = Camera(cam.pos, cam.pos + cam.fwd, cam.up_local, fov)
    return render_preview_mtl(ctx, pinhole, spacetime)
end

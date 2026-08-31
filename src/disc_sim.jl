"""
    Live fluid-simulated accretion disc (Metal).

A 2D stable-fluids solver (Stam 1999) on the disc volume's own (log s, φ)
grid: semi-Lagrangian advection of a perturbation velocity field riding on
the Keplerian background shear Ω(s) = √(M/s³), curl-noise forcing to sustain
eddies, Jacobi pressure projection, then extrusion through the vertical
scale-height envelope into the same `(nr, nphi, nz)` density texture the ray
tracer samples (`ctx.vol_gpu`) — the trace kernel is untouched.

Everything is GPU-resident; one `step_sim!` is ~20 small kernel dispatches
over a 192×256 grid plus one 2.4M-cell extrusion, well under a preview frame.
Units: sim time is M-time; velocities are grid cells per M-time.
"""

mutable struct DiscFluidSim
    nr::Int
    nphi::Int
    nz::Int
    # Perturbation velocity (u = radial cells, v = azimuthal cells), ping-pong.
    vu::MtlMatrix{Float32};  vv::MtlMatrix{Float32}
    vu2::MtlMatrix{Float32}; vv2::MtlMatrix{Float32}
    # Surface density (ping-pong + MacCormack scratch) and relaxation target.
    dens::MtlMatrix{Float32}; dens2::MtlMatrix{Float32}; dens3::MtlMatrix{Float32}
    base::MtlMatrix{Float32}
    # Pressure solve scratch.
    prs::MtlMatrix{Float32}; prs2::MtlMatrix{Float32}; div::MtlMatrix{Float32}
    # Keplerian shear per radial row (φ-cells per M-time) and vertical envelope.
    shear::MtlVector{Float32}
    env::MtlMatrix{Float32}          # (nr, nz)
    t::Float32                       # sim clock, M-time
    forcing::Float32                 # eddy forcing amplitude
    drag::Float32                    # velocity damping per M-time
    relax::Float32                   # density relaxation toward `base` per M-time
    kernels::Dict{Symbol, Any}
end

"""
    DiscFluidSim(vol::DiscVolume, disc::AccretionDisc; M=1.0,
                 forcing=6.0, drag=0.15, relax=0.05)

Build the solver on `vol`'s grid. The initial density is `vol`'s midplane
slice, so the sim takes over seamlessly from the procedural turbulence.
"""
function DiscFluidSim(vol::DiscVolume, disc::AccretionDisc; M::Real=1.0,
                      forcing::Real=6.0, drag::Real=0.15, relax::Real=0.05)
    nr, nphi, nz = size(vol.density)
    kmid = (nz + 1) ÷ 2
    dens0 = Array{Float32}(vol.density[:, :, kmid])
    base0 = Matrix{Float32}(undef, nr, nphi)
    env0 = Matrix{Float32}(undef, nr, nz)
    shear0 = Vector{Float32}(undef, nr)
    log_in, log_out = vol.log_s_in, vol.log_s_out
    for i in 1:nr
        s = exp(log_in + (i - 1) / (nr - 1) * (log_out - log_in))
        base0[i, :] .= Float32(_disc_radial_profile(s, disc, M))
        H = vol.scale_height * s
        for k in 1:nz
            z = -vol.z_max + (k - 1) / (nz - 1) * 2 * vol.z_max
            env0[i, k] = exp(-z^2 / (2.0f0 * H * H))
        end
        # Ω = √(M/s³) rad/M-time → azimuthal grid cells per M-time.
        shear0[i] = Float32(sqrt(M / s^3) * nphi / (2π))
    end
    # Normalise the initial slice so extrusion peaks near the volume's own peak.
    peak = maximum(dens0)
    peak > 0 && (dens0 ./= peak)
    z2 = () -> MtlArray(zeros(Float32, nr, nphi))
    sim = DiscFluidSim(nr, nphi, nz,
                       z2(), z2(), z2(), z2(),
                       MtlArray(dens0), z2(), z2(), MtlArray(base0),
                       z2(), z2(), z2(),
                       MtlArray(shear0), MtlArray(env0),
                       0.0f0, Float32(forcing), Float32(drag), Float32(relax),
                       Dict{Symbol, Any}())
    return sim
end

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

@inline function _sim_hash(x::Int32, y::Int32, z::Int32)
    h = reinterpret(UInt32, x) * 0x8da6b343 ⊻
        reinterpret(UInt32, y) * 0xd8163841 ⊻
        reinterpret(UInt32, z) * 0xcb1ab31f
    h = h ⊻ (h >> 13)
    h = h * 0x9e3779b9
    h = h ⊻ (h >> 16)
    return Float32(h & 0x00ffffff) * 5.9604645f-8   # → [0,1)
end

"""Value noise on a virtual hashed lattice (x, y periodic in ny cells, z=time)."""
@inline function _sim_noise(x::Float32, y::Float32, z::Float32)
    ix = unsafe_trunc(Int32, floor(x)); fx = x - floor(x)
    iy = unsafe_trunc(Int32, floor(y)); fy = y - floor(y)
    iz = unsafe_trunc(Int32, floor(z)); fz = z - floor(z)
    fx = fx * fx * (3.0f0 - 2.0f0 * fx)
    fy = fy * fy * (3.0f0 - 2.0f0 * fy)
    fz = fz * fz * (3.0f0 - 2.0f0 * fz)
    i1 = ix + Int32(1); j1 = iy + Int32(1); k1 = iz + Int32(1)
    c00 = _sim_hash(ix, iy, iz) + fx * (_sim_hash(i1, iy, iz) - _sim_hash(ix, iy, iz))
    c10 = _sim_hash(ix, j1, iz) + fx * (_sim_hash(i1, j1, iz) - _sim_hash(ix, j1, iz))
    c01 = _sim_hash(ix, iy, k1) + fx * (_sim_hash(i1, iy, k1) - _sim_hash(ix, iy, k1))
    c11 = _sim_hash(ix, j1, k1) + fx * (_sim_hash(i1, j1, k1) - _sim_hash(ix, j1, k1))
    c0 = c00 + fy * (c10 - c00)
    c1 = c01 + fy * (c11 - c01)
    return c0 + fz * (c1 - c0)
end

"""Bilinear sample of a (nr, nphi) field: φ periodic, radius clamped."""
@inline function _sim_sample(f, x::Float32, y::Float32, nr::Int32, nphi::Int32)
    x = clamp(x, 1.0f0, Float32(nr))
    i0 = clamp(unsafe_trunc(Int32, floor(x)), Int32(1), nr - Int32(1))
    tx = x - Float32(i0)
    yw = y - Float32(nphi) * floor((y - 1.0f0) / Float32(nphi))
    j0 = unsafe_trunc(Int32, floor(yw))
    ty = yw - Float32(j0)
    j0a = mod(j0 - Int32(1), nphi) + Int32(1)
    j1a = mod(j0, nphi) + Int32(1)
    a = f[i0, j0a] + tx * (f[i0 + Int32(1), j0a] - f[i0, j0a])
    b = f[i0, j1a] + tx * (f[i0 + Int32(1), j1a] - f[i0, j1a])
    return a + ty * (b - a)
end

@inline function _sim_ij(idx::Int32, nr::Int32)
    i = (idx - Int32(1)) % nr + Int32(1)
    j = (idx - Int32(1)) ÷ nr + Int32(1)
    return i, j
end

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

"""Semi-Lagrangian advection of the velocity field under shear + itself."""
function k_advect_vel!(vu2, vv2, vu, vv, shear, dt::Float32,
                       nr::Int32, nphi::Int32)
    idx = unsafe_trunc(Int32, thread_position_in_grid().x)
    idx > nr * nphi && return
    i, j = _sim_ij(idx, nr)
    x = Float32(i) - dt * vu[i, j]
    y = Float32(j) - dt * (vv[i, j] + shear[i])
    vu2[i, j] = _sim_sample(vu, x, y, nr, nphi)
    vv2[i, j] = _sim_sample(vv, x, y, nr, nphi)
    return
end

"""Curl-noise forcing + drag + radial edge taper."""
function k_force!(vu, vv, t::Float32, dt::Float32, amp::Float32,
                  drag::Float32, nr::Int32, nphi::Int32)
    idx = unsafe_trunc(Int32, thread_position_in_grid().x)
    idx > nr * nphi && return
    i, j = _sim_ij(idx, nr)
    # ψ on a coarse lattice (~12-cell eddies), drifting in time.
    sc = 0.085f0
    x = Float32(i) * sc
    y = Float32(j) * sc
    z = t * 0.35f0
    e = 0.35f0
    dpy = _sim_noise(x, y + e, z) - _sim_noise(x, y - e, z)
    dpx = _sim_noise(x + e, y, z) - _sim_noise(x - e, y, z)
    # F = ∇×(ψ ẑ): divergence-free in the grid metric.
    fu = dpy / (2.0f0 * e)
    fv = -dpx / (2.0f0 * e)
    edge = min(Float32(i) - 1.0f0, Float32(nr - i)) * 0.125f0
    edge = clamp(edge, 0.0f0, 1.0f0)
    damp = exp(-drag * dt)
    vu[i, j] = (vu[i, j] + dt * amp * fu * edge) * damp * edge
    vv[i, j] = (vv[i, j] + dt * amp * fv * edge) * damp
    return
end

function k_div!(dv, vu, vv, nr::Int32, nphi::Int32)
    idx = unsafe_trunc(Int32, thread_position_in_grid().x)
    idx > nr * nphi && return
    i, j = _sim_ij(idx, nr)
    ip = min(i + Int32(1), nr); im = max(i - Int32(1), Int32(1))
    jp = mod(j, nphi) + Int32(1); jm = mod(j - Int32(2), nphi) + Int32(1)
    dv[i, j] = 0.5f0 * (vu[ip, j] - vu[im, j] + vv[i, jp] - vv[i, jm])
    return
end

function k_jacobi!(p2, p, dv, nr::Int32, nphi::Int32)
    idx = unsafe_trunc(Int32, thread_position_in_grid().x)
    idx > nr * nphi && return
    i, j = _sim_ij(idx, nr)
    ip = min(i + Int32(1), nr); im = max(i - Int32(1), Int32(1))
    jp = mod(j, nphi) + Int32(1); jm = mod(j - Int32(2), nphi) + Int32(1)
    p2[i, j] = 0.25f0 * (p[ip, j] + p[im, j] + p[i, jp] + p[i, jm] - dv[i, j])
    return
end

function k_subgrad!(vu, vv, p, nr::Int32, nphi::Int32)
    idx = unsafe_trunc(Int32, thread_position_in_grid().x)
    idx > nr * nphi && return
    i, j = _sim_ij(idx, nr)
    ip = min(i + Int32(1), nr); im = max(i - Int32(1), Int32(1))
    jp = mod(j, nphi) + Int32(1); jm = mod(j - Int32(2), nphi) + Int32(1)
    vu[i, j] -= 0.5f0 * (p[ip, j] - p[im, j])
    vv[i, j] -= 0.5f0 * (p[i, jp] - p[i, jm])
    return
end

"""Plain semi-Lagrangian advection of a scalar (dt may be negative)."""
function k_adv_scalar!(dst, src, vu, vv, shear, dt::Float32,
                       nr::Int32, nphi::Int32)
    idx = unsafe_trunc(Int32, thread_position_in_grid().x)
    idx > nr * nphi && return
    i, j = _sim_ij(idx, nr)
    x = Float32(i) - dt * vu[i, j]
    y = Float32(j) - dt * (vv[i, j] + shear[i])
    dst[i, j] = _sim_sample(src, x, y, nr, nphi)
    return
end

"""
MacCormack correction + relaxation + contrast forcing. `d1` is the forward
semi-Lagrangian result, `d2` the backward re-advection of `d1`; the error
estimate `(d − d2)/2` cancels most numerical diffusion. The result is clamped
to the bilinear-corner range of `d` at the backtraced point (the standard
limiter), relaxed toward `base`, and given a slow multiplicative log-noise
kick so filament contrast is continuously replenished.
"""
function k_maccormack!(dnew, d, d1, d2, vu, vv, shear, base, t::Float32,
                       dt::Float32, relax::Float32, nr::Int32, nphi::Int32)
    idx = unsafe_trunc(Int32, thread_position_in_grid().x)
    idx > nr * nphi && return
    i, j = _sim_ij(idx, nr)
    x = Float32(i) - dt * vu[i, j]
    y = Float32(j) - dt * (vv[i, j] + shear[i])
    # Bilinear-corner min/max of the source field at the backtraced point.
    xc = clamp(x, 1.0f0, Float32(nr))
    i0 = clamp(unsafe_trunc(Int32, floor(xc)), Int32(1), nr - Int32(1))
    yw = y - Float32(nphi) * floor((y - 1.0f0) / Float32(nphi))
    j0 = unsafe_trunc(Int32, floor(yw))
    j0a = mod(j0 - Int32(1), nphi) + Int32(1)
    j1a = mod(j0, nphi) + Int32(1)
    i1 = i0 + Int32(1)
    lo = min(min(d[i0, j0a], d[i1, j0a]), min(d[i0, j1a], d[i1, j1a]))
    hi = max(max(d[i0, j0a], d[i1, j0a]), max(d[i0, j1a], d[i1, j1a]))
    a = clamp(d1[i, j] + 0.5f0 * (d[i, j] - d2[i, j]), lo, hi)
    # Relax toward a *structured* target: the mean profile modulated by
    # slowly-evolving log-normal noise. Bounded (unlike a multiplicative
    # random walk), it continuously feeds filament contrast that the shear
    # then draws out into spirals.
    n = _sim_noise(Float32(i) * 0.11f0, Float32(j) * 0.11f0, t * 0.25f0)
    tgt = base[i, j] * exp(1.2f0 * (2.0f0 * n - 1.0f0))
    dnew[i, j] = max(a + relax * dt * (tgt - a), 0.0f0)
    return
end

"""Extrude the 2D surface density through the vertical envelope."""
function k_extrude!(vol3, d, env, nr::Int32, nphi::Int32, nz::Int32)
    idx = unsafe_trunc(Int32, thread_position_in_grid().x)
    idx > nr * nphi * nz && return
    i = (idx - Int32(1)) % nr + Int32(1)
    rest = (idx - Int32(1)) ÷ nr
    j = rest % nphi + Int32(1)
    k = rest ÷ nphi + Int32(1)
    vol3[i, j, k] = d[i, j] * env[i, k]
    return
end

# ---------------------------------------------------------------------------
# Host-side stepping
# ---------------------------------------------------------------------------

function _sim_launch!(sim::DiscFluidSim, key::Symbol, f::Function, n::Int,
                      args...)
    if !haskey(sim.kernels, key)
        sim.kernels[key] = @metal launch=false f(args...)
    end
    k = sim.kernels[key]
    threads = min(k.pipeline.maxTotalThreadsPerThreadgroup, n)
    k(args...; threads=threads, groups=cld(n, threads))
    return nothing
end

"""
    step_sim!(sim::DiscFluidSim, ctx::MetalPreviewContext; dt=0.05, jacobi=14)

Advance the fluid one step of `dt` (M-time) and write the extruded density
into `ctx.vol_gpu`. All work stays on the GPU; launches are queued in order,
so the next trace sees the updated field with no explicit sync.
"""
function step_sim!(sim::DiscFluidSim, ctx::MetalPreviewContext;
                   dt::Real=0.05, jacobi::Int=8)
    n = sim.nr * sim.nphi
    nr32, nphi32 = Int32(sim.nr), Int32(sim.nphi)
    dtf = Float32(dt)

    _sim_launch!(sim, :adv, k_advect_vel!, n, sim.vu2, sim.vv2, sim.vu, sim.vv,
                 sim.shear, dtf, nr32, nphi32)
    sim.vu, sim.vu2 = sim.vu2, sim.vu
    sim.vv, sim.vv2 = sim.vv2, sim.vv
    _sim_launch!(sim, :force, k_force!, n, sim.vu, sim.vv, sim.t, dtf,
                 sim.forcing, sim.drag, nr32, nphi32)
    _sim_launch!(sim, :div, k_div!, n, sim.div, sim.vu, sim.vv, nr32, nphi32)
    fill!(sim.prs, 0.0f0)
    for _ in 1:cld(jacobi, 2)
        _sim_launch!(sim, :jac, k_jacobi!, n, sim.prs2, sim.prs, sim.div, nr32, nphi32)
        _sim_launch!(sim, :jac, k_jacobi!, n, sim.prs, sim.prs2, sim.div, nr32, nphi32)
    end
    _sim_launch!(sim, :sub, k_subgrad!, n, sim.vu, sim.vv, sim.prs, nr32, nphi32)
    # MacCormack density advection: forward, backward, corrected combine.
    # The combine writes into dens3 while reading it as `d2` — safe because
    # each thread touches only its own [i, j] element of that buffer.
    _sim_launch!(sim, :advf, k_adv_scalar!, n, sim.dens2, sim.dens, sim.vu,
                 sim.vv, sim.shear, dtf, nr32, nphi32)
    _sim_launch!(sim, :advb, k_adv_scalar!, n, sim.dens3, sim.dens2, sim.vu,
                 sim.vv, sim.shear, -dtf, nr32, nphi32)
    _sim_launch!(sim, :mac, k_maccormack!, n, sim.dens3, sim.dens, sim.dens2,
                 sim.dens3, sim.vu, sim.vv, sim.shear, sim.base, sim.t, dtf,
                 sim.relax, nr32, nphi32)
    sim.dens, sim.dens3 = sim.dens3, sim.dens
    _sim_launch!(sim, :ext, k_extrude!, sim.nr * sim.nphi * sim.nz,
                 ctx.vol_gpu, sim.dens, sim.env, nr32, nphi32, Int32(sim.nz))
    sim.t += dtf
    return nothing
end

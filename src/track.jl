# ---------------------------------------------------------------------------
# Racing tracks as timelike worldlines
# ---------------------------------------------------------------------------
#
# A track is not an authored spline. It is the worldline of a ship with bounded
# thrust — a timelike solution of the equations of motion, produced by a
# sequence of maneuvers. Everything the camera needs falls out of that worldline:
#
#   * position and coordinate velocity  -> where the camera is, and its boost
#   * heading                           -> where it looks
#   * a Fermi-Walker transported triad  -> the roll
#
# The roll is the point of the last one. A gyroscope carried along a worldline
# near a spinning hole does not keep a fixed orientation: it precesses, from
# geodetic (de Sitter) precession and from frame dragging (Lense-Thirring).
# Fermi-Walker transport *is* that gyroscope, so the track's roll schedule is
# a measured physical effect rather than a number an artist picked.
#
# Nothing here is Kerr-specific. The dynamics are driven by numerical
# derivatives of whatever `metric`/`metric_inverse` the spacetime provides, so
# a superposed multi-hole metric drops in without touching this file. That
# costs ~6 metric evaluations per derivative, which is irrelevant: this
# integrates ONE worldline on the CPU, not a million rays on the GPU.

const _E4 = (SVector(1.0, 0.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0, 0.0),
             SVector(0.0, 0.0, 1.0, 0.0), SVector(0.0, 0.0, 0.0, 1.0))

_fd_step(q) = 1.0e-6 * max(sqrt(q[2]^2 + q[3]^2 + q[4]^2), 1.0)

"""
    ham(spacetime, q, p)

Super-Hamiltonian `H = ½ g^μν p_μ p_ν`. `H = 0` on a photon, `−½` on a unit-mass
timelike worldline; the *flow* is identical either way, which is why the photon
integrator and the ship integrator share this function.
"""
@inline ham(bh::AbstractSpacetime, q::SVector{4,Float64}, p::SVector{4,Float64}) =
    0.5 * dot(p, metric_inverse(bh, q) * p)

"""
    dham_dq(spacetime, q, p)

`∂H/∂q^μ` by central differences. The `t` derivative is skipped rather than
computed: these spacetimes are stationary, so it is exactly zero, and
evaluating it would only add round-off to a conserved energy.
"""
function dham_dq(bh::AbstractSpacetime, q::SVector{4,Float64},
                 p::SVector{4,Float64})
    h = _fd_step(q)
    d1 = (ham(bh, q + h * _E4[2], p) - ham(bh, q - h * _E4[2], p)) / (2h)
    d2 = (ham(bh, q + h * _E4[3], p) - ham(bh, q - h * _E4[3], p)) / (2h)
    d3 = (ham(bh, q + h * _E4[4], p) - ham(bh, q - h * _E4[4], p)) / (2h)
    return SVector(0.0, d1, d2, d3)
end

"""
    christoffel(spacetime, q) -> Γ[λ, μ, ν]

Christoffel symbols of the second kind, `Γ^λ_μν`, from central differences of
`g_μν`. Used only by the Fermi-Walker transport; the worldline itself needs
only `∂H/∂q`.
"""
function christoffel(bh::AbstractSpacetime, q::SVector{4,Float64})
    h = _fd_step(q)
    ginv = metric_inverse(bh, q)
    dg = zeros(4, 4, 4)                       # dg[σ, μ, ν] = ∂_σ g_μν
    for σ in 2:4                              # stationary: ∂_t g = 0
        gp = metric(bh, q + h * _E4[σ])
        gm = metric(bh, q - h * _E4[σ])
        @inbounds for μ in 1:4, ν in 1:4
            dg[σ, μ, ν] = (gp[μ, ν] - gm[μ, ν]) / (2h)
        end
    end
    Γ = zeros(4, 4, 4)
    @inbounds for λ in 1:4, μ in 1:4, ν in 1:4
        s = 0.0
        for σ in 1:4
            s += ginv[λ, σ] * (dg[μ, σ, ν] + dg[ν, σ, μ] - dg[σ, μ, ν])
        end
        Γ[λ, μ, ν] = 0.5 * s
    end
    return Γ
end

"""
    normalize_timelike(spacetime, q, v3) -> u

Unit-normalised 4-velocity from a **coordinate** 3-velocity `v3 = dx/dt`.
Throws if `(1, v3)` is not timelike at `q` — which is a real failure, not a
rounding one: it means the requested velocity is not achievable there (inside
the ergosphere, for instance, no worldline can hold station).
"""
function normalize_timelike(bh::AbstractSpacetime, q::SVector{4,Float64},
                            v3::SVector{3,Float64})
    w = SVector(1.0, v3[1], v3[2], v3[3])
    n2 = -dot(w, metric(bh, q) * w)
    n2 > 0 || throw(ArgumentError(
        "coordinate velocity $v3 is not timelike at r=$(sqrt(q[2]^2+q[3]^2+q[4]^2)) " *
        "(g(w,w) = $(-n2) ≥ 0)"))
    return w / sqrt(n2)
end

"""
    orthonormal_frame(spacetime, q, u, fwd_hint, up_hint) -> (ef, er, eu)

Orthonormal spatial triad of the observer `u`, Gram-Schmidt under the full
metric with **forward first**, so the heading is exact and any error is pushed
into the roll reference. Metric-agnostic, unlike `ks_camera_tetrad`, which
hardcodes the Schwarzschild `f = 2M/r`.
"""
function orthonormal_frame(bh::AbstractSpacetime, q::SVector{4,Float64},
                           u::SVector{4,Float64},
                           fwd_hint::SVector{3,Float64},
                           up_hint::SVector{3,Float64})
    g = metric(bh, q)
    gd(A, B) = dot(A, g * B)
    ef = SVector(0.0, fwd_hint[1], fwd_hint[2], fwd_hint[3])
    eu = SVector(0.0, up_hint[1], up_hint[2], up_hint[3])
    ef = ef + gd(ef, u) * u
    ef = ef / sqrt(max(gd(ef, ef), 1e-300))
    eu = eu + gd(eu, u) * u - gd(eu, ef) * ef
    eu = eu / sqrt(max(gd(eu, eu), 1e-300))
    # Right-handed third leg: er = ±(u ∧ ef ∧ eu) dual. Cheaper and more stable
    # to Gram-Schmidt a coordinate axis that is not already nearly in the span.
    seed = abs(ef[2]) < 0.9 ? _E4[2] : _E4[3]
    er = seed + gd(seed, u) * u - gd(seed, ef) * ef - gd(seed, eu) * eu
    er = er / sqrt(max(gd(er, er), 1e-300))
    return ef, er, eu
end

"""
    boost(spacetime, q, u, frame, dv) -> u'

Apply an impulsive `Δv` (components on the ship's own `frame`, `|Δv| < 1`) to
the 4-velocity `u`. This is a Lorentz boost in the ship's instantaneous rest
frame, so `g(u', u') = −1` exactly, not to first order.
"""
function boost(bh::AbstractSpacetime, q::SVector{4,Float64},
               u::SVector{4,Float64},
               frame::NTuple{3,SVector{4,Float64}}, dv::SVector{3,Float64})
    b2 = dot(dv, dv)
    b2 < 1.0 || throw(ArgumentError("|Δv| = $(sqrt(b2)) must be < 1"))
    b2 < 1e-18 && return u
    γ = 1.0 / sqrt(1.0 - b2)
    bE = dv[1] * frame[1] + dv[2] * frame[2] + dv[3] * frame[3]
    return γ * (u + bE)
end

"""
    fw_rhs(Γ, g, u, a, e)

`de^μ/dτ` for Fermi-Walker transport along a worldline with 4-velocity `u` and
proper acceleration `a`:

    De^μ/dτ = (u^μ a_ν − a^μ u_ν) e^ν

Checked against `e = u`, which must reproduce `a`. With `a = 0` this is plain
parallel transport — the gyroscope of a coasting ship.
"""
function fw_rhs(Γ::Array{Float64,3}, g::SMatrix{4,4,Float64},
                u::SVector{4,Float64}, a::SVector{4,Float64},
                e::SVector{4,Float64})
    ae = dot(a, g * e)
    ue = dot(u, g * e)
    out = zeros(MVector{4,Float64})
    @inbounds for μ in 1:4
        s = 0.0
        for α in 1:4, β in 1:4
            s -= Γ[μ, α, β] * u[α] * e[β]
        end
        out[μ] = s + u[μ] * ae - a[μ] * ue
    end
    return SVector(out)
end

# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

"""
    Burn(τ, dv)

An impulsive maneuver: at proper time `τ`, boost by `Δv` (units of c) along the
ship's own axes. Between burns the ship coasts on an exact geodesic, so a track
is a chain of free-fall arcs joined by thrust — which is both how real mission
design works and what the game means by "jumps are free-fall".
"""
struct Burn
    τ::Float64
    dv::SVector{3,Float64}
end
Burn(τ::Real, dv) = Burn(Float64(τ), SVector{3,Float64}(dv))

"""
    Track

Sampled worldline of the ship. `fwd` is the heading (normalised coordinate
velocity), `up` and `gyro` are the Fermi-Walker up and forward legs — the
ship's gyroscope — and `roll` is the angle of `up` about the heading.

`fwd` and `gyro` are different vectors and the difference is the physics: the
heading is where the ship is going, the gyro leg is where an undisturbed
gyroscope still points. Their separation over an orbit is the geodetic +
Lense-Thirring precession. For an equatorial orbit `up` is preserved by
symmetry and carries none of it, so the in-plane `gyro` leg is the one to
measure.
`vel` is `dx/dt`, ready for `Camera(...; velocity = ...)`.
"""
struct Track
    spacetime::AbstractSpacetime
    τ::Vector{Float64}
    t::Vector{Float64}
    pos::Vector{SVector{3,Float64}}
    vel::Vector{SVector{3,Float64}}
    fwd::Vector{SVector{3,Float64}}
    up::Vector{SVector{3,Float64}}
    gyro::Vector{SVector{3,Float64}}
    roll::Vector{Float64}
    # Full 4-vector tetrad: observer + Fermi-Walker triad. The 3-vector fields
    # above are spatial *parts*, which are convenient for building a Camera but
    # are NOT orthonormal under a Euclidean dot product — measuring an angle
    # between two of them that way is wrong by O(2M/r), which at r = 10M is an
    # 8% error. Anything angular must go through `frame_components`.
    u4::Vector{SVector{4,Float64}}
    ef4::Vector{SVector{4,Float64}}
    er4::Vector{SVector{4,Float64}}
    eu4::Vector{SVector{4,Float64}}
    speed::Vector{Float64}
    captured::Bool
end

Base.length(tr::Track) = length(tr.τ)

_spatial(v::SVector{4,Float64}) = SVector(v[2], v[3], v[4])

"""
    frame_components(spacetime, q, tetrad, V) -> SVector{3}

Components of a 4-vector `V` on the orthonormal spatial triad `tetrad`,
`cᵢ = g(V, eᵢ)`. Because the triad is orthonormal, ordinary Euclidean vector
algebra is exact in component space — dot products, cross products and angles
all mean what they look like. This is the only correct way to ask an angular
question about a track; doing it on the raw spatial parts is wrong by O(2M/r).
"""
frame_components(bh::AbstractSpacetime, q::SVector{4,Float64},
                 tet::NTuple{3,SVector{4,Float64}}, V::SVector{4,Float64}) =
    (g = metric(bh, q); SVector(dot(V, g * tet[1]), dot(V, g * tet[2]),
                                dot(V, g * tet[3])))

"""
    _roll_angle(spacetime, q, u, tetrad, heading)

Roll of the gyroscope's up-leg about the heading, measured from the spin-axis
reference (ẑ projected perpendicular to the heading). Computed in the tetrad's
component space, so it is the proper angle rather than a coordinate one.
Returns `NaN` where the reference degenerates — a heading along the spin axis —
rather than silently snapping, so a track that flies up the pole is visibly
wrong instead of quietly wrong.
"""
function _roll_angle(bh::AbstractSpacetime, q::SVector{4,Float64},
                     u::SVector{4,Float64},
                     tet::NTuple{3,SVector{4,Float64}},
                     heading::SVector{3,Float64})
    g = metric(bh, q)
    proj(V) = V + dot(V, g * u) * u              # remove the u component
    H = proj(SVector(0.0, heading[1], heading[2], heading[3]))
    nH = dot(H, g * H); nH > 1e-18 || return NaN
    ĥ = frame_components(bh, q, tet, H / sqrt(nH))
    ẑc = frame_components(bh, q, tet, proj(SVector(0.0, 0.0, 0.0, 1.0)))
    perp = ẑc - dot(ẑc, ĥ) * ĥ
    np = norm(perp); np < 1e-6 && return NaN
    ref_up = perp / np
    ref_right = cross(ĥ, ref_up)
    up = SVector(0.0, 0.0, 1.0)                  # eu is (0,0,1) in its own frame
    up = up - dot(up, ĥ) * ĥ
    nu = norm(up); nu < 1e-12 && return NaN
    up = up / nu
    return atan(dot(up, ref_right), dot(up, ref_up))
end

"""
    integrate_track(spacetime, pos0, v0, τmax; dτ, burns, up_hint)

Integrate the ship's worldline from `pos0` with coordinate 3-velocity `v0`,
applying `burns` along the way, and Fermi-Walker transport an orthonormal triad
alongside it. Stops early if the ship is captured.

The worldline and the triad are advanced by the same RK4 step, so the triad
stays orthonormal to the same order the trajectory is accurate — the
`track_frame_error` check in the test suite is what holds this honest.
"""
function integrate_track(bh::AbstractSpacetime, pos0::SVector{3,Float64},
                         v0::SVector{3,Float64}, τmax::Real;
                         dτ::Real = 0.05, burns::Vector{Burn} = Burn[],
                         up_hint::SVector{3,Float64} = SVector(0.0, 0.0, 1.0))
    q = SVector(0.0, pos0[1], pos0[2], pos0[3])
    u = normalize_timelike(bh, q, v0)
    p = metric(bh, q) * u

    fwd0 = norm(v0) > 1e-12 ? v0 / norm(v0) : SVector(1.0, 0.0, 0.0)
    ef, er, eu = orthonormal_frame(bh, q, u, fwd0, up_hint)

    rh = horizon_radius(bh)
    pend = sort(burns; by = b -> b.τ)
    bi = 1

    τs = Float64[]; ts = Float64[]; ps = SVector{3,Float64}[]
    vs = SVector{3,Float64}[]; fs = SVector{3,Float64}[]
    us = SVector{3,Float64}[]; gy = SVector{3,Float64}[]
    rl = Float64[]; sp = Float64[]
    u4 = SVector{4,Float64}[]; e1 = SVector{4,Float64}[]
    e2 = SVector{4,Float64}[]; e3 = SVector{4,Float64}[]

    function rhs(q, p, e1, e2, e3)
        ginv = metric_inverse(bh, q)
        g = metric(bh, q)
        uu = ginv * p
        Γ = christoffel(bh, q)
        a = SVector(0.0, 0.0, 0.0, 0.0)          # coasting between burns
        return (uu, -dham_dq(bh, q, p),
                fw_rhs(Γ, g, uu, a, e1),
                fw_rhs(Γ, g, uu, a, e2),
                fw_rhs(Γ, g, uu, a, e3))
    end

    captured = false
    τ = 0.0
    nsteps = max(1, round(Int, abs(τmax / dτ)))
    dτ = copysign(abs(dτ), τmax)   # τmax < 0 integrates into the past
    for n in 0:nsteps
        ginv = metric_inverse(bh, q)
        u = ginv * p
        # Apply any burns whose time has arrived, in the current ship frame.
        # The triad is boosted by the *same* Lorentz transformation rather than
        # rebuilt: Fermi-Walker transport is by definition non-rotating, so a
        # burn boosts the gyroscope without spinning it. Rebuilding here would
        # silently erase the accumulated precession, which is the one thing the
        # track exists to carry.
        while bi <= length(pend) && pend[bi].τ <= τ + 1e-12
            g = metric(bh, q)
            sv = _spatial(u)
            fr = orthonormal_frame(bh, q, u, sv / max(norm(sv), 1e-12), _spatial(eu))
            dv = pend[bi].dv
            b2 = dot(dv, dv)
            if b2 > 1e-18
                γ = 1.0 / sqrt(1.0 - b2)
                bE = dv[1] * fr[1] + dv[2] * fr[2] + dv[3] * fr[3]
                k = (γ - 1.0) / b2
                shift = k * bE + γ * u
                # e = Σ cᵢ frᵢ  ⟹  e' = e + (c·Δv) · shift
                bump(e) = e + (dv[1] * dot(e, g * fr[1]) +
                               dv[2] * dot(e, g * fr[2]) +
                               dv[3] * dot(e, g * fr[3])) * shift
                ef = bump(ef); er = bump(er); eu = bump(eu)
                u = γ * (u + bE)
                p = g * u
            end
            bi += 1
        end

        v3 = SVector(u[2], u[3], u[4]) / u[1]
        nv = norm(v3)
        h = nv > 1e-12 ? v3 / nv : SVector(1.0, 0.0, 0.0)
        push!(τs, τ); push!(ts, q[1])
        push!(ps, SVector(q[2], q[3], q[4]))
        push!(vs, v3); push!(fs, h)
        push!(us, _spatial(eu)); push!(gy, _spatial(ef))
        push!(u4, u); push!(e1, ef); push!(e2, er); push!(e3, eu)
        push!(rl, _roll_angle(bh, q, u, (ef, er, eu), h))
        push!(sp, nv)

        n == nsteps && break

        # RK4 on (q, p, triad).
        a1 = rhs(q, p, ef, er, eu)
        a2 = rhs(q + 0.5dτ * a1[1], p + 0.5dτ * a1[2], ef + 0.5dτ * a1[3],
                 er + 0.5dτ * a1[4], eu + 0.5dτ * a1[5])
        a3 = rhs(q + 0.5dτ * a2[1], p + 0.5dτ * a2[2], ef + 0.5dτ * a2[3],
                 er + 0.5dτ * a2[4], eu + 0.5dτ * a2[5])
        a4 = rhs(q + dτ * a3[1], p + dτ * a3[2], ef + dτ * a3[3],
                 er + dτ * a3[4], eu + dτ * a3[5])
        c = dτ / 6
        q  = q  + c * (a1[1] + 2a2[1] + 2a3[1] + a4[1])
        p  = p  + c * (a1[2] + 2a2[2] + 2a3[2] + a4[2])
        ef = ef + c * (a1[3] + 2a2[3] + 2a3[3] + a4[3])
        er = er + c * (a1[4] + 2a2[4] + 2a3[4] + a4[4])
        eu = eu + c * (a1[5] + 2a2[5] + 2a3[5] + a4[5])
        τ += dτ

        rr = ks_radius(bh, SVector(q[2], q[3], q[4]))
        if rr < 1.02 * rh
            captured = true
            break
        end
    end

    return Track(bh, τs, ts, ps, vs, fs, us, gy, rl,
                 u4, e1, e2, e3, sp, captured)
end

# ---------------------------------------------------------------------------
# Track construction
# ---------------------------------------------------------------------------
#
# Tracks are NOT built by shooting from a start gate to a finish gate and
# hoping the result passes somewhere interesting. That is a badly conditioned
# two-point boundary value problem — trajectories near a black hole separate
# exponentially, so the map from initial conditions to final position is
# stiff, and an optimizer spends its whole budget fighting that.
#
# Build them the other way round. The interesting part of a track is the close
# pass, and the close pass is exactly what a level designer wants to author:
# how deep, at what inclination, with or against the spin. So construct the
# state AT periapsis, where the geometry is a direct choice, and integrate
# outward in BOTH time directions. Backwards gives you the approach the player
# must arrive on; forwards gives the exit. No solver, no convergence failure,
# and every track produced is a valid worldline by construction.
#
# Burns then only ever join one encounter's exit to the next one's entry, which
# is a small, local, well-conditioned problem instead of a global one.

"""
    ks_radius(spacetime, pos) -> r

The Kerr-Schild radial coordinate at `pos`: the root of
`r⁴ − (ρ² − a²)r² − a²z² = 0`. **Not** `|pos|` — surfaces of constant `r` are
confocal ellipsoids, so on the equator `|pos| = √(r² + a²)`. Every radial
comparison in this file goes through here, because mixing the two silently
places things on the wrong side of the horizon at high spin.
"""
function ks_radius(bh::AbstractSpacetime, pos::SVector{3,Float64})
    a = spin(bh)
    w = pos[1]^2 + pos[2]^2 + pos[3]^2 - a^2
    return sqrt(0.5 * (w + sqrt(w^2 + 4 * a^2 * pos[3]^2)))
end

"""
    grad_ks_r(spacetime, pos) -> ∇r

Gradient of the Kerr-Schild radial coordinate. Not `r̂`: in Kerr the surfaces of
constant `r` are confocal ellipsoids, so `∇r` tilts away from the position
vector off the equator, and using `pos/|pos|` instead would put a "periapsis"
at a point where `dr/dτ ≠ 0`.
"""
function grad_ks_r(bh::AbstractSpacetime, pos::SVector{3,Float64})
    rr(v) = ks_radius(bh, v)
    h = 1e-6 * max(norm(pos), 1.0)
    return SVector((rr(pos + h * SVector(1.0, 0, 0)) - rr(pos - h * SVector(1.0, 0, 0))) / 2h,
                   (rr(pos + h * SVector(0, 1.0, 0)) - rr(pos - h * SVector(0, 1.0, 0))) / 2h,
                   (rr(pos + h * SVector(0, 0, 1.0)) - rr(pos - h * SVector(0, 0, 1.0))) / 2h)
end

"""
    periapsis_state(spacetime, r_p, speed; inclination, prograde, phase)
        -> (pos, vel)

State at the closest approach of an encounter. `r_p` is the periapsis **Kerr-Schild** radius — the same `r` that
[`horizon_radius`](@ref) and [`photon_orbit_min`](@ref) are quoted in, not the
Cartesian distance from the origin,
`speed` the speed there **as measured by a static observer** (units of c, always valid in `[0,1)`), `inclination` the tilt of the
orbital plane from the equator in radians, and `prograde` whether the pass runs
with the hole's spin or against it.

The velocity direction is `n̂ × ∇r` — simultaneously in the chosen orbital plane
and tangent to the surface of constant `r`, which is what makes this point an
actual turning point rather than an approximate one.

`prograde` is the single biggest design knob at high spin: frame dragging moves
the innermost stable circular orbit from `9M` (retrograde) to `1.24M`
(prograde) at `a = 0.998`, so the two directions are nearly an order of
magnitude apart in how deep a pass is survivable.
"""
function periapsis_state(bh::AbstractSpacetime, r_p::Real, speed::Real;
                         inclination::Real = 0.0, prograde::Bool = true,
                         phase::Real = 0.0)
    0 < speed < 1 || throw(ArgumentError("speed must be in (0,1), got $speed"))
    r_p > horizon_radius(bh) ||
        throw(ArgumentError("periapsis $r_p is inside the horizon $(horizon_radius(bh))"))
    n̂ = SVector(sin(inclination) * sin(phase), -sin(inclination) * cos(phase),
                cos(inclination))
    ê = SVector(cos(phase), sin(phase), 0.0)
    ê = ê - dot(ê, n̂) * n̂
    ê = ê / norm(ê)
    # Place the point at Kerr-Schild radius `r_p`, NOT at Cartesian distance
    # `r_p`. Surfaces of constant r are confocal ellipsoids: on the equator
    # x² + y² = r² + a², so the two differ by up to `a`. Using the Cartesian
    # distance would put a requested r_p = 1.1M at a = 0.998 down at r = 0.46M,
    # inside the horizon, while the guard above happily passed it.
    a = spin(bh)
    ez2 = ê[3]^2
    s = 1.0 / sqrt((1 - ez2) / (r_p^2 + a^2) + ez2 / r_p^2)
    pos = s * ê
    t̂ = cross(n̂, grad_ks_r(bh, pos))
    nt = norm(t̂)
    nt > 1e-12 || throw(ArgumentError("degenerate orbital plane at inclination $inclination"))
    t̂ = (prograde ? 1.0 : -1.0) * t̂ / nt

    # `speed` is what a STATIC observer at the periapsis measures, not a
    # coordinate speed. The coordinate speed limit shrinks near the hole, so a
    # designer asking for 0.7c at r = 2.5M would be refused for a reason that
    # has nothing to do with the trajectory being unphysical. The static
    # observer's own 4-velocity is along ∂_t, which has no spatial part, so
    # boosting it along t̂ leaves dr/dτ = 0 exactly — the turning point survives.
    q = SVector(0.0, pos[1], pos[2], pos[3])
    g = metric(bh, q)
    g[1, 1] < 0 || throw(ArgumentError(
        "no static observer at r_p = $r_p (inside the ergosphere, g_tt = $(g[1,1]) ≥ 0). " *
        "A turning point there requires co-rotating with the hole, which this " *
        "constructor does not express — frame dragging leaves no choice about it."))
    et = SVector(1.0, 0.0, 0.0, 0.0) / sqrt(-g[1, 1])
    T = SVector(0.0, t̂[1], t̂[2], t̂[3])
    T = T + dot(T, g * et) * et
    T = T / sqrt(dot(T, g * T))
    γ = 1.0 / sqrt(1.0 - speed^2)
    u = γ * (et + speed * T)
    return pos, SVector(u[2], u[3], u[4]) / u[1]
end

"""
    encounter_track(spacetime, r_p, speed; τ_in, τ_out, kwargs...)

A complete flyby track built outward from its own periapsis: integrate `τ_in`
of proper time into the past, reverse, and append `τ_out` into the future. The
returned track is guaranteed timelike and guaranteed to make the requested pass
— there is nothing to converge, because the pass is the input rather than the
hoped-for output.

`τ` runs from `−τ_in` through `0` at periapsis to `+τ_out`.
"""
function encounter_track(bh::AbstractSpacetime, r_p::Real, speed::Real;
                         τ_in::Real = 120.0, τ_out::Real = 120.0,
                         dτ::Real = 0.05, inclination::Real = 0.0,
                         prograde::Bool = true, phase::Real = 0.0,
                         up_hint::SVector{3,Float64} = SVector(0.0, 0.0, 1.0))
    pos, vel = periapsis_state(bh, r_p, speed; inclination = inclination,
                               prograde = prograde, phase = phase)
    back = integrate_track(bh, pos, vel, -abs(τ_in); dτ = dτ, up_hint = up_hint)
    fwd  = integrate_track(bh, pos, vel,  abs(τ_out); dτ = dτ, up_hint = up_hint)
    # Drop the duplicated periapsis sample from the reversed half.
    keep = length(back):-1:2
    cat3(f) = vcat(getfield(back, f)[keep], getfield(fwd, f))
    return Track(bh, cat3(:τ), cat3(:t), cat3(:pos), cat3(:vel), cat3(:fwd),
                 cat3(:up), cat3(:gyro), cat3(:roll),
                 cat3(:u4), cat3(:ef4), cat3(:er4), cat3(:eu4),
                 cat3(:speed), back.captured || fwd.captured)
end

"""
    track_sample(track, τ) -> (pos, fwd, up, vel, roll)

Interpolate the track at proper time `τ`, clamped to its endpoints. Linear
between samples: the worldline is smooth and integrated at `dτ ≈ 0.02`, so at
any playback rate a viewer could follow, the interpolation error is far below a
pixel.

`vel` is the coordinate 3-velocity, which is what a `Camera` wants for
`velocity` — the renderer boosts the observer tetrad by it, so aberration,
Doppler and beaming come out of the same worldline that produced the position.
"""
function track_sample(tr::Track, τ::Real)
    n = length(tr)
    τ <= tr.τ[1]  && return (tr.pos[1], tr.fwd[1], tr.up[1], tr.vel[1], tr.roll[1])
    τ >= tr.τ[n]  && return (tr.pos[n], tr.fwd[n], tr.up[n], tr.vel[n], tr.roll[n])
    i = searchsortedfirst(tr.τ, τ)
    i = clamp(i, 2, n)
    w = (τ - tr.τ[i-1]) / (tr.τ[i] - tr.τ[i-1])
    lerp(a, b) = a + w * (b - a)
    nz(v) = (m = norm(v); m > 1e-12 ? v / m : v)
    return (lerp(tr.pos[i-1], tr.pos[i]),
            nz(lerp(tr.fwd[i-1], tr.fwd[i])),
            nz(lerp(tr.up[i-1],  tr.up[i])),
            lerp(tr.vel[i-1], tr.vel[i]),
            lerp(tr.roll[i-1], tr.roll[i]))
end

"""
    tidal_scalar(spacetime, pos) -> ~M/r³

Cheap stand-in for the tidal stress a ship feels in free fall. Proper
acceleration is zero on a geodesic — an accelerometer riding the slingshot
reads nothing — so this, not acceleration, is the quantity that can hurt you.
The real thing is the Riemann tensor contracted with the ship's frame; this is
its leading radial scaling, which is what a HUD needle actually needs.
"""
tidal_scalar(bh::AbstractSpacetime, pos::SVector{3,Float64}) =
    bh.M / max(ks_radius(bh, pos), 1e-6)^3

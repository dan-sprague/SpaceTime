# ---------------------------------------------------------------------------
# Ship dynamics: a massive camera on a true general-relativistic worldline
# ---------------------------------------------------------------------------
#
# The simulator's camera is not a ghost: it is a point mass whose worldline is
# integrated in Cartesian Kerr–Schild coordinates with the same geodesic
# right-hand side the renderer trusts (`ks_rhs`). Engines off means exact free
# fall — orbits, plunges, and zero on the accelerometer. Engines on apply a
# proper acceleration in the ship's own frame, implemented as a Strang split
# (half kick / geodesic drift / half kick), so hovering on thrust is stable.
#
# State is the covariant 4-momentum per unit mass: `p_t` is exactly conserved
# while coasting (the metric is static), which makes drift in the integrator
# directly observable — the test suite pins a full circular orbit on it.

"""
    ks_gdot(pos, M, A, B)

Kerr–Schild metric dot product of two **contravariant** 4-vectors `(t, x, y, z)`
at Cartesian position `pos`: `g = η + f l⊗l` with `f = 2M/r`, `l_μ = (1, x̂)`.
"""
function ks_gdot(pos::SVector{3,Float64}, M::Float64,
                 A::SVector{4,Float64}, B::SVector{4,Float64})
    r = norm(pos)
    f = 2.0 * M / r
    lA = A[1] + (pos[1] * A[2] + pos[2] * A[3] + pos[3] * A[4]) / r
    lB = B[1] + (pos[1] * B[2] + pos[2] * B[3] + pos[3] * B[4]) / r
    return -A[1] * B[1] + A[2] * B[2] + A[3] * B[3] + A[4] * B[4] + f * lA * lB
end

"""
    ks_raise(pos, M, p_t, p) -> u::SVector{4}

Contravariant 4-velocity `u^μ = g^{μν} p_ν` from the covariant momentum
`(p_t, p)`. Inverse metric: `g^{μν} = η^{μν} − f l^μ l^ν`, `l^μ = (−1, x̂)`.
"""
function ks_raise(pos::SVector{3,Float64}, M::Float64, p_t::Float64,
                  p::SVector{3,Float64})
    r = norm(pos)
    x̂ = pos / r
    f = 2.0 * M / r
    ℓ = -p_t + dot(x̂, p)                    # l^μ p_μ
    return SVector(-p_t + f * ℓ, (p - (f * ℓ) * x̂)...)
end

"""
    ks_lower(pos, M, u) -> (p_t, p)

Covariant momentum `p_μ = g_{μν} u^ν` from the contravariant 4-velocity.
"""
function ks_lower(pos::SVector{3,Float64}, M::Float64, u::SVector{4,Float64})
    r = norm(pos)
    x̂ = pos / r
    f = 2.0 * M / r
    ℓu = u[1] + x̂[1] * u[2] + x̂[2] * u[3] + x̂[3] * u[4]   # l_μ u^μ
    p_t = -u[1] + f * ℓu
    p = SVector(u[2], u[3], u[4]) + (f * ℓu) * x̂
    return p_t, p
end

"""
    ShipState(pos, M) -> ShipState

A massive camera-carrying ship on a timelike worldline, stored as Cartesian
Kerr–Schild position plus covariant 4-momentum per unit mass (`p·p = −1`).
The constructor starts the ship at rest relative to the local reference
observer of [`ks_camera_tetrad`](@ref) — static for `r ≥ 2.5M`, the
freeze-radius free-faller inside. `τ` and `t` accumulate proper time and
coordinate (far-away) time along the worldline, in units of M.
"""
mutable struct ShipState
    x::SVector{3,Float64}
    p::SVector{3,Float64}
    p_t::Float64
    τ::Float64
    t::Float64
end

function ShipState(pos::SVector{3,Float64}, M::Float64)
    # Any orthonormal camera triple gives the same reference observer u.
    fwd = abs(pos[3]) < 0.9 * norm(pos) ? SVector(0.0, 0.0, 1.0) :
                                          SVector(1.0, 0.0, 0.0)
    right = normalize(cross(fwd, pos))
    up = cross(right, fwd)
    u = ks_camera_tetrad(pos, fwd, right, up, M)[1]
    p_t, p = ks_lower(pos, M, u)
    return ShipState(pos, p, p_t, 0.0, 0.0)
end

"""
    ship_velocity(ship, M, fwd, right, up) -> (β, γ)

Decompose the ship's 4-velocity against the local reference observer's tetrad
built on the camera axes `fwd/right/up`: `u_ship = γ(u_ref + β¹Ef + β²Er +
β³Eu)`. `β` is exactly the `beta` argument of [`ks_camera_tetrad`](@ref), so
feeding it back reconstructs the ship's frame — the renderer then shows the
aberrated, Doppler-shifted view an observer riding this worldline sees.
"""
function ship_velocity(ship::ShipState, M::Float64, fwd::SVector{3,Float64},
                       right::SVector{3,Float64}, up::SVector{3,Float64})
    u = ks_raise(ship.x, M, ship.p_t, ship.p)
    u_ref, Ef, Er, Eu = ks_camera_tetrad(ship.x, fwd, right, up, M)
    γ = -ks_gdot(ship.x, M, u, u_ref)
    β = SVector(ks_gdot(ship.x, M, u, Ef),
                ks_gdot(ship.x, M, u, Er),
                ks_gdot(ship.x, M, u, Eu)) / γ
    return β, γ
end

# One proper-acceleration half-kick: u ← normalize(u + dτ·a^i e_i), where the
# axes are the ship-frame spatial triad supplied by the caller. Exact in the
# limit dτ → 0; renormalisation keeps u on the mass shell at any step size.
function _ship_kick(x::SVector{3,Float64}, M::Float64, u::SVector{4,Float64},
                    dτ::Float64, a::SVector{3,Float64},
                    axes::NTuple{3,SVector{4,Float64}})
    u2 = u + dτ * (a[1] * axes[1] + a[2] * axes[2] + a[3] * axes[3])
    return u2 / sqrt(max(-ks_gdot(x, M, u2, u2), 1.0e-6))
end

"""
    step_ship!(ship, M, dτ; accel=zero(SVector{3,Float64}), axes=nothing)

Advance the ship by proper time `dτ` (units of M). With `accel == 0` this is a
pure geodesic step — free fall. Otherwise `accel` is the proper acceleration
(units of c²/M) with components along `axes`, the ship-frame spatial triad
(pass the boosted `Ef/Er/Eu` from [`ks_camera_tetrad`](@ref); required when
`accel ≠ 0`). Thrust is Strang-split around radius-adaptive RK4 geodesic
substeps, and the momentum is renormalised to the mass shell once per call.
"""
function step_ship!(ship::ShipState, M::Float64, dτ::Float64;
                    accel::SVector{3,Float64}=SVector(0.0, 0.0, 0.0),
                    axes::Union{Nothing,NTuple{3,SVector{4,Float64}}}=nothing)
    dτ <= 0.0 && return ship
    thrusting = dot(accel, accel) > 0.0
    thrusting && axes === nothing &&
        throw(ArgumentError("step_ship!: thrust needs the ship-frame axes"))
    rem = dτ
    while rem > 1.0e-12
        r = norm(ship.x)
        h = min(rem, 0.015 * clamp(r / M, 0.4, 30.0) * M)
        if thrusting
            u = ks_raise(ship.x, M, ship.p_t, ship.p)
            u = _ship_kick(ship.x, M, u, 0.5 * h, accel, axes)
            ship.p_t, ship.p = ks_lower(ship.x, M, u)
        end
        # Coordinate-time bookkeeping: dt/dτ = u^t (midpoint-free but the
        # step is small; the clocks are HUD telemetry, not dynamics).
        ship.t += ks_raise(ship.x, M, ship.p_t, ship.p)[1] * h
        μ = rk4_step_preview(vcat(ship.x, ship.p), ship.p_t, M, h)
        ship.x = SVector(μ[1], μ[2], μ[3])
        ship.p = SVector(μ[4], μ[5], μ[6])
        if thrusting
            u = ks_raise(ship.x, M, ship.p_t, ship.p)
            u = _ship_kick(ship.x, M, u, 0.5 * h, accel, axes)
            ship.p_t, ship.p = ks_lower(ship.x, M, u)
        end
        ship.τ += h
        rem -= h
    end
    # Mass-shell renormalisation: RK4 drift is tiny per step but the flight
    # runs for minutes; projecting u back to g(u,u) = −1 costs nothing.
    u = ks_raise(ship.x, M, ship.p_t, ship.p)
    u = u / sqrt(max(-ks_gdot(ship.x, M, u, u), 1.0e-6))
    ship.p_t, ship.p = ks_lower(ship.x, M, u)
    return ship
end

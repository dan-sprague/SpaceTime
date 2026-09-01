abstract type AbstractSpacetime end

"""
    Schwarzschild(M)
Represents a Schwarzschild black hole with mass `M`.
"""
struct Schwarzschild <: AbstractSpacetime
    M::Float64
end


"""
    Kerr(M, a)

A rotating black hole of mass `M` and spin parameter `a` (`|a| ≤ M`; `a > 0`
spins about `+z`). Used by the Metal renderer's Kerr mode, which integrates
[`kerr_rhs_mtl`](@ref) in the same Cartesian Kerr–Schild chart the
Schwarzschild path uses, and reduces to it exactly at `a = 0`.

Kerr is axisymmetric rather than spherically symmetric, so deflection is a
function of two parameters `(λ, η) = (L_z/E, Q/E²)` rather than the single
angle `ψ`. That is what kills the deflection fan — a Kerr "fan" would be a 2D
table the size of the image. Everything else the renderer leans on survives:
`E`, `L_z` and Carter's constant `Q` are still conserved, so the geodesics are
still integrable and the wound-ray tests still have closed forms.
"""
struct Kerr <: AbstractSpacetime
    M::Float64
    a::Float64
    function Kerr(M::Real, a::Real)
        abs(a) <= M || throw(ArgumentError("|a| must be <= M (got a=$a, M=$M)"))
        new(Float64(M), Float64(a))
    end
end

"""
    horizon_radius(spacetime)

Outer horizon in Kerr–Schild `r`: `2M` for Schwarzschild, `M + √(M² − a²)` for
Kerr.
"""
horizon_radius(bh::Schwarzschild) = 2.0 * bh.M
horizon_radius(bh::Kerr) = bh.M + sqrt(max(bh.M^2 - bh.a^2, 0.0))

"""
    photon_orbit_min(spacetime)

Radius of the **prograde equatorial photon orbit** — the smallest periapsis any
null geodesic reaching infinity can have, so anything below it is captured:

    r_ph = 2M[1 + cos(⅔ arccos(−a/M))]

`3M` at `a = 0`, falling to `M` at `a = M`. This is the Kerr generalisation of
the Schwarzschild renderer's `2.95M` shadow test, and it is a *global* bound —
off-equatorial spherical photon orbits all sit at larger radii — so no
direction test is needed alongside it.
"""
photon_orbit_min(bh::Schwarzschild) = 3.0 * bh.M
photon_orbit_min(bh::Kerr) =
    2.0 * bh.M * (1.0 + cos((2.0 / 3.0) * acos(-bh.a / bh.M)))

"""
    spin(spacetime)

Spin parameter `a`; zero for Schwarzschild.
"""
spin(::Schwarzschild) = 0.0
spin(bh::Kerr) = bh.a

"""
    Schwarzschild Geodesic Equations of Motion

Defines the equations of motion for a photon in the Schwarzschild spacetime,
written in **Cartesian Kerr–Schild coordinates** — the same chart as the Metal
kernel (`ks_rhs_mtl`) and the CPU preview (`ks_rhs`). The metric is
`g = η + f l⊗l` with `f = 2M/r` and `l_μ = (1, x/r, y/r, z/r)`, giving the
Hamiltonian `H = ½(−p_t² + |p|² − f ℓ²)` with `ℓ = −p_t + (x·p)/r`. Unlike the
spherical chart this is regular at the poles and at the horizon — no
`1/sin²θ`, no `1/(r−2M)`.

The input `μ` is an 8-component state vector `(t, x, y, z, p_t, px, py, pz)`,
`p` is a tuple containing the black hole and metadata, and `t` is the affine
parameter. `p_t` is conserved (dp_t = 0).
"""
function (bh::Schwarzschild)(μ::SVector{8,T},p,t) where T
    bh, meta = p

    x, y, z = μ[2], μ[3], μ[4]
    p_t, px, py, pz = μ[5], μ[6], μ[7], μ[8]

    M = bh.M
    r2 = x * x + y * y + z * z
    inv_r = 1.0 / sqrt(r2)
    f = 2.0 * M * inv_r
    κ = (x * px + y * py + z * pz) * inv_r
    ℓ = -p_t + κ
    c1 = f * ℓ * inv_r                              # fℓ/r
    c2 = f * ℓ * (0.5 * ℓ + κ) * inv_r * inv_r      # f(ℓ²/2 + ℓκ)/r²

    dt = -p_t + f * ℓ
    dx = px - c1 * x
    dy = py - c1 * y
    dz = pz - c1 * z
    dpx = c1 * px - c2 * x
    dpy = c1 * py - c2 * y
    dpz = c1 * pz - c2 * z

    return SVector{8, T}(dt, dx, dy, dz, 0.0, dpx, dpy, dpz)
end

"""
    Schwarzschild Metric Inverse
Calculates the inverse of the Schwarzschild metric at a given position `q` in
Cartesian Kerr–Schild coordinates `(t, x, y, z)`. The Kerr–Schild form
`g_μν = η_μν + f l_μ l_ν` with null `l` inverts exactly to
`g^μν = η^μν − f l^μ l^ν`, where `l^μ = (−1, x/r, y/r, z/r)`. Returns a 4x4
matrix of the inverse metric components.
"""
function metric_inverse(bh::Schwarzschild, q::SVector{4,T}) where T
    x, y, z = q[2], q[3], q[4]
    r = sqrt(x^2 + y^2 + z^2)
    f = 2 * bh.M / r

    lU = SVector{4,T}(-1.0, x / r, y / r, z / r)
    η = @SMatrix [-one(T) zero(T) zero(T) zero(T);
                  zero(T)  one(T) zero(T) zero(T);
                  zero(T) zero(T)  one(T) zero(T);
                  zero(T) zero(T) zero(T)  one(T)]
    return η - f * (lU * lU')
end

"""
    Kerr Metric Inverse

Same Kerr–Schild structure as the Schwarzschild case, `g^μν = η^μν − f l^μ l^ν`,
with the Kerr `f` and `l` (see [`kerr_rhs_mtl`](@ref)) and `r` from the
implicit quartic `r⁴ − (ρ² − a²)r² − a²z² = 0`.
"""
function metric_inverse(bh::Kerr, q::SVector{4,T}) where T
    x, y, z = q[2], q[3], q[4]
    a = bh.a
    w = x^2 + y^2 + z^2 - a^2
    r2 = 0.5 * (w + sqrt(w^2 + 4 * a^2 * z^2))
    r = sqrt(max(r2, eps(Float64)))
    f = 2 * bh.M * r^3 / (r2^2 + a^2 * z^2)
    R2A = r2 + a^2
    lU = SVector{4,T}(-1.0, (r * x + a * y) / R2A, (r * y - a * x) / R2A, z / r)
    η = @SMatrix [-one(T) zero(T) zero(T) zero(T);
                  zero(T)  one(T) zero(T) zero(T);
                  zero(T) zero(T)  one(T) zero(T);
                  zero(T) zero(T) zero(T)  one(T)]
    return η - f * (lU * lU')
end

""" 
    hamiltonian(μ, bh)
Calculates the Hamiltonian for a photon in the given spacetime. The input `μ` is an 8-component state vector containing position and momentum information, and `bh` is the black hole spacetime. The function returns the value of the Hamiltonian, which should be zero for a photon following a geodesic.
"""
function hamiltonian(μ::SVector{8,T}, bh::AbstractSpacetime) where T
    q = μ[SVector{4}(1,2,3,4)]
    p = μ[SVector{4}(5,6,7,8)]
    
    g_inv = metric_inverse(bh, q)
    
    return 0.5 * dot(p, g_inv * p)
end


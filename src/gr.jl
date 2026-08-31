abstract type AbstractSpacetime end

"""
    Schwarzschild(M)
Represents a Schwarzschild black hole with mass `M`.
"""
struct Schwarzschild <: AbstractSpacetime
    M::Float64
end


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
    Kerr(M, a)
Represents a Kerr black hole with mass `M` and spin parameter `a`. The Kerr
"""
struct Kerr <: AbstractSpacetime
    M::Float64
    a::Float64
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

function metric_inverse(bh::Kerr, q::SVector{4,T}) where T
    @error "Kerr metric not implemented yet"
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


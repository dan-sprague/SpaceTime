abstract type AbstractSpacetime end

"""
    Schwarzschild(M)
Represents a Schwarzschild black hole with mass `M`.
"""
struct Schwarzschild <: AbstractSpacetime
    M::Float64
end


"""
    Schwarzchild Geodesic Equations of Motion

Defines the equations of motion for a photon in the Schwarzschild spacetime. The input `μ` is an 8-component state vector containing position and momentum information, `p` is a tuple containing the black hole and metadata, and `t` is the time parameter. The function returns the derivatives of the state vector according to the geodesic equations.
"""
function (bh::Schwarzschild)(μ::SVector{8,T},p,t) where T
    bh, meta = p
    
    r = μ[2]
    θ = μ[3]
    pt, pr, pθ, pϕ = μ[5], μ[6], μ[7], μ[8]
    
    M = bh.M
    sinθ, cosθ = sin(θ), cos(θ)
    
    r2 = r^2
    inv_r2 = 1.0 / r2
    inv_r3 = inv_r2 / r
    delta = r - 2 * M

    # 1. Position derivatives
    dt = -(r / delta) * pt
    dr = (delta / r) * pr
    dθ = inv_r2 * pθ
    dϕ = (inv_r2 / sinθ^2) * pϕ

    # 2. Momentum derivatives
    dpr = -0.5 * (
        (2 * M / delta^2) * pt^2 + 
        (2 * M * inv_r2) * pr^2 + 
        (-2 * inv_r3) * pθ^2 + 
        (-2 * inv_r3 / sinθ^2) * pϕ^2
    )

    dpθ = -0.5 * ((-2 * cosθ / (r2 * sinθ^3)) * pϕ^2)

    return SVector{8, T}(dt, dr, dθ, dϕ, 0.0, dpr, dpθ, 0.0)
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
Calculates the inverse of the Schwarzschild metric at a given position `q`. The input `q` is a 4-component vector containing the coordinates (t, r, θ, ϕ). The function returns a 4x4 matrix representing the inverse metric components.
"""
function metric_inverse(bh::Schwarzschild, q::SVector{4,T}) where T
    r = q[2]
    θ = q[3]

    M = bh.M

    Δ = r * (r - 2M)

    gUU_tt = -r^2 / Δ
    gUU_rr = Δ / r^2
    gUU_θθ = 1.0 / r^2
    gUU_pqpq = 1.0 / (r^2 * sin(θ)^2)

    z = zero(T)
    @SMatrix [gUU_tt  z    z    z   ;
              z   gUU_rr  z    z   ;
              z    z   gUU_θθ  z   ;
              z    z    z   gUU_pqpq]
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


abstract type AbstractSpacetime end


struct Schwarzschild <: AbstractSpacetime
    M::Float64
end


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

struct Kerr <: AbstractSpacetime
    M::Float64
    a::Float64
end

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

function hamiltonian(μ::SVector{8,T}, bh::AbstractSpacetime) where T
    q = μ[SVector{4}(1,2,3,4)]
    p = μ[SVector{4}(5,6,7,8)]
    
    g_inv = metric_inverse(bh, q)
    
    return 0.5 * dot(p, g_inv * p)
end


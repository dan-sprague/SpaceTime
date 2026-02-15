abstract type BlackHole end


struct Schwarzschild <: BlackHole
    M::Float64
end

struct Kerr <: BlackHole
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



function hamiltonian(μ::SVector{8,T}, bh::BlackHole) where T
    q = μ[SVector{4}(1,2,3,4)]
    p = μ[SVector{4}(5,6,7,8)]
    
    g_inv = metric_inverse(bh, q)
    
    return 0.5 * dot(p, g_inv * p)
end

function equations_of_motion(μ::SVector{8, T}, p, t) where T
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


function init_photon(cam::Camera, bh::BlackHole, u, v)
    d = get_ray_direction(cam, u, v)
    dx, dy, dz = d

    x, y, z = cam.pos
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = atan(y, x)
    q0 = @SVector [0.0, r, θ, ϕ]

    # Projections
    vr = dx * sin(θ) * cos(ϕ) + dy * sin(θ) * sin(ϕ) + dz * cos(θ)
    vθ = (dx * cos(θ) * cos(ϕ) + dy * cos(θ) * sin(ϕ) - dz * sin(θ)) / r
    vϕ = (-dx * sin(ϕ) + dy * cos(ϕ)) / (r * sin(θ))

    g_inv = metric_inverse(bh, q0)

    # Momentum components
    pr = vr / g_inv[2,2]
    pθ = vθ / g_inv[3,3]
    pϕ = vϕ / g_inv[4,4]

    # Null condition H=0
    spatial_part = g_inv[2,2]*pr^2 + g_inv[3,3]*pθ^2 + g_inv[4,4]*pϕ^2
    pt = -sqrt(abs(spatial_part / g_inv[1,1]))

    return vcat(q0, SVector(pt, pr, pθ, pϕ))
end



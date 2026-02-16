function boundary_condition(μ,t,integrator)
    r = μ[2]

    return (r - 2.00001) * (100.0 - r)
end

function horizon_affect!(integrator)
    terminate!(integrator)
end 

function disc_condition(μ,t,integrator)
    μ[3] - π/2
end

function disk_affect!(integrator)
    bh, meta = integrator.p 
    
    r = integrator.u[2]
    if DISK_INNER_RADIUS < r < DISK_OUTER_RADIUS
        meta.hit_disk = true
        meta.r_hit = r
        terminate!(integrator)
    end
end

function disk_affect_doppler!(integrator)
    bh = integrator.p[1]
    M = bh.M
    r, θ, ϕ = integrator.u[2], integrator.u[3], integrator.u[4]
    pr, pθ, pϕ = integrator.u[6], integrator.u[7], integrator.u[8]

    # Track closest approach
    meta = integrator.p[2]
    if r < meta.r_min
        meta.r_min = r
    end

    e_r = SVector(sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
    e_θ = SVector(cos(θ)*cos(ϕ), cos(θ)*sin(ϕ), -sin(θ))
    e_ϕ = SVector(-sin(ϕ), cos(ϕ), 0.0)

    v_r = (r - 2M) / r * pr
    p_cartesian = v_r * e_r + (pθ/r) * e_θ + (pϕ/(r*sin(θ))) * e_ϕ
    pos_cartesian = SVector(r*sin(θ)*cos(ϕ), r*sin(θ)*sin(ϕ), r*cos(θ))

    if DISK_INNER_RADIUS < r < DISK_OUTER_RADIUS
        local_color, T_obs = get_disc_color_doppler(r, pos_cartesian, p_cartesian, bh)

        R = r / (2M)
        R_inner = DISK_INNER_RADIUS / (2M)
        iscotaper = clamp((R^2 - R_inner^2) * 0.3, 0.0, 1.0)
        outertaper = clamp(T_obs / 1000.0, 0.0, 1.0)
        R_outer = DISK_OUTER_RADIUS / (2M)
        density = clamp((R_outer - R) / (R_outer - R_inner), 0, 1)
        #density = (R_inner / R)^1.05
        disc_opacity = iscotaper * outertaper * density^0.33

        integrator.p[2].acc_color += local_color * (integrator.p[2].alpha * disc_opacity)
        integrator.p[2].alpha *= (1.0 - disc_opacity)
    end
end




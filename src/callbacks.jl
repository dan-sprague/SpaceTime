"""
    boundary_condition(μ,t,integrator)
Defines a boundary condition for the ODE solver that checks if the photon has crossed the event horizon or has traveled too far away. The function returns a value that changes sign when the photon crosses these boundaries, allowing the ODE solver to trigger an event.
"""
function boundary_condition(μ,t,integrator)
    r = μ[2]

    return (r - 2.00001) * (100.0 - r)
end

"""
    horizon_affect!(integrator)
Defines the effect to be applied when the photon crosses the event horizon. In this case, it simply terminates the integration, as the photon is considered to be captured by the black hole.
"""
function horizon_affect!(integrator)
    terminate!(integrator)
end


"""
    disc_condition(μ,t,integrator)
Defines a condition for when the photon intersects with the accretion disc. The function returns a value that changes sign when the photon crosses the plane of the disc (θ = π/2), allowing the ODE solver to trigger an event.
"""
function disc_condition(μ,t,integrator)
    μ[3] - π/2
end


"""
    disc_affect!(integrator)
Defines the effect to be applied when the photon intersects with the accretion disc. It checks if the photon is within the radial bounds of the disc and, if so, it updates the accumulated color and alpha values in the metadata based on the local color of the disc at the point of intersection.
"""
function disc_affect!(integrator)
    bh, meta, disc = integrator.p

    r = integrator.u[2]
    if disc.inner_radius < r < disc.outer_radius
        meta.hit_disc = true
        meta.r_hit = r
        terminate!(integrator)
    end
end

"""
    disc_affect_doppler!(integrator)
Defines the effect to be applied when the photon intersects with the accretion disc, taking into account Doppler and gravitational redshift effects. It calculates the local color of the disc at the point of intersection and updates the accumulated color and alpha values in the metadata accordingly.
"""
function disc_affect_doppler!(integrator)
    bh = integrator.p[1]
    disc = integrator.p[3]
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

    if disc.inner_radius < r < disc.outer_radius
        local_color, T_obs = get_disc_color_doppler(r, pos_cartesian, p_cartesian, bh, disc)

        R = r / (2M)
        R_inner = disc.inner_radius / (2M)
        iscotaper = clamp((R^2 - R_inner^2) * 0.3, 0.0, 1.0)
        outertaper = clamp(T_obs / 1000.0, 0.0, 1.0)
        R_outer = disc.outer_radius / (2M)
        density = clamp((R_outer - R) / (R_outer - R_inner), 0, 1)
        #density = (R_inner / R)^1.05
        disc_opacity = iscotaper * outertaper * density^0.33

        integrator.p[2].acc_color += local_color * (integrator.p[2].alpha * disc_opacity)
        integrator.p[2].alpha *= (1.0 - disc_opacity)
    end
end

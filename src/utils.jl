function compare_hamiltonian_drift(cam::AbstractCamera, spacetime::AbstractSpacetime)
    u, v = 0.05, 0.05
    μ0 = init_photon(cam, spacetime, u, v)
    tspan = (0.0, 500.0)
    meta = RayData(RGBf(0.0,0.0,0.0),1.0,Inf,0.0)

    prob_sym = ODEProblem(spacetime, μ0, tspan, (spacetime, meta))
    sol_sym = solve(prob_sym, Vern9(), reltol=1e-8, abstol=1e-8)

    h_err = [hamiltonian(u, spacetime) for u in sol_sym.u]

    fig = Figure(resolution = (250, 250))
    ax = Makie.Axis(fig[1, 1],
        title = "Hamiltonian Constraint Violation (lower is better)",
        xlabel = "Radius (r / 2M)",
        ylabel = "log10|H|",
        yscale = log10)

    lines!(ax, sol_sym.u .|> x -> sqrt(x[2]^2 + x[3]^2 + x[4]^2) / (2 * spacetime.M), abs.(h_err), label="Functor", color=:blue)

    axislegend(ax)
    return fig
end


function visualize_solution(solutions::Vector{WorldLine}, spacetime::Schwarzschild; filename="rays.mp4", framerate=60, nframes=1000, rev=false, lim=10.0,
    title = "")
    # Precompute Cartesian trajectories (reverse if requested)
    trajectories = map(solutions) do wl
        # State is Cartesian Kerr–Schild: positions are components 2:4.
        pts = [Point3f(μ[2], μ[3], μ[4]) for μ in wl.μ]
        rev ? reverse(pts) : pts
    end

    # Time arrays for each trajectory (may differ due to early termination)
    # When reversed, remap times so they still increase monotonically
    times = if rev
        t_global_max = maximum(wl.t[end] for wl in solutions)
        [t_global_max .- reverse(wl.t) for wl in solutions]
    else
        [wl.t for wl in solutions]
    end

    # Trim to first point inside the axis limits (preserve relative offsets for sync)
    if rev
        for i in eachindex(trajectories)
            first_in = findfirst(p -> all(abs.(p) .<= lim), trajectories[i])
            if first_in !== nothing && first_in > 1
                trajectories[i] = trajectories[i][first_in:end]
                times[i] = times[i][first_in:end]
            end
        end
        # Shift so the first visible point starts at t=0, keeping relative sync
        t_min = minimum(t[1] for t in times)
        for i in eachindex(times)
            times[i] = times[i] .- t_min
        end
    end

    t_max = maximum(t[end] for t in times)

    fig = Figure(size = (400,400), figure_padding = 0, px_per_unit = 6)
    ax = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="z", aspect = :data,
    limits = (-lim, lim, -lim, lim, -lim, lim),title = title)
    mesh!(ax, Sphere(Point3f(0), 2*spacetime.M), color=:black)

    # One observable per trajectory for the line, one for the tip
    line_obs = [Observable(Point3f[first(traj)]) for traj in trajectories]
    tip_obs  = [Observable(Point3f[first(traj)]) for traj in trajectories]

    for i in eachindex(trajectories)
        lines!(ax, line_obs[i],color=:cornflowerblue, alpha = 0.5)
        scatter!(ax, tip_obs[i], color=:cornflowerblue, markersize=8)
    end

    record(fig, filename, 1:nframes; framerate) do frame
        t_current = frame / nframes * t_max
        for i in eachindex(trajectories)
            # Find how far along this trajectory we are in coordinate time
            idx = searchsortedlast(times[i], t_current)
            idx = clamp(idx, 1, length(trajectories[i]))
            line_obs[i][] = trajectories[i][1:idx]
            tip_obs[i][]  = [trajectories[i][idx]]
        end
    end
end

visualize_solution(sol::WorldLine, spacetime::Schwarzschild; kwargs...) = visualize_solution([sol], spacetime; kwargs...)

"""
    shadow_radius(cam, spacetime)

Critical screen-space radius below which rays are captured by the black hole.
For Schwarzschild, the critical impact parameter is b_c = 3√3 M, which maps
to a screen coordinate of tan(arcsin(b_c / r_cam)) / fov_factor.
"""
function shadow_radius(cam::AbstractCamera, spacetime::Schwarzschild)
    r_cam = norm(cam.pos)
    b_c = 3sqrt(3) * spacetime.M
    sin_α = b_c / r_cam
    sin_α >= 1.0 && return Inf
    tan(asin(sin_α)) / cam.fov_factor
end

function trace_fan(cam::AbstractCamera, spacetime::Schwarzschild;
                   u_range=range(-1.0, 1.0, length=10),
                   v_range=range(-0.5, 0.5, length=10),
                   tspan=(0.0, 1000.0),
                   solver=Tsit5())
    u_crit = shadow_radius(cam, spacetime)
    rays = WorldLine[]
    for u in u_range, v in v_range
        sqrt(u^2 + v^2) < u_crit && continue
        push!(rays, raytrace(spacetime, Photon(init_photon(cam, spacetime, u, v)); tspan, solver))
    end
    rays
end
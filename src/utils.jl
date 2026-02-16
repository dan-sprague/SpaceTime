function compare_hamiltonian_drift(cam::Camera, spacetime::AbstractSpacetime)
    u, v = 0.05, 0.05
    μ0 = init_photon(cam, spacetime, u, v)
    tspan = (0.0, 500.0)
    meta = RayData(RGBf(0.0,0.0,0.0),1.0,Inf)

    prob_sym = ODEProblem(spacetime, μ0, tspan, (spacetime, meta))
    sol_sym = solve(prob_sym, Vern9(), reltol=1e-8, abstol=1e-8)

    h_err = [hamiltonian(u, spacetime) for u in sol_sym.u]

    fig = Figure(resolution = (250, 250))
    ax = CairoMakie.Axis(fig[1, 1],
        title = "Hamiltonian Constraint Violation (lower is better)",
        xlabel = "Radius (r / 2M)",
        ylabel = "log10|H|",
        yscale = log10)

    lines!(ax, sol_sym.u .|> x -> x[2] / (2 * spacetime.M), abs.(h_err), label="Functor", color=:blue)

    axislegend(ax)
    return fig
end


function visualize_solution(solutions::Vector; M=1.0, filename="rays.mp4", framerate=30, nframes=120)
    # Precompute Cartesian trajectories
    trajectories = map(solutions) do sol
        r = sol[2,:]
        θ = sol[3,:]
        ϕ = sol[4,:]
        x = r .* sin.(θ) .* cos.(ϕ)
        y = r .* sin.(θ) .* sin.(ϕ)
        z = r .* cos.(θ)
        Point3f.(x, y, z)
    end

    npts = length(first(trajectories))
    fig = Figure()
    ax = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="z")
    mesh!(ax, Sphere(Point3f(0), 2M), color=(:black, 0.5))

    # One observable per trajectory for the line, one for the tip
    line_obs = [Observable(Point3f[first(traj)]) for traj in trajectories]
    tip_obs  = [Observable(Point3f[first(traj)]) for traj in trajectories]

    for i in eachindex(trajectories)
        lines!(ax, line_obs[i])
        scatter!(ax, tip_obs[i], color=:red, markersize=10, marker=:xcross)
    end

    record(fig, filename, 1:nframes; framerate) do frame
        idx = clamp(round(Int, frame / nframes * npts), 1, npts)
        for i in eachindex(trajectories)
            line_obs[i][] = trajectories[i][1:idx]
            tip_obs[i][]  = [trajectories[i][idx]]
        end
    end
end

visualize_solution(sol; kwargs...) = visualize_solution([sol]; kwargs...)

function trace_fan(cam::Camera, spacetime::AbstractSpacetime;
                   u_range=range(-0.3, 0.3, length=5),
                   v=0.0, tspan=(0.0, 500.0))
    [smooth_raytrace(spacetime, Photon(init_photon(cam, spacetime, u, v)), tspan)
     for u in u_range]
end
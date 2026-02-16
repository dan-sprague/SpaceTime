function compare_hamiltonian_drift(cam::Camera, bh::BlackHole)
    u, v = 0.05, 0.05 
    μ0 = init_photon(cam, bh, u, v)
    tspan = (0.0, 500.0)
    meta = RayData(RGBf(0.0,0.0,0.0),1.0,Inf)
    prob_auto = ODEProblem(equations_of_motion_autodiff, μ0, tspan, bh)
    sol_auto = solve(prob_auto, Vern9(), reltol=1e-8, abstol=1e-8)

    prob_sym = ODEProblem(equations_of_motion, μ0, tspan, (bh, meta))
    sol_sym = solve(prob_sym, Vern9(), reltol=1e-8, abstol=1e-8)

    h_err_auto = [hamiltonian(u, bh) for u in sol_auto.u]
    h_err_sym  = [hamiltonian(u, bh) for u in sol_sym.u]

    fig = Figure(resolution = (800, 600))
    ax = CairoMakie.Axis(fig[1, 1], 
        title = "Hamiltonian Constraint Violation (lower is better)",
        xlabel = "Coordinate Time (t)", 
        ylabel = "log10|H|",
        yscale = log10)

    lines!(ax, sol_auto.t, abs.(h_err_auto), label="AutoDiff", color=:blue)
    lines!(ax, sol_sym.t,  abs.(h_err_sym),  label="Symbolic", color=:red)
    
    axislegend(ax)
    return fig
end


function visualize_solution(solution)
    fig = Figure()
    ax = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="z")
    r = solution[2,:]
    θ = solution[3,:]
    ϕ = solution[4,:]
    x = r .* sin.(θ) .* cos.(ϕ)
    y = r .* sin.(θ) .* sin.(ϕ)
    z = r .* cos.(θ)
    lines!(ax, x, y, z)

    mesh!(ax, Sphere(Point3f(0), 2.0), color=(:black,0.5))
    scatter!(ax, x[end], y[end], z[end], color=:red, markersize=10,marker=:xcross)
    #mesh!(ax, Sphere(Point3f(0), 3.0), color=(:orange, 0.1))
    fig
end
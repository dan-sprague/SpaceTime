"""
    compare_hamiltonian_drift(cam, spacetime) -> Makie.Figure

Trace one ray with a high-order integrator and plot |H| (the null-geodesic
Hamiltonian, exactly zero for the true solution) against radius, as a check on
integrator drift. Needs a Makie backend loaded (`using CairoMakie`): the method
lives in the `SpaceTimeMakieExt` package extension.
"""
function compare_hamiltonian_drift end

"""
    visualize_solution(rays, spacetime; filename="rays.mp4", framerate=60,
                       nframes=1000, rev=false, lim=10.0, title="")

Record an animation of traced [`WorldLine`](@ref)s (from [`raytrace`](@ref) or
[`trace_fan`](@ref)) bending around the hole. Needs a Makie backend loaded
(`using CairoMakie`): the method lives in the `SpaceTimeMakieExt` extension.
"""
function visualize_solution end

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
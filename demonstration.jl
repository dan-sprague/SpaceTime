using Pkg
Pkg.activate(".")
Pkg.instantiate()

using SpaceTime
using StaticArrays
using LinearAlgebra
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0, blackbody=Blackbody(wb_temperature=6500.0))

spacetime = Schwarzschild(1.0)
cam = Camera(SVector(0.0, -10.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))

# Critical screen radius below which rays hit the black hole
u_crit = shadow_radius(cam, spacetime)

pad = 0.00000001
extent = u_crit + pad
sols = trace_fan(cam, spacetime;
    u_range=range(-extent, extent, length=15),
    v_range=range(-extent, extent, length=15),
    tspan=(0.0, 5000.0))

visualize_solution(sols, spacetime; nframes=500,rev=true)

# At r=3M, equatorial plane, purely tangential.
# Null condition: g^tt pt² + g^ϕϕ pϕ² = 0  →  pt = -pϕ / (3√3 M)
M = spacetime.M
pϕ = 1.0
pt = -pϕ / (3sqrt(3) * M)


photons = Photon[]
for (ϕ₀, frac) in [(i * 2π/10, sin(i * π/5)) for i in 0:9]
    pθ_i = frac * 1.0
    pϕ_i = sqrt(1.0 - pθ_i^2)
    # Null condition at r=3M: -3 pt² + (1/9M²)(pθ² + pϕ²/sin²θ) = 0
    pt_i = -sqrt((pθ_i^2 + pϕ_i^2) / (27M^2))
    push!(photons, Photon(SVector(0.0, 3M, π/2, ϕ₀, pt_i, 1e-6, pθ_i, pϕ_i)))
end

orbits = [raytrace(spacetime, p; tspan=(0.0, 500.0), npoints=5000) for p in photons]
visualize_solution(orbits, spacetime; nframes=500, filename="photon_sphere.mp4", rev=true,title = "Photon Sphere")

using LinearAlgebra
using GLMakie
using CairoMakie
bh = Schwarzschild(1.0)

# Camera transforms (yaw, pitch, roll, dolly, etc.) are now part of the
# SpaceTime package and can be used directly on any AbstractCamera.

"""
    verify_shadow_area(cam_orig, cam_offset, spacetime; height=400, rtol=0.05)

Compare the event-horizon pixel areas analytically. The shadow is a circle
with screen-space radius `u_crit = shadow_radius(cam, spacetime)`, mapping
to `π * (u_crit * height/2)²` pixels. Returns a named tuple with both areas
and whether the relative difference is within `rtol`.
"""
function verify_shadow_area(cam_orig, cam_offset, spacetime; height=400, rtol=0.05)
    function shadow_pixels(cam)
        u_crit = shadow_radius(cam, spacetime)
        return π * (u_crit * height / 2)^2
    end

    a_orig   = shadow_pixels(cam_orig)
    a_offset = shadow_pixels(cam_offset)
    rel_diff = abs(a_orig - a_offset) / max(a_orig, 1)
    passes   = rel_diff ≤ rtol

    @info "Shadow area check" a_orig a_offset rel_diff rtol passes
    return (; a_orig, a_offset, passes)
end

using FileIO
bg = load("starmap_g4k.jpg")
size = (4096,2048)
cam_pos = SVector(100.0, 30.1, 45.6)
target = SVector(0.0, 3.0, 0.0)
θ_roll = atan(size[2] / size[1])
world_up = SVector(0.0, 0.0, 1.0)
world_right = SVector(0.0, 1.0, 0.0)
tilted_up = normalize(world_up * cos(θ_roll) + world_right * sin(θ_roll))
disc2 = AccretionDisc(inner_radius=3.0, outer_radius=10.0, blackbody=Blackbody(wb_temperature=15000.0),
density_falloff=0.7)



θ = atan(size[2] / size[1])

cam_orig = SpaceTime.Camera(cam_pos, target, tilted_up, 
Lens(200.0))
#cam2 = @gimbal cam_orig |> yaw 0.0 |> pitch 0.0

@time img = render(cam_orig, bh, bg; disc=disc2, width=size[1], height=size[2], 
            samples=4)

post_img = postprocess(img;
    gain=1.0,
    exposure=0.1,
    gamma = 0.30,
    bloom_strength=1.0,
    threshold=0.2,
    bloom_radius=0.2,
    bloom_power=1.0,
    streak_strength=1.0,
    streak_length=0.10,
    streak_width=0.2,
    n_spikes=3,
    tonemap=:aces
) 

post_img
using CairoMakie
figure = Figure(size=size,px_per_unit=1,padding=0.0)
ax = Axis(figure[1,1],aspect=DataAspect())
image!(ax, post_img)
hidedecorations!(ax)
figure

save("blackhole9.png", figure)

# COOL SCENE ;)
# bh = Schwarzschild(1.0)
# cam_pos = SVector(30.0, 1.1, 1.6)
# target = SVector(0.0, 0.0, 0.0)
# roll_deg = 20.0
# θ_roll = deg2rad(roll_deg)
# world_up = SVector(0.0, 0.0, 1.0)
# world_right = SVector(0.0, 1.0, 0.0)
# tilted_up = normalize(world_up * cos(θ_roll) + world_right * sin(θ_roll))
# cam2 = SpaceTime.Camera(cam_pos, target, tilted_up, 0.55)
# disc2 = AccretionDisc(inner_radius=3.0, outer_radius=20.0, blackbody=Blackbody(wb_temperature=10000.0),
# density_falloff=0.8)

# post_img = postprocess(img;
#     gain=1.0,
#     exposure=1.2,
#     gamma = 0.1,
#     bloom_strength=1.0,
#     threshold=0.5,
#     bloom_radius=10.0,
#     bloom_power=1.5,
#     streak_strength=2.0,
#     streak_length=0.1,
#     streak_width=1,
#     n_spikes=4,
#     tonemap=:aces
# ) 
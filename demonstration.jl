using Pkg
Pkg.activate(".")
Pkg.instantiate()

using SpaceTime
using StaticArrays

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
cam_pos = SVector(30.0, 1.1, 1.6)
target = SVector(0.0, 0.0, 0.0)
roll_deg = 20.0
θ_roll = deg2rad(roll_deg)
world_up = SVector(0.0, 0.0, 1.0)
world_right = SVector(0.0, 1.0, 0.0)
tilted_up = normalize(world_up * cos(θ_roll) + world_right * sin(θ_roll))
cam2 = Camera(cam_pos, target, tilted_up, 0.55)
disc2 = AccretionDisc(inner_radius=3.0, outer_radius=20.0, blackbody=Blackbody(wb_temperature=29000.0),
density_falloff=0.8)

bg = fill(RGBf(0, 0, 0), 3840, 2160)

img = render(cam2, bh, bg; disc=disc2, width=3840, height=2160) 
post_img = postprocess(img;
    gain=1.0,
    exposure=1.5,
    gamma = 0.08,
    bloom_strength=1.0,
    threshold=0.4,
    bloom_radius=10.0,
    bloom_power=1.0,
    streak_strength=0.8,
    streak_length=0.2,
    streak_width=5,
    n_spikes=4,
    tonemap=:aces
)

figure = Figure()
ax = Axis(figure[1,1],aspect=DataAspect())
image!(ax, post_img)
hidedecorations!(ax)
figure
save("accretion_disc.png", figure)
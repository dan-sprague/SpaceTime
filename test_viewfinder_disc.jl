using Pkg
Pkg.activate(".")

using SpaceTime
using StaticArrays
using LinearAlgebra
using FileIO

bg = load("starmap_g4k.jpg")
spacetime = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0)

cam_pos = SVector(100.0, 30.1, 45.6)
target  = SVector(0.0, 3.0, 0.0)
world_up = SVector(0.0, 0.0, 1.0)

bg_size = (4096, 2048)
θ_roll = atan(bg_size[2] / bg_size[1])
world_right = SVector(0.0, 1.0, 0.0)
tilted_up = normalize(world_up * cos(θ_roll) + world_right * sin(θ_roll))

cam = Camera(cam_pos, target, tilted_up, Lens(200.0))

settings = PreviewSettings(width=320, height=240, dt=0.1, nmax=1000)

# Test CPU preview with disc
img_cpu = SpaceTime.render_preview(cam, spacetime, bg; settings=settings, disc=disc)
println("CPU preview with disc: ", size(img_cpu), " ", eltype(img_cpu))

# Test 3D scene construction with disc
fig = SpaceTime.viewfinder(cam, spacetime, bg;
                           settings=settings,
                           title="Black Hole Viewfinder (Metal)",
                           disc=disc)
println("Figure constructed successfully: ", typeof(fig))

ax3d = fig.content[2]
println("3D scene type: ", typeof(ax3d))
println("Number of plots in 3D scene: ", length(ax3d.scene.plots))
for (i, p) in enumerate(ax3d.scene.plots)
    println("  plot ", i, ": ", typeof(p))
end

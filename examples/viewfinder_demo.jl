# Black-hole viewfinder demo.
#
# Run with:
#     julia -t auto,1 --project examples/viewfinder_demo.jl
#
# The `-t auto,1` launch matters: the final render runs on default-pool worker
# threads while the main thread stays on the interactive pool driving the UI.
# Without it everything still works, but the final render shares one thread
# with the window.

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using SpaceTime
using GLMakie
using StaticArrays
using LinearAlgebra
using FileIO

# Load the same background used in demonstration.jl.
bg = load(joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"))

spacetime = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0),
                     density_falloff=0.8)

# Starting camera: the "Hectic" composition — close to the disc plane, rolled,
# looking at the origin. Use the fly-cam (drag / scroll / WASD) to explore.
world_up = SVector(0.0, 0.0, 1.0)
world_right = SVector(0.0, 1.0, 0.0)
θ_roll = deg2rad(20.0)
tilted_up = normalize(world_up * cos(θ_roll) + world_right * sin(θ_roll))
cam = SpaceTime.Camera(SVector(30.0, 1.1, 1.6), SVector(0.0, 0.0, 0.0),
                       tilted_up, 0.55)

# Volumetric disc: thick turbulent gas replacing the thin plane. Comment the
# `volume` argument out to get the flat disc back.
volume = DiscVolume(disc; M=spacetime.M)

# Preview resolution / integration settings. The Metal GPU preview easily
# sustains 320×240; raise the resolution if you have headroom.
settings = PreviewSettings(width=320, height=240, dt=0.1, nmax=2000)

fig = viewfinder(cam, spacetime, bg; settings=settings,
                 title="Black Hole Viewfinder", disc=disc, volume=volume)

# Keep the script alive until the window is closed.
if !isinteractive()
    wait(Makie.getscreen(fig.scene))
end

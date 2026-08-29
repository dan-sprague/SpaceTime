# Black-hole flythrough: real-time GPU-ray-traced flight around a
# Schwarzschild black hole with a volumetric accretion disc.
#
# Run with:
#     julia -t auto,1 --project fly_demo.jl
#
# Controls: drag to look, scroll to dolly, WASD to move, Q/E down/up,
# Z/C to roll, Shift for 5× speed. Auto speed slows you down near the
# horizon — try flying inside the photon sphere (r < 3M).

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using SpaceTime
using GLMakie
using StaticArrays
using FileIO

bg = load("starmap_g4k.jpg")

spacetime = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0),
                     density_falloff=0.8)
volume = DiscVolume(disc; M=spacetime.M)

# Start well outside, skimming the disc plane, looking at the hole — the
# flight inward (and through the photon sphere) is the show.
cam = SpaceTime.Camera(SVector(30.0, 1.1, 1.6), SVector(0.0, 0.0, 0.0),
                       SVector(0.0, 0.0, 1.0), Lens(24.0))

fig = flythrough(cam, spacetime, bg; disc=disc, volume=volume)

isinteractive() || wait(Makie.getscreen(fig.scene))

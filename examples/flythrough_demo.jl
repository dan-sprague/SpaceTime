# Black-hole flythrough: real-time GPU-ray-traced flight around a
# Schwarzschild black hole with a volumetric accretion disc.
#
# Run with:
#     julia -t auto,1 --project examples/flythrough_demo.jl
#
# The camera is a ship with mass on a true GR worldline: W/S A/D Q/E thrust
# in the ship frame, Space retro-burns to rest, Shift is a 4× burn, drag to
# look, Z/C to roll. Let go of the keys and you free-fall — try burning
# sideways near r = 8M and cutting the engines to enter an orbit, and watch
# the telemetry: the accelerometer reads zero while you fall.

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using SpaceTime
using GLMakie
using StaticArrays
using FileIO

bg = load(joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"))

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

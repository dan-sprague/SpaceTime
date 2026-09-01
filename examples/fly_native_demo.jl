# Spacetime Simulator, native shell: the flight simulator in a bare Metal
# window — GLFW + CAMetalLayer, no Makie. The traced frame never leaves the
# GPU.
#
# Run with:
#     julia --project examples/fly_native_demo.jl
#
# Drag to look; W/S A/D Q/E thrust in the ship frame; Space retro-burn;
# Shift ×4 burn; Z/C roll; [ ] thrust setting; - = time warp; V volumetric
# gas; R relativistic shading; L lens; X reset; Esc quit. Engines off is
# exact free fall — telemetry lives in the window title.

using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using SpaceTime
using StaticArrays
using FileIO

bg = load(joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"))

spacetime = Schwarzschild(1.0)
# White balance decides which radius renders neutral. At 10000 K that was
# r = 6M, leaving 93% of the disc's area cooler than white — the beige. At
# 5000 K neutral sits near 15M, so the hot inner disc reads white-to-blue and
# only the outer edge stays warm. Star colour is unaffected: it has its own
# white point (see `STAR_WB_TEMPERATURE`).
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=5000.0),
                     density_falloff=0.8)
# `haze` adds the diffuse envelope that greys the shadow. Erosion is off:
# it carves real fine structure into the gas, but it only ever removes gas, so
# it needs `opacity_scale` raised to compensate — the two move together.
volume = DiscVolume(disc; M=spacetime.M, haze=0.03, haze_height=4.0)

cam = SpaceTime.Camera(SVector(30.0, 1.1, 1.6), SVector(0.0, 0.0, 0.0),
                       SVector(0.0, 0.0, 1.0), Lens(24.0))

# RES=1080|1440|2160 selects the render resolution (default 1440).
res = get(Dict("1080" => (1920, 1080), "1440" => (2560, 1440),
               "2160" => (3840, 2160)), get(ENV, "RES", "1440"), (2560, 1440))
fly_native(cam, spacetime, bg; disc=disc, volume=volume,
           width=res[1], height=res[2],
           max_seconds=haskey(ENV, "SPACETIME_SMOKE") ? 8.0 : Inf)

# "Hectic" hero shot: camera skimming the disc plane, lensing wrapping the
# disc over the shadow, orbital motion blur, hot exposure with aggressive
# bloom and streaks.
#
# Full-quality 4K render (expect this to take a while):
#     julia -t auto --project examples/hero_shot.jl
# Fast low-res iteration pass:
#     HERO_LOWRES=1 julia -t auto --project examples/hero_shot.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpaceTime
using StaticArrays
using LinearAlgebra
using FileIO
using Random
using Images: clamp01nan

lowres = get(ENV, "HERO_LOWRES", "0") == "1"
W, H, S, TS = lowres ? (480, 270, 1, 2) : (3840, 2160, 4, 6)

bg = load(joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"))
bh = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0),
                     density_falloff=0.8)

# Volumetric turbulent gas disc (comment out to fall back to the thin plane).
# Seeded: unseeded this falls back to the global RNG, so the filament pattern
# was different on every run and no hero frame could ever be reproduced or
# compared against another.
volume = DiscVolume(disc; M=1.0, rng=Xoshiro(7))

# The "COOL SCENE" composition: just above the disc plane,
# rolled 20°, looking through the disc at the shadow.
world_up = SVector(0.0, 0.0, 1.0)
world_right = SVector(0.0, 1.0, 0.0)
θ_roll = deg2rad(20.0)
tilted_up = normalize(world_up * cos(θ_roll) + world_right * sin(θ_roll))
base_pos = SVector(30.0, 1.1, 1.6)
target = SVector(0.0, 0.0, 0.0)

# Slow orbital drift over the shutter interval gives tangential motion blur —
# the "energetic" smear — while thin-lens optics add shallow depth of field.
function camera_at(t)
    ϕ = 0.002 * t
    pos = SVector(base_pos[1] * cos(ϕ) - base_pos[2] * sin(ϕ),
                  base_pos[1] * sin(ϕ) + base_pos[2] * cos(ϕ),
                  base_pos[3])
    ThinLensCamera(pos, target, tilted_up;
                   focal_length=33.0, f_number=5.6, focus_distance=27.0)
end

println("Rendering $(W)×$(H), samples=$S, time_samples=$TS on ",
        Threads.nthreads(), " threads…")
@time img = render_motion(camera_at, 0.0, 1.0, bh, bg;
                          disc=disc, volume=volume, width=W, height=H,
                          samples=S, time_samples=TS)

# One shared, resolution-independent grade. `LOOK_HERO` is `LOOK_FILM` with
# glare kernels 6x tighter — the grade this still was tuned to. Both are
# fractions of frame height, so a HERO_LOWRES draft and the 4K deliverable are
# now the same picture at different sharpness rather than different looks.
post = apply_look!(img, LOOK_HERO; rng=Xoshiro(4242))

outfile = joinpath(@__DIR__, "..", lowres ? "hero_shot_lowres.png" : "hero_shot.png")
save(outfile, map(clamp01nan, rotr90(post)))
println("Saved: ", outfile)

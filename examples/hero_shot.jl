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
using Images: clamp01nan

lowres = get(ENV, "HERO_LOWRES", "0") == "1"
W, H, S, TS = lowres ? (480, 270, 1, 2) : (3840, 2160, 4, 6)

bg = load(joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"))
bh = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0),
                     density_falloff=0.8)

# Volumetric turbulent gas disc (comment out to fall back to the thin plane).
volume = DiscVolume(disc; M=1.0)

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

post = postprocess(img;
                   gain=1.0,
                   exposure=0.8,
                   gamma=0.2,
                   bloom_strength=1.0,
                   threshold=0.5,
                   bloom_radius=10.0,
                   bloom_power=1.5,
                   streak_strength=2.0,
                   streak_length=0.1,
                   streak_width=1.0,
                   n_spikes=4,
                   tonemap=:aces,
                   tonemap_hue_preserve=0.75)

sensor_expose!(post; iso=400.0, t_exp=1.0, read_noise_e=2.0, saturation=1.0e6)
apply_vignette!(post; strength=0.3)
apply_lens_distortion!(post; k1=-0.02)

outfile = joinpath(@__DIR__, "..", lowres ? "hero_shot_lowres.png" : "hero_shot.png")
save(outfile, map(clamp01nan, rotr90(post)))
println("Saved: ", outfile)

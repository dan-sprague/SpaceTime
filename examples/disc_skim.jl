# "Photon sphere dive" composition, saved from a viewfinder session
# (2026-08-29): camera at r ≈ 2.81M — INSIDE the photon sphere (3M), only
# 0.4 r_s above the horizon. From here the whole outside universe is
# compressed into a bright-rimmed circle (the escape cone, bounded by the
# photon-sphere critical curve) and the rest of the sky is horizon-black.
#
# Viewfinder state (for re-entering the composition live; move speed was 2.0):
#   pos (2.6, 0.7, 0.8) · yaw −286.3° · pitch 68.3° · roll 20°
#   focal 10mm · pinhole (no thin lens) · volumetric disc ON
#
# Full-quality 4K render:
#     julia -t auto --project examples/disc_skim.jl
# Fast low-res iteration pass:
#     SKIM_LOWRES=1 julia -t auto --project examples/disc_skim.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpaceTime
using StaticArrays
using FileIO
using Images: clamp01nan

lowres = get(ENV, "SKIM_LOWRES", "0") == "1"
W, H, S = lowres ? (480, 270, 1) : (3840, 2160, 4)

bg = load(joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"))
bh = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0),
                     density_falloff=0.8)
volume = DiscVolume(disc; M=1.0)

# Rebuild the camera through the same FlyCamState → Camera path the
# viewfinder uses, so the framing matches the live preview exactly.
state = SpaceTime.FlyCamState(SVector(2.6, 0.7, 0.8),
                              deg2rad(-286.3),   # yaw
                              deg2rad(68.3),     # pitch
                              deg2rad(20.0))     # roll
cam = SpaceTime.camera_from_state(state, 10.0, 5.6, 27.0, false)

println("Rendering $(W)×$(H), samples=$S on ", Threads.nthreads(), " threads…")
@time img = render(cam, bh, bg; disc=disc, volume=volume,
                   width=W, height=H, samples=S)

# Post: hero-shot defaults — tweak to taste (the saved state above is
# camera-only; post sliders weren't captured).
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
                   tonemap_hue_preserve=0.75,
                   # The 4K frame is the deliverable, so the look is anchored
                   # there and SKIM_LOWRES drafts scale down to match it.
                   ref_height=2160)

sensor_expose!(post; iso=400.0, t_exp=1.0, read_noise_e=2.0, saturation=1.0e6)
apply_vignette!(post; strength=0.3)

outfile = joinpath(@__DIR__, "..",
                   lowres ? "disc_skim_lowres.png" : "disc_skim.png")
save(outfile, map(clamp01nan, rotr90(post)))
println("Saved: ", outfile)

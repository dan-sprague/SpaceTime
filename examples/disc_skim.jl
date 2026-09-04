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
#     julia -t auto --project=examples examples/disc_skim.jl
# Fast low-res iteration pass:
#     SKIM_LOWRES=1 julia -t auto --project=examples examples/disc_skim.jl

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
# `LOOK_HERO` is the shared still grade, in fractions of frame height, so a
# SKIM_LOWRES draft and the 4K deliverable are the same picture at different
# sharpness. Diffraction and barrel distortion stay off, as they were here.
post = apply_look!(img, with_look(LOOK_HERO; f_number=0.0, distortion_k1=0.0);
                   rng=Xoshiro(4242))

outfile = joinpath(@__DIR__, "..",
                   lowres ? "disc_skim_lowres.png" : "disc_skim.png")
save(outfile, map(clamp01nan, rotr90(post)))
println("Saved: ", outfile)

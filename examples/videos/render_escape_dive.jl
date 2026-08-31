# Porthole escape + dive: fly out of the porthole, turn to the hero framing,
# then dive back through the disc and across the horizon — frame fades to
# black (website intro cut). Rendered WITH observer-frame relativity: the
# camera tetrad is boosted by the ship's velocity along the path, giving
# aberration, motion Doppler, and beaming for disc and sky alike.
# Run from the repo root:  julia -t auto,1 --project examples/videos/render_escape_dive.jl
#
# Env vars: RES=proxy|final, SAVE_TIFF=1|0, SAMPLES (default 2),
#           NFRAMES (default 900), OUT (mp4 path), T_M (video length in units
#           of M-time, default 220, peak ~0.5c — smaller = faster ship =
#           stronger aberration/Doppler/beaming; 110 blows out to blue-white)

using SpaceTime, StaticArrays, LinearAlgebra, FileIO, Random, Printf
using Images: clamp01nan, RGB
using FFMPEG_jll

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "escape_dive_path.jl"))
include(joinpath(@__DIR__, "video_common.jl"))

const RES = get(ENV, "RES", "proxy")
const FINAL = RES == "final"
const SAVE_TIFF = get(ENV, "SAVE_TIFF", "1") == "1"
const W, H = FINAL ? (1920, 1080) : (640, 360)
const SAMPLES = parse(Int, get(ENV, "SAMPLES", "2"))
const NFRAMES = parse(Int, get(ENV, "NFRAMES", "900"))
const T_M = parse(Float64, get(ENV, "T_M", "220.0"))
# Fade fully to black BEFORE the camera crosses r = 2M (frame ~872): the
# kernel's shadow-kill criterion flips discretely at the horizon and would
# otherwise show as a one-frame snap. Black holds to the end for the web cut.
const FADE_START = 820
const FADE_END = 866

frames = joinpath(ROOT, "renders", "escape_dive", RES)
mkpath(joinpath(frames, "png"))
SAVE_TIFF && mkpath(joinpath(frames, "linear"))
outmp4 = get(ENV, "OUT",
    joinpath(ROOT, FINAL ? "porthole_dive_1080p.mp4" : "porthole_dive_proxy_360p.mp4"))

bg = load(joinpath(ROOT, "starmap_g4k.jpg"))
st = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0), density_falloff=0.8)
vol = DiscVolume(disc; M=1.0, emission_scale=0.35, opacity_scale=0.3,
                 scale_height=0.18, turbulence=0.0, rng=Xoshiro(7))   # smooth gas
ctx = MetalPreviewContext(bg, 480, 270; dt=0.1, nmax=1000, disc=disc, volume=vol)

t0 = time()
for f in 1:NFRAMES
    t = (f - 1) / (NFRAMES - 1)
    fade = clamp((FADE_END * NFRAMES / 900 - f) /
                 ((FADE_END - FADE_START) * NFRAMES / 900), 0.0, 1.0)
    if fade <= 0.0   # fully black: skip the trace, write the frame directly
        save(joinpath(frames, "png", @sprintf("f%04d.png", f)), zeros(RGB{Float32}, H, W))
        continue
    end
    pos, tgt, up, fe = dive_path_at(t)
    cam = SpaceTime.Camera(pos, tgt, up, 0.55)
    v = dive_beta(t; T_M=T_M)
    βl = SVector(dot(v, cam.fwd), dot(v, cam.right), dot(v, cam.up_local))
    img = render_draft_mtl(ctx, cam, st; width=W, height=H, samples=SAMPLES,
                           dt=0.02, fisheye_deg=fe, relativistic=true, beta=βl)
    SAVE_TIFF && save_master(joinpath(frames, "linear", @sprintf("f%04d.tiff", f)), rotr90(img))
    post = postprocess(img; gain=1.0, exposure=0.8, gamma=0.2, bloom_strength=1.0,
                       threshold=0.5, bloom_radius=10.0, bloom_power=1.5,
                       streak_strength=2.0, streak_length=0.1, streak_width=1.0,
                       n_spikes=4, tonemap=:aces, tonemap_hue_preserve=0.75)
    sensor_expose!(post; iso=400.0, t_exp=1.0, read_noise_e=2.0, saturation=1.0e6)
    apply_vignette!(post; strength=0.3)
    if fade < 1.0
        fd = Float32(fade)
        post = map(c -> typeof(c)(fd * c.r, fd * c.g, fd * c.b), post)
    end
    save(joinpath(frames, "png", @sprintf("f%04d.png", f)), map(clamp01nan, rotr90(post)))
    f % 100 == 0 && (println("dive ", f, "/", NFRAMES, "  ",
        round((time() - t0) / 60, digits=1), " min"); flush(stdout))
end

pngpat = joinpath(frames, "png", "f%04d.png")
run(`$(ffmpeg()) -y -framerate 30 -i $pngpat -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium $outmp4`)
println("DIVE_DONE ", outmp4)

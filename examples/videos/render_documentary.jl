# "Ship-cam" documentary orbit: realistic 21-degree sweep over 30 s, thin lens
# (33mm f/5.6), OU mount jitter, lens dust, micro-streak events.
# Run from the repo root with:  julia -t auto,1 --project examples/videos/render_documentary.jl
#
# Env vars:
#   RES=proxy|final|4k  proxy = 640x360 (default, ~40 min), final = 1920x1080
#                       (~5.5 h), 4k = 3840x2160 (~22 h, ~70 GB of masters)
#   SAVE_TIFF=1|0     also save the raw linear render of every frame as Float32
#                     TIFF (default 1) into renders/documentary/<RES>/linear/
#                     for grading in post. Losslessly deflate-compressed in
#                     place (~1.3x — sampling noise limits it); final ~17 GB.
#   SAMPLES=n         override lens samples per axis (default 4 => 16 passes)
#   NFRAMES=n         frame count (default 900); OUT=path overrides the mp4
#   RESUME=1|0        skip frames already on disk (default 1) — a long render
#                     can be run in daily chunks; the last existing frame is
#                     re-rendered in case it was cut off mid-write. Delete the
#                     frames dir (or RESUME=0) after changing look parameters.
#   STOP_AFTER=h      exit cleanly after ~h hours (0 = no limit): finishes the
#                     current frame, prints progress, encodes only when all
#                     frames exist. Relaunch later to continue.

using SpaceTime, StaticArrays, LinearAlgebra, FileIO, Random, Printf
using Images: clamp01nan
using FFMPEG_jll

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "orbit_cam.jl"))
include(joinpath(@__DIR__, "video_common.jl"))

const RES = get(ENV, "RES", "proxy")
const SAVE_TIFF = get(ENV, "SAVE_TIFF", "1") == "1"
const W, H = RES == "4k" ? (3840, 2160) :
             RES == "final" ? (1920, 1080) : (640, 360)
const SAMPLES = parse(Int, get(ENV, "SAMPLES", "4"))
const NFRAMES = parse(Int, get(ENV, "NFRAMES", "900"))
# Pixel-unit effects (dust size, streak length) were tuned at 360p; scale with res.
const SC = H / 360.0

frames = joinpath(ROOT, "renders", "documentary", RES)
mkpath(joinpath(frames, "png"))
SAVE_TIFF && mkpath(joinpath(frames, "linear"))
outmp4 = get(ENV, "OUT",
    joinpath(ROOT, "renders", RES == "4k" ? "orbit_shipcam_4k.mp4" :
                   RES == "final" ? "orbit_shipcam_1080p.mp4" :
                                    "orbit_shipcam_proxy_360p.mp4"))

bg = load(joinpath(ROOT, "assets", "starmap_g4k.jpg"))
st = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0), density_falloff=0.8)
vol = DiscVolume(disc; M=1.0, rng=Xoshiro(3))
ctx = MetalPreviewContext(bg, 480, 270; dt=0.1, nmax=1000, disc=disc, volume=vol)

J = Jitter(21)
ev_rng = Xoshiro(555)
events = Dict(rand(ev_rng, 1:NFRAMES) => rand(ev_rng, UInt32) for _ in 1:14)

const RESUME = get(ENV, "RESUME", "1") == "1"
const STOP_AFTER = parse(Float64, get(ENV, "STOP_AFTER", "0"))   # hours
_pngf(f) = joinpath(frames, "png", @sprintf("f%04d.png", f))
_tiff(f) = joinpath(frames, "linear", @sprintf("f%04d.tiff", f))
_done(f) = isfile(_pngf(f)) && (!SAVE_TIFF || isfile(_tiff(f)))
# Redo the newest existing frame: it may have been cut off mid-write.
start_f = RESUME ? max(1, something(findfirst(!_done, 1:NFRAMES), NFRAMES + 1) - 1) : 1
start_f > 1 && println("resuming at frame ", start_f, "/", NFRAMES)

t0 = time()
stopped_early = false
for f in 1:NFRAMES
    t = (f - 1) / (NFRAMES - 1)
    jx = step!(J, 1 / 30)   # advance jitter even when skipping: keeps resumes deterministic
    f < start_f && continue
    if STOP_AFTER > 0 && (time() - t0) > STOP_AFTER * 3600
        println("STOPPED_AFTER_HOURS at frame ", f - 1, "/", NFRAMES,
                " — relaunch to resume"); flush(stdout)
        global stopped_early = true
        break
    end
    pos, tgt, up = orbit_cam(t, jx)
    cam = ThinLensCamera(pos, tgt, up; focal_length=33.0, f_number=5.6,
                         focus_distance=norm(pos))
    img = render_draft_mtl(ctx, cam, st; width=W, height=H, samples=SAMPLES,
                           dt=0.02, rng=Xoshiro(1000 + f))
    # Linear HDR frame, untouched by any grading — the master for post.
    SAVE_TIFF && save_master(_tiff(f), rotr90(img))
    post = postprocess(img; gain=1.0, exposure=0.8, gamma=0.2, bloom_strength=1.0,
                       threshold=0.5, bloom_radius=10.0, bloom_power=1.5,
                       streak_strength=2.0, streak_length=0.1, streak_width=1.0,
                       n_spikes=4, tonemap=:aces, tonemap_hue_preserve=0.75)
    apply_lens_dust!(post; lens_dust=LensDust(count=25, size_min=1.5 * SC, size_max=6.0 * SC,
                     opacity_min=0.05, opacity_max=0.22), rng=Xoshiro(99))
    if haskey(events, f)
        apply_micro_streaks!(post; streaks=MicroStreaks(count=rand(Xoshiro(events[f]), 1:2),
                             length_min=8.0 * SC, length_max=45.0 * SC),
                             rng=Xoshiro(events[f]))
    end
    sensor_expose!(post; iso=400.0, t_exp=1.0, read_noise_e=2.0, saturation=1.0e6)
    apply_vignette!(post; strength=0.3)
    apply_lens_distortion!(post; k1=-0.02)
    save(_pngf(f), map(clamp01nan, rotr90(post)))
    f % 100 == 0 && (println("documentary ", f, "/", NFRAMES, "  ",
        round((time() - t0) / 60, digits=1), " min"); flush(stdout))
end

if !stopped_early && all(f -> isfile(_pngf(f)), 1:NFRAMES)
    pngpat = joinpath(frames, "png", "f%04d.png")
    run(`$(ffmpeg()) -y -framerate 30 -i $pngpat -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium $outmp4`)
    println("DOCUMENTARY_DONE ", outmp4)
else
    n = count(f -> isfile(_pngf(f)), 1:NFRAMES)
    println("DOCUMENTARY_PARTIAL ", n, "/", NFRAMES, " frames on disk")
end

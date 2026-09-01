# Website background loop: 8 s seamless orbit hold at the hero composition,
# live fluid disc at 2.5x time-lapse, loop-exact sinusoidal drift/jitter, and a
# 1 s crossfade of the tail into the head so the (non-periodic) turbulence wraps.
# Run from the repo root with:  julia -t auto,1 --project examples/videos/render_loop.jl
#
# Env vars:
#   RES=proxy|final   proxy = 640x360 (default, ~40 min), final = 1920x1080 (~1.7 h)
#   SAVE_TIFF=1|0     keep the linear Float32 TIFF masters (default 1) in
#                     renders/loop/<RES>/linear/ — the 240 crossfaded loop
#                     frames, ready for grading in post. Losslessly deflate-
#                     compressed in place (~1.3x — sampling noise limits it);
#                     final ~5 GB.
#   SAMPLES=n         override lens samples per axis (default 4 => 16 passes)
#   NLOOP=n           loop frame count (default 240); OUT=path overrides the mp4
#
# Note: the crossfade is applied to the LINEAR frames (before grading), which is
# physically a double exposure; the graded PNGs are made from the blended result.

using SpaceTime, StaticArrays, LinearAlgebra, FileIO, Random, Printf
using Images: clamp01nan
using GLMakie: RGBf
using FFMPEG_jll

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "video_common.jl"))

const RES = get(ENV, "RES", "proxy")
const FINAL = RES == "final"
const SAVE_TIFF = get(ENV, "SAVE_TIFF", "1") == "1"
const W, H = FINAL ? (1920, 1080) : (640, 360)
const SAMPLES = parse(Int, get(ENV, "SAMPLES", "4"))
const NLOOP = parse(Int, get(ENV, "NLOOP", "240"))
const NX = min(30, NLOOP); const NT = NLOOP + NX   # 8 s loop + 1 s crossfade tail

# Shared, resolution-independent video grade, plus dust on the front element.
# Diffraction and barrel distortion stay off, as they have always been here.
const LOOK = with_look(LOOK_FILM; f_number=0.0, distortion_k1=0.0,
    dust = LensDust(count=25, size_min=1.5 / 360, size_max=6.0 / 360,
                    opacity_min=0.05, opacity_max=0.22))

frames = joinpath(ROOT, "renders", "loop", RES)
mkpath(joinpath(frames, "png"))
mkpath(joinpath(frames, "linear"))   # required intermediate for the crossfade
outmp4 = get(ENV, "OUT", joinpath(ROOT, "renders", FINAL ? "loop_1080p.mp4" : "loop_proxy_360p.mp4"))
lin(f) = joinpath(frames, "linear", @sprintf("f%04d.tiff", f))

bg = load(joinpath(ROOT, "assets", "starmap_g4k.jpg"))
st = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0), density_falloff=0.8)
vol = DiscVolume(disc; M=1.0, rng=Xoshiro(3))
ctx = MetalPreviewContext(bg, 480, 270; dt=0.1, nmax=1000, disc=disc, volume=vol)
sim = DiscFluidSim(vol, disc; M=1.0)
for _ in 1:150   # warm the eddies up before frame 1
    step_sim!(sim, ctx; dt=0.08)
end

p0 = SVector(30.0, 1.1, 1.6)
up0 = SVector(0.0, sind(20.0), cosd(20.0))
fwd0 = normalize(-p0)
right0 = normalize(cross(fwd0, up0))
upl0 = cross(right0, fwd0)

t0 = time()
for f in 1:NT
    ph = 2π * (f - 1) / NLOOP    # loop phase: integer cycles seam perfectly
    # Breathing drift + periodic mount jitter (all sinusoids, loop-exact).
    pos = p0 + 0.12 * sin(ph) * right0 + 0.06 * sin(2ph + 1.0) * upl0
    jr = 0.05 * sin(3ph + 0.7) + 0.03 * sin(7ph + 2.1)
    ju = 0.04 * sin(5ph + 1.9) + 0.03 * sin(11ph + 0.3)
    rl = deg2rad(0.1) * sin(2ph + 0.5)
    fwd = normalize(-pos)
    right = normalize(cross(fwd, up0))
    upl = cross(right, fwd)
    tgt = jr * right + ju * upl
    upj = normalize(cos(rl) * upl + sin(rl) * right)
    cam = ThinLensCamera(pos, tgt, upj; focal_length=33.0, f_number=5.6,
                         focus_distance=norm(pos))
    step_sim!(sim, ctx; dt=2.5 / 30)   # gas at 2.5x time-lapse
    img = render_draft_mtl(ctx, cam, st; width=W, height=H, samples=SAMPLES,
                           dt=0.02, rng=Xoshiro(4000 + f))
    save_master(lin(f), rotr90(img))
    f % 60 == 0 && (println("loop ", f, "/", NT, "  ",
        round((time() - t0) / 60, digits=1), " min"); flush(stdout))
end

# Crossfade the tail into the head (in linear light), grade, save PNGs.
for k in 1:NLOOP
    img = load(lin(k))
    if k <= NX
        α = Float32(k / NX)
        tl = load(lin(NLOOP + k))
        img = map((a, b) -> RGBf(α * a.r + (1 - α) * b.r, α * a.g + (1 - α) * b.g,
                                 α * a.b + (1 - α) * b.b), img, tl)
        save_master(lin(k), img)   # linear masters hold the seamless loop
    end
    post = apply_look!(img, LOOK; rng=Xoshiro(7000 + k), dust_rng=Xoshiro(99))
    save(joinpath(frames, "png", @sprintf("f%04d.png", k)), map(clamp01nan, post))
end
for k in NLOOP+1:NT   # tail frames are folded into the head; drop them
    rm(lin(k); force=true)
end
SAVE_TIFF || rm(joinpath(frames, "linear"); recursive=true, force=true)

pngpat = joinpath(frames, "png", "f%04d.png")
run(`$(ffmpeg()) -y -framerate 30 -i $pngpat -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium $outmp4`)
println("LOOP_DONE ", outmp4)

# Porthole escape: r=2.5M porthole -> fly out -> turn back through the disc.
# Run from the repo root with:  julia -t auto,1 --project examples/videos/render_escape.jl
#
# Env vars:
#   RES=proxy|final   proxy = 640x360 (default, ~8 min), final = 1920x1080 (~70 min)
#   SAVE_TIFF=1|0     also save the raw linear render of every frame as Float32
#                     TIFF (default 1) into renders/escape/<RES>/linear/ for
#                     grading in post. Losslessly deflate-compressed in place
#                     (~1.3x — sampling noise limits it); final ~17 GB total.
#   SAMPLES=n         override rays-per-pixel-axis (default 2)
#   NFRAMES=n         frame count (default 900); OUT=path overrides the mp4
#   REL=1|0           observer-frame relativity (default 0): boosts the camera
#                     tetrad by the ship velocity along the path — aberration,
#                     motion Doppler, beaming. Output name gains "_rel".
#   T_M=x             flight duration in M-time for REL (default 140, ~0.5c peak)
#   HUD=1|0           burn a small upper-left telemetry overlay into the graded
#                     frames (default = REL): radius + regime, speed in c, ship
#                     proper time vs earth (infinity) time for a 1e5 Msun hole.
#                     Linear TIFF masters stay clean.
#   GAS=live|static|smooth  disc gas: live fluid simulation stepped through the
#                     flight (default), frozen fBm filaments, or smooth slab.
#   BETA_SMOOTH=h     half-width (in path time) of the velocity smoothing
#                     window (default 0.015). The braking turn at t≈0.72
#                     sweeps beta through zero; small h makes the aberration
#                     "wobble" of the hole's apparent size abrupt.
#   ONLY=n            render just frame n at full settings and exit (no mp4) —
#                     the approval frame before committing to a long render.

using SpaceTime, StaticArrays, LinearAlgebra, FileIO, Random, Printf
using Images: clamp01nan
using FFMPEG_jll

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "escape_path.jl"))
include(joinpath(@__DIR__, "video_common.jl"))

const RES = get(ENV, "RES", "proxy")
const FINAL = RES == "final"
const SAVE_TIFF = get(ENV, "SAVE_TIFF", "1") == "1"
const W, H = FINAL ? (1920, 1080) : (640, 360)
const SAMPLES = parse(Int, get(ENV, "SAMPLES", "2"))
const NFRAMES = parse(Int, get(ENV, "NFRAMES", "900"))
const REL = get(ENV, "REL", "0") == "1"
const T_M = parse(Float64, get(ENV, "T_M", "140.0"))
const HUD = get(ENV, "HUD", REL ? "1" : "0") == "1"
const GAS = get(ENV, "GAS", "live")
const BETA_SMOOTH = parse(Float64, get(ENV, "BETA_SMOOTH", "0.015"))
const ONLY = parse(Int, get(ENV, "ONLY", "0"))   # 0 = full sequence

# The shared video grade, resolution independent: every length in `LOOK_FILM`
# is a fraction of frame height, so RES=proxy and RES=final differ only in
# sharpness. This shot has always run without an aperture-diffraction kernel
# and without barrel distortion, so both stay off here rather than being
# quietly switched on by the shared look.
const LOOK = with_look(LOOK_FILM; f_number=0.0, distortion_k1=0.0)

tag = REL ? "_rel" : ""
frames = joinpath(ROOT, "renders", "escape$(tag)", RES)
mkpath(joinpath(frames, "png"))
SAVE_TIFF && mkpath(joinpath(frames, "linear"))
outmp4 = get(ENV, "OUT",
    joinpath(ROOT, "renders", FINAL ? "porthole_escape$(tag)_1080p.mp4" :
                           "porthole_escape$(tag)_proxy_360p.mp4"))

bg = load(joinpath(ROOT, "assets", "starmap_g4k.jpg"))
st = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0), density_falloff=0.8)
vol = DiscVolume(disc; M=1.0, emission_scale=0.35, opacity_scale=0.3,
                 scale_height=0.18, turbulence=(GAS == "smooth" ? 0.0 : 1.0),
                 rng=Xoshiro(7))
ctx = MetalPreviewContext(bg, 480, 270; dt=0.1, nmax=1000, disc=disc, volume=vol)
sim = GAS == "live" ? DiscFluidSim(vol, disc; M=1.0) : nothing
if sim !== nothing
    for _ in 1:150   # warm the eddies up before frame 1
        step_sim!(sim, ctx; dt=0.08)
    end
end

# Ship velocity from the path: wide central difference (low-passes keyframe
# acceleration kinks), smooth tanh speed limit, smoothstep taper to zero at
# the r = 2.5M porthole (the observer there is the static porthole frame).
function escape_beta(t; h=BETA_SMOOTH)
    ta, tb = clamp(t - h, 0.0, 1.0), clamp(t + h, 0.0, 1.0)
    pa, _, _, _ = path_at(ta)
    pb, _, _, _ = path_at(tb)
    v = (pb - pa) / (max(tb - ta, 1.0e-9) * T_M)
    sp = norm(v)
    sp > 1.0e-9 && (v = v * (0.92 * tanh(sp / 0.92) / sp))
    pos, _, _, _ = path_at(t)
    x = clamp(norm(pos) - 2.5, 0.0, 1.0)
    v * (x * x * (3.0 - 2.0 * x))
end

# Ship proper time: dτ = dt·√(1−2M/r)/γ (gravitational × velocity dilation);
# earth time is Schwarzschild coordinate time. Seconds assume a 1e5 Msun hole
# (GM/c³ = 0.4926 s per M-time unit — same convention as orbit_cam.jl).
const TUNIT = 4.9255e-6 * 1.0e5
dtc = frame_span(T_M, NFRAMES)
τ_ship = 0.0

t0 = time()
for f in 1:NFRAMES
    t = frame_t(f, NFRAMES)
    pos, tgt, up, fe = path_at(t)
    # The ship's velocity rides on the camera, in world coordinates. It used to
    # be projected onto the camera basis here and passed to the renderer
    # alongside the camera, where the two could disagree.
    cam = SpaceTime.Camera(pos, tgt, up, 0.55;
                           velocity = REL ? escape_beta(t) : SVector(0.0, 0.0, 0.0))
    sim !== nothing && step_sim!(sim, ctx; dt=2.5 / 30)   # gas at 2.5x time-lapse
    ONLY > 0 && f != ONLY && continue   # sim still steps: frame ONLY is exact
    img = render_draft_mtl(ctx, cam, st; width=W, height=H, samples=SAMPLES,
                           dt=0.02, fisheye_deg=fe, relativistic=REL)
    # Linear HDR frame, untouched by any grading — the master for post.
    SAVE_TIFF && save_master(joinpath(frames, "linear", @sprintf("f%04d.tiff", f)), rotr90(img))
    post = apply_look!(img, LOOK; rng=Xoshiro(7000 + f))
    rot = map(clamp01nan, rotr90(post))
    if HUD
        r_h = norm(pos)
        sp_h = norm(cam.velocity)
        γ_h = 1.0 / sqrt(1.0 - min(sp_h^2, 0.999))
        f > 1 && (global τ_ship += dtc * sqrt(max(1.0 - 2.0 / r_h, 0.0)) / γ_h)
        draw_hud!(rot, (@sprintf("R %5.2f M  %s", r_h, hud_region(r_h)),
                        @sprintf("SPEED %.2f c", sp_h),
                        @sprintf("LENS FISHEYE %3.0f DEG", 2.0 * fe),
                        @sprintf("TIME US    %6.1f s", τ_ship * TUNIT),
                        @sprintf("TIME EARTH %6.1f s", (f - 1) * dtc * TUNIT)))
    end
    save(joinpath(frames, "png", @sprintf("f%04d.png", f)), rot)
    f % 100 == 0 && (println("escape ", f, "/", NFRAMES, "  ",
        round((time() - t0) / 60, digits=1), " min"); flush(stdout))
end

if ONLY > 0
    println("APPROVAL_FRAME ", joinpath(frames, "png", @sprintf("f%04d.png", ONLY)))
else
    pngpat = joinpath(frames, "png", "f%04d.png")
    run(`$(ffmpeg()) -y -framerate 30 -i $pngpat -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium $outmp4`)
    println("ESCAPE_DONE ", outmp4)
end

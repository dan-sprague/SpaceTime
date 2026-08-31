# Porthole escape, 4k24 delivery: r=2.5M porthole -> fly out -> turn back
# through the disc. The relativistic variant of render_escape.jl at 3840x2160,
# 24 fps, with OU mount jitter, static gas, and no TIFF masters — grading is
# approved from a uniformly sampled contact sheet instead (APPROVE=20).
# Run from the repo root with:
#   julia -t auto,1 --project examples/videos/render_escape_4k24.jl
#
# Env vars:
#   APPROVE=n         render n frames uniformly sampled over the flight into
#                     renders/escape4k*/approve/ and exit (no mp4). Frames are
#                     seeded per index, so approved frames are bit-identical
#                     to the same frames of the full render.
#   ONLY=n            render just frame n and exit (no mp4).
#   SAMPLES=n         rays-per-pixel-axis (default 2)
#   NFRAMES=n         frame count (default 720 = 30 s at 24 fps); OUT=path
#                     overrides the mp4 path
#   REL=1|0           observer-frame relativity (default 1): boosts the camera
#                     tetrad by the ship velocity along the path — aberration,
#                     motion Doppler, beaming. Output name gains "_rel".
#   T_M=x             flight duration in M-time for REL (default 140, ~0.5c peak)
#   HUD=1|0           burn the telemetry overlay into the frames (default = REL)
#   GAS=static|live|smooth  disc gas (default static: frozen fBm filaments —
#                     deterministic, no per-frame sim cost). live steps the
#                     fluid sim through the flight like render_escape.jl.
#   JITTER=1|0        OU mount jitter, pointing + roll (default 1). Applied to
#                     the camera basis only; the ship velocity (and hence the
#                     aberration) follows the smooth path.
#   BETA_SMOOTH=h     velocity smoothing half-width (default 0.015)
#   WIDTH/HEIGHT      frame size (default 3840x2160)
#
# The postprocess chain (bloom + diffraction streaks + ACES) and PNG saves run
# on worker threads behind a bounded channel, overlapped with the GPU render:
# at 4k the ~7 s of CPU post per frame hides entirely under the ~12 s render.

using SpaceTime, StaticArrays, LinearAlgebra, FileIO, Random, Printf
using Images: clamp01nan
using FFMPEG_jll

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "escape_path.jl"))
include(joinpath(@__DIR__, "orbit_cam.jl"))    # Jitter / step! (OU mount noise)
include(joinpath(@__DIR__, "video_common.jl"))

const FPS = 24
const W = parse(Int, get(ENV, "WIDTH", "3840"))
const H = parse(Int, get(ENV, "HEIGHT", "2160"))
const SAMPLES = parse(Int, get(ENV, "SAMPLES", "2"))
const NFRAMES = parse(Int, get(ENV, "NFRAMES", "720"))
const REL = get(ENV, "REL", "1") == "1"
const T_M = parse(Float64, get(ENV, "T_M", "140.0"))
const HUD = get(ENV, "HUD", REL ? "1" : "0") == "1"
const GAS = get(ENV, "GAS", "static")
const JITTER = get(ENV, "JITTER", "1") == "1"
const BETA_SMOOTH = parse(Float64, get(ENV, "BETA_SMOOTH", "0.015"))
const ONLY = parse(Int, get(ENV, "ONLY", "0"))     # 0 = full sequence
const APPROVE = parse(Int, get(ENV, "APPROVE", "0"))

tag = REL ? "_rel" : ""
frames = joinpath(ROOT, "renders", "escape4k$(tag)")
pngdir = joinpath(frames, APPROVE > 0 ? "approve" : "png")
mkpath(pngdir)
outmp4 = get(ENV, "OUT", joinpath(ROOT, "renders", "porthole_escape$(tag)_4k24.mp4"))

# Frames to actually render this run (jitter/sim state still advances through
# every frame, so any subset is exact against the full sequence).
render_set = ONLY > 0 ? Set([ONLY]) :
             APPROVE > 0 ? Set(unique(round.(Int, range(1, NFRAMES, length=APPROVE)))) :
             Set(1:NFRAMES)

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

# HUD telemetry is precomputed for the whole flight so post can run out of
# order on the workers: ship proper time is dτ = dt·√(1−2M/r)/γ, earth time
# is Schwarzschild coordinate time. Seconds assume a 1e5 Msun hole.
const TUNIT = 4.9255e-6 * 1.0e5
dtc = T_M / (NFRAMES - 1)
hud_lines = Vector{NTuple{5,String}}(undef, NFRAMES)
let τ = 0.0
    for f in 1:NFRAMES
        t = (f - 1) / (NFRAMES - 1)
        pos, _, _, fe = path_at(t)
        r_h = norm(pos)
        sp_h = REL ? norm(escape_beta(t)) : 0.0
        γ_h = 1.0 / sqrt(1.0 - min(sp_h^2, 0.999))
        f > 1 && (τ += dtc * sqrt(max(1.0 - 2.0 / r_h, 0.0)) / γ_h)
        hud_lines[f] = (@sprintf("R %5.2f M  %s", r_h, hud_region(r_h)),
                        @sprintf("SPEED %.2f c", sp_h),
                        @sprintf("LENS FISHEYE %3.0f DEG", 2.0 * fe),
                        @sprintf("TIME US    %6.1f s", τ * TUNIT),
                        @sprintf("TIME EARTH %6.1f s", (f - 1) * dtc * TUNIT))
    end
end

# --- Post + save workers behind a bounded channel ---------------------------
# ~100 MB per in-flight 4k frame; 4 slots bounds the producer comfortably.
const NPOST = 3
done_count = Threads.Atomic{Int}(0)
t0 = time()
ch = Channel{Tuple{Int,Matrix{RGB{Float32}}}}(4)
workers = map(1:NPOST) do _
    Threads.@spawn for (f, img) in ch
        post = postprocess(img; gain=1.0, exposure=0.8, gamma=0.2, bloom_strength=1.0,
                           threshold=0.5, bloom_radius=10.0, bloom_power=1.5,
                           streak_strength=2.0, streak_length=0.1, streak_width=1.0,
                           n_spikes=4, tonemap=:aces, tonemap_hue_preserve=0.75)
        sensor_expose!(post; iso=400.0, t_exp=1.0, read_noise_e=2.0,
                       saturation=1.0e6, rng=Xoshiro(70_000 + f))
        apply_vignette!(post; strength=0.3)
        rot = map(clamp01nan, rotr90(post))
        HUD && draw_hud!(rot, hud_lines[f])
        save(joinpath(pngdir, @sprintf("f%04d.png", f)), rot)
        n = Threads.atomic_add!(done_count, 1) + 1
        if n % 24 == 0 || length(render_set) < NFRAMES
            println("escape4k ", n, "/", length(render_set), "  ",
                    round((time() - t0) / 60, digits=1), " min"); flush(stdout)
        end
    end
end

# --- Producer: sim/jitter state advances every frame, GPU renders the set ---
J = Jitter(21)
for f in 1:NFRAMES
    t = (f - 1) / (NFRAMES - 1)
    jx = JITTER ? step!(J, 30.0 / (NFRAMES - 1);
                        rms=SVector(2.7e-3, 2.7e-3, deg2rad(0.15))) :
                  SVector(0.0, 0.0, 0.0)
    sim !== nothing && step_sim!(sim, ctx; dt=2.5 / FPS)
    f in render_set || continue

    pos, tgt, up, fe = path_at(t)
    # Mount jitter: pointing offset (in radians, scaled by target distance)
    # transverse to the view, plus roll about the view axis. Position and the
    # ship velocity stay on the smooth path, so the relativistic aberration
    # doesn't wobble with the mount.
    fwd = normalize(tgt - pos)
    right = normalize(cross(fwd, up))
    upl = cross(right, fwd)
    d = norm(tgt - pos)
    tgt = tgt + d * (jx[1] * right + jx[2] * upl)
    up_j = normalize(cos(jx[3]) * upl + sin(jx[3]) * right)
    cam = SpaceTime.Camera(pos, tgt, up_j, 0.55)

    v = REL ? escape_beta(t) : SVector(0.0, 0.0, 0.0)
    βl = SVector(dot(v, cam.fwd), dot(v, cam.right), dot(v, cam.up_local))
    img = render_draft_mtl(ctx, cam, st; width=W, height=H, samples=SAMPLES,
                           dt=0.02, fisheye_deg=fe, relativistic=REL, beta=βl,
                           rng=Xoshiro(9_000 + f))
    put!(ch, (f, img))
end
close(ch)
foreach(wait, workers)

if ONLY > 0
    println("APPROVAL_FRAME ", joinpath(pngdir, @sprintf("f%04d.png", ONLY)))
elseif APPROVE > 0
    println("APPROVAL_SET ", pngdir, "  (", length(render_set), " frames)")
else
    pngpat = joinpath(pngdir, "f%04d.png")
    run(`$(ffmpeg()) -y -framerate $FPS -i $pngpat -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium $outmp4`)
    println("ESCAPE_DONE ", outmp4)
end

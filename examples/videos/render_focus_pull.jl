# "Focus pull": the escape's ending, made the whole film. The camera starts
# near the horizon already fleeing at ~0.9c, looking back. Relativistic
# aberration compresses the rear view, so the 33mm rectilinear prime holds the
# whole shot (the escape's porthole needed a fisheye because it starts at
# rest; we don't) — and recession Doppler starts the disc as a dim red ember
# that blooms into color as the camera brakes. While it flies, the disc swings
# from vertical to the hero's 20-degree wrap (a roll schedule — the hero's
# "angle" IS the camera roll, so the landing is frame-exact). Once stationary
# on the hero pose, the operator racks focus: out past the hole, a hunting
# correction, then settling exactly on focus = norm(pos) — the photon-ring
# focus the hero still was measured to need. OU mount jitter runs throughout,
# heavy under the burn and calming to hand tremor for the rack; micro-streak
# events fire on a dozen scattered frames.
#
# Run from the repo root:
#   julia -t auto,1 --project examples/videos/render_focus_pull.jl
#
# Env vars:
#   RES=proxy|final|4k  proxy = 640x360 (default), final = 1920x1080,
#                       4k = 3840x2160
#   NFRAMES=n         frame count (default 576 = 24 s at 24 fps)
#   SAMPLES=n         rays-per-pixel-axis during flight (default 2)
#   TAIL_SAMPLES=n    rays-per-pixel-axis once stationary (default 4 — the
#                     rack needs the hero landing's sampling; flight motion
#                     hides 2-sample noise)
#   T_M=x             shot duration in M-time (default 147: peak ~0.9c at the
#                     start of the flight, stationary from the settle on)
#   BETA_CAP=x        tanh speed limit (default 0.92)
#   GAS_RATE=x        gas rotation rate multiplier (default 1.0: Keplerian at
#                     the shot's own M-time — ~4.5 inner-edge turns over 24 s)
#   REL=1|0           observer-frame relativity (default 1)
#   JITTER=1|0        OU mount jitter (default 1)
#   STREAKS=n         micro-streak events (default 12; 0 off)
#   ONLY=n / APPROVE=n / RESUME=1|0   as in render_escape_4k24.jl
#   OUT=path          override the mp4 path
#
# Timeline (t in [0,1], 24 s):
#   0.00-0.58  flight: rho 2.7 -> 30.06 along the hero azimuth, decelerating;
#              roll 90 -> 0 extra (disc vertical -> hero wrap), color blooming
#   0.58-0.71  settle: stationary, jitter calming, focus still short at 20M
#   0.71-0.94  rack: 20 -> 33 (overshoot) -> 29.3 -> 30.5 -> 30.063 (locked)
#   0.94-1.00  hold: the hero frame, breathing on hand tremor

using SpaceTime, StaticArrays, LinearAlgebra, FileIO, Random, Printf
using Images: clamp01nan, RGB
using FFMPEG_jll

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "orbit_cam.jl"))    # Jitter / step! (OU mount noise)
include(joinpath(@__DIR__, "video_common.jl"))

const FPS = 24
const RES = get(ENV, "RES", "proxy")
const W, H = RES == "4k" ? (3840, 2160) :
             RES == "final" ? (1920, 1080) : (640, 360)
const NFRAMES = parse(Int, get(ENV, "NFRAMES", "576"))
const SAMPLES = parse(Int, get(ENV, "SAMPLES", "2"))
const TAIL_SAMPLES = parse(Int, get(ENV, "TAIL_SAMPLES", "4"))
const T_M = parse(Float64, get(ENV, "T_M", "147.0"))
const BETA_CAP = parse(Float64, get(ENV, "BETA_CAP", "0.92"))
const GAS_RATE = parse(Float64, get(ENV, "GAS_RATE", "1.0"))
const REL = get(ENV, "REL", "1") == "1"
const JITTER = get(ENV, "JITTER", "1") == "1"
const NSTREAK = parse(Int, get(ENV, "STREAKS", "12"))
const ONLY = parse(Int, get(ENV, "ONLY", "0"))
const APPROVE = parse(Int, get(ENV, "APPROVE", "0"))
const RESUME = get(ENV, "RESUME", "1") == "1"

frames = joinpath(ROOT, "renders", "focus_pull", RES)
pngdir = joinpath(frames, APPROVE > 0 ? "approve" : "png")
mkpath(pngdir)
outmp4 = get(ENV, "OUT", joinpath(ROOT, "renders", "focus_pull_$(RES).mp4"))

# --- Timeline ---------------------------------------------------------------
const T_ARRIVE = 0.58     # position stationary from here
const T_CALM   = 0.71     # rack begins
const T_LOCK   = 0.94     # focus locked; hold to the end

smoothstep(x) = (x = clamp(x, 0.0, 1.0); x * x * (3.0 - 2.0 * x))

# Radial flight along the hero azimuth: rho(t) decelerates as (1-u)^2.8, so
# the shot opens at peak speed (we join the escape mid-flight) and the brake
# is continuous down to zero at arrival.
const P_HERO = SVector(30.0, 1.1, 1.6)
const RHO1 = norm(P_HERO)              # 30.063: the hero radius
const RHO0 = 2.7
const UHAT = P_HERO / RHO1
rho_at(t) = begin
    u = clamp(t / T_ARRIVE, 0.0, 1.0)
    RHO0 + (RHO1 - RHO0) * (1.0 - (1.0 - u)^2.8)
end
pos_at(t) = max(rho_at(t), 2.6) * UHAT

# The hero's up (20-degree wrap), plus an extra roll about the view axis that
# opens at 90 degrees (disc standing vertical in frame) and eases out by the
# time the settle ends. Rotating up about fwd IS "the disc rotating".
const UP_HERO = SVector(0.0, sind(20.0), cosd(20.0))
roll_extra(t) = deg2rad(90.0) * (1.0 - smoothstep(t / 0.62))

# Ship velocity: wide central difference of the position path over M-time,
# tanh-limited, smoothstep-tapered to zero at the r=2.6 floor (the escape's
# porthole treatment).
function beta_at(t; h=0.015)
    ta, tb = clamp(t - h, 0.0, 1.0), clamp(t + h, 0.0, 1.0)
    v = (pos_at(tb) - pos_at(ta)) / (max(tb - ta, 1.0e-9) * T_M)
    sp = norm(v)
    sp > 1.0e-9 && (v = v * (BETA_CAP * tanh(sp / BETA_CAP) / sp))
    x = clamp(norm(pos_at(t)) - 2.6, 0.0, 1.0)
    v * (x * x * (3.0 - 2.0 * x))
end

# Focus rack: short during flight (the frame arrives ALMOST right — ring
# soft), then the operator's pull: fast out with an overshoot, a hunting
# correction, a small recover, and the lock on norm(pos). Aperture stays
# fixed at the final focus (a real lens's pupil doesn't grow when you rack).
const FOCUS_LOCK = RHO1
const AP = FOCUS_LOCK / 5.6
seg(t, a, b, fa, fb) = fa + (fb - fa) * smoothstep((t - a) / (b - a))
focus_at(t) =
    t < T_CALM ? 20.0 :
    t < 0.80 ? seg(t, T_CALM, 0.80, 20.0, 33.0) :
    t < 0.86 ? seg(t, 0.80, 0.86, 33.0, 29.3) :
    t < 0.905 ? seg(t, 0.86, 0.905, 29.3, 30.5) :
                seg(t, 0.905, T_LOCK, 30.5, FOCUS_LOCK)

# Jitter amplitude: heavy under the burn, hand tremor once calm.
jitter_amp(t) = t < T_ARRIVE ? 1.0 + 1.5 * (1.0 - t / T_ARRIVE)^2 :
                t < T_CALM ? 1.0 : 0.55

# --- Scene ------------------------------------------------------------------
bg = load(joinpath(ROOT, "assets", "starmap_g4k.jpg"))
st = Schwarzschild(1.0)
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0),
                     density_falloff=0.8)
vol = DiscVolume(disc; M=1.0, rng=Xoshiro(7))
ctx = MetalPreviewContext(bg, 480, 270; dt=0.1, nmax=1000, disc=disc, volume=vol)

# Keplerian differential rotation of the gas grid at the shot's own M-time
# (Schwarzschild: Omega = s^-3/2). Pure function of t — pose_at rotates the
# grid per shutter stratum, so the gas motion-blurs with everything else.
const rot_scratch = similar(vol.density)
function rotate_volume!(t_M::Real)
    d = vol.density
    sc = rot_scratch
    nr, nphi, nz = size(d)
    lsin, lsout = Float64(vol.log_s_in), Float64(vol.log_s_out)
    Threads.@threads for i in 1:nr
        s = exp(lsin + (i - 1) / (nr - 1) * (lsout - lsin))
        fshift = (GAS_RATE * t_M / s^1.5) / (2π) * nphi
        j0 = floor(Int, fshift)
        tj = fshift - j0
        @inbounds for k in 1:nz, j in 1:nphi
            ja = mod(j - 1 - j0 - 1, nphi) + 1
            jb = mod(j - 1 - j0, nphi) + 1
            sc[i, j, k] = d[i, ja, k] * Float32(tj) +
                          d[i, jb, k] * Float32(1.0 - tj)
        end
    end
    copyto!(ctx.vol_gpu, sc)
    return nothing
end

# The settled hero grade (2026-09-02): exposure feeds the shadow halos, the
# raised bloom threshold holds the plume, streaks stay at 1.6.
const LOOK = with_look(LOOK_HERO; iso=640.0, exposure=0.7,
                       bloom_strength=0.75, streak_strength=1.6, threshold=0.7)

const FOV33 = 18.0 / 33.0

function cam_for(t, jx)
    pos = pos_at(t)
    fwd = normalize(-pos)
    up0 = UP_HERO
    # Extra roll about the view axis (the disc's vertical -> hero swing).
    ψ = roll_extra(t)
    right0 = normalize(cross(fwd, up0))
    upl0 = cross(right0, fwd)
    up = normalize(cos(ψ) * upl0 + sin(ψ) * right0)
    right = normalize(cross(fwd, up))
    upl = cross(right, fwd)
    amp = jitter_amp(t)
    tgt = norm(pos) * (amp * jx[1] * right + amp * jx[2] * upl)  # target ~ origin
    c, s = cos(amp * jx[3]), sin(amp * jx[3])
    upj = normalize(c * upl + s * right)
    v = REL ? beta_at(t) : SVector(0.0, 0.0, 0.0)
    SpaceTime.Camera(pos, tgt, upj, FOV33; velocity=v)
end

# --- Frame selection, jitter chain, streak events ---------------------------
render_set = ONLY > 0 ? Set([ONLY]) :
             APPROVE > 0 ? Set(unique(round.(Int, range(1, NFRAMES, length=APPROVE)))) :
             Set(1:NFRAMES)

jxs = fill(SVector(0.0, 0.0, 0.0), NFRAMES + 1)
if JITTER
    let Jp = Jitter(21)
        for f in 1:(NFRAMES + 1)
            jxs[f] = step!(Jp, frame_span(NFRAMES / FPS, NFRAMES);
                           rms=SVector(2.7e-3, 2.7e-3, deg2rad(0.15)))
        end
    end
end

ev_rng = Xoshiro(555)
events = NSTREAK > 0 ?
    Dict(rand(ev_rng, 1:NFRAMES) => rand(ev_rng, UInt32) for _ in 1:NSTREAK) :
    Dict{Int,UInt32}()

# --- Post + save workers behind a bounded channel ---------------------------
const NPOST = 3
done_count = Threads.Atomic{Int}(0)
t0 = time()
ch = Channel{Tuple{Int,Matrix{RGB{Float32}}}}(4)
workers = map(1:NPOST) do _
    Threads.@spawn for (f, img) in ch
        ev = get(events, f, nothing)
        look_f = ev === nothing ? LOOK :
            with_look(LOOK; streaks=MicroStreaks(count=rand(Xoshiro(ev), 1:2),
                                                 length_min=8.0 / 360,
                                                 length_max=45.0 / 360))
        post = apply_look!(img, look_f;
                           rng=Xoshiro(70_000 + f),
                           streak_rng=ev === nothing ? nothing : Xoshiro(ev))
        save(joinpath(pngdir, @sprintf("f%04d.png", f)),
             map(clamp01nan, rotr90(post)))
        n = Threads.atomic_add!(done_count, 1) + 1
        if n % 24 == 0 || length(render_set) < NFRAMES
            println("focus_pull ", n, "/", length(render_set), "  ",
                    round((time() - t0) / 60, digits=1), " min"); flush(stdout)
        end
    end
end

# --- Producer ---------------------------------------------------------------
_png(f) = joinpath(pngdir, @sprintf("f%04d.png", f))
const SHUTTER = frame_span(0.5, NFRAMES)   # 180-degree shutter in t units
for f in 1:NFRAMES
    f in render_set || continue
    RESUME && ONLY == 0 && APPROVE == 0 && isfile(_png(f)) && continue
    t = frame_t(f, NFRAMES)
    # Pose at shutter fraction s: path time and jitter advance across the open
    # shutter; the gas grid rotates to the same instant, so its motion blurs.
    pose_at(s) = begin
        ts = clamp(t + (s - 0.5) * SHUTTER, 0.0, 1.0)
        jx = jxs[f] + (s - 0.5) * 0.5 * (jxs[f + 1] - jxs[f])
        rotate_volume!(ts * T_M)
        cam_for(ts, jx)
    end
    img = render_draft_mtl(ctx, pose_at(0.5), st; width=W, height=H,
                           samples=(t < T_ARRIVE ? SAMPLES : TAIL_SAMPLES),
                           dt=0.02, relativistic=REL,
                           aperture_world=AP, focus_dist=focus_at(t),
                           camera_at=pose_at, rng=Xoshiro(9_000 + f))
    put!(ch, (f, img))
end
close(ch)
foreach(wait, workers)

if ONLY > 0
    println("APPROVAL_FRAME ", _png(ONLY))
elseif APPROVE > 0
    println("APPROVAL_SET ", pngdir, "  (", length(render_set), " frames)")
elseif all(f -> isfile(_png(f)), 1:NFRAMES)
    pngpat = joinpath(pngdir, "f%04d.png")
    run(`$(ffmpeg()) -y -framerate $FPS -i $pngpat -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium $outmp4`)
    println("FOCUS_PULL_DONE ", outmp4)
else
    println("FOCUS_PULL_PARTIAL ", count(f -> isfile(_png(f)), 1:NFRAMES),
            "/", NFRAMES, " frames on disk")
end

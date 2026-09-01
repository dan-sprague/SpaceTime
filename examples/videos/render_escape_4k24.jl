# Porthole escape, 4k24 delivery: r=2.5M porthole -> fly out -> turn back
# through the disc, landing on the hero-shot composition. The relativistic
# variant of render_escape.jl at 3840x2160, 24 fps, with the hero's dense gas,
# 180-degree-shutter motion blur, OU mount jitter, a fisheye -> 33mm lens
# transition that completes by the braking turn, deep thin-lens DoF on the
# rectilinear tail, static gas, and no TIFF masters — grading is approved from
# a uniformly sampled contact sheet instead (APPROVE=20).
# Run from the repo root with:
#   julia -t auto,1 --project examples/videos/render_escape_4k24.jl
#
# The lens: the porthole demands fisheye (100 deg > any rectilinear FOV), the
# hero landing is a 33mm rectilinear prime. Between t=0.50 and the turnaround
# at t=0.72 the fisheye half-angle eases to the FOV-matched 28.6 deg while a
# radial post-warp morphs the projection, landing exactly on the rectilinear
# mapping at t=0.72 — no distortion pop at the switch. From there the kernel's
# thin lens is active: deep DoF at F_NUMBER, focused just inside the hole.
#
# Env vars:
#   APPROVE=n         render n frames uniformly sampled over the flight into
#                     renders/escape4k*/approve/ and exit (no mp4). Frames are
#                     seeded per index, so approved frames are bit-identical
#                     to the same frames of the full render.
#   ONLY=n            render just frame n and exit (no mp4).
#   SAMPLES=n         rays-per-pixel-axis for the fisheye flight (default 2)
#   TAIL_SAMPLES=n    rays-per-pixel-axis for the rectilinear tail (default 4:
#                     the landing must match the CPU hero, and at 2 the gas
#                     halo speckles — verified sampling noise, not sensor
#                     grain. The flight hides 2-sample noise under motion.)
#   NFRAMES=n         frame count (default 720 = 30 s at 24 fps); OUT=path
#                     overrides the mp4 path
#   REL=1|0           observer-frame relativity (default 1): boosts the camera
#                     tetrad by the ship velocity along the path — aberration,
#                     motion Doppler, beaming. Output name gains "_rel".
#   T_M=x             flight duration in M-time for REL (default 140, ~0.5c peak)
#   HUD=1|0           burn the telemetry overlay into the frames (default 0)
#   GAS=static|live|smooth  disc gas (default static: frozen fBm filaments —
#                     deterministic, no per-frame sim cost). live steps the
#                     fluid sim through the flight like render_escape.jl.
#                     The volume parameters are the hero shot's (dense, thin,
#                     bright), not render_escape.jl's dilute flythrough haze.
#   FLARES=n          slow brightness flares on the static gas (default 8;
#                     0 disables). Gaussian arcs in the optically thin outer
#                     disc (12-19M) that swell over 2-4 s, decay over 4-8 s,
#                     and drift at their radius's Keplerian rate — subtle,
#                     and much slower than the live fluid sim. Ignored for
#                     GAS=live, which owns the density buffer itself.
#   F_NUMBER=n        f-stop of the 33mm tail (default 11 — deep DoF; the
#                     hero was f/5.6)
#   MOTION=1|0        180-degree-shutter motion blur (default 1): the samples^2
#                     supersampling passes double as stratified shutter times,
#                     so the blur is free. Covers path motion and mount jitter.
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
const TAIL_SAMPLES = parse(Int, get(ENV, "TAIL_SAMPLES", "4"))
const NFRAMES = parse(Int, get(ENV, "NFRAMES", "720"))
const REL = get(ENV, "REL", "1") == "1"
const T_M = parse(Float64, get(ENV, "T_M", "140.0"))
const HUD = get(ENV, "HUD", "0") == "1"
const GAS = get(ENV, "GAS", "static")
const F_NUMBER = parse(Float64, get(ENV, "F_NUMBER", "11.0"))

# The shared video grade. Every length in `LOOK_FILM` is a fraction of frame
# height, so the 4K deliverable and a 360p check render are the same look at
# different sharpness — no reference height to keep in sync. Aperture and
# distortion are set per frame below, since both change across the lens morph.
const LOOK = with_look(LOOK_FILM; f_number=0.0, distortion_k1=0.0)
const FLARES = parse(Int, get(ENV, "FLARES", "8"))
const MOTION = get(ENV, "MOTION", "1") == "1"
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
# The hero shot's gas: DiscVolume defaults (emission 0.8, opacity 1.2, scale
# height 0.08) — the dense luminous sheet that lensing wraps over the shadow.
# Seeded, unlike hero_shot.jl, so approval frames match the final run.
vol = DiscVolume(disc; M=1.0, turbulence=(GAS == "smooth" ? 0.0 : 0.8),
                 rng=Xoshiro(7))
ctx = MetalPreviewContext(bg, 480, 270; dt=0.1, nmax=1000, disc=disc, volume=vol)
sim = GAS == "live" ? DiscFluidSim(vol, disc; M=1.0) : nothing
if sim !== nothing
    for _ in 1:150   # warm the eddies up before frame 1
        step_sim!(sim, ctx; dt=0.08)
    end
end
# Slow flares modulate the static grid; the fluid sim owns that buffer itself,
# so the two are mutually exclusive. Events are scheduled over the footage
# duration at the flight's own M-time rate (T_M over the shot).
flares = (FLARES > 0 && sim === nothing) ?
    DiscFlares(vol; duration=NFRAMES / FPS, nflares=FLARES,
               M_per_s=T_M / (NFRAMES / FPS), rng=Xoshiro(11)) : nothing

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

# --- Lens schedule ----------------------------------------------------------
# Fisheye follows the path keys until T_LENS0, then eases to the FOV-matched
# half-angle FE33 by the turnaround T_LENS1 while the projection morph weight
# ramps 0 -> 1 (applied as a radial post-warp on the linear frame); from
# T_LENS1 on the render is true 33mm rectilinear with the kernel's thin lens.
const FOV33 = 18.0 / 33.0                 # 33mm full-frame: tan(vertical half-FOV)
const FE33 = rad2deg(atan(FOV33))         # FOV-matched fisheye half-angle ≈ 28.6°
const T_LENS0, T_LENS1 = 0.50, 0.72
fe_frame = Vector{Float64}(undef, NFRAMES)     # fisheye half-angle (0 = rectilinear)
morph_a = Vector{Float64}(undef, NFRAMES)      # projection morph weight
k1_frame = Vector{Float64}(undef, NFRAMES)     # hero barrel distortion, ramped in
for f in 1:NFRAMES
    t = (f - 1) / (NFRAMES - 1)
    _, _, _, fe_path = path_at(t)
    if t <= T_LENS0
        fe_frame[f], morph_a[f] = fe_path, 0.0
    elseif t < T_LENS1
        w = (t - T_LENS0) / (T_LENS1 - T_LENS0)
        w = w * w * (3.0 - 2.0 * w)
        fe_frame[f], morph_a[f] = (1.0 - w) * fe_path + w * FE33, w
    else
        fe_frame[f], morph_a[f] = 0.0, 0.0
    end
    # The hero's k1 = -0.02 barrel, eased in over the tail's first second so
    # the landing grades identically to hero_shot.jl without a mid-shot pop.
    k1_frame[f] = -0.02 * clamp((t - T_LENS1) / 0.033, 0.0, 1.0)
end

"""
Radial projection morph on a linear renderer-orientation frame: resample the
equidistant fisheye (half-angle `θm`) toward the target mapping
`θ(ρ) = (1−α)·ρ·θm + α·atan(ρ·FOV33)`. At α = 1 with θm = FE33 the output is
exactly the 33mm rectilinear frame, so the phase-3 switch is seamless. Every
angle the target needs lies inside the source frame for θm > 24°, which the
schedule guarantees.
"""
function lens_morph(img::Matrix{RGB{Float32}}, α::Float64, θm_deg::Float64)
    w, h = size(img)
    θm = deg2rad(θm_deg)
    out = similar(img)
    hh = h / 2.0
    black = RGB{Float32}(0, 0, 0)
    @inbounds for j in 1:h, i in 1:w
        u = (i - 0.5 - w / 2.0) / hh
        v = (j - 0.5 - h / 2.0) / hh
        ρ = sqrt(u * u + v * v)
        c = ρ > 1.0e-8 ?
            ((1.0 - α) * ρ * θm + α * atan(ρ * FOV33)) / (ρ * θm) : 1.0
        x = u * c * hh + w / 2.0 + 0.5
        y = v * c * hh + h / 2.0 + 0.5
        x0 = floor(Int, x); y0 = floor(Int, y)
        fx = Float32(x - x0); fy = Float32(y - y0)
        if x0 < 1 || y0 < 1 || x0 >= w || y0 >= h
            out[i, j] = black
        else
            out[i, j] = img[x0, y0]     * ((1 - fx) * (1 - fy)) +
                        img[x0 + 1, y0] * (fx * (1 - fy)) +
                        img[x0, y0 + 1] * ((1 - fx) * fy) +
                        img[x0 + 1, y0 + 1] * (fx * fy)
        end
    end
    return out
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
        pos, _, _, _ = path_at(t)
        r_h = norm(pos)
        sp_h = REL ? norm(escape_beta(t)) : 0.0
        γ_h = 1.0 / sqrt(1.0 - min(sp_h^2, 0.999))
        f > 1 && (τ += dtc * sqrt(max(1.0 - 2.0 / r_h, 0.0)) / γ_h)
        lens_str = fe_frame[f] > 0.0 ?
            @sprintf("LENS FISHEYE %3.0f DEG", 2.0 * fe_frame[f]) :
            @sprintf("LENS 33MM F/%.0f", F_NUMBER)
        hud_lines[f] = (@sprintf("R %5.2f M  %s", r_h, hud_region(r_h)),
                        @sprintf("SPEED %.2f c", sp_h),
                        lens_str,
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
        # Projection morph happens on the linear frame, before bloom/streaks,
        # so the glow follows the final geometry.
        morph_a[f] > 0.0 && (img = lens_morph(img, morph_a[f], fe_frame[f]))
        # Aperture diffraction, on the linear frame: the lens's own resolution
        # limit, which at f/11 is ~1.6 px at 4k. Applied only on the
        # rectilinear tail, where the thin lens is what we are modelling.
        # Diffraction runs only on the rectilinear tail, where the thin lens is
        # what we are modelling; barrel distortion follows the projection morph.
        # Both ride on the shared look rather than being spliced in by hand.
        look_f = with_look(LOOK;
            f_number = fe_frame[f] > 0.0 ? 0.0 : F_NUMBER,
            distortion_k1 = k1_frame[f])
        post = apply_look!(img, look_f; rng=Xoshiro(70_000 + f))
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

# --- Producer: sim state advances every frame, GPU renders the set ----------
# Mount-jitter states for every frame boundary, precomputed so the shutter can
# interpolate between them: real handheld blur is mostly the shake itself.
jxs = fill(SVector(0.0, 0.0, 0.0), NFRAMES + 1)
if JITTER
    let Jp = Jitter(21)
        for f in 1:(NFRAMES + 1)
            jxs[f] = step!(Jp, 30.0 / (NFRAMES - 1);
                           rms=SVector(2.7e-3, 2.7e-3, deg2rad(0.15)))
        end
    end
end

# Camera at path time `t` with mount jitter `jx`: pointing offset (radians,
# scaled by target distance) transverse to the view, plus roll about the view
# axis. Position and the ship velocity stay on the smooth path, so the
# relativistic aberration doesn't wobble with the mount.
function cam_for(t, jx)
    pos, tgt, up, _ = path_at(t)
    fwd = normalize(tgt - pos)
    right = normalize(cross(fwd, up))
    upl = cross(right, fwd)
    d = norm(tgt - pos)
    tgt = tgt + d * (jx[1] * right + jx[2] * upl)
    up_j = normalize(cos(jx[3]) * upl + sin(jx[3]) * right)
    return SpaceTime.Camera(pos, tgt, up_j, FOV33)
end

const SHUTTER = 0.5 / (NFRAMES - 1)   # 180-degree shutter, in path-time
for f in 1:NFRAMES
    t = (f - 1) / (NFRAMES - 1)
    sim !== nothing && step_sim!(sim, ctx; dt=2.5 / FPS)
    f in render_set || continue
    # Flare gain is a pure function of footage time, so a sampled subset of
    # frames is exact — no need to advance it on skipped frames.
    flares !== nothing && apply_flares!(ctx, vol, flares, (f - 1) / FPS)

    # Pose at shutter fraction s ∈ [0,1): path time and jitter both advance
    # across the open shutter (jitter lerped toward the next frame's state).
    pose_at(s) = begin
        ts = clamp(t + (s - 0.5) * SHUTTER, 0.0, 1.0)
        jx = jxs[f] + (s - 0.5) * 0.5 * (jxs[f + 1] - jxs[f])
        c = cam_for(ts, jx)
        v = REL ? escape_beta(ts) : SVector(0.0, 0.0, 0.0)
        (c, SVector(dot(v, c.fwd), dot(v, c.right), dot(v, c.up_local)))
    end
    cam, βl = pose_at(0.5)

    # Deep thin-lens DoF on the rectilinear tail, focused just inside the
    # camera radius like the hero (27M at r=30M). aperture = focus/f_number.
    fo = 0.9 * norm(cam.pos)
    ap = fe_frame[f] > 0.0 ? 0.0 : fo / F_NUMBER

    img = render_draft_mtl(ctx, cam, st; width=W, height=H,
                           samples=(fe_frame[f] > 0.0 ? SAMPLES : TAIL_SAMPLES),
                           dt=0.02, fisheye_deg=fe_frame[f], relativistic=REL,
                           beta=βl, aperture_world=ap, focus_dist=fo,
                           camera_at=(MOTION ? pose_at : nothing),
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

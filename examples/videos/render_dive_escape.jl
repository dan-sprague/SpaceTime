# "Dive & escape": an establishing shot at rest, a relativistic dive over the
# RECEDING limb of the disc, a full prograde revolution skimming just above
# the gas, a single steep plunge through the equatorial plane in the clean
# gap inside the disc's inner rim, then the escape — fleeing below the plane
# while the disc swings from vertical to the 20-degree wrap and recession
# Doppler lets the colors bloom back in — and a breathing hold on the
# 10-degrees-above-equator hero framing. Pinhole 10mm ultra-wide throughout
# (no DoF, no focus rack), so the whole disc stays in frame through the wrap
# and the drama is all aberration, Doppler, and lensing.
# Sky: procedural starfield (set_starfield!, texture replaced).
#
# Visibility rules learned from the first pass (which spent the wrap inside
# the rim's glow and printed five seconds of white):
#   - the wrap flies ~3 gas scale heights above the midplane (H = 0.08 s,
#     so z ~ +0.7 at rho ~ 3.3) — the disc glows below, doesn't envelop;
#   - the approach and wrap run over the RECEDING limb (azimuth increasing,
#     the +z-spin prograde side moving away from the camera), keeping the
#     beamed blueshifted limb across the frame instead of in the flight path;
#   - the plane is crossed ONCE, steeply, at rho ~ 2.88 — inside the 3M
#     inner edge the volume grid holds no gas, so the crossing is clean;
#   - the below-plane escape keeps ~3 scale heights of clearance.
# Velocity is a wide central difference of the path over M-time, tanh-capped
# at BETA_CAP and tapered out below r=3.0 (the Kerr static tetrad holds to
# the ergosphere; the taper is a comfort margin, not a validity one) while OU
# mount jitter (calm at rest, heavy in the dive, spiking at the turn,
# settling to hand tremor) shakes only the camera basis.
#
# Run from the repo root:
#   julia -t auto,1 --project=examples examples/videos/render_dive_escape.jl
#
# Env vars:
#   RES=120|proxy|final|4k  214x120 (default: motion preview, ~0.5 s/frame),
#                           640x360, 1920x1080, 3840x2160
#   NFRAMES=n         frame count (default 768 = 32 s at 24 fps)
#   SAMPLES=n         rays-per-pixel-axis in flight (default 2)
#   HOLD_SAMPLES=n    rays-per-pixel-axis in the hold (default 3)
#   T_M=x             shot duration in M-time (default 280: ~0.9c dive)
#   BETA_CAP=x        tanh speed limit (default 0.92)
#   GAS_RATE=x        gas rotation multiplier (default 1.0, Keplerian)
#   REL=1|0, JITTER=1|0, STREAKS=n, ONLY/APPROVE/RESUME, OUT   as elsewhere
#
# Timeline (t in [0,1], 32 s). The film frame carries a schematic 3D
# trajectory panel on its right (axis limits riding the camera radius), and
# the grade rides exposure like an iris pull (see `exp_at`):
#   0.00-0.09  establishing: pinned at rest at 45M, hole and disc framed wide
#   0.09-0.28  dive: rho 45 -> 5.5, accelerating over the receding limb,
#              aim leading through the turn (`w1` in `cam_for`)
#   0.28-0.56  the revolution: full prograde wrap at rho ~ 3.2-3.5M just
#              above the gas, ending at the rim/shadow gap
#   0.56-0.66  the horizon dip: plunge through the plane at rho ~ 2.88 and
#              down to r ~ 1.35 (r₊ = 1.063) — inside the ergosphere on the
#              blended faller tetrad, aim swung into the escape cone
#   0.66-0.86  escape: rho -> 30 below the plane, roll 90 -> 0, color bloom
#   0.86-0.93  settle: stationary on the 10-deg-up hero pose (20-deg roll)
#   0.93-1.00  hold: breathing on hand tremor

using SpaceTime, StaticArrays, LinearAlgebra, FileIO, Random, Printf
using Statistics: quantile!
using Images: clamp01nan, RGB
using FFMPEG_jll

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "orbit_cam.jl"))    # Jitter / step!
include(joinpath(@__DIR__, "video_common.jl"))

const FPS = 24
const RES = get(ENV, "RES", "120")
const W, H = RES == "4k" ? (3840, 2160) :
             RES == "1440" ? (2560, 1440) :
             RES == "final" ? (1920, 1080) :
             RES == "proxy" ? (640, 360) : (214, 120)
const NFRAMES = parse(Int, get(ENV, "NFRAMES", "768"))
const SAMPLES = parse(Int, get(ENV, "SAMPLES", "2"))
const HOLD_SAMPLES = parse(Int, get(ENV, "HOLD_SAMPLES", "3"))
const T_M = parse(Float64, get(ENV, "T_M", "240.0"))
const BETA_CAP = parse(Float64, get(ENV, "BETA_CAP", "0.92"))
const GAS_RATE = parse(Float64, get(ENV, "GAS_RATE", "1.0"))
const REL = get(ENV, "REL", "1") == "1"
const JITTER = get(ENV, "JITTER", "1") == "1"
const NSTREAK = parse(Int, get(ENV, "STREAKS", "12"))
const ONLY = parse(Int, get(ENV, "ONLY", "0"))
const APPROVE = parse(Int, get(ENV, "APPROVE", "0"))
const RESUME = get(ENV, "RESUME", "1") == "1"
# Output filename tag: render variants of one frame without overwriting each
# other (TAG=... ONLY=60 RESUME=0).
const TAG = get(ENV, "TAG", "")

frames = joinpath(ROOT, "renders", "dive_escape", RES)
pngdir = joinpath(frames, APPROVE > 0 ? "approve" : "png")
mkpath(pngdir)
outmp4 = get(ENV, "OUT", joinpath(ROOT, "renders", "dive_escape_$(RES).mp4"))

# --- Path -------------------------------------------------------------------
const T_EST = 0.09
const T_TURN0, T_TURN1 = 0.28, 0.66
const T_DIP0 = 0.56
const T_ARRIVE = 0.86
const T_CALM = 0.93

smoothstep(x) = (x = clamp(x, 0.0, 1.0); x * x * (3.0 - 2.0 * x))

# End pose: the approved 10-degrees-above-equator framing (screen-up, i.e.
# z < 0 under the rotr90 save convention), 20-degree roll via UP_HERO.
const P_END = SVector(29.5, 1.1, -5.0)
const RHO1 = norm(P_END)
const UHAT_OUT = P_END / RHO1
const AZ_OUT = atand(P_END[2], P_END[1])          # ~2.1 degrees
# Dive azimuth: 120 degrees around +z from the exit direction, arriving
# above the plane so the vertical disc reads during the approach.
_rzm(φ) = SMatrix{3,3}(cos(φ), sin(φ), 0.0, -sin(φ), cos(φ), 0.0, 0.0, 0.0, 1.0)
const UHAT_IN = normalize(_rzm(deg2rad(120.0)) * SVector(1.0, 0.0, 0.12))

const SPIN = 0.998
const A2 = SPIN * SPIN

# Cylindrical key helper: azimuth winds monotonically UP (prograde, +z spin).
pk(az, ρ, z) = SVector(ρ * cosd(az), ρ * sind(az), z)
# Deep keys are specified by KERR-SCHILD radius, not Euclidean: at this spin
# rk = √(ρ²−a²) on the equator, so the horizon r₊ = 1.063 sits at ρ ≈ 1.46
# — a ρ = 1.34 key is INSIDE the hole. ρ from the defining quartic:
# ρ² = (rk²+a²)(1 − z²/rk²).
pkr(az, rk, z) = pk(az, sqrt((rk^2 + A2) * (1.0 - z^2 / rk^2)), z)

# Catmull-Rom keys (t, pos). `pos_at` pins t <= T_EST to the first key
# exactly (no spline bow — the "zoom" a nonzero end tangent prints on a
# static radial hold). The whip winds azimuth 150 -> 1082 (~2.6 revolutions
# total): a full wrap riding z ~ +0.7 above the gas, then the plunge through
# the plane at rho ~ 2.88 (the clean gap inside the disc's 3M inner rim) and
# down to r ~ 1.35 — just above r₊ = 1.063, deep in the ergosphere, where
# the dense keys keep the spline from sagging into the kill radius — before
# the escape below the plane. Duplicated tail keys pin the hold still.
const KEYS = [
    (0.00, 45.0 * UHAT_IN),
    (0.05, 45.0 * UHAT_IN),
    (T_EST, 45.0 * UHAT_IN),
    (0.18, 24.0 * UHAT_IN),
    (T_TURN0, pk(150.0, 5.5, 0.95)),
    (0.318, pk(210.0, 4.0, 0.8)),
    (0.356, pk(270.0, 3.5, 0.75)),
    (0.395, pk(330.0, 3.3, 0.7)),
    (0.433, pk(390.0, 3.2, 0.65)),
    (0.471, pk(450.0, 3.2, 0.6)),
    (0.509, pk(510.0, 3.3, 0.6)),
    (0.541, pk(555.0, 3.05, 0.35)),
    (T_DIP0, pk(590.0, 2.88, 0.0)),
    (0.585, pkr(660.0, 2.1, -0.12)),
    (0.606, pkr(730.0, 1.55, -0.2)),
    (0.6225, pkr(790.0, 1.38, -0.25)),
    (0.639, pkr(850.0, 1.55, -0.3)),
    (T_TURN1, pkr(920.0, 2.1, -0.42)),
    (0.692, pk(985.0, 3.6, -0.9)),
    (0.724, pk(1030.0, 7.0, -1.7)),
    (0.778, pk(1062.0, 15.0, -3.3)),
    (T_ARRIVE, P_END),
    (0.94, P_END),
    (1.00, P_END),
]

function _cr(p0, p1, p2, p3, s)
    0.5 * ((2.0 * p1) + (-p0 + p2) * s + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * s^2 +
           (-p0 + 3.0*p1 - 3.0*p2 + p3) * s^3)
end

function pos_at(t)
    t = clamp(t, 0.0, 1.0)
    t <= T_EST && return KEYS[1][2]   # establishing shot: exactly pinned
    n = length(KEYS)
    k = min(something(findlast(K -> K[1] <= t, KEYS), 1), n - 1)
    t1, t2 = KEYS[k][1], KEYS[k+1][1]
    s = t2 > t1 ? (t - t1) / (t2 - t1) : 0.0
    i0, i3 = max(k - 1, 1), min(k + 2, n)
    p = _cr(KEYS[i0][2], KEYS[k][2], KEYS[k+1][2], KEYS[i3][2], s)
    # Floor in KERR-SCHILD r, above the ray kill radius (0.995x the prograde
    # photon orbit, ~1.068): Euclidean norm overstates rk by up to ~a near
    # the hole, which is how a "safe-looking" spline point ends up inside
    # the horizon. Two rescale passes converge amply.
    for _ in 1:2
        w = dot(p, p) - A2
        rk = sqrt(max(0.5 * (w + sqrt(w * w + 4.0 * A2 * p[3]^2)), 1.0e-12))
        rk < 1.16 && (p = p * (1.16 / rk))
    end
    return p
end

function beta_at(t; h=0.01)
    ta, tb = clamp(t - h, 0.0, 1.0), clamp(t + h, 0.0, 1.0)
    v = (pos_at(tb) - pos_at(ta)) / (max(tb - ta, 1.0e-9) * T_M)
    sp = norm(v)
    sp > 1.0e-9 && (v = v * (BETA_CAP * tanh(sp / BETA_CAP) / sp))
    # Full aberration essentially everywhere: the blended static/faller
    # tetrad is regular to the horizon, so the taper is only a soft floor
    # right at the r < 1.16 path clamp.
    x = clamp((norm(pos_at(t)) - 1.10) / 0.30, 0.0, 1.0)
    v * (x * x * (3.0 - 2.0 * x))
end

# Disc vertical through the dive and the whip; swings to the hero wrap on
# the way out. Rotating up about fwd IS the disc rotating on screen.
const UP_HERO = SVector(0.0, sind(20.0), cosd(20.0))
roll_extra(t) = deg2rad(90.0) *
    (1.0 - smoothstep((t - T_TURN1) / (0.84 - T_TURN1)))

# Jitter amplitude: calm at rest, ramping through the dive, settling to hand
# tremor on the hold. Scaled up ~2x for the 10mm (a wide lens shows a third
# of the angular shake per radian) but capped at 2.8 through the whip —
# shake sells speed against a calm background, and in the whip the frame
# dragging IS the drama; heavier shake just smears it.
# At rest on the 10mm wide the camera holds STILL: hand tremor on a static
# ultra-wide shot with no parallax reads as video-game screen shake (real
# wide lenses hide shake; our 10/focal scaling maximizes it). Jitter fades
# in only as the dive gets moving; the 33mm hold keeps its approved tremor.
jitter_amp(t) =
    t < T_EST ? 0.05 :
    t < T_TURN0 ? 0.05 + 2.75 * smoothstep((t - T_EST) / (T_TURN0 - T_EST)) :
    t < T_TURN1 ? 2.8 :
    t < T_ARRIVE ? 1.8 + 1.5 * (1.0 - (t - T_TURN1) / (T_ARRIVE - T_TURN1))^2 :
    t < T_CALM ? 2.9 : 1.8   # 10mm units; the 10/focal scale lands these on
                             # the approved 33mm hold (0.88 / 0.55)

# --- Scene ------------------------------------------------------------------
# Sky: PROCEDURAL STARFIELD (texture replaced) — the session default.
# Spacetime: Kerr at the Thorne limit — Bardeen-exact shadow, spin-aware
# disc Doppler, and the camera rides the blended static/faller Kerr tetrad
# (frame dragging in the boost/aberration; regular through the ergosphere
# to the horizon, which the r ~ 1.35 dip depends on).
bg = load(joinpath(ROOT, "assets", "starmap_g4k.jpg"))   # ctx buffer only
st = Kerr(1.0, SPIN)   # SPIN defined with the path constants
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0),
                     density_falloff=0.8)
vol = DiscVolume(disc; M=1.0, rng=Xoshiro(7))
# nmax 20000: at the rk ~ 1.4 periapsis every escaping photon is
# near-critical and winds the photon shell for tens of orbits before
# getting out; the default budget exhausts mid-winding and the rf < 4M rule
# shades the whole sky as shadow. Early-exit keeps the shallow frames cheap.
ctx = MetalPreviewContext(bg, 480, 270; dt=0.1, nmax=20000, disc=disc,
                          volume=vol)
# Dense/bright sky so lensing reads: 2× lattice + fill ≈ 5.8× stars, 2×
# flux buys back the point-source demagnification near the limb, psf 0.8 px
# is the anti-flicker sweet spot for a moving camera (set_starfield! docs).
#
# An isotropic random star field lenses into another random star field —
# invisible. The Milky Way band is the structure that makes deflection
# legible, but the default galactic normal (0,0,1) lays the band in the
# disc plane where its lensed ring hides behind the disc. Tilt it so the
# band is a vertical great circle through the sky point directly BEHIND
# the hole on the approach axis: its warp into rings around the shadow is
# the lensing shot. concentration 4.5 gives the band real contrast.
#
# Colour: temperatures draw uniformly, and the star white point is 10000 K
# (STAR_WB_TEMPERATURE), so the default 3000–16000 K span renders 54% of
# stars gold-to-red. 6500–22000 K puts the bulk at/above the white point,
# and saturation 0.35 pulls raw blackbody chroma back to measured star
# chromaticities (even O stars are pale blue-white — Charity's dataset;
# production sky shaders mix ~2/3 toward white for the same reason).
#
# psf 0.65 px keeps the median star a point (~1.5 px FWHM): real stars are
# sub-pixel, and the convincing look is brightness variation at fixed size,
# with only the bright tail spreading — via bloom in the post look, not PSF.
const GAL_N = normalize(cross(UHAT_IN, SVector(0.0, 0.0, 1.0)))
# The map's diffuse Milky-Way glow is what makes deflection VISIBLE at the
# far framings — lensing conserves surface brightness, so a uniform point
# field lenses into another uniform point field, and only continuous
# low-frequency structure (which "magnifies without loss") warps legibly
# around the shadow. But near the gas the same glow shows through the
# semi-transparent filaments as noisy blue mush, so `texture_weight` rides
# the timeline like everything else: full for the establishing/approach,
# zero through the wrap and dip (the swirled point field carries those
# frames), moderate for the hero hold. Stars stay procedural throughout.
tw_at(t) = 1.2 * (1.0 - smoothstep((t - 0.20) / 0.08)) +
           0.6 * smoothstep((t - 0.76) / 0.10)
function sky_at!(t)
    set_starfield!(ctx; height=H, density=1152, fill=0.85, flux=0.033,
                   psf_pixels=0.65, strength=1.6, saturation=0.35,
                   texture_weight=tw_at(t),
                   galactic=(GAL_N[1], GAL_N[2], GAL_N[3]), concentration=4.5,
                   temp_min=6500, temp_max=22000)
end
sky_at!(0.0)

const rot_scratch = similar(vol.density)
function rotate_volume!(t_M::Real)
    d = vol.density; sc = rot_scratch
    nr, nphi, nz = size(d)
    lsin, lsout = Float64(vol.log_s_in), Float64(vol.log_s_out)
    Threads.@threads for i in 1:nr
        s = exp(lsin + (i - 1) / (nr - 1) * (lsout - lsin))
        fshift = (GAS_RATE * t_M / (s^1.5 + SPIN)) / (2π) * nphi
        j0 = floor(Int, fshift); tj = fshift - j0
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

# Grade knobs, env-overridable for single-frame look probes. The default
# gamma=0.2 (x^5) is the hot, crushed film look; raise toward 1.0 (linear) or
# 2.2 (sRGB) to pull midtones back out of the blacks. Exposure is NOT here —
# it is overridden per-frame by `exp_at`; scale the whole ride with EXPOF.
const LOOK_GAMMA = parse(Float64, get(ENV, "LOOK_GAMMA", "0.2"))
const LOOK_CONTRAST = parse(Float64, get(ENV, "LOOK_CONTRAST", "0.0"))
const LOOK_BLOOM = parse(Float64, get(ENV, "LOOK_BLOOM", "0.75"))
const LOOK_STREAK = parse(Float64, get(ENV, "LOOK_STREAK", "1.6"))
const LOOK_THRESHOLD = parse(Float64, get(ENV, "LOOK_THRESHOLD", "0.7"))
const LOOK_HUE = parse(Float64, get(ENV, "LOOK_HUE", "0.75"))
const LOOK = with_look(LOOK_HERO; iso=640.0, exposure=0.7,
                       gamma=LOOK_GAMMA, contrast=LOOK_CONTRAST,
                       bloom_strength=LOOK_BLOOM, streak_strength=LOOK_STREAK,
                       threshold=LOOK_THRESHOLD, tonemap_hue_preserve=LOOK_HUE)
# Focal rack: 10mm ultra-wide from the establishing through the dip (the
# escape cone and shadow edge only fit in frame at that width), then racking
# up to 33mm across the escape so the settle lands exactly on the approved
# hero framing. `focal_at` is in mm; fov factor = 18/focal.
focal_at(t) = 10.0 + 23.0 * smoothstep((t - 0.68) / (T_ARRIVE - 0.68))
fov_at(t) = 18.0 / focal_at(t)

# Exposure ride — a DP's iris pull, keyed to the scene's radiance instead of
# a single fixed stop: baseline 0.7 for the establishing and dive, ~1.5
# stops down across the wrap (the rim seen edge-on), another ~2/3 stop into
# the dip's blueshifted exit flash, recovered by the roll-out.
# `EXPOF` scales the whole ride — probe knob for single-frame tests.
const EXPOF = parse(Float64, get(ENV, "EXPOF", "1.0"))
exp_at(t) =
    EXPOF * (0.7 - 0.44 * smoothstep((t - 0.24) / 0.10) -
                   0.16 * smoothstep((t - 0.52) / 0.05) +
                   0.60 * smoothstep((t - 0.72) / 0.11))

# Dip aim mix: weight on the velocity direction (the rest points outward).
# The reference observer down there is the KS faller, plunging at ~0.77c,
# and its aberration drags the escape cone — the mix is set empirically
# (probe single frames with DIPV=... ONLY=...).
const DIPV = parse(Float64, get(ENV, "DIPV", "0.35"))
# Probe knob: fisheye vertical half-angle in degrees (0 = the normal
# rectilinear lens) — for single-frame escape-cone hunts near the horizon.
const FISH = parse(Float64, get(ENV, "FISH", "0.0"))

# Aim assists, before the star-hold blend. Turn-in and wrap: lead the look
# ~52% toward the velocity (a driver looks through the corner) and HOLD the
# lead through the whole revolution — the hole and its paisley fringe sit
# frame-left, the dragged starfield stays up-frame. Horizon dip: from
# r ~ 1.35 the look-at-centre view lies entirely inside the capture cone —
# the shadow IS the forward sky and renders black — so the aim swings to
# DIPV·v̂ + (1−DIPV)·r̂, hunting the aberrated escape cone.
function aim_base(t)
    pos = pos_at(t)
    fwd = normalize(-pos)
    vβ = beta_at(t)
    nv = norm(vβ)
    v̂ = nv > 1.0e-9 ? vβ / nv : fwd
    w1 = 0.52 * smoothstep((t - 0.20) / 0.06) *
         (1.0 - smoothstep((t - 0.50) / 0.07))
    w1 > 1.0e-3 && (fwd = normalize((1.0 - w1) * fwd + w1 * v̂))
    w2 = smoothstep((t - 0.535) / 0.045) * (1.0 - smoothstep((t - 0.70) / 0.06))
    if w2 > 1.0e-3
        d = normalize(DIPV * v̂ + (1.0 - DIPV) * normalize(pos))
        fwd = normalize((1.0 - w2) * fwd + w2 * d)
    end
    return fwd
end
# The world-fixed direction the audience is appreciating at the end of the
# dive — the star whirl. Held through the turn-in so the stars are not
# snatched away the moment the camera starts to wrap.
const AIM_HOLD = aim_base(0.245)

function cam_for(t, jx)
    pos = pos_at(t)
    fwd = aim_base(t)
    # Star hold: keep the world-fixed direction the audience is admiring at
    # the end of the dive locked through most of the wrap (~7 s), releasing
    # just before the plunge — the paisley interior gets its due later, at
    # the dip, instead of owning the whole revolution.
    wh = smoothstep((t - 0.235) / 0.03) * (1.0 - smoothstep((t - 0.44) / 0.06))
    wh > 1.0e-3 && (fwd = normalize((1.0 - wh) * fwd + wh * AIM_HOLD))
    ψ = roll_extra(t)
    right0 = normalize(cross(fwd, UP_HERO))
    upl0 = cross(right0, fwd)
    up = normalize(cos(ψ) * upl0 + sin(ψ) * right0)
    right = normalize(cross(fwd, up))
    upl = cross(right, fwd)
    # Jitter amplitudes are tuned at 10mm; scale by 10/focal through the
    # rack so screen-space shake stays continuous, landing on the approved
    # 33mm hold feel.
    amp = jitter_amp(t) * (10.0 / focal_at(t))
    tgt = pos + norm(pos) * fwd +
          norm(pos) * (amp * jx[1] * right + amp * jx[2] * upl)
    c, s = cos(amp * jx[3]), sin(amp * jx[3])
    upj = normalize(c * upl + s * right)
    v = REL ? beta_at(t) : SVector(0.0, 0.0, 0.0)
    SpaceTime.Camera(pos, tgt, upj, fov_at(t); velocity=v)
end

# --- Frames, jitter chain, streak events ------------------------------------
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

# --- Trajectory panel: schematic 3D model composited to the film's right,
# orthographic from a fixed oblique view, axis limits riding the camera
# radius (max over a +-1.5 s window, box-smoothed) so the model stays
# legible from the 45M establishing shot to the r ~ 1.35 dip ---------------
const PVW = H                        # square panel, film height
const AZV, ELV = deg2rad(-125.0), deg2rad(24.0)
const PE1 = SVector(-sin(AZV), cos(AZV), 0.0)                        # right
const PE3 = SVector(cos(ELV) * cos(AZV), cos(ELV) * sin(AZV), sin(ELV))
const PE2 = cross(PE3, PE1)                                          # up
const RPLUS = 1.0 + sqrt(1.0 - SPIN^2)
# The panel draws Euclidean positions; the equatorial horizon sits at
# ρ = √(r₊² + a²), not at r₊.
const RHEQ = sqrt(RPLUS^2 + SPIN^2)
const PATHP = [pos_at(s) for s in range(0.0, 1.0; length=900)]
const LIMS = let
    rc = [norm(pos_at(frame_t(f, NFRAMES))) for f in 1:NFRAMES]
    L = [1.35 * maximum(rc[max(1, f - 36):min(NFRAMES, f + 36)])
         for f in 1:NFRAMES]
    for _ in 1:3
        L = [sum(L[max(1, f - 12):min(NFRAMES, f + 12)]) /
             length(max(1, f - 12):min(NFRAMES, f + 12)) for f in 1:NFRAMES]
    end
    max.(L, 4.0)
end

_pmix(a::RGB{Float32}, b::RGB{Float32}, α) = RGB{Float32}(
    a.r * (1 - α) + b.r * α, a.g * (1 - α) + b.g * α, a.b * (1 - α) + b.b * α)
function _pline!(img, p0, p1, c, α)
    steps = clamp(ceil(Int, max(abs(p1[1] - p0[1]), abs(p1[2] - p0[2]))), 1, 4000)
    for i in 0:steps
        τ = i / steps
        xi = round(Int, p0[1] + τ * (p1[1] - p0[1]))
        yi = round(Int, p0[2] + τ * (p1[2] - p0[2]))
        (1 <= yi <= size(img, 1) && 1 <= xi <= size(img, 2)) || continue
        @inbounds img[yi, xi] = _pmix(img[yi, xi], c, α)
    end
end
function _pdisk!(img, p, rad, c)
    ri = ceil(Int, rad)
    for dy in -ri:ri, dx in -ri:ri
        dx * dx + dy * dy <= rad * rad || continue
        yi, xi = round(Int, p[2]) + dy, round(Int, p[1]) + dx
        (1 <= yi <= size(img, 1) && 1 <= xi <= size(img, 2)) || continue
        @inbounds img[yi, xi] = c
    end
end

function draw_panel(f::Int, cam)
    img = fill(RGB{Float32}(0.020f0, 0.024f0, 0.038f0), H, PVW)
    L = LIMS[f]
    half = 0.5 * PVW
    prj(p) = (half + dot(p, PE1) / L * half * 0.92,
              half - dot(p, PE2) / L * half * 0.92)
    ring = RGB{Float32}(0.30f0, 0.24f0, 0.16f0)
    for s in (3.0, 20.0)                              # disc rim circles
        s > 2.2 * L && continue
        prev = prj(SVector(s, 0.0, 0.0))
        for adeg in 4:4:360
            q = prj(SVector(s * cosd(adeg), s * sind(adeg), 0.0))
            _pline!(img, prev, q, ring, 0.55)
            prev = q
        end
    end
    za = min(6.0, 0.6 * L)                            # spin axis
    _pline!(img, prj(SVector(0.0, 0.0, -za)), prj(SVector(0.0, 0.0, za)),
            RGB{Float32}(0.22f0, 0.30f0, 0.42f0), 0.8)
    ph = prj(SVector(0.0, 0.0, 0.0))                  # horizon
    rpx = max(RHEQ / L * half * 0.92, 1.6)
    _pdisk!(img, ph, rpx + 1.0, RGB{Float32}(0.45f0, 0.50f0, 0.62f0))
    _pdisk!(img, ph, rpx, RGB{Float32}(0.0f0, 0.0f0, 0.0f0))
    t = frame_t(f, NFRAMES)                           # path: done / to come
    kcut = clamp(round(Int, t * (length(PATHP) - 1)) + 1, 1, length(PATHP))
    for k in 2:length(PATHP)
        done = k <= kcut
        _pline!(img, prj(PATHP[k-1]), prj(PATHP[k]),
                done ? RGB{Float32}(1.0f0, 0.55f0, 0.15f0) :
                       RGB{Float32}(0.30f0, 0.30f0, 0.34f0),
                done ? 0.9 : 0.35)
    end
    pc = prj(cam.pos)                                 # camera + aim ray
    _pline!(img, pc, prj(cam.pos + 0.14 * L * normalize(cam.fwd)),
            RGB{Float32}(0.35f0, 0.85f0, 1.0f0), 0.9)
    _pdisk!(img, pc, max(1.4, 0.011 * PVW), RGB{Float32}(1.0f0, 1.0f0, 1.0f0))
    return img
end

# Auto-balance: `exp_at` is an artistic iris ride keyed against the original
# dim sky; the star/glow boosts moved absolute scene luminance, so the keyed
# values now blow out the bright sections. Meter each linear frame and yield
# when the keyed exposure would clip the mids: hold the 99.5th-percentile
# luminance to WHITE after exposure, floored at 0.35x the keyed value so the
# dive keeps its intended punch. The disc core still clips — it should.
const HL_WHITE = 1.25f0
function balanced_exposure(img, e_key)
    lum = Float32[]
    sizehint!(lum, length(img) ÷ 9 + 1)
    for j in 1:3:size(img, 2), i in 1:3:size(img, 1)
        c = img[i, j]
        push!(lum, 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b)
    end
    p = quantile!(lum, 0.995)
    return clamp(HL_WHITE / max(p, 1.0f-6), 0.35 * e_key, e_key)
end

const NPOST = 3
done_count = Threads.Atomic{Int}(0)
t0 = time()
ch = Channel{Tuple{Int,Matrix{RGB{Float32}}}}(4)
workers = map(1:NPOST) do _
    Threads.@spawn for (f, img) in ch
        tf = frame_t(f, NFRAMES)
        ev = get(events, f, nothing)
        look_f = with_look(LOOK; exposure=balanced_exposure(img, exp_at(tf)))
        ev === nothing ||
            (look_f = with_look(look_f;
                                streaks=MicroStreaks(count=rand(Xoshiro(ev), 1:2),
                                                     length_min=8.0 / 360,
                                                     length_max=45.0 / 360)))
        post = apply_look!(img, look_f;
                           rng=Xoshiro(70_000 + f),
                           streak_rng=ev === nothing ? nothing : Xoshiro(ev))
        panel = draw_panel(f, cam_for(tf, jxs[f]))
        save(joinpath(pngdir, @sprintf("f%04d%s.png", f, TAG)),
             map(clamp01nan, hcat(rotr90(post), panel)))
        n = Threads.atomic_add!(done_count, 1) + 1
        if n % 48 == 0 || length(render_set) < NFRAMES
            println("dive_escape ", n, "/", length(render_set), "  ",
                    round((time() - t0) / 60, digits=1), " min"); flush(stdout)
        end
    end
end

_png(f) = joinpath(pngdir, @sprintf("f%04d%s.png", f, TAG))
const SHUTTER = frame_span(0.5, NFRAMES)
for f in 1:NFRAMES
    f in render_set || continue
    RESUME && ONLY == 0 && APPROVE == 0 && isfile(_png(f)) && continue
    t = frame_t(f, NFRAMES)
    pose_at_s(s) = begin
        ts = clamp(t + (s - 0.5) * SHUTTER, 0.0, 1.0)
        jx = jxs[f] + (s - 0.5) * 0.5 * (jxs[f + 1] - jxs[f])
        # Texture anchored at the END of the film: differential rotation
        # shears the gas pattern ~19 deg per radial cell over the full
        # 240 M, which winds the approved paisley into smooth concentric
        # streaks. Anchoring at t = 1 puts zero net winding on the hero
        # hold (the pristine texture) and the wound state at the far-away
        # establishing shot, where inner-gas detail is unresolvable.
        rotate_volume!((ts - 1.0) * T_M)
        sky_at!(ts)
        cam_for(ts, jx)
    end
    # Adaptive Tsit5, free-running step (the Bardeen launch test owns the
    # shadow, so the radius-capped schedule is not needed); tol is a pure
    # accuracy dial.
    img = render_draft_mtl(ctx, pose_at_s(0.5), st; width=W, height=H,
                           samples=(t < T_ARRIVE ? SAMPLES : HOLD_SAMPLES),
                           dt=0.02, order=46, tol=1.0f-5,
                           relativistic=REL, fisheye_deg=FISH,
                           camera_at=pose_at_s, rng=Xoshiro(9_000 + f))
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
    println("DIVE_ESCAPE_DONE ", outmp4)
    if W < 640   # viewable upscale for the low-res motion preview (4x)
        upmp4 = replace(outmp4, ".mp4" => "_up.mp4")
        upvf = "scale=iw*4:ih*4:flags=lanczos"
        run(`$(ffmpeg()) -y -i $outmp4 -vf $upvf -c:v libx264 -pix_fmt yuv420p -crf 17 -preset medium $upmp4`)
        println("DIVE_ESCAPE_UP ", upmp4)
    end
else
    println("DIVE_ESCAPE_PARTIAL ", count(f -> isfile(_png(f)), 1:NFRAMES),
            "/", NFRAMES, " frames on disk")
end

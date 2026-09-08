# The hero composition on the GPU: same scene, gas seed, orbital-drift motion
# blur, and camera as examples/hero_shot.jl, rendered through the Metal draft
# path (lens/time strata fold into the samples^2 passes). The grade was
# tuned on 2026-09-02 and re-exposed on 2026-09-08:
#
#   - exposure −1.5, gamma 0.45, contrast 0.25: exposed for the disc. A
#     fully hue-preserving tonemap keeps the Doppler tint of the approaching
#     side instead of converging it to white. The
#     2026-09-02 grade (exposure 0.7, gamma 0.2 = an x^5 display curve) put
#     the disc face at ~9 linear and hid the star map's ~0.13 floor by
#     crushing it; the x^5 was also why the halos died below exposure 0.55.
#     Under x^2.2 the veil carries the halos and the blacks need the pull.
#   - bloom 0.75 / streaks 1.6 with threshold 0.7: the glare kernels.
#   - ISO 1000: two notches of grain over the CPU hero's 400.
#
# Point stars are absent by physics, not by omission: at focus = norm(pos) a
# star at infinity defocuses into a disk of ~aperture/focus radians (~10 deg
# at f/5.6), erasing it. The grey halos are the lensed sky wrapped into
# concentric arcs near the shadow — structure coherent along the ring
# direction survives the blur.
#
# Composition (2026-09-08): the camera is raised 10 deg of elevation about the
# hole from the original near-equatorial pose, aimed at the hole's centre,
# then yawed/pitched to put the shadow 80% of the way to the lower-left
# rule-of-thirds intersection with the plume trailing right. Note the saved
# frame is rotated 180 deg relative to the camera's up axis, so "up" on
# screen is world -z and negative yaw/pitch move the subject left/down.
#
# Lens effects on top of the grade: veiling glare (lifts the blacks around
# the plume), coating-tinted ghosts (which stack on the plume here, since it
# sits on the optical axis), and lateral chromatic aberration on the rim.
#
# Full-quality 4K render (~5 min on an M3-class GPU):
#     julia -t auto,1 --project=examples examples/hero_shot_gpu.jl
# Fast low-res iteration pass:
#     HERO_LOWRES=1 julia -t auto,1 --project=examples examples/hero_shot_gpu.jl

using Pkg
Pkg.activate(@__DIR__)   # examples env: SpaceTime (dev) + Images

using SpaceTime
using StaticArrays
using LinearAlgebra
using FileIO
using Random
using Images: clamp01nan

lowres = get(ENV, "HERO_LOWRES", "0") == "1"
# The low-res pass keeps the final's samples: lens and shutter strata fold
# into the samples² passes, and the grade is nonlinear, so a 4-pass preview
# is biased (speckle pumps bloom and veil), not merely noisier.
W, H, S = lowres ? (960, 540, 6) : (3840, 2160, 8)

bg = load(joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"))
st = Schwarzschild(1.0)
# The disc runs in to the photon sphere, but the shader switches from
# circular orbits to the plunge below the ISCO (6M): bounded Doppler factor,
# temperature frozen at the ISCO value, density fading as (r/r_isco)³. That
# is what turns the old Doppler-beamed white beam (γ ≈ 22 from circular
# orbits at 3M) into GRMHD's faint, redshifted inner glow.
disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                     blackbody=Blackbody(wb_temperature=10000.0),
                     density_falloff=0.8)
vol = DiscVolume(disc; M=1.0, rng=Xoshiro(7))
ctx = MetalPreviewContext(bg, 480, 270; dt=0.1, nmax=1000, disc=disc, volume=vol)

# The "COOL SCENE" composition: just above the disc plane, rolled 20°.
world_up = SVector(0.0, 0.0, 1.0)
world_right = SVector(0.0, 1.0, 0.0)
θ_roll = deg2rad(20.0)
tilted_up = normalize(world_up * cos(θ_roll) + world_right * sin(θ_roll))
target = SVector(0.0, 0.0, 0.0)

const FOCAL_MM = 33.0
const FOV = 18.0 / FOCAL_MM   # full-frame pinhole fov_factor (vertical half-fov tangent)
const ASPECT = W / H
const F_NUMBER = 5.6
const LIFT_DEG = 10.0       # elevation raised toward screen-up (world -z)
const THIRDS = 0.8          # 0 = hole centred, 1 = on the lower-left third

# Sky: procedural stars over the star map, with the map at 1.5× so the lensed
# Milky Way arcs sit a stop above the disc's exposure. Point stars
# themselves defocus into ~10° discs at this aperture and only add a floor.
set_starfield!(ctx; strength=1.0, texture_weight=2.0, height=H, fov_factor=FOV)

# The original pose, raised LIFT_DEG about the hole at the same radius.
function lifted(pos, deg)
    r = norm(pos)
    horiz = SVector(pos[1], pos[2], 0.0)
    el = atan(pos[3], norm(horiz)) - deg2rad(deg)
    r * (normalize(horiz) * cos(el) + world_up * sin(el))
end
base_pos = lifted(SVector(30.0, 1.1, 1.6), LIFT_DEG)

# Rule of thirds: after aiming at the hole, rotate the look so the shadow
# lands THIRDS of the way from centre to the lower-left third intersection.
yaw_deg   = -rad2deg(atan(THIRDS * FOV * ASPECT / 3))
pitch_deg = -rad2deg(atan(THIRDS * FOV / 3))

# Orbital drift over the shutter [0,1] — the hero's motion blur. The GPU draft
# path takes a pinhole pose per time stratum; aperture and focus ride as
# kwargs, converted with the CPU convention (diameter = focus/f_number).
pose_at(t) = begin
    ϕ = 0.002 * t
    pos = SVector(base_pos[1] * cos(ϕ) - base_pos[2] * sin(ϕ),
                  base_pos[1] * sin(ϕ) + base_pos[2] * cos(ϕ),
                  base_pos[3])
    cam = SpaceTime.Camera(pos, target, tilted_up, FOV)
    pitch(yaw(cam, yaw_deg), pitch_deg)
end

# Focus on the photon ring: it is formed by rays at the critical impact
# parameter b = 3√3 M, whose closest approach to the hole sits at depth
# √(r² − b²) along the line of sight — slightly nearer than the hole's centre.
b_crit = 3 * sqrt(3.0) * st.M
fo = sqrt(norm(base_pos)^2 - b_crit^2)
ap = fo / F_NUMBER

println("GPU hero $(W)×$(H), samples=$S ($(S^2) passes), f/$F_NUMBER, focus=$fo")
@time img = render_draft_mtl(ctx, pose_at(0.5), st; width=W, height=H,
                             samples=S, dt=0.02,
                             aperture_world=ap, focus_dist=fo,
                             camera_at=pose_at, rng=Xoshiro(2000))

# gamma 0.45 (x^2.2 display curve) in place of LOOK_FILM's 0.2 (x^5): the x^5
# crushed everything under ~0.7 linear. Contrast puts back the punch as an
# S-curve rather than a black crush, and the veil no longer has to fight it.
LOOK = with_look(LOOK_HERO; iso=1000.0, exposure=-1.5, gamma=0.45, contrast=0.25,
                 tonemap_hue_preserve=1.0,
                 bloom_strength=0.75, streak_strength=1.6, threshold=0.7,
                 veil=0.1, ghosts=0.03, chromatic_aberration=0.004)
post = apply_look!(img, LOOK; rng=Xoshiro(4242))

outfile = joinpath(@__DIR__, "..", "renders",
                   lowres ? "hero_gpu_plunge_ev15_draft.png" : "hero_gpu_plunge_ev15.png")
save(outfile, map(clamp01nan, rotr90(post)))
println("Saved: ", outfile)

# The hero composition on the GPU: same scene, gas seed, orbital-drift motion
# blur, and camera as examples/hero_shot.jl, rendered through the Metal draft
# path (lens/time strata fold into the samples^2 passes) with the grade that
# was tuned against it on 2026-09-02:
#
#   - exposure 0.7 with bloom threshold 0.7: exposure feeds the lensed-sky
#     halos around the shadow (they live in the tonemap's linear regime), the
#     raised threshold keeps the plume core from blowing back out. Exposure
#     below ~0.55 kills the halos before it touches the disc — the disc sits
#     on the ACES shoulder and barely responds.
#   - bloom 0.75 / streaks 1.6: the glare that survives the exposure pull.
#   - ISO 640: one notch of grain over the CPU hero's 400.
#
# Point stars are absent by physics, not by omission: at focus = norm(pos) a
# star at infinity defocuses into a disk of ~aperture/focus radians (~10 deg
# at f/5.6), erasing it. The grey halos are the lensed sky wrapped into
# concentric arcs near the shadow — structure coherent along the ring
# direction survives the blur.
#
# Full-quality 4K render (~5 min on an M3-class GPU):
#     julia -t auto,1 --project examples/hero_shot_gpu.jl
# Fast low-res iteration pass:
#     HERO_LOWRES=1 julia -t auto,1 --project examples/hero_shot_gpu.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpaceTime
using StaticArrays
using LinearAlgebra
using FileIO
using Random
using Images: clamp01nan

lowres = get(ENV, "HERO_LOWRES", "0") == "1"
W, H, S = lowres ? (960, 540, 4) : (3840, 2160, 6)

bg = load(joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"))
st = Schwarzschild(1.0)
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
base_pos = SVector(30.0, 1.1, 1.6)
target = SVector(0.0, 0.0, 0.0)

const FOV33 = 18.0 / 33.0   # 33mm full-frame pinhole fov_factor
const F_NUMBER = 5.6

# Orbital drift over the shutter [0,1] — the hero's motion blur. The GPU draft
# path takes a pinhole pose per time stratum; aperture and focus ride as
# kwargs, converted with the CPU convention (diameter = focus/f_number).
pose_at(t) = begin
    ϕ = 0.002 * t
    pos = SVector(base_pos[1] * cos(ϕ) - base_pos[2] * sin(ϕ),
                  base_pos[1] * sin(ϕ) + base_pos[2] * cos(ϕ),
                  base_pos[3])
    SpaceTime.Camera(pos, target, tilted_up, FOV33)
end

fo = norm(base_pos)         # focus ON the photon ring at the hole
ap = fo / F_NUMBER

println("GPU hero $(W)×$(H), samples=$S ($(S^2) passes), f/$F_NUMBER, focus=$fo")
@time img = render_draft_mtl(ctx, pose_at(0.5), st; width=W, height=H,
                             samples=S, dt=0.02,
                             aperture_world=ap, focus_dist=fo,
                             camera_at=pose_at, rng=Xoshiro(2000))

LOOK = with_look(LOOK_HERO; iso=640.0, exposure=0.7,
                 bloom_strength=0.75, streak_strength=1.6, threshold=0.7)
post = apply_look!(img, LOOK; rng=Xoshiro(4242))

outfile = joinpath(@__DIR__, "..", "renders",
                   lowres ? "hero_gpu_f5p6_draft.png" : "hero_gpu_f5p6.png")
save(outfile, map(clamp01nan, rotr90(post)))
println("Saved: ", outfile)

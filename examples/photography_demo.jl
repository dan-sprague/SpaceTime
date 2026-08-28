using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using SpaceTime
using StaticArrays
using LinearAlgebra
using Colors
using FileIO

const RGBf = RGB{Float32}

# -----------------------------------------------------------------------------
# Basic scene
# -----------------------------------------------------------------------------
bh = Schwarzschild(1.0)
bg = load(joinpath(dirname(@__DIR__), "starmap_g4k.jpg"))

disc = AccretionDisc(
    inner_radius=3.0,
    outer_radius=20.0,
    blackbody=Blackbody(wb_temperature=6500.0),
    density_falloff=0.0
)

cam_pos = SVector(100.0, 30.1, 45.6)
target  = SVector(0.0, 3.0, 0.0)
world_up = SVector(0.0, 0.0, 1.0)

# -----------------------------------------------------------------------------
# 1. Pinhole camera (default, sharp everywhere)
# -----------------------------------------------------------------------------
pinhole = Camera(cam_pos, target, world_up)
img_pinhole = render(pinhole, bh, bg; disc=disc, width=512, height=256, samples=2)

# -----------------------------------------------------------------------------
# 2. Thin-lens camera with depth of field
#    Focus on the black-hole shadow; foreground/background are softly blurred.
# -----------------------------------------------------------------------------
tl_cam = ThinLensCamera(cam_pos, target, world_up;
                        focal_length=200.0,
                        sensor_width=36.0,
                        f_number=2.8,
                        focus_distance=norm(cam_pos - target))
img_dof = render(tl_cam, bh, bg; disc=disc, width=512, height=256, samples=4)

# -----------------------------------------------------------------------------
# 3. Sensor exposure + noise
#    Simulate a 1/60 s exposure at ISO 800 with realistic read noise.
# -----------------------------------------------------------------------------
img_noisy = copy(img_dof)
sensor_expose!(img_noisy; iso=800.0, t_exp=1/60, read_noise_e=4.0, add_noise=true)

# -----------------------------------------------------------------------------
# 4. Lens post-effects: vignetting and barrel distortion
# -----------------------------------------------------------------------------
apply_vignette!(img_noisy; strength=0.5, falloff=1.8)
apply_lens_distortion!(img_noisy; k1=-0.03)

# -----------------------------------------------------------------------------
# 5. Motion blur: dolly past the black hole over 2 coordinate-time units
# -----------------------------------------------------------------------------
camera_path(t) = Camera(
    SVector(100.0 - 10t, 30.1, 45.6),
    target,
    world_up
)
img_motion = render_motion(camera_path, 0.0, 2.0, bh, bg;
                           disc=disc, width=256, height=128,
                           samples=2, time_samples=4)

# -----------------------------------------------------------------------------
# 6. Camera rig transform: orbit with roll
# -----------------------------------------------------------------------------
cam_orbit = @gimbal pinhole |> yaw 15.0 |> pitch -5.0 |> roll 10.0
img_orbit = render(cam_orbit, bh, bg; disc=disc, width=512, height=256, samples=2)

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
using CairoMakie
function save_image(path, img)
    fig = Figure(size=size(img), px_per_unit=1, padding=0.0)
    ax = Axis(fig[1,1], aspect=DataAspect())
    image!(ax, img)
    hidedecorations!(ax)
    save(path, fig)
end

save_image("pinhole.png", img_pinhole)
save_image("dof.png", img_dof)
save_image("noisy.png", img_noisy)
save_image("motion.png", img_motion)
save_image("orbit.png", img_orbit)

println("Photography demo outputs saved.")

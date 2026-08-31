"""
    SpaceTime

A general-relativistic ray tracer and virtual camera for Schwarzschild black
holes, built as an educational resource and a tech demo.

Light is traced along null geodesics — on the CPU with DifferentialEquations.jl
in Schwarzschild coordinates, and on Apple-silicon GPUs with a Metal kernel in
horizon-regular Kerr–Schild coordinates — through a volumetric accretion disc
shaded by Doppler-shifted blackbody emission. A physical camera pipeline
(pinhole / thin-lens / fisheye projections, sensor noise, bloom, tonemapping,
film-look post) turns the physics into photographs, and GLMakie apps
([`viewfinder`](@ref), [`flythrough`](@ref)) make it interactive, including
flight inside the photon sphere and across the horizon. Cameras may be given a
velocity: the tetrad is Lorentz-boosted, so relativistic aberration, Doppler
shift, and beaming appear in the image exactly as an on-board observer would
see them.

See the `examples/` directory for entry points, and the README for the physics
walkthrough.
"""
module SpaceTime

using SpecialFunctions: besselj1
using ImageFiltering
using Images
using FileIO
using DifferentialEquations, GLMakie, StaticArrays
using LinearAlgebra
using Colors
using FFTW
using Random
using Metal
using TOML
using Dates
using Printf

# Physics: metric, blackbody radiation, disc models
include("gr.jl")
include("blackbody.jl")
include("accretion_disc.jl")
include("disc_volume.jl")

# Cameras and CPU ray tracing
include("camera.jl")
include("dust.jl")
include("callbacks.jl")
include("raytrace.jl")

# Image pipeline: grading, sensor model, optical effects, raw I/O
include("postprocess.jl")
include("sensor.jl")
include("sensor_effects.jl")
include("rawio.jl")

# GPU renderer (Metal, Kerr–Schild) and interactive apps
include("utils.jl")
include("viewfinder.jl")
include("ship.jl")       # GR flight dynamics; uses the viewfinder integrator
include("metal.jl")
include("disc_sim.jl")   # live fluid disc; dispatches on MetalPreviewContext
include("flythrough.jl")
include("native_shell.jl") # Makie-free Metal window shell for the simulator
include("postapp.jl")
include("app.jl")        # julia_main: standalone-app entry point (create_app)

# --- Spacetimes and geodesics
export AbstractSpacetime, Schwarzschild, Kerr, metric_inverse, hamiltonian
export RayData, WorldLine, init_photon, raytrace
export visualize_solution, trace_fan, compare_hamiltonian_drift, shadow_radius

# --- Accretion disc and emission
export Blackbody, AccretionDisc, get_disc_color_doppler
export DiscVolume, sample_disc_volume
export DiscFluidSim, step_sim!

# --- Cameras
export AbstractCamera, Camera, PinholeCamera, ThinLensCamera, FisheyeCamera
export Lens, Photon
export yaw, pitch, roll, truck, pedestal, dolly, offset_camera, @gimbal
export sample_background, get_ray_direction, get_ray
export sensor_coordinate, jittered_grid, sample_lens_point

# --- CPU renderers
export render, render_no_doppler, render_motion

# --- GPU renderer (Metal)
export MetalPreviewContext, render_preview_mtl, render_preview_mtl!,
       render_draft_mtl, set_volume_enabled!, set_disc_enabled!

# --- Interactive apps and flight dynamics
export PreviewSettings, render_preview, viewfinder, flythrough, postprocessor
export ShipState, step_ship!, ship_velocity
export fly_native

# --- Image pipeline
export postprocess, airy_convolve, generate_psf, fft_convolve, aces_tonemap,
       auto_balance!
export apply_vignette!, apply_lens_distortion!
export SensorSettings, apply_iso_gain!, add_sensor_noise!, clip!, sensor_expose!
export save_raw, load_raw

# --- Dust and practical-effects
export InterstellarDust, LensDust, MicroStreaks
export apply_dust_extinction!, apply_dust_extinction, apply_dust_glow!,
       apply_dust_post!, apply_lens_dust!, apply_micro_streaks!
export dust_extinction_rgb, dust_glow_profile, henyey_greenstein

end # module SpaceTime

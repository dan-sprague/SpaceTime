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

include("gr.jl")
include("blackbody.jl")
include("accretion_disc.jl")
include("disc_volume.jl")
include("camera.jl")
include("dust.jl")
include("callbacks.jl")
include("raytrace.jl")
include("postprocess.jl")
include("sensor.jl")
include("sensor_effects.jl")
include("utils.jl")
include("viewfinder.jl")
include("metal.jl")

export AbstractSpacetime, Schwarzschild, Kerr, metric_inverse, hamiltonian
export Blackbody, AccretionDisc
export DiscVolume, sample_disc_volume
export Lens, Photon, Camera, AbstractCamera, PinholeCamera, ThinLensCamera
export RayData, WorldLine, init_photon, render_no_doppler, render, raytrace, render_motion
export sample_background, get_ray_direction, get_ray
export sensor_coordinate, jittered_grid, sample_lens_point
export PreviewSettings, render_preview, viewfinder
export MetalPreviewContext, render_preview_mtl, render_draft_mtl,
       set_volume_enabled!
export yaw, pitch, roll, truck, pedestal, dolly, offset_camera, @gimbal
export visualize_solution, trace_fan, compare_hamiltonian_drift, shadow_radius
export postprocess, airy_convolve, generate_psf, fft_convolve, aces_tonemap
export apply_vignette!, apply_lens_distortion!
export SensorSettings, apply_iso_gain!, add_sensor_noise!, clip!, sensor_expose!
export get_disc_color_doppler
# Dust & sensor effects
export InterstellarDust, LensDust, MicroStreaks
export apply_dust_extinction!, apply_dust_extinction, apply_dust_glow!
export apply_dust_post!
export apply_lens_dust!, apply_micro_streaks!
export dust_extinction_rgb, dust_glow_profile, henyey_greenstein
export StaticArrays

end # module SpaceTime

"""
    SpaceTime

A general-relativistic ray tracer and virtual camera for Schwarzschild and
Kerr black holes, built as an educational resource and a rendering library.

Light is traced along null geodesics in horizon-regular Cartesian Kerr–Schild
coordinates — on the CPU adaptively with DifferentialEquations.jl, and on
Apple-silicon GPUs with a Metal kernel — through a volumetric accretion disc
shaded by Doppler-shifted blackbody emission. A physical camera pipeline
(pinhole / thin-lens / fisheye projections, sensor noise, bloom, tonemapping,
film-look post) turns the physics into photographs, from a safe distance,
from inside the photon sphere, or across the horizon. Cameras may be given a
velocity: the tetrad is Lorentz-boosted, so relativistic aberration, Doppler
shift, and beaming appear in the image exactly as an on-board observer would
see them. A camera can also ride a timelike worldline ([`ShipState`](@ref),
[`Track`](@ref)): free fall is exact, thrust is proper acceleration in the
ship's own frame.

See the `examples/` directory for entry points, and the README for the physics
walkthrough.
"""
module SpaceTime

using SpecialFunctions: besselj1
using ImageFiltering   # only `imfilter` and `centered`; Images itself is not needed
using FileIO
using DifferentialEquations, StaticArrays
using LinearAlgebra
using Colors
const RGBf = RGB{Float32}   # used throughout as the linear working pixel type
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
include("track.jl")   # racing tracks as timelike worldlines; needs gr.jl only
include("camera.jl")
include("dust.jl")
include("callbacks.jl")
include("raytrace.jl")

# Image pipeline: grading, sensor model, optical effects, raw I/O
include("postprocess.jl")
include("sensor.jl")
include("sensor_effects.jl")
include("rawio.jl")
include("look.jl")       # resolution-independent Look / Sampling; see its header

# Camera tetrads, CPU preview, ship dynamics, GPU renderer (Metal, Kerr–Schild)
include("utils.jl")
include("preview.jl")    # ks_camera_tetrad / ks_init_photon + fixed-step CPU preview
include("ship.jl")       # GR flight dynamics; uses the preview integrator
include("metal.jl")
include("disc_sim.jl")   # live fluid disc; dispatches on MetalPreviewContext
include("starfield.jl")  # CPU port of the kernel starfield; shares _sim_hash

# --- Spacetimes and geodesics
export AbstractSpacetime, Schwarzschild, Kerr, metric_inverse, hamiltonian
export RayData, WorldLine, init_photon, raytrace
export visualize_solution, trace_fan, compare_hamiltonian_drift, shadow_radius
# (`visualize_solution` and `compare_hamiltonian_drift` plot; their methods live
# in the Makie package extension and appear once any Makie backend is loaded.)

# --- Accretion disc and emission
export Blackbody, AccretionDisc, get_disc_color_doppler
export DiscVolume, sample_disc_volume, volume_resolution
export DiscFlare, DiscFlares, apply_flares!
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
export bake_warp_map, bake_track_maps, render_baked!, BakedTrack
export MetalPreviewContext, render_preview_mtl, render_preview_mtl!,
       render_draft_mtl, set_volume_enabled!, set_disc_enabled!,
       set_march_stride!, set_starfield!

# --- Procedural sky (both renderers)
export Starfield, starfield_color, sky_color

# --- CPU preview and flight dynamics
export PreviewSettings, render_preview
export ShipState, step_ship!, ship_velocity

# Racing tracks: timelike worldlines with bounded thrust (src/track.jl)
export Track, Burn, integrate_track, solve_track
export metric, orthonormal_frame, christoffel, normalize_timelike
export periapsis_state, encounter_track, frame_components, ks_radius
export track_sample, tidal_scalar

# --- Look and Sampling: resolution-independent grade, device-independent effort
export Look, with_look, LOOK_FILM, LOOK_HERO, apply_look!
export Sampling, with_sampling, STILL, MOTION, sampling_rng, shutter_span

# --- Image pipeline
export postprocess, airy_kernel, apply_diffraction!, generate_psf, fft_convolve, aces_tonemap,
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

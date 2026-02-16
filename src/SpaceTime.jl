module SpaceTime

using SpecialFunctions: besselj1
using ImageFiltering
using Images
using DifferentialEquations,GLMakie,StaticArrays
using LinearAlgebra
using Colors
using FFTW

include("gr.jl")
include("blackbody.jl")
include("accretion_disc.jl")
include("raytrace.jl")
include("callbacks.jl")
include("postprocess.jl")
include("utils.jl")

export AbstractSpacetime, Schwarzschild, Kerr, metric_inverse, hamiltonian
export Blackbody, AccretionDisc
export Photon, Camera, RayData, WorldLine, init_photon, render_no_doppler, render, raytrace
export sample_background, get_ray_direction
export visualize_solution, trace_fan, compare_hamiltonian_drift, shadow_radius
export postprocess, airy_convolve, generate_psf, fft_convolve, aces_tonemap
export get_disc_color_doppler
export StaticArrays

end # module SpaceTime

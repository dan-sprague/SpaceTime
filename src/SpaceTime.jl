module SpaceTime

using SpecialFunctions: besselj1
using ImageFiltering
using Images
using DifferentialEquations,CairoMakie,StaticArrays
using LinearAlgebra
using Colors

include("gr.jl")
include("blackbody.jl")
include("accretion_disc.jl")
include("raytrace.jl")
include("callbacks.jl")
include("postprocess.jl")
include("utils.jl")

export AbstractSpacetime, Schwarzschild, Kerr, metric_inverse, hamiltonian
export Blackbody, AccretionDisc
export Photon, Camera, RayData, init_photon, render_no_doppler, render, smooth_raytrace
export sample_background, get_ray_direction
export visualize_solution, trace_fan, compare_hamiltonian_drift
export postprocess, airy_convolve
export get_disc_color_doppler

end # module SpaceTime

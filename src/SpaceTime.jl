module SpaceTime

using SpecialFunctions: besselj1       
using ImageFiltering       
using Images
using DifferentialEquations,CairoMakie,StaticArrays
using LinearAlgebra
using Colors



const DISK_INNER_RADIUS = 3.0
const DISK_OUTER_RADIUS = 20.0

const _WB_REF = blackbody_rgb(6500.0)
const WB_GAINS = SVector(1.0 / _WB_REF[1], 1.0 / _WB_REF[2], 1.0 / _WB_REF[3])


const BB_TABLE_MIN = 500.0
const BB_TABLE_MAX = 30000.0
const BB_TABLE_SIZE = 1024
const BB_TABLE = [wb_blackbody_color(T) for T in range(BB_TABLE_MIN, BB_TABLE_MAX, length=BB_TABLE_SIZE)]

export AbstractSpacetime, Schwarzschild, Kerr, metric_inverse, hamiltonian
export Photon, Camera, RayData, init_photon, render_no_doppler, render, smooth_raytrace
export sample_background, get_ray_direction
export visualize_solution, trace_fan, compare_hamiltonian_drift
export postprocess, airy_convolve
export get_disc_color_doppler

end # module SpaceTime

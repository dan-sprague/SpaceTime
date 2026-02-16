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


end # module SpaceTime

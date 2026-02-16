"""
    _cie_gauss(λ, μ, σ1, σ2)
Helper function to compute a Gaussian value for the CIE color matching functions. The function takes a wavelength `λ`, a mean `μ`, and two standard deviations `σ1` and `σ2` that determine the width of the Gaussian on either side of the mean. It returns the computed Gaussian value based on the input parameters.
"""
function _cie_gauss(λ, μ, σ1, σ2)
    σ = λ < μ ? σ1 : σ2
    exp(-0.5 * ((λ - μ) / σ)^2)
end

"""
    blackbody_rgb(T)

Computes the RGB color of a blackbody at temperature `T`. The function integrates the spectral radiance over the visible spectrum using the CIE color matching functions and converts the result to linear sRGB. The output is a normalized RGB vector representing the color of the blackbody.
"""
function blackbody_rgb(T)
    X, Y, Z = 0.0, 0.0, 0.0
    for λ in 380.0:5.0:780.0
        λm = λ * 1e-9
        B = 1.0 / (λm^5 * (exp(0.014388 / (λm * T)) - 1.0))
        x̄ = 1.056 * _cie_gauss(λ, 599.8, 37.9, 31.0) +
             0.362 * _cie_gauss(λ, 442.0, 16.0, 26.7) -
             0.065 * _cie_gauss(λ, 501.1, 20.4, 26.2)
        ȳ = 0.821 * _cie_gauss(λ, 568.8, 46.9, 40.5) +
             0.286 * _cie_gauss(λ, 530.9, 16.3, 31.1)
        z̄ = 1.217 * _cie_gauss(λ, 437.0, 11.8, 36.0) +
             0.681 * _cie_gauss(λ, 459.0, 26.0, 13.8)
        X += B * x̄; Y += B * ȳ; Z += B * z̄
    end
    # XYZ → linear sRGB
    r = max( 3.2406X - 1.5372Y - 0.4986Z, 0.0)
    g = max(-0.9689X + 1.8758Y + 0.0415Z, 0.0)
    b = max( 0.0557X - 0.2040Y + 1.0570Z, 0.0)
    m = max(r, g, b, 1e-10)
    SVector(r/m, g/m, b/m)
end

"""
    Blackbody(; wb_temperature=6500.0, table_min=500.0, table_max=30000.0, table_size=1024)

Represents a blackbody emission profile with a specified temperature and precomputed color lookup table. The `wb_temperature` parameter sets the reference temperature for white balancing, while `table_min`, `table_max`, and `table_size` define the range and resolution of the precomputed color table for efficient color retrieval during rendering. The struct contains the reference temperature, table parameters, and the precomputed color table itself, which can be used to quickly obtain the RGB color corresponding to any given temperature within the specified range.
"""
struct Blackbody
    wb_temperature::Float64
    table_min::Float64
    table_max::Float64
    table_size::Int
    wb_gains::SVector{3, Float64}
    table::Vector{SVector{3, Float64}}
end

function Blackbody(; wb_temperature=6500.0, table_min=500.0, table_max=30000.0, table_size=1024)
    wb_ref = blackbody_rgb(wb_temperature)
    wb_gains = SVector(1.0 / wb_ref[1], 1.0 / wb_ref[2], 1.0 / wb_ref[3])
    table = Vector{SVector{3, Float64}}(undef, table_size)
    for (i, T) in enumerate(range(table_min, table_max, length=table_size))
        c = blackbody_rgb(max(T, 1.0))
        table[i] = SVector(c[1] * wb_gains[1], c[2] * wb_gains[2], c[3] * wb_gains[3])
    end
    Blackbody(wb_temperature, table_min, table_max, table_size, wb_gains, table)
end

"""
    wb_blackbody_color(T, bb::Blackbody)
Calculates the white-balanced RGB color for a given temperature `T` using the provided `Blackbody` struct. The function retrieves the precomputed color from the blackbody's color table based on the input temperature, applying the appropriate white balance gains to ensure accurate color representation. The output is an RGB vector representing the color corresponding to the specified temperature, adjusted for white balance according to the reference temperature defined in the `Blackbody` struct.
"""
function wb_blackbody_color(T, bb::Blackbody)
    c = blackbody_rgb(max(T, 1.0))
    SVector(c[1] * bb.wb_gains[1], c[2] * bb.wb_gains[2], c[3] * bb.wb_gains[3])
end

"""
    wb_blackbody_color_fast(T, bb::Blackbody)
A fast version of the `wb_blackbody_color` function that retrieves the white-balanced RGB color for a given temperature `T` using the precomputed color table in the `Blackbody` struct. The function clamps the input temperature to the range defined by the blackbody's table parameters, calculates the corresponding index in the color table, and returns the precomputed color without performing the full blackbody calculation. This approach allows for efficient color retrieval while still providing accurate results based on the precomputed values in the blackbody's color table.
"""
function wb_blackbody_color_fast(T, bb::Blackbody)
    T_clamped = clamp(T, bb.table_min, bb.table_max)
    frac = (T_clamped - bb.table_min) / (bb.table_max - bb.table_min)
    idx = clamp(round(Int, frac * (bb.table_size - 1)) + 1, 1, bb.table_size)
    bb.table[idx]
end

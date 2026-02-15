function _cie_gauss(λ, μ, σ1, σ2)
    σ = λ < μ ? σ1 : σ2
    exp(-0.5 * ((λ - μ) / σ)^2)
end

# Blackbody temperature → normalized linear sRGB chromaticity
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

# White balance gains: 6500K blackbody → (1, 1, 1)
const _WB_REF = blackbody_rgb(6500.0)
const WB_GAINS = SVector(1.0 / _WB_REF[1], 1.0 / _WB_REF[2], 1.0 / _WB_REF[3])

function wb_blackbody_color(T)
    c = blackbody_rgb(max(T, 1.0))
    SVector(c[1] * WB_GAINS[1], c[2] * WB_GAINS[2], c[3] * WB_GAINS[3])
end



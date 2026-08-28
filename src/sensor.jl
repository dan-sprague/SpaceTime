using Random: AbstractRNG, default_rng

# Simple Poisson sampler to avoid a heavy dependency. Knuth's method for λ < 30,
# normal approximation with continuity correction for larger λ.
function _poisson_sample(rng::AbstractRNG, λ::Float64)
    λ <= 0.0 && return 0.0
    if λ < 30.0
        L = exp(-λ)
        k = 0
        p = 1.0
        while true
            k += 1
            p *= rand(rng)
            p <= L && return Float64(k - 1)
        end
    else
        # Normal approximation with continuity correction.
        return max(round(randn(rng) * sqrt(λ) + λ), 0.0)
    end
end

# -----------------------------------------------------------------------------
# Sensor / exposure simulation
# -----------------------------------------------------------------------------


"""
    SensorSettings(iso, t_exp, read_noise_e, saturation)

Container for camera sensor parameters.
- `iso`: sensitivity gain (linear; ISO 100 is the reference).
- `t_exp`: exposure time in seconds.
- `read_noise_e`: read noise in electrons RMS.
- `saturation`: maximum linear sensor value before hard clipping.
"""
struct SensorSettings
    iso::Float64
    t_exp::Float64
    read_noise_e::Float64
    saturation::Float64
end

SensorSettings(; iso=100.0, t_exp=1.0, read_noise_e=2.0, saturation=1.0e6) =
    SensorSettings(iso, t_exp, read_noise_e, saturation)

"""
    apply_iso_gain!(image, settings::SensorSettings)

Apply the linear ISO gain and exposure time to `image` in-place. The reference
is ISO 100 with 1 second exposure.
"""
function apply_iso_gain!(image, settings::SensorSettings)
    gain = settings.iso / 100.0 * settings.t_exp
    image .*= gain
    return image
end

# Full-well capacity in electrons for a signal of 1.0 at ISO 100. This sets
# the absolute photon-count scale of the shot-noise model: SNR at mid-gray and
# base ISO is √(0.5·10000) ≈ 70, i.e. clean but not noiseless, and grain grows
# photographically as ISO rises.
const SENSOR_FULL_WELL_E = 10_000.0

"""
    add_sensor_noise!(image, settings::SensorSettings; rng=default_rng())

Add photon shot noise (Poisson) and Gaussian read noise to `image` in-place.
Signal values are converted to electron counts via `SENSOR_FULL_WELL_E`
(scaled down as ISO rises, since higher ISO means fewer photons for the same
output level), noise is applied in electrons, and the result is converted
back. For colour images, noise is added independently to each channel.
"""
function add_sensor_noise!(image::Matrix{RGBf}, settings::SensorSettings;
                           rng::AbstractRNG=default_rng())
    e_per_unit = SENSOR_FULL_WELL_E * 100.0 / settings.iso
    inv_e_per_unit = 1.0 / e_per_unit
    read_sigma = settings.read_noise_e

    for idx in eachindex(image)
        c = image[idx]
        r_e = _poisson_sample(rng, max(c.r * e_per_unit, 0.0)) + read_sigma * randn(rng)
        g_e = _poisson_sample(rng, max(c.g * e_per_unit, 0.0)) + read_sigma * randn(rng)
        b_e = _poisson_sample(rng, max(c.b * e_per_unit, 0.0)) + read_sigma * randn(rng)

        image[idx] = RGBf(
            max(r_e * inv_e_per_unit, 0.0),
            max(g_e * inv_e_per_unit, 0.0),
            max(b_e * inv_e_per_unit, 0.0)
        )
    end
    return image
end

function add_sensor_noise!(image::Matrix{Float64}, settings::SensorSettings;
                           rng::AbstractRNG=default_rng())
    e_per_unit = SENSOR_FULL_WELL_E * 100.0 / settings.iso
    for idx in eachindex(image)
        photons = max(image[idx] * e_per_unit, 0.0)
        electrons = _poisson_sample(rng, photons) + settings.read_noise_e * randn(rng)
        image[idx] = max(electrons / e_per_unit, 0.0)
    end
    return image
end

"""
    clip!(image, max_value)

Hard-clip `image` values to `[0, max_value]` in-place.
"""
function clip!(image, max_value::Real)
    for idx in eachindex(image)
        image[idx] = clamp(image[idx], 0.0, max_value)
    end
    return image
end

function clip!(image::Matrix{RGBf}, max_value::Real)
    for idx in eachindex(image)
        c = image[idx]
        image[idx] = RGBf(clamp(c.r, 0.0, max_value),
                          clamp(c.g, 0.0, max_value),
                          clamp(c.b, 0.0, max_value))
    end
    return image
end

"""
    sensor_expose!(image; iso=100.0, t_exp=1.0, read_noise_e=2.0,
                   saturation=1.0e6, add_noise=true, rng=default_rng())

Convenience pipeline: apply ISO gain/exposure time, optionally add sensor
noise, then clip to sensor saturation.
"""
function sensor_expose!(image;
                          iso::Real=100.0,
                          t_exp::Real=1.0,
                          read_noise_e::Real=2.0,
                          saturation::Real=1.0e6,
                          add_noise::Bool=true,
                          rng::AbstractRNG=default_rng())
    settings = SensorSettings(iso, t_exp, read_noise_e, saturation)
    apply_iso_gain!(image, settings)
    add_noise && add_sensor_noise!(image, settings; rng=rng)
    clip!(image, settings.saturation)
    return image
end

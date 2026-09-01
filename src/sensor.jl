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
const SENSOR_FULL_WELL_E = 60_000.0   # modern full-frame sensor; sets shot-noise scale

"""
    _grain_gain_field(rng, image, e_per_unit, read_sigma, gw, gh)

Luminance-correlated noise gain sampled on a `gw × gh` lattice, from the
box-averaged luminance of `image` over each cell.

Grain *amplitude* deliberately does not fall with cell area. A larger cell here
does not model a larger photosite collecting more photons — it models the same
capture delivered at a lower resolution, where the film's grain is a property
of the stock rather than of the delivery. Size and amplitude stay independent
controls, which is how procedural grain behaves in a grading suite.
"""
function _grain_gain_field(rng::AbstractRNG, image::Matrix{RGBf},
                           e_per_unit::Float64, read_sigma::Float64,
                           gw::Int, gh::Int)
    w, h = size(image)
    acc = zeros(Float64, gw, gh)
    cnt = zeros(Int, gw, gh)
    for j in 1:h, i in 1:w
        ci = clamp(cld(i * gw, w), 1, gw)
        cj = clamp(cld(j * gh, h), 1, gh)
        c = image[i, j]
        acc[ci, cj] += 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
        cnt[ci, cj] += 1
    end
    gain = Matrix{Float64}(undef, gw, gh)
    for idx in eachindex(acc)
        Y = acc[idx] / max(cnt[idx], 1)
        e_mean = max(Y * e_per_unit, 0.0)
        e_obs = _poisson_sample(rng, e_mean) + read_sigma * randn(rng)
        gain[idx] = max(e_obs, 0.0) / max(e_mean, 1.0)
    end
    return gain
end

# Bilinear reconstruction averages four independent lattice deviates with
# weights that sum to 1, so it damps their variance by mean(Σw²) = 4/9 over the
# cell. Scaling the deviation by 1/√(4/9) restores the amplitude the caller
# asked for, so grain of a given size has the same strength at every
# resolution. Exact in the limit of a lattice much coarser than a pixel;
# slightly over-corrected as `grain_px` approaches 1, where there is little
# interpolation to compensate for in the first place.
const _GRAIN_RECON_GAIN = 1.5

# Bilinear sample of a coarse field at full-resolution pixel (i, j).
@inline function _bilinear(field::Matrix{Float64}, i::Int, j::Int,
                           w::Int, h::Int)
    gw, gh = size(field)
    u = (i - 0.5) * gw / w + 0.5
    v = (j - 0.5) * gh / h + 0.5
    i0 = clamp(floor(Int, u), 1, gw); i1 = clamp(i0 + 1, 1, gw)
    j0 = clamp(floor(Int, v), 1, gh); j1 = clamp(j0 + 1, 1, gh)
    fu = clamp(u - i0, 0.0, 1.0)
    fv = clamp(v - j0, 0.0, 1.0)
    return (1 - fu) * (1 - fv) * field[i0, j0] + fu * (1 - fv) * field[i1, j0] +
           (1 - fu) * fv * field[i0, j1] + fu * fv * field[i1, j1]
end

"""
    add_sensor_noise!(image, settings::SensorSettings; grain_px=0.0,
                      rng=default_rng())

Add photon shot noise (Poisson) and Gaussian read noise to `image` in-place.
Signal values are converted to electron counts via `SENSOR_FULL_WELL_E`
(scaled down as ISO rises, since higher ISO means fewer photons for the same
output level), noise is applied in electrons, and the result is converted back.

`grain_px` is the grain correlation length **in pixels**. At or below 1 the
noise is drawn independently per pixel — which means the grain covers a
sixth as much of a 2160-line frame as of a 360-line one, and effectively
vanishes at 4K. Above 1, deviates are drawn on a coarser lattice and bilinearly
interpolated, so a given grain size occupies the same fraction of the frame at
any resolution. Callers using [`Look`](@ref) get this from `grain_size`.
"""
function add_sensor_noise!(image::Matrix{RGBf}, settings::SensorSettings;
                           grain_px::Real=0.0,
                           rng::AbstractRNG=default_rng())
    e_per_unit = SENSOR_FULL_WELL_E * 100.0 / settings.iso
    read_sigma = settings.read_noise_e
    w, h = size(image)

    if grain_px > 1.0
        gw = max(1, round(Int, w / grain_px))
        gh = max(1, round(Int, h / grain_px))
        field = _grain_gain_field(rng, image, e_per_unit, read_sigma, gw, gh)
        for j in 1:h, i in 1:w
            g = max(1.0 + (_bilinear(field, i, j, w, h) - 1.0) *
                          _GRAIN_RECON_GAIN, 0.0)
            c = image[i, j]
            image[i, j] = RGBf(max(c.r * g, 0.0), max(c.g * g, 0.0),
                               max(c.b * g, 0.0))
        end
        return image
    end

    # Luminance-correlated grain: one Poisson deviate per pixel, applied as a
    # common gain to all three channels. Independent per-channel deviates
    # produce red/green chroma confetti that reads as pixelation — real
    # post-demosaic sensor noise is luma-dominant.
    for idx in eachindex(image)
        c = image[idx]
        Y = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
        e_mean = max(Y * e_per_unit, 0.0)
        e_obs = _poisson_sample(rng, e_mean) + read_sigma * randn(rng)
        gain = max(e_obs, 0.0) / max(e_mean, 1.0)
        image[idx] = RGBf(max(c.r * gain, 0.0), max(c.g * gain, 0.0),
                          max(c.b * gain, 0.0))
    end
    return image
end

function add_sensor_noise!(image::Matrix{Float64}, settings::SensorSettings;
                           grain_px::Real=0.0,
                           rng::AbstractRNG=default_rng())
    e_per_unit = SENSOR_FULL_WELL_E * 100.0 / settings.iso
    w, h = size(image)

    if grain_px > 1.0
        # Same lattice-and-interpolate model as the colour method: sample the
        # noise gain coarsely, then apply it per pixel.
        gw = max(1, round(Int, w / grain_px))
        gh = max(1, round(Int, h / grain_px))
        acc = zeros(Float64, gw, gh)
        cnt = zeros(Int, gw, gh)
        for j in 1:h, i in 1:w
            ci = clamp(cld(i * gw, w), 1, gw)
            cj = clamp(cld(j * gh, h), 1, gh)
            acc[ci, cj] += image[i, j]
            cnt[ci, cj] += 1
        end
        field = Matrix{Float64}(undef, gw, gh)
        for idx in eachindex(acc)
            e_mean = max(acc[idx] / max(cnt[idx], 1) * e_per_unit, 0.0)
            e_obs = _poisson_sample(rng, e_mean) +
                    settings.read_noise_e * randn(rng)
            field[idx] = max(e_obs, 0.0) / max(e_mean, 1.0)
        end
        for j in 1:h, i in 1:w
            g = max(1.0 + (_bilinear(field, i, j, w, h) - 1.0) *
                          _GRAIN_RECON_GAIN, 0.0)
            image[i, j] = max(image[i, j] * g, 0.0)
        end
        return image
    end

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
                   saturation=1.0e6, grain_size=0.0, add_noise=true,
                   rng=default_rng())

Convenience pipeline: apply ISO gain/exposure time, optionally add sensor
noise, then clip to sensor saturation.

`grain_size` is the grain correlation length as a **fraction of frame height**
(0 = one deviate per pixel, whose apparent grain shrinks as resolution rises).
It is converted to pixels here and handed to [`add_sensor_noise!`](@ref).
"""
function sensor_expose!(image;
                          iso::Real=100.0,
                          t_exp::Real=1.0,
                          read_noise_e::Real=2.0,
                          saturation::Real=1.0e6,
                          grain_size::Real=0.0,
                          add_noise::Bool=true,
                          rng::AbstractRNG=default_rng())
    settings = SensorSettings(iso, t_exp, read_noise_e, saturation)
    apply_iso_gain!(image, settings)
    add_noise && add_sensor_noise!(image, settings;
                                   grain_px=grain_size * size(image, 2),
                                   rng=rng)
    clip!(image, settings.saturation)
    return image
end

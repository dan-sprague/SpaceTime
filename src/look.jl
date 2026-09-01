# -----------------------------------------------------------------------------
# Look and Sampling: the two things that decide how a render turns out, split
# so that neither one depends on which device rendered it.
#
# The rule this file exists to enforce: **rendering the same shot at two
# resolutions must differ only in sharpness.** A parameter measured in raw
# pixels breaks that rule — 10 px of bloom is 2.8% of a 360-line frame and
# 0.46% of a 2160-line one, so the same number is a different look at every
# output size. Blender hit exactly this in its Glare node and fixed it the same
# way we do here: sizes became "linear, relative to the image size" in 4.4,
# rather than gaining a reference-resolution parameter to correct them by.
#
# So every length in `Look` is one of:
#   * a **fraction of frame height** (bloom radius, streak width, grain, dust) —
#     resolution independent by construction, no reference height anywhere;
#   * a **physical length on the sensor** in mm (aperture diffraction), which
#     becomes pixels via the pixel pitch and so scales on its own.
#
# There is deliberately no third category and no escape hatch.
# -----------------------------------------------------------------------------

"""
    Look

Everything that decides how a render *looks*: grade, glare, lens, sensor, and
practical effects. One `Look` is shared by every script and every output
resolution, so a still and a video frame of the same shot cannot silently
disagree.

All lengths are resolution independent. Fields documented "fraction of frame
height" are exactly that — `bloom_radius = 0.028` is 2.8% of the frame's height
whether that frame is 360 lines or 2160. `f_number` and `sensor_width_mm` are
physical and become pixels through the pixel pitch.

Construct a variant with [`with_look`](@ref):

    hero = with_look(LOOK_FILM; bloom_radius = 10 / 2160)

# Grade
- `gain`, `exposure`, `gamma`, `contrast`: exposure multiplier is
  `gain * 2^exposure`; `gamma` is applied as `x^(1/gamma)`.
- `tonemap`: `:aces`, `:reinhard`, or `:none`.
- `tonemap_hue_preserve`: 0 gives the per-channel filmic look, 1 tonemaps
  luminance only and keeps hue.

# Glare
- `bloom_strength`, `threshold`, `bloom_power`, `streak_strength`,
  `streak_length`, `n_spikes`: unitless, unchanged from `postprocess`.
- `bloom_radius`, `streak_width`: **fraction of frame height**.

# Optics
- `f_number`: aperture for the Airy diffraction kernel; `0` disables it. This
  is the *only* place a physical length enters, via `sensor_width_mm`.
- `vignette`, `distortion_k1`: applied before the sensor, because that is where
  they happen — see [`apply_look!`](@ref).

# Sensor
- `iso`, `t_exp`, `read_noise_e`, `saturation`: as [`SensorSettings`](@ref).
- `grain_size`: grain correlation length as a **fraction of frame height**.
  `1/360` reproduces the coarse, filmic grain of a 360-line render at every
  resolution; `0` falls back to one independent deviate per pixel, whose
  apparent grain shrinks as resolution rises until it is invisible.

# Practicals
- `dust`, `streaks`: optional [`LensDust`](@ref) / [`MicroStreaks`](@ref).
  Their own sizes are fractions of frame height too.
"""
Base.@kwdef struct Look
    # grade
    gain::Float64 = 1.0
    exposure::Float64 = 0.0
    gamma::Float64 = 2.2
    contrast::Float64 = 0.0
    tonemap::Symbol = :aces
    tonemap_hue_preserve::Float64 = 0.75
    # glare
    bloom_strength::Float64 = 0.6
    threshold::Float64 = 0.5
    bloom_radius::Float64 = 15.0 / 1080     # fraction of frame height
    bloom_power::Float64 = 1.5
    streak_strength::Float64 = 0.3
    streak_length::Float64 = 0.4            # already a fraction of max(w, h)
    streak_width::Float64 = 1.5 / 1080      # fraction of frame height
    n_spikes::Int = 4
    # optics
    f_number::Float64 = 0.0                 # 0 = no diffraction kernel
    sensor_width_mm::Float64 = SENSOR_WIDTH_MM
    vignette::Float64 = 0.0
    distortion_k1::Float64 = 0.0
    # sensor
    iso::Float64 = 100.0
    t_exp::Float64 = 1.0
    read_noise_e::Float64 = 2.0
    saturation::Float64 = 1.0e6
    grain_size::Float64 = 0.0               # fraction of frame height; 0 = per-pixel
    # practicals
    dust::Union{LensDust,Nothing} = nothing
    streaks::Union{MicroStreaks,Nothing} = nothing
end

"""
    with_look(look::Look; kwargs...)

Copy `look`, replacing the named fields. The way to express "the film look, but
with a tighter bloom" without restating twenty numbers.
"""
function with_look(look::Look; kwargs...)
    vals = map(f -> get(kwargs, f, getfield(look, f)), fieldnames(Look))
    return Look(vals...)
end

"""
    LOOK_FILM

The look the videos are graded to: hot exposure, wide bloom, four-point
streaks, coarse filmic grain. Its lengths are the values that were tuned by eye
on 360-line proxies, converted once to fractions of frame height — so the same
`Look` now renders that grade identically at 4K.
"""
const LOOK_FILM = Look(
    gain = 1.0, exposure = 0.8, gamma = 0.2, tonemap = :aces,
    tonemap_hue_preserve = 0.75,
    bloom_strength = 1.0, threshold = 0.5,
    bloom_radius = 10.0 / 360, bloom_power = 1.5,
    streak_strength = 2.0, streak_length = 0.1, streak_width = 1.0 / 360,
    n_spikes = 4,
    f_number = 5.6,
    vignette = 0.3, distortion_k1 = -0.02,
    iso = 400.0, t_exp = 1.0, read_noise_e = 2.0, saturation = 1.0e6,
    grain_size = 1.0 / 360,
)

"""
    LOOK_HERO

`LOOK_FILM` with the glare kernels six times tighter, reproducing the grade the
4K hero still was tuned to.

The 6× is not a resolution correction — both looks are resolution independent.
It is a real and unresolved *aesthetic* disagreement: the same `bloom_radius =
10.0` was tuned by eye against a 360-line proxy for the videos and against a
2160-line frame for the still, and nobody has since decided which halo width
the project actually wants. Keeping both as named constants makes the
disagreement visible in one file instead of hiding it in two scripts.
"""
const LOOK_HERO = with_look(LOOK_FILM;
    bloom_radius = 10.0 / 2160,
    streak_width = 1.0 / 2160,
    grain_size   = 1.0 / 2160,
)

# -----------------------------------------------------------------------------

"""
    Sampling

How many rays to spend and how to spread them. This is where a still and a
moving frame legitimately differ — and the *only* axis on which they should,
since neither the look nor the physics depends on how long you were willing to
wait.

- `samples`: rays per pixel per axis; `samples^2` in total.
- `shutter`: shutter angle as a fraction of the frame interval. `0.5` is the
  180° convention; `0` freezes motion.
- `seed`: base seed. Every render is reproducible from it.

Presets [`STILL`](@ref) and [`MOTION`](@ref) are the intended entry points.
"""
Base.@kwdef struct Sampling
    samples::Int = 4
    shutter::Float64 = 0.5
    seed::Int = 1
end

"""
    STILL

Sampling for a final still: many rays, no shutter. Nothing about it is
CPU-specific — it is the preset you would also hand the GPU to preview the same
frame quickly.
"""
const STILL = Sampling(samples = 8, shutter = 0.0, seed = 1)

"""
    MOTION

Sampling for a frame of a moving sequence: fewer rays, 180° shutter. Nothing
about it is GPU-specific.
"""
const MOTION = Sampling(samples = 4, shutter = 0.5, seed = 1)

"""
    with_sampling(s::Sampling; kwargs...)

Copy `s`, replacing the named fields.
"""
function with_sampling(s::Sampling; kwargs...)
    vals = map(f -> get(kwargs, f, getfield(s, f)), fieldnames(Sampling))
    return Sampling(vals...)
end

"""
    sampling_rng(s::Sampling, frame::Integer=0)

The RNG for one render under `s`. `frame` offsets the stream so a sequence gets
independent noise per frame while staying exactly reproducible — the same
`Sampling` and frame number always give the same image.
"""
sampling_rng(s::Sampling, frame::Integer=0) = Random.Xoshiro(s.seed + frame)

"""
    shutter_span(s::Sampling, frame_interval::Real)

Length of the open shutter in the same units as `frame_interval`. A shutter of
`0.5` on a 1/24 s frame is the 180-degree convention, 1/48 s. Callers use this
to build the `camera_at` they hand to a renderer; the shutter lives here rather
than inside the renderer because only the caller knows what the camera is doing
between frames.
"""
shutter_span(s::Sampling, frame_interval::Real) = s.shutter * frame_interval

"""
    render_draft_mtl(ctx, cam, spacetime, sampling::Sampling; frame=0, kwargs...)
    render_motion(camera_at, t0, t1, spacetime, background, sampling::Sampling;
                  frame=0, kwargs...)

Render under a [`Sampling`](@ref) preset. Both renderers take the same
`Sampling`, because effort and shutter are not properties of a device: `STILL`
on the GPU is a fast preview of a final frame, and `MOTION` on the CPU is the
reference a GPU frame can be checked against.

`sampling.shutter == 0` freezes motion — the GPU form ignores `camera_at`, and
the CPU form collapses to a single shutter sample.
"""
function render_draft_mtl(ctx, cam, spacetime, sampling::Sampling;
                          frame::Integer=0,
                          camera_at::Union{Function,Nothing}=nothing, kwargs...)
    return render_draft_mtl(ctx, cam, spacetime;
                            samples = sampling.samples,
                            rng = sampling_rng(sampling, frame),
                            camera_at = sampling.shutter > 0 ? camera_at : nothing,
                            kwargs...)
end

function render_motion(camera_at::Function, t0::Real, t1::Real,
                       spacetime::Schwarzschild, background, sampling::Sampling;
                       frame::Integer=0, kwargs...)
    # The CPU renderer multiplies its sample dimensions (samples² subpixel rays
    # at each of `time_samples` shutter steps) where the GPU folds them into one
    # set of passes. Matching the *total* rays per pixel keeps `Sampling` a
    # statement about effort rather than about which renderer is running.
    n = sampling.samples^2
    ts = sampling.shutter > 0 ? max(1, round(Int, sqrt(n))) : 1
    return render_motion(camera_at, t0, t1, spacetime, background;
                         samples = max(1, round(Int, sqrt(n / ts))),
                         time_samples = ts,
                         rng = sampling_rng(sampling, frame),
                         kwargs...)
end

# -----------------------------------------------------------------------------

"""
    postprocess(image, look::Look)

Grade and glare from a [`Look`](@ref). Converts the frame-relative kernel sizes
to pixels for this image's height and calls the keyword form.
"""
function postprocess(image::Matrix{RGBf}, look::Look)
    h = size(image, 2)
    return postprocess(image;
        gain = look.gain, exposure = look.exposure, gamma = look.gamma,
        contrast = look.contrast,
        bloom_strength = look.bloom_strength, threshold = look.threshold,
        bloom_radius = look.bloom_radius * h, bloom_power = look.bloom_power,
        streak_strength = look.streak_strength,
        streak_length = look.streak_length,
        streak_width = look.streak_width * h,
        n_spikes = look.n_spikes, tonemap = look.tonemap,
        tonemap_hue_preserve = look.tonemap_hue_preserve)
end

"""
    apply_look!(image, look::Look; rng, dust_rng=nothing, streak_rng=nothing)

Run the full image chain for `look` on a linear HDR render, in physical order:

1. **aperture diffraction** — in the lens, on linear light;
2. **grade and glare** — [`postprocess`](@ref);
3. **lens dust** — on the front element;
4. **micro-streaks** — optional, only when `streak_rng` is given, so a caller
   can fire them on chosen frames;
5. **vignette and distortion** — still in the lens;
6. **sensor exposure and grain** — last, because the sensor is last.

Steps 5 and 6 are in that order for a reason: grain is generated *by the
sensor*, so it must not be vignetted or distorted along with the image. The
scripts previously ran the sensor first and then distorted its grain.

The three streams are separate because they change on different clocks. Grain
is redrawn every frame, so `rng` should vary per frame; dust sits on the glass
and must not move, so `dust_rng` should be a fixed seed across a sequence. It
defaults to `rng`, which is right for a single still and wrong for a sequence —
pass it explicitly when rendering one.

Returns `image`, modified in place.
"""
function apply_look!(image::Matrix{RGBf}, look::Look;
                     rng::Random.AbstractRNG=Random.default_rng(),
                     dust_rng::Union{Random.AbstractRNG,Nothing}=nothing,
                     streak_rng::Union{Random.AbstractRNG,Nothing}=nothing)
    look.f_number > 0 && apply_diffraction!(image; f_number=look.f_number,
                                            sensor_width_mm=look.sensor_width_mm)
    out = postprocess(image, look)
    look.dust === nothing ||
        apply_lens_dust!(out; lens_dust=look.dust,
                         rng=something(dust_rng, rng))
    (look.streaks === nothing || streak_rng === nothing) ||
        apply_micro_streaks!(out; streaks=look.streaks, rng=streak_rng)
    look.vignette > 0 && apply_vignette!(out; strength=look.vignette)
    look.distortion_k1 == 0 || apply_lens_distortion!(out; k1=look.distortion_k1)
    sensor_expose!(out; iso=look.iso, t_exp=look.t_exp,
                   read_noise_e=look.read_noise_e, saturation=look.saturation,
                   grain_size=look.grain_size, rng=rng)
    return out
end

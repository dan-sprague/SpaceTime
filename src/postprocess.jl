# ---------------------------------------------------------------------------
# FFT-based bloom + star streak post-processing
# ---------------------------------------------------------------------------

# Chromatic scaling for bloom kernel per channel (R, G, B).
# Approximates diffraction: blue diffracts tighter, red wider.
const AIRY_SPECTRUM = SVector(1.0, 0.85, 0.7)

# Per-channel wavelengths in mm, matching the 610/550/465 nm convention the
# renderers use for relativistic sky tint and disc colour.
const CHANNEL_LAMBDA_MM = SVector(610.0e-6, 550.0e-6, 465.0e-6)

# 35 mm full-frame sensor width, the reference for `Lens`'s focal lengths.
const SENSOR_WIDTH_MM = 36.0

"""
    airy_kernel(f_number, pixel_pitch_mm, λ_mm; radius_px=nothing)

Airy point-spread function for a circular aperture, sampled on a pixel grid:
`I(r) = (2 J₁(v) / v)²` with `v = π r / (λ N)`, whose first zero sits at
`r = 1.22 λ N` (so the Airy *diameter* is the familiar `2.44 λ N`). Returns a
normalised, centred, odd-sized matrix.

`radius_px` defaults to three times the first-zero radius, which captures the
core and the first two rings.
"""
function airy_kernel(f_number::Real, pixel_pitch_mm::Real, λ_mm::Real;
                     radius_px::Union{Nothing,Int}=nothing)
    r_zero = 1.22 * λ_mm * f_number / pixel_pitch_mm      # first zero, pixels
    R = radius_px === nothing ? max(ceil(Int, 3 * r_zero), 1) : radius_px
    k = zeros(Float64, 2R + 1, 2R + 1)
    scale = π * pixel_pitch_mm / (λ_mm * f_number)        # v per pixel of r
    for j in -R:R, i in -R:R
        r = sqrt(Float64(i)^2 + Float64(j)^2)
        v = scale * r
        # (2J₁(v)/v)² → 1 as v → 0; besselj1(v)/v is removable there.
        k[i+R+1, j+R+1] = v < 1.0e-8 ? 1.0 : (2 * besselj1(v) / v)^2
    end
    s = sum(k)
    s > 0 && (k ./= s)
    return k
end

"""
    apply_diffraction!(image; f_number, sensor_width_mm=36.0)

Convolve `image` with the lens's Airy PSF, in place. This is the physical
resolution limit of the aperture: at f/11 on full-frame, the Airy diameter is
`2.44 λ N ≈ 14.8 µm`, which at 3840 px across a 36 mm sensor is about 1.6
pixels — so a real f/11 lens *cannot* render single-pixel detail, and an image
that contains it is optically impossible.

Apply this to the **linear** image, before `postprocess`: it is a property of
the glass, upstream of bloom and tonemapping. Each channel is blurred at its
own wavelength (610/550/465 nm), so the softening is chromatic like real
diffraction. A no-op when the PSF is narrower than a third of a pixel.
"""
function apply_diffraction!(image::Matrix{RGBf}; f_number::Real,
                            sensor_width_mm::Real=SENSOR_WIDTH_MM)
    w, _ = size(image)
    pitch = sensor_width_mm / w
    # Widest channel sets whether the PSF is resolvable at all.
    if 1.22 * CHANNEL_LAMBDA_MM[1] * f_number / pitch < 0.33
        return image
    end
    chans = (Float64.(getfield.(image, :r)), Float64.(getfield.(image, :g)),
             Float64.(getfield.(image, :b)))
    out = map(enumerate(chans)) do (c, ch)
        imfilter(ch, centered(airy_kernel(f_number, pitch, CHANNEL_LAMBDA_MM[c])),
                 "replicate")
    end
    @inbounds for idx in eachindex(image)
        image[idx] = RGBf(out[1][idx], out[2][idx], out[3][idx])
    end
    return image
end

"""
    fft_convolve(image_ch::Matrix{Float64}, kernel::Matrix{Float64})

Convolve a single-channel image with a kernel using FFT. The kernel must be
the same size as the image (already padded / centered).
"""
function fft_convolve(image_ch::Matrix{Float64}, kernel::Matrix{Float64})
    real.(ifft(fft(image_ch) .* fft(kernel)))
end

"""
    generate_bloom_kernel(w, h; bloom_radius=15.0, power=1.5)

Generate a w×h bloom kernel with a Moffat-like radial profile:
`1 / (1 + (r / bloom_radius)^2)^power`. Returns a 3-channel array
(w × h × 3) with chromatic scaling from `AIRY_SPECTRUM`.
"""
function generate_bloom_kernel(w, h; bloom_radius=15.0, power=1.5)
    kernel = zeros(w, h, 3)
    cx, cy = w ÷ 2 + 1, h ÷ 2 + 1
    for j in 1:h, i in 1:w
        r = sqrt((i - cx)^2 + (j - cy)^2)
        for c in 1:3
            # chromatic: scale radius per channel so blue blooms tighter
            rc = r / AIRY_SPECTRUM[c]
            kernel[i, j, c] = 1.0 / (1.0 + (rc / bloom_radius)^2)^power
        end
    end
    for c in 1:3
        s = sum(@view kernel[:, :, c])
        if s > 0
            kernel[:, :, c] ./= s
        end
    end
    kernel
end

"""
    generate_streak_kernel(w, h; n_spikes=4, streak_length=0.4, streak_width=1.5, angles=nothing)

Generate a w×h star-streak kernel. Each spike is a thin line through the
center with exponential fall-off along its length and Gaussian cross-section.
Default angles are n_spikes evenly spaced starting at 45°.
"""
function generate_streak_kernel(w, h; n_spikes=4, streak_length=0.4, streak_width=1.5, angles=nothing)
    if angles === nothing
        angles = [π/4 + k * π / n_spikes for k in 0:n_spikes-1]
    end
    kernel = zeros(w, h)
    cx, cy = w ÷ 2 + 1, h ÷ 2 + 1
    L = streak_length * max(w, h)

    for j in 1:h, i in 1:w
        dx = Float64(i - cx)
        dy = Float64(j - cy)
        for θ in angles
            cosθ, sinθ = cos(θ), sin(θ)
            d_along = dx * cosθ + dy * sinθ
            d_perp  = -dx * sinθ + dy * cosθ
            kernel[i, j] += exp(-abs(d_along) / L) * exp(-d_perp^2 / (2.0 * streak_width^2))
        end
    end
    s = sum(kernel)
    if s > 0
        kernel ./= s
    end
    kernel
end

"""
    generate_psf(w, h; bloom_radius=15.0, bloom_power=1.5, bloom_weight=0.7,
                 streak_length=0.4, streak_width=1.5, streak_weight=0.3,
                 n_spikes=4, angles=nothing)

Composite PSF = weighted bloom + weighted streaks. Returns a (w × h × 3)
array (chromatic bloom channels) with streaks added uniformly.
"""
function generate_psf(w, h; bloom_radius=15.0, bloom_power=1.5, bloom_weight=0.7,
                      streak_length=0.4, streak_width=1.5, streak_weight=0.3,
                      n_spikes=4, angles=nothing)
    bloom = generate_bloom_kernel(w, h; bloom_radius, power=bloom_power)
    streak = generate_streak_kernel(w, h; n_spikes, streak_length, streak_width, angles)

    psf = zeros(w, h, 3)
    for c in 1:3
        psf[:, :, c] .= bloom_weight .* bloom[:, :, c] .+ streak_weight .* streak
    end
    # Normalize each channel
    for c in 1:3
        s = sum(@view psf[:, :, c])
        if s > 0
            psf[:, :, c] ./= s
        end
    end
    psf
end

# -----------------------------------------------------------------------------
# Veiling glare and ghosts: the two glare terms that live at low resolution.
# -----------------------------------------------------------------------------

"""Box-average `ch` by an integer factor; edge remainders are dropped."""
function _downsample_box(ch::Matrix{Float64}, f::Int)
    f == 1 && return copy(ch)
    w, h = size(ch)
    ws, hs = w ÷ f, h ÷ f
    out = zeros(Float64, ws, hs)
    inv = 1.0 / (f * f)
    @inbounds for j in 1:hs, i in 1:ws
        s = 0.0
        for dj in 0:f-1, di in 0:f-1
            s += ch[(i - 1) * f + 1 + di, (j - 1) * f + 1 + dj]
        end
        out[i, j] = s * inv
    end
    out
end

"""Bilinear sample of `ch` at fractional (x, y); zero outside the image."""
@inline function _bilinear(ch::Matrix{Float64}, x::Float64, y::Float64)
    w, h = size(ch)
    (x < 1.0 || y < 1.0 || x > w || y > h) && return 0.0
    x0 = floor(Int, x); y0 = floor(Int, y)
    x1 = min(x0 + 1, w); y1 = min(y0 + 1, h)
    fx = x - x0; fy = y - y0
    @inbounds return ch[x0, y0] * (1 - fx) * (1 - fy) + ch[x1, y0] * fx * (1 - fy) +
                     ch[x0, y1] * (1 - fx) * fy + ch[x1, y1] * fx * fy
end

"""Bilinear upsample of a `_downsample_box` result back to `(w, h)`."""
function _upsample_bilinear(small::Matrix{Float64}, w::Int, h::Int, f::Int)
    f == 1 && return small
    ws, hs = size(small)
    out = zeros(Float64, w, h)
    @inbounds for j in 1:h, i in 1:w
        xs = clamp((i - 0.5) / f + 0.5, 1.0, Float64(ws))
        ys = clamp((j - 0.5) / f + 0.5, 1.0, Float64(hs))
        out[i, j] = _bilinear(small, xs, ys)
    end
    out
end

"""
    veil_kernel(w, h; radius, power=1.0)

Glare spread function for veiling glare: a wide, normalised power-law halo
`(1 + (r / radius)²)^-power`. Power 1 is the 1/r² tail of scatter inside the
lens body (the CIE/Stiles–Holladay glare law): equal energy in every octave of
radius, so the veil lifts the blacks across the whole frame rather than just
around the source. Centred, `w × h`.
"""
function veil_kernel(w, h; radius::Real, power::Real=1.0)
    k = zeros(Float64, w, h)
    cx, cy = w ÷ 2 + 1, h ÷ 2 + 1
    for j in 1:h, i in 1:w
        r2 = ((i - cx)^2 + (j - cy)^2) / radius^2
        k[i, j] = (1.0 + r2)^(-power)
    end
    k ./= sum(k)
    k
end

"""
    apply_veil!(chans; strength, radius_px)

Veiling glare on linear channels, in place. Following Talvala et al. (2007),
glare is a global linear transport: the recorded image is the direct image plus
the *whole scene* (not a bright pass) spread by a low-frequency glare spread
function. `strength` is the veil's energy as a fraction of the scene's, added
like bloom is. The ISO 9358 veiling-glare index of a real lens is a few
percent, but under a hard grade (`LOOK_FILM`'s x⁵ display gamma) a veil that
size is crushed to black; the values that read on screen are of order 1.

The veil is computed at reduced resolution, since it has no detail, and on a
zero-padded field so the plume's veil does not wrap round to the far edge.
"""
function apply_veil!(chans::NTuple{3,Matrix{Float64}}; strength::Real,
                     radius_px::Real, power::Real=1.0)
    w, h = size(chans[1])
    f = max(1, ceil(Int, h / 540))
    ws, hs = w ÷ f, h ÷ f
    kern = ifftshift(veil_kernel(2ws, 2hs; radius=radius_px / f, power=power))
    K = fft(kern)
    for ch in chans
        small = _downsample_box(ch, f)
        padded = zeros(Float64, 2ws, 2hs)
        padded[1:ws, 1:hs] .= small
        veil = real.(ifft(fft(padded) .* K))[1:ws, 1:hs]
        ch .+= strength .* _upsample_bilinear(veil, w, h, f)
    end
    chans
end

"""
    ngon_kernel(radius_px, n; rotation=0.0)

The image of an `n`-blade iris: a regular polygon of circumradius `radius_px`
with a one-pixel antialiased edge, normalised to unit sum. This is what a
defocused point — or a ghost, which is a defocused image of the aperture —
looks like through the lens.
"""
function ngon_kernel(radius_px::Real, n::Int; rotation::Real=0.0)
    R = max(ceil(Int, radius_px) + 1, 1)
    apothem = radius_px * cos(π / n)
    k = zeros(Float64, 2R + 1, 2R + 1)
    for j in -R:R, i in -R:R
        d = -Inf
        for m in 0:n-1
            θ = rotation + 2π * m / n
            d = max(d, i * cos(θ) + j * sin(θ) - apothem)
        end
        k[i + R + 1, j + R + 1] = clamp(0.5 - d, 0.0, 1.0)
    end
    k ./= sum(k)
    k
end

# The ghost train: each entry is one pair of reflecting surfaces. `m` is the
# magnification about frame centre (negative = flipped to the far side of
# centre, as most two-bounce ghosts are; |m| > 1 lands beyond the source),
# `size` the defocus of that ghost relative to `ghost_size`, `tint` the colour
# the anti-reflective coatings leave on it (coatings are tuned for green, so
# the residual reflection is cyan/magenta/amber), `energy` its relative
# brightness.
const GHOST_TRAIN = (
    (m = -0.55, size = 1.0, tint = (0.55, 0.85, 1.00), energy = 1.0),
    (m = -0.30, size = 0.6, tint = (1.00, 0.60, 0.85), energy = 0.7),
    (m = -0.15, size = 0.35, tint = (1.00, 0.85, 0.50), energy = 0.5),
    (m =  0.20, size = 0.45, tint = (0.60, 1.00, 0.70), energy = 0.5),
    (m =  0.45, size = 0.8, tint = (0.70, 0.80, 1.00), energy = 0.8),
    (m = -1.35, size = 1.8, tint = (0.50, 0.70, 1.00), energy = 1.2),
    (m =  1.60, size = 2.2, tint = (1.00, 0.65, 0.50), energy = 1.0),
)

# Per-channel magnification error of the ghosts: red is bent less than blue,
# so each ghost carries a red-outside / blue-inside fringe.
const GHOST_DISPERSION = (0.015, 0.0, -0.015)

"""
    apply_ghosts!(chans; strength, threshold, size_px, blades=7)

Lens-flare ghosts on linear channels, in place: the screen-space method of
Chapman (2017) / Froyok (2021), with the ghost shape from the physical model of
Hullin et al. (2011). Light above `threshold` is the source; each entry of
[`GHOST_TRAIN`](@ref) reflects it about frame centre with its own
magnification, blurs it with the [`ngon_kernel`](@ref) image of the iris,
tints it with its coating colour and fades it toward the frame edge. Energy is
conserved through the magnification (`1/m²`), so a small ghost is bright and a
large one is faint. Computed at reduced resolution, as the ghosts have no
detail finer than the iris image.
"""
function apply_ghosts!(chans::NTuple{3,Matrix{Float64}}; strength::Real,
                       threshold::Real, size_px::Real, blades::Int=7)
    w, h = size(chans[1])
    f = max(1, ceil(Int, h / 540))
    ws, hs = w ÷ f, h ÷ f
    cx, cy = (ws + 1) / 2.0, (hs + 1) / 2.0
    ρmax = hs / 2.0
    bright = ntuple(c -> _downsample_box(max.(chans[c] .- threshold, 0.0), f), 3)
    acc = ntuple(_ -> zeros(Float64, ws, hs), 3)
    for g in GHOST_TRAIN
        kern = centered(ngon_kernel(g.size * size_px / f, blades; rotation=0.3))
        for c in 1:3
            m = g.m * (1.0 + GHOST_DISPERSION[c])
            amp = strength * g.energy * g.tint[c] / (m * m)
            ghost = zeros(Float64, ws, hs)
            @inbounds for j in 1:hs, i in 1:ws
                xs = cx + (i - cx) / m
                ys = cy + (j - cy) / m
                ρ = sqrt((i - cx)^2 + (j - cy)^2) / ρmax
                t = clamp(ρ / 1.4, 0.0, 1.0)
                fade = 1.0 - t * t * (3.0 - 2.0 * t)
                ghost[i, j] = amp * fade * _bilinear(bright[c], xs, ys)
            end
            acc[c] .+= imfilter(ghost, kern, Fill(0.0))
        end
    end
    for c in 1:3
        chans[c] .+= _upsample_bilinear(acc[c], w, h, f)
    end
    chans
end

"""
    ifftshift(kernel::Matrix{Float64})

Shift a centered kernel so that its center is at index (1,1), which is the
layout expected by FFT convolution.
"""
function ifftshift(kernel::Matrix{Float64})
    w, h = size(kernel)
    circshift(kernel, (-(w ÷ 2), -(h ÷ 2)))
end

"""
    aces_tonemap(x)

ACES filmic tonemapping curve. Maps HDR values to [0, 1].
"""
function aces_tonemap(x)
    clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
end

"""
    postprocess(image; gain=1.0, exposure=0.0, gamma=2.2,
                bloom_strength=0.6, threshold=0.5,
                bloom_radius=15.0, bloom_power=1.5,
                streak_strength=0.3, streak_length=0.4, streak_width=1.5,
                n_spikes=4, angles=nothing, tonemap=:aces)

FFT-based HDR post-processing pipeline with bloom, star streaks, and tonemapping.

Pipeline (all operations in HDR linear space):
1. Gain + exposure adjustment (effective multiplier = gain * 2^exposure)
2. Bright-pass extraction (pixels above threshold)
3. FFT convolution of bright pass with composite PSF (bloom + streaks)
4. Combine with original HDR image
5. Tonemapping (ACES filmic or Reinhard)
6. Gamma correction (x^(1/gamma); 1.0 = linear, 2.2 = sRGB-like)
"""
function postprocess(image::Matrix{RGBf};
                     gain=1.0,
                     exposure=0.0,
                     gamma=2.2,
                     bloom_strength=0.6,
                     threshold=0.5,
                     bloom_radius=15.0,
                     bloom_power=1.5,
                     streak_strength=0.3,
                     streak_length=0.4,
                     streak_width=1.5,
                     n_spikes=4,
                     angles=nothing,
                     tonemap=:aces,
                     tonemap_hue_preserve=0.75,
                     contrast=0.0,
                     veil=0.0,
                     veil_radius=0.1 * size(image, 2),
                     ghosts=0.0,
                     ghost_size=0.02 * size(image, 2),
                     blades=7)
    w, h = size(image)

    # `bloom_radius` and `streak_width` are in raw pixels here, so the same
    # numbers give a 6x tighter bloom at 2160p than at 360p. This form is the
    # low-level one, used by the interactive apps where the preview is a fixed
    # size and pixels are what the sliders mean. Anything that renders at more
    # than one resolution should go through [`Look`](@ref) instead, whose
    # lengths are fractions of frame height and so cannot drift.

    # 1. Gain + exposure
    ev = gain * 2.0^exposure
    r_ch = Float64.(getfield.(image, :r)) .* ev
    g_ch = Float64.(getfield.(image, :g)) .* ev
    b_ch = Float64.(getfield.(image, :b)) .* ev

    # 2. Bright pass
    r_bright = max.(r_ch .- threshold, 0.0)
    g_bright = max.(g_ch .- threshold, 0.0)
    b_bright = max.(b_ch .- threshold, 0.0)

    # 3. Generate composite PSF and shift for FFT
    psf = generate_psf(w, h;
                       bloom_radius, bloom_power,
                       bloom_weight=bloom_strength,
                       streak_length, streak_width,
                       streak_weight=streak_strength,
                       n_spikes, angles)

    psf_r = ifftshift(psf[:, :, 1])
    psf_g = ifftshift(psf[:, :, 2])
    psf_b = ifftshift(psf[:, :, 3])

    # 4. FFT convolve each channel
    bloom_r = fft_convolve(r_bright, psf_r)
    bloom_g = fft_convolve(g_bright, psf_g)
    bloom_b = fft_convolve(b_bright, psf_b)

    # 5. Combine: original HDR + bloom result, then the low-frequency glare
    # terms — ghosts from the bright pass, veil from the whole scene.
    final_r = r_ch .+ bloom_r
    final_g = g_ch .+ bloom_g
    final_b = b_ch .+ bloom_b
    chans = (final_r, final_g, final_b)
    ghosts > 0 && apply_ghosts!(chans; strength=ghosts, threshold=threshold,
                                size_px=ghost_size, blades=blades)
    veil > 0 && apply_veil!(chans; strength=veil, radius_px=veil_radius)

    # 6. Tonemap. Per-channel filmic curves desaturate highlights (all three
    # channels converge to 1), so blend with a hue-preserving variant that
    # tonemaps luminance only and rescales the RGB triple by the ratio.
    # `tonemap_hue_preserve` = 0 gives the classic per-channel look, 1 keeps
    # hue/saturation fully at the cost of harsher-looking extreme highlights.
    if tonemap == :aces || tonemap == :reinhard
        tm = tonemap == :aces ? aces_tonemap : (x -> x / (1.0 + x))
        pc_r = tm.(final_r)
        pc_g = tm.(final_g)
        pc_b = tm.(final_b)
        k = clamp(tonemap_hue_preserve, 0.0, 1.0)
        if k > 0.0
            Y = 0.2126 .* final_r .+ 0.7152 .* final_g .+ 0.0722 .* final_b
            s = tm.(Y) ./ max.(Y, 1.0e-8)
            hp_r = clamp.(final_r .* s, 0.0, 1.0)
            hp_g = clamp.(final_g .* s, 0.0, 1.0)
            hp_b = clamp.(final_b .* s, 0.0, 1.0)
            final_r = (1.0 - k) .* pc_r .+ k .* hp_r
            final_g = (1.0 - k) .* pc_g .+ k .* hp_g
            final_b = (1.0 - k) .* pc_b .+ k .* hp_b
        else
            final_r = pc_r
            final_g = pc_g
            final_b = pc_b
        end
    end

    # 7. Gamma
    if gamma != 1.0
        inv_gamma = 1.0 / gamma
        final_r = final_r .^ inv_gamma
        final_g = final_g .^ inv_gamma
        final_b = final_b .^ inv_gamma
    end

    # 8. Contrast: blend toward (positive) or away from (negative) a
    # smoothstep S-curve in display space. Monotonic for contrast ∈ [-1, 1].
    if contrast != 0.0
        c = clamp(contrast, -1.0, 1.0)
        scurve(x) = (xc = clamp(x, 0.0, 1.0); xc + c * (xc * xc * (3.0 - 2.0 * xc) - xc))
        final_r = scurve.(final_r)
        final_g = scurve.(final_g)
        final_b = scurve.(final_b)
    end

    RGBf.(Float32.(final_r), Float32.(final_g), Float32.(final_b))
end

"""
    auto_balance!(image::Matrix{RGBf}; clip=0.001, midtone=0.45, max_gamma=2.2)

Photoshop-style "Auto Tone" (Enhance Per Channel Contrast): for each channel,
find the levels that clip `clip` (0.1% by default) of pixels at each end of
the histogram and stretch the channel across [0, 1] — which maximises
contrast *and* removes colour casts, since each channel is stretched
independently. Then a midtone pass: a single gamma nudges the median
luminance toward `midtone` (clamped to ±`max_gamma`), the equivalent of
Photoshop's automatic midtone slider. Apply as the last step of the photo
pipeline, on display-space values.
"""
function auto_balance!(image::Matrix{RGBf}; clip::Real=0.001,
                       midtone::Real=0.45, max_gamma::Real=2.2)
    n = length(image)
    n == 0 && return image
    stride = max(1, n ÷ 200_000)
    idxs = 1:stride:n

    lows = zeros(Float64, 3)
    highs = ones(Float64, 3)
    for (ch, getter) in enumerate((c -> c.r, c -> c.g, c -> c.b))
        vals = Float64[clamp(getter(image[k]), 0.0f0, 1.0f0) for k in idxs]
        sort!(vals)
        m = length(vals)
        lo = vals[clamp(round(Int, clip * m) + 1, 1, m)]
        hi = vals[clamp(m - round(Int, clip * m), 1, m)]
        if hi - lo > 1e-4
            lows[ch] = lo
            highs[ch] = hi
        end
    end

    inv_r = 1.0 / (highs[1] - lows[1])
    inv_g = 1.0 / (highs[2] - lows[2])
    inv_b = 1.0 / (highs[3] - lows[3])
    for k in eachindex(image)
        c = image[k]
        image[k] = RGBf(clamp((c.r - lows[1]) * inv_r, 0.0, 1.0),
                        clamp((c.g - lows[2]) * inv_g, 0.0, 1.0),
                        clamp((c.b - lows[3]) * inv_b, 0.0, 1.0))
    end

    # Midtone: median luminance -> `midtone` via a single gamma.
    lums = Float64[0.2126 * image[k].r + 0.7152 * image[k].g +
                   0.0722 * image[k].b for k in idxs]
    sort!(lums)
    med = clamp(lums[max(length(lums) ÷ 2, 1)], 1e-4, 1.0 - 1e-4)
    γ = clamp(log(midtone) / log(med), 1.0 / max_gamma, max_gamma)
    if !isapprox(γ, 1.0; atol=0.02)
        for k in eachindex(image)
            c = image[k]
            image[k] = RGBf(clamp(c.r, 0.0f0, 1.0f0)^γ,
                            clamp(c.g, 0.0f0, 1.0f0)^γ,
                            clamp(c.b, 0.0f0, 1.0f0)^γ)
        end
    end
    return image
end

# -----------------------------------------------------------------------------
# Lens / camera-body post-processing effects
# -----------------------------------------------------------------------------

"""
    apply_vignette!(image; strength=0.4, radius=1.0, falloff=1.5)

Apply natural vignetting in-place. Light falloff follows
`(1 - strength * (r / radius)^falloff)` clipped to [0, 1].
"""
function apply_vignette!(image::Matrix{RGBf}; strength::Real=0.4, radius::Real=1.0, falloff::Real=1.5)
    w, h = size(image)
    cx, cy = (w + 1) / 2.0, (h + 1) / 2.0
    r_max = sqrt(cx^2 + cy^2) * radius
    for j in 1:h, i in 1:w
        r = sqrt((i - cx)^2 + (j - cy)^2)
        factor = clamp(1.0 - strength * (r / r_max)^falloff, 0.0, 1.0)
        image[i, j] = image[i, j] * factor
    end
    return image
end

function apply_vignette!(image::Matrix{Float64}; strength::Real=0.4, radius::Real=1.0, falloff::Real=1.5)
    w, h = size(image)
    cx, cy = (w + 1) / 2.0, (h + 1) / 2.0
    r_max = sqrt(cx^2 + cy^2) * radius
    for j in 1:h, i in 1:w
        r = sqrt((i - cx)^2 + (j - cy)^2)
        factor = clamp(1.0 - strength * (r / r_max)^falloff, 0.0, 1.0)
        image[i, j] *= factor
    end
    return image
end

"""
    apply_lens_distortion!(image; k1=-0.05, k2=0.0)

Apply radial barrel/pincushion distortion in-place using a simple polynomial
model. `k1 < 0` gives barrel distortion; `k1 > 0` gives pincushion.
"""
function apply_lens_distortion!(image::Matrix{T}; k1::Real=-0.05, k2::Real=0.0) where T
    w, h = size(image)
    cx, cy = (w + 1) / 2.0, (h + 1) / 2.0
    r_max = sqrt(cx^2 + cy^2)
    out = similar(image)

    for j in 1:h, i in 1:w
        x = (i - cx) / r_max
        y = (j - cy) / r_max
        r2 = x^2 + y^2
        radial = 1.0 + k1 * r2 + k2 * r2^2

        xs = clamp(cx + x * radial * r_max, 1.0, w)
        ys = clamp(cy + y * radial * r_max, 1.0, h)

        x0 = floor(Int, xs)
        y0 = floor(Int, ys)
        x1 = min(x0 + 1, w)
        y1 = min(y0 + 1, h)
        fx = xs - x0
        fy = ys - y0

        c00 = image[x0, y0]
        c10 = image[x1, y0]
        c01 = image[x0, y1]
        c11 = image[x1, y1]
        out[i, j] = c00 * (1 - fx) * (1 - fy) +
                    c10 * fx * (1 - fy) +
                    c01 * (1 - fx) * fy +
                    c11 * fx * fy
    end
    image .= out
    return image
end

"""
    apply_chromatic_aberration!(image; strength=0.001)

Lateral (transverse) chromatic aberration in-place: the lens magnifies each
wavelength differently, so with green as the reference the red image is
scaled by `1 + strength` and the blue by `1 - strength` about frame centre.
The R/B mis-registration therefore grows linearly with radius — zero on axis,
`strength × half-diagonal` pixels in the corners — the red-outside /
blue-inside fringing of every wide lens. `strength` is unitless, so the look
is the same at every resolution.
"""
function apply_chromatic_aberration!(image::Matrix{RGBf}; strength::Real=0.001)
    w, h = size(image)
    cx, cy = (w + 1) / 2.0, (h + 1) / 2.0
    r_ch = Float64.(getfield.(image, :r))
    b_ch = Float64.(getfield.(image, :b))
    inv_r = 1.0 / (1.0 + strength)
    inv_b = 1.0 / (1.0 - strength)
    @inbounds for j in 1:h, i in 1:w
        xr = clamp(cx + (i - cx) * inv_r, 1.0, Float64(w))
        yr = clamp(cy + (j - cy) * inv_r, 1.0, Float64(h))
        xb = clamp(cx + (i - cx) * inv_b, 1.0, Float64(w))
        yb = clamp(cy + (j - cy) * inv_b, 1.0, Float64(h))
        c = image[i, j]
        image[i, j] = RGBf(_bilinear(r_ch, xr, yr), c.g, _bilinear(b_ch, xb, yb))
    end
    return image
end

"""
    lens_post(buckets; f_number, focus, fov_factor=0.55, fisheye_deg=0.0)

Depth-of-field as a post operation: composite the path-length buckets from
[`render_depth_mtl`](@ref), blurring each by its thin-lens circle of confusion
for the given aperture and focus distance. Aperture and focus become grading
dials — re-run at any f-stop without re-rendering. Returns a `Matrix{RGBf}`
in the renderer's `[width, height]` convention, ready for `postprocess`.

Approximation notes: blur is applied in image space per depth bucket, so
aperture rays are not re-traced — accurate for moderate apertures (~f/4 and
smaller); the exact-lens renderer remains the reference for fast glass near
the shadow edge.
"""
function lens_post(buckets::Array{Float32,3};
                   f_number::Real=8.0, focus::Real=30.0,
                   fov_factor::Real=0.55, fisheye_deg::Real=0.0)
    nb3, w, h = size(buckets)
    nb = nb3 ÷ 3
    ppr = fisheye_deg > 0 ? (h / 2) / deg2rad(fisheye_deg) :
                            (h / 2) / fov_factor      # pixels per radian
    A = focus / f_number                              # aperture diameter (M)
    s = (log(120.0) - log(1.5)) / (nb - 2)            # log bin width
    img = zeros(Float32, w, h, 3)
    for b in 1:nb
        d = b == nb ? Inf : 1.5 * exp((b - 0.5) * s)  # bucket centre distance
        θ = (A / 2) * abs(1.0 / d - 1.0 / focus)      # CoC angular radius
        rpx = θ * ppr
        if rpx < 0.6
            for c in 1:3
                img[:, :, c] .+= @view buckets[3 * (b - 1) + c, :, :]
            end
        else
            R = max(round(Int, rpx), 1)
            kmat = Float32[(x^2 + y^2) <= rpx^2 + 0.25 ? 1.0f0 : 0.0f0
                           for x in -R:R, y in -R:R]
            kmat ./= sum(kmat)
            kc = centered(kmat)
            for c in 1:3
                img[:, :, c] .+= imfilter(Array{Float32}(
                    @view buckets[3 * (b - 1) + c, :, :]), kc, "replicate")
            end
        end
    end
    return [RGBf(img[i, j, 1], img[i, j, 2], img[i, j, 3])
            for i in 1:w, j in 1:h]
end

# ---------------------------------------------------------------------------
# FFT-based bloom + star streak post-processing
# ---------------------------------------------------------------------------

# Chromatic scaling for bloom kernel per channel (R, G, B).
# Approximates diffraction: blue diffracts tighter, red wider.
const AIRY_SPECTRUM = SVector(1.0, 0.85, 0.7)

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
                     contrast=0.0)
    w, h = size(image)

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

    # 5. Combine: original HDR + bloom result
    final_r = r_ch .+ bloom_r
    final_g = g_ch .+ bloom_g
    final_b = b_ch .+ bloom_b

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

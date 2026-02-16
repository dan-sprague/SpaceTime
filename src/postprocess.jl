"""
    airy_disc(x)

The Airy disc function, which describes the diffraction pattern of a point light source. It is defined as (2 * J1(x) / x)^2, where J1 is the first-order Bessel function of the first kind.
"""
function airy_disc(x)
    (2 * besselj1(x) / x)^2
end

const AIRY_SPECTRUM = SVector(1.0, 0.86, 0.61)  # approximate R,G,B wavelength from starless python package https://github.com/rantonels/starless/blob/master/bloom.py

"""
    generate_kernel(scale, size)
Generates a 2D convolution kernel based on the Airy disc function for each color channel. The `scale` parameter controls the size of the Airy disc for each channel, and `size` determines the radius of the kernel.
"""
function generate_kernel(scale, size)
    coords = -size:size
    kernel = zeros(2size+1, 2size+1, 3)
    for (j, cy) in enumerate(coords), (i, cx) in enumerate(coords)
        r = sqrt(cx^2 + cy^2) + 1e-6
        for c in 1:3
            kernel[i, j, c] = airy_disc(r / scale[c])
        end
    end
    for c in 1:3
        kernel[:, :, c] ./= sum(@view kernel[:, :, c])
    end
    kernel
end

"""
    airy_convolve(image, radius; kernel_radius=25)

Applies an Airy disc convolution to the input image. The `radius` parameter controls the size of the Airy disc, and `kernel_radius` determines the radius of the convolution kernel.
"""
function airy_convolve(image::Matrix{<:RGB}, radius; kernel_radius=25)
    scale = radius .* AIRY_SPECTRUM
    kernel = generate_kernel(scale, kernel_radius)

    r_ch = Float64.(getfield.(image, :r))
    g_ch = Float64.(getfield.(image, :g))
    b_ch = Float64.(getfield.(image, :b))

    r_out = imfilter(r_ch, centered(kernel[:,:,1]), "symmetric")
    g_out = imfilter(g_ch, centered(kernel[:,:,2]), "symmetric")
    b_out = imfilter(b_ch, centered(kernel[:,:,3]), "symmetric")

    RGBf.(r_out, g_out, b_out)
end

# ---------------------------------------------------------------------------
# FFT-based bloom + star streak post-processing
# ---------------------------------------------------------------------------

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
    postprocess(image; gain=0.37, exposure=0.0, gamma=1.0,
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
                     gain=0.37,
                     exposure=0.0,
                     gamma=1.0,
                     bloom_strength=0.6,
                     threshold=0.5,
                     bloom_radius=15.0,
                     bloom_power=1.5,
                     streak_strength=0.3,
                     streak_length=0.4,
                     streak_width=1.5,
                     n_spikes=4,
                     angles=nothing,
                     tonemap=:aces)
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

    # 6. Tonemap
    if tonemap == :aces
        final_r = aces_tonemap.(final_r)
        final_g = aces_tonemap.(final_g)
        final_b = aces_tonemap.(final_b)
    elseif tonemap == :reinhard
        final_r = final_r ./ (1.0 .+ final_r)
        final_g = final_g ./ (1.0 .+ final_g)
        final_b = final_b ./ (1.0 .+ final_b)
    end

    # 7. Gamma
    if gamma != 1.0
        inv_gamma = 1.0 / gamma
        final_r = final_r .^ inv_gamma
        final_g = final_g .^ inv_gamma
        final_b = final_b .^ inv_gamma
    end

    RGBf.(Float32.(final_r), Float32.(final_g), Float32.(final_b))
end

# Keep old signature for backwards compatibility
function postprocess(image::Matrix{RGBf}, fov_factor; airy_radius=0.5, gain=0.37, glare_intensity=0.1)
    w, h = size(image)
    img = image .* gain

    threshold = 0.5
    bright_pass = map(c -> RGBf(max(0, c.r - threshold),
                                max(0, c.g - threshold),
                                max(0, c.b - threshold)), img)

    glow = zeros(RGBf, w, h)
    scales = [0.005, 0.02, 0.05, 0.1]
    weights = [0.5, 0.15, 0.1, 0.25]

    for (s, weight) in zip(scales, weights)
        sigma = w * s
        glow .+= imfilter(bright_pass, Kernel.gaussian(sigma)) .* (weight * glare_intensity)
    end

    img_combined = img .+ glow
    γ = 1.6
    img_combined = map(c -> RGBf(c.r^γ, c.g^γ, c.b^γ), img_combined)
    map(c -> RGB{Float32}(
        clamp(c.r, 0.0, Inf) / (1.0 + max(0.0, c.r)),
        clamp(c.g, 0.0, Inf) / (1.0 + max(0.0, c.g)),
        clamp(c.b, 0.0, Inf) / (1.0 + max(0.0, c.b))
    ), img_combined)
end

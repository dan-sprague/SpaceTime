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

"""
    postprocess(image, fov_factor; airy_radius=0.5, gain=0.37, glare_intensity=0.1)
Applies post-processing effects to the rendered image, including an Airy disc convolution for bright spots and a glare effect. The `fov_factor` can be used to adjust the intensity of the effects based on the field of view.
"""
function postprocess(image::Matrix{RGBf}, fov_factor; airy_radius=0.5, gain=0.37, glare_intensity=0.1)
    w, h = size(image)
    img = image .* gain

    threshold = 0.5
    bright_pass = map(c -> RGBf(max(0, c.r - threshold), 
                                max(0, c.g - threshold), 
                                max(0, c.b - threshold)), img)

    glow = zeros(RGBf, w, h)
    scales = [0.005, 0.02, 0.05, 0.1]
    weights = [0.5, 0.15, 0.1, 0.25]  # How much each scale contributes

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



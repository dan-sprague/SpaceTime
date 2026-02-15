  function veiling_flare(image::Matrix{<:RGB}; intensity=0.1)
    n = length(image)
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    for c in image
        sum_r += c.r
        sum_g += c.g
        sum_b += c.b
    end
    veil = RGBf(sum_r / n, sum_g / n, sum_b / n) * intensity
    image .+ Ref(veil)
end
                                                                                
  function airy_disk(x)
      (2 * besselj1(x) / x)^2
  end

  const AIRY_SPECTRUM = SVector(1.0, 0.86, 0.61)  # approximate R,G,B wavelength

  function generate_kernel(scale, size)
      coords = -size:size
      kernel = zeros(2size+1, 2size+1, 3)
      for (j, cy) in enumerate(coords), (i, cx) in enumerate(coords)
          r = sqrt(cx^2 + cy^2) + 1e-6
          for c in 1:3
              kernel[i, j, c] = airy_disk(r / scale[c])
          end
      end
      for c in 1:3
          kernel[:, :, c] ./= sum(@view kernel[:, :, c])
      end
      kernel
  end

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



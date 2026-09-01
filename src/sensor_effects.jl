# ==============================================================================
# Sensor / lens-plane effects — lens dust specks and micrometeoroid streaks.
# ==============================================================================

"""
    LensDust(; count=0, size_min=2.0, size_max=15.0, opacity_min=0.1,
              opacity_max=0.6, edge_softness=1.5)

Parameters for dust specks on the lens or sensor.

- `count`: number of dust particles.  0 = disabled.
- `size_min`, `size_max`: radius range in pixels.
- `opacity_min`, `opacity_max`: opacity range (0 = invisible, 1 = opaque).
- `edge_softness`: Gaussian falloff exponent for the speck edge (higher = sharper).
"""
struct LensDust
    count::Int
    size_min::Float64
    size_max::Float64
    opacity_min::Float64
    opacity_max::Float64
    edge_softness::Float64
end

LensDust(; count=0, size_min=2.0, size_max=15.0, opacity_min=0.1,
         opacity_max=0.6, edge_softness=1.5) =
    LensDust(count, size_min, size_max, opacity_min, opacity_max, edge_softness)

"""
    _lens_dust_speck!(image, cx, cy, radius, opacity, softness)

Draw one circular dust speck with soft Gaussian edges.
"""
function _lens_dust_speck!(image::Matrix{RGBf}, cx::Float64, cy::Float64,
                           radius::Float64, opacity::Float64, softness::Float64)
    w, h = size(image)
    # Bounding box
    i0 = max(floor(Int, cx - 2.0 * radius), 1)
    i1 = min(ceil(Int, cx + 2.0 * radius), w)
    j0 = max(floor(Int, cy - 2.0 * radius), 1)
    j1 = min(ceil(Int, cy + 2.0 * radius), h)

    sigma = radius / softness
    two_s2 = 2.0 * sigma^2
    for j in j0:j1, i in i0:i1
        r2 = (i - cx)^2 + (j - cy)^2
        if r2 < (2.5 * radius)^2
            # Soft edge: Gaussian profile with flat top
            if r2 <= radius^2
                factor = 1.0 - opacity
            else
                d = sqrt(r2) - radius
                edge = exp(-d^2 / two_s2)
                factor = 1.0 - opacity * edge
            end
            image[i, j] = RGBf(image[i, j].r * factor,
                               image[i, j].g * factor,
                               image[i, j].b * factor)
        end
    end
end

"""
    apply_lens_dust!(image; lens_dust::LensDust=LensDust(), ref_height=nothing,
                     rng=default_rng())

Apply random lens dust specks to `image` in-place.  Each speck is a small
dark circular patch with soft edges, simulating dust on the lens or sensor.

`LensDust` speck radii are in pixels, so the same numbers make specks that
cover six times less of the frame at 2160 lines than at 360 — but a real mote
on the glass covers a fixed *fraction* of the frame no matter what sensor is
behind it. Pass `ref_height`, the height the sizes were chosen at, to scale
them and keep the dust the same physical size. Left off, behaviour is
unchanged.
"""
function apply_lens_dust!(image::Matrix{RGBf}; lens_dust::LensDust=LensDust(),
                          ref_height::Union{Real,Nothing}=nothing,
                          rng::Random.AbstractRNG=Random.default_rng())
    lens_dust.count <= 0 && return image
    w, h = size(image)
    sc = ref_height === nothing ? 1.0 : h / Float64(ref_height)

    for _ in 1:lens_dust.count
        cx = rand(rng) * (w - 1) + 1.0
        cy = rand(rng) * (h - 1) + 1.0
        radius = sc * (lens_dust.size_min +
                       rand(rng) * (lens_dust.size_max - lens_dust.size_min))
        opacity = lens_dust.opacity_min + rand(rng) * (lens_dust.opacity_max - lens_dust.opacity_min)
        _lens_dust_speck!(image, cx, cy, radius, opacity, lens_dust.edge_softness)
    end
    return image
end

# ------------------------------------------------------------------------------
# Micrometeoroid streaks.
# ------------------------------------------------------------------------------

"""
    MicroStreaks(; count=0, length_min=20.0, length_max=120.0,
                  width_min=1.0, width_max=3.0, brightness_min=0.3,
                  brightness_max=1.5, angle_spread=π/6)

Parameters for micrometeoroid streaks across the image.

Micrometeoroids are small fast-moving particles that cross the field of view
during an exposure, leaving bright linear streaks (like meteors, but caused
by particles heated by the accretion disc or collisional heating).

- `count`: number of streaks.  0 = disabled.
- `length_min`, `length_max`: streak length range in pixels.
- `width_min`, `width_max`: streak half-width range in pixels (Gaussian σ).
- `brightness_min`, `brightness_max`: peak brightness over background.
- `angle_spread`: streaks are concentrated near the disc plane (± this angle
  from the horizontal axis in radians).
"""
struct MicroStreaks
    count::Int
    length_min::Float64
    length_max::Float64
    width_min::Float64
    width_max::Float64
    brightness_min::Float64
    brightness_max::Float64
    angle_spread::Float64
end

MicroStreaks(; count=0, length_min=20.0, length_max=120.0,
             width_min=1.0, width_max=3.0, brightness_min=0.3,
             brightness_max=1.5, angle_spread=π/6) =
    MicroStreaks(count, length_min, length_max, width_min, width_max,
                 brightness_min, brightness_max, angle_spread)

"""
    _draw_streak!(image, x0, y0, x1, y1, sigma, brightness)

Draw a single Gaussian-profile streak from (x0,y0) to (x1,y1).
"""
function _draw_streak!(image::Matrix{RGBf}, x0::Float64, y0::Float64,
                       x1::Float64, y1::Float64, sigma::Float64,
                       brightness::Float64)
    w, h = size(image)
    dx = x1 - x0
    dy = y1 - y0
    length_streak = sqrt(dx^2 + dy^2)
    length_streak < 1.0 && return

    ux = dx / length_streak  # unit along streak
    uy = dy / length_streak
    nx = -uy                  # perpendicular
    ny = ux

    # Bounding box in image coordinates
    margin = 4.0 * sigma
    x_min = min(x0, x1) - margin
    x_max = max(x0, x1) + margin
    y_min = min(y0, y1) - margin
    y_max = max(y0, y1) + margin

    i0 = max(floor(Int, x_min), 1)
    i1 = min(ceil(Int, x_max), w)
    j0 = max(floor(Int, y_min), 1)
    j1 = min(ceil(Int, y_max), h)

    two_s2 = 2.0 * sigma^2
    half_len = length_streak / 2.0
    mid_x = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0

    for j in j0:j1, i in i0:i1
        # Project onto streak coordinate system
        rx = i - mid_x
        ry = j - mid_y
        along = rx * ux + ry * uy   # distance along streak
        perp = rx * nx + ry * ny     # perpendicular distance

        # Check if within streak extent
        if abs(along) <= half_len + margin
            # Gaussian perpendicular falloff, exponential along
            perp_factor = exp(-perp^2 / two_s2)
            along_factor = exp(-abs(along) / (half_len * 0.7))
            factor = brightness * perp_factor * along_factor
            image[i, j] = RGBf(image[i, j].r + factor,
                               image[i, j].g + factor * 0.9,
                               image[i, j].b + factor * 0.7)
        end
    end
end

"""
    apply_micro_streaks!(image; streaks::MicroStreaks=MicroStreaks(),
                          ref_height=nothing, rng=default_rng())

Add random micrometeoroid streaks to `image` in-place.  Streaks are
concentrated near the equatorial plane (accretion disc plane) and have a
warm colour cast.

Streak lengths and widths are in pixels, so like lens dust they shrink
relative to the frame as resolution rises — a particle crossing the field
during an exposure sweeps a fixed fraction of the frame, not a fixed pixel
count. Pass `ref_height`, the height the sizes were chosen at, to scale them.
Left off, behaviour is unchanged.
"""
function apply_micro_streaks!(image::Matrix{RGBf};
                              streaks::MicroStreaks=MicroStreaks(),
                              ref_height::Union{Real,Nothing}=nothing,
                              rng::Random.AbstractRNG=Random.default_rng())
    streaks.count <= 0 && return image
    w, h = size(image)
    cx, cy = w / 2.0, h / 2.0
    sc = ref_height === nothing ? 1.0 : h / Float64(ref_height)

    for _ in 1:streaks.count
        # Random angle, biased toward the disc plane (horizontal)
        base_angle = rand(rng) * 2π
        disc_tilt = randn(rng) * streaks.angle_spread
        angle = base_angle + disc_tilt

        cos_a, sin_a = cos(angle), sin(angle)
        length_s = sc * (streaks.length_min +
                         rand(rng) * (streaks.length_max - streaks.length_min))
        sigma = sc * (streaks.width_min +
                      rand(rng) * (streaks.width_max - streaks.width_min))
        brightness = streaks.brightness_min + rand(rng) * (streaks.brightness_max - streaks.brightness_min)

        # Random midpoint position
        mx = cx + (rand(rng) - 0.5) * w * 1.2
        my = cy + (rand(rng) - 0.5) * h * 1.2

        half = length_s / 2.0
        x0 = mx - half * cos_a
        y0 = my - half * sin_a
        x1 = mx + half * cos_a
        y1 = my + half * sin_a

        _draw_streak!(image, x0, y0, x1, y1, sigma, brightness)
    end
    return image
end

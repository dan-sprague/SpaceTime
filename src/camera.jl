abstract type AbstractCamera end

"""
    Camera(pos, target, up, fov_factor=1.0)

A simple pinhole camera. All rays originate from `pos` and pass through an
imaginary sensor plane defined by `fov_factor = tan(half_fov)`. This is the
fastest camera model and matches the original `Camera` behaviour.
"""
struct Camera <: AbstractCamera
    pos::SVector{3, Float64}
    fwd::SVector{3, Float64}
    right::SVector{3, Float64}
    up_local::SVector{3, Float64}
    fov_factor::Float64

    function Camera(pos, target, up, fov_factor=1.0)
        fwd = normalize(target - pos)
        right = normalize(cross(fwd, up))
        up_local = cross(right, fwd)
        new(pos, fwd, right, up_local, fov_factor)
    end
end

"""Alias for the pinhole `Camera` type."""
const PinholeCamera = Camera

"""
    ThinLensCamera(pos, target, up; focal_length=50.0, sensor_width=36.0,
                   f_number=2.8, focus_distance=100.0)

A thin-lens camera with a circular aperture. `focal_length` and `sensor_width`
are in millimetres; `focus_distance` is in world units. The aperture diameter
is `focal_length / f_number`. Rays are sampled over the lens aperture and
converge at the focal plane, producing depth-of-field blur.
"""
struct ThinLensCamera <: AbstractCamera
    pos::SVector{3, Float64}
    fwd::SVector{3, Float64}
    right::SVector{3, Float64}
    up_local::SVector{3, Float64}
    focal_length::Float64       # mm
    sensor_width::Float64       # mm
    fov_factor::Float64         # (sensor_width / 2) / focal_length
    aperture::Float64           # mm
    focus_distance::Float64     # world units

    function ThinLensCamera(pos, target, up;
                              focal_length=50.0,
                              sensor_width=36.0,
                              f_number=2.8,
                              focus_distance=100.0)
        fwd = normalize(target - pos)
        right = normalize(cross(fwd, up))
        up_local = cross(right, fwd)
        fov_factor = (sensor_width / 2.0) / focal_length
        aperture = focal_length / f_number
        new(pos, fwd, right, up_local, focal_length, sensor_width, fov_factor,
            aperture, focus_distance)
    end
end

"""
    get_ray_direction(cam::Camera, u, v)

Direction of a ray through the normalised sensor coordinate `(u, v)` for a
pinhole camera. `u` and `v` are in the range [-1, 1] covering the sensor.
"""
function get_ray_direction(cam::Camera, u, v)
    dir_local = SVector{3}(u * cam.fov_factor, v * cam.fov_factor, 1.0)
    normalize(cam.right * dir_local[1] +
              cam.up_local * dir_local[2] +
              cam.fwd * dir_local[3])
end

"""
    sample_lens_point(aperture, rng=Random.default_rng())

Sample a point uniformly on a disk of diameter `aperture` in the local
(right, up) plane. Returns offsets `(dx, dy)` in camera-local coordinates.
"""
function sample_lens_point(aperture::Real, rng::Random.AbstractRNG=Random.default_rng())
    r = (aperture / 2.0) * sqrt(rand(rng))
    θ = 2π * rand(rng)
    r * cos(θ), r * sin(θ)
end

"""
    concentric_disk(u, v)

Shirley–Chiu low-distortion mapping from `(u, v) ∈ [0,1)²` to the unit disk
(PBRT's `SampleUniformDiskConcentric`). Adjacent sample strata map to adjacent
disk areas, so stratified lens samples cover the aperture far more evenly than
independent uniform draws — much less defocus noise at the same ray count.
"""
function concentric_disk(u::Real, v::Real)
    ox = 2.0 * u - 1.0
    oy = 2.0 * v - 1.0
    if ox == 0.0 && oy == 0.0
        return 0.0, 0.0
    end
    if abs(ox) > abs(oy)
        r = ox
        θ = (π / 4) * (oy / ox)
    else
        r = oy
        θ = π / 2 - (π / 4) * (ox / oy)
    end
    return r * cos(θ), r * sin(θ)
end

"""
    get_ray(cam::Camera, u, v)
    get_ray(cam::ThinLensCamera, u, v, rng=Random.default_rng())

Return `(origin::SVector{3,Float64}, direction::SVector{3,Float64})` for the
specified camera and sensor coordinate. For a thin lens the ray origin is
sampled on the aperture and the direction is chosen so that the ray passes
through the focal-plane point of a pinhole ray.
"""
function get_ray(cam::Camera, u, v, rng::Random.AbstractRNG=Random.default_rng())
    return cam.pos, get_ray_direction(cam, u, v)
end

"""
    FisheyeCamera(pos, target, up; theta_edge=deg2rad(100.0))

Equidistant fisheye camera: pixel radius maps linearly to view angle, with
`theta_edge` the half-angle at the top edge of the frame (the horizontal
edge extends by the aspect ratio). Fields with wider-than-180° views render
cleanly — a rectilinear pinhole cannot. Matches the Metal kernel's fisheye
lens (`fisheye_deg`).
"""
struct FisheyeCamera <: AbstractCamera
    pos::SVector{3, Float64}
    fwd::SVector{3, Float64}
    right::SVector{3, Float64}
    up_local::SVector{3, Float64}
    theta_edge::Float64

    function FisheyeCamera(pos, target, up; theta_edge=deg2rad(100.0))
        fwd = normalize(target - pos)
        right = normalize(cross(fwd, up))
        up_local = cross(right, fwd)
        new(pos, fwd, right, up_local, theta_edge)
    end
end

function get_ray(cam::FisheyeCamera, u, v,
                 rng::Random.AbstractRNG=Random.default_rng())
    ρ = sqrt(u^2 + v^2)
    ρ < 1e-12 && return cam.pos, cam.fwd
    θ = ρ * cam.theta_edge
    s, c = sincos(θ)
    dir = c * cam.fwd + (s / ρ) * (u * cam.right + v * cam.up_local)
    return cam.pos, dir
end

function _thin_lens_ray(cam::ThinLensCamera, u, v, dx, dy)
    # Pinhole direction and the point it hits on the focal plane.
    pinhole_dir = get_ray_direction(Camera(cam.pos, cam.pos + cam.fwd,
                                           cam.up_local, cam.fov_factor), u, v)
    focal_point = cam.pos + cam.focus_distance * pinhole_dir
    origin = cam.pos + dx * cam.right + dy * cam.up_local
    direction = normalize(focal_point - origin)
    return origin, direction
end

# Convert aperture from mm to world units.  The sensor half-width
# (sensor_width / 2) mm maps to fov_factor * focus_distance world units
# at the focal plane, so 1 mm = focus_distance / focal_length world units.
# The world-space aperture diameter then simplifies to focus_distance / f_number.
_aperture_world(cam::ThinLensCamera) =
    cam.aperture * cam.focus_distance / cam.focal_length

function get_ray(cam::ThinLensCamera, u, v, rng::Random.AbstractRNG=Random.default_rng())
    dx, dy = sample_lens_point(_aperture_world(cam), rng)
    return _thin_lens_ray(cam, u, v, dx, dy)
end

"""
    get_ray(cam, u, v, rng, lens::NTuple{2,Float64})

Stratified-lens variant: `lens` is a sample in `[0,1)²` mapped onto the
aperture disk via `concentric_disk`. Cameras without an aperture ignore it.
"""
function get_ray(cam::ThinLensCamera, u, v, rng::Random.AbstractRNG,
                 lens::NTuple{2, Float64})
    px, py = concentric_disk(lens[1], lens[2])
    half = _aperture_world(cam) / 2.0
    return _thin_lens_ray(cam, u, v, half * px, half * py)
end

get_ray(cam::AbstractCamera, u, v, rng::Random.AbstractRNG,
        lens::NTuple{2, Float64}) = get_ray(cam, u, v, rng)

# -----------------------------------------------------------------------------
# Camera transforms
# -----------------------------------------------------------------------------

"""Rebuild a camera from its stored fields with a new pos/fwd, preserving up_local."""
function _rebuild(cam::AbstractCamera, pos, fwd)
    target = pos + fwd
    if cam isa ThinLensCamera
        ThinLensCamera(pos, target, cam.up_local;
                       focal_length=cam.focal_length,
                       sensor_width=cam.sensor_width,
                       f_number=cam.focal_length / cam.aperture,
                       focus_distance=cam.focus_distance)
    else
        Camera(pos, target, cam.up_local, cam.fov_factor)
    end
end

"""Yaw: rotate the look direction horizontally by `deg` degrees (+right, −left)."""
function yaw(cam::AbstractCamera, deg)
    α = deg2rad(deg)
    new_fwd = normalize(cam.fwd * cos(α) + cam.right * sin(α))
    _rebuild(cam, cam.pos, new_fwd)
end

"""Pitch: rotate the look direction vertically by `deg` degrees (+up, −down)."""
function pitch(cam::AbstractCamera, deg)
    α = deg2rad(deg)
    new_fwd = normalize(cam.fwd * cos(α) + cam.up_local * sin(α))
    _rebuild(cam, cam.pos, new_fwd)
end

"""Roll: rotate the camera around its look direction by `deg` degrees."""
function roll(cam::AbstractCamera, deg)
    α = deg2rad(deg)
    new_up = normalize(cam.up_local * cos(α) + cam.right * sin(α))
    new_right = normalize(cross(cam.fwd, new_up))
    new_up = cross(new_right, cam.fwd)
    if cam isa ThinLensCamera
        target = cam.pos + cam.fwd
        ThinLensCamera(cam.pos, target, new_up;
                       focal_length=cam.focal_length,
                       sensor_width=cam.sensor_width,
                       f_number=cam.focal_length / cam.aperture,
                       focus_distance=cam.focus_distance)
    else
        Camera(cam.pos, cam.pos + cam.fwd, new_up, cam.fov_factor)
    end
end

"""Truck: translate the camera along its right vector by `dist` units (+right, −left)."""
function truck(cam::AbstractCamera, d)
    _rebuild(cam, cam.pos + d * cam.right, cam.fwd)
end

"""Pedestal: translate the camera along its up vector by `dist` units (+up, −down)."""
function pedestal(cam::AbstractCamera, d)
    _rebuild(cam, cam.pos + d * cam.up_local, cam.fwd)
end

"""Dolly: translate the camera along its forward vector by `dist` units (+fwd, −back)."""
function dolly(cam::AbstractCamera, d)
    _rebuild(cam, cam.pos + d * cam.fwd, cam.fwd)
end

"""
    @gimbal cam |> transform arg |> transform arg ...

Chain camera transforms: `@gimbal cam_orig |> yaw 18.0 |> pitch -10.0`
expands to `pitch(yaw(cam_orig, 18.0), -10.0)`.
"""
macro gimbal(cam, name_and_args...)
    args = [a for a in name_and_args if a !== :|>]
    length(args) % 2 == 0 || error("@gimbal: expected pairs of (transform, value)")
    result = esc(cam)
    for i in 1:2:length(args)
        result = Expr(:call, esc(args[i]), result, esc(args[i+1]))
    end
    return result
end

"""
    offset_camera(cam, bh_pos; mode=:rotate, angle=nothing, side=:left, aspect=3/2)

Reframe `cam` so the black hole sits at the 1/3 line (or at a custom `angle`
in degrees).
- `mode=:rotate` — yaw the look direction (preserves distance and shadow size).
- `mode=:translate` — horizontal truck (changes distance; shadow shrinks).
"""
function offset_camera(cam::AbstractCamera, bh_pos; mode=:rotate, angle=nothing, side=:left, aspect=3/2)
    dist  = norm(bh_pos - cam.pos)
    sign  = (side == :left) ? 1 : -1

    if mode == :rotate
        pan_deg = isnothing(angle) ? rad2deg(atan(cam.fov_factor * aspect / 3)) : angle
        return yaw(cam, sign * pan_deg)

    elseif mode == :translate
        pan_dir = normalize(SVector(cam.right[1], cam.right[2], 0.0))
        s = isnothing(angle) ? dist * cam.fov_factor * aspect / 3 : dist * tan(deg2rad(angle))
        new_pos = cam.pos + sign * s * pan_dir
        new_target = bh_pos + sign * s * pan_dir
        if cam isa ThinLensCamera
            return ThinLensCamera(new_pos, new_target, cam.up_local;
                                  focal_length=cam.focal_length,
                                  sensor_width=cam.sensor_width,
                                  f_number=cam.focal_length / cam.aperture,
                                  focus_distance=cam.focus_distance)
        else
            return Camera(new_pos, new_target, cam.up_local, cam.fov_factor)
        end
    else
        error("Unknown mode: $mode. Use :rotate or :translate.")
    end
end

# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------

"""
    jittered_grid(samples; rng=Random.default_rng())

Return a vector of `(du, dv)` offsets for stratified jittered supersampling.
Divides the pixel into `samples × samples` cells and picks one random point
inside each cell. This reduces aliasing compared to a regular grid.
"""
function jittered_grid(samples::Int; rng::Random.AbstractRNG=Random.default_rng())
    offsets = Vector{NTuple{2, Float64}}(undef, samples * samples)
    return jittered_grid!(offsets, samples, rng)
end

"""
    jittered_grid!(offsets, samples, rng)

In-place `jittered_grid` for per-pixel refresh without allocations. `offsets`
must have length `samples^2`.
"""
function jittered_grid!(offsets::Vector{NTuple{2, Float64}}, samples::Int,
                        rng::Random.AbstractRNG)
    inv_s = 1.0 / samples
    idx = 1
    for si in 0:(samples-1), sj in 0:(samples-1)
        du = (si + rand(rng)) * inv_s
        dv = (sj + rand(rng)) * inv_s
        offsets[idx] = (du, dv)
        idx += 1
    end
    return offsets
end

"""
    sensor_coordinate(i, j, width, height; du=0.5, dv=0.5)

Map integer pixel indices `(i, j)` and sub-pixel offsets `du, dv ∈ [0,1]` to
normalised sensor coordinates `(u, v)` in [-1, 1]. The mapping preserves aspect
ratio by scaling both axes by `height/2`.
"""
function sensor_coordinate(i, j, width, height; du=0.5, dv=0.5)
    half_h = height / 2.0
    u = (i - 0.5 - width / 2.0 + du) / half_h
    v = (j - 0.5 - height / 2.0 + dv) / half_h
    return u, v
end

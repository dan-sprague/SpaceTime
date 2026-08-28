"""
    Photon

Represents a photon in the spacetime, encapsulating its position and momentum
as an 8-component state vector. The first four components represent the
coordinates (t, r, θ, ϕ), while the last four represent the corresponding
momenta (pt, pr, pθ, pϕ).
"""
struct Photon
    μ::SVector{8, Float64}  # (t, r, θ, ϕ, pt, pr, pθ, pϕ)

    Photon(μ::SVector{8, Float64}) = new(μ)
    Photon(q::SVector{4, Float64}, p::SVector{4, Float64}) = new(vcat(q, p))
end

"""
    Lens(focal_length)

A camera lens defined by focal length (mm) assuming a 35mm full-frame sensor.
The `fov_factor` is `18.0 / focal_length`, i.e. `tan(half_fov)`.
"""
struct Lens
    focal_length::Float64
    fov_factor::Float64
    Lens(focal_length) = new(focal_length, 18.0 / focal_length)
end

"""Convenience constructor: build a pinhole camera from a `Lens`."""
Camera(pos, target, up, lens::Lens) = Camera(pos, target, up, lens.fov_factor)

"""
    RayData

Mutable per-thread metadata for each ray during integration. Holds the
accumulated colour, remaining alpha, and closest approach radius.
"""
mutable struct RayData
    acc_color::RGBf
    alpha::Float64
    r_min::Float64
    path_length::Float64
end

"""
    init_photon(origin, direction, spacetime::AbstractSpacetime)

Build a `Photon` from a ray `origin` and unit `direction` in Cartesian
coordinates. The position is converted to spherical coordinates and the
corresponding null momentum is computed from the inverse metric.
"""
function init_photon(origin::SVector{3,Float64}, direction::SVector{3,Float64},
                     spacetime::AbstractSpacetime)
    dx, dy, dz = direction
    x, y, z = origin

    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = atan(y, x)
    q0 = @SVector [0.0, r, θ, ϕ]

    # Project unit direction onto spherical basis vectors.
    vr = dx * sin(θ) * cos(ϕ) + dy * sin(θ) * sin(ϕ) + dz * cos(θ)
    vθ = (dx * cos(θ) * cos(ϕ) + dy * cos(θ) * sin(ϕ) - dz * sin(θ)) / r
    vϕ = (-dx * sin(ϕ) + dy * cos(ϕ)) / (r * sin(θ))

    g_inv = metric_inverse(spacetime, q0)

    pr = vr / g_inv[2,2]
    pθ = vθ / g_inv[3,3]
    pϕ = vϕ / g_inv[4,4]

    spatial_part = g_inv[2,2]*pr^2 + g_inv[3,3]*pθ^2 + g_inv[4,4]*pϕ^2
    pt = -sqrt(abs(spatial_part / g_inv[1,1]))

    return vcat(q0, SVector(pt, pr, pθ, pϕ))
end

"""
    init_photon(cam::AbstractCamera, spacetime::AbstractSpacetime, u, v; rng=Random.default_rng())

Convenience wrapper that samples a ray from `cam` at normalised sensor
coordinates `(u, v)` and converts it to a photon state vector.
"""
function init_photon(cam::AbstractCamera, spacetime::AbstractSpacetime, u, v;
                     rng::Random.AbstractRNG=Random.default_rng())
    origin, direction = get_ray(cam, u, v, rng)
    init_photon(origin, direction, spacetime)
end

"""
    sample_background(img, θ, ϕ)

Bilinearly sample the background image `img` at spherical coordinates
`(θ, ϕ)`.
"""
function sample_background(img, θ, ϕ)
    v = clamp(θ / π, 0.0, 1.0)
    u = clamp(mod2pi(ϕ) / (2π), 0.0, 1.0)

    w, h = size(img)
    xf = u * (w - 1) + 1
    yf = v * (h - 1) + 1

    x0 = clamp(floor(Int, xf), 1, w)
    x1 = clamp(x0 + 1, 1, w)
    y0 = clamp(floor(Int, yf), 1, h)
    y1 = clamp(y0 + 1, 1, h)

    fx = xf - x0
    fy = yf - y0

    c00 = img[x0, y0]
    c10 = img[x1, y0]
    c01 = img[x0, y1]
    c11 = img[x1, y1]

    return c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) +
           c01 * (1 - fx) * fy + c11 * fx * fy
end

# -----------------------------------------------------------------------------
# Core render loop helpers
# -----------------------------------------------------------------------------

"""
    _render_nchunks() -> Int

Number of column chunks (and hence worker tasks / per-chunk integrators) used
by the render loops. Several chunks per default-pool thread: columns through
the disc and shadow cost far more than sky columns, and finer chunks let the
scheduler balance that (measured 12–18% faster than one chunk per thread at
4× oversubscription; beyond ~4× the extra integrator setups start to win it
back).
"""
_render_nchunks() = 4 * max(1, Threads.nthreads(:default))

"""
    _foreach_column_chunk(body, width, nchunks)

Partition columns `1:width` into `nchunks` contiguous chunks and run
`body(chunk_index, columns)` on a spawned task per chunk. Unlike
`Threads.@threads :static`, this composes when called from an already-spawned
task and never runs work on the interactive/main thread when default-pool
threads exist.
"""
function _foreach_column_chunk(body::Function, width::Int, nchunks::Int)
    chunks = collect(Iterators.partition(1:width, cld(width, nchunks)))
    @sync for (ci, cols) in pairs(chunks)
        Threads.@spawn body(ci, cols)
    end
    nothing
end

"""
    _trace_color(integrator, meta, cam, spacetime, background, disc, u, v; rng, dust)

Trace one sample through `(u, v)` and return the accumulated RGB colour.
"""
function _trace_color(integrator, meta, cam::AbstractCamera,
                      spacetime::Schwarzschild, background, disc, u, v;
                      rng::Random.AbstractRNG=Random.default_rng(),
                      dust::Union{InterstellarDust,Nothing}=nothing)
    μ0 = init_photon(cam, spacetime, u, v; rng)

    meta.acc_color = RGBf(0, 0, 0)
    meta.alpha = 1.0
    meta.r_min = Inf
    meta.path_length = 0.0
    integrator.p = (spacetime, meta, disc)
    reinit!(integrator, μ0)
    solve!(integrator)

    final_r = integrator.sol.u[end][2]
    final_θ = integrator.sol.u[end][3]
    final_ϕ = integrator.sol.u[end][4]

    # Compute approximate path length for dust extinction.
    r_start = μ0[2]
    path_len = abs(r_start - final_r)  # radial component
    # Add transverse component for non-radial rays.
    Δθ = abs(μ0[3] - final_θ)
    path_len += r_start * Δθ * 0.5  # approximate
    meta.path_length = path_len

    color = if final_r < 2.1 * spacetime.M
        meta.acc_color
    else
        bg_color = sample_background(background, final_θ, final_ϕ)
        meta.acc_color + (bg_color * meta.alpha)
    end

    # Apply interstellar dust extinction along the line of sight.
    if !isnothing(dust) && dust.density > 0.0
        color = apply_dust_extinction(color, path_len, dust)
    end

    return color
end

"""
    _trace_grayscale(integrator, meta, cam, spacetime, u, v; rng)

Trace one sample through `(u, v)` and return a grayscale value.
"""
function _trace_grayscale(integrator, meta, cam::AbstractCamera,
                          spacetime::AbstractSpacetime, u, v;
                          rng::Random.AbstractRNG=Random.default_rng())
    μ0 = init_photon(cam, spacetime, u, v; rng)

    meta.acc_color = RGBf(0, 0, 0)
    meta.alpha = 1.0
    meta.r_min = Inf
    integrator.p = (spacetime, meta)
    reinit!(integrator, μ0)
    solve!(integrator)

    final_r = integrator.sol.u[end][2]
    return final_r < 2.1 * spacetime.M ? 0.0 : 0.5 + 0.5 * sin(10 * integrator.sol.u[end][4])
end

"""
    render(cam::AbstractCamera, spacetime::Schwarzschild, background;
           disc=AccretionDisc(), width=400, height=200, solver=Tsit5(),
           samples=1, jittered=true, rng=Random.default_rng())

Render a colour image with Doppler-shifted accretion disc and background.
Supports pinhole and thin-lens cameras, stratified jittered supersampling,
and per-thread RNG.
"""
function render(cam::AbstractCamera, spacetime::Schwarzschild, background;
                disc::Union{AccretionDisc,Nothing}=AccretionDisc(),
                dust::Union{InterstellarDust,Nothing}=nothing,
                volume::Union{DiscVolume,Nothing}=nothing,
                width::Int=400, height::Int=200,
                solver=Tsit5(),
                samples::Int=1,
                jittered::Bool=true,
                rng::Random.AbstractRNG=Random.default_rng(),
                progress::Union{Function,Nothing}=nothing)
    image = zeros(RGBf, width, height)
    cam_dist = norm(cam.pos)
    r_max = max(5.0 * cam_dist, 100.0)
    tspan = (0.0, max(10.0 * cam_dist, 500.0))
    # A volumetric disc replaces the thin-plane crossing callback.
    cb_for(rm) = isnothing(volume) ? make_cb_set(rm, disc) :
        CallbackSet(ContinuousCallback(make_boundary_condition(rm), horizon_affect!),
                    make_volume_cb(volume, disc))

    nchunks = _render_nchunks()
    thread_metas = [RayData(RGBf(0,0,0), 1.0, Inf, 0.0) for _ in 1:nchunks]
    thread_rngs = [copy(rng) for _ in 1:nchunks]
    μ0_dummy = init_photon(cam, spacetime, 0.0, 0.0; rng=rng)
    base_prob = ODEProblem(spacetime, μ0_dummy, tspan, (spacetime, thread_metas[1], disc))
    thread_integrators = [init(base_prob, solver, callback=cb_for(r_max),
                               dense=false, save_everystep=false,
                               reltol=1e-6, abstol=1e-6,
                               maxiters=50_000, verbose=false) for _ in 1:nchunks]

    inv_samples2 = 1.0 / samples^2
    subpixel_offsets = jittered ? jittered_grid(samples; rng=rng) :
                       [(du, dv) for du in range(0.5/samples, 1.0, samples),
                        dv in range(0.5/samples, 1.0, samples)]

    progress_counter = Threads.Atomic{Int}(0)
    progress_interval = max(1, width ÷ 100)

    _foreach_column_chunk(width, nchunks) do ci, cols
        integrator = thread_integrators[ci]
        meta = thread_metas[ci]
        local_rng = thread_rngs[ci]
        for i in cols
            for j in 1:height
                pixel_color = RGBf(0, 0, 0)
                for (du, dv) in subpixel_offsets
                    u, v = sensor_coordinate(i, j, width, height; du=du, dv=dv)
                    pixel_color += _trace_color(integrator, meta, cam, spacetime,
                                                background, disc, u, v;
                                                rng=local_rng, dust=dust)
                end
                image[i, j] = pixel_color * inv_samples2
            end
            c = Threads.atomic_add!(progress_counter, 1)
            if !isnothing(progress) && c % progress_interval == 0
                progress(c / width)
            end
        end
    end
    image
end

"""
    render_no_doppler(cam::AbstractCamera, spacetime::AbstractSpacetime;
                      width=200, height=100, solver=Tsit5(),
                      samples=1, jittered=true, rng=Random.default_rng())

Render a grayscale image without Doppler effects.
"""
function render_no_doppler(cam::AbstractCamera, spacetime::AbstractSpacetime;
                           width::Int=200, height::Int=100,
                           solver=Tsit5(),
                           samples::Int=1,
                           jittered::Bool=true,
                           rng::Random.AbstractRNG=Random.default_rng())
    image = zeros(Float64, width, height)
    cam_dist = norm(cam.pos)
    r_max = max(5.0 * cam_dist, 100.0)
    tspan = (0.0, max(10.0 * cam_dist, 500.0))
    cb = ContinuousCallback(make_boundary_condition(r_max), horizon_affect!)

    nchunks = _render_nchunks()
    thread_metas = [RayData(RGBf(0,0,0), 1.0, Inf, 0.0) for _ in 1:nchunks]
    thread_rngs = [copy(rng) for _ in 1:nchunks]
    μ0_dummy = init_photon(cam, spacetime, 0.0, 0.0; rng=rng)
    base_prob = ODEProblem(spacetime, μ0_dummy, tspan, (spacetime, thread_metas[1]))
    thread_integrators = [init(base_prob, solver, callback=cb,
                               save_everystep=false, dense=false,
                               reltol=1e-6, abstol=1e-6,
                               maxiters=50_000, verbose=false) for _ in 1:nchunks]

    inv_samples2 = 1.0 / samples^2
    subpixel_offsets = jittered ? jittered_grid(samples; rng=rng) :
                       [(du, dv) for du in range(0.5/samples, 1.0, samples),
                        dv in range(0.5/samples, 1.0, samples)]

    _foreach_column_chunk(width, nchunks) do ci, cols
        integrator = thread_integrators[ci]
        meta = thread_metas[ci]
        local_rng = thread_rngs[ci]
        for i in cols
            for j in 1:height
                pixel_value = 0.0
                for (du, dv) in subpixel_offsets
                    u, v = sensor_coordinate(i, j, width, height; du=du, dv=dv)
                    pixel_value += _trace_grayscale(integrator, meta, cam, spacetime,
                                                      u, v; rng=local_rng)
                end
                image[i, j] = pixel_value * inv_samples2
            end
        end
    end
    image
end

"""
    render_motion(camera_at, t0, t1, spacetime, background;
                  disc=AccretionDisc(), width=400, height=200,
                  solver=Tsit5(), samples=1, time_samples=8,
                  jittered=true, rng=Random.default_rng())

Render with motion blur by averaging rays over the shutter interval `[t0, t1]`.
`camera_at(t)` must return an `AbstractCamera` for coordinate time `t`.
"""
function render_motion(camera_at::Function, t0::Real, t1::Real,
                       spacetime::Schwarzschild, background;
                       disc::Union{AccretionDisc,Nothing}=AccretionDisc(),
                       volume::Union{DiscVolume,Nothing}=nothing,
                       width::Int=400, height::Int=200,
                       solver=Tsit5(),
                       samples::Int=1,
                       time_samples::Int=8,
                       jittered::Bool=true,
                       rng::Random.AbstractRNG=Random.default_rng())
    image = zeros(RGBf, width, height)
    inv_total = 1.0 / (samples^2 * time_samples)
    subpixel_offsets = jittered ? jittered_grid(samples; rng=rng) :
                       [(du, dv) for du in range(0.5/samples, 1.0, samples),
                        dv in range(0.5/samples, 1.0, samples)]

    # Each time sample gets its own set of integrators and metas.
    nchunks = _render_nchunks()

    # Pre-allocate per (time_sample, chunk) integrators. We use a serial outer
    # loop over time samples so that each time slice reuses its integrators.
    for ti in 1:time_samples
        t_shutter = t0 + (ti - rand(rng)) / time_samples * (t1 - t0)
        cam = camera_at(t_shutter)
        cam_dist = norm(cam.pos)
        rm = max(5.0 * cam_dist, 100.0)
        tspan = (0.0, max(10.0 * cam_dist, 500.0))

        thread_metas = [RayData(RGBf(0,0,0), 1.0, Inf, 0.0) for _ in 1:nchunks]
        thread_rngs = [copy(rng) for _ in 1:nchunks]
        μ0_dummy = init_photon(cam, spacetime, 0.0, 0.0; rng=rng)
        cb = isnothing(volume) ? make_cb_set(rm, disc) :
            CallbackSet(ContinuousCallback(make_boundary_condition(rm), horizon_affect!),
                        make_volume_cb(volume, disc))
        base_prob = ODEProblem(spacetime, μ0_dummy, tspan, (spacetime, thread_metas[1], disc))
        thread_integrators = [init(base_prob, solver, callback=cb,
                                   dense=false, save_everystep=false,
                                   reltol=1e-6, abstol=1e-6,
                                   maxiters=50_000, verbose=false) for _ in 1:nchunks]

        _foreach_column_chunk(width, nchunks) do ci, cols
            integrator = thread_integrators[ci]
            meta = thread_metas[ci]
            local_rng = thread_rngs[ci]
            for i in cols
                for j in 1:height
                    for (du, dv) in subpixel_offsets
                        u, v = sensor_coordinate(i, j, width, height; du=du, dv=dv)
                        image[i, j] += _trace_color(integrator, meta, cam, spacetime,
                                                    background, disc, u, v; rng=local_rng)
                    end
                end
            end
        end
    end

    image .*= inv_total
    return image
end

# -----------------------------------------------------------------------------
# WorldLine / single-ray tracing
# -----------------------------------------------------------------------------

"""
    WorldLine

Wraps the result of a ray trace as a sequence of 8-component state vectors
sampled at corresponding coordinate times.
"""
struct WorldLine
    t::Vector{Float64}
    μ::Vector{SVector{8, Float64}}
end

"""
    raytrace(spacetime, photon; tspan=(0.0, 500.0), npoints=1000, solver=Tsit5())

Trace a photon through `spacetime` and return a `WorldLine`.
"""
function raytrace(spacetime::AbstractSpacetime, photon::Photon;
                  tspan::Tuple{Float64,Float64}=(0.0, 500.0), npoints::Int=1000, solver=Tsit5())
    r_max = 2.0 * photon.μ[2]
    cb = ContinuousCallback(make_boundary_condition(r_max), horizon_affect!)
    prob = ODEProblem(spacetime, photon.μ, tspan, (spacetime, nothing))
    sol = solve(prob, solver, callback=cb, reltol=1e-6, abstol=1e-6)
    t_end = sol.t[end]
    t_smooth = range(tspan[1], t_end, length=npoints)
    sol_smooth = sol(t_smooth)
    WorldLine(collect(t_smooth), collect(sol_smooth.u))
end

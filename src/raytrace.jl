"""
    Photon

Represents a photon in the spacetime, encapsulating its position and momentum
as an 8-component state vector in Cartesian Kerr–Schild coordinates. The first
four components represent the coordinates (t, x, y, z), while the last four
represent the corresponding momenta (p_t, px, py, pz).
"""
struct Photon
    μ::SVector{8, Float64}  # (t, x, y, z, p_t, px, py, pz)

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
    # Camera/infinity frequency ratio for relativistic shading; 1 shades to
    # the observer at infinity (the default and historical behaviour).
    gcam::Float64
end
RayData(c, a, r, p) = RayData(c, a, r, p, 1.0)

"""
Blackbody-locus tint of a background star for camera/infinity shift `g`:
a ~5800 K source observed at `g`·5800 K, per-channel Planck ratios at
610/550/465 nm (matches the Metal kernel).
"""
function _relativistic_sky_tint(c::RGB, g::Float64)
    g == 1.0 && return RGBf(c)
    return RGBf(red(c) * 57.4 / (exp(4.067 / g) - 1.0),
                green(c) * 90.2 / (exp(4.513 / g) - 1.0),
                blue(c) * 206.5 / (exp(5.335 / g) - 1.0))
end

"""
    init_photon(origin, direction, spacetime::AbstractSpacetime)

Build a photon state vector from a ray `origin` and unit `direction` in
Cartesian Kerr–Schild coordinates `(t, x, y, z, p_t, px, py, pz)`. The
observer tetrad is [`ks_camera_tetrad`](@ref) — a static observer where one
exists, a radial free-faller inside r = 2.5M — and the null momentum comes
from [`ks_init_photon`](@ref): the ray is traced *backward* in time
(`q = n − u`), matching the GPU kernel, so sensor angles are proper angles in
the observer's frame and the conserved `p_t` carries the full frequency shift.
"""
function init_photon(origin::SVector{3,Float64}, direction::SVector{3,Float64},
                     spacetime::AbstractSpacetime)
    # Any flat-orthonormal pair completing `direction`: ks_init_photon only
    # uses them to decompose a direction that is entirely along `fwd`.
    a = abs(direction[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)
    right = normalize(cross(direction, a))
    up = cross(right, direction)

    tet = ks_camera_tetrad(origin, direction, right, up, spacetime.M)
    μ6, p_t = ks_init_photon(origin, direction, spacetime.M, tet,
                             direction, right, up)
    return SVector{8,Float64}(0.0, μ6[1], μ6[2], μ6[3],
                              p_t, μ6[4], μ6[5], μ6[6])
end

"""
    init_photon(cam::AbstractCamera, spacetime::AbstractSpacetime, u, v; rng=Random.default_rng())

Convenience wrapper that samples a ray from `cam` at normalised sensor
coordinates `(u, v)` and converts it to a photon state vector.
"""
function init_photon(cam::AbstractCamera, spacetime::AbstractSpacetime, u, v;
                     rng::Random.AbstractRNG=Random.default_rng(),
                     lens::Union{Nothing, NTuple{2, Float64}}=nothing)
    origin, direction = lens === nothing ? get_ray(cam, u, v, rng) :
                        get_ray(cam, u, v, rng, lens)
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
                      dust::Union{InterstellarDust,Nothing}=nothing,
                      lens::Union{Nothing, NTuple{2, Float64}}=nothing,
                      relativistic::Bool=false)
    μ0 = init_photon(cam, spacetime, u, v; rng, lens)

    meta.acc_color = RGBf(0, 0, 0)
    meta.alpha = 1.0
    meta.r_min = Inf
    meta.path_length = 0.0
    # Camera/infinity frequency ratio, per ray. The photon leaves the sensor at
    # unit frequency in the observer tetrad and `p_t` is conserved, so the shift
    # this ray carries is exactly 1/|p_t| — the same expression the Metal kernel
    # uses, with the same clamp.
    #
    # This used to be one number for the whole image, `1/√(1 − 2M/r)`. Outside
    # r = 2.5M that is not an approximation but the exact answer: the tetrad is
    # a static observer, the shift is purely gravitational, and it does not
    # depend on which way the ray left. Measured spread across a frame is 0.0%
    # at 20M, 6M and 3M, agreeing with the old scalar to four figures.
    #
    # Inside r = 2.5M the tetrad becomes a radial free-faller and the two part
    # company, because the infall gives the shift a direction dependence:
    #
    #   camera r    per-ray 1/|p_t|      old uniform scalar
    #     2.4M      3.466 .. 3.779       2.449
    #     2.1M      8.486 .. 17.589      4.583
    #
    # — 70% spread across one frame at 2.1M, against a single number wrong by
    # nearly 2x. That is the porthole and the horizon crossing, so this is the
    # regime the fix is for. It will also carry a boosted camera correctly if
    # the CPU ever gains one; today `init_photon` builds an unboosted tetrad,
    # so a moving CPU camera still shades as though it were at rest.
    meta.gcam = relativistic ? 1.0 / clamp(abs(μ0[5]), 0.05, 20.0) : 1.0
    integrator.p = (spacetime, meta, disc)
    reinit!(integrator, μ0)
    solve!(integrator)

    μf = integrator.sol.u[end]
    xf, yf, zf = μf[2], μf[3], μf[4]
    final_r = sqrt(xf^2 + yf^2 + zf^2)
    final_θ = acos(clamp(zf / final_r, -1.0, 1.0))
    final_ϕ = atan(yf, xf)

    # Compute approximate path length for dust extinction.
    x0, y0, z0 = μ0[2], μ0[3], μ0[4]
    r_start = sqrt(x0^2 + y0^2 + z0^2)
    path_len = abs(r_start - final_r)  # radial component
    # Add transverse component for non-radial rays.
    cosΔψ = (x0 * xf + y0 * yf + z0 * zf) / (r_start * final_r)
    path_len += r_start * acos(clamp(cosΔψ, -1.0, 1.0)) * 0.5  # approximate
    meta.path_length = path_len

    color = if final_r < 2.1 * spacetime.M
        meta.acc_color
    else
        bg_color = _relativistic_sky_tint(
            sample_background(background, final_θ, final_ϕ), meta.gcam)
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

    μf = integrator.sol.u[end]
    final_r = sqrt(μf[2]^2 + μf[3]^2 + μf[4]^2)
    return final_r < 2.1 * spacetime.M ? 0.0 : 0.5 + 0.5 * sin(10 * atan(μf[3], μf[2]))
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
                progress::Union{Function,Nothing}=nothing,
                relativistic::Bool=false)
    image = zeros(RGBf, width, height)
    cam_dist = norm(cam.pos)
    r_max = max(5.0 * cam_dist, 100.0)
    tspan = (0.0, max(10.0 * cam_dist, 500.0))
    # A volumetric disc replaces the thin-plane crossing callback.
    cb_for(rm) = isnothing(volume) ? make_cb_set(rm, disc) :
        CallbackSet(ContinuousCallback(make_boundary_condition(rm), horizon_affect!),
                    make_volume_cb(volume, disc))

    nchunks = _render_nchunks()
    # gcam is set per ray in `_trace_color`; this is only the initial value.
    thread_metas = [RayData(RGBf(0,0,0), 1.0, Inf, 0.0, 1.0) for _ in 1:nchunks]
    # Independently seeded per-chunk RNGs. `copy(rng)` would give every chunk
    # the same stream, tiling one noise pattern across all chunks.
    thread_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nchunks]
    μ0_dummy = init_photon(cam, spacetime, 0.0, 0.0; rng=rng)
    base_prob = ODEProblem(spacetime, μ0_dummy, tspan, (spacetime, thread_metas[1], disc))
    thread_integrators = [init(base_prob, solver, callback=cb_for(r_max),
                               dense=false, save_everystep=false,
                               reltol=1e-6, abstol=1e-6,
                               maxiters=50_000, verbose=false) for _ in 1:nchunks]

    n_sub = samples^2
    inv_samples2 = 1.0 / n_sub
    fixed_offsets = [(du, dv) for du in range(0.5/samples, 1.0, samples)
                     for dv in range(0.5/samples, 1.0, samples)]
    use_lens = cam isa ThinLensCamera

    progress_counter = Threads.Atomic{Int}(0)
    progress_interval = max(1, width ÷ 100)

    _foreach_column_chunk(width, nchunks) do ci, cols
        integrator = thread_integrators[ci]
        meta = thread_metas[ci]
        local_rng = thread_rngs[ci]
        pix_offs = Vector{NTuple{2, Float64}}(undef, n_sub)
        lens_offs = Vector{NTuple{2, Float64}}(undef, n_sub)
        for i in cols
            for j in 1:height
                # Fresh stratified jitter per pixel; a single grid reused for
                # every pixel correlates the sampling image-wide.
                jittered ? jittered_grid!(pix_offs, samples, local_rng) :
                           copyto!(pix_offs, fixed_offsets)
                if use_lens
                    # Stratified aperture samples (concentric-mapped in
                    # `get_ray`), shuffled so lens strata pair randomly with
                    # subpixel strata.
                    jittered_grid!(lens_offs, samples, local_rng)
                    Random.shuffle!(local_rng, lens_offs)
                end
                pixel_color = RGBf(0, 0, 0)
                for k in 1:n_sub
                    du, dv = pix_offs[k]
                    u, v = sensor_coordinate(i, j, width, height; du=du, dv=dv)
                    pixel_color += _trace_color(integrator, meta, cam, spacetime,
                                                background, disc, u, v;
                                                rng=local_rng, dust=dust,
                                                lens=use_lens ? lens_offs[k] : nothing,
                                                relativistic=relativistic)
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
    thread_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nchunks]
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
                       rng::Random.AbstractRNG=Random.default_rng(),
                       relativistic::Bool=false)
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
        thread_rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nchunks]
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
                                                    background, disc, u, v; rng=local_rng,
                                                    relativistic=relativistic)
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
    r_max = 2.0 * sqrt(photon.μ[2]^2 + photon.μ[3]^2 + photon.μ[4]^2)
    cb = ContinuousCallback(make_boundary_condition(r_max), horizon_affect!)
    prob = ODEProblem(spacetime, photon.μ, tspan, (spacetime, nothing))
    sol = solve(prob, solver, callback=cb, reltol=1e-6, abstol=1e-6)
    t_end = sol.t[end]
    t_smooth = range(tspan[1], t_end, length=npoints)
    sol_smooth = sol(t_smooth)
    WorldLine(collect(t_smooth), collect(sol_smooth.u))
end

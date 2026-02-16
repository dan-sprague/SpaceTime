"""
    Photon

Represents a photon in the spacetime, encapsulating its position and momentum as an 8-component state vector. The first four components represent the coordinates (t, r, θ, ϕ), while the last four components represent the corresponding momenta (pt, pr, pθ, pϕ). The `Photon` struct provides constructors for initializing the state vector from either a single 8-component vector or separate position and momentum vectors.
"""
struct Photon
    μ::SVector{8, Float64}  # (t, r, θ, ϕ, pt, pr, pθ, pϕ)

    Photon(μ::SVector{8, Float64}) = new(μ)
    Photon(q::SVector{4, Float64}, p::SVector{4, Float64}) = new(vcat(q, p))
end

"""
    Camera(pos, target, up, fov_factor)

Represents a camera in the spacetime, defined by its position `pos`, the point it is looking at `target`, an up vector `up` to define the orientation, and a field of view factor `fov_factor` that controls the width of the viewing frustum. The constructor calculates the forward, right, and local up vectors based on the input parameters to establish the camera's coordinate system.
"""
struct Camera
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

"""
    RayDat
    
A mutable struct to hold metadata for each ray during the integration process. It contains the accumulated color (`acc_color`), the alpha value for blending (`alpha`), and the minimum radius encountered along the ray's path (`r_min`). This struct is used to store intermediate results and state information as the photon is traced through the spacetime and interacts with various elements such as the accretion disc and the background.

"""
mutable struct RayData
    acc_color::RGBf
    alpha::Float64
    r_min::Float64
end



"""

    get_ray_direction(cam::Camera, u, v)

Calculates the direction of a ray in world coordinates based on the camera's orientation and the normalized screen coordinates `u` and `v`. The function constructs a local direction vector in the camera's coordinate system and then transforms it into world coordinates using the camera's right, up, and forward vectors. The resulting direction vector is normalized to ensure it has a unit length.
"""
function get_ray_direction(cam::Camera, u, v)
    dir_local = SVector{3}(u * cam.fov_factor, v * cam.fov_factor, 1.0)

    dir_world = normalize(
        cam.right * dir_local[1] +
        cam.up_local * dir_local[2] +
        cam.fwd * dir_local[3]
    )

    return dir_world
end


"""
    init_photon(cam::Camera, spacetime::AbstractSpacetime, u, v)
Initializes a photon's state vector based on the camera's position and orientation, as well as the normalized screen coordinates `u` and `v`. The function calculates the initial position of the photon in spherical coordinates (t, r, θ, ϕ) and computes the corresponding momentum components (pt, pr, pθ, pϕ) using the inverse metric of the spacetime. The resulting state vector is returned as an instance of the `Photon` struct, ready to be used for ray tracing through the spacetime.

"""
function init_photon(cam::Camera, spacetime::AbstractSpacetime, u, v)
    d = get_ray_direction(cam, u, v)
    dx, dy, dz = d

    x, y, z = cam.pos
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = atan(y, x)
    q0 = @SVector [0.0, r, θ, ϕ]

    # Projections
    vr = dx * sin(θ) * cos(ϕ) + dy * sin(θ) * sin(ϕ) + dz * cos(θ)
    vθ = (dx * cos(θ) * cos(ϕ) + dy * cos(θ) * sin(ϕ) - dz * sin(θ)) / r
    vϕ = (-dx * sin(ϕ) + dy * cos(ϕ)) / (r * sin(θ))

    g_inv = metric_inverse(spacetime, q0)

    # Momentum components
    pr = vr / g_inv[2,2]
    pθ = vθ / g_inv[3,3]
    pϕ = vϕ / g_inv[4,4]

    # Null condition H=0
    spatial_part = g_inv[2,2]*pr^2 + g_inv[3,3]*pθ^2 + g_inv[4,4]*pϕ^2
    pt = -sqrt(abs(spatial_part / g_inv[1,1]))

    return vcat(q0, SVector(pt, pr, pθ, pϕ))
end

"""
    sample_background(img, θ, ϕ)

Samples the background image `img` based on the spherical coordinates `θ` and `ϕ`. The function converts the spherical coordinates into normalized texture coordinates `u` and `v`, which are then used to index into the background image. The resulting color is returned as an RGB value, allowing the ray tracer to incorporate the background into the final rendered image based on the photon's trajectory and interactions with the spacetime and accretion disc.

"""
function sample_background(img, θ, ϕ)
    v = clamp(θ / π, 0.0, 1.0)
    u = clamp(mod2pi(ϕ) / (2π), 0.0, 1.0)
    
    w, h = size(img)
    px = round(Int, u * (w - 1)) + 1
    py = round(Int, v * (h - 1)) + 1
    
    return img[px, py]
end

"""
    render_no_doppler(cam::Camera, spacetime::AbstractSpacetime; width=200, height=100)

Renders an image of the spacetime without considering Doppler effects. The function initializes a 2D array to hold the pixel values and sets up ODE integrators for each thread to trace photons through the spacetime. For each pixel, it calculates the initial state of the photon based on the camera's position and orientation, and then integrates its trajectory until it either hits the black hole or escapes to infinity. The resulting image is a grayscale representation of the spacetime, where pixels corresponding to rays that hit the black hole are set to black, and others are colored based on their final azimuthal angle.

"""
function render_no_doppler(cam::Camera, spacetime::AbstractSpacetime; width=200, height=100)
    image = zeros(Float64, width, height)
    tspan = (0.0, 500.0)
    cb = ContinuousCallback(boundary_condition, horizon_affect!)

    maxtid = Threads.maxthreadid()
    thread_metas = [RayData(RGBf(0,0,0), 1.0, Inf) for _ in 1:maxtid]
    μ0_dummy = init_photon(cam, spacetime, 0.0, 0.0)
    base_prob = ODEProblem(spacetime, μ0_dummy, tspan, (spacetime, thread_metas[1]))
    thread_integrators = [init(base_prob, Tsit5(), callback=cb,
                               save_everystep=false, dense=false,
                               reltol=1e-6, abstol=1e-6) for _ in 1:maxtid]

    Threads.@threads :static for i in 1:width
        tid = Threads.threadid()
        integrator = thread_integrators[tid]
        meta = thread_metas[tid]
        for j in 1:height
            u = (i - width/2) / (width / 2)
            v = (j - height/2) / (height / 2)
            μ0 = init_photon(cam, spacetime, u, v)

            meta.acc_color = RGBf(0,0,0)
            meta.alpha = 1.0
            meta.r_min = Inf
            integrator.p = (spacetime, meta)
            reinit!(integrator, μ0)
            solve!(integrator)

            final_r = integrator.sol.u[end][2]
            if final_r < 2.1 * spacetime.M
                image[i, j] = 0.0
            else
                final_ϕ = integrator.sol.u[end][4]
                image[i, j] = 0.5 + 0.5 * sin(10 * final_ϕ)
            end
        end
    end
    image
end

"""
    render(cam::Camera, spacetime::T, background; disc::AccretionDisc=AccretionDisc(), width=400, height=200)

Renders an image of the spacetime with Doppler effects and an accretion disc. The function initializes a 2D array to hold the pixel values and sets up ODE integrators for each thread to trace photons through the spacetime. For each pixel, it calculates the initial state of the photon based on the camera's position and orientation, and then integrates its trajectory while accounting for interactions with the accretion disc and the background. The resulting image is a color representation of the spacetime, where pixels are colored based on their interactions with the disc and background, as well as their final positions.

"""
function render(cam::Camera, spacetime::Schwarzschild, background; disc::AccretionDisc=AccretionDisc(), width=400, height=200)
    image = zeros(RGBf, width, height)
    tspan = (0.0, 500.0)

    maxtid = Threads.maxthreadid()
    thread_metas = [RayData(RGBf(0,0,0), 1.0, Inf) for _ in 1:maxtid]
    μ0_dummy = init_photon(cam, spacetime, 0.0, 0.0)
    base_prob = ODEProblem(spacetime, μ0_dummy, tspan, (spacetime, thread_metas[1], disc))
    thread_integrators = [init(base_prob, Tsit5(), callback=cb_set,
                               dense=false, save_everystep=false,
                               reltol=1e-6, abstol=1e-6) for _ in 1:maxtid]

    Threads.@threads :static for i in 1:width
        tid = Threads.threadid()
        integrator = thread_integrators[tid]
        meta = thread_metas[tid]
        for j in 1:height
            u = (i - width/2) / (height/2)
            v = (j - height/2) / (height/2)
            μ0 = init_photon(cam, spacetime, u, v)

            meta.acc_color = RGBf(0,0,0)
            meta.alpha = 1.0
            meta.r_min = Inf
            integrator.p = (spacetime, meta, disc)
            reinit!(integrator, μ0)
            solve!(integrator)

            final_r = integrator.sol.u[end][2]
            final_θ = integrator.sol.u[end][3]
            final_ϕ = integrator.sol.u[end][4]

            if final_r < 2.1 * spacetime.M
                image[i, j] = meta.acc_color
            else
                bg_color = sample_background(background, final_θ, final_ϕ)
                image[i, j] = meta.acc_color + (bg_color * meta.alpha)
            end
        end
    end
    image
end

"""
    WorldLine

Wraps the result of a ray trace as a sequence of 8-component state vectors
(t, r, θ, ϕ, pₜ, pᵣ, pθ, pϕ) sampled at corresponding coordinate times.
"""
struct WorldLine
    t::Vector{Float64}
    μ::Vector{SVector{8, Float64}}
end

"""
    raytrace(spacetime, photon; tspan=(0.0, 500.0), npoints=1000)

Trace a photon through `spacetime` and return a `WorldLine`.
"""
function raytrace(spacetime::AbstractSpacetime, photon::Photon;
                  tspan::Tuple{Float64,Float64}=(0.0, 500.0), npoints::Int=1000)
    cb = ContinuousCallback(boundary_condition, horizon_affect!)
    prob = ODEProblem(spacetime, photon.μ, tspan, (spacetime, nothing))
    sol = solve(prob, Tsit5(), callback=cb, reltol=1e-6, abstol=1e-6)
    t_end = sol.t[end]
    t_smooth = range(tspan[1], t_end, length=npoints)
    sol_smooth = sol(t_smooth)
    WorldLine(collect(t_smooth), collect(sol_smooth.u))
end


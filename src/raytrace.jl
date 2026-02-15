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


mutable struct RayData
    acc_color::RGBf
    alpha::Float64
    r_min::Float64
end

function get_ray_direction(cam::Camera, u, v)
    dir_local = SVector{3}(u * cam.fov_factor, v * cam.fov_factor, 1.0)

    dir_world = normalize(
        cam.right * dir_local[1] +
        cam.up_local * dir_local[2] +
        cam.fwd * dir_local[3]
    )

    return dir_world
end

function init_photon(cam::Camera, bh::BlackHole, u, v)
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

    g_inv = metric_inverse(bh, q0)

    # Momentum components
    pr = vr / g_inv[2,2]
    pθ = vθ / g_inv[3,3]
    pϕ = vϕ / g_inv[4,4]

    # Null condition H=0
    spatial_part = g_inv[2,2]*pr^2 + g_inv[3,3]*pθ^2 + g_inv[4,4]*pϕ^2
    pt = -sqrt(abs(spatial_part / g_inv[1,1]))

    return vcat(q0, SVector(pt, pr, pθ, pϕ))
end

function sample_background(img, θ, ϕ)
    # Normalize angles: θ is usually [0, π], ϕ is [0, 2π]
    # Map to [0, 1] for image indexing
    v = clamp(θ / π, 0.0, 1.0)
    u = clamp(mod2pi(ϕ) / (2π), 0.0, 1.0)
    
    w, h = size(img)
    px = round(Int, u * (w - 1)) + 1
    py = round(Int, v * (h - 1)) + 1
    
    return img[px, py]
end

function render_no_doppler(cam::Camera, bh::BlackHole;width = 200,height = 100)
    image = zeros(Float64,width,height)

    tspan = (0.0,500.0)
    cb = ContinuousCallback(boundary_condition, horizon_affect!)
    meta = RayData(RGBf(0.0,0.0,0.0),1.0,Inf)
    Threads.@threads for i in 1:width
        for j in 1:height
            u = (i - width/2) / (width / 2)
            v = (j - height/2) / (height / 2)

            μ0 = init_photon(cam, bh, u, v)            
            problem = ODEProblem(equations_of_motion, μ0, tspan, (bh, meta))
            cb = ContinuousCallback(boundary_condition, horizon_affect!)
            sol = solve(problem, Tsit5(), callback=cb, save_everystep=false,
            reltol=1e-6, abstol=1e-6,dense = false)

            final_r = sol.u[end][2]
            if final_r < 2.1 * bh.M
                image[i, j] = 0.0
            else
                final_ϕ = sol.u[end][4]
                image[i,j] = 0.5 + 0.5 * sin(10 * final_ϕ)
            end
        end
    end
    image 
end

function render(cam::Camera, bh::BlackHole,background; width=400, height=200)
    image = zeros(RGBf, width, height)
    tspan = (0.0, 500.0)
    
    Threads.@threads for i in 1:width
        for j in 1:height
            meta = RayData(RGBf(0,0,0), 1.0, Inf)
            
            u = (i - width/2) / (height/2)
            v = (j - height/2) / (height/2)
            μ0 = init_photon(cam, bh, u, v)
            
            prob = ODEProblem(equations_of_motion, μ0, tspan, (bh, meta))
            
            sol = solve(prob, Tsit5(), callback=cb_set, 
            dense=false, save_everystep=false, 
            reltol=1e-6, abstol=1e-6)

            final_r = sol.u[end][2]
            final_θ = sol.u[end][3]
            final_ϕ = sol.u[end][4]
            
            if final_r < 2.1 * bh.M
                image[i, j] = meta.acc_color
            else
                bg_color = sample_background(background, final_θ, final_ϕ)
                image[i, j] = meta.acc_color + (bg_color * meta.alpha)
            end
        end
    end
    image 
end



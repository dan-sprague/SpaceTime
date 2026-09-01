using SpaceTime
using StaticArrays
using LinearAlgebra: norm, dot, cross, normalize, Diagonal
using Random
using Colors
using Test

const RGBf = RGB{Float32}

bh = Schwarzschild(1.0)
cam = Camera(SVector(0.0, -10.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))

@testset "Camera construction" begin
    @test cam isa Camera
    @test cam isa AbstractCamera
    @test ThinLensCamera(SVector(0.0,0.0,0.0), SVector(1.0,0.0,0.0), SVector(0.0,0.0,1.0)) isa AbstractCamera
end

@testset "Camera transforms" begin
    c2 = yaw(cam, 10.0)
    @test c2 isa Camera
    c3 = pitch(cam, 5.0)
    @test c3 isa Camera
    c4 = roll(cam, 3.0)
    @test c4 isa Camera
    c5 = dolly(cam, 1.0)
    @test norm(c5.pos - cam.pos - cam.fwd) < 1e-10
end

@testset "Sampling helpers" begin
    offsets = jittered_grid(2)
    @test length(offsets) == 4
    @test all(o -> 0.0 <= o[1] <= 1.0 && 0.0 <= o[2] <= 1.0, offsets)

    u, v = sensor_coordinate(1, 1, 10, 10)
    @test u < 0 && v < 0
    u, v = sensor_coordinate(10, 10, 10, 10)
    @test u > 0 && v > 0
end

@testset "Render no-doppler" begin
    img = render_no_doppler(cam, bh; width=20, height=10, samples=1)
    @test size(img) == (20, 10)
    @test all(x -> 0.0 <= x <= 1.0, img)

    tl = ThinLensCamera(SVector(0.0, -10.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))
    img2 = render_no_doppler(tl, bh; width=20, height=10, samples=2)
    @test size(img2) == (20, 10)
end

@testset "Sensor model" begin
    img = zeros(RGBf, 10, 10)
    img[5, 5] = RGBf(1.0, 0.5, 0.2)
    sensor_expose!(img; iso=100.0, t_exp=1.0, read_noise_e=0.0, add_noise=false)
    @test img[5, 5] == RGBf(1.0, 0.5, 0.2)

    img2 = fill(0.5, 10, 10)
    sensor_expose!(img2; iso=200.0, t_exp=2.0, add_noise=false)
    @test img2[1, 1] ≈ 0.5 * 4.0
end

# The camera tetrad in Kerr–Schild coordinates, optionally Lorentz-boosted by
# the camera's 3-velocity `beta`. These tests are executable statements of the
# theory: the tetrad must be orthonormal under the KS metric (g(e_a, e_b) =
# η_ab), and in flat space the per-ray frequency shift must reproduce the
# exact relativistic Doppler factor γ(1 + β).
@testset "Boosted camera tetrad" begin
    ks_metric(pos, M) = begin
        r = norm(pos)
        l = SVector(1.0, (pos / r)...)
        Matrix(Diagonal(SVector(-1.0, 1.0, 1.0, 1.0))) + (2M / r) * (l * l')
    end
    η = Diagonal([-1.0, 1.0, 1.0, 1.0])
    for (r, β) in [(30.0, SVector(0.5, 0.0, 0.0)),
                   (5.0, SVector(-0.6, 0.2, 0.1)),
                   (2.6, SVector(0.3, -0.3, 0.2)),
                   (100.0, SVector(0.0, 0.0, 0.9))]
        pos = r * normalize(SVector(1.0, 0.3, -0.2))
        fwd = normalize(-pos)
        right = normalize(cross(fwd, SVector(0.0, 0.0, 1.0)))
        upl = cross(right, fwd)
        u, Ef, Er, Eu = SpaceTime.ks_camera_tetrad(pos, fwd, right, upl, 1.0; beta=β)
        T = [u Ef Er Eu]
        @test maximum(abs.(T' * ks_metric(pos, 1.0) * T - η)) < 1.0e-12
    end

    # Flat-space Doppler through the kernel's ray-initialisation algebra.
    pos = SVector(1.0e6, 0.0, 0.0)
    fwd = SVector(-1.0, 0.0, 0.0)
    right = SVector(0.0, -1.0, 0.0)
    upl = SVector(0.0, 0.0, 1.0)
    for βf in (0.5, -0.5)
        u, Ef, _, _ = SpaceTime.ks_camera_tetrad(pos, fwd, right, upl, 1.0;
                                                 beta=SVector(βf, 0.0, 0.0))
        q = Ef - u                       # backward-traced centre-pixel ray
        x̂ = pos / norm(pos)
        p_t = -q[1] + (2.0 / norm(pos)) * (q[1] + dot(x̂, q[2:4]))
        γ = 1.0 / sqrt(1.0 - βf^2)
        @test 1.0 / abs(p_t) ≈ γ * (1.0 + βf) atol = 1.0e-5
    end

    # beta = 0 must reproduce the unboosted tetrad exactly.
    pos = SVector(10.0, 1.0, 0.5)
    fwd = normalize(-pos)
    right = normalize(cross(fwd, SVector(0.0, 0.0, 1.0)))
    upl = cross(right, fwd)
    t0 = SpaceTime.ks_camera_tetrad(pos, fwd, right, upl, 1.0)
    t1 = SpaceTime.ks_camera_tetrad(pos, fwd, right, upl, 1.0;
                                    beta=SVector(0.0, 0.0, 0.0))
    @test all(map((a, b) -> a == b, t0, t1))
end

@testset "Shadow radius" begin
    bh1 = Schwarzschild(1.0)
    # Far away, the shadow's angular radius approaches the critical impact
    # parameter over distance: α ≈ 3√3 M / r.
    far = Camera(SVector(0.0, -1.0e6, 0.0), SVector(0.0, 1.0, 0.0),
                 SVector(0.0, 0.0, 1.0))
    @test shadow_radius(far, bh1) * far.fov_factor ≈ 3.0 * sqrt(3.0) / 1.0e6 rtol = 1.0e-6
    # Closer than r = 3√3 M every rearward escape direction is cut off: the
    # shadow wraps the whole sky.
    near = Camera(SVector(0.0, -5.0, 0.0), SVector(0.0, 1.0, 0.0),
                  SVector(0.0, 0.0, 1.0))
    @test shadow_radius(near, bh1) == Inf
end

@testset "Post-processing effects" begin
    img = fill(RGBf(1.0, 1.0, 1.0), 20, 20)
    apply_vignette!(img; strength=0.5)
    @test img[10, 10] ≈ RGBf(1.0, 1.0, 1.0) atol=1e-2  # center nearly unaffected
    @test img[1, 1].r < 1.0

    img2 = fill(0.5, 20, 20)
    apply_lens_distortion!(img2; k1=0.0)
    @test size(img2) == (20, 20)
end

@testset "Resolution-independent post" begin
    # Bloom radius and streak width are given in pixels, so the same numbers
    # produce a tighter look on a larger frame unless `ref_height` rescales
    # them. Measure the glow around a lone bright pixel as a fraction of frame
    # height and require the two resolutions to agree.
    function glow_fraction(w, h; kwargs...)
        img = fill(RGBf(0, 0, 0), w, h)
        img[w ÷ 2, h ÷ 2] = RGBf(50, 50, 50)
        p = postprocess(img; gain=1.0, exposure=0.0, gamma=1.0,
                        bloom_strength=1.0, threshold=0.5, bloom_radius=10.0,
                        bloom_power=1.5, streak_strength=0.0, tonemap=:none,
                        kwargs...)
        cx, cy = w ÷ 2, h ÷ 2
        prof = [Float64(p[cx + k, cy].r) for k in 1:min(cx, cy) - 2]
        r = findfirst(<(0.05 * prof[1]), prof)
        return (r === nothing ? length(prof) : r) / h
    end

    small = glow_fraction(320, 180)
    big_unscaled = glow_fraction(960, 540)
    big_scaled = glow_fraction(960, 540; ref_height=180)

    # Unscaled, tripling the frame shrinks the glow to about a third of it.
    @test big_unscaled < 0.5 * small
    # Scaled, the glow covers the same fraction of the frame at both sizes.
    @test big_scaled ≈ small rtol = 0.15

    # Lens-plane effects take the same treatment: a speck of dust covers a
    # fixed fraction of the frame regardless of the sensor behind it.
    darkness(img) = 1.0 - sum(Float64(c.r) for c in img) / length(img)
    dust = LensDust(count=12, size_min=3.0, size_max=3.0,
                    opacity_min=0.8, opacity_max=0.8)
    a = fill(RGBf(1, 1, 1), 320, 180)
    b = fill(RGBf(1, 1, 1), 960, 540)
    apply_lens_dust!(a; lens_dust=dust, rng=Xoshiro(1))
    apply_lens_dust!(b; lens_dust=dust, ref_height=180, rng=Xoshiro(1))
    @test darkness(b) ≈ darkness(a) rtol = 0.25
end

@testset "Ship dynamics" begin
    M = 1.0

    # At rest far from the hole: β = 0 against the reference observer, and
    # the conserved energy is the static observer's −p_t = √(1 − 2M/r).
    pos = SVector(10.0, 2.0, 1.0)
    ship = ShipState(pos, M)
    fwd = normalize(-pos)
    right = normalize(cross(fwd, SVector(0.0, 0.0, 1.0)))
    upl = cross(right, fwd)
    β, γ = ship_velocity(ship, M, fwd, right, upl)
    @test norm(β) < 1.0e-12
    @test γ ≈ 1.0 atol = 1.0e-12
    @test -ship.p_t ≈ sqrt(1.0 - 2.0 / norm(pos)) atol = 1.0e-12

    # Free fall is a geodesic: a circular orbit at r = 8M closes on itself,
    # holds its radius, and conserves p_t to machine precision.
    r0 = 8.0
    Ω = sqrt(M / r0^3)
    ut = 1.0 / sqrt(1.0 - 3.0 * M / r0)
    x0 = SVector(r0, 0.0, 0.0)
    u = SVector(ut, 0.0, r0 * Ω * ut, 0.0)
    p_t, p = SpaceTime.ks_lower(x0, M, u)
    orb = ShipState(x0, p, p_t, 0.0, 0.0)
    @test SpaceTime.ks_gdot(x0, M, u, u) ≈ -1.0 atol = 1.0e-12
    τ_orbit = 2π / Ω / ut          # one coordinate-time period, in proper time
    pt0 = orb.p_t
    for _ in 1:100
        step_ship!(orb, M, τ_orbit / 100)
    end
    @test norm(orb.x) ≈ r0 atol = 1.0e-6
    @test norm(orb.x - x0) < 1.0e-4          # closed after one full lap
    @test orb.p_t ≈ pt0 atol = 1.0e-10
    @test orb.t ≈ 2π / Ω rtol = 1.0e-4       # coordinate clock: one period

    # Hovering: radial thrust a = (M/r²)/√(1−2M/r) balances gravity. Step
    # with the ship-frame axes recomputed every tick, like the flight loop.
    rh = 6.0
    hov = ShipState(SVector(rh, 0.0, 0.0), M)
    a_hover = (M / rh^2) / sqrt(1.0 - 2.0 * M / rh)
    f_h = SVector(1.0, 0.0, 0.0)             # camera looks radially outward
    r_h = SVector(0.0, 1.0, 0.0)
    u_h = SVector(0.0, 0.0, 1.0)
    for _ in 1:400
        βh, _ = ship_velocity(hov, M, f_h, r_h, u_h)
        tet = SpaceTime.ks_camera_tetrad(hov.x, f_h, r_h, u_h, M; beta=βh)
        step_ship!(hov, M, 0.05; accel=SVector(a_hover, 0.0, 0.0),
                   axes=(tet[2], tet[3], tet[4]))
    end
    @test norm(hov.x) ≈ rh atol = 2.0e-3
    βh, _ = ship_velocity(hov, M, f_h, r_h, u_h)
    @test norm(βh) < 2.0e-3

    # Mass shell after a hard burn: g(u, u) = −1 is maintained, and the
    # velocity decomposition round-trips through the boosted tetrad.
    burn = ShipState(pos, M)
    tet = SpaceTime.ks_camera_tetrad(pos, fwd, right, upl, M)
    step_ship!(burn, M, 2.0; accel=SVector(0.4, 0.1, 0.0),
               axes=(tet[2], tet[3], tet[4]))
    ub = SpaceTime.ks_raise(burn.x, M, burn.p_t, burn.p)
    @test SpaceTime.ks_gdot(burn.x, M, ub, ub) ≈ -1.0 atol = 1.0e-10
    βb, γb = ship_velocity(burn, M, fwd, right, upl)
    tb = SpaceTime.ks_camera_tetrad(burn.x, fwd, right, upl, M; beta=βb)
    @test norm(tb[1] - ub) < 1.0e-8 * γb
    @test γb > 1.1                            # the burn actually moved us
end

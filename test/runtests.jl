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

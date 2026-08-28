using SpaceTime
using StaticArrays
using LinearAlgebra: norm
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

@testset "Post-processing effects" begin
    img = fill(RGBf(1.0, 1.0, 1.0), 20, 20)
    apply_vignette!(img; strength=0.5)
    @test img[10, 10] ≈ RGBf(1.0, 1.0, 1.0) atol=1e-2  # center nearly unaffected
    @test img[1, 1].r < 1.0

    img2 = fill(0.5, 20, 20)
    apply_lens_distortion!(img2; k1=0.0)
    @test size(img2) == (20, 20)
end

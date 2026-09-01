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

@testset "Per-ray frequency shift" begin
    # The camera/infinity shift the CPU applies to shading is 1/|p_t| per ray,
    # the same expression the Metal kernel uses. Outside r = 2.5M the tetrad is
    # a static observer, so it must equal the closed-form gravitational shift
    # exactly and be identical for every ray in the frame — the scalar the CPU
    # used to apply was right there.
    st = Schwarzschild(1.0)
    shifts(R) = begin
        c = Camera(SVector(R, 0.0, 0.0), SVector(0.0, 0.0, 0.0),
                   SVector(0.0, 0.0, 1.0), 0.5)
        [1.0 / clamp(abs(init_photon(c, st, sensor_coordinate(i, j, 24, 14)...)[5]),
                     0.05, 20.0) for i in 1:24, j in 1:14]
    end
    # Agreement is to ~8 figures — the residual is round-trip error through the
    # tetrad construction. Set against the 70% spread inside 2.5M below, that is
    # six orders of magnitude of separation, so the loose bound keeps its teeth.
    for R in (20.0, 6.0, 3.0)
        g = shifts(R)
        @test maximum(g) - minimum(g) < 1.0e-7          # direction independent
        @test g[1] ≈ 1 / sqrt(1 - 2 / R) rtol = 1.0e-7  # and the static value
    end

    # Inside r = 2.5M the tetrad becomes a radial free-faller, the infall gives
    # the shift a direction dependence, and the single scalar stops being
    # correct. This is the porthole and the horizon crossing.
    g = shifts(2.1)
    @test (maximum(g) - minimum(g)) / (sum(g) / length(g)) > 0.5
    @test minimum(g) > 1 / sqrt(1 - 2 / 2.1)   # the old scalar underestimated
end

@testset "Camera velocity" begin
    st = Schwarzschild(1.0)
    pos = SVector(20.0, 0.0, 0.0)
    v = SVector(-0.6, 0.0, 0.0)                    # 0.6c straight at the hole
    c = Camera(pos, SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 1.0), 0.5;
               velocity=v)

    # Velocity is stored in world coordinates and projected onto the camera
    # axes on demand. Forward is -x here, so the whole boost is along forward.
    @test c.velocity == v
    @test SpaceTime.camera_beta(c) ≈ SVector(0.6, 0.0, 0.0) atol = 1e-12

    # A rotation of the mount must not change how the ship is moving — it just
    # re-resolves the same world velocity onto the new axes. This is the whole
    # reason velocity is world-frame: callers used to project it themselves and
    # hand the components to the renderer separately, where a stale projection
    # or a forgotten argument silently rendered a moving ship at rest.
    y = yaw(c, 25.0)
    @test y.velocity == v
    β = SpaceTime.camera_beta(y)
    @test norm(β) ≈ 0.6 atol = 1e-12               # speed is invariant
    @test β[1] ≈ 0.6 * cosd(25.0) atol = 1e-12     # and it rotates correctly
    @test abs(β[2]) ≈ 0.6 * sind(25.0) atol = 1e-12
    @test truck(c, 1.0).velocity == v              # translations too
    @test ThinLensCamera(pos, SVector(0.0,0.0,0.0), SVector(0.0,0.0,1.0);
                         velocity=v).velocity == v

    # Default is at rest, and then the tetrad must be the unboosted one.
    still = Camera(pos, SVector(0.0,0.0,0.0), SVector(0.0,0.0,1.0), 0.5)
    @test still.velocity == SVector(0.0, 0.0, 0.0)
    @test all(map((a, b) -> a == b, SpaceTime.camera_tetrad(still, st),
                  SpaceTime.ks_camera_tetrad(still.pos, still.fwd, still.right,
                                             still.up_local, 1.0)))

    # The CPU renderer can now render a moving observer at all, and it beams:
    # flying into the field blueshifts and brightens what is ahead.
    bg = fill(RGBf(0.05, 0.05, 0.08), 64, 32)
    mean(a) = sum(x -> (Float64(x.r) + Float64(x.g) + Float64(x.b)) / 3, a) / length(a)
    at_rest = render(still, st, bg; disc=nothing, width=28, height=16,
                     samples=1, rng=Xoshiro(1), relativistic=true)
    moving = render(c, st, bg; disc=nothing, width=28, height=16,
                    samples=1, rng=Xoshiro(1), relativistic=true)
    @test mean(moving) > 1.5 * mean(at_rest)
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
    # The invariant this whole file exists for: the same `Look` at two output
    # resolutions must give the same picture, differing only in sharpness. Every
    # length in a `Look` is a fraction of frame height, so there is no reference
    # resolution to get wrong.
    #
    # Measure the glow around a lone bright pixel as a fraction of frame height.
    function glow_fraction(w, h, look)
        img = fill(RGBf(0, 0, 0), w, h)
        img[w ÷ 2, h ÷ 2] = RGBf(50, 50, 50)
        p = postprocess(img, look)
        cx, cy = w ÷ 2, h ÷ 2
        prof = [Float64(p[cx + k, cy].r) for k in 1:min(cx, cy) - 2]
        r = findfirst(<(0.05 * prof[1]), prof)
        return (r === nothing ? length(prof) : r) / h
    end

    look = Look(gain=1.0, exposure=0.0, gamma=1.0, bloom_strength=1.0,
                threshold=0.5, bloom_radius=10.0 / 180, bloom_power=1.5,
                streak_strength=0.0, tonemap=:none)
    @test glow_fraction(960, 540, look) ≈ glow_fraction(320, 180, look) rtol = 0.15

    # ...and the test has teeth: a radius fixed in absolute pixels fails it,
    # which is the bug the fractional units replaced.
    function glow_fraction_px(w, h, radius_px)
        img = fill(RGBf(0, 0, 0), w, h)
        img[w ÷ 2, h ÷ 2] = RGBf(50, 50, 50)
        p = postprocess(img; gain=1.0, exposure=0.0, gamma=1.0,
                        bloom_strength=1.0, threshold=0.5,
                        bloom_radius=radius_px, bloom_power=1.5,
                        streak_strength=0.0, tonemap=:none)
        cx, cy = w ÷ 2, h ÷ 2
        prof = [Float64(p[cx + k, cy].r) for k in 1:min(cx, cy) - 2]
        r = findfirst(<(0.05 * prof[1]), prof)
        return (r === nothing ? length(prof) : r) / h
    end
    @test glow_fraction_px(960, 540, 10.0) < 0.5 * glow_fraction_px(320, 180, 10.0)

    # Lens-plane effects take the same treatment: a speck of dust covers a
    # fixed fraction of the frame regardless of the sensor behind it.
    darkness(img) = 1.0 - sum(Float64(c.r) for c in img) / length(img)
    dust = LensDust(count=12, size_min=3.0 / 180, size_max=3.0 / 180,
                    opacity_min=0.8, opacity_max=0.8)
    a = fill(RGBf(1, 1, 1), 320, 180)
    b = fill(RGBf(1, 1, 1), 960, 540)
    apply_lens_dust!(a; lens_dust=dust, rng=Xoshiro(1))
    apply_lens_dust!(b; lens_dust=dust, rng=Xoshiro(1))
    @test darkness(b) ≈ darkness(a) rtol = 0.25
end

@testset "Resolution-independent grain" begin
    # Grain is the one effect that was never scaled at all: one deviate per
    # pixel means the grain covers six times less of a 2160-line frame than of
    # a 360-line one, and effectively vanishes at 4K. That is why low-res
    # renders read as more filmic.
    #
    # Box-downsampling by F averages F² deviates. Independent ones lose a
    # factor F of RMS; ones correlated across an F-pixel grain cell survive.
    function down(img, f)
        w, h = size(img)
        o = zeros(RGBf, w ÷ f, h ÷ f)
        for j in 1:size(o, 2), i in 1:size(o, 1)
            s = 0.0
            for dj in 0:f-1, di in 0:f-1
                s += Float64(img[(i-1)*f + di + 1, (j-1)*f + dj + 1].g)
            end
            o[i, j] = RGBf(s / f^2, s / f^2, s / f^2)
        end
        o
    end
    # Relative RMS of the grain after downsampling to a common grid.
    function grain(w, h, gs, f)
        clean = fill(RGBf(0.18, 0.18, 0.18), w, h)
        sensor_expose!(clean; iso=400.0, add_noise=false)
        noisy = fill(RGBf(0.18, 0.18, 0.18), w, h)
        sensor_expose!(noisy; iso=400.0, grain_size=gs, rng=Xoshiro(11))
        c = f == 1 ? clean : down(clean, f)
        n = f == 1 ? noisy : down(noisy, f)
        μ = sum(x -> Float64(x.g), c) / length(c)
        sqrt(sum((Float64(n[i].g) - Float64(c[i].g))^2
                 for i in eachindex(n)) / length(n)) / μ
    end

    F = 4
    # Frame-relative grain keeps its strength across a 4x resolution change.
    @test grain(1280 * F, 720 * F, 1 / 360, F) ≈ grain(1280, 720, 1 / 360, 1) rtol = 0.1
    # Per-pixel grain loses a factor of F, which is the defect.
    @test grain(1280 * F, 720 * F, 0.0, F) < 0.45 * grain(1280, 720, 0.0, 1)
end

@testset "Look and Sampling" begin
    # The 6x disagreement between the video grade and the hero grade is real and
    # aesthetic, not a resolution artefact. Both are resolution independent;
    # they simply disagree about how wide the halo should be. Pinning it here
    # means changing it has to be deliberate.
    @test LOOK_FILM.bloom_radius / LOOK_HERO.bloom_radius ≈ 6.0
    @test LOOK_FILM.streak_width / LOOK_HERO.streak_width ≈ 6.0

    l = with_look(LOOK_FILM; exposure=1.5)
    @test l.exposure == 1.5
    @test l.gamma == LOOK_FILM.gamma          # everything else carries over
    @test l.bloom_radius == LOOK_FILM.bloom_radius

    s = with_sampling(MOTION; samples=6)
    @test s.samples == 6 && s.shutter == MOTION.shutter
    # A still is not "the CPU preset" and a moving frame is not "the GPU
    # preset": they differ only in effort and shutter.
    @test STILL.shutter == 0.0 && MOTION.shutter > 0.0
    @test STILL.samples > MOTION.samples

    # Reproducible by construction, and independent per frame.
    @test rand(sampling_rng(MOTION, 3)) == rand(sampling_rng(MOTION, 3))
    @test rand(sampling_rng(MOTION, 3)) != rand(sampling_rng(MOTION, 4))
    # 180-degree shutter on a 1/24 s frame is 1/48 s.
    @test shutter_span(MOTION, 1 / 24) ≈ 1 / 48
    @test shutter_span(STILL, 1 / 24) == 0.0

    # The CPU renderer takes the same `Sampling` the GPU does — effort is not a
    # property of a device. This is the CPU half; the GPU half needs Metal.
    bg = fill(RGBf(0.02, 0.02, 0.05), 64, 32)
    st = Schwarzschild(1.0)
    cam = Camera(SVector(28.0, 0.0, 3.0), SVector(0.0, 0.0, 0.0),
                 SVector(0.0, 0.0, 1.0), 0.35)
    m = render_motion(t -> cam, 0.0, 1.0, st, bg,
                      with_sampling(MOTION; samples=2);
                      disc=nothing, width=24, height=14)
    @test size(m) == (24, 14)
    @test all(c -> isfinite(c.r) && isfinite(c.g) && isfinite(c.b), m)

    # apply_look! runs the chain and leaves a sane image.
    img = fill(RGBf(0.2, 0.2, 0.2), 64, 36)
    img[32, 18] = RGBf(40, 40, 40)
    out = apply_look!(img, with_look(LOOK_FILM; f_number=0.0); rng=Xoshiro(3))
    @test size(out) == (64, 36)
    @test all(c -> isfinite(c.r) && isfinite(c.g) && isfinite(c.b), out)
    @test all(c -> c.r >= 0 && c.g >= 0 && c.b >= 0, out)
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

@testset "Gas erosion" begin
    disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                         blackbody=Blackbody(wb_temperature=10000.0),
                         density_falloff=0.8)
    small = (nr=48, nphi=64, nz=16)
    ref = DiscVolume(disc; M=1.0, rng=Xoshiro(7), small...)

    # erosion=0 is the shipped look, exactly: the carve must be off by default
    # and must short-circuit before it touches the Worley field.
    @test DiscVolume(disc; M=1.0, rng=Xoshiro(7), small...).density == ref.density
    @test DiscVolume(disc; M=1.0, rng=Xoshiro(7), erosion=0.0,
                     erosion_mode=:vein, small...).density == ref.density

    @test_throws ArgumentError DiscVolume(disc; M=1.0, erosion_mode=:nope, small...)

    # Erosion only ever removes gas, and it removes a real fraction of it —
    # the failure mode this guards is a remap whose threshold never reaches the
    # base distribution, which leaves a contrast gain and no holes at all.
    gas(v) = filter(>(0.0), vec(v.density))
    for mode in (:billow, :vein)
        e = DiscVolume(disc; M=1.0, rng=Xoshiro(7), erosion=0.9,
                       erosion_mode=mode, small...)
        @test count(<(0.01), gas(e)) / length(gas(e)) >
              count(<(0.01), gas(ref)) / length(gas(ref)) + 0.05
        @test all(isfinite, e.density)
        @test maximum(e.density) ≈ 1.0            # peak normalisation survives
    end

    # The remap's denominator is kept away from zero however hard it is driven.
    @test all(isfinite, DiscVolume(disc; M=1.0, rng=Xoshiro(7), erosion=1.0,
                                   small...).density)

    # Both Worley modes are range-matched, so `erosion` means the same thing in
    # each; an unmatched detail field is what made the first attempt inert.
    lat = SpaceTime._noise_lattice(Xoshiro(7))
    wv = [SpaceTime._worley_fbm(lat, 4x, 4y, 4z, 3; ridge=true)
          for x in 0:0.37:12, y in 0:0.41:12, z in 0:0.43:12]
    wb = [SpaceTime._worley_fbm(lat, 4x, 4y, 4z, 3; ridge=false)
          for x in 0:0.37:12, y in 0:0.41:12, z in 0:0.43:12]
    @test all(0 .<= wv .<= 1) && all(0 .<= wb .<= 1)
    @test abs(sum(wv) / length(wv) - sum(wb) / length(wb)) < 0.15
end

@testset "Star colour is independent of the disc" begin
    # Regression guard: star colour used to read the disc's blackbody LUT, so
    # regrading the disc re-tinted the whole sky, and a 10000 K disc white
    # point put every star below it and turned the field gold. Stars carry
    # their own white point now, and must not move when the disc's does.
    bg = [RGBf(0, 0, 0) for i in 1:8, j in 1:4]
    ctx(wb) = MetalPreviewContext(bg, 16, 16;
        disc=AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                           density_falloff=0.8,
                           blackbody=Blackbody(wb_temperature=wb)))
    hot, cool = ctx(10000.0), ctx(4000.0)
    @test Array(hot.star_lut) == Array(cool.star_lut)      # sky unmoved
    @test Array(hot.bb_lut) != Array(cool.bb_lut)          # disc did move

    # The starfield no longer needs a disc to borrow a LUT from.
    plain = MetalPreviewContext(bg, 16, 16)
    set_starfield!(plain; height=16, fov_factor=0.75)
    p = Array(plain.star_params)
    @test p[1] > 0 && p[14] > 1                            # on, with a real LUT

    # The default temperature range must straddle the star white point, or
    # every star tints one way — the original bug in its other form.
    @test p[10] < SpaceTime.STAR_WB_TEMPERATURE < p[10] + p[11]

    # Colour actually tracks temperature across that white point.
    lut = Array(plain.star_lut)
    idx(T) = clamp(round(Int, (T - p[12]) / (p[13] - p[12]) * (p[14] - 1)) + 1,
                   1, Int(p[14]))
    br(T) = (c = lut[:, idx(T)]; c[3] / c[1])
    @test br(3500.0) < 0.5                                  # cool star is warm
    @test br(SpaceTime.STAR_WB_TEMPERATURE) ≈ 1.0 atol = 0.02   # white point
    @test br(15000.0) > 1.2                                 # hot star is blue
end

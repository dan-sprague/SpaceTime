"""
    julia_main() -> Cint

Entry point for the standalone SpaceTime app built with PackageCompiler's
`create_app`: boots the real-time simulator ([`fly_native`](@ref)) with the
bundled starmap. `fly_native` runs the window loop on the calling thread and
returns when it closes, so no thread pinning is needed. Set `SPACETIME_SMOKE=1`
to auto-close after ~8 s (build smoke tests), and `RES=1080|1440|2160` to pick
the render resolution.

Pass `arcade` as an argument (or set `SPACETIME_MODE=arcade`) for the Kerr
arcade mode — low internal resolution, nearest upscale, six magma tones —
with `ARES=144|180|288|360` picking the internal height.
"""
function julia_main()::Cint
    try
        candidates = [
            joinpath(dirname(Sys.BINDIR), "assets", "starmap_g4k.jpg"), # app bundle
            joinpath(dirname(@__DIR__), "assets", "starmap_g4k.jpg"),   # dev checkout
            joinpath(pwd(), "assets", "starmap_g4k.jpg"),               # cwd fallback
        ]
        idx = findfirst(isfile, candidates)
        if idx === nothing
            @error "starmap_g4k.jpg not found next to the app or the package" candidates
            return 1
        end
        bg = FileIO.load(candidates[idx])

        spacetime = Schwarzschild(1.0)
        # White balance decides which radius renders neutral. At 10000 K that was
        # r = 6M, leaving 93% of the disc's area cooler than white — the beige. At
        # 5000 K neutral sits near 15M, so the hot inner disc reads white-to-blue and
        # only the outer edge stays warm. Star colour is unaffected: it has its own
        # white point (see `STAR_WB_TEMPERATURE`).
        disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                             blackbody=Blackbody(wb_temperature=5000.0),
                             density_falloff=0.8)
        # `haze` adds the diffuse envelope that greys the shadow. Erosion is off:
        # it carves real fine structure into the gas, but it only ever removes gas, so
        # it needs `opacity_scale` raised to compensate — the two move together.
        volume = DiscVolume(disc; M=spacetime.M, haze=0.03, haze_height=4.0)
        cam = Camera(SVector(30.0, 1.1, 1.6), SVector(0.0, 0.0, 0.0),
                     SVector(0.0, 0.0, 1.0), Lens(24.0))

        smoke = get(ENV, "SPACETIME_SMOKE", "0") == "1" ? 8.0 : Inf

        # Arcade mode: `SpaceTimeApp arcade`, or SPACETIME_MODE=arcade.
        # A spinning hole rendered at a low internal resolution, nearest-
        # upscaled and collapsed onto a handful of magma tones. It cannot use
        # the deflection fan (that needs spherical symmetry), so every pixel is
        # traced directly — which is only affordable at this resolution, which
        # is the look. ARES picks the internal resolution; keep it an integer
        # divisor of the display or the upscale gives uneven pixels.
        if "arcade" in ARGS || get(ENV, "SPACETIME_MODE", "") == "arcade"
            ares = get(Dict("144" => (256, 144), "180" => (320, 180),
                            "288" => (512, 288), "360" => (640, 360)),
                       get(ENV, "ARES", "144"), (256, 144))
            fly_native(cam, Kerr(1.0, 0.9), bg; disc=disc,
                       width=ares[1], height=ares[2],
                       winwidth=1280, winheight=720,
                       arcade=true, vsync=false,
                       palette=arcade_palette(6; ramp=:magma,
                                              lo=0.12, hi=0.92),
                       dither=1.0,
                       star_texture_weight=0.0, star_psf_pixels=0.9,
                       star_density=110,
                       title="Spacetime — Kerr arcade",
                       max_seconds=smoke)
            return 0
        end

        res = get(Dict("1080" => (1920, 1080), "1440" => (2560, 1440),
                       "2160" => (3840, 2160)),
                  get(ENV, "RES", "1440"), (2560, 1440))
        fly_native(cam, spacetime, bg; disc=disc, volume=volume,
                   width=res[1], height=res[2], max_seconds=smoke)
        return 0
    catch err
        Base.showerror(stderr, err, catch_backtrace())
        println(stderr)
        return 1
    end
end

"""
    julia_main() -> Cint

Entry point for the standalone SpaceTime app built with PackageCompiler's
`create_app`: boots the real-time simulator ([`fly_native`](@ref)) with the
bundled starmap. `fly_native` runs the window loop on the calling thread and
returns when it closes, so no thread pinning is needed. Set `SPACETIME_SMOKE=1`
to auto-close after ~8 s (build smoke tests), and `RES=1080|1440|2160` to pick
the render resolution.
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

        res = get(Dict("1080" => (1920, 1080), "1440" => (2560, 1440),
                       "2160" => (3840, 2160)),
                  get(ENV, "RES", "1440"), (2560, 1440))
        fly_native(cam, spacetime, bg; disc=disc, volume=volume,
                   width=res[1], height=res[2],
                   max_seconds=get(ENV, "SPACETIME_SMOKE", "0") == "1" ?
                               8.0 : Inf)
        return 0
    catch err
        Base.showerror(stderr, err, catch_backtrace())
        println(stderr)
        return 1
    end
end

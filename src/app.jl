"""
    julia_main() -> Cint

Entry point for the standalone SpaceTime app built with PackageCompiler's
`create_app`: boots the real-time flythrough with the bundled starmap. The
built executable should be launched with `--julia-args -t auto,1` so render
workers don't starve the UI thread. Set `SPACETIME_SMOKE=1` to auto-close
after ~10 s (build smoke tests).
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
        disc = AccretionDisc(inner_radius=3.0, outer_radius=20.0,
                             blackbody=Blackbody(wb_temperature=10000.0),
                             density_falloff=0.8)
        volume = DiscVolume(disc; M=spacetime.M)
        cam = Camera(SVector(30.0, 1.1, 1.6), SVector(0.0, 0.0, 0.0),
                     SVector(0.0, 0.0, 1.0), Lens(24.0))

        fig = flythrough(cam, spacetime, bg; disc=disc, volume=volume)

        if get(ENV, "SPACETIME_SMOKE", "0") == "1"
            @async begin
                sleep(10.0)
                GLMakie.closeall()
            end
        end
        wait(GLMakie.Makie.getscreen(fig.scene))
        return 0
    catch err
        Base.showerror(stderr, err, catch_backtrace())
        println(stderr)
        return 1
    end
end

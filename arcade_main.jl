# juliac entry point for the standalone Kerr arcade binary.
#
#   JC=$(dirname $(dirname $(which julia)))/share/julia/juliac/juliac.jl
#   julia --project=. "$JC" --output-exe build/SpaceTimeArcade \
#         --relative-rpath --verbose arcade_main.jl
#
# No --trim: this dependency tree (GLFW, ObjectiveC's @objc, Metal's kernel compilation) is not statically
# analysable, and trim=safe would reject it. Without trim juliac still gives a
# single executable, just a large one.

using SpaceTime
using StaticArrays
using Colors

# No image file is loaded, deliberately. Arcade mode runs with
# `star_texture_weight = 0`, so the equirectangular starmap contributes
# nothing — the sky is the procedural point-star field. Loading a JPEG would
# mean FileIO picking a decoder at *runtime*, which an AOT binary cannot do:
# it tries to precompile JpegTurbo_jll by spawning a `julia` that is not there,
# and dies with ENOENT. A few black texels stand in for the texture that gets
# multiplied by zero, and the binary needs no assets at all.
const BLANK_SKY = fill(RGB{Float32}(0, 0, 0), 8, 4)

function (@main)(args::Vector{String})::Cint
    try
        # Internal render height; keep it an integer divisor of the display or
        # the nearest upscale gives uneven pixels. 144 x10 and 180 x8 both land
        # exactly on 2560x1440.
        ares = get(Dict("144" => (256, 144), "180" => (320, 180),
                        "288" => (512, 288), "360" => (640, 360)),
                   get(ENV, "ARES", "144"), (256, 144))

        spin_a = something(tryparse(Float64, get(ENV, "SPIN", "0.9")), 0.9)
        disc = AccretionDisc(inner_radius = 3.0, outer_radius = 20.0,
                             blackbody = Blackbody(wb_temperature = 5000.0),
                             density_falloff = 0.8)
        cam = Camera(SVector(15.0, 0.0, 2.0), SVector(0.0, 0.0, 0.0),
                     SVector(0.0, 0.0, 1.0), Lens(24.0))

        fly_native(cam, Kerr(1.0, spin_a), BLANK_SKY;
                   disc = disc,
                   width = ares[1], height = ares[2],
                   winwidth = 1280, winheight = 720,
                   arcade = true, vsync = false,
                   palette = arcade_palette(6; ramp = :magma,
                                            lo = 0.12, hi = 0.92),
                   dither = 1.0,
                   star_texture_weight = 0.0, star_psf_pixels = 0.9,
                   star_density = 110,
                   title = "Spacetime — Kerr arcade",
                   max_seconds = get(ENV, "SPACETIME_SMOKE", "0") == "1" ?
                                 8.0 : Inf)
        return 0
    catch err
        Base.showerror(stderr, err, catch_backtrace())
        println(stderr)
        return 1
    end
end

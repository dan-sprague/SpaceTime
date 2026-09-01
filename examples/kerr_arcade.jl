# Arcade mode: a spinning black hole rendered at a low internal resolution and
# blown up with nearest-neighbour, palette-quantized and ordered-dithered.
#
# The point is not that the pixels hide sloppy physics. The geodesics are the
# real thing — exact Kerr in Cartesian Kerr–Schild, verified against the
# Schwarzschild kernel at a = 0 and against L_z conservation up to a = 0.998.
# What the low resolution buys is the ability to drop the deflection fan, which
# Kerr cannot use anyway (it needs spherical symmetry), and still hold a high
# frame rate: at 256x144 the whole frame is cheaper than the fan alone was.
#
#   julia -t auto,1 --project=. examples/kerr_arcade.jl
#
# Keys are the usual ones (W/S/A/D/Q/E fly, drag to look, 1-4 grade presets).
# Pick an internal resolution that integer-divides the display or the upscale
# gives uneven pixels: 256x144 x10 and 320x180 x8 both land on 2560x1440.

using SpaceTime, StaticArrays, FileIO

bg = FileIO.load(joinpath(@__DIR__, "..", "assets", "starmap_g4k.jpg"))

# a = 0.9: strong frame dragging (the shadow sits visibly off-centre and is
# flattened on the prograde side) while still resolving cleanly at dt = 0.1.
# Near-extremal (a > ~0.95) needs a finer step than arcade mode uses — the
# capture radius and the horizon converge and the shadow starts to leak sky.
spacetime = Kerr(1.0, 0.9)

disc = AccretionDisc(inner_radius = 3.0, outer_radius = 20.0,
                     blackbody = Blackbody(wb_temperature = 5000.0),
                     density_falloff = 0.8)

cam = Camera(SVector(15.0, 0.0, 2.0), SVector(0.0, 0.0, 0.0),
             SVector(0.0, 0.0, 1.0), Lens(24.0))

fly_native(cam, spacetime, bg;
           disc = disc,
           width = 256, height = 144,        # internal; x10 -> 2560x1440
           winwidth = 1280, winheight = 720,
           arcade = true,
           quantize = 12,                    # levels per channel
           dither = 1.0,                     # 4x4 Bayer, one quantization step
           star_texture_weight = 0.25,
           star_psf_pixels = 0.9,
           title = "Spacetime — Kerr arcade")

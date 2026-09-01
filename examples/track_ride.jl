# Ride a generated racing track past a spinning black hole.
#
#   julia -t auto,1 --project=. examples/track_ride.jl
#
# The camera is not animated. It is bolted to a timelike worldline produced by
# src/track.jl -- position, heading, roll and velocity all come from the same
# solution, so this is what a ship on that trajectory would actually see. The
# starfield sweep you get near periapsis is aberration from the ship's own
# 0.85c, not an effect dialled in afterwards.
#
# Nothing here is steering. The whole pass is FREE FALL: an accelerometer on
# board reads exactly zero the entire time, including the violent-looking part.
# The track was not shot from a start gate to a finish gate either -- it was
# built outward from its own periapsis, which is why it is guaranteed to make
# this pass and guaranteed to be flyable.
#
# Mouse-drag still looks around, as an offset from the track's own frame.
# Keys: ESC quits, 1-4 grade presets, B cycles the sky.

using SpaceTime, StaticArrays, FileIO

bg = FileIO.load(joinpath(@__DIR__, "..", "assets", "starmap_g4k.jpg"))
spacetime = Kerr(1.0, 0.9)          # horizon 1.436M, photon orbit 1.558M

# Periapsis 3M at 0.85c, tilted 20 degrees out of the equator.
#
# `speed` is what a STATIC observer at periapsis measures, so it is always a
# real speed in [0,1) -- the coordinate-speed limit shrinks near the hole and
# would make this parameter meaningless. Local escape speed here is
# sqrt(2M/r) = 0.816, and the boundary is sharp: 0.78 escapes only to r = 8.7M
# and 0.72 is captured outright. 0.85 comes in from ~60M, whips past at 3M and
# leaves. Drop it toward 0.79 for a much tighter, longer wrap around the hole.
#
# prograde=false is the other big knob. At a = 0.9 frame dragging moves the
# innermost stable orbit by nearly an order of magnitude, so the retrograde
# version of this same pass is a completely different trajectory.
track = encounter_track(spacetime, 3.0, 0.85;
                        τ_in = 70, τ_out = 70, dτ = 0.01,
                        inclination = 0.35, prograde = true)

disc = AccretionDisc(inner_radius = 3.0, outer_radius = 20.0,
                     blackbody = Blackbody(wb_temperature = 5000.0),
                     density_falloff = 0.8)

cam = Camera(track.pos[1], track.pos[1] + track.fwd[1], track.up[1], Lens(10.0))

fly_native(cam, spacetime, bg;
           disc = disc,
           track = track,
           track_speed = 2.5,       # proper seconds of ship time per wall second
           track_loop = true,
           focal = 10.0,             # 10mm: the wide field is where lensing reads
           # Live tracing. `baked = true` bakes the lensing into warp maps
           # along the track instead, which is ~41x cheaper per pixel -- but it
           # interpolates badly exactly where this track is interesting (see
           # the note in src/metal.jl), so it is off by default.
           baked = false,
           # 384x216 internal, x5 to a 1920x1080 window: an integer upscale, so
           # the pixels stay square and crisp. Worst case on this track (the
           # approach, where the disc fills a 10mm field) measures 13.6 ms, so
           # there is real headroom left for game logic.
           width = 384, height = 216,
           winwidth = 1920, winheight = 1080,
           arcade = true,
           # Full colour. Swap in
           #   palette = arcade_palette(6; ramp = :magma, lo = 0.12, hi = 0.92),
           #   dither  = 1.0,
           # for the six-tone arcade look.
           star_density = 384,
           vsync = false,
           star_texture_weight = 0.0,
           star_psf_pixels = 1.4,
           title = "Spacetime — track ride")

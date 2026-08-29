# ---------------------------------------------------------------------------
# Raw HDR frame I/O: the raytrace is the negative, post-processing is the
# print. Renders save linear unclamped Float32 and all grading happens later
# in the standalone post app (src/postapp.jl).
# ---------------------------------------------------------------------------

"""
    save_raw(path, img::Matrix{RGBf}; metadata=Dict())

Save a linear-light HDR raytrace as a 32-bit float TIFF (display
orientation — opens in Photoshop/Nuke/etc.) plus a TOML sidecar
`<path>.toml` recording how the frame was traced (camera, lens, spacetime,
sampling). Values are stored unclamped and untonemapped.

Metadata values must be TOML-representable (numbers, strings, booleans,
vectors); convert `SVector`s with `collect`.
"""
function save_raw(path::AbstractString, img::Matrix{RGBf};
                  metadata::AbstractDict=Dict{String,Any}())
    FileIO.save(path, rotr90(img))
    meta = Dict{String,Any}(metadata)
    meta["format"] = "SpaceTime raw v1: linear Float32 RGB, display orientation"
    meta["saved_at"] = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
    open(path * ".toml", "w") do io
        TOML.print(io, meta)
    end
    return path
end

"""
    load_raw(path) -> (img::Matrix{RGBf}, metadata::Dict)

Load a frame saved by [`save_raw`](@ref), returning it in the renderer's
`[width, height]` orientation together with its sidecar metadata (an empty
Dict when no sidecar is found).
"""
function load_raw(path::AbstractString)
    img = rotl90(RGBf.(FileIO.load(path)))
    sidecar = path * ".toml"
    meta = isfile(sidecar) ? TOML.parsefile(sidecar) : Dict{String,Any}()
    return img, meta
end

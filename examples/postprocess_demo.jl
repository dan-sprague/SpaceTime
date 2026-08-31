# Standalone post-processing app: grade a raw HDR frame saved by the
# flythrough's "Save raw" button (or save_raw).
#
# Run with:
#     julia -t auto,1 --project examples/postprocess_demo.jl <raw.tiff>

using Pkg
Pkg.activate(".")

using SpaceTime
using GLMakie

isempty(ARGS) &&
    error("usage: julia -t auto,1 --project examples/postprocess_demo.jl <raw.tiff>")

fig = postprocessor(ARGS[1])
isinteractive() || wait(Makie.getscreen(fig.scene))

# Build the standalone SpaceTime app with PackageCompiler.
#
#   julia build_app.jl              # -> build/SpaceTimeApp
#   build/SpaceTimeApp/bin/SpaceTimeApp            # the 1440p simulator
#   build/SpaceTimeApp/bin/SpaceTimeApp arcade     # Kerr arcade mode
#
# PackageCompiler is a build-time tool, so it lives in its own environment
# rather than in the package's Project.toml. The starmap is copied in beside
# the binary because `julia_main` looks for it at
# `dirname(Sys.BINDIR)/assets/starmap_g4k.jpg`.

const ROOT = @__DIR__
const DEST = joinpath(ROOT, "build", "SpaceTimeApp")
const BUILDENV = joinpath(ROOT, "build", "env")

using Pkg
mkpath(BUILDENV)
Pkg.activate(BUILDENV)
"PackageCompiler" in [p.name for p in values(Pkg.dependencies())] ||
    Pkg.add("PackageCompiler")
using PackageCompiler

# A precompile run that actually exercises the GPU paths: without this the
# first frames of the shipped app pay Metal shader compilation. Both modes,
# each closing itself after a few seconds.
precompile_script = joinpath(BUILDENV, "warm.jl")
open(precompile_script, "w") do io
    write(io, """
    ENV["SPACETIME_SMOKE"] = "1"
    ENV["RES"] = "1080"
    using SpaceTime
    SpaceTime.julia_main()
    ENV["SPACETIME_MODE"] = "arcade"
    SpaceTime.julia_main()
    """)
end

isdir(DEST) && rm(DEST; recursive=true)
create_app(ROOT, DEST;
           executables = ["SpaceTimeApp" => "julia_main"],
           precompile_execution_file = precompile_script,
           include_lazy_artifacts = true,
           force = true)

# Assets next to the binary, where julia_main looks for them.
assets_dst = joinpath(DEST, "assets")
mkpath(assets_dst)
cp(joinpath(ROOT, "assets", "starmap_g4k.jpg"),
   joinpath(assets_dst, "starmap_g4k.jpg"); force=true)

println("\nBuilt: ", joinpath(DEST, "bin", "SpaceTimeApp"))
println("Run:   ", joinpath(DEST, "bin", "SpaceTimeApp"), " arcade")

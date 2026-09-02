//! Shader build.
//!
//! Precompiling to a .metallib keeps the Metal front-end out of the launch
//! path, but `xcrun metal` ships with full Xcode, not the Command Line Tools.
//! When it is missing we fall back to the runtime compiler in the Metal
//! framework -- always present on macOS, and the path Metal.jl uses -- at the
//! cost of a few hundred ms at startup.
use std::{env, fs, path::PathBuf, process::Command};

const SHADERS: [&str; 2] = ["shaders/trace.metal", "shaders/post.metal"];

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // One translation unit for both kernels, so either path yields one library.
    let mut combined = String::new();
    for src in SHADERS {
        println!("cargo:rerun-if-changed={src}");
        combined.push_str(&fs::read_to_string(src).unwrap_or_else(|e| panic!("{src}: {e}")));
        combined.push('\n');
    }
    let combined_path = out.join("combined.metal");
    fs::write(&combined_path, &combined).unwrap();

    let lib = out.join("shaders.metallib");
    let precompiled = try_precompile(&combined_path, &lib);

    let glue = format!(
        "pub const SHADER_SRC: &str = include_str!({:?});\n\
         pub const SHADER_LIB: Option<&[u8]> = {};\n",
        combined_path,
        if precompiled { format!("Some(include_bytes!({:?}))", lib) } else { "None".into() }
    );
    fs::write(out.join("shaders.rs"), glue).unwrap();

    if !precompiled {
        println!("cargo:warning=xcrun metal not found (needs full Xcode); \
                  shaders will be compiled at startup instead");
    }
}

fn try_precompile(src: &PathBuf, lib: &PathBuf) -> bool {
    let air = lib.with_extension("air");
    let ok = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-O3", "-ffast-math", "-c"])
        .arg(src)
        .arg("-o")
        .arg(&air)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return false;
    }
    Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(lib)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

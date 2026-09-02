# SpaceTime — Rust + Metal

A port of the real-time path of [SpaceTime.jl](../README.md) to Rust with a
hand-written Metal Shading Language kernel. Same physics, same chart, same
look; the question it was built to answer is what the rewrite actually buys.

```bash
cargo run --release                # fly (arcade mode, Kerr a = 0.9)
cargo run --release -- --selftest  # executable physics checks
cargo run --release -- --bench     # where the frame time goes
```

`xcrun metal` ships with full Xcode, not the Command Line Tools. When it is
missing the build falls back to Metal's runtime compiler — the same path
Metal.jl uses — at the cost of a few hundred ms at startup. Nothing else
changes.

## What it measures

M3, Kerr a = 0.9, volumetric gas, procedural stars, RK4, dt = 0.1:

| resolution | trace | fps | Mray/s |
|---|---|---|---|
| 256×144 | 3.4 ms | 296 | 10.9 |
| 640×360 | 16.4 ms | 61 | 14.0 |
| 1280×720 | 62.2 ms | 16 | 14.8 |
| 1920×1080 | 137.5 ms | 7 | 15.1 |
| 2560×1440 | 242.7 ms | 4 | 15.2 |

Throughput is flat at ~15 Mray/s across a 56× range of pixel counts. That is
what an ALU-bound kernel looks like: the cost is the geodesic arithmetic, and
nothing about the host language or the dispatch is in the way.

Integrator, 1280×720, same scene:

| integrator | trace | fps | RMS vs RK4 |
|---|---|---|---|
| RK4 (4 evals/step) | 62.0 ms | 16 | — |
| midpoint (2 evals) | 37.4 ms | 27 | 0.018 |
| adaptive RK45, capped | 110.8 ms | 9 | 0.0014 |
| adaptive RK45, uncapped, tol 1e-3 | 7.5 ms | 133 | 0.389 |

Sky only — no disc, no gas — which separates the geodesic integration from
everything that piggybacks on its step:

| integrator | trace | fps | RMS vs RK4 |
|---|---|---|---|
| RK4 | 52.0 ms | 19 | — |
| adaptive RK45, uncapped, tol 1e-3 | 7.0 ms | 143 | 0.0013 |
| adaptive RK45, uncapped, tol 1e-5 | 9.3 ms | 108 | 0.0003 |

**On the geodesics alone, uncapped adaptive RK45 is 5–7× faster than RK4 and
the image is the same.** The shadow still lands on 3√3 M to a tenth of a pixel
with zero phantom sky inside it (`--selftest` checks exactly this), so the big
steps are not costing accuracy where it is visible.

The 0.389 RMS in the full scene is therefore *not* the geodesics. It is the
volumetric march, which samples the gas every Nth **integration** step: the
error controller bounds the ODE's local truncation error and has no idea the
gas exists, so adaptive steps resample it at wildly uneven arc lengths. The
disc-plane crossing has the same dependency. Decouple the gas march from the
integrator's step schedule — march it on its own arc-length stride — and the
5–7× is available with the picture intact. (Implemented on the Julia side in
`src/metal.jl`, together with the adaptive-integrator port — which there uses
the Tsit5 pair and a PI step controller rather than this Cash-Karp/I-control
pair; neither is ported back here yet.)

## What is ported

The real-time path, which is what determines frame rate:

- Null geodesics in Cartesian Kerr–Schild, Schwarzschild **and** Kerr, RK4 with
  a radius-adaptive step, plus midpoint and adaptive Cash-Karp RK45 selectable
  as Metal function constants (the MSL analogue of the Julia kernel's `Val`
  dispatch).
- The camera as an orthonormal tetrad: static observer outside 2.5M, radial
  free-faller inside, optional Lorentz boost for a moving ship.
- Volumetric disc with Doppler- and gravitationally-shifted Planck emission,
  alpha-composited; the thin-plane disc as the non-volumetric fallback.
- Procedural starfield on an equal-area grid, evaluated in the source sky so
  lensing magnifies it.
- Grade, arcade palette, ordered dither, and nearest upscale straight into the
  drawable — no frame ever touches the CPU.

## What is not

Deliberately, because none of it is on the frame-rate path: the offline
photography pipeline (FFT bloom, Airy PSF, sensor model, thin-lens depth
buckets), the deflection-fan / layered temporal accumulation, warp-map baking,
the live fluid simulation, and the CPU reference renderer. The Julia package
remains the reference implementation and the place those live.

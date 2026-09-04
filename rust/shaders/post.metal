// ---------------------------------------------------------------------------
// Display grade, arcade palette, and presentation. Translation of
// `_pack_bgra_kernel!` in src/native_shell.jl, minus the bloom/streak gather
// (arcade mode renders with scattered light off, and the offline film grade
// lives in the Julia postprocess path).
//
// Dispatched over the DRAWABLE, not the render target: the internal resolution
// is upscaled here by nearest-neighbour, and every per-pixel effect below --
// the ordered dither especially -- is evaluated at SOURCE coordinates, so the
// dither pattern scales up with the pixels instead of crawling across them.
// ---------------------------------------------------------------------------
#include <metal_stdlib>
using namespace metal;

// grade[0]  exposure      grade[1]  filmic on/off   grade[2]  crush gamma
// grade[3]  saturation    grade[4]  vignette        grade[5..7] white balance
// grade[10] hue preserve  grade[11] output gamma    grade[12] grain
// grade[13] upscale (0 nearest, 1 bilinear)
// grade[16] quant levels  grade[17] dither          grade[18] palette entries

static inline float aces(float x) {
    return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f),
                 0.0f, 1.0f);
}

// Signed-int hash matching `_grain_hash`; Julia's >> on Int32 is arithmetic,
// which is what `int` gives here.
static inline float grain_hash(int x, int y) {
    int h = x * 374761393 + y * 668265263;
    h = (h ^ (h >> 13)) * 1274126177;
    h = h ^ (h >> 16);
    return (float)(h & 0x00FFFFFF) * 5.9604645e-8f;
}

kernel void pack(texture2d<float, access::write> dst [[texture(0)]],
                 const device float     *src     [[buffer(0)]],
                 const device float     *grade   [[buffer(1)]],
                 const device float     *palette [[buffer(2)]],
                 constant uint4         &dims    [[buffer(3)]],  // srcW srcH dstW dstH
                 constant float         &escale  [[buffer(4)]],
                 uint2 gid [[thread_position_in_grid]])
{
    int sw = (int)dims.x, sh = (int)dims.y;
    int dw = (int)dims.z, dh = (int)dims.w;
    if ((int)gid.x >= dw || (int)gid.y >= dh) return;

    // Nearest-neighbour cell (also the dither/palette grid). Flip vertically:
    // the sensor's row 0 is the top of the image.
    int x   = min((int)((long)gid.x * sw / dw), sw - 1);
    int row = min((int)((long)gid.y * sh / dh), sh - 1);
    int j = sh - 1 - row;

    float g_exp = grade[0], g_film = grade[1], g_crush = grade[2];
    float g_sat = grade[3], g_vig = grade[4];
    float g_hue = grade[10], g_gp = grade[11], g_grain = grade[12];
    float g_up = grade[13];
    float g_qlev = grade[16], g_dith = grade[17], g_pal = grade[18];

    // Source colour: nearest for the pixel-art path (arcade), or bilinear when
    // smoothing is on. Bilinear samples in flipped source-row space so the four
    // taps stay adjacent.
    float3 c;
    if (g_up > 0.5f) {
        float fx  = ((float)gid.x + 0.5f) * (float)sw / (float)dw - 0.5f;
        float frw = ((float)gid.y + 0.5f) * (float)sh / (float)dh - 0.5f;
        float fjs = (float)(sh - 1) - frw;
        int x0 = clamp((int)floor(fx), 0, sw - 1), x1 = min(x0 + 1, sw - 1);
        int y0 = clamp((int)floor(fjs), 0, sh - 1), y1 = min(y0 + 1, sh - 1);
        float tx = clamp(fx - floor(fx), 0.0f, 1.0f);
        float ty = clamp(fjs - floor(fjs), 0.0f, 1.0f);
        int o00 = 3 * (x0 + sw * y0), o10 = 3 * (x1 + sw * y0);
        int o01 = 3 * (x0 + sw * y1), o11 = 3 * (x1 + sw * y1);
        float3 c00 = float3(src[o00], src[o00 + 1], src[o00 + 2]);
        float3 c10 = float3(src[o10], src[o10 + 1], src[o10 + 2]);
        float3 c01 = float3(src[o01], src[o01 + 1], src[o01 + 2]);
        float3 c11 = float3(src[o11], src[o11 + 1], src[o11 + 2]);
        c = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
    } else {
        int o = 3 * (x + sw * j);
        c = float3(src[o + 0], src[o + 1], src[o + 2]);
    }
    c = c * (g_exp * escale) * float3(grade[5], grade[6], grade[7]);
    // A NaN would survive every clamp below and land in the drawable.
    c = select(c, float3(0.0f), c != c);

    float l = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    c = max(l + g_sat * (c - l), 0.0f);

    if (g_film > 0.5f) {
        // ACES per channel, blended with the hue-preserving variant (tonemap
        // the luminance, rescale the triple) by the hue dial.
        float3 pc = float3(aces(c.r), aces(c.g), aces(c.b));
        if (g_hue > 0.0f) {
            float Y = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
            float sY = aces(Y) / max(Y, 1.0e-8f);
            c = (1.0f - g_hue) * pc + g_hue * clamp(c * sY, 0.0f, 1.0f);
        } else {
            c = pc;
        }
        c = exp(log(max(c, 1.0e-6f)) * g_gp);
    }
    if (g_crush != 1.0f) c = exp(log(max(c, 1.0e-6f)) * g_crush);

    if (g_grain > 0.0f) {
        float l2 = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
        c += g_grain * (grain_hash(x + 1, j + 1) - 0.5f) * 2.0f *
             sqrt(max(l2, 2.0e-3f));
    }
    if (g_vig > 0.0f) {
        float vu = ((float)x - 0.5f * (float)sw) / (0.5f * (float)sh);
        float vv = ((float)j - 0.5f * (float)sh) / (0.5f * (float)sh);
        c *= max(1.0f - g_vig * 0.25f * (vu * vu + vv * vv), 0.0f);
    }
    c = clamp(c, 0.0f, 1.0f);

    // Arcade palette. The dither is ORDERED, not random, on purpose: the
    // pattern is fixed to the pixel grid so it does not crawl when the camera
    // moves, which is the failure mode that makes low-resolution rendering
    // read as broken rather than stylised.
    if (g_pal > 1.5f || g_qlev > 1.5f) {
        int bx0 = x & 1,        by0 = j & 1;
        int bx1 = (x >> 1) & 1, by1 = (j >> 1) & 1;
        int m = 4 * (2 * bx1 + by1 * (3 - 4 * bx1)) + (2 * bx0 + by0 * (3 - 4 * bx0));
        float d = g_dith * (((float)m + 0.5f) * 0.0625f - 0.5f);
        if (g_pal > 1.5f) {
            // Collapse to N tones by LUMINANCE. Per-channel quantisation keeps
            // three independent ramps and so keeps the picture colourful;
            // mapping brightness onto one ramp is what actually reduces the
            // tone count, and lets entry 0 be true black so empty sky reads as
            // empty rather than tinted.
            float n = g_pal;
            float lum = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
            int k = (int)clamp(lum * n + d, 0.0f, n - 1.0f);
            c = float3(palette[3 * k + 0], palette[3 * k + 1], palette[3 * k + 2]);
        } else {
            float L = g_qlev - 1.0f;
            c = clamp(floor(c * L + 0.5f + d) / L, 0.0f, 1.0f);
        }
    }

    dst.write(float4(c, 1.0f), gid);
}

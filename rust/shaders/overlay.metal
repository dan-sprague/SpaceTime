// ---------------------------------------------------------------------------
// Menu overlay: blits a character grid over the presented drawable. Runs after
// `pack`, reading and writing the drawable in place, so the menu floats on top
// of the live render without a second render target.
//
// The grid is a console: one glyph slot and one attribute per cell. This kernel
// maps each drawable pixel to a cell, samples the 5x7 font atlas, and paints a
// foreground colour (by attribute) over a solid panel.
//
// It is WRITE-only: a CAMetalLayer drawable is not guaranteed to carry
// ShaderRead usage, so reading it back to blend is unsafe. Pixels outside the
// panel are left as `pack` wrote them (this kernel just returns), and inside
// the panel we paint an opaque dark card -- robust on every drawable.
// ---------------------------------------------------------------------------
#include <metal_stdlib>
using namespace metal;

// ip[0] cols       ip[1] rows        ip[2] origin_x   ip[3] origin_y
// ip[4] cell_w     ip[5] cell_h      ip[6] px_scale   ip[7] glyph_w
// ip[8] glyph_h    ip[9] pad         ip[10] glyph_count
kernel void overlay(texture2d<float, access::write> dst [[texture(0)]],
                    const device float *atlas  [[buffer(0)]],
                    const device int   *cells  [[buffer(1)]],
                    const device int   *attr   [[buffer(2)]],
                    constant int       *ip     [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]])
{
    int W = (int)dst.get_width(), H = (int)dst.get_height();
    if ((int)gid.x >= W || (int)gid.y >= H) return;

    int cols = ip[0], rows = ip[1];
    int ox = ip[2], oy = ip[3];
    int cw = ip[4], ch = ip[5];
    int ps = ip[6], gw = ip[7], gh = ip[8], pad = ip[9];

    int panel_w = cols * cw, panel_h = rows * ch;
    int lx = (int)gid.x - ox;
    int ly = (int)gid.y - oy;

    // Outside the padded panel: leave `pack`'s pixel in place.
    if (lx < -pad || ly < -pad || lx >= panel_w + pad || ly >= panel_h + pad) return;

    bool inside = lx >= 0 && ly >= 0 && lx < panel_w && ly < panel_h;
    // Opaque card: a dark interior with a slightly lighter border ring.
    float3 col = inside ? float3(0.03f, 0.04f, 0.06f) : float3(0.10f, 0.12f, 0.16f);

    if (inside) {
        int cx = lx / cw, cy = ly / ch;
        int fx = (lx - cx * cw) / ps;      // font column
        int fy = (ly - cy * ch) / ps;      // font row
        if (fx < gw && fy < gh && cx < cols && cy < rows) {
            int slot = cells[cy * cols + cx];
            if (slot > 0 && slot < ip[10]) {
                float on = atlas[(slot * gh + fy) * gw + fx];
                if (on > 0.5f) {
                    // Attribute -> colour. 1 dim label, 2 highlighted value,
                    // 3 title, else normal.
                    int a = attr[cy * cols + cx];
                    if      (a == 3) col = float3(1.0f, 0.86f, 0.42f);   // title amber
                    else if (a == 2) col = float3(0.45f, 0.92f, 1.0f);   // highlight cyan
                    else if (a == 1) col = float3(0.62f, 0.66f, 0.72f);  // dim label
                    else             col = float3(0.90f, 0.92f, 0.96f);  // normal
                }
            }
        }
    }
    dst.write(float4(col, 1.0f), gid);
}

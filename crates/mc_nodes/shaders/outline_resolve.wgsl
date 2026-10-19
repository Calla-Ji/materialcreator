// SPDX-License-Identifier: MIT OR Apache-2.0
// outline_resolve.wgsl — Filters / Spatial / Outline (resolve pass)
//
// Turns the propagated nearest-interface-midpoint seed into the node's
// stroke value: `E = clamp(width_px/2 + 0.5 - dist_px, 0, 1)`, exactly
// EDGE_OUTLINE_SPEC.md's own formula. `dist_px` is recovered from the
// doubled-coordinate seed (see `outline_seed.wgsl`/`outline_jfa.wgsl`'s own
// doc comments) by scaling this pixel's own coordinate by 2 and halving the
// final distance back to real pixel units — this is what makes `width = 0`
// read as an exact, everywhere-zero result (the closest any real pixel can
// ever be to an interface's own segment-center seed is exactly 0.5px, since
// seeds only ever sit at half-pixel offsets from pixel centers) and `width =
// 1` split exactly 0.5/0.5 across the two interface-adjacent pixels, both
// matching EDGE_OUTLINE_SPEC.md's own worked examples exactly for a long,
// isolated, axis-aligned interface.
//
// `invert` is applied here only when this resolve pass is genuinely the
// pipeline's last step (`mc_nodes::outline::dispatch` decides that — no
// rounding or softness requested) — see `outline_round.wgsl`/
// `outline_blur.wgsl`'s own `invert` params for the other two places this
// can happen instead, always exactly once.
//
// Inputs:  `seed_in`.
// Output:  `output`.
// Params:  resolution, wrap_flags, width_px, invert.

struct ResolveParams {
    resolution: u32,
    wrap_flags: u32,
    width_px: f32,
    invert: u32,
}

@group(0) @binding(0) var<uniform> params: ResolveParams;
@group(0) @binding(1) var seed_in: texture_2d<i32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

const NO_SEED: i32 = -1;

fn wrapped_delta(a: i32, b: i32, size: i32, wrap: bool) -> i32 {
    var d = b - a;
    if wrap {
        let half = size / 2;
        if d > half {
            d = d - size;
        } else if d < -half {
            d = d + size;
        }
    }
    return d;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let res = i32(params.resolution);
    let res2 = res * 2;
    let packed = textureLoad(seed_in, coord, 0).r;

    var dist_px = 1e9;
    if (packed != NO_SEED) {
        let sx2 = packed / res2;
        let sy2 = packed % res2;
        let dx = wrapped_delta(coord.x * 2, sx2, res2, wraps_x(params.wrap_flags));
        let dy = wrapped_delta(coord.y * 2, sy2, res2, wraps_y(params.wrap_flags));
        dist_px = sqrt(f32(dx * dx + dy * dy)) * 0.5;
    }

    var e = clamp(params.width_px * 0.5 + 0.5 - dist_px, 0.0, 1.0);
    if (params.invert != 0u) {
        e = 1.0 - e;
    }
    textureStore(output, coord, vec4<f32>(e, e, e, 1.0));
}

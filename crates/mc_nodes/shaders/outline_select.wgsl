// SPDX-License-Identifier: MIT OR Apache-2.0
// outline_select.wgsl — Filters / Spatial / Outline (select pass)
//
// First of several passes `mc_nodes::outline::dispatch` runs. For each
// pixel, compares its own value against its right neighbor and its down
// neighbor only — EDGE_OUTLINE_SPEC.md's own "enumerate each shared
// interface once" convention — and packs the result as 2 bits (bit 0 =
// right interface selected, bit 1 = down interface selected) into an
// `r32uint` texture for `outline_seed.wgsl` to turn into Jump Flooding
// seeds.
//
// Contrast mode compares linear-weighted luminance (the standard Rec. 709
// weights, 0.2126/0.7152/0.0722) with a strict `>` threshold — equality
// never selects a pair. Label mode compares the raw R channel for exact
// equality instead
// (no threshold, no luminance): meant for an identity signal like
// Distance's own Cell ID output, where only "same value or not" matters,
// never how close two different values happen to be.
//
// The last column/row only contributes an interface if that axis wraps —
// `EDGE_OUTLINE_SPEC.md`: "the last column/row contributes a seam interface
// only if that axis wraps."
//
// Inputs:  `input_tex`.
// Output:  `select_out`.
// Params:  resolution, mode, threshold, wrap_flags.

struct SelectParams {
    resolution: u32,
    mode: u32,
    threshold: f32,
    wrap_flags: u32,
}

const MODE_CONTRAST: u32 = 0u;
const MODE_LABEL: u32 = 1u;

@group(0) @binding(0) var<uniform> params: SelectParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var select_out: texture_storage_2d<r32uint, write>;

fn value_at(coord: vec2<i32>) -> f32 {
    let c = textureLoad(input_tex, coord, 0);
    if params.mode == MODE_LABEL {
        return c.r;
    }
    return dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn selected(a: f32, b: f32) -> bool {
    if params.mode == MODE_LABEL {
        return a != b;
    }
    return abs(a - b) > params.threshold;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let size = i32(params.resolution);
    let wrap_x = wraps_x(params.wrap_flags);
    let wrap_y = wraps_y(params.wrap_flags);
    let my_val = value_at(coord);

    var bits = 0u;
    if (coord.x < size - 1) || wrap_x {
        let nx = tile_coord(coord.x + 1, size, wrap_x);
        if selected(my_val, value_at(vec2<i32>(nx, coord.y))) {
            bits = bits | 1u;
        }
    }
    if (coord.y < size - 1) || wrap_y {
        let ny = tile_coord(coord.y + 1, size, wrap_y);
        if selected(my_val, value_at(vec2<i32>(coord.x, ny))) {
            bits = bits | 2u;
        }
    }
    textureStore(select_out, coord, vec4<u32>(bits, 0u, 0u, 0u));
}

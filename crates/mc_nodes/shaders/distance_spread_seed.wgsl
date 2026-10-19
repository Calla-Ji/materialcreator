// SPDX-License-Identifier: MIT OR Apache-2.0
// distance_spread_seed.wgsl — Filters / Spatial / Distance (seed pass)
//
// First of three passes `mc_nodes::distance_spread::dispatch` runs — marks
// every pixel at or above `threshold` in `mask_tex` as a seed, packing its
// own coordinate (`x * resolution + y`) into an `r32sint` texture for
// `distance_spread_jfa.wgsl`'s Jump Flooding passes to propagate outward;
// `-1` means "no seed here."
//
// Inputs:  `mask_tex`.
// Output:  `seed_out`.
// Params:  resolution, threshold.

struct SeedParams {
    resolution: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: SeedParams;
@group(0) @binding(1) var mask_tex: texture_2d<f32>;
@group(0) @binding(2) var seed_out: texture_storage_2d<r32sint, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let mask_val = textureLoad(mask_tex, coord, 0).r;
    var packed = -1;
    if (mask_val >= params.threshold) {
        packed = coord.x * i32(params.resolution) + coord.y;
    }
    textureStore(seed_out, coord, vec4<i32>(packed, 0, 0, 0));
}

// SPDX-License-Identifier: MIT OR Apache-2.0
// outline_round.wgsl — Filters / Spatial / Outline (rounding: disk closing)
//
// Grayscale disk closing (dilate then erode, `EDGE_OUTLINE_SPEC.md`'s own
// "rounding" contract) — two entry points sharing one brute per-pixel
// O(r²) disk-footprint scan (`dx²+dy² <= r²`), since `mc_nodes::outline
// ::dispatch` always runs `dilate_main` into a scratch texture, then
// `erode_main` back out. A real, new cost: no disk morphology exists
// anywhere else in this crate. Correctness-first for v1 — radius is capped
// at `mc_nodes::outline::MAX_ROUNDING_RADIUS_PX` on the Rust side regardless
// of the graph's resolution; a separable/diamond approximation is a
// follow-up only if benchmarking actually shows the capped radius is too
// slow, not attempted up front.
//
// `invert` only ever applies on `erode_main` (the second, genuinely-final
// half of the pair) — `dilate_main` ignores its own copy of the field,
// since its own output is always an intermediate, never the pipeline's last
// step.
//
// Inputs:  `input_tex`.
// Output:  `output`.
// Params:  resolution, radius, wrap_flags, invert.

struct RoundParams {
    resolution: u32,
    radius: i32,
    wrap_flags: u32,
    invert: u32,
}

@group(0) @binding(0) var<uniform> params: RoundParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn dilate_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let wrap_x = wraps_x(params.wrap_flags);
    let wrap_y = wraps_y(params.wrap_flags);
    let r = params.radius;

    var best = 0.0;
    for (var dy = -r; dy <= r; dy = dy + 1) {
        for (var dx = -r; dx <= r; dx = dx + 1) {
            if (dx * dx + dy * dy > r * r) {
                continue;
            }
            let x = tile_coord(i32(id.x) + dx, size, wrap_x);
            let y = tile_coord(i32(id.y) + dy, size, wrap_y);
            best = max(best, textureLoad(input_tex, vec2<i32>(x, y), 0).r);
        }
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(best, best, best, 1.0));
}

@compute @workgroup_size(8, 8)
fn erode_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let wrap_x = wraps_x(params.wrap_flags);
    let wrap_y = wraps_y(params.wrap_flags);
    let r = params.radius;

    var worst = 1.0;
    for (var dy = -r; dy <= r; dy = dy + 1) {
        for (var dx = -r; dx <= r; dx = dx + 1) {
            if (dx * dx + dy * dy > r * r) {
                continue;
            }
            let x = tile_coord(i32(id.x) + dx, size, wrap_x);
            let y = tile_coord(i32(id.y) + dy, size, wrap_y);
            worst = min(worst, textureLoad(input_tex, vec2<i32>(x, y), 0).r);
        }
    }
    var v = worst;
    if (params.invert != 0u) {
        v = 1.0 - v;
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(v, v, v, 1.0));
}

// SPDX-License-Identifier: MIT OR Apache-2.0
// outline_jfa.wgsl — Filters / Spatial / Outline (Jump Flooding iteration)
//
// Same Jump Flooding shape as `distance_spread_jfa.wgsl`'s own doc comment
// describes — each pixel looks at itself and its 8 neighbors `step` texels
// away in `seed_in`, keeps whichever is genuinely closest, writes that
// winner to `seed_out`. The one real difference: seed values here are
// packed in DOUBLED integer coordinates (see `outline_seed.wgsl`'s own doc
// comment), so distance comparisons scale `coord` by 2 and unpack/wrap
// against `resolution * 2`, not `resolution`. The neighbor *search* itself
// (`tile_coord(coord + offset * step, resolution, wrap)`) stays in ordinary
// per-pixel texel space — only the seed *values* being compared are in
// doubled units, converted back to real pixel distance in
// `outline_resolve.wgsl`.
//
// Inputs:  `seed_in`.
// Output:  `seed_out`.
// Params:  resolution, step, wrap_flags.

struct JfaParams {
    resolution: u32,
    step: i32,
    wrap_flags: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: JfaParams;
@group(0) @binding(1) var seed_in: texture_2d<i32>;
@group(0) @binding(2) var seed_out: texture_storage_2d<r32sint, write>;

const NO_SEED: i32 = -1;

// Duplicated from `distance_spread_jfa.wgsl` rather than shared — narrow-
// purpose to each node's own packed-seed-coordinate protocol (here, doubled
// units; there, plain pixel units), not generically useful to other nodes.
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
    let wrap_x = wraps_x(params.wrap_flags);
    let wrap_y = wraps_y(params.wrap_flags);

    var best_packed = NO_SEED;
    var best_dist2 = 0;
    for (var oy = -1; oy <= 1; oy = oy + 1) {
        for (var ox = -1; ox <= 1; ox = ox + 1) {
            let nx = tile_coord(coord.x + ox * params.step, res, wrap_x);
            let ny = tile_coord(coord.y + oy * params.step, res, wrap_y);
            let candidate = textureLoad(seed_in, vec2<i32>(nx, ny), 0).r;
            if (candidate == NO_SEED) {
                continue;
            }
            let sx2 = candidate / res2;
            let sy2 = candidate % res2;
            let dx = wrapped_delta(coord.x * 2, sx2, res2, wrap_x);
            let dy = wrapped_delta(coord.y * 2, sy2, res2, wrap_y);
            let dist2 = dx * dx + dy * dy;
            if (best_packed == NO_SEED || dist2 < best_dist2) {
                best_packed = candidate;
                best_dist2 = dist2;
            }
        }
    }
    textureStore(seed_out, coord, vec4<i32>(best_packed, 0, 0, 0));
}

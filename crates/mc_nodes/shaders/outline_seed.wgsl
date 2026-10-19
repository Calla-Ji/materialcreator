// SPDX-License-Identifier: MIT OR Apache-2.0
// outline_seed.wgsl — Filters / Spatial / Outline (seed pass)
//
// Turns `outline_select.wgsl`'s own per-pixel 2-bit interface selection into
// Jump Flooding seeds for `outline_jfa.wgsl`/`outline_resolve.wgsl`. Every
// pixel touches up to 4 interfaces: its own right/down (read directly from
// `select_in`), and its left/up neighbors' right/down (the interface
// between this pixel and that neighbor — read from the neighbor's own
// selection bits, not duplicated into this pixel's own).
//
// Each one of those (up to 4) touching interfaces, if selected, is exactly
// 0.5 pixels away from this pixel's own center — so there's no genuine
// "nearest" to compute here, just a deterministic pick among whichever are
// selected (priority order: own-right, left-neighbor, own-down,
// up-neighbor). The winning interface's exact segment center is packed in
// DOUBLED integer coordinates (e.g. a selected right interface at (x,y) has
// its segment center at real position (x+0.5, y), packed as (2x+1, 2y)) —
// this is what lets `outline_resolve.wgsl` recover the true half-pixel
// stroke placement exactly, with no floating-point approximation, answering
// EDGE_OUTLINE_SPEC.md's own warning that "a point-seed distance transform
// can shift segment placement by half a pixel."
//
// Packing uses modulus `2 * resolution` (not `resolution`, unlike
// `distance_spread_seed.wgsl`'s own plain-pixel packing) since doubled
// coordinates range over `[0, 2*resolution)`. `NO_SEED` (-1) means none of
// this pixel's touching interfaces are selected.
//
// Inputs:  `select_in`.
// Output:  `seed_out`.
// Params:  resolution, wrap_flags.

struct SeedParams {
    resolution: u32,
    wrap_flags: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: SeedParams;
@group(0) @binding(1) var select_in: texture_2d<u32>;
@group(0) @binding(2) var seed_out: texture_storage_2d<r32sint, write>;

const NO_SEED: i32 = -1;

fn wrap2(v: i32, m: i32) -> i32 {
    return ((v % m) + m) % m;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let size = i32(params.resolution);
    let size2 = size * 2;
    let wrap_x = wraps_x(params.wrap_flags);
    let wrap_y = wraps_y(params.wrap_flags);

    let self_bits = textureLoad(select_in, coord, 0).r;
    let right_self = (self_bits & 1u) != 0u;
    let down_self = (self_bits & 2u) != 0u;

    var left_right = false;
    if (coord.x > 0) || wrap_x {
        let lx = tile_coord(coord.x - 1, size, wrap_x);
        let lbits = textureLoad(select_in, vec2<i32>(lx, coord.y), 0).r;
        left_right = (lbits & 1u) != 0u;
    }
    var up_down = false;
    if (coord.y > 0) || wrap_y {
        let uy = tile_coord(coord.y - 1, size, wrap_y);
        let ubits = textureLoad(select_in, vec2<i32>(coord.x, uy), 0).r;
        up_down = (ubits & 2u) != 0u;
    }

    var sx2 = 0;
    var sy2 = 0;
    var found = false;
    if right_self {
        sx2 = coord.x * 2 + 1;
        sy2 = coord.y * 2;
        found = true;
    } else if left_right {
        sx2 = coord.x * 2 - 1;
        sy2 = coord.y * 2;
        found = true;
    } else if down_self {
        sx2 = coord.x * 2;
        sy2 = coord.y * 2 + 1;
        found = true;
    } else if up_down {
        sx2 = coord.x * 2;
        sy2 = coord.y * 2 - 1;
        found = true;
    }

    var packed = NO_SEED;
    if found {
        sx2 = wrap2(sx2, size2);
        sy2 = wrap2(sy2, size2);
        packed = sx2 * size2 + sy2;
    }
    textureStore(seed_out, coord, vec4<i32>(packed, 0, 0, 0));
}

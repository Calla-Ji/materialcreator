// SPDX-License-Identifier: MIT OR Apache-2.0
// distance_spread_label.wgsl — Filters / Spatial / Distance (blob-label
// propagation)
//
// Gives every seed pixel a label shared by every other seed pixel in the
// same *connected* blob (4-connected: touching left/right/up/down, wrap-
// aware per Tiling Mode) — used for Luminance Jitter, so each disconnected
// island in Mask gets its own random brightness, not a per-pixel-noisy one
// (see `mc_nodes::distance_spread::DistanceSpreadParams::luminance_jitter`'s
// own doc comment for why plain per-pixel hashing can't give that).
//
// This is deliberately NOT `distance_spread_jfa.wgsl`'s own algorithm
// reused: that one finds the geometrically *nearest* seed regardless of
// whether a path of mask pixels actually connects to it (exactly right for
// measuring distance, exactly wrong for labeling — it would "tunnel" across
// a real gap between two disconnected blobs and incorrectly merge them).
// Each round here instead combines two connectivity-safe operations:
//   1. A direct (one-texel) 4-neighbor relax — only ever looks at pixels
//      truly adjacent to this one, so it can never merge disconnected
//      blobs no matter how many rounds run.
//   2. Pointer-jump compression — also adopts this pixel's own *current*
//      label's own label, one hop further down an already-established
//      chain. Harmless on its own (it only ever follows a path this same
//      relaxation already proved connected), but makes a label discovered
//      anywhere in a blob reach the whole blob in O(log(diameter)) rounds
//      instead of O(diameter) — the same reason Jump Flooding itself
//      converges in O(log n) rounds rather than a linear one.
// `mc_nodes::distance_spread::dispatch` runs this the same number of
// rounds as the distance pipeline (`step_sizes(resolution).len()`, roughly
// log2(resolution)+1) — comfortably enough for any blob up to the full
// canvas size, since this algorithm's own convergence bound is also
// logarithmic in the longest chain length.
//
// Inputs:  `label_in`.
// Output:  `label_out`.
// Params:  resolution, wrap_flags.

struct LabelParams {
    resolution: u32,
    wrap_flags: u32,
}

@group(0) @binding(0) var<uniform> params: LabelParams;
@group(0) @binding(1) var label_in: texture_2d<i32>;
@group(0) @binding(2) var label_out: texture_storage_2d<r32sint, write>;

const NO_SEED: i32 = -1;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let res = i32(params.resolution);
    let packed_self = textureLoad(label_in, coord, 0).r;
    if (packed_self == NO_SEED) {
        textureStore(label_out, coord, vec4<i32>(NO_SEED, 0, 0, 0));
        return;
    }

    let wrap_x = wraps_x(params.wrap_flags);
    let wrap_y = wraps_y(params.wrap_flags);
    var best = packed_self;

    let offsets = array<vec2<i32>, 4>(
        vec2<i32>(1, 0),
        vec2<i32>(-1, 0),
        vec2<i32>(0, 1),
        vec2<i32>(0, -1),
    );
    for (var i = 0; i < 4; i = i + 1) {
        let nx = tile_coord(coord.x + offsets[i].x, res, wrap_x);
        let ny = tile_coord(coord.y + offsets[i].y, res, wrap_y);
        let n = textureLoad(label_in, vec2<i32>(nx, ny), 0).r;
        if (n != NO_SEED) {
            best = min(best, n);
        }
    }

    // Pointer-jump compression — see this file's own doc comment.
    let jump_to = vec2<i32>(packed_self / res, packed_self % res);
    let jumped = textureLoad(label_in, jump_to, 0).r;
    if (jumped != NO_SEED) {
        best = min(best, jumped);
    }

    textureStore(label_out, coord, vec4<i32>(best, 0, 0, 0));
}

// SPDX-License-Identifier: MIT OR Apache-2.0
// blur_box.wgsl — Filters / Spatial / Blur (box-pass, sigma >= 3px)
//
// One exact sliding-window box blur pass — same technique as Bevel's own
// `bevel_box_blur.wgsl` (duplicated, not cross-called, per this crate's own
// convention of keeping each node's shaders self-contained), chosen for the
// same reason stated there: a running sum is both *exact* (a true average
// of every texel in the window, not an estimate or a subsampled stride)
// and cheap regardless of how large the radius gets — one add and one
// subtract per output texel, so total cost per line is O(resolution), not
// O(resolution * radius).
//
// `mc_nodes::blur::dispatch` runs this shader 3 times (with 3 different
// box widths solved by `kovesi_box_sizes` to approximate a target Gaussian
// sigma — Kovesi, "Fast Almost-Gaussian Filtering", DICTA 2010), each time
// as a horizontal pass then a vertical pass — 6 dispatches total. This is
// what replaces the old `blur.wgsl`'s own tap/step scheme (which
// skip-sampled past a 16px radius) and last session's own downsample/
// upsample pyramid (no longer needed: this technique is already cost-
// independent of radius, which was the pyramid's only reason to exist).
//
// Params:
//   direction — 0 = horizontal pass, 1 = vertical pass
//   radius_px — half the box's own width (so the full window is
//               2*radius_px + 1), already solved by `kovesi_box_sizes` on
//               the Rust side.
//
// One thread per row (horizontal pass) or column (vertical pass).

struct BoxBlurParams {
    resolution: u32,
    direction: u32,
    radius_px: f32,
    wrap: u32,
}

@group(0) @binding(0) var<uniform> params: BoxBlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

fn load_along(line: i32, pos: i32, size: i32, horizontal: bool, wrap: bool) -> vec4<f32> {
    let wrapped = tile_coord(pos, size, wrap);
    if horizontal {
        return textureLoad(input_tex, vec2<i32>(wrapped, line), 0);
    }
    return textureLoad(input_tex, vec2<i32>(line, wrapped), 0);
}

fn store_along(line: i32, pos: i32, horizontal: bool, value: vec4<f32>) {
    if horizontal {
        textureStore(output, vec2<i32>(pos, line), value);
    } else {
        textureStore(output, vec2<i32>(line, pos), value);
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = i32(params.resolution);
    if i32(id.x) >= size {
        return;
    }
    let line = i32(id.x);
    let horizontal = params.direction == 0u;
    let wrap = params.wrap != 0u;

    let radius = i32(round(max(params.radius_px, 0.0)));
    if radius < 1 {
        for (var pos = 0; pos < size; pos = pos + 1) {
            store_along(line, pos, horizontal, load_along(line, pos, size, horizontal, wrap));
        }
        return;
    }

    let window = 2.0 * f32(radius) + 1.0;
    var sum = vec4<f32>(0.0);
    for (var dx = -radius; dx <= radius; dx = dx + 1) {
        sum = sum + load_along(line, dx, size, horizontal, wrap);
    }
    store_along(line, 0, horizontal, sum / window);

    for (var pos = 1; pos < size; pos = pos + 1) {
        sum = sum + load_along(line, pos + radius, size, horizontal, wrap);
        sum = sum - load_along(line, pos - radius - 1, size, horizontal, wrap);
        store_along(line, pos, horizontal, sum / window);
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
// distance_spread_resolve.wgsl — Filters / Spatial / Distance (resolve pass)
//
// Turns `distance_spread_jfa.wgsl`'s own final nearest-seed texture into
// the node's real output: either a 0..1 falloff that reaches 0 at
// `distance_px` pixels away (`combine_mode == COMBINE`), or a hard cutoff
// at that same distance with no fade at all (`combine_mode == ONLY_SOURCE`
// — Substance Designer's own "Only Source" mode: "solid color only from
// the Source input", not a gradient), scaling either a flat white
// (`has_source == 0u` — no Source connected, matching Substance's own "no
// Source" mode: without one it's a pure gradient/solid fill, not a copy of
// Mask's own possibly-sub-1.0 grayscale value at the seed) or `source_tex`'s
// own value at the nearest seed (`has_source != 0u`). A pixel with no seed
// anywhere in the whole mask reads 0 either way.
//
// Also scales by a per-blob random brightness (Luminance Jitter): looks up
// `distance_spread_label.wgsl`'s own converged label at the resolved
// nearest seed, and hashes *that* (not the raw seed coordinate, which
// varies continuously across a solid region) — every pixel resolving to a
// seed in the same connected blob shares the same label, so the whole blob
// gets one consistent jittered brightness, not per-pixel noise.
//
// Inputs:  `source_tex` (ignored when `has_source == 0u` — still bound to
//          something valid, just never read), `seed_in`, `label_in`.
// Output:  `output`.
// Params:  resolution, wrap_flags, distance_px, has_source, combine_mode,
//          luminance_jitter, luminance_seed.

struct ResolveParams {
    resolution: u32,
    wrap_flags: u32,
    distance_px: f32,
    has_source: u32,
    combine_mode: u32,
    luminance_jitter: f32,
    luminance_seed: u32,
    _pad0: u32,
}

const COMBINE: u32 = 0u;
const ONLY_SOURCE: u32 = 1u;

@group(0) @binding(0) var<uniform> params: ResolveParams;
@group(0) @binding(1) var source_tex: texture_2d<f32>;
@group(0) @binding(2) var seed_in: texture_2d<i32>;
@group(0) @binding(3) var label_in: texture_2d<i32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;

const NO_SEED: i32 = -1;

// See `distance_spread_jfa.wgsl`'s own copy of this function for why it's
// duplicated rather than shared via `common.wgsl`.
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
    let packed = textureLoad(seed_in, coord, 0).r;
    if (packed == NO_SEED) {
        textureStore(output, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let seed = vec2<i32>(packed / res, packed % res);
    let dx = wrapped_delta(coord.x, seed.x, res, wraps_x(params.wrap_flags));
    let dy = wrapped_delta(coord.y, seed.y, res, wraps_y(params.wrap_flags));
    let dist_px = sqrt(f32(dx * dx + dy * dy));
    let max_distance = max(params.distance_px, 0.0001);
    let source_value = textureLoad(source_tex, seed, 0).rgb;
    let base = select(vec3<f32>(1.0, 1.0, 1.0), source_value, params.has_source != 0u);
    var strength = clamp(1.0 - dist_px / max_distance, 0.0, 1.0);
    if params.combine_mode == ONLY_SOURCE {
        // A hard cutoff at the same distance Combine fades out at, instead
        // of a gradient — "solid color only from the Source input".
        strength = select(0.0, 1.0, dist_px <= max_distance);
    }

    var luminance_scale = 1.0;
    let label = textureLoad(label_in, seed, 0).r;
    if (label != NO_SEED) {
        let label_xy = vec2<i32>(label / res, label % res);
        luminance_scale = 1.0 - params.luminance_jitter * hash2(label_xy, params.luminance_seed);
    }

    textureStore(output, coord, vec4<f32>(base * strength * luminance_scale, 1.0));
}

// SPDX-License-Identifier: MIT OR Apache-2.0
// color_match.wgsl — Filters / Adjustments / Color Match
//
// Inputs:  `input_tex` — the texture to re-color.
// Output:  `output` — same resolution and alpha as `input_tex`; each pixel's
//          own luminance looked up through a `BUCKET_COUNT`-stop color
//          gradient built from `reference` (see `color_match.rs`'s own
//          `reference_gradient`), blended back against the untouched input
//          by `amount`, then a saturation boost on top. The gradient lookup
//          itself works exactly like Gradient Map's own stop interpolation
//          — just auto-populated from a reference image's own luminance
//          bands instead of hand-authored stops, and evenly spaced so no
//          per-stop position needs to be packed/stored at all.
// Params:
//   amount         — 0 = input unchanged, 1 = fully matched.
//   saturation     — post-match color intensity, 1 = unchanged.
//   buckets        — `BUCKET_COUNT` evenly-spaced gradient stops (only
//                    `.xyz` of each used), computed once on the CPU in
//                    `color_match.rs`'s own `dispatch` — a full-texture
//                    reduction isn't something a single per-pixel compute
//                    invocation can do on its own, and `reference` itself
//                    never needs its own texture binding in this shader at
//                    all as a result.

const BUCKET_COUNT: u32 = 32u;

struct ColorMatchParams {
    resolution: u32,
    amount: f32,
    saturation: f32,
    _pad0: f32,
    buckets: array<vec4<f32>, BUCKET_COUNT>,
}

@group(0) @binding(0) var<uniform> params: ColorMatchParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

fn gradient_lookup(luma: f32) -> vec3<f32> {
    let scaled = clamp(luma, 0.0, 1.0) * f32(BUCKET_COUNT - 1u);
    let i0 = u32(floor(scaled));
    let i1 = min(i0 + 1u, BUCKET_COUNT - 1u);
    let t = scaled - f32(i0);
    return mix(params.buckets[i0].xyz, params.buckets[i1].xyz, t);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }

    let value = textureLoad(input_tex, vec2<i32>(id.xy), 0);
    let luma = dot(value.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let matched = gradient_lookup(luma);
    var blended = mix(value.rgb, matched, params.amount);

    // Saturation: scales each channel's distance from the pixel's own
    // luminance (same Rec. 709 weights as above) rather than a full
    // RGB<->HSL round trip the way `hsl_adjust.wgsl` does its own
    // Saturation param — cheaper for a plain scale, and sidesteps HSL's
    // hue-wrap edge cases entirely since nothing here ever needs an actual
    // hue angle.
    let blended_luma = dot(blended, vec3<f32>(0.2126, 0.7152, 0.0722));
    blended = blended_luma + (blended - blended_luma) * params.saturation;

    let output_rgb = clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(output_rgb, value.a));
}

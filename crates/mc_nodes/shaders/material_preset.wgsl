// SPDX-License-Identifier: MIT OR Apache-2.0
// material_preset.wgsl — Filters / Adjustments / Material Presets
//
// Inputs:  `base_color_tex`, `roughness_tex`, `metallic_tex` — each either
//          a real connected upstream texture, or (unconnected) a flat fill
//          already set to the selected preset's own value by
//          `material_preset.rs`'s own `dispatch` — see that file's own doc
//          comment for why that means this shader never needs an "is this
//          actually connected" branch at all.
// Outputs: `base_color_out`, `roughness_out`, `metallic_out` — each input
//          blended toward the preset's own value by `amount`.
// Params:
//   amount                                      — 0 = inputs unchanged,
//                                                 1 = fully the preset.
//   preset_color/preset_roughness/preset_metalness — the selected
//                                                 material's own measured
//                                                 values (see
//                                                 `material_preset.rs`'s
//                                                 own `PRESETS` table).

struct MaterialPresetParams {
    resolution: u32,
    amount: f32,
    _pad0: f32,
    _pad1: f32,
    preset_color: vec4<f32>,
    preset_roughness: f32,
    preset_metalness: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> params: MaterialPresetParams;
@group(0) @binding(1) var base_color_tex: texture_2d<f32>;
@group(0) @binding(2) var roughness_tex: texture_2d<f32>;
@group(0) @binding(3) var metallic_tex: texture_2d<f32>;
@group(0) @binding(4) var base_color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var roughness_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var metallic_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let coord = vec2<i32>(id.xy);

    let base_color_in = textureLoad(base_color_tex, coord, 0);
    let base_color = mix(base_color_in.rgb, params.preset_color.xyz, params.amount);
    textureStore(base_color_out, coord, vec4<f32>(base_color, base_color_in.a));

    let roughness_in = textureLoad(roughness_tex, coord, 0).r;
    let roughness = mix(roughness_in, params.preset_roughness, params.amount);
    textureStore(roughness_out, coord, vec4<f32>(roughness, roughness, roughness, 1.0));

    let metallic_in = textureLoad(metallic_tex, coord, 0).r;
    let metallic = mix(metallic_in, params.preset_metalness, params.amount);
    textureStore(metallic_out, coord, vec4<f32>(metallic, metallic, metallic, 1.0));
}

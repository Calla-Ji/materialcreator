// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Adjustments / Material Presets.
//!
//! Display name: "Material Presets". Tooltip: "Pick a real, measured
//! material and drive Base Color/Roughness/Metallic toward it — like
//! Substance's Base Material, but backed by real spectrometer data."
//!
//! `PRESETS` is 116 real-world materials' own measured linear-sRGB Base
//! Color, Metallic, and Roughness — Autodesk Standard Surface values from
//! [AntonPalmqvist/physically-based-api](https://github.com/AntonPalmqvist/physically-based-api)
//! (CC0 1.0; see ASSETS.md for the full credit, given anyway even though
//! CC0 doesn't require it). Only the numeric `color`/`metalness`/`roughness`
//! fields are used — IOR and the rest of that schema have no matching
//! output channel in this app (CLAUDE.md §7's Outputs list), so they're
//! left out entirely rather than silently ignored partway through.
//!
//! Unconnected inputs output the selected preset's own flat value directly;
//! a connected input is blended toward it by `amount` instead — an artist's
//! own procedural Roughness/Metallic/Base Color variation carries through
//! rather than being flattened outright, same "Amount" idea Color Match
//! already uses for its own reference match.

use crate::shaders::full_source;

const NODE_KIND: &str = "material_preset";

/// (name, linear-sRGB Base Color, Metalness, Roughness) — see this module's
/// own doc comment for where these numbers come from. Kept in the exact
/// order physically-based-api itself lists them (already alphabetical) so
/// there's no separate "did I re-sort these the same way" step to get
/// wrong; `PRESET_NAMES` below must list the same 116 names in the same
/// order (`tests::names_match_data_in_the_same_order` guards that).
pub const PRESETS: &[(&str, [f32; 3], f32, f32)] = &[
    ("Aluminum", [0.916, 0.923, 0.924], 1.0, 0.0),
    ("Aluminum (Anodized Red)", [0.6, 0.0, 0.0], 1.0, 0.3),
    ("Amber", [0.83, 0.288, 0.036], 0.0, 0.0),
    ("Asphalt (Fresh)", [0.0429, 0.0414, 0.0401], 0.0, 0.5),
    ("Banana", [0.634, 0.532, 0.111], 0.0, 0.5),
    ("Beer (Pale Lager)", [0.889, 0.775, 0.558], 0.0, 0.0),
    ("Beryllium", [0.539, 0.533, 0.534], 1.0, 0.0),
    ("Blackboard", [0.039, 0.039, 0.039], 0.0, 0.9),
    ("Blood (Deoxygenated)", [0.415, 0.0, 0.0], 0.0, 0.0),
    ("Blood (Oxygenated)", [0.644, 0.003, 0.005], 0.0, 0.0),
    ("Bone", [0.793, 0.793, 0.664], 0.0, 0.9),
    ("Brass", [0.91, 0.778, 0.423], 1.0, 0.0),
    ("Brick", [0.262, 0.095, 0.061], 0.0, 1.0),
    ("Car Paint", [0.1, 0.1, 0.1], 0.0, 0.0),
    ("Cardboard", [0.351, 0.208, 0.11], 0.0, 0.5),
    ("Carrot", [0.713, 0.17, 0.026], 0.0, 0.5),
    ("Cast Iron", [0.027, 0.026, 0.026], 1.0, 0.3),
    ("Cesium", [0.718, 0.554, 0.237], 1.0, 0.0),
    ("Charcoal", [0.02, 0.02, 0.02], 0.0, 1.0),
    ("Chocolate", [0.162, 0.091, 0.06], 0.0, 0.5),
    ("Chromium", [0.654, 0.685, 0.701], 1.0, 0.0),
    ("Clay", [0.256, 0.118, 0.091], 0.0, 0.5),
    ("Cobalt", [0.699, 0.704, 0.671], 1.0, 0.0),
    ("Coffee", [0.447, 0.133, 0.034], 0.0, 0.0),
    ("Cola", [0.358, 0.186, 0.085], 0.0, 0.0),
    ("Concrete", [0.51, 0.51, 0.51], 0.0, 0.5),
    ("Cooking Oil", [0.737911, 0.687, 0.091], 0.0, 0.0),
    ("Copper", [0.932, 0.623, 0.522], 1.0, 0.0),
    ("Cream", [0.976, 0.9, 0.725], 0.0, 0.0),
    ("Diamond", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Egg Shell (Brown)", [0.493, 0.248, 0.123], 0.0, 0.9),
    ("Egg Shell (White)", [0.61, 0.624, 0.631], 0.0, 0.9),
    ("Ethanol", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Eye (Cornea)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Eye (Lens)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Eye (Sclera)", [0.652, 0.5, 0.394], 0.0, 0.0),
    ("Gasoline", [1.0, 0.97, 0.617], 0.0, 0.0),
    ("Germanium", [0.5, 0.517, 0.465], 1.0, 0.0),
    ("Glass (Borosilicate)", [0.988, 0.992, 0.985], 0.0, 0.0),
    ("Glass (Soda-lime)", [0.984, 0.995, 0.995], 0.0, 0.0),
    ("Gold", [1.059, 0.773, 0.307], 1.0, 0.0),
    ("Grass", [0.105, 0.133, 0.041], 0.0, 0.5),
    ("Gray Card", [0.18, 0.18, 0.18], 0.0, 0.9),
    ("Honey (Crystallized)", [0.449, 0.319, 0.011], 0.0, 0.3),
    ("Honey (Liquid)", [0.831, 0.571, 0.037], 0.0, 0.0),
    ("Ice", [0.973, 0.995, 1.0], 0.0, 0.0),
    ("Iridium", [0.745, 0.734, 0.704], 1.0, 0.0),
    ("Iron", [0.53, 0.513, 0.494], 1.0, 0.0),
    ("Juice (Apple)", [0.856, 0.765, 0.569], 0.0, 0.0),
    ("Juice (Cranberry)", [0.661, 0.381, 0.28], 0.0, 0.0),
    ("Juice (Grape)", [0.348, 0.091, 0.053], 0.0, 0.0),
    ("Juice (Ruby Grapefruit)", [0.09, 0.034, 0.016], 0.0, 0.0),
    ("Juice (White Grapefruit)", [0.973, 0.966, 0.93], 0.0, 0.0),
    ("Ketchup", [0.164, 0.006, 0.002], 0.0, 0.1),
    ("Lead", [0.626, 0.64, 0.693], 1.0, 0.0),
    ("Lemon", [0.617, 0.366, 0.045], 0.0, 0.5),
    ("Lithium", [0.916, 0.89, 0.807], 1.0, 0.0),
    ("Magnesium", [0.956, 0.953, 0.95], 1.0, 0.0),
    ("Manganese", [0.606, 0.592, 0.573], 1.0, 0.0),
    ("Marble", [0.83, 0.791, 0.753], 0.0, 0.0),
    ("Mercury", [0.781, 0.78, 0.778], 1.0, 0.0),
    ("Milk", [0.815, 0.813, 0.682], 0.0, 0.0),
    ("MIT Black", [5e-05, 5e-05, 5e-05], 0.0, 1.0),
    ("Molybdenum", [0.589, 0.612, 0.594], 1.0, 0.0),
    ("Musou Black", [0.006, 0.006, 0.006], 0.0, 1.0),
    ("Nickel", [0.697, 0.641, 0.563], 1.0, 0.0),
    ("Office Paper", [0.794, 0.834, 0.884], 0.0, 0.8),
    ("Orange", [0.615, 0.205, 0.01], 0.0, 0.5),
    ("Palladium", [0.734, 0.7036, 0.662], 1.0, 0.0),
    ("Pearl", [0.8, 0.75, 0.7], 0.0, 0.35),
    ("Petroleum", [0.03, 0.027, 0.024], 0.0, 0.0),
    ("Plastic (Acrylic)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Plastic (PC)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Plastic (PET)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Plastic (PLA)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Plastic (PP)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Plastic (PUR)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Plastic (PVC)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Platinum", [0.765, 0.73, 0.676], 1.0, 0.0),
    ("Polystyrene (Foam)", [0.8393, 0.8384, 0.841], 0.0, 0.5),
    ("Porcelain", [0.745, 0.745, 0.723], 0.0, 0.0),
    ("Potassium", [0.983, 0.956, 0.906], 1.0, 0.0),
    ("Quartz", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Rubidium", [0.919, 0.859, 0.747], 1.0, 0.0),
    ("Rust", [0.181, 0.057, 0.017], 0.0, 1.0),
    ("Salt", [1.0, 1.0, 1.0], 0.0, 0.2),
    ("Sand", [0.44, 0.386, 0.23074], 0.0, 1.0),
    ("Sapphire", [0.086, 0.323, 0.855], 0.0, 0.0),
    ("Silicon", [0.345, 0.369, 0.426], 1.0, 0.0),
    ("Silver", [0.991, 0.985, 0.974], 1.0, 0.0),
    ("Skin I", [0.847, 0.638, 0.552], 0.0, 0.5),
    ("Skin II", [0.799, 0.485, 0.347], 0.0, 0.5),
    ("Skin III", [0.623, 0.433, 0.343], 0.0, 0.5),
    ("Skin IV", [0.436, 0.227, 0.131], 0.0, 0.5),
    ("Skin V", [0.283, 0.148, 0.079], 0.0, 0.5),
    ("Skin VI", [0.09, 0.05, 0.02], 0.0, 0.5),
    ("Snow", [0.85, 0.85, 0.85], 0.0, 0.5),
    ("Soap Bubble", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Sodium", [0.977, 0.962, 0.936], 1.0, 0.0),
    ("Spectralon", [0.99, 0.99, 0.99], 0.0, 0.99),
    ("Stainless Steel", [0.669, 0.639, 0.598], 1.0, 0.0),
    ("Terracotta", [0.555, 0.212, 0.11], 0.0, 1.0),
    ("Tire", [0.023, 0.023, 0.023], 0.0, 0.7),
    ("Titanium", [0.441, 0.4, 0.361], 1.0, 0.0),
    ("Titanium (Anodized)", [0.441, 0.4, 0.361], 1.0, 0.0),
    ("Toilet Paper", [0.83, 0.835, 0.784], 0.0, 0.9),
    ("Toner (Black)", [0.05, 0.05, 0.05], 0.0, 0.5),
    ("Toothpaste", [0.9325, 0.9369, 0.9286], 0.0, 0.0),
    ("Tungsten", [0.537, 0.536, 0.519], 1.0, 0.0),
    ("Vanadium", [0.534, 0.526, 0.546], 1.0, 0.0),
    ("Vodka (40%)", [1.0, 1.0, 1.0], 0.0, 0.0),
    ("Water", [0.969, 0.996, 0.997], 0.0, 0.0),
    ("Whiteboard", [0.869, 0.867, 0.771], 0.0, 0.0),
    ("Wine (Red)", [0.31, 0.081, 0.053], 0.0, 0.0),
    ("Wine (White)", [0.896, 0.885, 0.783], 0.0, 0.0),
    ("Zinc", [0.808, 0.844, 0.865], 1.0, 0.0),
];

/// Same 116 names as `PRESETS`, same order — `defs.rs`'s own Enum param
/// needs a plain `&'static [&'static str]` it can use directly, and (unlike
/// a real Rust enum, which would give this for free via its own variants)
/// a name/data table pairing like this one has no built-in way to derive
/// one list from the other in a `static` initializer, so the two are kept
/// in sync by hand instead — guarded by
/// `tests::names_match_data_in_the_same_order`.
pub const PRESET_NAMES: [&str; 116] = [
    "Aluminum",
    "Aluminum (Anodized Red)",
    "Amber",
    "Asphalt (Fresh)",
    "Banana",
    "Beer (Pale Lager)",
    "Beryllium",
    "Blackboard",
    "Blood (Deoxygenated)",
    "Blood (Oxygenated)",
    "Bone",
    "Brass",
    "Brick",
    "Car Paint",
    "Cardboard",
    "Carrot",
    "Cast Iron",
    "Cesium",
    "Charcoal",
    "Chocolate",
    "Chromium",
    "Clay",
    "Cobalt",
    "Coffee",
    "Cola",
    "Concrete",
    "Cooking Oil",
    "Copper",
    "Cream",
    "Diamond",
    "Egg Shell (Brown)",
    "Egg Shell (White)",
    "Ethanol",
    "Eye (Cornea)",
    "Eye (Lens)",
    "Eye (Sclera)",
    "Gasoline",
    "Germanium",
    "Glass (Borosilicate)",
    "Glass (Soda-lime)",
    "Gold",
    "Grass",
    "Gray Card",
    "Honey (Crystallized)",
    "Honey (Liquid)",
    "Ice",
    "Iridium",
    "Iron",
    "Juice (Apple)",
    "Juice (Cranberry)",
    "Juice (Grape)",
    "Juice (Ruby Grapefruit)",
    "Juice (White Grapefruit)",
    "Ketchup",
    "Lead",
    "Lemon",
    "Lithium",
    "Magnesium",
    "Manganese",
    "Marble",
    "Mercury",
    "Milk",
    "MIT Black",
    "Molybdenum",
    "Musou Black",
    "Nickel",
    "Office Paper",
    "Orange",
    "Palladium",
    "Pearl",
    "Petroleum",
    "Plastic (Acrylic)",
    "Plastic (PC)",
    "Plastic (PET)",
    "Plastic (PLA)",
    "Plastic (PP)",
    "Plastic (PUR)",
    "Plastic (PVC)",
    "Platinum",
    "Polystyrene (Foam)",
    "Porcelain",
    "Potassium",
    "Quartz",
    "Rubidium",
    "Rust",
    "Salt",
    "Sand",
    "Sapphire",
    "Silicon",
    "Silver",
    "Skin I",
    "Skin II",
    "Skin III",
    "Skin IV",
    "Skin V",
    "Skin VI",
    "Snow",
    "Soap Bubble",
    "Sodium",
    "Spectralon",
    "Stainless Steel",
    "Terracotta",
    "Tire",
    "Titanium",
    "Titanium (Anodized)",
    "Toilet Paper",
    "Toner (Black)",
    "Toothpaste",
    "Tungsten",
    "Vanadium",
    "Vodka (40%)",
    "Water",
    "Whiteboard",
    "Wine (Red)",
    "Wine (White)",
    "Zinc",
];

fn preset_by_name(name: &str) -> &'static (&'static str, [f32; 3], f32, f32) {
    PRESETS
        .iter()
        .find(|(preset_name, ..)| *preset_name == name)
        .unwrap_or(&PRESETS[0])
}

pub struct MaterialPresetParams {
    pub preset: &'static str,
    pub amount: f32,
}

impl Default for MaterialPresetParams {
    fn default() -> Self {
        Self {
            preset: PRESETS[0].0,
            // Not 1.0: at full Amount a connected input's own grayscale
            // variation is completely overwritten by the preset's flat
            // value, which (confirmed by the owner actually using the
            // node) reads as "nothing I wired in matters" rather than
            // "pick a material" — 0.5 keeps a freshly-added node showing
            // its own inputs' detail blended with the preset, not just a
            // flat swatch.
            amount: 0.5,
        }
    }
}

impl MaterialPresetParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let preset = match params.get("preset") {
            Some(mc_core::ParamValue::Enum(name)) => PRESETS
                .iter()
                .find(|(preset_name, ..)| preset_name == name)
                .map_or(defaults.preset, |(preset_name, ..)| preset_name),
            _ => defaults.preset,
        };
        Self {
            preset,
            amount: params
                .get("amount")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.amount),
        }
    }
}

#[expect(clippy::too_many_arguments)]
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    base_color_key: Option<&str>,
    roughness_key: Option<&str>,
    metallic_key: Option<&str>,
    base_color_out_key: &str,
    roughness_out_key: &str,
    metallic_out_key: &str,
    resolution: u32,
    params: &MaterialPresetParams,
) {
    let (_, preset_color, preset_metalness, preset_roughness) = *preset_by_name(params.preset);

    // An unconnected input is filled with the preset's own value — blending
    // that against itself by `amount` is a no-op, so the shader below needs
    // no separate "is this connected" branch at all: connected or not, it's
    // always just `mix(input, preset, amount)`.
    let base_color_default_key = format!("{base_color_out_key}__default_base_color");
    let base_color_view = match base_color_key {
        Some(key) => pool
            .view(key)
            .unwrap_or_else(|| panic!("material_preset: no texture at base_color key {key:?}"))
            .clone(),
        None => {
            gpu.fill_rgba16f(
                pool,
                &base_color_default_key,
                resolution,
                [preset_color[0], preset_color[1], preset_color[2], 1.0],
            );
            pool.view(&base_color_default_key).unwrap().clone()
        }
    };
    let roughness_default_key = format!("{roughness_out_key}__default_roughness");
    let roughness_view = match roughness_key {
        Some(key) => pool
            .view(key)
            .unwrap_or_else(|| panic!("material_preset: no texture at roughness key {key:?}"))
            .clone(),
        None => {
            gpu.fill_rgba16f(
                pool,
                &roughness_default_key,
                resolution,
                [preset_roughness; 4],
            );
            pool.view(&roughness_default_key).unwrap().clone()
        }
    };
    let metallic_default_key = format!("{metallic_out_key}__default_metallic");
    let metallic_view = match metallic_key {
        Some(key) => pool
            .view(key)
            .unwrap_or_else(|| panic!("material_preset: no texture at metallic key {key:?}"))
            .clone(),
        None => {
            gpu.fill_rgba16f(
                pool,
                &metallic_default_key,
                resolution,
                [preset_metalness; 4],
            );
            pool.view(&metallic_default_key).unwrap().clone()
        }
    };

    let shader_source = full_source("material_preset.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(
        resolution,
        params,
        preset_color,
        preset_roughness,
        preset_metalness,
    );
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("material_preset_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let base_color_out = pool
        .get_or_create(
            &gpu.device,
            base_color_out_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let roughness_out = pool
        .get_or_create(
            &gpu.device,
            roughness_out_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let metallic_out = pool
        .get_or_create(
            &gpu.device,
            metallic_out_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("material_preset_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&base_color_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&roughness_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&metallic_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&base_color_out),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&roughness_out),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&metallic_out),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("material_preset_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("material_preset_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `MaterialPresetParams` in
/// `material_preset.wgsl`: `resolution`/`amount`/2 pad floats (16 bytes),
/// then `preset_color: vec4<f32>` (16 bytes, only `.xyz` used), then
/// `preset_roughness`/`preset_metalness`/2 pad floats (16 bytes) = 48 bytes
/// total.
fn pack_params(
    resolution: u32,
    params: &MaterialPresetParams,
    preset_color: [f32; 3],
    preset_roughness: f32,
    preset_metalness: f32,
) -> [u8; 48] {
    let mut bytes = [0u8; 48];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.amount.to_le_bytes());
    bytes[16..20].copy_from_slice(&preset_color[0].to_le_bytes());
    bytes[20..24].copy_from_slice(&preset_color[1].to_le_bytes());
    bytes[24..28].copy_from_slice(&preset_color[2].to_le_bytes());
    bytes[32..36].copy_from_slice(&preset_roughness.to_le_bytes());
    bytes[36..40].copy_from_slice(&preset_metalness.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, MaterialPresetParams, PRESETS, PRESET_NAMES};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    #[test]
    fn names_match_data_in_the_same_order() {
        assert_eq!(PRESET_NAMES.len(), PRESETS.len());
        for (name, (preset_name, ..)) in PRESET_NAMES.iter().zip(PRESETS.iter()) {
            assert_eq!(
                name, preset_name,
                "PRESET_NAMES and PRESETS have drifted out of sync"
            );
        }
    }

    #[test]
    fn default_amount_is_half_not_fully_flat() {
        // A fresh node shouldn't look like "my connected inputs don't
        // matter" — 0.5, not 1.0, so a freshly dropped node still shows a
        // blend with whatever's wired in rather than a flat preset swatch.
        let params = super::MaterialPresetParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(params.amount, 0.5);
    }

    #[test]
    fn every_preset_has_in_range_metalness_and_roughness() {
        for (name, _, metalness, roughness) in PRESETS {
            assert!(
                (0.0..=1.0).contains(metalness),
                "{name}: metalness {metalness} out of 0..1"
            );
            assert!(
                (0.0..=1.0).contains(roughness),
                "{name}: roughness {roughness} out of 0..1"
            );
        }
    }

    #[test]
    fn no_duplicate_preset_names() {
        let mut seen = std::collections::HashSet::new();
        for (name, ..) in PRESETS {
            assert!(seen.insert(name), "duplicate preset name: {name}");
        }
    }

    fn render(params: &MaterialPresetParams) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &mut gpu,
            &mut pool,
            None,
            None,
            None,
            "base_color",
            "roughness",
            "metallic",
            RESOLUTION,
            params,
        );
        (
            gpu.read_rgba8(pool.texture("base_color").unwrap(), RESOLUTION),
            gpu.read_rgba8(pool.texture("roughness").unwrap(), RESOLUTION),
            gpu.read_rgba8(pool.texture("metallic").unwrap(), RESOLUTION),
        )
    }

    #[test]
    fn unconnected_inputs_output_the_flat_preset_values() {
        let (_, preset_color, preset_metalness, preset_roughness) = PRESETS
            .iter()
            .find(|(name, ..)| *name == "Gold")
            .copied()
            .unwrap();
        let (base_color, roughness, metallic) = render(&MaterialPresetParams {
            preset: "Gold",
            amount: 1.0,
        });
        let first_pixel = &base_color[0..3];
        for (channel, expected) in first_pixel.iter().zip(preset_color) {
            let expected_byte = (expected.clamp(0.0, 1.0) * 255.0).round() as u8;
            assert!(
                channel.abs_diff(expected_byte) <= 1,
                "base color channel {channel} should read as the preset's own {expected_byte}"
            );
        }
        let expected_roughness = (preset_roughness * 255.0).round() as u8;
        let expected_metalness = (preset_metalness * 255.0).round() as u8;
        assert!(roughness[0].abs_diff(expected_roughness) <= 1);
        assert!(metallic[0].abs_diff(expected_metalness) <= 1);
    }

    #[test]
    fn amount_zero_with_a_connected_input_is_a_no_op() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::uniform_color::dispatch(
            &mut gpu,
            &mut pool,
            "roughness_in",
            RESOLUTION,
            &crate::uniform_color::UniformColorParams {
                color: [0.42, 0.42, 0.42, 1.0],
            },
        );
        let before = gpu.read_rgba8(pool.texture("roughness_in").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            None,
            Some("roughness_in"),
            None,
            "base_color",
            "roughness",
            "metallic",
            RESOLUTION,
            &MaterialPresetParams {
                preset: "Gold",
                amount: 0.0,
            },
        );
        let after = gpu.read_rgba8(pool.texture("roughness").unwrap(), RESOLUTION);
        assert_eq!(
            before, after,
            "amount = 0 must not change a connected input"
        );
    }

    #[test]
    fn is_deterministic() {
        let params = MaterialPresetParams {
            preset: "Copper",
            amount: 0.6,
        };
        let a = render(&params);
        let b = render(&params);
        assert_eq!(a, b, "same preset + amount produced different pixels");
    }

    #[test]
    fn preserves_tiling_of_a_connected_input() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "roughness_in",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            None,
            Some("roughness_in"),
            None,
            "base_color",
            "roughness",
            "metallic",
            RESOLUTION,
            &MaterialPresetParams {
                preset: "Rust",
                amount: 0.5,
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("roughness").unwrap(), RESOLUTION);
        assert_tiles_seamlessly(&pixels, RESOLUTION);
    }
}

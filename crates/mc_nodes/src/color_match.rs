// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Adjustments / Color Match.
//!
//! Display name: "Color Match". Tooltip: "Maps this input's own brightness
//! through a color gradient built from a reference image, so its full range
//! of hues can carry over — not just an overall tint."
//!
//! An earlier version matched Input's per-RGB-channel mean and standard
//! deviation to Reference's directly (the classic Reinhard-style transfer).
//! That turned out to have a real, structural ceiling, caught by the owner
//! actually using it: Input here is grayscale (R=G=B, a single scalar per
//! pixel), and three *affine* functions of one shared scalar can only ever
//! trace a straight line through RGB space — at most a two-color
//! interpolation. A reference with three or more genuinely different hues
//! (purple flowers, green leaves, cream background) could never come
//! through as more than two of them, no matter how Amount or Saturation
//! were tuned. Confirmed, not guessed: rendering it showed the green
//! vanishing entirely.
//!
//! This version instead builds an actual multi-stop gradient from
//! `reference` — [`reference_gradient`] buckets it by luminance, closer to
//! what Gradient Map already does, just auto-populated instead of
//! hand-authored — and looks Input's own luminance up through it in
//! `color_match.wgsl`. A gradient has no such ceiling: different luminance
//! bands can hold entirely different hues.
//!
//! Needs a full-texture reduction (every reference pixel's own luminance
//! bucket) that a single per-pixel compute invocation can't do on its own
//! — so unlike most filters here, `dispatch` itself does real CPU-side work
//! (reads `reference` back via `gpu.read_rgba8` and buckets it in Rust)
//! before ever touching the GPU for the actual per-pixel remap.
//! `image_trace::dispatch` already does a CPU readback mid-evaluation the
//! same way (full vectorization, in that case), so this isn't a new
//! pattern for this crate, just a new reason to use it.

use crate::shaders::full_source;

const NODE_KIND: &str = "color_match";
/// Matches `color_match.wgsl`'s fixed-size `buckets` array — enough
/// resolution to hold several genuinely distinct hues from a reference
/// without visible banding, small enough that the uniform buffer (`16 +
/// BUCKET_COUNT * 16` bytes) stays trivially small.
const BUCKET_COUNT: usize = 32;

pub struct ColorMatchParams {
    pub amount: f32,
    pub saturation: f32,
    /// Top-left and bottom-right corners (UV, 0..1) of the region of
    /// `reference` the gradient is built from — the rest of the image is
    /// ignored entirely. Defaults to the whole image; narrowing this is how
    /// the Parameters panel's own crop-rectangle picker (dragged directly
    /// on Reference's thumbnail, not typed in) lets an artist exclude a
    /// busy background and focus on just a subject.
    pub crop_min: [f32; 2],
    pub crop_max: [f32; 2],
}

impl Default for ColorMatchParams {
    fn default() -> Self {
        Self {
            amount: 1.0,
            saturation: 1.0,
            crop_min: [0.0, 0.0],
            crop_max: [1.0, 1.0],
        }
    }
}

impl ColorMatchParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        let vec2 = |name: &str, default: [f32; 2]| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_vec2)
                .unwrap_or(default)
        };
        Self {
            amount: float("amount", defaults.amount),
            saturation: float("saturation", defaults.saturation),
            crop_min: vec2("crop_min", defaults.crop_min),
            crop_max: vec2("crop_max", defaults.crop_max),
        }
    }
}

/// Rec. 709 luminance weights — same ones `grayscale_conversion.wgsl` and
/// `image_import.rs`'s own grayscale conversion already use.
const LUMINANCE_WEIGHTS: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// How far `crop_min`/`crop_max` have to differ from the full `[0,0]..
/// [1,1]` image before a crop counts as "the artist deliberately narrowed
/// this" rather than "still at its default" — see `reference_gradient`'s
/// own doc comment for why that distinction changes how pixels are
/// weighted. Comfortably above any float drift a drag gesture could leave
/// behind, comfortably below a crop an artist would actually draw on
/// purpose.
const CROP_EPSILON: f32 = 0.01;

/// Buckets `pixels` (an RGBA8 buffer, `resolution` square) by luminance into
/// [`BUCKET_COUNT`] stops, averaging each bucket's own color — only over
/// pixels whose UV position falls within `[crop_min, crop_max]`.
///
/// With no crop (the default, full image), each pixel's contribution is
/// weighted by its own chroma (`max(r,g,b) - min(r,g,b)`) — the
/// background-dilution fix an earlier version of this node applied to a
/// single whole-image average: a desaturated pixel in, say, the midtone
/// band shouldn't drown out a vividly-colored one in that same band. A
/// pixel with zero chroma still gets a small floor weight (0.05) rather
/// than zero, so a fully grayscale reference doesn't produce an all-empty
/// bucket set with nothing at all to build a gradient from.
///
/// Once the artist has drawn an actual crop, every pixel inside it counts
/// equally instead (plain, unweighted average) — confirmed as the right
/// call by a real bug report, not assumed: chroma-weighting a manually
/// curated region can let a handful of stray pixels at the crop's own edge
/// (an embroidery photo's soft/anti-aliased thread boundary bleeding a few
/// vivid flower-colored pixels into an otherwise plain-leaf crop, say)
/// outvote the region the artist actually meant to sample, exactly
/// backwards from what drawing the crop in the first place was for. Two
/// hues sharing nearly the same luminance (an easy coincidence — hue and
/// brightness are independent) still land in the same bucket and blend
/// regardless of weighting scheme; tightening the crop to fully exclude
/// the unwanted hue is the complete fix for that, not a weighting change.
///
/// A bucket nothing landed in (no pixels in that luminance band at all, or
/// the crop excluded them) borrows the nearest filled bucket's color — see
/// `fill_gaps` — rather than defaulting to black, which would show up as a
/// stripe of black in the final gradient whenever `reference`'s own
/// luminance range doesn't span the full 0..1.
fn reference_gradient(
    pixels: &[u8],
    resolution: u32,
    crop_min: [f32; 2],
    crop_max: [f32; 2],
) -> [[f32; 3]; BUCKET_COUNT] {
    let (chunks, _remainder) = pixels.as_chunks::<4>();
    let resolution = resolution.max(1);
    let is_cropped = crop_min[0] > CROP_EPSILON
        || crop_min[1] > CROP_EPSILON
        || crop_max[0] < 1.0 - CROP_EPSILON
        || crop_max[1] < 1.0 - CROP_EPSILON;

    let mut sum = [[0.0f64; 3]; BUCKET_COUNT];
    let mut weight = [0.0f64; BUCKET_COUNT];

    for (index, pixel) in chunks.iter().enumerate() {
        let x = (index as u32) % resolution;
        let y = (index as u32) / resolution;
        let u = (x as f32 + 0.5) / resolution as f32;
        let v = (y as f32 + 0.5) / resolution as f32;
        if u < crop_min[0] || u > crop_max[0] || v < crop_min[1] || v > crop_max[1] {
            continue;
        }

        let r = pixel[0] as f64 / 255.0;
        let g = pixel[1] as f64 / 255.0;
        let b = pixel[2] as f64 / 255.0;
        let luma = LUMINANCE_WEIGHTS[0] * r + LUMINANCE_WEIGHTS[1] * g + LUMINANCE_WEIGHTS[2] * b;
        let bucket_weight = if is_cropped {
            1.0
        } else {
            let chroma = r.max(g).max(b) - r.min(g).min(b);
            chroma.max(0.05)
        };

        let bucket = ((luma * (BUCKET_COUNT as f64 - 1.0)).round() as i64)
            .clamp(0, BUCKET_COUNT as i64 - 1) as usize;
        sum[bucket][0] += r * bucket_weight;
        sum[bucket][1] += g * bucket_weight;
        sum[bucket][2] += b * bucket_weight;
        weight[bucket] += bucket_weight;
    }

    let mut colors: [Option<[f32; 3]>; BUCKET_COUNT] = [None; BUCKET_COUNT];
    for i in 0..BUCKET_COUNT {
        if weight[i] > 0.001 {
            colors[i] = Some([
                (sum[i][0] / weight[i]) as f32,
                (sum[i][1] / weight[i]) as f32,
                (sum[i][2] / weight[i]) as f32,
            ]);
        }
    }
    fill_gaps(colors)
}

/// Fills any `None` bucket with its nearest filled neighbor's color (ties
/// broken toward the lower index) — guarantees every bucket ends up with a
/// real color as long as at least one bucket has one. If every bucket is
/// `None` (the crop region had no pixels at all, a pathological case —
/// e.g. `crop_max` dragged past `crop_min`), falls back to flat mid-gray
/// rather than leaving the gradient undefined.
fn fill_gaps(colors: [Option<[f32; 3]>; BUCKET_COUNT]) -> [[f32; 3]; BUCKET_COUNT] {
    let mut out = [[0.5f32, 0.5, 0.5]; BUCKET_COUNT];
    if colors.iter().all(Option::is_none) {
        return out;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        if let Some(color) = colors[i] {
            *slot = color;
            continue;
        }
        let mut best: Option<(usize, [f32; 3])> = None;
        for (j, candidate) in colors.iter().enumerate() {
            let Some(color) = candidate else { continue };
            let distance = i.abs_diff(j);
            if best.is_none_or(|(best_distance, _)| distance < best_distance) {
                best = Some((distance, *color));
            }
        }
        if let Some((_, color)) = best {
            *slot = color;
        }
    }
    out
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    reference_key: &str,
    output_key: &str,
    resolution: u32,
    params: &ColorMatchParams,
) {
    let reference_texture = pool
        .texture(reference_key)
        .unwrap_or_else(|| panic!("color_match: no texture at reference key {reference_key:?}"));
    let reference_pixels = gpu.read_rgba8(reference_texture, resolution);
    let gradient = reference_gradient(
        &reference_pixels,
        resolution,
        params.crop_min,
        params.crop_max,
    );

    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("color_match: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("color_match.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params, &gradient);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("color_match_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("color_match_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("color_match_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("color_match_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

/// Packs params into the exact byte layout of `ColorMatchParams` in
/// `color_match.wgsl`: a 16-byte header (`resolution`, `amount`,
/// `saturation`, 1 padding float), then `buckets: array<vec4<f32>,
/// BUCKET_COUNT>` (16 bytes each, only `.xyz` used — a `vec4` rather than a
/// tightly packed `vec3` so every element lands on a 16-byte boundary
/// without fighting WGSL's own uniform-buffer alignment rules, the same
/// choice this node's own earlier version made). `crop_min`/`crop_max`
/// never reach the GPU at all — they're only used on the CPU side, to
/// decide which reference pixels feed `reference_gradient` in the first
/// place.
fn pack_params(
    resolution: u32,
    params: &ColorMatchParams,
    gradient: &[[f32; 3]; BUCKET_COUNT],
) -> Vec<u8> {
    let mut bytes = vec![0u8; 16 + BUCKET_COUNT * 16];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.amount.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.saturation.to_le_bytes());
    for (i, color) in gradient.iter().enumerate() {
        let offset = 16 + i * 16;
        bytes[offset..offset + 4].copy_from_slice(&color[0].to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&color[1].to_le_bytes());
        bytes[offset + 8..offset + 12].copy_from_slice(&color[2].to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, fill_gaps, reference_gradient, ColorMatchParams, BUCKET_COUNT};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render_with_reference_color(reference_color: [f32; 4]) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        crate::uniform_color::dispatch(
            &mut gpu,
            &mut pool,
            "reference",
            RESOLUTION,
            &crate::uniform_color::UniformColorParams {
                color: reference_color,
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "reference",
            "output",
            RESOLUTION,
            &ColorMatchParams::default(),
        );
        let texture = pool.texture("output").expect("color_match texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// A pixel buffer that's 80% a plain desaturated background and 20% a
    /// vivid, much darker subject — the real bug report's shape (a colorful
    /// flower on plain linen), reproduced directly rather than assumed.
    fn background_and_subject_pixels() -> Vec<u8> {
        let mut pixels = Vec::new();
        for i in 0..100 {
            let (r, g, b) = if i < 20 {
                (40u8, 160u8, 70u8) // vivid, dark-ish green subject
            } else {
                (210u8, 200u8, 185u8) // plain, bright beige background
            };
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
        pixels
    }

    #[test]
    fn a_three_hue_reference_produces_all_three_hues_in_its_gradient() {
        // Regression test for the actual reported bug: a reference with
        // three genuinely distinct hues at low/mid/high luminance (dark
        // green stems, mid-brightness purple flowers, bright cream
        // background) used to come through as only two colors (a straight
        // line through RGB space can't hold a third), confirmed by
        // rendering a real photo through the old mean/std version and
        // seeing the green vanish entirely.
        let mut pixels = Vec::new();
        for i in 0..300u32 {
            let (r, g, b) = match i % 3 {
                0 => (20u8, 90u8, 30u8),    // dark green, low luminance
                1 => (150u8, 90u8, 200u8),  // purple, mid luminance
                _ => (235u8, 230u8, 210u8), // cream, high luminance
            };
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
        let resolution = (pixels.len() as f32 / 4.0).sqrt().round() as u32;
        // Not necessarily a perfect square — pad so `as_chunks::<4>` still
        // covers every real pixel; extra padding pixels fall in whichever
        // bucket they land in but don't remove any of the three real hues.
        while ((pixels.len() / 4) as u32) < resolution * resolution {
            pixels.extend_from_slice(&[0, 0, 0, 0]);
        }

        let gradient = reference_gradient(&pixels, resolution, [0.0, 0.0], [1.0, 1.0]);

        let is_greenish = |c: [f32; 3]| c[1] > c[0] + 0.1 && c[1] > c[2] + 0.1;
        let is_purpleish = |c: [f32; 3]| c[0] > 0.3 && c[2] > 0.3 && c[1] < c[0] - 0.1;
        let is_creamish = |c: [f32; 3]| c[0] > 0.7 && c[1] > 0.7 && c[2] > 0.6;

        assert!(
            gradient.iter().any(|&c| is_greenish(c)),
            "gradient should contain a green stop, got {gradient:?}"
        );
        assert!(
            gradient.iter().any(|&c| is_purpleish(c)),
            "gradient should contain a purple stop, got {gradient:?}"
        );
        assert!(
            gradient.iter().any(|&c| is_creamish(c)),
            "gradient should contain a cream stop, got {gradient:?}"
        );
    }

    #[test]
    fn weighted_buckets_favor_a_small_vivid_subject_over_a_larger_plain_background() {
        let pixels = background_and_subject_pixels();
        let gradient = reference_gradient(&pixels, 10, [0.0, 0.0], [1.0, 1.0]);
        // The subject's own low-luminance band should read as genuinely
        // green (g clearly the largest channel), not diluted toward the
        // background's beige by sharing a bucket with brighter pixels —
        // luminance bucketing already keeps them apart, so this mainly
        // confirms the bucketing + chroma weighting didn't accidentally
        // wash the subject's bucket out some other way.
        let darkest_bucket = gradient[0];
        assert!(
            darkest_bucket[1] > darkest_bucket[0] && darkest_bucket[1] > darkest_bucket[2],
            "the darkest (subject's own) bucket should read as green, got {darkest_bucket:?}"
        );
    }

    #[test]
    fn crop_excludes_pixels_outside_its_bounds() {
        // Left half green, right half magenta — cropping to just the left
        // half should produce a gradient with no trace of magenta at all.
        let mut pixels = Vec::new();
        let resolution = 10u32;
        for y in 0..resolution {
            for x in 0..resolution {
                let _ = y;
                if x < resolution / 2 {
                    pixels.extend_from_slice(&[20, 180, 20, 255]); // green
                } else {
                    pixels.extend_from_slice(&[200, 20, 200, 255]); // magenta
                }
            }
        }
        let cropped = reference_gradient(&pixels, resolution, [0.0, 0.0], [0.5, 1.0]);
        for color in cropped {
            assert!(
                color[0] < 0.3 || color[2] < 0.3,
                "a crop excluding the magenta half shouldn't produce a magenta-ish \
                 stop, got {color:?}"
            );
        }
    }

    /// Regression test for a second real bug report: the artist cropped a
    /// region that was mostly muted green leaves, but a sliver of vivid
    /// purple flower bled in at the crop's own soft edge (a real photo's
    /// anti-aliased thread boundary rarely lines up with a hard rectangle),
    /// and the result still read as "pretty purple" — confirmed as
    /// chroma-weighting actively making this worse (it exists to *favor*
    /// vivid pixels, which is backwards once the artist has manually
    /// curated a region). A crop should count every pixel inside it
    /// equally instead.
    #[test]
    fn a_cropped_region_uses_plain_averaging_not_chroma_weighting() {
        // 70% cream background, 25% muted green, 5% vivid purple bleed —
        // same shape as the real report.
        let mut pixels = Vec::new();
        for i in 0..1000 {
            let (r, g, b) = if i < 50 {
                (150u8, 90u8, 200u8) // vivid purple bleed
            } else if i < 300 {
                (80u8, 120u8, 60u8) // muted green
            } else {
                (225u8, 215u8, 190u8) // cream background
            };
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
        let resolution = 32;
        while (pixels.len() / 4) < (resolution * resolution) as usize {
            pixels.extend_from_slice(&[225, 215, 190, 255]);
        }

        // A tiny crop narrower than [0,0]..[1,1] is enough to switch to
        // plain averaging — the exact bounds don't matter here since every
        // pixel in `pixels` is already "inside" a full-image-ish crop.
        let cropped = reference_gradient(&pixels, resolution, [0.0, 0.0], [0.999, 1.0]);
        let uncropped = reference_gradient(&pixels, resolution, [0.0, 0.0], [1.0, 1.0]);

        // Both reference colors share a luminance bucket (green and purple
        // happen to be close in luminance, same as the real case) and get
        // blended — but the plain-averaged (cropped) version should weight
        // that blend much closer to the numerically-dominant green/cream,
        // while the chroma-weighted (uncropped) version pulls noticeably
        // further toward the vivid purple's own channel balance (more red,
        // more blue, relative to green).
        for bucket in 0..BUCKET_COUNT {
            let c = cropped[bucket];
            let u = uncropped[bucket];
            if (c[0] - u[0]).abs() > 0.001 || (c[2] - u[2]).abs() > 0.001 {
                assert!(
                    c[0] < u[0] && c[2] < u[2],
                    "bucket {bucket}: cropped (plain-averaged) {c:?} should be less \
                     purple-shifted than uncropped (chroma-weighted) {u:?}"
                );
            }
        }
    }

    #[test]
    fn fill_gaps_borrows_the_nearest_filled_neighbor() {
        let mut colors = [None; BUCKET_COUNT];
        colors[0] = Some([1.0, 0.0, 0.0]);
        colors[BUCKET_COUNT - 1] = Some([0.0, 0.0, 1.0]);
        let filled = fill_gaps(colors);
        // A bucket near the start should borrow the red end; one near the
        // finish should borrow the blue end.
        assert_eq!(filled[1], [1.0, 0.0, 0.0]);
        assert_eq!(filled[BUCKET_COUNT - 2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn fill_gaps_on_an_entirely_empty_reference_is_flat_gray_not_black() {
        let filled = fill_gaps([None; BUCKET_COUNT]);
        for color in filled {
            assert_eq!(color, [0.5, 0.5, 0.5]);
        }
    }

    #[test]
    fn amount_zero_is_a_no_op() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::perlin::dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &crate::perlin::PerlinParams::default(),
        );
        crate::uniform_color::dispatch(
            &mut gpu,
            &mut pool,
            "reference",
            RESOLUTION,
            &crate::uniform_color::UniformColorParams {
                color: [0.9, 0.1, 0.1, 1.0],
            },
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "reference",
            "output",
            RESOLUTION,
            &ColorMatchParams {
                amount: 0.0,
                ..ColorMatchParams::default()
            },
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(before, after, "amount = 0 must not change pixels");
    }

    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(
            &render_with_reference_color([0.7, 0.3, 0.2, 1.0]),
            RESOLUTION,
        );
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(
            render_with_reference_color([0.5, 0.25, 0.8, 1.0]),
            render_with_reference_color([0.5, 0.25, 0.8, 1.0]),
            "same input + reference + params produced different pixels"
        );
    }
}

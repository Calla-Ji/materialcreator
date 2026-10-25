// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Image Import.
//!
//! Display name: "Image Import". Tooltip: "Load your own color or
//! grayscale image to use as a texture or pattern source."
//!
//! Unlike every other node, this one does real file I/O and image decoding
//! at dispatch time instead of running a WGSL compute shader — there's
//! nothing to generate procedurally, the pixels already exist in a file.
//! The loaded image is stretched to fill the tile (`image` crate's own
//! resize, not tiled/repeated), then converted from sRGB to this engine's
//! internal linear convention (every node texture in the pool is linear —
//! see `mc_app::display_encode`'s own doc comment — and most real-world
//! photos/textures are sRGB-encoded, so leaving them as-is would read too
//! dark everywhere downstream).

use std::collections::BTreeMap;

use base64::Engine;

/// A real decode/read failure — distinct from "nothing's loaded yet" (not
/// an error, see [`load_pixels`]'s own doc comment). `dispatch` still
/// always writes *some* texture even when this is produced (the usual flat
/// gray fallback) — a node with no output texture at all risks a hard
/// panic in whatever's wired downstream of it (several nodes `.expect()` a
/// pool entry existing for a connected input), and skipping this node's
/// own `mark_evaluated` on failure would mean it keeps re-reading and
/// re-decoding on *every single frame* forever rather than only when the
/// user actually changes something — so this is purely a diagnostic
/// side-channel (surfaced via `mc_app::params_panel`'s own "load_error"
/// computed param), not a signal that changes graph evaluation itself.
/// Before this existed, a decode failure only ever reached `log::warn!`
/// (invisible outside a terminal) while the node quietly kept showing a
/// flat gray tile and the Parameters panel kept claiming "Linked"/
/// "Embedded" regardless — nothing distinguished a genuine failure from a
/// node nobody's loaded anything into yet.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("{0}")]
pub struct ImageImportError(String);

#[derive(Default)]
pub struct ImageImportParams {
    pub image: mc_core::ImageRef,
    /// Also drives this node's *effective output type* (Gray instead of
    /// Color) in `mc_nodes::defs::effective_output_type` — keep any change
    /// to the meaning of this field in sync with that.
    pub grayscale: bool,
}

impl ImageImportParams {
    pub fn from_graph(params: &BTreeMap<String, mc_core::ParamValue>) -> Self {
        Self {
            image: params
                .get("image")
                .and_then(mc_core::ParamValue::as_image)
                .cloned()
                .unwrap_or_default(),
            grayscale: params
                .get("grayscale")
                .and_then(mc_core::ParamValue::as_bool)
                .unwrap_or(false),
        }
    }
}

fn srgb_to_linear_u8(value: u8) -> u8 {
    let srgb = value as f32 / 255.0;
    let linear = if srgb <= 0.040_45 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    };
    (linear.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The source bytes to decode: embedded (self-contained in the graph) takes
/// priority over the external path, matching `ImageRef`'s own doc comment
/// (embedding is meant to replace needing the file again, not supplement
/// it). `Ok(None)` means nothing's configured at all (not an error, see
/// `load_pixels`); `Err` means something *is* configured but couldn't be
/// read — the embedded base64 is corrupt, or the linked file is missing,
/// moved, or unreadable.
fn source_bytes(image_ref: &mc_core::ImageRef) -> Result<Option<Vec<u8>>, ImageImportError> {
    if let Some(encoded) = &image_ref.embedded_base64 {
        return base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map(Some)
            .map_err(|err| ImageImportError(format!("embedded image data is corrupt: {err}")));
    }
    if let Some(path) = &image_ref.path {
        return std::fs::read(path)
            .map(Some)
            .map_err(|err| ImageImportError(format!("couldn't read {path}: {err}")));
    }
    Ok(None)
}

/// Rec. 709 luminance weights — same ones `grayscale_conversion.wgsl` and
/// `viewport3d.wgsl` already use, so a grayscale-converted import reads the
/// same as the dedicated Grayscale Conversion node would produce.
const LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

fn flat_gray(resolution: u32) -> Vec<u8> {
    [128, 128, 128, 255].repeat((resolution * resolution) as usize)
}

/// Decodes and resizes the loaded image to `resolution x resolution` RGBA8,
/// already sRGB-to-linear converted. Always returns pixels to actually
/// write — flat neutral gray (mid-gray, opaque) both when nothing's loaded
/// yet (the same "sensible default" convention every other node's optional
/// input falls back to, CLAUDE.md §5) and, now, when a read/decode genuinely
/// fails, with the failure reason returned alongside rather than only ever
/// reaching `log::warn!` as before — see `ImageImportError`'s own doc
/// comment for why a failure doesn't change *what's written*, only what's
/// reported.
fn load_pixels(
    image_ref: &mc_core::ImageRef,
    resolution: u32,
    grayscale: bool,
) -> (Vec<u8>, Option<ImageImportError>) {
    let bytes = match source_bytes(image_ref) {
        Ok(None) => return (flat_gray(resolution), None),
        Ok(Some(bytes)) => bytes,
        Err(err) => return (flat_gray(resolution), Some(err)),
    };
    let dynamic_image = match ::image::load_from_memory(&bytes) {
        Ok(image) => image,
        Err(err) => {
            return (
                flat_gray(resolution),
                Some(ImageImportError(format!(
                    "couldn't decode image data: {err}"
                ))),
            )
        }
    };

    let resized = dynamic_image
        .resize_exact(
            resolution,
            resolution,
            ::image::imageops::FilterType::Triangle,
        )
        .to_rgba8();
    let mut pixels = resized.into_raw();
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel[0] = srgb_to_linear_u8(pixel[0]);
        pixel[1] = srgb_to_linear_u8(pixel[1]);
        pixel[2] = srgb_to_linear_u8(pixel[2]);
        if grayscale {
            let luminance = (pixel[0] as f32 * LUMINANCE_WEIGHTS[0]
                + pixel[1] as f32 * LUMINANCE_WEIGHTS[1]
                + pixel[2] as f32 * LUMINANCE_WEIGHTS[2])
                .round() as u8;
            pixel[0] = luminance;
            pixel[1] = luminance;
            pixel[2] = luminance;
        }
    }
    (pixels, None)
}

/// Always writes a texture at `key`, same as before this node could ever
/// fail — the returned `Option<ImageImportError>` is a pure diagnostic
/// side-channel (see its own doc comment), not a graph-evaluation failure.
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &ImageImportParams,
) -> Option<ImageImportError> {
    let (pixels, error) = load_pixels(&params.image, resolution, params.grayscale);

    pool.get_or_create(
        &gpu.device,
        key,
        resolution,
        wgpu::TextureFormat::Rgba8Unorm,
    );
    let texture = pool.texture(key).expect("just created above");
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(resolution * 4),
            rows_per_image: Some(resolution),
        },
        wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        },
    );
    error
}

#[cfg(test)]
mod tests {
    use super::{dispatch, ImageImportParams};
    use crate::test_support::RESOLUTION;
    use base64::Engine;

    fn render(params: &ImageImportParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let error = dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        assert_eq!(error, None, "render() is for the success-path tests only");
        let texture = pool.texture("test").expect("image_import texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// A tiny deterministic PNG, generated in-memory rather than checked
    /// into the repo — half red, half blue, 4x4 pixels (small enough that
    /// resizing up to `RESOLUTION` exercises the real resize path).
    fn sample_png_bytes() -> Vec<u8> {
        let mut img = ::image::RgbaImage::new(4, 4);
        for (x, _y, pixel) in img.enumerate_pixels_mut() {
            *pixel = if x < 2 {
                ::image::Rgba([255, 0, 0, 255])
            } else {
                ::image::Rgba([0, 0, 255, 255])
            };
        }
        let mut bytes = Vec::new();
        ::image::DynamicImage::ImageRgba8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                ::image::ImageFormat::Png,
            )
            .expect("encoding the test PNG should never fail");
        bytes
    }

    #[test]
    fn nothing_loaded_is_a_flat_neutral_gray() {
        let pixels = render(&ImageImportParams::default());
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [128, 128, 128, 255]),
            "an Image Import with nothing loaded should be flat mid-gray"
        );
    }

    #[test]
    fn embedded_image_decodes_and_fills_the_tile() {
        let encoded = base64::engine::general_purpose::STANDARD.encode(sample_png_bytes());
        let params = ImageImportParams {
            image: mc_core::ImageRef {
                path: Some("sample.png".to_string()),
                embedded_base64: Some(encoded),
            },
            grayscale: false,
        };
        let pixels = render(&params);
        let r = RESOLUTION as usize;

        // Left half should read red-ish, right half blue-ish, once
        // sRGB->linear converted (255 stays 255, 0 stays 0, so the pure
        // primaries are unaffected by that conversion).
        let left = &pixels[(r / 2 * r + r / 4) * 4..][..4];
        let right = &pixels[(r / 2 * r + 3 * r / 4) * 4..][..4];
        assert!(left[0] > 200 && left[2] < 50, "left half should be red");
        assert!(right[2] > 200 && right[0] < 50, "right half should be blue");
    }

    /// A missing/moved/unreadable linked file is a real, reported error —
    /// see `ImageImportError`'s own doc comment for why the texture itself
    /// still comes through as the usual gray fallback (graph safety) while
    /// the failure is reported as a side channel instead of only ever
    /// reaching `log::warn!` as before.
    #[test]
    fn a_missing_file_is_reported_as_an_error() {
        let params = ImageImportParams {
            image: mc_core::ImageRef {
                path: Some("/nonexistent/path/that/should/not/exist.png".to_string()),
                embedded_base64: None,
            },
            grayscale: false,
        };
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let error = dispatch(&mut gpu, &mut pool, "test", RESOLUTION, &params);
        assert!(
            error.is_some(),
            "a missing file must be reported as an error"
        );
        let texture = pool
            .texture("test")
            .expect("a texture must still be written even on failure");
        let pixels = gpu.read_rgba8(texture, RESOLUTION);
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [128, 128, 128, 255]),
            "the fallback texture should still be flat gray"
        );
    }

    #[test]
    fn embedded_takes_priority_over_path() {
        // If both are set, the embedded bytes should win (see
        // `source_bytes`'s doc comment) — give it a path that would fail to
        // read, to prove the embedded data is what's actually used.
        let encoded = base64::engine::general_purpose::STANDARD.encode(sample_png_bytes());
        let params = ImageImportParams {
            image: mc_core::ImageRef {
                path: Some("/nonexistent/path.png".to_string()),
                embedded_base64: Some(encoded),
            },
            grayscale: false,
        };
        let pixels = render(&params);
        assert!(
            !pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [128, 128, 128, 255]),
            "embedded data should be used even though the path is bogus"
        );
    }

    #[test]
    fn grayscale_toggle_makes_every_pixel_a_neutral_rgb() {
        let encoded = base64::engine::general_purpose::STANDARD.encode(sample_png_bytes());
        let params = ImageImportParams {
            image: mc_core::ImageRef {
                path: Some("sample.png".to_string()),
                embedded_base64: Some(encoded),
            },
            grayscale: true,
        };
        let pixels = render(&params);
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == p[1] && p[1] == p[2]),
            "grayscale output should have R == G == B on every pixel"
        );
        // The red and blue halves of the source image should no longer read
        // as pure red/blue — proof this is a real luminance conversion, not
        // just relabeling the port type.
        let r = RESOLUTION as usize;
        let left = &pixels[(r / 2 * r + r / 4) * 4..][..4];
        assert!(
            left[0] < 200,
            "the red half should no longer read near-255 after grayscale conversion"
        );
    }

    #[test]
    fn corrupt_embedded_base64_is_reported_as_an_error() {
        let params = ImageImportParams {
            image: mc_core::ImageRef {
                path: None,
                embedded_base64: Some("not valid base64 at all!!".to_string()),
            },
            grayscale: false,
        };
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let error = dispatch(&mut gpu, &mut pool, "test", RESOLUTION, &params);
        assert!(error.is_some());
        assert!(
            pool.texture("test").is_some(),
            "the gray fallback must still be written"
        );
    }

    #[test]
    fn undecodable_image_bytes_are_reported_as_an_error() {
        // Valid base64, but not actual image data of any kind.
        let encoded = base64::engine::general_purpose::STANDARD.encode(b"definitely not a png");
        let params = ImageImportParams {
            image: mc_core::ImageRef {
                path: None,
                embedded_base64: Some(encoded),
            },
            grayscale: false,
        };
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let error = dispatch(&mut gpu, &mut pool, "test", RESOLUTION, &params);
        assert!(error.is_some());
        assert!(
            pool.texture("test").is_some(),
            "the gray fallback must still be written"
        );
    }
}

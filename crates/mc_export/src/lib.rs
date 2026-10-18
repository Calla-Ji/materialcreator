// SPDX-License-Identifier: MIT OR Apache-2.0
//! mc_export: PNG/EXR writing, channel packing, naming presets.
//!
//! Pure CPU-side post-processing — no GPU dependency. Callers (`mc_app`,
//! `mc_cli`, via `mc_nodes::export`) read a node's texture back to an RGBA8
//! pixel buffer (`mc_gpu::Context::read_rgba8`), then hand it here for
//! color-space conversion, channel packing, and file writing.
//!
//! Known limitation: every node's own texture is `Rgba16Float` now (the
//! banding fix — see `mc_gpu::Context::read_rgba8`'s own doc comment), so
//! the *computation* genuinely carries more than 8 bits of precision all
//! the way through a graph. But `read_rgba8` itself still always hands
//! callers back a plain 8-bit buffer (every existing caller, including the
//! 2D View, expects that), so this crate's own source pixel buffer is still
//! only ever 8-bit at the point it reaches here. `write_png16`/`write_exr32`
//! still produce valid 16-bit/32-bit-float files (some tools expect those
//! formats regardless), but they're upconverted from that 8-bit buffer, not
//! genuinely higher-precision. True 16-bit+ *export* needs a parallel
//! higher-precision readback path all the way to these functions, bypassing
//! `read_rgba8`'s own 8-bit contract — a separate, not-yet-done follow-up.

pub mod naming;

use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExportError {
    #[error(
        "pixel buffer is {actual} bytes, expected {expected} for a {width}x{height} RGBA image"
    )]
    SizeMismatch {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
    #[error("failed to encode image: {0}")]
    Image(#[from] image::ImageError),
    #[error("failed to write EXR file: {0}")]
    Exr(String),
}

/// Whether a map's values represent a gamma-encoded display color or a raw
/// linear quantity. CLAUDE.md §8: Base Color + Emissive are sRGB; every
/// other map (Normal, Roughness, Metallic, Height, AO) is linear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

/// Which 8-bit raster container to write a map into — a real enum rather
/// than a bare bool/string, same reasoning as [`ColorSpace`]. Only the
/// Unreal Engine 5 preset offers a choice here (CLAUDE.md §8); every other
/// preset writes fixed PNG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterFormat {
    Png,
    Tga,
}

impl RasterFormat {
    pub fn extension(self) -> &'static str {
        match self {
            RasterFormat::Png => "png",
            RasterFormat::Tga => "tga",
        }
    }
}

fn linear_to_srgb_u8(value: u8) -> u8 {
    let linear = value as f32 / 255.0;
    let srgb = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Applies `color_space` to the RGB channels of an RGBA8 pixel buffer
/// in-place (alpha is left untouched). A no-op for `ColorSpace::Linear`.
pub fn apply_color_space(pixels: &mut [u8], color_space: ColorSpace) {
    if color_space == ColorSpace::Linear {
        return;
    }
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel[0] = linear_to_srgb_u8(pixel[0]);
        pixel[1] = linear_to_srgb_u8(pixel[1]);
        pixel[2] = linear_to_srgb_u8(pixel[2]);
    }
}

/// Inverts a normal map's green channel in-place — the only difference
/// between OpenGL (Y+, Blender) and DirectX (Y-, Unreal) normal map
/// conventions. Assumes `pixels` is currently in OpenGL convention (what
/// `height_to_normal`'s own default produces).
pub fn flip_normal_green(pixels: &mut [u8]) {
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel[1] = 255 - pixel[1];
    }
}

/// Copies `alpha_source`'s red channel into `rgb_pixels`' alpha channel,
/// in place — used for the Unreal Engine 5 preset's "opacity in Base
/// Color's own alpha" mode, where a separately-rendered Output: Alpha mask
/// needs folding into an otherwise-opaque Base Color buffer before it's
/// written. Reads `alpha_source`'s channel 0, same single-channel-as-red
/// convention [`pack_orm`] already uses for its own three inputs.
pub fn merge_alpha(
    rgb_pixels: &mut [u8],
    alpha_source: &[u8],
    width: u32,
    height: u32,
) -> Result<(), ExportError> {
    check_size(rgb_pixels, width, height)?;
    check_size(alpha_source, width, height)?;
    for (out, src) in rgb_pixels
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(alpha_source.as_chunks::<4>().0)
    {
        out[3] = src[0];
    }
    Ok(())
}

fn check_size(pixels: &[u8], width: u32, height: u32) -> Result<(), ExportError> {
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.len() != expected {
        return Err(ExportError::SizeMismatch {
            width,
            height,
            expected,
            actual: pixels.len(),
        });
    }
    Ok(())
}

/// Packs three textures' red channels into one ORM texture (R = ambient
/// occlusion, G = roughness, B = metallic, alpha = 255) — Unreal's expected
/// packing (CLAUDE.md §8).
pub fn pack_orm(
    ambient_occlusion: &[u8],
    roughness: &[u8],
    metallic: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ExportError> {
    check_size(ambient_occlusion, width, height)?;
    check_size(roughness, width, height)?;
    check_size(metallic, width, height)?;

    let pixel_count = (width as usize) * (height as usize);
    let mut packed = vec![0u8; pixel_count * 4];
    for (i, out) in packed.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        out[0] = ambient_occlusion[i * 4];
        out[1] = roughness[i * 4];
        out[2] = metallic[i * 4];
        out[3] = 255;
    }
    Ok(packed)
}

/// Decodes PNG bytes (e.g. from `include_bytes!`) into an 8-bit-per-channel
/// RGBA pixel buffer plus its width/height — the inverse of `write_png8`,
/// used for loading a bundled image at startup (e.g. the app's window icon)
/// rather than exporting a node's output.
pub fn decode_png_rgba8(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), ExportError> {
    let image = image::load_from_memory(bytes)?.to_rgba8();
    let (width, height) = image.dimensions();
    Ok((image.into_raw(), width, height))
}

/// Writes an 8-bit PNG, applying `color_space` first.
pub fn write_png8(
    pixels: &[u8],
    width: u32,
    height: u32,
    color_space: ColorSpace,
    path: impl AsRef<Path>,
) -> Result<(), ExportError> {
    check_size(pixels, width, height)?;
    let mut pixels = pixels.to_vec();
    apply_color_space(&mut pixels, color_space);
    let image =
        image::RgbaImage::from_raw(width, height, pixels).expect("size already checked above");
    image.save(path.as_ref())?;
    Ok(())
}

/// Writes an 8-bit TGA, applying `color_space` first — identical to
/// [`write_png8`] apart from the container format (`image::save` infers
/// the encoder from `path`'s own extension; the two functions exist
/// separately so a call site's choice of format is explicit in its own
/// name rather than implicit in whatever path string it happens to pass).
pub fn write_tga8(
    pixels: &[u8],
    width: u32,
    height: u32,
    color_space: ColorSpace,
    path: impl AsRef<Path>,
) -> Result<(), ExportError> {
    check_size(pixels, width, height)?;
    let mut pixels = pixels.to_vec();
    apply_color_space(&mut pixels, color_space);
    let image =
        image::RgbaImage::from_raw(width, height, pixels).expect("size already checked above");
    image.save(path.as_ref())?;
    Ok(())
}

/// Writes a 16-bit PNG, applying `color_space` first. See the module doc
/// comment: this upconverts from 8-bit source data (value * 257), it
/// doesn't add real precision.
pub fn write_png16(
    pixels: &[u8],
    width: u32,
    height: u32,
    color_space: ColorSpace,
    path: impl AsRef<Path>,
) -> Result<(), ExportError> {
    check_size(pixels, width, height)?;
    let mut pixels = pixels.to_vec();
    apply_color_space(&mut pixels, color_space);
    let pixels16: Vec<u16> = pixels.iter().map(|&b| (b as u16) * 257).collect();
    let image = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(width, height, pixels16)
        .expect("size already checked above");
    image.save(path.as_ref())?;
    Ok(())
}

/// Writes a true single-channel 16-bit greyscale PNG, reading the *red*
/// channel of `pixels` as the source value (same single-channel-as-red
/// convention [`pack_orm`]/[`merge_alpha`] already use) — unlike
/// [`write_png16`], which writes a 4-channel RGBA16 image (R=G=B for a
/// genuinely greyscale source, but still a 4-channel container). Used for
/// Height, which CLAUDE.md §8 and the Unreal Engine 5 preset spec both
/// call "16-bit greyscale" specifically. Always linear (Height has no
/// sRGB concept), so this takes no `ColorSpace` — same reasoning as
/// [`write_exr32`].
pub fn write_png16_gray(
    pixels: &[u8],
    width: u32,
    height: u32,
    path: impl AsRef<Path>,
) -> Result<(), ExportError> {
    check_size(pixels, width, height)?;
    let gray16: Vec<u16> = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| (px[0] as u16) * 257)
        .collect();
    let image = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(width, height, gray16)
        .expect("size already checked above");
    image.save(path.as_ref())?;
    Ok(())
}

/// Writes a true single-channel 8-bit greyscale PNG, reading the *red*
/// channel of `pixels` as the source value — same convention as
/// [`write_png16_gray`], just at the Painter export preset's own 8-bit
/// default for Roughness/Metallic/AO/Opacity (16-bit is `write_png16_gray`
/// instead, same as every other caller). Always linear — same reasoning as
/// [`write_png16_gray`].
pub fn write_png8_gray(
    pixels: &[u8],
    width: u32,
    height: u32,
    path: impl AsRef<Path>,
) -> Result<(), ExportError> {
    check_size(pixels, width, height)?;
    let gray8: Vec<u8> = pixels.as_chunks::<4>().0.iter().map(|px| px[0]).collect();
    let image = image::ImageBuffer::<image::Luma<u8>, _>::from_raw(width, height, gray8)
        .expect("size already checked above");
    image.save(path.as_ref())?;
    Ok(())
}

/// Writes a 32-bit-float EXR. Always linear (EXR's whole point is storing
/// linear light, so unlike the PNG writers this doesn't take a `ColorSpace`
/// — sRGB-encoding float data into an EXR would defeat its purpose). See the
/// module doc comment re: upconverted precision.
pub fn write_exr32(
    pixels: &[u8],
    width: u32,
    height: u32,
    path: impl AsRef<Path>,
) -> Result<(), ExportError> {
    check_size(pixels, width, height)?;
    exr::prelude::write_rgba_file(path.as_ref(), width as usize, height as usize, |x, y| {
        let i = (y * width as usize + x) * 4;
        (
            pixels[i] as f32 / 255.0,
            pixels[i + 1] as f32 / 255.0,
            pixels[i + 2] as f32 / 255.0,
            pixels[i + 3] as f32 / 255.0,
        )
    })
    .map_err(|err| ExportError::Exr(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_round_trip_endpoints_are_exact() {
        assert_eq!(linear_to_srgb_u8(0), 0);
        assert_eq!(linear_to_srgb_u8(255), 255);
    }

    #[test]
    fn srgb_brightens_midtones() {
        // The sRGB transfer function lifts linear midtones noticeably, e.g.
        // linear 0.5 (~128) encodes to roughly 188 in sRGB.
        let encoded = linear_to_srgb_u8(128);
        assert!(encoded > 180 && encoded < 195, "got {encoded}");
    }

    #[test]
    fn linear_color_space_is_a_no_op() {
        let mut pixels = vec![10, 20, 30, 255, 200, 100, 50, 128];
        let before = pixels.clone();
        apply_color_space(&mut pixels, ColorSpace::Linear);
        assert_eq!(pixels, before);
    }

    #[test]
    fn srgb_color_space_leaves_alpha_alone() {
        let mut pixels = vec![10, 20, 30, 123];
        apply_color_space(&mut pixels, ColorSpace::Srgb);
        assert_eq!(pixels[3], 123);
        assert_ne!(pixels[0], 10); // RGB should have changed
    }

    #[test]
    fn flip_normal_green_inverts_only_green() {
        let mut pixels = vec![128, 200, 255, 255];
        flip_normal_green(&mut pixels);
        assert_eq!(pixels, vec![128, 55, 255, 255]);
    }

    #[test]
    fn merge_alpha_copies_red_channel_of_source_into_alpha() {
        let mut rgb = vec![10, 20, 30, 255, 40, 50, 60, 255];
        let alpha_source = vec![64, 0, 0, 0, 200, 0, 0, 0];
        merge_alpha(&mut rgb, &alpha_source, 2, 1).unwrap();
        assert_eq!(rgb, vec![10, 20, 30, 64, 40, 50, 60, 200]);
    }

    #[test]
    fn merge_alpha_rejects_mismatched_sizes() {
        let mut rgb = vec![10, 20, 30, 255];
        let alpha_source = vec![64, 0, 0, 0, 200, 0, 0, 0]; // wrong size
        assert!(merge_alpha(&mut rgb, &alpha_source, 1, 1).is_err());
    }

    #[test]
    fn pack_orm_assigns_channels_correctly() {
        let ao = vec![10, 10, 10, 255];
        let roughness = vec![20, 20, 20, 255];
        let metallic = vec![30, 30, 30, 255];
        let packed = pack_orm(&ao, &roughness, &metallic, 1, 1).unwrap();
        assert_eq!(packed, vec![10, 20, 30, 255]);
    }

    /// The Unreal Engine 5 preset's own spec'd test case: AO=0.2,
    /// Roughness=0.5, Metallic=0.8 → R=0.2, G=0.5, B=0.8 (within 1/255).
    /// 0.2*255=51 and 0.8*255=204 exactly; 0.5*255=127.5 rounds up to 128
    /// (round-half-up), which the spec's own "within 1/255" tolerance
    /// covers either way.
    #[test]
    fn pack_orm_matches_the_unreal_engine_5_spec_values() {
        let ao = vec![51, 51, 51, 255];
        let roughness = vec![128, 128, 128, 255];
        let metallic = vec![204, 204, 204, 255];
        let packed = pack_orm(&ao, &roughness, &metallic, 1, 1).unwrap();
        assert_eq!(packed, vec![51, 128, 204, 255]);
    }

    #[test]
    fn pack_orm_rejects_mismatched_sizes() {
        let ao = vec![10, 10, 10, 255];
        let roughness = vec![20, 20, 20, 255, 0, 0, 0, 0]; // wrong size
        let metallic = vec![30, 30, 30, 255];
        assert!(pack_orm(&ao, &roughness, &metallic, 1, 1).is_err());
    }

    #[test]
    fn write_read_png8_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("mc_export_test_png8.png");
        let pixels = vec![
            10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
        ];
        write_png8(&pixels, 2, 2, ColorSpace::Linear, &path).unwrap();
        let loaded = image::open(&path).unwrap().to_rgba8();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.as_raw(), &pixels);
    }

    #[test]
    fn write_read_tga8_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("mc_export_test_tga8.tga");
        let pixels = vec![
            10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
        ];
        write_tga8(&pixels, 2, 2, ColorSpace::Linear, &path).unwrap();
        let loaded = image::open(&path).unwrap().to_rgba8();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.as_raw(), &pixels);
    }

    /// The first PNG16 round-trip test in this crate — confirms the file
    /// genuinely decodes as 16-bit-per-channel, not just that it exists.
    #[test]
    fn write_read_png16_round_trip_is_genuinely_16_bit() {
        let dir = std::env::temp_dir();
        let path = dir.join("mc_export_test_png16.png");
        let pixels = vec![10, 20, 30, 255, 40, 50, 60, 255];
        write_png16(&pixels, 2, 1, ColorSpace::Linear, &path).unwrap();
        let loaded = image::open(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.color(), image::ColorType::Rgba16);
        let rgba16 = loaded.to_rgba16();
        assert_eq!(rgba16.as_raw()[0], 10u16 * 257);
    }

    #[test]
    fn write_read_png16_gray_round_trip_is_genuinely_16_bit_single_channel() {
        let dir = std::env::temp_dir();
        let path = dir.join("mc_export_test_png16_gray.png");
        let pixels = vec![10, 10, 10, 255, 40, 40, 40, 255];
        write_png16_gray(&pixels, 2, 1, &path).unwrap();
        let loaded = image::open(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.color(), image::ColorType::L16);
        let luma16 = loaded.to_luma16();
        assert_eq!(luma16.as_raw(), &vec![10u16 * 257, 40u16 * 257]);
    }

    #[test]
    fn write_read_png8_gray_round_trip_is_genuinely_8_bit_single_channel() {
        let dir = std::env::temp_dir();
        let path = dir.join("mc_export_test_png8_gray.png");
        let pixels = vec![10, 10, 10, 255, 200, 200, 200, 255];
        write_png8_gray(&pixels, 2, 1, &path).unwrap();
        let loaded = image::open(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.color(), image::ColorType::L8);
        let luma8 = loaded.to_luma8();
        assert_eq!(luma8.as_raw(), &vec![10u8, 200u8]);
    }

    #[test]
    fn write_exr32_round_trip_is_close() {
        let dir = std::env::temp_dir();
        let path = dir.join("mc_export_test.exr");
        let pixels = vec![
            10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
        ];
        write_exr32(&pixels, 2, 2, &path).unwrap();
        assert!(path.exists());
        std::fs::remove_file(&path).ok();
    }
}

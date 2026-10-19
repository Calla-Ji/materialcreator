// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bridges the graph (`mc_core` + [`crate::registry`]) to `mc_export`: finds
//! each Output node present in `graph`, reads its texture back at the
//! requested export resolution, and writes files per the naming preset.

use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ExportFlowError {
    #[error(transparent)]
    Eval(#[from] crate::registry::EvalError),
    #[error(transparent)]
    Export(#[from] mc_export::ExportError),
    #[error("failed to create output directory {0:?}: {1}")]
    CreateDir(PathBuf, std::io::Error),
    #[error("failed to write {0:?}: {1}")]
    WriteFile(PathBuf, std::io::Error),
}

/// How the Unreal Engine preset's Opacity output is handled — `Off`
/// writes nothing at all for it, regardless of whether an `output_alpha`
/// node is present in the graph. Unlike the plain per-map booleans on
/// [`ExportOptions`], this is a real mode choice (`BaseColorAlpha`
/// changes what's *inside* the Base Color file, not just whether a
/// separate one gets written), so it stays its own enum rather than a
/// plain "export Opacity?" bool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpacityMode {
    #[default]
    Off,
    BaseColorAlpha,
    SeparateFile,
}

/// Which maps to export, for any preset — the owner's own request:
/// Blender used to have no per-map choice at all (every map present in
/// the graph exported unconditionally), while Unreal only let you toggle
/// its "optional" trio (Emissive/Opacity/Height), with Base Color/Normal/
/// ORM forced always-on. Now every map is a real toggle for all three: a
/// map still only exports when both its own toggle is on *and* the graph
/// actually has that Output node (an unchecked toggle never *adds* a map
/// that wasn't there; it only ever removes one that was).
///
/// Field groups: `format` and `export_base_color`/`export_normal`/
/// `export_height`/`export_emissive` apply to every preset.
/// `export_roughness`/`export_metallic`/`export_ambient_occlusion`/
/// `export_alpha` are Blender's own trio-plus-alpha, reused as-is by
/// Painter (whose Opacity channel is the same "export this if present"
/// concept as Blender's Alpha, just under a different file name — see
/// `mc_export::naming::Preset::opacity_mask_file_name`). `export_orm`/
/// `opacity` are Unreal-only — it's the only preset that packs channels.
/// The `painter_*` fields are Painter-only: unlike the booleans above,
/// these aren't "export this or not" toggles, they're *how* — which
/// normal convention, which bit depth/format, and what to call the
/// material — so each preset's own export logic only ever reads the
/// fields that actually apply to it.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Unreal-only (PNG/TGA) — every other preset always writes PNG
    /// regardless.
    pub format: mc_export::RasterFormat,

    pub export_base_color: bool,
    pub export_normal: bool,
    pub export_height: bool,
    pub export_emissive: bool,

    /// Blender's own trio, reused by Painter for the same four channels
    /// (Roughness/Metallic/AmbientOcclusion/Opacity).
    pub export_roughness: bool,
    pub export_metallic: bool,
    pub export_ambient_occlusion: bool,
    pub export_alpha: bool,

    /// Unreal-only.
    pub export_orm: bool,
    pub opacity: OpacityMode,

    /// Painter-only: the material name used in every exported file name
    /// and the subfolder they're written into. Empty means "use the
    /// graph's own file name" — either way it's run through
    /// `mc_export::naming::sanitize_material_name` before use, so a
    /// user-typed name with spaces/special characters is always safe.
    pub painter_material_name: String,
    /// Painter-only: Roughness/Metallic/AmbientOcclusion/Opacity are 8-bit
    /// grayscale PNGs by default; `true` switches all four to 16-bit.
    pub painter_grayscale_16bit: bool,
    /// Painter-only: Height is a 16-bit grayscale PNG by default; `true`
    /// switches it to a 32-bit float EXR instead.
    pub painter_height_exr: bool,
    /// Painter-only: which normal map convention to write (embedded in the
    /// file name) — OpenGL by default, with a DirectX toggle.
    pub painter_normal_format: mc_core::NormalFormat,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: mc_export::RasterFormat::Png,
            export_base_color: true,
            export_normal: true,
            export_height: true,
            export_emissive: true,
            export_roughness: true,
            export_metallic: true,
            export_ambient_occlusion: true,
            export_alpha: true,
            export_orm: true,
            // Unlike every other toggle here, Opacity defaults off: unlike
            // a plain "export this extra file or not" choice, turning it
            // on changes *where* the alpha channel lands (either folded
            // into Base Color or its own file) — a more consequential
            // default to opt into than to opt out of.
            opacity: OpacityMode::Off,
            painter_material_name: String::new(),
            painter_grayscale_16bit: false,
            painter_height_exr: false,
            painter_normal_format: mc_core::NormalFormat::OpenGl,
        }
    }
}

fn read_output(
    kind: &str,
    resolution: u32,
    graph: &mc_core::Graph,
    pool: &mc_gpu::TexturePool,
    gpu: &mc_gpu::Context,
) -> Option<Vec<u8>> {
    let id = graph
        .nodes()
        .find(|(_, n)| n.kind == kind)
        .map(|(id, _)| id)?;
    let key = crate::registry::resolve_texture_key(graph, id)?;
    let texture = pool.texture(&key)?;
    Some(gpu.read_rgba8(texture, resolution))
}

/// Whether `graph` has an Output node of this `kind` (e.g.
/// `"output_emissive"`) at all — independent of any `ExportOptions`
/// toggle. `mc_app`'s export dialog uses this directly to grey out a
/// map's checkbox when the graph has nothing to export there in the first
/// place (the owner's own friend's testing report: ticking Emissive/Alpha
/// on a graph with neither wired up silently wrote fewer files than
/// expected, with no indication why — this is "can't possibly do
/// anything" made visible before the artist even clicks Export, not just
/// explained after).
pub fn has_output(graph: &mc_core::Graph, kind: &str) -> bool {
    graph.nodes().any(|(_, n)| n.kind == kind)
}

/// Painter's effective material name for this export: `options`' own
/// `painter_material_name` if set, falling back to `graph_name` otherwise —
/// either way always sanitized, so a user-typed name with spaces or special
/// characters is safe to use directly in a file/folder name.
fn painter_material_name(graph_name: &str, options: &ExportOptions) -> String {
    let raw = options.painter_material_name.trim();
    let source = if raw.is_empty() { graph_name } else { raw };
    mc_export::naming::sanitize_material_name(source)
}

/// Fills in a flat RGBA8 buffer (every pixel's RGB = `fallback_value`,
/// alpha = 255) when `buf` is `None` — the Unreal Engine 5 preset's ORM
/// fallback rule (AO=1.0/white, Roughness=0.5, Metallic=0.0) for whichever
/// of those three channels has no Output node in the graph.
fn resolve_or_fallback(buf: Option<Vec<u8>>, fallback_value: u8, resolution: u32) -> Vec<u8> {
    buf.unwrap_or_else(|| {
        let pixel_count = (resolution as usize) * (resolution as usize);
        let mut out = vec![0u8; pixel_count * 4];
        for pixel in out.as_chunks_mut::<4>().0 {
            *pixel = [fallback_value, fallback_value, fallback_value, 255];
        }
        out
    })
}

/// Writes an 8-bit raster, picking TGA only for the Unreal Engine 5 preset
/// when `format` asks for it — Blender always writes PNG, regardless of
/// `format` (which is meaningless for it; callers pass
/// `unreal_options.format` unconditionally rather than branching on preset
/// themselves).
fn write_8bit(
    preset: mc_export::naming::Preset,
    format: mc_export::RasterFormat,
    pixels: &[u8],
    resolution: u32,
    color_space: mc_export::ColorSpace,
    path: &Path,
) -> Result<(), mc_export::ExportError> {
    if preset == mc_export::naming::Preset::Unreal && format == mc_export::RasterFormat::Tga {
        mc_export::write_tga8(pixels, resolution, resolution, color_space, path)
    } else {
        mc_export::write_png8(pixels, resolution, resolution, color_space, path)
    }
}

/// Decides which output files a given export *would* produce — graph
/// presence + toggle checks only, no GPU work — used both to drive
/// [`export_material`]'s own write-gating below and by the export dialog's
/// pre-flight overwrite check. File names only (relative), not full paths;
/// see [`planned_output_paths`] for the `out_dir`-joined version.
fn planned_file_names(
    graph: &mc_core::Graph,
    preset: mc_export::naming::Preset,
    graph_name: &str,
    options: &ExportOptions,
) -> Vec<String> {
    let ext = options.format.extension();
    let height_ext = if options.painter_height_exr {
        "exr"
    } else {
        "png"
    };
    // Painter's own name (sanitized, user-editable) stands in for
    // `graph_name` everywhere below; for Blender/Unreal this is just
    // `graph_name` itself, unchanged.
    let name = if preset == mc_export::naming::Preset::Painter {
        painter_material_name(graph_name, options)
    } else {
        graph_name.to_string()
    };
    let mut names = Vec::new();

    if options.export_base_color && has_output(graph, "output_base_color") {
        names.push(preset.base_color_file_name(&name, ext));
    }
    if options.export_normal && has_output(graph, "output_normal") {
        names.push(preset.normal_file_name(&name, ext, options.painter_normal_format));
    }
    if options.export_height && has_output(graph, "output_height") {
        names.push(preset.height_file_name(&name, height_ext));
    }

    match preset {
        mc_export::naming::Preset::Blender | mc_export::naming::Preset::Painter => {
            for (wanted, file_name) in [
                (
                    options.export_roughness && has_output(graph, "output_roughness"),
                    preset.roughness_file_name(&name),
                ),
                (
                    options.export_metallic && has_output(graph, "output_metallic"),
                    preset.metallic_file_name(&name),
                ),
                (
                    options.export_ambient_occlusion
                        && has_output(graph, "output_ambient_occlusion"),
                    preset.ambient_occlusion_file_name(&name),
                ),
            ] {
                if let (true, Some(file_name)) = (wanted, file_name) {
                    names.push(file_name);
                }
            }
        }
        mc_export::naming::Preset::Unreal => {
            // Fallbacks mean ORM is produced regardless of which (if any)
            // of AO/Roughness/Metallic actually have a node, as long as
            // the ORM toggle itself is on.
            if options.export_orm {
                if let Some(file_name) = preset.orm_file_name(&name, ext) {
                    names.push(file_name);
                }
            }
        }
    }

    if options.export_emissive && has_output(graph, "output_emissive") {
        names.push(preset.emissive_file_name(&name, ext));
    }

    match preset {
        mc_export::naming::Preset::Blender => {
            if options.export_alpha && has_output(graph, "output_alpha") {
                names.push(preset.alpha_file_name(&name));
            }
        }
        mc_export::naming::Preset::Unreal => {
            if options.opacity == OpacityMode::SeparateFile && has_output(graph, "output_alpha") {
                if let Some(file_name) = preset.opacity_mask_file_name(&name, ext) {
                    names.push(file_name);
                }
            }
            // BaseColorAlpha folds into Base Color's own file above, not a
            // file of its own; Off writes nothing for opacity at all.
        }
        mc_export::naming::Preset::Painter => {
            // Painter's Opacity reuses Blender's `export_alpha` toggle —
            // same "export this if present" intent, different file name.
            if options.export_alpha && has_output(graph, "output_alpha") {
                if let Some(file_name) = preset.opacity_mask_file_name(&name, ext) {
                    names.push(file_name);
                }
            }
        }
    }

    names
}

/// The full paths a call to [`export_material`] with these exact arguments
/// would write — without evaluating the graph or touching the GPU. Used by
/// the export dialog to check which (if any) already exist before
/// exporting for real, so nothing is silently overwritten. Painter writes
/// into its own material-named subfolder (plus `README_Painter.txt`)
/// rather than flat into `out_dir`, unlike Blender/Unreal.
pub fn planned_output_paths(
    graph: &mc_core::Graph,
    preset: mc_export::naming::Preset,
    graph_name: &str,
    out_dir: &Path,
    options: &ExportOptions,
) -> Vec<PathBuf> {
    let names = planned_file_names(graph, preset, graph_name, options);
    if preset == mc_export::naming::Preset::Painter {
        let material_dir = out_dir.join(painter_material_name(graph_name, options));
        let mut paths: Vec<PathBuf> = names
            .into_iter()
            .map(|name| material_dir.join(name))
            .collect();
        paths.push(material_dir.join("README_Painter.txt"));
        paths
    } else {
        names.into_iter().map(|name| out_dir.join(name)).collect()
    }
}

/// Re-evaluates `graph` at `resolution` (restoring its original resolution
/// afterward, so the live preview isn't left at export resolution) and
/// writes every Output node present in it to `out_dir`, named per `preset`'s
/// convention. Returns the paths written; an Output kind missing from the
/// graph is silently skipped, not an error. Every map in `options` is its
/// own toggle for *either* preset now (the owner's own request — Blender
/// used to have no per-map choice at all, Unreal only for its "optional"
/// trio) — pass `&ExportOptions::default()` for "export everything the
/// graph has."
#[allow(clippy::too_many_arguments)] // each param is independently meaningful at every call site
pub fn export_material(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
    preset: mc_export::naming::Preset,
    resolution: u32,
    graph_name: &str,
    out_dir: &Path,
    options: &ExportOptions,
) -> Result<Vec<PathBuf>, ExportFlowError> {
    // Painter's own flow — one plain file per channel into a material-named
    // subfolder plus a README, never packed — is structurally different
    // enough from Blender/Unreal's shared logic below that it gets its own
    // function rather than threading a third case through every match in
    // this one.
    if preset == mc_export::naming::Preset::Painter {
        return export_material_painter(gpu, pool, graph, resolution, graph_name, out_dir, options);
    }

    std::fs::create_dir_all(out_dir)
        .map_err(|err| ExportFlowError::CreateDir(out_dir.to_path_buf(), err))?;

    let original_resolution = graph.settings.resolution;
    graph.settings.resolution = resolution;
    crate::registry::evaluate(gpu, pool, graph, false)?;

    let ext = options.format.extension();
    let mut written = Vec::new();

    // Read early: needed by Base Color (folded into its alpha channel)
    // before it's needed again, much later, by Opacity's own separate-file
    // case.
    let alpha = read_output("output_alpha", resolution, graph, pool, gpu);

    if options.export_base_color {
        if let Some(mut pixels) = read_output("output_base_color", resolution, graph, pool, gpu) {
            if preset == mc_export::naming::Preset::Unreal
                && options.opacity == OpacityMode::BaseColorAlpha
            {
                if let Some(alpha) = &alpha {
                    mc_export::merge_alpha(&mut pixels, alpha, resolution, resolution)?;
                }
            }
            let path = out_dir.join(preset.base_color_file_name(graph_name, ext));
            write_8bit(
                preset,
                options.format,
                &pixels,
                resolution,
                mc_export::ColorSpace::Srgb,
                &path,
            )?;
            written.push(path);
        }
    }

    if options.export_normal {
        if let Some(mut pixels) = read_output("output_normal", resolution, graph, pool, gpu) {
            if preset.normal_format() != mc_core::NormalFormat::OpenGl {
                mc_export::flip_normal_green(&mut pixels);
            }
            let path =
                out_dir.join(preset.normal_file_name(graph_name, ext, preset.normal_format()));
            if preset == mc_export::naming::Preset::Unreal {
                // 8-bit per the UE5 spec — Blender's own Normal export
                // stays 16-bit below, untouched.
                write_8bit(
                    preset,
                    options.format,
                    &pixels,
                    resolution,
                    mc_export::ColorSpace::Linear,
                    &path,
                )?;
            } else {
                mc_export::write_png16(
                    &pixels,
                    resolution,
                    resolution,
                    mc_export::ColorSpace::Linear,
                    &path,
                )?;
            }
            written.push(path);
        }
    }

    if options.export_height {
        if let Some(pixels) = read_output("output_height", resolution, graph, pool, gpu) {
            let path = out_dir.join(preset.height_file_name(graph_name, "png"));
            if preset == mc_export::naming::Preset::Unreal {
                mc_export::write_png16_gray(&pixels, resolution, resolution, &path)?;
            } else {
                mc_export::write_png16(
                    &pixels,
                    resolution,
                    resolution,
                    mc_export::ColorSpace::Linear,
                    &path,
                )?;
            }
            written.push(path);
        }
    }

    let ambient_occlusion = read_output("output_ambient_occlusion", resolution, graph, pool, gpu);
    let roughness = read_output("output_roughness", resolution, graph, pool, gpu);
    let metallic = read_output("output_metallic", resolution, graph, pool, gpu);
    // An explicit Channel Packer → Output: ORM wire (so the artist can see
    // the packed result before export) takes priority over auto-packing
    // from the three separate channels below, when present.
    let explicit_orm = read_output("output_orm", resolution, graph, pool, gpu);

    match preset {
        mc_export::naming::Preset::Blender => {
            for (wanted, pixels, file_name) in [
                (
                    options.export_roughness,
                    &roughness,
                    preset.roughness_file_name(graph_name),
                ),
                (
                    options.export_metallic,
                    &metallic,
                    preset.metallic_file_name(graph_name),
                ),
                (
                    options.export_ambient_occlusion,
                    &ambient_occlusion,
                    preset.ambient_occlusion_file_name(graph_name),
                ),
            ] {
                if let (true, Some(pixels), Some(file_name)) = (wanted, pixels, file_name) {
                    let path = out_dir.join(file_name);
                    mc_export::write_png8(
                        pixels,
                        resolution,
                        resolution,
                        mc_export::ColorSpace::Linear,
                        &path,
                    )?;
                    written.push(path);
                }
            }
        }
        mc_export::naming::Preset::Unreal => {
            if options.export_orm {
                // Already packed — a straight copy, no re-deriving
                // fallbacks (the Channel Packer node applies its own,
                // identical ones). Otherwise fall back to auto-packing
                // from the three separate channels, missing inputs
                // falling back rather than skipping ORM entirely — AO
                // defaults to white (no occlusion), Roughness to a
                // neutral mid-grey, Metallic to fully non-metal.
                let packed = match explicit_orm {
                    Some(pixels) => pixels,
                    None => {
                        let ao = resolve_or_fallback(ambient_occlusion, 255, resolution);
                        let roughness = resolve_or_fallback(roughness, 128, resolution);
                        let metallic = resolve_or_fallback(metallic, 0, resolution);
                        mc_export::pack_orm(&ao, &roughness, &metallic, resolution, resolution)?
                    }
                };
                let path = out_dir.join(
                    preset
                        .orm_file_name(graph_name, ext)
                        .expect("Unreal preset always has an ORM file name"),
                );
                write_8bit(
                    preset,
                    options.format,
                    &packed,
                    resolution,
                    mc_export::ColorSpace::Linear,
                    &path,
                )?;
                written.push(path);
            }
        }
        // Unreachable: Painter already returned via `export_material_painter`
        // above. Kept explicit (not `_`) so a future preset addition forces
        // a decision here instead of silently falling through.
        mc_export::naming::Preset::Painter => {}
    }

    if options.export_emissive {
        if let Some(pixels) = read_output("output_emissive", resolution, graph, pool, gpu) {
            let path = out_dir.join(preset.emissive_file_name(graph_name, ext));
            write_8bit(
                preset,
                options.format,
                &pixels,
                resolution,
                mc_export::ColorSpace::Srgb,
                &path,
            )?;
            written.push(path);
        }
    }

    match preset {
        mc_export::naming::Preset::Blender => {
            if options.export_alpha {
                if let Some(pixels) = &alpha {
                    let path = out_dir.join(preset.alpha_file_name(graph_name));
                    mc_export::write_png8(
                        pixels,
                        resolution,
                        resolution,
                        mc_export::ColorSpace::Linear,
                        &path,
                    )?;
                    written.push(path);
                }
            }
        }
        mc_export::naming::Preset::Unreal => {
            if options.opacity == OpacityMode::SeparateFile {
                if let Some(pixels) = &alpha {
                    let path = out_dir.join(
                        preset
                            .opacity_mask_file_name(graph_name, ext)
                            .expect("Unreal preset always has an opacity mask file name"),
                    );
                    write_8bit(
                        preset,
                        options.format,
                        pixels,
                        resolution,
                        mc_export::ColorSpace::Linear,
                        &path,
                    )?;
                    written.push(path);
                }
            }
            // BaseColorAlpha already handled above, folded into Base
            // Color's own file; Off writes nothing for opacity at all.
        }
        // Unreachable: Painter already returned via `export_material_painter`
        // above — same reasoning as the earlier match in this function.
        mc_export::naming::Preset::Painter => {}
    }

    graph.settings.resolution = original_resolution;
    crate::registry::evaluate(gpu, pool, graph, false)?;

    Ok(written)
}

/// The Painter preset's own export flow: writes one plain, unpacked image
/// per channel into a subfolder named after the material, plus a
/// `README_Painter.txt` pointing each file at its fill-layer slot. Never
/// packs channels (no ORM-style file) and never writes a placeholder for a
/// channel the graph doesn't have — same skip-if-missing behavior as
/// Blender/Unreal, just with no fallback values at all (Painter has no
/// "flat neutral default" concept like Unreal's ORM fallback).
#[allow(clippy::too_many_arguments)]
fn export_material_painter(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
    resolution: u32,
    graph_name: &str,
    out_dir: &Path,
    options: &ExportOptions,
) -> Result<Vec<PathBuf>, ExportFlowError> {
    let preset = mc_export::naming::Preset::Painter;
    let material_name = painter_material_name(graph_name, options);
    let material_dir = out_dir.join(&material_name);
    std::fs::create_dir_all(&material_dir)
        .map_err(|err| ExportFlowError::CreateDir(material_dir.clone(), err))?;

    let original_resolution = graph.settings.resolution;
    graph.settings.resolution = resolution;
    crate::registry::evaluate(gpu, pool, graph, false)?;

    let height_ext = if options.painter_height_exr {
        "exr"
    } else {
        "png"
    };
    let mut written = Vec::new();
    let mut readme_lines = Vec::new();

    if options.export_base_color {
        if let Some(pixels) = read_output("output_base_color", resolution, graph, pool, gpu) {
            let path = material_dir.join(preset.base_color_file_name(&material_name, "png"));
            mc_export::write_png8(
                &pixels,
                resolution,
                resolution,
                mc_export::ColorSpace::Srgb,
                &path,
            )?;
            readme_lines.push(format!(
                "{} -> Base Color",
                path.file_name().unwrap().to_string_lossy()
            ));
            written.push(path);
        }
    }

    if options.export_normal {
        if let Some(mut pixels) = read_output("output_normal", resolution, graph, pool, gpu) {
            if options.painter_normal_format != mc_core::NormalFormat::OpenGl {
                mc_export::flip_normal_green(&mut pixels);
            }
            let path = material_dir.join(preset.normal_file_name(
                &material_name,
                "png",
                options.painter_normal_format,
            ));
            mc_export::write_png8(
                &pixels,
                resolution,
                resolution,
                mc_export::ColorSpace::Linear,
                &path,
            )?;
            let convention = match options.painter_normal_format {
                mc_core::NormalFormat::OpenGl => "OpenGL",
                mc_core::NormalFormat::DirectX => "DirectX",
            };
            readme_lines.push(format!(
                "{} -> Normal (choose \"{convention}\" as the normal format for this project)",
                path.file_name().unwrap().to_string_lossy()
            ));
            written.push(path);
        }
    }

    if options.export_height {
        if let Some(pixels) = read_output("output_height", resolution, graph, pool, gpu) {
            let path = material_dir.join(preset.height_file_name(&material_name, height_ext));
            if options.painter_height_exr {
                mc_export::write_exr32(&pixels, resolution, resolution, &path)?;
            } else {
                mc_export::write_png16_gray(&pixels, resolution, resolution, &path)?;
            }
            readme_lines.push(format!(
                "{} -> Height",
                path.file_name().unwrap().to_string_lossy()
            ));
            written.push(path);
        }
    }

    for (wanted, kind, file_name, slot) in [
        (
            options.export_roughness,
            "output_roughness",
            preset.roughness_file_name(&material_name),
            "Roughness",
        ),
        (
            options.export_metallic,
            "output_metallic",
            preset.metallic_file_name(&material_name),
            "Metallic",
        ),
        (
            options.export_ambient_occlusion,
            "output_ambient_occlusion",
            preset.ambient_occlusion_file_name(&material_name),
            "Ambient Occlusion",
        ),
    ] {
        let (true, Some(file_name)) = (wanted, file_name) else {
            continue;
        };
        if let Some(pixels) = read_output(kind, resolution, graph, pool, gpu) {
            let path = material_dir.join(file_name);
            if options.painter_grayscale_16bit {
                mc_export::write_png16_gray(&pixels, resolution, resolution, &path)?;
            } else {
                mc_export::write_png8_gray(&pixels, resolution, resolution, &path)?;
            }
            readme_lines.push(format!(
                "{} -> {slot}",
                path.file_name().unwrap().to_string_lossy()
            ));
            written.push(path);
        }
    }

    if options.export_emissive {
        if let Some(pixels) = read_output("output_emissive", resolution, graph, pool, gpu) {
            let path = material_dir.join(preset.emissive_file_name(&material_name, "png"));
            mc_export::write_png8(
                &pixels,
                resolution,
                resolution,
                mc_export::ColorSpace::Srgb,
                &path,
            )?;
            readme_lines.push(format!(
                "{} -> Emissive",
                path.file_name().unwrap().to_string_lossy()
            ));
            written.push(path);
        }
    }

    if options.export_alpha {
        if let Some(pixels) = read_output("output_alpha", resolution, graph, pool, gpu) {
            if let Some(file_name) = preset.opacity_mask_file_name(&material_name, "png") {
                let path = material_dir.join(file_name);
                if options.painter_grayscale_16bit {
                    mc_export::write_png16_gray(&pixels, resolution, resolution, &path)?;
                } else {
                    mc_export::write_png8_gray(&pixels, resolution, resolution, &path)?;
                }
                readme_lines.push(format!(
                    "{} -> Opacity",
                    path.file_name().unwrap().to_string_lossy()
                ));
                written.push(path);
            }
        }
    }

    let readme_path = material_dir.join("README_Painter.txt");
    let readme_contents = if readme_lines.is_empty() {
        "This graph had no outputs to export.\n".to_string()
    } else {
        format!(
            "Drop each file below into the matching channel slot of a fill layer in Painter.\n\n{}\n",
            readme_lines.join("\n")
        )
    };
    std::fs::write(&readme_path, readme_contents)
        .map_err(|err| ExportFlowError::WriteFile(readme_path.clone(), err))?;
    written.push(readme_path);

    graph.settings.resolution = original_resolution;
    crate::registry::evaluate(gpu, pool, graph, false)?;

    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::{export_material, planned_output_paths, ExportOptions, OpacityMode};

    fn flat_color_graph(output_kind: &str, color: [f32; 4]) -> mc_core::Graph {
        let mut graph = mc_core::Graph::new();
        let fill = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color(color)),
        );
        let output = graph.add_node(mc_core::Node::new(output_kind));
        graph.connect(fill, "output", output, "input").unwrap();
        graph.settings.resolution = 8;
        graph
    }

    fn headless() -> (mc_gpu::Context, mc_gpu::TexturePool) {
        (
            mc_gpu::Context::new_headless().expect("headless GPU context"),
            mc_gpu::TexturePool::default(),
        )
    }

    #[test]
    fn exports_alpha_with_the_right_name_for_blender() {
        let mut graph = flat_color_graph("output_alpha", [0.5, 0.5, 0.5, 1.0]);
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_alpha_blender_test");
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Blender,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        assert_eq!(written.len(), 1, "expected exactly one file written");
        assert_eq!(written[0].file_name().unwrap(), "test_Alpha.png");
        assert!(written[0].exists());
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// The Unreal Engine 5 preset's own spec'd test case: AO=0.2,
    /// Roughness=0.5, Metallic=0.8 → ORM pixels R=0.2, G=0.5, B=0.8 (within
    /// 1/255), through the *real* read-back → pack → write path, not just
    /// `pack_orm` in isolation (already covered directly in `mc_export`).
    #[test]
    fn orm_channel_order_matches_the_unreal_engine_5_spec() {
        let mut graph = mc_core::Graph::new();
        let ao = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.2, 0.2, 0.2, 1.0])),
        );
        let roughness = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.5, 0.5, 0.5, 1.0])),
        );
        let metallic = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.8, 0.8, 0.8, 1.0])),
        );
        let ao_out = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));
        let roughness_out = graph.add_node(mc_core::Node::new("output_roughness"));
        let metallic_out = graph.add_node(mc_core::Node::new("output_metallic"));
        graph.connect(ao, "output", ao_out, "input").unwrap();
        graph
            .connect(roughness, "output", roughness_out, "input")
            .unwrap();
        graph
            .connect(metallic, "output", metallic_out, "input")
            .unwrap();
        graph.settings.resolution = 8;

        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_orm_order_test");
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Unreal,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let orm_path = written
            .iter()
            .find(|p| p.file_name().unwrap() == "T_test_ORM.png")
            .expect("ORM file was written");
        let decoded = image::open(orm_path).unwrap().to_rgba8();
        let px = decoded.get_pixel(0, 0);
        assert!(px[0].abs_diff(51) <= 1, "R (AO) = {}", px[0]);
        assert!(px[1].abs_diff(128) <= 1, "G (Roughness) = {}", px[1]);
        assert!(px[2].abs_diff(204) <= 1, "B (Metallic) = {}", px[2]);
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// An explicit Channel Packer → Output: ORM wire takes priority over
    /// auto-packing from the three separate channels — even when all three
    /// are *also* present in the graph with different values, the exported
    /// ORM must match the explicit node's own output, not a re-derived one.
    #[test]
    fn explicit_output_orm_node_takes_priority_over_auto_packing() {
        let mut graph = mc_core::Graph::new();

        // The three separate channels, present but deliberately different
        // from what the explicit Channel Packer below produces.
        let ao = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.9, 0.9, 0.9, 1.0])),
        );
        let ao_out = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));
        graph.connect(ao, "output", ao_out, "input").unwrap();

        // The explicit Channel Packer, wired to Output: ORM with its own,
        // distinct, exactly-known values.
        let r_fill = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.1, 0.1, 0.1, 1.0])),
        );
        let g_fill = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.4, 0.4, 0.4, 1.0])),
        );
        let b_fill = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.7, 0.7, 0.7, 1.0])),
        );
        let packer = graph.add_node(mc_core::Node::new("channel_pack"));
        graph.connect(r_fill, "output", packer, "r").unwrap();
        graph.connect(g_fill, "output", packer, "g").unwrap();
        graph.connect(b_fill, "output", packer, "b").unwrap();
        let orm_out = graph.add_node(mc_core::Node::new("output_orm"));
        graph.connect(packer, "output", orm_out, "input").unwrap();
        graph.settings.resolution = 8;

        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_explicit_orm_test");
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Unreal,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let orm_path = written
            .iter()
            .find(|p| p.file_name().unwrap() == "T_test_ORM.png")
            .expect("ORM file was written");
        let decoded = image::open(orm_path).unwrap().to_rgba8();
        let px = decoded.get_pixel(0, 0);
        // 0.1*255=26, 0.4*255=102, 0.7*255=179 — the explicit packer's own
        // values, not AO=0.9 (230) from the separate channel above.
        assert!(
            px[0].abs_diff(26) <= 1,
            "R = {} (expected packer's R)",
            px[0]
        );
        assert!(
            px[1].abs_diff(102) <= 1,
            "G = {} (expected packer's G)",
            px[1]
        );
        assert!(
            px[2].abs_diff(179) <= 1,
            "B = {} (expected packer's B)",
            px[2]
        );
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// A known OpenGL-convention normal exports with its green channel
    /// flipped (G = 1 - G) for the Unreal Engine 5 preset.
    #[test]
    fn normal_green_channel_flips_for_unreal_engine_5() {
        // G = 51/255 ~= 0.2 going in; flipped should read back as 255-51=204.
        let mut graph = flat_color_graph("output_normal", [0.5, 0.2, 0.5, 1.0]);
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_normal_flip_test");
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Unreal,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let normal_path = written
            .iter()
            .find(|p| p.file_name().unwrap() == "T_test_N.png")
            .expect("Normal file was written");
        let decoded = image::open(normal_path).unwrap().to_rgba8();
        let px = decoded.get_pixel(0, 0);
        assert!(px[1].abs_diff(204) <= 1, "flipped G = {}", px[1]);
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// No AO/Roughness/Metallic nodes at all in the graph — ORM still gets
    /// written, using the spec's fallback values (1.0/0.5/0.0).
    #[test]
    fn missing_orm_inputs_fall_back_to_the_unreal_engine_5_spec_defaults() {
        let mut graph = flat_color_graph("output_base_color", [0.5, 0.5, 0.5, 1.0]);
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_orm_fallback_test");
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Unreal,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let orm_path = written
            .iter()
            .find(|p| p.file_name().unwrap() == "T_test_ORM.png")
            .expect("ORM file is always written, even with no AO/Roughness/Metallic nodes");
        let decoded = image::open(orm_path).unwrap().to_rgba8();
        let px = decoded.get_pixel(0, 0);
        assert_eq!(px[0], 255, "AO fallback should be 1.0 (white)");
        assert_eq!(px[1], 128, "Roughness fallback should be 0.5");
        assert_eq!(px[2], 0, "Metallic fallback should be 0.0");
        std::fs::remove_dir_all(&out_dir).ok();
    }

    fn graph_with_every_optional_output() -> mc_core::Graph {
        let mut graph = mc_core::Graph::new();
        for kind in [
            "output_base_color",
            "output_normal",
            "output_emissive",
            "output_alpha",
            "output_height",
        ] {
            let fill = graph.add_node(mc_core::Node::new("uniform_color"));
            let output = graph.add_node(mc_core::Node::new(kind));
            graph.connect(fill, "output", output, "input").unwrap();
        }
        graph.settings.resolution = 8;
        graph
    }

    /// `ExportOptions::default()` now means "export everything the graph
    /// has" for either preset — the owner's own request, replacing the
    /// old asymmetry where Blender had no toggles at all and Unreal's
    /// "optional" trio defaulted off. Opacity is the one deliberate
    /// exception (see `ExportOptions::default`'s own comment), so it's
    /// the one file genuinely missing here despite an `output_alpha`
    /// node being present.
    #[test]
    fn default_options_export_everything_the_graph_has_for_unreal() {
        let mut graph = graph_with_every_optional_output();
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_default_unreal_test");
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Unreal,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let names: Vec<_> = written
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(
            names.len(),
            5,
            "BC, Normal, ORM, Emissive, Height -- Opacity defaults off, got {names:?}"
        );
        for expected in [
            "T_test_BC.png",
            "T_test_N.png",
            "T_test_ORM.png",
            "T_test_E.png",
            "T_test_H.png",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing {expected}, got {names:?}"
            );
        }
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// Turning every individual toggle off (not just the old "optional"
    /// trio — Base Color/Normal/ORM too, now that they're real toggles)
    /// writes nothing at all, even though the graph has every Output
    /// node connected.
    #[test]
    fn every_toggle_off_writes_nothing_for_unreal() {
        let mut graph = graph_with_every_optional_output();
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_optional_off_test");
        let options = ExportOptions {
            export_base_color: false,
            export_normal: false,
            export_height: false,
            export_emissive: false,
            export_orm: false,
            ..ExportOptions::default()
        };
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Unreal,
            8,
            "test",
            &out_dir,
            &options,
        )
        .expect("export succeeds");

        assert!(
            written.is_empty(),
            "every toggle off should write nothing, got {written:?}"
        );
        std::fs::remove_dir_all(&out_dir).ok();
    }

    #[test]
    fn optional_outputs_are_written_when_every_toggle_is_on() {
        let mut graph = graph_with_every_optional_output();
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_optional_on_test");
        let options = ExportOptions {
            opacity: OpacityMode::SeparateFile,
            ..ExportOptions::default()
        };
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Unreal,
            8,
            "test",
            &out_dir,
            &options,
        )
        .expect("export succeeds");

        let names: Vec<_> = written
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names.len(), 6, "expected all 6 files, got {names:?}");
        for expected in [
            "T_test_BC.png",
            "T_test_N.png",
            "T_test_ORM.png",
            "T_test_E.png",
            "T_test_M.png",
            "T_test_H.png",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing {expected}, got {names:?}"
            );
        }
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// Blender's own new per-map toggles — it used to export every
    /// present Output node unconditionally, with no way to turn any one
    /// of them off.
    #[test]
    fn blender_per_map_toggles_control_what_gets_written() {
        let mut graph = mc_core::Graph::new();
        for kind in [
            "output_base_color",
            "output_normal",
            "output_roughness",
            "output_metallic",
            "output_ambient_occlusion",
            "output_alpha",
        ] {
            let fill = graph.add_node(mc_core::Node::new("uniform_color"));
            let output = graph.add_node(mc_core::Node::new(kind));
            graph.connect(fill, "output", output, "input").unwrap();
        }
        graph.settings.resolution = 8;

        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_blender_toggles_test");
        let options = ExportOptions {
            export_normal: false,
            export_metallic: false,
            export_alpha: false,
            ..ExportOptions::default()
        };
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Blender,
            8,
            "test",
            &out_dir,
            &options,
        )
        .expect("export succeeds");

        let names: Vec<_> = written
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names.len(), 3, "expected 3 files, got {names:?}");
        for expected in ["test_BaseColor.png", "test_Roughness.png", "test_AO.png"] {
            assert!(
                names.contains(&expected.to_string()),
                "missing {expected}, got {names:?}"
            );
        }
        for excluded in ["test_Normal_GL.png", "test_Metallic.png", "test_Alpha.png"] {
            assert!(
                !names.contains(&excluded.to_string()),
                "{excluded} should have been turned off, got {names:?}"
            );
        }
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// Base-Color-alpha mode folds Opacity into Base Color's own alpha
    /// channel rather than writing a separate `_M` file.
    #[test]
    fn opacity_in_base_color_alpha_writes_no_separate_file() {
        let mut graph = mc_core::Graph::new();
        let bc_fill = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.5, 0.5, 0.5, 1.0])),
        );
        let bc_out = graph.add_node(mc_core::Node::new("output_base_color"));
        graph.connect(bc_fill, "output", bc_out, "input").unwrap();
        let alpha_fill = graph.add_node(
            mc_core::Node::new("uniform_color")
                .with_param("color", mc_core::ParamValue::Color([0.3, 0.3, 0.3, 1.0])),
        );
        let alpha_out = graph.add_node(mc_core::Node::new("output_alpha"));
        graph
            .connect(alpha_fill, "output", alpha_out, "input")
            .unwrap();
        graph.settings.resolution = 8;

        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_opacity_bc_alpha_test");
        let options = ExportOptions {
            opacity: OpacityMode::BaseColorAlpha,
            ..ExportOptions::default()
        };
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Unreal,
            8,
            "test",
            &out_dir,
            &options,
        )
        .expect("export succeeds");

        let names: Vec<_> = written
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(
            !names.iter().any(|n| n == "T_test_M.png"),
            "BaseColorAlpha mode shouldn't write a separate opacity file, got {names:?}"
        );
        let bc_path = written
            .iter()
            .find(|p| p.file_name().unwrap() == "T_test_BC.png")
            .expect("Base Color was written");
        let decoded = image::open(bc_path).unwrap().to_rgba8();
        let px = decoded.get_pixel(0, 0);
        assert!(
            px[3].abs_diff(77) <= 1,
            "Base Color's alpha should carry the opacity value (0.3*255≈77), got {}",
            px[3]
        );
        std::fs::remove_dir_all(&out_dir).ok();
    }

    fn graph_with_every_painter_output() -> mc_core::Graph {
        let mut graph = mc_core::Graph::new();
        for kind in [
            "output_base_color",
            "output_normal",
            "output_height",
            "output_roughness",
            "output_metallic",
            "output_ambient_occlusion",
            "output_emissive",
            "output_alpha",
        ] {
            let fill = graph.add_node(mc_core::Node::new("uniform_color"));
            let output = graph.add_node(mc_core::Node::new(kind));
            graph.connect(fill, "output", output, "input").unwrap();
        }
        graph.settings.resolution = 8;
        graph
    }

    /// Every Painter channel gets its own plain file, written into a
    /// subfolder named after the material, plus the README.
    #[test]
    fn painter_exports_every_channel_into_a_material_subfolder() {
        let mut graph = graph_with_every_painter_output();
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_painter_full_test");
        std::fs::remove_dir_all(&out_dir).ok();
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Painter,
            8,
            "bricks",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let material_dir = out_dir.join("bricks");
        let names: Vec<_> = written
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        let expected = [
            "bricks_BaseColor.png",
            "bricks_Normal_OpenGL.png",
            "bricks_Height.png",
            "bricks_Roughness.png",
            "bricks_Metallic.png",
            "bricks_AmbientOcclusion.png",
            "bricks_Emissive.png",
            "bricks_Opacity.png",
            "README_Painter.txt",
        ];
        for name in expected {
            assert!(
                names.contains(&name.to_string()),
                "missing {name}, got {names:?}"
            );
            assert!(
                material_dir.join(name).exists(),
                "{name} wasn't actually written to disk"
            );
        }
        assert_eq!(names.len(), expected.len(), "got {names:?}");
        for path in &written {
            assert_eq!(path.parent().unwrap(), material_dir);
        }
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// A DirectX-convention Painter normal is the OpenGL one with its green
    /// channel inverted — same flip Unreal already uses, just toggled via
    /// `ExportOptions::painter_normal_format` instead of being fixed per
    /// preset.
    #[test]
    fn painter_directx_normal_is_opengl_with_green_inverted() {
        // G = 51/255 ~= 0.2 going in; flipped should read back as 255-51=204.
        let mut graph = flat_color_graph("output_normal", [0.5, 0.2, 0.5, 1.0]);
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_painter_normal_directx_test");
        std::fs::remove_dir_all(&out_dir).ok();
        let options = ExportOptions {
            painter_normal_format: mc_core::NormalFormat::DirectX,
            ..ExportOptions::default()
        };
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Painter,
            8,
            "test",
            &out_dir,
            &options,
        )
        .expect("export succeeds");

        let path = written
            .iter()
            .find(|p| p.file_name().unwrap() == "test_Normal_DirectX.png")
            .expect("DirectX-named normal file was written");
        let decoded = image::open(path).unwrap().to_rgba8();
        let px = decoded.get_pixel(0, 0);
        assert!(px[1].abs_diff(204) <= 1, "flipped G = {}", px[1]);
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// The OpenGL default writes the convention's own name in the file and
    /// leaves green unflipped — the counterpart to the DirectX test above.
    #[test]
    fn painter_opengl_normal_is_unflipped_and_named_accordingly() {
        let mut graph = flat_color_graph("output_normal", [0.5, 0.2, 0.5, 1.0]);
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_painter_normal_opengl_test");
        std::fs::remove_dir_all(&out_dir).ok();
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Painter,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let path = written
            .iter()
            .find(|p| p.file_name().unwrap() == "test_Normal_OpenGL.png")
            .expect("OpenGL-named normal file was written");
        let decoded = image::open(path).unwrap().to_rgba8();
        let px = decoded.get_pixel(0, 0);
        assert!(px[1].abs_diff(51) <= 1, "unflipped G = {}", px[1]);
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// A graph with only Base Color writes only Base Color (+ the README) —
    /// no placeholder files for the other channels.
    #[test]
    fn painter_skips_missing_outputs_and_writes_no_placeholders() {
        let mut graph = flat_color_graph("output_base_color", [0.5, 0.5, 0.5, 1.0]);
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_painter_missing_test");
        std::fs::remove_dir_all(&out_dir).ok();
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Painter,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let names: Vec<_> = written
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(
            names.len(),
            2,
            "expected only Base Color + README, got {names:?}"
        );
        assert!(names.contains(&"test_BaseColor.png".to_string()));
        assert!(names.contains(&"README_Painter.txt".to_string()));
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// A user-typed material name with spaces/special characters is
    /// sanitized before it's used in either the subfolder or file names.
    #[test]
    fn painter_sanitizes_a_custom_material_name() {
        let mut graph = flat_color_graph("output_base_color", [0.5, 0.5, 0.5, 1.0]);
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_painter_sanitize_test");
        std::fs::remove_dir_all(&out_dir).ok();
        let options = ExportOptions {
            painter_material_name: "My Brick Wall!".to_string(),
            ..ExportOptions::default()
        };
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Painter,
            8,
            "test",
            &out_dir,
            &options,
        )
        .expect("export succeeds");

        let bc = written
            .iter()
            .find(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .ends_with("_BaseColor.png")
            })
            .expect("Base Color was written");
        assert_eq!(bc.file_name().unwrap(), "MyBrickWall_BaseColor.png");
        assert_eq!(bc.parent().unwrap().file_name().unwrap(), "MyBrickWall");
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// `painter_grayscale_16bit` affects Roughness/Metallic/AO/Opacity;
    /// `painter_height_exr` switches Height to a 32-bit float EXR — both
    /// off by default (8-bit PNG / 16-bit PNG respectively, per spec).
    #[test]
    fn painter_bit_depth_options_change_the_written_format() {
        let mut graph = mc_core::Graph::new();
        for kind in ["output_roughness", "output_height"] {
            let fill = graph.add_node(mc_core::Node::new("uniform_color"));
            let output = graph.add_node(mc_core::Node::new(kind));
            graph.connect(fill, "output", output, "input").unwrap();
        }
        graph.settings.resolution = 8;

        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_painter_bitdepth_test");
        std::fs::remove_dir_all(&out_dir).ok();
        let options = ExportOptions {
            painter_grayscale_16bit: true,
            painter_height_exr: true,
            ..ExportOptions::default()
        };
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Painter,
            8,
            "test",
            &out_dir,
            &options,
        )
        .expect("export succeeds");

        let roughness_path = written
            .iter()
            .find(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains("Roughness")
            })
            .expect("Roughness was written");
        let decoded = image::open(roughness_path).unwrap();
        assert_eq!(decoded.color(), image::ColorType::L16);

        let height_path = written
            .iter()
            .find(|p| p.file_name().unwrap().to_string_lossy().contains("Height"))
            .expect("Height was written");
        assert_eq!(height_path.extension().unwrap(), "exr");
        assert!(height_path.exists());
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// The README lists every written file and its fill-layer slot, and
    /// never mentions "Substance" anywhere (INDEPENDENCE.md).
    #[test]
    fn painter_readme_lists_each_written_file_and_its_slot() {
        let mut graph = flat_color_graph("output_base_color", [0.5, 0.5, 0.5, 1.0]);
        let (mut gpu, mut pool) = headless();
        let out_dir = std::env::temp_dir().join("mc_nodes_export_painter_readme_test");
        std::fs::remove_dir_all(&out_dir).ok();
        let written = export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            mc_export::naming::Preset::Painter,
            8,
            "test",
            &out_dir,
            &ExportOptions::default(),
        )
        .expect("export succeeds");

        let readme_path = written
            .iter()
            .find(|p| p.file_name().unwrap() == "README_Painter.txt")
            .expect("README was written");
        let contents = std::fs::read_to_string(readme_path).unwrap();
        assert!(contents.contains("test_BaseColor.png"));
        assert!(contents.contains("Base Color"));
        assert!(
            !contents.to_lowercase().contains("substance"),
            "README must never mention \"Substance\" (INDEPENDENCE.md), got:\n{contents}"
        );
        std::fs::remove_dir_all(&out_dir).ok();
    }

    /// The export dialog's pre-flight overwrite check needs Painter's
    /// planned paths to land inside the material subfolder, same as the
    /// real export does.
    #[test]
    fn painter_planned_output_paths_use_the_material_subfolder() {
        let graph = flat_color_graph("output_base_color", [0.5, 0.5, 0.5, 1.0]);
        let out_dir = std::env::temp_dir().join("mc_nodes_export_painter_planned_test");
        let paths = planned_output_paths(
            &graph,
            mc_export::naming::Preset::Painter,
            "test",
            &out_dir,
            &ExportOptions::default(),
        );
        assert!(paths.contains(&out_dir.join("test").join("test_BaseColor.png")));
        assert!(paths.contains(&out_dir.join("test").join("README_Painter.txt")));
    }
}

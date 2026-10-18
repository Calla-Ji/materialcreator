// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Export… dialog: pick a preset, resolution, output folder, and graph
//! name, then write the material via `mc_nodes::export::export_material`.

const RESOLUTIONS: [u32; 5] = [256, 512, 1024, 2048, 4096];

pub struct ExportDialogState {
    pub open: bool,
    preset: mc_export::naming::Preset,
    resolution: u32,
    graph_name: String,
    out_dir: Option<std::path::PathBuf>,
    status: Option<Result<Vec<std::path::PathBuf>, String>>,
    /// Which maps to export, plus Unreal's own Format/Opacity choices —
    /// shared between both presets now (the owner's own request: Blender
    /// used to have no per-map choice at all, only Unreal's "optional"
    /// trio was toggleable). See `mc_nodes::export::ExportOptions`'s own
    /// doc comment for which fields apply to which preset.
    options: mc_nodes::export::ExportOptions,
    /// `Some(paths)` while the "these already exist, overwrite?" confirm
    /// window is up — set by a first Export click that found existing
    /// files, cleared by either button in that window.
    pending_overwrite: Option<Vec<std::path::PathBuf>>,
    /// The folder-picker's own in-flight native dialog — see
    /// `async_dialog.rs`'s own doc comment for why this goes through
    /// `rfd::AsyncFileDialog` instead of the simpler-looking synchronous
    /// `rfd::FileDialog` API.
    pending_out_dir: Option<crate::async_dialog::PendingDialog<Option<rfd::FileHandle>>>,
}

impl Default for ExportDialogState {
    fn default() -> Self {
        Self {
            open: false,
            preset: mc_export::naming::Preset::Blender,
            resolution: 2048,
            graph_name: "material".to_string(),
            out_dir: None,
            status: None,
            options: mc_nodes::export::ExportOptions::default(),
            pending_overwrite: None,
            pending_out_dir: None,
        }
    }
}

/// Runs the real export and stores the result in `state.status`.
fn run_export(
    state: &mut ExportDialogState,
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
    out_dir: &std::path::Path,
) {
    let result = mc_nodes::export::export_material(
        gpu,
        pool,
        graph,
        state.preset,
        state.resolution,
        state.graph_name.trim(),
        out_dir,
        &state.options,
    )
    .map_err(|err| err.to_string());
    state.status = Some(result);
}

/// Renders the dialog if `state.open`. Mutates `graph` transiently (it's
/// re-evaluated at the export resolution, then restored) via
/// `mc_nodes::export::export_material`.
pub fn show(
    ctx: &egui::Context,
    state: &mut ExportDialogState,
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
) {
    if !state.open {
        return;
    }

    if let Some(pending) = &mut state.pending_out_dir {
        if let Some(result) = pending.poll(ctx) {
            state.pending_out_dir = None;
            if let Some(handle) = result {
                state.out_dir = Some(handle.path().to_path_buf());
                state.status = None;
            }
        }
    }

    let mut open = state.open;
    crate::dialog_chrome::show(
        ctx,
        "export_material_dialog",
        "Export Material",
        &mut open,
        true,
        crate::dialog_chrome::DEFAULT_WIDTH,
        |ui| {
            egui::ComboBox::from_label("Preset")
                .selected_text(state.preset.name())
                .show_ui(ui, |ui| {
                    for name in mc_export::naming::Preset::NAMES {
                        let preset = mc_export::naming::Preset::from_name(name)
                            .expect("NAMES and from_name must agree");
                        ui.selectable_value(&mut state.preset, preset, name);
                    }
                });

            egui::ComboBox::from_label("Resolution")
                .selected_text(format!("{0}x{0}", state.resolution))
                .show_ui(ui, |ui| {
                    for &resolution in &RESOLUTIONS {
                        ui.selectable_value(
                            &mut state.resolution,
                            resolution,
                            format!("{resolution}x{resolution}"),
                        );
                    }
                });

            let is_unreal = state.preset == mc_export::naming::Preset::Unreal;
            let is_painter = state.preset == mc_export::naming::Preset::Painter;

            if is_unreal {
                ui.separator();
                egui::ComboBox::from_label("Format")
                    .selected_text(match state.options.format {
                        mc_export::RasterFormat::Png => "PNG",
                        mc_export::RasterFormat::Tga => "TGA",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.options.format,
                            mc_export::RasterFormat::Png,
                            "PNG",
                        );
                        ui.selectable_value(
                            &mut state.options.format,
                            mc_export::RasterFormat::Tga,
                            "TGA",
                        );
                    });
            }

            ui.separator();
            ui.weak("Maps to export \u{2014} unchecked ones are left out entirely.");
            ui.checkbox(&mut state.options.export_base_color, "Base Color");
            ui.checkbox(&mut state.options.export_normal, "Normal");
            if is_unreal {
                ui.checkbox(&mut state.options.export_orm, "ORM");
            } else {
                ui.checkbox(&mut state.options.export_roughness, "Roughness");
                ui.checkbox(&mut state.options.export_metallic, "Metallic");
                ui.checkbox(
                    &mut state.options.export_ambient_occlusion,
                    "Ambient Occlusion",
                );
            }
            ui.checkbox(&mut state.options.export_height, "Height");
            ui.checkbox(&mut state.options.export_emissive, "Emissive");

            if is_unreal {
                let mut export_opacity =
                    state.options.opacity != mc_nodes::export::OpacityMode::Off;
                ui.checkbox(&mut export_opacity, "Opacity");
                if export_opacity {
                    if state.options.opacity == mc_nodes::export::OpacityMode::Off {
                        state.options.opacity = mc_nodes::export::OpacityMode::BaseColorAlpha;
                    }
                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        ui.radio_value(
                            &mut state.options.opacity,
                            mc_nodes::export::OpacityMode::BaseColorAlpha,
                            "In Base Color alpha",
                        );
                        ui.radio_value(
                            &mut state.options.opacity,
                            mc_nodes::export::OpacityMode::SeparateFile,
                            "Separate file (_M)",
                        );
                    });
                } else {
                    state.options.opacity = mc_nodes::export::OpacityMode::Off;
                }
            } else {
                // Painter's Opacity reuses the same `export_alpha` toggle as
                // Blender's Alpha (same "export this if present" intent,
                // different file name) — see `ExportOptions`'s own doc
                // comment.
                let label = if is_painter { "Opacity" } else { "Alpha" };
                ui.checkbox(&mut state.options.export_alpha, label);
            }

            if is_painter {
                ui.separator();
                ui.weak("Painter options");

                ui.horizontal(|ui| {
                    ui.label("Material Name:");
                    ui.text_edit_singleline(&mut state.options.painter_material_name)
                        .on_hover_text("Leave blank to use the Name field below.");
                });
                let effective_name = if state.options.painter_material_name.trim().is_empty() {
                    state.graph_name.trim()
                } else {
                    state.options.painter_material_name.trim()
                };
                ui.weak(format!(
                    "Files will be named using: {}",
                    mc_export::naming::sanitize_material_name(effective_name)
                ));

                ui.horizontal(|ui| {
                    ui.label("Normal format:");
                    ui.radio_value(
                        &mut state.options.painter_normal_format,
                        mc_core::NormalFormat::OpenGl,
                        "OpenGL",
                    );
                    ui.radio_value(
                        &mut state.options.painter_normal_format,
                        mc_core::NormalFormat::DirectX,
                        "DirectX",
                    );
                });

                ui.checkbox(
                    &mut state.options.painter_grayscale_16bit,
                    "16-bit Roughness/Metallic/AO/Opacity (default: 8-bit)",
                );
                ui.checkbox(
                    &mut state.options.painter_height_exr,
                    "32-bit EXR Height (default: 16-bit PNG)",
                );
            }
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.text_edit_singleline(&mut state.graph_name);
            });

            ui.horizontal(|ui| {
                ui.label("Folder:");
                let label = state
                    .out_dir
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(none selected)".to_string());
                ui.weak(label);
                if ui.button("Choose…").clicked() {
                    state.pending_out_dir = Some(crate::async_dialog::PendingDialog::new(
                        rfd::AsyncFileDialog::new().pick_folder(),
                    ));
                }
            });

            ui.separator();

            let can_export = state.out_dir.is_some() && !state.graph_name.trim().is_empty();
            if ui
                .add_enabled(can_export, egui::Button::new("Export"))
                .clicked()
            {
                let out_dir = state.out_dir.clone().expect("checked by can_export");
                let planned = mc_nodes::export::planned_output_paths(
                    graph,
                    state.preset,
                    state.graph_name.trim(),
                    &out_dir,
                    &state.options,
                );
                let existing: Vec<_> = planned.into_iter().filter(|p| p.exists()).collect();
                if existing.is_empty() {
                    run_export(state, gpu, pool, graph, &out_dir);
                } else {
                    state.pending_overwrite = Some(existing);
                }
            }

            if let Some(status) = &state.status {
                match status {
                    Ok(paths) => {
                        ui.colored_label(
                            egui::Color32::LIGHT_GREEN,
                            format!("Wrote {} file(s):", paths.len()),
                        );
                        for path in paths {
                            ui.weak(path.display().to_string());
                        }
                        if state.preset == mc_export::naming::Preset::Unreal {
                            ui.add_space(4.0);
                            ui.weak(
                                "In UE5: BC sRGB on; N sRGB off, compression Normalmap; ORM \
                                 sRGB off, compression Masks; H sRGB off, compression Grayscale.",
                            );
                        }
                        if state.preset == mc_export::naming::Preset::Painter {
                            ui.add_space(4.0);
                            ui.weak(
                                "Written into their own subfolder, alongside a README listing \
                                 which file goes in which fill-layer slot.",
                            );
                        }
                    }
                    Err(err) => {
                        ui.colored_label(egui::Color32::LIGHT_RED, format!("Export failed: {err}"));
                    }
                }
            }
        },
    );
    state.open = open;

    if let Some(existing) = state.pending_overwrite.clone() {
        // Not `closable`: Overwrite/Cancel are the only way out, on
        // purpose — see `dialog_chrome::show`'s own doc comment on why a
        // hard confirm like this never gets an X.
        let mut open = true;
        let mut still_pending = true;
        crate::dialog_chrome::show(
            ctx,
            "export_overwrite_confirm",
            "Overwrite existing files?",
            &mut open,
            false,
            crate::dialog_chrome::DEFAULT_WIDTH,
            |ui| {
                ui.label("These files already exist and will be overwritten:");
                ui.add_space(4.0);
                for path in &existing {
                    ui.weak(path.display().to_string());
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Overwrite").clicked() {
                        let out_dir = state.out_dir.clone().expect("set before export");
                        run_export(state, gpu, pool, graph, &out_dir);
                        still_pending = false;
                    }
                    if ui.button("Cancel").clicked() {
                        still_pending = false;
                    }
                });
            },
        );
        if !still_pending {
            state.pending_overwrite = None;
        }
    }
}

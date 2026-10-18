// SPDX-License-Identifier: MIT OR Apache-2.0
//! mc_app: the desktop application shell.
//!
//! Owns the window and the overall panel layout (menu, 2D/3D previews, node
//! graph, parameter panel). The node editor (`canvas.rs`) is a fully custom
//! pan/zoom/select/connect canvas built directly on egui, reading and
//! writing `mc_core::Graph` through the same `Command`/`Transaction` API as
//! every other part of the app: add nodes from a search menu, connect them
//! (with Gray/Color type checking), edit params in a generic panel, and
//! undo/redo.

mod add_node_menu;
mod async_dialog;
mod autosave;
mod canvas;
mod dialog_chrome;
mod display_encode;
mod dock;
mod export_dialog;
mod hdri;
mod mipgen;
mod node_icons;
mod params_panel;
mod path_editor;
mod preset_graph;
mod splash;
mod svg_import_dialog;
mod tess_gpu;
mod tess_mesh;
mod theme;
mod toast;
mod twod_toolbar;
mod viewport3d;
mod viewport3d_panels;

use std::collections::HashMap;
use std::sync::Arc;

use canvas::CanvasState;

/// Same list `export_dialog.rs` and `splash.rs` each already have their own
/// copy of — kept a third, separate copy here rather than sharing one,
/// matching how those two already don't share it with each other.
const RESOLUTIONS: [u32; 5] = [256, 512, 1024, 2048, 4096];

/// (display name, path relative to the repo root) — the same `.mcgraph`
/// files `mc_cli`'s own `main` builds and saves, picked up here by path
/// rather than re-building them in-process. Resolved by `resolve_example_path`
/// before use, since the process's working directory isn't reliably the repo
/// root once this is a packaged release build.
const EXAMPLE_GRAPHS: &[(&str, &str)] = &[
    ("Bricks", "examples/graphs/bricks.mcgraph"),
    ("Rock", "examples/graphs/rock.mcgraph"),
    ("Brushed Metal", "examples/graphs/brushed_metal.mcgraph"),
    ("Fabric", "examples/graphs/fabric.mcgraph"),
    ("Wooden Planks", "examples/graphs/wooden_planks.mcgraph"),
    (
        "Glitch & Dither Demo",
        "examples/graphs/glitch_dither_demo.mcgraph",
    ),
    (
        "Hard-Surface Demo",
        "examples/graphs/hardsurface_demo.mcgraph",
    ),
];

/// A packaged release ships `examples/` next to the executable (see
/// `release.yml`'s packaging step); `cargo run` instead relies on the
/// process's working directory being the repo root, where the same relative
/// path already resolves correctly. Prefer the exe-relative location so a
/// packaged build finds its own bundled copy regardless of launch method,
/// falling back to the plain relative path for the `cargo run` case.
fn resolve_example_path(relative: &str) -> std::path::PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(relative);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    std::path::PathBuf::from(relative)
}

/// 2D View's own pan/zoom/tiling-preview state — view-only, not part of the
/// graph or any node's params.
struct TwoDViewState {
    /// Screen-space drag offset, in points. At the default `zoom` in the
    /// single-tile view, the tile always exactly fills its square, so this
    /// only has visible effect once `tiling_preview` is on or `zoom` has
    /// zoomed in past 1x — kept regardless so toggling tiling/zooming back
    /// out restores where you left off.
    pan: egui::Vec2,
    /// Scroll-wheel/pinch zoom factor, centered on the pointer. 1.0 = fit
    /// (the single tile exactly fills its square; a tiling-preview "tile"
    /// is a third of the panel's shorter side, same as before zoom existed).
    zoom: f32,
    /// Shows the image tiled edge-to-edge across the whole panel (wrap-
    /// addressed, via `preview_texture`'s Repeat sampler) instead of a
    /// single centered copy, so a seam — or the lack of one — is visible at
    /// a glance.
    tiling_preview: bool,
    /// CLAUDE.md §9's "channel view (RGB / R / G / B / A)" — see
    /// `display_encode::ChannelView`'s own doc comment for why isolating
    /// one is sometimes the only way to actually judge it.
    channel_view: display_encode::ChannelView,
}

impl Default for TwoDViewState {
    fn default() -> Self {
        Self {
            pan: egui::Vec2::ZERO,
            zoom: 1.0,
            tiling_preview: false,
            channel_view: display_encode::ChannelView::default(),
        }
    }
}

/// Root application state for the eframe event loop.
struct MaterialCreatorApp {
    gpu: mc_gpu::Context,
    pool: mc_gpu::TexturePool,
    render_state: egui_wgpu::RenderState,

    graph: mc_core::Graph,
    history: mc_core::History,
    canvas: CanvasState,
    thumbnail_ids: HashMap<mc_core::NodeId, egui::TextureId>,

    /// The node the Parameters panel edits — the "pinned" node. Only
    /// changes on a deliberate action: double-clicking a node, the node
    /// header's own "Show in Parameters / 2D View" dot, or creating/adding
    /// a node (which naturally becomes both pinned and previewed at once).
    /// A single click on a node, by contrast, only moves `preview_node` —
    /// the owner's own request, so clicking around the graph to compare
    /// outputs in the 2D View doesn't keep yanking the Parameters panel
    /// away from whatever you're actually mid-edit on.
    selected: Option<mc_core::NodeId>,
    /// The node the 2D View shows. Moves on a plain single click/drag, not
    /// just the deliberate actions that also move `selected` — see that
    /// field's own doc comment for the full reasoning.
    preview_node: Option<mc_core::NodeId>,
    /// Which of `preview_node`'s own output ports to preview in the 2D
    /// View, for a node with more than one (currently just Thread) —
    /// tagged with the node it was set for, so switching to a *different*
    /// node automatically falls back to that node's own default/first
    /// output instead of silently reusing a stale port name that might not
    /// even exist on it. Set by clicking an output pin in the node graph
    /// (see `canvas::show`'s own doc comment on that pin's click
    /// handling).
    preview_port: Option<(mc_core::NodeId, String)>,
    add_node_menu: Option<add_node_menu::AddNodeMenu>,
    pending_wire_connect: Option<canvas::DroppedWire>,
    param_drag: params_panel::DragState,
    export_dialog: export_dialog::ExportDialogState,
    svg_import_dialog: svg_import_dialog::SvgImportDialogState,
    /// Set by the Image Trace node's own "Export SVG…" button
    /// (`params_panel.rs`) — consumed right after `export_dialog::show`,
    /// where `gpu`/`pool`/`graph` are already all in scope together, the
    /// same reason `export_dialog`'s own real export runs from there
    /// rather than from `params_panel.rs` itself.
    pending_svg_export: Option<mc_core::NodeId>,
    /// `do_open`/`do_save_as`/the HDRI loader's own in-flight native
    /// dialog — see `async_dialog.rs`'s own doc comment for why these go
    /// through `rfd::AsyncFileDialog` instead of the simpler-looking
    /// synchronous `rfd::FileDialog` API every one of these used to call
    /// directly: a real crash report showed the synchronous version
    /// aborting the whole process when a gesture event arrived while the
    /// native panel was open.
    pending_open: Option<async_dialog::PendingDialog<Option<rfd::FileHandle>>>,
    pending_save_as: Option<async_dialog::PendingDialog<Option<rfd::FileHandle>>>,
    pending_hdri_load: Option<async_dialog::PendingDialog<Option<rfd::FileHandle>>>,
    /// Same idea, for the Image Trace "Export SVG…" flow above — also
    /// carries the already-computed SVG text across the async gap, since
    /// it has to wait for the save panel's own result before it knows
    /// where to write it.
    pending_svg_export_save: Option<(String, async_dialog::PendingDialog<Option<rfd::FileHandle>>)>,
    /// Image Import's own in-flight dialog — see
    /// `params_panel::PendingImageImport`'s own doc comment.
    pending_image_import: Option<params_panel::PendingImageImport>,
    viewport3d: viewport3d::Viewport3d,
    viewport_panels: viewport3d_panels::ViewportPanels,
    dock: dock::DockLayout,

    preview_texture_id: Option<egui::TextureId>,
    two_d_view: TwoDViewState,
    path_editor: path_editor::PathEditorState,

    /// Where the current graph was last loaded from or saved to — `None`
    /// for a never-saved graph, in which case Save behaves like Save As.
    current_path: Option<std::path::PathBuf>,

    toasts: toast::Toasts,
    /// The last graph-evaluation error's message, if the graph is currently
    /// failing to evaluate — `evaluate()` runs every frame, so without this
    /// a persistent error would spam a fresh toast on every single frame
    /// instead of once when it first appears.
    last_eval_error: Option<String>,
    /// `true` for exactly one frame before a heavy recompute (a Resolution
    /// change or a freshly loaded/new graph, both of which mark everything
    /// dirty at once) actually runs — see this field's own use in `ui()`
    /// for why: it lets a "Recalculating…" pill actually get painted and
    /// presented *before* `registry::evaluate`'s own call blocks the UI
    /// thread, rather than the screen just freezing on whatever was on
    /// screen before the edit with no feedback at all (confirmed by
    /// hands-on testing to look exactly like a crash). Set only from the
    /// two specific, deliberate triggers above — *not* inferred from "how
    /// many nodes are currently dirty," which an earlier version of this
    /// tried and got wrong: adding even one new node to a small graph can
    /// make a majority of it "dirty" by that measure, which flickered the
    /// pill/dim-veil on completely routine node-graph editing (reported
    /// directly as "the screen started flimmering" while adding nodes).
    recompute_in_progress: bool,
    /// The Resolution this app last actually evaluated at — compared
    /// against `self.graph.settings.resolution` each frame purely to
    /// detect a *change*, which is what sets `recompute_in_progress`
    /// (every node's evaluation hash depends on it, so changing it marks
    /// the entire graph dirty at once — see `Graph::evaluation_hash`).
    last_known_resolution: u32,

    autosave: autosave::Autosave,
    /// Set at startup if a recovery file from a previous, uncleanly-ended
    /// session was found — shows the restore/discard prompt until the
    /// artist answers it.
    pending_recovery: bool,
    /// Shown on every launch, once `pending_recovery` (if any) is resolved
    /// — see `splash.rs`'s own doc comment.
    splash: splash::SplashState,
}

impl eframe::App for MaterialCreatorApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("menu_bar")
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(12, 8)))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = 10.0;
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("MaterialCreator")
                            .strong()
                            .color(theme::SUPERNOVA_LIGHT),
                    );
                    ui.add_space(4.0);
                    ui.separator();
                    if ui.button("New").clicked() {
                        self.do_new();
                    }
                    if ui.button("Open").clicked() {
                        self.do_open();
                    }
                    if ui.button("Save").clicked() {
                        self.do_save();
                    }
                    if ui.button("Save As").clicked() {
                        self.do_save_as();
                    }
                    if ui.button("Export…").clicked() {
                        self.export_dialog.open = true;
                    }
                    ui.add_space(4.0);
                    ui.separator();
                    // Live working resolution — changing it needs no special
                    // handling beyond the plain assignment below: every
                    // node's own evaluation hash already includes
                    // `settings.resolution` (`Graph::evaluation_hash`), so
                    // the whole graph re-evaluates at the new size on the
                    // very next frame; `TexturePool::get_or_create` already
                    // recreates a texture when the requested size changed;
                    // and both the 2D View's own preview texture and the 3D
                    // viewport's texture bind group are rebuilt fresh every
                    // frame (never cached across frames), so neither one can
                    // end up holding a stale, wrong-sized texture. Not
                    // undo-tracked — same precedent as `set_node_position`
                    // and the splash screen's own initial resolution choice,
                    // neither of which go through the Command system either.
                    egui::ComboBox::from_label("Resolution")
                        .selected_text(format!("{0}x{0}", self.graph.settings.resolution))
                        .show_ui(ui, |ui| {
                            for &resolution in &RESOLUTIONS {
                                ui.selectable_value(
                                    &mut self.graph.settings.resolution,
                                    resolution,
                                    format!("{resolution}x{resolution}"),
                                );
                            }
                        });
                    ui.add_space(4.0);
                    ui.separator();
                    if ui
                        .add_enabled(self.history.can_undo(), egui::Button::new("Undo"))
                        .clicked()
                    {
                        self.history.undo(&mut self.graph);
                    }
                    if ui
                        .add_enabled(self.history.can_redo(), egui::Button::new("Redo"))
                        .clicked()
                    {
                        self.history.redo(&mut self.graph);
                    }
                    ui.add_space(4.0);
                    ui.separator();
                    ui.menu_button("Examples", |ui| {
                        for &(name, path) in EXAMPLE_GRAPHS {
                            if ui.button(name).clicked() {
                                self.load_example(path);
                                ui.close();
                            }
                        }
                    });
                });
            });

        self.toasts.show(ui.ctx());
        self.autosave.maybe_save(&self.graph);

        if self.pending_recovery {
            // Not `closable`: Restore/Discard are the only way out, on
            // purpose — see `dialog_chrome::show`'s own doc comment on
            // why a hard confirm like this never gets an X.
            let mut open = true;
            dialog_chrome::show(
                ui.ctx(),
                "recover_unsaved_work_confirm",
                "Recover unsaved work?",
                &mut open,
                false,
                dialog_chrome::DEFAULT_WIDTH,
                |ui| {
                    ui.label(
                        "MaterialCreator didn't close cleanly last time — an autosave from \
                         that session is available.",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Restore").clicked() {
                            match mc_core::Graph::load_from_file(autosave::recovery_path()) {
                                Ok(graph) => {
                                    // Clears the just-read recovery file too
                                    // (see `replace_graph`'s doc comment).
                                    self.replace_graph(graph, None);
                                    self.toasts.info("Restored autosave from previous session");
                                    // Real recovered work is already loaded —
                                    // the splash screen's own "pick a
                                    // starting point" choice doesn't apply.
                                    self.splash.open = false;
                                }
                                Err(err) => {
                                    log::warn!("failed to restore autosave: {err}");
                                    self.toasts
                                        .error(format!("Couldn't restore autosave: {err}"));
                                    autosave::Autosave::clear();
                                }
                            }
                            self.pending_recovery = false;
                        }
                        if ui.button("Discard").clicked() {
                            autosave::Autosave::clear();
                            self.pending_recovery = false;
                        }
                    });
                },
            );
        }

        // Shown on every launch, once any crash-recovery prompt above is
        // resolved — picking "Restore" there already closes this (see that
        // handler), so it only ever appears when there's genuinely nothing
        // to recover, or the artist chose "Discard".
        if !self.pending_recovery {
            if let Some((start, resolution)) = splash::show(ui.ctx(), &mut self.splash) {
                let graph = match start {
                    splash::StartChoice::Empty => {
                        let mut graph = mc_core::Graph::new();
                        graph.settings.resolution = resolution;
                        graph
                    }
                    splash::StartChoice::Blender => preset_graph::build_blender(resolution),
                    splash::StartChoice::Unreal => preset_graph::build_unreal_engine_5(resolution),
                };
                self.replace_graph(graph, None);
                self.splash.open = false;
            }
        }

        // Re-evaluate whatever the previous frame's edits (including
        // anything undo/redo just changed) left dirty, before any panel
        // reads from `self.pool` — runs every single frame, same as always,
        // *except* the one frame right after a heavy-recompute trigger
        // fires (a Resolution change, or `replace_graph` for New/Open),
        // which instead skips straight to painting the "Recalculating…"
        // pill below so it has a chance to actually reach the screen before
        // the deferred `evaluate` call (next frame) blocks the UI thread.
        //
        // An earlier version of this gated `evaluate` itself behind
        // `recompute_in_progress` — i.e. only ever called it on the one
        // deferred frame — which silently stopped *all* routine graph
        // editing (adding a node, connecting a wire, a param tweak) from
        // ever evaluating again, since none of those set that flag. Caught
        // via a direct report: "it seems to apply to all nodes... No node
        // is showing a previsualization right now."
        if self.recompute_in_progress {
            // The deferred frame: do the (possibly heavy) work now.
            self.run_graph_evaluation();
            self.recompute_in_progress = false;
        } else {
            let resolution_changed = self.graph.settings.resolution != self.last_known_resolution;
            self.last_known_resolution = self.graph.settings.resolution;
            if resolution_changed {
                self.recompute_in_progress = true;
                // eframe only repaints on input/animation by default — this
                // forces the next frame (the one that actually calls
                // `evaluate`) to run immediately instead of waiting idle
                // for a mouse move that may not come.
                ui.ctx().request_repaint();
            } else {
                // The ordinary case, every other frame: evaluate immediately,
                // same as this app has always done.
                self.run_graph_evaluation();
            }
        }

        if self.recompute_in_progress {
            // Same dim veil `splash.rs`'s own first-launch prompt uses
            // (`Order::Middle`, non-interactable — purely visual, a click
            // still reaches whatever's underneath, which doesn't matter
            // anyway since input is about to go unresponsive once
            // `evaluate` actually blocks) — makes the pill read as a real,
            // distinct state rather than just another floating label
            // competing with the panels behind it.
            egui::Area::new(egui::Id::new("materialcreator_recompute_veil"))
                .order(egui::Order::Middle)
                .fixed_pos(egui::Pos2::ZERO)
                .interactable(false)
                .fade_in(false)
                .show(ui.ctx(), |ui| {
                    ui.painter().rect_filled(
                        ui.ctx().content_rect(),
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(0x13, 0x17, 0x18, 140),
                    );
                });

            egui::Area::new(egui::Id::new("materialcreator_recompute_pill"))
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .order(egui::Order::Foreground)
                // `Area` fades in new content over its first few frames by
                // default — this pill often only lives for exactly one
                // frame before the heavy recompute it's warning about
                // blocks the UI thread, so with the default fade it never
                // got past "barely visible" before the freeze hit (reported
                // directly: "I couldn't see any pill"). It needs to be at
                // full opacity on its very first frame, since that's the
                // frame that stays frozen on screen for the whole stall.
                .fade_in(false)
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style())
                        .fill(crate::theme::NEUTRON_STAR)
                        .inner_margin(egui::Margin::symmetric(32, 24))
                        .corner_radius(12.0)
                        .show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 16.0;
                                    ui.add(
                                        egui::Spinner::new()
                                            .color(crate::theme::SUPERNOVA_LIGHT)
                                            .size(28.0),
                                    );
                                    ui.label(
                                        egui::RichText::new("Recalculating…")
                                            .color(crate::theme::SUPERNOVA_LIGHT)
                                            .size(22.0),
                                    );
                                });
                                ui.add_space(6.0);
                                ui.label(
                                    egui::RichText::new("This might take up to 1 minute.")
                                        .color(crate::theme::SUPERNOVA_LIGHT.gamma_multiply(0.6))
                                        .size(14.0),
                                );
                            });
                        });
                });
        }

        // 2D View, 3D View, Node Graph, and Parameters are dock cells (see
        // dock.rs) laid out from raw rects we own, rather than
        // `egui::Window`/Panel/columns — none of those auto-size to content,
        // so nothing here can corrupt a sibling's measured size the way the
        // old nested layout did, and dragging a border always keeps both
        // neighbors flush since they read the same stored split ratio.
        let bounds = ui.available_rect_before_wrap();

        self.viewport_panels.update_anim(ui.ctx());
        let drawer_width = self.viewport_panels.current_width();
        let preview_texture_id = self.preview_texture();
        let mut canvas_output = None;
        // Set when the Path editor just deleted a selected point this
        // frame — read below by the global Delete/Backspace handler so it
        // doesn't *also* delete the whole node on the same keypress (see
        // `path_editor::show`'s own doc comment on its return value).
        let mut path_editor_deleted_a_point = false;

        // Disjoint borrows of the fields each dock cell's content needs, so
        // the closure below doesn't have to borrow `self` as a whole while
        // `self.dock` is already borrowed by the `show` call it's passed to.
        let MaterialCreatorApp {
            dock,
            graph,
            history,
            canvas,
            thumbnail_ids,
            selected,
            preview_node,
            preview_port,
            param_drag,
            viewport3d,
            viewport_panels,
            gpu,
            pool,
            render_state,
            two_d_view,
            path_editor,
            svg_import_dialog,
            pending_svg_export,
            pending_hdri_load,
            pending_image_import,
            ..
        } = self;

        dock.show(
            ui,
            bounds,
            |id| match id {
                dock::CrateId::TwoDView => "2D View",
                dock::CrateId::ThreeDView => "3D View",
                dock::CrateId::NodeGraph => "Node Graph",
                dock::CrateId::Parameters => "Parameters",
                dock::CrateId::NodeLibrary => "Nodes",
            },
            &mut |ui, id| match id {
                dock::CrateId::Parameters => {
                    params_panel::show(
                        ui,
                        graph,
                        history,
                        *selected,
                        canvas.selected_annotation,
                        param_drag,
                        path_editor,
                        pending_svg_export,
                        pending_image_import,
                        &*thumbnail_ids,
                    );
                }
                dock::CrateId::TwoDView => {
                    let full_rect = ui.available_rect_before_wrap();
                    let toolbar_content_rect = twod_toolbar::show(
                        ui,
                        full_rect,
                        &mut two_d_view.tiling_preview,
                        &mut two_d_view.channel_view,
                        path_editor,
                        svg_import_dialog,
                        graph,
                        history,
                        *selected,
                    );

                    match preview_texture_id {
                        Some(texture_id) => {
                            const PADDING: f32 = 16.0;

                            let content_rect = toolbar_content_rect.shrink(PADDING);

                            // Scroll wheel and pinch both zoom here — unlike
                            // the node graph canvas, there's nothing else
                            // for a plain scroll to do in a single-image
                            // preview (drag already covers panning) —
                            // centered on the pointer so the spot being
                            // inspected stays under the cursor instead of
                            // the view recentering under it. Gated the same
                            // way canvas.rs gates its own pan/zoom: pointer
                            // inside this panel's content rect *and* this
                            // panel's layer is the topmost one there, so
                            // scrolling over e.g. a popup sitting on top of
                            // the 2D View doesn't also zoom the view
                            // underneath it.
                            if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                                if content_rect.contains(pointer)
                                    && ui.ctx().layer_id_at(pointer) == Some(ui.layer_id())
                                {
                                    let pinch = ui.input(|i| i.zoom_delta());
                                    let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
                                    // A mouse wheel/trackpad notch is tens
                                    // of points; this scale was tuned by
                                    // feel to land in the same ballpark as
                                    // a single pinch gesture's own
                                    // zoom_delta step.
                                    let scroll_zoom = (scroll_y * 0.002).exp();
                                    let raw_delta = pinch * scroll_zoom;
                                    if raw_delta != 1.0 {
                                        let old_zoom = two_d_view.zoom;
                                        two_d_view.zoom = (old_zoom * raw_delta).clamp(0.25, 32.0);
                                        // Re-derive the factor actually
                                        // applied (clamping can shrink it)
                                        // so the anchor correction below
                                        // stays exact instead of
                                        // overshooting at the zoom limits.
                                        let applied = two_d_view.zoom / old_zoom;
                                        let reference = if two_d_view.tiling_preview {
                                            content_rect.min
                                        } else {
                                            content_rect.center()
                                        };
                                        let anchor = pointer - reference;
                                        two_d_view.pan =
                                            anchor * (1.0 - applied) + two_d_view.pan * applied;
                                    }
                                }
                            }

                            // Drag-to-pan, under the editor's own point/handle
                            // dragging — while a Path node is selected, this
                            // same response drives that instead (see below);
                            // panning a vector editor mid-edit would be more
                            // confusing than useful, and reusing one interact
                            // region avoids two senses racing for the same
                            // pointer input.
                            let drag_response = ui.interact(
                                content_rect,
                                ui.id().with("two_d_view_drag"),
                                egui::Sense::click_and_drag(),
                            );
                            let editing_path = selected
                                .and_then(|id| graph.node(id))
                                .is_some_and(|n| n.kind == "path");
                            if drag_response.dragged() && !editing_path {
                                two_d_view.pan += drag_response.drag_delta();
                            }

                            if two_d_view.tiling_preview {
                                // Each tile is a third of the shorter side, so
                                // the panel always shows a handful of full
                                // repeats regardless of its own size/aspect —
                                // "fills the window" rather than a fixed 2x2/3x3
                                // count that could leave empty space in a large
                                // or oddly-shaped panel. Scaled by `zoom` on
                                // top of that base size.
                                let tile_px = (content_rect.width().min(content_rect.height())
                                    / 3.0)
                                    .max(16.0)
                                    * two_d_view.zoom;
                                let uv_rect = egui::Rect::from_min_size(
                                    egui::pos2(
                                        -two_d_view.pan.x / tile_px,
                                        -two_d_view.pan.y / tile_px,
                                    ),
                                    egui::vec2(
                                        content_rect.width() / tile_px,
                                        content_rect.height() / tile_px,
                                    ),
                                );
                                ui.painter().image(
                                    texture_id,
                                    content_rect,
                                    uv_rect,
                                    egui::Color32::WHITE,
                                );
                            } else {
                                let size = content_rect.width().min(content_rect.height())
                                    * two_d_view.zoom;
                                // Clamped so a drag can slide the tile around its
                                // padded box but never off it entirely — `pan`
                                // itself is left unclamped (not reassigned here)
                                // so a wilder pan is still there, restored
                                // automatically, if tiling is turned back on.
                                let half_slack =
                                    ((content_rect.size() - egui::Vec2::splat(size)) * 0.5).abs();
                                let pan = egui::vec2(
                                    two_d_view.pan.x.clamp(-half_slack.x, half_slack.x),
                                    two_d_view.pan.y.clamp(-half_slack.y, half_slack.y),
                                );
                                let image_rect = egui::Rect::from_center_size(
                                    content_rect.center() + pan,
                                    egui::Vec2::splat(size),
                                );
                                // Clipped to `content_rect` — once zoomed in,
                                // `size` can exceed `content_rect`'s own
                                // dimensions, and an unclipped painter let
                                // the image bleed out over the toolbar rail
                                // and neighboring panels instead of stopping
                                // at the 2D View's own edge.
                                ui.painter_at(content_rect).image(
                                    texture_id,
                                    image_rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );

                                if editing_path {
                                    if let Some(id) = *selected {
                                        path_editor_deleted_a_point = path_editor::show(
                                            ui,
                                            image_rect,
                                            content_rect,
                                            graph,
                                            history,
                                            id,
                                            path_editor,
                                            &drag_response,
                                        );
                                    }
                                }
                            }
                        }
                        None => {
                            // A plain `ui.add_space`/`ui.horizontal` here
                            // would lay out from the panel's own default
                            // cursor, starting under the toolbar rail —
                            // anchoring to `toolbar_content_rect` keeps
                            // this text clear of it, same as the toolbar
                            // itself being anchored rather than laid out
                            // inline.
                            let mut content_ui =
                                ui.new_child(egui::UiBuilder::new().max_rect(toolbar_content_rect));
                            content_ui.add_space(8.0);
                            content_ui.horizontal(|ui| {
                                ui.add_space(8.0);
                                ui.weak("(select a node to preview it)");
                            });
                        }
                    }
                }
                dock::CrateId::ThreeDView => {
                    const TOOLBAR_WIDTH: f32 = 38.0;
                    let full_rect = ui.available_rect_before_wrap();
                    let divider_stroke = ui.visuals().window_stroke();

                    let toolbar_rect = egui::Rect::from_min_size(
                        full_rect.min,
                        egui::vec2(TOOLBAR_WIDTH, full_rect.height()),
                    );
                    {
                        let mut toolbar_ui =
                            ui.new_child(egui::UiBuilder::new().max_rect(toolbar_rect));
                        toolbar_ui.set_clip_rect(toolbar_rect);
                        viewport_panels.toolbar_vertical(&mut toolbar_ui);
                    }
                    ui.painter().vline(
                        toolbar_rect.right(),
                        toolbar_rect.y_range(),
                        divider_stroke,
                    );

                    // The viewport always fills everything right of the
                    // toolbar — the side panel overlaps it instead of
                    // squeezing it narrower, matching the classic N-panel
                    // behavior (Blender) rather than a push-aside layout.
                    let viewport_rect = egui::Rect::from_min_max(
                        egui::pos2(toolbar_rect.right() + 1.0, full_rect.top()),
                        full_rect.max,
                    );
                    let response = ui.interact(
                        viewport_rect,
                        ui.id().with("viewport3d_interact"),
                        egui::Sense::click_and_drag(),
                    );
                    viewport3d.handle_input(ui, &response);
                    let texture_id = viewport3d.render(
                        gpu,
                        pool,
                        graph,
                        render_state,
                        viewport_rect.width().max(1.0) as u32,
                        viewport_rect.height().max(1.0) as u32,
                    );
                    ui.painter().image(
                        texture_id,
                        viewport_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );

                    if drawer_width > 0.5 {
                        let drawer_rect = egui::Rect::from_min_size(
                            egui::pos2(viewport_rect.right() - drawer_width, viewport_rect.top()),
                            egui::vec2(drawer_width, viewport_rect.height()),
                        );
                        ui.painter()
                            .rect_filled(drawer_rect, 0.0, theme::NEUTRON_STAR);
                        ui.painter().rect_stroke(
                            drawer_rect,
                            0.0,
                            divider_stroke,
                            egui::StrokeKind::Inside,
                        );
                        // Absorbs clicks/drags over empty space in the
                        // drawer so they don't fall through to the
                        // viewport's orbit-drag interact underneath it.
                        ui.interact(
                            drawer_rect,
                            ui.id().with("viewport3d_drawer_bg"),
                            egui::Sense::click_and_drag(),
                        );

                        let inner_rect = drawer_rect.shrink2(egui::vec2(10.0, 8.0));
                        let mut drawer_ui =
                            ui.new_child(egui::UiBuilder::new().max_rect(inner_rect));
                        drawer_ui.set_clip_rect(inner_rect);

                        // A short 3D View panel (e.g. squeezed by the Node
                        // Graph dock cell below it) can leave less room than
                        // a panel's content needs — scrollable so every
                        // control stays reachable instead of just clipping
                        // off the bottom with no way back to it.
                        egui::ScrollArea::vertical()
                            .id_salt("viewport3d_drawer_scroll")
                            .auto_shrink([false, false])
                            .show(&mut drawer_ui, |ui| {
                                let kind = viewport_panels.visible_kind();
                                ui.heading(kind.title());
                                ui.weak(kind.description());
                                ui.separator();
                                match kind {
                                    viewport3d_panels::PanelKind::Mesh => {
                                        egui::ComboBox::from_label("Shape")
                                            .selected_text(viewport3d.mesh_kind.label())
                                            .show_ui(ui, |ui| {
                                                for mesh_kind in viewport3d::MeshKind::ALL {
                                                    ui.selectable_value(
                                                        &mut viewport3d.mesh_kind,
                                                        mesh_kind,
                                                        mesh_kind.label(),
                                                    );
                                                }
                                            });
                                    }
                                    viewport3d_panels::PanelKind::Material => {
                                        ui.spacing_mut().slider_width = 80.0;
                                        ui.add(
                                            egui::Slider::new(
                                                &mut viewport3d.height_scale,
                                                0.0..=0.5,
                                            )
                                            .text("Height Scale"),
                                        );
                                        // Ceiling is fixed, independent of the graph's own
                                        // output Resolution — see `TESSELLATION_MAX`'s own doc
                                        // comment for why a lower-resolution graph shouldn't
                                        // also silently cap this.
                                        ui.add(
                                            egui::Slider::new(
                                                &mut viewport3d.tessellation,
                                                viewport3d::TESSELLATION_MIN
                                                    ..=viewport3d::TESSELLATION_MAX,
                                            )
                                            .text("Tessellation")
                                            .logarithmic(true),
                                        )
                                        .on_hover_text(
                                            "Drag, or click the number to type an exact value.",
                                        );
                                        ui.separator();
                                        let mut unlit = viewport3d.shading_mode
                                            == viewport3d::ShadingMode::UnlitBaseColor;
                                        if ui
                                            .checkbox(&mut unlit, "Unlit Base Color")
                                            .on_hover_text(
                                                "Shows Base Color directly, with no lighting, \
                                         exposure, or tonemapping \u{2014} for checking it \
                                         matches the 2D View and an exported PNG exactly.",
                                            )
                                            .changed()
                                        {
                                            viewport3d.shading_mode = if unlit {
                                                viewport3d::ShadingMode::UnlitBaseColor
                                            } else {
                                                viewport3d::ShadingMode::Lit
                                            };
                                        }
                                        ui.add_enabled(
                                            !unlit,
                                            egui::Slider::new(&mut viewport3d.exposure, 0.1..=4.0)
                                                .text("Exposure"),
                                        );
                                        ui.separator();
                                        let mut pixel_filter = viewport3d.texture_filter
                                            == viewport3d::TextureFilter::Pixel;
                                        if ui
                                    .checkbox(&mut pixel_filter, "Pixel / Nearest Filtering")
                                    .on_hover_text(
                                        "Disables all texture smoothing (texel and mip) \u{2014} \
                                         for inspecting dither and pixel-art materials crisply \
                                         instead of blended.",
                                    )
                                    .changed()
                                {
                                    viewport3d.texture_filter = if pixel_filter {
                                        viewport3d::TextureFilter::Pixel
                                    } else {
                                        viewport3d::TextureFilter::Smooth
                                    };
                                }
                                    }
                                    viewport3d_panels::PanelKind::Environment => {
                                        let selected_label = match &viewport3d.environment_source {
                                            viewport3d::EnvironmentSource::Preset(preset) => {
                                                preset.label()
                                            }
                                            viewport3d::EnvironmentSource::Custom(name) => {
                                                name.as_str()
                                            }
                                        };
                                        egui::ComboBox::from_label("HDRI")
                                            .selected_text(selected_label)
                                            .show_ui(ui, |ui| {
                                                for preset in hdri::HdriPreset::ALL {
                                                    let is_selected = viewport3d.environment_source
                                                        == viewport3d::EnvironmentSource::Preset(
                                                            preset,
                                                        );
                                                    if ui
                                                        .selectable_label(
                                                            is_selected,
                                                            preset.label(),
                                                        )
                                                        .clicked()
                                                        && !is_selected
                                                    {
                                                        viewport3d
                                                            .set_environment_preset(gpu, preset);
                                                    }
                                                }
                                            });
                                        if let Some(texture_id) = viewport3d.environment_thumbnail()
                                        {
                                            ui.add_space(4.0);
                                            let width = ui.available_width();
                                            ui.image(egui::load::SizedTexture::new(
                                                texture_id,
                                                egui::vec2(width, width * 0.5),
                                            ));
                                            ui.add_space(4.0);
                                        }
                                        if ui.button("Load custom HDRI\u{2026}").clicked() {
                                            *pending_hdri_load =
                                                Some(async_dialog::PendingDialog::new(
                                                    rfd::AsyncFileDialog::new()
                                                        .add_filter("HDRI", &["hdr", "exr"])
                                                        .pick_file(),
                                                ));
                                        }
                                        ui.separator();
                                        ui.spacing_mut().slider_width = 80.0;
                                        ui.add(
                                            egui::Slider::new(
                                                &mut viewport3d.environment_intensity,
                                                0.0..=4.0,
                                            )
                                            .text("Brightness"),
                                        )
                                        .on_hover_text(
                                            "Scales just the HDRI's contribution (background \
                                     reflections and ambient fill) \u{2014} independent of the \
                                     Material panel's overall Exposure.",
                                        );
                                        ui.add(
                                    egui::Slider::new(&mut viewport3d.environment_blur, 0.0..=1.0)
                                        .text("Blur"),
                                )
                                .on_hover_text(
                                    "Softens the environment reflected in the material, for \
                                     shiny/metallic surfaces that would otherwise mirror the \
                                     HDRI back sharply.",
                                );
                                    }
                                    viewport3d_panels::PanelKind::Camera => {}
                                }
                            });
                    }
                }
                dock::CrateId::NodeGraph => {
                    canvas_output = Some(canvas::show(
                        ui,
                        graph,
                        history,
                        canvas,
                        selected,
                        preview_node,
                        preview_port,
                        gpu,
                        pool,
                        render_state,
                        thumbnail_ids,
                    ));
                }
                dock::CrateId::NodeLibrary => {
                    show_node_library(ui, graph, history, canvas, selected, preview_node);
                }
            },
        );

        let canvas_output = canvas_output.unwrap_or(canvas::CanvasOutput {
            open_add_menu_at: None,
            dropped_wire: None,
            connect_rejected: None,
        });

        if let Some(reason) = &canvas_output.connect_rejected {
            self.toasts.error(reason.clone());
        }

        // Cmd+A (select all) / Delete, skipped while a text field (e.g. the
        // add-node search box, a parameter's numeric entry) has focus, so
        // those keys keep doing normal text editing instead of being
        // hijacked into graph-wide actions.
        if ui.memory(|m| m.focused()).is_none() {
            let (select_all, delete_pressed, new_pressed, open_pressed, save_pressed) =
                ui.ctx().input(|i| {
                    (
                        i.modifiers.command && i.key_pressed(egui::Key::A),
                        i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace),
                        i.modifiers.command && i.key_pressed(egui::Key::N),
                        i.modifiers.command && i.key_pressed(egui::Key::O),
                        i.modifiers.command && i.key_pressed(egui::Key::S),
                    )
                });
            // Shift+Cmd+S (Save As) vs. plain Cmd+S (Save): read `modifiers`
            // separately from the tuple above, since `key_pressed` already
            // consumed the event by the time `save_pressed` is true.
            let save_as = save_pressed && ui.ctx().input(|i| i.modifiers.shift);

            if select_all {
                self.canvas.selected = self.graph.nodes().map(|(id, _)| id).collect();
            }

            if delete_pressed && !path_editor_deleted_a_point && !self.canvas.selected.is_empty() {
                let ids: Vec<mc_core::NodeId> = self.canvas.selected.iter().copied().collect();
                let transaction = self.graph.remove_nodes_transaction(&ids);
                let _ = self.history.do_transaction(&mut self.graph, transaction);
                if self.selected.is_some_and(|id| ids.contains(&id)) {
                    self.selected = None;
                }
                if self.preview_node.is_some_and(|id| ids.contains(&id)) {
                    self.preview_node = None;
                }
                self.canvas.selected.clear();
            }

            if delete_pressed && !path_editor_deleted_a_point {
                match self.canvas.selected_annotation {
                    Some(canvas::AnnotationSelection::Comment(id)) => {
                        let transaction = self.graph.remove_comment_transaction(id);
                        let _ = self.history.do_transaction(&mut self.graph, transaction);
                        self.canvas.selected_annotation = None;
                    }
                    Some(canvas::AnnotationSelection::Frame(id)) => {
                        let transaction = self.graph.remove_frame_transaction(id);
                        let _ = self.history.do_transaction(&mut self.graph, transaction);
                        self.canvas.selected_annotation = None;
                    }
                    None => {}
                }
            }

            if new_pressed {
                self.do_new();
            }
            if open_pressed {
                self.do_open();
            }
            if save_pressed {
                if save_as {
                    self.do_save_as();
                } else {
                    self.do_save();
                }
            }
        }

        if self.add_node_menu.is_none() {
            if let Some(dropped) = canvas_output.dropped_wire {
                self.add_node_menu = Some(add_node_menu::AddNodeMenu::open_for_dropped_wire(
                    dropped.screen_pos,
                    dropped.graph_pos,
                    add_node_menu::DropFilter {
                        port_type: dropped.from_port_type,
                        need_input: dropped.from_is_output,
                    },
                ));
                self.pending_wire_connect = Some(dropped);
            } else if let Some((screen_pos, graph_pos)) = canvas_output.open_add_menu_at {
                self.add_node_menu = Some(add_node_menu::AddNodeMenu::open(screen_pos, graph_pos));
                self.pending_wire_connect = None;
            }
        }
        if let Some((def, graph_pos)) = add_node_menu::show(ui.ctx(), &mut self.add_node_menu) {
            let node = def.instantiate();
            let (core_id, transaction) = self.graph.add_node_transaction(node);
            if self
                .history
                .do_transaction(&mut self.graph, transaction)
                .is_ok()
            {
                let _ = self
                    .graph
                    .set_node_position(core_id, [graph_pos.x, graph_pos.y]);
                self.selected = Some(core_id);
                self.preview_node = Some(core_id);
                self.canvas.selected.clear();
                self.canvas.selected.insert(core_id);
                if let Some(dropped) = self.pending_wire_connect.take() {
                    canvas::connect_dropped_wire(
                        &mut self.graph,
                        &mut self.history,
                        &dropped,
                        core_id,
                    );
                }
            }
        }

        export_dialog::show(
            ui.ctx(),
            &mut self.export_dialog,
            &mut self.gpu,
            &mut self.pool,
            &mut self.graph,
        );

        svg_import_dialog::show(
            ui.ctx(),
            &mut self.svg_import_dialog,
            &mut self.graph,
            &mut self.history,
        );

        // Image Trace's own "Export SVG…" button (`params_panel.rs`) —
        // handled here rather than there since it needs `gpu`/`pool`
        // together with `graph`, same reasoning as `export_dialog`'s own
        // real export. Re-traces from the current Image input rather
        // than re-deriving an SVG from the already-computed (and
        // tile-unit, collinear-point-dropped) `path` param, so the
        // export is a faithful re-run of the same pipeline Preview Mask
        // and Path came from, not a second, lossier conversion of their
        // own output.
        if let Some(id) = self.pending_svg_export.take() {
            let resolution = self.graph.settings.resolution;
            let svg = (|| {
                let (input_id, from_port) = self.graph.upstream_from(id, "image")?;
                let input_key = mc_nodes::registry::texture_key(input_id, &from_port);
                let texture = self.pool.texture(&input_key)?;
                let bytes = self.gpu.read_rgba8(texture, resolution);
                let params = self
                    .graph
                    .node(id)
                    .map(|n| mc_nodes::image_trace::TraceParams::from_graph(&n.params))
                    .unwrap_or_default();
                let contours = mc_nodes::image_trace::trace_from_rgba8(
                    bytes, resolution, resolution, &params,
                )?;
                Some(mc_nodes::image_trace::to_svg(
                    &contours, resolution, resolution,
                ))
            })();

            match svg {
                Some(svg) => {
                    self.pending_svg_export_save = Some((
                        svg,
                        async_dialog::PendingDialog::new(
                            rfd::AsyncFileDialog::new()
                                .add_filter("SVG", &["svg"])
                                .set_file_name("trace.svg")
                                .save_file(),
                        ),
                    ));
                }
                None => log::warn!("image_trace export: nothing to export for node {id:?}"),
            }
        }

        if let Some((svg, pending)) = &mut self.pending_svg_export_save {
            if let Some(result) = pending.poll(ui.ctx()) {
                if let Some(handle) = result {
                    let path = handle.path().to_path_buf();
                    if let Err(err) = std::fs::write(&path, &*svg) {
                        log::warn!("failed to write {}: {err}", path.display());
                    }
                }
                self.pending_svg_export_save = None;
            }
        }

        if let Some(pending) = &mut self.pending_open {
            if let Some(result) = pending.poll(ui.ctx()) {
                self.pending_open = None;
                if let Some(handle) = result {
                    self.open_path(handle.path().to_path_buf());
                }
            }
        }

        if let Some(pending) = &mut self.pending_save_as {
            if let Some(result) = pending.poll(ui.ctx()) {
                self.pending_save_as = None;
                if let Some(handle) = result {
                    self.save_to(handle.path().to_path_buf());
                }
            }
        }

        if let Some(pending) = &mut self.pending_hdri_load {
            if let Some(result) = pending.poll(ui.ctx()) {
                self.pending_hdri_load = None;
                if let Some(handle) = result {
                    let path = handle.path();
                    if let Err(err) = self.viewport3d.set_environment_custom(&self.gpu, path) {
                        log::warn!("failed to load HDRI {}: {err}", path.display());
                        self.toasts
                            .error(format!("Couldn't load {}: {err}", path.display()));
                    }
                }
            }
        }

        // egui/eframe only repaints in response to new OS input (mouse
        // move, click, key) — it does not automatically notice that a graph
        // edit this frame (connect, param change, add/remove node,
        // undo/redo) left a node dirty. Without this, the node this frame's
        // edit affects sits stale/blank until some unrelated input event
        // happens to trigger another frame; this guarantees the follow-up
        // frame that actually runs `evaluate()` on it.
        if self
            .graph
            .nodes()
            .any(|(id, _)| self.graph.needs_evaluation(id))
        {
            ui.ctx().request_repaint();
        }
    }

    /// A graceful shutdown means there's nothing left for the recovery
    /// snapshot to protect against — leaving it behind is what signals an
    /// *unclean* exit (crash, force-quit) to the next launch.
    fn on_exit(&mut self) {
        autosave::Autosave::clear();
    }
}

impl MaterialCreatorApp {
    /// Swaps in a different graph wholesale (New/Open) — resets every piece
    /// of state that's keyed off the *previous* graph's nodes (history,
    /// selection, cached thumbnails/preview texture, the GPU texture pool),
    /// not just the graph field itself, so nothing stale lingers (a node ID
    /// from the old graph reused by the new one would otherwise find an
    /// orphaned texture already sitting in the pool under that key).
    /// Evaluates whatever's currently dirty and reports the result exactly
    /// like every frame always has — factored out so both call sites in
    /// `ui()` (the ordinary every-frame case, and the one deferred frame
    /// right after a heavy-recompute trigger) share the same error
    /// handling instead of risking it drifting apart between two copies.
    fn run_graph_evaluation(&mut self) {
        match mc_nodes::registry::evaluate(&mut self.gpu, &mut self.pool, &mut self.graph) {
            Ok(()) => self.last_eval_error = None,
            // A required input not being connected yet is completely normal
            // mid-edit state (e.g. a node just dragged in, not wired up
            // yet) — `registry::evaluate`'s own doc comment calls this out
            // explicitly. It isn't worth alarming the artist over; the
            // node's pin/wire coloring already shows what's unconnected.
            Err(mc_nodes::registry::EvalError::MissingInput(..)) => {
                self.last_eval_error = None;
            }
            Err(err) => {
                log::warn!("graph evaluation error: {err}");
                let message = err.to_string();
                if self.last_eval_error.as_deref() != Some(message.as_str()) {
                    self.toasts
                        .error(format!("Graph evaluation failed: {message}"));
                }
                self.last_eval_error = Some(message);
            }
        }
    }

    fn replace_graph(&mut self, graph: mc_core::Graph, path: Option<std::path::PathBuf>) {
        self.graph = graph;
        self.current_path = path;
        self.history = mc_core::History::new();
        self.canvas = CanvasState::default();
        self.thumbnail_ids.clear();
        self.selected = None;
        self.preview_node = None;
        self.add_node_menu = None;
        self.pending_wire_connect = None;
        self.preview_texture_id = None;
        self.pool = mc_gpu::TexturePool::default();
        // A fresh or just-opened graph has no unsaved history of its own
        // yet, so any leftover crash-recovery snapshot from a previous
        // session is now stale.
        autosave::Autosave::clear();
        // A just-loaded graph's very first evaluation is everything, at
        // whatever resolution it was saved at — worth the same
        // "Recalculating…" warning as a Resolution change (see
        // `recompute_in_progress`'s own doc comment).
        self.recompute_in_progress = true;
        self.last_known_resolution = self.graph.settings.resolution;
        self.viewport3d.height_scale = self.graph.settings.height_scale;
    }

    fn do_new(&mut self) {
        // Same "pick a starting point" choice as a fresh launch, rather
        // than silently dropping straight to an empty graph — see
        // `splash.rs`'s own doc comment.
        self.splash.open = true;
    }

    fn do_open(&mut self) {
        self.pending_open = Some(async_dialog::PendingDialog::new(
            rfd::AsyncFileDialog::new()
                .add_filter("MaterialCreator Graph", &["mcgraph"])
                .pick_file(),
        ));
    }

    fn do_save(&mut self) {
        match self.current_path.clone() {
            Some(path) => self.save_to(path),
            None => self.do_save_as(),
        }
    }

    fn do_save_as(&mut self) {
        self.pending_save_as = Some(async_dialog::PendingDialog::new(
            rfd::AsyncFileDialog::new()
                .add_filter("MaterialCreator Graph", &["mcgraph"])
                .set_file_name("untitled.mcgraph")
                .save_file(),
        ));
    }

    /// Opens `path` as the current graph — the part of `do_open` that
    /// still has to happen once its own async dialog (see `pending_open`)
    /// resolves, factored out so it isn't duplicated between there and
    /// any other future "load this specific path" caller.
    fn open_path(&mut self, path: std::path::PathBuf) {
        match mc_core::Graph::load_from_file(&path) {
            Ok(graph) => self.replace_graph(graph, Some(path)),
            Err(err) => {
                log::warn!("failed to open {}: {err}", path.display());
                self.toasts
                    .error(format!("Couldn't open {}: {err}", path.display()));
            }
        }
    }

    /// Loads one of `EXAMPLE_GRAPHS` as a fresh starting point — deliberately
    /// `replace_graph(graph, None)`, not `open_path`'s own `Some(path)`:
    /// an example is a template to build on, the same way the splash
    /// screen's own Blender/Unreal presets are, not a real file the artist
    /// is now editing in place. Leaving `current_path` unset means "Save"
    /// prompts "Save As" instead of silently overwriting the checked-in
    /// example.
    fn load_example(&mut self, path: &str) {
        let resolved = resolve_example_path(path);
        match mc_core::Graph::load_from_file(&resolved) {
            Ok(graph) => self.replace_graph(graph, None),
            Err(err) => {
                log::warn!("failed to open example {}: {err}", resolved.display());
                self.toasts
                    .error(format!("Couldn't open {}: {err}", resolved.display()));
            }
        }
    }

    fn save_to(&mut self, path: std::path::PathBuf) {
        self.graph.settings.height_scale = self.viewport3d.height_scale;
        match self.graph.save_to_file(&path) {
            Ok(()) => {
                self.toasts.info(format!("Saved {}", path.display()));
                self.current_path = Some(path);
                // The file just written is now the authoritative copy, so
                // any crash-recovery snapshot from before this save is
                // stale.
                autosave::Autosave::clear();
            }
            Err(err) => {
                log::warn!("failed to save {}: {err}", path.display());
                self.toasts
                    .error(format!("Couldn't save {}: {err}", path.display()));
            }
        }
    }

    /// `preview_node`'s own resolved output texture, registered/updated as
    /// an egui texture for the 2D view. Prefers `preview_port` (a specific
    /// output someone clicked on, for a multi-output node) when it's
    /// actually still valid for `id` — still checked against the node's own
    /// `NodeDef` here rather than trusted outright, since `preview_port`
    /// can outlive the node selection that set it (e.g. undo recreating the
    /// node with the same id but a different kind is extremely unlikely,
    /// but a stale port name silently resolving to *some other* node's
    /// texture under the same key format would be a confusing bug to hit).
    fn preview_texture(&mut self) -> Option<egui::TextureId> {
        let id = self.preview_node?;
        let kind = self.graph.node(id)?.kind.clone();
        // Resolves to the explicitly-previewed output *port* (not just a
        // texture key) when `preview_port` names one of `id`'s own current
        // outputs, so its real `PortType` can be checked too — reusing this
        // same resolution for both the key and the type keeps them from
        // ever disagreeing about which port is actually on screen.
        let explicit_port = match &self.preview_port {
            Some((port_node, port_name)) if *port_node == id => mc_nodes::defs::node_def(&kind)
                .and_then(|def| def.outputs.iter().find(|p| p.name == port_name)),
            _ => None,
        };
        let (key, is_sdf) = if let Some(port) = explicit_port {
            let port_type =
                mc_nodes::defs::effective_output_type(&self.graph, id, &kind, port.port_type);
            (
                mc_nodes::registry::texture_key(id, port.name),
                port_type == mc_core::PortType::Sdf,
            )
        } else {
            let key = mc_nodes::registry::resolve_texture_key(&self.graph, id)?;
            let is_sdf = mc_nodes::registry::resolve_output_port_type(&self.graph, id)
                == Some(mc_core::PortType::Sdf);
            (key, is_sdf)
        };
        // Node textures are linear; egui never decodes what it's given, so
        // this view has to already be sRGB-encoded or the preview reads too
        // dark/off relative to the 3D viewport and an exported PNG — see
        // `display_encode.rs`. `is_sdf` renders iso-bands instead, in which
        // case the Channel View selector below has no effect (an Sdf port
        // is always single-channel) — not hiding that selector's UI yet,
        // this step is plumbing only (HARDSURFACE_NODES.md step 1).
        let view = crate::display_encode::encode_for_display(
            &mut self.gpu,
            &mut self.pool,
            &key,
            self.two_d_view.channel_view,
            is_sdf,
        )?;

        // Repeat (not egui's default ClampToEdge) addressing: identical to
        // before for the normal single-tile view (UV never leaves [0, 1]
        // there), but also lets the 2D View's tiling-preview toggle sample
        // UVs past 1.0 and get the real wrapped texture back instead of a
        // smeared repeat of the edge texel.
        let sampler_descriptor = wgpu::SamplerDescriptor {
            label: Some("preview_texture_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        };

        Some(match self.preview_texture_id {
            Some(texture_id) => {
                self.render_state
                    .renderer
                    .write()
                    .update_egui_texture_from_wgpu_texture_with_sampler_options(
                        &self.gpu.device,
                        &view,
                        sampler_descriptor,
                        texture_id,
                    );
                texture_id
            }
            None => {
                let texture_id = self
                    .render_state
                    .renderer
                    .write()
                    .register_native_texture_with_sampler_options(
                        &self.gpu.device,
                        &view,
                        sampler_descriptor,
                    );
                self.preview_texture_id = Some(texture_id);
                texture_id
            }
        })
    }
}

/// Display order for the Node Library panel's category groups — the
/// `mc_nodes::defs` category strings, just in a deliberate reading order
/// rather than their incidental declaration order in that file.
const NODE_CATEGORY_ORDER: [&str; 10] = [
    "Generators/Noise",
    "Generators/Patterns",
    "Generators/Path",
    "Filters/Adjustments",
    "Filters/Blending & Masking",
    "Filters/Spatial",
    "Filters/Stylize",
    "Filters/Glitch",
    "Converters",
    "Output",
];

/// A short label for a category group header — the panel is narrow, so the
/// full "Filters/Blending & Masking"-style string wouldn't fit on one line.
fn category_short_name(category: &'static str) -> &'static str {
    match category {
        "Generators/Noise" => "Noise",
        "Generators/Patterns" => "Patterns",
        "Generators/Path" => "Path",
        "Filters/Adjustments" => "Adjust",
        "Filters/Blending & Masking" => "Blend/Mask",
        "Filters/Spatial" => "Spatial",
        "Filters/Stylize" => "Stylize",
        "Filters/Glitch" => "Glitch",
        "Converters" => "Convert",
        "Output" => "Output",
        _ => category,
    }
}

/// The Node Library panel's content: every node, grouped by category and
/// color-coded to match (`theme::category_color`), shown as a Material
/// Icons Round glyph, or a hand-drawn exception (`node_icons::draw_icon`)
/// — clicking one adds that
/// node to the graph, the same transaction/selection flow the add-node menu
/// uses, just without a specific click position to place it at.
fn show_node_library(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    canvas: &mut canvas::CanvasState,
    selected: &mut Option<mc_core::NodeId>,
    preview: &mut Option<mc_core::NodeId>,
) {
    const ICON_SIZE: f32 = 32.0;
    const ICON_FONT_SIZE: f32 = 18.0;

    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for &category in &NODE_CATEGORY_ORDER {
                        ui.add_space(6.0);
                        ui.colored_label(
                            theme::category_color(category),
                            category_short_name(category),
                        );
                        ui.add_space(2.0);
                        ui.horizontal_wrapped(|ui| {
                            for def in mc_nodes::defs::all_node_defs()
                                .iter()
                                .filter(|def| def.category == category)
                            {
                                let (rect, response) = ui.allocate_exact_size(
                                    egui::Vec2::splat(ICON_SIZE),
                                    egui::Sense::click(),
                                );
                                if ui.is_rect_visible(rect) {
                                    let bg_alpha = if response.hovered() { 0.55 } else { 0.3 };
                                    ui.painter().rect_filled(
                                        rect,
                                        4.0,
                                        theme::category_color(category).gamma_multiply(bg_alpha),
                                    );
                                    node_icons::draw_icon(
                                        ui.painter(),
                                        rect,
                                        def.kind,
                                        ICON_FONT_SIZE,
                                        theme::SUPERNOVA_LIGHT,
                                    );
                                }
                                let response = response.on_hover_text(def.display_name);
                                if response.clicked() {
                                    add_node_from_library(
                                        graph, history, canvas, selected, preview, def,
                                    );
                                }
                            }
                        });
                    }
                    // Trailing breathing room so the last category's icons
                    // don't sit flush against the scroll area's own bottom
                    // edge.
                    ui.add_space(10.0);
                });
        });
}

fn add_node_from_library(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    canvas: &mut canvas::CanvasState,
    selected: &mut Option<mc_core::NodeId>,
    preview: &mut Option<mc_core::NodeId>,
    def: &'static mc_nodes::defs::NodeDef,
) {
    let node = def.instantiate();
    let (core_id, transaction) = graph.add_node_transaction(node);
    if history.do_transaction(graph, transaction).is_ok() {
        let drop_pos = canvas.last_click_pos();
        let _ = graph.set_node_position(core_id, [drop_pos.x, drop_pos.y]);
        *selected = Some(core_id);
        *preview = Some(core_id);
        canvas.selected.clear();
        canvas.selected.insert(core_id);
    }
}

/// Replaces egui's default fonts with Geist (body text) and registers
/// Material Icons Round under its own named family, used explicitly wherever
/// an icon glyph is needed (see `viewport3d_panels.rs`) rather than mixed
/// into the body font — both bundled under `assets/fonts/`, see ASSETS.md.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    fonts.font_data.insert(
        "geist".to_owned(),
        egui::FontData::from_static(include_bytes!("../../../assets/fonts/Geist-Regular.ttf"))
            .into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "geist".to_owned());

    fonts.font_data.insert(
        "material_icons_round".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../../../assets/fonts/MaterialIconsRound-Regular.otf"
        ))
        .into(),
    );
    fonts.families.insert(
        egui::FontFamily::Name("material_icons_round".into()),
        vec!["material_icons_round".to_owned()],
    );

    ctx.set_fonts(fonts);
}

/// The app's window/taskbar icon: the bundled Material Icons Round "texture"
/// glyph (Apache-2.0, see ASSETS.md) on Neutron Star, in Supernova Light —
/// generated once from the brand palette and committed as a PNG rather than
/// rendered at startup, so this has zero runtime cost beyond decoding it.
fn app_icon() -> egui::IconData {
    let bytes = include_bytes!("../../../assets/icon/icon_256.png");
    let (rgba, width, height) =
        mc_export::decode_png_rgba8(bytes).expect("bundled app icon PNG is well-formed");
    egui::IconData {
        rgba,
        width,
        height,
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("MaterialCreator")
            .with_icon(app_icon())
            // Opens filling the screen (maximized, not exclusive
            // fullscreen — the OS menu bar/dock stay reachable) so the
            // splash screen's own centering lands in the middle of the
            // actual screen on first paint, not a small default window
            // size the splash would otherwise look lost inside of.
            .with_maximized(true),
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
                // eframe's own default asks for `wgpu::Limits::default()`,
                // which caps every GPU buffer at 256 MiB regardless of what
                // the actual hardware supports — a real wall an earlier,
                // since-replaced CPU-built mesh for the 3D viewport's
                // Tessellation slider actually hit. The current GPU vertex-
                // pulling tessellation path (`viewport3d.wgsl`'s
                // `vs_main_tess`) doesn't upload anything that scales with
                // Tessellation, so this is now a safety margin rather than a
                // known-reachable limit — still worth asking for the
                // adapter's own real limits instead of the conservative
                // default. On the GL backend (no real adapter-reported
                // limits to trust) keep the conservative WebGL2 defaults.
                device_descriptor: Arc::new(|adapter| {
                    let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        adapter.limits()
                    };
                    wgpu::DeviceDescriptor {
                        label: Some("mc_app wgpu device"),
                        required_limits: wgpu::Limits {
                            max_texture_dimension_2d: 8192,
                            ..base_limits
                        },
                        ..Default::default()
                    }
                }),
                ..egui_wgpu::WgpuSetupCreateNew::without_display_handle()
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "MaterialCreator",
        native_options,
        Box::new(|cc| {
            theme::apply(&cc.egui_ctx);
            install_fonts(&cc.egui_ctx);

            let render_state = cc
                .wgpu_render_state
                .clone()
                .expect("mc_app requires eframe's wgpu backend");
            let gpu = mc_gpu::Context::new(render_state.device.clone(), render_state.queue.clone());
            let viewport3d = viewport3d::Viewport3d::new(&gpu);

            let mut toasts = toast::Toasts::default();
            if !viewport3d.tess_supported() {
                // This GPU/driver can't supply enough VERTEX-stage storage
                // buffers for the Cube/Rounded Cube tessellation technique
                // (see `viewport3d::device_supports_tess`) — surfaced here
                // rather than silently downgrading, since it's a hardware
                // limit the artist has no way to notice from the render
                // alone otherwise.
                toasts.info(
                    "This GPU doesn't support the extra-detail technique used for Rounded \
                     Cube/Cube's Height displacement \u{2014} using a standard mesh instead \
                     (sharp height edges will look a bit more faceted).",
                );
            }

            Ok(Box::new(MaterialCreatorApp {
                gpu,
                pool: mc_gpu::TexturePool::default(),
                render_state,
                graph: mc_core::Graph::new(),
                history: mc_core::History::new(),
                canvas: CanvasState::default(),
                thumbnail_ids: HashMap::new(),
                selected: None,
                preview_node: None,
                preview_port: None,
                add_node_menu: None,
                pending_wire_connect: None,
                param_drag: None,
                export_dialog: export_dialog::ExportDialogState::default(),
                svg_import_dialog: svg_import_dialog::SvgImportDialogState::default(),
                pending_svg_export: None,
                pending_open: None,
                pending_save_as: None,
                pending_hdri_load: None,
                pending_svg_export_save: None,
                pending_image_import: None,
                viewport3d,
                viewport_panels: viewport3d_panels::ViewportPanels::default(),
                dock: dock::DockLayout::default(),
                preview_texture_id: None,
                two_d_view: TwoDViewState::default(),
                path_editor: path_editor::PathEditorState::default(),
                current_path: None,
                toasts,
                last_eval_error: None,
                recompute_in_progress: false,
                // Matches `mc_core::Graph::new()`'s own default resolution
                // (the empty placeholder graph above, before the splash
                // screen's own choice calls `replace_graph`) — so the very
                // first frame doesn't spuriously see a "change".
                last_known_resolution: 1024,
                autosave: autosave::Autosave::default(),
                pending_recovery: autosave::recovery_path().exists(),
                splash: splash::SplashState::default(),
            }))
        }),
    )
}

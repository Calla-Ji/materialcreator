// SPDX-License-Identifier: MIT OR Apache-2.0
//! A generic parameter panel: every widget comes from the selected node's
//! `NodeDef` (CLAUDE.md §6.6) — no per-node-kind UI code here, so a new node
//! added in `mc_nodes` gets a working panel for free.

/// A param edit mid-drag: the value it had *before* the drag started, so
/// dragging a slider over many frames produces one undo step, not one per
/// frame (CLAUDE.md §9).
pub type DragState = Option<(mc_core::NodeId, String, mc_core::ParamValue)>;

/// Image Import's own "Load Image…" button's in-flight native dialog —
/// see `async_dialog.rs`'s own doc comment for why this goes through
/// `rfd::AsyncFileDialog` instead of the simpler-looking synchronous
/// `rfd::FileDialog` API. Captures everything `commit_once` needs at the
/// moment the dialog was launched (not read back out of "whatever's
/// selected" once it resolves, which may no longer even be this node by
/// then).
pub struct PendingImageImport {
    node_id: mc_core::NodeId,
    param_name: &'static str,
    embed: bool,
    dialog: crate::async_dialog::PendingDialog<Option<rfd::FileHandle>>,
}

/// One SplitMix64 step: advances `state` and returns a well-distributed
/// 64-bit output. Shared by [`random_seed`] (one output) and
/// [`randomize_all_params`] (a whole stream of them, one call per param) —
/// no `rand` dependency needed for UI convenience features that aren't
/// security-sensitive.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Every font family actually installed on this machine, queried once and
/// cached for the rest of the process — real OS font discovery
/// (`mc_nodes::type_text::list_font_families`) is too slow to re-run every
/// single frame the Type node's params panel happens to be open, and the
/// installed-font set can't change while this app is running anyway.
fn installed_font_families() -> &'static [String] {
    static FAMILIES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    FAMILIES.get_or_init(mc_nodes::type_text::list_font_families)
}

/// A new seed for the dice button — previously just `value.wrapping_add(1)`,
/// which isn't random at all (confirmed bug report: it "just adds one more
/// number"). Seeded from the current wall-clock time for a visually
/// "random" result each click.
fn random_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default();
    splitmix64_next(&mut nanos.wrapping_add(1)) as u32
}

/// Upgrades the Seed dice button into a full "randomize everything" action,
/// for *any* node with a Seed param, not a Knit/Thread-specific hack: every
/// node's own `NodeDef` already carries the min/max/options each
/// `ParamSpecKind` needs to pick a sane random value, the same metadata
/// this whole panel already renders sliders from (this file's own doc
/// comment: "no per-node-kind UI code here"). Owner's explicit call after
/// Seed turned out to only touch Stitch ID on Knit (and only a ply phase
/// offset on Thread): "I want it to touch all available parameters in the
/// node and return random results to the user" — every slider gets
/// reassigned, not just Seed itself.
///
/// `Gradient`/`ImagePath` are left untouched (no sane random gradient, and
/// nothing to randomly pick for a file path). `Text` resets to its own
/// declared default rather than being randomized — Knit's own Chart is the
/// motivating case: leftover custom chart text silently overrides whatever
/// Preset this same randomize pass just picked, which would make Preset
/// look broken, not random; resetting to blank (its default — "use the
/// preset's own chart") avoids that, and the same reasoning is sound for
/// any future Text param in the same situation.
fn randomize_all_params(
    def: &'static mc_nodes::defs::NodeDef,
    seed_param_name: &str,
    new_seed: u32,
) -> Vec<(&'static str, mc_core::ParamValue)> {
    let mut state = (new_seed as u64) ^ 0x2545_F491_4F6C_DD1D;
    let next_unit_f32 = |state: &mut u64| -> f32 {
        // Top 24 bits -> [0, 1): plenty of precision for a UI slider.
        ((splitmix64_next(state) >> 40) as f32) / (1u64 << 24) as f32
    };

    let mut out = Vec::new();
    for param in def.params {
        if param.name == seed_param_name {
            continue; // the new seed itself is set by the caller
        }
        let value = match &param.kind {
            mc_nodes::defs::ParamSpecKind::Float { min, max, .. } => {
                let t = next_unit_f32(&mut state);
                mc_core::ParamValue::Float(min + t * (max - min))
            }
            mc_nodes::defs::ParamSpecKind::Int { min, max, .. } => {
                let span = (*max - *min + 1).max(1) as u64;
                let v = *min + (splitmix64_next(&mut state) % span) as i32;
                mc_core::ParamValue::Int(v)
            }
            mc_nodes::defs::ParamSpecKind::Bool { .. } => {
                mc_core::ParamValue::Bool(splitmix64_next(&mut state).is_multiple_of(2))
            }
            mc_nodes::defs::ParamSpecKind::Enum { options, .. } => {
                let idx = (splitmix64_next(&mut state) as usize) % options.len();
                mc_core::ParamValue::Enum(options[idx].to_string())
            }
            mc_nodes::defs::ParamSpecKind::Color { .. } => mc_core::ParamValue::Color([
                next_unit_f32(&mut state),
                next_unit_f32(&mut state),
                next_unit_f32(&mut state),
                1.0,
            ]),
            mc_nodes::defs::ParamSpecKind::Vec2 { min, max, .. } => {
                let x = min + next_unit_f32(&mut state) * (max - min);
                let y = min + next_unit_f32(&mut state) * (max - min);
                mc_core::ParamValue::Vec2([x, y])
            }
            mc_nodes::defs::ParamSpecKind::Seed { .. } => {
                mc_core::ParamValue::Seed(splitmix64_next(&mut state) as u32)
            }
            mc_nodes::defs::ParamSpecKind::Text { default } => {
                mc_core::ParamValue::Text(default.to_string())
            }
            mc_nodes::defs::ParamSpecKind::Gradient { .. }
            | mc_nodes::defs::ParamSpecKind::ImagePath
            | mc_nodes::defs::ParamSpecKind::Path
            | mc_nodes::defs::ParamSpecKind::FontFamily => continue,
        };
        out.push((param.name, value));
    }
    out
}

/// An explanatory aside under a param — e.g. "this only matters once X is
/// set" — styled to actually read as a comment rather than another label:
/// three-quarters of the body text size (half read as too small to be
/// comfortable; this is the settled middle ground) and italic, matching how
/// comments look in code. The established convention for this kind of
/// per-param note going forward (first used for Thread's Fiber Depth/Fibers
/// per Ply hint below) — reach for this instead of a plain `ui.weak(...)`
/// whenever the text is explaining a param's behavior rather than its live
/// connection state (the `[x]`/`[ ] connected` status lines above stay
/// full-size, since those are closer to a label than an aside).
fn param_comment(ui: &mut egui::Ui, text: &str) {
    let body_size = ui
        .style()
        .text_styles
        .get(&egui::TextStyle::Body)
        .map(|font| font.size)
        .unwrap_or(14.0);
    ui.label(
        egui::RichText::new(text)
            .size(body_size * 0.75)
            .italics()
            .weak(),
    );
}

/// Fiber Depth carves relative to a *ply's own* radius, but Height
/// normalizes against the whole bundle's radius — so how much of Height's
/// own 0..1 range the groove can ever reach is capped at
/// `Fiber Depth * (ply radius / bundle radius)`, and that ratio is set by
/// Plies and Cable Groups (packing() 's own fixed ratios), *not* Threads —
/// confirmed by rendering the same comparison at Threads 1 and Threads 4
/// and getting an identical difference, after an earlier version of this
/// hint claimed Threads mattered here too. The full, undiluted groove
/// pattern is always in the Fiber output itself (its whole point, per the
/// spec — wire it in as a separate roughness/AO detail instead of judging
/// Fiber Twist from Height alone).
fn thread_fiber_param_hint(ui: &mut egui::Ui, node_kind: &str, param_name: &str) {
    if node_kind == "thread_shape"
        && matches!(param_name, "fibers_per_ply" | "fiber_depth" | "fiber_mode")
    {
        param_comment(
            ui,
            "How visible this is in Height (not in the Fiber output itself) is capped by \
             Plies/Cable Groups, not Threads \u{2014} for the full pattern, use the Fiber \
             output directly.",
        );
    }
}

/// Seed only ever shifts where a *ply* sits in its own orbit around the
/// thread axis — a single (unplied) yarn has no orbit to shift, so Seed is
/// a genuine no-op at Plies 1 (confirmed: `thread_shape`'s own
/// `seed_only_has_an_effect_with_two_or_more_plies` test). Explained here
/// rather than hiding the control outright — a user who already has a
/// seed dialed in on a Plies-2+ thread shouldn't lose that value from view
/// just because they're trying Plies 1.
fn thread_seed_hint(
    ui: &mut egui::Ui,
    node_kind: &str,
    param_name: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "thread_shape" || param_name != "seed" {
        return;
    }
    let plies = current_params
        .get("plies")
        .and_then(mc_core::ParamValue::as_int)
        .unwrap_or(2);
    if plies <= 1 {
        param_comment(
            ui,
            "No effect at Plies 1 \u{2014} a single yarn has no ply orbit for Seed to \
             shift.",
        );
    }
}

/// `knit_reference.py`'s own fallback for `plies <= 1`: Ply ID reads
/// exactly the same as Mask (there's only one strand, so "which strand is
/// on top" has nothing to distinguish) — confirmed by `mc_nodes::knit`'s
/// own `ply_id_matches_mask_for_a_single_ply` test. Without this, a user
/// wiring up Ply ID at the default Plies 1 sees what looks like a second
/// copy of Mask and reasonably assumes something's broken (reported
/// exactly that way: "Ply ID... does not work").
fn knit_ply_id_hint(
    ui: &mut egui::Ui,
    node_kind: &str,
    param_name: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "knit" || param_name != "plies" {
        return;
    }
    let plies = current_params
        .get("plies")
        .and_then(mc_core::ParamValue::as_int)
        .unwrap_or(1);
    if plies <= 1 {
        param_comment(
            ui,
            "At Plies 1, Ply ID reads identically to Mask \u{2014} there's only one strand \
             to tell apart. Try Plies 2+ to see each one colored separately.",
        );
    }
}

/// Stitch Type is 1 (white) for every Knit stitch, 0 (black) for every
/// Purl one — so a chart that's entirely one or the other (Jersey's `K`,
/// Reverse Jersey's `P`) reads as a flat, uninterrupted color, not a flaw
/// in the output. Shown right under Preset, the most common place that
/// chart gets picked (editing Chart directly has the same effect, but
/// doesn't need its own copy of this hint — a user already typing a chart
/// by hand is already thinking about which stitches are which).
fn knit_stitch_type_hint(
    ui: &mut egui::Ui,
    node_kind: &str,
    param_name: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "knit" || param_name != "preset" {
        return;
    }
    let params = mc_nodes::knit::KnitParams::from_graph(current_params);
    let g = mc_nodes::knit::resolve(&params);
    let all_knit = g.chart.iter().all(|row| row.iter().all(|&cell| cell));
    let all_purl = g.chart.iter().all(|row| row.iter().all(|&cell| !cell));
    if all_knit || all_purl {
        param_comment(
            ui,
            "This chart is all one stitch type, so Stitch Type reads as a flat color (no \
             flaw \u{2014} try a Rib or Purl preset to see knit/purl contrast).",
        );
    }
}

/// `knit_reference.py`'s own `course_tubes()` returns the plain centerline
/// untouched for `plies <= 1` — the rotation-minimizing-frame math Ply
/// Twist and Twist Angle feed into never runs, so neither one changes the
/// render at all until there's a second strand to arrange (verified:
/// rendering the same tile at Plies 2 with Ply Twist S vs Z differs in
/// about two-thirds of the image's bytes; at Plies 1 the two are
/// byte-identical). Same root cause as Ply ID reading as Mask above.
fn knit_ply_arrangement_hint(
    ui: &mut egui::Ui,
    node_kind: &str,
    param_name: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "knit" || !matches!(param_name, "ply_direction" | "twist_angle") {
        return;
    }
    let plies = current_params
        .get("plies")
        .and_then(mc_core::ParamValue::as_int)
        .unwrap_or(1);
    if plies <= 1 {
        param_comment(
            ui,
            "No effect at Plies 1 \u{2014} a single yarn has no ply arrangement for this to \
             change. Try Plies 2+.",
        );
    }
}

/// A dropdown of common named yarn constructions (values confirmed against
/// the owner's own screenshots of each one actually rendered, per
/// `mc_nodes::thread_shape::PRESETS`'s own doc comment) that set several of
/// Thread's params at once — the same one-click-starting-point idea as
/// Pattern Generator's own Shape input, just for an entire construction
/// instead of one silhouette. Every param a preset touches stays a normal,
/// live, editable slider afterward; this is a one-shot action, not a mode
/// — there's deliberately no "which preset is currently active" indicator,
/// since editing even one slider after applying a preset already makes
/// that concept meaningless (the result isn't "that preset" anymore, it's
/// just whatever the sliders currently say).
fn thread_preset_picker(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    node_kind: &str,
) {
    if node_kind != "thread_shape" {
        return;
    }
    let mut chosen = None;
    egui::ComboBox::from_label("Preset")
        .selected_text("Choose\u{2026}")
        .show_ui(ui, |ui| {
            for preset in mc_nodes::thread_shape::PRESETS {
                if ui.selectable_label(false, preset.name).clicked() {
                    chosen = Some(preset);
                }
            }
        });
    if let Some(preset) = chosen {
        commit_once_multi(
            graph,
            history,
            id,
            &[
                ("threads", mc_core::ParamValue::Int(preset.threads as i32)),
                ("thickness", mc_core::ParamValue::Float(preset.thickness)),
                ("plies", mc_core::ParamValue::Int(preset.plies as i32)),
                (
                    "ply_direction",
                    mc_core::ParamValue::Enum(preset.ply_direction.name().to_string()),
                ),
                (
                    "twist_angle",
                    mc_core::ParamValue::Float(preset.twist_angle),
                ),
                (
                    "fiber_mode",
                    mc_core::ParamValue::Enum(preset.fiber_mode.name().to_string()),
                ),
                (
                    "fibers_per_ply",
                    mc_core::ParamValue::Int(preset.fibers_per_ply as i32),
                ),
                (
                    "fiber_depth",
                    mc_core::ParamValue::Float(preset.fiber_depth),
                ),
                (
                    "cable_groups",
                    mc_core::ParamValue::Int(preset.cable_groups as i32),
                ),
                (
                    "cable_angle",
                    mc_core::ParamValue::Float(preset.cable_angle),
                ),
            ],
        );
    }
}

/// Pattern Generator's own version of [`thread_preset_picker`] — a one-shot
/// "Stagger Preset" starting point that sets `offset`/`gap`, not a persisted
/// mode. Deliberately not labeled "Layout" in the UI — that name belongs to
/// the real `layout` param (Grid/Herringbone), an ongoing generation mode
/// the shader checks every frame, which this single-shot action isn't.
/// Unlike Thread's own preset picker, this one *does* show which preset
/// (if any) the current values still match — Offset/Gap are both common,
/// frequently hand-tweaked sliders on their own, so landing on a preset's
/// exact numbers by coincidence (or by picking it) is worth confirming at a
/// glance; it reverts to "Choose…" the moment either is edited away from an
/// exact match, the same "no longer that preset" logic every picker here
/// already has, just surfaced instead of hidden.
fn tile_generator_stagger_preset_picker(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    node_kind: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "tile_generator" {
        return;
    }
    let offset = current_params
        .get("offset")
        .and_then(mc_core::ParamValue::as_float);
    let gap = current_params
        .get("gap")
        .and_then(mc_core::ParamValue::as_float);
    let current = mc_nodes::tile_generator::LAYOUT_PRESETS
        .iter()
        .find(|preset| Some(preset.offset) == offset && Some(preset.gap) == gap)
        .map_or("Choose\u{2026}", |preset| preset.name);

    let mut chosen = None;
    egui::ComboBox::from_label("Stagger Preset")
        .selected_text(current)
        .show_ui(ui, |ui| {
            for preset in mc_nodes::tile_generator::LAYOUT_PRESETS {
                if ui.selectable_label(false, preset.name).clicked() {
                    chosen = Some(preset);
                }
            }
        });
    if let Some(preset) = chosen {
        commit_once_multi(
            graph,
            history,
            id,
            &[
                ("offset", mc_core::ParamValue::Float(preset.offset)),
                ("gap", mc_core::ParamValue::Float(preset.gap)),
            ],
        );
    }
}

/// Knit's own version of [`thread_preset_picker`], one load-bearing
/// difference: `preset` is a real, ongoing field `mc_nodes::knit::resolve`
/// reads every frame (it picks the chart and its stagger/thickness/
/// loop-height modifiers), not a one-shot template with no lasting
/// identity the way Thread's presets are — so this *does* show which one
/// is currently selected (`selected_text`), and renders in place of the
/// generic Enum widget the per-param loop would otherwise give "Preset"
/// (skipped there specifically for Knit — see the `continue` in the main
/// loop below). Picking one resets Wales to that preset's own
/// `canonical_wales()` and every other construction slider back to
/// `KnitParams::default`'s own values (plus clearing Chart, so a leftover
/// custom chart can't silently keep overriding the preset just picked) —
/// regardless of whatever those sliders previously held, including a
/// prior "randomize everything" roll. Side and Seed are left alone, same
/// reasoning as Thread's own table: Side is orthogonal to "which chart",
/// Seed is the artist's own choice, not part of "what kind of fabric is
/// this."
fn knit_preset_picker(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    node_kind: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "knit" {
        return;
    }
    let current = match current_params.get("preset") {
        Some(mc_core::ParamValue::Enum(s)) => s.clone(),
        _ => "Jersey".to_string(),
    };
    let mut chosen = None;
    egui::ComboBox::from_label("Preset")
        .selected_text(current)
        .show_ui(ui, |ui| {
            for name in mc_nodes::knit::KnitPreset::NAMES {
                if ui.selectable_label(false, name).clicked() {
                    chosen = Some(name);
                }
            }
        });
    if let Some(name) = chosen {
        let preset = mc_nodes::knit::KnitPreset::from_name(name)
            .expect("KnitPreset::NAMES/from_name always agree");
        let defaults = mc_nodes::knit::KnitParams::default();
        commit_once_multi(
            graph,
            history,
            id,
            &[
                ("preset", mc_core::ParamValue::Enum(name.to_string())),
                ("chart", mc_core::ParamValue::Text(String::new())),
                (
                    "wales",
                    mc_core::ParamValue::Int(preset.canonical_wales() as i32),
                ),
                ("auto_courses", mc_core::ParamValue::Bool(true)),
                (
                    "roundness",
                    mc_core::ParamValue::Float(defaults.roundness as f32),
                ),
                ("depth", mc_core::ParamValue::Float(defaults.depth as f32)),
                (
                    "tension",
                    mc_core::ParamValue::Enum(defaults.tension.name().to_string()),
                ),
                ("thickness", mc_core::ParamValue::Float(0.0)),
                ("loop_height", mc_core::ParamValue::Float(0.0)),
                ("plies", mc_core::ParamValue::Int(defaults.plies as i32)),
                (
                    "ply_direction",
                    mc_core::ParamValue::Enum(defaults.ply_direction.name().to_string()),
                ),
                (
                    "twist_angle",
                    mc_core::ParamValue::Float(defaults.twist_angle as f32),
                ),
                (
                    "rib_depth",
                    mc_core::ParamValue::Float(defaults.rib_depth as f32),
                ),
                (
                    "rib_compaction",
                    mc_core::ParamValue::Float(defaults.rib_compaction as f32),
                ),
            ],
        );
    }
}

/// Stitch's "Look" param (ParamSpec::defs.rs's own tooltip: "Picking a Look
/// resets Thread Diameter, Plies, Twist Angle, Regularity, Stem Width and
/// Density to its own defaults — each stays independently editable
/// afterward") — same "preset resets these sliders, they stay independently
/// editable after" pattern as `knit_preset_picker` above, just triggered by
/// the node's own real "look" param rather than a separate "preset" one.
fn stitch_look_picker(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    node_kind: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "stitch" {
        return;
    }
    let current = match current_params.get("look") {
        Some(mc_core::ParamValue::Enum(s)) => s.clone(),
        _ => "hand".to_string(),
    };
    let mut chosen = None;
    egui::ComboBox::from_label("Look")
        .selected_text(&current)
        .show_ui(ui, |ui| {
            for name in ["hand", "machine"] {
                if ui.selectable_label(current == name, name).clicked() {
                    chosen = Some(name);
                }
            }
        });
    if let Some(name) = chosen {
        let look = match name {
            "machine" => mc_nodes::stitch::Look::Machine,
            _ => mc_nodes::stitch::Look::Hand,
        };
        let defaults = mc_nodes::stitch::look_defaults(look);
        commit_once_multi(
            graph,
            history,
            id,
            &[
                ("look", mc_core::ParamValue::Enum(name.to_string())),
                (
                    "thread_mm",
                    mc_core::ParamValue::Float(defaults.thread_mm as f32),
                ),
                ("plies", mc_core::ParamValue::Int(defaults.plies as i32)),
                (
                    "twist_angle",
                    mc_core::ParamValue::Float(defaults.twist_angle_deg as f32),
                ),
                (
                    "regularity",
                    mc_core::ParamValue::Float(defaults.regularity as f32),
                ),
                (
                    "stem_width_mm",
                    mc_core::ParamValue::Float(defaults.thread_mm as f32),
                ),
                (
                    "density_mm",
                    mc_core::ParamValue::Float(
                        (defaults.thread_mm * defaults.density_factor) as f32,
                    ),
                ),
            ],
        );
    }
}

/// French Knot's own "Knot Look" picker — same "preset resets these
/// sliders, they stay independently editable after" pattern as
/// `stitch_look_picker` just above, for the exact same reason
/// (`knot_look` is its own param, separate from Stitch's own `look`,
/// precisely because it needs this kind of multi-slider reset with
/// different values — see `mc_nodes::defs`'s own Stitch `NodeDef` doc
/// comment on `knot_look`).
fn knot_look_picker(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    node_kind: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    let stitch_kind = match current_params.get("stitch") {
        Some(mc_core::ParamValue::Enum(s)) => s.as_str(),
        _ => "running",
    };
    if node_kind != "stitch" || stitch_kind != "french_knot" {
        return;
    }
    let current = match current_params.get("knot_look") {
        Some(mc_core::ParamValue::Enum(s)) => s.clone(),
        _ => "hand".to_string(),
    };
    let mut chosen = None;
    egui::ComboBox::from_label("Knot Look")
        .selected_text(&current)
        .show_ui(ui, |ui| {
            for name in ["hand", "neat", "loose"] {
                if ui.selectable_label(current == name, name).clicked() {
                    chosen = Some(name);
                }
            }
        });
    if let Some(name) = chosen {
        let look = match name {
            "neat" => mc_nodes::french_knot::KnotLook::Neat,
            "loose" => mc_nodes::french_knot::KnotLook::Loose,
            _ => mc_nodes::french_knot::KnotLook::Hand,
        };
        let defaults = mc_nodes::french_knot::look_defaults(look);
        commit_once_multi(
            graph,
            history,
            id,
            &[
                ("knot_look", mc_core::ParamValue::Enum(name.to_string())),
                (
                    "knot_twist_angle",
                    mc_core::ParamValue::Float(defaults.twist_angle_deg as f32),
                ),
                (
                    "regularity",
                    mc_core::ParamValue::Float(defaults.regularity as f32),
                ),
            ],
        );
    }
}

/// KNIT_NODE_SPEC.md §8 item 6: "snapping shown live in the panel (for
/// example '12 wales x 13 courses')" — Wales/Courses both get rounded up to
/// a multiple of the chart's own width/height (`mc_nodes::knit::resolve`),
/// so what a slider shows isn't necessarily the number actually used.
/// Recomputes `resolve()` from the current param snapshot every frame
/// (cheap — plain CPU math, no GPU dispatch) rather than caching, so it's
/// never stale by even one frame behind whatever's currently on the
/// sliders.
fn knit_resolved_hint(
    ui: &mut egui::Ui,
    node_kind: &str,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) {
    if node_kind != "knit" {
        return;
    }
    let params = mc_nodes::knit::KnitParams::from_graph(current_params);
    let g = mc_nodes::knit::resolve(&params);
    ui.weak(format!(
        "Resolved: {} wales \u{d7} {} courses",
        g.wales, g.courses
    ));
}

/// The Seed dice button, upgraded by [`randomize_all_params`] into a
/// "randomize everything on this node" action — its own section at the
/// very top of the panel, above any preset picker. Owner's own call: a
/// whole-node action like this reads as a different kind of thing than a
/// preset (which only resets a handful of *dependent* sliders), so the two
/// shouldn't visually blend into one group. Previously rendered wherever
/// "seed" fell in the node's own declared param order — the bottom, since
/// every node declares it last — which buried a whole-node action under
/// everything it was about to reset.
fn seed_randomizer_section(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    dragging: &mut DragState,
    id: mc_core::NodeId,
    def: &'static mc_nodes::defs::NodeDef,
    current_params: &std::collections::BTreeMap<String, mc_core::ParamValue>,
) -> bool {
    let Some(param) = def
        .params
        .iter()
        .find(|p| matches!(p.kind, mc_nodes::defs::ParamSpecKind::Seed { .. }))
    else {
        return false;
    };
    let mc_nodes::defs::ParamSpecKind::Seed { default } = &param.kind else {
        unreachable!("just matched Seed above")
    };
    let mut value = current_params
        .get(param.name)
        .and_then(mc_core::ParamValue::as_seed)
        .unwrap_or(*default);
    ui.horizontal(|ui| {
        let response = ui
            .add(egui::DragValue::new(&mut value).prefix(format!("{}: ", param.display_name)))
            .on_hover_text(param.tooltip);
        commit_live(
            ui,
            graph,
            history,
            dragging,
            id,
            param.name,
            &response,
            mc_core::ParamValue::Seed(value),
        );
        if ui
            .button("🎲")
            .on_hover_text("Randomizes every parameter on this node, not just Seed.")
            .clicked()
        {
            let new_seed = random_seed();
            let mut changes = vec![(param.name, mc_core::ParamValue::Seed(new_seed))];
            changes.extend(randomize_all_params(def, param.name, new_seed));
            commit_once_multi(graph, history, id, &changes);
        }
    });
    thread_seed_hint(ui, def.kind, param.name, current_params);
    ui.separator();
    true
}

#[expect(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    selected: Option<mc_core::NodeId>,
    selected_annotation: Option<crate::canvas::AnnotationSelection>,
    dragging: &mut DragState,
    path_editor: &mut crate::path_editor::PathEditorState,
    pending_svg_export: &mut Option<mc_core::NodeId>,
    pending_image_import: &mut Option<PendingImageImport>,
) {
    if let Some(pending) = pending_image_import {
        if let Some(result) = pending.dialog.poll(ui.ctx()) {
            if let Some(handle) = result {
                let path = handle.path();
                let path_string = path.display().to_string();
                let embedded_base64 = pending
                    .embed
                    .then(|| std::fs::read(path).ok())
                    .flatten()
                    .map(|bytes| {
                        use base64::Engine;
                        base64::engine::general_purpose::STANDARD.encode(bytes)
                    });
                commit_once(
                    graph,
                    history,
                    pending.node_id,
                    pending.param_name,
                    mc_core::ParamValue::Image(mc_core::ImageRef {
                        path: Some(path_string),
                        embedded_base64,
                    }),
                );
            }
            *pending_image_import = None;
        }
    }

    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            // A node with a long param list (Knit, Thread) can need more
            // vertical room than a squeezed/small window has to give —
            // scrollable so every control stays reachable instead of
            // clipping off the bottom with no way back to it (same
            // reasoning as the 3D View drawer's own ScrollArea).
            egui::ScrollArea::vertical()
                .id_salt("params_panel_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if let Some(annotation) = selected_annotation {
                        show_annotation(ui, graph, history, annotation);
                        return;
                    }
                    show_inner(
                        ui,
                        graph,
                        history,
                        selected,
                        dragging,
                        path_editor,
                        pending_svg_export,
                        pending_image_import,
                    );
                });
        });
}

/// A Comment's or Frame's own small, hand-written editor — these aren't
/// nodes (no `kind`/`params`/`NodeDef`), so unlike `show_inner` below they
/// can't get a panel "for free" from a node definition's declared params.
/// Comment: just its text. Frame: title and color (size/position are
/// dragged directly on the canvas instead, same as a node's own position).
fn show_annotation(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    annotation: crate::canvas::AnnotationSelection,
) {
    match annotation {
        crate::canvas::AnnotationSelection::Comment(id) => {
            let Some(comment) = graph.comment(id) else {
                ui.weak("(selected comment no longer exists)");
                return;
            };
            let mut text = comment.text.clone();
            ui.label("Comment");
            let response = ui.add(
                egui::TextEdit::multiline(&mut text)
                    .desired_width(f32::INFINITY)
                    .desired_rows(4),
            );
            if response.changed() {
                commit_comment_text(graph, history, id, text);
            }
        }
        crate::canvas::AnnotationSelection::Frame(id) => {
            let Some(frame) = graph.frame(id) else {
                ui.weak("(selected frame no longer exists)");
                return;
            };
            let mut title = frame.title.clone();
            let mut color = frame.color;
            ui.horizontal(|ui| {
                ui.label("Title");
                let response =
                    ui.add(egui::TextEdit::singleline(&mut title).desired_width(f32::INFINITY));
                if response.changed() {
                    commit_frame_title(graph, history, id, title.clone());
                }
            });
            ui.horizontal(|ui| {
                ui.label("Color");
                let response = ui.color_edit_button_rgba_unmultiplied(&mut color);
                if response.changed() {
                    commit_frame_color(graph, history, id, color);
                }
            });
        }
    }
}

fn commit_comment_text(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::CommentId,
    new_text: String,
) {
    let old = graph.comment(id).map(|c| c.text.clone());
    if graph.set_comment_text(id, new_text).is_ok() {
        if let Some(old_text) = old {
            history.record_applied(vec![mc_core::Command::SetCommentText {
                id,
                text: old_text,
            }]);
        }
    }
}

fn commit_frame_title(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::FrameId,
    new_title: String,
) {
    let old = graph.frame(id).map(|f| f.title.clone());
    if graph.set_frame_title(id, new_title).is_ok() {
        if let Some(old_title) = old {
            history.record_applied(vec![mc_core::Command::SetFrameTitle {
                id,
                title: old_title,
            }]);
        }
    }
}

fn commit_frame_color(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::FrameId,
    new_color: [f32; 4],
) {
    let old = graph.frame(id).map(|f| f.color);
    if graph.set_frame_color(id, new_color).is_ok() {
        if let Some(old_color) = old {
            history.record_applied(vec![mc_core::Command::SetFrameColor {
                id,
                color: old_color,
            }]);
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn show_inner(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    selected: Option<mc_core::NodeId>,
    dragging: &mut DragState,
    path_editor: &mut crate::path_editor::PathEditorState,
    pending_svg_export: &mut Option<mc_core::NodeId>,
    pending_image_import: &mut Option<PendingImageImport>,
) {
    let Some(id) = selected else {
        ui.weak("(select a node to edit its parameters)");
        return;
    };
    let Some(node) = graph.node(id) else {
        ui.weak("(selected node no longer exists)");
        return;
    };
    let Some(def) = mc_nodes::defs::node_def(&node.kind) else {
        ui.weak(format!("(unknown node kind {:?})", node.kind));
        return;
    };
    // Snapshot params (and kind) as owned data so the loop below is free to
    // mutate `graph` without fighting the borrow checker over `node`.
    let current_params = node.params.clone();
    let node_kind = node.kind.clone();

    ui.heading(def.display_name);
    ui.weak(def.tooltip);

    let has_seed_randomizer =
        seed_randomizer_section(ui, graph, history, dragging, id, def, &current_params);

    thread_preset_picker(ui, graph, history, id, &node_kind);
    tile_generator_stagger_preset_picker(ui, graph, history, id, &node_kind, &current_params);
    knit_preset_picker(ui, graph, history, id, &node_kind, &current_params);
    knit_resolved_hint(ui, &node_kind, &current_params);
    stitch_look_picker(ui, graph, history, id, &node_kind, &current_params);
    knot_look_picker(ui, graph, history, id, &node_kind, &current_params);

    // A connected-vs-fallback status line for inputs where that's genuinely
    // ambiguous from the rendered result alone (e.g. a connected Fiber using
    // default params that happen to match the built-in fallback's own
    // defaults looks identical either way) — see
    // `mc_nodes::defs::input_is_optional`'s doc comment.
    for port in def.inputs {
        if !mc_nodes::defs::input_is_optional(&node_kind, port.name) {
            continue;
        }
        // Plain ASCII, not a Unicode check mark/circle glyph — the proportional
        // font (Geist) doesn't cover those, which rendered as a tofu box
        // instead of a status symbol.
        let status = if graph.upstream(id, port.name).is_some() {
            format!("[x] {}: connected", port.display_name)
        } else {
            format!(
                "[ ] {}: not connected \u{2014} using the built-in default below",
                port.display_name
            )
        };
        ui.weak(status);
    }
    ui.separator();

    // Pattern Generator: once a custom Shape is wired in, Rows and Columns
    // collapse into one linked "Tile Count" slider instead of their normal
    // independent ones below — a custom shape stamped into an uneven grid
    // would read as visibly stretched, unlike the built-in brick/grid shape
    // (which *wants* independent Rows/Columns, e.g. the classic 6x4 brick
    // layout). Gap/Edge Darkening only ever affected the built-in shape, so
    // they're hidden too once Shape is connected, replaced by Shape Size
    // (which conversely has no effect on the built-in shape, so it's hidden
    // the rest of the time). Only this one node/case; not a generic
    // ParamSpecKind or `param_is_relevant` rule, since both of those work
    // off param *values*, not port connection state.
    let shape_connected = node_kind == "tile_generator" && graph.upstream(id, "shape").is_some();
    if shape_connected {
        let rows = current_params
            .get("rows")
            .and_then(mc_core::ParamValue::as_int)
            .unwrap_or(4);
        let columns = current_params
            .get("columns")
            .and_then(mc_core::ParamValue::as_int)
            .unwrap_or(4);
        let mut tile_count = rows.max(columns);
        let response = ui
            .add(egui::Slider::new(&mut tile_count, 1..=32).text("Tile Count"))
            .on_hover_text(
                "Rows and Columns together, locked to the same value while a custom Shape \
                 is wired in, so it isn't stretched across an uneven grid.",
            );
        if response.changed() {
            commit_once_multi(
                graph,
                history,
                id,
                &[
                    ("rows", mc_core::ParamValue::Int(tile_count)),
                    ("columns", mc_core::ParamValue::Int(tile_count)),
                ],
            );
        }
    }

    // Fabric Generator: "Thread Width"/"Thread Roundness" only ever affect
    // the built-in fallback fiber this node renders when nothing's wired
    // into Thread — once a Thread node *is* connected, dragging them did
    // nothing, which read exactly like "the connection isn't working" (a
    // real bug report — the sliders were right there, visibly draggable,
    // producing zero visible change). Stockinette is the one exception:
    // it never samples Thread at all (see fabric_generator.wgsl's own doc
    // comment), so Thread Width keeps governing its loop width regardless
    // of whether something's connected.
    let fabric_thread_connected =
        node_kind == "fabric_generator" && graph.upstream(id, "fiber").is_some();
    let fabric_weave_type = current_params
        .get("weave_type")
        .and_then(|v| match v {
            mc_core::ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        })
        .unwrap_or("Plain Weave");
    let fabric_is_stockinette = fabric_weave_type == "Stockinette";
    let hide_thread_width = fabric_thread_connected && !fabric_is_stockinette;

    for param in def.params {
        if shape_connected && matches!(param.name, "rows" | "columns" | "gap" | "bevel") {
            continue;
        }
        if !shape_connected && param.name == "shape_size" {
            continue;
        }
        if hide_thread_width && param.name == "thread_width" {
            continue;
        }
        // Rendered by knit_preset_picker above instead — a plain Enum
        // widget here would duplicate it without the multi-slider reset.
        if node_kind == "knit" && param.name == "preset" {
            continue;
        }
        // Rendered by stitch_look_picker/knot_look_picker above instead,
        // for the same reason.
        if node_kind == "stitch" && matches!(param.name, "look" | "knot_look") {
            continue;
        }
        // Image Trace's and Path Simplify's own "path" params are each
        // computed fresh every evaluation (`mc_nodes::image_trace`'s and
        // `mc_nodes::path_simplify`'s own `dispatch`/`simplify`), not
        // hand-edited — the generic `ParamSpecKind::Path` case below
        // (built for the Path node's own, genuinely user-editable "Wrap"
        // toggle) would let a click here get silently overwritten on the
        // very next evaluation, so both render their own read-only
        // summary instead. Image Trace also gets the "Export SVG…"
        // action TRACE_SPEC.md's own node UI calls for.
        if matches!(node_kind.as_str(), "image_trace" | "path_simplify") && param.name == "path" {
            let path_value = current_params
                .get("path")
                .and_then(mc_core::ParamValue::as_path)
                .cloned()
                .unwrap_or_default();
            let point_count: usize = path_value.subpaths.iter().map(|sp| sp.points.len()).sum();
            let source = if node_kind == "image_trace" {
                "traced from Image"
            } else {
                "simplified from Path"
            };
            ui.weak(format!(
                "{} subpath{}, {} point{} \u{2014} {source}",
                path_value.subpaths.len(),
                if path_value.subpaths.len() == 1 {
                    ""
                } else {
                    "s"
                },
                point_count,
                if point_count == 1 { "" } else { "s" },
            ))
            .on_hover_text(param.tooltip);
            if node_kind == "image_trace"
                && ui
                    .add_enabled(
                        !path_value.subpaths.is_empty(),
                        egui::Button::new("Export SVG\u{2026}"),
                    )
                    .clicked()
            {
                *pending_svg_export = Some(id);
            }
            continue;
        }
        // Rendered by seed_randomizer_section above instead, for the same
        // reason — moved to its own section at the top of the panel.
        if has_seed_randomizer && matches!(param.kind, mc_nodes::defs::ParamSpecKind::Seed { .. }) {
            continue;
        }
        if !mc_nodes::defs::param_is_relevant(&node_kind, param.name, &current_params) {
            continue;
        }
        let current = current_params.get(param.name).cloned();
        match &param.kind {
            mc_nodes::defs::ParamSpecKind::Float { min, max, default } => {
                let mut value = current.and_then(|v| v.as_float()).unwrap_or(*default);
                let response = ui
                    .add(egui::Slider::new(&mut value, *min..=*max).text(param.display_name))
                    .on_hover_text(param.tooltip);
                commit_live(
                    ui,
                    graph,
                    history,
                    dragging,
                    id,
                    param.name,
                    &response,
                    mc_core::ParamValue::Float(value),
                );
                thread_fiber_param_hint(ui, &node_kind, param.name);
                knit_ply_arrangement_hint(ui, &node_kind, param.name, &current_params);
            }
            mc_nodes::defs::ParamSpecKind::Int { min, max, default } => {
                let mut value = current.and_then(|v| v.as_int()).unwrap_or(*default);
                let response = ui
                    .add(egui::Slider::new(&mut value, *min..=*max).text(param.display_name))
                    .on_hover_text(param.tooltip);
                commit_live(
                    ui,
                    graph,
                    history,
                    dragging,
                    id,
                    param.name,
                    &response,
                    mc_core::ParamValue::Int(value),
                );
                thread_fiber_param_hint(ui, &node_kind, param.name);
                knit_ply_id_hint(ui, &node_kind, param.name, &current_params);
            }
            // No Seed arm here — every node with a Seed param gets
            // `has_seed_randomizer == true` above, so the skip earlier in
            // this loop always continues past it before reaching this
            // match; it's rendered once, at the top of the panel, by
            // `seed_randomizer_section` instead.
            mc_nodes::defs::ParamSpecKind::Seed { .. } => {}
            mc_nodes::defs::ParamSpecKind::Enum { options, default } => {
                let value = match &current {
                    Some(mc_core::ParamValue::Enum(s)) => s.clone(),
                    _ => (*default).to_string(),
                };
                let mut chosen = None;
                egui::ComboBox::from_label(param.display_name)
                    .selected_text(value.clone())
                    .show_ui(ui, |ui| {
                        for option in *options {
                            if ui.selectable_label(value == *option, *option).clicked() {
                                chosen = Some((*option).to_string());
                            }
                        }
                    });
                if let Some(new_value) = chosen {
                    commit_once(
                        graph,
                        history,
                        id,
                        param.name,
                        mc_core::ParamValue::Enum(new_value),
                    );
                }
                thread_fiber_param_hint(ui, &node_kind, param.name);
                knit_stitch_type_hint(ui, &node_kind, param.name, &current_params);
                knit_ply_arrangement_hint(ui, &node_kind, param.name, &current_params);
            }
            mc_nodes::defs::ParamSpecKind::Color { default } => {
                let mut value = match &current {
                    Some(mc_core::ParamValue::Color(c)) => *c,
                    _ => *default,
                };
                ui.horizontal(|ui| {
                    ui.label(param.display_name);
                    let response = ui.color_edit_button_rgba_unmultiplied(&mut value);
                    if response.changed() {
                        commit_once(
                            graph,
                            history,
                            id,
                            param.name,
                            mc_core::ParamValue::Color(value),
                        );
                    }
                });
            }
            mc_nodes::defs::ParamSpecKind::Bool { default } => {
                let mut value = current.and_then(|v| v.as_bool()).unwrap_or(*default);
                if ui
                    .checkbox(&mut value, param.display_name)
                    .on_hover_text(param.tooltip)
                    .changed()
                {
                    commit_once(
                        graph,
                        history,
                        id,
                        param.name,
                        mc_core::ParamValue::Bool(value),
                    );
                }
            }
            mc_nodes::defs::ParamSpecKind::Text { default } => {
                let mut value = current
                    .and_then(|v| v.as_text().map(str::to_string))
                    .unwrap_or_else(|| default.to_string());
                let mut changed = false;
                ui.horizontal(|ui| {
                    ui.label(param.display_name);
                    let response = ui
                        .add(
                            egui::TextEdit::singleline(&mut value)
                                .hint_text(param.tooltip)
                                .desired_width(f32::INFINITY),
                        )
                        .on_hover_text(param.tooltip);
                    changed = response.changed();
                });
                if changed {
                    if node_kind == "knit" && param.name == "chart" {
                        // Spec: "editing switches the preset to Custom" —
                        // committed together as one undo step, same as
                        // Pattern Generator's linked Tile Count above.
                        commit_once_multi(
                            graph,
                            history,
                            id,
                            &[
                                ("chart", mc_core::ParamValue::Text(value)),
                                (
                                    "preset",
                                    mc_core::ParamValue::Enum(
                                        mc_nodes::knit::KnitPreset::Custom.name().to_string(),
                                    ),
                                ),
                            ],
                        );
                    } else {
                        commit_once(
                            graph,
                            history,
                            id,
                            param.name,
                            mc_core::ParamValue::Text(value),
                        );
                    }
                }
            }
            mc_nodes::defs::ParamSpecKind::FontFamily => {
                let value = current
                    .and_then(|v| v.as_text().map(str::to_string))
                    .unwrap_or_default();
                let families = installed_font_families();
                // Installed-font lists routinely run into the hundreds —
                // a plain scrolling list made finding one slow, so this
                // keeps a small free-typed filter alongside it. The filter
                // text itself lives in egui's own per-widget memory (keyed
                // by this exact param) rather than a field threaded through
                // every caller, since it's pure transient UI state no other
                // code needs to see.
                let search_id = ui.id().with((id, param.name, "font_search"));
                let mut chosen = None;
                egui::ComboBox::from_label(param.display_name)
                    .selected_text(if value.is_empty() {
                        "(System Default)"
                    } else {
                        value.as_str()
                    })
                    .show_ui(ui, |ui| {
                        let mut search = ui
                            .memory_mut(|mem| mem.data.get_temp::<String>(search_id))
                            .unwrap_or_default();
                        ui.add(
                            egui::TextEdit::singleline(&mut search)
                                .hint_text("Search fonts…")
                                .desired_width(f32::INFINITY),
                        )
                        .request_focus();
                        ui.memory_mut(|mem| mem.data.insert_temp(search_id, search.clone()));
                        ui.separator();
                        let query = search.to_lowercase();
                        egui::ScrollArea::vertical()
                            .max_height(240.0)
                            .show(ui, |ui| {
                                if query.is_empty()
                                    && ui
                                        .selectable_label(value.is_empty(), "(System Default)")
                                        .clicked()
                                {
                                    chosen = Some(String::new());
                                }
                                for family in families.iter().filter(|f| {
                                    query.is_empty() || f.to_lowercase().contains(&query)
                                }) {
                                    if ui.selectable_label(*family == value, family).clicked() {
                                        chosen = Some(family.clone());
                                    }
                                }
                            });
                    })
                    .response
                    .on_hover_text(param.tooltip);
                if let Some(new_value) = chosen {
                    commit_once(
                        graph,
                        history,
                        id,
                        param.name,
                        mc_core::ParamValue::Text(new_value),
                    );
                }
            }
            mc_nodes::defs::ParamSpecKind::Vec2 { min, max, default } => {
                let mut value = current.and_then(|v| v.as_vec2()).unwrap_or(*default);
                ui.horizontal(|ui| {
                    ui.label(param.display_name);
                    let x_response = ui
                        .add(egui::Slider::new(&mut value[0], *min..=*max).text("X"))
                        .on_hover_text(param.tooltip);
                    let y_response = ui
                        .add(egui::Slider::new(&mut value[1], *min..=*max).text("Y"))
                        .on_hover_text(param.tooltip);
                    // One slider's `changed()` already reflects both axes
                    // together, since `value` is read fresh each frame —
                    // committing on either response alone (not a merged
                    // one) just means only the axis actually dragged
                    // triggers a write.
                    commit_live(
                        ui,
                        graph,
                        history,
                        dragging,
                        id,
                        param.name,
                        &x_response,
                        mc_core::ParamValue::Vec2(value),
                    );
                    commit_live(
                        ui,
                        graph,
                        history,
                        dragging,
                        id,
                        param.name,
                        &y_response,
                        mc_core::ParamValue::Vec2(value),
                    );
                });
            }
            mc_nodes::defs::ParamSpecKind::Gradient { default } => {
                let mut stops = match &current {
                    Some(mc_core::ParamValue::Gradient(s)) if !s.is_empty() => s.clone(),
                    _ => default.to_vec(),
                };
                ui.label(param.display_name);
                let stop_count = stops.len();
                let mut changed = false;
                let mut remove_index = None;
                for (i, stop) in stops.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::Slider::new(&mut stop.position, 0.0..=1.0).text("pos"))
                            .changed()
                        {
                            changed = true;
                        }
                        if ui
                            .color_edit_button_rgba_unmultiplied(&mut stop.color)
                            .changed()
                        {
                            changed = true;
                        }
                        if stop_count > 2 && ui.small_button("✕").clicked() {
                            remove_index = Some(i);
                        }
                    });
                }
                if let Some(i) = remove_index {
                    stops.remove(i);
                    changed = true;
                }
                if ui.small_button("+ Add Stop").clicked() {
                    stops.push(mc_core::GradientStop {
                        position: 0.5,
                        color: [0.5, 0.5, 0.5, 1.0],
                    });
                    changed = true;
                }
                if changed {
                    commit_once(
                        graph,
                        history,
                        id,
                        param.name,
                        mc_core::ParamValue::Gradient(stops),
                    );
                }
            }
            mc_nodes::defs::ParamSpecKind::ImagePath => {
                let image_ref = current
                    .as_ref()
                    .and_then(mc_core::ParamValue::as_image)
                    .cloned()
                    .unwrap_or_default();
                // Whether the *next* load embeds the file's bytes into the
                // graph or just remembers its path — read from the sibling
                // "embed_in_graph" Bool param (its own checkbox renders via
                // the ordinary Bool case above, same loop).
                let embed_next_load = current_params
                    .get("embed_in_graph")
                    .and_then(mc_core::ParamValue::as_bool)
                    .unwrap_or(false);

                ui.horizontal(|ui| {
                    ui.label(param.display_name);
                    if ui.button("Load Image\u{2026}").clicked() {
                        *pending_image_import = Some(PendingImageImport {
                            node_id: id,
                            param_name: param.name,
                            embed: embed_next_load,
                            dialog: crate::async_dialog::PendingDialog::new(
                                rfd::AsyncFileDialog::new()
                                    .add_filter(
                                        "Image",
                                        &["png", "jpg", "jpeg", "bmp", "tga", "gif", "webp"],
                                    )
                                    .pick_file(),
                            ),
                        });
                    }
                });
                let status = match (&image_ref.path, image_ref.embedded_base64.is_some()) {
                    (Some(path), true) => format!(
                        "Embedded: {}",
                        std::path::Path::new(path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(path)
                    ),
                    (Some(path), false) => format!(
                        "Linked: {}",
                        std::path::Path::new(path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(path)
                    ),
                    (None, _) => "(none loaded)".to_string(),
                };
                ui.weak(status).on_hover_text(param.tooltip);
            }
            mc_nodes::defs::ParamSpecKind::Path => {
                let path_value = current
                    .as_ref()
                    .and_then(mc_core::ParamValue::as_path)
                    .cloned()
                    .unwrap_or_default();
                let point_count: usize = path_value.subpaths.iter().map(|sp| sp.points.len()).sum();
                ui.weak(format!(
                    "{} subpath{}, {} point{} — edit in the 2D View",
                    path_value.subpaths.len(),
                    if path_value.subpaths.len() == 1 {
                        ""
                    } else {
                        "s"
                    },
                    point_count,
                    if point_count == 1 { "" } else { "s" },
                ))
                .on_hover_text(param.tooltip);

                let mut wrap = path_value.wrap;
                if ui
                    .checkbox(&mut wrap, "Wrap")
                    .on_hover_text(
                        "Points may be dragged past the tile edge and reappear on the \
                         other side, so a border can tile seamlessly.",
                    )
                    .changed()
                {
                    commit_once(
                        graph,
                        history,
                        id,
                        param.name,
                        mc_core::ParamValue::Path(mc_core::Path {
                            wrap,
                            ..path_value.clone()
                        }),
                    );
                }

                path_point_list(ui, graph, history, dragging, id, path_editor, &path_value);
            }
        }
    }
}

/// Every point, grouped by subpath, with its own X/Y `DragValue` fields —
/// added at the owner's request so a point can be nudged to an exact
/// coordinate here instead of only by eye in the 2D View. Clicking a
/// point's own label selects it there too, the same link clicking a point
/// in the 2D View already has back to its own fields appearing here.
#[allow(clippy::too_many_arguments)]
fn path_point_list(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    dragging: &mut DragState,
    id: mc_core::NodeId,
    path_editor: &mut crate::path_editor::PathEditorState,
    path_value: &mc_core::Path,
) {
    if path_value.subpaths.is_empty() {
        return;
    }
    ui.add_space(6.0);
    ui.separator();
    for (sp_i, sp) in path_value.subpaths.iter().enumerate() {
        // Collapsed by default: a hand-drawn curve might only have a
        // handful of points, but an Image Trace result can have hundreds
        // of subpaths with thousands of points between them — rendering
        // every one of those rows unconditionally made this list the
        // dominant cost in the whole panel, and an unusable wall of
        // fields to begin with. `CollapsingHeader`'s own body closure
        // only runs while open, so a closed subpath costs nothing here
        // beyond its own header row.
        egui::CollapsingHeader::new(format!(
            "Subpath {} ({}, {} point{})",
            sp_i + 1,
            if sp.closed { "closed" } else { "open" },
            sp.points.len(),
            if sp.points.len() == 1 { "" } else { "s" },
        ))
        .id_salt(("path_point_list_subpath", id, sp_i))
        .default_open(false)
        .show(ui, |ui| {
            for (pt_i, pt) in sp.points.iter().enumerate() {
                ui.horizontal(|ui| {
                    let is_selected = path_editor.is_selected(sp_i, pt_i);
                    if ui
                        .selectable_label(is_selected, format!("Point {}", pt_i + 1))
                        .clicked()
                    {
                        path_editor.select_only(sp_i, pt_i);
                    }

                    let mut new_pos = pt.pos;
                    let x_response = ui.add(
                        egui::DragValue::new(&mut new_pos[0])
                            .speed(0.001)
                            .prefix("X: "),
                    );
                    let y_response = ui.add(
                        egui::DragValue::new(&mut new_pos[1])
                            .speed(0.001)
                            .prefix("Y: "),
                    );

                    // Called every frame regardless of whether either
                    // value actually changed — same as the Vec2 case
                    // above, `commit_live` itself is the thing deciding
                    // whether this frame is a drag-start/live-apply/
                    // drag-stop no-op, same reasoning as why that case
                    // doesn't gate on `changed()` either: gating here on
                    // our own would risk skipping the exact frame
                    // `commit_live` needs to see to open or close its
                    // undo snapshot.
                    let mut new_path = path_value.clone();
                    new_path.subpaths[sp_i].points[pt_i].pos = new_pos;
                    commit_live(
                        ui,
                        graph,
                        history,
                        dragging,
                        id,
                        "path",
                        &x_response,
                        mc_core::ParamValue::Path(new_path.clone()),
                    );
                    commit_live(
                        ui,
                        graph,
                        history,
                        dragging,
                        id,
                        "path",
                        &y_response,
                        mc_core::ParamValue::Path(new_path),
                    );

                    let trash = ui
                        .add(egui::Button::new(
                            egui::RichText::new('\u{e872}'.to_string()) // "delete"
                                .family(egui::FontFamily::Name("material_icons_round".into())),
                        ))
                        .on_hover_text("Delete this point.");
                    if trash.clicked() {
                        path_editor.delete_point(graph, history, id, sp_i, pt_i);
                    }
                });
            }
        });
    }
}

/// Applies a value live every frame (so the preview tracks the drag), and
/// records a single undo step spanning the whole drag gesture once it ends.
#[allow(clippy::too_many_arguments)]
fn commit_live(
    _ui: &egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    dragging: &mut DragState,
    id: mc_core::NodeId,
    name: &str,
    response: &egui::Response,
    new_value: mc_core::ParamValue,
) {
    if response.drag_started() {
        if let Some(old) = graph.node(id).and_then(|n| n.params.get(name).cloned()) {
            *dragging = Some((id, name.to_string(), old));
        }
    }
    if response.changed() {
        let _ = graph.set_param(id, name, new_value);
    }
    if response.drag_stopped() {
        if let Some((drag_id, drag_name, old_value)) = dragging.take() {
            if drag_id == id && drag_name == name {
                history.record_applied(vec![mc_core::Command::SetParam {
                    id,
                    name: name.to_string(),
                    value: old_value,
                }]);
            }
        }
    }
}

/// For instantaneous edits (a combo box choice, a dice-button click) that
/// aren't a drag gesture — each one is its own undo step.
fn commit_once(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    name: &str,
    new_value: mc_core::ParamValue,
) {
    let old = graph.node(id).and_then(|n| n.params.get(name).cloned());
    if graph.set_param(id, name, new_value).is_ok() {
        if let Some(old_value) = old {
            history.record_applied(vec![mc_core::Command::SetParam {
                id,
                name: name.to_string(),
                value: old_value,
            }]);
        }
    }
}

/// Like `commit_once`, but for several params that should feel like one
/// user action (e.g. Pattern Generator's linked Tile Count setting both
/// Rows and Columns) — one undo step for all of them together, rather than
/// needing Undo pressed once per param to fully revert the edit.
fn commit_once_multi(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    id: mc_core::NodeId,
    new_values: &[(&str, mc_core::ParamValue)],
) {
    let mut inverse = Vec::new();
    for (name, new_value) in new_values {
        let old = graph.node(id).and_then(|n| n.params.get(*name).cloned());
        if graph.set_param(id, name, new_value.clone()).is_ok() {
            if let Some(old_value) = old {
                inverse.push(mc_core::Command::SetParam {
                    id,
                    name: name.to_string(),
                    value: old_value,
                });
            }
        }
    }
    if !inverse.is_empty() {
        history.record_applied(inverse);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knit_def() -> &'static mc_nodes::defs::NodeDef {
        mc_nodes::defs::node_def("knit").expect("knit is a registered node kind")
    }

    fn thread_def() -> &'static mc_nodes::defs::NodeDef {
        mc_nodes::defs::node_def("thread_shape").expect("thread_shape is a registered node kind")
    }

    #[test]
    fn randomize_all_params_never_touches_seed_itself() {
        let def = knit_def();
        let out = randomize_all_params(def, "seed", 42);
        assert!(
            out.iter().all(|(name, _)| *name != "seed"),
            "the caller sets the new seed value itself; randomize_all_params shouldn't \
             duplicate it"
        );
    }

    #[test]
    fn randomize_all_params_stays_within_every_declared_range() {
        for def in [knit_def(), thread_def()] {
            for seed in [0u32, 1, 42, u32::MAX, 123456] {
                let out = randomize_all_params(def, "seed", seed);
                for (name, value) in &out {
                    let spec = def
                        .params
                        .iter()
                        .find(|p| p.name == *name)
                        .unwrap_or_else(|| panic!("{name} not declared on {}", def.kind));
                    match (&spec.kind, value) {
                        (
                            mc_nodes::defs::ParamSpecKind::Float { min, max, .. },
                            mc_core::ParamValue::Float(v),
                        ) => {
                            assert!(
                                v >= min && v <= max,
                                "{}.{name}: {v} outside [{min}, {max}]",
                                def.kind
                            );
                        }
                        (
                            mc_nodes::defs::ParamSpecKind::Int { min, max, .. },
                            mc_core::ParamValue::Int(v),
                        ) => {
                            assert!(
                                v >= min && v <= max,
                                "{}.{name}: {v} outside [{min}, {max}]",
                                def.kind
                            );
                        }
                        (
                            mc_nodes::defs::ParamSpecKind::Enum { options, .. },
                            mc_core::ParamValue::Enum(v),
                        ) => {
                            assert!(
                                options.contains(&v.as_str()),
                                "{}.{name}: {v:?} not one of {options:?}",
                                def.kind
                            );
                        }
                        (
                            mc_nodes::defs::ParamSpecKind::Vec2 { min, max, .. },
                            mc_core::ParamValue::Vec2(v),
                        ) => {
                            assert!(
                                v.iter().all(|c| c >= min && c <= max),
                                "{}.{name}: {v:?} outside [{min}, {max}]",
                                def.kind
                            );
                        }
                        (
                            mc_nodes::defs::ParamSpecKind::Bool { .. },
                            mc_core::ParamValue::Bool(_),
                        )
                        | (
                            mc_nodes::defs::ParamSpecKind::Seed { .. },
                            mc_core::ParamValue::Seed(_),
                        )
                        | (
                            mc_nodes::defs::ParamSpecKind::Color { .. },
                            mc_core::ParamValue::Color(_),
                        )
                        | (
                            mc_nodes::defs::ParamSpecKind::Text { .. },
                            mc_core::ParamValue::Text(_),
                        ) => {}
                        (_, value) => {
                            panic!(
                                "{}.{name}: unexpected value {value:?} for its declared kind",
                                def.kind
                            )
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn randomize_all_params_resets_text_to_its_declared_default() {
        let def = knit_def();
        let out = randomize_all_params(def, "seed", 7);
        let chart = out
            .iter()
            .find(|(name, _)| *name == "chart")
            .map(|(_, v)| v.clone());
        assert_eq!(chart, Some(mc_core::ParamValue::Text(String::new())));
    }

    #[test]
    fn randomize_all_params_is_deterministic_for_the_same_seed() {
        let def = knit_def();
        let a = randomize_all_params(def, "seed", 99);
        let b = randomize_all_params(def, "seed", 99);
        assert_eq!(a, b);
    }

    #[test]
    fn randomize_all_params_varies_with_the_seed() {
        let def = knit_def();
        let a = randomize_all_params(def, "seed", 1);
        let b = randomize_all_params(def, "seed", 2);
        assert_ne!(a, b);
    }

    /// Reproduces "randomize, then apply a named preset" at the data layer
    /// (no `egui::Ui` needed — `thread_preset_picker`'s own commit call is
    /// exactly `commit_once_multi` with a preset's field values, so this
    /// exercises the same path it would on a real click) to check whether
    /// Thread's preset system genuinely fails to overwrite a prior
    /// randomized value, or whether it already works at this layer and
    /// the reported issue is specific to live UI interaction.
    #[test]
    fn applying_a_thread_preset_after_randomizing_lands_on_its_exact_values() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let id = graph.add_node(thread_def().instantiate());

        // Simulate a prior "randomize everything" click.
        let randomized = randomize_all_params(thread_def(), "seed", 7);
        commit_once_multi(&mut graph, &mut history, id, &randomized);

        // Now simulate clicking the "4-Ply Cable" preset.
        let preset = mc_nodes::thread_shape::PRESETS
            .iter()
            .find(|p| p.name == "4-Ply Cable")
            .expect("4-Ply Cable preset exists");
        commit_once_multi(
            &mut graph,
            &mut history,
            id,
            &[
                ("threads", mc_core::ParamValue::Int(preset.threads as i32)),
                ("thickness", mc_core::ParamValue::Float(preset.thickness)),
                ("plies", mc_core::ParamValue::Int(preset.plies as i32)),
                (
                    "ply_direction",
                    mc_core::ParamValue::Enum(preset.ply_direction.name().to_string()),
                ),
                (
                    "twist_angle",
                    mc_core::ParamValue::Float(preset.twist_angle),
                ),
                (
                    "fiber_mode",
                    mc_core::ParamValue::Enum(preset.fiber_mode.name().to_string()),
                ),
                (
                    "fibers_per_ply",
                    mc_core::ParamValue::Int(preset.fibers_per_ply as i32),
                ),
                (
                    "fiber_depth",
                    mc_core::ParamValue::Float(preset.fiber_depth),
                ),
                (
                    "cable_groups",
                    mc_core::ParamValue::Int(preset.cable_groups as i32),
                ),
                (
                    "cable_angle",
                    mc_core::ParamValue::Float(preset.cable_angle),
                ),
            ],
        );

        let node = graph.node(id).expect("node still exists");
        assert_eq!(
            node.params.get("threads"),
            Some(&mc_core::ParamValue::Int(preset.threads as i32))
        );
        assert_eq!(
            node.params.get("plies"),
            Some(&mc_core::ParamValue::Int(preset.plies as i32))
        );
        assert_eq!(
            node.params.get("cable_groups"),
            Some(&mc_core::ParamValue::Int(preset.cable_groups as i32))
        );
        assert_eq!(
            node.params.get("fiber_mode"),
            Some(&mc_core::ParamValue::Enum(
                preset.fiber_mode.name().to_string()
            ))
        );
    }

    /// Same shape as the Thread test above, for Knit's own preset system —
    /// picking "2x2 Rib" after randomizing everything (including a
    /// leftover custom Chart string) should land exactly on that preset's
    /// canonical numbers, not some mix of random and reset values.
    #[test]
    fn applying_a_knit_preset_after_randomizing_lands_on_its_canonical_values() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let id = graph.add_node(knit_def().instantiate());

        let randomized = randomize_all_params(knit_def(), "seed", 13);
        commit_once_multi(&mut graph, &mut history, id, &randomized);
        // A leftover custom chart specifically, since that's the case the
        // picker's own Chart-clearing exists to guard against.
        commit_once(
            &mut graph,
            &mut history,
            id,
            "chart",
            mc_core::ParamValue::Text("KP/PK".to_string()),
        );

        let preset = mc_nodes::knit::KnitPreset::Rib2x2;
        let defaults = mc_nodes::knit::KnitParams::default();
        commit_once_multi(
            &mut graph,
            &mut history,
            id,
            &[
                (
                    "preset",
                    mc_core::ParamValue::Enum(preset.name().to_string()),
                ),
                ("chart", mc_core::ParamValue::Text(String::new())),
                (
                    "wales",
                    mc_core::ParamValue::Int(preset.canonical_wales() as i32),
                ),
                ("auto_courses", mc_core::ParamValue::Bool(true)),
                (
                    "roundness",
                    mc_core::ParamValue::Float(defaults.roundness as f32),
                ),
                ("depth", mc_core::ParamValue::Float(defaults.depth as f32)),
                (
                    "tension",
                    mc_core::ParamValue::Enum(defaults.tension.name().to_string()),
                ),
                ("thickness", mc_core::ParamValue::Float(0.0)),
                ("loop_height", mc_core::ParamValue::Float(0.0)),
                ("plies", mc_core::ParamValue::Int(defaults.plies as i32)),
                (
                    "ply_direction",
                    mc_core::ParamValue::Enum(defaults.ply_direction.name().to_string()),
                ),
                (
                    "twist_angle",
                    mc_core::ParamValue::Float(defaults.twist_angle as f32),
                ),
                (
                    "rib_depth",
                    mc_core::ParamValue::Float(defaults.rib_depth as f32),
                ),
                (
                    "rib_compaction",
                    mc_core::ParamValue::Float(defaults.rib_compaction as f32),
                ),
            ],
        );

        let node = graph.node(id).expect("node still exists");
        assert_eq!(
            node.params.get("preset"),
            Some(&mc_core::ParamValue::Enum("2x2 Rib".to_string()))
        );
        assert_eq!(
            node.params.get("chart"),
            Some(&mc_core::ParamValue::Text(String::new())),
            "a leftover custom chart should be cleared, or it would silently keep \
             overriding the preset just picked"
        );
        assert_eq!(
            node.params.get("wales"),
            Some(&mc_core::ParamValue::Int(12))
        );
        assert_eq!(
            node.params.get("auto_courses"),
            Some(&mc_core::ParamValue::Bool(true))
        );
        assert_eq!(
            node.params.get("rib_depth"),
            Some(&mc_core::ParamValue::Float(1.5))
        );
    }
}

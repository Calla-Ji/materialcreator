// SPDX-License-Identifier: MIT OR Apache-2.0
//! A fully custom node-graph canvas: pan, zoom, node dragging, click/shift/
//! box selection, and drag-to-connect wires — built directly on egui
//! primitives instead of egui-snarl.
//!
//! `mc_core::Graph` stays the only source of truth for node kind/params/
//! connections, exactly as before; this module owns purely-visual state
//! (camera pan/zoom, multi-selection, in-progress drag/wire gestures) and
//! edits `Graph` through the same `Command`/`Transaction` API the rest of
//! the app already uses, so undo/redo keeps working with no extra plumbing.
//! Node position from plain dragging is intentionally *not* undo-tracked
//! (CLAUDE.md: "purely cosmetic, ignored by evaluation"), matching the old
//! behavior — [`auto_layout`] is the one deliberate exception, since it can
//! move every node in the graph in a single click; see its own doc comment.
//!
//! Built to replace egui-snarl after its selection model turned out to have
//! no public way to clear or replace its own internal (and visually
//! hardcoded) selection — a plain click elsewhere could never deselect, and
//! that limitation sat underneath both rendering *and* input handling with
//! no seam to patch around. Owning both here means every gesture (what
//! clears a selection, what a drag does, how box-select works) is ours to
//! define and change going forward.
//!
//! Also draws and handles Comments and Frames (`mc_core::Comment`/`Frame`)
//! — free-floating text notes and titled grouping boxes, neither one a
//! real node (no kind/params/ports/evaluation). Added via the two corner
//! buttons below Auto Layout/Snap to Grid; edited through the Parameters
//! panel (see `AnnotationSelection`) rather than inline on the canvas.

use std::collections::{HashMap, HashSet};

use egui::{Color32, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use mc_core::{CommentId, FrameId, NodeId, PortType};
use serde::{Deserialize, Serialize};

const NODE_WIDTH: f32 = 170.0;
const HEADER_HEIGHT: f32 = 24.0;
const PORT_ROW_HEIGHT: f32 = 20.0;
const BODY_PADDING: f32 = 6.0;
/// Fills the node's full width (minus the usual side margin) rather than
/// being a small fixed square with dead space around it — stays square, so
/// the node ends up a bit taller too.
const THUMBNAIL_SIZE: f32 = NODE_WIDTH - 2.0 * BODY_PADDING;
const PIN_RADIUS: f32 = 5.0;
const PIN_HIT_RADIUS: f32 = 10.0;
const WIRE_HIT_DISTANCE: f32 = 12.0;
const PREVIEW_DOT_RADIUS: f32 = 4.0;
const PREVIEW_DOT_MARGIN: f32 = 12.0;
const DOT_SPACING: f32 = 16.0;
const HEADER_TEXT_LEFT: f32 = 8.0;
/// Space reserved on the header's right side for the preview/duplicate/
/// delete dots, so the title text never runs underneath them.
const HEADER_TEXT_RIGHT_RESERVED: f32 = PREVIEW_DOT_MARGIN + DOT_SPACING * 2.0 + 6.0;
/// Extra right-side space the performance overlay's own cost badge (colored
/// dot + "N.Nms" text) needs, added to `HEADER_TEXT_RIGHT_RESERVED` only
/// while the overlay's on (`node_graph_toolbar`'s toggle) — the title just
/// truncates a bit sooner on a long node name while it's active, same
/// truncation `node_title_text` already does for a narrow zoomed-out node.
const PERF_BADGE_RESERVED: f32 = 52.0;
const PERF_BADGE_FONT_SIZE: f32 = 9.0;

/// Which pin a wire-drag gesture started from, or that a drawn wire ends at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinRef {
    Output { node: NodeId, index: usize },
    Input { node: NodeId, index: usize },
}

struct PinInfo {
    r: PinRef,
    pos: Pos2,
    port_type: PortType,
}

/// Which annotation (Comment or Frame — see their own doc comments in
/// `mc_core`) is currently shown in the Parameters panel, if any —
/// independent of `CanvasState::selected`/the app's own `Option<NodeId>`,
/// since an annotation never has a 2D/3D preview and isn't part of a
/// node's own multi-select/group-drag/Auto Layout machinery. Only one of
/// "a node is selected" / "an annotation is selected" is ever true at a
/// time — selecting one clears the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationSelection {
    Comment(CommentId),
    Frame(FrameId),
}

/// A frame's fields, copied out of `graph` up front for the duration of one
/// frame-pass loop iteration — see that loop's own comment for why.
struct FrameSnapshot {
    id: FrameId,
    title: String,
    color: [f32; 4],
    position: [f32; 2],
    size: [f32; 2],
}

/// Visual/interaction state for the node canvas: camera, multi-selection,
/// and any in-progress drag gesture. Everything structural lives in
/// `mc_core::Graph` instead.
pub struct CanvasState {
    pan: Vec2,
    zoom: f32,
    pub selected: HashSet<NodeId>,
    /// Which Comment or Frame (if either) the Parameters panel is currently
    /// showing — see `AnnotationSelection`'s own doc comment.
    pub selected_annotation: Option<AnnotationSelection>,
    /// Toggled by the Snap to Grid corner button — persistent, not a
    /// held-key modifier (that was the first version; the owner found it
    /// "not really amazing" and asked for a toggle instead).
    pub snap_to_grid: bool,
    /// While Snap to Grid is on and a node drag is in progress, each
    /// dragged node's true, unrounded running position — see the drag
    /// handler's own comment for why this exists (rounding the node's
    /// already-rounded *stored* position every frame made a slow drag
    /// stick permanently at whatever grid cell it first snapped to).
    /// Cleared per-node the instant its drag stops.
    drag_unsnapped: HashMap<NodeId, [f32; 2]>,
    wire_drag: Option<PinRef>,
    /// Screen-space drag-start position, while a Shift+drag box-select is in
    /// progress.
    box_select_start: Option<Pos2>,
    /// Graph-space position of the last left- or right-click directly on the
    /// canvas (background or a node, not a UI widget elsewhere) — where the
    /// Node Library panel drops a newly-added node, so it lands wherever the
    /// artist was last pointing rather than a fixed, easy-to-lose corner.
    last_click_pos: Pos2,
    /// How many times Paste has been pressed since the last Copy — each
    /// paste offsets the new nodes by this many multiples of
    /// `DUPLICATE_OFFSET`, so repeatedly pressing Cmd+V fans the copies out
    /// instead of stacking every one exactly on top of the last. Reset to 0
    /// by `copy_selection_to_clipboard`.
    paste_count: u32,
    /// `(texture key, is_sdf)` as of each node's own last redrawn thumbnail
    /// — skips re-running `display_encode::encode_for_display`'s compute
    /// pass for every visible node, every single frame, when nothing about
    /// that node changed (panning/dragging/hovering the canvas repaints
    /// every frame without dirtying anything) and its thumbnail would come
    /// out identical anyway. See `mc_app::MaterialCreatorApp::
    /// last_evaluated_nodes`'s own doc comment for the matching 2D View fix.
    thumbnail_signatures: HashMap<NodeId, (String, bool)>,
}

impl Default for CanvasState {
    fn default() -> Self {
        Self {
            pan: Vec2::ZERO,
            zoom: 1.0,
            selected: HashSet::new(),
            selected_annotation: None,
            snap_to_grid: false,
            drag_unsnapped: HashMap::new(),
            wire_drag: None,
            box_select_start: None,
            last_click_pos: Pos2::ZERO,
            paste_count: 0,
            thumbnail_signatures: HashMap::new(),
        }
    }
}

impl CanvasState {
    /// Where the Node Library panel should drop a newly-added node — see
    /// `last_click_pos`'s own doc comment.
    pub fn last_click_pos(&self) -> Pos2 {
        self.last_click_pos
    }
}

/// What the caller still needs to do after `show` returns — `main.rs` drives
/// `add_node_menu` the same way it already did before this rewrite.
pub struct CanvasOutput {
    pub open_add_menu_at: Option<(Pos2, Pos2)>,
    /// A wire was dragged out from a pin and dropped on empty canvas (no
    /// compatible pin under the release point) — `main.rs` should open the
    /// add-node menu filtered to the given port type/direction, then connect
    /// whatever gets picked back to this pin.
    pub dropped_wire: Option<DroppedWire>,
    /// A wire was dropped on a real pin, but the connection was rejected —
    /// `main.rs` should show this as a toast. Before this field existed, a
    /// rejected connection (most often: dragging "upward" into a node that
    /// would create a cycle, correctly refused) failed completely silently
    /// — the wire preview just vanished on release, with nothing telling
    /// the artist why. See `try_connect`'s own `ConnectFailure` for the
    /// actual reasons this can happen.
    pub connect_rejected: Option<String>,
}

pub struct DroppedWire {
    pub screen_pos: Pos2,
    pub graph_pos: Pos2,
    pub from_node: NodeId,
    pub from_port_index: usize,
    pub from_is_output: bool,
    pub from_port_type: PortType,
}

fn graph_to_screen(origin: Pos2, pan: Vec2, zoom: f32, p: Pos2) -> Pos2 {
    origin + (p.to_vec2() - pan) * zoom
}

fn screen_to_graph(origin: Pos2, pan: Vec2, zoom: f32, s: Pos2) -> Pos2 {
    (pan + (s - origin) / zoom).to_pos2()
}

/// A faint grid of dots over the canvas background ("a light dot grid in
/// twinkling grey") — drawn in graph space so it pans/zooms with the camera
/// like every other node editor's background grid, rather than staying
/// fixed to the screen.
const GRID_BASE_SPACING: f32 = 24.0;

fn draw_dot_grid(painter: &egui::Painter, canvas_rect: Rect, origin: Pos2, pan: Vec2, zoom: f32) {
    // Keeps on-screen dot spacing within a sane range regardless of zoom: at
    // a fixed graph-space spacing, zooming out would either draw an
    // impractical number of dots or (if simply capped) show visibly uneven
    // density. Doubling the spacing in graph space instead keeps the
    // on-screen density roughly constant.
    let mut spacing = GRID_BASE_SPACING;
    while spacing * zoom < 16.0 {
        spacing *= 2.0;
    }

    let top_left = screen_to_graph(origin, pan, zoom, canvas_rect.min);
    let bottom_right = screen_to_graph(origin, pan, zoom, canvas_rect.max);
    let start_x = (top_left.x / spacing).floor() as i64;
    let end_x = (bottom_right.x / spacing).ceil() as i64;
    let start_y = (top_left.y / spacing).floor() as i64;
    let end_y = (bottom_right.y / spacing).ceil() as i64;

    let color = crate::theme::TWINKLING_GREY.gamma_multiply(0.15);
    for gy in start_y..=end_y {
        for gx in start_x..=end_x {
            let p = graph_to_screen(
                origin,
                pan,
                zoom,
                Pos2::new(gx as f32 * spacing, gy as f32 * spacing),
            );
            if canvas_rect.contains(p) {
                painter.circle_filled(p, 1.0, color);
            }
        }
    }
}

fn port_color(port_type: PortType) -> Color32 {
    match port_type {
        PortType::Gray => Color32::from_gray(200),
        PortType::Color => crate::theme::PINK_DWARF,
        PortType::Path => crate::theme::PATH_CYAN,
        PortType::Sdf => crate::theme::SDF_TEAL,
    }
}

/// The node header's title text: the generic `"Output: "` prefix (every
/// Output node def has one — see `mc_nodes::defs`) is swapped for a much
/// shorter `"ON "`, and whatever still doesn't fit the node's fixed width
/// after that is truncated with `"..."` — long display names were bleeding
/// right past the node's own edge, over its neighbors. Applies to any node,
/// not just Output ones, since nothing stops a future node def from having
/// just as long a name.
fn node_title_text(
    painter: &egui::Painter,
    display_name: &str,
    font_id: egui::FontId,
    max_width: f32,
) -> String {
    let shortened = match display_name.strip_prefix("Output: ") {
        Some(rest) => format!("ON {rest}"),
        None => display_name.to_string(),
    };

    let fits = |s: &str| -> bool {
        painter
            .layout_no_wrap(s.to_string(), font_id.clone(), Color32::WHITE)
            .size()
            .x
            <= max_width
    };

    if fits(&shortened) {
        return shortened;
    }

    let mut truncated = shortened;
    while !truncated.is_empty() {
        let candidate = format!("{truncated}...");
        if fits(&candidate) {
            return candidate;
        }
        truncated.pop();
    }
    "...".to_string()
}

/// Node layout constants are defined at `zoom == 1` and scaled by `zoom`
/// everywhere they're used for screen-space geometry — a real proportional
/// zoom, matching every other node editor, rather than only scaling node
/// *positions* and leaving each node's own size fixed (which reads as nodes
/// sliding closer/further apart instead of the view zooming).
fn node_size(def: &mc_nodes::defs::NodeDef, zoom: f32) -> Vec2 {
    let port_rows = def.inputs.len().max(def.outputs.len()).max(1) as f32;
    let height = HEADER_HEIGHT
        + BODY_PADDING
        + port_rows * PORT_ROW_HEIGHT
        + BODY_PADDING
        + THUMBNAIL_SIZE
        + BODY_PADDING;
    Vec2::new(NODE_WIDTH, height) * zoom
}

fn pin_pos(rect: Rect, is_output: bool, index: usize, zoom: f32) -> Pos2 {
    let y =
        rect.min.y + (HEADER_HEIGHT + BODY_PADDING + (index as f32 + 0.5) * PORT_ROW_HEIGHT) * zoom;
    let x = if is_output { rect.max.x } else { rect.min.x };
    Pos2::new(x, y)
}

fn cubic_bezier(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
    let mt = 1.0 - t;
    let a = mt * mt * mt;
    let b = 3.0 * mt * mt * t;
    let c = 3.0 * mt * t * t;
    let d = t * t * t;
    Pos2::new(
        a * p0.x + b * p1.x + c * p2.x + d * p3.x,
        a * p0.y + b * p1.y + c * p2.y + d * p3.y,
    )
}

/// A wire's own curve is drawn with this many straight segments regardless
/// of its length or zoom — plenty for a thin 2px stroke (a true curve and a
/// 10-segment polyline approximation of it are visually indistinguishable
/// at that stroke width), not the 24 this used before for no real reason.
const WIRE_SEGMENTS: i32 = 10;
/// The curve's own minimum "reach" before the two nodes get close enough
/// that `(to.x - from.x) * 0.5` alone would pinch it flat — in *graph*
/// space (scaled by `zoom` at each call site below), not screen pixels.
/// Using a fixed screen-pixel floor here used to mean the curve visibly
/// got curvier as you zoomed out: node spacing shrinks on screen with
/// everything else, but a screen-space floor doesn't, so it increasingly
/// dominates the curve's own shape. Not a deliberate design choice, just
/// what using screen-space pins directly happened to produce.
const WIRE_MIN_REACH: f32 = 40.0;

fn wire_points(from: Pos2, to: Pos2, zoom: f32) -> Vec<Pos2> {
    let control_offset = ((to.x - from.x).abs() * 0.5).max(WIRE_MIN_REACH * zoom);
    let c1 = from + Vec2::new(control_offset, 0.0);
    let c2 = to - Vec2::new(control_offset, 0.0);
    (0..=WIRE_SEGMENTS)
        .map(|i| cubic_bezier(from, c1, c2, to, i as f32 / WIRE_SEGMENTS as f32))
        .collect()
}

fn draw_wire(painter: &egui::Painter, from: Pos2, to: Pos2, zoom: f32, color: Color32) {
    painter.line(wire_points(from, to, zoom), Stroke::new(2.0, color));
}

/// Distance from `p` to the actual (curved) wire, not the straight line
/// between its endpoints — a right-click near a strongly curved wire could
/// be nowhere near that straight chord, which is why disconnect-by-right-
/// click could silently miss.
fn dist_to_wire(p: Pos2, from: Pos2, to: Pos2, zoom: f32) -> f32 {
    let points = wire_points(from, to, zoom);
    points
        .windows(2)
        .map(|w| dist_to_segment(p, w[0], w[1]))
        .fold(f32::INFINITY, f32::min)
}

fn dist_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 <= f32::EPSILON {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length()
}

fn node_def_of(graph: &mc_core::Graph, id: NodeId) -> Option<&'static mc_nodes::defs::NodeDef> {
    mc_nodes::defs::node_def(&graph.node(id)?.kind)
}

/// If `node`'s `port_name` input currently has a connection, disconnects it
/// (so the old wire disappears the instant it's grabbed, not lingering until
/// the new connection lands) and returns the *output* pin it was fed from.
///
/// Rewiring an existing connection should keep its output end anchored and
/// let you drag the input end to a new home — not the other way around. So
/// picking up a connected input re-anchors the drag at that fixed output,
/// via the same `PinRef::Output` a fresh drag from that output would use,
/// rather than anchoring at the input pin that was actually clicked.
fn pick_up_connected_input(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    node: NodeId,
    port_name: &str,
) -> Option<PinRef> {
    let connection = graph
        .connections()
        .iter()
        .find(|c| c.to_node == node && c.to_port == port_name)
        .cloned()?;
    let from_index = node_def_of(graph, connection.from_node).and_then(|d| {
        d.outputs
            .iter()
            .position(|p| p.name == connection.from_port)
    })?;
    let _ = history.do_transaction(
        graph,
        vec![mc_core::Command::Disconnect(connection.clone())],
    );
    Some(PinRef::Output {
        node: connection.from_node,
        index: from_index,
    })
}

/// Connects two pins if exactly one is an output and one is an input and
/// their port types are compatible; rejected otherwise (e.g. dragging
/// between two inputs, incompatible types, or a connection that would
/// create a cycle) — see `ConnectFailure`'s own doc comment for why the
/// caller should surface this rather than swallow it.
fn try_connect(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    a: PinRef,
    b: PinRef,
) -> Result<(), ConnectFailure> {
    let (
        PinRef::Output {
            node: from_node,
            index: from_index,
        },
        PinRef::Input {
            node: to_node,
            index: to_index,
        },
    ) = (match (a, b) {
        (PinRef::Output { .. }, PinRef::Input { .. }) => (a, b),
        (PinRef::Input { .. }, PinRef::Output { .. }) => (b, a),
        _ => return Err(ConnectFailure::NotOneInputOneOutput),
    })
    else {
        return Err(ConnectFailure::NotOneInputOneOutput);
    };

    let Some(from_port) = node_def_of(graph, from_node).and_then(|d| d.outputs.get(from_index))
    else {
        return Err(ConnectFailure::NotOneInputOneOutput);
    };
    let Some(to_port) = node_def_of(graph, to_node).and_then(|d| d.inputs.get(to_index)) else {
        return Err(ConnectFailure::NotOneInputOneOutput);
    };
    let from_kind = graph
        .node(from_node)
        .map(|n| n.kind.clone())
        .unwrap_or_default();
    let from_type =
        mc_nodes::defs::effective_output_type(graph, from_node, &from_kind, from_port.port_type);
    if !mc_nodes::defs::ports_compatible(from_type, to_port.port_type) {
        return Err(ConnectFailure::IncompatibleTypes);
    }
    match graph.connect_transaction(from_node, from_port.name, to_node, to_port.name) {
        Ok(tx) => {
            let _ = history.do_transaction(graph, tx);
            Ok(())
        }
        Err(_) => Err(ConnectFailure::WouldCreateCycle),
    }
}

/// Why a dropped wire didn't connect — surfaced to the artist as a toast
/// (`CanvasOutput::connect_rejected`) rather than left silent. Before this
/// existed, every one of these cases just made the in-progress wire
/// preview vanish on release with no explanation — most confusingly for
/// `WouldCreateCycle`, since the graph's own cycle check is correct,
/// required behavior (evaluation assumes a DAG), but dragging a wire
/// "upward" into an indirect source of the node you started from hits it
/// constantly in normal use and looks exactly like "the connection just
/// doesn't register."
enum ConnectFailure {
    NotOneInputOneOutput,
    IncompatibleTypes,
    WouldCreateCycle,
}

impl ConnectFailure {
    fn message(&self) -> &'static str {
        match self {
            ConnectFailure::NotOneInputOneOutput => {
                "Can't connect: need one output pin and one input pin"
            }
            ConnectFailure::IncompatibleTypes => "Can't connect: incompatible port types",
            ConnectFailure::WouldCreateCycle => "Can't connect: would create a loop",
        }
    }
}

/// Duplicates `id` — or, if it's part of the current multi-selection, every
/// selected node — with the same kind/params/label, offset slightly so
/// copies never sit exactly on top of the originals. The new node(s) become
/// the selection, so they're immediately ready to drag away, matching
/// duplicate in most other node editors.
const DUPLICATE_OFFSET: f32 = 30.0;

/// Gathers `ids` as `(NodeId, Node)` pairs, plus whichever of `graph`'s own
/// connections have both endpoints inside `ids` — the common "selection plus
/// its internal wiring" shape both Duplicate and Copy need.
fn gather_subgraph(
    graph: &mc_core::Graph,
    ids: &[NodeId],
) -> (Vec<(NodeId, mc_core::Node)>, Vec<mc_core::Connection>) {
    let id_set: HashSet<NodeId> = ids.iter().copied().collect();
    let nodes = ids
        .iter()
        .filter_map(|&id| graph.node(id).cloned().map(|node| (id, node)))
        .collect();
    let connections = graph
        .connections()
        .iter()
        .filter(|c| id_set.contains(&c.from_node) && id_set.contains(&c.to_node))
        .cloned()
        .collect();
    (nodes, connections)
}

/// Clones `nodes` (fresh ids, `offset` applied to each position) plus
/// `connections` (whichever of the original connections had both endpoints
/// among the cloned nodes, as returned by `gather_subgraph`), as two undo
/// steps — see `add_nodes_transaction`'s own doc comment for why not one.
/// Returns the new node ids, same order as `nodes`. Shared by
/// `duplicate_nodes` (source: the live graph's own selected nodes — this is
/// also the fix for Duplicate's own pre-existing "drops internal
/// connections" gap) and `paste_from_clipboard` (source: a deserialized
/// clipboard payload) — same operation, different data source.
fn clone_subgraph(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    nodes: &[(NodeId, mc_core::Node)],
    connections: &[mc_core::Connection],
    offset: [f32; 2],
) -> Vec<NodeId> {
    if nodes.is_empty() {
        return Vec::new();
    }

    let cloned_nodes: Vec<mc_core::Node> = nodes
        .iter()
        .map(|(_, node)| {
            let mut node = node.clone();
            node.position = [node.position[0] + offset[0], node.position[1] + offset[1]];
            node
        })
        .collect();
    let (new_ids, transaction) = graph.add_nodes_transaction(cloned_nodes);
    if history.do_transaction(graph, transaction).is_err() {
        return Vec::new();
    }

    let remap: HashMap<NodeId, NodeId> = nodes
        .iter()
        .map(|(old_id, _)| *old_id)
        .zip(new_ids.iter().copied())
        .collect();
    let mut wiring = mc_core::Transaction::new();
    for connection in connections {
        let (Some(&from), Some(&to)) = (
            remap.get(&connection.from_node),
            remap.get(&connection.to_node),
        ) else {
            continue;
        };
        if let Ok(connect) =
            graph.connect_transaction(from, &connection.from_port, to, &connection.to_port)
        {
            wiring.extend(connect);
        }
    }
    if !wiring.is_empty() {
        let _ = history.do_transaction(graph, wiring);
    }

    new_ids
}

fn duplicate_nodes(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    canvas: &mut CanvasState,
    selected: &mut Option<NodeId>,
    preview: &mut Option<NodeId>,
    id: NodeId,
) {
    let targets: Vec<NodeId> = if canvas.selected.contains(&id) {
        canvas.selected.iter().copied().collect()
    } else {
        vec![id]
    };
    let (nodes, connections) = gather_subgraph(graph, &targets);
    let new_ids = clone_subgraph(
        graph,
        history,
        &nodes,
        &connections,
        [DUPLICATE_OFFSET, DUPLICATE_OFFSET],
    );
    if !new_ids.is_empty() {
        canvas.selected.clear();
        canvas.selected.extend(new_ids.iter().copied());
        *selected = new_ids.last().copied();
        *preview = *selected;
    }
}

/// What gets serialized to the OS clipboard by Copy, and parsed back by
/// Paste — `NodeId`/`Node`/`Connection` already derive Serialize/Deserialize,
/// reused as-is rather than inventing a parallel type. `version` isn't read
/// yet (there's only ever been one), but its presence means a future format
/// change can detect and reject an old payload instead of silently
/// misreading it.
#[derive(Serialize, Deserialize)]
struct ClipboardPayload {
    version: u32,
    nodes: Vec<(NodeId, mc_core::Node)>,
    connections: Vec<mc_core::Connection>,
}

const CLIPBOARD_PAYLOAD_VERSION: u32 = 1;

/// Copies the current selection (nodes plus their internal wiring) to the OS
/// clipboard as JSON, via egui's own clipboard bridge. A no-op if nothing is
/// selected. Resets `canvas.paste_count`, so the next Paste lands at the
/// base offset again rather than continuing to fan out from a previous
/// paste spree. Returns how many nodes were copied (0 for a no-op) — purely
/// so `main.rs` can show a toast; this module has no toast UI of its own.
pub fn copy_selection_to_clipboard(
    ctx: &egui::Context,
    graph: &mc_core::Graph,
    canvas: &mut CanvasState,
) -> usize {
    if canvas.selected.is_empty() {
        return 0;
    }
    let ids: Vec<NodeId> = canvas.selected.iter().copied().collect();
    let (nodes, connections) = gather_subgraph(graph, &ids);
    let count = nodes.len();
    let payload = ClipboardPayload {
        version: CLIPBOARD_PAYLOAD_VERSION,
        nodes,
        connections,
    };
    match serde_json::to_string(&payload) {
        Ok(json) => {
            ctx.copy_text(json);
            canvas.paste_count = 0;
            count
        }
        Err(_) => 0,
    }
}

/// Looks for a `Paste` event this frame and, if its text parses as a
/// `ClipboardPayload`, clones it into the graph via `clone_subgraph` and
/// selects the result — otherwise (no paste event, or clipboard text isn't
/// one of ours) does nothing. Pasting unrelated clipboard text must never
/// panic or create garbage nodes, so a parse failure is silently ignored.
/// Returns how many nodes were pasted (0 if there was nothing to do) —
/// purely so `main.rs` can show a toast on a real paste, not on every frame.
pub fn paste_from_clipboard(
    ctx: &egui::Context,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    canvas: &mut CanvasState,
    selected: &mut Option<NodeId>,
    preview: &mut Option<NodeId>,
) -> usize {
    let pasted_text = ctx.input(|i| {
        i.events.iter().find_map(|event| match event {
            egui::Event::Paste(text) => Some(text.clone()),
            _ => None,
        })
    });
    let Some(text) = pasted_text else {
        return 0;
    };
    let Ok(payload) = serde_json::from_str::<ClipboardPayload>(&text) else {
        return 0;
    };

    canvas.paste_count += 1;
    let offset = DUPLICATE_OFFSET * canvas.paste_count as f32;
    let new_ids = clone_subgraph(
        graph,
        history,
        &payload.nodes,
        &payload.connections,
        [offset, offset],
    );
    if !new_ids.is_empty() {
        canvas.selected.clear();
        canvas.selected.extend(new_ids.iter().copied());
        *selected = new_ids.last().copied();
        *preview = *selected;
    }
    new_ids.len()
}

/// Horizontal gap between layout columns, and vertical gap between
/// stacked nodes within a column — both graph units, independent of
/// `NODE_WIDTH`/node height so a tall multi-port node still gets a clean
/// gap below it rather than touching its neighbor.
const AUTO_LAYOUT_COLUMN_GAP: f32 = 80.0;
const AUTO_LAYOUT_ROW_GAP: f32 = 30.0;

/// What "Add Frame" creates when `selection` is non-empty: a frame sized and
/// positioned to just enclose every selected node's own bounding box (via
/// `node_size`, at `zoom = 1.0` — graph-space, not screen-space), padded by
/// `FRAME_SELECTION_PADDING` on every side but the top. A selected id whose
/// node no longer exists, or whose kind has no `NodeDef` (shouldn't happen,
/// but cheaper to skip than to unwrap), is simply left out of the bounding
/// box rather than failing the whole thing.
fn frame_around_selection(graph: &mc_core::Graph, selection: &HashSet<NodeId>) -> mc_core::Frame {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for &id in selection {
        let Some(node) = graph.node(id) else {
            continue;
        };
        let Some(def) = mc_nodes::defs::node_def(&node.kind) else {
            continue;
        };
        let size = node_size(def, 1.0);
        min_x = min_x.min(node.position[0]);
        min_y = min_y.min(node.position[1]);
        max_x = max_x.max(node.position[0] + size.x);
        max_y = max_y.max(node.position[1] + size.y);
    }
    if !min_x.is_finite() {
        return mc_core::Frame::new("New Frame", [0.0, 0.0], mc_core::DEFAULT_FRAME_SIZE);
    }
    let position = [
        min_x - FRAME_SELECTION_PADDING,
        min_y - FRAME_SELECTION_PADDING - FRAME_TITLE_BAR_HEIGHT,
    ];
    let size = [
        (max_x - min_x) + FRAME_SELECTION_PADDING * 2.0,
        (max_y - min_y) + FRAME_SELECTION_PADDING * 2.0 + FRAME_TITLE_BAR_HEIGHT,
    ];
    mc_core::Frame::new("New Frame", position, size)
}

/// Rearranges nodes in `graph` into a left-to-right layered layout: column =
/// longest-path distance from a source node (no inputs, *within the laid-
/// out set* — see `selection` below), row = stacked within that column by
/// each node's own real height (via `node_size`), ordered by a one-pass
/// barycenter heuristic (each node sorted by the average row of its own
/// upstream nodes) to cut down on wire crossings.
///
/// `selection` scopes the layout: empty applies it to every node in the
/// graph (anchored at the graph origin, the original whole-graph
/// behavior); non-empty applies it only to those nodes, with every other
/// node left untouched — connections to/from outside the selection are
/// ignored for column/row purposes, so the selected subgraph is laid out
/// by its own internal topology alone. A selection-scoped layout is
/// anchored at the selected nodes' own current top-left corner (rather
/// than the graph origin) so tidying up one small group doesn't fling it
/// across the canvas into unrelated nodes.
///
/// Recorded as one undo step of `Command::MoveNode`s — deliberately *not*
/// following plain dragging's own "position isn't undo-tracked" rule (see
/// this module's own doc comment): a single drag moves one node a few
/// pixels, but this can silently rewrite every node's hand-placed position
/// in a 50+ node graph in a single click, which is a big enough blast
/// radius to deserve a real undo step.
pub fn auto_layout(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    selection: &HashSet<NodeId>,
) {
    let Ok(full_order) = graph.topological_order() else {
        return;
    };
    if full_order.is_empty() {
        return;
    }
    let scoped = !selection.is_empty();
    let order: Vec<NodeId> = full_order
        .into_iter()
        .filter(|id| !scoped || selection.contains(id))
        .collect();
    if order.is_empty() {
        return;
    }

    let mut column: HashMap<NodeId, u32> = HashMap::new();
    for &id in &order {
        let col = graph
            .connections()
            .iter()
            .filter(|c| c.to_node == id && (!scoped || selection.contains(&c.from_node)))
            .filter_map(|c| column.get(&c.from_node))
            .max()
            .map(|&upstream_col| upstream_col + 1)
            .unwrap_or(0);
        column.insert(id, col);
    }

    let column_count = column.values().max().map(|&c| c + 1).unwrap_or(0) as usize;
    let mut columns: Vec<Vec<NodeId>> = vec![Vec::new(); column_count];
    for &id in &order {
        columns[column[&id] as usize].push(id);
    }

    // Every `output_*` node is always a graph leaf (nothing ever reads
    // from one), so moving them all into one shared final column — to the
    // right of everything else, regardless of each one's own natural
    // topological depth — is purely cosmetic and never disturbs a real
    // dependency, scoped selection or not.
    let mut output_ids: Vec<NodeId> = Vec::new();
    for nodes_in_column in &mut columns {
        nodes_in_column.retain(|&id| {
            let is_output = graph
                .node(id)
                .is_some_and(|node| node.kind.starts_with("output_"));
            if is_output {
                output_ids.push(id);
            }
            !is_output
        });
    }
    columns.retain(|nodes_in_column| !nodes_in_column.is_empty());
    if !output_ids.is_empty() {
        columns.push(output_ids);
    }

    // Barycenter row ordering: processed column by column, left to right,
    // so by the time a node's own row is needed as an upstream reference
    // for a later column, it's already been assigned.
    let mut row_of: HashMap<NodeId, f32> = HashMap::new();
    for nodes_in_column in &mut columns {
        nodes_in_column.sort_by(|&a, &b| {
            let barycenter = |id: NodeId| -> f32 {
                let upstream_rows: Vec<f32> = graph
                    .connections()
                    .iter()
                    .filter(|c| c.to_node == id && (!scoped || selection.contains(&c.from_node)))
                    .filter_map(|c| row_of.get(&c.from_node).copied())
                    .collect();
                if upstream_rows.is_empty() {
                    0.0
                } else {
                    upstream_rows.iter().sum::<f32>() / upstream_rows.len() as f32
                }
            };
            barycenter(a)
                .partial_cmp(&barycenter(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (row, &id) in nodes_in_column.iter().enumerate() {
            row_of.insert(id, row as f32);
        }
    }

    let (start_x, start_y) = if scoped {
        let mut min = Vec2::new(f32::INFINITY, f32::INFINITY);
        for &id in &order {
            if let Some(node) = graph.node(id) {
                min.x = min.x.min(node.position[0]);
                min.y = min.y.min(node.position[1]);
            }
        }
        (min.x, min.y)
    } else {
        (0.0, 0.0)
    };

    let mut transaction: mc_core::Transaction = Vec::new();
    let mut x = start_x;
    for nodes_in_column in &columns {
        let mut y = start_y;
        for &id in nodes_in_column {
            transaction.push(mc_core::Command::MoveNode {
                id,
                position: [x, y],
            });
            let height = node_def_of(graph, id)
                .map(|def| node_size(def, 1.0).y)
                .unwrap_or(0.0);
            y += height + AUTO_LAYOUT_ROW_GAP;
        }
        x += NODE_WIDTH + AUTO_LAYOUT_COLUMN_GAP;
    }

    if !transaction.is_empty() {
        let _ = history.do_transaction(graph, transaction);
    }
}

/// One frame of (optionally grid-snapped) node dragging: adds `delta` onto
/// `base_unsnapped` (the running, full-precision position — *not* the
/// node's own stored position, which may already be rounded from a
/// previous frame) and returns `(position_to_store, new_unsnapped_total)`.
/// Snapping only the returned *store* value, while the caller keeps
/// feeding `new_unsnapped_total` back in as next frame's `base_unsnapped`,
/// is what makes a slow drag actually progress instead of sticking:
/// re-rounding an already-rounded position every frame means a small
/// per-frame delta just rounds straight back to the same grid cell,
/// forever, since nothing anywhere retains the sub-grid precision needed
/// to ever cross into the next one.
fn apply_drag_delta(
    base_unsnapped: [f32; 2],
    delta: Vec2,
    snap_to_grid: bool,
) -> ([f32; 2], [f32; 2]) {
    let raw = [base_unsnapped[0] + delta.x, base_unsnapped[1] + delta.y];
    let out = if snap_to_grid {
        [
            (raw[0] / GRID_BASE_SPACING).round() * GRID_BASE_SPACING,
            (raw[1] / GRID_BASE_SPACING).round() * GRID_BASE_SPACING,
        ]
    } else {
        raw
    };
    (out, raw)
}

/// How close to the canvas edge (screen pixels) the pointer needs to be,
/// while dragging a wire, before the view starts scrolling toward it, and
/// the top speed (screen pixels/second) reached right at — or past — that
/// edge, tapering linearly to zero at the margin's own inner edge.
const WIRE_DRAG_AUTOPAN_MARGIN: f32 = 60.0;
const WIRE_DRAG_AUTOPAN_MAX_SPEED: f32 = 700.0;

/// The raw pan velocity (graph-camera screen-pixels/second, before `dt` or
/// zoom are applied) wire-drag autopan wants for `pointer`'s current
/// position relative to `canvas_rect` — split out from
/// [`autopan_for_wire_drag`] so this edge-margin math is testable without
/// a real `egui::Ui`/`Context`. A pointer outside `canvas_rect` entirely
/// (dragged clean off the canvas) clamps to the same top speed as right at
/// the edge, rather than accelerating further or doing nothing.
fn wire_drag_autopan_velocity(canvas_rect: Rect, pointer: Pos2) -> Vec2 {
    let edge_speed = |dist_from_edge: f32| -> f32 {
        ((WIRE_DRAG_AUTOPAN_MARGIN - dist_from_edge) / WIRE_DRAG_AUTOPAN_MARGIN).clamp(0.0, 1.0)
            * WIRE_DRAG_AUTOPAN_MAX_SPEED
    };
    let mut v = Vec2::ZERO;
    v.x -= edge_speed(pointer.x - canvas_rect.left());
    v.x += edge_speed(canvas_rect.right() - pointer.x);
    v.y -= edge_speed(pointer.y - canvas_rect.top());
    v.y += edge_speed(canvas_rect.bottom() - pointer.y);
    v
}

/// While a wire is being dragged, scrolls `canvas`'s own camera toward
/// wherever `pointer` is hovering near (or past) `canvas_rect`'s edge —
/// see [`wire_drag_autopan_velocity`] for the actual math. Requests a
/// repaint every active frame, since this has to keep moving even while
/// the pointer itself is perfectly still (egui only redraws on its own
/// otherwise).
fn autopan_for_wire_drag(
    canvas: &mut CanvasState,
    canvas_rect: Rect,
    pointer: Pos2,
    ui: &egui::Ui,
) {
    let velocity = wire_drag_autopan_velocity(canvas_rect, pointer);
    if velocity != Vec2::ZERO {
        let dt = ui.input(|i| i.stable_dt);
        canvas.pan += velocity * dt / canvas.zoom;
        ui.ctx().request_repaint();
    }
}

const FRAME_TITLE_BAR_HEIGHT: f32 = 26.0;
const FRAME_RESIZE_HANDLE_SIZE: f32 = 14.0;
/// A resize drag can never shrink a frame past this — small enough to still
/// be a meaningful grouping box, large enough that the title bar and resize
/// handle never fully overlap each other.
const FRAME_MIN_SIZE: [f32; 2] = [120.0, 80.0];
/// How much breathing room "Add Frame" leaves around the selected nodes'
/// own bounding box, on every side but the top (which instead reserves
/// `FRAME_TITLE_BAR_HEIGHT` so the title bar doesn't sit directly on top of
/// the topmost node).
const FRAME_SELECTION_PADDING: f32 = 40.0;

fn frame_color32(color: [f32; 4]) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
        (color[3].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

// Matches `twod_toolbar.rs`'s and `viewport3d_panels.rs`'s own BUTTON_SIZE/
// ICON_FONT_SIZE exactly — the owner's own request, so these corner
// buttons read as the same button style as the 2D/3D view sidebars rather
// than a visually distinct one-off.
const CORNER_BUTTON_SIZE: f32 = 30.0;
const CORNER_BUTTON_GAP: f32 = 6.0;
const CORNER_BUTTON_MARGIN: f32 = 10.0;
const CORNER_ICON_FONT_SIZE: f32 = 20.0;
// Codepoints confirmed against Google's own MaterialIconsRound-Regular.
// codepoints file, same convention every other glyph constant in this app
// follows (see `twod_toolbar.rs`'s own module doc comment).
const AUTO_LAYOUT_GLYPH: char = '\u{e8f0}'; // "view_module" — the owner's own pick.
                                            // "magnet" isn't in this app's bundled font (that's a Material *Symbols*
                                            // icon, a different, newer set this app doesn't ship) — "grid_on",
                                            // confirmed with the owner as her second choice once that was explained.
const SNAP_TO_GRID_GLYPH: char = '\u{e3ec}'; // "grid_on"
const ADD_COMMENT_GLYPH: char = '\u{e0b9}'; // "comment"
const ADD_FRAME_GLYPH: char = '\u{e3c2}'; // "crop_free"
                                          // Confirmed against the bundled `MaterialIconsRound-Regular.otf`'s own
                                          // cmap (glyph name "speed").
const SPEED_GLYPH: char = '\u{e9e4}';

/// Square icon button for the canvas's own top-right corner controls
/// (Auto Layout, Snap to Grid) — same visual recipe as
/// `twod_toolbar::icon_button` (that one's private to its own file, and a
/// third near-identical copy here follows the same precedent
/// `viewport3d_panels::icon_button` already set rather than factor out a
/// shared helper three different UI styles would all have to agree on).
fn corner_icon_button(
    ui: &mut egui::Ui,
    rect: Rect,
    id_source: &str,
    glyph: char,
    selected: bool,
    tooltip: &str,
) -> egui::Response {
    let response = ui.interact(rect, ui.id().with(id_source), Sense::click());
    let visuals = ui.style().interact_selectable(&response, selected);
    let painter = ui.painter();
    if selected || response.hovered() {
        painter.rect_filled(rect, 4.0, visuals.bg_fill);
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        glyph.to_string(),
        egui::FontId::new(
            CORNER_ICON_FONT_SIZE,
            egui::FontFamily::Name("material_icons_round".into()),
        ),
        visuals.fg_stroke.color,
    );
    response.on_hover_text(tooltip)
}

#[expect(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    canvas: &mut CanvasState,
    selected: &mut Option<NodeId>,
    preview: &mut Option<NodeId>,
    preview_port: &mut Option<(NodeId, String)>,
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    render_state: &egui_wgpu::RenderState,
    thumbnail_ids: &mut HashMap<NodeId, egui::TextureId>,
    perf_overlay_enabled: &mut bool,
    node_cost_ms: &HashMap<NodeId, f32>,
    last_evaluated_nodes: &HashSet<NodeId>,
) -> CanvasOutput {
    let canvas_rect = ui.available_rect_before_wrap();
    let origin = canvas_rect.min;
    let modifiers = ui.input(|i| i.modifiers);
    let mut output = CanvasOutput {
        open_add_menu_at: None,
        dropped_wire: None,
        connect_rejected: None,
    };

    // Background interact region first, so it sits "under" the per-node/
    // per-pin regions added later this frame — same z-order-by-add-order
    // egui-snarl itself relied on, and the reason this works without any
    // special-casing: whichever interactive region was added *last* for a
    // given screen position wins.
    let bg_id = ui.id().with("mc_canvas_bg");
    let bg_response = ui.interact(canvas_rect, bg_id, Sense::click_and_drag());

    // Trackpad two-finger scroll and a mouse wheel both land in
    // `smooth_scroll_delta` and pan; pinch (and ctrl/cmd-scroll, which egui
    // already folds into `zoom_delta()` separately from the scroll delta)
    // zooms. Keeping these on separate input signals is what makes a plain
    // trackpad swipe pan instead of zooming — they used to be the same
    // scroll-wheel signal, which is why a swipe looked like "nodes pushing
    // closer together" instead of the view sliding.
    //
    // Gated on the pointer being anywhere inside the canvas *and* this
    // canvas's own layer being the topmost one there — not on
    // `bg_response.hovered()` (a node's own interact region, added later,
    // z-order-wins the hover over the background's, which used to make
    // pinch-zoom and pan silently stop working the moment the cursor
    // crossed over a node), but also not on `canvas_rect.contains(pointer)`
    // alone (that doesn't know about a foreground popup like the add-node
    // menu sitting on top — its own scroll area would otherwise fight the
    // canvas for the same scroll gesture, with the canvas panning *behind*
    // the popup at the same time the popup scrolled its list).
    let canvas_layer = ui.layer_id();
    if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
        if canvas_rect.contains(pointer) && ui.ctx().layer_id_at(pointer) == Some(canvas_layer) {
            let zoom_delta = ui.input(|i| i.zoom_delta());
            if zoom_delta != 1.0 {
                let graph_pt = screen_to_graph(origin, canvas.pan, canvas.zoom, pointer);
                canvas.zoom = (canvas.zoom * zoom_delta).clamp(0.2, 3.0);
                canvas.pan = graph_pt.to_vec2() - (pointer - origin) / canvas.zoom;
            }
            let scroll = ui.input(|i| i.smooth_scroll_delta);
            if scroll != Vec2::ZERO {
                canvas.pan -= scroll / canvas.zoom;
            }
        }
    }

    let painter = ui.painter_at(canvas_rect);
    draw_dot_grid(&painter, canvas_rect, origin, canvas.pan, canvas.zoom);

    // Frame pass: drawn and interacted with first (background grouping
    // boxes, meant to sit visually *behind* everything else) — added
    // before node rects are even computed so a node's own interact region,
    // added later this frame, naturally wins any click on the overlap
    // (the file's own "whichever interactive region was added last wins"
    // rule, same one the background-pan region at the top of `show` relies
    // on). Frames don't drag the nodes visually inside them (see `Frame`'s
    // own `mc_core` doc comment) — purely a background rectangle with a
    // title and a resize handle.
    // Snapshotted as owned values up front, rather than holding a borrow of
    // `graph.frame(id)` across the loop body — several things below (drag,
    // resize, the context menu's delete) need `graph`/`history` mutably at
    // the same time.
    let frames: Vec<FrameSnapshot> = graph
        .frames()
        .map(|(id, frame)| FrameSnapshot {
            id,
            title: frame.title.clone(),
            color: frame.color,
            position: frame.position,
            size: frame.size,
        })
        .collect();
    for FrameSnapshot {
        id,
        title,
        color,
        position,
        size,
    } in frames
    {
        let rect = Rect::from_min_size(
            graph_to_screen(
                origin,
                canvas.pan,
                canvas.zoom,
                Pos2::new(position[0], position[1]),
            ),
            Vec2::new(size[0], size[1]) * canvas.zoom,
        );
        let is_selected = canvas.selected_annotation == Some(AnnotationSelection::Frame(id));
        let fill = frame_color32(color);
        painter.rect_filled(rect, 6.0 * canvas.zoom, fill);
        let border_color = if is_selected {
            crate::theme::PINK_DWARF
        } else {
            fill.gamma_multiply(1.6)
        };
        painter.rect_stroke(
            rect,
            6.0 * canvas.zoom,
            Stroke::new(if is_selected { 2.0 } else { 1.0 }, border_color),
            StrokeKind::Outside,
        );

        let title_bar_height = FRAME_TITLE_BAR_HEIGHT * canvas.zoom;
        let title_rect = Rect::from_min_size(rect.min, Vec2::new(rect.width(), title_bar_height));
        painter.rect_filled(title_rect, 6.0 * canvas.zoom, fill.gamma_multiply(1.3));
        painter.text(
            title_rect.left_center() + Vec2::new(8.0 * canvas.zoom, 0.0),
            egui::Align2::LEFT_CENTER,
            &title,
            egui::FontId::proportional(13.0 * canvas.zoom),
            ui.visuals().text_color(),
        );

        let title_response = ui.interact(
            title_rect,
            ui.id().with(("mc_frame_title", id)),
            Sense::click_and_drag(),
        );
        if title_response.dragged() {
            let delta = title_response.drag_delta() / canvas.zoom;
            let new_position = [position[0] + delta.x, position[1] + delta.y];
            let _ = graph.set_frame_position(id, new_position);
        }
        if title_response.clicked() {
            if let Some(screen_pos) = title_response.interact_pointer_pos() {
                canvas.last_click_pos =
                    screen_to_graph(origin, canvas.pan, canvas.zoom, screen_pos);
            }
            canvas.selected_annotation = Some(AnnotationSelection::Frame(id));
            *selected = None;
        }
        title_response.context_menu(|ui| {
            if ui.button("Delete Frame").clicked() {
                let transaction = graph.remove_frame_transaction(id);
                let _ = history.do_transaction(graph, transaction);
                if canvas.selected_annotation == Some(AnnotationSelection::Frame(id)) {
                    canvas.selected_annotation = None;
                }
                ui.close();
            }
        });

        let handle_size = FRAME_RESIZE_HANDLE_SIZE * canvas.zoom;
        let handle_rect = Rect::from_min_size(
            rect.max - Vec2::splat(handle_size),
            Vec2::splat(handle_size),
        );
        painter.rect_filled(handle_rect, 2.0, fill.gamma_multiply(1.6));
        let handle_response = ui.interact(
            handle_rect,
            ui.id().with(("mc_frame_resize", id)),
            Sense::drag(),
        );
        if handle_response.dragged() {
            let delta = handle_response.drag_delta() / canvas.zoom;
            let new_size = [
                (size[0] + delta.x).max(FRAME_MIN_SIZE[0]),
                (size[1] + delta.y).max(FRAME_MIN_SIZE[1]),
            ];
            let _ = graph.set_frame_size(id, new_size);
        }
    }

    // Pass 1: node screen rects (read-only — needed before any drawing so
    // wires can be drawn *behind* nodes).
    let mut node_rects: HashMap<NodeId, Rect> = HashMap::new();
    for (id, node) in graph.nodes() {
        let Some(def) = mc_nodes::defs::node_def(&node.kind) else {
            continue;
        };
        let pos = graph_to_screen(
            origin,
            canvas.pan,
            canvas.zoom,
            Pos2::new(node.position[0], node.position[1]),
        );
        node_rects.insert(id, Rect::from_min_size(pos, node_size(def, canvas.zoom)));
    }

    // Pass 2: pin positions (read-only), used for both wire endpoints and
    // wire-drag hit-testing.
    let mut pins: Vec<PinInfo> = Vec::new();
    for (id, node) in graph.nodes() {
        let Some(def) = mc_nodes::defs::node_def(&node.kind) else {
            continue;
        };
        let Some(&rect) = node_rects.get(&id) else {
            continue;
        };
        for (i, port) in def.outputs.iter().enumerate() {
            let port_type =
                mc_nodes::defs::effective_output_type(graph, id, &node.kind, port.port_type);
            pins.push(PinInfo {
                r: PinRef::Output { node: id, index: i },
                pos: pin_pos(rect, true, i, canvas.zoom),
                port_type,
            });
        }
        for (i, port) in def.inputs.iter().enumerate() {
            pins.push(PinInfo {
                r: PinRef::Input { node: id, index: i },
                pos: pin_pos(rect, false, i, canvas.zoom),
                port_type: port.port_type,
            });
        }
    }

    // Pass 3: wires (drawn before nodes so node bodies paint on top of the
    // wire ends, not the other way around).
    let mut wires: Vec<(mc_core::Connection, Pos2, Pos2)> = Vec::new();
    for connection in graph.connections() {
        let from_index = node_def_of(graph, connection.from_node).and_then(|d| {
            d.outputs
                .iter()
                .position(|p| p.name == connection.from_port)
        });
        let to_index = node_def_of(graph, connection.to_node)
            .and_then(|d| d.inputs.iter().position(|p| p.name == connection.to_port));
        let (Some(fi), Some(ti)) = (from_index, to_index) else {
            continue;
        };
        let from_pos = pins
            .iter()
            .find(|p| {
                p.r == PinRef::Output {
                    node: connection.from_node,
                    index: fi,
                }
            })
            .map(|p| p.pos);
        let to_pos = pins
            .iter()
            .find(|p| {
                p.r == PinRef::Input {
                    node: connection.to_node,
                    index: ti,
                }
            })
            .map(|p| p.pos);
        if let (Some(fp), Some(tp)) = (from_pos, to_pos) {
            wires.push((connection.clone(), fp, tp));
        }
    }
    for (connection, from_pos, to_pos) in &wires {
        let color = node_def_of(graph, connection.from_node)
            .and_then(|d| d.output(&connection.from_port))
            .map_or(Color32::from_gray(160), |p| port_color(p.port_type));
        draw_wire(&painter, *from_pos, *to_pos, canvas.zoom, color);
    }

    // Comment pass: drawn above wires (so they read as sitting "on" the
    // graph, not buried under it) but before nodes, so an overlapping
    // node's own later-added interact region still wins clicks on the
    // overlap — same z-order-by-add-order reasoning the Frame pass above
    // already uses.
    let comments: Vec<(CommentId, String, [f32; 2])> = graph
        .comments()
        .map(|(id, comment)| (id, comment.text.clone(), comment.position))
        .collect();
    for (id, text, position) in comments {
        let font_id = egui::FontId::proportional(14.0 * canvas.zoom);
        let text_color = ui.visuals().text_color();
        let size = painter
            .layout_no_wrap(text.clone(), font_id.clone(), text_color)
            .size();
        let padding = Vec2::splat(8.0 * canvas.zoom);
        let screen_pos = graph_to_screen(
            origin,
            canvas.pan,
            canvas.zoom,
            Pos2::new(position[0], position[1]),
        );
        let rect = Rect::from_min_size(screen_pos, size + padding * 2.0);
        let is_selected = canvas.selected_annotation == Some(AnnotationSelection::Comment(id));

        painter.rect_filled(
            rect,
            4.0 * canvas.zoom,
            Color32::from_rgba_unmultiplied(40, 38, 20, 210),
        );
        painter.rect_stroke(
            rect,
            4.0 * canvas.zoom,
            Stroke::new(
                if is_selected { 2.0 } else { 1.0 },
                if is_selected {
                    crate::theme::PINK_DWARF
                } else {
                    Color32::from_gray(120)
                },
            ),
            StrokeKind::Outside,
        );
        painter.text(
            rect.min + padding,
            egui::Align2::LEFT_TOP,
            &text,
            font_id,
            text_color,
        );

        let response = ui.interact(
            rect,
            ui.id().with(("mc_comment", id)),
            Sense::click_and_drag(),
        );
        if response.dragged() {
            let delta = response.drag_delta() / canvas.zoom;
            let new_position = [position[0] + delta.x, position[1] + delta.y];
            let _ = graph.set_comment_position(id, new_position);
        }
        if response.clicked() {
            if let Some(screen_pos) = response.interact_pointer_pos() {
                canvas.last_click_pos =
                    screen_to_graph(origin, canvas.pan, canvas.zoom, screen_pos);
            }
            canvas.selected_annotation = Some(AnnotationSelection::Comment(id));
            *selected = None;
        }
        response.context_menu(|ui| {
            if ui.button("Delete Comment").clicked() {
                let transaction = graph.remove_comment_transaction(id);
                let _ = history.do_transaction(graph, transaction);
                if canvas.selected_annotation == Some(AnnotationSelection::Comment(id)) {
                    canvas.selected_annotation = None;
                }
                ui.close();
            }
        });
    }

    // Pass 4: nodes themselves — drawing, selection, dragging, pins.
    let node_ids: Vec<NodeId> = graph.nodes().map(|(id, _)| id).collect();
    for id in node_ids {
        let Some(def) = node_def_of(graph, id) else {
            continue;
        };
        let Some(&rect) = node_rects.get(&id) else {
            continue;
        };
        let is_selected = canvas.selected.contains(&id);

        let node_id = ui.id().with(("mc_node", id));
        let node_response = ui.interact(rect, node_id, Sense::click_and_drag());

        if node_response.drag_started_by(egui::PointerButton::Primary) && !is_selected {
            canvas.selected.clear();
            canvas.selected.insert(id);
            *preview = Some(id);
        }
        if node_response.dragged_by(egui::PointerButton::Primary) {
            let delta = node_response.drag_delta() / canvas.zoom;
            let targets: Vec<NodeId> = if canvas.selected.contains(&id) {
                canvas.selected.iter().copied().collect()
            } else {
                vec![id]
            };
            // Snap to Grid (the corner toggle button) locks dragged nodes
            // to the same `GRID_BASE_SPACING` the dot grid is drawn at —
            // not whatever multiple the dots render at when zoomed out,
            // which only doubles for on-screen readability, not layout.
            // Snapped on every frame, not just on release, so it reads as
            // the node locking to the grid as it moves rather than
            // jumping once at the end.
            //
            // The per-frame delta is added onto `drag_unsnapped`'s own
            // running total, not onto the node's actual (already-rounded)
            // stored position — a real bug once fixed: rounding the
            // already-rounded position every frame meant a slow drag's
            // small per-frame delta kept rounding straight back to the
            // same grid cell it started at, forever, since it never
            // accumulated anywhere with full precision. From the node's
            // own point of view that reads as the drag simply stopping
            // dead the moment it first snapped. See `apply_drag_delta`'s
            // own doc comment.
            for target in targets {
                let base = canvas
                    .drag_unsnapped
                    .get(&target)
                    .copied()
                    .or_else(|| graph.node(target).map(|n| n.position));
                if let Some(pos) = base {
                    let (out_pos, raw) = apply_drag_delta(pos, delta, canvas.snap_to_grid);
                    if canvas.snap_to_grid {
                        canvas.drag_unsnapped.insert(target, raw);
                    } else {
                        canvas.drag_unsnapped.remove(&target);
                    }
                    let _ = graph.set_node_position(target, out_pos);
                }
            }
        }
        if node_response.drag_stopped_by(egui::PointerButton::Primary) {
            let targets: Vec<NodeId> = if canvas.selected.contains(&id) {
                canvas.selected.iter().copied().collect()
            } else {
                vec![id]
            };
            // Only for a single dragged node — splicing a whole multi-
            // selection into one wire has no obvious meaning (which node
            // would feed which?), so a group drag ending on a wire is just
            // an ordinary drop, same as landing on empty canvas.
            if let [target] = targets.as_slice() {
                splice_node_into_wire(graph, history, &wires, *target, rect, canvas.zoom);
            }
            for target in targets {
                canvas.drag_unsnapped.remove(&target);
            }
        }
        if node_response.clicked_by(egui::PointerButton::Primary) {
            if let Some(screen_pos) = node_response.interact_pointer_pos() {
                canvas.last_click_pos =
                    screen_to_graph(origin, canvas.pan, canvas.zoom, screen_pos);
            }
            if modifiers.shift {
                if !canvas.selected.remove(&id) {
                    canvas.selected.insert(id);
                }
            } else {
                canvas.selected.clear();
                canvas.selected.insert(id);
                // A plain click only moves the Parameters panel, not the
                // 2D/3D View's own preview — the owner's own request
                // (reversed from the original "click moves preview,
                // double-click pins Parameters" behavior), so clicking
                // around the graph to inspect/tweak params doesn't keep
                // yanking the preview away from whatever's actually being
                // looked at. Double-click pins the preview instead.
                *selected = Some(id);
            }
        }
        if node_response.double_clicked_by(egui::PointerButton::Primary) {
            canvas.selected.clear();
            canvas.selected.insert(id);
            *selected = Some(id);
            *preview = Some(id);
        }
        node_response.context_menu(|ui| {
            if ui.button("Duplicate Node").clicked() {
                duplicate_nodes(graph, history, canvas, selected, preview, id);
                ui.close();
            }
            if ui.button("Delete Node").clicked() {
                let transaction = graph.remove_node_transaction(id);
                let _ = history.do_transaction(graph, transaction);
                canvas.selected.remove(&id);
                if *selected == Some(id) {
                    *selected = None;
                }
                if *preview == Some(id) {
                    *preview = None;
                }
                ui.close();
            }
        });

        let zoom = canvas.zoom;
        let is_selected = canvas.selected.contains(&id);
        let fill = crate::theme::WORMHOLE_GREY;
        // Selection outline is Pink Dwarf, same color `theme::PINK_DWARF`
        // already uses for a Color-type port pin on this same card — safe
        // to share here since a thick full-card border and a tiny port dot
        // never read as the same affordance (unlike the duplicate dot,
        // which sits right next to a same-sized delete dot and would).
        let stroke = if is_selected {
            Stroke::new(2.0, crate::theme::PINK_DWARF)
        } else {
            Stroke::new(1.0, crate::theme::TWINKLING_GREY)
        };
        let card_radius = 4.0 * zoom;
        painter.rect_filled(rect, card_radius, fill);
        // Header accent colored by the node's category (`theme::
        // category_color`, same palette the Node Library panel uses) — top
        // corners rounded to match the card, bottom corners square so it
        // sits flush with the body fill below rather than floating as a
        // separately-rounded shape. Dimmed the same way the Node Library
        // panel's own icon badges already dim it (`main.rs`'s
        // `show_node_library`, hovered-badge alpha): at full brightness
        // this header sits directly behind the Supernova Light title text,
        // and several category hues (the lighter yellows/greens especially)
        // don't contrast well with that near-white against a full-strength
        // fill the way they do against the sidebar's own dimmed badges.
        let header_rect =
            Rect::from_min_size(rect.min, Vec2::new(rect.width(), HEADER_HEIGHT * zoom));
        let header_radius = card_radius.round() as u8;
        painter.rect_filled(
            header_rect,
            egui::CornerRadius {
                nw: header_radius,
                ne: header_radius,
                sw: 0,
                se: 0,
            },
            crate::theme::category_color(def.category).gamma_multiply(0.55),
        );
        painter.rect_stroke(rect, card_radius, stroke, StrokeKind::Inside);
        painter.line_segment(
            [
                Pos2::new(rect.min.x, rect.min.y + HEADER_HEIGHT * zoom),
                Pos2::new(rect.max.x, rect.min.y + HEADER_HEIGHT * zoom),
            ],
            Stroke::new(1.0, crate::theme::TWINKLING_GREY.gamma_multiply(0.5)),
        );
        let title_font = egui::FontId::proportional(14.0 * zoom);
        let right_reserved = HEADER_TEXT_RIGHT_RESERVED
            + if *perf_overlay_enabled {
                PERF_BADGE_RESERVED
            } else {
                0.0
            };
        let title_max_width = (NODE_WIDTH - HEADER_TEXT_LEFT - right_reserved) * zoom;
        let title_text = node_title_text(
            &painter,
            def.display_name,
            title_font.clone(),
            title_max_width,
        );
        painter.text(
            Pos2::new(
                rect.min.x + HEADER_TEXT_LEFT * zoom,
                rect.min.y + HEADER_HEIGHT * zoom / 2.0,
            ),
            egui::Align2::LEFT_CENTER,
            title_text,
            title_font,
            crate::theme::SUPERNOVA_LIGHT,
        );

        // Delete indicator: same idea as the preview dot below, for the
        // right-click-menu "Delete Node" action — a one-click way to remove
        // a single node without having to select it first. Rightmost of the
        // three header dots.
        let delete_dot_pos = Pos2::new(
            rect.max.x - PREVIEW_DOT_MARGIN * zoom,
            rect.min.y + HEADER_HEIGHT * zoom / 2.0,
        );
        painter.circle_filled(
            delete_dot_pos,
            PREVIEW_DOT_RADIUS * zoom,
            Color32::from_rgb(210, 70, 70),
        );
        let delete_dot_id = ui.id().with(("mc_delete_dot", id));
        let delete_dot_rect =
            Rect::from_center_size(delete_dot_pos, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
        if ui
            .interact(delete_dot_rect, delete_dot_id, Sense::click())
            .on_hover_text("Delete node")
            .clicked()
        {
            let transaction = graph.remove_node_transaction(id);
            let _ = history.do_transaction(graph, transaction);
            canvas.selected.remove(&id);
            if *selected == Some(id) {
                *selected = None;
            }
            if *preview == Some(id) {
                *preview = None;
            }
        }

        // Duplicate indicator: same idea as the delete dot, for the
        // right-click-menu "Duplicate Node" action's one-click equivalent —
        // the counterpart to delete, so it sits right next to it. Middle of
        // the three header dots. A custom blue, like delete's custom red —
        // not `theme::PINK_DWARF`, which already means "Color port" on this
        // same node card and would read ambiguously reused here.
        let duplicate_dot_pos = delete_dot_pos - Vec2::new(DOT_SPACING * zoom, 0.0);
        painter.circle_filled(
            duplicate_dot_pos,
            PREVIEW_DOT_RADIUS * zoom,
            Color32::from_rgb(90, 150, 210),
        );
        let duplicate_dot_id = ui.id().with(("mc_duplicate_dot", id));
        let duplicate_dot_rect =
            Rect::from_center_size(duplicate_dot_pos, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
        if ui
            .interact(duplicate_dot_rect, duplicate_dot_id, Sense::click())
            .on_hover_text("Duplicate node")
            .clicked()
        {
            duplicate_nodes(graph, history, canvas, selected, preview, id);
        }

        // Preview-target indicator: a small dot in the header, rather than a
        // full "Preview" button in the body — frees up the whole body for
        // the thumbnail, and sidesteps real egui widgets (which don't scale
        // with our zoom) inside the node entirely. Leftmost of the three
        // header dots.
        let preview_dot_pos = duplicate_dot_pos - Vec2::new(DOT_SPACING * zoom, 0.0);
        painter.circle_filled(
            preview_dot_pos,
            PREVIEW_DOT_RADIUS * zoom,
            if *selected == Some(id) {
                crate::theme::PINK_DWARF
            } else {
                crate::theme::TWINKLING_GREY.gamma_multiply(0.6)
            },
        );
        let preview_dot_id = ui.id().with(("mc_preview_dot", id));
        let preview_dot_rect =
            Rect::from_center_size(preview_dot_pos, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
        if ui
            .interact(preview_dot_rect, preview_dot_id, Sense::click())
            .on_hover_text("Show in Parameters / 2D View")
            .clicked()
        {
            *selected = Some(id);
            *preview = Some(id);
        }

        // Performance overlay's own cost label — the corner "speed" button
        // below toggles it. Placed left of the delete/duplicate/preview
        // dot trio, its own reserved slot (`PERF_BADGE_RESERVED`) rather
        // than folded into that row — plain text, no color-coding or
        // marker dot (the owner's own call, after an earlier version added
        // both: a flat "N.Nms" reading is enough to judge a node by, and a
        // marker dot next to the three header dots already there read as
        // one too many).
        if *perf_overlay_enabled {
            let label = match node_cost_ms.get(&id) {
                // Never measured yet — the overlay was just turned on and
                // this node hasn't re-dispatched (so hasn't been timed)
                // since, not an error state.
                Some(&ms) => format!("{ms:.1}ms"),
                None => "–".to_string(),
            };
            painter.text(
                preview_dot_pos - Vec2::new(DOT_SPACING * zoom, 0.0),
                egui::Align2::RIGHT_CENTER,
                label,
                egui::FontId::proportional(PERF_BADGE_FONT_SIZE * zoom),
                crate::theme::SUPERNOVA_LIGHT,
            );
        }

        for (i, port) in def.inputs.iter().enumerate() {
            let p = pin_pos(rect, false, i, zoom);
            draw_pin(
                ui,
                &painter,
                p,
                port.port_type,
                port.display_name,
                zoom,
                PinStyle {
                    is_output: false,
                    is_previewed: false,
                },
            );
            let pin_id = ui.id().with(("mc_pin_in", id, i));
            let pin_rect = Rect::from_center_size(p, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
            let pin_resp = ui.interact(pin_rect, pin_id, Sense::click_and_drag());
            if pin_resp.drag_started_by(egui::PointerButton::Primary) {
                // If this input is already wired, re-anchor the drag at the
                // output it came from (see `pick_up_connected_input`) so
                // that end stays fixed while this one follows the cursor to
                // a new input. An empty input just starts a fresh drag from
                // itself, looking for an output, as before.
                canvas.wire_drag = pick_up_connected_input(graph, history, id, port.name)
                    .or(Some(PinRef::Input { node: id, index: i }));
            }
        }
        let node_kind = graph.node(id).map(|n| n.kind.clone()).unwrap_or_default();
        // Which output is actually shown in the 2D View right now for this
        // node: whichever `preview_port` names, if that's still a real
        // output of this node, else the first/default one
        // `preview_texture`'s own fallback would use — so the highlight
        // below never lies about what's on screen, including before anyone
        // has clicked an output pin at all.
        // Owned (not borrowed from `preview_port`) specifically so the
        // click handler below is free to write back into `*preview_port`
        // within the same loop that reads this.
        let effective_preview_port: Option<String> = match &preview_port {
            Some((pid, name))
                if *pid == id && def.outputs.iter().any(|p| p.name == name.as_str()) =>
            {
                Some(name.clone())
            }
            _ => def.outputs.first().map(|p| p.name.to_string()),
        };
        let mut previewed_output_type = PortType::Gray;
        for (i, port) in def.outputs.iter().enumerate() {
            let p = pin_pos(rect, true, i, zoom);
            let port_type =
                mc_nodes::defs::effective_output_type(graph, id, &node_kind, port.port_type);
            let is_this_port_previewed = effective_preview_port.as_deref() == Some(port.name);
            if is_this_port_previewed {
                previewed_output_type = port_type;
            }
            // Only worth highlighting which output is active when there's
            // more than one to choose from — a single-output node's pin
            // changing color for no reason would just be visual noise.
            let is_previewed = def.outputs.len() > 1 && is_this_port_previewed;
            draw_pin(
                ui,
                &painter,
                p,
                port_type,
                port.display_name,
                zoom,
                PinStyle {
                    is_output: true,
                    is_previewed,
                },
            );
            let pin_id = ui.id().with(("mc_pin_out", id, i));
            let pin_rect = Rect::from_center_size(p, Vec2::splat(PIN_HIT_RADIUS * zoom * 2.0));
            let pin_resp = ui
                .interact(pin_rect, pin_id, Sense::click_and_drag())
                .on_hover_text(if def.outputs.len() > 1 {
                    "Drag to connect \u{2014} click to show in the 2D View"
                } else {
                    "Drag to connect"
                });
            if pin_resp.drag_started_by(egui::PointerButton::Primary) {
                canvas.wire_drag = Some(PinRef::Output { node: id, index: i });
            }
            // A plain click (not a drag) previews this specific output
            // instead of wiring it — only moves `preview` (2D View), not
            // `selected` (Parameters) — matching this pin's own tooltip,
            // which only ever promised the 2D View. Deliberately different
            // from a plain click on the node body itself (which moves
            // `selected`, not `preview` — see that click handler's own
            // doc comment): this pin's own tooltip is specific to the 2D
            // View, so it keeps doing exactly that regardless of which way
            // the node-body click/double-click roles are assigned.
            // `Sense::click_and_drag`'s own `clicked()` already only fires
            // for an actual click (no meaningful pointer movement), the
            // same way the input pins above already rely on
            // `drag_started_by` alone to mean a real drag — no extra guard
            // needed to tell the two gestures apart.
            if pin_resp.clicked() {
                *preview = Some(id);
                *preview_port = Some((id, port.name.to_string()));
            }
        }

        // Thumbnail: drawn straight onto the canvas via the painter, not a
        // real egui `Image` widget — a plain widget's layout doesn't know
        // about our zoom, which is exactly what made the old "Preview"
        // button (and, before that, this same image) overflow or get
        // clipped at low zoom.
        let port_rows = def.inputs.len().max(def.outputs.len()).max(1) as f32;
        let body_top = rect.min.y
            + (HEADER_HEIGHT + BODY_PADDING + port_rows * PORT_ROW_HEIGHT + BODY_PADDING) * zoom;
        let thumbnail_rect = Rect::from_min_size(
            Pos2::new(rect.min.x + BODY_PADDING * zoom, body_top),
            Vec2::splat(THUMBNAIL_SIZE * zoom),
        );
        // Mirrors whatever the 2D View is showing for this node (see
        // `effective_preview_port` above) rather than always the default
        // output — otherwise clicking "Mask" to preview it would leave
        // this node's own thumbnail still showing Height, which would
        // read as the click not having done anything.
        let thumbnail_key = match &effective_preview_port {
            Some(port_name) => Some(mc_nodes::registry::texture_key(id, port_name)),
            None => mc_nodes::registry::resolve_texture_key(graph, id),
        };
        if let Some(key) = thumbnail_key {
            let is_sdf = previewed_output_type == PortType::Sdf;
            // Skip the real re-encode below when nothing was (re-)evaluated
            // since last frame and this node's own thumbnail is still
            // showing the same key/sdf-ness as last time — the already-
            // registered egui texture is still correct, just needs
            // repainting (every frame, not just this fast path) at
            // `thumbnail_rect`'s current position/zoom. See
            // `thumbnail_signatures`'s own doc comment.
            let signature = (key.clone(), is_sdf);
            let cached = last_evaluated_nodes.is_empty()
                && canvas.thumbnail_signatures.get(&id) == Some(&signature);
            let texture_id = if cached {
                thumbnail_ids.get(&id).copied()
            } else {
                None
            };
            let texture_id = texture_id.or_else(|| {
                // Thumbnails are node output, which is linear — see
                // `display_encode.rs` for why this has to be a real encode
                // pass, not just handing egui the raw (linear) texture view.
                let view = crate::display_encode::encode_for_display(
                    gpu,
                    pool,
                    &key,
                    crate::display_encode::ChannelView::Rgb,
                    is_sdf,
                )?;
                canvas.thumbnail_signatures.insert(id, signature);
                Some(match thumbnail_ids.get(&id).copied() {
                    Some(tex_id) => {
                        render_state
                            .renderer
                            .write()
                            .update_egui_texture_from_wgpu_texture(
                                &gpu.device,
                                &view,
                                wgpu::FilterMode::Linear,
                                tex_id,
                            );
                        tex_id
                    }
                    None => {
                        let tex_id = render_state.renderer.write().register_native_texture(
                            &gpu.device,
                            &view,
                            wgpu::FilterMode::Linear,
                        );
                        thumbnail_ids.insert(id, tex_id);
                        tex_id
                    }
                })
            });
            if let Some(texture_id) = texture_id {
                painter.image(
                    texture_id,
                    thumbnail_rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        }
    }

    // Pass 5: in-progress wire-drag — live preview plus release handling.
    if let Some(from) = canvas.wire_drag {
        let current = ui.input(|i| i.pointer.interact_pos().or(i.pointer.hover_pos()));
        if let (Some(from_pin), Some(cur)) = (pins.iter().find(|p| p.r == from), current) {
            draw_wire(
                &painter,
                from_pin.pos,
                cur,
                canvas.zoom,
                port_color(from_pin.port_type),
            );
        }
        // Scrolls the view toward wherever the pointer is near (or past)
        // the canvas edge, so a node outside the current viewport — most
        // often one further "up" the graph than what's visible, the
        // reported case this exists for — is still reachable without
        // letting go of the wire.
        if let Some(cur) = current {
            autopan_for_wire_drag(canvas, canvas_rect, cur, ui);
        }
        if ui.input(|i| i.pointer.primary_released()) {
            if let Some(cur) = current {
                let target = pins
                    .iter()
                    .filter(|p| p.r != from)
                    .min_by(|a, b| a.pos.distance(cur).total_cmp(&b.pos.distance(cur)));
                let connected =
                    target.is_some_and(|t| t.pos.distance(cur) <= PIN_HIT_RADIUS * canvas.zoom);
                if let Some(target) = target.filter(|_| connected) {
                    if let Err(reason) = try_connect(graph, history, from, target.r) {
                        output.connect_rejected = Some(reason.message().to_string());
                    }
                } else if let Some(from_pin) = pins.iter().find(|p| p.r == from) {
                    let (from_node, from_port_index, from_is_output) = match from {
                        PinRef::Output { node, index } => (node, index, true),
                        PinRef::Input { node, index } => (node, index, false),
                    };
                    output.dropped_wire = Some(DroppedWire {
                        screen_pos: cur,
                        graph_pos: screen_to_graph(origin, canvas.pan, canvas.zoom, cur),
                        from_node,
                        from_port_index,
                        from_is_output,
                        from_port_type: from_pin.port_type,
                    });
                }
            }
            canvas.wire_drag = None;
        }
    }

    // Pass 6: background gestures — pan, box-select, click-to-clear,
    // Cmd+drag-to-rewire, right-click (wire disconnect or add-node menu).

    // Cmd+drag starting on an existing wire picks its input end back up —
    // an alternative to grabbing the (small) input pin directly, for
    // relocating a connection without having to aim precisely at the pin.
    if modifiers.command
        && canvas.wire_drag.is_none()
        && bg_response.drag_started_by(egui::PointerButton::Primary)
    {
        if let Some(start) = bg_response.interact_pointer_pos() {
            if let Some((connection, _, _)) = wires
                .iter()
                .find(|(_, a, b)| dist_to_wire(start, *a, *b, canvas.zoom) <= WIRE_HIT_DISTANCE)
            {
                let to_node = connection.to_node;
                let to_port = connection.to_port.clone();
                canvas.wire_drag = pick_up_connected_input(graph, history, to_node, &to_port);
            }
        }
    }

    if modifiers.shift {
        if bg_response.drag_started_by(egui::PointerButton::Primary) {
            canvas.box_select_start = bg_response.interact_pointer_pos();
        }
    } else if canvas.wire_drag.is_none() && bg_response.dragged_by(egui::PointerButton::Primary) {
        canvas.pan -= bg_response.drag_delta() / canvas.zoom;
    }

    if let Some(start) = canvas.box_select_start {
        if let Some(cur) = bg_response.interact_pointer_pos() {
            let select_rect = Rect::from_two_pos(start, cur);
            let accent = crate::theme::SUPERNOVA_LIGHT;
            painter.rect(
                select_rect,
                0.0,
                Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 40),
                Stroke::new(1.0, accent),
                StrokeKind::Inside,
            );
        }
        if bg_response.drag_stopped_by(egui::PointerButton::Primary) {
            if let Some(cur) = bg_response.interact_pointer_pos() {
                let select_rect = Rect::from_two_pos(start, cur);
                for (&id, &rect) in &node_rects {
                    if select_rect.intersects(rect) {
                        canvas.selected.insert(id);
                    }
                }
            }
            canvas.box_select_start = None;
        }
    }

    // A plain left click (no drag, no modifiers) disconnects a wire if it
    // landed on one, or clears the selection otherwise. Previously this was
    // on right-click, which turned out not to register reliably; left-click
    // reuses the exact same `clicked_by(Primary)` path that click-to-clear
    // already used and was confirmed working, and reserves right-click
    // purely for the add-node menu so it can never compete with the
    // Cmd+drag-to-rewire gesture below.
    if bg_response.clicked_by(egui::PointerButton::Primary) && !modifiers.shift {
        if let Some(screen_pos) = bg_response.interact_pointer_pos() {
            canvas.last_click_pos = screen_to_graph(origin, canvas.pan, canvas.zoom, screen_pos);
        }
        let hit_wire = bg_response.interact_pointer_pos().and_then(|screen_pos| {
            wires.iter().find(|(_, a, b)| {
                dist_to_wire(screen_pos, *a, *b, canvas.zoom) <= WIRE_HIT_DISTANCE
            })
        });
        if let Some((connection, _, _)) = hit_wire {
            let _ = history.do_transaction(
                graph,
                vec![mc_core::Command::Disconnect(connection.clone())],
            );
        } else {
            canvas.selected.clear();
            canvas.selected_annotation = None;
        }
    }

    if bg_response.secondary_clicked() {
        if let Some(screen_pos) = bg_response.interact_pointer_pos() {
            let graph_pos = screen_to_graph(origin, canvas.pan, canvas.zoom, screen_pos);
            canvas.last_click_pos = graph_pos;
            output.open_add_menu_at = Some((screen_pos, graph_pos));
        }
    }

    // Top-right corner controls — Auto Layout above Snap to Grid, per the
    // owner's own request to move both out of the main menu bar and group
    // them here. Added last, so (per this file's own z-order-by-add-order
    // rule, see Pass 4's own comment) they sit on top of and intercept
    // clicks over the plain background-pan gesture underneath them.
    let auto_layout_rect = Rect::from_min_size(
        Pos2::new(
            canvas_rect.right() - CORNER_BUTTON_MARGIN - CORNER_BUTTON_SIZE,
            canvas_rect.top() + CORNER_BUTTON_MARGIN,
        ),
        Vec2::splat(CORNER_BUTTON_SIZE),
    );
    let snap_rect = Rect::from_min_size(
        Pos2::new(
            auto_layout_rect.min.x,
            auto_layout_rect.max.y + CORNER_BUTTON_GAP,
        ),
        Vec2::splat(CORNER_BUTTON_SIZE),
    );
    if corner_icon_button(
        ui,
        auto_layout_rect,
        "canvas_auto_layout_button",
        AUTO_LAYOUT_GLYPH,
        false,
        if canvas.selected.is_empty() {
            "Auto Layout — rearranges every node into a clean left-to-right layout."
        } else {
            "Auto Layout — rearranges the selected nodes into a clean left-to-right layout."
        },
    )
    .clicked()
    {
        auto_layout(graph, history, &canvas.selected);
    }
    if corner_icon_button(
        ui,
        snap_rect,
        "canvas_snap_to_grid_button",
        SNAP_TO_GRID_GLYPH,
        canvas.snap_to_grid,
        "Snap to Grid — dragged nodes lock to the grid while this is on.",
    )
    .clicked()
    {
        canvas.snap_to_grid = !canvas.snap_to_grid;
    }

    // Add Comment / Add Frame — grouped below Auto Layout/Snap to Grid,
    // same corner, per the owner's own request.
    let add_comment_rect = Rect::from_min_size(
        Pos2::new(snap_rect.min.x, snap_rect.max.y + CORNER_BUTTON_GAP),
        Vec2::splat(CORNER_BUTTON_SIZE),
    );
    let add_frame_rect = Rect::from_min_size(
        Pos2::new(
            add_comment_rect.min.x,
            add_comment_rect.max.y + CORNER_BUTTON_GAP,
        ),
        Vec2::splat(CORNER_BUTTON_SIZE),
    );
    // Performance overlay toggle — grouped right under Add Frame, same
    // corner, the owner's own request after an earlier version of this
    // put it in its own separate sidebar rail instead.
    let perf_overlay_rect = Rect::from_min_size(
        Pos2::new(
            add_frame_rect.min.x,
            add_frame_rect.max.y + CORNER_BUTTON_GAP,
        ),
        Vec2::splat(CORNER_BUTTON_SIZE),
    );
    if corner_icon_button(
        ui,
        add_comment_rect,
        "canvas_add_comment_button",
        ADD_COMMENT_GLYPH,
        false,
        "Add Comment — drops a text note on the canvas; edit it in the Parameters panel.",
    )
    .clicked()
    {
        let pos = canvas.last_click_pos();
        let mut comment = mc_core::Comment::new("New comment");
        comment.position = [pos.x, pos.y];
        let (id, transaction) = graph.add_comment_transaction(comment);
        if history.do_transaction(graph, transaction).is_ok() {
            canvas.selected.clear();
            canvas.selected_annotation = Some(AnnotationSelection::Comment(id));
            *selected = None;
        }
    }
    if corner_icon_button(
        ui,
        add_frame_rect,
        "canvas_add_frame_button",
        ADD_FRAME_GLYPH,
        false,
        if canvas.selected.is_empty() {
            "Add Frame — drops an empty grouping box on the canvas; edit it in the Parameters panel."
        } else {
            "Add Frame — sizes a titled box to fit the selected nodes; edit it in the Parameters panel."
        },
    )
    .clicked()
    {
        let frame = if canvas.selected.is_empty() {
            let pos = canvas.last_click_pos();
            mc_core::Frame::new("New Frame", [pos.x, pos.y], mc_core::DEFAULT_FRAME_SIZE)
        } else {
            frame_around_selection(graph, &canvas.selected)
        };
        let (id, transaction) = graph.add_frame_transaction(frame);
        if history.do_transaction(graph, transaction).is_ok() {
            canvas.selected_annotation = Some(AnnotationSelection::Frame(id));
            *selected = None;
        }
    }
    if corner_icon_button(
        ui,
        perf_overlay_rect,
        "canvas_perf_overlay_button",
        SPEED_GLYPH,
        *perf_overlay_enabled,
        "Show each node's measured GPU cost. Turning this on makes every node that \
         re-evaluates while it's on wait for the GPU to actually finish, instead of just \
         queuing its work — a small, deliberate slowdown in exchange for a real reading, not \
         a free background measurement.",
    )
    .clicked()
    {
        *perf_overlay_enabled = !*perf_overlay_enabled;
        if *perf_overlay_enabled {
            // Without this, every already-evaluated (not dirty) node would
            // sit at "never measured" until something happened to mark it
            // dirty on its own — turning the overlay on should show a real
            // reading for the whole graph right away, not just whatever
            // gets edited from this point forward.
            graph.mark_all_dirty();
        }
    }

    output
}

/// Bundles `draw_pin`'s two booleans (which otherwise pushed it past
/// clippy's argument-count limit) — "is this an output pin" and "is this
/// specific output the one currently shown in the 2D View" are different
/// questions (the latter only ever true for one of possibly several
/// outputs), so this stays two fields rather than collapsing into one.
struct PinStyle {
    is_output: bool,
    is_previewed: bool,
}

fn draw_pin(
    ui: &egui::Ui,
    painter: &egui::Painter,
    pos: Pos2,
    port_type: PortType,
    label: &str,
    zoom: f32,
    style: PinStyle,
) {
    let PinStyle {
        is_output,
        is_previewed,
    } = style;
    painter.circle_filled(pos, PIN_RADIUS * zoom, port_color(port_type));
    let offset = (PIN_RADIUS + 4.0) * zoom;
    let (anchor, text_pos) = if is_output {
        (egui::Align2::RIGHT_CENTER, pos - Vec2::new(offset, 0.0))
    } else {
        (egui::Align2::LEFT_CENTER, pos + Vec2::new(offset, 0.0))
    };
    // Same accent color the node-header preview dot uses for "this is what
    // the 2D View is currently showing" — only ever true for an output
    // pin, and only when that node has more than one (so a normal
    // single-output node's label never changes color for no visible
    // reason).
    let color = if is_previewed {
        crate::theme::SUPERNOVA_LIGHT
    } else {
        ui.visuals().text_color()
    };
    painter.text(
        text_pos,
        anchor,
        label,
        egui::FontId::proportional(12.0 * zoom),
        color,
    );
}

/// Connects a freshly-added `new_node` back to the pin a [`DroppedWire`]
/// started from, picking the first port on `new_node` compatible with it —
/// the "drag a wire out, drop it on nothing, pick a node, it's already
/// wired up" gesture.
pub fn connect_dropped_wire(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    dropped: &DroppedWire,
    new_node: NodeId,
) {
    let Some(new_def) = node_def_of(graph, new_node) else {
        return;
    };
    if dropped.from_is_output {
        let Some(from_port) = node_def_of(graph, dropped.from_node)
            .and_then(|d| d.outputs.get(dropped.from_port_index))
        else {
            return;
        };
        let Some(to_port) = new_def
            .inputs
            .iter()
            .find(|p| mc_nodes::defs::ports_compatible(dropped.from_port_type, p.port_type))
        else {
            return;
        };
        if let Ok(tx) =
            graph.connect_transaction(dropped.from_node, from_port.name, new_node, to_port.name)
        {
            let _ = history.do_transaction(graph, tx);
        }
    } else {
        let Some(to_port) = node_def_of(graph, dropped.from_node)
            .and_then(|d| d.inputs.get(dropped.from_port_index))
        else {
            return;
        };
        let Some(from_port) = new_def
            .outputs
            .iter()
            .find(|p| mc_nodes::defs::ports_compatible(p.port_type, dropped.from_port_type))
        else {
            return;
        };
        if let Ok(tx) =
            graph.connect_transaction(new_node, from_port.name, dropped.from_node, to_port.name)
        {
            let _ = history.do_transaction(graph, tx);
        }
    }
}

/// `connection`'s own effective from/to port types — the from side follows
/// the same `effective_output_type` Gray/Color resolution `pins` (Pass 2
/// above) already uses for every output pin, not just the raw declared
/// type, so this agrees with what the wire's own color on screen actually
/// promises. `None` if either end's node/port has since gone missing
/// (shouldn't happen for a connection that's still in `graph`, but this
/// runs a frame after it was last confirmed to exist).
fn connection_port_types(
    graph: &mc_core::Graph,
    connection: &mc_core::Connection,
) -> Option<(PortType, PortType)> {
    let from_kind = graph.node(connection.from_node)?.kind.clone();
    let from_port = node_def_of(graph, connection.from_node)?
        .outputs
        .iter()
        .find(|p| p.name == connection.from_port)?;
    let from_type = mc_nodes::defs::effective_output_type(
        graph,
        connection.from_node,
        &from_kind,
        from_port.port_type,
    );
    let to_type = node_def_of(graph, connection.to_node)?
        .inputs
        .iter()
        .find(|p| p.name == connection.to_port)?
        .port_type;
    Some((from_type, to_type))
}

/// If `node` was just dropped on top of an existing wire (not one of its
/// own — `node` can't already be either endpoint, both to avoid a trivial
/// self-loop and because a node being dragged is never one of its own
/// wire's endpoints anyway), splices it in: the original connection is
/// replaced by two new ones, `from -> node` and `node -> to`, picking
/// `node`'s own first input/output port compatible with the wire's
/// existing types — same rule `connect_dropped_wire` already uses for a
/// wire dropped on empty canvas, reused here for the same "automatically
/// wire up whatever's actually compatible" reasoning. One undo step, not
/// two: both `connect_transaction` calls below are built from the same
/// unmodified graph snapshot (neither has been applied yet when the second
/// is built), so if the new node already participates elsewhere in the
/// graph in a way that would make inserting it here a cycle, the *whole*
/// splice is abandoned — no half-wired state ever reaches `history`. A
/// no-op, same as a drop that didn't land on any wire, if `node` has no
/// input or output port compatible with the wire it landed on.
fn splice_node_into_wire(
    graph: &mut mc_core::Graph,
    history: &mut mc_core::History,
    wires: &[(mc_core::Connection, Pos2, Pos2)],
    node: NodeId,
    node_rect: Rect,
    zoom: f32,
) {
    let Some(def) = node_def_of(graph, node) else {
        return;
    };
    let Some((connection, _, _)) = wires.iter().find(|(connection, from, to)| {
        connection.from_node != node
            && connection.to_node != node
            && wire_points(*from, *to, zoom)
                .iter()
                .any(|p| node_rect.contains(*p))
    }) else {
        return;
    };
    let Some((from_type, to_type)) = connection_port_types(graph, connection) else {
        return;
    };
    let Some(in_port) = def
        .inputs
        .iter()
        .find(|p| mc_nodes::defs::ports_compatible(from_type, p.port_type))
    else {
        return;
    };
    let Some(out_port) = def
        .outputs
        .iter()
        .find(|p| mc_nodes::defs::ports_compatible(p.port_type, to_type))
    else {
        return;
    };

    let (from_node, from_port, to_node, to_port) = (
        connection.from_node,
        connection.from_port.clone(),
        connection.to_node,
        connection.to_port.clone(),
    );
    let Ok(mut transaction) = graph.connect_transaction(from_node, &from_port, node, in_port.name)
    else {
        return;
    };
    // `to`'s own input already has the old `from -> to` connection on it —
    // `connect_transaction` itself generates the `Disconnect` for whatever
    // currently occupies that port (see its own doc comment), so the old
    // wire's removal doesn't need to be spelled out separately here.
    let Ok(connect_out) = graph.connect_transaction(node, out_port.name, to_node, &to_port) else {
        return;
    };
    transaction.extend(connect_out);
    let _ = history.do_transaction(graph, transaction);
}

#[cfg(test)]
mod tests {
    use super::{
        apply_drag_delta, auto_layout, copy_selection_to_clipboard, duplicate_nodes,
        frame_around_selection, paste_from_clipboard, splice_node_into_wire, try_connect,
        wire_drag_autopan_velocity, CanvasState, PinRef, DUPLICATE_OFFSET, FRAME_MIN_SIZE,
        GRID_BASE_SPACING, WIRE_DRAG_AUTOPAN_MAX_SPEED,
    };
    use egui::{Pos2, Rect, Vec2};
    use mc_core::{Node, NodeId, ParamValue};
    use std::collections::HashSet;

    /// Regression test for a reported bug: with Snap to Grid on, dragging a
    /// node slowly made it stop moving entirely partway through the drag.
    /// Root cause was re-rounding the node's own already-snapped stored
    /// position every frame — each small per-frame delta just rounded
    /// straight back to the same cell forever, since nowhere kept the
    /// sub-grid precision needed to ever cross into the next one. This
    /// simulates exactly that: many frames of a 2-unit delta (far smaller
    /// than the 24-unit grid's own 12-unit rounding threshold), each frame
    /// feeding the *unsnapped* total back in as the next frame's base —
    /// what `canvas.rs`'s own drag handler does — and asserts real distance
    /// is still covered by the end, not stuck at the starting cell.
    #[test]
    fn snap_to_grid_drag_makes_progress_across_many_small_frames_instead_of_sticking() {
        let mut unsnapped = [0.0f32, 0.0];
        let mut last_output = [0.0f32, 0.0];
        for _ in 0..24 {
            let (out, raw) = apply_drag_delta(unsnapped, Vec2::new(2.0, 0.0), true);
            unsnapped = raw;
            last_output = out;
        }
        assert!(
            last_output[0] > 0.0,
            "24 frames of a 2-unit drag (48 units total) should cross at least one 24-unit \
             grid cell, not stick at the starting position: got {last_output:?}"
        );
    }

    #[test]
    fn snap_to_grid_rounds_to_the_nearest_grid_multiple() {
        let (out, _) = apply_drag_delta([0.0, 0.0], Vec2::new(GRID_BASE_SPACING * 2.4, 0.0), true);
        assert_eq!(out[0], GRID_BASE_SPACING * 2.0);
    }

    #[test]
    fn snap_to_grid_off_passes_the_delta_through_unrounded() {
        let (out, raw) = apply_drag_delta([10.0, 10.0], Vec2::new(3.0, -1.0), false);
        assert_eq!(out, [13.0, 9.0]);
        assert_eq!(raw, [13.0, 9.0]);
    }

    #[test]
    fn autopan_is_zero_well_inside_the_canvas() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let v = wire_drag_autopan_velocity(rect, Pos2::new(400.0, 300.0));
        assert_eq!(v, Vec2::ZERO);
    }

    #[test]
    fn autopan_scrolls_toward_the_top_edge() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let v = wire_drag_autopan_velocity(rect, Pos2::new(400.0, 5.0));
        assert!(v.y < 0.0, "near the top edge should pan toward negative y");
        assert_eq!(v.x, 0.0);
    }

    #[test]
    fn autopan_scrolls_toward_the_bottom_edge() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let v = wire_drag_autopan_velocity(rect, Pos2::new(400.0, 598.0));
        assert!(
            v.y > 0.0,
            "near the bottom edge should pan toward positive y"
        );
    }

    #[test]
    fn autopan_scrolls_toward_the_left_edge() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let v = wire_drag_autopan_velocity(rect, Pos2::new(2.0, 300.0));
        assert!(v.x < 0.0, "near the left edge should pan toward negative x");
    }

    #[test]
    fn autopan_clamps_to_the_max_speed_well_past_the_edge() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let v = wire_drag_autopan_velocity(rect, Pos2::new(400.0, -500.0));
        assert_eq!(v.y, -WIRE_DRAG_AUTOPAN_MAX_SPEED);
    }

    #[test]
    fn autopan_in_a_corner_moves_diagonally() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let v = wire_drag_autopan_velocity(rect, Pos2::new(3.0, 3.0));
        assert!(v.x < 0.0, "near the left edge should also pan in x");
        assert!(v.y < 0.0, "near the top edge should also pan in y");
    }

    #[test]
    fn try_connect_links_a_compatible_output_to_an_input() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let source = graph.add_node(Node::new("perlin_noise"));
        let filter = graph.add_node(Node::new("levels"));

        let result = try_connect(
            &mut graph,
            &mut history,
            PinRef::Output {
                node: source,
                index: 0,
            },
            PinRef::Input {
                node: filter,
                index: 0,
            },
        );

        assert!(result.is_ok());
        assert_eq!(graph.connections().len(), 1);
    }

    #[test]
    fn try_connect_rejects_two_pins_of_the_same_direction() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("voronoi"));

        let result = try_connect(
            &mut graph,
            &mut history,
            PinRef::Output { node: a, index: 0 },
            PinRef::Output { node: b, index: 0 },
        );

        assert!(result.is_err());
        assert!(graph.connections().is_empty());
    }

    #[test]
    fn try_connect_rejects_a_connection_that_would_create_a_cycle() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let a = graph.add_node(Node::new("levels"));
        let b = graph.add_node(Node::new("levels"));
        graph
            .connect(a, "output", b, "input")
            .expect("a -> b is valid on its own");

        // Dragging a wire from b's output "up" into a's input would feed
        // b back into its own upstream — exactly the reported "can't
        // connect to a node higher up" symptom; the graph is right to
        // refuse it, but this test is really about `try_connect` giving a
        // real reason back rather than nothing.
        let result = try_connect(
            &mut graph,
            &mut history,
            PinRef::Output { node: b, index: 0 },
            PinRef::Input { node: a, index: 0 },
        );

        assert!(result.is_err());
        assert_eq!(
            graph.connections().len(),
            1,
            "the original a -> b connection should be untouched"
        );
    }

    #[test]
    fn auto_layout_places_downstream_nodes_in_a_later_column() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let source = graph.add_node(Node::new("perlin_noise"));
        let filter = graph.add_node(Node::new("levels"));
        graph
            .connect(source, "output", filter, "input")
            .expect("valid connection");
        // Deliberately start both nodes at the same spot, so a passing
        // test can only mean the layout actually separated them by
        // dependency, not that they happened to already be far apart.
        let _ = graph.set_node_position(source, [500.0, 500.0]);
        let _ = graph.set_node_position(filter, [500.0, 500.0]);

        auto_layout(&mut graph, &mut history, &HashSet::new());

        let source_x = graph.node(source).unwrap().position[0];
        let filter_x = graph.node(filter).unwrap().position[0];
        assert!(
            filter_x > source_x,
            "levels (downstream) should land in a later column than perlin_noise (its source): \
             source_x={source_x}, filter_x={filter_x}"
        );
    }

    #[test]
    fn auto_layout_is_a_single_undoable_step() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("voronoi"));
        let _ = graph.set_node_position(a, [12.0, 34.0]);
        let _ = graph.set_node_position(b, [56.0, 78.0]);

        auto_layout(&mut graph, &mut history, &HashSet::new());
        assert_ne!(graph.node(a).unwrap().position, [12.0, 34.0]);

        assert!(history.can_undo());
        history.undo(&mut graph);
        assert_eq!(graph.node(a).unwrap().position, [12.0, 34.0]);
        assert_eq!(graph.node(b).unwrap().position, [56.0, 78.0]);
    }

    #[test]
    fn auto_layout_on_an_empty_graph_does_nothing() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        auto_layout(&mut graph, &mut history, &HashSet::new());
        assert!(!history.can_undo());
    }

    #[test]
    fn auto_layout_with_a_selection_only_moves_the_selected_nodes() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("voronoi"));
        let c = graph.add_node(Node::new("levels"));
        graph
            .connect(a, "output", c, "input")
            .expect("valid connection");
        let _ = graph.set_node_position(a, [500.0, 500.0]);
        let _ = graph.set_node_position(b, [999.0, 999.0]);
        let _ = graph.set_node_position(c, [500.0, 500.0]);

        let selection: HashSet<NodeId> = [a, c].into_iter().collect();
        auto_layout(&mut graph, &mut history, &selection);

        // `b` wasn't selected — left exactly where it was.
        assert_eq!(graph.node(b).unwrap().position, [999.0, 999.0]);
        // `a`/`c` were — `c` (downstream of `a`) lands in a later column.
        let a_x = graph.node(a).unwrap().position[0];
        let c_x = graph.node(c).unwrap().position[0];
        assert!(c_x > a_x, "a_x={a_x} c_x={c_x}");
    }

    #[test]
    fn auto_layout_with_a_selection_anchors_at_the_groups_own_top_left() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("voronoi"));
        // Far from the graph origin, so a passing test can only mean the
        // group was anchored at its own position, not the origin.
        let _ = graph.set_node_position(a, [2000.0, 3000.0]);
        let _ = graph.set_node_position(b, [2100.0, 3100.0]);

        let selection: HashSet<NodeId> = [a, b].into_iter().collect();
        auto_layout(&mut graph, &mut history, &selection);

        let a_pos = graph.node(a).unwrap().position;
        let b_pos = graph.node(b).unwrap().position;
        assert!(
            a_pos[0] >= 2000.0 && a_pos[1] >= 3000.0,
            "expected the group to stay near its own top-left, got a_pos={a_pos:?}"
        );
        assert!(
            b_pos[0] >= 2000.0 && b_pos[1] >= 3000.0,
            "expected the group to stay near its own top-left, got b_pos={b_pos:?}"
        );
    }

    #[test]
    fn duplicate_copies_kind_and_params_with_an_offset_position_and_selects_the_copy() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let mut canvas = CanvasState::default();
        let mut selected = None;
        let mut preview = None;

        let original =
            graph.add_node(Node::new("perlin_noise").with_param("seed", ParamValue::Seed(7)));
        let _ = graph.set_node_position(original, [100.0, 200.0]);

        duplicate_nodes(
            &mut graph,
            &mut history,
            &mut canvas,
            &mut selected,
            &mut preview,
            original,
        );

        assert_eq!(
            graph.nodes().count(),
            2,
            "duplicate should add exactly one new node"
        );
        let copy_id = *canvas
            .selected
            .iter()
            .next()
            .expect("the copy should be selected");
        assert_ne!(copy_id, original, "the copy must be a distinct node");
        assert_eq!(selected, Some(copy_id));

        let copy = graph.node(copy_id).expect("copy exists");
        assert_eq!(copy.kind, "perlin_noise");
        assert_eq!(copy.params.get("seed"), Some(&ParamValue::Seed(7)));
        assert_eq!(
            copy.position,
            [100.0 + DUPLICATE_OFFSET, 200.0 + DUPLICATE_OFFSET]
        );
    }

    #[test]
    fn duplicating_part_of_a_multi_selection_duplicates_the_whole_selection() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let mut canvas = CanvasState::default();
        let mut selected = None;
        let mut preview = None;

        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("levels"));
        graph.connect(a, "output", b, "input").unwrap();
        canvas.selected.insert(a);
        canvas.selected.insert(b);

        duplicate_nodes(
            &mut graph,
            &mut history,
            &mut canvas,
            &mut selected,
            &mut preview,
            a,
        );

        assert_eq!(
            graph.nodes().count(),
            4,
            "both selected nodes should be duplicated"
        );
        assert_eq!(
            canvas.selected.len(),
            2,
            "the two copies should become the new selection"
        );
        assert!(
            !canvas.selected.contains(&a) && !canvas.selected.contains(&b),
            "the new selection should be the copies, not the originals"
        );

        let mut copies = canvas.selected.iter().copied();
        let (copy1, copy2) = (copies.next().unwrap(), copies.next().unwrap());
        let wired = graph.upstream(copy1, "input") == Some(copy2)
            || graph.upstream(copy2, "input") == Some(copy1);
        assert!(
            wired,
            "the internal a->b connection must survive into the duplicated copies"
        );
        assert_eq!(
            graph.upstream(b, "input"),
            Some(a),
            "the originals' own connection must be untouched"
        );
    }

    #[test]
    fn copy_then_paste_round_trips_nodes_and_internal_wiring() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let mut canvas = CanvasState::default();
        let mut selected = None;
        let mut preview = None;
        let ctx = egui::Context::default();

        let a = graph.add_node(Node::new("perlin_noise").with_param("seed", ParamValue::Seed(9)));
        let b = graph.add_node(Node::new("levels"));
        let _ = graph.set_node_position(a, [10.0, 20.0]);
        let _ = graph.set_node_position(b, [110.0, 20.0]);
        graph.connect(a, "output", b, "input").unwrap();
        canvas.selected.insert(a);
        canvas.selected.insert(b);

        ctx.begin_pass(egui::RawInput::default());
        copy_selection_to_clipboard(&ctx, &graph, &mut canvas);
        let mut copy_output = ctx.end_pass();
        copy_output.textures_delta.clear();
        let copied = copy_output
            .platform_output
            .commands
            .into_iter()
            .find_map(|c| match c {
                egui::OutputCommand::CopyText(text) => Some(text),
                _ => None,
            })
            .expect("copy with a non-empty selection must write to the clipboard");

        let mut raw_input = egui::RawInput::default();
        raw_input.events.push(egui::Event::Paste(copied));
        ctx.begin_pass(raw_input);
        paste_from_clipboard(
            &ctx,
            &mut graph,
            &mut history,
            &mut canvas,
            &mut selected,
            &mut preview,
        );
        // `end_pass`'s `TexturesDelta` (default-font upload on the first
        // pass) must be acknowledged or it panics on drop — this test never
        // renders anything, so there's nothing to actually apply it to.
        ctx.end_pass().textures_delta.clear();

        assert_eq!(
            graph.nodes().count(),
            4,
            "paste should add new copies, not reuse the originals"
        );
        assert_eq!(canvas.selected.len(), 2, "the pasted nodes become selected");
        assert!(!canvas.selected.contains(&a) && !canvas.selected.contains(&b));

        let mut pasted = canvas.selected.iter().copied();
        let (p1, p2) = (pasted.next().unwrap(), pasted.next().unwrap());
        let (pasted_a, pasted_b) = if graph.node(p1).unwrap().kind == "perlin_noise" {
            (p1, p2)
        } else {
            (p2, p1)
        };
        assert_eq!(
            graph.node(pasted_a).unwrap().params.get("seed"),
            Some(&ParamValue::Seed(9)),
            "params must survive the round trip"
        );
        assert_eq!(
            graph.upstream(pasted_b, "input"),
            Some(pasted_a),
            "the internal connection must survive the round trip"
        );
        assert_eq!(
            graph.upstream(b, "input"),
            Some(a),
            "the original selection and its wiring must be untouched"
        );
    }

    #[test]
    fn pasting_garbage_clipboard_text_does_nothing() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let mut canvas = CanvasState::default();
        let mut selected = None;
        let mut preview = None;
        let ctx = egui::Context::default();

        let mut raw_input = egui::RawInput::default();
        raw_input
            .events
            .push(egui::Event::Paste("not json at all".to_string()));
        ctx.begin_pass(raw_input);
        paste_from_clipboard(
            &ctx,
            &mut graph,
            &mut history,
            &mut canvas,
            &mut selected,
            &mut preview,
        );
        ctx.end_pass().textures_delta.clear();

        assert_eq!(
            graph.nodes().count(),
            0,
            "garbage clipboard text must not create any nodes"
        );
        assert!(selected.is_none());
        assert!(canvas.selected.is_empty());
    }

    #[test]
    fn frame_around_selection_encloses_every_selected_node() {
        let mut graph = mc_core::Graph::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("voronoi"));
        let _ = graph.set_node_position(a, [0.0, 0.0]);
        let _ = graph.set_node_position(b, [500.0, 300.0]);
        let mut selection = HashSet::new();
        selection.insert(a);
        selection.insert(b);

        let frame = frame_around_selection(&graph, &selection);

        let def_a = mc_nodes::defs::node_def("perlin_noise").unwrap();
        let def_b = mc_nodes::defs::node_def("voronoi").unwrap();
        let size_a = super::node_size(def_a, 1.0);
        let size_b = super::node_size(def_b, 1.0);
        let min_x = 0.0;
        let min_y = 0.0;
        let max_x = 500.0 + size_b.x;
        let max_y = (size_a.y).max(300.0 + size_b.y);

        assert!(
            frame.position[0] <= min_x,
            "frame must start at/before the leftmost node"
        );
        assert!(
            frame.position[1] <= min_y,
            "frame must start at/before the topmost node"
        );
        assert!(
            frame.position[0] + frame.size[0] >= max_x,
            "frame must extend past the rightmost node's own right edge"
        );
        assert!(
            frame.position[1] + frame.size[1] >= max_y,
            "frame must extend past the lowest node's own bottom edge"
        );
    }

    #[test]
    fn frame_around_selection_falls_back_to_default_size_for_an_empty_or_stale_selection() {
        let graph = mc_core::Graph::new();
        let frame = frame_around_selection(&graph, &HashSet::new());
        assert_eq!(frame.size, mc_core::DEFAULT_FRAME_SIZE);
        assert!(frame.size[0] >= FRAME_MIN_SIZE[0] && frame.size[1] >= FRAME_MIN_SIZE[1]);
    }

    /// A straight wire from `(0,0)` to `(200,0)` at `zoom = 1.0` — its own
    /// bezier control points land on the same line the endpoints do (see
    /// `wire_points`'s own math), so every sampled point sits at `y = 0`,
    /// `x` somewhere in `[0, 200]`. A rect straddling `x = 100` always
    /// contains at least the midpoint sample.
    fn straight_wire(connection: mc_core::Connection) -> (mc_core::Connection, Pos2, Pos2) {
        (connection, Pos2::new(0.0, 0.0), Pos2::new(200.0, 0.0))
    }

    #[test]
    fn splice_inserts_a_dropped_node_in_between_and_is_one_undo_step() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let source = graph.add_node(Node::new("perlin_noise"));
        let sink = graph.add_node(Node::new("levels"));
        graph.connect(source, "output", sink, "input").unwrap();
        let dropped = graph.add_node(Node::new("invert"));

        let connection = graph.connections()[0].clone();
        let wires = vec![straight_wire(connection)];
        let rect = Rect::from_min_max(Pos2::new(80.0, -20.0), Pos2::new(120.0, 20.0));

        splice_node_into_wire(&mut graph, &mut history, &wires, dropped, rect, 1.0);

        assert_eq!(
            graph.upstream(sink, "input"),
            Some(dropped),
            "sink's input should now come from the dropped node"
        );
        assert_eq!(
            graph.upstream(dropped, "input"),
            Some(source),
            "the dropped node's own input should come from the original source"
        );

        assert!(history.undo(&mut graph), "splice must be undoable");
        assert_eq!(
            graph.upstream(sink, "input"),
            Some(source),
            "undo must restore the original direct connection in one step"
        );
        assert_eq!(graph.upstream(dropped, "input"), None);
    }

    #[test]
    fn splice_does_nothing_without_a_compatible_port() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let source = graph.add_node(Node::new("perlin_noise"));
        let sink = graph.add_node(Node::new("levels"));
        graph.connect(source, "output", sink, "input").unwrap();
        // A generator has no input port at all, so it can never be spliced
        // into an existing wire.
        let dropped = graph.add_node(Node::new("voronoi"));

        let connection = graph.connections()[0].clone();
        let wires = vec![straight_wire(connection)];
        let rect = Rect::from_min_max(Pos2::new(80.0, -20.0), Pos2::new(120.0, 20.0));

        splice_node_into_wire(&mut graph, &mut history, &wires, dropped, rect, 1.0);

        assert_eq!(
            graph.connections().len(),
            1,
            "the original wire must be untouched"
        );
        assert_eq!(graph.upstream(sink, "input"), Some(source));
        assert!(
            !history.can_undo(),
            "a no-op splice must not record a history step"
        );
    }

    #[test]
    fn splice_does_nothing_when_the_dropped_node_is_already_an_endpoint() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let source = graph.add_node(Node::new("perlin_noise"));
        let sink = graph.add_node(Node::new("levels"));
        graph.connect(source, "output", sink, "input").unwrap();

        let connection = graph.connections()[0].clone();
        let wires = vec![straight_wire(connection)];
        let rect = Rect::from_min_max(Pos2::new(80.0, -20.0), Pos2::new(120.0, 20.0));

        // Dropping either endpoint back "onto its own wire" must be a no-op,
        // not a self-loop attempt.
        splice_node_into_wire(&mut graph, &mut history, &wires, source, rect, 1.0);
        splice_node_into_wire(&mut graph, &mut history, &wires, sink, rect, 1.0);

        assert_eq!(graph.connections().len(), 1);
        assert_eq!(graph.upstream(sink, "input"), Some(source));
    }

    #[test]
    fn splice_does_nothing_when_the_rect_misses_the_wire() {
        let mut graph = mc_core::Graph::new();
        let mut history = mc_core::History::new();
        let source = graph.add_node(Node::new("perlin_noise"));
        let sink = graph.add_node(Node::new("levels"));
        graph.connect(source, "output", sink, "input").unwrap();
        let dropped = graph.add_node(Node::new("invert"));

        let connection = graph.connections()[0].clone();
        let wires = vec![straight_wire(connection)];
        // Nowhere near the wire, which runs along y = 0 between x = 0 and
        // x = 200.
        let rect = Rect::from_min_max(Pos2::new(1000.0, 1000.0), Pos2::new(1040.0, 1040.0));

        splice_node_into_wire(&mut graph, &mut history, &wires, dropped, rect, 1.0);

        assert_eq!(graph.connections().len(), 1);
        assert_eq!(graph.upstream(sink, "input"), Some(source));
    }
}

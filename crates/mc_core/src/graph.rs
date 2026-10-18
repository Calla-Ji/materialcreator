// SPDX-License-Identifier: MIT OR Apache-2.0
//! The graph model: nodes, connections, settings, evaluation order, and the
//! dirty/cache tracking that decides what needs re-evaluating.
//!
//! This crate is deliberately GPU- and UI-free (CLAUDE.md §5) so it can be
//! unit-tested on its own; actually running a node's shader lives in
//! `mc_nodes`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::comment::Comment;
use crate::connection::Connection;
use crate::frame::Frame;
use crate::id::{CommentId, FrameId, NodeId};
use crate::node::Node;
use crate::param::ParamValue;

/// Current `.mcgraph` JSON schema version. Bump this and add a migration
/// whenever `Graph`'s shape changes in a way old files can't parse directly
/// — never break loading of existing example graphs.
pub const FORMAT_VERSION: u32 = 1;

/// The working resolution range CLAUDE.md documents (256 up to the optional
/// 8192 "if VRAM allows" tier) — `from_json` clamps a loaded file's
/// resolution into this range rather than trusting it outright. There's no
/// UI control for this field today (only the export dialog's own, separate
/// 256-4096 list is user-facing), so the only way it ever takes an extreme
/// value is a hand-edited or corrupted `.mcgraph` — an unclamped absurd
/// value (e.g. 100000) would ask the GPU to allocate a texture it can't.
pub const MIN_RESOLUTION: u32 = 256;
pub const MAX_RESOLUTION: u32 = 8192;

/// Normal map convention: Blender expects OpenGL (Y+), Unreal expects
/// DirectX (Y-). A graph-wide default, overridable at export (CLAUDE.md §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalFormat {
    OpenGl,
    DirectX,
}

/// Matches `Viewport3d::new()`'s own real-world default
/// (`crates/mc_app/src/viewport3d.rs`) — kept in sync by hand since
/// `mc_core` can't depend on the app crate.
fn default_height_scale() -> f32 {
    0.15
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSettings {
    pub resolution: u32,
    pub seed: u32,
    pub normal_format: NormalFormat,
    /// How strongly Height displaces the 3D viewport's mesh. Saved per
    /// graph (rather than a single global UI setting) so each example's
    /// own relief — a few millimeters of brushed-metal scratch versus a
    /// whole hard-surface panel bevel — gets a sensible starting point
    /// instead of one slider value fighting every graph at once.
    #[serde(default = "default_height_scale")]
    pub height_scale: f32,
}

impl Default for GraphSettings {
    fn default() -> Self {
        Self {
            resolution: 1024,
            seed: 0,
            normal_format: NormalFormat::OpenGl,
            height_scale: default_height_scale(),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("node {0:?} does not exist")]
    NoSuchNode(NodeId),
    #[error("connecting {from:?}:{from_port} to {to:?}:{to_port} would create a cycle")]
    WouldCreateCycle {
        from: NodeId,
        from_port: String,
        to: NodeId,
        to_port: String,
    },
    #[error("graph contains a cycle")]
    Cycle,
    #[error("comment {0:?} does not exist")]
    NoSuchComment(CommentId),
    #[error("frame {0:?} does not exist")]
    NoSuchFrame(FrameId),
}

/// Failure loading a `.mcgraph` file.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("failed to read graph file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse graph JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "this file was saved by a newer version of MaterialCreator (format {found}, this build \
         only supports up to {supported}) \u{2014} update the app to open it"
    )]
    UnsupportedFormatVersion { found: u32, supported: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub format_version: u32,
    #[serde(default)]
    pub settings: GraphSettings,
    pub(crate) nodes: BTreeMap<NodeId, Node>,
    pub(crate) connections: Vec<Connection>,
    /// Free-floating text annotations — see `Comment`'s own doc comment.
    #[serde(default)]
    pub(crate) comments: BTreeMap<CommentId, Comment>,
    /// Visual node-grouping rectangles — see `Frame`'s own doc comment.
    #[serde(default)]
    pub(crate) frames: BTreeMap<FrameId, Frame>,

    /// Derived from `nodes`/`comments`/`frames` after load, never persisted
    /// (see `from_json`).
    #[serde(skip, default)]
    pub(crate) next_id: u64,
    /// Which nodes have been explicitly invalidated (new, edited, or
    /// downstream of something that was) since they were last evaluated.
    #[serde(skip, default)]
    pub(crate) dirty: HashSet<NodeId>,
    /// The evaluation hash each node had the last time it was evaluated.
    #[serde(skip, default)]
    pub(crate) evaluated_hashes: HashMap<NodeId, u64>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            settings: GraphSettings::default(),
            nodes: BTreeMap::new(),
            connections: Vec::new(),
            comments: BTreeMap::new(),
            frames: BTreeMap::new(),
            next_id: 0,
            dirty: HashSet::new(),
            evaluated_hashes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.next_id);
        self.do_command(crate::command::Command::AddNode { id, node })
            .expect("adding a node at a fresh id never fails");
        id
    }

    /// Remove a node and any connections touching it (as a single undoable
    /// transaction — see `remove_node_transaction`). Anything downstream of
    /// it loses an input and is marked dirty (it'll fall back to that
    /// input's default).
    pub fn remove_node(&mut self, id: NodeId) {
        for command in self.remove_node_transaction(id) {
            let _ = self.do_command(command);
        }
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes.iter().map(|(&id, node)| (id, node))
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    pub fn comment(&self, id: CommentId) -> Option<&Comment> {
        self.comments.get(&id)
    }

    pub fn comments(&self) -> impl Iterator<Item = (CommentId, &Comment)> {
        self.comments.iter().map(|(&id, comment)| (id, comment))
    }

    /// Plain direct mutation, not undo-tracked — same convention as
    /// `set_node_position` (dragging a comment around is "purely cosmetic,
    /// ignored by evaluation" the same way a node's position is).
    pub fn set_comment_position(
        &mut self,
        id: CommentId,
        position: [f32; 2],
    ) -> Result<(), GraphError> {
        let comment = self
            .comments
            .get_mut(&id)
            .ok_or(GraphError::NoSuchComment(id))?;
        comment.position = position;
        Ok(())
    }

    /// For UI callers that want undo tracking (the Parameters panel) —
    /// mirrors `set_param`'s own "mutate, discard the returned inverse"
    /// shape; such a caller reads the old value itself first and records
    /// its own inverse (see `mc_app::params_panel::commit_once`'s doc
    /// comment for why).
    pub fn set_comment_text(&mut self, id: CommentId, text: String) -> Result<(), GraphError> {
        self.do_command(crate::command::Command::SetCommentText { id, text })
            .map(|_| ())
    }

    pub fn frame(&self, id: FrameId) -> Option<&Frame> {
        self.frames.get(&id)
    }

    pub fn frames(&self) -> impl Iterator<Item = (FrameId, &Frame)> {
        self.frames.iter().map(|(&id, frame)| (id, frame))
    }

    /// Plain direct mutation, not undo-tracked — see
    /// `set_comment_position`'s own doc comment.
    pub fn set_frame_position(
        &mut self,
        id: FrameId,
        position: [f32; 2],
    ) -> Result<(), GraphError> {
        let frame = self
            .frames
            .get_mut(&id)
            .ok_or(GraphError::NoSuchFrame(id))?;
        frame.position = position;
        Ok(())
    }

    /// Plain direct mutation, not undo-tracked — a resize drag updates this
    /// every frame the same way a position drag does.
    pub fn set_frame_size(&mut self, id: FrameId, size: [f32; 2]) -> Result<(), GraphError> {
        let frame = self
            .frames
            .get_mut(&id)
            .ok_or(GraphError::NoSuchFrame(id))?;
        frame.size = size;
        Ok(())
    }

    /// For UI callers that want undo tracking — see `set_comment_text`'s own
    /// doc comment.
    pub fn set_frame_title(&mut self, id: FrameId, title: String) -> Result<(), GraphError> {
        self.do_command(crate::command::Command::SetFrameTitle { id, title })
            .map(|_| ())
    }

    /// For UI callers that want undo tracking — see `set_comment_text`'s own
    /// doc comment.
    pub fn set_frame_color(&mut self, id: FrameId, color: [f32; 4]) -> Result<(), GraphError> {
        self.do_command(crate::command::Command::SetFrameColor { id, color })
            .map(|_| ())
    }

    /// Update a node's editor canvas position. Purely cosmetic — it doesn't
    /// affect evaluation, so unlike other edits it isn't undo-tracked (a
    /// dragged node snapping back on undo would be surprising).
    pub fn set_node_position(&mut self, id: NodeId, position: [f32; 2]) -> Result<(), GraphError> {
        let node = self.nodes.get_mut(&id).ok_or(GraphError::NoSuchNode(id))?;
        node.position = position;
        Ok(())
    }

    pub fn set_param(
        &mut self,
        id: NodeId,
        name: &str,
        value: ParamValue,
    ) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&id) {
            return Err(GraphError::NoSuchNode(id));
        }
        self.do_command(crate::command::Command::SetParam {
            id,
            name: name.to_string(),
            value,
        })
        .map(|_| ())
    }

    /// Writes `value` into node `id`'s own params directly — no undo
    /// history, no [`mark_dirty_downstream`](Self::mark_dirty_downstream)
    /// cascade, unlike [`set_param`](Self::set_param). For a node's own
    /// evaluation to record a *computed* result into its own params
    /// (e.g. Image Trace's traced path, read back out by a downstream
    /// Path node's `Reference` input — [`mc_core::PortType::Path`]'s own
    /// doc comment), not a user edit.
    ///
    /// This has to skip the dirty cascade specifically *because* it only
    /// ever runs from inside a node's own evaluation: calling the normal,
    /// cascading `set_param` there would re-mark this node's entire
    /// downstream chain dirty on every single evaluation pass forever
    /// (even once nothing has actually changed), since the write always
    /// happens as a side effect of evaluating this node again — defeating
    /// [`needs_evaluation`](Self::needs_evaluation)'s caching for
    /// everything downstream. Whatever actually triggered this
    /// re-evaluation (a param edit, an upstream change) already cascaded
    /// the dirty flag correctly through the normal edit path, so nothing
    /// downstream is missed.
    pub fn set_computed_param(
        &mut self,
        id: NodeId,
        name: &str,
        value: ParamValue,
    ) -> Result<(), GraphError> {
        let node = self.nodes.get_mut(&id).ok_or(GraphError::NoSuchNode(id))?;
        node.params.insert(name.to_string(), value);
        Ok(())
    }

    /// Connect `from_node:from_port` to `to_node:to_port`, replacing any
    /// existing connection into `to_node:to_port` (each input takes one
    /// connection; outputs can fan out). Rejected without changing the graph
    /// if it would create a cycle.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_port: &str,
        to_node: NodeId,
        to_port: &str,
    ) -> Result<(), GraphError> {
        for command in self.connect_transaction(from_node, from_port, to_node, to_port)? {
            self.do_command(command)?;
        }
        Ok(())
    }

    pub fn disconnect(&mut self, to_node: NodeId, to_port: &str) {
        if let Some(existing) = self
            .connections
            .iter()
            .find(|c| c.to_node == to_node && c.to_port == to_port)
            .cloned()
        {
            let _ = self.do_command(crate::command::Command::Disconnect(existing));
        }
    }

    /// The node feeding `(to_node, to_port)`, if any.
    pub fn upstream(&self, to_node: NodeId, to_port: &str) -> Option<NodeId> {
        self.connections
            .iter()
            .find(|c| c.to_node == to_node && c.to_port == to_port)
            .map(|c| c.from_node)
    }

    /// Same as [`upstream`](Self::upstream), but also returns *which* of the
    /// upstream node's own output ports feeds this input — needed once a
    /// node can have more than one output (e.g. Thread's Height/Ply
    /// ID/Fiber/Mask), where "the upstream node" alone isn't enough to know
    /// which of its textures to read.
    pub fn upstream_from(&self, to_node: NodeId, to_port: &str) -> Option<(NodeId, String)> {
        self.connections
            .iter()
            .find(|c| c.to_node == to_node && c.to_port == to_port)
            .map(|c| (c.from_node, c.from_port.clone()))
    }

    /// A deterministic evaluation order (every node after everything it
    /// depends on). Graphs built through `connect` can never contain a
    /// cycle, but a hand-edited or corrupted `.mcgraph` file in principle
    /// could, hence this stays fallible.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, GraphError> {
        Self::topo_order(&self.nodes, &self.connections).ok_or(GraphError::Cycle)
    }

    pub(crate) fn topo_order(
        nodes: &BTreeMap<NodeId, Node>,
        connections: &[Connection],
    ) -> Option<Vec<NodeId>> {
        let mut in_degree: BTreeMap<NodeId, usize> = nodes.keys().map(|&id| (id, 0)).collect();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for connection in connections {
            *in_degree.get_mut(&connection.to_node)? += 1;
            adjacency
                .entry(connection.from_node)
                .or_default()
                .push(connection.to_node);
        }

        // A `BTreeSet` frontier (rather than whatever order a `HashSet`
        // happens to give) makes the result deterministic for a given graph.
        let mut ready: BTreeSet<NodeId> = in_degree
            .iter()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::with_capacity(nodes.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);
            if let Some(children) = adjacency.get(&id) {
                for &child in children {
                    let degree = in_degree.get_mut(&child)?;
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(child);
                    }
                }
            }
        }

        (order.len() == nodes.len()).then_some(order)
    }

    /// Every node reachable by following connections forward from `start`,
    /// not including `start` itself.
    pub(crate) fn downstream_of(&self, start: NodeId) -> HashSet<NodeId> {
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for connection in &self.connections {
            adjacency
                .entry(connection.from_node)
                .or_default()
                .push(connection.to_node);
        }

        let mut seen = HashSet::new();
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            if seen.insert(id) {
                if let Some(children) = adjacency.get(&id) {
                    stack.extend(children.iter().copied());
                }
            }
        }
        seen.remove(&start);
        seen
    }

    pub(crate) fn mark_dirty_downstream(&mut self, start: NodeId) {
        self.dirty.insert(start);
        self.dirty.extend(self.downstream_of(start));
    }

    /// A hash of everything that determines this node's output: its kind,
    /// params, the graph's resolution/seed, and (recursively) its inputs'
    /// own hashes. Two evaluations with the same hash would produce the same
    /// pixels.
    pub fn evaluation_hash(&self, id: NodeId) -> Option<u64> {
        let mut memo = HashMap::new();
        self.evaluation_hash_memo(id, &mut memo)
    }

    fn evaluation_hash_memo(&self, id: NodeId, memo: &mut HashMap<NodeId, u64>) -> Option<u64> {
        if let Some(&hash) = memo.get(&id) {
            return Some(hash);
        }
        let node = self.nodes.get(&id)?;

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        node.kind.hash(&mut hasher);
        for (name, value) in &node.params {
            name.hash(&mut hasher);
            value.hash(&mut hasher);
        }
        self.settings.resolution.hash(&mut hasher);
        self.settings.seed.hash(&mut hasher);

        let mut inputs: Vec<&Connection> = self
            .connections
            .iter()
            .filter(|c| c.to_node == id)
            .collect();
        inputs.sort_by(|a, b| a.to_port.cmp(&b.to_port));
        for connection in inputs {
            connection.to_port.hash(&mut hasher);
            let input_hash = self
                .evaluation_hash_memo(connection.from_node, memo)
                .unwrap_or(0);
            input_hash.hash(&mut hasher);
        }

        let hash = hasher.finish();
        memo.insert(id, hash);
        Some(hash)
    }

    /// Whether `id` needs (re-)evaluating: it's explicitly dirty, has never
    /// been evaluated, or its evaluation hash changed since it last was.
    pub fn needs_evaluation(&self, id: NodeId) -> bool {
        if self.dirty.contains(&id) {
            return true;
        }
        match (self.evaluated_hashes.get(&id), self.evaluation_hash(id)) {
            (Some(&cached), Some(current)) => cached != current,
            _ => true,
        }
    }

    /// Record that `id` has been evaluated at its current hash and clear its
    /// dirty flag — the next `needs_evaluation` call is a cache hit unless
    /// something changes first.
    pub fn mark_evaluated(&mut self, id: NodeId) {
        if let Some(hash) = self.evaluation_hash(id) {
            self.evaluated_hashes.insert(id, hash);
        }
        self.dirty.remove(&id);
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// `next_id` is deliberately not persisted — it's derived from the
    /// loaded nodes' ids here, rather than trusted from the file, so a
    /// hand-edited `.mcgraph` can't desync it from reality. Likewise,
    /// `format_version` and `settings.resolution` aren't trusted outright
    /// either: a file from a newer app build is refused outright (its shape
    /// may not even match what this build expects), and resolution is
    /// clamped into `MIN_RESOLUTION..=MAX_RESOLUTION` rather than handed
    /// straight to the GPU.
    pub fn from_json(json: &str) -> Result<Self, LoadError> {
        let mut graph: Graph = serde_json::from_str(json)?;
        if graph.format_version > FORMAT_VERSION {
            return Err(LoadError::UnsupportedFormatVersion {
                found: graph.format_version,
                supported: FORMAT_VERSION,
            });
        }
        graph.next_id = graph
            .nodes
            .keys()
            .map(|id| id.0 + 1)
            .chain(graph.comments.keys().map(|id| id.0 + 1))
            .chain(graph.frames.keys().map(|id| id.0 + 1))
            .max()
            .unwrap_or(0);
        graph.settings.resolution = graph
            .settings
            .resolution
            .clamp(MIN_RESOLUTION, MAX_RESOLUTION);
        graph.settings.height_scale = graph.settings.height_scale.clamp(0.0, 0.5);
        Ok(graph)
    }

    pub fn save_to_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(
            path,
            self.to_json().expect("Graph always serializes to JSON"),
        )
    }

    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, LoadError> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json(&text)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_graph() -> (Graph, NodeId, NodeId) {
        let mut graph = Graph::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("levels"));
        graph.connect(a, "output", b, "input").unwrap();
        (graph, a, b)
    }

    #[test]
    fn connect_rejects_cycle() {
        let (mut graph, a, b) = linear_graph();
        let err = graph.connect(b, "output", a, "input").unwrap_err();
        assert_eq!(
            err,
            GraphError::WouldCreateCycle {
                from: b,
                from_port: "output".to_string(),
                to: a,
                to_port: "input".to_string(),
            }
        );
        // Rejected connection must not have been applied.
        assert_eq!(graph.connections().len(), 1);
    }

    #[test]
    fn topological_order_respects_dependencies() {
        let (graph, a, b) = linear_graph();
        let order = graph.topological_order().unwrap();
        let pos_a = order.iter().position(|&id| id == a).unwrap();
        let pos_b = order.iter().position(|&id| id == b).unwrap();
        assert!(pos_a < pos_b, "producer must come before consumer");
    }

    #[test]
    fn connecting_to_an_occupied_input_replaces_it() {
        let mut graph = Graph::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("perlin_noise"));
        let c = graph.add_node(Node::new("levels"));
        graph.connect(a, "output", c, "input").unwrap();
        graph.connect(b, "output", c, "input").unwrap();
        assert_eq!(graph.connections().len(), 1);
        assert_eq!(graph.upstream(c, "input"), Some(b));
    }

    #[test]
    fn cache_hit_after_mark_evaluated() {
        let mut graph = Graph::new();
        let id = graph.add_node(Node::new("perlin_noise").with_param("seed", ParamValue::Seed(1)));

        assert!(graph.needs_evaluation(id), "never-evaluated node is dirty");
        graph.mark_evaluated(id);
        assert!(
            !graph.needs_evaluation(id),
            "cache hit right after evaluating"
        );

        graph.set_param(id, "seed", ParamValue::Seed(2)).unwrap();
        assert!(
            graph.needs_evaluation(id),
            "changed param must invalidate the cache"
        );

        graph.mark_evaluated(id);
        assert!(
            !graph.needs_evaluation(id),
            "cache hit again after re-evaluating"
        );
    }

    #[test]
    fn changing_an_upstream_param_dirties_downstream_nodes() {
        let (mut graph, a, b) = linear_graph();
        graph.mark_evaluated(a);
        graph.mark_evaluated(b);

        graph.set_param(a, "seed", ParamValue::Seed(99)).unwrap();
        assert!(graph.needs_evaluation(a));
        assert!(
            graph.needs_evaluation(b),
            "downstream node must be dirtied when its input changes"
        );
    }

    #[test]
    fn save_load_round_trip() {
        let mut original = Graph::new();
        let a = original.add_node(
            Node::new("perlin_noise")
                .with_param("scale", ParamValue::Int(4))
                .with_param("seed", ParamValue::Seed(7)),
        );
        let b = original.add_node(Node::new("levels"));
        original.connect(a, "output", b, "input").unwrap();
        original.settings.resolution = 2048;

        let json = original.to_json().unwrap();
        let loaded = Graph::from_json(&json).unwrap();

        assert_eq!(loaded.format_version, original.format_version);
        assert_eq!(loaded.settings.resolution, original.settings.resolution);
        assert_eq!(loaded.connections(), original.connections());
        for (id, node) in original.nodes() {
            let loaded_node = loaded.node(id).expect("node survives round trip");
            assert_eq!(loaded_node.kind, node.kind);
            assert_eq!(loaded_node.params, node.params);
        }

        // A freshly loaded graph must be able to add a node without
        // colliding with an id from the file.
        let mut loaded = loaded;
        let c = loaded.add_node(Node::new("output_base_color"));
        assert!(c.0 > a.0 && c.0 > b.0);
    }

    #[test]
    fn comments_and_frames_survive_a_save_load_round_trip() {
        let mut original = Graph::new();
        let (comment_id, tx) = original.add_comment_transaction(Comment::new("hello"));
        original.do_command(tx.into_iter().next().unwrap()).unwrap();
        original
            .set_comment_position(comment_id, [12.0, 34.0])
            .unwrap();
        let (frame_id, tx) =
            original.add_frame_transaction(Frame::new("Wood prep", [1.0, 2.0], [300.0, 200.0]));
        original.do_command(tx.into_iter().next().unwrap()).unwrap();

        let json = original.to_json().unwrap();
        let loaded = Graph::from_json(&json).unwrap();

        let comment = loaded.comment(comment_id).expect("comment survives");
        assert_eq!(comment.text, "hello");
        assert_eq!(comment.position, [12.0, 34.0]);
        let frame = loaded.frame(frame_id).expect("frame survives");
        assert_eq!(frame.title, "Wood prep");
        assert_eq!(frame.position, [1.0, 2.0]);
        assert_eq!(frame.size, [300.0, 200.0]);
    }

    /// A graph with only a comment/frame (no nodes at all) must still derive
    /// a `next_id` past whichever of them has the highest id, exactly like
    /// `save_load_round_trip` already checks for nodes — comments and
    /// frames share the same counter (see `CommentId`'s own doc comment).
    #[test]
    fn next_id_accounts_for_comments_and_frames_with_no_nodes() {
        let mut original = Graph::new();
        let (_comment_id, tx) = original.add_comment_transaction(Comment::new("note"));
        original.do_command(tx.into_iter().next().unwrap()).unwrap();
        let (frame_id, tx) =
            original.add_frame_transaction(Frame::new("Group", [0.0, 0.0], [100.0, 100.0]));
        original.do_command(tx.into_iter().next().unwrap()).unwrap();

        let json = original.to_json().unwrap();
        let mut loaded = Graph::from_json(&json).unwrap();

        let new_node = loaded.add_node(Node::new("perlin_noise"));
        assert!(new_node.0 > frame_id.0);
    }

    #[test]
    fn file_round_trip() {
        let mut graph = Graph::new();
        graph.add_node(Node::new("perlin_noise"));
        let path = std::env::temp_dir().join("mc_core_test_graph.mcgraph");

        graph.save_to_file(&path).unwrap();
        let loaded = Graph::load_from_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.nodes().count(), graph.nodes().count());
    }

    #[test]
    fn from_json_clamps_an_absurdly_high_resolution() {
        let mut graph = Graph::new();
        graph.settings.resolution = 100_000;
        let json = graph.to_json().unwrap();

        let loaded = Graph::from_json(&json).unwrap();
        assert_eq!(loaded.settings.resolution, MAX_RESOLUTION);
    }

    #[test]
    fn from_json_clamps_a_zero_resolution() {
        let mut graph = Graph::new();
        graph.settings.resolution = 0;
        let json = graph.to_json().unwrap();

        let loaded = Graph::from_json(&json).unwrap();
        assert_eq!(loaded.settings.resolution, MIN_RESOLUTION);
    }

    #[test]
    fn from_json_leaves_an_in_range_resolution_untouched() {
        let mut graph = Graph::new();
        graph.settings.resolution = 2048;
        let json = graph.to_json().unwrap();

        let loaded = Graph::from_json(&json).unwrap();
        assert_eq!(loaded.settings.resolution, 2048);
    }

    #[test]
    fn from_json_refuses_a_newer_format_version() {
        let mut graph = Graph::new();
        graph.format_version = FORMAT_VERSION + 1;
        let json = graph.to_json().unwrap();

        let err = Graph::from_json(&json).unwrap_err();
        match err {
            LoadError::UnsupportedFormatVersion { found, supported } => {
                assert_eq!(found, FORMAT_VERSION + 1);
                assert_eq!(supported, FORMAT_VERSION);
            }
            other => panic!("expected UnsupportedFormatVersion, got {other:?}"),
        }
    }

    #[test]
    fn from_json_accepts_an_older_or_equal_format_version() {
        let mut graph = Graph::new();
        graph.format_version = FORMAT_VERSION;
        let json = graph.to_json().unwrap();
        assert!(Graph::from_json(&json).is_ok());
    }
}

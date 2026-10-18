// SPDX-License-Identifier: MIT OR Apache-2.0
//! Command-based undo/redo (CLAUDE.md §9, §11).
//!
//! Every graph mutation — `add_node`, `remove_node`, `connect`, `disconnect`,
//! `set_param` — is implemented in terms of one or more [`Command`]s applied
//! through [`Graph::do_command`], which returns the command that would undo
//! it. A single user-visible action (e.g. "delete this node and its
//! connections") is a [`Transaction`]: a list of atomic commands that
//! undo/redo together as one step. [`History`] is the undo/redo stack built
//! on top of that.

use crate::comment::Comment;
use crate::connection::Connection;
use crate::frame::Frame;
use crate::graph::{Graph, GraphError};
use crate::id::{CommentId, FrameId, NodeId};
use crate::node::Node;
use crate::param::ParamValue;

/// One atomic, self-inverting graph edit.
#[derive(Debug, Clone)]
pub enum Command {
    AddNode {
        id: NodeId,
        node: Node,
    },
    RemoveNode {
        id: NodeId,
        node: Node,
    },
    Connect(Connection),
    Disconnect(Connection),
    SetParam {
        id: NodeId,
        name: String,
        value: ParamValue,
    },
    RemoveParam {
        id: NodeId,
        name: String,
    },
    /// A node's editor canvas position, undo-tracked — unlike
    /// `Graph::set_node_position` (plain drag-to-move, deliberately *not*
    /// undo-tracked; see that method's own doc comment), this exists for
    /// the one case where silently rewriting position is a bigger deal
    /// than a single accidental drag: Auto Layout, which can move every
    /// node in the graph in one click.
    MoveNode {
        id: NodeId,
        position: [f32; 2],
    },
    AddComment {
        id: CommentId,
        comment: Comment,
    },
    RemoveComment {
        id: CommentId,
        comment: Comment,
    },
    SetCommentText {
        id: CommentId,
        text: String,
    },
    AddFrame {
        id: FrameId,
        frame: Frame,
    },
    RemoveFrame {
        id: FrameId,
        frame: Frame,
    },
    SetFrameTitle {
        id: FrameId,
        title: String,
    },
    SetFrameColor {
        id: FrameId,
        color: [f32; 4],
    },
}

/// One or more [`Command`]s that undo/redo together as a single step.
pub type Transaction = Vec<Command>;

/// An undo/redo stack of [`Transaction`]s.
#[derive(Debug, Default)]
pub struct History {
    undo_stack: Vec<Transaction>,
    redo_stack: Vec<Transaction>,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply every command in `transaction` to `graph`, then push its
    /// inverse onto the undo stack as one step (clearing the redo stack, as
    /// any new edit normally does).
    pub fn do_transaction(
        &mut self,
        graph: &mut Graph,
        transaction: Transaction,
    ) -> Result<(), GraphError> {
        let inverse = apply_and_invert(graph, transaction)?;
        self.undo_stack.push(inverse);
        self.redo_stack.clear();
        Ok(())
    }

    /// Undo the most recent transaction, if any. Returns whether there was
    /// one to undo.
    pub fn undo(&mut self, graph: &mut Graph) -> bool {
        let Some(transaction) = self.undo_stack.pop() else {
            return false;
        };
        let inverse =
            apply_and_invert(graph, transaction).expect("a recorded undo must re-apply cleanly");
        self.redo_stack.push(inverse);
        true
    }

    /// Redo the most recently undone transaction, if any. Returns whether
    /// there was one to redo.
    pub fn redo(&mut self, graph: &mut Graph) -> bool {
        let Some(transaction) = self.redo_stack.pop() else {
            return false;
        };
        let inverse =
            apply_and_invert(graph, transaction).expect("a recorded redo must re-apply cleanly");
        self.undo_stack.push(inverse);
        true
    }

    /// Record a transaction's inverse directly, without applying it — for
    /// edits the caller already applied live outside the command system
    /// (e.g. a slider dragged over several frames: each frame calls
    /// `Graph::set_param` directly for an immediately-responsive UI, and
    /// only one undo step is recorded when the drag ends, with `inverse`
    /// restoring the value from *before* the drag started).
    pub fn record_applied(&mut self, inverse: Transaction) {
        self.undo_stack.push(inverse);
        self.redo_stack.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}

/// Apply each command in order, collecting their inverses in reverse order
/// (so applying the result undoes the whole transaction).
fn apply_and_invert(
    graph: &mut Graph,
    transaction: Transaction,
) -> Result<Transaction, GraphError> {
    let mut inverses = Vec::with_capacity(transaction.len());
    for command in transaction {
        inverses.push(graph.do_command(command)?);
    }
    inverses.reverse();
    Ok(inverses)
}

impl Graph {
    /// Apply one atomic command, mutating the graph, and return the command
    /// that would undo it. This is a low-level primitive: it trusts the
    /// command is well-formed (e.g. `Connect`/`Disconnect` don't themselves
    /// check for cycles or an occupied input port — see
    /// `connect_transaction`, which does, and which `connect()` is built on).
    pub fn do_command(&mut self, command: Command) -> Result<Command, GraphError> {
        match command {
            Command::AddNode { id, node } => {
                self.next_id = self.next_id.max(id.0 + 1);
                self.dirty.insert(id);
                self.nodes.insert(id, node.clone());
                Ok(Command::RemoveNode { id, node })
            }
            Command::RemoveNode { id, node: _ } => {
                let existing = self.nodes.remove(&id).ok_or(GraphError::NoSuchNode(id))?;
                self.evaluated_hashes.remove(&id);
                self.dirty.remove(&id);
                Ok(Command::AddNode { id, node: existing })
            }
            Command::Connect(connection) => {
                self.mark_dirty_downstream(connection.to_node);
                let inverse = Command::Disconnect(connection.clone());
                self.connections.push(connection);
                Ok(inverse)
            }
            Command::Disconnect(connection) => {
                self.connections.retain(|c| *c != connection);
                self.mark_dirty_downstream(connection.to_node);
                Ok(Command::Connect(connection))
            }
            Command::SetParam { id, name, value } => {
                let node = self.nodes.get_mut(&id).ok_or(GraphError::NoSuchNode(id))?;
                let previous = node.params.insert(name.clone(), value);
                self.mark_dirty_downstream(id);
                Ok(match previous {
                    Some(old) => Command::SetParam {
                        id,
                        name,
                        value: old,
                    },
                    None => Command::RemoveParam { id, name },
                })
            }
            Command::RemoveParam { id, name } => {
                let node = self.nodes.get_mut(&id).ok_or(GraphError::NoSuchNode(id))?;
                let previous = node.params.remove(&name);
                self.mark_dirty_downstream(id);
                Ok(match previous {
                    Some(old) => Command::SetParam {
                        id,
                        name,
                        value: old,
                    },
                    None => Command::RemoveParam { id, name },
                })
            }
            Command::MoveNode { id, position } => {
                let node = self.nodes.get_mut(&id).ok_or(GraphError::NoSuchNode(id))?;
                let previous = node.position;
                node.position = position;
                // Cosmetic only, same as `set_node_position` — doesn't
                // affect evaluation, so no `mark_dirty_downstream`.
                Ok(Command::MoveNode {
                    id,
                    position: previous,
                })
            }
            Command::AddComment { id, comment } => {
                self.next_id = self.next_id.max(id.0 + 1);
                self.comments.insert(id, comment.clone());
                Ok(Command::RemoveComment { id, comment })
            }
            Command::RemoveComment { id, comment: _ } => {
                let existing = self
                    .comments
                    .remove(&id)
                    .ok_or(GraphError::NoSuchComment(id))?;
                Ok(Command::AddComment {
                    id,
                    comment: existing,
                })
            }
            Command::SetCommentText { id, text } => {
                let comment = self
                    .comments
                    .get_mut(&id)
                    .ok_or(GraphError::NoSuchComment(id))?;
                let previous = std::mem::replace(&mut comment.text, text);
                Ok(Command::SetCommentText { id, text: previous })
            }
            Command::AddFrame { id, frame } => {
                self.next_id = self.next_id.max(id.0 + 1);
                self.frames.insert(id, frame.clone());
                Ok(Command::RemoveFrame { id, frame })
            }
            Command::RemoveFrame { id, frame: _ } => {
                let existing = self.frames.remove(&id).ok_or(GraphError::NoSuchFrame(id))?;
                Ok(Command::AddFrame {
                    id,
                    frame: existing,
                })
            }
            Command::SetFrameTitle { id, title } => {
                let frame = self
                    .frames
                    .get_mut(&id)
                    .ok_or(GraphError::NoSuchFrame(id))?;
                let previous = std::mem::replace(&mut frame.title, title);
                Ok(Command::SetFrameTitle {
                    id,
                    title: previous,
                })
            }
            Command::SetFrameColor { id, color } => {
                let frame = self
                    .frames
                    .get_mut(&id)
                    .ok_or(GraphError::NoSuchFrame(id))?;
                let previous = std::mem::replace(&mut frame.color, color);
                Ok(Command::SetFrameColor {
                    id,
                    color: previous,
                })
            }
        }
    }

    /// The transaction that connects `from_node:from_port` to
    /// `to_node:to_port`: a `Disconnect` of whatever currently occupies that
    /// input port (if anything), then the new `Connect`. Rejected without
    /// building anything if it would create a cycle.
    pub fn connect_transaction(
        &self,
        from_node: NodeId,
        from_port: &str,
        to_node: NodeId,
        to_port: &str,
    ) -> Result<Transaction, GraphError> {
        if !self.nodes.contains_key(&from_node) {
            return Err(GraphError::NoSuchNode(from_node));
        }
        if !self.nodes.contains_key(&to_node) {
            return Err(GraphError::NoSuchNode(to_node));
        }

        let mut trial = self.connections.clone();
        let mut transaction = Transaction::new();
        if let Some(existing) = trial
            .iter()
            .find(|c| c.to_node == to_node && c.to_port == to_port)
            .cloned()
        {
            trial.retain(|c| *c != existing);
            transaction.push(Command::Disconnect(existing));
        }

        let new_connection = Connection {
            from_node,
            from_port: from_port.to_string(),
            to_node,
            to_port: to_port.to_string(),
        };
        trial.push(new_connection.clone());

        if Self::topo_order(&self.nodes, &trial).is_none() {
            return Err(GraphError::WouldCreateCycle {
                from: from_node,
                from_port: from_port.to_string(),
                to: to_node,
                to_port: to_port.to_string(),
            });
        }

        transaction.push(Command::Connect(new_connection));
        Ok(transaction)
    }

    /// The id a node added right now would get, and the transaction that
    /// adds it — for UI flows (e.g. an add-node menu) where creating a node
    /// should be undoable like any other edit. Plain `add_node` is for
    /// non-interactive callers (tests, `mc_cli`) that don't track history.
    pub fn add_node_transaction(&self, node: Node) -> (NodeId, Transaction) {
        let id = NodeId(self.next_id);
        (id, vec![Command::AddNode { id, node }])
    }

    /// The transaction that removes `id` and every connection touching it,
    /// so undoing it restores both the node and its connections.
    pub fn remove_node_transaction(&self, id: NodeId) -> Transaction {
        let mut transaction: Transaction = self
            .connections
            .iter()
            .filter(|c| c.from_node == id || c.to_node == id)
            .cloned()
            .map(Command::Disconnect)
            .collect();
        if let Some(node) = self.nodes.get(&id) {
            transaction.push(Command::RemoveNode {
                id,
                node: node.clone(),
            });
        }
        transaction
    }

    /// Like [`Self::remove_node_transaction`], but for a whole batch of
    /// nodes as a single undoable step — e.g. a "select all, then delete"
    /// gesture. A connection between two nodes that are *both* in `ids`
    /// would otherwise show up in each node's own transaction and get
    /// disconnected twice; this dedupes by connection, not by node.
    pub fn remove_nodes_transaction(&self, ids: &[NodeId]) -> Transaction {
        let id_set: std::collections::HashSet<NodeId> = ids.iter().copied().collect();
        let mut transaction: Transaction = self
            .connections
            .iter()
            .filter(|c| id_set.contains(&c.from_node) || id_set.contains(&c.to_node))
            .cloned()
            .map(Command::Disconnect)
            .collect();
        for id in &id_set {
            if let Some(node) = self.nodes.get(id) {
                transaction.push(Command::RemoveNode {
                    id: *id,
                    node: node.clone(),
                });
            }
        }
        transaction
    }

    /// The id a comment added right now would get, and the transaction that
    /// adds it — same shape as [`Self::add_node_transaction`], for the
    /// canvas's own "Add Comment" button.
    pub fn add_comment_transaction(&self, comment: Comment) -> (CommentId, Transaction) {
        let id = CommentId(self.next_id);
        (id, vec![Command::AddComment { id, comment }])
    }

    /// The transaction that removes comment `id` — a comment has no
    /// connections of its own to clean up, unlike a node.
    pub fn remove_comment_transaction(&self, id: CommentId) -> Transaction {
        match self.comments.get(&id) {
            Some(comment) => vec![Command::RemoveComment {
                id,
                comment: comment.clone(),
            }],
            None => Transaction::new(),
        }
    }

    /// The id a frame added right now would get, and the transaction that
    /// adds it — same shape as [`Self::add_node_transaction`], for the
    /// canvas's own "Add Frame" button.
    pub fn add_frame_transaction(&self, frame: Frame) -> (FrameId, Transaction) {
        let id = FrameId(self.next_id);
        (id, vec![Command::AddFrame { id, frame }])
    }

    /// The transaction that removes frame `id` — a frame has no connections
    /// of its own, and (by design — see `Frame`'s own doc comment) doesn't
    /// own the nodes visually inside it, so nothing else needs cleaning up.
    pub fn remove_frame_transaction(&self, id: FrameId) -> Transaction {
        match self.frames.get(&id) {
            Some(frame) => vec![Command::RemoveFrame {
                id,
                frame: frame.clone(),
            }],
            None => Transaction::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param::ParamValue;

    #[test]
    fn undo_add_node_removes_it() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let id = graph.add_node(Node::new("perlin_noise"));
        // `add_node` itself isn't undo-tracked (it's a convenience wrapper);
        // record it explicitly to exercise History, mirroring how the UI
        // would call `do_transaction` for a user-visible "add node" action.
        let node = graph.node(id).unwrap().clone();
        history
            .do_transaction(&mut graph, vec![Command::RemoveNode { id, node }])
            .unwrap();
        assert!(graph.node(id).is_none());

        assert!(history.undo(&mut graph));
        assert!(graph.node(id).is_some(), "undo must bring the node back");

        assert!(history.redo(&mut graph));
        assert!(graph.node(id).is_none(), "redo must remove it again");
    }

    #[test]
    fn undo_remove_node_restores_its_connections() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("levels"));
        graph.connect(a, "output", b, "input").unwrap();

        let transaction = graph.remove_node_transaction(a);
        history.do_transaction(&mut graph, transaction).unwrap();
        assert!(graph.node(a).is_none());
        assert_eq!(graph.upstream(b, "input"), None);

        assert!(history.undo(&mut graph));
        assert!(graph.node(a).is_some());
        assert_eq!(
            graph.upstream(b, "input"),
            Some(a),
            "undo must restore the connection, not just the node"
        );
    }

    #[test]
    fn remove_nodes_transaction_deletes_a_shared_connection_exactly_once() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("levels"));
        let c = graph.add_node(Node::new("invert"));
        // a -> b -> c: the a-b connection touches two nodes that are both
        // being removed together, which is the case that would double up a
        // Disconnect command if the transaction weren't deduped by
        // connection.
        graph.connect(a, "output", b, "input").unwrap();
        graph.connect(b, "output", c, "input").unwrap();

        let transaction = graph.remove_nodes_transaction(&[a, b]);
        history
            .do_transaction(&mut graph, transaction)
            .expect("a shared connection must only be disconnected once");
        assert!(graph.node(a).is_none());
        assert!(graph.node(b).is_none());
        assert!(
            graph.node(c).is_some(),
            "c wasn't in the batch, must survive"
        );
        assert_eq!(graph.upstream(c, "input"), None);

        assert!(history.undo(&mut graph));
        assert!(graph.node(a).is_some());
        assert!(graph.node(b).is_some());
        assert_eq!(
            graph.upstream(b, "input"),
            Some(a),
            "undo must restore the a->b connection"
        );
        assert_eq!(
            graph.upstream(c, "input"),
            Some(b),
            "undo must restore the b->c connection"
        );
    }

    #[test]
    fn undo_set_param_restores_old_value() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let id = graph.add_node(Node::new("perlin_noise").with_param("seed", ParamValue::Seed(1)));

        history
            .do_transaction(
                &mut graph,
                vec![Command::SetParam {
                    id,
                    name: "seed".to_string(),
                    value: ParamValue::Seed(2),
                }],
            )
            .unwrap();
        assert_eq!(graph.node(id).unwrap().params["seed"], ParamValue::Seed(2));

        assert!(history.undo(&mut graph));
        assert_eq!(graph.node(id).unwrap().params["seed"], ParamValue::Seed(1));
    }

    #[test]
    fn new_transaction_after_undo_clears_redo_stack() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("perlin_noise"));

        let node_a = graph.node(a).unwrap().clone();
        history
            .do_transaction(
                &mut graph,
                vec![Command::RemoveNode {
                    id: a,
                    node: node_a,
                }],
            )
            .unwrap();
        history.undo(&mut graph);
        assert!(history.can_redo());

        let node_b = graph.node(b).unwrap().clone();
        history
            .do_transaction(
                &mut graph,
                vec![Command::RemoveNode {
                    id: b,
                    node: node_b,
                }],
            )
            .unwrap();
        assert!(
            !history.can_redo(),
            "a fresh action must invalidate the old redo branch"
        );
    }

    #[test]
    fn connect_transaction_still_rejects_cycles() {
        let mut graph = Graph::new();
        let a = graph.add_node(Node::new("perlin_noise"));
        let b = graph.add_node(Node::new("levels"));
        graph.connect(a, "output", b, "input").unwrap();

        assert!(graph.connect_transaction(b, "output", a, "input").is_err());
    }

    #[test]
    fn undo_add_comment_removes_it() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let (id, transaction) = graph.add_comment_transaction(Comment::new("note"));
        history.do_transaction(&mut graph, transaction).unwrap();
        assert!(graph.comment(id).is_some());

        assert!(history.undo(&mut graph));
        assert!(graph.comment(id).is_none());

        assert!(history.redo(&mut graph));
        assert!(graph.comment(id).is_some());
    }

    #[test]
    fn undo_set_comment_text_restores_old_text() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let (id, transaction) = graph.add_comment_transaction(Comment::new("before"));
        history.do_transaction(&mut graph, transaction).unwrap();

        let old = graph.comment(id).unwrap().text.clone();
        graph.set_comment_text(id, "after".to_string()).unwrap();
        history.record_applied(vec![Command::SetCommentText { id, text: old }]);
        assert_eq!(graph.comment(id).unwrap().text, "after");

        assert!(history.undo(&mut graph));
        assert_eq!(graph.comment(id).unwrap().text, "before");
    }

    #[test]
    fn undo_remove_frame_restores_it() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let (id, transaction) =
            graph.add_frame_transaction(Frame::new("Group", [0.0, 0.0], [100.0, 100.0]));
        history.do_transaction(&mut graph, transaction).unwrap();

        let remove = graph.remove_frame_transaction(id);
        history.do_transaction(&mut graph, remove).unwrap();
        assert!(graph.frame(id).is_none());

        assert!(history.undo(&mut graph));
        let restored = graph.frame(id).expect("undo must bring the frame back");
        assert_eq!(restored.title, "Group");
    }

    #[test]
    fn undo_set_frame_color_restores_old_color() {
        let mut graph = Graph::new();
        let mut history = History::new();
        let (id, transaction) =
            graph.add_frame_transaction(Frame::new("Group", [0.0, 0.0], [100.0, 100.0]));
        history.do_transaction(&mut graph, transaction).unwrap();

        let old = graph.frame(id).unwrap().color;
        let new_color = [1.0, 0.0, 0.0, 1.0];
        graph.set_frame_color(id, new_color).unwrap();
        history.record_applied(vec![Command::SetFrameColor { id, color: old }]);
        assert_eq!(graph.frame(id).unwrap().color, new_color);

        assert!(history.undo(&mut graph));
        assert_eq!(graph.frame(id).unwrap().color, old);
    }
}

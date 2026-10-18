// SPDX-License-Identifier: MIT OR Apache-2.0
//! mc_core: the graph model — nodes, connections, parameter values,
//! topological evaluation order, and dirty/cache tracking.
//!
//! Deliberately independent of the GPU and UI (CLAUDE.md §5) so it can be
//! unit-tested on its own. Actually running a node's shader lives in
//! `mc_nodes`.

mod command;
mod comment;
mod connection;
mod frame;
mod graph;
mod id;
mod node;
mod param;
mod path;
mod port;

pub use command::{Command, History, Transaction};
pub use comment::Comment;
pub use connection::Connection;
pub use frame::{Frame, DEFAULT_FRAME_COLOR, DEFAULT_FRAME_SIZE};
pub use graph::{Graph, GraphError, GraphSettings, LoadError, NormalFormat, FORMAT_VERSION};
pub use id::{CommentId, FrameId, NodeId};
pub use node::Node;
pub use param::{GradientStop, ImageRef, ParamValue};
pub use path::{Path, Point, PointKind, SubPath};
pub use port::PortType;

//! Arena-based IR transformation passes.
//!
//! These passes operate directly on `IrContext` with in-place mutations,
//! leveraging use-chains for efficient dead code detection and RAUW for
//! value remapping.

pub mod call_graph;
pub mod canonicalize;
pub mod dce;
pub mod global_dce;
pub mod inline;
pub mod scf_control_flow;
pub mod scf_to_cf;

pub use call_graph::{CallGraph, build_call_graph, recursive_functions, tarjan_scc};
pub use canonicalize::Canonicalize;
pub use dce::{DceConfig, DceResult, EliminateDeadCode, eliminate_dead_code};
pub use global_dce::{
    GlobalDce, GlobalDceConfig, GlobalDceResult, eliminate_dead_definitions,
    eliminate_dead_definitions_with_config, reachable_definitions_with_config,
};
pub use inline::{InlineConfig, InlineError, InlineFunctions, InlineResult, inline_single_call};
pub use scf_to_cf::ScfToCf;

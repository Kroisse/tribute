//! Type-Directed Name Resolution (TDNR) for the AST.
//!
//! This module resolves method calls using type information gathered during type checking.
//! It transforms `MethodCall` expressions into regular `Call` expressions by finding
//! the appropriate function based on the receiver's type.
//!
//! ## UFCS Resolution
//!
//! UFCS (Uniform Function Call Syntax) allows `x.method(y)` to be resolved as `method(x, y)`
//! where `method` is a function whose first parameter type matches the type of `x`.
//!
//! ```text
//! x.len()       // MethodCall → Call { callee: "len", args: [x] }
//! list.map(f)   // MethodCall → Call { callee: "map", args: [list, f] }
//! ```
//!
//! ## Pipeline Position
//!
//! TDNR runs after type checking, when concrete types are available:
//! ```text
//! astgen → resolve → typecheck → tdnr
//! ```

mod resolver;

pub use resolver::TdnrResolver;

use crate::ast::{Module, TypedRef};

/// Resolve method calls in a module using type information.
///
/// This is the main entry point for TDNR.
pub fn resolve_tdnr<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
) -> Module<TypedRef<'db>> {
    let resolver = TdnrResolver::new(db);
    resolver.resolve_module(module)
}

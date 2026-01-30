//! AST to TrunkIR lowering.
//!
//! This module transforms a type-checked AST (`Module<TypedRef<'db>>`) into TrunkIR.
//! Unlike tirgen which works directly from CST, this pass has access to:
//! - Resolved names (all references point to their definitions)
//! - Type information (every expression has a known type)
//!
//! ## Pipeline Position
//!
//! ```text
//! CST → AST → resolve → typecheck → tdnr → ast_to_ir → TrunkIR
//! ```
//!
//! ## Output Format
//!
//! The generated TrunkIR uses dialects:
//! - `func`: Functions, calls, returns
//! - `arith`: Arithmetic operations, constants
//! - `adt`: Struct/enum construction, field access
//! - `scf`: Structured control flow (if, case)
//! - `tribute`: Unresolved operations (for gradual migration)
//!
//! ## Status
//!
//! This module is under development. Currently provides basic structure lowering.
//! Full expression and pattern lowering will be added incrementally.

mod context;
mod lower;

use std::collections::HashMap;

use trunk_ir::dialect::core;
use trunk_ir::{PathId, Symbol};

use crate::ast::{Module, SpanMap, TypeScheme, TypedRef};

pub use context::IrLoweringCtx;
pub use lower::lower_module;

/// Lower a typed AST module to TrunkIR.
///
/// This is the main entry point for AST-to-IR transformation.
///
/// # Parameters
/// - `db`: Salsa database
/// - `module`: Type-checked AST module
/// - `span_map`: SpanMap for looking up source locations
/// - `source_uri`: URI of the source file
/// - `function_types`: Function type schemes from type checking
pub fn lower_ast_to_ir<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
    span_map: SpanMap,
    source_uri: &str,
    function_types: HashMap<Symbol, TypeScheme<'db>>,
) -> core::Module<'db> {
    let path = PathId::new(db, source_uri.to_owned());
    lower_module(db, path, span_map, module, function_types)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::PathId;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    #[test]
    fn test_context_creation() {
        let db = test_db();
        let path = PathId::new(&db, "test.trb".to_owned());
        let span_map = SpanMap::default();
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            span_map,
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
        );

        // Verify context provides expected types
        let int_ty = ctx.int_type();
        let bool_ty = ctx.bool_type();
        let unit_ty = ctx.nil_type();

        // Types should be different
        assert_ne!(int_ty, bool_ty);
        assert_ne!(int_ty, unit_ty);
        assert_ne!(bool_ty, unit_ty);
    }

    #[test]
    fn test_context_scopes() {
        let db = test_db();
        let path = PathId::new(&db, "test.trb".to_owned());
        let span_map = SpanMap::default();
        let mut ctx = IrLoweringCtx::new(
            &db,
            path,
            span_map,
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
        );

        // Initially no binding
        let local_id = crate::ast::LocalId::new(0);
        assert!(ctx.lookup(local_id).is_none());

        // After entering scope and binding, can look up
        ctx.enter_scope();
        // Note: We can't easily test bind without creating a Value,
        // so just verify scope entry/exit doesn't panic
        ctx.exit_scope();
    }

    #[test]
    fn test_context_location() {
        let db = test_db();
        let path = PathId::new(&db, "test.trb".to_owned());
        let span_map = SpanMap::default();
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            span_map,
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
        );

        // Verify location creation doesn't panic
        let node_id = crate::ast::NodeId::from_raw(42);
        let location = ctx.location(node_id);

        // Location should have the correct path
        assert_eq!(location.path, path);
    }
}

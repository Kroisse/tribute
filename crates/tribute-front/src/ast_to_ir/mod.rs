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
//! CST → AST → resolve → typecheck → tdnr → ast_to_ir → TrunkIR (arena)
//! ```
//!
//! ## Output Format
//!
//! The generated TrunkIR uses dialects:
//! - `func`: Functions, calls, returns
//! - `arith`: Arithmetic operations, constants
//! - `adt`: Struct/enum construction, field access
//! - `scf`: Structured control flow (if, case)
//! - `cont`: Continuation-based control flow (handle, shift)
//! - `closure`: Closure creation
//!
//! ## Arena IR
//!
//! This module emits arena-based IR (`IrContext` + `Module`) directly,
//! bypassing the Salsa-interned IR layer.

mod context;
mod lower;

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::rewrite::Module as IrModule;

use crate::ast::{Module, NodeId, SpanMap, Type, TypeScheme, TypedRef};

pub use context::IrLoweringCtx;
pub use lower::lower_module;

/// Lower a typed AST module to arena TrunkIR.
///
/// This is the main entry point for AST-to-IR transformation.
/// Emits directly into the given `IrContext`, returning an `Module`.
///
/// # Parameters
/// - `db`: Salsa database (for AST type access and diagnostics)
/// - `ir`: Arena IR context to emit into
/// - `module`: Type-checked AST module
/// - `span_map`: SpanMap for looking up source locations
/// - `source_uri`: URI of the source file
/// - `function_types`: Function type schemes from type checking
/// - `node_types`: Node types from type checking (NodeId → Type)
pub fn lower_ast_to_ir<'db>(
    db: &'db dyn salsa::Database,
    ir: &mut IrContext,
    module: Module<TypedRef<'db>>,
    span_map: SpanMap,
    source_uri: &str,
    function_types: HashMap<Symbol, TypeScheme<'db>>,
    node_types: HashMap<NodeId, Type<'db>>,
) -> IrModule {
    let path = ir.paths.intern(source_uri.to_owned());
    lower::lower_module(db, ir, path, span_map, module, function_types, node_types)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::context::IrContext;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    #[test]
    fn test_context_creation() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let span_map = SpanMap::default();
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            span_map,
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        // Verify context provides expected types
        let mut ir2 = IrContext::new();
        let int_ty = ctx.i32_type(&mut ir2);
        let bool_ty = ctx.bool_type(&mut ir2);
        let unit_ty = ctx.nil_type(&mut ir2);

        // Types should be different
        assert_ne!(int_ty, bool_ty);
        assert_ne!(int_ty, unit_ty);
        assert_ne!(bool_ty, unit_ty);
    }

    /// Create a dummy ValueRef for testing by creating a block with an arg.
    fn dummy_value(ir: &mut IrContext, path: trunk_ir::refs::PathRef) -> trunk_ir::refs::ValueRef {
        use trunk_ir::context::{BlockArgData, BlockData};
        use trunk_ir::types::Location;
        let location = Location::new(path, Default::default());
        let nil_ty = trunk_ir::dialect::core::nil(ir).as_type_ref();
        let block = ir.create_block(BlockData {
            location,
            args: vec![BlockArgData {
                ty: nil_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        ir.block_arg(block, 0)
    }

    #[test]
    fn test_scope_guard_cleanup() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let mut ctx = IrLoweringCtx::new(
            &db,
            path,
            SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let local_id = crate::ast::LocalId::new(0);
        let val = dummy_value(&mut ir, path);

        // Binding is not visible before scope
        assert!(ctx.lookup(local_id).is_none());

        // Binding is visible inside scope guard
        {
            let mut scope = ctx.scope();
            scope.bind(local_id, Symbol::new("x"), val);
            assert_eq!(scope.lookup(local_id), Some(val));
        }

        // Binding is cleaned up after scope guard drops
        assert!(ctx.lookup(local_id).is_none());
    }

    #[test]
    fn test_scope_guard_cleanup_on_early_return() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let mut ctx = IrLoweringCtx::new(
            &db,
            path,
            SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let local_id = crate::ast::LocalId::new(0);
        let val = dummy_value(&mut ir, path);

        // Simulate early return: helper creates scope, binds, returns early
        fn bind_and_bail(
            ctx: &mut IrLoweringCtx<'_>,
            local_id: crate::ast::LocalId,
            val: trunk_ir::refs::ValueRef,
        ) -> Option<()> {
            let mut scope = ctx.scope();
            scope.bind(local_id, Symbol::new("x"), val);
            None // early return — scope guard still drops
        }
        let _ = bind_and_bail(&mut ctx, local_id, val);

        // Binding must be cleaned up despite early return
        assert!(ctx.lookup(local_id).is_none());
    }

    #[test]
    fn test_prompt_tag_guard_cleanup() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let mut ctx = IrLoweringCtx::new(
            &db,
            path,
            SpanMap::default(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        // No active prompt tag initially
        assert_eq!(ctx.active_prompt_tag(), None);

        // Prompt tag is active inside guard
        {
            let prompt = ctx.prompt_tag_scope();
            assert_eq!(prompt.active_prompt_tag(), Some(prompt.tag()));
        }

        // Prompt tag is cleaned up after guard drops
        assert_eq!(ctx.active_prompt_tag(), None);
    }

    #[test]
    fn test_context_location() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.paths.intern("test.trb".to_owned());
        let span_map = SpanMap::default();
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            span_map,
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        // Verify location creation doesn't panic
        let node_id = crate::ast::NodeId::from_raw(42);
        let location = ctx.location(node_id);

        // Location should have the correct path
        assert_eq!(location.path, path);
    }
}

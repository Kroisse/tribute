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
//! The source-logical frontend output uses dialects:
//! - `tribute_control`: Source callables, ability operations, and handlers
//! - `arith`: Arithmetic operations, constants
//! - `adt`: Struct/enum construction, field access
//! - `scf`: Structured control flow (if, case)
//! - `list`: Source list values and observations
//!
//! The old physical `func`/continuation/closure route is kept only through the
//! explicit temporary legacy entry below until shared CPS composition moves to
//! the driver.
//!
//! ## Arena IR
//!
//! This module emits arena-based IR (`IrContext` + `Module`) directly,
//! bypassing the Salsa-interned IR layer.

mod context;
mod lower;
mod normalize;

use std::collections::HashMap;

use tribute_ir::dialect::tribute_control::OperationDeclaration;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::rewrite::Module as IrModule;

use crate::ast::{
    AbilityId, CallingConvention, Module as AstModule, NodeId, SpanMap, Type, TypeScheme, TypedRef,
};

pub use context::IrLoweringCtx;

/// The source-logical frontend boundary.
///
/// `operation_declarations` is semantic metadata, deliberately kept outside
/// textual TrunkIR.  The shared CPS conversion consumes this exact set rather
/// than attempting to reconstruct source declarations from lowered ops.
pub struct FrontendIrModule {
    pub module: IrModule,
    pub operation_declarations: Vec<OperationDeclaration>,
}

/// Policy for compiler-generated identity done continuations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Update)]
pub enum DoneContinuationPolicy {
    /// Emit a separate helper function at every use site.
    PerUse,
    /// Share one helper function across the compilation unit.
    PerCompilationUnit,
}

/// Independently selectable AST-to-IR policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Update)]
pub struct AstToIrOptions {
    pub done_continuation: DoneContinuationPolicy,
}

impl AstToIrOptions {
    pub const fn production() -> Self {
        Self {
            done_continuation: DoneContinuationPolicy::PerCompilationUnit,
        }
    }

    pub const fn baseline() -> Self {
        Self {
            done_continuation: DoneContinuationPolicy::PerUse,
        }
    }
}

impl Default for AstToIrOptions {
    fn default() -> Self {
        Self::production()
    }
}

/// A type-checked module and the metadata required to lower it to TrunkIR.
///
/// This models the boundary between the typed frontend and IR lowering as one
/// value, instead of passing parallel metadata collections positionally.
pub struct TypedModule<'db> {
    pub ast: AstModule<TypedRef<'db>>,
    pub span_map: SpanMap,
    pub function_types: HashMap<Symbol, TypeScheme<'db>>,
    pub constructor_types: HashMap<crate::ast::CtorId<'db>, TypeScheme<'db>>,
    pub node_types: HashMap<NodeId, Type<'db>>,
    /// Exact instantiated callable types for direct source calls.
    pub call_callee_types: HashMap<NodeId, Type<'db>>,
    /// Exact call sites whose callee metadata was specialized by monomorphization.
    pub specialized_call_callee_nodes: std::collections::HashSet<NodeId>,
    pub ability_conventions: HashMap<AbilityId<'db>, CallingConvention>,
    pub ability_definitions: HashMap<AbilityId<'db>, crate::typeck::AbilityInfo<'db>>,
    pub handler_operations: HashMap<NodeId, crate::typeck::InstantiatedHandlerOperation<'db>>,
    pub perform_operations: HashMap<NodeId, crate::typeck::InstantiatedPerformOperation<'db>>,
    /// Solved source-callable signatures for lambda expressions.
    pub lambda_signatures: HashMap<NodeId, crate::typeck::LambdaSignature<'db>>,
    /// Case expressions whose source coverage is known to be exhaustive.
    pub exhaustive_cases: std::collections::HashSet<NodeId>,
    pub well_known_types: crate::typeck::WellKnownTypes<'db>,
}

impl<'db> TypedModule<'db> {
    /// Lower this typed module to arena TrunkIR.
    ///
    /// This is the main entry point for AST-to-IR transformation.
    pub fn lower_to_ir(
        self,
        db: &'db dyn salsa::Database,
        ir: &mut IrContext,
        source_uri: &str,
    ) -> FrontendIrModule {
        let path = ir.paths.intern(source_uri.to_owned());
        self.lower_module(db, ir, path)
    }

    /// Temporary compatibility entry for the pre-#825 driver routing.
    ///
    /// New frontend consumers must use [`Self::lower_to_ir`].  This explicit
    /// legacy entry is configured by `AstToIrOptions`, which apply only to
    /// that physical compatibility route, until the driver composes shared CPS.
    #[doc(hidden)]
    pub fn lower_to_legacy_ir_with_options(
        self,
        db: &'db dyn salsa::Database,
        ir: &mut IrContext,
        source_uri: &str,
        options: AstToIrOptions,
    ) -> IrModule {
        let path = ir.paths.intern(source_uri.to_owned());
        self.lower_module_legacy(db, ir, path, options)
    }
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

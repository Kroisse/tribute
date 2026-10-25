//! AST to TrunkIR lowering.
//!
//! This module transforms a type-checked AST (`Module<TypedRef<'db>>`) into TrunkIR.
//! Lowering consumes:
//! - Resolved names (all references point to their definitions)
//! - Type information (every expression has a known type)
//!
//! ## Pipeline Position
//!
//! ```text
//! CST → AST → resolve → typecheck → ast_to_ir → source-logical TrunkIR
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
//! ## Arena IR
//!
//! This module emits arena-based IR (`IrContext` + `Module`) directly.

mod context;
mod lower;
mod row_binders;
pub mod runtime_items;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::sync::LazyLock;

use tribute_ir::dialect::tribute_control::CompilerIntrinsicDeclaration;
use trunk_ir::context::IrContext;
use trunk_ir::rewrite::Module as IrModule;
use trunk_ir::{Symbol, SymbolPath};

use crate::SortedMap;
use crate::ast::{AbilityId, DeclTable, NodeId, SpanMaps, Type, TypeScheme, TypedRef};

pub use context::IrLoweringCtx;

/// The source-logical frontend boundary. The module declares the ability
/// instances it names with `ability.decl`.
pub struct FrontendIrModule {
    pub module: IrModule,
    pub compiler_intrinsics: Vec<CompilerIntrinsicDeclaration>,
}

static SUPPORTED_COMPILER_INTRINSICS: LazyLock<HashSet<Symbol>> = LazyLock::new(|| {
    [
        "std::collections::List::__tribute_list_prepend_intrinsic",
        "std::Int::+",
        "std::Int::-",
        "std::Int::*",
        "std::Int::/",
        "std::Int::%",
        "std::Int::==",
        "std::Int::!=",
        "std::Int::<",
        "std::Int::<=",
        "std::Int::>",
        "std::Int::>=",
        "std::Nat::+",
        "std::Nat::-",
        "std::Nat::*",
        "std::Nat::/",
        "std::Nat::%",
        "std::Nat::==",
        "std::Nat::!=",
        "std::Nat::<",
        "std::Nat::<=",
        "std::Nat::>",
        "std::Nat::>=",
        "std::Float::+",
        "std::Float::-",
        "std::Float::*",
        "std::Float::/",
        "std::Float::==",
        "std::Float::!=",
        "std::Float::<",
        "std::Float::<=",
        "std::Float::>",
        "std::Float::>=",
        "std::__bytes_get_or_panic",
        "std::io::__tribute_io_write",
        "std::io::__tribute_io_read_line",
    ]
    .into_iter()
    .chain(tribute_ir::runtime_items::SLOTS_INTRINSICS)
    .map(Symbol::new)
    .collect()
});

fn is_supported_compiler_intrinsic(identity: &Symbol) -> bool {
    SUPPORTED_COMPILER_INTRINSICS.contains(identity)
}

/// An unsupported directive and its source declaration for diagnostics.
#[derive(Debug)]
pub struct UnsupportedCompilerIntrinsic {
    pub node: NodeId,
    pub identity: Symbol,
}

/// Register supported compiler intrinsic directives by canonical source name.
///
/// `extern "intrinsic"` is compiler-reserved: any source declaration using it
/// requests lowering under its qualified name. The fixed supported set remains
/// the validation boundary; return every unknown directive for source diagnostics.
pub fn registered_compiler_intrinsics<V>(
    module: &DeclTable<V>,
) -> Result<HashMap<NodeId, Symbol>, Vec<UnsupportedCompilerIntrinsic>> {
    let mut result = HashMap::default();
    let mut unsupported = Vec::new();
    for entry in &module.entries {
        let crate::ast::Decl::ExternFunction(function) = &entry.decl else {
            continue;
        };
        if function.abi != "intrinsic" {
            continue;
        }
        let symbol = entry.module.child(function.name.clone()).to_symbol();
        if is_supported_compiler_intrinsic(&symbol) {
            result.insert(function.id, symbol);
        } else {
            unsupported.push(UnsupportedCompilerIntrinsic {
                node: function.id,
                identity: symbol,
            });
        }
    }
    if unsupported.is_empty() {
        Ok(result)
    } else {
        Err(unsupported)
    }
}

/// A type-checked module and the metadata required to lower it to TrunkIR.
///
/// This models the boundary between the typed frontend and IR lowering as one
/// value, instead of passing parallel metadata collections positionally.
pub struct TypedModule<'db> {
    pub ast: DeclTable<TypedRef<'db>>,
    /// The span maps of the module's merged sources.
    pub span_map: SpanMaps<'db>,
    pub function_types: HashMap<SymbolPath, TypeScheme<'db>>,
    pub constructor_types: HashMap<crate::ast::CtorId<'db>, TypeScheme<'db>>,
    pub specialized_enum_variants: SortedMap<NodeId, TypeScheme<'db>>,
    pub node_types: SortedMap<NodeId, Type<'db>>,
    pub local_instances: SortedMap<NodeId, crate::typeck::LocalCallableInstance<'db>>,
    /// The instance each reference to a generic function selects.
    pub function_instances: SortedMap<NodeId, crate::typeck::FunctionInstance<'db>>,
    pub ability_definitions: HashMap<AbilityId<'db>, crate::typeck::AbilityInfo<'db>>,
    pub handler_operations: SortedMap<NodeId, crate::typeck::InstantiatedHandlerOperation<'db>>,
    pub perform_operations: SortedMap<NodeId, crate::typeck::InstantiatedPerformOperation<'db>>,
    /// Solved source-callable signatures for lambda expressions.
    pub lambda_signatures: SortedMap<NodeId, crate::typeck::LambdaSignature<'db>>,
    /// Case expressions whose source coverage is known to be exhaustive.
    pub exhaustive_cases: HashSet<NodeId>,
    /// Non-identity evidence selections of calls, resumes, and handles.
    pub evidence_plans: SortedMap<NodeId, Vec<crate::typeck::EvidenceStep<'db>>>,
    pub well_known_types: crate::typeck::WellKnownTypes<'db>,
    /// Exact intrinsic-directive declaration IDs and canonical identities.
    pub compiler_intrinsics: HashMap<NodeId, Symbol>,
    /// The runtime structs to publish as prelude types.
    pub runtime_types: runtime_items::RuntimeTypes,
    /// URIs of the other sources whose declarations were merged into `ast`.
    pub merged_sources: Vec<String>,
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
        let path = ir.intern_path(source_uri);
        self.lower_module(db, ir, path)
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
    fn compiler_intrinsic_registry_is_explicit() {
        assert!(is_supported_compiler_intrinsic(&Symbol::new(
            "std::Float::=="
        )));
        assert!(is_supported_compiler_intrinsic(&Symbol::new(
            "std::io::__tribute_io_read_line"
        )));
        assert!(!is_supported_compiler_intrinsic(&Symbol::new(
            "user_intrinsic"
        )));
    }

    #[test]
    fn test_context_creation() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb");
        let span_map = SpanMaps::default();
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            span_map,
            HashMap::default(),
            SortedMap::default(),
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
        let nil_ty = trunk_ir::dialect::core::Nil::params()
            .intern(ir)
            .as_type_ref();
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
        let path = ir.intern_path("test.trb");
        let mut ctx = IrLoweringCtx::new(
            &db,
            path,
            SpanMaps::default(),
            HashMap::default(),
            SortedMap::default(),
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
        let path = ir.intern_path("test.trb");
        let mut ctx = IrLoweringCtx::new(
            &db,
            path,
            SpanMaps::default(),
            HashMap::default(),
            SortedMap::default(),
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
    fn test_context_location() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb");
        let span_map = SpanMaps::default();
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            span_map,
            HashMap::default(),
            SortedMap::default(),
        );

        // Verify location creation doesn't panic
        let node_id = crate::ast::NodeId::from_raw(42);
        let location = ctx.location(node_id);

        // Location should have the correct path
        assert_eq!(location.path, path);
    }
}

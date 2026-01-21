//! Type-Directed Name Resolution (TDNR) pass.
//!
//! This pass resolves remaining `tribute.call` operations that couldn't be resolved
//! during initial name resolution because they require type information.
//!
//! ## UFCS Resolution
//!
//! UFCS (Uniform Function Call Syntax) transforms `x.method(y)` into `method(x, y)`,
//! then finds the function `method` in the current namespace where the first
//! parameter type matches the receiver's type.
//!
//! ```text
//! x.len()           // tribute.call(x, "len") → len(x) where first param matches x's type
//! list.map(f)       // tribute.call(list, f, "map") → map(list, f)
//! ```
//!
//! ## Pipeline Position
//!
//! TDNR runs after type inference, when concrete types are available:
//! ```text
//! stage_resolve → stage_typecheck → stage_tdnr
//! ```
//!
//! Uses `RewritePattern` + `PatternApplicator` for declarative transformation.

use tracing::trace;

use crate::resolve::{Binding, ModuleEnv, build_env};
use tribute_ir::ModulePathExt;
use tribute_ir::dialect::tribute;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func;
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewritePattern, RewriteResult, TypeConverter,
};
use trunk_ir::{Attribute, DialectOp, DialectType, Operation, Region, Symbol, Type, Value};

// =============================================================================
// TDNR Pattern
// =============================================================================

/// Pattern to resolve `tribute.call` operations using UFCS.
///
/// For `x.method(y)` (represented as `tribute.call` with receiver `x` and name `method`):
/// 1. Look up `method` in the module environment
/// 2. If it's a function and its first parameter matches `x`'s type, resolve it
/// 3. Transform to `func.call(method, x, y, ...)`
struct ResolveTributeCallPattern<'db> {
    env: ModuleEnv<'db>,
}

impl<'db> RewritePattern<'db> for ResolveTributeCallPattern<'db> {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: tribute.call
        if tribute::Call::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        }

        trace!("TDNR: found tribute.call operation");

        let operands = adaptor.operands();
        if operands.is_empty() {
            return RewriteResult::Unchanged; // No receiver
        }

        // Get method name from attributes
        let attrs = op.attributes(db);
        let Some(Attribute::Symbol(qual_name)) = attrs.get(&Symbol::new("name")) else {
            return RewriteResult::Unchanged;
        };
        trace!("TDNR: method name = {:?}", qual_name);

        let receiver = operands[0];

        // Get the receiver's type using adaptor (handles both OpResult and BlockArg)
        let Some(receiver_type) = adaptor.get_value_type(db, receiver) else {
            trace!("TDNR: could not get receiver type");
            return RewriteResult::Unchanged;
        };
        trace!(
            "TDNR: receiver_type = {}.{}",
            receiver_type.dialect(db),
            receiver_type.name(db)
        );

        // Look up the function - handle both simple and qualified names
        let Some((func_path, func_ty)) = self.lookup_function(db, *qual_name, receiver_type) else {
            trace!("TDNR: could not resolve method");
            return RewriteResult::Unchanged;
        };

        // Check if the first parameter type matches the receiver type
        let Some(func_type) = core::Func::from_type(db, func_ty) else {
            return RewriteResult::Unchanged;
        };
        let params = func_type.params(db);
        let Some(first_param) = params.first() else {
            return RewriteResult::Unchanged;
        };

        // Match receiver type with first parameter type
        if !types_compatible(receiver_type, *first_param) {
            return RewriteResult::Unchanged;
        }

        // Create func.call with the resolved function
        let location = op.location(db);
        let Some(result_ty) = op.results(db).first().copied() else {
            return RewriteResult::Unchanged;
        };
        let args: Vec<Value<'db>> = operands.iter().copied().collect();

        let new_op = func::call(db, location, args, result_ty, func_path);
        trace!("TDNR: resolved to func.call");

        RewriteResult::Replace(new_op.as_operation())
    }
}

impl<'db> ResolveTributeCallPattern<'db> {
    /// Look up the function for a method call.
    fn lookup_function(
        &self,
        db: &'db dyn salsa::Database,
        qual_name: Symbol,
        receiver_type: Type<'db>,
    ) -> Option<(Symbol, Type<'db>)> {
        if qual_name.is_simple() {
            // Simple name: first try direct lookup
            if let Some(Binding::Function { path, ty }) = self.env.lookup(qual_name.last_segment())
            {
                return Some((*path, *ty));
            }
            // Direct lookup failed - try type-based namespace lookup
            self.lookup_method_in_type_namespace(db, qual_name.last_segment(), receiver_type)
        } else {
            // Qualified name: look up by full path, fall back to namespace lookup
            let binding = self.env.lookup_path(qual_name).or_else(|| {
                let namespace = qual_name.parent_path()?;
                self.env
                    .lookup_qualified(namespace, qual_name.last_segment())
            })?;
            let Binding::Function { ty, .. } = binding else {
                return None;
            };
            Some((qual_name, *ty))
        }
    }

    /// Look up a method in the namespace of a type that matches the receiver type.
    fn lookup_method_in_type_namespace(
        &self,
        db: &'db dyn salsa::Database,
        method_name: Symbol,
        receiver_type: Type<'db>,
    ) -> Option<(Symbol, Type<'db>)> {
        trace!(
            "TDNR lookup_method_in_type_namespace: method='{}', receiver={}.{}",
            method_name,
            receiver_type.dialect(db),
            receiver_type.name(db)
        );

        for (ns_name, namespace) in self.env.namespaces_iter() {
            if let Some(Binding::Function { path, ty }) = namespace.get(&method_name)
                && let Some(func_type) = core::Func::from_type(db, *ty)
                && let Some(first_param) = func_type.params(db).first()
                && types_compatible(receiver_type, *first_param)
            {
                trace!(
                    "  found method {}::{} with matching first param",
                    ns_name, method_name
                );
                return Some((*path, *ty));
            }
        }
        trace!("  no matching method found");
        None
    }
}

/// Check if two types are compatible for UFCS resolution.
fn types_compatible(actual: Type<'_>, expected: Type<'_>) -> bool {
    actual == expected
}

// =============================================================================
// Pipeline Integration
// =============================================================================

/// Run TDNR on a module.
///
/// This builds a `ModuleEnv` from the module and uses it for UFCS resolution.
/// For `x.method(y)`, it looks up `method` in the environment and checks
/// if the first parameter type matches `x`'s type.
///
/// Uses `PatternApplicator` for declarative transformation.
pub fn resolve_tdnr<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    trace!("TDNR: starting resolution");

    // Sanity check: verify all operand references point to operations in the same module
    #[cfg(debug_assertions)]
    verify_operand_references(db, module, "TDNR input");

    let env = build_env(db, &module);
    trace!(
        "TDNR: built environment with {} definitions",
        env.definitions_iter().count()
    );

    // Use PatternApplicator for declarative transformation
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(ResolveTributeCallPattern { env });
    let target = ConversionTarget::new();
    let result = applicator.apply_partial(db, module, target).module;

    trace!("TDNR: resolution complete");
    result
}

#[cfg(debug_assertions)]
fn verify_operand_references<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
    context: &str,
) {
    use std::collections::HashSet;

    // Collect all operations in the module
    let mut all_ops: HashSet<trunk_ir::Operation<'db>> = HashSet::new();
    collect_ops_in_region(db, module.body(db), &mut all_ops);

    // Verify all operand references point to operations in the set
    verify_refs_in_region(db, module.body(db), &all_ops, context);
}

#[cfg(debug_assertions)]
fn collect_ops_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    ops: &mut std::collections::HashSet<trunk_ir::Operation<'db>>,
) {
    use std::ops::ControlFlow;
    use trunk_ir::{OperationWalk, WalkAction};

    let _ = region.walk_all::<()>(db, |op| {
        ops.insert(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
}

#[cfg(debug_assertions)]
fn verify_refs_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    all_ops: &std::collections::HashSet<trunk_ir::Operation<'db>>,
    context: &str,
) {
    use std::ops::ControlFlow;
    use trunk_ir::{OperationWalk, ValueDef, WalkAction};

    let _ = region.walk_all::<()>(db, |op| {
        for operand in op.operands(db).iter() {
            if let ValueDef::OpResult(ref_op) = operand.def(db)
                && !all_ops.contains(&ref_op)
            {
                tracing::warn!(
                    "STALE REFERENCE DETECTED in {}!\n  \
                     Operation {}.{} references {}.{} which is NOT in the module",
                    context,
                    op.dialect(db),
                    op.name(db),
                    ref_op.dialect(db),
                    ref_op.name(db)
                );
            }
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_env_lookup() {
        // Basic smoke test - ModuleEnv is used for UFCS resolution
        let env: ModuleEnv<'_> = ModuleEnv::new();
        assert!(env.lookup(Symbol::new("nonexistent")).is_none());
    }
}

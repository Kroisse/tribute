//! Core lowering logic.
//!
//! Transforms AST declarations and expressions to arena TrunkIR operations.

mod case;
mod decl;
mod expr;
mod handle;
mod lambda;

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core};
use trunk_ir::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use super::context::IrLoweringCtx;

use crate::ast::{
    CtorId, Expr, NodeId, Pattern, PatternKind, ResolvedRef, TypeAnnotation, TypeAnnotationKind,
    TypeKind, TypedRef,
};

// Re-export lower_module as the public entry point
pub use self::decl::lower_module;

// =============================================================================
// IrBuilder
// =============================================================================

/// Builder for emitting arena TrunkIR operations within a block.
///
/// Combines the lowering context, IrContext, and current block to provide
/// a unified API for expression lowering.
pub(super) struct IrBuilder<'a, 'db> {
    pub ctx: &'a mut IrLoweringCtx<'db>,
    pub ir: &'a mut IrContext,
    pub block: BlockRef,
}

impl<'a, 'db> IrBuilder<'a, 'db> {
    pub fn new(ctx: &'a mut IrLoweringCtx<'db>, ir: &'a mut IrContext, block: BlockRef) -> Self {
        Self { ctx, ir, block }
    }

    pub fn db(&self) -> &'db dyn salsa::Database {
        self.ctx.db
    }

    pub fn location(&self, id: NodeId) -> Location {
        self.ctx.location(id)
    }

    /// Emit a nil value (Tribute's unit type).
    pub fn emit_nil(&mut self, location: Location) -> ValueRef {
        let ty = self.ctx.nil_type(self.ir);
        let op = arith::r#const(self.ir, location, ty, Attribute::Unit);
        self.ir.push_op(self.block, op.op_ref());
        op.result(self.ir)
    }

    /// Emit diagnostic for unimplemented expression and return nil placeholder.
    pub fn emit_unsupported(&mut self, location: Location, feature: &str) -> Option<ValueRef> {
        Diagnostic::new(
            format!("{feature} not yet supported in IR lowering"),
            location.span,
            DiagnosticSeverity::Warning,
            CompilationPhase::Lowering,
        )
        .accumulate(self.db());
        Some(self.emit_nil(location))
    }

    /// Get the result type from a function type, or use the type directly.
    pub fn call_result_type(&mut self, ty: &crate::ast::Type<'db>) -> TypeRef {
        match ty.kind(self.db()) {
            TypeKind::Func { result, .. } => {
                let result = *result;
                self.ctx.convert_type(self.ir, result)
            }
            _ => {
                let ty = *ty;
                self.ctx.convert_type(self.ir, ty)
            }
        }
    }

    /// Insert an unrealized_conversion_cast if the value's type differs from target_ty.
    pub fn cast_if_needed(
        &mut self,
        location: Location,
        value: ValueRef,
        target_ty: TypeRef,
    ) -> ValueRef {
        let value_ty = self.ir.value_ty(value);

        if value_ty == target_ty {
            return value;
        }

        // Skip cast for closure → func conversions
        if self.ctx.is_closure_type(self.ir, value_ty) && self.ctx.is_func_type(self.ir, target_ty)
        {
            return value;
        }

        // Skip cast for func → func conversions with compatible signatures
        if self.ctx.is_func_type(self.ir, value_ty)
            && self.ctx.is_func_type(self.ir, target_ty)
            && self.ctx.func_type_param_count(self.ir, value_ty)
                == self.ctx.func_type_param_count(self.ir, target_ty)
        {
            return value;
        }

        // Insert unrealized_conversion_cast
        let cast_op = core::unrealized_conversion_cast(self.ir, location, value, target_ty);
        self.ir.push_op(self.block, cast_op.op_ref());
        cast_op.result(self.ir)
    }

    /// Lower arguments and propagate errors properly.
    pub fn collect_args(
        &mut self,
        args: impl IntoIterator<Item = Expr<TypedRef<'db>>>,
    ) -> Option<Vec<ValueRef>> {
        args.into_iter()
            .map(|a| expr::lower_expr(self, a))
            .collect()
    }
}

// =============================================================================
// Shared utility functions
// =============================================================================

/// Derive a qualified type name from a CtorId for use as a type_map key.
pub(super) fn qualified_type_name(db: &dyn salsa::Database, ctor_id: &CtorId<'_>) -> Symbol {
    ctor_id.qualified(db)
}

/// Resolve the ADT (enum/struct) type attribute for a constructor.
pub(super) fn resolve_enum_type_attr<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    ctor_ty: crate::ast::Type<'db>,
) -> TypeRef {
    let result_ty = match ctor_ty.kind(ctx.db) {
        TypeKind::Func { result, .. } => *result,
        _ => ctor_ty,
    };
    ctx.resolve_adt_type(result_ty)
        .unwrap_or_else(|| ctx.anyref_type(ir))
}

/// Extract the type name from a ResolvedRef.
pub(super) fn extract_type_name<'db>(
    db: &'db dyn salsa::Database,
    resolved: &ResolvedRef<'db>,
) -> Symbol {
    match resolved {
        ResolvedRef::Constructor { id, .. } => id.name(db),
        _ => unreachable!("Record type must be a constructor: {:?}", resolved),
    }
}

/// Extract the CtorId from a ResolvedRef.
pub(super) fn extract_ctor_id<'db>(resolved: &ResolvedRef<'db>) -> CtorId<'db> {
    match resolved {
        ResolvedRef::Constructor { id, .. } => *id,
        _ => unreachable!("Record type must be a constructor: {:?}", resolved),
    }
}

/// Create (or reuse) an `adt.struct` type for a tuple and register it in the type map.
pub(super) fn get_or_create_tuple_type<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    node_id: NodeId,
) -> Option<(Symbol, TypeRef)> {
    let ast_ty = ctx.get_node_type(node_id)?;
    let TypeKind::Tuple(elem_tys) = ast_ty.kind(ctx.db) else {
        return None;
    };
    let ir_fields: Vec<(Symbol, TypeRef)> = elem_tys
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let name = Symbol::from_dynamic(&i.to_string());
            let ir_ty = ctx.convert_type(ir, *ty);
            (name, ir_ty)
        })
        .collect();
    let type_names: Vec<String> = ir_fields
        .iter()
        .map(|(_, ty)| {
            let td = ir.types.get(*ty);
            td.name.to_string()
        })
        .collect();
    let tuple_name = Symbol::from_dynamic(&format!("__tuple_{}", type_names.join("_")));

    if let Some(struct_ty) = ctx.get_type(tuple_name) {
        return Some((tuple_name, struct_ty));
    }

    let struct_ty = ctx.adt_struct_type(ir, tuple_name, &ir_fields);
    ctx.register_type(tuple_name, struct_ty);
    Some((tuple_name, struct_ty))
}

/// Convert a type annotation to an arena TypeRef.
pub(super) fn convert_annotation_to_ir_type<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    annotation: Option<&TypeAnnotation>,
) -> TypeRef {
    let Some(ann) = annotation else {
        return ctx.i32_type(ir);
    };

    match &ann.kind {
        TypeAnnotationKind::Named(name) => {
            if *name == "Int" || *name == "Nat" {
                ctx.i32_type(ir)
            } else if *name == "Float" {
                ctx.f64_type(ir)
            } else if *name == "Bool" {
                ctx.bool_type(ir)
            } else if *name == "Bytes" {
                ctx.bytes_type(ir)
            } else if *name == "Rune" {
                ctx.i32_type(ir)
            } else if *name == "Nil" {
                ctx.nil_type(ir)
            } else {
                ctx.anyref_type(ir)
            }
        }
        TypeAnnotationKind::Path(_) => ctx.anyref_type(ir),
        TypeAnnotationKind::App { ctor, .. } => convert_annotation_to_ir_type(ctx, ir, Some(ctor)),
        _ => ctx.anyref_type(ir),
    }
}

/// Validate that a natural number literal fits in the i31 range.
///
/// Returns the value as `i32` if valid, or emits a diagnostic and returns `None`.
pub(super) fn validate_nat_i31(
    db: &dyn salsa::Database,
    location: Location,
    n: u64,
) -> Option<i32> {
    const I31_MAX: u64 = (1 << 30) - 1;
    if n > I31_MAX {
        Diagnostic::new(
            format!(
                "natural number literal {} exceeds i31 range (max: {})",
                n, I31_MAX
            ),
            location.span,
            DiagnosticSeverity::Error,
            CompilationPhase::Lowering,
        )
        .accumulate(db);
        return None;
    }
    Some(n as i32)
}

/// Validate that an integer literal fits in the i31 range.
///
/// Returns the value as `i32` if valid, or emits a diagnostic and returns `None`.
pub(super) fn validate_int_i31(
    db: &dyn salsa::Database,
    location: Location,
    n: i64,
) -> Option<i32> {
    const I31_MIN: i64 = -(1 << 30);
    const I31_MAX: i64 = (1 << 30) - 1;
    if !(I31_MIN..=I31_MAX).contains(&n) {
        Diagnostic::new(
            format!(
                "integer literal {} exceeds i31 range ({} to {})",
                n, I31_MIN, I31_MAX
            ),
            location.span,
            DiagnosticSeverity::Error,
            CompilationPhase::Lowering,
        )
        .accumulate(db);
        return None;
    }
    Some(n as i32)
}

/// Check if a pattern is unconditional (always matches without runtime checks).
pub(super) fn is_irrefutable_pattern<R: salsa::Update>(pattern: &Pattern<R>) -> bool {
    matches!(
        &*pattern.kind,
        PatternKind::Wildcard | PatternKind::Bind { .. }
    )
}

/// Create an identity `done_k` closure: `fn(result: anyref) -> anyref { return result }`.
///
/// Built as a direct `func.func` + `closure.new` (NOT `closure.lambda`), so it
/// bypasses `lower_closure_lambda` and has a fixed, known signature:
/// `(evidence, env, result) -> anyref`.
///
/// This is an internal mechanism closure, not a user lambda.
pub(super) fn create_identity_done_k(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
) -> ValueRef {
    use trunk_ir::context::{BlockArgData, BlockData, RegionData};
    use trunk_ir::dialect::{adt, func};

    use tribute_ir::dialect::{ability, closure};

    let anyref_ty = builder.ctx.anyref_type(builder.ir);
    let evidence_ty = ability::evidence_adt_type_ref(builder.ir);

    let dk_name = builder.ctx.gen_lambda_name();

    // func.func @identity_dk(%evidence, %env, %result) -> anyref { return %result }
    let dk_block = builder.ir.create_block(BlockData {
        location,
        args: vec![
            BlockArgData {
                ty: evidence_ty,
                attrs: Default::default(),
            },
            BlockArgData {
                ty: anyref_ty,
                attrs: Default::default(),
            },
            BlockArgData {
                ty: anyref_ty,
                attrs: Default::default(),
            },
        ],
        ops: Default::default(),
        parent_region: None,
    });
    let result_param = builder.ir.block_arg(dk_block, 2); // result is 3rd arg
    let ret = func::r#return(builder.ir, location, [result_param]);
    builder.ir.push_op(dk_block, ret.op_ref());

    let dk_region = builder.ir.create_region(RegionData {
        location,
        blocks: trunk_ir::smallvec::smallvec![dk_block],
        parent_op: None,
    });

    let all_param_types = vec![evidence_ty, anyref_ty, anyref_ty];
    let dk_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &all_param_types, anyref_ty, None);
    let dk_func_op = func::func(builder.ir, location, dk_name, dk_func_ty, dk_region);

    // Push to module block
    let module_block = builder
        .ctx
        .module_block()
        .expect("module block should be set");
    builder.ir.push_op(module_block, dk_func_op.op_ref());

    // closure.new @identity_dk, null_env
    let null_op = adt::ref_null(builder.ir, location, anyref_ty, anyref_ty);
    builder.ir.push_op(builder.block, null_op.op_ref());
    let null_env = null_op.result(builder.ir);

    let closure_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &[anyref_ty], anyref_ty, None);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
    let closure_op = closure::new(builder.ir, location, null_env, closure_ty, dk_name);
    builder.ir.push_op(builder.block, closure_op.op_ref());
    closure_op.result(builder.ir)
}

/// Emit a call to the `done_k` continuation closure with a result value,
/// followed by `func.return` with the call's result.
///
/// Used by effectful functions in CPS mode: instead of `func.return result`,
/// they call `done_k(result)` and return the call's result.
///
/// Done_k closures use their own calling convention: `fn(result) -> anyref`.
/// They are internal mechanism closures, not user-visible lambdas.
pub(super) fn emit_done_k_call(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    done_k: ValueRef,
    result: ValueRef,
) {
    use trunk_ir::dialect::func;

    let anyref_ty = builder.ctx.anyref_type(builder.ir);

    // Cast done_k to closure type so closure_lower can decompose the call.
    let closure_func_ty =
        builder
            .ctx
            .func_type_with_effect(builder.ir, &[anyref_ty], anyref_ty, None);
    let closure_ty = builder.ctx.closure_type(builder.ir, closure_func_ty);
    let done_k_closure = builder.cast_if_needed(location, done_k, closure_ty);
    let result_anyref = builder.cast_if_needed(location, result, anyref_ty);

    let call = func::call_indirect(
        builder.ir,
        location,
        done_k_closure,
        vec![result_anyref],
        anyref_ty,
    );
    builder.ir.push_op(builder.block, call.op_ref());

    let ret = func::r#return(builder.ir, location, [call.result(builder.ir)]);
    builder.ir.push_op(builder.block, ret.op_ref());
}

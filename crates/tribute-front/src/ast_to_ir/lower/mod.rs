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
use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::dialect::{arith, core};
use trunk_ir::arena::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::arena::types::{Attribute, Location};

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
        let result = op.result(self.ir);
        self.ctx.track_value_type(result, ty);
        result
    }

    /// Emit diagnostic for unimplemented expression and return nil placeholder.
    pub fn emit_unsupported(&mut self, location: Location, feature: &str) -> Option<ValueRef> {
        Diagnostic {
            message: format!("{feature} not yet supported in IR lowering"),
            span: location.span,
            severity: DiagnosticSeverity::Warning,
            phase: CompilationPhase::Lowering,
        }
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
        // Get the current type of the value from its definition
        let value_ty = {
            let value_data = self.ir.value(value);
            match value_data.def {
                trunk_ir::arena::refs::ValueDef::OpResult(op, _idx) => {
                    let results = self.ir.op_result_types(op);
                    let index = match value_data.def {
                        trunk_ir::arena::refs::ValueDef::OpResult(_, i) => i as usize,
                        _ => 0,
                    };
                    results.get(index).copied()
                }
                trunk_ir::arena::refs::ValueDef::BlockArg(_, _) => self
                    .ctx
                    .get_value_type(value)
                    .or_else(|| Some(self.ir.value_ty(value))),
            }
        };

        let Some(value_ty) = value_ty else {
            return value;
        };

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
        let casted = cast_op.result(self.ir);
        self.ctx.track_value_type(casted, target_ty);
        casted
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
    Symbol::from_dynamic(&ctor_id.qualified_name(db).to_string())
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
        .unwrap_or_else(|| ctx.any_type(ir))
}

/// Extract the type name from a ResolvedRef.
pub(super) fn extract_type_name<'db>(
    db: &'db dyn salsa::Database,
    resolved: &ResolvedRef<'db>,
) -> Symbol {
    match resolved {
        ResolvedRef::Constructor { id, .. } => id.ctor_name(db),
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
            } else if *name == "String" {
                ctx.string_type(ir)
            } else if *name == "Bytes" {
                ctx.bytes_type(ir)
            } else if *name == "Rune" {
                ctx.i32_type(ir)
            } else if *name == "Nil" {
                ctx.nil_type(ir)
            } else {
                ctx.any_type(ir)
            }
        }
        TypeAnnotationKind::Path(_) => ctx.any_type(ir),
        TypeAnnotationKind::App { ctor, .. } => convert_annotation_to_ir_type(ctx, ir, Some(ctor)),
        _ => ctx.any_type(ir),
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
        Diagnostic {
            message: format!(
                "natural number literal {} exceeds i31 range (max: {})",
                n, I31_MAX
            ),
            span: location.span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::Lowering,
        }
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
        Diagnostic {
            message: format!(
                "integer literal {} exceeds i31 range ({} to {})",
                n, I31_MIN, I31_MAX
            ),
            span: location.span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::Lowering,
        }
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

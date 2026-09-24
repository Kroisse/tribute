//! Core lowering logic.
//!
//! Transforms AST declarations and expressions to arena TrunkIR operations.

mod case;
mod decl;
mod expr;
mod logical;

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core};
use trunk_ir::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};

use super::context::IrLoweringCtx;

use crate::ast::{CallingConvention, CtorId, NodeId, ResolvedRef, TypeKind};

/// IR-level function signature extracted from a TypeScheme.
#[derive(Clone)]
pub(super) struct FuncSignature {
    pub param_types: Vec<TypeRef>,
    pub return_type: TypeRef,
    pub convention: CallingConvention,
}

impl FuncSignature {
    pub fn lookup_logical<'db>(
        ctx: &IrLoweringCtx<'db>,
        ir: &mut IrContext,
        name: Symbol,
    ) -> Option<Self> {
        if let Some(signature) = ctx.lookup_logical_generated_signature(name) {
            return Some(Self {
                param_types: signature.param_types.clone(),
                return_type: signature.return_type,
                convention: signature.convention,
            });
        }
        let scheme = *ctx.lookup_function_type(name)?;
        let body = scheme.body(ctx.db);
        match body.kind(ctx.db) {
            TypeKind::Func { params, result, .. } => Some(Self {
                param_types: params
                    .iter()
                    .map(|ty| ctx.convert_logical_type(ir, *ty))
                    .collect(),
                return_type: ctx.convert_logical_type(ir, *result),
                convention: ctx.calling_convention_for_type(body)?,
            }),
            _ => None,
        }
    }
}

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
        let op = arith::Const::operands()
            .value(Attribute::Unit)
            .results(ty)
            .build(self.ir, location);
        self.ir.push_op(self.block, op.op_ref());
        op.result(self.ir)
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

        // Insert unrealized_conversion_cast
        let cast_op = core::UnrealizedConversionCast::operands(value)
            .results(target_ty)
            .build(self.ir, location);
        self.ir.push_op(self.block, cast_op.op_ref());
        cast_op.result(self.ir)
    }
}

// =============================================================================
// Shared utility functions
// =============================================================================

/// Derive a qualified type name from a CtorId for use as a type_map key.
pub(super) fn qualified_type_name(db: &dyn salsa::Database, ctor_id: &CtorId<'_>) -> Symbol {
    ctor_id.qualified(db)
}

/// Extract the type name from a ResolvedRef.
pub(super) fn extract_type_name<'db>(
    db: &'db dyn salsa::Database,
    resolved: &ResolvedRef<'db>,
) -> Symbol {
    match resolved {
        ResolvedRef::Constructor { id, .. } => id.qualified(db),
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

/// Create or reuse a source-logical tuple layout, preserving nested callable types.
pub(super) fn get_or_create_logical_tuple_type<'db>(
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
            let ir_ty = ctx.convert_logical_type(ir, *ty);
            (name, ir_ty)
        })
        .collect();
    let tuple_name = ctx.logical_tuple_name(*ast_ty);

    if let Some(struct_ty) = ctx.get_type(tuple_name) {
        return Some((tuple_name, struct_ty));
    }

    let struct_ty = ctx.adt_struct_type(ir, tuple_name, &ir_fields);
    ctx.register_type(tuple_name, struct_ty);
    Some((tuple_name, struct_ty))
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

/// Resolve an enum type attribute from a constructor identity.
///
/// Monomorphization rewrites constructor IDs to their specialized enum name,
/// while a constructor's function type can remain polymorphic. The ID is the
/// authoritative layout identity for construction and pattern matching.
pub(super) fn resolve_enum_type_attr_for_constructor<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    resolved: &ResolvedRef<'db>,
    fallback_ctor_ty: crate::ast::Type<'db>,
) -> TypeRef {
    match resolved {
        ResolvedRef::Constructor { id, .. } => ctx.get_type(id.qualified(ctx.db)),
        _ => None,
    }
    .unwrap_or_else(|| resolve_enum_type_attr(ctx, ir, fallback_ctor_ty))
}

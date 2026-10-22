//! Core lowering logic.
//!
//! Transforms AST declarations and expressions to arena TrunkIR operations.

mod case;
mod decl;
mod expr;
mod logical;

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::dialect::tribute_control::{Row, RowBinder, RowUnion};
use tribute_ir::dialect::{tribute_control, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::types::Location;

use super::context::IrLoweringCtx;

use crate::ast::{CtorId, NodeId, ResolvedRef, TypeKind};

/// IR-level function signature extracted from a TypeScheme.
#[derive(Clone)]
pub(super) struct FuncSignature {
    pub param_types: Vec<TypeRef>,
    pub return_type: TypeRef,
    /// The callable's row, in its own binders.
    pub row: Row,
    pub binders: Vec<RowBinder>,
    pub unions: Vec<RowUnion>,
}

impl FuncSignature {
    pub fn lookup_logical<'db>(
        ctx: &mut IrLoweringCtx<'db>,
        ir: &mut IrContext,
        name: &[Symbol],
    ) -> Option<Self> {
        let outer = ctx.enter_definition(name);
        let signature = Self::lookup_in_definition(ctx, ir, name);
        ctx.leave_definition(outer);
        signature
    }

    fn lookup_in_definition<'db>(
        ctx: &IrLoweringCtx<'db>,
        ir: &mut IrContext,
        name: &[Symbol],
    ) -> Option<Self> {
        if let Some(signature) = ctx.lookup_logical_generated_signature(name) {
            return Some(Self {
                param_types: signature.param_types.clone(),
                return_type: signature.return_type,
                row: signature.row.clone(),
                binders: signature.binders.clone(),
                unions: vec![],
            });
        }
        let scheme = *ctx.lookup_function_type(name)?;
        let body = scheme.body(ctx.db);
        match body.kind(ctx.db) {
            TypeKind::Func {
                params,
                result,
                effect,
            } => Some(Self {
                param_types: params
                    .iter()
                    .map(|ty| ctx.convert_logical_type(ir, *ty))
                    .collect(),
                return_type: ctx.convert_logical_type(ir, *result),
                row: ctx.ability_row(*effect),
                binders: ctx
                    .current_row_binders()
                    .iter()
                    .map(|(_, binder)| *binder)
                    .collect(),
                unions: ctx.row_unions(scheme),
            }),
            _ => None,
        }
    }

    /// This signature as a reference passing the row arguments `args` sees
    /// it, in the referring definition's binders.
    pub fn instantiate(&self, ir: &mut IrContext, args: &[Row]) -> Self {
        let mut substitute = |ty| {
            tribute_control::substitute_rows(ir, ty, args)
                .expect("a signature's row tails name its quantified binders")
        };
        Self {
            param_types: self.param_types.iter().map(|ty| substitute(*ty)).collect(),
            return_type: substitute(self.return_type),
            row: self
                .row
                .substitute(args)
                .expect("a signature's row tail names a quantified binder"),
            binders: vec![],
            unions: vec![],
        }
    }

    /// The `row_args` attribute passing `args`, absent when there are none.
    pub fn row_args_attribute(
        ir: &mut IrContext,
        args: &[Row],
    ) -> Option<trunk_ir::types::Attribute> {
        (!args.is_empty()).then(|| {
            trunk_ir::types::Attribute::List(args.iter().map(|row| row.to_attribute(ir)).collect())
        })
    }

    /// The logical callable type of this signature.
    pub fn callable_type(&self, ir: &mut IrContext) -> TypeRef {
        tribute_control::FuncSig::params(self.param_types.iter().copied(), self.return_type)
            .row(self.row.clone())
            .intern(ir)
            .as_type_ref()
    }

    /// Mark `function`, a declaration or definition of this signature, with
    /// its row binders and unions.
    pub fn attach_rows(&self, ir: &mut IrContext, function: trunk_ir::refs::OpRef) {
        if !self.binders.is_empty() {
            let binders: Vec<_> = self
                .binders
                .iter()
                .map(|binder| ir.string_attr(binder.keyword()))
                .collect();
            ir.op_attrs_mut(function).insert(
                tribute_control::ROW_BINDERS_ATTR,
                trunk_ir::types::Attribute::List(binders),
            );
        }
        if !self.unions.is_empty() {
            let unions: Vec<_> = self
                .unions
                .iter()
                .map(|union| union.to_attribute(ir))
                .collect();
            ir.op_attrs_mut(function).insert(
                tribute_control::ROW_UNIONS_ATTR,
                trunk_ir::types::Attribute::List(unions),
            );
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
        let op = core::NilValue::operands().build(self.ir, location);
        self.ir.push_op(self.block, op.op_ref());
        op.result(self.ir)
    }

    /// Convert `value` to `target_ty` if its type differs: a box or unbox
    /// where the representation changes, otherwise an
    /// unrealized_conversion_cast.
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

        // A named function takes the callable type its use expects: a
        // conversion cast cannot change a calling convention.
        if let Some(value) = self.function_ref_as(location, value, target_ty) {
            return value;
        }

        let op = tribute_rt::convert_value(self.ir, location, value, target_ty);
        self.ir.push_op(self.block, op);
        self.ir.op_result(op, 0)
    }

    /// `value` again as a reference to the same function under the callable
    /// type `target_ty`, if `value` is a function reference and `target_ty`
    /// has its parameters and result under a convention at least as strong.
    fn function_ref_as(
        &mut self,
        location: Location,
        value: ValueRef,
        target_ty: TypeRef,
    ) -> Option<ValueRef> {
        let trunk_ir::refs::ValueDef::OpResult(producer, _) = self.ir.value_def(value) else {
            return None;
        };
        let named = tribute_control::FuncRef::from_op(self.ir, producer).ok()?;
        let source = tribute_control::FuncSig::from_type_ref(self.ir, self.ir.value_ty(value))?;
        let target = tribute_control::FuncSig::from_type_ref(self.ir, target_ty)?;
        if target.result(self.ir) != source.result(self.ir)
            || target.inputs(self.ir) != source.inputs(self.ir)
            || !target.row(self.ir).includes(&source.row(self.ir))
        {
            return None;
        }
        let function = named.func_ref(self.ir).clone();
        let row_args = self
            .ir
            .op_attrs(producer)
            .get(tribute_control::ROW_ARGS_ATTR)
            .cloned();
        let reference = tribute_control::FuncRef::operands()
            .func_ref(function)
            .results(target_ty)
            .build(self.ir, location);
        if let Some(row_args) = row_args {
            self.ir
                .op_attrs_mut(reference.op_ref())
                .insert(tribute_control::ROW_ARGS_ATTR, row_args);
        }
        self.ir.push_op(self.block, reference.op_ref());
        if let Some(path) = self.ctx.function_ref_path(producer).cloned() {
            self.ctx.record_function_ref(reference.op_ref(), path);
        }
        Some(reference.result(self.ir))
    }
}

// =============================================================================
// Shared utility functions
// =============================================================================

/// Derive a qualified type name from a CtorId for use as a type_map key.
pub(super) fn qualified_type_name(db: &dyn salsa::Database, ctor_id: &CtorId<'_>) -> Symbol {
    ctor_id.qualified(db).clone()
}

/// Extract the type name from a ResolvedRef.
pub(super) fn extract_type_name<'db>(
    db: &'db dyn salsa::Database,
    resolved: &ResolvedRef<'db>,
) -> Symbol {
    match resolved {
        ResolvedRef::Constructor { id, .. } => id.qualified(db).clone(),
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
    let elements: Vec<TypeRef> = elem_tys
        .iter()
        .map(|ty| ctx.convert_logical_type(ir, *ty))
        .collect();
    tribute_control::logical_tuple_type(ir, &elements);
    let name = tribute_control::logical_tuple_name(ir, &elements);
    let layout = ir
        .type_alias_by_text(&name)
        .expect("a logical tuple layout is registered under its name");
    Some((Symbol::new(&name), layout))
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
        ResolvedRef::Constructor { id, .. } => ctx.get_type(&id.qualified(ctx.db)),
        _ => None,
    }
    .unwrap_or_else(|| resolve_enum_type_attr(ctx, ir, fallback_ctor_ty))
}

//! Materialization and reconciliation of `core.unrealized_conversion_cast`.
//!
//! Dialect conversion leaves `unrealized_conversion_cast` placeholders where
//! a value's type and the type its uses declare disagree. Three passes handle
//! them, following MLIR's split between materialization and reconciliation:
//!
//! - [`materialize_unrealized_casts`] runs before target type conversion and
//!   replaces only the casts whose conversion needs real operations (such as
//!   boxing). Casts that only retype a value stay.
//! - [`convert_unrealized_casts`] runs with the target converter and converts
//!   each cast's result type, materializing the conversions that remain.
//! - [`reconcile_unrealized_casts`] needs no converter. It removes casts that
//!   fold away: identities, cast chains that return to an earlier type, and
//!   dead casts. A cast it cannot remove stays for the target's legality
//!   boundary to reject.

use std::collections::HashSet;
use std::ops::ControlFlow;

use crate::context::IrContext;
use crate::dialect::core;
use crate::ops::DialectOp;
use crate::refs::{BlockRef, OpRef, TypeRef, ValueDef, ValueRef};
use crate::rewrite::{Module, TypeConverter};
use crate::walk::{WalkAction, walk_op};

/// Replace the casts whose conversion needs real operations.
///
/// Runs before target type conversion. A cast is replaced only when its
/// declared result type does not convert, its source differs from that type,
/// and materialization emits operations. Other casts stay: they retype the
/// value until the target conversion makes both sides agree. In particular,
/// materializing toward a converted type would retype the uses before the
/// target conversion, so such casts are left without calling the materializer.
pub fn materialize_unrealized_casts(ctx: &mut IrContext, module: Module, tc: &TypeConverter) {
    for op in collect_casts(ctx, module) {
        let Some((block, input, declared)) = cast_parts(ctx, op) else {
            continue;
        };
        let from = ctx.value_ty(input);
        if from == declared || tc.convert_type_or_identity(ctx, declared) != declared {
            continue;
        }
        let location = ctx.op(op).location;
        match tc.materialize(ctx, location, input, from, declared) {
            Some(mat) if !mat.ops.is_empty() => {
                for mat_op in mat.ops {
                    ctx.insert_op_before(block, op, mat_op);
                }
                replace_cast(ctx, op, mat.value);
            }
            // A no-op materialization still changes the static type; keep the
            // cast until the target conversion unifies both sides.
            _ => {}
        }
    }
}

/// Convert every cast's result type with the target converter.
///
/// This is the tail of the target type conversion. Each cast's result is
/// retyped to its converted form; a cast whose source then differs is
/// materialized, and a materialization without operations forwards the
/// source. Identities are left to [`reconcile_unrealized_casts`], and a cast
/// the converter cannot materialize stays with its converted result type.
pub fn convert_unrealized_casts(ctx: &mut IrContext, module: Module, tc: &TypeConverter) {
    for op in collect_casts(ctx, module) {
        let Some((block, input, declared)) = cast_parts(ctx, op) else {
            continue;
        };
        let to = tc.convert_type_or_identity(ctx, declared);
        ctx.set_op_result_type(op, 0, to);
        let from = ctx.value_ty(input);
        if from == to {
            continue;
        }
        let location = ctx.op(op).location;
        if let Some(mat) = tc.materialize(ctx, location, input, from, to) {
            for mat_op in mat.ops {
                ctx.insert_op_before(block, op, mat_op);
            }
            replace_cast(ctx, op, mat.value);
        }
    }
}

/// Remove the casts that fold away without a type converter.
///
/// Mirrors MLIR's `reconcileUnrealizedCasts`. Casts are processed bottom to
/// top. A cast without uses is erased. Otherwise the chain of casts feeding
/// it is followed; when a cast in the chain (including the cast itself) takes
/// an input of the result type, uses of the result are replaced by that input.
/// Casts that feed a removed cast are revisited, so dead chains disappear.
///
/// A cast that cannot be removed stays. Reconciliation never fails and can
/// run at any point: it only forwards a value to uses of exactly its type.
pub fn reconcile_unrealized_casts(ctx: &mut IrContext, module: Module) {
    let mut worklist = collect_casts(ctx, module);
    let mut queued: HashSet<OpRef> = worklist.iter().copied().collect();
    let mut erased: HashSet<OpRef> = HashSet::new();

    while let Some(op) = worklist.pop() {
        queued.remove(&op);
        if erased.contains(&op) || cast_parts(ctx, op).is_none() {
            continue;
        }
        let result = ctx.op_result(op, 0);
        if !ctx.has_uses(result) {
            enqueue_input_cast(ctx, op, &mut worklist, &mut queued);
            crate::rewrite::erase_op(ctx, op);
            erased.insert(op);
            continue;
        }

        let result_ty = ctx.value_ty(result);
        let mut next = Some(op);
        while let Some(cast) = next {
            let input = ctx.op_operands(cast)[0];
            if ctx.value_ty(input) == result_ty {
                enqueue_input_cast(ctx, op, &mut worklist, &mut queued);
                replace_cast(ctx, op, input);
                erased.insert(op);
                break;
            }
            next = input_cast(ctx, cast);
        }
    }
}

/// Queue the cast that defines the input of `op`, if it is not queued yet.
fn enqueue_input_cast(
    ctx: &IrContext,
    op: OpRef,
    worklist: &mut Vec<OpRef>,
    queued: &mut HashSet<OpRef>,
) {
    if let Some(producer) = input_cast(ctx, op)
        && queued.insert(producer)
    {
        worklist.push(producer);
    }
}

/// Collect every well-formed cast under `module` in walk order.
fn collect_casts(ctx: &IrContext, module: Module) -> Vec<OpRef> {
    let mut casts = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if core::UnrealizedConversionCast::matches(ctx, op) {
            casts.push(op);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    casts
}

/// The parent block, input, and declared result type of a 1:1 cast still
/// attached to a block.
fn cast_parts(ctx: &IrContext, op: OpRef) -> Option<(BlockRef, ValueRef, TypeRef)> {
    let block = ctx.op(op).parent_block?;
    let (&[input], &[result_ty]) = (ctx.op_operands(op), ctx.op_result_types(op)) else {
        tracing::warn!(location = ?ctx.op(op).location, "malformed unrealized_conversion_cast");
        return None;
    };
    Some((block, input, result_ty))
}

/// The cast that defines the input of `op`, if any.
fn input_cast(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let &[input] = ctx.op_operands(op) else {
        return None;
    };
    match ctx.value_def(input) {
        ValueDef::OpResult(producer, _)
            if core::UnrealizedConversionCast::matches(ctx, producer) =>
        {
            Some(producer)
        }
        _ => None,
    }
}

/// Forward uses of the cast's result to `value` and erase the cast.
fn replace_cast(ctx: &mut IrContext, op: OpRef, value: ValueRef) {
    let result = ctx.op_result(op, 0);
    ctx.replace_all_uses(result, value);
    crate::rewrite::erase_op(ctx, op);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OperationDataBuilder;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::type_converter::MaterializeResult;
    use crate::symbol::Symbol;
    use crate::types::TypeDataBuilder;

    fn cast_count(ctx: &IrContext, module: Module) -> usize {
        collect_casts(ctx, module).len()
    }

    /// Assert that `module` prints the same as the module parsed from `expected`.
    #[track_caller]
    fn assert_ir(ctx: &IrContext, module: Module, expected: &str) {
        let mut expected_ctx = IrContext::new();
        let expected_module = parse_test_module(&mut expected_ctx, expected);
        assert_eq!(
            print_module(ctx, module.op()),
            print_module(&expected_ctx, expected_module.op())
        );
    }

    fn named_type(ctx: &mut IrContext, name: &'static str) -> TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", name).build())
    }

    /// Converts `core.i64` to `core.i32` and materializes `core.i32` →
    /// `core.f64` with a `test.box` operation. Every other pair is a no-op.
    fn test_converter(ctx: &mut IrContext) -> TypeConverter {
        let i32_ty = named_type(ctx, "i32");
        let i64_ty = named_type(ctx, "i64");
        let f64_ty = named_type(ctx, "f64");
        let mut tc = TypeConverter::new();
        tc.add_conversion(move |_ctx, ty| (ty == i64_ty).then_some(i32_ty));
        tc.set_materializer(move |ctx, loc, value, from, to| {
            if from == i32_ty && to == f64_ty {
                let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("box"))
                    .operand(value)
                    .result(to)
                    .build(ctx);
                let op = ctx.create_op(data);
                return Some(MaterializeResult {
                    value: ctx.op_result(op, 0),
                    ops: vec![op],
                });
            }
            Some(MaterializeResult { value, ops: vec![] })
        });
        tc
    }

    #[test]
    fn materialize_replaces_only_casts_that_emit_operations() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %boxed = core.unrealized_conversion_cast %x : core.f64
    %retyped = core.unrealized_conversion_cast %x : core.i64
    %noop = core.unrealized_conversion_cast %x : core.i16
    %same = core.unrealized_conversion_cast %x : core.i32
    func.return %same
  }
}"#,
        );
        let tc = test_converter(&mut ctx);

        materialize_unrealized_casts(&mut ctx, module, &tc);

        // Only the boxing cast is materialized; the conversion-only, no-op,
        // and identity casts are kept for later stages.
        assert_eq!(cast_count(&ctx, module), 3);
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("test.box"), "{printed}");
    }

    #[test]
    fn materialize_keeps_casts_without_materialization() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %r = core.unrealized_conversion_cast %x : core.f32
    func.return %x
  }
}"#,
        );
        let mut tc = TypeConverter::new();
        tc.set_materializer(|_, _, _, _, _| None);

        materialize_unrealized_casts(&mut ctx, module, &tc);

        assert_eq!(cast_count(&ctx, module), 1);
    }

    #[test]
    fn materialize_does_not_materialize_toward_a_converted_type() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %r = core.unrealized_conversion_cast %x : core.i64
    func.return %r
  }
}"#,
        );
        // i64 converts to f64, and the materializer would emit a real op.
        let i64_ty = named_type(&mut ctx, "i64");
        let f64_ty = named_type(&mut ctx, "f64");
        let mut tc = TypeConverter::new();
        tc.add_conversion(move |_ctx, ty| (ty == i64_ty).then_some(f64_ty));
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let counter = calls.clone();
        tc.set_materializer(move |ctx, loc, value, _from, to| {
            counter.set(counter.get() + 1);
            let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("convert"))
                .operand(value)
                .result(to)
                .build(ctx);
            let op = ctx.create_op(data);
            Some(MaterializeResult {
                value: ctx.op_result(op, 0),
                ops: vec![op],
            })
        });
        let before = print_module(&ctx, module.op());

        materialize_unrealized_casts(&mut ctx, module, &tc);

        assert_eq!(print_module(&ctx, module.op()), before, "the cast stays");
        assert_eq!(calls.get(), 0, "no materialization op is created");
    }

    #[test]
    fn convert_retypes_results_and_materializes() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %boxed = core.unrealized_conversion_cast %x : core.f64
    %retyped = core.unrealized_conversion_cast %x : core.i64
    %noop = core.unrealized_conversion_cast %x : core.i16
    func.return %retyped
  }
}"#,
        );
        let tc = test_converter(&mut ctx);

        convert_unrealized_casts(&mut ctx, module, &tc);

        // `%retyped` becomes an i32 → i32 identity for reconciliation; the
        // others are materialized.
        assert_eq!(cast_count(&ctx, module), 1);
        reconcile_unrealized_casts(&mut ctx, module);
        assert_eq!(cast_count(&ctx, module), 0);
    }

    #[test]
    fn convert_keeps_unmaterializable_cast_with_converted_result() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.f32) -> core.i64 {
    %r = core.unrealized_conversion_cast %x : core.i64
    func.return %r
  }
}"#,
        );
        let i32_ty = named_type(&mut ctx, "i32");
        let i64_ty = named_type(&mut ctx, "i64");
        let mut tc = TypeConverter::new();
        tc.add_conversion(move |_ctx, ty| (ty == i64_ty).then_some(i32_ty));
        tc.set_materializer(|_, _, _, _, _| None);

        convert_unrealized_casts(&mut ctx, module, &tc);
        reconcile_unrealized_casts(&mut ctx, module);

        let casts = collect_casts(&ctx, module);
        assert_eq!(casts.len(), 1);
        assert_eq!(ctx.op_result_types(casts[0]), [i32_ty]);
    }

    #[test]
    fn reconcile_removes_identity() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %r = core.unrealized_conversion_cast %x : core.i32
    func.return %r
  }
}"#,
        );

        reconcile_unrealized_casts(&mut ctx, module);

        assert_ir(
            &ctx,
            module,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    func.return %x
  }
}"#,
        );
    }

    #[test]
    fn reconcile_folds_round_trip_and_erases_dead_inner_cast() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i32
    %b = core.unrealized_conversion_cast %a : core.i64
    func.return %b
  }
}"#,
        );

        reconcile_unrealized_casts(&mut ctx, module);

        assert_ir(
            &ctx,
            module,
            r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i64 {
    func.return %x
  }
}"#,
        );
    }

    #[test]
    fn reconcile_folds_longer_chain() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i32
    %b = core.unrealized_conversion_cast %a : core.i16
    %c = core.unrealized_conversion_cast %b : core.i64
    func.return %c
  }
}"#,
        );

        reconcile_unrealized_casts(&mut ctx, module);

        assert_eq!(cast_count(&ctx, module), 0);
    }

    #[test]
    fn reconcile_erases_dead_cast_chain() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i32
    %b = core.unrealized_conversion_cast %a : core.i16
    func.return %x
  }
}"#,
        );

        reconcile_unrealized_casts(&mut ctx, module);

        assert_eq!(cast_count(&ctx, module), 0);
    }

    #[test]
    fn reconcile_keeps_live_branch_of_shared_cast() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i32 {
    %a = core.unrealized_conversion_cast %x : core.i32
    %b = core.unrealized_conversion_cast %a : core.i64
    %c = core.unrealized_conversion_cast %b : core.i32
    func.return %a
  }
}"#,
        );

        reconcile_unrealized_casts(&mut ctx, module);

        // `%c` folds to `%a` and dies with `%b`; `%a` still feeds the return.
        assert_ir(
            &ctx,
            module,
            r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i32 {
    %a = core.unrealized_conversion_cast %x : core.i32
    func.return %a
  }
}"#,
        );
    }

    #[test]
    fn reconcile_keeps_live_cast_that_does_not_fold() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i16
    %b = core.unrealized_conversion_cast %a : core.i64
    func.return %b
  }
}"#,
        );

        reconcile_unrealized_casts(&mut ctx, module);

        assert_eq!(cast_count(&ctx, module), 2);
    }
}

//! Legalization and reconciliation of `core.unrealized_conversion_cast`.
//!
//! Dialect conversion leaves `unrealized_conversion_cast` placeholders where
//! a value's type and the type its uses declare disagree. A pass that creates
//! such a cast emits any representation change (such as boxing) itself, so a
//! cast only retypes a value. Two steps then handle the casts, following
//! MLIR's split between materialization and reconciliation:
//!
//! - [`UnrealizedCastConversionPattern`] is the cast legalization pattern of a
//!   target conversion. It converts each cast's result type and materializes
//!   only conversions that need real operations; it never forwards a value of
//!   another type.
//! - [`ReconcileUnrealizedCasts`] needs no converter. It removes casts that
//!   fold away: identities, cast chains that return to an earlier type, and
//!   dead casts. A cast it cannot remove stays for the target's legality
//!   boundary to reject.

use rustc_hash::FxHashSet as HashSet;
use std::ops::ControlFlow;

use crate::analysis::AnalysisCache;
use crate::context::IrContext;
use crate::dialect::core;
use crate::ops::DialectOp;
use crate::pass::{Pass, PassRunResult};
use crate::refs::{BlockRef, OpRef, TypeRef, ValueDef, ValueRef};
use crate::rewrite::{Module, OpRewritePattern, PatternRewriter};
use crate::walk::{WalkAction, walk_op};

/// Cast legalization pattern for a target type conversion.
///
/// Retypes a cast's result to its converted form. When the source still
/// differs from that type, the cast is replaced by the operations the
/// converter materializes. A materialization without operations, or none at
/// all, keeps the cast: forwarding the source would give its uses a value of
/// another type. Identities are left to [`ReconcileUnrealizedCasts`], and a
/// cast that stays is rejected by the target's emission boundary unless a
/// target pass erases it (such as a WasmGC reference upcast).
pub struct UnrealizedCastConversionPattern;

impl OpRewritePattern for UnrealizedCastConversionPattern {
    type Op = core::UnrealizedConversionCast;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        unrealized_conversion_cast: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = unrealized_conversion_cast.op_ref();
        let (&[input], &[declared]) = (
            ctx.op_operands(op).as_array()?,
            ctx.op_result_types(op).as_array()?,
        );
        let to = rewriter
            .type_converter()
            .convert_type_or_identity(ctx, declared);
        let from = ctx.value_ty(input);
        let location = ctx.op(op).location;
        if from != to
            && let Some(mat) = rewriter
                .type_converter()
                .materialize(ctx, location, input, from, to)
            && !mat.ops.is_empty()
        {
            for mat_op in mat.ops {
                rewriter.insert_op(mat_op);
            }
            rewriter.erase_op(vec![mat.value]);
            return Some(());
        }
        if to == declared {
            return None;
        }
        let cast = core::UnrealizedConversionCast::operands(input)
            .results(to)
            .build(ctx, location);
        rewriter.replace_op(cast.op_ref());
        Some(())
    }
}

/// Pass that removes the casts that fold away without a type converter.
///
/// It removes identities, cast chains that return to an earlier type, and dead
/// casts. A cast it cannot remove stays for the target's legality boundary to
/// reject. See the module documentation.
pub struct ReconcileUnrealizedCasts;

impl Pass for ReconcileUnrealizedCasts {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "reconcile-unrealized-casts"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        reconcile_unrealized_casts(ctx, target.into());
        Ok(())
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
fn reconcile_unrealized_casts(ctx: &mut IrContext, module: Module) {
    let mut worklist = collect_casts(ctx, module);
    let mut queued: HashSet<OpRef> = worklist.iter().copied().collect();
    let mut erased: HashSet<OpRef> = HashSet::default();

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
    let &[input] = ctx.op_operands(op).as_array()?;
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
    use crate::dialect::core;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::TypeConverter;
    use crate::rewrite::type_converter::MaterializeResult;

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

    /// Converts `core.i64` to `core.i32` and materializes `core.i32` →
    /// `core.f64` with a `test.box` operation. Every other pair is a no-op.
    fn test_converter(ctx: &mut IrContext) -> TypeConverter {
        let i32_ty = core::I32::type_ref(ctx);
        let i64_ty = core::I64::type_ref(ctx);
        let f64_ty = core::F64::type_ref(ctx);
        let mut tc = TypeConverter::new();
        tc.add_conversion(move |_ctx, ty| (ty == i64_ty).then_some(i32_ty));
        tc.set_materializer(move |ctx, loc, value, from, to| {
            if from == i32_ty && to == f64_ty {
                let data = OperationDataBuilder::new(loc, "test.box")
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

    fn apply_cast_pattern(ctx: &mut IrContext, module: Module, tc: TypeConverter) {
        crate::rewrite::PatternApplicator::new(tc)
            .add_pattern(UnrealizedCastConversionPattern)
            .apply_partial(ctx, module);
    }

    #[test]
    fn cast_pattern_retypes_result_to_an_identity() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %r = core.unrealized_conversion_cast %x : core.i64
    func.return %r
  }
}"#,
        );
        let tc = test_converter(&mut ctx);

        apply_cast_pattern(&mut ctx, module, tc);

        assert_ir(
            &ctx,
            module,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %r = core.unrealized_conversion_cast %x : core.i32
    func.return %r
  }
}"#,
        );
        reconcile_unrealized_casts(&mut ctx, module);
        assert_eq!(cast_count(&ctx, module), 0);
    }

    #[test]
    fn cast_pattern_materializes_real_operations() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.f64 {
    %r = core.unrealized_conversion_cast %x : core.f64
    func.return %r
  }
}"#,
        );
        let tc = test_converter(&mut ctx);

        apply_cast_pattern(&mut ctx, module, tc);

        assert_eq!(cast_count(&ctx, module), 0);
        let printed = print_module(&ctx, module.op());
        assert!(printed.contains("test.box"), "{printed}");
    }

    #[test]
    fn cast_pattern_keeps_casts_without_real_materialization() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32, %y: core.f32) -> core.i16 {
    %noop = core.unrealized_conversion_cast %x : core.i16
    %none = core.unrealized_conversion_cast %y : core.i64
    func.return %noop
  }
}"#,
        );
        let i32_ty = core::I32::type_ref(&mut ctx);
        let i64_ty = core::I64::type_ref(&mut ctx);
        let mut tc = TypeConverter::new();
        tc.add_conversion(move |_ctx, ty| (ty == i64_ty).then_some(i32_ty));
        let f32_ty = core::F32::type_ref(&mut ctx);
        tc.set_materializer(move |_ctx, _loc, value, from, _to| {
            (from != f32_ty).then_some(MaterializeResult { value, ops: vec![] })
        });

        apply_cast_pattern(&mut ctx, module, tc);

        // Neither the no-op nor the failed materialization forwards the
        // source; the second cast only gets its converted result type.
        assert_ir(
            &ctx,
            module,
            r#"core.module @test {
  func.func @f(%x: core.i32, %y: core.f32) -> core.i16 {
    %noop = core.unrealized_conversion_cast %x : core.i16
    %none = core.unrealized_conversion_cast %y : core.i32
    func.return %noop
  }
}"#,
        );
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

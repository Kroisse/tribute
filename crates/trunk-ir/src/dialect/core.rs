//! Arena-based core dialect.

// === Operation registrations ===
crate::register_isolated_op!(core.module);

#[crate::dialect(crate = crate)]
mod core {
    #[attr(sym_name: Symbol)]
    fn module() {
        #[region(body)]
        {}
    }

    fn unrealized_conversion_cast(value: ()) -> result {}

    struct Nil;
    struct Never;
    struct Bytes;
    struct Ptr;
    struct Array<Element>;
    #[attr(nullable: bool)]
    struct Ref<Pointee>;
    struct Tuple<#[rest] Elements>;
    struct Func<Return, #[rest] Params>;
}

// =========================================================================
// Canonicalization folds
//
// Owned by this dialect and aggregated by `transforms::canonicalize` via
// [`folds`]. Folds are looked up by (dialect, op_name) so they don't
// self-filter — they assume the dispatcher already decided this op is
// `core.unrealized_conversion_cast`.
// =========================================================================

use crate::context::IrContext;
use crate::ops::DialectOp;
use crate::refs::{OpRef, ValueDef};
use crate::symbol::Symbol;
use crate::transforms::canonicalize::{FoldFn, FoldResult};

/// Folds this dialect contributes to `transforms::canonicalize`.
pub(crate) fn folds() -> Vec<(Symbol, Symbol, FoldFn)> {
    vec![(
        Symbol::new("core"),
        Symbol::new("unrealized_conversion_cast"),
        fold_unrealized_conversion_cast as FoldFn,
    )]
}

/// `core.unrealized_conversion_cast` folds:
///
/// - **Identity** (`%x : T → T`): drop the cast and forward `%x`.
/// - **Round-trip** (`cast<A → B>(cast<B → A>(%x))`): forward the inner
///   cast's input; the now-dead inner cast falls to DCE.
///
/// Safe *specifically* because both ops are
/// `core.unrealized_conversion_cast` — dialect-conversion placeholders
/// that carry no value-level conversion semantics. A resolved cast pair
/// like `arith.trunc` followed by `arith.extend` is *not* safe to collapse
/// the same way (narrower intermediate types lose information). Once
/// `resolve_unrealized_casts` has run, no `unrealized_conversion_cast`
/// ops remain and this fold is a no-op.
pub(crate) fn fold_unrealized_conversion_cast(ctx: &IrContext, op: OpRef) -> Option<FoldResult> {
    let operands = ctx.op_operands(op);
    let result_types = ctx.op_result_types(op);
    if operands.len() != 1 || result_types.len() != 1 {
        return None;
    }
    let input = operands[0];
    let result_ty = result_types[0];

    // Identity: T → T
    if ctx.value_ty(input) == result_ty {
        return Some(FoldResult::Forward(input));
    }

    // Round-trip: A → B → A
    if let ValueDef::OpResult(producer, _) = ctx.value_def(input)
        && UnrealizedConversionCast::matches(ctx, producer)
        && let Some(&inner_input) = ctx.op_operands(producer).first()
        && ctx.value_ty(inner_input) == result_ty
    {
        return Some(FoldResult::Forward(inner_input));
    }

    None
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod canonicalize_tests {
    use super::*;
    use crate::parser::parse_test_module;
    use crate::printer::print_module;
    use crate::rewrite::{ApplyResult, Module, PatternApplicator, TypeConverter};
    use crate::symbol::Symbol;
    use crate::walk::{WalkAction, walk_op};
    use std::ops::ControlFlow;

    use crate::transforms::canonicalize::FoldDispatchPattern;

    fn run_core_patterns(ctx: &mut IrContext, module: Module) -> ApplyResult {
        let dispatcher = FoldDispatchPattern::from_folds(folds());
        PatternApplicator::new(TypeConverter::new())
            .add_pattern_box(Box::new(dispatcher))
            .apply_partial(ctx, module)
    }

    fn count_ops(ctx: &IrContext, module: Module, dialect: &str, name: &str) -> usize {
        let dialect_sym = Symbol::from_dynamic(dialect);
        let name_sym = Symbol::from_dynamic(name);
        let mut count = 0usize;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            let data = ctx.op(op);
            if data.dialect == dialect_sym && data.name == name_sym {
                count += 1;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        count
    }

    #[test]
    fn unrealized_cast_identity_drops_same_type_cast() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %r = core.unrealized_conversion_cast %x : core.i32
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_core_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            0
        );
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn unrealized_cast_identity_does_not_match_when_types_differ() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %r = core.unrealized_conversion_cast %x : core.i64
    func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_core_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            1
        );
    }

    #[test]
    fn unrealized_cast_round_trip_collapses_pair() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i64) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i32
    %b = core.unrealized_conversion_cast %a : core.i64
    func.return %b
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_core_patterns(&mut ctx, module);
        assert!(result.total_changes >= 1);
        // Inner cast remains (now dead); outer is gone.
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            1
        );
        insta::assert_snapshot!(print_module(&ctx, module.op()));
    }

    #[test]
    fn unrealized_cast_round_trip_does_not_match_three_step_chain() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i64 {
    %a = core.unrealized_conversion_cast %x : core.i16
    %b = core.unrealized_conversion_cast %a : core.i64
    func.return %b
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        let result = run_core_patterns(&mut ctx, module);
        assert_eq!(result.total_changes, 0);
        assert_eq!(
            count_ops(&ctx, module, "core", "unrealized_conversion_cast"),
            2
        );
    }
}

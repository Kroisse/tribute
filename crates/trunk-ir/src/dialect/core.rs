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
// Canonicalization patterns
//
// Owned by this dialect and aggregated by `transforms::canonicalize` via
// [`canonicalization_patterns`]. Each pattern self-filters on
// `core.unrealized_conversion_cast` via the typed wrapper.
// =========================================================================

use crate::context::IrContext;
use crate::ops::DialectOp;
use crate::refs::{OpRef, ValueDef};
use crate::rewrite::{PatternRewriter, RewritePattern};

/// Patterns this dialect contributes to `transforms::canonicalize`.
pub fn canonicalization_patterns() -> Vec<Box<dyn RewritePattern>> {
    vec![
        Box::new(UnrealizedCastIdentity),
        Box::new(UnrealizedCastRoundTrip),
    ]
}

/// `core.unrealized_conversion_cast %x : T → T` → `%x`.
///
/// A no-op cast left over from a type conversion that ended up matching the
/// input type (e.g. after surrounding ops were rewritten). The op carries no
/// useful work — drop it and forward the operand.
pub struct UnrealizedCastIdentity;

impl RewritePattern for UnrealizedCastIdentity {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !UnrealizedConversionCast::matches(ctx, op) {
            return false;
        }
        let operands = ctx.op_operands(op);
        let result_types = ctx.op_result_types(op);
        if operands.len() != 1 || result_types.len() != 1 {
            return false;
        }
        let input = operands[0];
        let result_ty = result_types[0];
        if ctx.value_ty(input) != result_ty {
            return false;
        }
        rewriter.erase_op(vec![input]);
        true
    }

    fn name(&self) -> &'static str {
        "UnrealizedCastIdentity"
    }
}

/// `cast<A → B>(cast<B → A>(%x))` → `%x`.
///
/// A pair of casts that collapse back to the input type — common when one
/// rewrite materializes `B` and a later rewrite expects `A` again. We forward
/// the inner cast's input. The inner cast is left in place; if it had no
/// other users it becomes dead and is collected by DCE.
///
/// Safe *specifically* because both ops are `core.unrealized_conversion_cast`
/// — dialect-conversion placeholders that carry no value-level conversion
/// semantics. Resolved casts like `arith.trunc` followed by `arith.extend`
/// would *not* be safe to collapse this way: a narrower intermediate type
/// loses information that the outer cast cannot recover (e.g.
/// `i64 → i32 → i64` discards the upper 32 bits). The pattern's self-filter
/// on `UnrealizedConversionCast` is what restricts the rewrite to the
/// information-preserving placeholder case; once `resolve_unrealized_casts`
/// has run the materializer, no `unrealized_conversion_cast` ops remain and
/// this pattern is a no-op.
pub struct UnrealizedCastRoundTrip;

impl RewritePattern for UnrealizedCastRoundTrip {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        if !UnrealizedConversionCast::matches(ctx, op) {
            return false;
        }
        let operands = ctx.op_operands(op);
        let result_types = ctx.op_result_types(op);
        if operands.len() != 1 || result_types.len() != 1 {
            return false;
        }
        let outer_input = operands[0];
        let outer_result_ty = result_types[0];
        let producer = match ctx.value_def(outer_input) {
            ValueDef::OpResult(p, _) => p,
            ValueDef::BlockArg(_, _) => return false,
        };
        if !UnrealizedConversionCast::matches(ctx, producer) {
            return false;
        }
        let inner_operands = ctx.op_operands(producer);
        let Some(&inner_input) = inner_operands.first() else {
            return false;
        };
        // Round-trip iff the outer's result type matches the inner cast's
        // input type, i.e. types form `A → B → A`.
        if ctx.value_ty(inner_input) != outer_result_ty {
            return false;
        }
        rewriter.erase_op(vec![inner_input]);
        true
    }

    fn name(&self) -> &'static str {
        "UnrealizedCastRoundTrip"
    }
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

    fn run_core_patterns(ctx: &mut IrContext, module: Module) -> ApplyResult {
        let applicator = canonicalization_patterns().into_iter().fold(
            PatternApplicator::new(TypeConverter::new()),
            PatternApplicator::add_pattern_box,
        );
        applicator.apply_partial(ctx, module)
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

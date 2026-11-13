//! Operation flags: properties an operation's definition declares, with no
//! behavior of their own.
//!
//! A flag is a unit struct declared with [`#[op_flag]`](macro@crate::op_flag).
//! An operation declares its flags with `#[flags(..)]` in `#[dialect]`, and
//! its [`OpDef`] holds them. Unlike an operation interface, a flag has no
//! per-operation implementation and cannot be added to an operation after its
//! definition. A flag whose property is an invariant may check it with a
//! verifier, `#[op_flag(verify)]`, which every operation declaring the flag
//! runs as part of [`OpDef::verify`].

use crate::op_def::OpDef;
use crate::ops::VerificationError;
use crate::refs::{RegionRef, ValueDef, ValueRef};
use crate::{IrContext, OpRef};
use trunk_ir_macros::op_flag;

/// The identity of one operation flag, compared by address.
#[derive(Debug)]
pub struct OpFlagDef {
    pub name: &'static str,
    /// Calls the flag's [`VerifyFlag`] impl for `#[op_flag(verify)]` flags.
    pub verifier: Option<FlagVerifier>,
}

/// The invariant check of a flag declared `#[op_flag(verify)]`.
pub type FlagVerifier = fn(&IrContext, OpRef) -> Result<(), VerificationError>;

/// The invariant a flag states, checked for every operation that declares
/// it, as an MLIR trait's `verifyTrait` is.
///
/// Required by `#[op_flag(verify)]`. [`OpDef::verify`] runs it after the
/// operation passed its declarative schema and before its own `#[verify]`
/// hook. It may walk the operation's regions.
#[diagnostic::on_unimplemented(
    message = "`#[op_flag(verify)]` requires `VerifyFlag` for `{Self}`",
    note = "add `impl VerifyFlag for {Self} {{ fn verify(ctx: &IrContext, op: OpRef) -> Result<(), VerificationError> {{ .. }} }}`"
)]
pub trait VerifyFlag: OpFlag {
    fn verify(ctx: &IrContext, op: OpRef) -> Result<(), VerificationError>;
}

/// A type naming an operation flag, as `#[flags(..)]` lists it.
///
/// Implemented by `#[op_flag]`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an operation flag",
    label = "listed in `#[flags(..)]`",
    note = "an operation interface is listed in `#[implements(..)]`"
)]
pub trait OpFlag {
    const DEF: &'static OpFlagDef;

    /// Whether `op`'s operation declares this flag.
    fn is_set(ctx: &IrContext, op: OpRef) -> bool {
        OpDef::of(ctx, op).is_some_and(OpDef::has_flag::<Self>)
    }
}

/// Pure operations: no side effects, safe to remove if unused.
///
/// DCE may eliminate an operation with this flag if its results are unused.
#[op_flag]
pub struct Pure;

/// Operations whose regions cannot reference values from above.
///
/// The regions of an operation with this flag are "isolated" from the
/// enclosing scope - they cannot directly reference SSA values defined outside
/// the operation. This is important for:
///
/// 1. **Verification**: Check that isolated regions don't have stale references
/// 2. **Rewrite passes**: Different handling for isolated vs non-isolated ops
/// 3. **Code generation**: Isolated regions can be compiled independently
///
/// Its verifier rejects an operand anywhere in the operation's regions whose
/// value is defined outside the operation, as MLIR's `IsIsolatedFromAbove`
/// does. It does not enter a nested operation with this flag, which checks
/// its own regions.
///
/// Examples of isolated operations:
/// - `func.func` - function bodies (must receive values via parameters)
/// - `core.module` - module bodies
///
/// Examples of non-isolated operations (can capture outer values):
/// - `scf.if`, `scf.for` - control flow
/// - `closure.new` - closure creation
#[op_flag(verify)]
pub struct IsolatedFromAbove;

impl VerifyFlag for IsolatedFromAbove {
    fn verify(ctx: &IrContext, op: OpRef) -> Result<(), VerificationError> {
        let mut pending: Vec<RegionRef> = ctx.op_regions(op).collect();
        while let Some(region) = pending.pop() {
            for &block in &ctx.region(region).blocks {
                for &user in &ctx.block(block).ops {
                    for (index, &operand) in ctx.op_operands(user).iter().enumerate() {
                        if !is_defined_within(ctx, operand, op) {
                            return Err(format!(
                                "operand #{index} of {} uses {operand}, which is defined \
                                 outside this isolated operation",
                                ctx.op(user).name.display(ctx)
                            )
                            .into());
                        }
                    }
                    if !Self::is_set(ctx, user) {
                        pending.extend(ctx.op_regions(user));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Whether `value` is defined in a region nested under `owner`.
fn is_defined_within(ctx: &IrContext, value: ValueRef, owner: OpRef) -> bool {
    let block = match ctx.value_def(value) {
        ValueDef::OpResult(op, _) => ctx.op(op).parent_block,
        ValueDef::BlockArg(block, _) => Some(block),
    };
    let mut region = block.and_then(|block| ctx.block(block).parent_region);
    while let Some(current) = region {
        let Some(parent) = ctx.region(current).parent_op else {
            return false;
        };
        if parent == owner {
            return true;
        }
        region = ctx
            .op(parent)
            .parent_block
            .and_then(|block| ctx.block(block).parent_region);
    }
    false
}

/// Operations that must be the last operation of their block, as MLIR's
/// `IsTerminator` is.
///
/// The operation verifier rejects a flagged operation that some operation
/// follows. The flag says nothing about where control goes: `CallableExit`,
/// `Branch`, and `RegionBranchTerminator` describe that. An operation without
/// the flag may still end its block.
#[op_flag]
pub struct Terminator;

/// Operations whose regions' blocks need not end in a [`Terminator`], as
/// MLIR's `NoTerminator` trait.
///
/// Every other operation's blocks end in a `Terminator`. A region that
/// holds declarations rather than control flow, such as a module body, opts
/// out with this flag.
#[op_flag]
pub struct NoTerminator;

#[cfg(test)]
mod tests {
    use super::{IsolatedFromAbove, NoTerminator};
    use crate::IrContext;
    use crate::op_def::OpDef;
    use crate::parser::parse_test_module;
    use crate::validation::{validate_operation_verifiers, validate_value_integrity};

    #[crate::dialect]
    mod isolation_test {
        #[flags(IsolatedFromAbove, NoTerminator)]
        fn sealed() {
            #[region(body)]
            {}
        }

        #[flags(NoTerminator)]
        fn open() {
            #[region(body)]
            {}
        }

        fn make() -> Value<_> {}

        fn consume(value: Value<_>) {}
    }

    /// The flag violations `OpDef::verify` reports, one per operation, as
    /// `(operation, message)`.
    fn flag_violations(input: &str) -> Vec<(String, String)> {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let mut found = Vec::new();
        let _ = crate::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
            let def = OpDef::of(&ctx, op).expect("registered");
            for violation in def.verify(&ctx, op) {
                found.push((ctx.op(op).name.to_text(&ctx).into(), violation.to_string()));
            }
            std::ops::ControlFlow::Continue(crate::walk::WalkAction::Advance)
        });
        found
    }

    #[test]
    fn isolated_operation_uses_only_values_defined_inside_it() {
        let input = r#"core.module @test {
  "isolation_test.sealed"() ({
    %inner = "isolation_test.make"() : core.i32
    "isolation_test.consume"(%inner)
    "isolation_test.open"() ({
      %nested = "isolation_test.make"() : core.i32
      "isolation_test.consume"(%inner)
      "isolation_test.consume"(%nested)
    })
    "isolation_test.sealed"() ({
      %own = "isolation_test.make"() : core.i32
      "isolation_test.consume"(%own)
    })
  })
}"#;
        assert_eq!(flag_violations(input), []);
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        assert!(validate_operation_verifiers(&ctx, module).is_ok());
        assert!(validate_value_integrity(&ctx, module).is_ok());
    }

    #[test]
    fn isolated_operation_rejects_a_value_from_above() {
        let input = r#"core.module @test {
  %outer = "isolation_test.make"() : core.i32
  "isolation_test.sealed"() ({
    %inner = "isolation_test.make"() : core.i32
    "isolation_test.open"() ({
      "isolation_test.consume"(%outer)
    })
    "isolation_test.sealed"() ({
      "isolation_test.consume"(%inner)
    })
  })
}"#;
        // The outer operation reports the value captured through its
        // non-isolated region and leaves the nested isolated operation to
        // report its own capture, so each is reported once.
        let violations = flag_violations(input);
        assert_eq!(violations.len(), 2, "{violations:?}");
        for (op, message) in &violations {
            assert_eq!(op, "isolation_test.sealed");
            assert!(
                message.starts_with("IsolatedFromAbove: operand #0 of isolation_test.consume uses")
                    && message.ends_with("defined outside this isolated operation"),
                "{violations:?}"
            );
        }

        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let verified = validate_operation_verifiers(&ctx, module);
        assert_eq!(verified.errors.len(), 2, "{verified}");
        let scoped = validate_value_integrity(&ctx, module);
        assert_eq!(scoped.errors.len(), 2, "{scoped}");
    }
}

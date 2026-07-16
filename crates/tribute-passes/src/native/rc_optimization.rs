//! Local reference-counting optimizations for the native backend.
//!
//! This pass runs immediately after RC insertion. It removes a retain/release
//! pair only when both operations are in the same block, refer to the same SSA
//! value, and every intervening use is explicitly known not to let the value
//! escape. Unknown uses are barriers by design; later RC optimization work can
//! widen the safe-use set with dedicated borrow and alias analyses.

use tribute_ir::dialect::tribute_rt;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::ops::DialectOp;
use trunk_ir::rewrite::Module;
use trunk_ir::rewrite::helpers::erase_op;
use trunk_ir::{BlockRef, OpRef, RegionRef, ValueRef};

/// Eliminate provably redundant retain/release pairs in `module`.
pub fn eliminate_paired_rc(ctx: &mut IrContext, module: Module) {
    let Some(body) = module.body(ctx) else {
        return;
    };
    optimize_region(ctx, body);
}

fn optimize_region(ctx: &mut IrContext, region: RegionRef) {
    let blocks = ctx.region(region).blocks.to_vec();
    for block in blocks {
        while eliminate_one_pair(ctx, block) {}

        let ops = ctx.block(block).ops.to_vec();
        for op in ops {
            let nested_regions = ctx.op(op).regions.to_vec();
            for nested in nested_regions {
                optimize_region(ctx, nested);
            }
        }
    }
}

fn eliminate_one_pair(ctx: &mut IrContext, block: BlockRef) -> bool {
    let ops = ctx.block(block).ops.to_vec();

    for (retain_index, &retain_ref) in ops.iter().enumerate() {
        let Ok(retain) = tribute_rt::Retain::from_op(ctx, retain_ref) else {
            continue;
        };
        let ptr = retain.ptr(ctx);
        let retained = retain.result(ctx);

        for &op in &ops[retain_index + 1..] {
            if let Ok(release) = tribute_rt::Release::from_op(ctx, op) {
                let released = release.ptr(ctx);
                if released == ptr || released == retained {
                    erase_op(ctx, op);
                    ctx.replace_all_uses(retained, ptr);
                    erase_op(ctx, retain_ref);
                    return true;
                }
            }

            if is_barrier_for(ctx, op, ptr, retained) {
                break;
            }
        }
    }

    false
}

fn is_barrier_for(ctx: &IrContext, op: OpRef, ptr: ValueRef, retained: ValueRef) -> bool {
    // Captures by nested regions are not represented as ordinary operands on
    // the containing op, so do not move RC operations across region boundaries.
    if !ctx.op(op).regions.is_empty() {
        return true;
    }

    ctx.op_operands(op)
        .iter()
        .enumerate()
        .any(|(operand_index, &operand)| {
            (operand == ptr || operand == retained)
                && !is_proven_non_escaping_use(ctx, op, operand_index)
        })
}

fn is_proven_non_escaping_use(ctx: &IrContext, op: OpRef, operand_index: usize) -> bool {
    // Reading through a reference does not transfer ownership.
    if clif::Load::matches(ctx, op) {
        return operand_index == 0;
    }

    // `clif.store(value, addr)`: the address is borrowed, while storing the
    // reference as `value` lets it escape into memory.
    if clif::Store::matches(ctx, op) {
        return operand_index == 1;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::validation::validate_use_chains;

    fn run_pass(ir: &str) -> (IrContext, Module) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        eliminate_paired_rc(&mut ctx, module);
        (ctx, module)
    }

    fn printed_after_pass(ir: &str) -> String {
        let (ctx, module) = run_pass(ir);
        print_module(&ctx, module.op())
    }

    #[test]
    fn eliminates_adjacent_pair_and_rewrites_retain_result() {
        let (ctx, module) = run_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.i32 {
    %1 = tribute_rt.retain %0 : core.ptr
    %2 = clif.load %1 {offset = 0} : core.i32
    tribute_rt.release %0 {alloc_size = 12}
    clif.return %2
  }
}"#,
        );
        let output = print_module(&ctx, module.op());
        assert!(!output.contains("tribute_rt.retain"));
        assert!(!output.contains("tribute_rt.release"));
        assert!(output.contains("clif.load %0"));
        assert!(validate_use_chains(&ctx, module).is_ok());
    }

    #[test]
    fn allows_borrowed_load_and_store_address_uses() {
        let output = printed_after_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr, %1: core.i32) -> core.i32 {
    %2 = tribute_rt.retain %0 : core.ptr
    %3 = clif.load %0 {offset = 0} : core.i32
    clif.store %1, %2 {offset = 4}
    tribute_rt.release %0 {alloc_size = 12}
    clif.return %3
  }
}"#,
        );
        assert!(!output.contains("tribute_rt."));
    }

    #[test]
    fn store_value_is_an_escape_barrier() {
        let output = printed_after_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr, %1: core.ptr) -> core.nil {
    %2 = tribute_rt.retain %0 : core.ptr
    clif.store %0, %1 {offset = 0}
    tribute_rt.release %0 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        assert!(output.contains("tribute_rt.retain"));
        assert!(output.contains("tribute_rt.release"));
    }

    #[test]
    fn call_argument_is_an_escape_barrier() {
        let output = printed_after_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.nil {
  ^bb0:
    %1 = tribute_rt.retain %0 : core.ptr
    %2 = clif.call %0 {callee = @consume} : core.nil
    tribute_rt.release %0 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        assert!(output.contains("tribute_rt.retain"));
        assert!(output.contains("tribute_rt.release"));
    }

    #[test]
    fn branch_argument_is_an_escape_barrier() {
        let output = printed_after_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.nil {
  ^bb0:
    %1 = tribute_rt.retain %0 : core.ptr
    clif.jump %0 [^bb1]
  ^bb1(%2: core.ptr):
    tribute_rt.release %2 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        assert!(output.contains("tribute_rt.retain"));
        assert!(output.contains("tribute_rt.release"));
    }

    #[test]
    fn cast_alias_is_a_barrier() {
        let output = printed_after_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.nil {
    %1 = tribute_rt.retain %0 : core.ptr
    %2 = core.unrealized_conversion_cast %0 : core.ptr
    tribute_rt.release %0 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        assert!(output.contains("tribute_rt.retain"));
        assert!(output.contains("tribute_rt.release"));
    }

    #[test]
    fn does_not_match_a_different_pointer_or_cross_blocks() {
        let different = printed_after_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr, %1: core.ptr) -> core.nil {
    %2 = tribute_rt.retain %0 : core.ptr
    tribute_rt.release %1 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        assert!(different.contains("tribute_rt.retain"));
        assert!(different.contains("tribute_rt.release"));

        let cross_block = printed_after_pass(
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.nil {
  ^bb0:
    %1 = tribute_rt.retain %0 : core.ptr
    clif.jump [^bb1]
  ^bb1:
    tribute_rt.release %0 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        assert!(cross_block.contains("tribute_rt.retain"));
        assert!(cross_block.contains("tribute_rt.release"));
    }

    #[test]
    fn is_idempotent() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.func @f(%0: core.ptr) -> core.nil {
    %1 = tribute_rt.retain %0 : core.ptr
    tribute_rt.release %0 {alloc_size = 12}
    clif.return
  }
}"#,
        );
        eliminate_paired_rc(&mut ctx, module);
        let once = print_module(&ctx, module.op());
        eliminate_paired_rc(&mut ctx, module);
        assert_eq!(once, print_module(&ctx, module.op()));
    }
}

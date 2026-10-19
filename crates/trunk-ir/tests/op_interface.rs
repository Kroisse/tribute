//! An operation interface declared outside trunk-ir, implemented by a
//! dialect of this crate, registered with `#[implements(..)]`, and by an
//! operation of trunk-ir, registered with `register_interfaces!`.

use trunk_ir::dialect::arith;
use trunk_ir::ops::DialectOp;
use trunk_ir::parser::parse_test_module;
use trunk_ir::walk::{WalkAction, walk_region};
use trunk_ir::{IrContext, OpRef, ValueRef};

mod probe {
    #[trunk_ir::dialect]
    mod probe {
        #[implements(super::Consumes)]
        fn pair(lhs: Value<_>, rhs: Value<_>) {}

        fn other() {}
    }
}

/// Operands the operation consumes.
#[trunk_ir::op_interface]
pub trait Consumes: DialectOp {
    /// The operands the operation reads, at most `max` of them.
    fn consumed<'a>(self, ctx: &'a IrContext, max: &usize) -> &'a [ValueRef];

    fn count(&self, ctx: &IrContext) -> usize {
        ctx.op_operands(self.op_ref()).len()
    }

    /// Whether `op` is this operation; `this` is a tag returned on a match.
    fn same_as(self, _ctx: &IrContext, op: OpRef, this: u32) -> Option<u32> {
        (self.op_ref() == op).then_some(this)
    }

    fn forget_last(self, ctx: &mut IrContext, keep: u32) -> bool {
        let op = self.op_ref();
        if ctx.op_operands(op).len() <= keep as usize {
            return false;
        }
        ctx.remove_op_operand(op, keep);
        true
    }
}

impl Consumes for probe::Pair {
    fn consumed<'a>(self, ctx: &'a IrContext, max: &usize) -> &'a [ValueRef] {
        let operands = ctx.op_operands(self.op_ref());
        &operands[..operands.len().min(*max)]
    }
}

impl Consumes for arith::Addi {
    fn consumed<'a>(self, ctx: &'a IrContext, _max: &usize) -> &'a [ValueRef] {
        &ctx.op_operands(self.op_ref())[..1]
    }

    fn count(&self, _ctx: &IrContext) -> usize {
        1
    }
}

trunk_ir::register_interfaces!(arith::Addi: Consumes);

fn find(ctx: &IrContext, body: trunk_ir::RegionRef, name: trunk_ir::OpName) -> OpRef {
    let mut found = None;
    let _ = walk_region::<()>(ctx, body, &mut |op| {
        if ctx.op(op).name == name {
            found = Some(op);
        }
        std::ops::ControlFlow::Continue(WalkAction::Advance)
    });
    found.expect("operation in the test module")
}

#[test]
fn queries_dispatch_to_each_implementation() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  func.func @main(%a: core.i32, %b: core.i32) -> core.i32 {
    probe.pair %a, %b
    %sum = arith.addi %a, %b : core.i32
    probe.other
    func.return %sum
  }
}"#,
    );
    let body = module.body(&ctx).unwrap();
    let pair = find(&ctx, body, probe::Pair::NAME);
    let other = find(&ctx, body, probe::Other::NAME);
    let addi = find(&ctx, body, arith::Addi::NAME);

    let pair_interface = DynConsumes::get(&ctx, pair).expect("probe.pair implements Consumes");
    assert_eq!(pair_interface.op(), pair);
    assert_eq!(
        pair_interface.consumed(&ctx, &usize::MAX),
        ctx.op_operands(pair)
    );
    assert_eq!(
        pair_interface.consumed(&ctx, &1),
        &ctx.op_operands(pair)[..1]
    );
    assert_eq!(pair_interface.count(&ctx), 2);
    assert_eq!(pair_interface.same_as(&ctx, pair, 7), Some(7));
    assert_eq!(pair_interface.same_as(&ctx, addi, 7), None);

    let addi_interface = DynConsumes::get(&ctx, addi).expect("arith.addi implements Consumes");
    assert_eq!(
        addi_interface.consumed(&ctx, &usize::MAX),
        &ctx.op_operands(addi)[..1]
    );
    assert_eq!(addi_interface.count(&ctx), 1);

    assert!(DynConsumes::get(&ctx, other).is_none());
    assert!(!DynConsumes::is_implemented(&ctx, other));
    assert!(DynConsumes::is_implemented(&ctx, pair));

    let mut implementors: Vec<_> = DynConsumes::implementors()
        .map(|name| name.display(&ctx).to_string())
        .collect();
    implementors.sort();
    assert_eq!(implementors, ["arith.addi", "probe.pair"]);

    assert!(pair_interface.forget_last(&mut ctx, 1));
    assert_eq!(ctx.op_operands(pair).len(), 1);
    assert!(!pair_interface.forget_last(&mut ctx, 1));
}

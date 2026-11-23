use trunk_ir::context::IrContext;
use trunk_ir::op_def::OpDef;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::parser::parse_module;
use trunk_ir::rewrite::Module;

use super::*;
use crate::dialect::tribute_rtti::DynManagedRefType;

fn parse(input: &str) -> (IrContext, Module) {
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input)
        .unwrap_or_else(|error| panic!("failed to parse:\n{error}\n\n{input}"));
    let module = Module::new(&ctx, module).unwrap();
    (ctx, module)
}

const CELL: &str = r#"core.module @test {
  func.func @flip(%0: core.i1, %1: core.i1) -> core.i1 {
      %2 = heap.cell_new %0 : heap.cell<core.i1>
      heap.cell_set %2, %1
      %3 = heap.cell_get %2 : core.i1
      func.return %3
  }
}
"#;

fn body_ops(ctx: &IrContext, module: Module) -> Vec<trunk_ir::OpRef> {
    let func = trunk_ir::dialect::func::Func::from_op(ctx, module.ops(ctx)[0]).unwrap();
    let entry = ctx.region(func.body(ctx)).blocks[0];
    ctx.block(entry).ops.to_vec()
}

#[test]
fn cell_operations_round_trip() {
    let (ctx, module) = parse(CELL);
    assert_eq!(module.op().display(&ctx).to_string(), CELL);
    let ops = body_ops(&ctx, module);
    let new = CellNew::from_op(&ctx, ops[0]).unwrap();
    let cell_ty = ctx.value_ty(new.result(&ctx));
    assert!(DynManagedRefType::is_implemented(&ctx, cell_ty));
    let i1 = ctx.value_ty(new.value(&ctx));
    assert_eq!(
        Cell::from_type_ref(&ctx, cell_ty).unwrap().contents(&ctx),
        i1
    );
    for op in ops {
        assert!(OpDef::of(&ctx, op).unwrap().verify(&ctx, op).is_empty());
    }
}

#[test]
fn the_builder_infers_the_read_type() {
    let (mut ctx, module) = parse(CELL);
    let ops = body_ops(&ctx, module);
    let new = CellNew::from_op(&ctx, ops[0]).unwrap();
    let location = ctx.op(ops[0]).location;
    let get = CellGet::operands(new.result(&ctx)).build(&mut ctx, location);
    assert_eq!(
        ctx.value_ty(get.result(&ctx)),
        ctx.value_ty(new.value(&ctx))
    );
}

#[test]
fn a_value_of_another_type_is_rejected() {
    for text in [
        r#"core.module @test {
  func.func @f(%0: core.i32) {
    %1 = heap.cell_new %0 : heap.cell<core.i1>
    func.return
  }
}
"#,
        r#"core.module @test {
  func.func @f(%0: heap.cell<core.i1>, %1: core.i32) {
    heap.cell_set %0, %1
    func.return
  }
}
"#,
        r#"core.module @test {
  func.func @f(%0: heap.cell<core.i1>) {
    %1 = heap.cell_get %0 : core.i32
    func.return
  }
}
"#,
        r#"core.module @test {
  func.func @f(%0: core.i1) {
    %1 = heap.cell_get %0 : core.i1
    func.return
  }
}
"#,
    ] {
        let (ctx, module) = parse(text);
        let op = body_ops(&ctx, module)[0];
        assert!(
            !OpDef::of(&ctx, op).unwrap().verify(&ctx, op).is_empty(),
            "accepted:\n{text}"
        );
    }
}

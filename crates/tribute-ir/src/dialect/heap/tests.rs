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

const BOXES: &str = r#"core.module @test {
  !Node = adt.struct<"Node", (core.i32)>
  !Node_box_null = heap.box<null !Node>
  !Point = adt.struct<"Point", (core.f64, tribute_rt.anyref)>
  !Point_box = heap.box<!Point>
  !Shape = adt.enum<"Shape", [(), (core.f64)]>
  !Shape_box = heap.box<!Shape>

  func.func @values(%0: !Point_box, %1: !Shape_box, %2: heap.box<adt.array<core.i32>>) -> !Point_box {
      %3 = heap.unbox_value %0 : !Point
      %4 = heap.box_value %3 : !Point_box
      %5 = heap.unbox_value %1 : !Shape
      %6 = heap.box_value %5 : !Shape_box
      %7 = heap.unbox_value %2 : adt.array<core.i32>
      %8 = heap.box_value %7 : heap.box<adt.array<core.i32>>
      func.return %4
  }
  func.func @nulls(%0: heap.box<!Node>) -> core.i1 {
      %1 = heap.null : !Node_box_null
      %2 = heap.cast %0 : heap.box<!Node> to !Node_box_null
      %3 = heap.is_null %2 : !Node_box_null
      func.return %3
  }
}
"#;

fn all_body_ops(ctx: &IrContext, module: Module) -> Vec<trunk_ir::OpRef> {
    module
        .ops(ctx)
        .iter()
        .filter_map(|&op| trunk_ir::dialect::func::Func::from_op(ctx, op).ok())
        .flat_map(|func| {
            let entry = ctx.region(func.body(ctx)).blocks[0];
            ctx.block(entry).ops.to_vec()
        })
        .collect()
}

/// The layout of the box `boxing` builds.
fn boxed_layout(ctx: &IrContext, boxing: trunk_ir::OpRef) -> TypeRef {
    let boxing = BoxValue::from_op(ctx, boxing).unwrap();
    Box::from_type_ref(ctx, ctx.value_ty(boxing.result(ctx)))
        .unwrap()
        .layout(ctx)
}

#[test]
fn box_operations_round_trip_and_verify() {
    let (ctx, module) = parse(BOXES);
    assert_eq!(module.op().display(&ctx).to_string(), BOXES);
    for op in all_body_ops(&ctx, module) {
        let violations = OpDef::of(&ctx, op).unwrap().verify(&ctx, op);
        assert!(
            violations.is_empty(),
            "{}: {violations:?}",
            op.display(&ctx)
        );
    }
}

#[test]
fn a_nullable_box_writes_null_before_its_layout() {
    let (ctx, module) = parse(
        r#"core.module @test {
  !Node = adt.struct<"Node", (core.i32)>
  func.func @f(%0: heap.box<null !Node>, %1: heap.box<!Node, {nullable = true}>, %2: heap.box<!Node>) {
    func.return
  }
}
"#,
    );
    let func = trunk_ir::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
    let entry = ctx.region(func.body(&ctx)).blocks[0];
    let [null, generic, plain] = ctx.block_args(entry)[..] else {
        panic!("three parameters");
    };
    // The keyword and the generic attribute spell the same type.
    assert_eq!(ctx.value_ty(null), ctx.value_ty(generic));
    assert!(
        Box::from_type_ref(&ctx, ctx.value_ty(null))
            .unwrap()
            .is_nullable(&ctx)
    );
    assert!(
        !Box::from_type_ref(&ctx, ctx.value_ty(plain))
            .unwrap()
            .is_nullable(&ctx)
    );
    let printed = module.op().display(&ctx).to_string();
    assert!(printed.contains("heap.box<null !Node>"), "{printed}");
    assert!(!printed.contains("nullable"), "{printed}");
}

#[test]
fn null_and_a_nullable_attribute_are_not_written_together() {
    let mut ctx = IrContext::new();
    let text = r#"core.module @test {
  func.func @f(%0: heap.box<null adt.array<core.i32>, {nullable = true}>) {
    func.return
  }
}
"#;
    assert!(parse_module(&mut ctx, text).is_err());
}

#[test]
fn a_box_suggests_the_alias_of_its_layout() {
    use trunk_ir::asm_format::suggest_type_alias_name;

    let (ctx, module) = parse(
        r#"core.module @test {
  func.func @f(%0: heap.box<null adt.struct<"Node", (core.i32)>>, %1: heap.box<adt.struct<"Node", (core.i32)>>, %2: heap.box<adt.array<core.i32>>) {
    func.return
  }
}
"#,
    );
    let func = trunk_ir::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
    let entry = ctx.region(func.body(&ctx)).blocks[0];
    let names = ctx
        .block_args(entry)
        .iter()
        .map(|&arg| suggest_type_alias_name(&ctx, ctx.value_ty(arg)))
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            Some("Node_box_null".to_owned()),
            Some("Node_box".to_owned()),
            None
        ]
    );
    let printed = module.op().display(&ctx).to_string();
    assert!(
        printed.contains("!Node_box_null = heap.box<null !Node>"),
        "{printed}"
    );
}

#[test]
fn a_box_is_a_managed_reference() {
    let (ctx, module) = parse(BOXES);
    let boxing = BoxValue::from_op(&ctx, all_body_ops(&ctx, module)[1]).unwrap();
    let boxed = Box::from_type_ref(&ctx, ctx.value_ty(boxing.result(&ctx))).unwrap();
    assert!(DynManagedRefType::is_implemented(&ctx, boxed.as_type_ref()));
    assert!(!boxed.is_nullable(&ctx));
}

#[test]
fn a_null_test_borrows_and_a_cast_forwards() {
    use crate::dialect::tribute_rtti::{DynOperandOwnership, OperandRole};

    let (ctx, module) = parse(BOXES);
    let ops = all_body_ops(&ctx, module);
    let roles = |op| {
        DynOperandOwnership::get(&ctx, op)
            .unwrap()
            .operand_roles(&ctx)
            .unwrap()
    };
    let find = |matches: fn(&IrContext, trunk_ir::OpRef) -> bool| {
        ops.iter().copied().find(|&op| matches(&ctx, op)).unwrap()
    };
    assert_eq!(roles(find(IsNull::matches)), [OperandRole::Borrow]);
    assert_eq!(roles(find(Cast::matches)), [OperandRole::Forward]);
}

/// The function body of a module that declares `!Point`, `!Shape`, and the
/// nominal `!Node`.
fn rejected(body: &str) -> String {
    format!(
        r#"core.module @test {{
  !Point = adt.struct<"Point", (core.f64, core.f64)>
  !Shape = adt.enum<"Shape", [(), (core.f64)]>
  !Node = adt.struct<"Node", (core.i32)>
  func.func @f(%0: core.f64, %1: heap.box<!Point>, %2: heap.box<!Shape>, %3: heap.box<null !Node>, %4: heap.box<adt.array<core.i32>>) {{
    {body}
    func.return
  }}
}}
"#
    )
}

#[test]
fn reference_operations_that_do_not_fit_their_box_are_rejected() {
    for body in [
        // Null of a non-null box, and a null test of one.
        "%5 = heap.null : heap.box<!Point>",
        "%5 = heap.is_null %1 : heap.box<!Point>",
        // A cast between unrelated layouts, and an identity cast.
        "%5 = heap.cast %1 : heap.box<!Point> to heap.box<!Shape>",
        "%5 = heap.cast %1 : heap.box<!Point> to heap.box<!Point>",
        // A cast that changes the layout and the nullability at once.
        "%5 = heap.cast %3 : heap.box<null !Node> to heap.box<!Point>",
        // The unboxing of a nullable box.
        "%5 = heap.unbox_value %3 : !Node",
    ] {
        let text = rejected(body);
        let (ctx, module) = parse(&text);
        let op = body_ops(&ctx, module)[0];
        assert!(
            !OpDef::of(&ctx, op).unwrap().verify(&ctx, op).is_empty(),
            "accepted: {body}"
        );
    }
}

#[test]
fn a_value_boxes_only_as_a_non_null_box_of_its_own_layout() {
    for body in [
        "%4 = heap.box_value %0 : heap.box<null !Point>",
        "%4 = heap.box_value %0 : heap.box<!Shape>",
        "%4 = heap.box_value %1 : heap.box<adt.enum<\"Ghost\">>",
        "%4 = heap.unbox_value %2 : adt.enum<\"Ghost\">",
        "%4 = heap.box_value %3 : heap.box<adt.struct<\"Hollow\">>",
    ] {
        let text = format!(
            r#"core.module @test {{
  !Point = adt.struct<"Point", (core.f64, core.f64)>
  !Shape = adt.enum<"Shape", [(), (core.f64)]>
  func.func @f(%0: !Point, %1: adt.enum<"Ghost">, %2: heap.box<adt.enum<"Ghost">>, %3: adt.struct<"Hollow">) {{
    {body}
    func.return
  }}
}}
"#
        );
        let (ctx, module) = parse(&text);
        let op = body_ops(&ctx, module)[0];
        assert!(
            !OpDef::of(&ctx, op).unwrap().verify(&ctx, op).is_empty(),
            "accepted: {body}"
        );
    }
}

#[test]
fn a_row_instance_boxes_as_the_value_of_its_def() {
    let text = r#"core.module @test {
  !Callback = adt.struct<"Callback"['a0], (tribute_control.func_sig<(core.i64) -> core.i64 row(@Log | 'a0)>)>
  !Ticked = adt.struct<"Callback"[row(@Tick)]>
  func.func @f(%0: !Ticked) {
    %1 = heap.box_value %0 : heap.box<!Ticked>
    %2 = heap.unbox_value %1 : !Ticked
    func.return
  }
}
"#;
    let (ctx, module) = parse(text);
    for op in &body_ops(&ctx, module)[..2] {
        let violations = OpDef::of(&ctx, *op).unwrap().verify(&ctx, *op);
        assert!(violations.is_empty(), "{violations:?}");
    }
}

#[test]
fn a_box_of_a_non_layout_is_not_a_box() {
    let mut ctx = IrContext::new();
    let i32_ty = trunk_ir::types::TypeDataBuilder::new("core.i32").intern(&mut ctx);
    assert!(Box::params(i32_ty).intern(&mut ctx).is_err());
}

#[test]
fn a_non_null_box_has_one_spelling() {
    let (mut ctx, module) = parse(BOXES);
    let point = boxed_layout(&ctx, all_body_ops(&ctx, module)[1]);
    assert!(Box::params(point).nullable(true).intern(&mut ctx).is_ok());
    assert!(Box::params(point).nullable(false).intern(&mut ctx).is_err());
}

#[test]
fn a_capture_box_and_its_base_box_cast_both_ways() {
    let text = r#"core.module @test {
  !base = adt.struct<(core.ptr), {layout = "closure"}>
  !captures = adt.struct<(core.ptr, core.i32), {closure_base = !base, layout = "closure_captures"}>
  func.func @f(%0: heap.box<!captures>) {
    %1 = heap.cast %0 : heap.box<!captures> to heap.box<!base>
    %2 = heap.cast %1 : heap.box<!base> to heap.box<!captures>
    func.return
  }
}
"#;
    let (ctx, module) = parse(text);
    for op in &body_ops(&ctx, module)[..2] {
        let violations = OpDef::of(&ctx, *op).unwrap().verify(&ctx, *op);
        assert!(violations.is_empty(), "{violations:?}");
    }
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

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
  !Point = adt.struct<"Point", (core.f64, tribute_rt.anyref)>
  !Shape = adt.enum<"Shape", [(), (core.f64)]>
  !t0 = heap.box<!Node, {nullable = true}>

  func.func @structs(%0: core.f64, %1: tribute_rt.anyref, %2: tribute_rt.anyref) -> core.f64 {
      %3 = heap.struct_new(%0, %1) : (core.f64, tribute_rt.anyref) -> heap.box<!Point>
      %4 = heap.struct_get %3 {field = 1} : tribute_rt.anyref
      %5 = heap.struct_update %3, %2 {field = 1} : heap.box<!Point>
      %6 = heap.struct_get %5 {field = 0} : core.f64
      func.return %6
  }
  func.func @variants(%0: core.f64) -> core.f64 {
      %1 = heap.variant_new 1(%0) : (core.f64) -> heap.box<!Shape>
      %2 = heap.variant_is %1 {tag = 1} : core.i1
      %3 = heap.variant_tag %1 : heap.box<!Shape>
      %4 = heap.variant_get %1 {field = 0, tag = 1} : core.f64
      func.return %4
  }
  func.func @arrays(%0: core.index, %1: core.i32, %2: core.index) -> core.index {
      %3 = heap.array_new %0, %1 : heap.box<adt.array<core.i32>>
      %4 = heap.array_of %1, %1 : heap.box<adt.array<core.i32>>
      %5 = heap.array_splice %4 [%2, %0] (%1) : heap.box<adt.array<core.i32>>
      %6 = heap.array_get %5, %2 : core.i32
      %7 = heap.array_set %3, %2, %6 : heap.box<adt.array<core.i32>>
      %8 = heap.array_len %7 : heap.box<adt.array<core.i32>>
      %9 = heap.array_const dense<"0x0102"> : heap.box<adt.array<core.i8>>
      func.return %8
  }
  func.func @nulls(%0: heap.box<!Node>) -> core.i1 {
      %1 = heap.null : !t0
      %2 = heap.cast %0 : heap.box<!Node> to !t0
      %3 = heap.is_null %2 : !t0
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
fn a_box_is_a_managed_reference_and_its_layout_is_not_one() {
    let (ctx, module) = parse(BOXES);
    let new = StructNew::from_op(&ctx, all_body_ops(&ctx, module)[0]).unwrap();
    let boxed = Box::from_type_ref(&ctx, ctx.value_ty(new.result(&ctx))).unwrap();
    assert!(DynManagedRefType::is_implemented(&ctx, boxed.as_type_ref()));
    assert!(!boxed.is_nullable(&ctx));
}

#[test]
fn updates_consume_their_source_and_reads_project() {
    use crate::dialect::tribute_rtti::{DynOperandOwnership, OperandRole};

    let (ctx, module) = parse(BOXES);
    let ops = all_body_ops(&ctx, module);
    let roles = |op| {
        DynOperandOwnership::get(&ctx, op)
            .unwrap()
            .operand_roles(&ctx)
            .unwrap()
    };
    let anyref = ctx.value_ty(StructNew::from_op(&ctx, ops[0]).unwrap().fields(&ctx)[1]);
    let point = ctx.value_ty(StructNew::from_op(&ctx, ops[0]).unwrap().result(&ctx));
    let point = Box::from_type_ref(&ctx, point).unwrap().layout(&ctx);
    assert_eq!(
        roles(ops[1]),
        [OperandRole::Project {
            object: point,
            field: anyref
        }]
    );
    assert_eq!(
        roles(ops[2]),
        [
            OperandRole::Consume,
            OperandRole::Initialize { field: anyref }
        ]
    );
    let array_set = ops
        .iter()
        .copied()
        .find(|&op| ArraySet::matches(&ctx, op))
        .unwrap();
    assert!(matches!(
        roles(array_set)[..],
        [
            OperandRole::Consume,
            OperandRole::Borrow,
            OperandRole::Initialize { .. }
        ]
    ));
    let cast = ops
        .iter()
        .copied()
        .find(|&op| Cast::matches(&ctx, op))
        .unwrap();
    assert_eq!(roles(cast), [OperandRole::Forward]);
}

#[test]
fn every_box_operation_states_the_role_of_each_operand() {
    use crate::dialect::tribute_rtti::{DynOperandOwnership, OperandRole};

    let (ctx, module) = parse(BOXES);
    let ops = all_body_ops(&ctx, module);
    let roles = |op| {
        DynOperandOwnership::get(&ctx, op)
            .map(|op| op.operand_roles(&ctx).unwrap())
            .unwrap_or_default()
    };
    let initialize = |role: &OperandRole| matches!(role, OperandRole::Initialize { .. });
    for &op in &ops {
        let roles = roles(op);
        if DynOperandOwnership::is_implemented(&ctx, op) {
            assert_eq!(
                roles.len(),
                ctx.op_operands(op).len(),
                "{}",
                op.display(&ctx)
            );
        }
        if StructNew::matches(&ctx, op) || VariantNew::matches(&ctx, op) {
            assert!(roles.iter().all(initialize), "{}", op.display(&ctx));
        } else if VariantGet::matches(&ctx, op) {
            assert!(matches!(roles[..], [OperandRole::Project { .. }]));
        } else if ArrayOf::matches(&ctx, op) {
            assert!(roles.iter().all(initialize));
        } else if ArraySplice::matches(&ctx, op) {
            assert!(matches!(
                roles[..],
                [
                    OperandRole::Borrow,
                    OperandRole::Borrow,
                    OperandRole::Borrow,
                    OperandRole::Initialize { .. }
                ]
            ));
        } else if ArrayGet::matches(&ctx, op) {
            assert!(matches!(
                roles[..],
                [OperandRole::Project { .. }, OperandRole::Borrow]
            ));
        } else if ArrayNew::matches(&ctx, op)
            || ArrayLen::matches(&ctx, op)
            || VariantIs::matches(&ctx, op)
            || VariantTag::matches(&ctx, op)
            || IsNull::matches(&ctx, op)
        {
            assert!(
                roles.iter().all(|role| *role == OperandRole::Borrow),
                "{}",
                op.display(&ctx)
            );
        }
    }
}

/// The function body of a module that declares `!Point`, `!Shape`, and the
/// nominal `!Node`.
fn rejected(body: &str) -> String {
    format!(
        r#"core.module @test {{
  !Point = adt.struct<"Point", (core.f64, core.f64)>
  !Shape = adt.enum<"Shape", [(), (core.f64)]>
  !Node = adt.struct<"Node", (core.i32)>
  func.func @f(%0: core.f64, %1: heap.box<!Point>, %2: heap.box<!Shape>, %3: heap.box<!Node, {{nullable = true}}>, %4: heap.box<adt.array<core.i32>>) {{
    {body}
    func.return
  }}
}}
"#
    )
}

#[test]
fn box_operations_that_do_not_fit_their_layout_are_rejected() {
    for body in [
        // One field too few.
        "%5 = heap.struct_new(%0) : (core.f64) -> heap.box<!Point>",
        // A struct built as a box of an enum.
        "%5 = heap.struct_new(%0) : (core.f64) -> heap.box<!Shape>",
        // A struct built as a nullable box.
        "%5 = heap.struct_new(%0, %0) : (core.f64, core.f64) -> heap.box<!Point, {nullable = true}>",
        // A field past the layout's.
        "%5 = heap.struct_get %1 {field = 2} : core.f64",
        // A read of the wrong type.
        "%5 = heap.struct_get %1 {field = 0} : core.i32",
        // A read through a nullable box.
        "%5 = heap.struct_get %3 {field = 0} : core.i32",
        // A variant the enum does not have.
        "%5 = heap.variant_new 2(%0) : (core.f64) -> heap.box<!Shape>",
        "%5 = heap.variant_get %2 {field = 0, tag = 0} : core.f64",
        // A variant read of a struct.
        "%5 = heap.variant_tag %1 : heap.box<!Point>",
        // An update of the wrong type.
        "%5 = heap.struct_update %1, %4 {field = 0} : heap.box<!Point>",
        // Null of a non-null box, and a null test of one.
        "%5 = heap.null : heap.box<!Point>",
        "%5 = heap.is_null %1 : heap.box<!Point>",
        // A cast between unrelated layouts, and an identity cast.
        "%5 = heap.cast %1 : heap.box<!Point> to heap.box<!Shape>",
        "%5 = heap.cast %1 : heap.box<!Point> to heap.box<!Point>",
        // An array operation on a struct.
        "%5 = heap.array_len %1 : heap.box<!Point>",
        // A variant test and a variant build with a tag or fields the enum
        // does not have.
        "%5 = heap.variant_is %2 {tag = 2} : core.i1",
        "%5 = heap.variant_new 1() : () -> heap.box<!Shape>",
        // A variant read of a struct, and a struct read of an enum.
        "%5 = heap.variant_get %1 {field = 0, tag = 0} : core.f64",
        "%5 = heap.struct_get %2 {field = 0} : core.f64",
        // An update and an array read through a nullable box.
        "%5 = heap.struct_update %3, %0 {field = 0} : heap.box<!Node, {nullable = true}>",
        "%5 = heap.array_get %3, %0 : core.i32",
        // An array built as a nullable box.
        "%5 = heap.array_of : heap.box<adt.array<core.i32>, {nullable = true}>",
        // A cast that changes the layout and the nullability at once.
        "%5 = heap.cast %3 : heap.box<!Node, {nullable = true}> to heap.box<!Point>",
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
fn an_operation_on_a_box_of_a_key_without_a_body_is_rejected() {
    for body in [
        "%1 = heap.variant_is %0 {tag = 0} : core.i1",
        "%1 = heap.variant_tag %0 : heap.box<adt.enum<\"Ghost\">>",
        "%1 = heap.variant_get %0 {field = 0, tag = 0} : core.i32",
    ] {
        let text = format!(
            r#"core.module @test {{
  func.func @f(%0: heap.box<adt.enum<"Ghost">>) {{
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
fn a_box_of_a_non_layout_is_not_a_box() {
    let mut ctx = IrContext::new();
    let i32_ty = trunk_ir::types::TypeDataBuilder::new("core.i32").intern(&mut ctx);
    assert!(Box::params(i32_ty).intern(&mut ctx).is_err());
}

#[test]
fn a_non_null_box_has_one_spelling() {
    let (mut ctx, module) = parse(BOXES);
    let new = StructNew::from_op(&ctx, all_body_ops(&ctx, module)[0]).unwrap();
    let point = Box::from_type_ref(&ctx, ctx.value_ty(new.result(&ctx)))
        .unwrap()
        .layout(&ctx);
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
fn a_read_through_a_row_instance_sees_its_row_arguments() {
    let text = r#"core.module @test {
  !Callback = adt.struct<"Callback"['a0], (tribute_control.func_sig<(core.i64) -> core.i64 row(@Log | 'a0)>)>
  !Ticked = adt.struct<"Callback"[row(@Tick)]>
  func.func @f(%0: heap.box<!Ticked>) {
    func.return
  }
}
"#;
    let (mut ctx, module) = parse(text);
    let func = trunk_ir::dialect::func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
    let entry = ctx.region(func.body(&ctx)).blocks[0];
    let boxed = ctx.block_args(entry)[0];
    let ticked = Box::from_type_ref(&ctx, ctx.value_ty(boxed))
        .unwrap()
        .layout(&ctx);
    let substituted = adt::field_type(&mut ctx, ticked, 0).unwrap();
    let callback = adt::layout_of(&ctx, ticked).unwrap();
    let declared = Struct::from_type_ref(&ctx, callback)
        .unwrap()
        .field_type(&ctx, 0)
        .unwrap();
    assert_ne!(substituted, declared);
    let location = ctx.op(module.op()).location;
    for (result, accepted) in [(substituted, true), (declared, false)] {
        let get = StructGet::operands(boxed)
            .field(0)
            .results(result)
            .build(&mut ctx, location);
        let violations = OpDef::of(&ctx, get.op_ref())
            .unwrap()
            .verify(&ctx, get.op_ref());
        assert_eq!(violations.is_empty(), accepted, "{violations:?}");
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

use trunk_ir::op_def::OpDef;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::parser::parse_module;
use trunk_ir::printer::print_module;

use super::*;
use crate::dialect::tribute_control;

const MODULE: &str = r#"core.module @test {
  !t0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !t1 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>

  tribute_cps.func @declared(%arg0: core.i32) -> core.i32 convention(direct)
  tribute_cps.func @apply(%0: !t0, %1: core.i32) -> core.i32 convention(direct) {
      %2 = tribute_cps.call_indirect %0, %1 {convention = "direct"} : core.i32
      tribute_control.return %2
  }
  tribute_cps.func @outer['a0](%0: core.i32) -> core.i32 row(| 'a0) convention(cps) attributes {visibility = "private"} {
      %1 = tribute_cps.lambda(%2: core.i32) -> core.i32 row(| 'a0) captures [%0] entries {
        cps {
            tribute_control.return %0
        }
      }
      %3 = tribute_cps.func_ref {convention = "cps", func_ref = @declared} : !t0
      tribute_cps.tail_call_indirect %1, %0 {convention = "cps"}
  }
}
"#;

fn parse(input: &str) -> (IrContext, trunk_ir::rewrite::Module) {
    let mut ctx = IrContext::new();
    let module = parse_module(&mut ctx, input)
        .unwrap_or_else(|error| panic!("failed to parse:\n{error}\n\n{input}"));
    let module = trunk_ir::rewrite::Module::new(&ctx, module).unwrap();
    (ctx, module)
}

#[test]
fn concise_forms_write_the_operation_convention() {
    let (ctx, module) = parse(MODULE);
    assert_eq!(print_module(&ctx, module.op()), MODULE);

    let ops: Vec<_> = module.ops(&ctx).to_vec();
    assert_eq!(
        declared_convention(&ctx, ops[0]),
        Some(CallingConvention::Direct)
    );
    let outer = Func::from_op(&ctx, ops[2]).unwrap();
    assert_eq!(outer.convention(&ctx), CallingConvention::Cps);
    let signature = tribute_control::FuncSig::from_type_ref(&ctx, outer.r#type(&ctx)).unwrap();
    assert_eq!(signature.assigned_convention(&ctx), None);
    let body = ctx.region(outer.body(&ctx)).blocks[0];
    let lambda = Lambda::from_op(&ctx, ctx.block(body).ops[0]).unwrap();
    assert_eq!(lambda.convention(&ctx), CallingConvention::Cps);
    assert_eq!(lambda.captures(&ctx).len(), 1);
}

#[test]
fn a_concise_function_must_write_its_convention() {
    let mut ctx = IrContext::new();
    let input = r#"core.module @test {
  tribute_cps.func @f(%x: core.i32) -> core.i32 {
    tribute_control.return %x
  }
}
"#;
    assert!(parse_module(&mut ctx, input).is_err());
}

#[test]
fn a_convention_must_be_a_keyword() {
    let (ctx, module) = parse(
        r#"core.module @test {
  tribute_cps.func {convention = "fast", sym_name = "f", type = tribute_control.func_sig<() -> core.i32>}
}
"#,
    );
    let op = module.ops(&ctx)[0];
    let errors = OpDef::of(&ctx, op).unwrap().verify(&ctx, op);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("`convention` must be one of"),
        "{errors:?}"
    );
    assert!(
        print_module(&ctx, module.op()).contains("tribute_cps.func {"),
        "an invalid convention prints generically"
    );
}

fn messages(input: &str) -> Vec<String> {
    let (ctx, module) = parse(input);
    tribute_control::validate(&ctx, module, &[], &mut Default::default())
        .errors
        .into_iter()
        .map(|error| error.message)
        .collect()
}

#[test]
fn a_specialized_module_validates() {
    assert_eq!(messages(MODULE), Vec::<String>::new());
}

#[test]
fn a_module_holds_one_form_of_its_callables() {
    let errors = messages(
        r#"core.module @test {
  tribute_cps.func @f(%x: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %x
  }
  tribute_control.func @g(%x: core.i32) -> core.i32 {
    tribute_control.return %x
  }
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("class specialization replaces every")),
        "{errors:?}"
    );
}

#[test]
fn an_indirect_tail_call_is_not_stronger_than_its_callable() {
    let errors = messages(
        r#"core.module @test {
  tribute_cps.func @f(%k: tribute_control.func_sig<() -> core.i32>) -> core.i32 convention(direct) {
    tribute_cps.tail_call_indirect %k {convention = "cps"}
  }
}
"#,
    );
    assert_eq!(
        errors,
        ["callee convention must not be stronger than the enclosing callable"]
    );
}

#[test]
fn a_function_reference_is_not_weaker_than_its_instance() {
    let errors = messages(
        r#"core.module @test {
  tribute_cps.func @target() -> core.i32 convention(cps) {
    %0 = arith.const {value = 1} : core.i32
    tribute_control.return %0
  }
  tribute_cps.func @f() -> core.i32 convention(direct) {
    %0 = tribute_cps.func_ref {convention = "direct", func_ref = @target} : tribute_control.func_sig<() -> core.i32>
    %1 = tribute_cps.call_indirect %0 {convention = "direct"} : core.i32
    tribute_control.return %1
  }
}
"#,
    );
    assert_eq!(
        errors,
        ["func_ref result convention must be at least as strong as its target"]
    );
}

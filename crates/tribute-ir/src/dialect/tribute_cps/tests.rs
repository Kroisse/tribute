use trunk_ir::op_def::OpDef;
use trunk_ir::ops::DialectOp;
use trunk_ir::parser::parse_module;
use trunk_ir::printer::print_module;

use super::*;
use crate::dialect::tribute_control;

const MODULE: &str = r#"core.module @test {
  !t0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !t1 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 0}}>

  tribute_cps.func @declared(%arg0: core.i32) -> core.i32 convention(direct)
  tribute_cps.func_table @"declared$8" = {direct = @declared}
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
      %3 = tribute_cps.func_ref @"declared$8" : !t0
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
    let outer = Func::from_op(&ctx, ops[3]).unwrap();
    assert_eq!(outer.convention(&ctx), CallingConvention::Cps);
    let body = ctx.region(outer.body(&ctx)).blocks[0];
    let lambda = Lambda::from_op(&ctx, ctx.block(body).ops[0]).unwrap();
    itertools::assert_equal(lambda.conventions(&ctx), [CallingConvention::Cps]);
    assert_eq!(lambda.captures(&ctx).len(), 1);
}

const TWO_BODIES: &str = r#"core.module @test {
  !t0 = tribute_control.func_sig<(core.i32) -> core.i32>

  tribute_cps.func @outer(%0: core.i32) -> core.i32 convention(cps) {
      %1 = tribute_cps.lambda(%2: core.i32) -> core.i32 captures [%0] entries {
        direct {
            tribute_control.return %0
        }
        cps {
            tribute_control.return %2
        }
      }
      tribute_cps.tail_call_indirect %1, %0 {convention = "cps"}
  }
}
"#;

#[test]
fn a_lambda_writes_a_body_per_convention() {
    let (ctx, module) = parse(TWO_BODIES);
    assert_eq!(print_module(&ctx, module.op()), TWO_BODIES);
    let outer = Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
    let body = ctx.region(outer.body(&ctx)).blocks[0];
    let lambda = Lambda::from_op(&ctx, ctx.block(body).ops[0]).unwrap();
    itertools::assert_equal(
        lambda.conventions(&ctx),
        [CallingConvention::Direct, CallingConvention::Cps],
    );
    let bodies: Vec<_> = lambda.bodies(&ctx).collect();
    assert_eq!(bodies.len(), 2);
    // Each body takes the parameters, and the bodies share the captures.
    for body in bodies {
        let entry = ctx.region(body).blocks[0];
        assert_eq!(ctx.block_args(entry).len(), 1);
    }
    assert_eq!(messages(TWO_BODIES), Vec::<String>::new());
}

#[test]
fn a_lambda_names_each_body_once_in_order() {
    for conventions in [
        &[CallingConvention::Cps, CallingConvention::Direct][..],
        &[CallingConvention::Direct, CallingConvention::Direct],
        &[CallingConvention::Direct],
    ] {
        let (mut ctx, module) = parse(TWO_BODIES);
        let outer = Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let body = ctx.region(outer.body(&ctx)).blocks[0];
        let lambda = ctx.block(body).ops[0];
        let list = conventions
            .iter()
            .map(|convention| ctx.string_attr(convention.keyword()))
            .collect();
        ctx.op_attrs_mut(lambda)
            .insert(CONVENTIONS_ATTR, trunk_ir::types::Attribute::List(list));
        let errors = OpDef::of(&ctx, lambda).unwrap().verify(&ctx, lambda);
        assert!(
            errors.iter().any(|error| error
                .to_string()
                .contains("`conventions` must be strictly increasing")),
            "{conventions:?}: {errors:?}"
        );
    }
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

/// A module whose function table enters `@target`, an instance of
/// convention `target`, through `entry`.
fn table_module(target: &str, entry: &str) -> String {
    format!(
        r#"core.module @test {{
  tribute_cps.func @target() -> core.i32 convention({target}) {{
    %0 = arith.const {{value = 1}} : core.i32
    tribute_control.return %0
  }}
  tribute_cps.func_table @"target$8" = {{{entry} = @target}}
  tribute_cps.func @f() -> core.i32 convention(direct) {{
    %0 = tribute_cps.func_ref {{func_ref = @"target$8"}} : tribute_control.func_sig<() -> core.i32>
    %1 = tribute_cps.call_indirect %0 {{convention = "direct"}} : core.i32
    tribute_control.return %1
  }}
}}
"#
    )
}

#[test]
fn a_function_table_entry_has_the_convention_of_its_target() {
    assert_eq!(
        messages(&table_module("direct", "direct")),
        Vec::<String>::new()
    );
    for entry in ["direct", "cps"] {
        assert_eq!(
            messages(&table_module("evidence_direct", entry)),
            [format!(
                "function table entry `{entry}` is not the convention of its target @target"
            )]
        );
    }
}

/// A classless tail can make the target `Cps` while the value's row needs
/// a weaker entry.
#[test]
fn a_function_table_entry_may_be_weaker_than_a_cps_target() {
    assert_eq!(
        messages(&table_module("cps", "direct")),
        Vec::<String>::new()
    );
}

#[test]
fn a_function_reference_names_a_function_table() {
    let errors = messages(
        r#"core.module @test {
  tribute_cps.func @target() -> core.i32 convention(direct) {
    %0 = arith.const {value = 1} : core.i32
    tribute_control.return %0
  }
  tribute_cps.func @f() -> core.i32 convention(direct) {
    %0 = tribute_cps.func_ref {func_ref = @target} : tribute_control.func_sig<() -> core.i32>
    %1 = tribute_cps.call_indirect %0 {convention = "direct"} : core.i32
    tribute_control.return %1
  }
}
"#,
    );
    assert!(
        errors.contains(&"unresolved function table @target".to_owned()),
        "{errors:?}"
    );
}

#[test]
fn a_function_table_names_its_targets_by_convention_keyword() {
    let (ctx, module) = parse(
        r#"core.module @test {
  tribute_cps.func_table @t = {cps = @a, fast = @b}
}
"#,
    );
    let op = module.ops(&ctx)[0];
    let errors = OpDef::of(&ctx, op).unwrap().verify(&ctx, op);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].to_string().contains("convention keywords"),
        "{errors:?}"
    );
}

/// A body after the first reads the parameters the header names, and
/// makes transfers through its own convention.
#[test]
fn every_body_takes_the_parameters_and_its_own_convention() {
    let input = r#"core.module @test {
  !t0 = tribute_control.func_sig<() -> core.i32>
  !t1 = tribute_control.func_sig<(!t0) -> core.i32>

  tribute_cps.func @outer(%0: core.i32) -> core.i32 convention(cps) {
      %1 = tribute_cps.lambda(%2: !t0) -> core.i32 captures [%0] entries {
        direct {
            tribute_control.return %0
        }
        cps {
            %3 = tribute_cps.call_indirect %2 {convention = "cps"} : core.i32
            tribute_control.return %3
        }
      }
      tribute_control.return %0
  }
}
"#;
    let (ctx, module) = parse(input);
    assert_eq!(print_module(&ctx, module.op()), input);
    assert_eq!(messages(input), Vec::<String>::new());

    // A transfer in the Direct body may not be stronger than that body.
    let stronger = input.replace(
        "direct {\n            tribute_control.return %0",
        "direct {\n            tribute_cps.tail_call_indirect %2 {convention = \"cps\"}",
    );
    assert!(
        messages(&stronger)
            .iter()
            .any(|message| message.contains("must not be stronger than the enclosing callable")),
        "{:?}",
        messages(&stronger)
    );
}

#[test]
fn a_function_reference_writes_its_target_and_row_arguments() {
    let input = r#"core.module @test {
  !t0 = tribute_control.func_sig<() -> core.i32, {ability_row = {tail = 0}}>

  tribute_cps.func @f['a0]() -> core.i32 row(| 'a0) convention(cps) {
      %0 = tribute_cps.func_ref @inner::@"g$8" rows [{tail = 0}] : !t0
      %1 = tribute_control.func_ref @h : !t0
      tribute_cps.tail_call_indirect %0 {convention = "cps"}
  }
}
"#;
    let (ctx, module) = parse(input);
    assert_eq!(print_module(&ctx, module.op()), input);
}

/// A reference and its table's targets are in one module, where
/// legalization builds the table's entries.
#[test]
fn a_function_table_and_its_targets_are_in_the_module_of_its_references() {
    let errors = messages(
        r#"core.module @test {
  core.module @inner {
    tribute_cps.func @id() -> core.i32 convention(direct) {
      %0 = arith.const {value = 1} : core.i32
      tribute_control.return %0
    }
    tribute_cps.func_table @"id$8" = {direct = @id}
  }
  tribute_cps.func_table @"outer$8" = {direct = @inner::@id}
  tribute_cps.func @f() -> core.i32 convention(direct) {
    %0 = tribute_cps.func_ref @inner::@"id$8" : tribute_control.func_sig<() -> core.i32>
    %1 = tribute_cps.call_indirect %0 {convention = "direct"} : core.i32
    tribute_control.return %1
  }
}
"#,
    );
    assert_eq!(
        errors,
        [
            "function table target @inner::id must be in the table's own module",
            "func_ref @inner::id$8 must name a table of its own module",
        ]
    );
}

#[test]
fn the_targets_of_a_function_table_share_one_source_signature() {
    let errors = messages(
        r#"core.module @test {
  tribute_cps.func @a() -> core.i32 convention(direct) {
    %0 = arith.const {value = 1} : core.i32
    tribute_control.return %0
  }
  tribute_cps.func @b(%x: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %x
  }
  tribute_cps.func_table @t = {direct = @a, cps = @b}
}
"#,
    );
    assert_eq!(
        errors,
        ["function table targets must share one source signature"]
    );
}

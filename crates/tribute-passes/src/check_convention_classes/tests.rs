use trunk_ir::parser::parse_test_module;

use super::*;

/// The rules `input` disagrees with, in report order.
fn rules(input: &str) -> Vec<&'static str> {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    check(&ctx, module)
        .into_iter()
        .map(|disagreement| disagreement.rule)
        .collect()
}

const CTL: &str = r#"  ability.decl {sym_name = "Ctl"} {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }"#;

/// `app` has one class variable, the row of its callback; `main` selects
/// its `Direct` instance with a pure lambda.
fn calls(callback: &str, own: &str, lambda: &str, callee: &str) -> String {
    format!(
        r#"core.module @m {{
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {{ability_row = {{tail = 0}}}}>
{CTL}
  tribute_cps.func @"app$9D"['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) convention({own}) {{
    %r = tribute_cps.call_indirect %f, %x {{convention = "{callback}"}} : core.i32
    tribute_control.return %r
  }}
  tribute_cps.func @main(%x: core.i32) -> core.i32 convention(direct) {{
    %f = tribute_cps.lambda(%y: core.i32) -> core.i32 captures [] entries {{
      {lambda} {{
        tribute_control.return %y
      }}
    }}
    %r = tribute_control.call %x, %f {{callee = @"{callee}", row_args = [{{}}]}} : core.i32
    tribute_control.return %r
  }}
}}"#
    )
}

#[test]
fn a_reference_selects_the_instance_of_its_rows() {
    assert_eq!(
        rules(&calls("direct", "direct", "direct", "app$9D")),
        [""; 0]
    );
    assert_eq!(
        rules(&calls("direct", "direct", "direct", "app")),
        ["reference"]
    );
}

#[test]
fn an_instance_has_its_classes_in_its_transfers_and_convention() {
    assert_eq!(
        rules(&calls("evidence_direct", "direct", "direct", "app$9D")),
        ["transfer"]
    );
    assert_eq!(rules(&calls("direct", "cps", "direct", "app$9D")), ["own"]);
    assert_eq!(
        rules(&calls("direct", "direct", "cps", "app$9D")),
        ["lambda"]
    );
}

#[test]
fn a_suffix_names_one_class_per_class_variable() {
    // The reference names the instance, whose name claims two classes.
    let input = calls("direct", "direct", "direct", "app$9DD")
        .replace("@\"app$9D\"['a0]", "@\"app$9DD\"['a0]");
    assert_eq!(rules(&input), ["suffix", "reference"]);
}

#[test]
fn instances_of_one_definition_select_the_same_class_variables() {
    let input = calls("direct", "direct", "direct", "app$9D").replace(
        "  tribute_cps.func @main",
        "  tribute_cps.func @\"app$9E\"['a0](%x: core.i32) -> core.i32 row(| 'a0) convention(cps) {\n    tribute_control.return %x\n  }\n  tribute_cps.func @main",
    );
    assert!(rules(&input).contains(&"siblings"), "{:?}", rules(&input));
}

const LOCAL: &str = r#"core.module @m {
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !lc = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {instances = [@Ctl]}}>
  !l1 = tribute_control.func_sig<(core.i32) -> core.i32, {ability_row = {tail = 1}}>
  !h0 = tribute_control.func_sig<() -> core.i32>
CTL
  tribute_cps.func @f[local 'a0, local 'a1](%x: core.i32) -> core.i32 convention(direct) {
    %g = tribute_cps.lambda['a0](%y: core.i32) -> core.i32 row(| 'a0) captures [] entries {
      direct {
        tribute_control.return %y
      }
    }
    %h = tribute_cps.lambda['a1]() -> core.i32 row(| 'a1) captures [%g] entries {
      direct {
        %s = tribute_control.instantiate %g {row_args = [{tail = 1}]} : !l1
        %c = arith.const {value = 1} : core.i32
        %v = tribute_cps.call_indirect %s, %c {convention = "direct"} : core.i32
        tribute_control.return %v
      }
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !l0
    %i = tribute_control.instantiate %h {row_args = [{}]} : !h0
    %r = tribute_cps.call_indirect %i {convention = "direct"} : core.i32
    USE
    tribute_control.return %r
  }
}"#;

#[test]
fn a_local_class_settles_from_direct_through_other_locals() {
    let input = LOCAL.replace("CTL", CTL).replace("USE", "");
    assert_eq!(rules(&input), [""; 0]);
}

#[test]
fn every_instantiation_of_a_lambda_selects_the_same_classes() {
    let input = LOCAL.replace("CTL", CTL).replace(
        "USE",
        "%b = tribute_control.instantiate %g {row_args = [{instances = [@Ctl]}]} : !lc",
    );
    let found = rules(&input);
    assert!(found.contains(&"instantiate"), "{found:?}");
}

#[test]
fn a_func_ref_has_the_class_of_the_row_of_its_value() {
    // The value's row is wider than the target's, so its class is stronger.
    let input = |convention: &str| {
        format!(
            r#"core.module @m {{
  !open = tribute_control.func_sig<(core.i32) -> core.i32, {{ability_row = {{instances = [@Ctl]}}}}>
{CTL}
  tribute_cps.func @pure(%x: core.i32) -> core.i32 convention(direct) {{
    tribute_control.return %x
  }}
  tribute_cps.func @f(%x: core.i32) -> core.i32 convention(direct) {{
    %p = tribute_cps.func_ref {{convention = "{convention}", func_ref = @pure}} : !open
    tribute_control.return %x
  }}
}}"#
        )
    };
    assert_eq!(rules(&input("cps")), [""; 0]);
    assert_eq!(rules(&input("direct")), ["func_ref"]);
}

#[test]
fn an_inferred_row_reads_its_tail_only_when_it_is_a_class_variable() {
    let inferred = |convention: &str| {
        format!(
            r#"core.module @m {{
  tribute_cps.func @g['a0](%x: core.i32) -> core.i32 row(| 'a0) inferred convention({convention}) {{
    tribute_control.return %x
  }}
}}"#
        )
    };
    assert_eq!(rules(&inferred("direct")), [""; 0]);
    assert_eq!(rules(&inferred("cps")), ["own"]);
}

#[test]
fn the_pass_fails_with_every_disagreement() {
    let run = |input: &str| {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let module =
            trunk_ir::dialect::core::Module::from_op(&ctx, module.op()).expect("a core.module");
        let mut pm = trunk_ir::pass::PassManager::new();
        pm.add_pass(CheckConventionClasses);
        pm.run(
            &mut ctx,
            module,
            &mut trunk_ir::analysis::AnalysisCache::new(),
        )
        .map_err(|error| error.to_string())
    };
    assert_eq!(run(&calls("direct", "direct", "direct", "app$9D")), Ok(()));
    let error = run(&calls("direct", "direct", "direct", "app")).unwrap_err();
    assert!(
        error.contains("check-convention-classes")
            && error
                .contains("convention classes disagree with rows: (reference) main refers to app,"),
        "{error}"
    );
}

#[test]
fn a_definition_without_a_body_has_no_class_variables() {
    // With a class variable, the closed row would select `external$9D`.
    let input = format!(
        r#"core.module @m {{
  !cb = tribute_control.func_sig<(core.i32) -> core.i32, {{ability_row = {{tail = 0}}}}>
  !closed = tribute_control.func_sig<(core.i32) -> core.i32>
{CTL}
  tribute_cps.func @external['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) convention(cps)
  tribute_cps.func @pure(%y: core.i32) -> core.i32 convention(direct) {{
    tribute_control.return %y
  }}
  tribute_cps.func @main(%x: core.i32) -> core.i32 convention(direct) {{
    %f = tribute_cps.func_ref {{convention = "direct", func_ref = @pure}} : !closed
    %r = tribute_control.call %x, %f {{callee = @external, row_args = [{{}}]}} : core.i32
    tribute_control.return %r
  }}
}}"#
    );
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, &input);
    let validation =
        tribute_ir::dialect::tribute_control::validate(&ctx, module, &[], &mut Default::default());
    assert!(validation.is_ok(), "{validation}");
    assert_eq!(rules(&input), [""; 0]);
}

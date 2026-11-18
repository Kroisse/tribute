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

const CTL: &str = r#"  ability.decl @Ctl {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }"#;

/// The bodies of a lambda that returns its parameter `%y`, one per
/// convention.
fn identity_bodies(conventions: &[&str]) -> String {
    conventions
        .iter()
        .map(|convention| {
            format!("      {convention} {{\n        tribute_control.return %y\n      }}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `app` has one class variable, the row of its callback; `main` selects
/// its `Direct` instance with a pure lambda that has a body for each of
/// `lambda`.
fn calls(callback: &str, own: &str, lambda: &[&str], callee: &str) -> String {
    let bodies = identity_bodies(lambda);
    format!(
        r#"core.module @m {{
  !cb = tribute_control.func_sig<(core.i32) -> core.i32 row(| 'a0)>
{CTL}
  tribute_cps.func @"app$9D"['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) convention({own}) {{
    %r = tribute_cps.call_indirect %f(%x) attributes {{convention = "{callback}"}} : (core.i32) -> core.i32 row(| 'a0)
    tribute_control.return %r
  }}
  tribute_cps.func @main(%x: core.i32) -> core.i32 convention(direct) {{
    %f = tribute_cps.lambda(%y: core.i32) -> core.i32 entries {{
{bodies}
    }}
    %r = tribute_control.call @"{callee}"(%x, %f) attributes {{row_args = [{{}}]}} : (core.i32, tribute_control.func_sig<(core.i32) -> core.i32>) -> core.i32
    tribute_control.return %r
  }}
}}"#
    )
}

/// The body of a pure lambda whose own flow hands no frame on: its own
/// class alone.
const PURE: &[&str] = &["direct"];

#[test]
fn a_reference_selects_the_instance_of_its_rows() {
    assert_eq!(rules(&calls("direct", "direct", PURE, "app$9D")), [""; 0]);
    assert_eq!(
        rules(&calls("direct", "direct", PURE, "app")),
        ["reference"]
    );
}

#[test]
fn an_instance_has_its_classes_in_its_transfers_and_convention() {
    assert_eq!(
        rules(&calls("evidence_direct", "direct", PURE, "app$9D")),
        ["transfer"]
    );
    assert_eq!(rules(&calls("direct", "cps", PURE, "app$9D")), ["own"]);
    assert_eq!(
        rules(&calls("direct", "direct", &["cps"], "app$9D")),
        ["lambda"]
    );
}

/// A lambda whose own flow hands no frame on has no `Cps` entry body.
#[test]
fn a_lambda_without_a_frame_transfer_has_no_cps_entry_body() {
    assert_eq!(
        rules(&calls("direct", "direct", &["direct", "cps"], "app$9D")),
        ["lambda"]
    );
}

/// `main` calls a lambda whose bodies are `OUTER`. A body of a convention
/// in `transfers` ends with a `become` of `id`'s instance for that flow;
/// the others return their parameter.
const TRANSFER: &str = r#"core.module @m {
  tribute_cps.func @id(%x: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %x
  }
  tribute_cps.func @"id$7"(%x: core.i32) -> core.i32 convention(cps) attributes {entry_of = "id"} {
    tribute_control.return %x
  }
  tribute_cps.func @main(%x: core.i32) -> core.i32 convention(direct) {
    %f = tribute_cps.lambda(%y: core.i32) -> core.i32 entries {
OUTER
    }
    %r = tribute_cps.call_indirect %f(%x) attributes {convention = "direct"} : (core.i32) -> core.i32
    tribute_control.return %r
  }
}"#;

/// Bodies of a lambda over `%y`, one per convention; a body of a convention
/// in `transfers` ends with a `become` of `id`.
fn transfer_bodies(conventions: &[&str], transfers: &[&str]) -> String {
    conventions
        .iter()
        .map(|&convention| {
            let exit = match (transfers.contains(&convention), convention) {
                (false, _) => "tribute_control.return %y".to_owned(),
                (true, "cps") => {
                    r#"tribute_control.tail_call @"id$7"(%y) : (core.i32) -> ()"#.to_owned()
                }
                (true, _) => "tribute_control.tail_call @id(%y) : (core.i32) -> ()".to_owned(),
            };
            format!("      {convention} {{\n        {exit}\n      }}")
        })
        .join("\n")
}

/// A lambda whose own flow has a `become` has a `Cps` entry body.
#[test]
fn a_lambda_with_a_frame_transfer_has_a_cps_entry_body() {
    let input = |bodies: &[&str]| TRANSFER.replace("OUTER", &transfer_bodies(bodies, bodies));
    assert_eq!(rules(&input(&["direct"])), ["lambda"]);
    assert_eq!(rules(&input(&["direct", "cps"])), [""; 0]);
}

/// A nested lambda's `become` belongs to its own flow, not to the lambda
/// that holds it.
#[test]
fn a_nested_frame_transfer_gives_only_the_nested_lambda_a_cps_entry_body() {
    let input = |outer: &[&str], inner: &[&str]| {
        let inner = transfer_bodies(inner, inner)
            .replace("%y", "%z")
            .replace('\n', "\n    ");
        let outer = outer
            .iter()
            .map(|convention| {
                format!(
                    "      {convention} {{\n        %g = tribute_cps.lambda(%z: core.i32) -> core.i32 entries {{\n    {inner}\n        }}\n        tribute_control.return %y\n      }}"
                )
            })
            .join("\n");
        TRANSFER.replace("OUTER", &outer)
    };
    assert_eq!(rules(&input(&["direct"], &["direct", "cps"])), [""; 0]);
    assert_eq!(
        rules(&input(&["direct", "cps"], &["direct", "cps"])),
        ["lambda"]
    );
    assert_eq!(rules(&input(&["direct"], &["direct"])), ["lambda"]);
}

/// An instance of an upper bound calls through the bound, and a lambda over
/// its class variable has a body for each class up to it.
#[test]
fn an_instance_of_an_upper_bound_has_a_body_per_class_up_to_it() {
    let input = |bodies: &str| {
        format!(
            r#"core.module @m {{
  !cb = tribute_control.func_sig<(core.i32) -> core.i32 row(| 'a0)>
  tribute_cps.func @"wrap$9e"['a0](%x: core.i32) -> core.i32 row(| 'a0) convention(evidence_direct) {{
    %f = tribute_cps.lambda(%y: core.i32) -> core.i32 row(| 'a0) entries {{
{bodies}
    }}
    %r = tribute_cps.call_indirect %f(%x) attributes {{convention = "evidence_direct"}} : (core.i32) -> core.i32 row(| 'a0)
    tribute_control.return %r
  }}
}}"#
        )
    };
    assert_eq!(rules(&input(&identity_bodies(&["direct"]))), ["lambda"]);
    let both = identity_bodies(&["direct", "evidence_direct"]);
    assert_eq!(rules(&input(&both)), [""; 0]);
}

#[test]
fn a_suffix_names_one_class_per_class_variable() {
    // The reference names the instance, whose name claims two classes.
    let input =
        calls("direct", "direct", PURE, "app$9DD").replace("@\"app$9D\"['a0]", "@\"app$9DD\"['a0]");
    assert_eq!(rules(&input), ["suffix", "reference"]);
}

#[test]
fn instances_of_one_definition_select_the_same_class_variables() {
    let input = calls("direct", "direct", PURE, "app$9D").replace(
        "  tribute_cps.func @main",
        "  tribute_cps.func @\"app$9E\"['a0](%x: core.i32) -> core.i32 row(| 'a0) convention(cps) {\n    tribute_control.return %x\n  }\n  tribute_cps.func @main",
    );
    assert!(rules(&input).contains(&"siblings"), "{:?}", rules(&input));
}

const LOCAL: &str = r#"core.module @m {
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !la = tribute_control.func_sig<(core.i32) -> core.i32 row(@Ask)>
  !l1 = tribute_control.func_sig<(core.i32) -> core.i32 row(| 'a1)>
  !h0 = tribute_control.func_sig<() -> core.i32>
  ability.decl @Ask {
    ability.operation {kind = "fn", op_name = "ask", param_types = [], result_type = core.i32}
  }
  tribute_cps.func @f[local 'a0, local 'a1](%x: core.i32) -> core.i32 convention(direct) {
    %g = tribute_cps.lambda['a0](%y: core.i32) -> core.i32 row(| 'a0) entries {
BODIES
    }
    %h = tribute_cps.lambda['a1]() -> core.i32 row(| 'a1) entries {
      direct {
        %s = tribute_control.instantiate %g {row_args = [{tail = 1}]} : !l1
        %c = arith.constant 1 : core.i32
        %v = tribute_cps.call_indirect %s(%c) attributes {convention = "direct"} : (core.i32) -> core.i32 row(| 'a1)
        tribute_control.return %v
      }
    }
    %a = tribute_control.instantiate %g {row_args = [{}]} : !l0
    %i = tribute_control.instantiate %h {row_args = [{}]} : !h0
    %r = tribute_cps.call_indirect %i() attributes {convention = "direct"} : () -> core.i32
    USE
    tribute_control.return %r
  }
}"#;

#[test]
fn a_local_class_settles_from_direct_through_other_locals() {
    let input = LOCAL
        .replace("BODIES", &identity_bodies(PURE))
        .replace("USE", "");
    assert_eq!(rules(&input), [""; 0]);
}

#[test]
fn a_lambda_has_a_body_for_each_class_its_instantiations_select() {
    let input = |bodies: &[&str]| {
        LOCAL.replace("BODIES", &identity_bodies(bodies)).replace(
            "USE",
            "%b = tribute_control.instantiate %g {row_args = [{instances = [@Ask]}]} : !la",
        )
    };
    assert_eq!(rules(&input(&["direct"])), ["lambda"]);
    assert_eq!(rules(&input(&["direct", "evidence_direct"])), [""; 0]);
}

/// Instantiations that select one body may pass different classes for a
/// binder, which that body reads as their upper bound.
#[test]
fn instantiations_that_select_one_class_may_pass_other_classes() {
    let input = |second: &str| {
        format!(
            r#"core.module @m {{
  !l0 = tribute_control.func_sig<(core.i32) -> core.i32 row(@Ctl)>
  !la = tribute_control.func_sig<(core.i32) -> core.i32 row(@Ask, @Ctl)>
  ability.decl @Ask {{
    ability.operation {{kind = "fn", op_name = "ask", param_types = [], result_type = core.i32}}
  }}
{CTL}
  tribute_cps.func @f[local 'a0](%x: core.i32) -> core.i32 convention(direct) {{
    %g = tribute_cps.lambda['a0](%y: core.i32) -> core.i32 row(@Ctl | 'a0) entries {{
      cps {{
        tribute_control.return %y
      }}
    }}
    %a = tribute_control.instantiate %g {{row_args = [{{}}]}} : !l0
    %b = tribute_control.instantiate %g {{row_args = [{second}]}} : {}
    tribute_control.return %x
  }}
}}"#,
            if second == "{}" { "!l0" } else { "!la" }
        )
    };
    assert_eq!(rules(&input("{}")), [""; 0]);
    assert_eq!(rules(&input("{instances = [@Ask]}")), [""; 0]);
}

/// The `Cps` entry instance of `pure`, the `cps` target of its table.
const PURE_ENTRY: &str = r#"tribute_cps.func @"pure$7"(%y: core.i32) -> core.i32 convention(cps) attributes {entry_of = "pure"} {
    tribute_control.return %y
  }"#;

#[test]
fn a_func_ref_enters_its_table_no_stronger_than_the_row_of_its_value() {
    let input = |value: &str, entries: &str| {
        format!(
            r#"core.module @m {{
  !open = tribute_control.func_sig<(core.i32) -> core.i32 row(@Ctl)>
  !closed = tribute_control.func_sig<(core.i32) -> core.i32>
{CTL}
  tribute_cps.func @pure(%y: core.i32) -> core.i32 convention(direct) {{
    tribute_control.return %y
  }}
  {PURE_ENTRY}
  tribute_cps.func_table @"pure$8" = {{{entries}}}
  tribute_cps.func @f(%x: core.i32) -> core.i32 convention(direct) {{
    %p = tribute_cps.func_ref @"pure$8" : {value}
    tribute_control.return %x
  }}
}}"#
        )
    };
    let both = r#"cps = @"pure$7", direct = @pure"#;
    // A value whose row is wider than the target's enters a weaker entry
    // through an adapter.
    assert_eq!(rules(&input("!open", both)), [""; 0]);
    assert_eq!(rules(&input("!closed", both)), [""; 0]);
    assert_eq!(rules(&input("!closed", r#"cps = @"pure$7""#)), ["func_ref"]);
    // The table of a definition with a body has a `cps` target.
    assert_eq!(rules(&input("!closed", "direct = @pure")), ["table"]);
}

#[test]
fn an_own_row_reads_its_tail_only_when_it_is_a_class_variable() {
    let own = |inferred: &str, convention: &str| {
        format!(
            r#"core.module @m {{
  tribute_cps.func @g['a0](%x: core.i32) -> core.i32 row(| 'a0){inferred} convention({convention}) {{
    tribute_control.return %x
  }}
}}"#
        )
    };
    for inferred in ["", " inferred"] {
        assert_eq!(rules(&own(inferred, "direct")), [""; 0]);
        assert_eq!(rules(&own(inferred, "cps")), ["own"]);
    }
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
    assert_eq!(run(&calls("direct", "direct", PURE, "app$9D")), Ok(()));
    let error = run(&calls("direct", "direct", PURE, "app")).unwrap_err();
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
  !cb = tribute_control.func_sig<(core.i32) -> core.i32 row(| 'a0)>
  !closed = tribute_control.func_sig<(core.i32) -> core.i32>
{CTL}
  tribute_cps.func @external['a0](%x: core.i32, %f: !cb) -> core.i32 row(| 'a0) convention(cps)
  tribute_cps.func @pure(%y: core.i32) -> core.i32 convention(direct) {{
    tribute_control.return %y
  }}
  {PURE_ENTRY}
  tribute_cps.func_table @"pure$8" = {{cps = @"pure$7", direct = @pure}}
  tribute_cps.func @main(%x: core.i32) -> core.i32 convention(direct) {{
    %f = tribute_cps.func_ref @"pure$8" : !closed
    %r = tribute_control.call @external(%x, %f) attributes {{row_args = [{{}}]}} : (core.i32, !closed) -> core.i32
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

/// `step`'s row implies `Direct`. The `op` arm of its tail handle is a
/// `Cps` flow, so its `become` names `callee`, which `entry` defines.
fn arm_become(callee: &str, entry: &str) -> String {
    format!(
        r#"core.module @m {{
{CTL}
  tribute_cps.func @step(%x: core.i32) -> core.i32 convention(direct) {{
    %r = tribute_control.handle : core.i32 {{
      %c = tribute_control.perform op @Ctl @ctl() : () -> core.i32
      tribute_control.yield %c
    }} do(%v: core.i32) {{
        tribute_control.yield %v
    }} op @Ctl @ctl() : core.i32 {{
        tribute_control.tail_call @"{callee}"(%x) : (core.i32) -> ()
    }}
    tribute_control.return %r
  }}
  {entry}
}}"#
    )
}

/// The `Cps` entry instance of `step`, declaring `convention` and
/// `entry_of`.
fn step_entry(name: &str, convention: &str, entry_of: &str) -> String {
    format!(
        r#"tribute_cps.func @"{name}"(%x: core.i32) -> core.i32 convention({convention}) attributes {{entry_of = "{entry_of}"}} {{
    tribute_control.return %x
  }}"#
    )
}

/// A `become` in a `Cps` flow enters the `Cps` entry instance of a callee
/// whose row implies a weaker convention, which declares `Cps`.
#[test]
fn a_become_in_a_cps_flow_names_the_cps_entry_instance() {
    let entry = step_entry("step$7", "cps", "step");
    assert_eq!(rules(&arm_become("step$7", &entry)), [""; 0]);
    assert_eq!(rules(&arm_become("step", &entry)), ["reference"]);
    let direct = step_entry("step$7", "direct", "step");
    assert_eq!(rules(&arm_become("step$7", &direct)), ["own"]);
    let misnamed = step_entry("step$8", "cps", "step");
    assert_eq!(
        rules(&arm_become("step$8", &misnamed)),
        ["reference", "entry"]
    );
}

/// The call that makes the body result of `step`'s tail handle names
/// `callee` and calls `%k` through `convention`, and `step$7` is `step`'s
/// `Cps` entry instance.
fn body_result_calls(callee: &str, convention: &str) -> String {
    format!(
        r#"core.module @m {{
  !k = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_cps.func @step(%x: core.i32) -> core.i32 convention(direct) {{
    tribute_control.return %x
  }}
  {entry}
  tribute_cps.func @named(%x: core.i32) -> core.i32 convention(direct) {{
    %r = tribute_control.handle : core.i32 {{
      %c = tribute_control.call @"{callee}"(%x) : (core.i32) -> core.i32
      tribute_control.yield %c
    }} do(%v: core.i32) {{
        tribute_control.yield %v
    }}
    tribute_control.return %r
  }}
  tribute_cps.func @value(%x: core.i32, %k: !k) -> core.i32 convention(direct) {{
    %r = tribute_control.handle : core.i32 {{
      %c = tribute_cps.call_indirect %k(%x) attributes {{convention = "{convention}"}} : (core.i32) -> core.i32
      tribute_control.yield %c
    }} do(%v: core.i32) {{
        tribute_control.yield %v
    }}
    tribute_control.return %r
  }}
}}"#,
        entry = step_entry("step$7", "cps", "step"),
    )
}

/// A call that hands the frame of a tail handle's body on enters a callee
/// whose row implies a weaker convention at its `Cps` entry instance, and
/// a callable value at its `Cps` entry.
#[test]
fn a_body_result_call_of_a_tail_handle_enters_as_cps() {
    assert_eq!(rules(&body_result_calls("step$7", "cps")), [""; 0]);
    assert_eq!(
        rules(&body_result_calls("step", "direct")),
        ["reference", "transfer"]
    );
}

/// An indirect `become` in a `Cps` flow enters the value's `Cps` entry,
/// whatever class the callee type reads.
#[test]
fn an_indirect_become_in_a_cps_flow_is_cps() {
    let input = |convention: &str| {
        format!(
            r#"core.module @m {{
  !k = tribute_control.func_sig<(core.i32) -> core.i32>
{CTL}
  tribute_cps.func @step(%x: core.i32, %k: !k) -> core.i32 convention(direct) {{
    %r = tribute_control.handle : core.i32 {{
      %c = tribute_control.perform op @Ctl @ctl() : () -> core.i32
      tribute_control.yield %c
    }} do(%v: core.i32) {{
        tribute_control.yield %v
    }} op @Ctl @ctl() : core.i32 {{
        tribute_cps.tail_call_indirect %k(%x) attributes {{convention = "{convention}"}} : (core.i32) -> core.i32
    }}
    tribute_control.return %r
  }}
}}"#
        )
    };
    assert_eq!(rules(&input("cps")), [""; 0]);
    assert_eq!(rules(&input("direct")), ["transfer"]);
}

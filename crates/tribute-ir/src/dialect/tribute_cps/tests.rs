use trunk_ir::op_def::OpDef;
use trunk_ir::ops::DialectOp;
use trunk_ir::parser::parse_module;

use super::*;
use crate::dialect::tribute_control;

const MODULE: &str = r#"core.module @test {
  !t0 = tribute_control.func_sig<(core.i32) -> core.i32>
  !t1 = tribute_control.func_sig<(core.i32) -> core.i32 row(| 'a0)>

  tribute_cps.func @declared(%arg0: core.i32) -> core.i32 convention(direct)
  tribute_cps.func_table @"declared$8" = {direct = @declared}
  tribute_cps.func @apply(%0: !t0, %1: core.i32) -> core.i32 convention(direct) {
      %2 = tribute_cps.call_indirect %0(%1) attributes {convention = "direct"} : (core.i32) -> core.i32
      tribute_control.return %2
  }
  tribute_cps.func @outer['a0](%0: core.i32) -> core.i32 row(| 'a0) convention(cps) attributes {visibility = "private"} {
      %1 = tribute_cps.lambda(%2: core.i32) -> core.i32 row(| 'a0) entries {
        cps {
            tribute_control.return %0
        }
      }
      %3 = tribute_cps.func_ref @"declared$8" : !t0
      tribute_cps.tail_call_indirect %1(%0) attributes {convention = "cps"} : (core.i32) -> core.i32 row(| 'a0)
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
    assert_eq!(module.op().display(&ctx).to_string(), MODULE);

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
}

const TWO_BODIES: &str = r#"core.module @test {
  !t0 = tribute_control.func_sig<(core.i32) -> core.i32>

  tribute_cps.func @outer(%0: core.i32) -> core.i32 convention(cps) {
      %1 = tribute_cps.lambda(%2: core.i32) -> core.i32 entries {
        direct {
            tribute_control.return %0
        }
        cps {
            tribute_control.return %2
        }
      }
      tribute_cps.tail_call_indirect %1(%0) attributes {convention = "cps"} : (core.i32) -> core.i32
  }
}
"#;

#[test]
fn a_lambda_writes_a_body_per_convention() {
    let (ctx, module) = parse(TWO_BODIES);
    assert_eq!(module.op().display(&ctx).to_string(), TWO_BODIES);
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
  "tribute_cps.func"() {convention = "fast", sym_name = "f", type = tribute_control.func_sig<() -> core.i32>}
}
"#,
    );
    let op = module.ops(&ctx)[0];
    let errors = OpDef::of(&ctx, op).unwrap().verify(&ctx, op);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .to_string()
            .contains("attribute `convention` must be one of"),
        "{errors:?}"
    );
    assert!(
        module
            .op()
            .display(&ctx)
            .to_string()
            .contains("\"tribute_cps.func\"() {"),
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
fn an_indirect_tail_call_is_not_stronger_than_its_flow() {
    let errors = messages(
        r#"core.module @test {
  tribute_cps.func @f(%k: tribute_control.func_sig<() -> core.i32>) -> core.i32 convention(direct) {
    tribute_cps.tail_call_indirect %k() attributes {convention = "cps"} : () -> core.i32
  }
}
"#,
    );
    assert_eq!(
        errors,
        ["callee convention must not be stronger than the flow that holds the transfer"]
    );
}

/// A tail transfer in a `Cps` flow enters its callee as `Cps`: the arms of
/// a handle in a `Direct` callable are one (`design/tribute-dialects.md`,
/// "Convention과 class 특수화").
#[test]
fn a_tail_transfer_in_a_cps_flow_enters_its_callee_as_cps() {
    let module = |convention: &str, callee: &str| {
        format!(
            r#"core.module @test {{
  ability.decl @Ctl {{
    ability.operation {{kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}}
  }}
  tribute_cps.func @g(%x: core.i32) -> core.i32 convention({callee}) {{
    tribute_control.return %x
  }}
  tribute_cps.func @f(%x: core.i32, %k: tribute_control.func_sig<(core.i32) -> core.i32>) -> core.i32 convention(direct) {{
    %r = tribute_control.handle : core.i32 {{
      %c = tribute_control.perform op @Ctl @ctl() : () -> core.i32
      tribute_control.yield %c
    }} do(%v: core.i32) {{
        tribute_cps.tail_call_indirect %k(%v) attributes {{convention = "{convention}"}} : (core.i32) -> core.i32
    }} op @Ctl @ctl() : core.i32 {{
        tribute_control.tail_call @g(%x) : (core.i32) -> ()
    }}
    tribute_control.return %r
  }}
}}
"#
        )
    };
    assert_eq!(messages(&module("cps", "cps")), Vec::<String>::new());
    assert_eq!(
        messages(&module("direct", "direct")),
        ["a tail transfer in a Cps flow must enter its callee as Cps"; 2]
    );
}

/// A module whose function table enters `@target`, an instance of
/// convention `target`, through `entry`.
fn table_module(target: &str, entry: &str) -> String {
    format!(
        r#"core.module @test {{
  tribute_cps.func @target() -> core.i32 convention({target}) {{
    %0 = arith.constant 1 : core.i32
    tribute_control.return %0
  }}
  tribute_cps.func_table @"target$8" = {{{entry} = @target}}
  tribute_cps.func @f() -> core.i32 convention(direct) {{
    %0 = tribute_cps.func_ref @"target$8" : tribute_control.func_sig<() -> core.i32>
    %1 = tribute_cps.call_indirect %0() attributes {{convention = "direct"}} : () -> core.i32
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
    %0 = arith.constant 1 : core.i32
    tribute_control.return %0
  }
  tribute_cps.func @f() -> core.i32 convention(direct) {
    %0 = tribute_cps.func_ref @target : tribute_control.func_sig<() -> core.i32>
    %1 = tribute_cps.call_indirect %0() attributes {convention = "direct"} : () -> core.i32
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
      %1 = tribute_cps.lambda(%2: !t0) -> core.i32 entries {
        direct {
            tribute_control.return %0
        }
        cps {
            %3 = tribute_cps.call_indirect %2() attributes {convention = "cps"} : () -> core.i32
            tribute_control.return %3
        }
      }
      tribute_control.return %0
  }
}
"#;
    let (ctx, module) = parse(input);
    assert_eq!(module.op().display(&ctx).to_string(), input);
    assert_eq!(messages(input), Vec::<String>::new());

    // A transfer in the Direct body may not be stronger than that body.
    let stronger = input.replace(
        "direct {\n            tribute_control.return %0",
        "direct {\n            tribute_cps.tail_call_indirect %2() attributes {convention = \"cps\"} : () -> core.i32",
    );
    assert!(
        messages(&stronger)
            .iter()
            .any(|message| message.contains("must not be stronger than the flow")),
        "{:?}",
        messages(&stronger)
    );
}

#[test]
fn a_function_reference_writes_its_target_and_row_arguments() {
    let input = r#"core.module @test {
  !t0 = tribute_control.func_sig<() -> core.i32 row(| 'a0)>

  tribute_cps.func @f['a0]() -> core.i32 row(| 'a0) convention(cps) {
      %0 = tribute_cps.func_ref @inner::@"g$8" rows [{tail = 0}] : !t0
      %1 = tribute_control.func_ref @h : !t0
      tribute_cps.tail_call_indirect %0() attributes {convention = "cps"} : () -> core.i32 row(| 'a0)
  }
}
"#;
    let (ctx, module) = parse(input);
    assert_eq!(module.op().display(&ctx).to_string(), input);
}

/// A reference and its table's targets are in one module, where
/// legalization builds the table's entries.
#[test]
fn a_function_table_and_its_targets_are_in_the_module_of_its_references() {
    let errors = messages(
        r#"core.module @test {
  core.module @inner {
    tribute_cps.func @id() -> core.i32 convention(direct) {
      %0 = arith.constant 1 : core.i32
      tribute_control.return %0
    }
    tribute_cps.func_table @"id$8" = {direct = @id}
  }
  tribute_cps.func_table @"outer$8" = {direct = @inner::@id}
  tribute_cps.func @f() -> core.i32 convention(direct) {
    %0 = tribute_cps.func_ref @inner::@"id$8" : tribute_control.func_sig<() -> core.i32>
    %1 = tribute_cps.call_indirect %0() attributes {convention = "direct"} : () -> core.i32
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

/// Each function holds one call; `passes_body_frame_on` holds for the call
/// that makes the body result of a handle in tail position, directly or
/// through a structured arm, in a function, an `op` arm, or a lambda.
const BODY_RESULT_CALLS: &str = r#"core.module @test {
  !k = tribute_control.func_sig<(core.i32) -> core.i32>
  ability.decl @Ctl {
    ability.operation {kind = "op", op_name = "ctl", param_types = [], result_type = core.i32}
  }
  tribute_cps.func @g(%x: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %x
  }
  tribute_cps.func @named(%x: core.i32) -> core.i32 convention(direct) {
    %r = tribute_control.handle : core.i32 {
      %c = tribute_control.call @g(%x) : (core.i32) -> core.i32
      tribute_control.yield %c
    } do(%v: core.i32) {
        tribute_control.yield %v
    }
    tribute_control.return %r
  }
  tribute_cps.func @value(%x: core.i32, %k: !k) -> core.i32 convention(direct) {
    %r = tribute_control.handle : core.i32 {
      %c = tribute_cps.call_indirect %k(%x) attributes {convention = "cps"} : (core.i32) -> core.i32
      tribute_control.yield %c
    } do(%v: core.i32) {
        tribute_control.yield %v
    }
    tribute_control.return %r
  }
  tribute_cps.func @select(%x: core.i32, %b: core.i1) -> core.i32 convention(direct) {
    %r = tribute_control.handle : core.i32 {
      %s = scf.if %b -> (core.i32) {
        %c = tribute_control.call @g(%x) : (core.i32) -> core.i32
        scf.yield %c
      } else {
        scf.yield %x
      }
      tribute_control.yield %s
    } do(%v: core.i32) {
        tribute_control.yield %v
    }
    tribute_control.return %r
  }
  tribute_cps.func @in_arm(%x: core.i32) -> core.i32 convention(direct) {
    %r = tribute_control.handle : core.i32 {
      %p = tribute_control.perform op @Ctl @ctl() : () -> core.i32
      tribute_control.yield %p
    } do(%v: core.i32) {
        tribute_control.yield %v
    } op @Ctl @ctl() : core.i32 {
        %i = tribute_control.handle : core.i32 {
          %c = tribute_control.call @g(%x) : (core.i32) -> core.i32
          tribute_control.yield %c
        } do(%w: core.i32) {
            tribute_control.yield %w
        }
        tribute_control.yield %i
    }
    tribute_control.return %r
  }
  tribute_cps.func @in_lambda(%x: core.i32) -> !k convention(direct) {
    %l = tribute_cps.lambda(%y: core.i32) -> core.i32 entries {
      direct {
        %r = tribute_control.handle : core.i32 {
          %c = tribute_control.call @g(%y) : (core.i32) -> core.i32
          tribute_control.yield %c
        } do(%v: core.i32) {
            tribute_control.yield %v
        }
        tribute_control.return %r
      }
    }
    tribute_control.return %l
  }
  tribute_cps.func @not_tail(%x: core.i32) -> core.i32 convention(direct) {
    %r = tribute_control.handle : core.i32 {
      %c = tribute_control.call @g(%x) : (core.i32) -> core.i32
      tribute_control.yield %c
    } do(%v: core.i32) {
        tribute_control.yield %v
    }
    %s = arith.addi %r, %x : core.i32
    tribute_control.return %s
  }
  tribute_cps.func @in_body(%x: core.i32) -> core.i32 convention(direct) {
    %r = tribute_control.handle : core.i32 {
      %i = tribute_control.handle : core.i32 {
        %c = tribute_control.call @g(%x) : (core.i32) -> core.i32
        tribute_control.yield %c
      } do(%w: core.i32) {
          tribute_control.yield %w
      }
      tribute_control.yield %i
    } do(%v: core.i32) {
        tribute_control.yield %v
    }
    tribute_control.return %r
  }
  tribute_cps.func @planned(%x: core.i32) -> core.i32 convention(direct) {
    %r = tribute_control.handle : core.i32 {
      %c = tribute_control.call @g(%x) attributes {evidence_plan = [{mask = @Ctl}]} : (core.i32) -> core.i32
      tribute_control.yield %c
    } do(%v: core.i32) {
        tribute_control.yield %v
    }
    tribute_control.return %r
  }
  tribute_cps.func @other_result(%x: core.i32) -> core.i32 convention(direct) {
    %r = tribute_control.handle : core.i32 {
      %c = tribute_control.call @g(%x) : (core.i32) -> core.i32
      tribute_control.yield %x
    } do(%v: core.i32) {
        tribute_control.yield %v
    }
    tribute_control.return %r
  }
  tribute_cps.func @no_handle(%x: core.i32) -> core.i32 convention(direct) {
    %c = tribute_control.call @g(%x) : (core.i32) -> core.i32
    tribute_control.return %c
  }
}
"#;

/// Only the calls at the body result of a tail handle hand its frame on:
/// not one in a handle whose result is used, one in the body of another
/// handle, one with an `evidence_plan`, one whose result the body does not
/// yield, and a tail call outside any handle (`design/cps-effects.md`,
/// "Frame을 그대로 넘기는 자리").
#[test]
fn only_a_call_at_the_body_result_of_a_tail_handle_passes_its_frame_on() {
    let (ctx, module) = parse(BODY_RESULT_CALLS);
    let passes: Vec<(&str, bool)> = module
        .ops(&ctx)
        .iter()
        .filter_map(|&func| {
            let name = ctx.op_attrs(func).get_str("sym_name")?;
            let mut call = None;
            let _ = trunk_ir::walk::walk_op::<()>(&ctx, func, &mut |op| {
                if tribute_control::Call::matches(&ctx, op) || CallIndirect::matches(&ctx, op) {
                    call = Some(op);
                }
                std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
            });
            Some((name, passes_body_frame_on(&ctx, call?)))
        })
        .collect();
    assert_eq!(
        passes,
        [
            ("named", true),
            ("value", true),
            ("select", true),
            ("in_arm", true),
            ("in_lambda", true),
            ("not_tail", false),
            ("in_body", false),
            ("planned", false),
            ("other_result", false),
            ("no_handle", false),
        ]
    );
}

#[test]
fn the_targets_of_a_function_table_share_one_source_signature() {
    let errors = messages(
        r#"core.module @test {
  tribute_cps.func @a() -> core.i32 convention(direct) {
    %0 = arith.constant 1 : core.i32
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

const CPS_ENTRY: &str = r#"core.module @test {
  !t0 = tribute_control.func_sig<(core.i32) -> core.i32>

  tribute_cps.func @f(%0: core.i32) -> core.i32 convention(direct) {
      tribute_control.return %0
  }
  tribute_cps.func @"f$7"(%0: core.i32) -> core.i32 convention(cps) attributes {entry_of = "f"} {
      tribute_control.return %0
  }
  tribute_cps.func @outer(%0: core.i32) -> core.i32 convention(direct) {
      %1 = tribute_cps.lambda(%2: core.i32) -> core.i32 attributes {cps_entry = true} entries {
        direct {
            tribute_control.tail_call @f(%2) attributes {cps_callee = @"f$7"} : (core.i32) -> ()
        }
      }
      tribute_control.return %0
  }
}
"#;

/// `cps_entry` and `cps_callee` survive printing and parsing, and read back
/// through their accessors and helpers.
#[test]
fn cps_entry_and_cps_callee_round_trip() {
    let (ctx, module) = parse(CPS_ENTRY);
    assert_eq!(module.op().display(&ctx).to_string(), CPS_ENTRY);

    let outer = Func::from_op(&ctx, module.ops(&ctx)[2]).unwrap();
    let block = ctx.region(outer.body(&ctx)).blocks[0];
    let lambda = ctx.block(block).ops[0];
    assert!(has_cps_entry(&ctx, lambda));
    assert!(Lambda::from_op(&ctx, lambda).unwrap().cps_entry(&ctx));

    let body = ctx.op_regions(lambda).last().unwrap();
    let entry = ctx.region(body).blocks[0];
    let call = ctx.block(entry).ops[0];
    assert!(in_cps_entry_flow(&ctx, call));
    assert_eq!(
        ctx.op_attrs(call)
            .get_symbol_ref(CPS_CALLEE_ATTR)
            .and_then(|path| path.leaf())
            .map(ToString::to_string),
        Some("f$7".to_owned())
    );
    assert_eq!(messages(CPS_ENTRY), Vec::<String>::new());

    // Without the declaration the lambda is an ordinary one.
    let plain = CPS_ENTRY.replace(" attributes {cps_entry = true}", "");
    let (ctx, module) = parse(&plain);
    let outer = Func::from_op(&ctx, module.ops(&ctx)[2]).unwrap();
    let block = ctx.region(outer.body(&ctx)).blocks[0];
    let lambda = ctx.block(block).ops[0];
    assert!(!has_cps_entry(&ctx, lambda));
    assert_eq!(module.op().display(&ctx).to_string(), plain);
}

/// A lambda with a `cps` body already has a `Cps` entry, so it does not
/// declare `cps_entry`.
#[test]
fn cps_entry_is_rejected_with_a_cps_body() {
    let input = r#"core.module @test {
  tribute_cps.func @outer(%0: core.i32) -> core.i32 convention(direct) {
      %1 = tribute_cps.lambda(%2: core.i32) -> core.i32 attributes {cps_entry = true} entries {
        direct {
            tribute_control.return %2
        }
        cps {
            tribute_control.return %2
        }
      }
      tribute_control.return %0
  }
}
"#;
    let lambda_errors = |input: &str| {
        let (ctx, module) = parse(input);
        let outer = Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
        let block = ctx.region(outer.body(&ctx)).blocks[0];
        let lambda = ctx.block(block).ops[0];
        OpDef::of(&ctx, lambda)
            .unwrap()
            .verify(&ctx, lambda)
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
    };
    let errors = lambda_errors(input);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("`cps_entry` belongs only")),
        "{errors:?}"
    );
    let accepted = input.replace(" attributes {cps_entry = true}", "");
    assert_eq!(lambda_errors(&accepted), Vec::<String>::new());
}

use super::*;
use tribute_ir::calling_convention::{
    CALLING_CONVENTION_ATTR, DynConventionBearing, cps_closure_function_type,
};
use tribute_ir::dialect::{adt, effect};
use trunk_ir::SymbolPath;
use trunk_ir::ops::DialectType;
use trunk_ir::parser::parse_test_module;
use trunk_ir::printer::print_module;
use trunk_ir::symbol_table::nearest_table;

use crate::lower_continuation_frames::lower_continuation_frames;
use crate::test_support::assert_unchanged_on_error;

/// Run the conversion without operation declarations or intrinsics.
fn run_pre_cps(ctx: &mut IrContext, module: Module) -> Result<(), TributeControlToCpsError> {
    tribute_control_to_cps(ctx, module, &[], &mut Default::default())?;
    lower_continuation_frames(ctx, module)
}

fn parse(input: &str) -> (IrContext, Module) {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, input);
    (ctx, module)
}

#[test]
fn shared_function_conversion_preserves_lists_and_nested_attributes() {
    let (mut ctx, module) = parse("core.module @m { func.func @placeholder() { func.return } }");
    let block = ctx.region(module.body(&ctx).unwrap()).blocks[0];
    let i32_ty = trunk_ir::types::TypeDataBuilder::new("core.i32").intern(&mut ctx);
    let source_callable = tribute_control::FuncSig::params([i32_ty], i32_ty)
        .intern(&mut ctx)
        .as_type_ref();
    for results in [vec![], vec![source_callable]] {
        let attrs =
            AttributeMap::from_entries(&mut ctx, [("nested", Attribute::Type(source_callable))]);
        let source = func::func_sig_with_attrs(&mut ctx, [source_callable], results.clone(), attrs)
            .as_type_ref();
        let mut converter = Converter::new(&mut ctx, block, CallableGraph::default());
        let converted = converter.convert_type(source);
        assert_eq!(
            converter.convert_type(source),
            converted,
            "cached conversion must be stable"
        );
        let converted_callable = converter.convert_type(source_callable);
        let function = func::FuncSig::from_type_ref(converter.ctx, converted).unwrap();
        assert_ne!(converted_callable, source_callable);
        assert_eq!(function.inputs(converter.ctx), [converted_callable]);
        assert_eq!(function.results(converter.ctx).len(), results.len());
        if !results.is_empty() {
            assert_eq!(function.results(converter.ctx), [converted_callable]);
        }
        assert_eq!(
            converter.ctx.type_attrs(converted).get_type("nested"),
            Some(converted_callable)
        );
    }
}

/// The first ability instance an operation of `module` names.
fn first_ability(ctx: &IrContext, module: Module) -> SymbolPath {
    let mut found = None;
    let _ = trunk_ir::walk::walk_op::<()>(ctx, module.op(), &mut |op| {
        if found.is_none() {
            found = ctx.op_attrs(op).get_symbol_ref("ability").cloned();
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    found.expect("test module names an ability instance")
}

fn operation_declarations(
    ctx: &mut IrContext,
    operations: &[(&str, &str)],
) -> Vec<tribute_control::OperationDeclaration> {
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(ctx, "core.i32")).then_some(ty))
        .expect("test module declares core.i32");
    operations
        .iter()
        .map(|&(ability_name, op_name)| {
            tribute_control::OperationDeclaration::new(
                SymbolPath::new([ability_name]),
                ctx.intern_str(op_name),
                ctx.intern_str("op"),
                [i32_type],
                i32_type,
            )
        })
        .collect()
}

/// `converted` is the module after CPS conversion and `printed` after its
/// frames are lowered.
fn assert_nested_resume_frames(converted: &str, printed: &str) {
    // Each handle is installed in place, and again wherever a resumed
    // continuation rebuilds its layer.
    assert_eq!(
        printed.matches("effect.fresh_prompt_tag").count(),
        2,
        "{printed}"
    );
    assert_eq!(
        printed.matches("ability.handle_dispatch").count(),
        4,
        "{printed}"
    );
    assert_eq!(
        converted.matches("ability.perform").count(),
        2,
        "{converted}"
    );
    assert!(
        printed
            .matches("tribute.cps_continuation_frame_result")
            .count()
            >= 2,
        "{printed}"
    );
    assert!(
        printed.matches("adt.struct_get").count() >= 6,
        "dynamic frames must be unpacked to recover done and dispatch: {printed}"
    );
    assert!(
        printed.matches("adt.struct_new").count() >= 4,
        "suffixes and resumes must repack the handle-layer dispatcher with a new done target: {printed}"
    );
    assert!(
        converted
            .lines()
            .filter(|line| line.contains("ability.perform"))
            .all(|line| !line.contains(": core.i32")),
        "final ability.perform must be resultless: {converted}"
    );
    assert!(!printed.contains("tribute_control."), "{printed}");
}

/// The conventions of the bodies of `lambda`, a `closure.lambda`.
fn lambda_conventions(ctx: &IrContext, lambda: OpRef) -> Vec<CallingConvention> {
    closure::Lambda::from_op(ctx, lambda)
        .ok()
        .and_then(|lambda| lambda.entries(ctx))
        .into_iter()
        .flatten()
        .map(|entry| entry.convention)
        .collect()
}

fn collect_lambdas(ctx: &IrContext, op: OpRef, lambdas: &mut Vec<OpRef>) {
    if closure::Lambda::matches(ctx, op) {
        lambdas.push(op);
        return;
    }
    for region in ctx.op_regions(op) {
        for &block in &ctx.region(region).blocks {
            for &child in &ctx.block(block).ops {
                collect_lambdas(ctx, child, lambdas);
            }
        }
    }
}

fn collect_cps_tail_calls(ctx: &IrContext, op: OpRef, tails: &mut Vec<OpRef>) {
    if func::TailCallIndirect::matches(ctx, op)
        && DynConventionBearing::of(ctx, op) == Some(CallingConvention::Cps)
    {
        tails.push(op);
    }
    for region in ctx.op_regions(op) {
        for &block in &ctx.region(region).blocks {
            for &child in &ctx.block(block).ops {
                collect_cps_tail_calls(ctx, child, tails);
            }
        }
    }
}

#[test]
fn textual_callable_graph_converts_and_reparses() {
    let input = r#"core.module @test {
  tribute_cps.func @decl(%left: core.i32, %right: core.i32) -> core.i32 convention(direct)
  tribute_cps.func @identity(%value: core.i32) -> core.i32 convention(evidence_direct) {
    tribute_control.return %value
  }
  tribute_cps.func @cps_identity(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_cps.func @outer(%first: core.i32) -> core.i32 convention(direct) {
    %captured = tribute_cps.lambda(%value: core.i32) -> core.i32 captures [%first] entries {
      direct {
        tribute_control.return %first
      }
    }
    tribute_control.return %first
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();
    verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(!printed.contains("tribute_control.func "));
    assert!(!printed.contains("tribute_control.func_ref "));
    assert!(!printed.contains("tribute_control.call_indirect "));
    assert!(printed.contains("func.func @decl"));
    assert!(printed.contains("closure.lambda"));
    assert!(printed.contains("func.tail_call_indirect"));
    assert!(printed.contains("tribute.calling_convention = \"cps\""));

    let mut reparsed = IrContext::new();
    let reparsed_module = parse_test_module(&mut reparsed, &printed);
    verify_tribute_control_post_cps(&reparsed, reparsed_module, &mut Default::default()).unwrap();
}

#[test]
fn pass_wrapper_runs_the_verified_conversion() {
    let input = r#"core.module @test {
  tribute_cps.func @identity(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
}"#;
    let (mut ctx, module) = parse(input);
    let target = core::Module::from_op(&ctx, module.op()).unwrap();
    let mut pass = TributeControlToCps::new();
    assert_eq!(pass.name(), "tribute-control-to-cps");
    pass.run(&mut ctx, target, &mut Default::default()).unwrap();
    verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap();
}

#[test]
fn textual_direct_evidence_and_cps_transfers_preserve_exact_abis() {
    let input = r#"core.module @test {
  !direct = tribute_control.func_sig<(core.i32) -> core.i32>
  !evidence = tribute_control.func_sig<(core.i32) -> core.i32>
  !cps = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_cps.func @direct(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_cps.func @evidence(%value: core.i32) -> core.i32 convention(evidence_direct) {
    tribute_control.return %value
  }
  tribute_cps.func @cps(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_cps.func_table @"direct$8" = {direct = @direct}
  tribute_cps.func_table @"evidence$8" = {evidence_direct = @evidence}
  tribute_cps.func_table @"cps$8" = {cps = @cps}
  tribute_cps.func @exercise(%value: core.i32) -> core.i32 convention(cps) {
    %direct_result = tribute_control.call %value {callee = @direct} : core.i32
    %evidence_result = tribute_control.call %direct_result {callee = @evidence} : core.i32
    %direct_ref = tribute_cps.func_ref {func_ref = @"direct$8"} : !direct
    %direct_indirect = tribute_cps.call_indirect %direct_ref, %evidence_result {convention = "direct"} : core.i32
    %evidence_ref = tribute_cps.func_ref {func_ref = @"evidence$8"} : !evidence
    %evidence_indirect = tribute_cps.call_indirect %evidence_ref, %direct_indirect {convention = "evidence_direct"} : core.i32
    %cps_ref = tribute_cps.func_ref {func_ref = @"cps$8"} : !cps
    %cps_indirect = tribute_cps.call_indirect %cps_ref, %evidence_indirect {convention = "cps"} : core.i32
    tribute_control.return %cps_indirect
  }
  tribute_cps.func @known_cps(%value: core.i32) -> core.i32 convention(cps) {
    %result = tribute_control.call %value {callee = @cps} : core.i32
    tribute_control.return %result
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(printed.contains("func.call "));
    assert!(printed.contains("func.call_indirect"));
    assert!(printed.contains("func.tail_call "));
    assert!(printed.contains("func.tail_call_indirect"));
    assert!(printed.contains("tribute.calling_convention = \"direct\""));
    assert!(printed.contains("tribute.calling_convention = \"evidence_direct\""));
    assert!(printed.contains("tribute.calling_convention = \"cps\""));
    assert!(!printed.contains("tribute_control."), "{printed}");

    // The closure type records no convention; the lambda's own convention
    // places the environment.
    let mut lambdas = Vec::new();
    collect_lambdas(&ctx, module.op(), &mut lambdas);
    let one_parameter_cps = lambdas.into_iter().find(|lambda| {
        let closure_ty = ctx.op_result_types(*lambda)[0];
        lambda_conventions(&ctx, *lambda) == [CallingConvention::Cps]
            && closure::Closure::from_type_ref(&ctx, closure_ty)
                .and_then(|closure| func::FuncSig::from_type_ref(&ctx, closure.func_type(&ctx)))
                .is_some_and(|callable| callable.inputs(&ctx).len() == 1)
    });
    let lambda =
        one_parameter_cps.expect("conversion should produce a one-parameter CPS continuation");
    let closure_ty = ctx.op_result_types(lambda)[0];
    assert!(ctx.get_type(closure_ty).attrs.is_empty());
    crate::lower_closure_lambda::lower_closure_lambda(&mut ctx, module);
    crate::closure_lower::lower_prepared_closures(&mut ctx, module).unwrap();
    let lowered = print_module(&ctx, module.op());
    assert!(
        !lowered.contains("closure.lambda") && !lowered.contains("closure.new"),
        "{lowered}"
    );
}

/// Every converted indirect transfer that carries convention metadata, in
/// source order.
fn convention_bearing_transfers(ctx: &IrContext, op: OpRef) -> Vec<OpRef> {
    fn visit(ctx: &IrContext, op: OpRef, transfers: &mut Vec<OpRef>) {
        if (func::CallIndirect::matches(ctx, op) || func::TailCallIndirect::matches(ctx, op))
            && DynConventionBearing::of(ctx, op).is_some()
        {
            transfers.push(op);
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    visit(ctx, child, transfers);
                }
            }
        }
    }
    let mut transfers = Vec::new();
    visit(ctx, op, &mut transfers);
    transfers
}

/// A source-data indirect call must reach the target ABI boundary with the
/// exact callable contract its callee was declared with. Emitting convention
/// metadata without that contract is what the validator rejects, and the
/// contract must never be reconstructed from the physical operands later.
///
/// The callee can be a callable parameter, a named `func_ref`, or a local or
/// capturing lambda, so every one of those forms must carry its contract.
#[test]
fn source_data_indirect_calls_carry_their_exact_signature() {
    let input = r#"core.module @test {
  !direct = tribute_control.func_sig<(core.i32, core.i32) -> core.i32>
  !evidence = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_cps.func @add(%left: core.i32, %right: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %left
  }
  tribute_cps.func @pure(%f: !direct, %left: core.i32, %right: core.i32) -> core.i32 convention(direct) {
    %result = tribute_cps.call_indirect %f, %left, %right {convention = "direct"} : core.i32
    tribute_control.return %result
  }
  tribute_cps.func_table @"add$8" = {direct = @add}
  tribute_cps.func @named_ref(%left: core.i32, %right: core.i32) -> core.i32 convention(direct) {
    %ref = tribute_cps.func_ref {func_ref = @"add$8"} : !direct
    %called = tribute_cps.call_indirect %ref, %left, %right {convention = "direct"} : core.i32
    tribute_control.return %called
  }
  tribute_cps.func @capturing(%captured: core.i32, %value: core.i32) -> core.i32 convention(direct) {
    %lambda = tribute_cps.lambda(%inner: core.i32) -> core.i32 captures [%captured] entries {
      direct {
        %sum = arith.addi %inner, %captured : core.i32
        tribute_control.return %sum
      }
    }
    %called = tribute_cps.call_indirect %lambda, %value {convention = "direct"} : core.i32
    tribute_control.return %called
  }
  tribute_cps.func @thunk(%f: !evidence, %value: core.i32) -> core.i32 convention(evidence_direct) {
    %called = tribute_cps.call_indirect %f, %value {convention = "evidence_direct"} : core.i32
    tribute_control.return %called
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();

    // The adapters that fill the closures' stronger slots make transfers of
    // their own.
    let transfers: Vec<_> = module
        .ops(&ctx)
        .iter()
        .copied()
        .filter(|&op| {
            func::Func::from_op(&ctx, op).is_ok_and(|function| {
                !function
                    .sym_name(&ctx)
                    .starts_with("__tribute_entry_adapter_")
            })
        })
        .flat_map(|op| convention_bearing_transfers(&ctx, op))
        .collect();
    let mut conventions: Vec<_> = transfers
        .iter()
        .map(|transfer| DynConventionBearing::of(&ctx, *transfer).unwrap())
        .collect();
    conventions.sort_unstable();
    assert_eq!(
        conventions,
        [
            CallingConvention::Direct,
            CallingConvention::Direct,
            CallingConvention::Direct,
            CallingConvention::EvidenceDirect,
        ],
        "every source-data indirect call must convert: {}",
        print_module(&ctx, module.op())
    );

    for transfer in &transfers {
        let convention = DynConventionBearing::of(&ctx, *transfer).unwrap();
        let call = trunk_ir::op_interface::DynIndirectCallLike::get(&ctx, *transfer).unwrap();
        let callee = call.indirect_callee(&ctx).unwrap();
        // The callee's closure type is the source callable; the transfer's
        // signature adds the hidden parameters of its convention.
        let source = closure::Closure::from_type_ref(&ctx, ctx.value_ty(callee))
            .map(|closure| closure.func_type(&ctx))
            .and_then(|ty| func::FuncSig::from_type_ref(&ctx, ty))
            .expect("the converted callee is a closure of its source callable");
        let entry = call
            .exact_signature(&ctx)
            .and_then(|ty| func::FuncSig::from_type_ref(&ctx, ty))
            .expect("a converted transfer carries its entry signature");
        let hidden = usize::from(convention.needs_evidence());
        assert_eq!(
            (&entry.inputs(&ctx)[hidden..], entry.results(&ctx)),
            (source.inputs(&ctx), source.results(&ctx)),
            "{convention:?} transfer must carry its exact entry contract: {}",
            print_module(&ctx, module.op())
        );
    }

    // The converted contract must survive the passes that run before the
    // target ABI boundary, which validates and physicalizes it.
    crate::lower_closure_lambda::lower_closure_lambda(&mut ctx, module);
    crate::target_abi::lower_cps_signatures_to_physical(&mut ctx, module)
        .expect("exact signatures must let the converted transfers cross the target ABI boundary");
}

#[test]
fn nested_textual_module_converts_its_callable_graph_atomically() {
    let input = r#"core.module @outer {
  core.module @inner {
    tribute_cps.func @nested(%value: core.i32) -> core.i32 convention(cps) {
      tribute_control.return %value
    }
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(printed.contains("core.module @inner"));
    assert!(printed.contains("func.func @nested"));
    assert!(printed.contains("func.tail_call_indirect"));
    assert!(!printed.contains("tribute_control."));

    let mut reparsed = IrContext::new();
    let reparsed_module = parse_test_module(&mut reparsed, &printed);
    verify_tribute_control_post_cps(&reparsed, reparsed_module, &mut Default::default()).unwrap();
}

#[test]
fn nested_modules_resolve_same_named_callables_in_their_own_module() {
    let input = r#"core.module @outer {
  tribute_cps.func @same(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_cps.func @outer_call(%value: core.i32) -> core.i32 convention(direct) {
    %result = tribute_control.call %value {callee = @same} : core.i32
    tribute_control.return %result
  }
  core.module @inner {
    tribute_cps.func @same(%value: core.i1) -> core.i1 convention(evidence_direct) {
      tribute_control.return %value
    }
    tribute_cps.func @inner_call(%value: core.i1) -> core.i1 convention(evidence_direct) {
      %result = tribute_control.call %value {callee = @same} : core.i1
      tribute_control.return %result
    }
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap();
    let printed = print_module(&ctx, module.op());
    assert_eq!(printed.matches("func.func @same").count(), 2, "{printed}");
    assert!(printed.contains("func.func @outer_call"), "{printed}");
    assert!(printed.contains("func.func @inner_call"), "{printed}");
    assert!(printed.contains("tribute.calling_convention = \"direct\""));
    assert!(printed.contains("tribute.calling_convention = \"evidence_direct\""));

    let mut reparsed = IrContext::new();
    let reparsed_module = parse_test_module(&mut reparsed, &printed);
    verify_tribute_control_post_cps(&reparsed, reparsed_module, &mut Default::default()).unwrap();
}

#[test]
fn textual_nested_attribute_types_convert_atomically() {
    let input = r#"core.module @test {
  !callback = tribute_control.func_sig<(core.i32) -> (core.i32, {metadata = [core.array<core.i32>, [7, @Callback]]})>
  !record = adt.struct<CallbackRecord(callback: !callback)>
  tribute_cps.func @identity(%value: !record) -> !record convention(direct) {
    tribute_control.return %value
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(printed.contains("adt.struct<CallbackRecord("));
    assert!(
        printed.contains("closure.closure<func.func_sig<(core.i32) -> (core.i32, {metadata"),
        "{printed}"
    );
    assert!(
        printed.contains("metadata = [core.array<core.i32>, [7, @Callback]]"),
        "{printed}"
    );
    assert!(!printed.contains("tribute_control."));

    let mut reparsed = IrContext::new();
    let reparsed_module = parse_test_module(&mut reparsed, &printed);
    verify_tribute_control_post_cps(&reparsed, reparsed_module, &mut Default::default()).unwrap();
}

#[test]
fn source_signature_metadata_roundtrips_before_conversion() {
    let source = r#"core.module @test {
  !inner = tribute_control.func_sig<(core.i32) -> core.i32>
  !outer = tribute_control.func_sig<() -> (core.i32, {metadata = [[!inner, @function]]})>
  !lambda = tribute_control.func_sig<() -> (core.i32, {metadata = [[!inner, @lambda]]})>
  tribute_cps.func {convention = "direct", sym_name = "outer", type = !outer} {
    %lambda = tribute_cps.lambda {conventions = ["direct"]} : !lambda {
      %inner = arith.const {value = 1} : core.i32
      tribute_control.return %inner
    }
    %result = arith.const {value = 2} : core.i32
    tribute_control.return %result
  }
}"#;
    let (ctx, module) = parse(source);
    let printed_source = print_module(&ctx, module.op());
    assert!(
        printed_source.contains(
            "!outer = tribute_control.func_sig<() -> (core.i32, {metadata = [[!inner, @function]]})>"
        ),
        "{printed_source}"
    );
    assert!(
        printed_source.contains(
            "!lambda = tribute_control.func_sig<() -> (core.i32, {metadata = [[!inner, @lambda]]})>"
        ),
        "{printed_source}"
    );
    let (mut ctx, module) = parse(&printed_source);

    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    let lowered = module
        .ops(&ctx)
        .iter()
        .copied()
        .find(|&op| {
            func::Func::from_op(&ctx, op).is_ok_and(|function| function.sym_name(&ctx) == "outer")
        })
        .unwrap();
    let physical = ctx.op_attrs(lowered).get_type("type").unwrap();
    let physical = func::FuncSig::from_type_ref(&ctx, physical).unwrap();
    let Attribute::List(function_metadata) = ctx
        .type_attrs(physical.as_type_ref())
        .get("metadata")
        .unwrap()
    else {
        panic!("function metadata must remain a list");
    };
    let [Attribute::List(function_pair)] = function_metadata.as_slice() else {
        panic!("function metadata must preserve its nested pair");
    };
    let [
        Attribute::Type(function_nested),
        Attribute::SymbolRef(function_tag),
    ] = function_pair.as_slice()
    else {
        panic!("function metadata must preserve its nested source signature");
    };
    assert_eq!(*function_tag, SymbolPath::new(["function"]));
    assert!(closure::Closure::matches(&ctx, *function_nested));
    assert!(
        ctx.type_attrs(physical.as_type_ref())
            .get(CALLING_CONVENTION_ATTR)
            .is_none()
    );
    assert_eq!(
        DynConventionBearing::of(&ctx, lowered),
        Some(CallingConvention::Direct)
    );

    let mut lambdas = Vec::new();
    collect_lambdas(&ctx, module.op(), &mut lambdas);
    let lambda_metadata = lambdas.into_iter().find_map(|lambda| {
        let closure_ty = ctx.op_result_types(lambda)[0];
        let closure = closure::Closure::from_type_ref(&ctx, closure_ty)?;
        let signature = func::FuncSig::from_type_ref(&ctx, closure.func_type(&ctx))?;
        let Attribute::List(metadata) = ctx.type_attrs(signature.as_type_ref()).get("metadata")?
        else {
            return None;
        };
        let [Attribute::List(pair)] = metadata.as_slice() else {
            return None;
        };
        let [Attribute::Type(nested), Attribute::SymbolRef(tag)] = pair.as_slice() else {
            return None;
        };
        (tag == ["lambda"]).then_some(*nested)
    });
    assert!(
        lambda_metadata.is_some_and(|nested| closure::Closure::matches(&ctx, nested)),
        "lambda metadata must retain a converted nested source signature"
    );
}

#[test]
fn malformed_pre_boundary_is_atomic() {
    let input = r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    %illegal = func.call %value {callee = @broken} : core.i32
    tribute_control.return %illegal
  }
}"#;
    let (mut ctx, module) = parse(input);
    let error = assert_unchanged_on_error(&mut ctx, module, run_pre_cps);
    assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
    assert!(error.to_string().contains("func.call"));
}

#[test]
fn malformed_or_retired_source_signatures_fail_before_conversion() {
    for (name, attrs, expected) in [
        (
            "tribute_control.func_sig",
            vec![
                (func::NUM_INPUTS_ATTR, Attribute::Int(2)),
                (func::NUM_RESULTS_ATTR, Attribute::Int(1)),
            ],
            "malformed tribute_control.func_sig",
        ),
        (
            "tribute_control.callable",
            vec![
                (func::NUM_INPUTS_ATTR, Attribute::Int(1)),
                (func::NUM_RESULTS_ATTR, Attribute::Int(1)),
            ],
            "unsupported tribute_control type 'callable'",
        ),
    ] {
        let (mut ctx, module) = parse(
            r#"core.module @test {
  tribute_cps.func @identity(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
}"#,
        );
        let i32_ty = ctx
            .types()
            .iter()
            .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
            .unwrap();
        let mut builder = TypeDataBuilder::new(name).params([i32_ty]);
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        let malformed = builder.intern(&mut ctx);
        let function = module.ops(&ctx)[0];
        ctx.op_attrs_mut(function)
            .insert("type", Attribute::Type(malformed));

        let before = print_module(&ctx, module.op());
        let error =
            tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap_err();
        assert_eq!(error.boundary, PRE_CPS_BOUNDARY, "{name}: {error}");
        assert!(error.to_string().contains(expected), "{name}: {error}");
        assert_eq!(print_module(&ctx, module.op()), before, "{name}");
    }
}

#[test]
fn raw_malformed_source_signature_storage_fails_before_conversion() {
    let input = r#"core.module @test {
  tribute_control.func {sym_name = "broken", type = tribute_control.func_sig<core.i32, {num_inputs = 2, num_results = 1}>}
  %lambda = tribute_control.lambda : tribute_control.func_sig<core.i32, {num_inputs = 2, num_results = 1}>
}"#;
    let (mut ctx, module) = parse(input);
    let error = assert_unchanged_on_error(&mut ctx, module, run_pre_cps);
    assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
    assert!(
        error
            .to_string()
            .contains("malformed tribute_control.func_sig"),
        "{error}"
    );
}

#[test]
fn malformed_lookup_inputs_fail_before_conversion_and_remain_unchanged() {
    let malformed = [
        (
            r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    %result = tribute_control.call %value : core.i32
    tribute_control.return %result
  }
}"#,
            "missing required attribute `callee`",
        ),
        (
            r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    %result = tribute_control.call %value {callee = @missing} : core.i32
    tribute_control.return %result
  }
}"#,
            "unresolved callee @missing",
        ),
        (
            r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    %result = tribute_cps.call_indirect %value, %value {convention = "direct"} : core.i32
    tribute_control.return %result
  }
}"#,
            "expected S: tribute_control.func_sig",
        ),
        (
            r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(cps) {
    %result = tribute_control.perform %value : core.i32
    tribute_control.return %result
  }
}"#,
            "missing required attribute `ability`",
        ),
        (
            r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(cps) {
    %result = tribute_control.handle : core.i32
    tribute_control.return %result
  }
}"#,
            "expected 3 region(s)",
        ),
    ];
    for (input, expected) in malformed {
        let (mut ctx, module) = parse(input);
        let error = assert_unchanged_on_error(&mut ctx, module, run_pre_cps);
        assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn source_successors_fail_before_conversion_and_leave_ir_unchanged() {
    let input = r#"core.module @test {
  ^entry:
    scf.br [^exit]
  ^exit:
}"#;
    let (mut ctx, module) = parse(input);
    let error = assert_unchanged_on_error(&mut ctx, module, run_pre_cps);
    assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
    assert!(
        error.to_string().contains("forbids block successors"),
        "{error}"
    );
}

#[test]
fn malformed_switch_case_fails_before_conversion_and_is_atomic() {
    let input = r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(cps) {
    scf.switch %value {
      scf.case {
        %identity = tribute_cps.lambda(%nested: core.i32) -> core.i32 captures [] entries {
          direct {
            tribute_control.return %nested
          }
        }
        scf.yield
      }
    }
    tribute_control.return %value
  }
}"#;
    let (mut ctx, module) = parse(input);
    let error = assert_unchanged_on_error(&mut ctx, module, run_pre_cps);
    assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
    assert!(
        error.to_string().contains("scf.case requires a value"),
        "{error}"
    );
}

#[test]
fn source_shape_validation_rejects_every_malformed_switch_component() {
    let malformed = [
        (
            r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch {
    scf.default {
      scf.yield
    }
  }
}"#,
            "one discriminant, no results, and one body region",
        ),
        (
            r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  %result = scf.switch %value : core.i32 {
    scf.default {
      scf.yield
    }
  }
}"#,
            "one discriminant, no results, and one body region",
        ),
        (
            r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch %value {
    ^first:
      scf.default {
        scf.yield
      }
    ^second:
  }
}"#,
            "body region requires exactly one block",
        ),
        (
            r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch %value {
    %invalid = arith.const {value = 1} : core.i32
  }
}"#,
            "body may contain only scf.case and scf.default",
        ),
        (
            r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch %value {
    scf.case {value = 0}
  }
}"#,
            "arm requires exactly one region",
        ),
        (
            r#"core.module @test {
  %value = arith.const {value = 0} : core.i32
  scf.switch %value {
    scf.default {
      ^first:
        scf.yield
      ^second:
    }
  }
}"#,
            "arm region requires exactly one block",
        ),
    ];

    for (input, expected) in malformed {
        let (ctx, module) = parse(input);
        let failures = verify_source_conversion_shapes(&ctx, module);
        assert!(
            failures
                .iter()
                .any(|failure| failure.message.contains(expected)),
            "{failures:?}"
        );
    }
}

#[test]
fn post_candidate_failure_restores_alias_maps_and_canonical_ir() {
    let input = r#"core.module @candidate {
  !callback = tribute_control.func_sig<(core.i32) -> core.i32>
  func.func @broken(%value: core.i32) -> core.i32 {
    func.return %value
  }
}"#;
    let (mut ctx, candidate) = parse(input);
    let before = print_module(&ctx, candidate.op());
    let source_aliases = ctx.type_aliases().to_vec();
    let (alias_name, source_type) = source_aliases[0].clone();
    let converted_type = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    ctx.register_type_alias(alias_name.clone(), converted_type);

    let error = verify_candidate_or_restore_aliases(
        &mut ctx,
        candidate,
        &source_aliases,
        &mut Default::default(),
    )
    .unwrap_err();
    assert_eq!(error.boundary, POST_CPS_BOUNDARY);
    assert_eq!(ctx.type_alias_by_name(&alias_name), Some(source_type));
    assert_eq!(ctx.type_alias_by_type(source_type), Some(alias_name));
    assert_eq!(ctx.type_alias_by_type(converted_type), None);
    assert_eq!(print_module(&ctx, candidate.op()), before);
}

#[test]
fn named_boundaries_report_local_and_core_validation_errors() {
    let local_input = r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return
  }
}"#;
    let (ctx, module) = parse(local_input);
    let error =
        verify_tribute_control_pre_cps(&ctx, module, &[], &mut Default::default()).unwrap_err();
    assert!(error.to_string().contains("expected 1 operand"), "{error}");

    let core_input = r#"core.module @test {
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(direct) {
    func.tail_call_indirect
    tribute_control.return %value
  }
}"#;
    let (ctx, module) = parse(core_input);
    let error =
        verify_tribute_control_pre_cps(&ctx, module, &[], &mut Default::default()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected at least 1 operand(s), found 0"),
        "{error}"
    );

    let post_input = r#"core.module @test {
  func.func @broken() -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call_indirect
  }
}"#;
    let (ctx, module) = parse(post_input);
    let error = verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected at least 1 operand(s), found 0"),
        "{error}"
    );

    let malformed_delimiter = r#"core.module @test {
  !evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
  func.func @broken() -> core.never attributes {tribute.calling_convention = "cps"} {
    ability.handle_dispatch {abilities = []} {
      ^body(%inner: !evidence):
        func.unreachable
    }
  }
}"#;
    let (ctx, module) = parse(malformed_delimiter);
    let error = verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
    assert!(error.to_string().contains("requires an evidence operand"));
}

#[test]
fn post_boundary_rejects_a_convention_on_an_operation_that_cannot_carry_one() {
    let (ctx, module) = parse(
        r#"core.module @test {
  func.func @main() -> core.nil attributes {tribute.calling_convention = "direct"} {
    %n = core.nil_value {tribute.calling_convention = "direct"} : core.nil
    func.return %n
  }
}"#,
    );
    let error = verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
    assert!(
        error.to_string().contains(
            "core.nil_value: calling-convention metadata on an operation that cannot carry it"
        ),
        "{error}"
    );
}

#[test]
fn post_boundary_rejects_recursively_nested_control_types() {
    let input = r#"core.module @test {
  !nested = core.tuple<closure.closure<func.func_sig<(tribute_control.resume_token<core.i32, core.i32>) -> core.i32>>>
}"#;
    let (ctx, module) = parse(input);
    let error = verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
    assert!(error.to_string().contains("forbidden type"), "{error}");
}

#[test]
fn pre_boundary_rejects_recursively_nested_physical_signatures() {
    let input = r#"core.module @test {
  !nested = core.tuple<func.func_sig<(core.i32) -> core.i32>>
}"#;
    let (ctx, module) = parse(input);
    let error =
        verify_tribute_control_pre_cps(&ctx, module, &[], &mut Default::default()).unwrap_err();
    assert!(error.to_string().contains("forbidden type"), "{error}");
}

#[test]
fn post_boundary_rejects_malformed_physical_callable_transfers() {
    let malformed = [
        (
            r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call %value {tribute.calling_convention = "cps"}
  }
}"#,
            "requires a resolved callee symbol",
        ),
        (
            r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call %value {callee = @callee}
  }
}"#,
            "func.tail_call must carry exact Direct, EvidenceDirect, or Cps metadata",
        ),
        (
            r#"core.module @test {
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call %value {callee = @missing, tribute.calling_convention = "cps"}
  }
}"#,
            "unresolved callee @missing",
        ),
        (
            r#"core.module @test {
  func.func @callee(%value: core.i64) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = "cps"}
  }
}"#,
            "operands do not match the target signature",
        ),
        (
            r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = "cps"} {
    func.return %value
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = "cps"}
  }
}"#,
            "target must have core.never result",
        ),
        (
            r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = "evidence_direct"} {
    func.unreachable
  }
  func.func @caller(%value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = "cps"}
  }
}"#,
            "must preserve exact Cps metadata",
        ),
        (
            r#"core.module @test {
  func.func @caller(%callee: closure.closure<func.func_sig<(core.i32) -> core.never>>, %value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    func.tail_call_indirect %callee, %value
  }
}"#,
            "func.tail_call_indirect must carry exact Direct, EvidenceDirect, or Cps metadata",
        ),
        (
            r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    func.return %value
  }
  func.func @caller(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %result = func.call %value {callee = @callee, tribute.calling_convention = "evidence_direct"} : core.i32
    func.return %result
  }
}"#,
            "func.call metadata does not match its target",
        ),
        (
            r#"core.module @test {
  func.func @caller(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = "direct"} {
    %result = func.call %value {callee = @missing, tribute.calling_convention = "direct"} : core.i32
    func.return %result
  }
}"#,
            "func.call references unresolved callee @missing",
        ),
        (
            r#"core.module @test {
  func.func @caller(%callee: closure.closure<func.func_sig<(core.i32) -> core.never>>, %value: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    %result = func.call_indirect %callee, %value {tribute.calling_convention = "cps"} : core.never
    func.unreachable
  }
}"#,
            "dynamic Cps transfers must use func.tail_call_indirect",
        ),
    ];
    for (input, expected) in malformed {
        let (ctx, module) = parse(input);
        let error =
            verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn post_boundary_rejects_nonphysical_dispatchers_and_residual_control_ops() {
    let dispatcher_input = r#"core.module @test {
  !evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
  !tr = closure.closure<func.func_sig<(!evidence, core.i32, tribute_rt.anyref) -> tribute_rt.anyref>>
  func.func @caller(%ev: !evidence, %prompt: core.i32, %tr: !tr) -> core.never attributes {tribute.calling_convention = "cps"} {
    ability.handle_dispatch %ev, %prompt, %tr {abilities = [@State]} {
      ^body(%inner: !evidence):
        func.unreachable
    }
  }
}"#;
    let (ctx, module) = parse(dispatcher_input);
    let error = verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("dispatcher must be a physical closure result"),
        "{error}"
    );

    let wrong_abi_input = r#"core.module @test {
  !evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
  func.func @caller(%ev: !evidence, %prompt: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    %tr = closure.lambda evidence_direct(%inner: !evidence) -> tribute_rt.anyref {
      func.unreachable
    }
    ability.handle_dispatch %ev, %prompt, %tr {abilities = [@State]} {
      ^body(%inner: !evidence):
        func.unreachable
    }
  }
}"#;
    let (ctx, module) = parse(wrong_abi_input);
    let error = verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
    let text = error.to_string();
    assert!(
        text.contains("tail-resumptive dispatcher has the wrong"),
        "{text}"
    );

    let wrong_metadata_input = r#"core.module @test {
  !evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
  func.func @caller(%ev: !evidence, %prompt: core.i32) -> core.never attributes {tribute.calling_convention = "cps"} {
    %tr = closure.lambda direct(%inner: !evidence, %op_idx: core.i32, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
      func.unreachable
    }
    ability.handle_dispatch %ev, %prompt, %tr {abilities = [@State]} {
      ^body(%inner: !evidence):
        func.unreachable
    }
  }
}"#;
    let (ctx, module) = parse(wrong_metadata_input);
    let error = verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
    let text = error.to_string();
    assert!(
        text.contains("dispatcher must be a closure.lambda with one evidence_direct body"),
        "{text}"
    );

    let residual_input = r#"core.module @test {
  tribute_cps.func @residual(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
}"#;
    let (ctx, module) = parse(residual_input);
    let error = verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
    assert!(
        error.to_string().contains("residual tribute_cps.func"),
        "{error}"
    );
}

#[test]
fn conversion_error_display_handles_diagnostics_without_an_operation() {
    let error = TributeControlToCpsError::one(PRE_CPS_BOUNDARY, None, None, "synthetic failure");
    let text = error.to_string();
    assert!(text.contains("1 error(s)"));
    assert!(text.contains("  - synthetic failure"));
}

#[test]
fn textual_resumptive_handle_emits_one_resultless_delimiter() {
    let input = r#"core.module @test {
  tribute_cps.func @run(%input: core.i32) -> core.i32 convention(cps) {
    %handled = tribute_control.handle : core.i32 {
      %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      tribute_control.yield %performed
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
        ^arm(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %handled
  }
}"#;
    let (mut ctx, module) = parse(input);
    let mut ability = None;
    fn find_ability(ctx: &IrContext, region: RegionRef, found: &mut Option<SymbolPath>) {
        for block in ctx.region(region).blocks.iter().copied() {
            for op in ctx.block(block).ops.iter().copied() {
                if tribute_control::Handler::matches(ctx, op) {
                    *found = ctx.op_attrs(op).get_symbol_ref("ability").cloned();
                }
                for nested in ctx.op_regions(op) {
                    find_ability(ctx, nested, found);
                }
            }
        }
    }
    find_ability(&ctx, module.body(&ctx).unwrap(), &mut ability);
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .unwrap();
    let declarations = [tribute_control::OperationDeclaration::new(
        ability.unwrap(),
        ctx.intern_str("get"),
        ctx.intern_str("op"),
        vec![i32_type],
        i32_type,
    )];
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();
    let mut perform_resume = None;
    fn find_perform_resume(ctx: &IrContext, op: OpRef, found: &mut Option<ValueRef>) {
        if let Ok(perform) = effect::DispatchCps::from_op(ctx, op) {
            *found = Some(perform.resume(ctx));
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    find_perform_resume(ctx, child, found);
                }
            }
        }
    }
    find_perform_resume(&ctx, module.op(), &mut perform_resume);
    let perform_resume = perform_resume.expect("resumptive handler emits ability.perform");
    let erased_function = cps_closure_function_type(&ctx, ctx.value_ty(perform_resume))
        .expect("ability.perform resume has CPS callable provenance");
    let erased = func::FuncSig::from_type_ref(&ctx, erased_function)
        .expect("ability.perform resume has a function signature");
    assert_eq!(erased.inputs(&ctx).len(), 3);
    assert!(
        tribute_ir::dialect::tribute_rt::Anyref::matches(&ctx, erased.inputs(&ctx)[2]),
        "ability.perform must receive Resume<anyref, R>, not the exact continuation: {}",
        print_module(&ctx, module.op())
    );
    let trunk_ir::ValueDef::OpResult(wrapper_op, _) = ctx.value_def(perform_resume) else {
        panic!("ability.perform resume must be the one-shot adapter closure");
    };
    assert!(closure::Lambda::matches(&ctx, wrapper_op));
    assert!(
        ctx.op_operands(wrapper_op).iter().copied().any(|capture| {
            cps_closure_function_type(&ctx, ctx.value_ty(capture))
                .and_then(|function| func::FuncSig::from_type_ref(&ctx, function))
                .is_some_and(|function| {
                    function.inputs(&ctx).len() == 3
                        && trunk_ir::dialect::core::I32::matches(&ctx, function.inputs(&ctx)[2])
                })
        }),
        "the erased adapter must capture the declared ResumeExact<I, R>: {}",
        print_module(&ctx, module.op())
    );
    let mut consumed_get = None;
    let mut consumed_set = None;
    fn find_one_shot_state_ops(
        ctx: &IrContext,
        op: OpRef,
        consumed_get: &mut Option<ValueRef>,
        consumed_set: &mut Option<ValueRef>,
    ) {
        let is_one_shot_type = |ty: TypeRef| {
            ctx.type_attrs(ty)
                .get_str("name")
                .is_some_and(|name| name.starts_with("__tribute_one_shot_state"))
        };
        if let Ok(get) = adt::StructGet::from_op(ctx, op)
            && is_one_shot_type(get.r#type(ctx))
        {
            *consumed_get = Some(get.r#ref(ctx));
        }
        if let Ok(set) = adt::StructSet::from_op(ctx, op)
            && is_one_shot_type(set.r#type(ctx))
        {
            assert!(ctx.op_results(op).is_empty(), "struct_set mutates in place");
            *consumed_set = Some(set.r#ref(ctx));
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    find_one_shot_state_ops(ctx, child, consumed_get, consumed_set);
                }
            }
        }
    }
    find_one_shot_state_ops(&ctx, module.op(), &mut consumed_get, &mut consumed_set);
    assert_eq!(consumed_get, consumed_set);
    assert!(consumed_get.is_some(), "one-shot state was not emitted");
    let printed = print_module(&ctx, module.op());
    // One installation under a fresh prompt, and one more in the function
    // every resumption of the handle shares to install the layer again.
    assert_eq!(printed.matches("effect.fresh_prompt_tag").count(), 1);
    assert_eq!(printed.matches("ability.handle_dispatch").count(), 2);
    assert!(printed.contains("abilities = [@State]"), "{printed}");
    assert!(printed.contains("func.tail_call_indirect"));
    assert!(printed.contains("adt.struct_set"));
    assert!(!printed.contains("tribute_control."));
    assert!(!printed.contains("handler_metadata"));
    assert!(!printed.contains("adt.ref_null"));
    assert!(!printed.contains("__tribute_cps_control"));
}

#[test]
fn multiple_arms_for_one_ability_emit_one_dispatcher() {
    let input = r#"core.module @test {
  tribute_cps.func @run(%input: core.i32) -> core.i32 convention(cps) {
    %handled = tribute_control.handle : core.i32 {
      %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      tribute_control.yield %performed
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
        ^get(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
      tribute_control.handler {ability = @State, kind = "op", op_name = "set", operation_result_type = core.i32} {
        ^set(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
          %fallback = arith.const {value = 9} : core.i32
          tribute_control.yield %fallback
      }
    }
    tribute_control.return %handled
  }
}"#;
    let (mut ctx, module) = parse(input);
    let ability = first_ability(&ctx, module);
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .unwrap();
    let declarations = [
        tribute_control::OperationDeclaration::new(
            ability.clone(),
            ctx.intern_str("get"),
            ctx.intern_str("op"),
            [i32_type],
            i32_type,
        ),
        tribute_control::OperationDeclaration::new(
            ability.clone(),
            ctx.intern_str("set"),
            ctx.intern_str("op"),
            [i32_type],
            i32_type,
        ),
    ];
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();

    let mut delimiters = Vec::new();
    fn collect_delimiters(ctx: &IrContext, op: OpRef, found: &mut Vec<OpRef>) {
        if ability::HandleDispatch::matches(ctx, op) {
            found.push(op);
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    collect_delimiters(ctx, child, found);
                }
            }
        }
    }
    collect_delimiters(&ctx, module.op(), &mut delimiters);
    // The installation, and the one shared by every resumption that installs
    // the layer again: the count does not grow with the resumptive arms.
    assert_eq!(delimiters.len(), 2, "expected the handle's two delimiters");
    for delimiter in delimiters {
        assert_eq!(ctx.op_operands(delimiter).len(), 3);
        let Some(Attribute::List(abilities)) = ctx.op_attrs(delimiter).get("abilities") else {
            panic!("final delimiter must have abilities");
        };
        assert_eq!(abilities.len(), 1);
    }
    let printed = print_module(&ctx, module.op());
    assert!(printed.contains("value = 9"));
    assert!(printed.contains("func.tail_call_indirect"));
    assert!(
        printed.contains("@__tribute_make_dispatch_adapter_"),
        "the dispatch-adapter factory must be present: {printed}"
    );
    let mut tails = Vec::new();
    collect_cps_tail_calls(&ctx, module.op(), &mut tails);
    assert!(
        tails.len() >= 3,
        "the dispatch adapter and separate handler paths must tail-transfer: {printed}"
    );
    for tail in tails {
        let signature = trunk_ir::op_interface::DynIndirectCallLike::get(&ctx, tail)
            .and_then(|call| call.exact_signature(&ctx))
            .expect("every emitted CPS indirect tail has its exact closure contract");
        let callable = func::FuncSig::from_type_ref(&ctx, signature)
            .expect("the indirect tail signature is a func.func_sig");
        assert_eq!(
            callable.single_result(&ctx).unwrap(),
            core::Never::params().intern(&mut ctx).as_type_ref()
        );
        assert_eq!(
            callable.inputs(&ctx).len() + 1,
            ctx.op_operands(tail).len(),
            "the exact signature covers every tail operand: {printed}"
        );
    }
    assert!(printed.contains("effect.fresh_prompt_tag"));
}

#[test]
fn textual_scf_branch_captures_only_the_selected_suffix() {
    let input = r#"core.module @test {
  tribute_cps.func @branch(%input: core.i32) -> core.i32 convention(cps) {
    %condition = arith.const {value = true} : core.i1
    %selected = scf.if %condition : core.i32 {
      %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      scf.yield %performed
    } {
      %fallback = arith.const {value = 7} : core.i32
      scf.yield %fallback
    }
    %one = arith.const {value = 1} : core.i32
    %sum = arith.addi %selected, %one : core.i32
    tribute_control.return %sum
  }
}"#;
    let (mut ctx, module) = parse(input);
    let ability = first_ability(&ctx, module);
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .unwrap();
    let declarations = [tribute_control::OperationDeclaration::new(
        ability.clone(),
        ctx.intern_str("get"),
        ctx.intern_str("op"),
        vec![i32_type],
        i32_type,
    )];
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    let converted = print_module(&ctx, module.op());
    lower_continuation_frames(&mut ctx, module).unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(printed.contains("scf.if"));
    assert!(printed.contains(" : core.never"));
    assert!(converted.contains("ability.perform"));
    assert!(printed.contains("func.tail_call_indirect"));
    assert!(printed.contains("arith.addi"));
    assert!(!printed.contains("tribute_control."));
}

#[test]
fn textual_zero_result_cps_and_direct_scf_branches_lower() {
    let input = r#"core.module @test {
  tribute_cps.func @identity(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_cps.func @branch(%input: core.i32, %condition: core.i1) -> core.i32 convention(cps) {
    %direct = tribute_cps.lambda(%value: core.i32) -> core.i32 captures [%condition] entries {
      direct {
        %selected = scf.if %condition : core.i32 {
          %called = tribute_control.call %value {callee = @identity} : core.i32
          scf.yield %called
        } {
          scf.yield %value
        }
        tribute_control.return %selected
      }
    }
    scf.if %condition {
      %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      scf.yield
    } {
      %performed = tribute_control.perform %input {ability = @State, op_name = "set", operation_kind = "op"} : core.i32
      scf.yield
    }
    tribute_control.return %input
  }
}"#;
    let (mut ctx, module) = parse(input);
    let ability = first_ability(&ctx, module);
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .unwrap();
    let declarations = [
        tribute_control::OperationDeclaration::new(
            ability.clone(),
            ctx.intern_str("get"),
            ctx.intern_str("op"),
            [i32_type],
            i32_type,
        ),
        tribute_control::OperationDeclaration::new(
            ability.clone(),
            ctx.intern_str("set"),
            ctx.intern_str("op"),
            [i32_type],
            i32_type,
        ),
    ];
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    let converted = print_module(&ctx, module.op());
    lower_continuation_frames(&mut ctx, module).unwrap();
    let printed = print_module(&ctx, module.op());
    let scf_ifs: Vec<_> = printed
        .lines()
        .filter(|line| line.contains("scf.if"))
        .collect();
    let value_if_count = scf_ifs
        .iter()
        .filter(|line| line.contains(": core.i32"))
        .count();
    assert_eq!(value_if_count, 1, "{printed}");
    assert!(
        scf_ifs.iter().any(|line| line.contains(": core.never")),
        "{printed}"
    );
    assert_eq!(
        converted.matches("ability.perform").count(),
        2,
        "{converted}"
    );
    assert!(printed.contains("func.call") && printed.contains("func.tail_call_indirect"));
    assert!(!printed.contains("tribute_control."));

    let mut tails = Vec::new();
    collect_cps_tail_calls(&ctx, module.op(), &mut tails);
    assert!(
        tails.len() >= 3,
        "the dispatch adapter and independent CPS exits must both tail-transfer: {printed}"
    );
    for tail in tails {
        let signature = trunk_ir::op_interface::DynIndirectCallLike::get(&ctx, tail)
            .and_then(|call| call.exact_signature(&ctx))
            .expect("every emitted CPS indirect tail has its exact closure contract");
        let callable = func::FuncSig::from_type_ref(&ctx, signature)
            .expect("the indirect tail signature is a func.func_sig");
        assert_eq!(
            callable.single_result(&ctx).unwrap(),
            core::Never::params().intern(&mut ctx).as_type_ref()
        );
        assert_eq!(
            callable.inputs(&ctx).len() + 1,
            ctx.op_operands(tail).len(),
            "the exact signature covers every tail operand: {printed}"
        );
        assert!(
            callable
                .inputs(&ctx)
                .iter()
                .zip(&ctx.op_operands(tail)[1..])
                .all(|(expected, actual)| *expected == ctx.value_ty(*actual)),
            "the exact signature must not be inferred from erased operands: {printed}"
        );
    }

    let mut lambdas = Vec::new();
    collect_lambdas(&ctx, module.op(), &mut lambdas);
    let mut generated_param_counts = Vec::new();
    for lambda in lambdas {
        if lambda_conventions(&ctx, lambda) != [CallingConvention::Cps] {
            continue;
        }
        let closure_ty = ctx.op_result_types(lambda)[0];
        assert!(ctx.get_type(closure_ty).attrs.is_empty());
        let callable = closure::Closure::from_type_ref(&ctx, closure_ty)
            .and_then(|closure| func::FuncSig::from_type_ref(&ctx, closure.func_type(&ctx)))
            .unwrap();
        generated_param_counts.push(callable.inputs(&ctx).len());
    }
    assert!(generated_param_counts.contains(&2), "{printed}");

    crate::lower_closure_lambda::lower_closure_lambda(&mut ctx, module);
    let lifted = print_module(&ctx, module.op());
    assert!(!lifted.contains("closure.lambda"), "{lifted}");
    crate::closure_lower::lower_prepared_closures(&mut ctx, module).unwrap();
    let lowered = print_module(&ctx, module.op());
    assert!(!lowered.contains("closure.new"), "{lowered}");
    assert!(lowered.contains("signature"), "{lowered}");
}

#[test]
fn cps_indirect_tail_without_a_provenance_bearing_closure_fails_before_insertion() {
    let (mut ctx, module) = parse(
        r#"core.module @test {
  func.func @caller() -> core.never {
    func.unreachable
  }
}"#,
    );
    let module_block = ctx.region(module.body(&ctx).unwrap()).blocks[0];
    let location = ctx.op(module.op()).location;
    let never = core::Never::params().intern(&mut ctx).as_type_ref();
    let raw_type = func::func_sig(&mut ctx, [], [never]).as_type_ref();
    let raw = func::Constant::operands()
        .func_ref(SymbolPath::new(["raw"]))
        .results(raw_type)
        .build(&mut ctx, location);
    let before = ctx.block(module_block).ops.clone();
    let callee = raw.result(&ctx);
    let error =
        emit_cps_tail_call_indirect(&mut ctx, module_block, location, callee, std::iter::empty())
            .unwrap_err();

    assert!(
        error.to_string().contains("no exact closure contract"),
        "{error}"
    );
    assert_eq!(ctx.block(module_block).ops.as_slice(), before.as_slice());
}

#[test]
fn malformed_multi_result_effectful_scf_if_remains_unchanged() {
    let input = r#"core.module @test {
  tribute_cps.func @broken(%input: core.i32, %condition: core.i1) -> core.i32 convention(cps) {
    %left, %right = scf.if %condition : core.i32, core.i32 {
      %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      scf.yield %performed, %input
    } {
      scf.yield %input, %input
    }
    tribute_control.return %left
  }
}"#;
    let (mut ctx, module) = parse(input);
    let ability = first_ability(&ctx, module);
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .unwrap();
    let declarations = [tribute_control::OperationDeclaration::new(
        ability.clone(),
        ctx.intern_str("get"),
        ctx.intern_str("op"),
        [i32_type],
        i32_type,
    )];
    tribute_control::declare_operations(&mut ctx, module, &declarations);
    let before = print_module(&ctx, module.op());
    let error = tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap_err();
    assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
    assert!(
        error
            .to_string()
            .contains("scf.if (op3): expected 0 to 1 result(s), found 2"),
        "{error}"
    );
    assert_eq!(print_module(&ctx, module.op()), before);
}

/// A function reference called through a stronger convention than its
/// target's enters the table's adapter, which needs no null environment.
#[test]
fn stronger_func_ref_builds_a_cps_adapter_without_a_null_environment() {
    let input = r#"core.module @test {
  !cps = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_cps.func @id(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_cps.func_table @"id$8" = {direct = @id}
  tribute_cps.func @run(%value: core.i32) -> core.i32 convention(cps) {
    %callee = tribute_cps.func_ref {func_ref = @"id$8"} : !cps
    %result = tribute_cps.call_indirect %callee, %value {convention = "cps"} : core.i32
    tribute_control.return %result
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(printed.contains("__tribute_func_ref_adapter"));
    assert!(printed.contains("closure.new"));
    assert!(printed.contains("func.tail_call_indirect"));
    assert!(printed.contains("tribute.calling_convention = \"cps\""));
    assert!(!printed.contains("adt.ref_null"));
    assert!(!printed.contains("tribute_control.func "));
    assert!(!printed.contains("tribute_control.func_ref "));
    assert!(!printed.contains("tribute_control.call_indirect "));
}

#[test]
fn parameter_attributes_follow_their_parameters_to_the_physical_abi() {
    let input = r#"core.module @test {
  !cps = tribute_control.func_sig<(core.i32 {k = @v}) -> (core.i32 {r = @x})>
  tribute_cps.func @id(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_cps.func_table @"id$8" = {cps = @id}
  tribute_cps.func @run(%value: core.i32) -> core.i32 convention(cps) {
    %callee = tribute_cps.func_ref {func_ref = @"id$8"} : !cps
    %result = tribute_cps.call_indirect %callee, %value {convention = "cps"} : core.i32
    tribute_control.return %result
  }
}"#;
    let (mut ctx, module) = parse(input);
    let marked = AttributeMap::from_entries(
        &mut ctx,
        [("k", Attribute::SymbolRef(SymbolPath::new(["v"])))],
    );
    let empty = AttributeMap::new;
    let adapter_type = |ctx: &IrContext| {
        let adapter = module
            .ops(ctx)
            .iter()
            .copied()
            .find_map(|op| {
                let function = func::Func::from_op(ctx, op).ok()?;
                (function.sym_name(ctx) == "__tribute_func_ref_adapter_0").then_some(function)
            })
            .expect("func_ref adapter");
        func::FuncSig::from_type_ref(ctx, adapter.r#type(ctx)).unwrap()
    };

    // Evidence, environment, and frame are hidden parameters; the Cps
    // source result is replaced by `core.never` and loses its attributes.
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    let logical = adapter_type(&ctx);
    itertools::assert_equal(
        logical
            .inputs_with_attrs(&ctx)
            .map(|(_, attrs)| attrs.clone()),
        [empty(), empty(), empty(), marked.clone()],
    );
    assert!(logical.result_attrs(&ctx).all(AttributeMap::is_empty));

    // The physical Cps callable has no result and consumes every input,
    // including the hidden ones. Its definition and the indirect call agree
    // on one exact type, and the closure's convention table holds it.
    crate::lower_closure_lambda::lower_closure_lambda(&mut ctx, module);
    crate::target_abi::lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
    crate::closure_lower::lower_prepared_closures(&mut ctx, module).unwrap();
    let physical = adapter_type(&ctx);
    let contract = crate::target_abi::physical_parameter_attrs(&mut ctx, CallingConvention::Cps);
    let consumed = || contract.clone();
    let mut marked_consumed = marked;
    marked_consumed.view_mut(&mut ctx).extend(consumed());
    itertools::assert_equal(
        physical
            .inputs_with_attrs(&ctx)
            .map(|(_, attrs)| attrs.clone()),
        [consumed(), consumed(), consumed(), marked_consumed],
    );
    assert!(physical.results(&ctx).is_empty());
    let mut references = Vec::new();
    let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
        if let Ok(table) = tribute_ir::dialect::tribute_rt::ConventionTable::from_op(&ctx, op)
            && table
                .entries(&ctx)
                .any(|entry| entry == ["__tribute_func_ref_adapter_0"])
        {
            references.push(physical.as_type_ref());
        }
        if let Ok(call) = func::TailCallIndirect::from_op(&ctx, op)
            && call.signature(&ctx) == physical.as_type_ref()
        {
            references.push(call.signature(&ctx));
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    assert_eq!(
        references,
        [physical.as_type_ref(), physical.as_type_ref()],
        "{}",
        print_module(&ctx, module.op())
    );
}

/// References that share a table share its entry functions, whatever
/// convention their values are called with.
#[test]
fn references_sharing_a_table_share_its_entries() {
    let input = r#"core.module @test {
  !direct = tribute_control.func_sig<(core.i32) -> core.i32>
  !evidence = tribute_control.func_sig<(core.i32) -> core.i32>
  !cps = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_cps.func @direct(%value: core.i32) -> core.i32 convention(direct) {
    tribute_control.return %value
  }
  tribute_cps.func @evidence(%value: core.i32) -> core.i32 convention(evidence_direct) {
    tribute_control.return %value
  }
  tribute_cps.func @cps(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_cps.func_table @"direct$8" = {direct = @direct}
  tribute_cps.func_table @"evidence$8" = {evidence_direct = @evidence}
  tribute_cps.func_table @"cps$8" = {cps = @cps}
  tribute_cps.func @refs(%value: core.i32) -> core.i32 convention(cps) {
    %direct_direct = tribute_cps.func_ref {func_ref = @"direct$8"} : !direct
    %direct_evidence = tribute_cps.func_ref {func_ref = @"direct$8"} : !evidence
    %direct_cps = tribute_cps.func_ref {func_ref = @"direct$8"} : !cps
    %evidence_evidence = tribute_cps.func_ref {func_ref = @"evidence$8"} : !evidence
    %evidence_cps = tribute_cps.func_ref {func_ref = @"evidence$8"} : !cps
    %cps_cps = tribute_cps.func_ref {func_ref = @"cps$8"} : !cps
    tribute_control.return %value
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap();
    let printed = print_module(&ctx, module.op());
    assert_eq!(
        printed
            .matches("func.func @__tribute_func_ref_adapter")
            .count(),
        3,
        "{printed}"
    );
    let mut tables = Vec::new();
    let _ = walk_op::<()>(&ctx, module.op(), &mut |op| {
        if let Ok(closure) = closure::New::from_op(&ctx, op) {
            tables.push(closure.entries(&ctx).cloned().collect::<Vec<_>>());
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    assert_eq!(tables.len(), 6, "{printed}");
    assert_eq!(tables[0], tables[1]);
    assert_eq!(tables[1], tables[2]);
    assert_eq!(tables[3], tables[4]);
    assert_ne!(tables[0], tables[3]);
    assert!(!printed.contains("tribute_control."));
}

/// A func_ref adapter is defined in the root module, which a reference in a
/// nested module cannot name.
#[test]
fn func_ref_in_a_nested_module_is_adapted_in_that_module() {
    let input = r#"core.module @test {
  core.module @inner {
    !direct = tribute_control.func_sig<(core.i32) -> core.i32>
    tribute_cps.func @id(%value: core.i32) -> core.i32 convention(direct) {
      tribute_control.return %value
    }
    tribute_cps.func_table @"id$8" = {direct = @id}
    tribute_cps.func @refs(%value: core.i32) -> core.i32 convention(direct) {
      %reference = tribute_cps.func_ref {func_ref = @"id$8"} : !direct
      tribute_control.return %value
    }
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
        .expect("a nested func_ref is adapted in its own module");
    let symbols = SymbolTable::collect(&ctx, module);
    let mut adapters = Vec::new();
    let _ = walk_op::<()>(&ctx, module.op(), &mut |op| {
        if let Ok(closure) = closure::New::from_op(&ctx, op) {
            adapters.extend(
                closure
                    .entries(&ctx)
                    .map(|entry| symbols.resolve(&ctx, op, entry)),
            );
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    // The adapter fills the Direct slot; trap entries fill the others.
    let [Some(_), Some(_), Some(_)] = adapters[..] else {
        panic!("the closure names its adapter and trap entries: {adapters:?}");
    };
    let inner = symbols
        .resolve_in(&ctx, module.op(), &[Symbol::new("inner")])
        .expect("inner module");
    for entry in adapters.into_iter().flatten() {
        assert_eq!(nearest_table(&ctx, entry), Some(inner));
    }
}

/// A classless tail lets a table enter a `Cps` target through a weaker
/// convention, whose value-delimiter adapter legalization does not build
/// yet.
#[test]
fn a_table_entry_weaker_than_its_cps_target_is_rejected_before_mutation() {
    let input = r#"core.module @test {
  !evidence = tribute_control.func_sig<(core.i32) -> core.i32>
  tribute_cps.func @cps(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_cps.func_table @"cps$8" = {evidence_direct = @cps}
  tribute_cps.func @broken(%value: core.i32) -> core.i32 convention(evidence_direct) {
    %callee = tribute_cps.func_ref {func_ref = @"cps$8"} : !evidence
    tribute_control.return %value
  }
}"#;
    let (mut ctx, module) = parse(input);
    let error = assert_unchanged_on_error(&mut ctx, module, run_pre_cps);
    assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
    assert!(
        error
            .to_string()
            .contains("a function table entry weaker than its target is not supported"),
        "{error}"
    );
}

#[test]
fn raw_pointer_managed_masquerade_is_rejected_before_mutation() {
    let input = r#"core.module @test {
  !S = adt.struct<S()>
  !R = adt.typeref<{name = "S"}>
  tribute_cps.func @broken(%raw: core.ptr) -> !R convention(direct) {
    %middle = core.unrealized_conversion_cast %raw : core.i64
    %managed = core.unrealized_conversion_cast %middle : !R
    tribute_control.return %managed
  }
}"#;
    let (mut ctx, module) = parse(input);
    let error = assert_unchanged_on_error(&mut ctx, module, run_pre_cps);
    assert_eq!(error.boundary, PRE_CPS_BOUNDARY);
    assert!(error.to_string().contains("core.ptr cast chain"), "{error}");
}

#[test]
fn handles_stay_abstract_until_the_frames_are_lowered() {
    let input = r#"core.module @test {
  tribute_cps.func @nested(%input: core.i32) -> core.i32 convention(cps) {
    %outer = tribute_control.handle : core.i32 {
      %inner = tribute_control.handle : core.i32 {
        tribute_control.yield %input
      } {
        ^inner_completion(%value: core.i32):
          tribute_control.yield %value
      } {
        ^inner_handlers:
      }
      tribute_control.yield %inner
    } {
      ^outer_completion(%value: core.i32):
        tribute_control.yield %value
    } {
      ^outer_handlers:
    }
    tribute_control.return %outer
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    let printed = print_module(&ctx, module.op());
    assert_eq!(printed.matches("ability.handle ").count(), 2, "{printed}");
    assert!(!printed.contains("ability.handle_dispatch"), "{printed}");
    assert!(!printed.contains("effect.fresh_prompt_tag"), "{printed}");
    assert!(!printed.contains("adt.struct_new"), "{printed}");
    lower_continuation_frames(&mut ctx, module).unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(!printed.contains("ability.handle "), "{printed}");
    assert!(!printed.contains("ability.frame"), "{printed}");
}

#[test]
fn nested_textual_handles_keep_distinct_delimiters() {
    let input = r#"core.module @test {
  tribute_cps.func @nested(%input: core.i32) -> core.i32 convention(cps) {
    %outer = tribute_control.handle : core.i32 {
      %inner = tribute_control.handle : core.i32 {
        tribute_control.yield %input
      } {
        ^inner_completion(%value: core.i32):
          tribute_control.yield %value
      } {
        ^inner_handlers:
      }
      tribute_control.yield %inner
    } {
      ^outer_completion(%value: core.i32):
        tribute_control.yield %value
    } {
      ^outer_handlers:
    }
    tribute_control.return %outer
  }
}"#;
    let (mut ctx, module) = parse(input);
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();
    let printed = print_module(&ctx, module.op());
    // Each handle is installed in place, and again in the layer a
    // resumed continuation rebuilds.
    assert_eq!(printed.matches("effect.fresh_prompt_tag").count(), 2);
    assert_eq!(printed.matches("ability.handle_dispatch").count(), 4);
    assert!(printed.matches("func.tail_call_indirect").count() >= 3);
    assert!(!printed.contains("__tribute_cps_control"));
}

#[test]
fn nested_same_ability_resumes_rebuild_the_dynamic_frame_dispatcher() {
    let input = r#"core.module @test {
  tribute_cps.func @nested_same(%input: core.i32) -> core.i32 convention(cps) {
    %outer = tribute_control.handle : core.i32 {
      %inner = tribute_control.handle : core.i32 {
        %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
        tribute_control.yield %performed
      } {
        ^inner_completion(%value: core.i32):
          tribute_control.yield %value
      } {
        tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
          ^inner_arm(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.yield %resumed
        }
      }
      %performed = tribute_control.perform %inner {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      tribute_control.yield %performed
    } {
      ^outer_completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
        ^outer_arm(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %outer
  }
}"#;
    let (mut ctx, module) = parse(input);
    let declarations = operation_declarations(&mut ctx, &[("State", "get")]);
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    let converted = print_module(&ctx, module.op());
    lower_continuation_frames(&mut ctx, module).unwrap();
    assert_nested_resume_frames(&converted, &print_module(&ctx, module.op()));
}

#[test]
fn nested_cross_ability_resumes_rebuild_the_dynamic_frame_dispatcher() {
    let input = r#"core.module @test {
  tribute_cps.func @nested_cross(%input: core.i32) -> core.i32 convention(cps) {
    %outer = tribute_control.handle : core.i32 {
      %inner = tribute_control.handle : core.i32 {
        %performed = tribute_control.perform %input {ability = @Console, op_name = "read", operation_kind = "op"} : core.i32
        tribute_control.yield %performed
      } {
        ^inner_completion(%value: core.i32):
          tribute_control.yield %value
      } {
        tribute_control.handler {ability = @Console, kind = "op", op_name = "read", operation_result_type = core.i32} {
          ^inner_arm(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
            %resumed = tribute_control.resume %token, %argument : core.i32
            tribute_control.yield %resumed
        }
      }
      %performed = tribute_control.perform %inner {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      tribute_control.yield %performed
    } {
      ^outer_completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
        ^outer_arm(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %outer
  }
}"#;
    let (mut ctx, module) = parse(input);
    let declarations = operation_declarations(&mut ctx, &[("State", "get"), ("Console", "read")]);
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    let converted = print_module(&ctx, module.op());
    lower_continuation_frames(&mut ctx, module).unwrap();
    assert_nested_resume_frames(&converted, &print_module(&ctx, module.op()));
}
#[test]
fn op_to_never_uses_a_typed_zero_capture_reject_continuation() {
    let input = r#"core.module @test {
  tribute_cps.func @abortable(%input: core.i32) -> core.i32 convention(cps) {
    %handled = tribute_control.handle : core.i32 {
      %never = tribute_control.perform %input {ability = @Abort, op_name = "abort", operation_kind = "op"} : core.never
      %unreachable_suffix = arith.const {value = 99} : core.i32
      tribute_control.yield %unreachable_suffix
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability = @Abort, kind = "op", op_name = "abort", operation_result_type = core.never} {
        ^arm(%argument: core.i32):
          %fallback = arith.const {value = 7} : core.i32
          tribute_control.yield %fallback
      }
    }
    tribute_control.return %handled
  }
}"#;
    let (mut ctx, module) = parse(input);
    let ability = first_ability(&ctx, module);
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .unwrap();
    let never_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.never")).then_some(ty))
        .unwrap();
    let declarations = [tribute_control::OperationDeclaration::new(
        ability.clone(),
        ctx.intern_str("abort"),
        ctx.intern_str("op"),
        vec![i32_type],
        never_type,
    )];
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();
    let mut perform = None;
    fn find_perform(ctx: &IrContext, op: OpRef, found: &mut Option<effect::DispatchCps>) {
        if let Ok(candidate) = effect::DispatchCps::from_op(ctx, op) {
            assert!(
                found.replace(candidate).is_none(),
                "expected one ability.perform"
            );
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    find_perform(ctx, child, found);
                }
            }
        }
    }
    find_perform(&ctx, module.op(), &mut perform);
    let perform = perform.expect("op -> Never emits ability.perform");
    let resume = perform.resume(&ctx);
    let dispatch = perform.dispatch(&ctx);
    let dispatch_signature = cps_closure_function_type(&ctx, ctx.value_ty(dispatch))
        .and_then(|ty| func::FuncSig::from_type_ref(&ctx, ty))
        .expect("ability.perform dispatch has a CPS callable signature");
    assert_eq!(
        dispatch_signature.inputs(&ctx).get(1),
        Some(&ctx.value_ty(resume)),
        "reject continuation must have the exact resume type required by dispatch"
    );
    let resume_signature = cps_closure_function_type(&ctx, ctx.value_ty(resume))
        .and_then(|ty| func::FuncSig::from_type_ref(&ctx, ty))
        .expect("reject continuation has a CPS callable signature");
    assert!(
        tribute_ir::dialect::tribute_rt::Anyref::matches(&ctx, resume_signature.inputs(&ctx)[2]),
        "reject continuation must accept the canonical erased resume input"
    );
    let trunk_ir::ValueDef::OpResult(reject, _) = ctx.value_def(resume) else {
        panic!("reject continuation must be a closure.lambda");
    };
    assert!(closure::Lambda::matches(&ctx, reject));
    assert!(ctx.op_operands(reject).is_empty());
    let body = ctx.op_region(reject, 0).unwrap();
    let body = ctx.region(body).blocks[0];
    assert!(func::Unreachable::matches(&ctx, ctx.block(body).ops[0]));
    crate::lower_ability_call::lower_ability_call(&mut ctx, module);
    let printed = print_module(&ctx, module.op());
    assert!(!printed.contains("ability.perform"), "{printed}");
    assert!(printed.contains("effect.dispatch_cps"), "{printed}");
    let mut dispatch_cps = None;
    fn find_dispatch_cps(ctx: &IrContext, op: OpRef, found: &mut Option<OpRef>) {
        if effect::DispatchCps::matches(ctx, op) {
            assert!(
                found.replace(op).is_none(),
                "expected one effect.dispatch_cps"
            );
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    find_dispatch_cps(ctx, child, found);
                }
            }
        }
    }
    find_dispatch_cps(&ctx, module.op(), &mut dispatch_cps);
    let dispatch_cps = dispatch_cps.expect("ability.perform lowers to effect.dispatch_cps");
    assert!(ctx.op_results(dispatch_cps).is_empty());
    assert_eq!(
        ctx.op_attrs(dispatch_cps).get_type("answer_type"),
        Some(i32_type)
    );
    assert!(printed.contains("func.unreachable"));
    assert!(!printed.contains("value = 99"));
    assert!(!printed.contains("tribute.ownership"));
    assert!(!printed.contains("adt.struct_set"));
    assert!(!printed.contains("adt.ref_null"));
}

#[test]
fn fn_operation_stays_evidence_direct_without_continuation_capture() {
    let input = r#"core.module @test {
  tribute_cps.func @read(%input: core.i32) -> core.i32 convention(cps) {
    %handled = tribute_control.handle : core.i32 {
      %value = tribute_control.perform %input {ability = @Reader, op_name = "read", operation_kind = "fn"} : core.i32
      tribute_control.yield %value
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability = @Reader, kind = "fn", op_name = "read", operation_result_type = core.i32} {
        ^arm(%argument: core.i32):
          tribute_control.yield %argument
      }
    }
    tribute_control.return %handled
  }
}"#;
    let (mut ctx, module) = parse(input);
    let ability = first_ability(&ctx, module);
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .unwrap();
    let declarations = [tribute_control::OperationDeclaration::new(
        ability.clone(),
        ctx.intern_str("read"),
        ctx.intern_str("fn"),
        vec![i32_type],
        i32_type,
    )];
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(printed.contains("ability.call"));
    assert!(!printed.contains("ability.perform"));
    assert!(!printed.contains("tribute.ownership"));
    assert!(!printed.contains("adt.struct_set"));
    assert!(printed.contains("tribute.calling_convention = \"evidence_direct\""));
    let mut indirect_calls = Vec::new();
    fn collect_indirect_calls(ctx: &IrContext, op: OpRef, calls: &mut Vec<OpRef>) {
        if func::CallIndirect::matches(ctx, op) {
            calls.push(op);
        }
        for region in ctx.op_regions(op) {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    collect_indirect_calls(ctx, child, calls);
                }
            }
        }
    }
    collect_indirect_calls(&ctx, module.op(), &mut indirect_calls);
    let call = indirect_calls
        .into_iter()
        .find(|op| DynConventionBearing::of(&ctx, *op) == Some(CallingConvention::EvidenceDirect))
        .expect("fn handler dispatcher call");
    let call = func::CallIndirect::from_op(&ctx, call).unwrap();
    let expected = tribute_ir::calling_convention::physical_closure_function_type(
        &ctx,
        ctx.value_ty(call.callee(&ctx)),
        CallingConvention::EvidenceDirect,
    )
    .expect("fn arm retains an exact closure contract");
    assert_eq!(call.signature(&ctx), expected);
}

#[test]
fn textual_scf_switch_reenters_the_shared_suffix() {
    let input = r#"core.module @test {
  tribute_cps.func @switching(%input: core.i32) -> core.i32 convention(cps) {
    %choice = arith.const {value = 0} : core.i32
    scf.switch %choice {
      scf.case {value = 0} {
        %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
        scf.yield
      }
      scf.default {
        scf.yield
      }
    }
    tribute_control.return %input
  }
}"#;
    let (mut ctx, module) = parse(input);
    let ability = first_ability(&ctx, module);
    let i32_type = ctx
        .types()
        .iter()
        .find_map(|(ty, data)| (data.name.is(&ctx, "core.i32")).then_some(ty))
        .unwrap();
    let declarations = [tribute_control::OperationDeclaration::new(
        ability.clone(),
        ctx.intern_str("get"),
        ctx.intern_str("op"),
        vec![i32_type],
        i32_type,
    )];
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    let printed = print_module(&ctx, module.op());
    assert!(printed.contains("scf.switch"));
    assert!(printed.contains("scf.case"));
    assert!(printed.contains("ability.perform"));
    assert!(printed.matches("func.tail_call_indirect").count() >= 2);
    assert!(!printed.contains("tribute_control."));
}

#[cfg(test)]
mod shared_contract_boundary_regressions {
    use super::verify_tribute_control_post_cps;
    #[test]
    fn post_cps_boundary_checks_known_ordinary_calls_and_returns() {
        let valid = "core.module @m {
          func.func @identity(%x: core.i32) -> core.i32 attributes {tribute.calling_convention = \"direct\"} {
            func.return %x
          }
          func.func @caller(%x: core.i32) -> core.i32 attributes {tribute.calling_convention = \"direct\"} {
            %r = func.call %x {callee = @identity, tribute.calling_convention = \"direct\"} : core.i32
            func.return %r
          }
        }";
        let mut ctx = trunk_ir::IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, valid);
        verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap();
        for (from, to, expected) in [
            ("} : core.i32", "} : core.nil", "call result list mismatch"),
            ("func.return %x", "func.return", "return count mismatch"),
        ] {
            let mut ctx = trunk_ir::IrContext::new();
            let module = trunk_ir::parser::parse_test_module(&mut ctx, &valid.replace(from, to));
            let error =
                verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}

/// The value delimiters of `name`, each with whether it starts on the
/// initial evidence and the type of its answer.
fn delimiters(ctx: &IrContext, module: Module, name: &str) -> Vec<(bool, TypeRef)> {
    let function = module
        .ops(ctx)
        .iter()
        .copied()
        .find(|&op| func::Func::from_op(ctx, op).is_ok_and(|f| f.sym_name(ctx) == name))
        .unwrap();
    let body = ctx.op_region(function, 0).unwrap();
    ctx.region(body)
        .blocks
        .iter()
        .flat_map(|&block| ctx.block(block).ops.iter().copied())
        .filter_map(|op| effect::Delimit::from_op(ctx, op).ok())
        .map(|delimit| {
            let initial = matches!(
                ctx.value_def(delimit.evidence(ctx)),
                trunk_ir::ValueDef::OpResult(op, _) if effect::InitialEvidence::matches(ctx, op)
            );
            (initial, delimit.result_ty(ctx))
        })
        .collect()
}

#[test]
fn a_cps_call_in_a_callable_that_is_not_cps_runs_under_a_value_delimiter() {
    let input = r#"core.module @test {
  tribute_cps.func @cps_identity(%value: core.i32) -> core.i32 convention(cps) {
    tribute_control.return %value
  }
  tribute_cps.func @direct(%value: core.i32) -> core.i32 convention(direct) {
    %answer = tribute_control.call %value {callee = @cps_identity} : core.i32
    tribute_control.return %answer
  }
  tribute_cps.func @with_evidence(%value: core.i32) -> core.i32 convention(evidence_direct) {
    %answer = tribute_control.call %value {callee = @cps_identity} : core.i32
    tribute_control.return %answer
  }
}"#;
    let (mut ctx, module) = parse(input);
    run_pre_cps(&mut ctx, module).unwrap();
    verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap();
    let i32_type = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    // A Direct flow receives no evidence; its delimiter starts on the
    // initial one.
    assert_eq!(delimiters(&ctx, module, "direct"), [(true, i32_type)]);
    assert_eq!(
        delimiters(&ctx, module, "with_evidence"),
        [(false, i32_type)]
    );
    let printed = print_module(&ctx, module.op());
    assert!(!printed.contains("ability.frame"), "{printed}");
}

#[test]
fn a_handle_in_a_direct_callable_runs_under_a_value_delimiter() {
    let input = r#"core.module @test {
  tribute_cps.func @run(%input: core.i32) -> core.i32 convention(direct) {
    %handled = tribute_control.handle : core.i32 {
      %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
      tribute_control.yield %performed
    } {
      ^completion(%value: core.i32):
        tribute_control.yield %value
    } {
      tribute_control.handler {ability = @State, kind = "op", op_name = "get", operation_result_type = core.i32} {
        ^arm(%argument: core.i32, %token: tribute_control.resume_token<core.i32, core.i32>):
          %resumed = tribute_control.resume %token, %argument : core.i32
          tribute_control.yield %resumed
      }
    }
    tribute_control.return %handled
  }
}"#;
    let (mut ctx, module) = parse(input);
    let declarations = operation_declarations(&mut ctx, &[("State", "get")]);
    {
        tribute_control::declare_operations(&mut ctx, module, &declarations);
        tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default())
    }
    .unwrap();
    lower_continuation_frames(&mut ctx, module).unwrap();
    verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap();
    let i32_type = TypeDataBuilder::new("core.i32").intern(&mut ctx);
    assert_eq!(delimiters(&ctx, module, "run"), [(true, i32_type)]);
    let printed = print_module(&ctx, module.op());
    // The handle and its operation are in the delimited Cps closure.
    assert!(printed.contains("ability.handle_dispatch"), "{printed}");
    assert!(printed.contains("effect.dispatch_cps"), "{printed}");
}

#[test]
fn a_general_operation_outside_a_handle_is_rejected_in_a_direct_callable() {
    let input = r#"core.module @test {
  tribute_cps.func @run(%input: core.i32) -> core.i32 convention(direct) {
    %performed = tribute_control.perform %input {ability = @State, op_name = "get", operation_kind = "op"} : core.i32
    tribute_control.return %performed
  }
}"#;
    let (mut ctx, module) = parse(input);
    let declarations = operation_declarations(&mut ctx, &[("State", "get")]);
    tribute_control::declare_operations(&mut ctx, module, &declarations);
    let error = assert_unchanged_on_error(&mut ctx, module, |ctx, module| {
        tribute_control_to_cps(ctx, module, &[], &mut Default::default())
    });
    assert!(
        error
            .to_string()
            .contains("general operation appears in a non-CPS callable"),
        "{error}"
    );
}

#[test]
fn legalization_drops_ability_rows_and_row_metadata() {
    let with_rows = r#"core.module @test {
  !open = tribute_control.func_sig<(core.i32) -> (core.i32, {ability_row = {tail = 0}})>
  ability.decl {sym_name = "A"} {
    ability.operation {kind = "fn", op_name = "get", param_types = [], result_type = core.i32}
  }
  tribute_cps.func @apply['a0 = 'a1, body 'a1](%f: !open, %x: core.i32) -> core.i32 row(| 'a0) inferred convention(cps) {
    %r = tribute_cps.call_indirect %f, %x {convention = "cps"} : core.i32
    tribute_control.return %r
  }
  tribute_cps.func @main(%x: core.i32) -> core.i32 row(@A) convention(cps) {
    %f = tribute_cps.lambda(%y: core.i32) -> core.i32 row(@A) captures [] entries {
      cps {
        tribute_control.return %y
      }
    }
    %r = tribute_control.call %f, %x {callee = @apply, row_args = [{instances = [@A]}]} : core.i32
    tribute_control.return %r
  }
}"#;
    let without_rows = with_rows
        .replace(", {ability_row = {tail = 0}}", "")
        .replace("['a0 = 'a1, body 'a1]", "")
        .replace(" row(| 'a0) inferred", "")
        .replace(" row(@A)", "")
        .replace(", row_args = [{instances = [@A]}]", "");
    let legalize = |input: &str| {
        let (mut ctx, module) = parse(input);
        run_pre_cps(&mut ctx, module).unwrap();
        verify_tribute_control_post_cps(&ctx, module, &mut Default::default()).unwrap();
        print_module(&ctx, module.op())
    };
    assert_eq!(legalize(with_rows), legalize(&without_rows));
}

/// An instantiated local lambda legalizes to the same callable graph as the
/// lambda used directly: rows and their instantiation leave with legalization.
#[test]
fn instantiate_legalizes_to_its_lambda() {
    let module = |header: &str, use_: &str| {
        let kind = if header.is_empty() { "body" } else { "local" };
        format!(
            r#"core.module @m {{
  !step0 = tribute_control.func_sig<(core.i32) -> (core.i32, {{ability_row = {{tail = 0}}}})>
  tribute_cps.func @f['a0, {kind} 'a1](%x: core.i32) -> core.i32 row(| 'a0) convention(direct) {{
    %g = tribute_cps.lambda{header}(%y: core.i32) -> core.i32 row(| 'a1) captures [] entries {{
      direct {{
        tribute_control.return %y
      }}
    }}
    {use_}
    tribute_control.return %r
  }}
}}"#
        )
    };
    let legalized = |input: String| {
        let (mut ctx, module) = parse(&input);
        run_pre_cps(&mut ctx, module).expect("legalizes");
        print_module(&ctx, module.op())
    };
    let instantiated = legalized(module(
        "['a1]",
        "%s = tribute_control.instantiate %g {row_args = [{tail = 0}]} : !step0\n    %r = tribute_cps.call_indirect %s, %x {convention = \"direct\"} : core.i32",
    ));
    let direct = legalized(module(
        "",
        "%r = tribute_cps.call_indirect %g, %x {convention = \"direct\"} : core.i32",
    ));
    assert_eq!(instantiated, direct);
    assert!(!instantiated.contains("quantified_rows"), "{instantiated}");
}

/// Legalization reads conventions, so it takes only a module whose
/// signatures class specialization has assigned them.
#[test]
fn legalization_requires_assigned_conventions() {
    let (mut ctx, module) = parse(
        r#"core.module @m {
  tribute_control.func @f(%x: core.i32) -> core.i32 {
    tribute_control.return %x
  }
}"#,
    );
    let error = run_pre_cps(&mut ctx, module).expect_err("no conventions yet");
    assert!(
        error
            .to_string()
            .contains("class specialization must run before legalization"),
        "{error}"
    );
}

/// A lambda with a body per convention legalizes to one closure with a
/// lifted entry per body, sharing its captures, and an adapter for the slot
/// between them.
#[test]
fn a_lambda_with_two_bodies_lifts_an_entry_per_body() {
    let (mut ctx, module) = parse(
        r#"core.module @test {
  tribute_cps.func @outer(%captured: core.i32, %value: core.i32) -> core.i32 convention(cps) {
    %lambda = tribute_cps.lambda(%inner: core.i32) -> core.i32 captures [%captured] entries {
      direct {
        tribute_control.return %inner
      }
      cps {
        tribute_control.return %captured
      }
    }
    %called = tribute_cps.call_indirect %lambda, %value {convention = "cps"} : core.i32
    tribute_control.return %called
  }
}"#,
    );
    tribute_control_to_cps(&mut ctx, module, &[], &mut Default::default()).unwrap();
    let mut lambdas = Vec::new();
    collect_lambdas(&ctx, module.op(), &mut lambdas);
    let source = lambdas
        .iter()
        .filter_map(|&op| closure::Lambda::from_op(&ctx, op).ok())
        .find(|lambda| lambda.adapters(&ctx).is_some())
        .expect("the source lambda has adapters");
    let entries = source.entries(&ctx).expect("the lists describe the bodies");
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.convention)
            .collect::<Vec<_>>(),
        [CallingConvention::Direct, CallingConvention::Cps]
    );
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.signature.inputs(&ctx).len())
            .collect::<Vec<_>>(),
        [1, 3],
        "the Cps body takes the evidence and its frame"
    );
    let adapters: Vec<_> = source
        .adapters(&ctx)
        .unwrap()
        .iter()
        .map(|(keyword, _)| ctx.str(keyword).to_owned())
        .collect();
    assert_eq!(adapters, ["evidence_direct"]);
    assert_eq!(ctx.op_operands(source.op_ref()).len(), 1);

    crate::lower_closure_lambda::lower_closure_lambda(&mut ctx, module);
    let lifted = print_module(&ctx, module.op());
    assert!(!lifted.contains("closure.lambda"), "{lifted}");
    let mut news = Vec::new();
    let _ = trunk_ir::walk::walk_op::<()>(&ctx, module.op(), &mut |op| {
        if let Ok(new) = closure::New::from_op(&ctx, op)
            && new
                .entries(&ctx)
                .any(|entry| entry.to_string().contains("__tribute_entry_adapter_"))
        {
            news.push(new);
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    let [new] = news[..] else {
        panic!("one closure holds the adapter: {lifted}");
    };
    let entries: Vec<_> = new.entries(&ctx).map(|entry| entry.to_string()).collect();
    assert!(
        entries[0] != entries[2] && entries.iter().all(|entry| !entry.contains("trap")),
        "a body fills the Direct and Cps slots: {entries:?}\n{lifted}"
    );
    assert_eq!(ctx.op_operands(new.op_ref()).len(), 1, "{lifted}");
    crate::closure_lower::lower_prepared_closures(&mut ctx, module).unwrap();
    let lowered = print_module(&ctx, module.op());
    assert!(!lowered.contains("closure.new"), "{lowered}");
}

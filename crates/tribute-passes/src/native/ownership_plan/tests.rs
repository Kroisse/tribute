use super::*;
use crate::native::evidence::lower_evidence_to_native;
use crate::native::rc_materialization::materialize;
use crate::native::type_converter::native_type_converter;
use trunk_ir::parser::parse_test_module;
use trunk_ir::printer::print_module;
use trunk_ir::types::TypeDataBuilder;
use trunk_ir_cranelift_backend::passes::func_to_clif;

fn build(ir: &str) -> (IrContext, Module, NativeOwnershipPlan) {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    let before = print_module(&ctx, module.op());
    let plan = build_native_ownership_plan(&ctx, module).expect("typed ownership plan");
    assert_eq!(print_module(&ctx, module.op()), before);
    (ctx, module, plan)
}

fn count(function: &FunctionOwnershipPlan, kind: ActionKind) -> usize {
    function
        .actions()
        .iter()
        .filter(|action| action.kind == kind)
        .count()
}

fn assert_plan_error_unchanged(ir: &str, expected: &str) {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    let before = print_module(&ctx, module.op());
    let error = build_native_ownership_plan(&ctx, module).expect_err("invalid ownership input");
    assert!(
        error.to_string().contains(expected),
        "unexpected error: {error}"
    );
    assert_eq!(print_module(&ctx, module.op()), before);
}

#[test]
fn typed_plan_options_preserve_or_elide_only_proven_parameter_and_field_borrows() {
    let ir = r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Box = adt.struct() {name = @Box, fields = [[@child, !ChildRef]]}
  !BoxRef = adt.typeref() {name = @Box}
  func.func @observe(%child: !ChildRef) -> core.i32 {
    %value = adt.struct_get %child {field = 0, type = !Child} : core.i32
    func.return %value
  }
  func.func @forward(%child: !ChildRef) -> core.i32 {
    %value = func.call %child {callee = @observe} : core.i32
    func.return %value
  }
  func.func @load(%owner: !BoxRef) -> core.i32 {
    %child = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    %value = func.call %child {callee = @observe} : core.i32
    func.return %value
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    let preserved = build_native_ownership_plan_with_options(
        &ctx,
        module,
        NativeOwnershipPlanOptions {
            elide_proven_borrowed_parameters: false,
            elide_proven_field_borrows: false,
        },
    )
    .expect("preserved typed plan");
    let elided = build_native_ownership_plan_with_options(
        &ctx,
        module,
        NativeOwnershipPlanOptions::production(),
    )
    .expect("elided typed plan");

    let preserved_forward = preserved.function(Symbol::new("forward")).unwrap();
    let elided_forward = elided.function(Symbol::new("forward")).unwrap();
    assert_eq!(preserved_forward.entries(), [EntryOwnership::Retained]);
    assert_eq!(elided_forward.entries(), [EntryOwnership::Borrowed]);
    assert_eq!(count(preserved_forward, ActionKind::EntryAcquire), 1);
    assert_eq!(count(elided_forward, ActionKind::EntryAcquire), 0);

    let load_op = ctx
        .op(elided.function(Symbol::new("load")).unwrap().operation())
        .regions[0];
    let load = ctx.op_result(ctx.block(ctx.region(load_op).blocks[0]).ops[0], 0);
    let projection = ctx.block(ctx.region(load_op).blocks[0]).ops[0];
    let preserved_load = preserved.function(Symbol::new("load")).unwrap();
    let elided_load = elided.function(Symbol::new("load")).unwrap();
    assert!(preserved_load.actions().iter().any(|action| {
        action.kind == ActionKind::CopyAcquire
            && action.value == load
            && action.anchor == ActionAnchor::After(projection)
    }));
    assert!(
        preserved_load
            .actions()
            .iter()
            .any(|action| action.kind == ActionKind::FinalRelease && action.value == load)
    );
    assert!(elided_load.actions().iter().any(|action| {
        action.kind == ActionKind::BorrowLoad
            && action.value == load
            && action.anchor == ActionAnchor::After(projection)
    }));
    assert!(
        !elided_load
            .actions()
            .iter()
            .any(|action| action.kind == ActionKind::FinalRelease && action.value == load)
    );
}

#[test]
fn continuation_frame_capture_has_entry_store_and_deep_release_plan() {
    let (_ctx, _module, plan) = build(
        r#"core.module @test {
  !Frame = adt.struct() {name = @ContinuationFrame, fields = [[@value, core.i32]]}
  !FrameRef = adt.typeref() {name = @ContinuationFrame}
  !Env = adt.struct() {name = @continuation_env, fields = [[@frame, !FrameRef], [@code, core.func(core.nil)]]}
  func.func @capture(%frame: !FrameRef, %code: core.func(core.nil)) -> core.nil {
    %env = adt.struct_new %frame, %code {type = !Env} : !Env
    func.return
  }
}"#,
    );
    let function = plan.function(Symbol::new("capture")).unwrap();
    assert_eq!(
        function.entries(),
        [EntryOwnership::Retained, EntryOwnership::Plain]
    );
    assert_eq!(count(function, ActionKind::EntryAcquire), 1);
    assert_eq!(count(function, ActionKind::StoreAcquire), 1);
    assert_eq!(count(function, ActionKind::FinalRelease), 2);
    assert!(matches!(
        &plan.rtti_types()[0].fields,
        ManagedFieldBitmap::Struct(fields) if fields == &[true, false]
    ));
}

#[test]
fn continuation_frame_capture_materializes_the_typed_entry_and_store_actions() {
    let ir = r#"core.module @test {
  !Frame = adt.struct() {name = @ContinuationFrame, fields = [[@value, core.i32]]}
  !FrameRef = adt.typeref() {name = @ContinuationFrame}
  !Env = adt.struct() {name = @continuation_env, fields = [[@frame, !FrameRef], [@code, core.func(core.nil)]]}
  func.func @capture(%frame: !FrameRef, %code: core.func(core.nil)) -> core.nil {
    %env = adt.struct_new %frame, %code {type = !Env} : !Env
    func.return
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    let plan = build_native_ownership_plan(&ctx, module).expect("typed ownership plan");
    let capture = plan.function(Symbol::new("capture")).unwrap();
    assert_eq!(
        count(capture, ActionKind::EntryAcquire) + count(capture, ActionKind::StoreAcquire),
        2
    );

    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    let materialized = print_module(&ctx, module.op());
    assert_eq!(
        materialized.matches("tribute_rt.retain").count(),
        2,
        "{materialized}"
    );
    assert!(materialized.contains("adt.struct_new"), "{materialized}");
    assert_eq!(
        materialized.matches("tribute_rt.release").count(),
        2,
        "{materialized}"
    );
    // Frame payload is i32 (4 bytes) and the environment payload is two
    // native pointers (16 bytes); each release carries the 8-byte RC header.
    assert!(materialized.contains("alloc_size = 12"), "{materialized}");
    assert!(materialized.contains("alloc_size = 24"), "{materialized}");
}

#[test]
fn nested_continuation_frame_closure_releases_use_exact_header_inclusive_sizes() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Frame = adt.struct() {name = @ContinuationFrame, fields = [[@value, core.i32]]}
  !FrameRef = adt.typeref() {name = @ContinuationFrame}
  !_closure = adt.struct() {name = @_closure, fields = [[@func_ptr, core.i32], [@env, !FrameRef]]}
  func.func @capture(%frame: !FrameRef) -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %closure = adt.struct_new %code, %frame {type = !_closure} : !_closure
    func.return
  }
}"#,
    );
    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    let materialized = print_module(&ctx, module.op());
    // Frame: i32 payload (4) + header (8). Closure: i32 plus aligned frame
    // pointer payload (16) + header (8).
    assert!(materialized.contains("alloc_size = 12"), "{materialized}");
    assert!(materialized.contains("alloc_size = 24"), "{materialized}");
}

#[test]
fn compiler_owned_evidence_closure_handoffs_keep_internal_layouts_owned() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !_closure = adt.struct() {name = @_closure, fields = [[@func_ptr, core.i32], [@env, tribute_rt.anyref]]}
  func.func @install(%evidence: core.ptr, %prompt: core.i32) -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %closure = adt.struct_new %code, %env {type = !_closure} : !_closure
    %tr = arith.const {value = 0} : core.ptr
    %extended = effect.extend %evidence, %prompt, %tr, %closure {ability_ref = core.ability_ref() {name = @State}} : core.ptr
    func.return
  }
}"#,
    );
    lower_evidence_to_native(&mut ctx, module);
    let mut plan = build_native_ownership_plan(&ctx, module).expect("typed ownership plan");
    let install = plan.function(Symbol::new("install")).unwrap();
    let mut handoff = None;
    walk_module(&ctx, module, |op| {
        if core::UnrealizedConversionCast::matches(&ctx, op) {
            handoff = Some(op);
        }
    });
    let handoff = handoff.expect("native evidence handler handoff");
    assert_eq!(count(install, ActionKind::EvidenceClosureTransfer), 1);
    assert!(
        !install.actions().iter().any(|action| {
            action.kind == ActionKind::FinalRelease && action.anchor == ActionAnchor::After(handoff)
        }),
        "the compiler-marked runtime handoff retains this exact closure pointer beyond the raw ABI call"
    );

    let transfer = plan
        .functions
        .iter_mut()
        .find(|function| function.symbol == Symbol::new("install"))
        .expect("install ownership plan")
        .actions
        .iter_mut()
        .find(|action| action.kind == ActionKind::EvidenceClosureTransfer)
        .expect("evidence closure transfer");
    transfer.destination = 3;
    let before = print_module(&ctx, module.op());
    assert!(materialize(&mut ctx, module, &plan).is_err());
    assert_eq!(print_module(&ctx, module.op()), before);
}

#[test]
fn compiler_owned_evidence_closure_transfers_cover_both_runtime_destinations() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !_closure = adt.struct() {name = @_closure, fields = [[@func_ptr, core.i32], [@env, tribute_rt.anyref]]}
  func.func @install(%evidence: core.ptr, %prompt: core.i32) -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %tr = adt.struct_new %code, %env {type = !_closure} : !_closure
    %handler = adt.struct_new %code, %env {type = !_closure} : !_closure
    %extended = effect.extend %evidence, %prompt, %tr, %handler {ability_ref = core.ability_ref() {name = @State}} : core.ptr
    func.return
  }
}"#,
    );
    lower_evidence_to_native(&mut ctx, module);
    let plan = build_native_ownership_plan(&ctx, module).expect("typed ownership plan");
    let install = plan.function(Symbol::new("install")).unwrap();
    let transfers = install
        .actions()
        .iter()
        .filter(|action| action.kind == ActionKind::EvidenceClosureTransfer)
        .map(|action| action.destination)
        .collect::<Vec<_>>();
    assert_eq!(transfers, [3, 4]);
}

#[test]
fn internal_closure_raw_pointer_handoff_outside_native_evidence_fails_closed() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !_closure = adt.struct() {name = @_closure, fields = [[@func_ptr, core.i32], [@env, tribute_rt.anyref]]}
  func.func @escape(%value: core.ptr) -> core.nil attributes {abi = "C"} {
    func.unreachable
  }
  func.func @install() -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %closure = adt.struct_new %code, %env {type = !_closure} : !_closure
    %raw = core.unrealized_conversion_cast %closure : core.ptr
    %result = func.call %raw {callee = @escape} : core.nil
    func.return
  }
}"#,
        "internal _closure to core.ptr handoff lacks compiler-owned native evidence provenance",
    );
}

#[test]
fn materialization_releases_the_exact_replaced_field_and_fails_before_mutation() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !NodeRef = adt.typeref() {name = @Node}
  !Node = adt.struct() {name = @Node, fields = [[@next, !NodeRef]]}
  func.func @replace(%node: !Node, %old: !NodeRef, %new: !NodeRef) -> core.nil {
    adt.struct_set %node, %new {field = 0, type = !Node}
    func.return
  }
}"#,
    );
    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    let materialized = print_module(&ctx, module.op());
    let retain = materialized.rfind("tribute_rt.retain").unwrap();
    let get = materialized.find("adt.struct_get").unwrap();
    let release = materialized.find("tribute_rt.release %").unwrap();
    let set = materialized.find("adt.struct_set").unwrap();
    assert!(
        retain < get && get < release && release < set,
        "{materialized}"
    );
    assert!(materialized.contains("alloc_size = 16"), "{materialized}");

    let (mut ctx, module, mut stale) = build(
        r#"core.module @test {
  !NodeRef = adt.typeref() {name = @Node}
  !Node = adt.struct() {name = @Node, fields = [[@next, !NodeRef]]}
  func.func @replace(%node: !Node, %new: !NodeRef) -> core.nil {
    adt.struct_set %node, %new {field = 0, type = !Node}
    func.return
  }
}"#,
    );
    let action = stale.functions[0]
        .actions
        .iter_mut()
        .find(|action| action.kind == ActionKind::ReleaseReplacedField)
        .unwrap();
    action.destination = 1;
    let before = print_module(&ctx, module.op());
    assert!(materialize(&mut ctx, module, &stale).is_err());
    assert_eq!(print_module(&ctx, module.op()), before);
}

#[test]
fn duplicate_owning_destinations_and_null_are_explicit() {
    let (_ctx, _module, plan) = build(
        r#"core.module @test {
  !NodeRef = adt.typeref() {name = @Node}
  !Node = adt.struct() {name = @Node, fields = [[@next, !NodeRef], [@other, !NodeRef]]}
  func.func @duplicate(%value: !NodeRef) -> !NodeRef {
    %node = adt.struct_new %value, %value {type = !Node} : !NodeRef
    func.return %node
  }
  func.func @null() -> !NodeRef {
    %null = adt.ref_null {type = !NodeRef} : !NodeRef
    func.return %null
  }
  func.func @replace(%node: !NodeRef, %value: !NodeRef) -> core.nil {
    adt.struct_set %node, %value {field = 0, type = !Node}
    func.return
  }
}"#,
    );
    let duplicate = plan.function(Symbol::new("duplicate")).unwrap();
    assert_eq!(count(duplicate, ActionKind::StoreAcquire), 2);
    assert_eq!(count(duplicate, ActionKind::ReturnTransfer), 1);
    let null = plan.function(Symbol::new("null")).unwrap();
    assert_eq!(count(null, ActionKind::ReturnTransfer), 1);
    assert_eq!(count(null, ActionKind::EntryAcquire), 0);
    assert_eq!(count(null, ActionKind::FinalRelease), 0);
    let replace = plan.function(Symbol::new("replace")).unwrap();
    assert_eq!(count(replace, ActionKind::StoreAcquire), 1);
    assert_eq!(count(replace, ActionKind::ReleaseReplacedField), 1);
}

#[test]
fn borrowed_load_return_acquires_a_transfer_unit() {
    let (_ctx, _module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Box = adt.struct() {name = @Box, fields = [[@child, !ChildRef]]}
  !BoxRef = adt.typeref() {name = @Box}
  func.func @load(%owner: !BoxRef) -> !ChildRef {
    %child = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    func.return %child
  }
}"#,
    );
    let function = plan.function(Symbol::new("load")).unwrap();
    assert_eq!(function.entries(), [EntryOwnership::Borrowed]);
    assert_eq!(count(function, ActionKind::BorrowLoad), 1);
    assert_eq!(count(function, ActionKind::CopyAcquire), 1);
    assert_eq!(count(function, ActionKind::ReturnTransfer), 1);
    assert_eq!(count(function, ActionKind::FinalRelease), 0);
}

#[test]
fn compatible_cast_and_enum_projection_preserve_borrowed_ownership() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Choice = adt.enum() {name = @Choice, variants = [[@Some, [!ChildRef]]]}
  !ChoiceRef = adt.typeref() {name = @Choice}
  func.func @load(%choice: !ChoiceRef) -> !ChildRef {
    %erased = adt.ref_cast %choice {type = tribute_rt.anyref} : tribute_rt.anyref
    %restored = adt.ref_cast %erased {type = !ChoiceRef} : !ChoiceRef
    %child = adt.variant_get %restored {type = !Choice, tag = @Some, field = 0} : !ChildRef
    func.return %child
  }
}"#,
    );
    let function = plan.function(Symbol::new("load")).unwrap();
    assert_eq!(function.entries(), [EntryOwnership::Borrowed]);
    assert_eq!(count(function, ActionKind::BorrowLoad), 1);
    assert_eq!(count(function, ActionKind::CopyAcquire), 1);
    assert_eq!(count(function, ActionKind::ReturnTransfer), 1);
    assert_eq!(count(function, ActionKind::FinalRelease), 0);

    let mut projection = None;
    walk_module(&ctx, module, |op| {
        if adt::VariantGet::matches(&ctx, op) {
            projection = Some(op);
        }
    });
    let projection = projection.expect("variant projection");
    for (key, invalid) in [
        (
            Symbol::new("tag"),
            trunk_ir::Attribute::Symbol(Symbol::new("Missing")),
        ),
        (Symbol::new("field"), trunk_ir::Attribute::Int(1)),
    ] {
        let original = ctx
            .op(projection)
            .attributes
            .get(key)
            .expect("projection attribute")
            .clone();
        ctx.op_mut(projection).attributes.insert(key, invalid);
        let before = print_module(&ctx, module.op());
        assert!(build_native_ownership_plan(&ctx, module).is_err());
        assert_eq!(print_module(&ctx, module.op()), before);
        ctx.op_mut(projection).attributes.insert(key, original);
    }
}

#[test]
fn malformed_projection_arity_fails_before_mutation() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Box = adt.struct() {name = @Box, fields = [[@child, !ChildRef]]}
  !BoxRef = adt.typeref() {name = @Box}
  func.func @load(%owner: !BoxRef) -> !ChildRef {
    %child, %extra = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef, !ChildRef
    func.return %child
  }
}"#,
        "ADT projection must have exactly one result",
    );
}

#[test]
fn early_native_terminators_and_successors_fail_before_mutation() {
    for ir in [
        r#"core.module @test {
  func.func @f() -> core.nil {
    func.return
    func.return
  }
}"#,
        r#"core.module @test {
  func.func @f() -> core.nil {
    ^entry:
      test.jump [^exit]
      func.return
    ^exit:
      func.return
  }
}"#,
    ] {
        assert_plan_error_unchanged(
            ir,
            "control-flow operation precedes the final block operation",
        );
    }
}

#[test]
fn physical_empty_results_reject_values_before_mutation() {
    for result in ["core.nil", "core.never"] {
        let ir = format!(
            r#"core.module @test {{
  func.func @f() -> {result} {{
    %value = arith.const {{value = 1}} : core.i32
    func.return %value
  }}
}}"#
        );
        assert_plan_error_unchanged(
            &ir,
            "function return differs from the exact callable signature",
        );
    }
}

#[test]
fn cross_block_borrowed_load_keeps_owner_alive_without_releasing_the_load() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Box = adt.struct() {name = @Box, fields = [[@child, !ChildRef]]}
  !BoxRef = adt.typeref() {name = @Box}
  func.func @observe(%child: !ChildRef) -> core.i32 {
    %value = adt.struct_get %child {field = 0, type = !Child} : core.i32
    func.return %value
  }
  func.func @load(%child: !ChildRef) -> core.nil {
    ^entry:
      %owner = adt.struct_new %child {type = !Box} : !BoxRef
      %loaded = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
      cf.br [^next]
    ^next:
      %seen = func.call %loaded {callee = @observe} : core.i32
      func.return
  }
}"#,
    );
    let function = plan.function(Symbol::new("load")).unwrap();
    let body = ctx.op(function.operation()).regions[0];
    let [entry, next] = ctx.region(body).blocks.as_slice() else {
        panic!("two-block fixture")
    };
    let owner = ctx.op_result(ctx.block(*entry).ops[0], 0);
    let loaded = ctx.op_result(ctx.block(*entry).ops[1], 0);
    let observe = ctx.block(*next).ops[0];

    assert!(function.actions().iter().any(|action| {
        action.kind == ActionKind::BorrowLoad
            && action.value == loaded
            && action.anchor == ActionAnchor::After(ctx.block(*entry).ops[1])
    }));
    assert!(
        !function
            .actions()
            .iter()
            .any(|action| { action.kind == ActionKind::FinalRelease && action.value == loaded })
    );
    assert!(function.actions().iter().any(|action| {
        action.kind == ActionKind::FinalRelease
            && action.value == owner
            && action.anchor == ActionAnchor::After(observe)
    }));
}

#[test]
fn cfg_copy_and_tail_dying_value_actions_are_complete() {
    let (ctx, module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @branch(%value: !R) -> core.nil attributes {tribute.calling_convention = 2} {
    ^entry:
      cf.br %value, %value [^merge]
    ^merge(%left: !R, %right: !R):
      func.unreachable
  }
  func.func @tail(%sent: !R, %dying: !R) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %sent {callee = @sink, tribute.calling_convention = 2}
  }
  func.func @sink(%value: !R) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
}"#,
    );
    let branch = plan.function(Symbol::new("branch")).unwrap();
    assert_eq!(count(branch, ActionKind::CopyAcquire), 1);
    assert_eq!(count(branch, ActionKind::FinalRelease), 2);
    let tail = plan.function(Symbol::new("tail")).unwrap();
    assert_eq!(count(tail, ActionKind::TailTransfer), 1);
    assert_eq!(count(tail, ActionKind::FinalRelease), 1);
    assert!(
        tail.actions()
            .iter()
            .all(|action| { !matches!(action.anchor, ActionAnchor::After(_)) })
    );

    let tail_index = plan
        .functions
        .iter()
        .position(|function| function.symbol == Symbol::new("tail"))
        .unwrap();
    let body = ctx.op(plan.functions[tail_index].operation).regions[0];
    let block = ctx.region(body).blocks[0];
    let tail_op = *ctx.block(block).ops.last().unwrap();
    let action_index = plan.functions[tail_index]
        .actions
        .iter()
        .position(|action| action.kind == ActionKind::TailTransfer)
        .unwrap();
    let mut after_tail = plan.clone();
    after_tail.functions[tail_index].actions[action_index].anchor = ActionAnchor::After(tail_op);
    assert!(after_tail.validate_against(&ctx, module).is_err());
}

#[test]
fn enum_rtti_uses_the_same_nested_managed_predicate() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Choice = adt.enum() {name = @Choice, variants = [[@None, []], [@Some, [!ChildRef, core.ptr]], [@Bytes, [core.bytes]]]}
  func.func @some(%child: !ChildRef, %raw: core.ptr) -> !Choice {
    %choice = adt.variant_new %child, %raw {tag = @Some, type = !Choice} : !Choice
    func.return %choice
  }
}"#,
    );
    let entry = plan
        .rtti_types()
        .iter()
        .find(|entry| {
            ctx.types.get(entry.ty).attrs.get_symbol("name") == Some(Symbol::new("Choice"))
        })
        .unwrap();
    assert!(matches!(
        &entry.fields,
        ManagedFieldBitmap::Enum(variants)
            if variants == &[vec![], vec![true, false], vec![false]]
    ));
}

#[test]
fn unmanaged_physical_and_buffer_types_never_receive_actions() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Marker = adt.struct() {name = @EvidenceMarker, fields = [[@code, core.ptr]]}
  !Evidence = core.array(!Marker)
  func.func @raw(%raw: core.ptr, %bytes: core.bytes, %array: core.array(core.i32), %evidence: !Evidence, %code: core.func(core.nil)) -> core.ptr {
    func.return %raw
  }
}"#,
    );
    let function = plan.function(Symbol::new("raw")).unwrap();
    assert_eq!(function.entries(), [EntryOwnership::Plain; 5]);
    assert!(function.actions().is_empty());
    for ty in ctx
        .op(function.operation())
        .attributes
        .get_type("type")
        .and_then(|ty| core::Func::from_type_ref(&ctx, ty))
        .unwrap()
        .params(&ctx)
    {
        assert!(!plan.is_managed_type(&ctx, *ty));
    }
}

#[test]
fn semantic_closure_values_are_managed_by_the_typed_contract() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Closure = closure.closure(core.func(core.i32, core.i32))
  func.func @identity(%closure: !Closure) -> !Closure {
    func.return %closure
  }
}"#,
    );
    let function = plan.function(Symbol::new("identity")).unwrap();
    let body = ctx.op(function.operation()).regions[0];
    let entry = ctx.region(body).blocks[0];
    assert!(plan.is_managed_type(&ctx, ctx.value_ty(ctx.block_args(entry)[0])));
    assert_eq!(function.entries(), [EntryOwnership::Retained]);
    assert_eq!(count(function, ActionKind::EntryAcquire), 1);
    assert_eq!(count(function, ActionKind::ReturnTransfer), 1);
}

#[test]
fn semantic_closure_release_uses_its_compiler_generated_allocation_layout() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Closure = closure.closure(core.func(core.nil, tribute_rt.anyref))
  func.func @target(%environment: tribute_rt.anyref) -> core.nil {
    func.return
  }
  func.func @consume(%closure: !Closure) -> core.nil {
    func.return
  }
  func.func @main() -> core.nil {
    %environment = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %closure = closure.new %environment {func_ref = @target} : !Closure
    func.call %closure {callee = @consume} : core.nil
    func.return
  }
}"#,
    );
    crate::closure_lower::lower_closures(&mut ctx, module);

    let plan = build_native_ownership_plan(&ctx, module).expect("typed ownership plan");
    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    let materialized = print_module(&ctx, module.op());

    // The generated closure pair has an i32 function slot and an anyref
    // environment slot: 16-byte payload plus the 8-byte RC header.
    assert!(materialized.contains("alloc_size = 24"), "{materialized}");
}

#[test]
fn direct_indirect_return_and_tail_contracts_are_typed() {
    let (_ctx, _module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @ordinary(%value: !R) -> !R {
    func.return %value
  }
  func.func @observe(%value: !R) -> core.i32 {
    %seen = adt.struct_get %value {field = 0, type = !Layout} : core.i32
    func.return %seen
  }
  func.func @caller(%value: !R, %callee: core.func(!R, !R)) -> !R {
    %seen = func.call %value {callee = @observe} : core.i32
    %direct = func.call %value {callee = @ordinary} : !R
    %indirect = func.call_indirect %callee, %direct {func.indirect_call_signature = core.func(!R, !R)} : !R
    func.return %indirect
  }
  func.func @tail(%value: !R) -> core.nil attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @sink, tribute.calling_convention = 2}
  }
  func.func @sink(%value: !R) -> core.nil attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
}"#,
    );
    let caller = plan.function(Symbol::new("caller")).unwrap();
    assert_eq!(count(caller, ActionKind::CallBorrow), 1);
    assert_eq!(count(caller, ActionKind::CallRetain), 2);
    assert_eq!(count(caller, ActionKind::ReturnTransfer), 1);
    let tail = plan.function(Symbol::new("tail")).unwrap();
    assert_eq!(count(tail, ActionKind::TailTransfer), 1);
    assert!(
        !tail
            .actions()
            .iter()
            .any(|action| matches!(action.anchor, ActionAnchor::After(_)))
    );
}

#[test]
fn stale_identity_unsupported_regions_and_malformed_calls_fail_unchanged() {
    for ir in [
        r#"core.module @test {
  !R = adt.typeref() {name = @Missing}
  func.func @f(%value: !R) -> !R { func.return %value }
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%value: !R) -> !R {
    %x = scf.if %value : !R { func.return %value }
    func.return %x
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%value: !R, %callee: core.func(!R, !R)) -> !R {
    %x = func.call_indirect %callee {func.indirect_call_signature = core.func(!R, !R)} : !R
    func.return %x
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !First = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  !Second = adt.struct() {name = @R, fields = [[@x, core.i64]]}
  func.func @f(%value: !R) -> !R { func.return %value }
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @foreign(%value: !R) -> !R
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%raw: core.ptr) -> !R {
    %value = adt.ref_cast %raw {type = !R} : !R
    func.return %value
  }
}"#,
        r#"core.module @test {
  !A = adt.typeref() {name = @A}
  !ALayout = adt.struct() {name = @A, fields = [[@x, core.i32]]}
  !B = adt.typeref() {name = @B}
  !BLayout = adt.struct() {name = @B, fields = [[@x, core.i32]]}
  func.func @f(%value: !A) -> !B {
    %wrong = adt.ref_cast %value {type = !B} : !B
    func.return %wrong
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%value: !R) -> !R {
    %result = func.call %value {callee = @missing} : !R
    func.return %result
  }
}"#,
        r#"core.module @test {
  !ARef = adt.typeref() {name = @A}
  !A = adt.struct() {name = @A, fields = [[@x, core.i32]]}
  !BRef = adt.typeref() {name = @B}
  !B = adt.struct() {name = @B, fields = [[@x, core.i32]]}
  !Env = adt.struct() {name = @Env, fields = [[@value, !ARef]]}
  func.func @f(%value: !BRef) -> core.nil {
    %env = adt.struct_new %value {type = !Env} : !Env
    func.return
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%value: !R, %callee: core.func(!R, !R)) -> !R {
    %result = func.call_indirect %callee, %value : !R
    func.return %result
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%value: !R) -> !R {
    ^entry:
      cf.br [^next]
    ^next(%next: !R):
      func.return %next
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%value: !R) -> !R {
    func.return
  }
}"#,
    ] {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let before = print_module(&ctx, module.op());
        assert!(build_native_ownership_plan(&ctx, module).is_err());
        assert_eq!(print_module(&ctx, module.op()), before);
    }
}

#[test]
fn nominal_layout_lookup_ignores_unreachable_interner_entries() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%value: !R) -> !R { func.return %value }
}"#,
    );
    let i64_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i64")).build());
    let stale = ctx.types.intern(
        TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("name", trunk_ir::Attribute::Symbol(Symbol::new("R")))
            .attr(
                "fields",
                trunk_ir::Attribute::List(vec![trunk_ir::Attribute::List(vec![
                    trunk_ir::Attribute::Symbol(Symbol::new("x")),
                    trunk_ir::Attribute::Type(i64_ty),
                ])]),
            )
            .build(),
    );
    assert!(ctx.type_alias_by_type(stale).is_none());
    build_native_ownership_plan(&ctx, module)
        .expect("an unreachable stale layout must not shadow the module declaration");

    let mut ambiguous_ctx = IrContext::new();
    let ambiguous = parse_test_module(
        &mut ambiguous_ctx,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !First = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  !Second = adt.struct() {name = @R, fields = [[@x, core.i64]]}
  func.func @f(%value: !R, %first: !First, %second: !Second) -> !R {
    func.return %value
  }
}"#,
    );
    assert!(build_native_ownership_plan(&ambiguous_ctx, ambiguous).is_err());
}

#[test]
fn plan_order_is_deterministic_and_duplicate_actions_fail_validation() {
    let (ctx, module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @f(%value: !R) -> !R { func.return %value }
}"#,
    );
    let second = build_native_ownership_plan(&ctx, module).unwrap();
    assert_eq!(plan.functions(), second.functions());
    assert_eq!(plan.rtti_types(), second.rtti_types());
    let mut invalid = plan.clone();
    let action = invalid.functions[0].actions[0];
    invalid.functions[0].actions.push(action);
    assert!(invalid.validate_against(&ctx, module).is_err());

    let mut conflicting = plan.clone();
    let mut action = conflicting.functions[0].actions[0];
    action.kind = ActionKind::FinalRelease;
    conflicting.functions[0].actions[0] = action;
    conflicting.functions[0].actions.push(OwnershipAction {
        kind: ActionKind::ReturnTransfer,
        ..action
    });
    assert!(conflicting.validate_against(&ctx, module).is_err());

    let mut duplicate_function = plan.clone();
    duplicate_function.functions.push(plan.functions[0].clone());
    assert!(duplicate_function.validate_against(&ctx, module).is_err());
}

#[test]
fn stale_plan_and_ambiguous_rtti_rewrites_fail_without_mutation() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !A = adt.struct() {name = @A, fields = [[@x, core.i32]]}
  !B = adt.struct() {name = @B, fields = [[@x, core.i32]]}
  !R = adt.typeref() {name = @A}
  func.func @make(%x: core.i32, %value: !R, %raw: core.ptr) -> !R {
    %a = adt.struct_new %x {type = !A} : !R
    %b = adt.struct_new %x {type = !B} : !B
    func.return %a
  }
  func.func @other(%value: !R) -> !R { func.return %value }
}"#,
    );
    let before = print_module(&ctx, module.op());
    plan.validate_against(&ctx, module)
        .expect("freshly built plan must validate");
    let [first, second] = plan.rtti_types() else {
        panic!("two exact RTTI layouts")
    };
    assert!(
        plan.remap_rtti_types(
            &ctx,
            module,
            &[TypeRewrite {
                source: first.ty,
                target: second.ty,
            }],
        )
        .is_err(),
        "two typed layouts may not collapse onto one physical identity"
    );

    let mut stale_module = plan.clone();
    let other = parse_test_module(&mut ctx, "core.module @other {}");
    stale_module.module = other.op();
    assert!(stale_module.validate_against(&ctx, module).is_err());

    let mut duplicate_rtti = plan.clone();
    duplicate_rtti.rtti_types.push(first.clone());
    assert!(duplicate_rtti.validate_against(&ctx, module).is_err());

    let mut stale_bitmap = plan.clone();
    stale_bitmap.rtti_types[0].fields = ManagedFieldBitmap::Struct(vec![true]);
    assert!(stale_bitmap.validate_against(&ctx, module).is_err());

    let function = &plan.functions[0];
    let body = ctx.op(function.operation).regions[0];
    let entry = ctx.region(body).blocks[0];
    let raw = ctx.block_args(entry)[2];
    let mut unmanaged_action = plan.clone();
    unmanaged_action.functions[0].actions[0].value = raw;
    assert!(unmanaged_action.validate_against(&ctx, module).is_err());

    let mut stale_function = plan.clone();
    stale_function.functions[0].operation = module.op();
    assert!(stale_function.validate_against(&ctx, module).is_err());

    let mut stale_entry = plan.clone();
    stale_entry.functions[0].entries[1] = EntryOwnership::Plain;
    assert!(stale_entry.validate_against(&ctx, module).is_err());

    let mut stale_anchor = plan.clone();
    stale_anchor.functions[0].actions[0].anchor = ActionAnchor::Before(module.op());
    assert!(stale_anchor.validate_against(&ctx, module).is_err());

    let other = plan.function(Symbol::new("other")).unwrap();
    let other_body = ctx.op(other.operation).regions[0];
    let other_entry = ctx.region(other_body).blocks[0];
    let mut stale_value = plan.clone();
    stale_value.functions[0].actions[0].value = ctx.block_args(other_entry)[0];
    assert!(stale_value.validate_against(&ctx, module).is_err());
    assert_eq!(print_module(&ctx, module.op()), before);
}

#[test]
fn plan_revalidation_requires_the_exact_reachable_function_set() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  func.func @first() -> core.nil { func.return }
  func.func @second() -> core.nil { func.return }
}"#,
    );

    let mut incomplete = plan.clone();
    incomplete.functions.pop();
    let before = print_module(&ctx, module.op());
    let error = incomplete
        .validate_against(&ctx, module)
        .expect_err("a plan may not omit a reachable function");
    assert!(error.to_string().contains("reachable function identities"));
    assert_eq!(print_module(&ctx, module.op()), before);

    let detached = plan.functions[1].operation;
    let parent = ctx.op(detached).parent_block.expect("module block");
    ctx.remove_op_from_block(parent, detached);
    let before = print_module(&ctx, module.op());
    let error = plan
        .validate_against(&ctx, module)
        .expect_err("a plan may not retain a detached function");
    assert!(error.to_string().contains("reachable function identities"));
    assert_eq!(print_module(&ctx, module.op()), before);
}

#[test]
fn closure_rtti_bitmap_follows_the_exact_func_to_clif_type_rewrite() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Closure = adt.struct() {name = @_closure, fields = [[@func_ptr, core.i32], [@env, tribute_rt.anyref]]}
  func.func @make(%code: core.i32, %env: tribute_rt.anyref) -> !Closure {
    %closure = adt.struct_new %code, %env {type = !Closure} : !Closure
    func.return %closure
  }
}"#,
    );
    let plan = build_native_ownership_plan(&ctx, module).expect("typed ownership plan");
    let semantic = plan.rtti_types()[0].ty;
    assert!(matches!(
        &plan.rtti_types()[0].fields,
        ManagedFieldBitmap::Struct(fields) if fields == &[false, true]
    ));

    let (type_converter, _) = native_type_converter(&mut ctx);
    let lowering = func_to_clif::lower(&mut ctx, module, type_converter).expect("func_to_clif");
    assert_eq!(lowering.rtti_layout_rewrites().len(), 1);
    assert_eq!(lowering.rtti_layout_rewrites()[0].source, semantic);
    let remapped = plan
        .remap_rtti_types(&ctx, module, lowering.rtti_layout_rewrites())
        .expect("exact closure layout rewrite");
    assert_ne!(remapped[0].ty, semantic);
    assert_eq!(remapped[0].fields, plan.rtti_types()[0].fields);
}

#[test]
fn rtti_identity_never_falls_back_to_same_name_or_shape() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Exact = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  !SameNameShape = adt.struct() {name = @R, fields = [[@x, core.i32]], test.identity = 1}
  !Stale = adt.struct() {name = @Stale, fields = [[@x, core.i32]]}
  func.func @make(%x: core.i32) -> !Exact {
    %value = adt.struct_new %x {type = !Exact} : !Exact
    func.return %value
  }
}"#,
    );
    let plan = build_native_ownership_plan(&ctx, module).expect("typed ownership plan");
    let mut allocation = None;
    walk_module(&ctx, module, |op| {
        if adt::StructNew::matches(&ctx, op) {
            allocation = Some(op);
        }
    });
    let allocation = allocation.unwrap();
    let candidates = ctx
        .types
        .iter()
        .filter_map(|(ty, data)| {
            (data.dialect == Symbol::new("adt")
                && data.name == Symbol::new("struct")
                && ty != plan.rtti_types()[0].ty)
                .then_some(ty)
        })
        .collect::<Vec<_>>();
    assert_eq!(candidates.len(), 2);

    for &candidate in &candidates {
        ctx.op_mut(allocation)
            .attributes
            .insert(Symbol::new("type"), trunk_ir::Attribute::Type(candidate));
        assert!(plan.remap_rtti_types(&ctx, module, &[]).is_err());
        assert!(
            plan.remap_rtti_types(
                &ctx,
                module,
                &[TypeRewrite {
                    source: plan.rtti_types()[0].ty,
                    target: candidate,
                }],
            )
            .is_ok(),
            "only an explicit exact identity rewrite may carry the bitmap"
        );
    }

    let exact = plan.rtti_types()[0].ty;
    assert!(
        plan.remap_rtti_types(
            &ctx,
            module,
            &[TypeRewrite {
                source: candidates[0],
                target: candidates[1],
            }],
        )
        .is_err(),
        "a rewrite source not present in the typed plan is stale"
    );
    assert!(
        plan.remap_rtti_types(
            &ctx,
            module,
            &[
                TypeRewrite {
                    source: exact,
                    target: candidates[0],
                },
                TypeRewrite {
                    source: exact,
                    target: candidates[1],
                },
            ],
        )
        .is_err(),
        "one typed identity cannot map ambiguously"
    );
}

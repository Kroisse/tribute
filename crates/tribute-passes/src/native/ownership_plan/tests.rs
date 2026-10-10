use super::*;
use crate::native::evidence::lower_evidence_to_native;
use crate::native::rc_materialization::materialize;
use crate::native::type_converter::native_type_converter;
use trunk_ir::op_interface::{
    BranchModel, BranchOps, BranchSuccessor, BranchSuccessors, ControlFlowInterfaceError,
};
use trunk_ir::parser::parse_test_module;
use trunk_ir::printer::print_module;
use trunk_ir::types::TypeDataBuilder;
use trunk_ir_cranelift_backend::passes::func_to_clif;

use crate::test_support::assert_unchanged_on_error;

/// Build the plan with the production ownership policy and a fresh cache.
fn production_plan(
    ctx: &IrContext,
    module: Module,
) -> Result<NativeOwnershipPlan, OwnershipPlanError> {
    build_native_ownership_plan(
        ctx,
        module,
        NativeOwnershipPlanOptions::production(),
        &mut Default::default(),
    )
}

// Branch terminators whose interface models are deliberately wrong.
#[trunk_ir::dialect]
mod test {
    fn incomplete_branch() {
        #[successor(exit)]
        {}
    }

    fn missing_edge_branch() {
        #[successor(exit)]
        {}
    }

    fn reversed_branch() {
        #[successor(left)]
        {}
        #[successor(right)]
        {}
    }

    fn multi_forwarding_branch(value: Value<_>) {
        #[successor(left)]
        {}
        #[successor(right)]
        {}
    }
}

impl BranchModel for IncompleteBranch {
    fn successors(self, _ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        Err(ControlFlowInterfaceError::new(
            "test branch interface is incomplete",
        ))
    }
}

impl BranchModel for MissingEdgeBranch {
    fn successors(self, _ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        Ok(BranchSuccessors::default())
    }
}

impl BranchModel for ReversedBranch {
    fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        Ok(BranchSuccessors::new([
            BranchSuccessor::new(self.right(ctx), []),
            BranchSuccessor::new(self.left(ctx), []),
        ]))
    }
}

impl BranchModel for MultiForwardingBranch {
    fn successors(self, ctx: &IrContext) -> Result<BranchSuccessors, ControlFlowInterfaceError> {
        let forwarded = self.value(ctx);
        Ok(BranchSuccessors::new([
            BranchSuccessor::new(self.left(ctx), [forwarded]),
            BranchSuccessor::new(self.right(ctx), [forwarded]),
        ]))
    }
}

trunk_ir::inventory::submit! { BranchOps::register::<IncompleteBranch>() }
trunk_ir::inventory::submit! { BranchOps::register::<MissingEdgeBranch>() }
trunk_ir::inventory::submit! { BranchOps::register::<ReversedBranch>() }
trunk_ir::inventory::submit! { BranchOps::register::<MultiForwardingBranch>() }

fn build(ir: &str) -> (IrContext, Module, NativeOwnershipPlan) {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    let before = print_module(&ctx, module.op());
    let plan = production_plan(&ctx, module).expect("typed ownership plan");
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
    let error =
        assert_unchanged_on_error(&mut ctx, module, |ctx, module| production_plan(ctx, module));
    assert!(
        error.to_string().contains(expected),
        "unexpected error: {error}"
    );
}

#[test]
fn bodyless_scalar_declarations_need_no_target_binding_or_ownership_actions() {
    let (ctx, module, plan) = build(
        r#"core.module @test {
        func.func @external(%value: core.i32) -> core.i32
        func.func @caller(%value: core.i32) -> core.i32 {
            %result = func.call %value {callee = @external} : core.i32
            func.return %result
        }
    }"#,
    );
    assert!(plan.function(&Symbol::new("external")).is_none());
    assert!(
        plan.function(&Symbol::new("caller"))
            .unwrap()
            .actions()
            .is_empty()
    );
    plan.validate_against(&ctx, module).unwrap();
    assert_plan_error_unchanged(
        r#"core.module @test {
            func.func @external(%value: core.i32) -> core.i32
            func.func @caller() -> core.i32 {
                %result = func.call {callee = @external} : core.i32
                func.return %result
            }
        }"#,
        "call arguments differ from the exact callable signature",
    );
    assert_plan_error_unchanged(
        "core.module @test { func.func {sym_name = \"bad\", type = core.i32} }",
        "bodyless function lacks exact signature",
    );
    assert_plan_error_unchanged(
        "core.module @test { func.func @managed(%value: tribute_rt.anyref) -> tribute_rt.anyref }",
        "bodyless native declaration exposes a managed reference",
    );
}

#[test]
fn malformed_callable_bodies_fail_before_ownership_analysis_without_mutation() {
    assert_plan_error_unchanged(
        "core.module @test { func.func {sym_name = \"empty\", type = func.func_sig<() -> ()>} {} }",
        "func.func @empty: body has no entry block",
    );
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        "core.module @test { func.func {sym_name = \"extra\", type = func.func_sig<() -> ()>} { func.return } }",
    );
    let op = module.ops(&ctx)[0];
    let extra = ctx.create_region(trunk_ir::RegionData {
        location: ctx.op(op).location,
        blocks: Default::default(),
        parent_op: None,
    });
    ctx.push_op_region(op, extra);
    let before: trunk_ir::RegionList = ctx.op_regions(op).collect();
    let error = production_plan(&ctx, module).expect_err("multiple bodies");
    assert!(
        error
            .to_string()
            .contains("func.func @extra: has more than one body region"),
        "{error}"
    );
    assert!(ctx.op_regions(op).eq(before));
}

#[test]
fn revalidation_rejects_a_declaration_changed_to_an_empty_body() {
    let (mut ctx, module, plan) = build(
        "core.module @test { func.func {sym_name = \"external\", type = func.func_sig<() -> ()>} }",
    );
    let op = module.ops(&ctx)[0];
    let body = ctx.create_region(trunk_ir::RegionData {
        location: ctx.op(op).location,
        blocks: Default::default(),
        parent_op: None,
    });
    ctx.push_op_region(op, body);
    let before = print_module(&ctx, module.op());
    let error = plan
        .validate_against(&ctx, module)
        .expect_err("malformed declaration must not be omitted");
    assert!(
        error
            .to_string()
            .contains("func.func @external: body has no entry block"),
        "{error}"
    );
    assert_eq!(print_module(&ctx, module.op()), before);
}

#[test]
fn ordinary_result_contract_requires_one_value_and_preserves_zero_width_results() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
        func.func @values(%one: core.i32, %two: core.i32, %nil: core.nil, %never: core.never) { func.unreachable }
    }"#,
    );
    let function = func::Func::from_op(&ctx, module.ops(&ctx)[0]).unwrap();
    let entry = ctx.region(function.body(&ctx)).blocks[0];
    let values = ctx.block_args(entry);
    let managed = HashSet::default();
    let check = |values: &[ValueRef], expected: &[TypeRef]| {
        actions::validate_result_contract(&ctx, values, expected, &managed, "test result")
    };
    let scalar = ctx.value_ty(values[0]);
    assert!(check(&[], &[scalar]).is_err());
    assert!(check(&values[..1], &[scalar]).is_ok());
    assert!(check(&values[..2], &[scalar]).is_err());
    assert!(check(&[], &[]).is_ok());
    assert!(check(&values[..1], &[]).is_err());
    for &value in &values[2..] {
        let ty = ctx.value_ty(value);
        assert!(check(&[], &[ty]).is_ok());
        assert!(check(&[value], &[ty]).is_ok());
        assert!(check(&[value, value], &[ty]).is_err());
    }
}

#[test]
fn typed_plan_options_preserve_or_elide_only_proven_parameter_and_field_borrows() {
    let ir = r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
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
    let preserved = build_native_ownership_plan(
        &ctx,
        module,
        NativeOwnershipPlanOptions {
            elide_proven_borrowed_parameters: false,
            elide_proven_field_borrows: false,
        },
        &mut Default::default(),
    )
    .expect("preserved typed plan");
    let elided = production_plan(&ctx, module).expect("elided typed plan");

    let preserved_forward = preserved.function(&Symbol::new("forward")).unwrap();
    let elided_forward = elided.function(&Symbol::new("forward")).unwrap();
    assert_eq!(preserved_forward.entries(), [EntryOwnership::Retained]);
    assert_eq!(elided_forward.entries(), [EntryOwnership::Borrowed]);
    assert_eq!(count(preserved_forward, ActionKind::EntryAcquire), 1);
    assert_eq!(count(elided_forward, ActionKind::EntryAcquire), 0);

    let load_op = ctx
        .op_region(
            elided.function(&Symbol::new("load")).unwrap().operation(),
            0,
        )
        .unwrap();
    let load = ctx.op_result(ctx.block(ctx.region(load_op).blocks[0]).ops[0], 0);
    let projection = ctx.block(ctx.region(load_op).blocks[0]).ops[0];
    let preserved_load = preserved.function(&Symbol::new("load")).unwrap();
    let elided_load = elided.function(&Symbol::new("load")).unwrap();
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
  !Frame = adt.struct<ContinuationFrame(value: core.i32)>
  !FrameRef = adt.typeref<{name = "ContinuationFrame"}>
  !Env = adt.struct<continuation_env(frame: !FrameRef, code: func.func_sig<() -> core.nil>)>
  func.func @capture(%frame: !FrameRef, %code: func.func_sig<() -> core.nil>) -> core.nil {
    %env = adt.struct_new %frame, %code {type = !Env} : !Env
    func.return
  }
}"#,
    );
    let function = plan.function(&Symbol::new("capture")).unwrap();
    assert_eq!(
        function.entries(),
        [EntryOwnership::Retained, EntryOwnership::Plain]
    );
    assert_eq!(count(function, ActionKind::EntryAcquire), 1);
    assert_eq!(count(function, ActionKind::StoreAcquire), 1);
    assert_eq!(count(function, ActionKind::FinalRelease), 2);
    assert_eq!(
        plan.rtti_types()[0].fields,
        [FieldKind::Managed, FieldKind::Raw]
    );
}

#[test]
fn continuation_frame_capture_materializes_the_typed_entry_and_store_actions() {
    let ir = r#"core.module @test {
  !Frame = adt.struct<ContinuationFrame(value: core.i32)>
  !FrameRef = adt.typeref<{name = "ContinuationFrame"}>
  !Env = adt.struct<continuation_env(frame: !FrameRef, code: func.func_sig<() -> core.nil>)>
  func.func @capture(%frame: !FrameRef, %code: func.func_sig<() -> core.nil>) -> core.nil {
    %env = adt.struct_new %frame, %code {type = !Env} : !Env
    func.return
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    let plan = production_plan(&ctx, module).expect("typed ownership plan");
    let capture = plan.function(&Symbol::new("capture")).unwrap();
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
  !Frame = adt.struct<ContinuationFrame(value: core.i32)>
  !FrameRef = adt.typeref<{name = "ContinuationFrame"}>
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: !FrameRef), {layout = "closure"}>
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
fn native_evidence_lowers_managed_closure_handoff_to_into_raw() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  !Evidence = core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>
  func.func @install(%evidence: !Evidence, %prompt: core.i32) -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %closure = adt.struct_new %code, %env {type = !_closure} : !_closure
    %extended = effect.extend %evidence, %prompt, %closure, %evidence {ability_ref = core.ability_ref<{name = "State"}>} : !Evidence
    func.return
  }
}"#,
    );
    lower_evidence_to_native(&mut ctx, module);
    let mut plan = production_plan(&ctx, module).expect("typed ownership plan");
    let install = plan.function(&Symbol::new("install")).unwrap();
    let transfer = install
        .actions()
        .iter()
        .find(|action| action.kind == ActionKind::IntoRawTransfer)
        .expect("into_raw transfer");
    let ActionAnchor::Before(into_raw) = transfer.anchor else {
        panic!("into_raw transfer must anchor the into_raw operation");
    };
    assert_eq!(count(install, ActionKind::IntoRawTransfer), 1);
    let lowered = print_module(&ctx, module.op());
    assert!(!lowered.contains("native_evidence_closure_transfer_destinations"));
    assert_eq!(
        lowered.matches("tribute_rt.into_raw").count(),
        1,
        "{lowered}"
    );
    assert!(
        !install.actions().iter().any(|action| {
            action.kind == ActionKind::FinalRelease
                && action.anchor == ActionAnchor::After(into_raw)
        }),
        "into_raw consumes this exact closure ownership unit"
    );

    let transfer = plan
        .functions
        .iter_mut()
        .find(|function| function.symbol == "install")
        .expect("install ownership plan")
        .actions
        .iter_mut()
        .find(|action| action.kind == ActionKind::IntoRawTransfer)
        .expect("into_raw transfer");
    transfer.destination = 1;
    assert_unchanged_on_error(&mut ctx, module, |ctx, module| {
        materialize(ctx, module, &plan)
    });
}

#[test]
fn native_evidence_lowers_a_managed_dispatcher_to_into_raw() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  !Evidence = core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>
  func.func @install(%evidence: !Evidence, %prompt: core.i32) -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %tr = adt.struct_new %code, %env {type = !_closure} : !_closure
    %extended = effect.extend %evidence, %prompt, %tr, %evidence {ability_ref = core.ability_ref<{name = "State"}>} : !Evidence
    func.return
  }
}"#,
    );
    lower_evidence_to_native(&mut ctx, module);
    let plan = production_plan(&ctx, module).expect("typed ownership plan");
    let install = plan.function(&Symbol::new("install")).unwrap();
    assert_eq!(count(install, ActionKind::IntoRawTransfer), 1);
    let lowered = print_module(&ctx, module.op());
    assert_eq!(
        lowered.matches("tribute_rt.into_raw").count(),
        1,
        "{lowered}"
    );
}

#[test]
fn internal_closure_raw_pointer_handoff_outside_native_evidence_fails_closed() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
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
        "is viewed as a raw pointer; use tribute_rt.into_raw",
    );
}

#[test]
fn into_raw_transfers_one_exact_closure_unit_without_materializing_rc() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  func.func @install() -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %closure = adt.struct_new %code, %env {type = !_closure} : !_closure
    %raw = tribute_rt.into_raw %closure : core.ptr
    func.return
  }
}"#,
    );
    let install = plan.function(&Symbol::new("install")).unwrap();
    assert_eq!(count(install, ActionKind::IntoRawTransfer), 1);
    let transferred = install
        .actions()
        .iter()
        .find(|action| action.kind == ActionKind::IntoRawTransfer)
        .expect("into_raw transfer")
        .value;
    assert!(
        !install.actions().iter().any(|action| {
            action.kind == ActionKind::FinalRelease && action.value == transferred
        })
    );

    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    let materialized = print_module(&ctx, module.op());
    assert!(
        materialized.contains("tribute_rt.into_raw"),
        "{materialized}"
    );
}

fn into_raw_fixture(transfers: &str) -> String {
    [
        r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  func.func @transfers() -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %closure = adt.struct_new %code, %env {type = !_closure} : !_closure
"#,
        transfers,
        r#"
    func.return
  }
}"#,
    ]
    .concat()
}

#[test]
fn nested_field_borrow_keeps_the_outer_owner_alive_through_the_last_use() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Inner = adt.struct<Inner(child: !ChildRef)>
  !InnerRef = adt.typeref<{name = "Inner"}>
  !Box = adt.struct<Box(inner: !InnerRef)>
  func.func @observe(%child: !ChildRef) -> core.i32 {
    %value = adt.struct_get %child {field = 0, type = !Child} : core.i32
    func.return %value
  }
  func.func @load(%child: !ChildRef) -> core.nil {
    %inner = adt.struct_new %child {type = !Inner} : !InnerRef
    %owner = adt.struct_new %inner {type = !Box} : !Box
    %loaded = adt.struct_get %owner {field = 0, type = !Box} : !InnerRef
    %erased = adt.ref_cast %loaded {type = tribute_rt.anyref} : tribute_rt.anyref
    %restored = adt.ref_cast %erased {type = !InnerRef} : !InnerRef
    %nested = adt.struct_get %restored {field = 0, type = !Inner} : !ChildRef
    %value = func.call %nested {callee = @observe} : core.i32
    func.return
  }
}"#,
    );
    let function = plan.function(&Symbol::new("load")).unwrap();
    let body = ctx.op_region(function.operation(), 0).unwrap();
    let block = ctx.region(body).blocks[0];
    let box_layout = ctx.type_alias_by_text("Box").unwrap();
    let mut owner = None;
    let mut call = None;
    for &op in &ctx.block(block).ops {
        if adt::StructNew::from_op(&ctx, op).is_ok_and(|new| new.r#type(&ctx) == box_layout) {
            owner = Some(ctx.op_result(op, 0));
        }
        if func::Call::matches(&ctx, op) {
            call = Some(op);
        }
    }
    let owner = owner.expect("outer owner");
    let call = call.expect("nested borrowed field use");
    assert!(function.actions().iter().any(|action| {
        action.kind == ActionKind::FinalRelease
            && action.value == owner
            && action.anchor == ActionAnchor::After(call)
    }));
}

#[test]
fn into_raw_grouped_transfers_acquire_exact_extra_units_before_the_first_transfer() {
    for (count, transfers) in [
        (
            2,
            "    %first = tribute_rt.into_raw %closure : core.ptr\n    %second = tribute_rt.into_raw %closure : core.ptr",
        ),
        (
            3,
            "    %first = tribute_rt.into_raw %closure : core.ptr\n    %second = tribute_rt.into_raw %closure : core.ptr\n    %third = tribute_rt.into_raw %closure : core.ptr",
        ),
    ] {
        let (mut ctx, module, plan) = build(&into_raw_fixture(transfers));
        let function = plan.function(&Symbol::new("transfers")).unwrap();
        let transfer_actions = function
            .actions()
            .iter()
            .filter(|action| action.kind == ActionKind::IntoRawTransfer)
            .collect::<Vec<_>>();
        let source = transfer_actions[0].value;
        let ActionAnchor::Before(first) = transfer_actions[0].anchor else {
            panic!("first transfer must have a before anchor");
        };
        assert_eq!(transfer_actions.len(), count);
        assert!(transfer_actions.iter().all(|action| action.value == source));
        let copies = function
            .actions()
            .iter()
            .filter(|action| action.kind == ActionKind::CopyAcquire && action.value == source)
            .collect::<Vec<_>>();
        assert_eq!(copies.len(), count - 1);
        assert!(copies.iter().enumerate().all(|(index, action)| {
            action.anchor == ActionAnchor::Before(first) && action.destination == (index + 1) as u32
        }));
        assert!(
            !function.actions().iter().any(|action| {
                action.kind == ActionKind::FinalRelease && action.value == source
            })
        );

        materialize(&mut ctx, module, &plan).expect("typed RC materialization");
        let materialized = print_module(&ctx, module.op());
        let retain = materialized
            .find("tribute_rt.retain")
            .expect("grouped transfers require a retain");
        let transfer = materialized
            .find("tribute_rt.into_raw")
            .expect("fixture contains an into_raw transfer");
        assert!(retain < transfer, "{materialized}");
    }
}

#[test]
fn into_raw_group_validation_ignores_preserved_field_borrow_acquire() {
    let ir = r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  !Owner = adt.struct<Owner(closure: !_closure)>
  func.func @transfers() -> core.nil {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %stored = adt.struct_new %code, %env {type = !_closure} : !_closure
    %owner = adt.struct_new %stored {type = !Owner} : !Owner
    %closure = adt.struct_get %owner {field = 0, type = !Owner} : !_closure
    %first = tribute_rt.into_raw %closure : core.ptr
    %second = tribute_rt.into_raw %closure : core.ptr
    func.return
  }
}"#;
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ir);
    let plan = build_native_ownership_plan(
        &ctx,
        module,
        NativeOwnershipPlanOptions {
            elide_proven_borrowed_parameters: true,
            elide_proven_field_borrows: false,
        },
        &mut Default::default(),
    )
    .expect("typed ownership plan with preserved field borrows");
    let function = plan.function(&Symbol::new("transfers")).unwrap();
    let transfers = function
        .actions()
        .iter()
        .filter(|action| action.kind == ActionKind::IntoRawTransfer)
        .collect::<Vec<_>>();
    let source = transfers[0].value;
    assert!(function.actions().iter().any(|action| {
        action.kind == ActionKind::CopyAcquire
            && action.value == source
            && matches!(action.anchor, ActionAnchor::After(_))
    }));

    materialize(&mut ctx, module, &plan)
        .expect("field-borrow acquisition is not a grouped-transfer acquisition");
}

#[test]
fn into_raw_rejects_non_transfer_and_cross_block_uses_before_mutation() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  func.func @later_use() -> !_closure {
    %code = arith.const {value = 0} : core.i32
    %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
    %closure = adt.struct_new %code, %env {type = !_closure} : !_closure
    %raw = tribute_rt.into_raw %closure : core.ptr
    func.return %closure
  }
}"#,
        "all direct closure uses to be exact same-block transfers",
    );
    assert_plan_error_unchanged(
        r#"core.module @test {
  !_closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  func.func @cross_block() -> core.nil {
    ^entry:
      %code = arith.const {value = 0} : core.i32
      %env = adt.ref_null {type = tribute_rt.anyref} : tribute_rt.anyref
      %closure = adt.struct_new %code, %env {type = !_closure} : !_closure
      %first = tribute_rt.into_raw %closure : core.ptr
      cf.br [^next]
    ^next:
      %second = tribute_rt.into_raw %closure : core.ptr
      func.return
  }
}"#,
        "all direct closure uses to be exact same-block transfers",
    );
}

#[test]
fn stale_grouped_into_raw_plan_fails_before_materialization() {
    let (mut ctx, module, mut plan) = build(&into_raw_fixture(
        "    %first = tribute_rt.into_raw %closure : core.ptr\n    %second = tribute_rt.into_raw %closure : core.ptr",
    ));
    let actions = &mut plan
        .functions
        .iter_mut()
        .find(|function| function.symbol == "transfers")
        .unwrap()
        .actions;
    let second = actions
        .iter()
        .rposition(|action| action.kind == ActionKind::IntoRawTransfer)
        .unwrap();
    actions.remove(second);
    assert_unchanged_on_error(&mut ctx, module, |ctx, module| {
        materialize(ctx, module, &plan)
    });
}

#[test]
fn into_raw_rejects_non_closure_input_before_mutation() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  func.func @invalid(%raw: core.ptr) -> core.nil {
    %again = tribute_rt.into_raw %raw : core.ptr
    func.return
  }
}"#,
        "requires the exact managed closure layout and a core.ptr result",
    );
}

#[test]
fn materialization_releases_the_exact_replaced_field_and_fails_before_mutation() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !NodeRef = adt.typeref<{name = "Node"}>
  !Node = adt.struct<Node(next: !NodeRef)>
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
  !NodeRef = adt.typeref<{name = "Node"}>
  !Node = adt.struct<Node(next: !NodeRef)>
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
    assert_unchanged_on_error(&mut ctx, module, |ctx, module| {
        materialize(ctx, module, &stale)
    });
}

#[test]
fn duplicate_owning_destinations_and_null_are_explicit() {
    let (_ctx, _module, plan) = build(
        r#"core.module @test {
  !NodeRef = adt.typeref<{name = "Node"}>
  !Node = adt.struct<Node(next: !NodeRef, other: !NodeRef)>
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
    let duplicate = plan.function(&Symbol::new("duplicate")).unwrap();
    assert_eq!(count(duplicate, ActionKind::StoreAcquire), 2);
    assert_eq!(count(duplicate, ActionKind::ReturnTransfer), 1);
    let null = plan.function(&Symbol::new("null")).unwrap();
    assert_eq!(count(null, ActionKind::ReturnTransfer), 1);
    assert_eq!(count(null, ActionKind::EntryAcquire), 0);
    assert_eq!(count(null, ActionKind::FinalRelease), 0);
    let replace = plan.function(&Symbol::new("replace")).unwrap();
    assert_eq!(count(replace, ActionKind::StoreAcquire), 1);
    assert_eq!(count(replace, ActionKind::ReleaseReplacedField), 1);
}

#[test]
fn borrowed_load_return_acquires_a_transfer_unit() {
    let (_ctx, _module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  func.func @load(%owner: !BoxRef) -> !ChildRef {
    %child = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    func.return %child
  }
}"#,
    );
    let function = plan.function(&Symbol::new("load")).unwrap();
    assert_eq!(function.entries(), [EntryOwnership::Borrowed]);
    assert_eq!(count(function, ActionKind::BorrowLoad), 1);
    assert_eq!(count(function, ActionKind::CopyAcquire), 1);
    assert_eq!(count(function, ActionKind::ReturnTransfer), 1);
    assert_eq!(count(function, ActionKind::FinalRelease), 0);
}

/// The projection each `@read*` fixture function loads from its box.
fn box_child_projection(
    ctx: &IrContext,
    plan: &NativeOwnershipPlan,
    name: &'static str,
) -> (OpRef, ValueRef) {
    let function = plan.function(&Symbol::new(name)).unwrap();
    let body = ctx.op_region(function.operation(), 0).unwrap();
    let entry = ctx.region(body).blocks[0];
    let get = ctx.block(entry).ops[0];
    assert!(adt::StructGet::matches(ctx, get));
    (get, ctx.op_result(get, 0))
}

#[test]
fn projection_of_a_written_layout_keeps_its_own_unit() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  !Pair = adt.struct<Pair(child: !ChildRef)>
  !PairRef = adt.typeref<{name = "Pair"}>
  func.func @read_then_replace(%owner: !BoxRef, %new: !ChildRef) -> !ChildRef {
    %old = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    adt.struct_set %owner, %new {field = 0, type = !Box}
    func.return %old
  }
  func.func @read_only(%owner: !BoxRef) -> !ChildRef {
    %child = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    func.return %child
  }
  func.func @read_unwritten(%owner: !PairRef) -> !ChildRef {
    %child = adt.struct_get %owner {field = 0, type = !Pair} : !ChildRef
    func.return %child
  }
}"#,
    );
    // A writer in the same function, or anywhere else in the module, releases
    // the previous field value, so neither reader of `Box` may borrow it.
    for name in ["read_then_replace", "read_only"] {
        let function = plan.function(&Symbol::new(name)).unwrap();
        let (get, child) = box_child_projection(&ctx, &plan, name);
        assert_eq!(count(function, ActionKind::BorrowLoad), 0, "{name}");
        assert!(
            function.actions().iter().any(|action| {
                action.kind == ActionKind::CopyAcquire
                    && action.value == child
                    && action.anchor == ActionAnchor::After(get)
            }),
            "{name}"
        );
    }
    let unwritten = plan.function(&Symbol::new("read_unwritten")).unwrap();
    assert_eq!(count(unwritten, ActionKind::BorrowLoad), 1);

    let (get, old) = box_child_projection(&ctx, &plan, "read_then_replace");
    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    let block = ctx.op(get).parent_block.unwrap();
    let ops = &ctx.block(block).ops;
    let position = |found: &dyn Fn(OpRef) -> bool| ops.iter().position(|&op| found(op)).unwrap();
    let acquire = position(&|op| {
        tribute_ir::dialect::tribute_rt::Retain::matches(&ctx, op) && ctx.op_operands(op) == [old]
    });
    let replaced_release =
        position(&|op| tribute_ir::dialect::tribute_rt::Release::matches(&ctx, op));
    assert!(position(&|op| op == get) < acquire && acquire < replaced_release);
}

/// The single managed `adt.struct_get` of a fixture function: the operation,
/// its borrowed result and the owner it reads.
fn borrowed_projection(
    ctx: &IrContext,
    plan: &NativeOwnershipPlan,
    name: &'static str,
) -> (OpRef, ValueRef, ValueRef) {
    let function = plan.function(&Symbol::new(name)).unwrap();
    let body = ctx.op_region(function.operation(), 0).unwrap();
    let mut gets = ctx.region(body).blocks.iter().flat_map(|&block| {
        ctx.block(block).ops.iter().copied().filter(|&op| {
            adt::StructGet::matches(ctx, op)
                && plan.is_managed_type(ctx, ctx.op_result_types(op)[0])
        })
    });
    let get = gets.next().expect("one managed projection");
    assert!(gets.next().is_none(), "one managed projection");
    assert!(
        function.actions().iter().any(|action| {
            action.kind == ActionKind::BorrowLoad && action.anchor == ActionAnchor::After(get)
        }),
        "{name} borrows its projection"
    );
    (get, ctx.op_result(get, 0), ctx.op_operands(get)[0])
}

fn has_action(
    function: &FunctionOwnershipPlan,
    kind: ActionKind,
    value: ValueRef,
    anchor: ActionAnchor,
) -> bool {
    function
        .actions()
        .iter()
        .any(|action| action.kind == kind && action.value == value && action.anchor == anchor)
}

/// Position in `block` of the materialized retain or release of `value`.
fn rc_position(ctx: &IrContext, block: BlockRef, retain: bool, value: ValueRef) -> usize {
    use tribute_ir::dialect::tribute_rt::{Release, Retain};
    ctx.block(block)
        .ops
        .iter()
        .position(|&op| {
            let matches = if retain {
                Retain::matches(ctx, op)
            } else {
                Release::matches(ctx, op)
            };
            matches && ctx.op_operands(op) == [value]
        })
        .expect("materialized RC operation")
}

fn op_position(ctx: &IrContext, op: OpRef) -> (BlockRef, usize) {
    let block = ctx.op(op).parent_block.unwrap();
    let position = ctx.block(block).ops.iter().position(|&other| other == op);
    (block, position.unwrap())
}

#[test]
fn borrowed_projection_proper_tail_transfer_acquires_before_the_owner_dies() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  func.func @sink(%value: !ChildRef) attributes {type = func.func_sig<(!ChildRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
  func.func @sink_both(%owner: !BoxRef, %value: !ChildRef) attributes {type = func.func_sig<(!BoxRef {tribute.ownership = "consumed"}, !ChildRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
  func.func @owner_dies(%owner: !BoxRef) attributes {type = func.func_sig<(!BoxRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    %child = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    func.tail_call %child {callee = @sink}
  }
  func.func @owner_moves(%owner: !BoxRef) attributes {type = func.func_sig<(!BoxRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    %child = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    func.tail_call %owner, %child {callee = @sink_both}
  }
}"#,
    );
    let (get, child, owner) = borrowed_projection(&ctx, &plan, "owner_dies");
    let (block, _) = op_position(&ctx, get);
    let tail = *ctx.block(block).ops.last().unwrap();
    let dies = plan.function(&Symbol::new("owner_dies")).unwrap();
    assert!(has_action(
        dies,
        ActionKind::CopyAcquire,
        child,
        ActionAnchor::Before(tail)
    ));
    assert!(has_action(
        dies,
        ActionKind::TailTransfer,
        child,
        ActionAnchor::Before(tail)
    ));
    assert!(has_action(
        dies,
        ActionKind::FinalRelease,
        owner,
        ActionAnchor::Before(tail)
    ));

    let (moves_get, moves_child, moves_owner) = borrowed_projection(&ctx, &plan, "owner_moves");
    let (moves_block, _) = op_position(&ctx, moves_get);
    let moves_tail = *ctx.block(moves_block).ops.last().unwrap();
    let moves = plan.function(&Symbol::new("owner_moves")).unwrap();
    assert!(has_action(
        moves,
        ActionKind::CopyAcquire,
        moves_child,
        ActionAnchor::Before(moves_tail)
    ));
    assert!(has_action(
        moves,
        ActionKind::TailTransfer,
        moves_owner,
        ActionAnchor::Before(moves_tail)
    ));
    assert!(has_action(
        moves,
        ActionKind::TailTransfer,
        moves_child,
        ActionAnchor::Before(moves_tail)
    ));
    assert_eq!(count(moves, ActionKind::TailTransfer), 2);
    assert_eq!(count(moves, ActionKind::FinalRelease), 0);

    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    // The projection's unit exists before the owner that kept it alive is
    // released, and nothing follows the proper-tail terminator.
    assert!(rc_position(&ctx, block, true, child) < rc_position(&ctx, block, false, owner));
    assert_eq!(ctx.block(block).ops.last(), Some(&tail));
    assert!(rc_position(&ctx, moves_block, true, moves_child) < op_position(&ctx, moves_tail).1);
    assert_eq!(ctx.block(moves_block).ops.last(), Some(&moves_tail));
}

#[test]
fn borrowed_projection_ordinary_call_to_a_consumed_parameter_acquires_its_unit() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  func.func @sink(%value: !ChildRef) attributes {type = func.func_sig<(!ChildRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
  func.func @load(%child: !ChildRef) -> core.nil {
    %owner = adt.struct_new %child {type = !Box} : !BoxRef
    %loaded = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    func.call %loaded {callee = @sink}
    func.return
  }
}"#,
    );
    let (get, loaded, owner) = borrowed_projection(&ctx, &plan, "load");
    let (block, get_position) = op_position(&ctx, get);
    let call = ctx.block(block).ops[get_position + 1];
    assert!(func::Call::matches(&ctx, call));
    let function = plan.function(&Symbol::new("load")).unwrap();
    assert!(has_action(
        function,
        ActionKind::CallAcquire,
        loaded,
        ActionAnchor::Before(call)
    ));
    assert!(has_action(
        function,
        ActionKind::FinalRelease,
        owner,
        ActionAnchor::After(call)
    ));
    assert!(
        !function
            .actions()
            .iter()
            .any(|action| action.kind == ActionKind::FinalRelease && action.value == loaded)
    );

    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    // The callee consumes the acquired unit; the caller's owner outlives the call.
    let call_position = op_position(&ctx, call).1;
    assert!(rc_position(&ctx, block, true, loaded) < call_position);
    assert!(call_position < rc_position(&ctx, block, false, owner));
}

#[test]
fn borrowed_projection_stored_in_an_aggregate_acquires_the_field_unit() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  !Capture = adt.struct<Capture(child: !ChildRef)>
  !CaptureRef = adt.typeref<{name = "Capture"}>
  func.func @capture(%child: !ChildRef) -> !CaptureRef {
    %owner = adt.struct_new %child {type = !Box} : !BoxRef
    %loaded = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    %captured = adt.struct_new %loaded {type = !Capture} : !CaptureRef
    func.return %captured
  }
}"#,
    );
    let (get, loaded, owner) = borrowed_projection(&ctx, &plan, "capture");
    let (block, get_position) = op_position(&ctx, get);
    let store = ctx.block(block).ops[get_position + 1];
    assert!(adt::StructNew::matches(&ctx, store));
    let function = plan.function(&Symbol::new("capture")).unwrap();
    assert!(has_action(
        function,
        ActionKind::StoreAcquire,
        loaded,
        ActionAnchor::Before(store)
    ));
    assert!(has_action(
        function,
        ActionKind::FinalRelease,
        owner,
        ActionAnchor::After(store)
    ));

    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    // The stored field owns its unit before the owner it was read from dies.
    let store_position = op_position(&ctx, store).1;
    assert!(rc_position(&ctx, block, true, loaded) < store_position);
    assert!(store_position < rc_position(&ctx, block, false, owner));
}

#[test]
fn loop_carried_borrowed_projection_acquires_before_the_previous_owner_dies() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !NodeRef = adt.typeref<{name = "Node"}>
  !Node = adt.struct<Node(next: !NodeRef)>
  func.func @walk(%condition: core.i1, %tail: !NodeRef) -> core.nil {
    ^entry:
      %head = adt.struct_new %tail {type = !Node} : !NodeRef
      cf.br %head [^loop]
    ^loop(%current: !NodeRef):
      %next = adt.struct_get %current {field = 0, type = !Node} : !NodeRef
      cf.cond_br %condition [^latch, ^exit]
    ^latch:
      cf.br %next [^loop]
    ^exit:
      func.return
  }
}"#,
    );
    let (_, next, current) = borrowed_projection(&ctx, &plan, "walk");
    let function = plan.function(&Symbol::new("walk")).unwrap();
    let body = ctx.op_region(function.operation(), 0).unwrap();
    let latch = ctx.region(body).blocks[2];
    let back_edge = *ctx.block(latch).ops.last().unwrap();
    assert!(has_action(
        function,
        ActionKind::CopyAcquire,
        next,
        ActionAnchor::Before(back_edge)
    ));
    assert!(has_action(
        function,
        ActionKind::FinalRelease,
        current,
        ActionAnchor::Before(back_edge)
    ));
    // Leaving the loop drops the iteration's owner on the exit edge.
    let exit = ctx.region(body).blocks[3];
    assert!(has_action(
        function,
        ActionKind::FinalRelease,
        current,
        ActionAnchor::BlockStart(exit)
    ));

    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    // The next iteration's owner holds its unit before this iteration's dies.
    assert!(rc_position(&ctx, latch, true, next) < rc_position(&ctx, latch, false, current));
}

#[test]
fn enum_release_leaves_the_size_to_the_descriptor() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Choice = adt.enum<Choice { None(), Some(core.i64, core.i64) }>
  !ChoiceRef = adt.typeref<{name = "Choice"}>
  func.func @drop() -> core.nil {
    %choice = adt.variant_new {tag = "None", type = !Choice} : !ChoiceRef
    func.return
  }
}"#,
    );
    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    let materialized = print_module(&ctx, module.op());

    // Each variant has its own allocation size, so the release of an enum
    // value carries the dynamic-size signal instead of one static size.
    assert!(
        materialized.contains("tribute_rt.release %0 {alloc_size = 0}"),
        "{materialized}"
    );
}

#[test]
fn compatible_cast_and_enum_projection_preserve_borrowed_ownership() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Choice = adt.enum<Choice { Some(!ChildRef) }>
  !ChoiceRef = adt.typeref<{name = "Choice"}>
  func.func @load(%choice: !ChoiceRef) -> !ChildRef {
    %erased = adt.ref_cast %choice {type = tribute_rt.anyref} : tribute_rt.anyref
    %restored = adt.ref_cast %erased {type = !ChoiceRef} : !ChoiceRef
    %child = adt.variant_get %restored {type = !Choice, tag = "Some", field = 0} : !ChildRef
    func.return %child
  }
}"#,
    );
    let function = plan.function(&Symbol::new("load")).unwrap();
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
    let missing_tag = ctx.string_attr("Missing");
    for (key, invalid) in [
        (Symbol::new("tag"), missing_tag),
        (Symbol::new("field"), trunk_ir::Attribute::Int(1)),
    ] {
        let original = ctx
            .op(projection)
            .attributes
            .get(key.clone())
            .expect("projection attribute")
            .clone();
        ctx.op_mut(projection)
            .attributes
            .insert(key.clone(), invalid);
        assert_unchanged_on_error(&mut ctx, module, |ctx, module| production_plan(ctx, module));
        ctx.op_mut(projection).attributes.insert(key, original);
    }
}

#[test]
fn malformed_projection_arity_fails_before_mutation() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  func.func @load(%owner: !BoxRef) -> !ChildRef {
    %child, %extra = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef, !ChildRef
    func.return %child
  }
}"#,
        "ADT projection must have exactly one result",
    );
}

#[test]
fn mismatched_projection_managed_types_fail_before_mutation() {
    for (source_ty, result_ty) in [("!BoxRef", "!BoxRef"), ("!ChildRef", "!ChildRef")] {
        assert_plan_error_unchanged(
            &format!(
                r#"core.module @test {{
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{{name = "Child"}}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{{name = "Box"}}>
  func.func @load(%owner: {source_ty}) -> {result_ty} {{
    %child = adt.struct_get %owner {{field = 0, type = !Box}} : {result_ty}
    func.return %child
  }}
}}"#
            ),
            "ADT projection managed type contract is malformed",
        );
    }
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
fn branch_interfaces_fail_closed_before_mutation() {
    for (ir, expected) in [
        (
            r#"core.module @test {
  func.func @f() -> core.nil {
    ^entry:
      test.incomplete_branch [^exit]
    ^exit:
      func.return
  }
}"#,
            "Branch interface is incomplete",
        ),
        (
            r#"core.module @test {
  func.func @f() -> core.nil {
    ^entry:
      test.unregistered_branch [^exit]
    ^exit:
      func.return
  }
}"#,
            "unsupported native CFG terminator",
        ),
        (
            r#"core.module @test {
  func.func @f() -> core.nil {
    ^entry:
      test.missing_edge_branch [^exit]
    ^exit:
      func.return
  }
}"#,
            "Branch successors leave the function or are incomplete",
        ),
        (
            r#"core.module @test {
  func.func @f() -> core.nil {
    ^entry:
      test.reversed_branch [^left, ^right]
    ^left:
      func.return
    ^right:
      func.return
  }
}"#,
            "Branch successors leave the function or are incomplete",
        ),
        (
            r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R) -> core.nil {
    ^entry:
      test.multi_forwarding_branch %value [^left, ^right]
    ^left(%left: !R):
      func.return
    ^right(%right: !R):
      func.return
  }
}"#,
            "multi-successor Branch forwarding is unsupported",
        ),
    ] {
        assert_plan_error_unchanged(ir, expected);
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
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
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
    let function = plan.function(&Symbol::new("load")).unwrap();
    let body = ctx.op_region(function.operation(), 0).unwrap();
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
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @branch(%value: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    ^entry:
      cf.br %value, %value [^merge]
    ^merge(%left: !R, %right: !R):
      func.unreachable
  }
  func.func @tail(%sent: !R, %dying: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}, !R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.tail_call %sent {callee = @sink}
  }
  func.func @sink(%value: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
}"#,
    );
    let branch = plan.function(&Symbol::new("branch")).unwrap();
    assert_eq!(count(branch, ActionKind::CopyAcquire), 1);
    assert_eq!(count(branch, ActionKind::FinalRelease), 2);
    let tail = plan.function(&Symbol::new("tail")).unwrap();
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
        .position(|function| function.symbol == "tail")
        .unwrap();
    let body = ctx
        .op_region(plan.functions[tail_index].operation, 0)
        .unwrap();
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

    materialize(&mut ctx, module, &plan).expect("branch ownership plan materializes");
    let materialized = print_module(&ctx, module.op());
    assert_eq!(materialized.matches("tribute_rt.retain").count(), 1);
    assert_eq!(materialized.matches("tribute_rt.release").count(), 4);
}

#[test]
fn branch_transfer_of_a_value_live_afterwards_acquires_the_destination_unit() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  func.func @sink(%value: !BoxRef) attributes {type = func.func_sig<(!BoxRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
  func.func @observe(%child: !ChildRef) -> core.i32 {
    %value = adt.struct_get %child {field = 0, type = !Child} : core.i32
    func.return %value
  }
  func.func @used_after(%child: !ChildRef) -> core.nil {
    ^entry:
      %value = adt.struct_new %child {type = !Box} : !BoxRef
      cf.br %value [^next]
    ^next(%moved: !BoxRef):
      func.call %value {callee = @sink}
      func.return
  }
  func.func @projection_used_after(%child: !ChildRef) -> core.nil {
    ^entry:
      %owner = adt.struct_new %child {type = !Box} : !BoxRef
      %loaded = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
      cf.br %owner [^next]
    ^next(%moved: !BoxRef):
      %seen = func.call %loaded {callee = @observe} : core.i32
      func.return
  }
  func.func @moved(%child: !ChildRef) -> core.nil {
    ^entry:
      %value = adt.struct_new %child {type = !Box} : !BoxRef
      cf.br %value [^next]
    ^next(%moved: !BoxRef):
      func.return
  }
}"#,
    );
    let branch_source = |name: &'static str| {
        let function = plan.function(&Symbol::new(name)).unwrap();
        let body = ctx.op_region(function.operation(), 0).unwrap();
        let entry = ctx.region(body).blocks[0];
        let branch = *ctx.block(entry).ops.last().unwrap();
        (function, branch, ctx.op_operands(branch)[0])
    };
    // The source keeps its unit for the later use, or for the projection
    // borrowed from it, so the block argument gets a unit of its own.
    for name in ["used_after", "projection_used_after"] {
        let (function, branch, source) = branch_source(name);
        assert!(
            function.actions().iter().any(|action| {
                action.kind == ActionKind::CopyAcquire
                    && action.value == source
                    && action.anchor == ActionAnchor::Before(branch)
            }),
            "{name}"
        );
        assert_eq!(
            function
                .actions()
                .iter()
                .filter(|action| action.kind == ActionKind::FinalRelease && action.value == source)
                .count(),
            1,
            "{name}"
        );
    }
    // A source that dies at the branch moves its only unit.
    let (moved, _, source) = branch_source("moved");
    assert_eq!(count(moved, ActionKind::CopyAcquire), 0);
    assert!(
        !moved
            .actions()
            .iter()
            .any(|action| action.kind == ActionKind::FinalRelease && action.value == source)
    );

    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
}

/// The blocks of a fixture function, in layout order.
fn function_blocks(
    ctx: &IrContext,
    plan: &NativeOwnershipPlan,
    name: &'static str,
) -> trunk_ir::BlockList {
    let function = plan.function(&Symbol::new(name)).unwrap();
    let body = ctx.op_region(function.operation(), 0).unwrap();
    ctx.region(body).blocks.clone()
}

fn final_releases(function: &FunctionOwnershipPlan, value: ValueRef) -> Vec<ActionAnchor> {
    function
        .actions()
        .iter()
        .filter(|action| action.kind == ActionKind::FinalRelease && action.value == value)
        .map(|action| action.anchor)
        .collect()
}

const EDGE_DEATHS: &str = r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  func.func @sink(%value: !BoxRef) attributes {type = func.func_sig<(!BoxRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
  func.func @observe(%child: !ChildRef) -> core.i32 {
    %value = adt.struct_get %child {field = 0, type = !Child} : core.i32
    func.return %value
  }
  func.func @one_arm(%condition: core.i1, %child: !ChildRef) -> core.nil {
    ^entry:
      %value = adt.struct_new %child {type = !Box} : !BoxRef
      cf.cond_br %condition [^use, ^skip]
    ^use:
      func.call %value {callee = @sink}
      cf.br [^join]
    ^skip:
      cf.br [^join]
    ^join:
      func.return
  }
  func.func @nested(%outer: core.i1, %inner: core.i1, %child: !ChildRef) -> core.nil {
    ^entry:
      %value = adt.struct_new %child {type = !Box} : !BoxRef
      cf.cond_br %outer [^inside, ^outer_skip]
    ^inside:
      cf.cond_br %inner [^use, ^inner_skip]
    ^use:
      func.call %value {callee = @sink}
      cf.br [^join]
    ^inner_skip:
      cf.br [^join]
    ^outer_skip:
      cf.br [^join]
    ^join:
      func.return
  }
  func.func @loop_exit(%condition: core.i1, %child: !ChildRef) -> core.nil {
    ^entry:
      %value = adt.struct_new %child {type = !Box} : !BoxRef
      cf.br [^loop]
    ^loop:
      func.call %value {callee = @sink}
      cf.cond_br %condition [^loop, ^exit]
    ^exit:
      func.return
  }
  func.func @projection_one_arm(%condition: core.i1, %child: !ChildRef) -> core.nil {
    ^entry:
      %owner = adt.struct_new %child {type = !Box} : !BoxRef
      %loaded = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
      cf.cond_br %condition [^use, ^skip]
    ^use:
      %seen = func.call %loaded {callee = @observe} : core.i32
      cf.br [^join]
    ^skip:
      cf.br [^join]
    ^join:
      func.return
  }
}"#;

#[test]
fn value_dying_on_an_edge_is_released_at_the_successor_start() {
    let (mut ctx, module, plan) = build(EDGE_DEATHS);
    let entry_value = |name: &'static str| {
        let entry = function_blocks(&ctx, &plan, name)[0];
        ctx.op_result(ctx.block(entry).ops[0], 0)
    };

    // One arm consumes the value; the other releases it on entry.
    let blocks = function_blocks(&ctx, &plan, "one_arm");
    let [_, used, skip, join] = blocks.as_slice() else {
        panic!("four-block fixture")
    };
    let value = entry_value("one_arm");
    let mut released_on_entry = vec![(*skip, value)];
    let call = ctx.block(*used).ops[0];
    let one_arm = plan.function(&Symbol::new("one_arm")).unwrap();
    assert_eq!(
        final_releases(one_arm, value),
        [ActionAnchor::After(call), ActionAnchor::BlockStart(*skip)]
    );
    assert!(
        !one_arm
            .actions()
            .iter()
            .any(|action| action.anchor == ActionAnchor::BlockStart(*join))
    );

    // Each dead arm releases at its own start, at the depth where it diverges.
    let blocks = function_blocks(&ctx, &plan, "nested");
    let [_, _, used, inner_skip, outer_skip, _] = blocks.as_slice() else {
        panic!("six-block fixture")
    };
    let value = entry_value("nested");
    released_on_entry.extend([(*inner_skip, value), (*outer_skip, value)]);
    let call = ctx.block(*used).ops[0];
    let nested = plan.function(&Symbol::new("nested")).unwrap();
    assert_eq!(
        final_releases(nested, value),
        [
            ActionAnchor::After(call),
            ActionAnchor::BlockStart(*inner_skip),
            ActionAnchor::BlockStart(*outer_skip)
        ]
    );

    // A value kept live around a loop is released where the loop is left.
    let blocks = function_blocks(&ctx, &plan, "loop_exit");
    let [_, _, exit] = blocks.as_slice() else {
        panic!("three-block fixture")
    };
    let value = entry_value("loop_exit");
    released_on_entry.push((*exit, value));
    let loop_exit = plan.function(&Symbol::new("loop_exit")).unwrap();
    assert_eq!(
        final_releases(loop_exit, value),
        [ActionAnchor::BlockStart(*exit)]
    );

    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    for (block, value) in released_on_entry {
        assert_eq!(rc_position(&ctx, block, false, value), 0);
    }
}

#[test]
fn borrowed_projection_owner_dying_on_an_edge_follows_the_selected_liveness_view() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, EDGE_DEATHS);
    let preserved = build_native_ownership_plan(
        &ctx,
        module,
        NativeOwnershipPlanOptions {
            elide_proven_borrowed_parameters: false,
            elide_proven_field_borrows: false,
        },
        &mut Default::default(),
    )
    .expect("preserved typed plan");
    let elided = production_plan(&ctx, module).expect("elided typed plan");

    let blocks = function_blocks(&ctx, &elided, "projection_one_arm");
    let [entry, used, skip, _] = blocks.as_slice() else {
        panic!("four-block fixture")
    };
    let owner = ctx.op_result(ctx.block(*entry).ops[0], 0);
    let get = ctx.block(*entry).ops[1];
    let loaded = ctx.op_result(get, 0);
    let call = ctx.block(*used).ops[0];

    // Borrowing keeps the owner live into the arm that reads the projection,
    // so the owner dies after that read or on the edge into the other arm.
    let elided = elided.function(&Symbol::new("projection_one_arm")).unwrap();
    assert_eq!(
        final_releases(elided, owner),
        [ActionAnchor::After(call), ActionAnchor::BlockStart(*skip)]
    );
    assert!(final_releases(elided, loaded).is_empty());

    // Without the borrow the projection owns a unit, and that unit is the one
    // that dies on the edge.
    let preserved = preserved
        .function(&Symbol::new("projection_one_arm"))
        .unwrap();
    assert_eq!(final_releases(preserved, owner), [ActionAnchor::After(get)]);
    assert_eq!(
        final_releases(preserved, loaded),
        [ActionAnchor::After(call), ActionAnchor::BlockStart(*skip)]
    );
}

#[test]
fn value_dying_on_an_edge_into_a_shared_successor_is_rejected() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  func.func @sink(%value: !BoxRef) attributes {type = func.func_sig<(!BoxRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
  func.func @shared(%outer: core.i1, %inner: core.i1, %child: !ChildRef) -> core.nil {
    ^entry:
      %value = adt.struct_new %child {type = !Box} : !BoxRef
      cf.cond_br %outer [^inside, ^other]
    ^inside:
      cf.cond_br %inner [^join, ^use]
    ^use:
      func.call %value {callee = @sink}
      cf.br [^join]
    ^other:
      cf.br [^join]
    ^join:
      func.return
  }
}"#,
        "dies on a control-flow edge whose successor has another predecessor",
    );
}

#[test]
fn value_dying_on_a_back_edge_to_the_entry_block_is_rejected() {
    // The function entry reaches `^entry` without the value, so a release at
    // its start would run before the first definition.
    assert_plan_error_unchanged(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Box = adt.struct<Box(child: !ChildRef)>
  !BoxRef = adt.typeref<{name = "Box"}>
  func.func @sink(%value: !BoxRef) attributes {type = func.func_sig<(!BoxRef {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
  func.func @restart() -> core.nil {
    ^entry:
      %child = adt.ref_null {type = !ChildRef} : !ChildRef
      %value = adt.struct_new %child {type = !Box} : !BoxRef
      %again = adt.ref_is_null %child : core.i1
      cf.cond_br %again [^back, ^use]
    ^back:
      cf.cond_br %again [^entry, ^use]
    ^use:
      func.call %value {callee = @sink}
      func.return
  }
}"#,
        "dies on a control-flow edge whose successor has another predecessor",
    );
}

#[test]
fn cfg_accepts_conditional_branch_with_duplicate_successors() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @duplicate_successor(%condition: core.i1, %value: !R) attributes {type = func.func_sig<(core.i1 {tribute.ownership = "consumed"}, !R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    ^entry:
      cf.cond_br %condition [^exit, ^exit]
    ^exit:
      func.unreachable
  }
}"#,
    );
    let function = plan.function(&Symbol::new("duplicate_successor")).unwrap();
    assert_eq!(count(function, ActionKind::EntryAcquire), 0);
    assert_eq!(count(function, ActionKind::FinalRelease), 1);

    materialize(&mut ctx, module, &plan).expect("duplicate-successor plan materializes");
    assert_eq!(
        print_module(&ctx, module.op())
            .matches("cf.cond_br")
            .count(),
        1
    );
}

#[test]
fn enum_rtti_uses_the_same_nested_managed_predicate() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Choice = adt.enum<Choice { None(), Some(!ChildRef, core.ptr), Bytes(core.bytes) }>
  func.func @some(%child: !ChildRef, %raw: core.ptr) -> !Choice {
    %choice = adt.variant_new %child, %raw {tag = "Some", type = !Choice} : !Choice
    func.return %choice
  }
}"#,
    );
    let [entry] = plan.rtti_types() else {
        panic!("one RTTI descriptor per allocated variant")
    };
    assert_eq!(adt::nominal_name(&ctx, entry.ty), Some("Choice"));
    assert_eq!(entry.tag.map(|tag| ctx.str(tag)), Some("Some"));
    assert_eq!(entry.fields, [FieldKind::Managed, FieldKind::Raw]);
}

#[test]
fn unmanaged_physical_and_buffer_types_never_receive_actions() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Marker = adt.struct<EvidenceMarker(code: core.ptr)>
  !Evidence = core.array<!Marker>
  func.func @raw(%raw: core.ptr, %array: core.array<core.i32>, %evidence: !Evidence, %code: func.func_sig<() -> core.nil>) -> core.ptr {
    func.return %raw
  }
}"#,
    );
    let function = plan.function(&Symbol::new("raw")).unwrap();
    assert_eq!(function.entries(), [EntryOwnership::Plain; 4]);
    assert!(function.actions().is_empty());
    for ty in ctx
        .op(function.operation())
        .attributes
        .get_type("type")
        .and_then(|ty| func::FuncSig::from_type_ref(&ctx, ty))
        .unwrap()
        .inputs(&ctx)
    {
        assert!(!plan.is_managed_type(&ctx, *ty));
    }
}

#[test]
fn bytes_is_a_managed_reference_with_its_own_units() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Leaf = adt.struct<Leaf(bytes: core.bytes)>
  !LeafRef = adt.typeref<{name = "Leaf"}>
  func.func @wrap(%raw: core.ptr) -> !LeafRef {
    %bytes = tribute_rt.from_raw %raw : core.bytes
    %leaf = adt.struct_new %bytes {type = !Leaf} : !LeafRef
    func.return %leaf
  }
}"#,
    );

    // `from_raw` yields a fresh owned value: the field takes its own unit and
    // the local one is released.
    let wrap = plan.function(&Symbol::new("wrap")).unwrap();
    let [from_raw, new, _] = ctx.block(function_blocks(&ctx, &plan, "wrap")[0]).ops[..] else {
        panic!("three-operation fixture")
    };
    let bytes = ctx.op_result(from_raw, 0);
    assert!(plan.is_managed_type(&ctx, ctx.value_ty(bytes)));
    assert!(has_action(
        wrap,
        ActionKind::StoreAcquire,
        bytes,
        ActionAnchor::Before(new)
    ));
    assert_eq!(final_releases(wrap, bytes), [ActionAnchor::After(new)]);
    let leaf = plan
        .rtti_types()
        .iter()
        .find(|entry| entry.ty == ctx.type_alias_by_text("Leaf").unwrap())
        .unwrap();
    assert_eq!(leaf.fields, [FieldKind::Managed]);
}

#[test]
fn only_explicit_transfers_cross_between_raw_pointers_and_managed_references() {
    // A cast cannot say whether a unit moves, in either direction.
    assert_plan_error_unchanged(
        r#"core.module @test {
  func.func @receive(%raw: core.ptr) -> core.bytes {
    %bytes = core.unrealized_conversion_cast %raw : core.bytes
    func.return %bytes
  }
}"#,
        "masquerades as a managed reference; use tribute_rt.from_raw",
    );
    assert_plan_error_unchanged(
        r#"core.module @test {
  func.func @view(%bytes: core.bytes) -> core.ptr {
    %raw = core.unrealized_conversion_cast %bytes : core.ptr
    func.return %raw
  }
}"#,
        "is viewed as a raw pointer; use tribute_rt.into_raw",
    );
    // `from_raw` itself takes a raw pointer and produces a managed reference.
    assert_plan_error_unchanged(
        r#"core.module @test {
  func.func @retype(%bytes: core.bytes) -> core.bytes {
    %again = tribute_rt.from_raw %bytes : core.bytes
    func.return %again
  }
}"#,
        "tribute_rt.from_raw must take a core.ptr and produce a managed reference",
    );
    assert_plan_error_unchanged(
        r#"core.module @test {
  func.func @copy(%raw: core.ptr) -> core.ptr {
    %same = tribute_rt.from_raw %raw : core.ptr
    func.return %same
  }
}"#,
        "tribute_rt.from_raw must take a core.ptr and produce a managed reference",
    );
}

#[test]
fn evidence_is_a_managed_reference_the_runtime_returns_owned() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Evidence = core.array<adt.struct<_Marker(ability_id: core.i32, prompt_tag: core.i32, tr_dispatch_fn: core.ptr, shadowed: core.ptr, outer: core.ptr), {layout = "evidence_marker"}>, {layout = "evidence"}>
  func.func @__tribute_evidence_mask(%ev: !Evidence, %id: core.i32) -> !Evidence attributes {abi = "C"}
  func.func @__tribute_evidence_lookup(%ev: !Evidence, %id: core.i32) -> core.i32 attributes {abi = "C"}
  func.func @derive(%ev: !Evidence, %id: core.i32) -> core.i32 {
    %masked = func.call %ev, %id {callee = @__tribute_evidence_mask} : !Evidence
    %tag = func.call %masked, %id {callee = @__tribute_evidence_lookup} : core.i32
    func.return %tag
  }
}"#,
    );
    let derive = plan.function(&Symbol::new("derive")).unwrap();
    let [mask, lookup, _] = ctx.block(function_blocks(&ctx, &plan, "derive")[0]).ops[..] else {
        panic!("three-operation fixture")
    };
    // The runtime borrows its evidence argument and returns a unit the caller
    // owns, which is released after the last use.
    let masked = ctx.op_result(mask, 0);
    assert!(plan.is_managed_type(&ctx, ctx.value_ty(masked)));
    assert_eq!(count(derive, ActionKind::CallAcquire), 0);
    assert_eq!(
        final_releases(derive, masked),
        [ActionAnchor::After(lookup)]
    );
}

#[test]
fn closure_release_uses_its_compiler_generated_allocation_layout() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Closure = closure.closure<func.func_sig<(tribute_rt.anyref) -> core.nil>>
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
    crate::closure_lower::lower_prepared_closures(&mut ctx, module).unwrap();
    crate::closure_lower::finalize_closure_storage_layout(&mut ctx, module);

    let plan = production_plan(&ctx, module).expect("typed ownership plan");
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
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @ordinary(%value: !R) -> !R {
    func.return %value
  }
  func.func @observe(%value: !R) -> core.i32 {
    %seen = adt.struct_get %value {field = 0, type = !Layout} : core.i32
    func.return %seen
  }
  func.func @caller(%value: !R, %callee: func.func_sig<(!R) -> !R>) -> !R {
    %seen = func.call %value {callee = @observe} : core.i32
    %direct = func.call %value {callee = @ordinary} : !R
    %indirect = func.call_indirect %callee, %direct {signature = func.func_sig<(!R) -> !R>} : !R
    func.return %indirect
  }
  func.func @tail(%value: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.tail_call %value {callee = @sink}
  }
  func.func @sink(%value: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
}"#,
    );
    let caller = plan.function(&Symbol::new("caller")).unwrap();
    assert_eq!(count(caller, ActionKind::CallBorrow), 1);
    assert_eq!(count(caller, ActionKind::CallRetain), 2);
    assert_eq!(count(caller, ActionKind::ReturnTransfer), 1);
    let tail = plan.function(&Symbol::new("tail")).unwrap();
    assert_eq!(count(tail, ActionKind::TailTransfer), 1);
    assert!(
        !tail
            .actions()
            .iter()
            .any(|action| matches!(action.anchor, ActionAnchor::After(_)))
    );
}

#[test]
fn retained_parameter_calls_balance_retains_and_releases() {
    // The callee of a retained parameter acquires its own unit at entry, so
    // neither a direct nor an indirect caller retains for the call.
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @keep(%value: !R) -> !R {
    func.return %value
  }
  func.func @direct(%value: !R) -> core.i32 {
    %kept = func.call %value {callee = @keep} : !R
    %seen = adt.struct_get %kept {field = 0, type = !Layout} : core.i32
    func.return %seen
  }
  func.func @indirect(%value: !R, %callee: func.func_sig<(!R) -> !R>) -> core.i32 {
    %kept = func.call_indirect %callee, %value {signature = func.func_sig<(!R) -> !R>} : !R
    %seen = adt.struct_get %kept {field = 0, type = !Layout} : core.i32
    func.return %seen
  }
}"#,
    );
    assert_eq!(
        plan.function(&Symbol::new("keep")).unwrap().entries(),
        [EntryOwnership::Retained]
    );
    for caller in ["direct", "indirect"] {
        assert_eq!(
            count(
                plan.function(&Symbol::new(caller)).unwrap(),
                ActionKind::CallRetain
            ),
            1
        );
    }
    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
    let materialized = print_module(&ctx, module.op());
    // Each function retains only its own entry parameter. Each caller
    // releases that unit and the one the call returns.
    assert_eq!(
        materialized.matches("tribute_rt.retain").count(),
        3,
        "{materialized}"
    );
    assert_eq!(
        materialized.matches("tribute_rt.release").count(),
        4,
        "{materialized}"
    );
}

#[test]
fn signature_consumed_contracts_drive_entries_and_call_sites() {
    let (mut ctx, module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  !Consuming = func.func_sig<(!R {tribute.ownership = "consumed"}, core.i32 {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>
  func.func @sink(%value: !R, %count: core.i32) attributes {type = !Consuming} {
    func.unreachable
  }
  func.func @direct(%value: !R) -> core.i32 {
    func.call %value, %value {callee = @pair}
    %seen = adt.struct_get %value {field = 0, type = !Layout} : core.i32
    func.return %seen
  }
  func.func @pair(%left: !R, %right: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}, !R {tribute.ownership = "consumed"}) -> ()>} {
    func.return
  }
  func.func @indirect(%value: !R, %callee: !Consuming, %count: core.i32) -> core.i32 {
    func.call_indirect %callee, %value, %count {signature = !Consuming}
    %seen = adt.struct_get %value {field = 0, type = !Layout} : core.i32
    func.return %seen
  }
  func.func @tail(%value: !R, %callee: !Consuming, %count: core.i32) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}, !Consuming {tribute.ownership = "consumed"}, core.i32 {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.tail_call_indirect %callee, %value, %count {signature = !Consuming}
  }
  func.func @duplicate(%value: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.tail_call %value, %value {callee = @pair_tail}
  }
  func.func @pair_tail(%left: !R, %right: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}, !R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
}"#,
    );
    // A consumed managed input is Consumed; the marker on an unmanaged input
    // is inert.
    assert_eq!(
        plan.function(&Symbol::new("sink")).unwrap().entries(),
        [EntryOwnership::Consumed, EntryOwnership::Plain]
    );
    // An ordinary call acquires one unit per consumed destination, direct or
    // indirect, and leaves the caller's own unit live.
    assert_eq!(
        count(
            plan.function(&Symbol::new("direct")).unwrap(),
            ActionKind::CallAcquire
        ),
        2
    );
    assert_eq!(
        count(
            plan.function(&Symbol::new("indirect")).unwrap(),
            ActionKind::CallAcquire
        ),
        1
    );
    // A tail edge transfers its unit; supplying one value to two consumed
    // parameters transfers one unit and acquires the other.
    assert_eq!(
        count(
            plan.function(&Symbol::new("tail")).unwrap(),
            ActionKind::TailTransfer
        ),
        1
    );
    let duplicate = plan.function(&Symbol::new("duplicate")).unwrap();
    assert_eq!(count(duplicate, ActionKind::TailTransfer), 2);
    assert_eq!(count(duplicate, ActionKind::CopyAcquire), 1);
    materialize(&mut ctx, module, &plan).expect("typed RC materialization");
}

#[test]
fn proper_tail_edges_without_a_consumed_contract_are_rejected() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @tail(%value: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.tail_call %value {callee = @sink}
  }
  func.func @sink(%value: !R) attributes {type = func.func_sig<(!R) -> (), {call_conv = "tail"}>} {
    func.unreachable
  }
}"#,
        "proper-tail managed parameter is not consumed",
    );
    assert_plan_error_unchanged(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  !Unmarked = func.func_sig<(!R) -> (), {call_conv = "tail"}>
  func.func @tail(%value: !R, %callee: !Unmarked) attributes {type = func.func_sig<(!R {tribute.ownership = "consumed"}, !Unmarked {tribute.ownership = "consumed"}) -> (), {call_conv = "tail"}>} {
    func.tail_call_indirect %callee, %value {signature = !Unmarked}
  }
}"#,
        "proper-tail managed parameter is not consumed",
    );
}

#[test]
fn unknown_parameter_ownership_contracts_are_rejected() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @run(%value: !R) attributes {type = func.func_sig<(!R {tribute.ownership = "borrowed"}) -> ()>} {
    func.return
  }
}"#,
        "unknown tribute.ownership parameter contract",
    );
}

#[test]
fn bodyless_c_ffi_borrows_managed_arguments_and_transfers_managed_results() {
    let (_ctx, _module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(value: core.i32)>
  func.func @foreign(%value: !R) -> !R attributes {abi = "C"}
  func.func @caller(%value: !R) -> !R {
    %result = func.call %value {callee = @foreign} : !R
    func.return %result
  }
}"#,
    );
    let caller = plan.function(&Symbol::new("caller")).unwrap();
    assert_eq!(count(caller, ActionKind::CallBorrow), 1);
    assert_eq!(count(caller, ActionKind::CallRetain), 0);
    assert_eq!(count(caller, ActionKind::ReturnTransfer), 1);
}

#[test]
fn stale_identity_unsupported_regions_and_malformed_calls_fail_unchanged() {
    for ir in [
        r#"core.module @test {
  !R = adt.typeref<{name = "Missing"}>
  func.func @f(%value: !R) -> !R { func.return %value }
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R) -> !R {
    %x = scf.if %value : !R { func.return %value }
    func.return %x
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R, %callee: func.func_sig<(!R) -> !R>) -> !R {
    %x = func.call_indirect %callee {signature = func.func_sig<(!R) -> !R>} : !R
    func.return %x
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !First = adt.struct<R(x: core.i32)>
  !Second = adt.struct<R(x: core.i64)>
  func.func @f(%value: !R) -> !R { func.return %value }
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @foreign(%value: !R) -> !R
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%raw: core.ptr) -> !R {
    %value = adt.ref_cast %raw {type = !R} : !R
    func.return %value
  }
}"#,
        r#"core.module @test {
  !A = adt.typeref<{name = "A"}>
  !ALayout = adt.struct<A(x: core.i32)>
  !B = adt.typeref<{name = "B"}>
  !BLayout = adt.struct<B(x: core.i32)>
  func.func @f(%value: !A) -> !B {
    %wrong = adt.ref_cast %value {type = !B} : !B
    func.return %wrong
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R) -> !R {
    %result = func.call %value {callee = @missing} : !R
    func.return %result
  }
}"#,
        r#"core.module @test {
  !ARef = adt.typeref<{name = "A"}>
  !A = adt.struct<A(x: core.i32)>
  !BRef = adt.typeref<{name = "B"}>
  !B = adt.struct<B(x: core.i32)>
  !Env = adt.struct<Env(value: !ARef)>
  func.func @f(%value: !BRef) -> core.nil {
    %env = adt.struct_new %value {type = !Env} : !Env
    func.return
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R, %callee: func.func_sig<(!R) -> !R>) -> !R {
    %result = func.call_indirect %callee, %value : !R
    func.return %result
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R) -> !R {
    ^entry:
      cf.br [^next]
    ^next(%next: !R):
      func.return %next
  }
}"#,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R) -> !R {
    func.return
  }
}"#,
    ] {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        assert_unchanged_on_error(&mut ctx, module, |ctx, module| production_plan(ctx, module));
    }
}

#[test]
fn unused_frame_alias_with_missing_nominal_result_is_not_a_live_ownership_root() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Missing = adt.typeref<{name = "Missing"}>
  !DeadFrame = adt.struct<ContinuationFrame(result: !Missing)>
  func.func @live() -> core.nil { func.return }
}"#,
    );

    let plan = production_plan(&ctx, module)
        .expect("unused continuation-frame aliases must not affect ownership planning");
    let dead_frame = ctx
        .type_alias_by_text("DeadFrame")
        .expect("parsed dead frame alias");
    assert!(!plan.is_managed_type(&ctx, dead_frame));
    assert!(plan.rtti_types().is_empty());
}

#[test]
fn reachable_missing_nominal_typeref_still_fails_before_planning() {
    assert_plan_error_unchanged(
        r#"core.module @test {
  !Missing = adt.typeref<{name = "Missing"}>
  !DeadFrame = adt.struct<ContinuationFrame(result: !Missing)>
  func.func @live() -> core.nil {
    ^entry:
      func.return
    ^unreachable(%value: !Missing):
      func.unreachable
  }
}"#,
        "adt.typeref \"Missing\" has no unique native layout",
    );
}

#[test]
fn reachable_recursive_nominal_layout_is_validated_once() {
    let (_ctx, _module, plan) = build(
        r#"core.module @test {
  !CursorRef = adt.typeref<{name = "Cursor"}>
  !Cursor = adt.struct<Cursor(next: !CursorRef)>
  func.func @make(%next: !CursorRef) -> !CursorRef {
    %cursor = adt.struct_new %next {type = !Cursor} : !CursorRef
    func.return %cursor
  }
}"#,
    );

    assert_eq!(plan.rtti_types().len(), 1);
    assert_eq!(plan.rtti_types()[0].fields, [FieldKind::Managed]);
}

#[test]
fn direct_layout_in_live_null_metadata_is_managed() {
    let (ctx, _module, plan) = build(
        r#"core.module @test {
  !Node = adt.struct<__native_node(value: core.i32)>
  func.func @empty() -> tribute_rt.anyref {
    %null = adt.ref_null {type = !Node} : tribute_rt.anyref
    func.return %null
  }
}"#,
    );

    let node = ctx
        .type_alias_by_text("Node")
        .expect("parsed native node alias");
    assert!(plan.is_managed_type(&ctx, node));
    assert!(plan.rtti_types().is_empty());
}

#[test]
fn nominal_layout_lookup_ignores_unreachable_interner_entries() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R) -> !R { func.return %value }
}"#,
    );
    let i64_ty = ctx.intern_type(TypeDataBuilder::new("core", "i64").build());
    let stale = tribute_ir::dialect::adt::struct_type(
        &mut ctx,
        "R",
        [("x", i64_ty)],
        trunk_ir::types::AttributeMap::new(),
    )
    .as_type_ref();
    assert!(ctx.type_alias_by_type(stale).is_none());
    production_plan(&ctx, module)
        .expect("an unreachable stale layout must not shadow the module declaration");

    let mut ambiguous_ctx = IrContext::new();
    let ambiguous = parse_test_module(
        &mut ambiguous_ctx,
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !First = adt.struct<R(x: core.i32)>
  !Second = adt.struct<R(x: core.i64)>
  func.func @f(%value: !R, %first: !First, %second: !Second) -> !R {
    func.return %value
  }
}"#,
    );
    assert!(production_plan(&ambiguous_ctx, ambiguous).is_err());
}

#[test]
fn plan_order_is_deterministic_and_duplicate_actions_fail_validation() {
    let (ctx, module, plan) = build(
        r#"core.module @test {
  !R = adt.typeref<{name = "R"}>
  !Layout = adt.struct<R(x: core.i32)>
  func.func @f(%value: !R) -> !R { func.return %value }
}"#,
    );
    let second = production_plan(&ctx, module).unwrap();
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
  !A = adt.struct<A(x: core.i32)>
  !B = adt.struct<B(x: core.i32)>
  !R = adt.typeref<{name = "A"}>
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
    let [first, _second] = plan.rtti_types() else {
        panic!("two exact RTTI layouts")
    };

    let mut stale_module = plan.clone();
    let other = parse_test_module(&mut ctx, "core.module @other {}");
    stale_module.module = other.op();
    assert!(stale_module.validate_against(&ctx, module).is_err());

    let mut duplicate_rtti = plan.clone();
    duplicate_rtti.rtti_types.push(first.clone());
    assert!(duplicate_rtti.validate_against(&ctx, module).is_err());

    let mut stale_bitmap = plan.clone();
    stale_bitmap.rtti_types[0].fields = vec![FieldKind::Raw];
    assert!(stale_bitmap.validate_against(&ctx, module).is_err());

    let function = &plan.functions[0];
    let body = ctx.op_region(function.operation, 0).unwrap();
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

    let other = plan.function(&Symbol::new("other")).unwrap();
    let other_body = ctx.op_region(other.operation, 0).unwrap();
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
fn closure_rtti_declaration_follows_the_native_closure_layout() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Closure = adt.struct<_closure(func_ptr: core.i32, env: tribute_rt.anyref), {layout = "closure"}>
  func.func @make(%code: core.i32, %env: tribute_rt.anyref) -> !Closure {
    %closure = adt.struct_new %code, %env {type = !Closure} : !Closure
    func.return %closure
  }
}"#,
    );
    let plan = production_plan(&ctx, module).expect("typed ownership plan");
    let semantic = plan.rtti_types()[0].ty;
    assert_eq!(
        plan.rtti_types()[0].fields,
        [
            FieldKind::Int {
                width: 32,
                signed: false
            },
            FieldKind::Dynamic
        ]
    );

    crate::native::rtti::declare_rtti_layouts(&mut ctx, module, plan.rtti_types());
    crate::native::adapt_closure_layout::lower(&mut ctx, module);
    let (type_converter, _) = native_type_converter(&mut ctx);
    func_to_clif::lower(&mut ctx, module, type_converter).expect("func_to_clif");

    let [layout] = tribute_ir::dialect::tribute_rtti::Layout::declared(&ctx, module)[..] else {
        panic!("one closure RTTI declaration")
    };
    assert_ne!(layout.r#type(&ctx), semantic);
    assert_eq!(layout.field_kinds(&ctx), plan.rtti_types()[0].fields);
    let (type_converter, _) = native_type_converter(&mut ctx);
    crate::native::rtti::generate_rtti(&mut ctx, module, &type_converter)
        .expect("the declaration names the adapted allocation layout");
}

#[test]
fn rtti_identity_never_falls_back_to_same_name_or_shape() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Exact = adt.struct<R(x: core.i32)>
  !SameNameShape = adt.struct<R(x: core.i32), {test.identity = 1}>
  !Stale = adt.struct<Stale(x: core.i32)>
  func.func @make(%x: core.i32) -> !Exact {
    %value = adt.struct_new %x {type = !Exact} : !Exact
    func.return %value
  }
}"#,
    );
    let plan = production_plan(&ctx, module).expect("typed ownership plan");
    crate::native::rtti::declare_rtti_layouts(&mut ctx, module, plan.rtti_types());
    let mut allocation = None;
    walk_module(&ctx, module, |op| {
        if adt::StructNew::matches(&ctx, op) {
            allocation = Some(op);
        }
    });
    let allocation = allocation.unwrap();
    let candidates = ctx
        .types()
        .iter()
        .filter_map(|(ty, data)| {
            (data.dialect == "adt" && data.name == "struct" && ty != plan.rtti_types()[0].ty)
                .then_some(ty)
        })
        .collect::<Vec<_>>();
    assert_eq!(candidates.len(), 2);

    let (type_converter, _) = native_type_converter(&mut ctx);
    for &candidate in &candidates {
        ctx.op_mut(allocation)
            .attributes
            .insert("type", trunk_ir::Attribute::Type(candidate));
        assert!(plan.validate_against(&ctx, module).is_err());
        assert!(
            crate::native::rtti::generate_rtti(&mut ctx, module, &type_converter).is_err(),
            "a same-name or same-shape layout must not reuse the exact declaration"
        );
    }
}

#[test]
fn nested_same_named_functions_are_distinct_qualified_definitions() {
    let (_, _, plan) = build(
        r#"core.module @test {
  core.module @left {
    func.func @helper(%value: core.i32) -> core.i32 {
      func.return %value
    }
  }
  core.module @right {
    func.func @helper(%value: core.i32) -> core.i32 {
      func.return %value
    }
  }
  func.func @main(%value: core.i32) -> core.i32 {
    %left = func.call %value {callee = @"left::helper"} : core.i32
    %right = func.call %left {callee = @"right::helper"} : core.i32
    func.return %right
  }
}"#,
    );
    assert_eq!(plan.functions().len(), 3);
}

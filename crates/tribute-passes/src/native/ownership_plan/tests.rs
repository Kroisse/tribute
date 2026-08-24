use super::*;
use crate::native::ownership_summary;
use crate::native::rc_insertion::{
    BorrowedParameterPolicy, TemporaryBorrowPolicy, insert_rc_with_policies_and_trusted_summaries,
};
use crate::native::type_converter::native_type_converter;
use trunk_ir::parser::parse_test_module;
use trunk_ir::printer::print_module;
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
fn continuation_frame_exposes_the_old_post_erasure_zero_retain_failure() {
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

    let (type_converter, _) = native_type_converter(&mut ctx);
    let trusted = ownership_summary::compute_and_attach(&mut ctx, module, &type_converter)
        .expect("legacy ownership summaries");
    let lowering = func_to_clif::lower(&mut ctx, module, type_converter).expect("func_to_clif");
    assert!(
        plan.remap_rtti_types(&ctx, module, lowering.rtti_layout_rewrites())
            .is_ok(),
        "func_to_clif must report every exact ADT allocation layout rewrite"
    );
    insert_rc_with_policies_and_trusted_summaries(
        &mut ctx,
        module,
        BorrowedParameterPolicy::ElideProvenBorrowed,
        TemporaryBorrowPolicy::Preserve,
        &trusted,
    )
    .expect("legacy RC insertion");
    let lowered = print_module(&ctx, module.op());
    assert_eq!(lowered.matches("tribute_rt.retain").count(), 0, "{lowered}");
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
fn cfg_copy_and_tail_dying_value_actions_are_complete() {
    let (_ctx, _module, plan) = build(
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
    ] {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let before = print_module(&ctx, module.op());
        assert!(build_native_ownership_plan(&ctx, module).is_err());
        assert_eq!(print_module(&ctx, module.op()), before);
    }
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
}"#,
    );
    let before = print_module(&ctx, module.op());
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

//! Cached liveness values and their dependency on typed ownership facts.

use std::collections::HashSet;
use std::sync::Arc;

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::parser::parse_test_module;

use super::*;

const CROSS_BLOCK_BORROW: &str = r#"core.module @test {
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
}"#;

const BRANCH_LOOP: &str = r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @flow(%condition: core.i1, %value: !R) -> core.nil {
    ^entry:
      cf.cond_br %condition [^left, ^right]
    ^left:
      cf.br [^join]
    ^right:
      cf.br [^join]
    ^join:
      cf.br [^loop]
    ^loop:
      %field = adt.struct_get %value {field = 0, type = !Layout} : core.i32
      cf.cond_br %condition [^loop, ^exit]
    ^exit:
      func.return
  }
}"#;

const ALIAS_PROJECTION: &str = r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Choice = adt.enum() {name = @Choice, variants = [[@Some, [!ChildRef]]]}
  !ChoiceRef = adt.typeref() {name = @Choice}
  func.func @load(%choice: !ChoiceRef) -> !ChildRef {
    ^entry:
      %erased = adt.ref_cast %choice {type = tribute_rt.anyref} : tribute_rt.anyref
      %restored = adt.ref_cast %erased {type = !ChoiceRef} : !ChoiceRef
      %child = adt.variant_get %restored {type = !Choice, tag = @Some, field = 0} : !ChildRef
      cf.br [^next]
    ^next:
      func.return %child
  }
}"#;

const NESTED_PROJECTION: &str = r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Inner = adt.struct() {name = @Inner, fields = [[@child, !ChildRef]]}
  !InnerRef = adt.typeref() {name = @Inner}
  !Box = adt.struct() {name = @Box, fields = [[@inner, !InnerRef]]}
  func.func @observe(%child: !ChildRef) -> core.i32 {
    %value = adt.struct_get %child {field = 0, type = !Child} : core.i32
    func.return %value
  }
  func.func @load(%child: !ChildRef) -> core.nil {
    ^entry:
      %inner = adt.struct_new %child {type = !Inner} : !InnerRef
      %owner = adt.struct_new %inner {type = !Box} : !Box
      %loaded = adt.struct_get %owner {field = 0, type = !Box} : !InnerRef
      %erased = adt.ref_cast %loaded {type = tribute_rt.anyref} : tribute_rt.anyref
      %restored = adt.ref_cast %erased {type = !InnerRef} : !InnerRef
      %nested = adt.struct_get %restored {field = 0, type = !Inner} : !ChildRef
      cf.br [^next]
    ^next:
      %seen = func.call %nested {callee = @observe} : core.i32
      func.return
  }
}"#;

fn function_op(ctx: &IrContext, module: Module, name: &'static str) -> OpRef {
    let mut found = None;
    walk_module(ctx, module, |op| {
        if let Ok(function) = func::Func::from_op(ctx, op)
            && function.sym_name(ctx) == Symbol::new(name)
        {
            found = Some(op);
        }
    });
    found.expect("defined function")
}

#[test]
fn both_views_reuse_facts_but_extend_only_proven_owner_liveness() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, CROSS_BLOCK_BORROW);
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();
    let liveness = cache.get::<NativeManagedLiveness>(&ctx, op).unwrap();
    assert_eq!(liveness.computed_views(), (false, false));
    let conservative = liveness.view(false);
    assert_eq!(liveness.computed_views(), (true, false));
    let extended = liveness.view(true);
    assert_eq!(liveness.computed_views(), (true, true));
    let facts = cache
        .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
        .unwrap();
    let [entry, next] = facts.cfg().blocks() else {
        panic!("two-block fixture")
    };
    let owner = ctx.op_result(ctx.block(*entry).ops[0], 0);
    let loaded = ctx.op_result(ctx.block(*entry).ops[1], 0);
    let child = ctx.block_args(*entry)[0];

    for view in [conservative, extended] {
        assert_eq!(
            view.defs(*entry).unwrap(),
            &HashSet::from([child, owner, loaded])
        );
        assert!(view.defs(*next).unwrap().is_empty());
        assert!(view.live_in(*entry).unwrap().is_empty());
        assert!(view.live_out(*next).unwrap().is_empty());
    }
    assert_eq!(
        conservative.live_in(*next).unwrap(),
        &HashSet::from([loaded])
    );
    assert_eq!(
        conservative.live_out(*entry).unwrap(),
        &HashSet::from([loaded])
    );
    assert_eq!(
        extended.live_in(*next).unwrap(),
        &HashSet::from([loaded, owner])
    );
    assert_eq!(
        extended.live_out(*entry).unwrap(),
        &HashSet::from([loaded, owner])
    );
    assert!(Arc::ptr_eq(
        &liveness,
        &cache.get::<NativeManagedLiveness>(&ctx, op).unwrap()
    ));
    assert!(std::ptr::eq(extended, liveness.view(true)));
}

#[test]
fn branch_merge_and_loop_reach_a_stable_fixed_point() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, BRANCH_LOOP);
    let op = function_op(&ctx, module, "flow");
    let mut cache = AnalysisCache::new();
    let liveness = cache.get::<NativeManagedLiveness>(&ctx, op).unwrap();
    let conservative = liveness.view(false);
    let extended = liveness.view(true);
    let facts = cache
        .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
        .unwrap();
    let [entry, left, right, join, loop_block, exit] = facts.cfg().blocks() else {
        panic!("six-block fixture")
    };
    let value = ctx.block_args(*entry)[1];
    for view in [conservative, extended] {
        assert_eq!(view.defs(*entry).unwrap(), &HashSet::from([value]));
        assert!(view.live_in(*entry).unwrap().is_empty());
        for block in [*left, *right, *join, *loop_block] {
            assert!(view.defs(block).unwrap().is_empty());
            assert_eq!(view.live_in(block).unwrap(), &HashSet::from([value]));
            assert_eq!(view.live_out(block).unwrap(), &HashSet::from([value]));
        }
        assert_eq!(view.live_out(*entry).unwrap(), &HashSet::from([value]));
        assert!(view.defs(*exit).unwrap().is_empty());
        assert!(view.live_in(*exit).unwrap().is_empty());
        assert!(view.live_out(*exit).unwrap().is_empty());
    }
}

#[test]
fn exact_aliases_do_not_become_separate_managed_definitions() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, ALIAS_PROJECTION);
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();
    let liveness = cache.get::<NativeManagedLiveness>(&ctx, op).unwrap();
    let conservative = liveness.view(false);
    let extended = liveness.view(true);
    let facts = cache
        .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
        .unwrap();
    let [entry, next] = facts.cfg().blocks() else {
        panic!("two-block fixture")
    };
    let root = ctx.block_args(*entry)[0];
    let erased = ctx.op_result(ctx.block(*entry).ops[0], 0);
    let restored = ctx.op_result(ctx.block(*entry).ops[1], 0);
    let child = ctx.op_result(ctx.block(*entry).ops[2], 0);
    for view in [conservative, extended] {
        assert_eq!(view.defs(*entry).unwrap(), &HashSet::from([root, child]));
        assert!(view.defs(*next).unwrap().is_empty());
    }
    assert!(!facts.managed_values().contains(&erased));
    assert!(!facts.managed_values().contains(&restored));
    assert_eq!(
        conservative.live_in(*next).unwrap(),
        &HashSet::from([child])
    );
    assert_eq!(
        extended.live_in(*next).unwrap(),
        &HashSet::from([root, child])
    );
}

#[test]
fn nested_projection_extends_the_outer_owner_through_cross_block_use() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, NESTED_PROJECTION);
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();
    let liveness = cache.get::<NativeManagedLiveness>(&ctx, op).unwrap();
    let conservative = liveness.view(false);
    let extended = liveness.view(true);
    let facts = cache
        .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
        .unwrap();
    let [entry, next] = facts.cfg().blocks() else {
        panic!("two-block fixture")
    };
    let outer = ctx.op_result(ctx.block(*entry).ops[1], 0);
    let loaded = ctx.op_result(ctx.block(*entry).ops[2], 0);
    let nested = ctx.op_result(ctx.block(*entry).ops[5], 0);
    assert!(facts.projection_owners().contains_key(&loaded));
    assert!(facts.projection_owners().contains_key(&nested));
    for view in [conservative, extended] {
        assert!(view.defs(*entry).unwrap().contains(&outer));
        assert!(view.defs(*entry).unwrap().contains(&loaded));
        assert!(view.defs(*entry).unwrap().contains(&nested));
        assert!(view.live_in(*next).unwrap().contains(&nested));
    }
    assert!(!conservative.live_in(*next).unwrap().contains(&outer));
    assert!(extended.live_in(*next).unwrap().contains(&outer));
    assert!(!extended.live_in(*next).unwrap().contains(&loaded));
}

#[test]
fn liveness_invalidation_is_precise_and_transitive() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, CROSS_BLOCK_BORROW);
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();
    let first = cache.get::<NativeManagedLiveness>(&ctx, op).unwrap();
    first.view(false);
    first.view(true);
    let facts = cache
        .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
        .unwrap();

    cache.invalidate::<NativeManagedLiveness>(op);
    assert!(
        cache
            .get_cached::<NativeManagedLiveness>(&ctx, op)
            .is_none()
    );
    assert!(Arc::ptr_eq(
        &facts,
        &cache
            .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
            .unwrap()
    ));
    let recomputed = cache.get::<NativeManagedLiveness>(&ctx, op).unwrap();
    assert!(!Arc::ptr_eq(&first, &recomputed));
    assert_eq!(recomputed.computed_views(), (false, false));
    recomputed.view(false);
    assert_eq!(recomputed.computed_views(), (true, false));

    cache.invalidate::<NativeOwnershipFunctionFacts>(op);
    assert!(
        cache
            .get_cached::<NativeManagedLiveness>(&ctx, op)
            .is_none()
    );
    assert!(
        cache
            .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
            .is_none()
    );
    assert!(
        cache
            .get_cached::<NativeOwnershipModuleFacts>(&ctx, module.op())
            .is_some()
    );
}

#[test]
fn planner_selects_liveness_only_from_field_borrow_policy() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, CROSS_BLOCK_BORROW);
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();

    let preserved = build_native_ownership_plan_with_analyses(
        &ctx,
        module,
        NativeOwnershipPlanOptions {
            elide_proven_borrowed_parameters: false,
            elide_proven_field_borrows: false,
        },
        &mut cache,
    )
    .unwrap();
    let liveness = cache.get_cached::<NativeManagedLiveness>(&ctx, op).unwrap();
    assert_eq!(liveness.computed_views(), (true, false));

    build_native_ownership_plan_with_analyses(
        &ctx,
        module,
        NativeOwnershipPlanOptions {
            elide_proven_borrowed_parameters: true,
            elide_proven_field_borrows: false,
        },
        &mut cache,
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        &liveness,
        &cache.get_cached::<NativeManagedLiveness>(&ctx, op).unwrap()
    ));
    assert_eq!(liveness.computed_views(), (true, false));

    let field_borrows = build_native_ownership_plan_with_analyses(
        &ctx,
        module,
        NativeOwnershipPlanOptions {
            elide_proven_borrowed_parameters: false,
            elide_proven_field_borrows: true,
        },
        &mut cache,
    )
    .unwrap();
    assert_eq!(liveness.computed_views(), (true, true));
    assert_ne!(
        preserved.function(Symbol::new("load")).unwrap().actions(),
        field_borrows
            .function(Symbol::new("load"))
            .unwrap()
            .actions()
    );
    build_native_ownership_plan_with_analyses(
        &ctx,
        module,
        NativeOwnershipPlanOptions::production(),
        &mut cache,
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        &liveness,
        &cache.get_cached::<NativeManagedLiveness>(&ctx, op).unwrap()
    ));
}

#[test]
fn production_plan_computes_only_owner_extended_view() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, CROSS_BLOCK_BORROW);
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();

    build_native_ownership_plan_with_analyses(
        &ctx,
        module,
        NativeOwnershipPlanOptions::production(),
        &mut cache,
    )
    .unwrap();
    let liveness = cache.get_cached::<NativeManagedLiveness>(&ctx, op).unwrap();
    assert_eq!(liveness.computed_views(), (false, true));
    assert!(Arc::ptr_eq(
        &liveness,
        &cache.get::<NativeManagedLiveness>(&ctx, op).unwrap()
    ));
    assert_eq!(liveness.computed_views(), (false, true));
}

#[test]
fn failed_facts_lookup_does_not_publish_liveness() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !R = adt.typeref() {name = @R}
  !Layout = adt.struct() {name = @R, fields = [[@x, core.i32]]}
  func.func @bad(%value: !R) -> core.i32 {
    %field = adt.struct_get %value {field = 1, type = !Layout} : core.i32
    func.return %field
  }
}"#,
    );
    let op = function_op(&ctx, module, "bad");
    let mut cache = AnalysisCache::new();
    let direct = cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .err()
        .expect("invalid facts");
    let error = cache
        .get::<NativeManagedLiveness>(&ctx, op)
        .err()
        .expect("invalid liveness");
    assert_eq!(error.analysis_type(), direct.analysis_type());
    assert_eq!(error.target(), direct.target());
    assert_eq!(error.to_string(), direct.to_string());
    assert!(
        cache
            .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
            .is_none()
    );
    assert!(
        cache
            .get_cached::<NativeManagedLiveness>(&ctx, op)
            .is_none()
    );
}

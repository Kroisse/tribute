//! Tests for the policy-neutral ownership flow analyses.
//!
//! These cover caching and invalidation behavior plus the fact sets the
//! planner consumes. Planner-level parity is covered by `tests.rs`.

use std::sync::Arc;

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::parser::parse_test_module;
use trunk_ir::printer::print_module;

use super::*;

const BORROW_FIXTURE: &str = r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  !Box = adt.struct() {name = @Box, fields = [[@child, !ChildRef]]}
  !BoxRef = adt.typeref() {name = @Box}
  func.func @observe(%child: !ChildRef) -> core.i32 {
    %value = adt.struct_get %child {field = 0, type = !Child} : core.i32
    func.return %value
  }
  func.func @load(%owner: !BoxRef) -> core.i32 {
    %child = adt.struct_get %owner {field = 0, type = !Box} : !ChildRef
    %value = func.call %child {callee = @observe} : core.i32
    func.return %value
  }
}"#;

const CAST_FIXTURE: &str = r#"core.module @test {
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
    found.expect("defined function op")
}

#[test]
fn both_planner_modes_reuse_one_cached_fact_set() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, BORROW_FIXTURE);
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
    .expect("preserved typed plan");
    let preserved_facts = cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .expect("function facts");

    let elided = build_native_ownership_plan_with_analyses(
        &ctx,
        module,
        NativeOwnershipPlanOptions::production(),
        &mut cache,
    )
    .expect("elided typed plan");
    let elided_facts = cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .expect("function facts");

    // The policy differs, so the plans differ...
    assert_eq!(
        preserved.function(Symbol::new("load")).unwrap().entries(),
        [EntryOwnership::Retained]
    );
    assert_eq!(
        elided.function(Symbol::new("load")).unwrap().entries(),
        [EntryOwnership::Borrowed]
    );
    // ...while the policy-neutral facts are computed once and shared.
    assert!(Arc::ptr_eq(&preserved_facts, &elided_facts));
}

#[test]
fn repeated_lookups_reuse_facts_and_invalidation_recomputes_them() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, BORROW_FIXTURE);
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();

    assert!(
        cache
            .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
            .is_none()
    );
    let first = cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .expect("function facts");
    let reused = cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .expect("function facts");
    assert!(Arc::ptr_eq(&first, &reused));

    cache.invalidate::<NativeOwnershipFunctionFacts>(op);
    let recomputed = cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .expect("function facts");
    assert!(!Arc::ptr_eq(&first, &recomputed));
    // Invalidating only the dependent keeps the reusable prerequisite.
    assert!(
        cache
            .get_cached::<NativeOwnershipModuleFacts>(&ctx, module.op())
            .is_some()
    );

    // Invalidating the prerequisite revalidates and recomputes the dependent.
    cache.invalidate::<NativeOwnershipModuleFacts>(module.op());
    assert!(
        cache
            .get_cached::<NativeOwnershipModuleFacts>(&ctx, module.op())
            .is_none()
    );
    let cascaded = cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .expect("function facts");
    assert!(!Arc::ptr_eq(&recomputed, &cascaded));
}

#[test]
fn malformed_projection_fails_closed_without_publishing_facts() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @test {
  !Child = adt.struct() {name = @Child, fields = [[@value, core.i32]]}
  !ChildRef = adt.typeref() {name = @Child}
  func.func @load(%child: !ChildRef) -> core.i32 {
    %value = adt.struct_get %child {field = 1, type = !Child} : core.i32
    func.return %value
  }
}"#,
    );
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();
    let before = print_module(&ctx, module.op());

    // The module boundary still validates; only the function contract fails.
    assert!(
        cache
            .get::<NativeOwnershipModuleFacts>(&ctx, module.op())
            .is_ok()
    );
    assert!(cache.get::<NativeOwnershipFunctionFacts>(&ctx, op).is_err());
    assert!(
        cache
            .get_cached::<NativeOwnershipFunctionFacts>(&ctx, op)
            .is_none()
    );
    // A retry revalidates from scratch instead of reusing a partial result.
    assert!(cache.get::<NativeOwnershipFunctionFacts>(&ctx, op).is_err());
    assert_eq!(print_module(&ctx, module.op()), before);
}

#[test]
fn nested_functions_depend_on_the_outermost_module_facts() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(
        &mut ctx,
        r#"core.module @outer {
  core.module @inner {
    func.func @load(%value: core.i32) -> core.i32 {
      func.return %value
    }
  }
}"#,
    );
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();
    cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .expect("function facts");

    // The function keys to the outermost module facts the planner consumes,
    // so no narrower inner-module entry is ever published.
    assert!(
        cache
            .get_cached::<NativeOwnershipModuleFacts>(&ctx, module.op())
            .is_some()
    );
    let mut inner = None;
    walk_module(&ctx, module, |candidate| {
        if candidate != module.op() && Module::new(&ctx, candidate).is_some() {
            inner = Some(candidate);
        }
    });
    let inner = inner.expect("nested module");
    assert!(
        cache
            .get_cached::<NativeOwnershipModuleFacts>(&ctx, inner)
            .is_none()
    );
}

#[test]
fn facts_record_exact_alias_roots_and_projection_owners() {
    let mut ctx = IrContext::new();
    let module = parse_test_module(&mut ctx, CAST_FIXTURE);
    let op = function_op(&ctx, module, "load");
    let mut cache = AnalysisCache::new();
    let facts = cache
        .get::<NativeOwnershipFunctionFacts>(&ctx, op)
        .expect("function facts");

    let mut casts = Vec::new();
    let mut projection = None;
    walk_module(&ctx, module, |candidate| {
        if adt::RefCast::matches(&ctx, candidate) {
            casts.push(ctx.op_result(candidate, 0));
        }
        if adt::VariantGet::matches(&ctx, candidate) {
            projection = Some(ctx.op_result(candidate, 0));
        }
    });
    let body = ctx.op(op).regions[0];
    let root = ctx.block_args(ctx.region(body).blocks[0])[0];
    let [first, second] = casts.as_slice() else {
        panic!("expected two casts");
    };
    let projection = projection.expect("variant projection");

    // Alias roots come from the typed cast contract, not from physical shape.
    assert_eq!(facts.aliases().get(first), Some(&root));
    assert_eq!(facts.aliases().get(second), Some(&root));
    assert_eq!(facts.projection_owners().get(&projection), Some(&root));

    // Aliased results fold into their root instead of becoming units.
    assert!(facts.managed_values().contains(&root));
    assert!(!facts.managed_values().contains(first));
    assert!(!facts.managed_values().contains(second));
    // The projection keeps its own unit and owns a managed root.
    assert!(facts.managed_values().contains(&projection));
}

use super::*;
use salsa_test_macros::salsa_test;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use trunk_ir::dialect::{arith, cont, func, trampoline};
use trunk_ir::ir::BlockBuilder;
use trunk_ir::rewrite::{OpAdaptor, RewriteContext, RewritePattern, RewriteResult, TypeConverter};
use trunk_ir::{Attribute, BlockArg, BlockId, IdVec, PathId, Span};

/// Create a shared test location with a fixed span `(0, 0)`.
///
/// **Caution:** `ShiftAnalysis` keys by `Span`, so tests exercising multiple
/// shift operations must use distinct spans (not this helper) to avoid
/// key collisions.
fn test_location(db: &dyn salsa::Database) -> trunk_ir::Location<'_> {
    let path = PathId::new(db, "test.trb".to_owned());
    trunk_ir::Location::new(path, Span::new(0, 0))
}

// ========================================================================
// Test: Multi-suspend block handling
// ========================================================================

/// Test helper: builds handler_dispatch with multiple suspend blocks and applies pattern.
/// Helper function to count nested scf.if operations in a region.
/// This counts the total number of scf.if operations, including nested ones.
fn count_scf_if_in_region<'db>(db: &'db dyn salsa::Database, region: &Region<'db>) -> usize {
    let mut count = 0;
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if scf::If::from_operation(db, *op).is_ok() {
                count += 1;
                // Count nested scf.if in the then/else regions
                for nested_region in op.regions(db).iter() {
                    count += count_scf_if_in_region(db, nested_region);
                }
            }
        }
    }
    count
}

/// Returns the number of nested scf.if operations in the suspend region.
/// With the new dispatch structure, each suspend arm becomes a branch in nested scf.if.
/// For 2 suspend arms: if (op_idx == 0) { arm0 } else { if (true) { arm1 } else { arm1 } }
/// So we expect 2 scf.if operations (outer dispatch + inner for last arm).
#[salsa::tracked]
fn handler_dispatch_scf_if_count(db: &dyn salsa::Database) -> usize {
    let location = test_location(db);
    let step_ty = trampoline::Step::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();

    // Create 3 blocks: done block + 2 suspend blocks
    let done_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());

    // Create marker block args for suspend blocks (required by collect_suspend_arms)
    let marker_arg1 = {
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            Symbol::new("op_name"),
            Attribute::Symbol(Symbol::new("get")),
        );
        BlockArg::new(db, i32_ty, attrs)
    };

    let marker_arg2 = {
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            Symbol::new("op_name"),
            Attribute::Symbol(Symbol::new("set")),
        );
        BlockArg::new(db, i32_ty, attrs)
    };

    let mut builder1 = BlockBuilder::new(db, location);
    let zero1 = builder1.op(arith::Const::i32(db, location, 1));
    let step1 = builder1.op(trampoline::step_done(
        db,
        location,
        zero1.result(db),
        step_ty,
    ));
    builder1.op(scf::r#yield(db, location, vec![step1.result(db)]));
    let suspend_block1 = {
        let block = builder1.build();
        // Rebuild block with marker arg
        Block::new(
            db,
            block.id(db),
            location,
            IdVec::from(vec![marker_arg1]),
            block.operations(db).clone(),
        )
    };

    let mut builder2 = BlockBuilder::new(db, location);
    let zero2 = builder2.op(arith::Const::i32(db, location, 2));
    let step2 = builder2.op(trampoline::step_done(
        db,
        location,
        zero2.result(db),
        step_ty,
    ));
    builder2.op(scf::r#yield(db, location, vec![step2.result(db)]));
    let suspend_block2 = {
        let block = builder2.build();
        // Rebuild block with marker arg
        Block::new(
            db,
            block.id(db),
            location,
            IdVec::from(vec![marker_arg2]),
            block.operations(db).clone(),
        )
    };

    let body_region = Region::new(
        db,
        location,
        IdVec::from(vec![done_block, suspend_block1, suspend_block2]),
    );

    // Create a dummy result value for handler_dispatch
    let dummy_const = arith::Const::i32(db, location, 0);
    let result_val = dummy_const.as_operation().result(db, 0);

    // Create handler_dispatch with 3 blocks
    let test_tag: u32 = 12345; // Dummy tag for testing
    let i32_ty = core::I32::new(db).as_type();
    let dispatch_op = cont::handler_dispatch(
        db,
        location,
        result_val,
        step_ty,
        test_tag,
        i32_ty,
        body_region,
    )
    .as_operation();

    // Apply pattern
    let effectful_funcs = Rc::new(HashSet::new());
    let handlers_in_effectful_funcs = Rc::new(HashSet::new());
    let pattern = LowerHandlerDispatchPattern {
        effectful_funcs,
        handlers_in_effectful_funcs,
    };
    let ctx = RewriteContext::new();
    let type_converter = TypeConverter::new();
    let adaptor = OpAdaptor::new(
        dispatch_op,
        dispatch_op.operands(db).clone(),
        vec![],
        &ctx,
        &type_converter,
    );
    let result = pattern.match_and_rewrite(db, &dispatch_op, &adaptor);

    // Count scf.if operations in the loop body
    // With trampoline loop, the result is a single scf.loop operation
    match result {
        RewriteResult::Expand(ops) if ops.len() == 1 => {
            let loop_op = &ops[0];
            let regions = loop_op.regions(db);
            if !regions.is_empty() {
                // Count scf.if in the loop body region
                count_scf_if_in_region(db, &regions[0])
            } else {
                0
            }
        }
        _ => 0,
    }
}

#[salsa_test]
fn test_handler_dispatch_collects_all_suspend_blocks(db: &salsa::DatabaseImpl) {
    // With trampoline loop structure and 2 suspend arms, we expect:
    // - 1 scf.if for is_done check (done vs shift branch)
    // - 1 scf.if for tag_matches check (in shift branch)
    // - 2 scf.if for arm dispatch (outer dispatch + last arm always-true)
    // Total: 4 scf.if operations
    let count = handler_dispatch_scf_if_count(db);
    assert_eq!(
        count, 4,
        "Loop body should have 4 scf.if: is_done + tag_matches + 2 arm dispatch"
    );
}

// ========================================================================
// Test: is_step_value with scf.if type verification
// ========================================================================

/// Test helper: creates scf.if returning Step type and checks is_step_value.
#[salsa::tracked]
fn is_step_value_for_step_if(db: &dyn salsa::Database) -> bool {
    let location = test_location(db);
    let step_ty = trampoline::Step::new(db).as_type();

    // Create scf.if returning Step type
    let cond_op = arith::Const::i32(db, location, 1);
    let cond_val = cond_op.as_operation().result(db, 0);

    let mut then_builder = BlockBuilder::new(db, location);
    let zero = then_builder.op(arith::Const::i32(db, location, 0));
    let step = then_builder.op(trampoline::step_done(
        db,
        location,
        zero.result(db),
        step_ty,
    ));
    then_builder.op(scf::r#yield(db, location, vec![step.result(db)]));
    let then_region = Region::new(db, location, IdVec::from(vec![then_builder.build()]));

    let mut else_builder = BlockBuilder::new(db, location);
    let one = else_builder.op(arith::Const::i32(db, location, 1));
    let step2 = else_builder.op(trampoline::step_done(db, location, one.result(db), step_ty));
    else_builder.op(scf::r#yield(db, location, vec![step2.result(db)]));
    let else_region = Region::new(db, location, IdVec::from(vec![else_builder.build()]));

    let if_op = scf::r#if(db, location, cond_val, step_ty, then_region, else_region);
    let if_result = if_op.as_operation().result(db, 0);

    let empty_map = HashMap::new();
    is_step_value(db, if_result, &empty_map)
}

/// Test helper: creates scf.if returning i32 (non-Step) and checks is_step_value.
#[salsa::tracked]
fn is_step_value_for_i32_if(db: &dyn salsa::Database) -> bool {
    let location = test_location(db);
    let i32_ty = core::I32::new(db).as_type();

    // Create scf.if returning i32 (not Step)
    let cond_op = arith::Const::i32(db, location, 1);
    let cond_val = cond_op.as_operation().result(db, 0);

    let mut then_builder = BlockBuilder::new(db, location);
    let val1 = then_builder.op(arith::Const::i32(db, location, 42));
    then_builder.op(scf::r#yield(db, location, vec![val1.result(db)]));
    let then_region = Region::new(db, location, IdVec::from(vec![then_builder.build()]));

    let mut else_builder = BlockBuilder::new(db, location);
    let val2 = else_builder.op(arith::Const::i32(db, location, 0));
    else_builder.op(scf::r#yield(db, location, vec![val2.result(db)]));
    let else_region = Region::new(db, location, IdVec::from(vec![else_builder.build()]));

    let if_op = scf::r#if(db, location, cond_val, i32_ty, then_region, else_region);
    let if_result = if_op.as_operation().result(db, 0);

    let empty_map = HashMap::new();
    is_step_value(db, if_result, &empty_map)
}

#[salsa_test]
fn test_is_step_value_scf_if_with_step_type(db: &salsa::DatabaseImpl) {
    assert!(
        is_step_value_for_step_if(db),
        "scf.if returning Step should be detected as Step value"
    );
}

#[salsa_test]
fn test_is_step_value_scf_if_with_non_step_type(db: &salsa::DatabaseImpl) {
    assert!(
        !is_step_value_for_i32_if(db),
        "scf.if returning i32 should NOT be detected as Step value"
    );
}

// ========================================================================
// Test: Resume function generation (no global state)
// ========================================================================

#[test]
fn test_fresh_resume_name_generates_unique_names() {
    let counter = Rc::new(RefCell::new(0u32));

    let name1 = fresh_resume_name(&counter);
    let name2 = fresh_resume_name(&counter);
    let name3 = fresh_resume_name(&counter);

    assert_eq!(name1, "__resume_0");
    assert_eq!(name2, "__resume_1");
    assert_eq!(name3, "__resume_2");
    assert_eq!(*counter.borrow(), 3);
}

#[test]
fn test_resume_specs_isolation() {
    // Test that different ResumeCounter instances are independent
    let counter1 = Rc::new(RefCell::new(0u32));
    let counter2 = Rc::new(RefCell::new(0u32));

    // Increment counter1 twice
    *counter1.borrow_mut() += 1;
    *counter1.borrow_mut() += 1;

    // counter2 should still be 0
    assert_eq!(*counter1.borrow(), 2);
    assert_eq!(*counter2.borrow(), 0, "counter2 should be independent");
}

// ========================================================================
// Test: Utility functions
// ========================================================================

#[test]
fn test_compute_op_idx_deterministic() {
    // Same inputs should produce same output
    let idx1 = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
    let idx2 = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
    assert_eq!(idx1, idx2, "Same inputs should produce same op_idx");

    // Different op names should produce different indices
    let idx_get = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
    let idx_set = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("set")));
    assert_ne!(
        idx_get, idx_set,
        "Different ops should have different indices"
    );

    // Different abilities should produce different indices
    let idx_state = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
    let idx_console = compute_op_idx(Some(Symbol::new("Console")), Some(Symbol::new("get")));
    assert_ne!(
        idx_state, idx_console,
        "Different abilities should have different indices"
    );
}

#[test]
fn test_state_type_name_deterministic() {
    let module = Symbol::new("test_module");

    // Same inputs should produce same output
    let name1 = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module,
        span: Span::new(10, 20),
        shift_index: 0,
    });
    let name2 = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module,
        span: Span::new(10, 20),
        shift_index: 0,
    });
    assert_eq!(name1, name2, "Same inputs should produce same name");

    // Name should start with __State_ prefix
    assert!(
        name1.starts_with("__State_"),
        "State type name should have __State_ prefix"
    );

    // Different spans should produce different names
    let name_span1 = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module,
        span: Span::new(10, 20),
        shift_index: 0,
    });
    let name_span2 = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module,
        span: Span::new(30, 40),
        shift_index: 0,
    });
    assert_ne!(
        name_span1, name_span2,
        "Different spans should produce different names"
    );

    // Different ops should produce different names
    let name_get = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module,
        span: Span::new(10, 20),
        shift_index: 0,
    });
    let name_set = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("set")),
        module_name: module,
        span: Span::new(10, 20),
        shift_index: 0,
    });
    assert_ne!(
        name_get, name_set,
        "Different ops should produce different names"
    );

    // Different shift indices should produce different names
    let name_idx0 = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module,
        span: Span::new(10, 20),
        shift_index: 0,
    });
    let name_idx1 = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module,
        span: Span::new(10, 20),
        shift_index: 1,
    });
    assert_ne!(
        name_idx0, name_idx1,
        "Different shift indices should produce different names"
    );
}

#[test]
fn test_state_type_name_module_uniqueness() {
    let module1 = Symbol::new("test_module1");
    let module2 = Symbol::new("test_module2");

    // Different module_names with same span should produce different names
    let name_mod1 = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module1,
        span: Span::new(10, 20),
        shift_index: 0,
    });
    let name_mod2 = state_type_name(StateTypeKey {
        ability_name: Some(Symbol::new("State")),
        op_name: Some(Symbol::new("get")),
        module_name: module2,
        span: Span::new(10, 20),
        shift_index: 0,
    });
    assert_ne!(
        name_mod1, name_mod2,
        "Different module_names should produce different names"
    );
}

// ========================================================================
// Test: build_nested_dispatch i1 condition types
// ========================================================================

/// Verify that build_nested_dispatch produces i1-typed conditions.
///
/// With 2 SuspendArms, the first arm generates arith.cmp_eq (result: i1)
/// and the last arm generates arith.const true (result: i1).
#[salsa::tracked]
fn run_build_nested_dispatch_i1_test(db: &dyn salsa::Database) -> Result<(), String> {
    let location = test_location(db);
    let step_ty = trampoline::Step::new(db).as_type();
    let i32_ty = core::I32::new(db).as_type();
    let i1_ty = core::I::<1>::new(db).as_type();

    // Create 2 simple suspend arm blocks with marker args
    let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));

    let block_get = {
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            Symbol::new("ability_ref"),
            Attribute::Type(state_ref.as_type()),
        );
        attrs.insert(
            Symbol::new("op_name"),
            Attribute::Symbol(Symbol::new("get")),
        );
        let marker_arg = BlockArg::new(db, i32_ty, attrs);
        let mut b = BlockBuilder::new(db, location);
        let c = b.op(arith::Const::i32(db, location, 0));
        let step = b.op(trampoline::step_done(db, location, c.result(db), step_ty));
        b.op(scf::r#yield(db, location, vec![step.result(db)]));
        let block = b.build();
        Block::new(
            db,
            block.id(db),
            location,
            IdVec::from(vec![marker_arg]),
            block.operations(db).clone(),
        )
    };

    let block_set = {
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            Symbol::new("ability_ref"),
            Attribute::Type(state_ref.as_type()),
        );
        attrs.insert(
            Symbol::new("op_name"),
            Attribute::Symbol(Symbol::new("set")),
        );
        let marker_arg = BlockArg::new(db, i32_ty, attrs);
        let mut b = BlockBuilder::new(db, location);
        let c = b.op(arith::Const::i32(db, location, 0));
        let step = b.op(trampoline::step_done(db, location, c.result(db), step_ty));
        b.op(scf::r#yield(db, location, vec![step.result(db)]));
        let block = b.build();
        Block::new(
            db,
            block.id(db),
            location,
            IdVec::from(vec![marker_arg]),
            block.operations(db).clone(),
        )
    };

    let arms = vec![
        SuspendArm {
            expected_op_idx: compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get"))),
            block: block_get,
        },
        SuspendArm {
            expected_op_idx: compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("set"))),
            block: block_set,
        },
    ];

    // Build a dummy current_op_idx value
    let mut builder = BlockBuilder::new(db, location);
    let op_idx_const = builder.op(arith::Const::i32(db, location, 0));

    build_nested_dispatch(
        db,
        &mut builder,
        location,
        step_ty,
        op_idx_const.result(db),
        &arms,
        0,
        &HashSet::new(),
    );

    let block = builder.build();
    let ops = block.operations(db);

    // Find arith.cmp_eq and check its result type is i1
    let mut found_cmp_eq = false;
    let mut found_const_i1 = false;
    for op in ops.iter() {
        if let Ok(cmp_op) = arith::CmpEq::from_operation(db, *op) {
            let result_ty = cmp_op.as_operation().results(db);
            if result_ty.len() == 1 && result_ty[0] == i1_ty {
                found_cmp_eq = true;
            }
        }
    }

    // Check the scf.if's then/else regions for the last arm's arith.const with i1 type
    for op in ops.iter() {
        if let Ok(if_op) = scf::If::from_operation(db, *op) {
            // The else region contains the last arm dispatch with arith.const true (i1)
            let else_region = if_op.r#else(db);
            for else_block in else_region.blocks(db).iter() {
                for else_op in else_block.operations(db).iter() {
                    if arith::Const::from_operation(db, *else_op).is_ok() {
                        let results = else_op.results(db);
                        if results.len() == 1 && results[0] == i1_ty {
                            found_const_i1 = true;
                        }
                    }
                }
            }
        }
    }

    if !found_cmp_eq {
        return Err("arith.cmp_eq with i1 result type not found in dispatch".to_string());
    }
    if !found_const_i1 {
        return Err(
            "arith.const with i1 result type not found in last-arm else region".to_string(),
        );
    }

    Ok(())
}

#[salsa_test]
fn test_build_nested_dispatch_uses_i1_conditions(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_build_nested_dispatch_i1_test(db) {
        panic!("{}", msg);
    }
}

// ========================================================================
// Test: Hash-based dispatch (collect_suspend_arms)
// ========================================================================

/// Helper tracked function to test hash-based indexing.
#[salsa::tracked]
fn run_collect_suspend_arms_table_based_test(db: &dyn salsa::Database) -> Result<(), String> {
    let location = test_location(db);
    let i32_ty = core::I32::new(db).as_type();

    // Create done block (block 0)
    let done_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());

    // Create suspend blocks with ability_ref and op_name attributes
    let state_ref = core::AbilityRefType::simple(db, Symbol::new("State"));

    let marker_arg_get = {
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            Symbol::new("ability_ref"),
            Attribute::Type(state_ref.as_type()),
        );
        attrs.insert(
            Symbol::new("op_name"),
            Attribute::Symbol(Symbol::new("get")),
        );
        BlockArg::new(db, i32_ty, attrs)
    };

    let marker_arg_set = {
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            Symbol::new("ability_ref"),
            Attribute::Type(state_ref.as_type()),
        );
        attrs.insert(
            Symbol::new("op_name"),
            Attribute::Symbol(Symbol::new("set")),
        );
        BlockArg::new(db, i32_ty, attrs)
    };

    let marker_arg_modify = {
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            Symbol::new("ability_ref"),
            Attribute::Type(state_ref.as_type()),
        );
        attrs.insert(
            Symbol::new("op_name"),
            Attribute::Symbol(Symbol::new("modify")),
        );
        BlockArg::new(db, i32_ty, attrs)
    };

    // Create suspend blocks
    let suspend_block_get = Block::new(
        db,
        BlockId::fresh(),
        location,
        IdVec::from(vec![marker_arg_get]),
        IdVec::new(),
    );
    let suspend_block_set = Block::new(
        db,
        BlockId::fresh(),
        location,
        IdVec::from(vec![marker_arg_set]),
        IdVec::new(),
    );
    let suspend_block_modify = Block::new(
        db,
        BlockId::fresh(),
        location,
        IdVec::from(vec![marker_arg_modify]),
        IdVec::new(),
    );

    // Collect blocks: done + 3 suspend blocks
    let blocks = IdVec::from(vec![
        done_block,
        suspend_block_get,
        suspend_block_set,
        suspend_block_modify,
    ]);

    // Collect suspend arms
    let arms = collect_suspend_arms(db, &blocks);

    // Should have 3 arms (skip done block)
    if arms.len() != 3 {
        return Err(format!("Expected 3 suspend arms, got {}", arms.len()));
    }

    // With hash-based dispatch, expected_op_idx is compute_op_idx(ability, op_name)
    let expected_get = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("get")));
    let expected_set = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("set")));
    let expected_modify = compute_op_idx(Some(Symbol::new("State")), Some(Symbol::new("modify")));

    if arms[0].expected_op_idx != expected_get {
        return Err(format!(
            "First arm (get) should have op_idx {}, got {}",
            expected_get, arms[0].expected_op_idx
        ));
    }
    if arms[1].expected_op_idx != expected_set {
        return Err(format!(
            "Second arm (set) should have op_idx {}, got {}",
            expected_set, arms[1].expected_op_idx
        ));
    }
    if arms[2].expected_op_idx != expected_modify {
        return Err(format!(
            "Third arm (modify) should have op_idx {}, got {}",
            expected_modify, arms[2].expected_op_idx
        ));
    }

    Ok(())
}

/// Test that collect_suspend_arms assigns hash-based indices.
#[salsa_test]
fn test_collect_suspend_arms_hash_based_indexing(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_collect_suspend_arms_table_based_test(db) {
        panic!("{}", msg);
    }
}

/// Helper tracked function to test done-only case.
#[salsa::tracked]
fn run_collect_suspend_arms_done_only_test(db: &dyn salsa::Database) -> Result<(), String> {
    let location = test_location(db);

    // Create only done block (block 0)
    let done_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());

    let blocks = IdVec::from(vec![done_block]);

    // Collect suspend arms
    let arms = collect_suspend_arms(db, &blocks);

    // Should have no arms (only done block)
    if !arms.is_empty() {
        return Err(format!(
            "Expected no suspend arms when only done block exists, got {}",
            arms.len()
        ));
    }

    Ok(())
}

/// Test that collect_suspend_arms handles empty blocks correctly.
#[salsa_test]
fn test_collect_suspend_arms_done_only(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_collect_suspend_arms_done_only_test(db) {
        panic!("{}", msg);
    }
}

// ========================================================================
// compute_op_idx tests
// ========================================================================

#[test]
fn test_compute_op_idx_is_deterministic() {
    let ability = Some(Symbol::new("State"));
    let op = Some(Symbol::new("get"));
    let idx1 = compute_op_idx(ability, op);
    let idx2 = compute_op_idx(ability, op);
    assert_eq!(idx1, idx2, "compute_op_idx should be deterministic");
}

#[test]
fn test_compute_op_idx_differs_for_different_ops() {
    let ability = Some(Symbol::new("State"));
    let get_idx = compute_op_idx(ability, Some(Symbol::new("get")));
    let set_idx = compute_op_idx(ability, Some(Symbol::new("set")));
    assert_ne!(
        get_idx, set_idx,
        "different operations should produce different indices"
    );
}

#[test]
fn test_compute_op_idx_differs_for_different_abilities() {
    let op = Some(Symbol::new("get"));
    let state_idx = compute_op_idx(Some(Symbol::new("State")), op);
    let console_idx = compute_op_idx(Some(Symbol::new("Console")), op);
    assert_ne!(
        state_idx, console_idx,
        "different abilities should produce different indices"
    );
}

#[test]
fn test_compute_op_idx_handles_none() {
    // Should not panic with None values
    let idx_none_both = compute_op_idx(None, None);
    let idx_none_ability = compute_op_idx(None, Some(Symbol::new("get")));
    let idx_none_op = compute_op_idx(Some(Symbol::new("State")), None);

    // None cases should differ from populated cases
    assert_ne!(idx_none_both, idx_none_ability);
    assert_ne!(idx_none_both, idx_none_op);
}

// ========================================================================
// Test: truncate_scf_if_block — pure branch regression
// ========================================================================

/// Pure branch (no effectful calls): original ops are preserved, and the
/// yield operand is wrapped in `trampoline.step_done` + `scf.yield`.
#[salsa::tracked]
fn run_truncate_scf_if_block_pure_branch(db: &dyn salsa::Database) -> Result<(), String> {
    let location = test_location(db);
    let step_ty = trampoline::Step::new(db).as_type();

    // Build a block: arith.const 42 ; scf.yield(%const)
    let mut builder = BlockBuilder::new(db, location);
    let c = builder.op(arith::Const::i32(db, location, 42));
    builder.op(scf::r#yield(db, location, vec![c.result(db)]));
    let block = builder.build();

    let effectful_funcs: HashSet<Symbol> = HashSet::new();
    let result = truncate_scf_if_block(db, &block, &effectful_funcs, step_ty);

    let ops: Vec<_> = result.operations(db).iter().copied().collect();

    // Expect 3 ops: arith.const, trampoline.step_done, scf.yield
    if ops.len() != 3 {
        return Err(format!(
            "Expected 3 ops (const + step_done + yield), got {}",
            ops.len()
        ));
    }

    // First op should be the original arith.const
    if arith::Const::from_operation(db, ops[0]).is_err() {
        return Err("ops[0] should be arith.const".to_string());
    }

    // Second op should be trampoline.step_done
    let step_done = trampoline::StepDone::from_operation(db, ops[1])
        .map_err(|_| "ops[1] should be trampoline.step_done".to_string())?;
    let step_done_result_ty = step_done.as_operation().results(db);
    if step_done_result_ty.first() != Some(&step_ty) {
        return Err("step_done result type should be Step".to_string());
    }

    // Third op should be scf.yield whose operand is the step_done result
    let yield_op = scf::Yield::from_operation(db, ops[2])
        .map_err(|_| "ops[2] should be scf.yield".to_string())?;
    let yield_operands = yield_op.as_operation().operands(db);
    if yield_operands.len() != 1 {
        return Err(format!(
            "scf.yield should have 1 operand, got {}",
            yield_operands.len()
        ));
    }
    if yield_operands[0] != step_done.result(db) {
        return Err("scf.yield operand should be the step_done result".to_string());
    }

    Ok(())
}

#[salsa_test]
fn test_truncate_scf_if_block_pure_branch(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_truncate_scf_if_block_pure_branch(db) {
        panic!("{}", msg);
    }
}

// ========================================================================
// Test: truncate_scf_if_block — effectful branch
// ========================================================================

/// Effectful branch: call to an effectful function gets its result type
/// changed to Step, and ops after the call are truncated.
#[salsa::tracked]
fn run_truncate_scf_if_block_effectful_branch(db: &dyn salsa::Database) -> Result<(), String> {
    let location = test_location(db);
    let i32_ty = core::I32::new(db).as_type();
    let step_ty = trampoline::Step::new(db).as_type();
    let callee = Symbol::new("effectful_fn");

    // Build: func.call @effectful_fn() -> i32 ; arith.const 99 ; scf.yield(%call)
    let mut builder = BlockBuilder::new(db, location);
    let call = builder.op(func::call(db, location, vec![], i32_ty, callee));
    let _dead = builder.op(arith::Const::i32(db, location, 99));
    builder.op(scf::r#yield(db, location, vec![call.result(db)]));
    let block = builder.build();

    let mut effectful_funcs: HashSet<Symbol> = HashSet::new();
    effectful_funcs.insert(callee);
    let result = truncate_scf_if_block(db, &block, &effectful_funcs, step_ty);

    let ops: Vec<_> = result.operations(db).iter().copied().collect();

    // Expect 2 ops: func.call (with Step result) + scf.yield
    if ops.len() != 2 {
        return Err(format!("Expected 2 ops (call + yield), got {}", ops.len()));
    }

    // First op: func.call with Step result type
    let new_call = func::Call::from_operation(db, ops[0])
        .map_err(|_| "ops[0] should be func.call".to_string())?;
    let call_results = new_call.as_operation().results(db);
    if call_results.first() != Some(&step_ty) {
        return Err(format!(
            "call result should be Step, got {:?}",
            call_results.first()
        ));
    }

    // Second op: scf.yield whose operand is the call's Step result
    let yield_op = scf::Yield::from_operation(db, ops[1])
        .map_err(|_| "ops[1] should be scf.yield".to_string())?;
    let yield_operands = yield_op.as_operation().operands(db);
    if yield_operands.len() != 1 {
        return Err(format!(
            "scf.yield should have 1 operand, got {}",
            yield_operands.len()
        ));
    }
    if yield_operands[0] != new_call.as_operation().result(db, 0) {
        return Err("scf.yield operand should be the call's Step result".to_string());
    }

    Ok(())
}

#[salsa_test]
fn test_truncate_scf_if_block_effectful_branch(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_truncate_scf_if_block_effectful_branch(db) {
        panic!("{}", msg);
    }
}

// ========================================================================
// Test: truncate_scf_if_branch — mixed branches (integration)
// ========================================================================

/// Integration test: scf.if with effectful then-branch and pure else-branch.
/// Both branches should produce Step-typed yield values after processing.
#[salsa::tracked]
fn run_truncate_scf_if_block_mixed_branches(db: &dyn salsa::Database) -> Result<(), String> {
    let location = test_location(db);
    let i32_ty = core::I32::new(db).as_type();
    let step_ty = trampoline::Step::new(db).as_type();
    let callee = Symbol::new("effectful_fn");

    let mut effectful_funcs: HashSet<Symbol> = HashSet::new();
    effectful_funcs.insert(callee);

    // Then branch (effectful): func.call @effectful_fn() ; scf.yield(%call)
    let mut then_builder = BlockBuilder::new(db, location);
    let call = then_builder.op(func::call(db, location, vec![], i32_ty, callee));
    then_builder.op(scf::r#yield(db, location, vec![call.result(db)]));
    let then_block = then_builder.build();
    let then_region = Region::new(db, location, IdVec::from(vec![then_block]));

    // Else branch (pure): arith.const 7 ; scf.yield(%const)
    let mut else_builder = BlockBuilder::new(db, location);
    let c = else_builder.op(arith::Const::i32(db, location, 7));
    else_builder.op(scf::r#yield(db, location, vec![c.result(db)]));
    let else_block = else_builder.build();
    let else_region = Region::new(db, location, IdVec::from(vec![else_block]));

    // Process both branches
    let new_then = truncate_scf_if_branch(db, &then_region, &effectful_funcs, step_ty);
    let new_else = truncate_scf_if_branch(db, &else_region, &effectful_funcs, step_ty);

    // --- Verify then branch ---
    let then_ops: Vec<_> = new_then.blocks(db)[0]
        .operations(db)
        .iter()
        .copied()
        .collect();

    // Then: call (Step) + yield
    if then_ops.len() != 2 {
        return Err(format!(
            "Then branch: expected 2 ops, got {}",
            then_ops.len()
        ));
    }
    let then_call = func::Call::from_operation(db, then_ops[0])
        .map_err(|_| "Then ops[0] should be func.call".to_string())?;
    if then_call.as_operation().results(db).first() != Some(&step_ty) {
        return Err("Then call should have Step result".to_string());
    }
    let then_yield = scf::Yield::from_operation(db, then_ops[1])
        .map_err(|_| "Then ops[1] should be scf.yield".to_string())?;
    // The yield operand should be the call result (which has Step type)
    if then_yield.as_operation().operands(db)[0] != then_call.as_operation().result(db, 0) {
        return Err("Then yield operand should be the call's Step result".to_string());
    }

    // --- Verify else branch ---
    let else_ops: Vec<_> = new_else.blocks(db)[0]
        .operations(db)
        .iter()
        .copied()
        .collect();

    // Else: const + step_done + yield
    if else_ops.len() != 3 {
        return Err(format!(
            "Else branch: expected 3 ops, got {}",
            else_ops.len()
        ));
    }
    if arith::Const::from_operation(db, else_ops[0]).is_err() {
        return Err("Else ops[0] should be arith.const".to_string());
    }
    let else_step_done = trampoline::StepDone::from_operation(db, else_ops[1])
        .map_err(|_| "Else ops[1] should be trampoline.step_done".to_string())?;
    let else_yield = scf::Yield::from_operation(db, else_ops[2])
        .map_err(|_| "Else ops[2] should be scf.yield".to_string())?;
    // The yield operand should be the step_done result (which has Step type)
    if else_yield.as_operation().operands(db)[0] != else_step_done.result(db) {
        return Err("Else yield operand should be the step_done result".to_string());
    }
    // Verify step_done result type is Step
    if else_step_done.as_operation().results(db).first() != Some(&step_ty) {
        return Err("step_done result type should be Step".to_string());
    }

    Ok(())
}

#[salsa_test]
fn test_truncate_scf_if_block_mixed_branches(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_truncate_scf_if_block_mixed_branches(db) {
        panic!("{}", msg);
    }
}

// ========================================================================
// Test: push_prompt body does not propagate effectfulness
// ========================================================================

/// A function that only calls effectful functions inside a push_prompt body
/// should NOT be identified as effectful itself, because push_prompt bodies
/// are handled by the enclosing handler.
#[salsa::tracked]
fn run_push_prompt_does_not_propagate_effectfulness(
    db: &dyn salsa::Database,
) -> Result<(), String> {
    let location = test_location(db);
    let i32_ty = core::I32::new(db).as_type();

    // Create an effectful function (has non-empty effect row in its type)
    let ability_ty = core::AbilityRefType::simple(db, Symbol::new("State")).as_type();
    let effect_row = core::EffectRowType::concrete(db, IdVec::from(vec![ability_ty]));
    let effectful_fn = func::Func::build_with_effect(
        db,
        location,
        "effectful_fn",
        IdVec::new(),
        i32_ty,
        Some(effect_row.as_type()),
        |entry| {
            let c = entry.op(arith::Const::i32(db, location, 42));
            entry.op(func::Return::value(db, location, c.result(db)));
        },
    );

    // Create a "caller" function that calls effectful_fn ONLY inside a push_prompt body.
    // This function should NOT be marked as effectful.
    let caller_fn = func::Func::build(db, location, "caller_fn", IdVec::new(), i32_ty, |entry| {
        // Build push_prompt body region that calls effectful_fn
        let mut body_builder = BlockBuilder::new(db, location);
        let call = body_builder.op(func::call(
            db,
            location,
            vec![],
            i32_ty,
            Symbol::new("effectful_fn"),
        ));
        body_builder.op(func::Return::value(db, location, call.result(db)));
        let body_region = Region::new(db, location, IdVec::from(vec![body_builder.build()]));
        let handlers_region = Region::new(db, location, IdVec::new());

        let pp = entry.op(cont::push_prompt(
            db,
            location,
            i32_ty,
            0,
            body_region,
            handlers_region,
        ));
        entry.op(func::Return::value(db, location, pp.result(db)));
    });

    // Build module with both functions
    let module = core::Module::build(db, location, "test".into(), |top| {
        top.op(effectful_fn);
        top.op(caller_fn);
    });

    let effectful = identify_effectful_functions(db, &module);

    // effectful_fn should be in the set (it has an effect row)
    if !effectful.contains(&Symbol::new("effectful_fn")) {
        return Err("effectful_fn should be identified as effectful".to_string());
    }

    // caller_fn should NOT be in the set — its call to effectful_fn is inside push_prompt
    if effectful.contains(&Symbol::new("caller_fn")) {
        return Err(
            "caller_fn should NOT be effectful: its effectful call is inside push_prompt"
                .to_string(),
        );
    }

    Ok(())
}

#[salsa_test]
fn test_push_prompt_does_not_propagate_effectfulness(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_push_prompt_does_not_propagate_effectfulness(db) {
        panic!("{}", msg);
    }
}

// ========================================================================
// Test: build_arm_region emits unreachable for empty arm
// ========================================================================

/// When an arm block has no operations (and thus no result to yield),
/// build_arm_region should emit a func.unreachable as a defensive terminator.
#[salsa::tracked]
fn run_build_arm_region_empty_arm_has_terminator(db: &dyn salsa::Database) -> Result<(), String> {
    let location = test_location(db);

    // Create an empty arm block (no operations, no results)
    let empty_block = Block::new(db, BlockId::fresh(), location, IdVec::new(), IdVec::new());

    let effectful_funcs = HashSet::new();
    let region = build_arm_region(db, location, &empty_block, &effectful_funcs);

    let blocks = region.blocks(db);
    if blocks.is_empty() {
        return Err("build_arm_region should produce at least one block".to_string());
    }

    let ops: Vec<_> = blocks[0].operations(db).iter().copied().collect();
    if ops.is_empty() {
        return Err("arm region block should have at least one operation (terminator)".to_string());
    }

    // The last op should be func.unreachable (defensive fallback)
    let last_op = ops.last().unwrap();
    if func::Unreachable::from_operation(db, *last_op).is_err() {
        return Err(format!(
            "last op should be func.unreachable, got {}.{}",
            last_op.dialect(db),
            last_op.name(db),
        ));
    }

    Ok(())
}

#[salsa_test]
fn test_build_arm_region_empty_arm_has_terminator(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_build_arm_region_empty_arm_has_terminator(db) {
        panic!("{}", msg);
    }
}

// ========================================================================
// Test: truncate_scf_if_block — no yield defensive fallback
// ========================================================================

/// Defensive fallback: a block with no scf.yield and no effectful calls
/// should produce func.unreachable as a terminator (malformed IR recovery).
#[salsa::tracked]
fn run_truncate_scf_if_block_no_yield_unreachable(db: &dyn salsa::Database) -> Result<(), String> {
    let location = test_location(db);
    let step_ty = trampoline::Step::new(db).as_type();

    // Build a block with only arith.const — no scf.yield, no effectful calls
    let mut builder = BlockBuilder::new(db, location);
    builder.op(arith::Const::i32(db, location, 42));
    let block = builder.build();

    let effectful_funcs: HashSet<Symbol> = HashSet::new();
    let result = truncate_scf_if_block(db, &block, &effectful_funcs, step_ty);

    let ops: Vec<_> = result.operations(db).iter().copied().collect();

    // Expect 2 ops: arith.const + func.unreachable
    if ops.len() != 2 {
        return Err(format!(
            "Expected 2 ops (const + unreachable), got {}",
            ops.len()
        ));
    }

    if arith::Const::from_operation(db, ops[0]).is_err() {
        return Err("ops[0] should be arith.const".to_string());
    }

    if func::Unreachable::from_operation(db, ops[1]).is_err() {
        return Err(format!(
            "ops[1] should be func.unreachable, got {}.{}",
            ops[1].dialect(db),
            ops[1].name(db),
        ));
    }

    Ok(())
}

#[salsa_test]
fn test_truncate_scf_if_block_no_yield_unreachable(db: &salsa::DatabaseImpl) {
    if let Err(msg) = run_truncate_scf_if_block_no_yield_unreachable(db) {
        panic!("{}", msg);
    }
}

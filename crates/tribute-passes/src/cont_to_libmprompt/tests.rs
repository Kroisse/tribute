//! Tests for cont_to_libmprompt pass.

use salsa_test_macros::salsa_test;
use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::scf;
use trunk_ir::parser::parse_test_module;
use trunk_ir::{DialectOp, Operation, Region, Type, Value};

use super::lower_cont_to_libmprompt;
use super::push_prompt::compute_live_ins;

// ============================================================================
// FFI declarations
// ============================================================================

#[salsa::tracked]
fn run_ffi_declarations_test(db: &dyn salsa::Database) -> Result<(), String> {
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @_placeholder() -> core.nil {
    func.return
  }
}"#,
    );

    let module = super::ffi::ensure_libmprompt_ffi(db, module);

    // Verify all FFI functions are declared
    let body = module.body(db);
    let blocks = body.blocks(db);
    let entry = blocks.first().ok_or("no entry block")?;

    let mut declared: Vec<String> = Vec::new();
    for op in entry.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            declared.push(func_op.sym_name(db).to_string());
        }
    }

    for name in super::ffi::FFI_NAMES {
        if !declared.contains(&name.to_string()) {
            return Err(format!("Missing FFI declaration: {name}"));
        }
    }

    // Verify idempotency: count should not change on second call
    let func_count_before = entry
        .operations(db)
        .iter()
        .filter(|op| func::Func::from_operation(db, **op).is_ok())
        .count();

    let module2 = super::ffi::ensure_libmprompt_ffi(db, module);
    let body2 = module2.body(db);
    let blocks2 = body2.blocks(db);
    let entry2 = blocks2.first().ok_or("no entry block after second call")?;
    let func_count_after = entry2
        .operations(db)
        .iter()
        .filter(|op| func::Func::from_operation(db, **op).is_ok())
        .count();

    if func_count_after != func_count_before {
        return Err(format!(
            "Idempotency failed: expected {func_count_before} funcs, got {func_count_after}",
        ));
    }

    Ok(())
}

#[salsa_test]
fn test_ffi_declarations(db: &salsa::DatabaseImpl) {
    let result = run_ffi_declarations_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Shift lowering
// ============================================================================

#[salsa::tracked]
fn run_shift_lowering_test(db: &dyn salsa::Database) -> Result<(), String> {
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @test_fn() -> core.ptr {
    %tag = arith.const {value = 42} : core.i32
    %val = arith.const {value = 1} : core.i32
    %val_ptr = core.unrealized_conversion_cast %val : core.ptr
    %result = cont.shift %tag, %val_ptr {ability_ref = core.ability_ref() {name = @State}, op_name = @get} : core.ptr {
    }
    func.return %result
  }
}"#,
    );

    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            if !find_call_in_module(db, &lowered, "__tribute_yield") {
                return Err("Expected func.call @__tribute_yield after lowering".into());
            }
            Ok(())
        }
        Err(e) => Err(format!("Lowering failed: {e:?}")),
    }
}

#[salsa_test]
fn test_lower_shift_basic(db: &salsa::DatabaseImpl) {
    let result = run_shift_lowering_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Resume lowering
// ============================================================================

#[salsa::tracked]
fn run_resume_lowering_test(db: &dyn salsa::Database) -> Result<(), String> {
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @test_resume(%k: core.ptr, %val: core.ptr) -> core.ptr {
    %result = cont.resume %k, %val : core.ptr
    func.return %result
  }
}"#,
    );

    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            if !find_call_in_module(db, &lowered, "__tribute_resume") {
                return Err("Expected func.call @__tribute_resume after lowering".into());
            }
            Ok(())
        }
        Err(e) => Err(format!("Lowering failed: {e:?}")),
    }
}

#[salsa_test]
fn test_lower_resume_basic(db: &salsa::DatabaseImpl) {
    let result = run_resume_lowering_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Drop lowering
// ============================================================================

#[salsa::tracked]
fn run_drop_lowering_test(db: &dyn salsa::Database) -> Result<(), String> {
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @test_drop(%k: core.ptr) -> core.nil {
    cont.drop %k
    func.return
  }
}"#,
    );

    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            if !find_call_in_module(db, &lowered, "__tribute_resume_drop") {
                return Err("Expected func.call @__tribute_resume_drop after lowering".into());
            }
            Ok(())
        }
        Err(e) => Err(format!("Lowering failed: {e:?}")),
    }
}

#[salsa_test]
fn test_lower_drop_basic(db: &salsa::DatabaseImpl) {
    let result = run_drop_lowering_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Helper: find func.call in module
// ============================================================================

fn find_call_in_module<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
    callee_name: &str,
) -> bool {
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if find_call_in_op(db, op, callee_name) {
                return true;
            }
        }
    }
    false
}

fn find_call_in_op<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    callee_name: &str,
) -> bool {
    // Check if this op is the target call
    if let Ok(call) = func::Call::from_operation(db, *op)
        && call.callee(db) == callee_name
    {
        return true;
    }

    // Check nested regions
    for region in op.regions(db).iter() {
        for block in region.blocks(db).iter() {
            for nested_op in block.operations(db).iter() {
                if find_call_in_op(db, nested_op, callee_name) {
                    return true;
                }
            }
        }
    }

    false
}

// ============================================================================
// Live-in analysis: nested region definitions excluded
// ============================================================================

/// Helper: find the first scf.if operation inside a function, return its regions.
fn find_scf_if_regions<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
) -> Option<(Region<'db>, Region<'db>)> {
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if let Ok(func_op) = func::Func::from_operation(db, *op) {
                let func_body = func_op.body(db);
                for fblock in func_body.blocks(db).iter() {
                    for fop in fblock.operations(db).iter() {
                        if let Ok(if_op) = scf::If::from_operation(db, *fop) {
                            return Some((if_op.then(db), if_op.r#else(db)));
                        }
                    }
                }
            }
        }
    }
    None
}

#[salsa::tracked]
fn run_live_ins_excludes_nested_defs_test(db: &dyn salsa::Database) -> Result<(), String> {
    // Parse a module with nested scf.if inside the then-branch.
    // The then-branch uses %x (func param, defined outside) and has
    // nested regions with inner-defined values that must NOT appear as live-ins.
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @test_fn(%x: core.i32) -> core.i32 {
    %cond = arith.const {value = true} : core.i1
    %r = scf.if %cond : core.i32 {
      %inner = arith.const {value = 42} : core.i32
      scf.yield %inner
    } {
      scf.yield %x
    }
    func.return %r
  }
}"#,
    );

    // Extract the else-region of scf.if: it uses %x (defined in func, not in the region)
    let (_, else_region) = find_scf_if_regions(db, &module).ok_or("Could not find scf.if")?;

    let no_lookup = &(|_: &dyn salsa::Database, _: Value<'_>| -> Option<Type<'_>> { None });
    let live_ins = compute_live_ins(db, &else_region, no_lookup);

    // %x should be the only live-in (used in else branch via scf.yield)
    if live_ins.len() != 1 {
        return Err(format!(
            "Expected exactly 1 live-in (%x), got {}",
            live_ins.len()
        ));
    }

    // Now test that the THEN-region has NO live-ins
    // (all values used inside are also defined inside)
    let (then_region, _) = find_scf_if_regions(db, &module).ok_or("Could not find scf.if")?;
    let then_live_ins = compute_live_ins(db, &then_region, no_lookup);
    if !then_live_ins.is_empty() {
        return Err(format!(
            "Expected 0 live-ins in then-region, got {}",
            then_live_ins.len()
        ));
    }

    Ok(())
}

#[salsa_test]
fn test_live_ins_excludes_nested_defs(db: &salsa::DatabaseImpl) {
    let result = run_live_ins_excludes_nested_defs_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Live-in analysis: nested scf.if defs not leaked
// ============================================================================

#[salsa::tracked]
fn run_live_ins_nested_defs_not_leaked_test(db: &dyn salsa::Database) -> Result<(), String> {
    // Parse a module where the then-branch has a NESTED scf.if.
    // Without the recursive definition collection fix, values defined
    // inside the inner scf.if would incorrectly appear as live-ins of
    // the outer then-region.
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @test_fn(%x: core.i32) -> core.i32 {
    %cond = arith.const {value = true} : core.i1
    %outer_r = scf.if %cond : core.i32 {
      %inner_r = scf.if %cond : core.i32 {
        %deep = arith.const {value = 99} : core.i32
        scf.yield %deep
      } {
        scf.yield %x
      }
      scf.yield %inner_r
    } {
      scf.yield %x
    }
    func.return %outer_r
  }
}"#,
    );

    // Extract the outer then-region (contains nested scf.if)
    let (outer_then, _) = find_scf_if_regions(db, &module).ok_or("Could not find outer scf.if")?;

    let no_lookup = &(|_: &dyn salsa::Database, _: Value<'_>| -> Option<Type<'_>> { None });
    let live_ins = compute_live_ins(db, &outer_then, no_lookup);

    // Live-ins should be: %cond (scf.if condition) and %x (used in inner else).
    // %deep and other inner-defined values must NOT appear.
    if live_ins.len() != 2 {
        return Err(format!(
            "Expected 2 live-ins (%cond, %x), got {}",
            live_ins.len()
        ));
    }

    Ok(())
}

#[salsa_test]
fn test_live_ins_nested_defs_not_leaked(db: &salsa::DatabaseImpl) {
    let result = run_live_ins_nested_defs_not_leaked_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Handler dispatch: done-only (no suspend arms, triggers unreachable)
// ============================================================================

#[salsa::tracked]
fn run_handler_dispatch_done_only_test(db: &dyn salsa::Database) -> Result<(), String> {
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @test_done_only() -> core.i32 {
    %0 = cont.push_prompt {tag = 1} : core.i32 {
      %1 = arith.const {value = 42} : core.i32
      scf.yield %1
    } {
    }
    %2 = cont.handler_dispatch %0 {tag = 1, result_type = core.i32} : core.i32 {
      cont.done {
        ^bb0(%done_val: core.i32):
          scf.yield %done_val
      }
    }
    func.return %2
  }
}"#,
    );

    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            // Verify __tribute_prompt call exists (from push_prompt)
            if !find_call_in_module(db, &lowered, "__tribute_prompt") {
                return Err("Expected __tribute_prompt call".into());
            }
            // Verify __tribute_yield_active call exists (from handler_dispatch)
            if !find_call_in_module(db, &lowered, "__tribute_yield_active") {
                return Err("Expected __tribute_yield_active call".into());
            }
            Ok(())
        }
        Err(e) => Err(format!("Lowering failed: {e:?}")),
    }
}

#[salsa_test]
fn test_handler_dispatch_done_only(db: &salsa::DatabaseImpl) {
    let result = run_handler_dispatch_done_only_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Handler dispatch: single suspend arm
// ============================================================================

#[salsa::tracked]
fn run_handler_dispatch_single_suspend_arm_test(db: &dyn salsa::Database) -> Result<(), String> {
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @__tribute_resume(%k: core.ptr, %sv: core.ptr) -> core.ptr {
    func.return %k
  }
  func.func @test_single_arm() -> core.i32 {
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c = arith.const {value = 10} : core.i32
      scf.yield %c
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.i32} : core.i32 {
      cont.done {
        ^bb0(%v: core.i32):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = func.call %k, %sv {callee = @__tribute_resume} : core.ptr
          scf.yield %r
      }
    }
    func.return %1
  }
}"#,
    );

    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            if !find_call_in_module(db, &lowered, "__tribute_yield_active") {
                return Err("Expected __tribute_yield_active call (dispatch loop)".into());
            }
            if !find_call_in_module(db, &lowered, "__tribute_get_yield_op_idx") {
                return Err("Expected __tribute_get_yield_op_idx call (shift branch)".into());
            }
            if !find_call_in_module(db, &lowered, "__tribute_resume") {
                return Err("Expected __tribute_resume call (suspend arm body)".into());
            }
            Ok(())
        }
        Err(e) => Err(format!("Lowering failed: {e:?}")),
    }
}

#[salsa_test]
fn test_handler_dispatch_single_suspend_arm(db: &salsa::DatabaseImpl) {
    let result = run_handler_dispatch_single_suspend_arm_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Handler dispatch: multiple suspend arms
// ============================================================================

#[salsa::tracked]
fn run_handler_dispatch_multi_suspend_arms_test(db: &dyn salsa::Database) -> Result<(), String> {
    let module = parse_test_module(
        db,
        r#"core.module @test {
  func.func @__tribute_resume(%k: core.ptr, %sv: core.ptr) -> core.ptr {
    func.return %k
  }
  func.func @test_multi_arms() -> core.i32 {
    %0 = cont.push_prompt {tag = 1} : core.ptr {
      %c = arith.const {value = 10} : core.i32
      scf.yield %c
    } {
    }
    %1 = cont.handler_dispatch %0 {tag = 1, result_type = core.i32} : core.i32 {
      cont.done {
        ^bb0(%v: core.i32):
          scf.yield %v
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @get} {
        ^bb0(%k: core.ptr, %sv: core.ptr):
          %r = func.call %k, %sv {callee = @__tribute_resume} : core.ptr
          scf.yield %r
      }
      cont.suspend {ability_ref = core.ability_ref() {name = @State}, op_name = @set} {
        ^bb0(%k2: core.ptr, %sv2: core.ptr):
          %r2 = func.call %k2, %sv2 {callee = @__tribute_resume} : core.ptr
          scf.yield %r2
      }
    }
    func.return %1
  }
}"#,
    );

    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            if !find_call_in_module(db, &lowered, "__tribute_get_yield_op_idx") {
                return Err(
                    "Expected __tribute_get_yield_op_idx call (dispatch needs op_idx)".into(),
                );
            }
            // With multiple arms, nested if-else dispatch uses arith.cmp_eq
            if !find_op_in_module(db, &lowered, "arith", "cmp_eq") {
                return Err(
                    "Expected arith.cmp_eq (nested if-else dispatch for op_idx comparison)".into(),
                );
            }
            Ok(())
        }
        Err(e) => Err(format!("Lowering failed: {e:?}")),
    }
}

#[salsa_test]
fn test_handler_dispatch_multi_suspend_arms(db: &salsa::DatabaseImpl) {
    let result = run_handler_dispatch_multi_suspend_arms_test(db);
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

// ============================================================================
// Helper: find operation by dialect and name in module
// ============================================================================

fn find_op_in_module<'db>(
    db: &'db dyn salsa::Database,
    module: &core::Module<'db>,
    dialect: &str,
    name: &str,
) -> bool {
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            if find_op_recursive(db, op, dialect, name) {
                return true;
            }
        }
    }
    false
}

fn find_op_recursive<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    dialect: &str,
    name: &str,
) -> bool {
    if op.dialect(db) == dialect && op.name(db) == name {
        return true;
    }
    for region in op.regions(db).iter() {
        for block in region.blocks(db).iter() {
            for nested_op in block.operations(db).iter() {
                if find_op_recursive(db, nested_op, dialect, name) {
                    return true;
                }
            }
        }
    }
    false
}

//! Tests for cont_to_libmprompt pass.

use salsa_test_macros::salsa_test;
use trunk_ir::dialect::core::{self};
use trunk_ir::dialect::func::{self};
use trunk_ir::dialect::{arith, cont};
use trunk_ir::ir::BlockBuilder;
use trunk_ir::{
    Block, BlockArg, BlockId, DialectOp, DialectType, IdVec, Location, Operation, PathId, Region,
    Span, Symbol, Value, ValueDef, idvec,
};

use super::lower_cont_to_libmprompt;

fn test_location(db: &dyn salsa::Database) -> Location<'_> {
    let path = PathId::new(db, "test.trb".to_owned());
    Location::new(path, Span::new(0, 0))
}

// ============================================================================
// FFI declarations
// ============================================================================

#[salsa::tracked]
fn run_ffi_declarations_test(db: &dyn salsa::Database) -> Result<(), String> {
    let loc = test_location(db);
    let body = Region::new(
        db,
        loc,
        idvec![Block::new(
            db,
            BlockId::fresh(),
            loc,
            IdVec::new(),
            IdVec::new()
        )],
    );
    let module = core::Module::create(db, loc, Symbol::new("test"), body);

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

    // Verify idempotency
    let module2 = super::ffi::ensure_libmprompt_ffi(db, module);
    let body2 = module2.body(db);
    let blocks2 = body2.blocks(db);
    let entry2 = blocks2.first().ok_or("no entry block after second call")?;
    let count2 = entry2
        .operations(db)
        .iter()
        .filter(|op| func::Func::from_operation(db, **op).is_ok())
        .count();

    if count2 != super::ffi::FFI_NAMES.len() {
        return Err(format!(
            "Idempotency failed: expected {} funcs, got {count2}",
            super::ffi::FFI_NAMES.len()
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
    let loc = test_location(db);
    let ptr_ty = core::Ptr::new(db).as_type();
    let state_ref_ty = core::AbilityRefType::simple(db, Symbol::new("State")).as_type();

    // Build a function with cont.shift
    let mut func_builder = BlockBuilder::new(db, loc);

    // %tag = arith.const 42
    let tag_const = func_builder.op(arith::Const::i32(db, loc, 42));
    let tag = tag_const.result(db);

    // %val = arith.const 1
    let val_const = func_builder.op(arith::Const::i32(db, loc, 1));

    // Cast val to ptr for shift
    let val_ptr = func_builder.op(core::unrealized_conversion_cast(
        db,
        loc,
        val_const.result(db),
        ptr_ty,
    ));

    // %result = cont.shift(%tag, %val) { ability_ref: State, op_name: get }
    let handler_region = Region::new(db, loc, IdVec::from(Vec::<Block>::new()));
    let shift_op = func_builder.op(cont::shift(
        db,
        loc,
        tag,
        vec![val_ptr.result(db)],
        ptr_ty,
        state_ref_ty,
        Symbol::new("get"),
        None,
        None,
        handler_region,
    ));

    func_builder.op(func::r#return(db, loc, Some(shift_op.result(db))));

    let func_body = Region::new(db, loc, IdVec::from(vec![func_builder.build()]));
    let func_ty = core::Func::new(db, IdVec::from(vec![]), ptr_ty);
    let func_op = func::func(db, loc, Symbol::new("test_fn"), *func_ty, func_body);

    // Build module
    let entry_block = Block::new(
        db,
        BlockId::fresh(),
        loc,
        IdVec::new(),
        idvec![func_op.as_operation()],
    );
    let body = Region::new(db, loc, idvec![entry_block]);
    let module = core::Module::create(db, loc, Symbol::new("test"), body);

    // Lower
    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            // Verify __tribute_yield call exists
            let has_yield_call = find_call_in_module(db, &lowered, "__tribute_yield");
            if !has_yield_call {
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
    let loc = test_location(db);
    let ptr_ty = core::Ptr::new(db).as_type();

    // Build a function with cont.resume using Block::new pattern
    let func_block_id = BlockId::fresh();

    // Use block args as continuation and value
    let cont_val = Value::new(db, ValueDef::BlockArg(func_block_id), 0);
    let val = Value::new(db, ValueDef::BlockArg(func_block_id), 1);

    // Build operations manually
    let resume_op = cont::resume(db, loc, cont_val, val, ptr_ty);
    let ret_op = func::r#return(db, loc, Some(resume_op.result(db)));

    let func_block = Block::new(
        db,
        func_block_id,
        loc,
        IdVec::from(vec![
            BlockArg::of_type(db, ptr_ty),
            BlockArg::of_type(db, ptr_ty),
        ]),
        idvec![resume_op.as_operation(), ret_op.as_operation()],
    );
    let func_body = Region::new(db, loc, IdVec::from(vec![func_block]));
    let func_ty = core::Func::new(db, IdVec::from(vec![ptr_ty, ptr_ty]), ptr_ty);
    let func_op = func::func(db, loc, Symbol::new("test_resume"), *func_ty, func_body);

    let entry_block = Block::new(
        db,
        BlockId::fresh(),
        loc,
        IdVec::new(),
        idvec![func_op.as_operation()],
    );
    let body = Region::new(db, loc, idvec![entry_block]);
    let module = core::Module::create(db, loc, Symbol::new("test"), body);

    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            let has_resume_call = find_call_in_module(db, &lowered, "__tribute_resume");
            if !has_resume_call {
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
    let loc = test_location(db);
    let ptr_ty = core::Ptr::new(db).as_type();
    let nil_ty = core::Nil::new(db).as_type();

    let func_block_id = BlockId::fresh();

    let cont_val = Value::new(db, ValueDef::BlockArg(func_block_id), 0);

    let drop_op = cont::drop(db, loc, cont_val);
    let ret_op = func::r#return(db, loc, None);

    let func_block = Block::new(
        db,
        func_block_id,
        loc,
        IdVec::from(vec![BlockArg::of_type(db, ptr_ty)]),
        idvec![drop_op.as_operation(), ret_op.as_operation()],
    );
    let func_body = Region::new(db, loc, IdVec::from(vec![func_block]));
    let func_ty = core::Func::new(db, IdVec::from(vec![ptr_ty]), nil_ty);
    let func_op = func::func(db, loc, Symbol::new("test_drop"), *func_ty, func_body);

    let entry_block = Block::new(
        db,
        BlockId::fresh(),
        loc,
        IdVec::new(),
        idvec![func_op.as_operation()],
    );
    let body = Region::new(db, loc, idvec![entry_block]);
    let module = core::Module::create(db, loc, Symbol::new("test"), body);

    let result = lower_cont_to_libmprompt(db, module);
    match result {
        Ok(lowered) => {
            let has_drop_call = find_call_in_module(db, &lowered, "__tribute_resume_drop");
            if !has_drop_call {
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

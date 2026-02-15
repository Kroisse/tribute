//! Native entrypoint generation pass.
//!
//! Generates a C ABI `main` entrypoint that wraps the user's `main` function.
//! The user's `main` is renamed to `_tribute_main`, and a new `main` is created
//! that calls it and returns exit code 0.

use trunk_ir::dialect::{arith, core, func};
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol,
};

/// Generate a native C ABI entrypoint wrapper for the user's `main` function.
///
/// This pass:
/// 1. Renames `func.func @main` to `func.func @_tribute_main`
/// 2. Creates a new `func.func @main() -> i32` that calls `_tribute_main()`
///    and returns 0
///
/// This ensures the native binary has a C-compatible `main` entry point
/// while preserving the user's function semantics (which may return `Nil`).
pub fn generate_native_entrypoint<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    let Some(entry_block) = blocks.first() else {
        return module;
    };

    // Find the user's main function and its return type
    let mut found_main = false;
    let mut main_return_ty = None;

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op)
            && func_op.sym_name(db) == Symbol::new("main")
        {
            found_main = true;
            if let Some(func_ty) = core::Func::from_type(db, func_op.r#type(db)) {
                main_return_ty = Some(func_ty.result(db));
            }
        }
    }

    if !found_main {
        tracing::warn!("No main function found; skipping entrypoint generation");
        return module;
    }

    let location = module.location(db);
    let i32_ty = core::I32::new(db).as_type();
    let tribute_main_sym = Symbol::new("_tribute_main");
    let tribute_main_return_ty = main_return_ty.unwrap_or_else(|| core::Nil::new(db).as_type());

    // Rebuild module: rename main -> _tribute_main, then append new entrypoint
    let mut new_ops: Vec<Operation<'db>> = Vec::new();

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op)
            && func_op.sym_name(db) == Symbol::new("main")
        {
            let renamed = rebuild_func_with_name(db, &func_op, tribute_main_sym);
            new_ops.push(renamed);
            continue;
        }
        new_ops.push(*op);
    }

    // Append the C ABI entrypoint
    let entrypoint = build_entrypoint(db, location, tribute_main_return_ty, i32_ty);
    new_ops.push(entrypoint);

    // Rebuild module
    let new_block = Block::new(
        db,
        entry_block.id(db),
        entry_block.location(db),
        entry_block.args(db).clone(),
        new_ops.into_iter().collect(),
    );
    let new_body = Region::new(db, body.location(db), IdVec::from(vec![new_block]));
    core::Module::create(db, module.location(db), module.name(db), new_body)
}

/// Rebuild a `func.func` operation with a new name, preserving all other attributes and body.
fn rebuild_func_with_name<'db>(
    db: &'db dyn salsa::Database,
    func_op: &func::Func<'db>,
    new_name: Symbol,
) -> Operation<'db> {
    let op = func_op.as_operation();
    let location = op.location(db);
    let sym_name_key = Symbol::new("sym_name");

    let mut builder = Operation::of(db, location, func::DIALECT_NAME(), func::FUNC())
        .attr(sym_name_key, Attribute::Symbol(new_name));

    // Copy all attributes except sym_name
    for (key, value) in op.attributes(db).iter() {
        if *key != sym_name_key {
            builder = builder.attr(*key, value.clone());
        }
    }

    // Copy regions (function body)
    for region in op.regions(db).iter() {
        builder = builder.region(*region);
    }

    builder.build()
}

/// Build the C ABI entrypoint function:
///
/// ```text
/// func.func @main() -> i32 {
///     call @_tribute_main()
///     %zero = arith.const 0 : i32
///     return %zero
/// }
/// ```
fn build_entrypoint<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    tribute_main_return_ty: trunk_ir::Type<'db>,
    i32_ty: trunk_ir::Type<'db>,
) -> Operation<'db> {
    func::Func::build(db, location, "main", IdVec::new(), i32_ty, |builder| {
        // Call _tribute_main() — result is ignored
        builder.op(func::call(
            db,
            location,
            vec![],
            tribute_main_return_ty,
            Symbol::new("_tribute_main"),
        ));
        // Return exit code 0
        let zero = builder.op(arith::Const::i32(db, location, 0));
        builder.op(func::r#return(db, location, vec![zero.result(db)]));
    })
    .as_operation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::{BlockId, PathId, Span, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    /// Build a module with a single main function returning i32.
    #[salsa::tracked]
    fn make_main_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let main_fn = func::Func::build(db, location, "main", IdVec::new(), i32_ty, |builder| {
            let c42 = builder.op(arith::Const::i32(db, location, 42));
            builder.op(func::r#return(db, location, vec![c42.result(db)]));
        });

        let ops: IdVec<Operation<'_>> = idvec![main_fn.as_operation()];
        let block = Block::new(db, BlockId::fresh(), location, idvec![], ops);
        let body = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), body)
    }

    /// Apply entrypoint pass to the main module.
    #[salsa::tracked]
    fn apply_entrypoint_to_main(db: &dyn salsa::Database) -> core::Module<'_> {
        let module = make_main_module(db);
        generate_native_entrypoint(db, module)
    }

    #[salsa_test]
    fn test_entrypoint_renames_main(db: &salsa::DatabaseImpl) {
        let result = apply_entrypoint_to_main(db);

        let body = result.body(db);
        let ops = body.blocks(db)[0].operations(db);
        let mut names: Vec<String> = Vec::new();
        for op in ops.iter() {
            if let Ok(f) = func::Func::from_operation(db, *op) {
                names.push(f.sym_name(db).to_string());
            }
        }

        assert!(
            names.contains(&"_tribute_main".to_string()),
            "Expected _tribute_main, got: {:?}",
            names
        );
        assert!(
            names.contains(&"main".to_string()),
            "Expected main wrapper, got: {:?}",
            names
        );
        assert_eq!(
            names.len(),
            2,
            "Expected exactly 2 functions, got: {:?}",
            names
        );
    }

    /// Build a module with no main function.
    #[salsa::tracked]
    fn make_no_main_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let i32_ty = core::I32::new(db).as_type();

        let helper_fn =
            func::Func::build(db, location, "helper", IdVec::new(), i32_ty, |builder| {
                let c1 = builder.op(arith::Const::i32(db, location, 1));
                builder.op(func::r#return(db, location, vec![c1.result(db)]));
            });

        let ops: IdVec<Operation<'_>> = idvec![helper_fn.as_operation()];
        let block = Block::new(db, BlockId::fresh(), location, idvec![], ops);
        let body = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), body)
    }

    /// Apply entrypoint pass to a module without main.
    #[salsa::tracked]
    fn apply_entrypoint_no_main(db: &dyn salsa::Database) -> core::Module<'_> {
        let module = make_no_main_module(db);
        generate_native_entrypoint(db, module)
    }

    #[salsa_test]
    fn test_entrypoint_no_main(db: &salsa::DatabaseImpl) {
        let result = apply_entrypoint_no_main(db);

        let body = result.body(db);
        let ops = body.blocks(db)[0].operations(db);
        assert_eq!(ops.len(), 1);
        if let Ok(f) = func::Func::from_operation(db, ops[0]) {
            assert_eq!(f.sym_name(db).to_string(), "helper");
        }
    }

    /// Build a module with main returning Nil.
    #[salsa::tracked]
    fn make_nil_main_module(db: &dyn salsa::Database) -> core::Module<'_> {
        let location = test_location(db);
        let nil_ty = core::Nil::new(db).as_type();

        let main_fn = func::Func::build(db, location, "main", IdVec::new(), nil_ty, |builder| {
            builder.op(func::Return::empty(db, location));
        });

        let ops: IdVec<Operation<'_>> = idvec![main_fn.as_operation()];
        let block = Block::new(db, BlockId::fresh(), location, idvec![], ops);
        let body = Region::new(db, location, idvec![block]);
        core::Module::create(db, location, "test".into(), body)
    }

    /// Apply entrypoint pass to nil-returning main.
    #[salsa::tracked]
    fn apply_entrypoint_nil_main(db: &dyn salsa::Database) -> core::Module<'_> {
        let module = make_nil_main_module(db);
        generate_native_entrypoint(db, module)
    }

    #[salsa_test]
    fn test_entrypoint_wrapper_returns_i32(db: &salsa::DatabaseImpl) {
        let result = apply_entrypoint_nil_main(db);
        let i32_ty = core::I32::new(db).as_type();

        let body = result.body(db);
        for op in body.blocks(db)[0].operations(db).iter() {
            if let Ok(f) = func::Func::from_operation(db, *op)
                && f.sym_name(db) == Symbol::new("main")
            {
                let func_ty = core::Func::from_type(db, f.r#type(db)).unwrap();
                assert_eq!(func_ty.result(db), i32_ty);
            }
        }
    }
}

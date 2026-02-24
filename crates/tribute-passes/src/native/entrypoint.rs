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
    sanitize: bool,
) -> core::Module<'db> {
    let body = module.body(db);
    let blocks = body.blocks(db);

    assert!(
        blocks.len() == 1,
        "Module body must be single-block, got {} blocks",
        blocks.len()
    );

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

    let main_sym = Symbol::new("main");

    // Rebuild module: rename main -> _tribute_main, rewrite call sites, then append new entrypoint
    let mut new_ops: Vec<Operation<'db>> = Vec::new();
    let mut has_tribute_init = false;
    let mut has_asan_init = false;
    let init_sym = Symbol::new("__tribute_init");
    let asan_init_sym = Symbol::new("__asan_init");

    for op in entry_block.operations(db).iter() {
        if let Ok(func_op) = func::Func::from_operation(db, *op) {
            if func_op.sym_name(db) == init_sym {
                has_tribute_init = true;
            }
            if func_op.sym_name(db) == asan_init_sym {
                has_asan_init = true;
            }
            if func_op.sym_name(db) == main_sym {
                let renamed = rebuild_func_with_name(db, &func_op, tribute_main_sym);
                // Also rewrite any call @main inside this function's body to @_tribute_main
                let rewritten = rewrite_symbol_refs(db, &renamed, main_sym, tribute_main_sym);
                new_ops.push(rewritten);
                continue;
            }
        }
        // Rewrite call @main in other functions too
        let rewritten = rewrite_symbol_refs(db, op, main_sym, tribute_main_sym);
        new_ops.push(rewritten);
    }

    // Ensure __tribute_init is declared (may already exist from cont_to_libmprompt FFI pass)
    if !has_tribute_init {
        let nil_ty = core::Nil::new(db).as_type();
        let init_decl = func::Func::build_extern(
            db,
            location,
            "__tribute_init",
            None,
            [],
            nil_ty,
            None,
            Some("C"),
        );
        new_ops.insert(0, init_decl.as_operation());
    }

    // Declare __asan_init when AddressSanitizer is enabled
    if sanitize && !has_asan_init {
        let nil_ty = core::Nil::new(db).as_type();
        let asan_init_decl = func::Func::build_extern(
            db,
            location,
            "__asan_init",
            None,
            [],
            nil_ty,
            None,
            Some("C"),
        );
        new_ops.insert(0, asan_init_decl.as_operation());
    }

    // Append the C ABI entrypoint
    let entrypoint = build_entrypoint(db, location, tribute_main_return_ty, i32_ty, sanitize);
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

/// Recursively rewrite symbol references in an operation's attributes and regions.
///
/// Rewrites `func.call`, `func.tail_call` callee attributes and `func.constant` func_ref
/// attributes from `old_sym` to `new_sym`.
fn rewrite_symbol_refs<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    old_sym: Symbol,
    new_sym: Symbol,
) -> Operation<'db> {
    let mut attrs_changed = false;
    let mut new_attrs = op.attributes(db).clone();

    // Rewrite callee attribute on func.call / func.tail_call
    let callee_key = Symbol::new("callee");
    if let Some(Attribute::Symbol(sym)) = new_attrs.get(&callee_key)
        && *sym == old_sym
    {
        new_attrs.insert(callee_key, Attribute::Symbol(new_sym));
        attrs_changed = true;
    }

    // Rewrite func_ref attribute on func.constant
    let func_ref_key = Symbol::new("func_ref");
    if let Some(Attribute::Symbol(sym)) = new_attrs.get(&func_ref_key)
        && *sym == old_sym
    {
        new_attrs.insert(func_ref_key, Attribute::Symbol(new_sym));
        attrs_changed = true;
    }

    // Recurse into regions
    let regions = op.regions(db);
    let new_regions: IdVec<Region<'db>> = regions
        .iter()
        .map(|r| rewrite_symbol_refs_in_region(db, r, old_sym, new_sym))
        .collect();
    let regions_changed = new_regions != *regions;

    if !attrs_changed && !regions_changed {
        return *op;
    }

    Operation::new(
        db,
        op.location(db),
        op.dialect(db),
        op.name(db),
        op.operands(db).clone(),
        op.results(db).clone(),
        new_attrs,
        if regions_changed {
            new_regions
        } else {
            regions.clone()
        },
        op.successors(db).clone(),
    )
}

fn rewrite_symbol_refs_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
    old_sym: Symbol,
    new_sym: Symbol,
) -> Region<'db> {
    let blocks = region.blocks(db);
    let new_blocks: IdVec<Block<'db>> = blocks
        .iter()
        .map(|block| {
            let new_ops: IdVec<Operation<'db>> = block
                .operations(db)
                .iter()
                .map(|op| rewrite_symbol_refs(db, op, old_sym, new_sym))
                .collect();
            if new_ops == *block.operations(db) {
                return *block;
            }
            Block::new(
                db,
                block.id(db),
                block.location(db),
                block.args(db).clone(),
                new_ops,
            )
        })
        .collect();
    if new_blocks == *blocks {
        return *region;
    }
    Region::new(db, region.location(db), new_blocks)
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
    sanitize: bool,
) -> Operation<'db> {
    let nil_ty = core::Nil::new(db).as_type();

    func::Func::build(db, location, "main", IdVec::new(), i32_ty, |builder| {
        // Initialize ASan before anything else (must precede libmprompt's SIGSEGV handler)
        if sanitize {
            builder.op(func::call(
                db,
                location,
                vec![],
                nil_ty,
                Symbol::new("__asan_init"),
            ));
        }
        // Initialize libmprompt runtime before any ability use
        builder.op(func::call(
            db,
            location,
            vec![],
            nil_ty,
            Symbol::new("__tribute_init"),
        ));
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
        generate_native_entrypoint(db, module, false)
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
            names.contains(&"__tribute_init".to_string()),
            "Expected __tribute_init, got: {:?}",
            names
        );
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
            3,
            "Expected exactly 3 functions, got: {:?}",
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
        generate_native_entrypoint(db, module, false)
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
        generate_native_entrypoint(db, module, false)
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

//! Salsa IR ↔ Arena IR bidirectional conversion.
//!
//! Provides [`import_salsa_module`] to convert a Salsa `Operation<'db>` (module)
//! into an arena-based `(IrContext, ArenaModule)`, and [`export_to_salsa`] to
//! convert back. Together they enable arena-based passes to be spliced into
//! the existing Salsa pipeline.
//!
//! # Design
//!
//! Both directions use a 2-pass approach for block handling:
//! - **Pass 1**: Allocate all blocks (with args but no ops) so that forward
//!   references to successor blocks can be resolved.
//! - **Pass 2**: Convert operations (which may reference successor blocks and
//!   values defined in earlier ops).

use std::collections::{BTreeMap, HashMap};

use smallvec::SmallVec;

use super::context::{BlockArgData, BlockData, IrContext, OperationDataBuilder, RegionData};
use super::refs::{BlockRef, OpRef, PathRef, RegionRef, TypeRef, ValueRef};
use super::rewrite::ArenaModule;
use super::types as arena_types;
use crate::ir::{self, BlockId, Symbol};
use crate::{Attribute, IdVec, Location, PathId, Type};

// ============================================================================
// Public API
// ============================================================================

/// Import a Salsa module operation into a fresh arena `IrContext`.
///
/// Returns `(ctx, arena_module)` where `ctx` owns all arena entities and
/// `arena_module` is a thin wrapper around the root `core.module` operation.
///
/// # Panics
///
/// Panics if the given operation is not a `core.module`.
pub fn import_salsa_module<'db>(
    db: &'db dyn salsa::Database,
    module: ir::Operation<'db>,
) -> (IrContext, ArenaModule) {
    assert!(
        module.dialect(db) == Symbol::new("core") && module.name(db) == Symbol::new("module"),
        "import_salsa_module: expected core.module, got {}.{}",
        module.dialect(db),
        module.name(db),
    );

    let mut converter = SalsaToArena::new(db);
    let op = converter.convert_operation(module);
    let ctx = converter.ctx;
    let arena_module =
        ArenaModule::new(&ctx, op).expect("converted operation should be core.module");
    (ctx, arena_module)
}

/// Export an arena module back to a Salsa `Operation<'db>`.
pub fn export_to_salsa<'db>(
    db: &'db dyn salsa::Database,
    ctx: &IrContext,
    module: ArenaModule,
) -> ir::Operation<'db> {
    let mut converter = ArenaToSalsa::new(ctx, db);
    converter.convert_operation(module.op())
}

// ============================================================================
// Salsa → Arena
// ============================================================================

struct SalsaToArena<'db> {
    db: &'db dyn salsa::Database,
    ctx: IrContext,
    value_map: HashMap<ir::Value<'db>, ValueRef>,
    /// Maps Salsa BlockId → Arena BlockRef.
    block_map: HashMap<BlockId, BlockRef>,
    type_cache: HashMap<Type<'db>, TypeRef>,
    path_cache: HashMap<PathId<'db>, PathRef>,
}

impl<'db> SalsaToArena<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            ctx: IrContext::new(),
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            type_cache: HashMap::new(),
            path_cache: HashMap::new(),
        }
    }

    // ---- Leaf conversions ----

    fn convert_path(&mut self, path: PathId<'db>) -> PathRef {
        if let Some(&cached) = self.path_cache.get(&path) {
            return cached;
        }
        let uri = path.uri(self.db).to_owned();
        let r = self.ctx.paths.intern(uri);
        self.path_cache.insert(path, r);
        r
    }

    fn convert_location(&mut self, loc: Location<'db>) -> arena_types::Location {
        let path = self.convert_path(loc.path);
        arena_types::Location::new(path, loc.span)
    }

    fn convert_type(&mut self, ty: Type<'db>) -> TypeRef {
        if let Some(&cached) = self.type_cache.get(&ty) {
            return cached;
        }
        let db = self.db;
        let params: SmallVec<[TypeRef; 4]> = ty
            .params(db)
            .iter()
            .map(|&p| self.convert_type(p))
            .collect();
        let attrs: BTreeMap<Symbol, arena_types::Attribute> = ty
            .attrs(db)
            .iter()
            .map(|(k, v)| (*k, self.convert_attribute(v)))
            .collect();
        let data = arena_types::TypeData {
            dialect: ty.dialect(db),
            name: ty.name(db),
            params,
            attrs,
        };
        let r = self.ctx.types.intern(data);
        self.type_cache.insert(ty, r);
        r
    }

    fn convert_attribute(&mut self, attr: &Attribute<'db>) -> arena_types::Attribute {
        match attr {
            Attribute::Unit => arena_types::Attribute::Unit,
            Attribute::Bool(b) => arena_types::Attribute::Bool(*b),
            Attribute::IntBits(v) => arena_types::Attribute::IntBits(*v),
            Attribute::FloatBits(v) => arena_types::Attribute::FloatBits(*v),
            Attribute::String(s) => arena_types::Attribute::String(s.clone()),
            Attribute::Bytes(b) => arena_types::Attribute::Bytes(b.as_slice().into()),
            Attribute::Type(ty) => arena_types::Attribute::Type(self.convert_type(*ty)),
            Attribute::Symbol(s) => arena_types::Attribute::Symbol(*s),
            Attribute::List(list) => arena_types::Attribute::List(
                list.iter().map(|a| self.convert_attribute(a)).collect(),
            ),
            Attribute::Location(loc) => {
                arena_types::Attribute::Location(self.convert_location(*loc))
            }
        }
    }

    fn lookup_value(&self, v: ir::Value<'db>) -> ValueRef {
        *self
            .value_map
            .get(&v)
            .unwrap_or_else(|| panic!("SalsaToArena: unmapped value {:?}", v))
    }

    fn lookup_block(&self, block: &ir::Block<'db>) -> BlockRef {
        let id = block.id(self.db);
        *self
            .block_map
            .get(&id)
            .unwrap_or_else(|| panic!("SalsaToArena: unmapped block {:?}", id))
    }

    // ---- Structural conversions ----

    fn convert_region(&mut self, region: ir::Region<'db>) -> RegionRef {
        let db = self.db;
        let loc = self.convert_location(region.location(db));
        let blocks = region.blocks(db);

        // Pass 1: Create all blocks (args only, no ops) and register mappings.
        let mut arena_blocks: SmallVec<[BlockRef; 4]> = SmallVec::new();
        for salsa_block in blocks.iter() {
            let block_loc = self.convert_location(salsa_block.location(db));
            let args: Vec<BlockArgData> = salsa_block
                .args(db)
                .iter()
                .map(|arg| {
                    let ty = self.convert_type(arg.ty(db));
                    let attrs = arg
                        .attrs(db)
                        .iter()
                        .map(|(k, v)| (*k, self.convert_attribute(v)))
                        .collect();
                    BlockArgData { ty, attrs }
                })
                .collect();

            let arena_block = self.ctx.create_block(BlockData {
                location: block_loc,
                args,
                ops: SmallVec::new(),
                parent_region: None,
            });

            // Register block mapping
            let block_id = salsa_block.id(db);
            self.block_map.insert(block_id, arena_block);

            // Register block arg value mappings
            for i in 0..salsa_block.args(db).len() {
                let salsa_val = ir::Value::new(db, ir::ValueDef::BlockArg(block_id), i);
                let arena_val = self.ctx.block_arg(arena_block, i as u32);
                self.value_map.insert(salsa_val, arena_val);
            }

            arena_blocks.push(arena_block);
        }

        // Pass 2: Convert operations in each block.
        for (block_idx, salsa_block) in blocks.iter().enumerate() {
            let arena_block = arena_blocks[block_idx];
            for salsa_op in salsa_block.operations(db).iter() {
                let arena_op = self.convert_operation(*salsa_op);
                self.ctx.push_op(arena_block, arena_op);
            }
        }

        self.ctx.create_region(RegionData {
            location: loc,
            blocks: arena_blocks,
            parent_op: None,
        })
    }

    fn convert_operation(&mut self, op: ir::Operation<'db>) -> OpRef {
        let db = self.db;
        let loc = self.convert_location(op.location(db));

        // Convert operands
        let operands: Vec<ValueRef> = op
            .operands(db)
            .iter()
            .map(|v| self.lookup_value(*v))
            .collect();

        // Convert result types
        let result_types: Vec<TypeRef> = op
            .results(db)
            .iter()
            .map(|ty| self.convert_type(*ty))
            .collect();

        // Convert attributes
        let attributes: BTreeMap<Symbol, arena_types::Attribute> = op
            .attributes(db)
            .iter()
            .map(|(k, v)| (*k, self.convert_attribute(v)))
            .collect();

        // Convert regions (recursive)
        let regions: SmallVec<[RegionRef; 4]> = op
            .regions(db)
            .iter()
            .map(|r| self.convert_region(*r))
            .collect();

        // Convert successors
        let successors: SmallVec<[BlockRef; 4]> = op
            .successors(db)
            .iter()
            .map(|b| self.lookup_block(b))
            .collect();

        // Build and create the operation
        let mut builder = OperationDataBuilder::new(loc, op.dialect(db), op.name(db));
        for v in operands {
            builder = builder.operand(v);
        }
        for ty in result_types {
            builder = builder.result(ty);
        }
        for (k, v) in attributes {
            builder = builder.attr(k, v);
        }
        for r in regions {
            builder = builder.region(r);
        }
        for b in successors {
            builder = builder.successor(b);
        }
        let data = builder.build(&mut self.ctx);
        let arena_op = self.ctx.create_op(data);

        // Register result value mappings
        for (i, _) in op.results(db).iter().enumerate() {
            let salsa_val = ir::Value::new(db, ir::ValueDef::OpResult(op), i);
            let arena_val = self.ctx.op_result(arena_op, i as u32);
            self.value_map.insert(salsa_val, arena_val);
        }

        arena_op
    }
}

// ============================================================================
// Arena → Salsa
// ============================================================================

struct ArenaToSalsa<'a, 'db> {
    ctx: &'a IrContext,
    db: &'db dyn salsa::Database,
    value_map: HashMap<ValueRef, ir::Value<'db>>,
    block_map: HashMap<BlockRef, ir::Block<'db>>,
    type_cache: HashMap<TypeRef, Type<'db>>,
    path_cache: HashMap<PathRef, PathId<'db>>,
}

impl<'a, 'db> ArenaToSalsa<'a, 'db> {
    fn new(ctx: &'a IrContext, db: &'db dyn salsa::Database) -> Self {
        Self {
            ctx,
            db,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            type_cache: HashMap::new(),
            path_cache: HashMap::new(),
        }
    }

    // ---- Leaf conversions ----

    fn convert_path(&mut self, path: PathRef) -> PathId<'db> {
        if let Some(&cached) = self.path_cache.get(&path) {
            return cached;
        }
        let uri = self.ctx.paths.get(path).to_owned();
        let r = PathId::new(self.db, uri);
        self.path_cache.insert(path, r);
        r
    }

    fn convert_location(&mut self, loc: arena_types::Location) -> Location<'db> {
        let path = self.convert_path(loc.path);
        Location::new(path, loc.span)
    }

    fn convert_type(&mut self, ty: TypeRef) -> Type<'db> {
        if let Some(&cached) = self.type_cache.get(&ty) {
            return cached;
        }
        let data = self.ctx.types.get(ty);
        let params: IdVec<Type<'db>> = data.params.iter().map(|&p| self.convert_type(p)).collect();
        let attrs: BTreeMap<Symbol, Attribute<'db>> = data
            .attrs
            .iter()
            .map(|(k, v)| (*k, self.convert_attribute(v)))
            .collect();
        let r = Type::new(self.db, data.dialect, data.name, params, attrs);
        self.type_cache.insert(ty, r);
        r
    }

    fn convert_attribute(&mut self, attr: &arena_types::Attribute) -> Attribute<'db> {
        match attr {
            arena_types::Attribute::Unit => Attribute::Unit,
            arena_types::Attribute::Bool(b) => Attribute::Bool(*b),
            arena_types::Attribute::IntBits(v) => Attribute::IntBits(*v),
            arena_types::Attribute::FloatBits(v) => Attribute::FloatBits(*v),
            arena_types::Attribute::String(s) => Attribute::String(s.clone()),
            arena_types::Attribute::Bytes(b) => Attribute::Bytes(b.to_vec()),
            arena_types::Attribute::Type(ty) => Attribute::Type(self.convert_type(*ty)),
            arena_types::Attribute::Symbol(s) => Attribute::Symbol(*s),
            arena_types::Attribute::List(list) => {
                Attribute::List(list.iter().map(|a| self.convert_attribute(a)).collect())
            }
            arena_types::Attribute::Location(loc) => {
                Attribute::Location(self.convert_location(*loc))
            }
        }
    }

    fn lookup_value(&self, v: ValueRef) -> ir::Value<'db> {
        *self
            .value_map
            .get(&v)
            .unwrap_or_else(|| panic!("ArenaToSalsa: unmapped value {v}"))
    }

    fn lookup_block(&self, b: BlockRef) -> ir::Block<'db> {
        *self
            .block_map
            .get(&b)
            .unwrap_or_else(|| panic!("ArenaToSalsa: unmapped block {b}"))
    }

    // ---- Structural conversions ----

    fn convert_region(&mut self, region: RegionRef) -> ir::Region<'db> {
        let db = self.db;
        let region_data = self.ctx.region(region);
        let loc = self.convert_location(region_data.location);
        let arena_blocks: SmallVec<[BlockRef; 4]> = region_data.blocks.clone();

        // Pass 1: Allocate BlockIds and register block arg value mappings.
        // We create placeholder blocks first (empty ops) to populate block_map.
        let mut block_ids: Vec<BlockId> = Vec::with_capacity(arena_blocks.len());
        let mut salsa_blocks: Vec<ir::Block<'db>> = Vec::with_capacity(arena_blocks.len());

        for &arena_block in &arena_blocks {
            let block_data = self.ctx.block(arena_block);
            let block_loc = self.convert_location(block_data.location);
            let block_id = BlockId::fresh();
            block_ids.push(block_id);

            // Convert block args
            let args: IdVec<ir::BlockArg<'db>> = block_data
                .args
                .iter()
                .map(|arg| {
                    let ty = self.convert_type(arg.ty);
                    let attrs: BTreeMap<Symbol, Attribute<'db>> = arg
                        .attrs
                        .iter()
                        .map(|(k, v)| (*k, self.convert_attribute(v)))
                        .collect();
                    ir::BlockArg::new(db, ty, attrs)
                })
                .collect();

            // Register block arg value mappings
            let arena_args = self.ctx.block_args(arena_block);
            for (i, &arena_val) in arena_args.iter().enumerate() {
                let salsa_val = ir::Value::new(db, ir::ValueDef::BlockArg(block_id), i);
                self.value_map.insert(arena_val, salsa_val);
            }

            // Create placeholder block (empty ops, will be replaced)
            let placeholder = ir::Block::new(db, block_id, block_loc, args, IdVec::new());
            self.block_map.insert(arena_block, placeholder);
            salsa_blocks.push(placeholder);
        }

        // Pass 2: Convert operations and rebuild blocks with their ops.
        for (i, &arena_block) in arena_blocks.iter().enumerate() {
            let block_data = self.ctx.block(arena_block);
            let ops: IdVec<ir::Operation<'db>> = block_data
                .ops
                .iter()
                .map(|&op| self.convert_operation(op))
                .collect();

            if !ops.is_empty() {
                // Rebuild block with operations
                let old = salsa_blocks[i];
                let new_block =
                    ir::Block::new(db, old.id(db), old.location(db), old.args(db).clone(), ops);
                salsa_blocks[i] = new_block;
                self.block_map.insert(arena_block, new_block);
            }
        }

        let blocks: IdVec<ir::Block<'db>> = salsa_blocks.into_iter().collect();
        ir::Region::new(db, loc, blocks)
    }

    fn convert_operation(&mut self, op: OpRef) -> ir::Operation<'db> {
        let db = self.db;
        let op_data = self.ctx.op(op);
        let loc = self.convert_location(op_data.location);

        // Convert operands
        let arena_operands = self.ctx.op_operands(op);
        let operands: IdVec<ir::Value<'db>> = arena_operands
            .iter()
            .map(|&v| self.lookup_value(v))
            .collect();

        // Convert result types
        let arena_result_types = self.ctx.op_result_types(op);
        let results: IdVec<Type<'db>> = arena_result_types
            .iter()
            .map(|&ty| self.convert_type(ty))
            .collect();

        // Convert attributes
        let attributes: BTreeMap<Symbol, Attribute<'db>> = op_data
            .attributes
            .iter()
            .map(|(k, v)| (*k, self.convert_attribute(v)))
            .collect();

        // Convert regions (recursive)
        let regions: IdVec<ir::Region<'db>> = op_data
            .regions
            .iter()
            .map(|&r| self.convert_region(r))
            .collect();

        // Convert successors
        let successors: IdVec<ir::Block<'db>> = op_data
            .successors
            .iter()
            .map(|&b| self.lookup_block(b))
            .collect();

        let salsa_op = ir::Operation::new(
            db,
            loc,
            op_data.dialect,
            op_data.name,
            operands,
            results,
            attributes,
            regions,
            successors,
        );

        // Register result value mappings
        let arena_results = self.ctx.op_results(op);
        for (i, &arena_val) in arena_results.iter().enumerate() {
            let salsa_val = ir::Value::new(db, ir::ValueDef::OpResult(salsa_op), i);
            self.value_map.insert(arena_val, salsa_val);
        }

        salsa_op
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::printer as arena_printer;
    use crate::printer;
    use crate::{
        BlockBuilder, DialectOp, DialectType, Span,
        dialect::{arith, core, func, scf},
        idvec,
    };
    use salsa_test_macros::salsa_test;

    /// Tracked wrapper for building and printing a module.
    #[salsa::tracked]
    fn do_export_roundtrip(db: &dyn salsa::Database, original: ir::Operation<'_>) -> String {
        let (ctx, arena_module) = import_salsa_module(db, original);
        let exported = export_to_salsa(db, &ctx, arena_module);
        printer::print_op(db, exported)
    }

    // ------------------------------------------------------------------
    // Basic import tests
    // ------------------------------------------------------------------

    #[salsa::tracked]
    fn build_simple_module(db: &dyn salsa::Database) -> ir::Operation<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        let main_func = func::Func::build(
            db,
            location,
            "main",
            idvec![],
            core::I32::new(db).as_type(),
            |entry| {
                let c0 = entry.op(arith::Const::i32(db, location, 40));
                let c1 = entry.op(arith::Const::i32(db, location, 2));
                let add = entry.op(arith::add(
                    db,
                    location,
                    c0.result(db),
                    c1.result(db),
                    core::I32::new(db).as_type(),
                ));
                entry.op(func::Return::value(db, location, add.result(db)));
            },
        );

        core::Module::build(db, location, "main".into(), |top| {
            top.op(main_func);
        })
        .as_operation()
    }

    #[salsa_test]
    fn import_simple_module(db: &salsa::DatabaseImpl) {
        let module_op = build_simple_module(db);
        let (ctx, arena_module) = import_salsa_module(db, module_op);

        // Verify module name
        assert_eq!(arena_module.name(&ctx), Some(Symbol::new("main")));

        // Verify there's a function inside
        let ops = arena_module.ops(&ctx);
        assert_eq!(ops.len(), 1);
        let func_op = ctx.op(ops[0]);
        assert_eq!(func_op.dialect, Symbol::new("func"));
        assert_eq!(func_op.name, Symbol::new("func"));
    }

    #[salsa_test]
    fn import_preserves_arena_text(db: &salsa::DatabaseImpl) {
        let module_op = build_simple_module(db);
        let (ctx, arena_module) = import_salsa_module(db, module_op);

        let text = arena_printer::print_module(&ctx, arena_module.op());
        assert!(text.contains("core.module @main"));
        assert!(text.contains("func.func @main"));
        assert!(text.contains("arith.const {value = 40}"));
        assert!(text.contains("arith.const {value = 2}"));
        assert!(text.contains("arith.add"));
        assert!(text.contains("func.return"));
    }

    // ------------------------------------------------------------------
    // Round-trip tests
    // ------------------------------------------------------------------

    #[salsa_test]
    fn roundtrip_simple_module(db: &salsa::DatabaseImpl) {
        let original = build_simple_module(db);
        let original_text = printer::print_op(db, original);
        let exported_text = do_export_roundtrip(db, original);
        assert_eq!(original_text, exported_text);
    }

    #[salsa::tracked]
    fn build_module_with_block_args(db: &dyn salsa::Database) -> ir::Operation<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 10));
        let i32_ty = core::I32::new(db).as_type();

        let add_func = func::Func::build(
            db,
            location,
            "add",
            idvec![i32_ty, i32_ty],
            i32_ty,
            |entry| {
                let x = entry.block_arg(db, 0);
                let y = entry.block_arg(db, 1);
                let sum = entry.op(arith::add(db, location, x, y, i32_ty));
                entry.op(func::Return::value(db, location, sum.result(db)));
            },
        );

        core::Module::build(db, location, "test".into(), |top| {
            top.op(add_func);
        })
        .as_operation()
    }

    #[salsa_test]
    fn roundtrip_block_args(db: &salsa::DatabaseImpl) {
        let original = build_module_with_block_args(db);
        let original_text = printer::print_op(db, original);
        let exported_text = do_export_roundtrip(db, original);
        assert_eq!(original_text, exported_text);
    }

    #[salsa::tracked]
    fn build_module_with_nested_regions(db: &dyn salsa::Database) -> ir::Operation<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let main_func = func::Func::build(db, location, "nested", idvec![], i32_ty, |entry| {
            let cond = entry.op(arith::Const::u64(db, location, 1));

            let then_region = {
                let mut bb = BlockBuilder::new(db, location);
                let c = bb.op(arith::Const::i32(db, location, 1));
                bb.op(scf::r#yield(db, location, vec![c.result(db)]));
                ir::Region::new(db, location, idvec![bb.build()])
            };

            let else_region = {
                let mut bb = BlockBuilder::new(db, location);
                let c = bb.op(arith::Const::i32(db, location, 2));
                bb.op(scf::r#yield(db, location, vec![c.result(db)]));
                ir::Region::new(db, location, idvec![bb.build()])
            };

            let if_op = entry.op(scf::r#if(
                db,
                location,
                cond.result(db),
                i32_ty,
                then_region,
                else_region,
            ));

            entry.op(func::Return::value(db, location, if_op.result(db)));
        });

        core::Module::build(db, location, "test".into(), |top| {
            top.op(main_func);
        })
        .as_operation()
    }

    #[salsa_test]
    fn roundtrip_nested_regions(db: &salsa::DatabaseImpl) {
        let original = build_module_with_nested_regions(db);
        let original_text = printer::print_op(db, original);
        let exported_text = do_export_roundtrip(db, original);
        assert_eq!(original_text, exported_text);
    }

    #[salsa::tracked]
    fn build_module_with_multiple_functions(db: &dyn salsa::Database) -> ir::Operation<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let add_func = func::Func::build(
            db,
            location,
            "add",
            idvec![i32_ty, i32_ty],
            i32_ty,
            |entry| {
                let x = entry.block_arg(db, 0);
                let y = entry.block_arg(db, 1);
                let sum = entry.op(arith::add(db, location, x, y, i32_ty));
                entry.op(func::Return::value(db, location, sum.result(db)));
            },
        );

        let main_func = func::Func::build(db, location, "main", idvec![], i32_ty, |entry| {
            let c1 = entry.op(arith::Const::i32(db, location, 10));
            let c2 = entry.op(arith::Const::i32(db, location, 20));
            let call = entry.op(func::call(
                db,
                location,
                vec![c1.result(db), c2.result(db)],
                i32_ty,
                "add".into(),
            ));
            entry.op(func::Return::value(db, location, call.result(db)));
        });

        core::Module::build(db, location, "test".into(), |top| {
            top.op(add_func);
            top.op(main_func);
        })
        .as_operation()
    }

    #[salsa_test]
    fn roundtrip_multiple_functions(db: &salsa::DatabaseImpl) {
        let original = build_module_with_multiple_functions(db);
        let original_text = printer::print_op(db, original);
        let exported_text = do_export_roundtrip(db, original);
        assert_eq!(original_text, exported_text);
    }

    #[salsa::tracked]
    fn build_empty_module(db: &dyn salsa::Database) -> ir::Operation<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        core::Module::build(db, location, "empty".into(), |_top| {}).as_operation()
    }

    #[salsa_test]
    fn roundtrip_empty_module(db: &salsa::DatabaseImpl) {
        let original = build_empty_module(db);
        let original_text = printer::print_op(db, original);
        let exported_text = do_export_roundtrip(db, original);
        assert_eq!(original_text, exported_text);
    }
}

#[cfg(test)]
mod seed_corpus {
    use super::*;
    use crate::arena::printer as arena_printer;
    use crate::printer;

    // Salsa input to hold IR text
    #[salsa::input]
    struct TextInput {
        #[returns(ref)]
        text: String,
    }

    /// Tracked function: parse → import → export → print.
    #[salsa::tracked]
    fn roundtrip_via_arena(db: &dyn salsa::Database, input: TextInput) -> String {
        let salsa_module =
            crate::parser::parse_module(db, input.text(db)).expect("parse should succeed");
        let (ctx, arena_module) = import_salsa_module(db, salsa_module);
        let exported = export_to_salsa(db, &ctx, arena_module);
        printer::print_op(db, exported)
    }

    /// Tracked function: parse → print (canonical form).
    #[salsa::tracked]
    fn parse_and_print(db: &dyn salsa::Database, input: TextInput) -> String {
        let salsa_module =
            crate::parser::parse_module(db, input.text(db)).expect("parse should succeed");
        printer::print_op(db, salsa_module)
    }

    /// Tracked function: parse → import → arena print.
    #[salsa::tracked]
    fn import_and_arena_print(db: &dyn salsa::Database, input: TextInput) -> String {
        let salsa_module =
            crate::parser::parse_module(db, input.text(db)).expect("parse should succeed");
        let (ctx, arena_module) = import_salsa_module(db, salsa_module);
        arena_printer::print_module(&ctx, arena_module.op())
    }

    /// Seed corpus patterns reused from `crates/trunk-ir/src/parser/mod.rs`.
    static SEED_CORPUS: &[&str] = &[
        // Minimal module
        "core.module @test {\n}\n",
        // Module with simple function
        "\
core.module @test {
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 42} : core.i32
    func.return %0
  }
}
",
        // Function with arguments
        "\
core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
    %2 = arith.add %0, %1 : core.i32
    func.return %2
  }
}
",
        // Module with multiple functions
        "\
core.module @test {
  func.func @id(%0: core.i32) -> core.i32 {
    func.return %0
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 0} : core.i32
    func.return %0
  }
}
",
    ];

    /// Parse IR text → Salsa module → import to Arena → export back to Salsa → print.
    /// Verify that the output matches the original text.
    #[test]
    fn seed_corpus_roundtrip() {
        let db = salsa::DatabaseImpl::default();
        salsa::Database::attach(&db, |db| {
            for (i, &ir_text) in SEED_CORPUS.iter().enumerate() {
                let input = TextInput::new(db, ir_text.to_string());

                let original_text = parse_and_print(db, input);

                // Verify arena text is valid
                let arena_text = import_and_arena_print(db, input);
                assert!(
                    !arena_text.is_empty(),
                    "seed corpus [{i}]: arena text is empty"
                );

                // Round-trip: parse → import → export → print
                let exported_text = roundtrip_via_arena(db, input);

                assert_eq!(
                    original_text, exported_text,
                    "seed corpus [{i}] round-trip mismatch:\n--- original ---\n{original_text}\n--- exported ---\n{exported_text}"
                );
            }
        });
    }
}

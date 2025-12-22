//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This pass transforms mid-level IR operations (func, arith, scf, adt) to
//! wasm dialect operations. Phase 1 handles basic arithmetic and function calls.

use std::collections::HashMap;

use trunk_ir::DialectOp;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::{adt, arith, func, scf, src, wasm};
use trunk_ir::{
    Attribute, Block, DialectType, IdVec, Location, Operation, QualifiedName, Region, Symbol,
    Type, Value, ValueDef,
};

/// Entry point for lowering mid-level IR to wasm dialect.
#[salsa::tracked]
pub fn lower_to_wasm<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let mut lowerer = WasmLowerer::new(db);
    let lowered = lowerer.lower_module(module);

    // Debug: Verify all operations are now in wasm dialect
    if cfg!(debug_assertions) {
        check_all_wasm_dialect(db, &lowered);
    }

    lowered
}

/// Debug helper to check if all operations in function bodies are in wasm dialect
fn check_all_wasm_dialect<'db>(db: &'db dyn salsa::Database, module: &Module<'db>) {
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            let dialect = op.dialect(db);
            let name = op.name(db);

            // Check func.func operations and their bodies
            if dialect == Symbol::new("func") && name == Symbol::new("func") {
                check_function_body(db, op);
            }
        }
    }
}

/// Check if all operations in a function body are in wasm dialect
fn check_function_body<'db>(db: &'db dyn salsa::Database, func_op: &Operation<'db>) {
    if let Some(body_region) = func_op.regions(db).first() {
        for block in body_region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                let dialect = op.dialect(db);
                if dialect != Symbol::new("wasm") {
                    eprintln!(
                        "ERROR: Found non-wasm operation in function body: {}.{}",
                        dialect,
                        op.name(db)
                    );
                }
            }
        }
    }
}

/// Lowers mid-level IR to wasm dialect operations.
struct WasmLowerer<'db> {
    db: &'db dyn salsa::Database,
    value_map: HashMap<Value<'db>, Value<'db>>,
    /// Maps Types to their WasmGC type indices
    type_registry: HashMap<Type<'db>, u32>,
    /// Counter for next type index to assign
    next_type_idx: u32,
    module_location: Option<Location<'db>>,
    data_segments: DataSegments<'db>,
    memory_plan: MemoryPlan,
    wasi_plan: WasiPlan,
    main_exports: MainExports<'db>,
}

impl<'db> WasmLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            value_map: HashMap::new(),
            type_registry: HashMap::new(),
            next_type_idx: 0,
            module_location: None,
            data_segments: DataSegments::new(),
            memory_plan: MemoryPlan::new(),
            wasi_plan: WasiPlan::new(),
            main_exports: MainExports::new(),
        }
    }

    /// Get or create a type index for a given type
    fn get_type_index(&mut self, ty: Type<'db>) -> u32 {
        if let Some(&idx) = self.type_registry.get(&ty) {
            idx
        } else {
            let idx = self.next_type_idx;
            self.type_registry.insert(ty, idx);
            self.next_type_idx += 1;
            idx
        }
    }

    fn lower_module(&mut self, module: Module<'db>) -> Module<'db> {
        self.module_location = Some(module.location(self.db));
        let body = module.body(self.db);
        let lowered_body = self.lower_region(body);
        let new_body = self.finish_module_body(lowered_body);
        Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        )
    }

    fn lower_region(&mut self, region: Region<'db>) -> Region<'db> {
        let blocks = region
            .blocks(self.db)
            .iter()
            .map(|block| self.lower_block(*block))
            .collect::<Vec<_>>();
        Region::new(self.db, region.location(self.db), IdVec::from(blocks))
    }

    fn finish_module_body(&mut self, lowered_body: Region<'db>) -> Region<'db> {
        let blocks = lowered_body.blocks(self.db);
        if blocks.is_empty() {
            return lowered_body;
        }

        let mut new_blocks = Vec::new();
        for (idx, block) in blocks.iter().enumerate() {
            if idx == 0 {
                let location = block.location(self.db);
                let mut ops = self.module_preamble_ops(location);
                ops.extend(block.operations(self.db).iter().copied());
                ops.extend(self.module_data_ops(location));
                ops.extend(self.module_extra_ops(location));

                let new_block = Block::new(
                    self.db,
                    location,
                    block.args(self.db).clone(),
                    IdVec::from(ops),
                );
                new_blocks.push(new_block);
            } else {
                new_blocks.push(*block);
            }
        }

        Region::new(
            self.db,
            lowered_body.location(self.db),
            IdVec::from(new_blocks),
        )
    }

    fn lower_block(&mut self, block: Block<'db>) -> Block<'db> {
        let mut new_ops = IdVec::new();
        for op in block.operations(self.db).iter().copied() {
            let rewritten = self.lower_op(op);
            new_ops.extend(rewritten);
        }
        Block::new(
            self.db,
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    fn module_preamble_ops(&mut self, location: Location<'db>) -> Vec<Operation<'db>> {
        let mut ops = Vec::new();
        let module_location = self.module_location.unwrap_or(location);

        if self.wasi_plan.needs_fd_write {
            let i32_ty = core::I32::new(self.db).as_type();
            let params = IdVec::from(vec![i32_ty, i32_ty, i32_ty, i32_ty]);
            let import_ty = core::Func::new(self.db, params, i32_ty).as_type();
            ops.push(
                wasm::import_func(
                    self.db,
                    module_location,
                    Attribute::String("wasi_snapshot_preview1".into()),
                    Attribute::String("fd_write".into()),
                    Attribute::Symbol(Symbol::new("fd_write")),
                    Attribute::Type(import_ty),
                )
                .as_operation(),
            );
        }

        if self.memory_plan.needs_memory && !self.memory_plan.has_memory {
            let required_pages = self
                .memory_plan
                .required_pages(self.data_segments.end_offset());
            ops.push(
                wasm::memory(
                    self.db,
                    module_location,
                    Attribute::IntBits(required_pages as u64),
                    Attribute::Unit,
                    Attribute::Bool(false),
                    Attribute::Bool(false),
                )
                .as_operation(),
            );
            self.memory_plan.has_memory = true;
        }

        ops
    }

    fn module_data_ops(&mut self, location: Location<'db>) -> Vec<Operation<'db>> {
        let module_location = self.module_location.unwrap_or(location);
        self.data_segments
            .take_segments()
            .into_iter()
            .map(|(offset, bytes)| {
                wasm::data(
                    self.db,
                    module_location,
                    Attribute::IntBits(offset as u64),
                    Attribute::Bytes(bytes),
                )
                .as_operation()
            })
            .collect()
    }

    fn module_extra_ops(&mut self, location: Location<'db>) -> Vec<Operation<'db>> {
        let mut ops = Vec::new();
        let module_location = self.module_location.unwrap_or(location);

        if self.memory_plan.needs_memory
            && self.memory_plan.has_memory
            && !self.memory_plan.has_exported_memory
        {
            ops.push(
                wasm::export_memory(
                    self.db,
                    module_location,
                    Attribute::String("memory".into()),
                    Attribute::IntBits(0),
                )
                .as_operation(),
            );
            self.memory_plan.has_exported_memory = true;
        }

        if self.main_exports.saw_main && !self.main_exports.main_exported {
            ops.push(
                wasm::export_func(
                    self.db,
                    module_location,
                    Attribute::String("main".into()),
                    Attribute::QualifiedName(QualifiedName::simple(Symbol::new("main"))),
                )
                .as_operation(),
            );
            self.main_exports.main_exported = true;
        }

        if self.wasi_plan.needs_fd_write && self.main_exports.saw_main {
            if let Some(start_func) = self.build_start_function(module_location) {
                ops.push(start_func);
                ops.push(
                    wasm::export_func(
                        self.db,
                        module_location,
                        Attribute::String("_start".into()),
                        Attribute::QualifiedName(QualifiedName::simple(Symbol::new("_start"))),
                    )
                    .as_operation(),
                );
            }
        }

        ops
    }

    fn build_start_function(&self, location: Location<'db>) -> Option<Operation<'db>> {
        let nil_ty = core::Nil::new(self.db).as_type();
        let main_result = self.main_exports.main_result_type.unwrap_or(nil_ty);

        let mut call_builder = Operation::of_name(self.db, location, "wasm.call")
            .operands(IdVec::new())
            .attr(
                "callee",
                Attribute::QualifiedName(QualifiedName::simple(Symbol::new("main"))),
            );
        if !self.is_nil_type(main_result) {
            call_builder = call_builder.results(IdVec::from(vec![main_result]));
        }
        let call = call_builder.build();
        let mut ops: Vec<Operation<'db>> = vec![call];

        if !self.is_nil_type(main_result) {
            let call_val = ops.last().expect("call inserted").result(self.db, 0);
            ops.push(wasm::drop(self.db, location, call_val).as_operation());
        }

        ops.push(wasm::r#return(self.db, location, Vec::new()).as_operation());

        let body_block = Block::new(self.db, location, IdVec::new(), IdVec::from(ops));
        let body = Region::new(self.db, location, IdVec::from(vec![body_block]));
        let func_ty = core::Func::new(self.db, IdVec::new(), nil_ty).as_type();

        Some(func::func(self.db, location, Symbol::new("_start"), func_ty, body).as_operation())
    }

    fn lower_op(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let remapped_operands = self.remap_operands(op);
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);
        if dialect == wasm::DIALECT_NAME() {
            self.observe_wasm_module_op(&op, name);
        }

        // Transform operations based on dialect
        if dialect == arith::DIALECT_NAME() {
            return self.lower_arith_op(op, name, remapped_operands);
        }

        if dialect == func::DIALECT_NAME() {
            return self.lower_func_op(op, name, remapped_operands);
        }

        if dialect == scf::DIALECT_NAME() {
            return self.lower_scf_op(op, name, remapped_operands);
        }

        if dialect == adt::DIALECT_NAME() {
            return self.lower_adt_op(op, name, remapped_operands);
        }

        // src.var operations are intentionally preserved for IDE hover information.
        // They represent resolved local bindings whose actual values are in value_map.
        if dialect == src::DIALECT_NAME() && name == src::VAR() {
            let new_op = op.modify(self.db).operands(remapped_operands).build();
            self.map_results(op, new_op);
            return vec![new_op];
        }

        // Operations we're not handling should not appear in function bodies
        // They should have been processed by earlier pipeline stages
        if cfg!(debug_assertions) {
            let dialect_str = dialect.to_string();
            if !["wasm", "core", "type", "ty", "src"].contains(&dialect_str.as_str()) {
                eprintln!(
                    "WARNING: Unhandled operation in lowering: {}.{} (this may cause emit errors)",
                    dialect, name
                );
            }
        }

        // Keep other operations as-is (recursively lower regions)
        let new_regions = op
            .regions(self.db)
            .iter()
            .copied()
            .map(|region| self.lower_region(region))
            .collect::<Vec<_>>();

        let new_op = op
            .modify(self.db)
            .operands(remapped_operands)
            .regions(IdVec::from(new_regions))
            .build();
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn lower_arith_op(
        &mut self,
        op: Operation<'db>,
        name: Symbol,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let result_type = op.results(self.db).first().copied();

        let wasm_name = if name == arith::CONST() {
            self.arith_const_to_wasm_name(result_type)
        } else {
            self.arith_op_to_wasm_name(name, result_type)
        };

        let mut new_op_builder =
            Operation::of_name(self.db, location, wasm_name).operands(operands.clone());

        // For const operations, copy the value attribute
        if name == arith::CONST()
            && let Some(attr @ (Attribute::IntBits(_) | Attribute::FloatBits(_))) =
                op.attributes(self.db).get(&Symbol::new("value"))
        {
            new_op_builder = new_op_builder.attr("value", attr.clone());
        }

        // Set result types for arithmetic operations
        let new_op = new_op_builder.results(op.results(self.db).clone()).build();

        self.map_results(op, new_op);
        vec![new_op]
    }

    fn arith_const_to_wasm_name(&self, ty: Option<Type<'db>>) -> &'static str {
        match ty {
            Some(t) => {
                let name = t.name(self.db);
                if name == Symbol::new("i32") {
                    "wasm.i32_const"
                } else if name == Symbol::new("i64") {
                    "wasm.i64_const"
                } else if name == Symbol::new("f32") {
                    "wasm.f32_const"
                } else if name == Symbol::new("f64") {
                    "wasm.f64_const"
                } else {
                    // Default to i32 for unknown types
                    "wasm.i32_const"
                }
            }
            None => "wasm.i32_const", // Default to i32
        }
    }

    fn arith_op_to_wasm_name(&self, name: Symbol, ty: Option<Type<'db>>) -> &'static str {
        let type_suffix = self.type_suffix(ty);

        if name == arith::ADD() {
            match type_suffix {
                "i32" => "wasm.i32_add",
                "i64" => "wasm.i64_add",
                "f32" => "wasm.f32_add",
                "f64" => "wasm.f64_add",
                _ => "wasm.i32_add",
            }
        } else if name == arith::SUB() {
            match type_suffix {
                "i32" => "wasm.i32_sub",
                "i64" => "wasm.i64_sub",
                "f32" => "wasm.f32_sub",
                "f64" => "wasm.f64_sub",
                _ => "wasm.i32_sub",
            }
        } else if name == arith::MUL() {
            match type_suffix {
                "i32" => "wasm.i32_mul",
                "i64" => "wasm.i64_mul",
                "f32" => "wasm.f32_mul",
                "f64" => "wasm.f64_mul",
                _ => "wasm.i32_mul",
            }
        } else if name == arith::DIV() {
            match type_suffix {
                "i32" => "wasm.i32_div_s",
                "i64" => "wasm.i64_div_s",
                "f32" => "wasm.f32_div",
                "f64" => "wasm.f64_div",
                _ => "wasm.i32_div_s",
            }
        } else if name == arith::CMP_EQ() {
            match type_suffix {
                "i32" => "wasm.i32_eq",
                "i64" => "wasm.i64_eq",
                "f32" => "wasm.f32_eq",
                "f64" => "wasm.f64_eq",
                _ => "wasm.i32_eq",
            }
        } else if name == arith::CMP_NE() {
            match type_suffix {
                "i32" => "wasm.i32_ne",
                "i64" => "wasm.i64_ne",
                "f32" => "wasm.f32_ne",
                "f64" => "wasm.f64_ne",
                _ => "wasm.i32_ne",
            }
        } else if name == arith::CMP_LT() {
            match type_suffix {
                "i32" => "wasm.i32_lt_s",
                "i64" => "wasm.i64_lt_s",
                "f32" => "wasm.f32_lt",
                "f64" => "wasm.f64_lt",
                _ => "wasm.i32_lt_s",
            }
        } else if name == arith::CMP_LE() {
            match type_suffix {
                "i32" => "wasm.i32_le_s",
                "i64" => "wasm.i64_le_s",
                "f32" => "wasm.f32_le",
                "f64" => "wasm.f64_le",
                _ => "wasm.i32_le_s",
            }
        } else if name == arith::CMP_GT() {
            match type_suffix {
                "i32" => "wasm.i32_gt_s",
                "i64" => "wasm.i64_gt_s",
                "f32" => "wasm.f32_gt",
                "f64" => "wasm.f64_gt",
                _ => "wasm.i32_gt_s",
            }
        } else if name == arith::CMP_GE() {
            match type_suffix {
                "i32" => "wasm.i32_ge_s",
                "i64" => "wasm.i64_ge_s",
                "f32" => "wasm.f32_ge",
                "f64" => "wasm.f64_ge",
                _ => "wasm.i32_ge_s",
            }
        } else {
            // Unsupported arith operation; keep as-is
            "wasm.i32_add" // placeholder; should not reach here
        }
    }

    fn type_suffix(&self, ty: Option<Type<'db>>) -> &'static str {
        match ty {
            Some(t) => {
                let name = t.name(self.db);
                if name == Symbol::new("i32") {
                    "i32"
                } else if name == Symbol::new("i64") {
                    "i64"
                } else if name == Symbol::new("f32") {
                    "f32"
                } else if name == Symbol::new("f64") {
                    "f64"
                } else {
                    "i32" // Default to i32
                }
            }
            None => "i32", // Default to i32
        }
    }

    fn lower_func_op(
        &mut self,
        op: Operation<'db>,
        name: Symbol,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        // func.call -> wasm.call
        if name == func::CALL() {
            if let Some(rewritten) = self.lower_intrinsic_call(&op, operands.clone()) {
                return rewritten;
            }
            let mut new_op_builder = Operation::of_name(self.db, location, "wasm.call")
                .operands(operands)
                .results(op.results(self.db).clone());

            // Copy callee attribute
            if let Some(callee) = op.attributes(self.db).get(&Symbol::new("callee")) {
                new_op_builder = new_op_builder.attr("callee", callee.clone());
            }

            let new_op = new_op_builder.build();
            self.map_results(op, new_op);
            return vec![new_op];
        }

        // func.return -> wasm.return
        if name == func::RETURN() {
            let new_op = Operation::of_name(self.db, location, "wasm.return")
                .operands(operands)
                .build();
            return vec![new_op];
        }

        // func.func - keep as-is but lower regions
        // (emit_wasm handles func.func directly)
        if name == func::FUNC() {
            self.record_func_metadata(&op);
        }
        let new_regions = op
            .regions(self.db)
            .iter()
            .copied()
            .map(|region| self.lower_region(region))
            .collect::<Vec<_>>();

        let new_op = op
            .modify(self.db)
            .operands(operands)
            .regions(IdVec::from(new_regions))
            .build();
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn lower_scf_op(
        &mut self,
        op: Operation<'db>,
        name: Symbol,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        if name == scf::IF() {
            return self.lower_scf_if(op, location, operands);
        }

        if name == scf::LOOP() {
            return self.lower_scf_loop(op, location, operands);
        }

        if name == scf::YIELD() {
            // Yields are implicit in wasm - remove them
            return vec![];
        }

        if name == scf::CONTINUE() {
            return self.lower_scf_continue(location);
        }

        if name == scf::BREAK() {
            return self.lower_scf_break(location, operands);
        }

        // Unhandled scf operations (switch, case, default) - keep as-is
        let new_regions = op
            .regions(self.db)
            .iter()
            .copied()
            .map(|region| self.lower_region(region))
            .collect::<Vec<_>>();

        let new_op = op
            .modify(self.db)
            .operands(operands)
            .regions(IdVec::from(new_regions))
            .build();
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn lower_scf_if(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let regions = op.regions(self.db);
        let then_region = regions
            .first()
            .copied()
            .expect("scf.if missing then region");
        let else_region = regions.get(1).copied();

        // Lower regions and strip scf.yield terminators
        let lowered_then = self.lower_region_strip_yield(then_region);
        let lowered_else = else_region.map(|r| self.lower_region_strip_yield(r));

        // Build wasm.if with same structure
        let mut wasm_if = Operation::of_name(self.db, location, "wasm.if").operands(operands);

        if let Some(result_ty) = op.results(self.db).first().copied() {
            wasm_if = wasm_if.results(IdVec::from(vec![result_ty]));
        }

        let mut new_regions = vec![lowered_then];
        if let Some(else_r) = lowered_else {
            new_regions.push(else_r);
        }

        let new_op = wasm_if.regions(IdVec::from(new_regions)).build();
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn lower_region_strip_yield(&mut self, region: Region<'db>) -> Region<'db> {
        let lowered = self.lower_region(region);
        let blocks = lowered.blocks(self.db);
        let block = &blocks[0];
        let ops = block.operations(self.db);

        // Remove trailing scf.yield
        let mut new_ops = IdVec::new();
        for (i, op) in ops.iter().enumerate() {
            let is_last = i == ops.len() - 1;
            let is_yield = op.dialect(self.db) == Symbol::new("scf")
                && op.name(self.db) == Symbol::new("yield");

            if !(is_last && is_yield) {
                new_ops.push(*op);
            }
        }

        let new_block = Block::new(
            self.db,
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        );

        Region::new(
            self.db,
            lowered.location(self.db),
            IdVec::from(vec![new_block]),
        )
    }

    fn lower_scf_loop(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        _operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let regions = op.regions(self.db);
        let body_region = *regions.first().expect("scf.loop missing body");

        // Lower loop body (contains scf.continue/scf.break)
        let lowered_body = self.lower_region(body_region);

        // Create wasm.loop with lowered body
        let wasm_loop = Operation::of_name(self.db, location, "wasm.loop")
            .regions(IdVec::from(vec![lowered_body]))
            .build();

        // Wrap in wasm.block for break target
        let result_ty = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Nil::new(self.db).as_type());

        let block_body_block = Block::new(
            self.db,
            location,
            IdVec::new(),
            IdVec::from(vec![wasm_loop]),
        );
        let block_body = Region::new(self.db, location, IdVec::from(vec![block_body_block]));

        let wasm_block = Operation::of_name(self.db, location, "wasm.block")
            .results(IdVec::from(vec![result_ty]))
            .regions(IdVec::from(vec![block_body]))
            .build();

        self.map_results(op, wasm_block);
        vec![wasm_block]
    }

    fn lower_scf_continue(&mut self, location: Location<'db>) -> Vec<Operation<'db>> {
        // Branch to enclosing wasm.loop (depth 1)
        let br_op = Operation::of_name(self.db, location, "wasm.br")
            .attr("target", Attribute::IntBits(1))
            .build();
        vec![br_op]
    }

    fn lower_scf_break(
        &mut self,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        // Branch to enclosing wasm.block (depth 0)
        // Operand (result value) stays on stack
        let br_op = Operation::of_name(self.db, location, "wasm.br")
            .attr("target", Attribute::IntBits(0))
            .operands(operands)
            .build();
        vec![br_op]
    }

    fn lower_adt_op(
        &mut self,
        op: Operation<'db>,
        name: Symbol,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        if name == adt::STRING_CONST() {
            return self.lower_string_const(op, location);
        }

        if name == adt::BYTES_CONST() {
            return self.lower_bytes_const(op, location);
        }

        if name == adt::STRUCT_NEW() {
            return self.lower_adt_struct_new(op, location, operands);
        }

        if name == adt::STRUCT_GET() {
            return self.lower_adt_struct_get(op, location, operands);
        }

        if name == adt::STRUCT_SET() {
            return self.lower_adt_struct_set(op, location, operands);
        }

        if name == adt::VARIANT_NEW() {
            return self.lower_adt_variant_new(op, location, operands);
        }

        if name == adt::VARIANT_TAG() {
            return self.lower_adt_variant_tag(op, location, operands);
        }

        if name == adt::VARIANT_GET() {
            return self.lower_adt_variant_get(op, location, operands);
        }

        if name == adt::ARRAY_NEW() {
            return self.lower_adt_array_new(op, location, operands);
        }

        if name == adt::ARRAY_GET() {
            return self.lower_adt_array_get(op, location, operands);
        }

        if name == adt::ARRAY_SET() {
            return self.lower_adt_array_set(op, location, operands);
        }

        if name == adt::ARRAY_LEN() {
            return self.lower_adt_array_len(op, location, operands);
        }

        if name == adt::REF_NULL() {
            return self.lower_adt_ref_null(op, location);
        }

        if name == adt::REF_IS_NULL() {
            return self.lower_adt_ref_is_null(op, location, operands);
        }

        if name == adt::REF_CAST() {
            return self.lower_adt_ref_cast(op, location, operands);
        }

        // Unhandled ADT operations - keep as-is with region lowering
        let new_regions = op
            .regions(self.db)
            .iter()
            .copied()
            .map(|region| self.lower_region(region))
            .collect::<Vec<_>>();

        let new_op = op
            .modify(self.db)
            .operands(operands)
            .regions(IdVec::from(new_regions))
            .build();
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn lower_adt_struct_new(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        // Get the struct type from the type attribute
        let attrs = op.attributes(self.db);
        let struct_type = match attrs.get(&Symbol::new("type")) {
            Some(Attribute::Type(ty)) => *ty,
            _ => {
                // Fallback: keep as-is if type attribute is missing
                return vec![op];
            }
        };

        let type_idx = self.get_type_index(struct_type);

        // Create wasm.struct_new with type index
        let wasm_struct_new = Operation::of_name(self.db, location, "wasm.struct_new")
            .operands(operands)
            .attr("type_idx", Attribute::IntBits(type_idx as u64))
            .results(op.results(self.db).clone())
            .build();

        self.map_results(op, wasm_struct_new);
        vec![wasm_struct_new]
    }

    fn lower_adt_struct_get(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(struct_ref) = operands.first().copied() else {
            return vec![op];
        };
        let field_attr = attrs.get(&Symbol::new("field"));
        let Some(field_idx) = self.field_index_from_attr(field_attr) else {
            return vec![op];
        };

        // Create wasm.struct_get with field index
        let mut wasm_struct_get = Operation::of_name(self.db, location, "wasm.struct_get")
            .operands(IdVec::from(vec![struct_ref]))
            .attr("field_idx", Attribute::IntBits(field_idx as u64));
        if let Some(type_idx) = self.type_idx_for_value(struct_ref) {
            wasm_struct_get = wasm_struct_get.attr("type_idx", Attribute::IntBits(type_idx as u64));
        }
        let wasm_struct_get = wasm_struct_get.results(op.results(self.db).clone()).build();

        self.map_results(op, wasm_struct_get);
        vec![wasm_struct_get]
    }

    fn lower_adt_struct_set(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(struct_ref) = operands.first().copied() else {
            return vec![op];
        };
        let field_attr = attrs.get(&Symbol::new("field"));
        let Some(field_idx) = self.field_index_from_attr(field_attr) else {
            return vec![op];
        };

        let mut wasm_struct_set = Operation::of_name(self.db, location, "wasm.struct_set")
            .operands(operands)
            .attr("field_idx", Attribute::IntBits(field_idx as u64));
        if let Some(type_idx) = self.type_idx_for_value(struct_ref) {
            wasm_struct_set = wasm_struct_set.attr("type_idx", Attribute::IntBits(type_idx as u64));
        }
        vec![wasm_struct_set.build()]
    }

    fn lower_adt_variant_new(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        // Get the variant type and tag from attributes
        let attrs = op.attributes(self.db);
        let variant_type = match attrs.get(&Symbol::new("type")) {
            Some(Attribute::Type(ty)) => *ty,
            _ => {
                // Fallback: keep as-is if type attribute is missing
                return vec![op];
            }
        };

        let tag = match attrs.get(&Symbol::new("tag")) {
            Some(Attribute::IntBits(tag)) => *tag as u32,
            Some(Attribute::Symbol(tag_sym)) => Self::name_hash_u32(&tag_sym.to_string()),
            Some(Attribute::String(tag_str)) => Self::name_hash_u32(tag_str),
            _ => {
                // Fallback: keep as-is if tag is missing
                return vec![op];
            }
        };

        let type_idx = self.get_type_index(variant_type);

        // Variants are represented as structs with an i32 tag field (field 0) + payload fields
        // We need to insert the tag as the first field
        let mut variant_fields = IdVec::new();

        // First field is the tag (as i32 const)
        // If the tag isn't a direct discriminant, we fall back to a deterministic hash.
        let tag_value = u64::from(tag);
        let tag_const = Operation::of_name(self.db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(tag_value & 0xFFFFFFFF))
            .results(IdVec::from(vec![core::I32::new(self.db).as_type()]))
            .build();
        let tag_value = tag_const.result(self.db, 0);
        variant_fields.push(tag_value);

        // Add payload fields
        for &operand in operands.iter() {
            variant_fields.push(operand);
        }

        // Create wasm.struct_new for the variant (includes tag + fields)
        let wasm_variant_new = Operation::of_name(self.db, location, "wasm.struct_new")
            .operands(variant_fields)
            .attr("type_idx", Attribute::IntBits(type_idx as u64))
            .results(op.results(self.db).clone())
            .build();

        // The tag const operation is implicit - just return the struct_new
        self.map_results(op, wasm_variant_new);
        vec![tag_const, wasm_variant_new]
    }

    fn lower_adt_variant_tag(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        // Extract tag from variant (which is field 0 of the struct)
        let Some(variant_ref) = operands.first().copied() else {
            return vec![op];
        };

        // Create wasm.struct_get with field_idx=0 (the tag field)
        let mut wasm_tag_get = Operation::of_name(self.db, location, "wasm.struct_get")
            .operands(IdVec::from(vec![variant_ref]))
            .attr("field_idx", Attribute::IntBits(0));
        if let Some(type_idx) = self.type_idx_for_value(variant_ref) {
            wasm_tag_get = wasm_tag_get.attr("type_idx", Attribute::IntBits(type_idx as u64));
        }
        let wasm_tag_get = wasm_tag_get.results(op.results(self.db).clone()).build();

        self.map_results(op, wasm_tag_get);
        vec![wasm_tag_get]
    }

    fn lower_adt_variant_get(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        // Extract field from variant (field index offset by 1 because field 0 is the tag)
        let Some(variant_ref) = operands.first().copied() else {
            return vec![op];
        };

        let attrs = op.attributes(self.db);
        let field_attr = attrs.get(&Symbol::new("field"));

        // Similar to struct_get, extract field index
        let Some(base_field_idx) = self.field_index_from_attr(field_attr) else {
            return vec![op];
        };

        // Add 1 to skip the tag field
        let field_idx = base_field_idx + 1;

        // Create wasm.struct_get with offset field index
        let mut wasm_variant_get = Operation::of_name(self.db, location, "wasm.struct_get")
            .operands(IdVec::from(vec![variant_ref]))
            .attr("field_idx", Attribute::IntBits(field_idx as u64));
        if let Some(type_idx) = self.type_idx_for_value(variant_ref) {
            wasm_variant_get =
                wasm_variant_get.attr("type_idx", Attribute::IntBits(type_idx as u64));
        }
        let wasm_variant_get = wasm_variant_get
            .results(op.results(self.db).clone())
            .build();

        self.map_results(op, wasm_variant_get);
        vec![wasm_variant_get]
    }

    fn lower_adt_array_new(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(type_idx) = self.type_idx_from_attr(attrs.get(&Symbol::new("type"))) else {
            return vec![op];
        };

        let (wasm_name, operands) = if operands.len() <= 1 {
            ("wasm.array_new_default", operands)
        } else {
            ("wasm.array_new", operands)
        };

        let wasm_array_new = Operation::of_name(self.db, location, wasm_name)
            .operands(operands)
            .attr("type_idx", Attribute::IntBits(type_idx as u64))
            .results(op.results(self.db).clone())
            .build();

        self.map_results(op, wasm_array_new);
        vec![wasm_array_new]
    }

    fn lower_adt_array_get(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let Some(array_ref) = operands.first().copied() else {
            return vec![op];
        };
        let mut wasm_array_get =
            Operation::of_name(self.db, location, "wasm.array_get").operands(operands);
        if let Some(type_idx) = self.type_idx_for_value(array_ref) {
            wasm_array_get = wasm_array_get.attr("type_idx", Attribute::IntBits(type_idx as u64));
        }
        let wasm_array_get = wasm_array_get.results(op.results(self.db).clone()).build();

        self.map_results(op, wasm_array_get);
        vec![wasm_array_get]
    }

    fn lower_adt_array_set(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let Some(array_ref) = operands.first().copied() else {
            return vec![op];
        };
        let mut wasm_array_set =
            Operation::of_name(self.db, location, "wasm.array_set").operands(operands);
        if let Some(type_idx) = self.type_idx_for_value(array_ref) {
            wasm_array_set = wasm_array_set.attr("type_idx", Attribute::IntBits(type_idx as u64));
        }
        vec![wasm_array_set.build()]
    }

    fn lower_adt_array_len(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let wasm_array_len = Operation::of_name(self.db, location, "wasm.array_len")
            .operands(operands)
            .results(op.results(self.db).clone())
            .build();

        self.map_results(op, wasm_array_len);
        vec![wasm_array_len]
    }

    fn lower_adt_ref_null(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
    ) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(type_idx) = self.type_idx_from_attr(attrs.get(&Symbol::new("type"))) else {
            return vec![op];
        };

        let wasm_ref_null = Operation::of_name(self.db, location, "wasm.ref_null")
            .attr("heap_type", Attribute::IntBits(type_idx as u64))
            .results(op.results(self.db).clone())
            .build();

        self.map_results(op, wasm_ref_null);
        vec![wasm_ref_null]
    }

    fn lower_adt_ref_is_null(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let wasm_ref_is_null = Operation::of_name(self.db, location, "wasm.ref_is_null")
            .operands(operands)
            .results(op.results(self.db).clone())
            .build();

        self.map_results(op, wasm_ref_is_null);
        vec![wasm_ref_is_null]
    }

    fn lower_adt_ref_cast(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(type_idx) = self.type_idx_from_attr(attrs.get(&Symbol::new("type"))) else {
            return vec![op];
        };

        let wasm_ref_cast = Operation::of_name(self.db, location, "wasm.ref_cast")
            .operands(operands)
            .attr("target_type", Attribute::IntBits(type_idx as u64))
            .results(op.results(self.db).clone())
            .build();

        self.map_results(op, wasm_ref_cast);
        vec![wasm_ref_cast]
    }

    fn lower_string_const(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
    ) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(Attribute::String(value)) = attrs.get(&Symbol::new("value")) else {
            return vec![op];
        };
        let (offset, len) = self
            .data_segments
            .allocate_bytes(value.clone().into_bytes());
        self.memory_plan.needs_memory = true;
        let new_op = self.build_pointer_const(location, op.results(self.db).clone(), offset);
        let new_value = new_op.result(self.db, 0);
        self.data_segments.record_literal(new_value, offset, len);
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn lower_bytes_const(
        &mut self,
        op: Operation<'db>,
        location: Location<'db>,
    ) -> Vec<Operation<'db>> {
        let attrs = op.attributes(self.db);
        let Some(Attribute::Bytes(value)) = attrs.get(&Symbol::new("value")) else {
            return vec![op];
        };
        let (offset, len) = self.data_segments.allocate_bytes(value.clone());
        self.memory_plan.needs_memory = true;
        let new_op = self.build_pointer_const(location, op.results(self.db).clone(), offset);
        let new_value = new_op.result(self.db, 0);
        self.data_segments.record_literal(new_value, offset, len);
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn lower_intrinsic_call(
        &mut self,
        op: &Operation<'db>,
        operands: IdVec<Value<'db>>,
    ) -> Option<Vec<Operation<'db>>> {
        let results = op.results(self.db);
        let returns_unit = results
            .first()
            .copied()
            .map(|ty| self.is_nil_type(ty))
            .unwrap_or(false);

        if (results.is_empty() || returns_unit)
            && self.is_print_line_call(op)
            && let Some(arg) = operands.first().copied()
            && let Some((ptr, len)) = self.data_segments.literal_for(arg)
        {
            self.wasi_plan.needs_fd_write = true;
            self.memory_plan.needs_memory = true;

            let location = op.location(self.db);
            let i32_ty = core::I32::new(self.db).as_type();
            let iovec_ptr = self.data_segments.ensure_iovec(ptr, len);
            let nwritten_ptr = self.data_segments.ensure_nwritten();

            let fd_const = self.wasm_i32_const(location, 1, i32_ty);
            let iovec_const = self.wasm_i32_const(location, iovec_ptr, i32_ty);
            let iovec_len_const = self.wasm_i32_const(location, 1, i32_ty);
            let nwritten_const = self.wasm_i32_const(location, nwritten_ptr, i32_ty);

            let callee = QualifiedName::simple(Symbol::new("fd_write"));
            let call = Operation::of_name(self.db, location, "wasm.call")
                .operands(IdVec::from(vec![
                    fd_const.result(self.db, 0),
                    iovec_const.result(self.db, 0),
                    iovec_len_const.result(self.db, 0),
                    nwritten_const.result(self.db, 0),
                ]))
                .results(IdVec::from(vec![i32_ty]))
                .attr("callee", Attribute::QualifiedName(callee))
                .build();
            let drop = Operation::of_name(self.db, location, "wasm.drop")
                .operands(IdVec::from(vec![call.result(self.db, 0)]))
                .build();
            if !results.is_empty() {
                let old_result = op.result(self.db, 0);
                let replacement = nwritten_const.result(self.db, 0);
                self.value_map.insert(old_result, replacement);
            }

            return Some(vec![
                fd_const,
                iovec_const,
                iovec_len_const,
                nwritten_const,
                call,
                drop,
            ]);
        }
        None
    }

    fn record_func_metadata(&mut self, op: &Operation<'db>) {
        let attrs = op.attributes(self.db);
        let Some(Attribute::Symbol(sym)) = attrs.get(&Symbol::new("sym_name")) else {
            return;
        };
        if *sym != Symbol::new("main") {
            return;
        }

        self.main_exports.saw_main = true;
        if let Some(Attribute::Type(ty)) = attrs.get(&Symbol::new("type")) {
            if let Some(func_ty) = core::Func::from_type(self.db, *ty) {
                self.main_exports.main_result_type = Some(func_ty.result(self.db));
            }
        }
    }

    fn observe_wasm_module_op(&mut self, op: &Operation<'db>, name: Symbol) {
        if name == wasm::MEMORY() {
            self.memory_plan.has_memory = true;
        } else if name == wasm::EXPORT_MEMORY() {
            self.memory_plan.has_exported_memory = true;
        } else if name == wasm::EXPORT_FUNC() {
            if let Some(Attribute::String(export)) =
                op.attributes(self.db).get(&Symbol::new("name"))
            {
                if export == "main" {
                    self.main_exports.main_exported = true;
                }
            }
        }
    }

    fn wasm_i32_const(
        &self,
        location: Location<'db>,
        value: u32,
        result_ty: Type<'db>,
    ) -> Operation<'db> {
        Operation::of_name(self.db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(u64::from(value)))
            .results(IdVec::from(vec![result_ty]))
            .build()
    }

    fn build_pointer_const(
        &self,
        location: Location<'db>,
        results: IdVec<Type<'db>>,
        offset: u32,
    ) -> Operation<'db> {
        Operation::of_name(self.db, location, "wasm.i32_const")
            .attr("value", Attribute::IntBits(offset as u64))
            .results(results)
            .build()
    }

    fn is_print_line_call(&self, op: &Operation<'db>) -> bool {
        let attrs = op.attributes(self.db);
        let Some(Attribute::QualifiedName(path)) = attrs.get(&Symbol::new("callee")) else {
            return false;
        };
        path.name() == Symbol::new("print_line")
    }

    fn is_nil_type(&self, ty: Type<'db>) -> bool {
        ty.dialect(self.db) == core::DIALECT_NAME() && ty.name(self.db) == Symbol::new("nil")
    }

    fn remap_operands(&self, op: Operation<'db>) -> IdVec<Value<'db>> {
        let mut operands = IdVec::new();
        for &operand in op.operands(self.db).iter() {
            let mapped = self.value_map.get(&operand).copied().unwrap_or(operand);
            operands.push(mapped);
        }
        operands
    }

    fn map_results(&mut self, old_op: Operation<'db>, new_op: Operation<'db>) {
        let old_results = old_op.results(self.db);
        let new_results = new_op.results(self.db);
        let count = old_results.len().min(new_results.len());
        for i in 0..count {
            let old_val = old_op.result(self.db, i);
            let new_val = new_op.result(self.db, i);
            self.value_map.insert(old_val, new_val);
        }
    }

    fn value_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        match value.def(self.db) {
            ValueDef::OpResult(op) => op.results(self.db).get(value.index(self.db)).copied(),
            ValueDef::BlockArg(block) => block.args(self.db).get(value.index(self.db)).copied(),
        }
    }

    fn type_idx_for_value(&mut self, value: Value<'db>) -> Option<u32> {
        let ty = self.value_type(value)?;
        Some(self.get_type_index(ty))
    }

    fn type_idx_from_attr(&mut self, attr: Option<&Attribute<'db>>) -> Option<u32> {
        match attr {
            Some(Attribute::Type(ty)) => Some(self.get_type_index(*ty)),
            _ => None,
        }
    }

    fn field_index_from_attr(&self, attr: Option<&Attribute<'db>>) -> Option<u32> {
        match attr {
            Some(Attribute::IntBits(idx)) => Some(*idx as u32),
            Some(Attribute::Symbol(sym)) => Some(Self::name_hash_u32(&sym.to_string())),
            Some(Attribute::String(name)) => Some(Self::name_hash_u32(name)),
            _ => None,
        }
    }

    fn name_hash_u32(name: &str) -> u32 {
        name.as_bytes()
            .iter()
            .fold(0u32, |h, &b| h.wrapping_mul(31).wrapping_add(u32::from(b)))
    }
}

#[derive(Default)]
struct WasiPlan {
    needs_fd_write: bool,
}

impl WasiPlan {
    fn new() -> Self {
        Self::default()
    }
}

#[derive(Default)]
struct MemoryPlan {
    has_memory: bool,
    has_exported_memory: bool,
    needs_memory: bool,
}

impl MemoryPlan {
    fn new() -> Self {
        Self::default()
    }

    fn required_pages(&self, end_offset: u32) -> u32 {
        std::cmp::max(1, (end_offset + 0xFFFF) / 0x10000)
    }
}

#[derive(Default)]
struct MainExports<'db> {
    saw_main: bool,
    main_result_type: Option<Type<'db>>,
    main_exported: bool,
}

impl<'db> MainExports<'db> {
    fn new() -> Self {
        Self::default()
    }
}

#[derive(Default)]
struct DataSegments<'db> {
    next_offset: u32,
    segments: Vec<(u32, Vec<u8>)>,
    literal_data: HashMap<Value<'db>, (u32, u32)>,
    iovec_offsets: HashMap<(u32, u32), u32>,
    nwritten_offset: Option<u32>,
}

impl<'db> DataSegments<'db> {
    fn new() -> Self {
        Self::default()
    }

    fn end_offset(&self) -> u32 {
        self.next_offset
    }

    fn allocate_bytes(&mut self, bytes: Vec<u8>) -> (u32, u32) {
        let offset = Self::align_to(self.next_offset, 4);
        let len = bytes.len() as u32;
        self.segments.push((offset, bytes));
        self.next_offset = offset + len;
        (offset, len)
    }

    fn record_literal(&mut self, value: Value<'db>, offset: u32, len: u32) {
        self.literal_data.insert(value, (offset, len));
    }

    fn literal_for(&self, value: Value<'db>) -> Option<(u32, u32)> {
        self.literal_data.get(&value).copied()
    }

    fn ensure_iovec(&mut self, ptr: u32, len: u32) -> u32 {
        if let Some(&offset) = self.iovec_offsets.get(&(ptr, len)) {
            return offset;
        }
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&ptr.to_le_bytes());
        bytes.extend_from_slice(&len.to_le_bytes());
        let (offset, _) = self.allocate_bytes(bytes);
        self.iovec_offsets.insert((ptr, len), offset);
        offset
    }

    fn ensure_nwritten(&mut self) -> u32 {
        if let Some(offset) = self.nwritten_offset {
            return offset;
        }
        let (offset, _) = self.allocate_bytes(vec![0, 0, 0, 0]);
        self.nwritten_offset = Some(offset);
        offset
    }

    fn take_segments(&mut self) -> Vec<(u32, Vec<u8>)> {
        std::mem::take(&mut self.segments)
    }

    fn align_to(value: u32, align: u32) -> u32 {
        if align == 0 {
            return value;
        }
        ((value + align - 1) / align) * align
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lower_arith_const() {
        // Basic test placeholder
    }

    #[test]
    fn test_lower_arith_add() {
        // Basic test placeholder
    }
}

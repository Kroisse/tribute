//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This pass transforms mid-level IR operations (func, arith, scf, adt) to
//! wasm dialect operations. Phase 1 handles basic arithmetic and function calls.

use std::collections::HashMap;

use crate::plan::{DataSegments, MainExports, MemoryPlan, WasiPlan};

use trunk_ir::DialectOp;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::{func, wasm};

use crate::passes::const_to_wasm::ConstAnalysis;
use trunk_ir::ir::BlockBuilder;
use trunk_ir::{
    Attribute, Block, DialectType, IdVec, Location, Operation, QualifiedName, Region, Symbol,
    Type, Value, idvec,
};

/// Entry point for lowering mid-level IR to wasm dialect.
#[salsa::tracked]
pub fn lower_to_wasm<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // Phase 1: Pattern-based lowering passes
    let module = crate::passes::arith_to_wasm::lower(db, module);
    let module = crate::passes::scf_to_wasm::lower(db, module);
    let module = crate::passes::func_to_wasm::lower(db, module);
    let module = crate::passes::adt_to_wasm::lower(db, module);

    // Const analysis and lowering (string/bytes constants to data segments)
    let const_analysis = crate::passes::const_to_wasm::analyze_consts(db, module);
    let module = crate::passes::const_to_wasm::lower(db, module, const_analysis);

    // Phase 2: Remaining lowering via WasmLowerer
    let mut lowerer = WasmLowerer::new(db, const_analysis);
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

            // Check wasm.func operations and their bodies
            if dialect == wasm::DIALECT_NAME() && name == wasm::FUNC() {
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
    module_location: Option<Location<'db>>,
    const_analysis: ConstAnalysis<'db>,
    data_segments: DataSegments,
    memory_plan: MemoryPlan,
    wasi_plan: WasiPlan,
    main_exports: MainExports<'db>,
}

impl<'db> WasmLowerer<'db> {
    fn new(db: &'db dyn salsa::Database, const_analysis: ConstAnalysis<'db>) -> Self {
        // Start data segment allocation after const analysis data
        let data_segments = DataSegments::with_offset(const_analysis.total_size(db));
        Self {
            db,
            value_map: HashMap::new(),
            module_location: None,
            const_analysis,
            data_segments,
            memory_plan: MemoryPlan::new(),
            wasi_plan: WasiPlan::new(),
            main_exports: MainExports::new(),
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
            .collect();
        Region::new(self.db, region.location(self.db), blocks)
    }

    fn finish_module_body(&mut self, lowered_body: Region<'db>) -> Region<'db> {
        let blocks = lowered_body.blocks(self.db);
        let Some(first_block) = blocks.first() else {
            return lowered_body;
        };

        let location = first_block.location(self.db);
        let mut builder =
            BlockBuilder::new(self.db, location).args(first_block.args(self.db).clone());

        self.module_preamble_ops(&mut builder, location);
        for op in first_block.operations(self.db).iter() {
            builder.op(*op);
        }
        self.module_data_ops(&mut builder, location);
        self.module_extra_ops(&mut builder, location);

        let mut new_blocks = blocks.clone();
        new_blocks[0] = builder.build();
        Region::new(self.db, lowered_body.location(self.db), new_blocks)
    }

    fn lower_block(&mut self, block: Block<'db>) -> Block<'db> {
        let location = block.location(self.db);
        let args = block.args(self.db).clone();
        let mut builder = BlockBuilder::new(self.db, location).args(args);

        for op in block.operations(self.db).iter().copied() {
            self.lower_op(&mut builder, op);
        }

        builder.build()
    }

    fn module_preamble_ops(&mut self, builder: &mut BlockBuilder<'db>, location: Location<'db>) {
        let module_location = self.module_location.unwrap_or(location);

        if self.wasi_plan.needs_fd_write {
            let i32_ty = core::I32::new(self.db).as_type();
            let params = idvec![i32_ty, i32_ty, i32_ty, i32_ty];
            let import_ty = core::Func::new(self.db, params, i32_ty).as_type();
            builder.op(wasm::import_func(
                self.db,
                module_location,
                Attribute::String("wasi_snapshot_preview1".into()),
                Attribute::String("fd_write".into()),
                Attribute::Symbol(Symbol::new("fd_write")),
                Attribute::Type(import_ty),
            ));
        }

        // Check if const analysis has data that requires memory
        let const_size = self.const_analysis.total_size(self.db);
        if const_size > 0 {
            self.memory_plan.needs_memory = true;
        }

        if self.memory_plan.needs_memory && !self.memory_plan.has_memory {
            // Calculate total data size: const analysis data + runtime data segments
            let total_data_size = const_size + self.data_segments.end_offset();
            let required_pages = self.memory_plan.required_pages(total_data_size);
            builder.op(wasm::memory(
                self.db,
                module_location,
                Attribute::IntBits(required_pages as u64),
                Attribute::Unit,
                Attribute::Bool(false),
                Attribute::Bool(false),
            ));
            self.memory_plan.has_memory = true;
        }
    }

    fn module_data_ops(&mut self, builder: &mut BlockBuilder<'db>, location: Location<'db>) {
        let module_location = self.module_location.unwrap_or(location);

        // Emit data segments from const analysis (string/bytes constants)
        for (content, offset, _len) in self.const_analysis.allocations(self.db).iter() {
            builder.op(wasm::data(
                self.db,
                module_location,
                Attribute::IntBits(u64::from(*offset)),
                Attribute::Bytes(content.clone()),
            ));
        }

        // Emit data segments from runtime allocations (iovec, nwritten, etc.)
        for (offset, bytes) in self.data_segments.take_segments() {
            builder.op(wasm::data(
                self.db,
                module_location,
                Attribute::IntBits(u64::from(offset)),
                Attribute::Bytes(bytes),
            ));
        }
    }

    fn module_extra_ops(&mut self, builder: &mut BlockBuilder<'db>, location: Location<'db>) {
        let module_location = self.module_location.unwrap_or(location);

        if self.memory_plan.needs_memory
            && self.memory_plan.has_memory
            && !self.memory_plan.has_exported_memory
        {
            builder.op(wasm::export_memory(
                self.db,
                module_location,
                Attribute::String("memory".into()),
                Attribute::IntBits(0),
            ));
            self.memory_plan.has_exported_memory = true;
        }

        if self.main_exports.saw_main && !self.main_exports.main_exported {
            builder.op(wasm::export_func(
                self.db,
                module_location,
                Attribute::String("main".into()),
                Attribute::QualifiedName(QualifiedName::simple(Symbol::new("main"))),
            ));
            self.main_exports.main_exported = true;
        }

        if self.wasi_plan.needs_fd_write && self.main_exports.saw_main {
            builder.op(self.build_start_function(module_location));
            builder.op(wasm::export_func(
                self.db,
                module_location,
                Attribute::String("_start".into()),
                Attribute::QualifiedName(QualifiedName::simple(Symbol::new("_start"))),
            ));
        }
    }

    fn build_start_function(&self, location: Location<'db>) -> func::Func<'db> {
        let main_result = self.main_exports.main_result_type;

        // Collect operations for the function body
        let mut builder = BlockBuilder::new(self.db, location);

        // Build wasm.call to main
        let call = builder.op(wasm::call(
            self.db,
            location,
            None,
            main_result,
            Attribute::QualifiedName(QualifiedName::simple(Symbol::new("main"))),
        ));

        // Drop result if main returns non-nil (use typed helper)
        if main_result.is_some() {
            let call_val = call.result(self.db, 0);
            builder.op(wasm::drop(self.db, location, call_val));
        }

        // Return (use typed helper)
        builder.op(wasm::r#return(self.db, location, None));

        // Build region and func operation
        let body_block = builder.build();
        let region = Region::new(self.db, location, idvec![body_block]);
        let func_ty =
            core::Func::new(self.db, idvec![], core::Nil::new(self.db).as_type()).as_type();

        func::func(self.db, location, Symbol::new("_start"), func_ty, region)
    }

    fn lower_op(&mut self, builder: &mut BlockBuilder<'db>, op: Operation<'db>) {
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);

        // Handle wasm dialect metadata collection and intrinsic calls
        if dialect == wasm::DIALECT_NAME() {
            self.observe_wasm_module_op(&op, name);

            if name == wasm::FUNC() {
                if let Ok(op_func) = wasm::Func::from_operation(self.db, op) {
                    self.record_wasm_func_metadata(&op_func);
                }
            }

            // Handle intrinsic calls (e.g., print_line -> fd_write)
            if name == wasm::CALL() {
                let remapped_operands = self.remap_operands(op);
                if self.lower_intrinsic_call(builder, &op, remapped_operands) {
                    return;
                }
            }
        }

        // Debug warning for unexpected dialects in function bodies
        if cfg!(debug_assertions) {
            let dialect_str = dialect.to_string();
            let allowed = ["wasm", "core", "type", "ty", "src", "func", "adt"];
            if !allowed.contains(&dialect_str.as_str()) {
                eprintln!(
                    "WARNING: Unhandled operation in lowering: {}.{} (this may cause emit errors)",
                    dialect, name
                );
            }
        }

        // Default: preserve operation with remapped operands and recursively lowered regions
        self.preserve_op_with_lowered_regions(builder, op);
    }

    /// Preserve an operation as-is, but remap operands and recursively lower nested regions.
    fn preserve_op_with_lowered_regions(
        &mut self,
        builder: &mut BlockBuilder<'db>,
        op: Operation<'db>,
    ) {
        let remapped_operands = self.remap_operands(op);
        let new_regions: Vec<_> = op
            .regions(self.db)
            .iter()
            .copied()
            .map(|region| self.lower_region(region))
            .collect();

        let new_op = op
            .modify(self.db)
            .operands(remapped_operands)
            .regions(IdVec::from(new_regions))
            .build();
        self.map_results(&op, &new_op);
        builder.op(new_op);
    }

    /// Try to get literal pointer and length from a value's defining operation.
    /// Returns (ptr_offset, length) if the value is from a wasm.i32_const with literal_len.
    fn get_literal_info(&self, value: Value<'db>) -> Option<(u32, u32)> {
        let def = value.def(self.db);
        let trunk_ir::ValueDef::OpResult(op) = def else {
            return None;
        };
        if op.dialect(self.db) != wasm::DIALECT_NAME() {
            return None;
        }
        if op.name(self.db) != Symbol::new("i32_const") {
            return None;
        }
        let attrs = op.attributes(self.db);
        let Attribute::IntBits(ptr) = attrs.get(&Symbol::new("value"))? else {
            return None;
        };
        let Attribute::IntBits(len) = attrs.get(&Symbol::new("literal_len"))? else {
            return None;
        };
        Some((*ptr as u32, *len as u32))
    }

    fn lower_intrinsic_call(
        &mut self,
        builder: &mut BlockBuilder<'db>,
        op: &Operation<'db>,
        operands: IdVec<Value<'db>>,
    ) -> bool {
        let results = op.results(self.db);
        let returns_unit = results
            .first()
            .copied()
            .map(|ty| self.is_nil_type(ty))
            .unwrap_or(false);

        if (results.is_empty() || returns_unit)
            && self.is_print_line_call(op)
            && let Some(arg) = operands.first().copied()
            && let Some((ptr, len)) = self.get_literal_info(arg)
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
                .results(idvec![i32_ty])
                .attr("callee", Attribute::QualifiedName(callee))
                .build();
            let drop = Operation::of_name(self.db, location, "wasm.drop")
                .operands(idvec![call.result(self.db, 0)])
                .build();
            if !results.is_empty() {
                let old_result = op.result(self.db, 0);
                let replacement = nwritten_const.result(self.db, 0);
                self.value_map.insert(old_result, replacement);
            }

            builder.op(fd_const);
            builder.op(iovec_const);
            builder.op(iovec_len_const);
            builder.op(nwritten_const);
            builder.op(call);
            builder.op(drop);
            return true;
        }
        false
    }

    fn record_wasm_func_metadata(&mut self, op: &wasm::Func<'db>) {
        if op.sym_name(self.db) != Symbol::new("main") {
            return;
        }

        self.main_exports.saw_main = true;
        if let Some(func_ty) = core::Func::from_type(self.db, op.r#type(self.db)) {
            self.main_exports.main_result_type = Some(func_ty.result(self.db));
        }
    }

    fn observe_wasm_module_op(&mut self, op: &Operation<'db>, name: Symbol) {
        if name == wasm::MEMORY() {
            self.memory_plan.has_memory = true;
        } else if name == wasm::EXPORT_MEMORY() {
            self.memory_plan.has_exported_memory = true;
        } else if name == wasm::EXPORT_FUNC()
            && let Some(Attribute::String(export)) =
                op.attributes(self.db).get(&Symbol::new("name"))
            && export == "main"
        {
            self.main_exports.main_exported = true;
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
            .results(idvec![result_ty])
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

    fn map_results(&mut self, old_op: &Operation<'db>, new_op: &Operation<'db>) {
        let old_results = old_op.results(self.db);
        let new_results = new_op.results(self.db);
        let count = old_results.len().min(new_results.len());
        for i in 0..count {
            let old_val = old_op.result(self.db, i);
            let new_val = new_op.result(self.db, i);
            self.value_map.insert(old_val, new_val);
        }
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

//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This pass transforms mid-level IR operations (func, arith, scf, adt) to
//! wasm dialect operations. Phase 1 handles basic arithmetic and function calls.

use std::collections::HashMap;

use tracing::{error, warn};

use crate::plan::{MainExports, MemoryPlan};

use trunk_ir::DialectOp;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::src;
use trunk_ir::dialect::wasm;

use crate::passes::const_to_wasm::ConstAnalysis;
use crate::passes::intrinsic_to_wasm::IntrinsicAnalysis;
use trunk_ir::ir::BlockBuilder;
use trunk_ir::{
    Attribute, Block, DialectType, IdVec, Location, Operation, QualifiedName, Region, Symbol,
    Value, idvec,
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

    // Intrinsic analysis and lowering (print_line -> fd_write)
    let intrinsic_analysis = crate::passes::intrinsic_to_wasm::analyze_intrinsics(
        db,
        module,
        const_analysis.total_size(db),
    );
    let module = crate::passes::intrinsic_to_wasm::lower(db, module, intrinsic_analysis);

    // Phase 2: Remaining lowering via WasmLowerer
    let mut lowerer = WasmLowerer::new(db, const_analysis, intrinsic_analysis);
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
                // Skip src.var - kept for LSP hover support, no runtime effect
                if dialect == src::DIALECT_NAME() && op.name(db) == src::VAR() {
                    continue;
                }
                if dialect != Symbol::new("wasm") {
                    error!(
                        "Found non-wasm operation in function body: {}.{}",
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
    intrinsic_analysis: IntrinsicAnalysis<'db>,
    memory_plan: MemoryPlan,
    main_exports: MainExports<'db>,
}

impl<'db> WasmLowerer<'db> {
    fn new(
        db: &'db dyn salsa::Database,
        const_analysis: ConstAnalysis<'db>,
        intrinsic_analysis: IntrinsicAnalysis<'db>,
    ) -> Self {
        Self {
            db,
            value_map: HashMap::new(),
            module_location: None,
            const_analysis,
            intrinsic_analysis,
            memory_plan: MemoryPlan::new(),
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

        // Emit fd_write import if intrinsics need it
        if self.intrinsic_analysis.needs_fd_write(self.db) {
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

        // Check if any analysis has data that requires memory
        let const_size = self.const_analysis.total_size(self.db);
        let intrinsic_size = self.intrinsic_analysis.total_size(self.db);
        if const_size > 0 || intrinsic_size > 0 {
            self.memory_plan.needs_memory = true;
        }

        if self.memory_plan.needs_memory && !self.memory_plan.has_memory {
            // Calculate total data size: const analysis + intrinsic analysis
            let total_data_size = const_size + intrinsic_size;
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

        // Emit data segments from intrinsic analysis (iovec structures)
        for (ptr, len, offset) in self.intrinsic_analysis.iovec_allocations(self.db).iter() {
            let mut iovec_bytes = Vec::with_capacity(8);
            iovec_bytes.extend_from_slice(&ptr.to_le_bytes());
            iovec_bytes.extend_from_slice(&len.to_le_bytes());
            builder.op(wasm::data(
                self.db,
                module_location,
                Attribute::IntBits(u64::from(*offset)),
                Attribute::Bytes(iovec_bytes),
            ));
        }

        // Emit nwritten buffer if needed
        if let Some(nwritten_offset) = self.intrinsic_analysis.nwritten_offset(self.db) {
            builder.op(wasm::data(
                self.db,
                module_location,
                Attribute::IntBits(u64::from(nwritten_offset)),
                Attribute::Bytes(vec![0, 0, 0, 0]),
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

        if self.intrinsic_analysis.needs_fd_write(self.db) && self.main_exports.saw_main {
            builder.op(self.build_start_function(module_location));
            builder.op(wasm::export_func(
                self.db,
                module_location,
                Attribute::String("_start".into()),
                Attribute::QualifiedName(QualifiedName::simple(Symbol::new("_start"))),
            ));
        }
    }

    fn build_start_function(&self, location: Location<'db>) -> Operation<'db> {
        let nil_ty = core::Nil::new(self.db).as_type();

        // Normalize result type: nil has no runtime representation in wasm
        let main_result = self
            .main_exports
            .main_result_type
            .and_then(|ty| if ty == nil_ty { None } else { Some(ty) });

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

        // Drop result if main returns a value
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

        // Create wasm.func directly (not func.func) since we're past the func_to_wasm pass
        Operation::of_name(self.db, location, "wasm.func")
            .attr("sym_name", Attribute::Symbol(Symbol::new("_start")))
            .attr("type", Attribute::Type(func_ty))
            .region(region)
            .build()
    }

    fn lower_op(&mut self, builder: &mut BlockBuilder<'db>, op: Operation<'db>) {
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);

        // Skip src.var operations - they're kept for LSP hover support but have no runtime effect.
        // The resolver marks them as resolved_local and maps their results to actual values.
        if dialect == src::DIALECT_NAME() && name == src::VAR() {
            return;
        }

        // Handle wasm dialect metadata collection
        if dialect == wasm::DIALECT_NAME() {
            self.observe_wasm_module_op(&op, name);

            if name == wasm::FUNC() {
                if let Ok(op_func) = wasm::Func::from_operation(self.db, op) {
                    self.record_wasm_func_metadata(&op_func);
                }
            }
        }

        // Debug warning for unexpected dialects in function bodies
        // Note: Some dialects are allowed for edge cases:
        // - "func": func.call_indirect (closure support pending)
        // - "adt": adt.string_const (handled by const_to_wasm pass)
        // - "src": src.var operations kept for LSP (filtered above)
        if cfg!(debug_assertions) {
            let dialect_str = dialect.to_string();
            let allowed = ["wasm", "core", "type", "ty", "func", "adt", "case", "scf", "src"];
            if !allowed.contains(&dialect_str.as_str()) {
                warn!(
                    "Unhandled operation in lowering: {}.{} (this may cause emit errors)",
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

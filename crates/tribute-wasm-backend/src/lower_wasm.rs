//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This pass transforms mid-level IR operations (func, arith, scf, adt) to
//! wasm dialect operations. Phase 1 handles basic arithmetic and function calls.

use tracing::{error, warn};

use crate::plan::{MainExports, MemoryPlan};

use tribute_ir::ModulePathExt;
use tribute_ir::dialect::tribute;
use trunk_ir::DialectOp;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::RewriteContext;

use crate::gc_types::{STEP_IDX, STEP_TAG_DONE, step_marker_type};
use crate::passes::const_to_wasm::ConstAnalysis;
use crate::passes::intrinsic_to_wasm::IntrinsicAnalysis;
use trunk_ir::ir::BlockBuilder;
use trunk_ir::{
    Attribute, Block, DialectType, IdVec, Location, Operation, Region, Symbol, Type, Value, idvec,
};

/// Entry point for lowering mid-level IR to wasm dialect.
#[salsa::tracked]
pub fn lower_to_wasm<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // Phase 1: Pattern-based lowering passes
    tracing::debug!("=== BEFORE arith_to_wasm ===\n{:?}", module);
    let module = crate::passes::arith_to_wasm::lower(db, module);
    tracing::debug!("=== AFTER arith_to_wasm ===\n{:?}", module);
    let module = crate::passes::scf_to_wasm::lower(db, module);
    let module = crate::passes::func_to_wasm::lower(db, module);
    tracing::debug!("=== AFTER func_to_wasm ===\n{:?}", module);
    let module = crate::passes::closure_to_wasm::lower(db, module);

    // Convert trampoline global state ops to wasm ops
    let module = crate::passes::trampoline_to_wasm::lower(db, module);

    // Convert ALL adt ops to wasm (including those from trampoline_to_adt)
    let module = crate::passes::adt_to_wasm::lower(db, module);

    // Lower tribute_rt operations (box_int, unbox_int) to wasm operations
    let module = crate::passes::tribute_rt_to_wasm::lower(db, module);

    // Concretize type variables in wasm operations (resolve tribute.type_var)
    let module = crate::passes::wasm_type_concrete::lower(db, module);

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

    // Phase 3: Assign unique type_idx to GC struct operations before emit
    crate::passes::wasm_gc_type_assign::assign_gc_type_indices(db, lowered)
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
    ctx: RewriteContext<'db>,
    module_location: Option<Location<'db>>,
    const_analysis: ConstAnalysis<'db>,
    intrinsic_analysis: IntrinsicAnalysis<'db>,
    memory_plan: MemoryPlan,
    main_exports: MainExports<'db>,
    /// Whether the module uses continuations and needs yield globals.
    /// Currently always true since cont_to_trampoline runs in the main pipeline.
    has_continuations: bool,
}

impl<'db> WasmLowerer<'db> {
    fn new(
        db: &'db dyn salsa::Database,
        const_analysis: ConstAnalysis<'db>,
        intrinsic_analysis: IntrinsicAnalysis<'db>,
    ) -> Self {
        // Always emit yield globals - they have negligible overhead
        // and cont_to_trampoline runs in the main pipeline
        let has_continuations = true;

        Self {
            db,
            ctx: RewriteContext::new(),
            module_location: None,
            const_analysis,
            intrinsic_analysis,
            memory_plan: MemoryPlan::new(),
            main_exports: MainExports::new(),
            has_continuations,
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
            BlockBuilder::new(self.db, location).block_args(first_block.args(self.db).clone());

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
        let mut builder = BlockBuilder::new(self.db, location)
            .id(block.id(self.db))
            .block_args(args);

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
                Symbol::new("wasi_snapshot_preview1"),
                Symbol::new("fd_write"),
                Symbol::new("fd_write"),
                import_ty,
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
                required_pages,
                0,
                false,
                false,
            ));
            self.memory_plan.has_memory = true;
        }

        // Emit yield globals for continuation support
        // Note: $yield_value is no longer needed - shift_value is stored in continuation struct
        if self.has_continuations {
            // $yield_state: i32 (0 = normal, 1 = yielding)
            builder.op(wasm::global(
                self.db,
                module_location,
                Symbol::new("i32"),
                true,
                0,
            ));
            // $yield_tag: i32 (prompt tag being yielded to)
            builder.op(wasm::global(
                self.db,
                module_location,
                Symbol::new("i32"),
                true,
                0,
            ));
            // $yield_cont: anyref (captured continuation, GC-managed)
            builder.op(wasm::global(
                self.db,
                module_location,
                Symbol::new("anyref"),
                true,
                0,
            ));
            // $yield_op_idx: i32 (operation index within ability for multi-op dispatch)
            builder.op(wasm::global(
                self.db,
                module_location,
                Symbol::new("i32"),
                true,
                0,
            ));
        }
    }

    fn module_data_ops(&mut self, builder: &mut BlockBuilder<'db>, location: Location<'db>) {
        let module_location = self.module_location.unwrap_or(location);

        // Emit active data segments for string constants (linear memory)
        for (content, offset, _len) in self.const_analysis.string_allocations(self.db).iter() {
            builder.op(wasm::data(
                self.db,
                module_location,
                *offset as i32,
                Attribute::Bytes(content.clone()),
                false, // active segment
            ));
        }

        // Emit passive data segments for bytes constants (for array.new_data)
        for (content, _data_idx, _len) in self.const_analysis.bytes_allocations(self.db).iter() {
            builder.op(wasm::data(
                self.db,
                module_location,
                0, // offset not used for passive segments
                Attribute::Bytes(content.clone()),
                true, // passive segment
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
                *offset as i32,
                Attribute::Bytes(iovec_bytes),
                false, // active segment
            ));
        }

        // Emit nwritten buffer if needed
        if let Some(nwritten_offset) = self.intrinsic_analysis.nwritten_offset(self.db) {
            builder.op(wasm::data(
                self.db,
                module_location,
                nwritten_offset as i32,
                Attribute::Bytes(vec![0, 0, 0, 0]),
                false, // active segment
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
                "memory".into(),
                0,
            ));
            self.memory_plan.has_exported_memory = true;
        }

        if self.main_exports.saw_main && !self.main_exports.main_exported {
            // Check if main returns Step (effectful function)
            let main_returns_step = self
                .main_exports
                .main_result_type
                .map(|ty| self.is_step_type(ty))
                .unwrap_or(false);

            if main_returns_step {
                // Generate trampoline wrapper and export it instead
                builder.op(self.build_trampoline_main(module_location));
                builder.op(wasm::export_func(
                    self.db,
                    module_location,
                    "main".into(),
                    Symbol::new("main_trampoline"),
                ));
            } else {
                // Export main directly
                builder.op(wasm::export_func(
                    self.db,
                    module_location,
                    "main".into(),
                    Symbol::new("main"),
                ));
            }
            self.main_exports.main_exported = true;
        }

        if self.intrinsic_analysis.needs_fd_write(self.db) && self.main_exports.saw_main {
            builder.op(self.build_start_function(module_location));
            builder.op(wasm::export_func(
                self.db,
                module_location,
                "_start".into(),
                Symbol::new("_start"),
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
            Symbol::new("main"),
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
        wasm::func(self.db, location, Symbol::new("_start"), func_ty, region).as_operation()
    }

    /// Check if a type is the Step type (trampoline return type).
    fn is_step_type(&self, ty: Type<'db>) -> bool {
        ty == step_marker_type(self.db)
    }

    /// Build a trampoline wrapper function that unwraps Step values from main.
    ///
    /// The trampoline:
    /// 1. Calls the original main function (which returns Step)
    /// 2. Extracts the tag field from Step
    /// 3. If tag == DONE: extracts and returns the value (with proper unboxing)
    /// 4. If tag != DONE: traps (unhandled effect)
    fn build_trampoline_main(&self, location: Location<'db>) -> Operation<'db> {
        use tribute_ir::dialect::tribute_rt;

        let anyref_ty = wasm::Anyref::new(self.db).as_type();
        let i32_ty = core::I32::new(self.db).as_type();
        let nil_ty = core::Nil::new(self.db).as_type();
        let step_ty = step_marker_type(self.db);

        // Determine the actual return type from the original main signature
        let original_result_ty = self.main_exports.original_result_type.unwrap_or(anyref_ty);

        // Check for special return types
        let returns_nil = core::Nil::from_type(self.db, original_result_ty).is_some();
        let is_nat = tribute_rt::is_nat(self.db, original_result_ty);
        let needs_i32_unbox = tribute_rt::is_int(self.db, original_result_ty)
            || is_nat
            || core::I32::from_type(self.db, original_result_ty).is_some();

        // The trampoline's return type
        let trampoline_result_ty = if returns_nil {
            nil_ty
        } else if needs_i32_unbox {
            i32_ty
        } else {
            anyref_ty
        };

        let mut builder = BlockBuilder::new(self.db, location);

        // Call the original main function which returns Step
        let call_main = builder.op(wasm::call(
            self.db,
            location,
            None,
            Some(step_ty),
            Symbol::new("main"),
        ));
        let step_result = call_main.results(self.db).first().copied().unwrap();

        // Extract tag field (field 0) from Step
        let get_tag = builder.op(wasm::struct_get(
            self.db,
            location,
            step_result,
            i32_ty,
            STEP_IDX,
            0,
        ));
        let tag_val = get_tag.result(self.db);

        // Extract value field (field 1) from Step
        let get_value = builder.op(wasm::struct_get(
            self.db,
            location,
            step_result,
            anyref_ty,
            STEP_IDX,
            1,
        ));
        let value_val = get_value.result(self.db);

        // Compare tag with DONE (0)
        let done_const = builder.op(wasm::i32_const(self.db, location, i32_ty, STEP_TAG_DONE));
        let done_val = done_const.result(self.db);

        let cmp_eq = builder.op(wasm::i32_eq(self.db, location, tag_val, done_val, i32_ty));
        let is_done = cmp_eq.result(self.db);

        // If tag == DONE, return the value; otherwise trap
        // Build the if-else construct
        let then_block = {
            let mut then_builder = BlockBuilder::new(self.db, location);

            if returns_nil {
                // Nil return - no value to return
                then_builder.op(wasm::r#return(self.db, location, None));
            } else if needs_i32_unbox {
                // Unbox Int/Nat to i32
                // Cast anyref to i31ref and extract i32 (abstract type, no type_idx)
                let i31ref_ty = wasm::I31ref::new(self.db).as_type();
                let cast = then_builder.op(wasm::ref_cast(
                    self.db, location, value_val, i31ref_ty, i31ref_ty, None,
                ));
                let i31_val = cast.result(self.db);
                // Use unsigned extraction for Nat, signed for Int
                let return_val = if is_nat {
                    let unbox =
                        then_builder.op(wasm::i31_get_u(self.db, location, i31_val, i32_ty));
                    unbox.result(self.db)
                } else {
                    let unbox =
                        then_builder.op(wasm::i31_get_s(self.db, location, i31_val, i32_ty));
                    unbox.result(self.db)
                };
                then_builder.op(wasm::r#return(self.db, location, Some(return_val)));
            } else {
                // Return anyref directly
                then_builder.op(wasm::r#return(self.db, location, Some(value_val)));
            }

            then_builder.build()
        };

        let else_block = {
            let mut else_builder = BlockBuilder::new(self.db, location);
            // Unhandled effect - trap
            else_builder.op(wasm::unreachable(self.db, location));
            else_builder.build()
        };

        let then_region = Region::new(self.db, location, idvec![then_block]);
        let else_region = Region::new(self.db, location, idvec![else_block]);

        // Note: wasm.if returns a value when both branches return the same type
        // Since we return from both branches, we need a different approach
        // Use scf.if or just emit the if without result and rely on return
        builder.op(wasm::r#if(
            self.db,
            location,
            is_done,
            nil_ty, // no result type - branches use return
            then_region,
            else_region,
        ));

        // This point is unreachable, but we need to return something for validation
        builder.op(wasm::unreachable(self.db, location));

        let body_block = builder.build();
        let region = Region::new(self.db, location, idvec![body_block]);
        let func_ty = core::Func::new(self.db, idvec![], trampoline_result_ty).as_type();

        wasm::func(
            self.db,
            location,
            Symbol::new("main_trampoline"),
            func_ty,
            region,
        )
        .as_operation()
    }

    fn lower_op(&mut self, builder: &mut BlockBuilder<'db>, op: Operation<'db>) {
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);

        // Skip tribute dialect metadata operations - they have no runtime representation.
        if dialect == tribute::DIALECT_NAME() {
            // These are all metadata/definition ops that don't produce wasm code:
            // - var: variable references (resolved during name resolution)
            // - ability_def: ability type definitions
            // - enum_def, variant_def, field_def, struct_def: type definitions
            return;
        }

        // Handle wasm dialect metadata collection
        if dialect == wasm::DIALECT_NAME() {
            self.observe_wasm_module_op(&op, name);

            if let Ok(op_func) = wasm::Func::from_operation(self.db, op) {
                self.record_wasm_func_metadata(&op_func);
            }
        }

        // Debug warning for unexpected dialects in function bodies
        // Note: Some dialects are allowed for edge cases:
        // - "func": func.call_indirect (closure support pending)
        // - "adt": adt.string_const (handled by const_to_wasm pass)
        // - "tribute": tribute.var operations kept for LSP (filtered above)
        if cfg!(debug_assertions) {
            let dialect_str = dialect.to_string();
            let allowed = [
                "wasm",
                "core",
                "func",
                "adt",
                "scf",
                "tribute",
                "tribute_pat",
            ];
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
        let remapped_operands = self.remap_operand_values(op);
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
        self.ctx.map_results(self.db, &op, &new_op);
        builder.op(new_op);
    }

    fn record_wasm_func_metadata(&mut self, op: &wasm::Func<'db>) {
        let sym_name = op.sym_name(self.db);
        // Only match root-level main, not foo::main
        if !(sym_name.is_simple() && sym_name.last_segment() == Symbol::new("main")) {
            return;
        }

        self.main_exports.saw_main = true;
        if let Some(func_ty) = core::Func::from_type(self.db, op.r#type(self.db)) {
            self.main_exports.main_result_type = Some(func_ty.result(self.db));
        }

        // Read original_result_type attribute if present (set by cont_to_wasm for effectful functions)
        if let Some(Attribute::Type(original_ty)) = op
            .as_operation()
            .attributes(self.db)
            .get(&Symbol::new("original_result_type"))
        {
            self.main_exports.original_result_type = Some(*original_ty);
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

    fn remap_operand_values(&self, op: Operation<'db>) -> IdVec<Value<'db>> {
        let mut operands = IdVec::new();
        for &operand in op.operands(self.db).iter() {
            operands.push(self.ctx.lookup(operand));
        }
        operands
    }
}

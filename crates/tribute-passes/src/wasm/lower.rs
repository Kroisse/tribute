//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This pass transforms mid-level IR operations (func, arith, scf, adt) to
//! wasm dialect operations. Uses arena IR for in-place mutation within a
//! single arena session.

use tracing::{error, warn};
use tribute_ir::ModulePathExt;
use trunk_ir::DialectOp;
use trunk_ir::arena::context::{BlockData, IrContext, RegionData};
use trunk_ir::arena::dialect::func as arena_func;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::arena::rewrite::{
    ArenaModule, PatternApplicator, WasmFuncSignatureConversionPattern,
};
use trunk_ir::arena::types::{Attribute as ArenaAttribute, Location, TypeDataBuilder};
use trunk_ir::ir::Symbol;
use trunk_ir::smallvec::smallvec;

use super::const_to_wasm::ConstAnalysis;
use super::intrinsic_to_wasm::IntrinsicAnalysis;
use super::type_converter::{self, wasm_type_converter};
use trunk_ir::arena::bridge::{export_to_salsa, import_salsa_module};
use trunk_ir::dialect::core::Module;
use trunk_ir_wasm_backend::gc_types::{STEP_IDX, STEP_TAG_DONE};

/// Entry point for lowering mid-level IR to wasm dialect.
#[salsa::tracked]
#[tracing::instrument(skip_all)]
pub fn lower_to_wasm<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // Import Salsa module into a fresh arena context
    let (mut ctx, arena_module) = import_salsa_module(db, module.operation());

    // Run the entire WASM lowering pipeline in a single arena session
    lower_to_wasm_arena(&mut ctx, arena_module);

    // Export back to Salsa
    let salsa_op = export_to_salsa(db, &ctx, arena_module);
    Module::from_operation(db, salsa_op).expect("exported operation should be core.module")
}

/// Run the full WASM lowering pipeline on arena IR.
fn lower_to_wasm_arena(ctx: &mut IrContext, module: ArenaModule) {
    // Phase 1: Pattern-based lowering passes (using trunk-ir-wasm-backend)
    {
        let _span = tracing::info_span!("arith_to_wasm").entered();
        let tc = wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::arith_to_wasm::lower(ctx, module, tc);
    }
    {
        let _span = tracing::info_span!("scf_to_wasm").entered();
        let tc = wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::scf_to_wasm::lower(ctx, module, tc);
    }

    // Normalize tribute_rt primitive types (int, nat, bool, float) to core types
    // BEFORE trampoline_to_wasm so downstream passes don't need to handle tribute_rt
    {
        let _span = tracing::info_span!("normalize_primitive_types").entered();
        super::normalize_primitive_types::lower(ctx, module);
    }

    // Convert trampoline types/ops BEFORE func_to_wasm so function signatures
    // have ADT types (not trampoline.Step) when converted to wasm.func
    {
        let _span = tracing::info_span!("trampoline_to_wasm").entered();
        trunk_ir_wasm_backend::passes::trampoline_to_wasm::lower(ctx, module);
    }

    {
        let _span = tracing::info_span!("func_to_wasm").entered();
        let tc = wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::func_to_wasm::lower(ctx, module, tc);
    }
    debug_func_params(ctx, module, "after func_to_wasm");

    // Convert wasm.func signature types (e.g., core.array(Marker) → wasm.arrayref)
    // This must run AFTER func_to_wasm so wasm.func operations exist, and BEFORE
    // emit so function type attributes have WASM-level types.
    {
        let _span = tracing::info_span!("wasm_func_signature_conversion").entered();
        let tc = wasm_type_converter(ctx);
        PatternApplicator::new(tc)
            .add_pattern(WasmFuncSignatureConversionPattern)
            .apply_partial(ctx, module);
    }
    debug_func_params(ctx, module, "after wasm_func_signature_conversion");

    // Convert ALL adt ops to wasm (including those from trampoline_to_adt)
    {
        let _span = tracing::info_span!("adt_to_wasm").entered();
        let tc = wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::adt_to_wasm::lower(ctx, module, tc);
    }
    debug_func_params(ctx, module, "after adt_to_wasm");

    // Lower tribute_rt operations (box_int, unbox_int) to wasm operations
    {
        let _span = tracing::info_span!("tribute_rt_to_wasm").entered();
        super::tribute_rt_to_wasm::lower(ctx, module);
    }
    debug_func_params(ctx, module, "after tribute_rt_to_wasm");

    // Lower evidence runtime function stubs (prepare for inline WASM operations)
    {
        let _span = tracing::info_span!("evidence_to_wasm").entered();
        super::evidence_to_wasm::lower_evidence_to_wasm(ctx, module);
    }

    // Const analysis and lowering (string/bytes constants to data segments)
    let const_analysis = {
        let _span = tracing::info_span!("const_to_wasm").entered();
        let const_analysis = super::const_to_wasm::analyze_consts(ctx, module);
        super::const_to_wasm::lower(ctx, module, &const_analysis);
        const_analysis
    };

    // Intrinsic analysis and lowering (print_line -> fd_write)
    let intrinsic_analysis =
        super::intrinsic_to_wasm::analyze_intrinsics(ctx, module, const_analysis.total_size());
    {
        let _span = tracing::info_span!("intrinsic_to_wasm").entered();
        super::intrinsic_to_wasm::lower(ctx, module, &intrinsic_analysis);
    }

    // Phase 2: Module-level operations via WasmLowerer (in-place)
    {
        let _span = tracing::info_span!("wasm_lowerer").entered();
        let mut lowerer = WasmLowerer::new(&const_analysis, &intrinsic_analysis);
        lowerer.lower_module(ctx, module);
    }

    // Debug: Verify all operations are now in wasm dialect
    if cfg!(debug_assertions) {
        check_all_wasm_dialect(ctx, module);
    }

    // Phase 3: Assign unique type_idx to GC struct operations before emit
    {
        let _span = tracing::info_span!("assign_gc_type_indices").entered();
        super::wasm_gc_type_assign::assign_gc_type_indices(ctx, module);
    }
}

// =============================================================================
// Debug helpers
// =============================================================================

/// Debug helper to check if all operations in function bodies are in wasm dialect
fn check_all_wasm_dialect(ctx: &IrContext, module: ArenaModule) {
    let Some(body) = module.body(ctx) else {
        return;
    };
    for &block in ctx.region(body).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            let data = ctx.op(op);
            if data.dialect == arena_wasm::DIALECT_NAME() && data.name == Symbol::new("func") {
                check_function_body(ctx, op);
            }
        }
    }
}

/// Debug helper to trace function parameter types through the pipeline.
fn debug_func_params(ctx: &IrContext, module: ArenaModule, phase: &str) {
    let Some(body) = module.body(ctx) else {
        return;
    };
    for &block in ctx.region(body).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            let data = ctx.op(op);
            // Check for func.func or wasm.func operations
            if data.dialect == arena_func::DIALECT_NAME() && data.name == Symbol::new("func") {
                if let Some(ArenaAttribute::Type(fn_ty)) = data.attributes.get(&Symbol::new("type"))
                {
                    let fn_data = ctx.types.get(*fn_ty);
                    let params: Vec<_> = fn_data
                        .params
                        .iter()
                        .map(|t| {
                            let td = ctx.types.get(*t);
                            format!("{}.{}", td.dialect, td.name)
                        })
                        .collect();
                    let sym_name = data
                        .attributes
                        .get(&Symbol::new("sym_name"))
                        .map(|a| format!("{a:?}"))
                        .unwrap_or_default();
                    tracing::debug!("[{phase}] func.func {sym_name}: params={params:?}");
                }
            } else if data.dialect == arena_wasm::DIALECT_NAME()
                && data.name == Symbol::new("func")
                && let Some(ArenaAttribute::Type(fn_ty)) = data.attributes.get(&Symbol::new("type"))
            {
                let fn_data = ctx.types.get(*fn_ty);
                let params: Vec<_> = fn_data
                    .params
                    .iter()
                    .map(|t| {
                        let td = ctx.types.get(*t);
                        format!("{}.{}", td.dialect, td.name)
                    })
                    .collect();
                let sym_name = data
                    .attributes
                    .get(&Symbol::new("sym_name"))
                    .map(|a| format!("{a:?}"))
                    .unwrap_or_default();
                tracing::debug!("[{phase}] wasm.func {sym_name}: params={params:?}");
            }
        }
    }
}

/// Check if all operations in a function body are in wasm dialect
fn check_function_body(ctx: &IrContext, func_op: OpRef) {
    let regions = &ctx.op(func_op).regions;
    if let Some(&body_region) = regions.first() {
        for &block in ctx.region(body_region).blocks.iter() {
            for &op in ctx.block(block).ops.iter() {
                let dialect = ctx.op(op).dialect;
                if dialect != Symbol::new("wasm") {
                    error!(
                        "Found non-wasm operation in function body: {}.{}",
                        dialect,
                        ctx.op(op).name
                    );
                }
            }
        }
    }
}

// =============================================================================
// Type helpers
// =============================================================================

fn intern_type(ctx: &mut IrContext, dialect: &'static str, name: &'static str) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new(dialect), Symbol::new(name)).build())
}

fn intern_func_type(ctx: &mut IrContext, params: Vec<TypeRef>, result: TypeRef) -> TypeRef {
    // core.func layout: params[0] = return type, params[1..] = param types (matches Salsa convention)
    let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).param(result);
    for p in params {
        builder = builder.param(p);
    }
    ctx.types.intern(builder.build())
}

fn is_type(ctx: &IrContext, ty: TypeRef, dialect: &'static str, name: &'static str) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new(dialect) && data.name == Symbol::new(name)
}

// =============================================================================
// MainExports (arena version)
// =============================================================================

/// Tracks whether the `main` function was found and its type info.
struct MainExports {
    saw_main: bool,
    main_result_type: Option<TypeRef>,
    original_result_type: Option<TypeRef>,
    main_exported: bool,
}

impl MainExports {
    fn new() -> Self {
        Self {
            saw_main: false,
            main_result_type: None,
            original_result_type: None,
            main_exported: false,
        }
    }
}

// =============================================================================
// MemoryPlan (arena version)
// =============================================================================

struct ArenaMemoryPlan {
    has_memory: bool,
    has_exported_memory: bool,
    needs_memory: bool,
}

impl ArenaMemoryPlan {
    fn new() -> Self {
        Self {
            has_memory: false,
            has_exported_memory: false,
            needs_memory: false,
        }
    }

    fn required_pages(&self, end_offset: u32) -> u32 {
        std::cmp::max(1, end_offset.div_ceil(0x10000))
    }
}

// =============================================================================
// WasmLowerer (arena version)
// =============================================================================

/// Lowers mid-level IR to wasm dialect operations (arena version).
///
/// In the arena version, the lowerer does NOT walk/rebuild the entire IR tree.
/// Instead it:
/// 1. Scans module ops to collect metadata (main function info, memory presence)
/// 2. Inserts preamble ops (imports, memory, globals) at the front of the module
/// 3. Appends data segment ops and module-level extras (exports, _start function)
struct WasmLowerer<'a> {
    const_analysis: &'a ConstAnalysis,
    intrinsic_analysis: &'a IntrinsicAnalysis,
    memory_plan: ArenaMemoryPlan,
    main_exports: MainExports,
    has_continuations: bool,
}

impl<'a> WasmLowerer<'a> {
    fn new(const_analysis: &'a ConstAnalysis, intrinsic_analysis: &'a IntrinsicAnalysis) -> Self {
        Self {
            const_analysis,
            intrinsic_analysis,
            memory_plan: ArenaMemoryPlan::new(),
            main_exports: MainExports::new(),
            has_continuations: true,
        }
    }

    fn lower_module(&mut self, ctx: &mut IrContext, module: ArenaModule) {
        let Some(body) = module.body(ctx) else {
            return;
        };

        let module_block = module
            .first_block(ctx)
            .expect("module should have a body block");
        let location = ctx.op(module.op()).location;

        // Phase 1: Scan existing ops to collect metadata
        self.scan_module_ops(ctx, body);

        // Phase 2: Build and insert preamble ops at the front
        let first_existing_op = ctx.block(module_block).ops.first().copied();
        self.insert_preamble_ops(ctx, module_block, first_existing_op, location);

        // Phase 3: Append data segment and extra ops at the end
        self.append_data_ops(ctx, module_block, location);
        self.append_extra_ops(ctx, module_block, location);
    }

    /// Scan all module-level ops to collect metadata about main function,
    /// memory declarations, and exports.
    fn scan_module_ops(&mut self, ctx: &IrContext, body: RegionRef) {
        for &block in ctx.region(body).blocks.iter() {
            for &op in ctx.block(block).ops.iter() {
                let data = ctx.op(op);

                if data.dialect == arena_wasm::DIALECT_NAME() {
                    // Track wasm module-level metadata
                    if data.name == Symbol::new("memory") {
                        self.memory_plan.has_memory = true;
                    } else if data.name == Symbol::new("export_memory") {
                        self.memory_plan.has_exported_memory = true;
                    } else if data.name == Symbol::new("export_func")
                        && let Some(ArenaAttribute::String(name)) =
                            data.attributes.get(&Symbol::new("name"))
                    {
                        if name == "main" {
                            self.main_exports.main_exported = true;
                        }
                    } else if data.name == Symbol::new("func") {
                        self.scan_wasm_func(ctx, op);
                    }
                }

                // Debug: warn about unexpected dialects
                if cfg!(debug_assertions) {
                    let dialect_str = data.dialect.to_string();
                    let allowed = ["wasm", "core"];
                    if !allowed.contains(&dialect_str.as_str()) {
                        warn!(
                            "Unhandled operation in lowering: {}.{} (this may cause emit errors)",
                            data.dialect, data.name
                        );
                    }
                }
            }
        }
    }

    /// Check if a wasm.func op is the main function and record its metadata.
    fn scan_wasm_func(&mut self, ctx: &IrContext, op: OpRef) {
        let data = ctx.op(op);
        let Some(ArenaAttribute::Symbol(sym_name)) = data.attributes.get(&Symbol::new("sym_name"))
        else {
            return;
        };

        // Only match root-level main, not foo::main
        if !(sym_name.is_simple() && sym_name.last_segment() == Symbol::new("main")) {
            return;
        }

        self.main_exports.saw_main = true;

        if let Some(ArenaAttribute::Type(fn_ty)) = data.attributes.get(&Symbol::new("type")) {
            let fn_data = ctx.types.get(*fn_ty);
            // In arena core.func type, params[0] is the return type
            if let Some(&result_ty) = fn_data.params.first() {
                self.main_exports.main_result_type = Some(result_ty);
            }
        }

        // Read original_result_type attribute if present
        if let Some(ArenaAttribute::Type(original_ty)) =
            data.attributes.get(&Symbol::new("original_result_type"))
        {
            self.main_exports.original_result_type = Some(*original_ty);
        }
    }

    /// Size of the runtime print buffer used by `_start` for itoa + fd_write.
    const PRINT_BUF_TOTAL: u32 = 32;

    /// Base offset for `_start`'s runtime print buffer in linear memory.
    fn start_buf_base(&self) -> u32 {
        let end = self.const_analysis.total_size() + self.intrinsic_analysis.total_size;
        (end + 3) & !3
    }

    /// Insert preamble ops (imports, memory, globals) before existing module ops.
    fn insert_preamble_ops(
        &mut self,
        ctx: &mut IrContext,
        module_block: BlockRef,
        insert_before: Option<OpRef>,
        location: Location,
    ) {
        let mut preamble_ops: Vec<OpRef> = Vec::new();

        // Emit fd_write import if intrinsics need it OR if main exists (_start needs it)
        if self.intrinsic_analysis.needs_fd_write || self.main_exports.saw_main {
            let i32_ty = intern_type(ctx, "core", "i32");
            let import_ty = intern_func_type(ctx, vec![i32_ty, i32_ty, i32_ty, i32_ty], i32_ty);
            let op = arena_wasm::import_func(
                ctx,
                location,
                Symbol::new("wasi_snapshot_preview1"),
                Symbol::new("fd_write"),
                Symbol::new("fd_write"),
                import_ty,
            );
            preamble_ops.push(op.op_ref());
        }

        // Check if memory is needed
        let const_size = self.const_analysis.total_size();
        let intrinsic_size = self.intrinsic_analysis.total_size;
        let start_buf_size = if self.main_exports.saw_main {
            Self::PRINT_BUF_TOTAL
        } else {
            0
        };
        if const_size > 0 || intrinsic_size > 0 || start_buf_size > 0 {
            self.memory_plan.needs_memory = true;
        }

        if self.memory_plan.needs_memory && !self.memory_plan.has_memory {
            let total_data_size = const_size + intrinsic_size + start_buf_size;
            let required_pages = self.memory_plan.required_pages(total_data_size);
            let op = arena_wasm::memory(ctx, location, required_pages, 0, false, false);
            preamble_ops.push(op.op_ref());
            self.memory_plan.has_memory = true;
        }

        // Emit yield globals for continuation support
        if self.has_continuations {
            // Index 0 ($yield_state): i32
            let g0 = arena_wasm::global(
                ctx,
                location,
                Symbol::new("i32"),
                true,
                ArenaAttribute::IntBits(0),
            );
            preamble_ops.push(g0.op_ref());
            // Index 1 ($yield_tag): i32
            let g1 = arena_wasm::global(
                ctx,
                location,
                Symbol::new("i32"),
                true,
                ArenaAttribute::IntBits(0),
            );
            preamble_ops.push(g1.op_ref());
            // Index 2 ($yield_cont): anyref
            let g2 = arena_wasm::global(
                ctx,
                location,
                Symbol::new("anyref"),
                true,
                ArenaAttribute::IntBits(0),
            );
            preamble_ops.push(g2.op_ref());
            // Index 3 ($yield_op_idx): i32
            let g3 = arena_wasm::global(
                ctx,
                location,
                Symbol::new("i32"),
                true,
                ArenaAttribute::IntBits(0),
            );
            preamble_ops.push(g3.op_ref());
        }

        // Insert all preamble ops before the first existing op
        for op in preamble_ops {
            if let Some(before) = insert_before {
                ctx.insert_op_before(module_block, before, op);
            } else {
                ctx.push_op(module_block, op);
            }
        }
    }

    /// Append data segment ops at the end of the module block.
    fn append_data_ops(&self, ctx: &mut IrContext, module_block: BlockRef, location: Location) {
        // Emit active data segments for string constants (linear memory)
        for (content, offset, _len) in self.const_analysis.string_allocations.iter() {
            let op = arena_wasm::data(
                ctx,
                location,
                *offset,
                ArenaAttribute::Bytes(content.as_slice().into()),
                false,
            );
            ctx.push_op(module_block, op.op_ref());
        }

        // Emit passive data segments for bytes constants (for array.new_data)
        for (content, _data_idx, _len) in self.const_analysis.bytes_allocations.iter() {
            let op = arena_wasm::data(
                ctx,
                location,
                0,
                ArenaAttribute::Bytes(content.as_slice().into()),
                true,
            );
            ctx.push_op(module_block, op.op_ref());
        }

        // Emit data segments from intrinsic analysis (iovec structures)
        for (ptr, len, offset) in self.intrinsic_analysis.iovec_allocations.iter() {
            let mut iovec_bytes = Vec::with_capacity(8);
            iovec_bytes.extend_from_slice(&ptr.to_le_bytes());
            iovec_bytes.extend_from_slice(&len.to_le_bytes());
            let op = arena_wasm::data(
                ctx,
                location,
                *offset,
                ArenaAttribute::Bytes(iovec_bytes.as_slice().into()),
                false,
            );
            ctx.push_op(module_block, op.op_ref());
        }

        // Emit nwritten buffer if needed
        if let Some(nwritten_offset) = self.intrinsic_analysis.nwritten_offset {
            let op = arena_wasm::data(
                ctx,
                location,
                nwritten_offset,
                ArenaAttribute::Bytes(smallvec![0, 0, 0, 0]),
                false,
            );
            ctx.push_op(module_block, op.op_ref());
        }
    }

    /// Append module-level extra ops (exports, _start function).
    fn append_extra_ops(
        &mut self,
        ctx: &mut IrContext,
        module_block: BlockRef,
        location: Location,
    ) {
        if self.memory_plan.needs_memory
            && self.memory_plan.has_memory
            && !self.memory_plan.has_exported_memory
        {
            let op = arena_wasm::export_memory(ctx, location, "memory".into(), 0);
            ctx.push_op(module_block, op.op_ref());
            self.memory_plan.has_exported_memory = true;
        }

        if self.main_exports.saw_main && !self.main_exports.main_exported {
            let op = arena_wasm::export_func(ctx, location, "main".into(), Symbol::new("main"));
            ctx.push_op(module_block, op.op_ref());
            self.main_exports.main_exported = true;
        }

        if self.main_exports.saw_main {
            let start_func = self.build_start_function(ctx, location);
            ctx.push_op(module_block, start_func);

            let export_op =
                arena_wasm::export_func(ctx, location, "_start".into(), Symbol::new("_start"));
            ctx.push_op(module_block, export_op.op_ref());
        }
    }

    /// Build the `_start` function that calls main and handles the result.
    fn build_start_function(&self, ctx: &mut IrContext, location: Location) -> OpRef {
        let i32_ty = intern_type(ctx, "core", "i32");
        let nil_ty = intern_type(ctx, "core", "nil");

        let main_returns_step = self
            .main_exports
            .main_result_type
            .map(|ty| is_type(ctx, ty, "adt", "struct") && is_step_adt(ctx, ty))
            .unwrap_or(false);

        // Build the body block
        let body_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        if main_returns_step {
            self.build_start_step_body(ctx, body_block, location, i32_ty, nil_ty);
        } else {
            self.build_start_simple_body(ctx, body_block, location, i32_ty, nil_ty);
        }

        let body_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![body_block],
            parent_op: None,
        });

        let func_ty = intern_func_type(ctx, vec![], nil_ty);
        let func_op = arena_wasm::func(ctx, location, Symbol::new("_start"), func_ty, body_region);
        func_op.op_ref()
    }

    /// Build _start body for main returning Step type.
    fn build_start_step_body(
        &self,
        ctx: &mut IrContext,
        body_block: BlockRef,
        location: Location,
        i32_ty: TypeRef,
        nil_ty: TypeRef,
    ) {
        let step_ty = type_converter::step_adt_type(ctx);
        let anyref_ty = intern_type(ctx, "wasm", "anyref");

        let orig_ty = self.main_exports.original_result_type;
        let is_int_like = orig_ty
            .map(|ty| {
                is_type(ctx, ty, "tribute_rt", "int")
                    || is_type(ctx, ty, "tribute_rt", "nat")
                    || is_type(ctx, ty, "core", "i32")
            })
            .unwrap_or(false);
        let is_nat = orig_ty
            .map(|ty| is_type(ctx, ty, "tribute_rt", "nat"))
            .unwrap_or(false);
        let returns_nil = orig_ty
            .map(|ty| is_type(ctx, ty, "core", "nil"))
            .unwrap_or(false);

        // Call main — returns Step
        let call_main = arena_wasm::call(ctx, location, vec![], vec![step_ty], Symbol::new("main"));
        ctx.push_op(body_block, call_main.op_ref());
        let step_result = call_main.results(ctx)[0];

        // Extract tag field (field 0) from Step
        let get_tag = arena_wasm::struct_get(ctx, location, step_result, i32_ty, STEP_IDX, 0);
        ctx.push_op(body_block, get_tag.op_ref());
        let tag_val = get_tag.result(ctx);

        // Compare tag with DONE (0)
        let done_const = arena_wasm::i32_const(ctx, location, i32_ty, STEP_TAG_DONE);
        ctx.push_op(body_block, done_const.op_ref());
        let cmp_eq = arena_wasm::i32_eq(ctx, location, tag_val, done_const.result(ctx), i32_ty);
        ctx.push_op(body_block, cmp_eq.op_ref());
        let is_done = cmp_eq.result(ctx);

        // Build then (Done) branch
        let then_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        if !returns_nil {
            // Extract value field (field 1) from Step
            let get_value =
                arena_wasm::struct_get(ctx, location, step_result, anyref_ty, STEP_IDX, 1);
            ctx.push_op(then_block, get_value.op_ref());
            let value_val = get_value.result(ctx);

            if is_int_like {
                // Unbox to i32 and print
                let i31ref_ty = intern_type(ctx, "wasm", "i31ref");
                let cast = arena_wasm::ref_cast(
                    ctx,
                    location,
                    value_val,
                    i31ref_ty,
                    Symbol::new("i31ref"),
                    None,
                );
                ctx.push_op(then_block, cast.op_ref());

                let unbox_val = if is_nat {
                    let get_u = arena_wasm::i31_get_u(ctx, location, cast.result(ctx), i32_ty);
                    ctx.push_op(then_block, get_u.op_ref());
                    get_u.result(ctx)
                } else {
                    let get_s = arena_wasm::i31_get_s(ctx, location, cast.result(ctx), i32_ty);
                    ctx.push_op(then_block, get_s.op_ref());
                    get_s.result(ctx)
                };
                self.build_print_i32_body(ctx, then_block, location, unbox_val);
            } else {
                // Drop the anyref value
                let drop_op = arena_wasm::drop(ctx, location, value_val);
                ctx.push_op(then_block, drop_op.op_ref());
            }
        }

        let ret_then = arena_wasm::r#return(ctx, location, vec![]);
        ctx.push_op(then_block, ret_then.op_ref());

        // Build else (unhandled effect) branch — trap
        let else_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let trap = arena_wasm::unreachable(ctx, location);
        ctx.push_op(else_block, trap.op_ref());

        let then_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![then_block],
            parent_op: None,
        });
        let else_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![else_block],
            parent_op: None,
        });

        let if_op = arena_wasm::r#if(ctx, location, is_done, nil_ty, then_region, else_region);
        ctx.push_op(body_block, if_op.op_ref());

        let trap_end = arena_wasm::unreachable(ctx, location);
        ctx.push_op(body_block, trap_end.op_ref());
    }

    /// Build _start body for main NOT returning Step.
    fn build_start_simple_body(
        &self,
        ctx: &mut IrContext,
        body_block: BlockRef,
        location: Location,
        _i32_ty: TypeRef,
        nil_ty: TypeRef,
    ) {
        let result_ty = self
            .main_exports
            .main_result_type
            .and_then(|ty| if ty == nil_ty { None } else { Some(ty) });
        let result_types: Vec<TypeRef> = result_ty.into_iter().collect();
        let is_i32 = result_ty
            .map(|ty| is_type(ctx, ty, "core", "i32"))
            .unwrap_or(false);

        let call = arena_wasm::call(ctx, location, vec![], result_types, Symbol::new("main"));
        ctx.push_op(body_block, call.op_ref());

        if is_i32 {
            let result_val = call.results(ctx)[0];
            self.build_print_i32_body(ctx, body_block, location, result_val);
        } else if result_ty.is_some() {
            let result_val = call.results(ctx)[0];
            let drop_op = arena_wasm::drop(ctx, location, result_val);
            ctx.push_op(body_block, drop_op.op_ref());
        }

        let ret = arena_wasm::r#return(ctx, location, vec![]);
        ctx.push_op(body_block, ret.op_ref());
    }

    /// Build IR operations to print an i32 value as decimal to stdout via fd_write.
    fn build_print_i32_body(
        &self,
        ctx: &mut IrContext,
        block: BlockRef,
        location: Location,
        result_val: ValueRef,
    ) {
        let i32_ty = intern_type(ctx, "core", "i32");
        let nil_ty = intern_type(ctx, "core", "nil");

        // Memory layout offsets
        let buf_base = self.start_buf_base();
        let scratch_rem = buf_base;
        let scratch_pos = buf_base + 4;
        let buf_end = buf_base + 8 + 11;
        let iovec_off = buf_base + 20;
        let nwr_off = buf_base + 28;

        let align_i32: u32 = 2;
        let align_i8: u32 = 0;

        // === Step 1: Handle sign — compute absolute value ===
        let zero = arena_wasm::i32_const(ctx, location, i32_ty, 0);
        ctx.push_op(block, zero.op_ref());
        let zero_val = zero.result(ctx);

        let is_neg = arena_wasm::i32_lt_s(ctx, location, result_val, zero_val, i32_ty);
        ctx.push_op(block, is_neg.op_ref());
        let is_neg_val = is_neg.result(ctx);

        // if (result < 0) then (0 - result) else result → abs_value
        let then_abs_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let neg = arena_wasm::i32_sub(ctx, location, zero_val, result_val, i32_ty);
        ctx.push_op(then_abs_block, neg.op_ref());
        let yield_neg = arena_wasm::r#yield(ctx, location, neg.result(ctx));
        ctx.push_op(then_abs_block, yield_neg.op_ref());

        let else_abs_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let yield_pos = arena_wasm::r#yield(ctx, location, result_val);
        ctx.push_op(else_abs_block, yield_pos.op_ref());

        let then_abs_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![then_abs_block],
            parent_op: None,
        });
        let else_abs_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![else_abs_block],
            parent_op: None,
        });
        let if_abs = arena_wasm::r#if(
            ctx,
            location,
            is_neg_val,
            i32_ty,
            then_abs_region,
            else_abs_region,
        );
        ctx.push_op(block, if_abs.op_ref());
        let abs_value = if_abs.result(ctx);

        // === Step 2: Initialize scratch memory ===
        let rem_addr = arena_wasm::i32_const(ctx, location, i32_ty, scratch_rem as i32);
        ctx.push_op(block, rem_addr.op_ref());
        let store_rem = arena_wasm::i32_store(
            ctx,
            location,
            rem_addr.result(ctx),
            abs_value,
            0,
            align_i32,
            0,
        );
        ctx.push_op(block, store_rem.op_ref());

        let pos_addr_c = arena_wasm::i32_const(ctx, location, i32_ty, scratch_pos as i32);
        ctx.push_op(block, pos_addr_c.op_ref());
        let buf_end_c = arena_wasm::i32_const(ctx, location, i32_ty, buf_end as i32);
        ctx.push_op(block, buf_end_c.op_ref());
        let store_pos = arena_wasm::i32_store(
            ctx,
            location,
            pos_addr_c.result(ctx),
            buf_end_c.result(ctx),
            0,
            align_i32,
            0,
        );
        ctx.push_op(block, store_pos.op_ref());

        // === Step 3: Do-while digit extraction loop ===
        let loop_body_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        {
            let b = loop_body_block;

            let rem_addr_l = arena_wasm::i32_const(ctx, location, i32_ty, scratch_rem as i32);
            ctx.push_op(b, rem_addr_l.op_ref());
            let rem = arena_wasm::i32_load(
                ctx,
                location,
                rem_addr_l.result(ctx),
                i32_ty,
                0,
                align_i32,
                0,
            );
            ctx.push_op(b, rem.op_ref());

            let pos_addr_l = arena_wasm::i32_const(ctx, location, i32_ty, scratch_pos as i32);
            ctx.push_op(b, pos_addr_l.op_ref());
            let pos = arena_wasm::i32_load(
                ctx,
                location,
                pos_addr_l.result(ctx),
                i32_ty,
                0,
                align_i32,
                0,
            );
            ctx.push_op(b, pos.op_ref());

            let ten = arena_wasm::i32_const(ctx, location, i32_ty, 10);
            ctx.push_op(b, ten.op_ref());
            let digit =
                arena_wasm::i32_rem_u(ctx, location, rem.result(ctx), ten.result(ctx), i32_ty);
            ctx.push_op(b, digit.op_ref());
            let next =
                arena_wasm::i32_div_u(ctx, location, rem.result(ctx), ten.result(ctx), i32_ty);
            ctx.push_op(b, next.op_ref());

            let ascii_0 = arena_wasm::i32_const(ctx, location, i32_ty, 48);
            ctx.push_op(b, ascii_0.op_ref());
            let ascii = arena_wasm::i32_add(
                ctx,
                location,
                digit.result(ctx),
                ascii_0.result(ctx),
                i32_ty,
            );
            ctx.push_op(b, ascii.op_ref());
            let store8 = arena_wasm::i32_store8(
                ctx,
                location,
                pos.result(ctx),
                ascii.result(ctx),
                0,
                align_i8,
                0,
            );
            ctx.push_op(b, store8.op_ref());

            let rem_addr_s = arena_wasm::i32_const(ctx, location, i32_ty, scratch_rem as i32);
            ctx.push_op(b, rem_addr_s.op_ref());
            let store_next = arena_wasm::i32_store(
                ctx,
                location,
                rem_addr_s.result(ctx),
                next.result(ctx),
                0,
                align_i32,
                0,
            );
            ctx.push_op(b, store_next.op_ref());

            let one = arena_wasm::i32_const(ctx, location, i32_ty, 1);
            ctx.push_op(b, one.op_ref());
            let new_pos =
                arena_wasm::i32_sub(ctx, location, pos.result(ctx), one.result(ctx), i32_ty);
            ctx.push_op(b, new_pos.op_ref());
            let pos_addr_s = arena_wasm::i32_const(ctx, location, i32_ty, scratch_pos as i32);
            ctx.push_op(b, pos_addr_s.op_ref());
            let store_pos2 = arena_wasm::i32_store(
                ctx,
                location,
                pos_addr_s.result(ctx),
                new_pos.result(ctx),
                0,
                align_i32,
                0,
            );
            ctx.push_op(b, store_pos2.op_ref());

            let br_if = arena_wasm::br_if(ctx, location, next.result(ctx), 0);
            ctx.push_op(b, br_if.op_ref());
        }

        let loop_body_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![loop_body_block],
            parent_op: None,
        });
        let loop_op = arena_wasm::r#loop(ctx, location, vec![], nil_ty, loop_body_region);
        ctx.push_op(block, loop_op.op_ref());

        // === Step 4: Read final position ===
        let final_pos_addr = arena_wasm::i32_const(ctx, location, i32_ty, scratch_pos as i32);
        ctx.push_op(block, final_pos_addr.op_ref());
        let final_pos = arena_wasm::i32_load(
            ctx,
            location,
            final_pos_addr.result(ctx),
            i32_ty,
            0,
            align_i32,
            0,
        );
        ctx.push_op(block, final_pos.op_ref());

        // === Step 5: Handle negative sign prefix ===
        let then_sign_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        {
            let minus = arena_wasm::i32_const(ctx, location, i32_ty, 45); // '-'
            ctx.push_op(then_sign_block, minus.op_ref());
            let store_sign = arena_wasm::i32_store8(
                ctx,
                location,
                final_pos.result(ctx),
                minus.result(ctx),
                0,
                align_i8,
                0,
            );
            ctx.push_op(then_sign_block, store_sign.op_ref());
            let yield_sign = arena_wasm::r#yield(ctx, location, final_pos.result(ctx));
            ctx.push_op(then_sign_block, yield_sign.op_ref());
        }

        let else_sign_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        {
            let one = arena_wasm::i32_const(ctx, location, i32_ty, 1);
            ctx.push_op(else_sign_block, one.op_ref());
            let start = arena_wasm::i32_add(
                ctx,
                location,
                final_pos.result(ctx),
                one.result(ctx),
                i32_ty,
            );
            ctx.push_op(else_sign_block, start.op_ref());
            let yield_start = arena_wasm::r#yield(ctx, location, start.result(ctx));
            ctx.push_op(else_sign_block, yield_start.op_ref());
        }

        let then_sign_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![then_sign_block],
            parent_op: None,
        });
        let else_sign_region = ctx.create_region(RegionData {
            location,
            blocks: smallvec![else_sign_block],
            parent_op: None,
        });
        let if_sign = arena_wasm::r#if(
            ctx,
            location,
            is_neg_val,
            i32_ty,
            then_sign_region,
            else_sign_region,
        );
        ctx.push_op(block, if_sign.op_ref());
        let start_ptr = if_sign.result(ctx);

        // === Step 6: Calculate string length ===
        let buf_end_plus1 = arena_wasm::i32_const(ctx, location, i32_ty, (buf_end + 1) as i32);
        ctx.push_op(block, buf_end_plus1.op_ref());
        let len = arena_wasm::i32_sub(ctx, location, buf_end_plus1.result(ctx), start_ptr, i32_ty);
        ctx.push_op(block, len.op_ref());

        // === Step 7: Set up iovec and call fd_write ===
        let iovec_addr = arena_wasm::i32_const(ctx, location, i32_ty, iovec_off as i32);
        ctx.push_op(block, iovec_addr.op_ref());
        let store_iovec_ptr = arena_wasm::i32_store(
            ctx,
            location,
            iovec_addr.result(ctx),
            start_ptr,
            0,
            align_i32,
            0,
        );
        ctx.push_op(block, store_iovec_ptr.op_ref());

        let iovec_len_addr = arena_wasm::i32_const(ctx, location, i32_ty, (iovec_off + 4) as i32);
        ctx.push_op(block, iovec_len_addr.op_ref());
        let store_iovec_len = arena_wasm::i32_store(
            ctx,
            location,
            iovec_len_addr.result(ctx),
            len.result(ctx),
            0,
            align_i32,
            0,
        );
        ctx.push_op(block, store_iovec_len.op_ref());

        let stdout = arena_wasm::i32_const(ctx, location, i32_ty, 1);
        ctx.push_op(block, stdout.op_ref());
        let iovs_count = arena_wasm::i32_const(ctx, location, i32_ty, 1);
        ctx.push_op(block, iovs_count.op_ref());
        let nwr_addr = arena_wasm::i32_const(ctx, location, i32_ty, nwr_off as i32);
        ctx.push_op(block, nwr_addr.op_ref());

        let fd_result = arena_wasm::call(
            ctx,
            location,
            vec![
                stdout.result(ctx),
                iovec_addr.result(ctx),
                iovs_count.result(ctx),
                nwr_addr.result(ctx),
            ],
            vec![i32_ty],
            Symbol::new("fd_write"),
        );
        ctx.push_op(block, fd_result.op_ref());
        let drop_fd = arena_wasm::drop(ctx, location, fd_result.results(ctx)[0]);
        ctx.push_op(block, drop_fd.op_ref());
    }
}

/// Check if a type is the Step ADT type.
fn is_step_adt(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    if let Some(ArenaAttribute::Symbol(name)) = data.attrs.get(&Symbol::new("name")) {
        *name == Symbol::new("_Step")
    } else {
        false
    }
}

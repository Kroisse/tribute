//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This pass transforms mid-level IR operations (func, arith, scf, adt) to
//! wasm dialect operations. Uses arena IR for in-place mutation within a
//! single arena session.

use std::fmt;

use tracing::{error, warn};
use tribute_core::{CallingConvention, get_calling_convention};
use tribute_ir::ModulePathExt;
use trunk_ir::Symbol;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{PassError, PassManager};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, TypeConverter,
    WasmFuncSignatureConversionPattern,
};
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

use super::const_to_wasm::ConstAnalysis;
use super::io::IoAnalysis;
use super::type_converter::{self, wasm_type_converter};
use trunk_ir_wasm_backend::gc_types::{EVIDENCE_IDX, STEP_IDX, STEP_TAG_DONE};

const WASM_BACKEND_READY_BOUNDARY: &str = "wasm-backend-ready";

#[derive(Debug)]
pub enum WasmLowerError {
    Conversion(ConversionError),
    Pass(PassError),
    Const(super::const_to_wasm::ConstValidationError),
}

impl fmt::Display for WasmLowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conversion(error) => error.fmt(f),
            Self::Pass(error) => error.fmt(f),
            Self::Const(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for WasmLowerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Conversion(error) => Some(error),
            Self::Pass(error) => Some(error),
            Self::Const(error) => Some(error),
        }
    }
}

impl From<ConversionError> for WasmLowerError {
    fn from(error: ConversionError) -> Self {
        Self::Conversion(error)
    }
}

impl From<PassError> for WasmLowerError {
    fn from(error: PassError) -> Self {
        Self::Pass(error)
    }
}

impl From<super::const_to_wasm::ConstValidationError> for WasmLowerError {
    fn from(error: super::const_to_wasm::ConstValidationError) -> Self {
        Self::Const(error)
    }
}

/// Conversion target for IR after Wasm-specific lowering has removed shared
/// ability/effect ABI operations.
pub fn wasm_backend_ready_target() -> ConversionTarget {
    ConversionTarget::new()
        .illegal_dialect("ability")
        .illegal_dialect("effect")
}

/// Conversion target immediately before WebAssembly emission.
pub fn wasm_emission_ready_target() -> ConversionTarget {
    wasm_backend_ready_target().illegal_dialect("wasm_gc")
}

/// Run the full WASM lowering pipeline on arena IR.
pub fn lower_to_wasm(ctx: &mut IrContext, module: Module) -> Result<(), WasmLowerError> {
    let const_analysis = super::const_to_wasm::analyze_consts(ctx, module);
    let io_analysis = IoAnalysis::analyze(ctx, module, const_analysis.total_size());
    {
        let _span = tracing::info_span!("io_to_wasm").entered();
        super::io::lower(ctx, module, &io_analysis)?;
    }

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

    // Lower tribute_rt operations (box_int, unbox_int) to wasm operations
    // This must run BEFORE adt_to_wasm because float boxing/unboxing emits
    // adt.struct_new / adt.ref_cast / adt.struct_get that need conversion.
    {
        let _span = tracing::info_span!("tribute_rt_to_wasm").entered();
        super::tribute_rt_to_wasm::lower(ctx, module);
    }
    debug_func_params(ctx, module, "after tribute_rt_to_wasm");

    // Lower constants before adt_to_wasm so string constants can become the
    // ordinary prelude String::Leaf variant.
    {
        let _span = tracing::info_span!("const_to_wasm").entered();
        super::const_to_wasm::validate_for_wasm(ctx, module, &const_analysis)?;
        super::const_to_wasm::lower(ctx, module, &const_analysis);
    }

    // Convert ALL adt ops to wasm (including String::Leaf from const lowering).
    {
        let _span = tracing::info_span!("adt_to_wasm").entered();
        let tc = wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::adt_to_wasm::lower(ctx, module, tc);
    }
    debug_func_params(ctx, module, "after adt_to_wasm");

    // Lower evidence runtime function stubs (prepare for inline WASM operations)
    {
        let _span = tracing::info_span!("evidence_to_wasm").entered();
        if let Ok(core_module) = core::Module::from_op(ctx, module.op()) {
            super::evidence_to_wasm::prepare_wasm_evidence_runtime(ctx, module);
            let mut pm = PassManager::new();
            pm.nest::<wasm_dialect::Func>()
                .add_pass(super::evidence_to_wasm::LowerEvidenceToWasm);
            pm.run(ctx, core_module)?;
        } else {
            super::evidence_to_wasm::lower_evidence_to_wasm(ctx, module);
        }
    }

    {
        let _span = tracing::info_span!("intrinsic_to_wasm").entered();
        super::intrinsic_to_wasm::lower(ctx, module);
    }

    // Phase 2: Module-level operations via WasmLowerer (in-place)
    {
        let _span = tracing::info_span!("wasm_lowerer").entered();
        let mut lowerer = WasmLowerer::new(&const_analysis, &io_analysis);
        lowerer.lower_module(ctx, module);
    }

    verify_wasm_backend_ready(ctx, module)?;
    Ok(())
}

/// Assign module-local GC indices after all type-conversion materialization.
pub fn finalize_wasm_gc_types(ctx: &mut IrContext, module: Module) -> Result<(), ConversionError> {
    trunk_ir_wasm_backend::passes::wasm_gc_to_wasm::lower(ctx, module);
    PatternApplicator::new(TypeConverter::new())
        .with_target(wasm_emission_ready_target())
        .apply_partial_conversion(ctx, module, "wasm-emission-ready")?;
    if cfg!(debug_assertions) {
        check_all_wasm_dialect(ctx, module);
    }
    Ok(())
}

/// Verify the partial Wasm backend boundary.
///
/// Unknown operations are still allowed because unresolved casts and backend
/// infrastructure may remain for later pipeline stages, but residual
/// `ability.*` and `effect.*` operations are compiler bugs at this point.
pub fn verify_wasm_backend_ready(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), ConversionError> {
    PatternApplicator::new(TypeConverter::new())
        .with_target(wasm_backend_ready_target())
        .apply_partial_conversion(ctx, module, WASM_BACKEND_READY_BOUNDARY)?;
    Ok(())
}

// =============================================================================
// Debug helpers
// =============================================================================

/// Debug helper to check if all operations in function bodies are in wasm dialect
fn check_all_wasm_dialect(ctx: &IrContext, module: Module) {
    let Some(body) = module.body(ctx) else {
        return;
    };
    for &block in ctx.region(body).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            let data = ctx.op(op);
            if data.dialect == wasm_dialect::DIALECT_NAME() && data.name == Symbol::new("func") {
                check_function_body(ctx, op);
            }
        }
    }
}

/// Debug helper to trace function parameter types through the pipeline.
fn debug_func_params(ctx: &IrContext, module: Module, phase: &str) {
    let Some(body) = module.body(ctx) else {
        return;
    };
    for &block in ctx.region(body).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            let data = ctx.op(op);
            // Check for func.func or wasm.func operations
            if data.dialect == func::DIALECT_NAME() && data.name == Symbol::new("func") {
                if let Some(fn_ty) = data.attributes.get_type("type") {
                    let fn_data = ctx.types.get(fn_ty);
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
                        .get_text("sym_name")
                        .map(|text| text.to_string())
                        .unwrap_or_default();
                    tracing::debug!("[{phase}] func.func {sym_name}: params={params:?}");
                }
            } else if data.dialect == wasm_dialect::DIALECT_NAME()
                && data.name == Symbol::new("func")
                && let Some(fn_ty) = data.attributes.get_type("type")
            {
                let fn_data = ctx.types.get(fn_ty);
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
                    .get_text("sym_name")
                    .map(|text| text.to_string())
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
    core::func(ctx, result, params).as_type_ref()
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
    main_param_types: Vec<TypeRef>,
    main_convention: CallingConvention,
    main_exported: bool,
}

impl MainExports {
    fn new() -> Self {
        Self {
            saw_main: false,
            main_result_type: None,
            main_param_types: Vec::new(),
            main_convention: CallingConvention::Direct,
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
    io_analysis: &'a IoAnalysis,
    memory_plan: ArenaMemoryPlan,
    main_exports: MainExports,
    has_continuations: bool,
}

impl<'a> WasmLowerer<'a> {
    fn new(const_analysis: &'a ConstAnalysis, io_analysis: &'a IoAnalysis) -> Self {
        Self {
            const_analysis,
            io_analysis,
            memory_plan: ArenaMemoryPlan::new(),
            main_exports: MainExports::new(),
            has_continuations: true,
        }
    }

    fn lower_module(&mut self, ctx: &mut IrContext, module: Module) {
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

                if data.dialect == wasm_dialect::DIALECT_NAME() {
                    // Track wasm module-level metadata
                    if data.name == Symbol::new("memory") {
                        self.memory_plan.has_memory = true;
                    } else if data.name == Symbol::new("export_memory") {
                        self.memory_plan.has_exported_memory = true;
                    } else if data.name == Symbol::new("export_func") {
                        if data.attributes.get_str("name") == Some("main") {
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
        let Some(sym_name) = data.attributes.get_symbol("sym_name") else {
            return;
        };

        // Only match root-level main, not foo::main
        if !(sym_name.is_simple() && sym_name.last_segment() == Symbol::new("main")) {
            return;
        }

        self.main_exports.saw_main = true;
        self.main_exports.main_convention = get_calling_convention(ctx, op).unwrap_or_default();

        if let Some(fn_ty) = data.attributes.get_type("type") {
            let fn_data = ctx.types.get(fn_ty);
            // In arena core.func type, params[0] is the return type
            if let Some(&result_ty) = fn_data.params.first() {
                self.main_exports.main_result_type = Some(result_ty);
            }
            self.main_exports.main_param_types = fn_data.params[1..].to_vec();
        }
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

        if self.io_analysis.needs_fd_write {
            let i32_ty = intern_type(ctx, "core", "i32");
            let import_ty = intern_func_type(ctx, vec![i32_ty, i32_ty, i32_ty, i32_ty], i32_ty);
            let op = wasm_dialect::import_func(
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
        let io_size = self.io_analysis.total_size;
        if const_size > 0 || io_size > 0 {
            self.memory_plan.needs_memory = true;
        }

        if self.memory_plan.needs_memory && !self.memory_plan.has_memory {
            let total_data_size = const_size + io_size;
            let required_pages = self.memory_plan.required_pages(total_data_size);
            let op = wasm_dialect::memory(ctx, location, required_pages, 0, false, false);
            preamble_ops.push(op.op_ref());
            self.memory_plan.has_memory = true;
        }

        // Emit yield globals for continuation support
        if self.has_continuations {
            // Index 0 ($yield_state): i32
            let g0 =
                wasm_dialect::global(ctx, location, Symbol::new("i32"), true, Attribute::Int(0));
            preamble_ops.push(g0.op_ref());
            // Index 1 ($yield_tag): i32
            let g1 =
                wasm_dialect::global(ctx, location, Symbol::new("i32"), true, Attribute::Int(0));
            preamble_ops.push(g1.op_ref());
            // Index 2 ($yield_cont): anyref
            let g2 = wasm_dialect::global(
                ctx,
                location,
                Symbol::new("anyref"),
                true,
                Attribute::Int(0),
            );
            preamble_ops.push(g2.op_ref());
            // Index 3 ($yield_op_idx): i32
            let g3 =
                wasm_dialect::global(ctx, location, Symbol::new("i32"), true, Attribute::Int(0));
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
        // String and Bytes constants share passive data segments.
        for (content, _data_idx, _len) in self.const_analysis.allocations.iter() {
            let op = wasm_dialect::data(
                ctx,
                location,
                0,
                Attribute::Bytes(content.as_slice().into()),
                true,
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
            let op = wasm_dialect::export_memory(ctx, location, "memory".into(), 0);
            ctx.push_op(module_block, op.op_ref());
            self.memory_plan.has_exported_memory = true;
        }

        if self.main_exports.saw_main && !self.main_exports.main_exported {
            let op = wasm_dialect::export_func(ctx, location, "main".into(), Symbol::new("main"));
            ctx.push_op(module_block, op.op_ref());
            self.main_exports.main_exported = true;
        }

        if self.main_exports.saw_main {
            let start_func = self.build_start_function(ctx, location);
            ctx.push_op(module_block, start_func);

            let export_op =
                wasm_dialect::export_func(ctx, location, "_start".into(), Symbol::new("_start"));
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
        let func_op =
            wasm_dialect::func(ctx, location, Symbol::new("_start"), func_ty, body_region);
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

        // Call main — returns Step
        let main_args = self.build_main_args(ctx, body_block, location, i32_ty);
        let call_main =
            wasm_dialect::call(ctx, location, main_args, vec![step_ty], Symbol::new("main"));
        ctx.push_op(body_block, call_main.op_ref());
        let step_result = call_main.results(ctx)[0];

        // Extract tag field (field 0) from Step
        let get_tag = wasm_dialect::struct_get(ctx, location, step_result, i32_ty, STEP_IDX, 0);
        ctx.push_op(body_block, get_tag.op_ref());
        let tag_val = get_tag.result(ctx);

        // Compare tag with DONE (0)
        let done_const = wasm_dialect::i32_const(ctx, location, i32_ty, STEP_TAG_DONE);
        ctx.push_op(body_block, done_const.op_ref());
        let cmp_eq = wasm_dialect::i32_eq(ctx, location, tag_val, done_const.result(ctx), i32_ty);
        ctx.push_op(body_block, cmp_eq.op_ref());
        let is_done = cmp_eq.result(ctx);

        // Build then (Done) branch
        let then_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });

        let ret_then = wasm_dialect::r#return(ctx, location, vec![]);
        ctx.push_op(then_block, ret_then.op_ref());

        // Build else (unhandled effect) branch — trap
        let else_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let trap = wasm_dialect::unreachable(ctx, location);
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

        let if_op = wasm_dialect::r#if(ctx, location, is_done, nil_ty, then_region, else_region);
        ctx.push_op(body_block, if_op.op_ref());

        let trap_end = wasm_dialect::unreachable(ctx, location);
        ctx.push_op(body_block, trap_end.op_ref());
    }

    /// Build `_start` for a pure `main`, whose source-level result is always Nil.
    fn build_start_simple_body(
        &self,
        ctx: &mut IrContext,
        body_block: BlockRef,
        location: Location,
        i32_ty: TypeRef,
        _nil_ty: TypeRef,
    ) {
        let main_args = self.build_main_args(ctx, body_block, location, i32_ty);
        let call = wasm_dialect::call(ctx, location, main_args, vec![], Symbol::new("main"));
        ctx.push_op(body_block, call.op_ref());

        let ret = wasm_dialect::r#return(ctx, location, vec![]);
        ctx.push_op(body_block, ret.op_ref());
    }

    fn build_main_args(
        &self,
        ctx: &mut IrContext,
        body_block: BlockRef,
        location: Location,
        i32_ty: TypeRef,
    ) -> Vec<ValueRef> {
        match self.main_exports.main_convention {
            CallingConvention::Direct => {
                assert!(
                    self.main_exports.main_param_types.is_empty(),
                    "Wasm entrypoint: Direct `main` must not have hidden parameters"
                );
                Vec::new()
            }
            CallingConvention::EvidenceDirect => {
                assert_eq!(
                    self.main_exports.main_param_types.len(),
                    1,
                    "Wasm entrypoint: EvidenceDirect `main` must have one evidence parameter"
                );
                let zero = wasm_dialect::i32_const(ctx, location, i32_ty, 0);
                ctx.push_op(body_block, zero.op_ref());
                let empty = wasm_dialect::array_new_default(
                    ctx,
                    location,
                    zero.result(ctx),
                    self.main_exports.main_param_types[0],
                    EVIDENCE_IDX,
                );
                ctx.push_op(body_block, empty.op_ref());
                vec![empty.result(ctx)]
            }
            CallingConvention::Cps => {
                panic!(
                    "Wasm entrypoint: Cps `main` is invalid; frontend must reject residual control effects"
                )
            }
        }
    }
}

/// Check if a type is the Step ADT type.
fn is_step_adt(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.attrs.get_symbol("name") == Some(Symbol::new("_Step"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Span;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::refs::PathRef;
    use trunk_ir::rewrite::LegalityCheck;

    fn lower_text(ir: &str) -> String {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        lower_to_wasm(&mut ctx, module).expect("test module should lower to wasm");
        print_module(&ctx, module.op())
    }

    fn empty_module_with_block(ctx: &mut IrContext) -> Module {
        let module = parse_test_module(
            ctx,
            r#"core.module @test {
  func.func @placeholder() -> core.nil {
    func.return
  }
}"#,
        );
        let placeholder = module.ops(ctx)[0];
        trunk_ir::rewrite::erase_op(ctx, placeholder);
        module
    }

    #[test]
    fn wasm_start_passes_empty_evidence_to_evidence_direct_main() {
        let mut ctx = IrContext::new();
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let evidence_ty = intern_type(&mut ctx, "wasm", "arrayref");
        let nil_ty = intern_type(&mut ctx, "core", "nil");
        let const_analysis = ConstAnalysis {
            allocations: vec![],
            string_enum_ty: None,
        };
        let io_analysis = IoAnalysis {
            needs_fd_write: false,
            iovec_offset: 0,
            nwritten_offset: 0,
            scratch_offset: 0,
            total_size: 0,
        };
        let mut lowerer = WasmLowerer::new(&const_analysis, &io_analysis);
        lowerer.main_exports.saw_main = true;
        lowerer.main_exports.main_result_type = Some(nil_ty);
        lowerer.main_exports.main_param_types = vec![evidence_ty];
        lowerer.main_exports.main_convention = CallingConvention::EvidenceDirect;

        let start = lowerer.build_start_function(&mut ctx, location);
        let start = wasm_dialect::Func::from_op(&ctx, start).expect("wasm _start function");
        let entry = ctx.region(start.body(&ctx)).blocks[0];
        let ops = ctx.block(entry).ops.to_vec();
        let empty = ops
            .iter()
            .copied()
            .find(|op| ctx.op(*op).name == Symbol::new("array_new_default"))
            .expect("_start should allocate empty evidence");
        let call_main = ops
            .iter()
            .copied()
            .find(|op| {
                ctx.op(*op).name == Symbol::new("call")
                    && ctx.op(*op).attributes.get_symbol("callee") == Some(Symbol::new("main"))
            })
            .expect("_start should call main");

        assert_eq!(ctx.op_operands(call_main), &[ctx.op_result(empty, 0)]);
    }

    #[test]
    fn wasm_start_checks_step_result_before_returning() {
        let mut ctx = IrContext::new();
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let step_ty = type_converter::step_adt_type(&mut ctx);
        let const_analysis = ConstAnalysis {
            allocations: vec![],
            string_enum_ty: None,
        };
        let io_analysis = IoAnalysis {
            needs_fd_write: false,
            iovec_offset: 0,
            nwritten_offset: 0,
            scratch_offset: 0,
            total_size: 0,
        };
        let mut lowerer = WasmLowerer::new(&const_analysis, &io_analysis);
        lowerer.main_exports.saw_main = true;
        lowerer.main_exports.main_result_type = Some(step_ty);

        let start = lowerer.build_start_function(&mut ctx, location);
        let start = wasm_dialect::Func::from_op(&ctx, start).expect("wasm _start function");
        let entry = ctx.region(start.body(&ctx)).blocks[0];
        let names: Vec<_> = ctx
            .block(entry)
            .ops
            .iter()
            .map(|&op| ctx.op(op).name)
            .collect();

        assert!(names.contains(&Symbol::new("call")));
        assert!(names.contains(&Symbol::new("struct_get")));
        assert!(names.contains(&Symbol::new("i32_eq")));
        assert!(names.contains(&Symbol::new("if")));
        assert!(names.contains(&Symbol::new("unreachable")));
        let i32_ty = intern_type(&mut ctx, "core", "i32");
        assert!(!is_step_adt(&ctx, i32_ty));
    }

    #[test]
    #[should_panic(expected = "Cps `main` is invalid")]
    fn wasm_start_rejects_cps_main() {
        let mut ctx = IrContext::new();
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let i32_ty = intern_type(&mut ctx, "core", "i32");
        let body_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let const_analysis = ConstAnalysis {
            allocations: vec![],
            string_enum_ty: None,
        };
        let io_analysis = IoAnalysis {
            needs_fd_write: false,
            iovec_offset: 0,
            nwritten_offset: 0,
            scratch_offset: 0,
            total_size: 0,
        };
        let mut lowerer = WasmLowerer::new(&const_analysis, &io_analysis);
        lowerer.main_exports.main_convention = CallingConvention::Cps;

        lowerer.build_main_args(&mut ctx, body_block, location, i32_ty);
    }

    #[test]
    fn module_lowerer_emits_io_and_passive_data_requirements() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run() -> core.i32 {
    %value = arith.const {value = 1} : core.i32
    func.return %value
  }
}"#,
        );
        let const_analysis = ConstAnalysis {
            allocations: vec![(b"text".to_vec(), 0, 4), (vec![1, 2], 1, 2)],
            string_enum_ty: None,
        };
        let io_analysis = IoAnalysis {
            needs_fd_write: true,
            iovec_offset: 4,
            nwritten_offset: 12,
            scratch_offset: 16,
            total_size: 20,
        };
        let mut lowerer = WasmLowerer::new(&const_analysis, &io_analysis);
        let module_block = module.first_block(&ctx).expect("module body block");
        let placeholder = ctx.block(module_block).ops[0];

        lowerer.lower_module(&mut ctx, module);
        trunk_ir::rewrite::erase_op(&mut ctx, placeholder);

        let location = Location::new(PathRef::from_u32(0), Span::default());
        let i32_ty = intern_type(&mut ctx, "core", "i32");
        let i64_ty = intern_type(&mut ctx, "core", "i64");
        let f32_ty = intern_type(&mut ctx, "core", "f32");
        let f64_ty = intern_type(&mut ctx, "core", "f64");
        let nil_ty = intern_type(&mut ctx, "core", "nil");
        let body_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let address = wasm_dialect::i32_const(&mut ctx, location, i32_ty, 0);
        let address_result = address.result(&ctx);
        ctx.push_op(body_block, address.op_ref());
        let i64_load = wasm_dialect::i64_load(&mut ctx, location, address_result, i64_ty, 0, 0, 0);
        let f32_load = wasm_dialect::f32_load(&mut ctx, location, address_result, f32_ty, 0, 0, 0);
        let f64_load = wasm_dialect::f64_load(&mut ctx, location, address_result, f64_ty, 0, 0, 0);
        let i64_result = i64_load.result(&ctx);
        let f32_result = f32_load.result(&ctx);
        let f64_result = f64_load.result(&ctx);
        for load in [i64_load.op_ref(), f32_load.op_ref(), f64_load.op_ref()] {
            ctx.push_op(body_block, load);
        }
        for store in [
            wasm_dialect::i64_store(&mut ctx, location, address_result, i64_result, 0, 0, 0)
                .op_ref(),
            wasm_dialect::f32_store(&mut ctx, location, address_result, f32_result, 0, 0, 0)
                .op_ref(),
            wasm_dialect::f64_store(&mut ctx, location, address_result, f64_result, 0, 0, 0)
                .op_ref(),
        ] {
            ctx.push_op(body_block, store);
        }
        let ret = wasm_dialect::r#return(&mut ctx, location, vec![]);
        ctx.push_op(body_block, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location,
            blocks: smallvec![body_block],
            parent_op: None,
        });
        let func_ty = intern_func_type(&mut ctx, vec![], nil_ty);
        let memory_func =
            wasm_dialect::func(&mut ctx, location, Symbol::new("memory_ops"), func_ty, body);
        ctx.push_op(module_block, memory_func.op_ref());

        let output = print_module(&ctx, module.op());
        assert!(output.contains("wasm.import_func"), "{output}");
        assert!(output.contains("wasm.memory"), "{output}");
        assert!(output.contains("wasm.export_memory"), "{output}");
        assert_eq!(output.matches("wasm.data").count(), 2, "{output}");

        let binary = trunk_ir_wasm_backend::emit_module_to_wasm(&mut ctx, module)
            .expect("lowered module requirements should emit");
        assert_eq!(&binary.bytes[..4], b"\0asm");
    }

    #[test]
    fn module_lowerer_recognizes_existing_exports_memory_and_main_signature() {
        let mut ctx = IrContext::new();
        let module = empty_module_with_block(&mut ctx);
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let module_block = module.first_block(&ctx).expect("module body block");
        let i32_ty = intern_type(&mut ctx, "core", "i32");
        let nil_ty = intern_type(&mut ctx, "core", "nil");

        let memory = wasm_dialect::memory(&mut ctx, location, 1, 0, false, false);
        let export_memory = wasm_dialect::export_memory(&mut ctx, location, "memory".into(), 0);
        let export_main =
            wasm_dialect::export_func(&mut ctx, location, "main".into(), Symbol::new("main"));
        for op in [
            memory.op_ref(),
            export_memory.op_ref(),
            export_main.op_ref(),
        ] {
            ctx.push_op(module_block, op);
        }

        let main_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret = wasm_dialect::r#return(&mut ctx, location, vec![]);
        ctx.push_op(main_block, ret.op_ref());
        let main_body = ctx.create_region(RegionData {
            location,
            blocks: smallvec![main_block],
            parent_op: None,
        });
        let main_ty = intern_func_type(&mut ctx, vec![i32_ty], nil_ty);
        let main = wasm_dialect::func(&mut ctx, location, Symbol::new("main"), main_ty, main_body);
        ctx.push_op(module_block, main.op_ref());

        let const_analysis = ConstAnalysis {
            allocations: vec![],
            string_enum_ty: None,
        };
        let io_analysis = IoAnalysis {
            needs_fd_write: false,
            iovec_offset: 0,
            nwritten_offset: 0,
            scratch_offset: 0,
            total_size: 0,
        };
        let mut lowerer = WasmLowerer::new(&const_analysis, &io_analysis);
        lowerer.scan_module_ops(&ctx, module.body(&ctx).expect("module body"));

        assert!(lowerer.memory_plan.has_memory);
        assert!(lowerer.memory_plan.has_exported_memory);
        assert!(lowerer.main_exports.saw_main);
        assert!(lowerer.main_exports.main_exported);
        assert_eq!(lowerer.main_exports.main_result_type, Some(nil_ty));
        assert_eq!(lowerer.main_exports.main_param_types, vec![i32_ty]);

        ctx.op_mut(main.op_ref()).attributes.remove("sym_name");
        lowerer.main_exports.saw_main = false;
        lowerer.scan_wasm_func(&ctx, main.op_ref());
        assert!(!lowerer.main_exports.saw_main);
    }

    #[test]
    fn module_lowerer_populates_an_empty_module() {
        let mut ctx = IrContext::new();
        let module = empty_module_with_block(&mut ctx);
        let const_analysis = ConstAnalysis {
            allocations: vec![],
            string_enum_ty: None,
        };
        let io_analysis = IoAnalysis {
            needs_fd_write: false,
            iovec_offset: 0,
            nwritten_offset: 0,
            scratch_offset: 0,
            total_size: 0,
        };
        let mut lowerer = WasmLowerer::new(&const_analysis, &io_analysis);

        lowerer.lower_module(&mut ctx, module);

        assert!(!module.ops(&ctx).is_empty());
    }

    #[test]
    fn debug_func_params_accepts_source_functions_with_parameters() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @identity(%value: core.i32) -> core.i32 {
    func.return %value
  }
}"#,
        );

        debug_func_params(&ctx, module, "test");

        assert_eq!(module.ops(&ctx).len(), 1);
    }

    #[test]
    fn module_lowerer_ignores_modules_without_a_body() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, "core.module @test {}");
        check_all_wasm_dialect(&ctx, module);
        debug_func_params(&ctx, module, "test");
        let const_analysis = ConstAnalysis {
            allocations: vec![],
            string_enum_ty: None,
        };
        let io_analysis = IoAnalysis {
            needs_fd_write: false,
            iovec_offset: 0,
            nwritten_offset: 0,
            scratch_offset: 0,
            total_size: 0,
        };
        let mut lowerer = WasmLowerer::new(&const_analysis, &io_analysis);

        lowerer.lower_module(&mut ctx, module);

        assert!(module.body(&ctx).is_none());
    }

    #[test]
    fn wasm_dialect_check_visits_function_bodies() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    %value = arith.const {value = 1} : core.i32
    wasm.return
  }
}"#,
        );

        check_all_wasm_dialect(&ctx, module);
    }

    #[test]
    fn wasm_backend_ready_rejects_residual_effect_ops() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%ev: wasm.arrayref, %payload: wasm.anyref) -> wasm.anyref {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : wasm.anyref
    func.return %result
  }
}"#,
        );

        let error = verify_wasm_backend_ready(&mut ctx, module)
            .expect_err("residual effect operation should fail wasm backend boundary");

        assert_eq!(error.boundary(), WASM_BACKEND_READY_BOUNDARY);
        assert_eq!(error.operations().len(), 1);
        assert_eq!(error.operations()[0].dialect, Symbol::new("effect"));
        assert_eq!(error.operations()[0].name, Symbol::new("dispatch_tail"));
        assert_eq!(error.operations()[0].legality, LegalityCheck::Illegal);
    }

    #[test]
    fn wasm_backend_ready_allows_unknown_later_stage_ops() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run() -> core.i32 {
    %value = arith.const {value = 1} : core.i32
    func.return %value
  }
}"#,
        );

        verify_wasm_backend_ready(&mut ctx, module)
            .expect("partial wasm backend boundary should allow unknown later-stage ops");
    }

    #[test]
    fn lower_to_wasm_rejects_non_reference_string_constant_result() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !String = adt.enum() {name = @String, variants = [[@Leaf, [core.bytes]], [@Branch, [tribute_rt.anyref, tribute_rt.anyref, core.i32]]]}
  func.func @main() -> core.nil {
    %string = adt.string_const {value = "wrong"} : core.i32
    func.return
  }
}"#,
        );
        let string_ty = ctx
            .type_aliases()
            .iter()
            .find_map(|(name, ty)| (*name == "String").then_some(*ty))
            .expect("String alias");
        tribute_ir::metadata::WellKnownTypes {
            string: Some(string_ty),
        }
        .attach(&mut ctx, module.op());

        let error = lower_to_wasm(&mut ctx, module)
            .expect_err("non-reference string constant result must be rejected");

        assert!(
            error
                .to_string()
                .contains("adt.string_const must produce wasm.anyref"),
            "{error}"
        );
    }

    #[test]
    fn lower_to_wasm_removes_effect_dispatch_tail() {
        let output = lower_text(
            r#"core.module @test {
  func.func @run(%ev: wasm.arrayref, %payload: wasm.anyref) -> wasm.anyref {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : wasm.anyref
    func.return %result
  }
}"#,
        );

        assert!(!output.contains("effect.dispatch_tail"), "{output}");
        assert!(output.contains("__tribute_evidence_lookup"), "{output}");
        assert!(output.contains("wasm.call_indirect"), "{output}");
    }

    #[test]
    fn wasm_lower_error_wraps_conversion_error() {
        let conversion = ConversionError::new(WASM_BACKEND_READY_BOUNDARY, vec![]);
        let error = WasmLowerError::from(conversion);

        assert!(error.to_string().contains(WASM_BACKEND_READY_BOUNDARY));
        assert!(std::error::Error::source(&error).is_some());
    }
}

//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This pass transforms mid-level IR operations (func, arith, scf, adt) to
//! wasm dialect operations. Uses arena IR for in-place mutation within a
//! single arena session.

use std::fmt;

use tracing::{error, warn};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::conversion::UnrealizedCastConversionPattern;
use trunk_ir::dialect::core;
use trunk_ir::dialect::func;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassError, PassManager, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef};
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, TypeConverter,
};
use trunk_ir::types::Location;
use trunk_ir_wasm_backend::passes::reference_upcast::ReferenceUpcastPattern;
use trunk_ir_wasm_backend::passes::signature_conversion::WasmFuncSignatureConversionPattern;

use super::type_converter::wasm_type_converter;
use trunk_ir::SymbolPath;

const WASM_BACKEND_READY_BOUNDARY: &str = "wasm-backend-ready";

#[derive(Debug)]
pub enum WasmLowerError {
    Conversion(ConversionError),
    Pass(PassError),
}

impl fmt::Display for WasmLowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conversion(error) => error.fmt(f),
            Self::Pass(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for WasmLowerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Conversion(error) => Some(error),
            Self::Pass(error) => Some(error),
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

/// Conversion target for IR after Wasm-specific lowering has removed shared
/// ability/effect ABI operations.
pub fn wasm_backend_ready_target() -> ConversionTarget {
    ConversionTarget::new()
        .illegal_dialect(tribute_ir::dialect::ability::DIALECT_NAME)
        .illegal_dialect(tribute_ir::dialect::effect::DIALECT_NAME)
}

/// Conversion target immediately before WebAssembly emission.
///
/// Every `core.unrealized_conversion_cast` must have been materialized or
/// reconciled by now; a remaining one is a conversion bug.
pub fn wasm_emission_ready_target() -> ConversionTarget {
    wasm_backend_ready_target()
        .illegal_dialect(trunk_ir::dialect::wasm_gc::DIALECT_NAME)
        .illegal_op("core.unrealized_conversion_cast")
}

/// Run the full WASM lowering pipeline on arena IR.
///
/// The structured-control boundary check runs first, before any mutation.
/// The lowering steps then run as passes of one [`PassManager`], so a debug
/// build verifies IR invariants after every step that changed the IR, and
/// every step shares `analyses`. Each step reads only its input IR: a step
/// that relies on a module-level resource, such as a data segment, an import,
/// or linear memory, declares it in the IR.
pub(crate) fn lower_to_wasm(
    ctx: &mut IrContext,
    module: Module,
    analyses: &mut AnalysisCache,
) -> Result<(), WasmLowerError> {
    trunk_ir_wasm_backend::passes::scf_to_wasm::validate_lowerable_structured_control(
        ctx, module, analyses,
    )?;

    let core_module =
        core::Module::from_op(ctx, module.op()).expect("Wasm lowering requires a core.module");
    wasm_lowering_passes().run(ctx, core_module, analyses)?;

    verify_wasm_backend_ready(ctx, module)?;
    Ok(())
}

/// The Wasm lowering steps, in order.
fn wasm_lowering_passes() -> PassManager {
    let mut pm = PassManager::new();
    pm.add_pass(ConvertBuiltinLayouts)
        .add_pass(IoToWasm)
        // Pattern-based lowering (trunk-ir-wasm-backend).
        .add_pass(ArithToWasm)
        .add_pass(ScfToWasm)
        // Convert function and indirect-call signatures before target lowering.
        .add_pass(ConvertSignatures)
        .add_pass(FuncToWasm)
        // Convert wasm.func signature types (e.g., adt.array<Marker> →
        // wasm.arrayref). Runs after func_to_wasm creates the wasm.func operations
        // and before emission reads their type attributes.
        .add_pass(WasmFuncSignatureConversion)
        // Lower tribute_rt operations (box_int, unbox_int). Runs before
        // adt_to_wasm because float boxing emits adt operations that it converts.
        .add_pass(TributeRtToWasm)
        // Lower constants before adt_to_wasm so string constants can become the
        // ordinary prelude String::Leaf variant.
        .add_pass(ConstToWasm)
        // Convert all adt operations, including String::Leaf from const lowering,
        // after declaring the runtime type descriptors they store.
        .add_pass(AdtToWasm)
        // Replace the evidence type with its GC array once no `adt` operation
        // reads it, which leaves the casts `adt` lowering made to it as identities.
        .add_pass(ConvertEvidenceArrays)
        // Fold the round trips that array operands of parameters took through
        // their `adt.array` type while signature conversion ran ahead of `adt`
        // lowering.
        .add_pass(trunk_ir::conversion::ReconcileUnrealizedCasts)
        // Bind the prompt tag allocator that evidence resolution declared to its
        // Wasm implementation.
        .add_pass(BindPromptTagRuntime)
        .add_pass(IntrinsicToWasm)
        // Module-level operations via WasmLowerer (in-place).
        .add_pass(LowerWasmModule)
        // Replace user struct layouts with their structural GC types once no
        // `adt` operation reads a nominal layout.
        .add_pass(ConvertStructLayouts);
    pm.with_debug_verifier()
        .with_instrumentation(|ctx, name, op| {
            if let Some(module) = Module::new(ctx, op) {
                debug_func_params(ctx, module, name);
            }
        });
    pm
}

struct ConvertBuiltinLayouts;

impl Pass for ConvertBuiltinLayouts {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "convert-builtin-layouts"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::type_converter::convert_builtin_layouts(ctx, target.into());
        Ok(())
    }
}

struct IoToWasm;

impl Pass for IoToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "io-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::io::lower(ctx, target.into())?;
        Ok(())
    }
}

struct ArithToWasm;

impl Pass for ArithToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "arith-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let tc = wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::arith_to_wasm::lower(ctx, target.into(), tc);
        Ok(())
    }
}

struct ScfToWasm;

impl Pass for ScfToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "scf-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let tc = wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::scf_to_wasm::lower(ctx, target.into(), tc, analyses)?;
        Ok(())
    }
}

struct ConvertSignatures;

impl Pass for ConvertSignatures {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "convert-signatures"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::convert_signatures::lower(ctx, target.into());
        Ok(())
    }
}

struct FuncToWasm;

impl Pass for FuncToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "func-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let tc = wasm_type_converter(ctx);
        trunk_ir_wasm_backend::passes::func_to_wasm::lower(ctx, target.into(), tc);
        Ok(())
    }
}

struct WasmFuncSignatureConversion;

impl Pass for WasmFuncSignatureConversion {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "wasm-func-signature-conversion"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let tc = wasm_type_converter(ctx);
        PatternApplicator::new(tc)
            .add_pattern(WasmFuncSignatureConversionPattern)
            .apply_partial(ctx, Module::from(target));
        Ok(())
    }
}

struct TributeRtToWasm;

impl Pass for TributeRtToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "tribute-rt-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::tribute_rt_to_wasm::lower(ctx, target.into());
        Ok(())
    }
}

struct ConstToWasm;

impl Pass for ConstToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "const-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let const_analysis = super::const_to_wasm::analyze_consts(ctx, target.into());
        super::const_to_wasm::validate_for_wasm(ctx, target.into(), &const_analysis)?;
        super::const_to_wasm::lower(ctx, target.into(), &const_analysis);
        Ok(())
    }
}

struct AdtToWasm;

impl Pass for AdtToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "adt-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::descriptors::declare(ctx, target.into());
        let tc = wasm_type_converter(ctx);
        crate::wasm::adt_to_wasm::lower(ctx, target.into(), tc);
        Ok(())
    }
}

struct ConvertEvidenceArrays;

impl Pass for ConvertEvidenceArrays {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "convert-evidence-arrays"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::type_converter::convert_evidence_arrays(ctx, target.into());
        Ok(())
    }
}

struct BindPromptTagRuntime;

impl Pass for BindPromptTagRuntime {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "bind-prompt-tag-runtime"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::prompt_tag::bind_wasm_prompt_tag_runtime(ctx, target.into());
        Ok(())
    }
}

struct IntrinsicToWasm;

impl Pass for IntrinsicToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "intrinsic-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::intrinsic_to_wasm::lower(ctx, target.into());
        Ok(())
    }
}

struct LowerWasmModule;

impl Pass for LowerWasmModule {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "wasm-lowerer"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        WasmLowerer::new().lower_module(ctx, target.into());
        Ok(())
    }
}

struct ConvertStructLayouts;

impl Pass for ConvertStructLayouts {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "convert-struct-layouts"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        super::struct_layouts::convert(ctx, target.into());
        Ok(())
    }
}

/// Assign module-local GC indices after all type-conversion materialization.
pub(crate) fn finalize_wasm_gc_types(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), ConversionError> {
    trunk_ir_wasm_backend::passes::wasm_gc_to_wasm::lower(ctx, module);
    PatternApplicator::new(TypeConverter::new())
        .with_target(wasm_emission_ready_target())
        .apply_partial_conversion(ctx, module, "wasm-emission-ready")?;
    if cfg!(debug_assertions) {
        check_all_wasm_dialect(ctx, module);
    }
    Ok(())
}

/// Run the full WASM lowering pipeline on arena IR.
///
/// The structured-control boundary check runs first, before any mutation.
/// The lowering steps then run as passes of one [`PassManager`], so a debug
/// build verifies IR invariants after every step that changed the IR, and
/// every step shares the run's analyses. Each step reads only its input IR:
/// a step that relies on a module-level resource, such as a data segment, an
/// import, or linear memory, declares it in the IR.
pub struct LowerToWasm;

impl Pass for LowerToWasm {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-to-wasm"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower_to_wasm(ctx, target.into(), analyses)?;
        Ok(())
    }
}

/// A module pass that legalizes the `unrealized_conversion_cast` operations
/// left after Wasm lowering with the Wasm type converter: it converts their
/// result types, materializes real representation changes, and writes WasmGC
/// reference upcasts as `wasm.ref_upcast`.
/// [`trunk_ir::conversion::ReconcileUnrealizedCasts`] then
/// folds the identities left behind, and the emission boundary in
/// [`FinalizeWasmGcTypes`] rejects any cast that remains.
pub struct LegalizeUnrealizedCasts;

impl Pass for LegalizeUnrealizedCasts {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "legalize-wasm-unrealized-casts"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let tc = wasm_type_converter(ctx);
        let result = PatternApplicator::new(tc)
            .add_pattern(UnrealizedCastConversionPattern)
            .add_pattern(ReferenceUpcastPattern)
            .apply_partial(ctx, Module::from(target));
        if !result.reached_fixpoint {
            warn!("wasm cast legalization did not reach a fixpoint");
        }
        Ok(())
    }
}

/// Assign module-local GC indices after all type-conversion materialization,
/// and reject any operation the Wasm emitter does not accept.
pub struct FinalizeWasmGcTypes;

impl Pass for FinalizeWasmGcTypes {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "finalize-wasm-gc-types"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        finalize_wasm_gc_types(ctx, target.into())?;
        Ok(())
    }
}

/// Verify the partial Wasm backend boundary.
///
/// Unknown operations are still allowed because unrealized casts and backend
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
            if wasm_dialect::Func::matches(ctx, op) {
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
            // Check for func.func or wasm.func operations
            if func::Func::matches(ctx, op) {
                if let Some(fn_ty) = ctx.op_attrs(op).get_type("type") {
                    let Some(function) = func::FuncSig::from_type_ref(ctx, fn_ty) else {
                        continue;
                    };
                    let params: Vec<_> = function
                        .inputs(ctx)
                        .iter()
                        .map(|t| {
                            let td = ctx.get_type(*t);
                            format!("{}", td.name.display(ctx))
                        })
                        .collect();
                    let sym_name = ctx.op_attrs(op).get_str("sym_name").unwrap_or_default();
                    tracing::debug!("[{phase}] func.func {sym_name}: params={params:?}");
                }
            } else if wasm_dialect::Func::matches(ctx, op)
                && let Some(fn_ty) = ctx.op_attrs(op).get_type("type")
            {
                let Some(function) = wasm_dialect::FuncSig::from_type_ref(ctx, fn_ty) else {
                    continue;
                };
                let params: Vec<_> = function
                    .inputs(ctx)
                    .iter()
                    .map(|t| {
                        let td = ctx.get_type(*t);
                        format!("{}", td.name.display(ctx))
                    })
                    .collect();
                let sym_name = ctx.op_attrs(op).get_str("sym_name").unwrap_or_default();
                tracing::debug!("[{phase}] wasm.func {sym_name}: params={params:?}");
            }
        }
    }
}

/// Check if all operations in a function body are in wasm dialect
fn check_function_body(ctx: &IrContext, func_op: OpRef) {
    if let Some(body_region) = ctx.op_region(func_op, 0) {
        for &block in ctx.region(body_region).blocks.iter() {
            for &op in ctx.block(block).ops.iter() {
                let name = ctx.op(op).name;
                if name.dialect(ctx) != wasm_dialect::DIALECT_NAME {
                    error!(
                        "Found non-wasm operation in function body: {}",
                        name.display(ctx)
                    );
                }
            }
        }
    }
}

// =============================================================================
// Type helpers
// =============================================================================

// =============================================================================
// MemoryPlan (arena version)
// =============================================================================

struct ArenaMemoryPlan {
    has_memory: bool,
    has_exported_memory: bool,
}

impl ArenaMemoryPlan {
    fn new() -> Self {
        Self {
            has_memory: false,
            has_exported_memory: false,
        }
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
/// 2. Appends module-level extras (memory and `_start` exports)
///
/// Imports, memory, and data segments are declared by the steps that need
/// them; the lowerer only exports what the module already declares.
struct WasmLowerer {
    memory_plan: ArenaMemoryPlan,
    /// Whether the module defines the root `main`.
    saw_main: bool,
}

impl WasmLowerer {
    fn new() -> Self {
        Self {
            memory_plan: ArenaMemoryPlan::new(),
            saw_main: false,
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

        // Phase 2: Append extra ops at the end
        self.append_extra_ops(ctx, module_block, location);
    }

    /// Scan all module-level ops to collect metadata about main function,
    /// memory declarations, and exports.
    fn scan_module_ops(&mut self, ctx: &IrContext, body: RegionRef) {
        for &block in ctx.region(body).blocks.iter() {
            for &op in ctx.block(block).ops.iter() {
                let data = ctx.op(op);

                if data.name.dialect(ctx) == wasm_dialect::DIALECT_NAME {
                    // Track wasm module-level metadata
                    if wasm_dialect::Memory::matches(ctx, op) {
                        self.memory_plan.has_memory = true;
                    } else if wasm_dialect::ExportMemory::matches(ctx, op) {
                        self.memory_plan.has_exported_memory = true;
                    } else if wasm_dialect::Func::matches(ctx, op) {
                        self.scan_wasm_func(ctx, op);
                    }
                }

                // Debug: warn about unexpected dialects
                if cfg!(debug_assertions) {
                    let allowed = ["wasm", "core"];
                    if !allowed.contains(&data.name.dialect(ctx)) {
                        warn!(
                            "Unhandled operation in lowering: {} (this may cause emit errors)",
                            data.name.display(ctx)
                        );
                    }
                }
            }
        }
    }

    /// Check if a wasm.func op is the main function and record its metadata.
    fn scan_wasm_func(&mut self, ctx: &IrContext, op: OpRef) {
        let Some(sym_name) = ctx.op_attrs(op).get_str("sym_name") else {
            return;
        };

        // Only match root-level main, not foo::main
        if sym_name != "main" {
            return;
        }

        self.saw_main = true;
    }

    /// Append module-level extra ops (memory and `_start` exports).
    fn append_extra_ops(
        &mut self,
        ctx: &mut IrContext,
        module_block: BlockRef,
        location: Location,
    ) {
        if self.memory_plan.has_memory && !self.memory_plan.has_exported_memory {
            let name = ctx.intern_str("memory");
            let op = wasm_dialect::ExportMemory::operands()
                .name(name)
                .index(0)
                .build(ctx, location);
            ctx.push_op(module_block, op.op_ref());
            self.memory_plan.has_exported_memory = true;
        }

        // The boundary exit verifies that the root `main` is the parameterless
        // wrapper that nothing references, so it is the WASI command entry
        // itself.
        if self.saw_main {
            let name = ctx.intern_str("_start");
            let export_op = wasm_dialect::ExportFunc::operands()
                .name(name)
                .func(SymbolPath::new(["main"]))
                .build(ctx, location);
            ctx.push_op(module_block, export_op.op_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::SymbolPath;
    use trunk_ir::context::{BlockData, RegionData};
    use trunk_ir::dialect::core;
    use trunk_ir::refs::TypeRef;
    use trunk_ir::smallvec::smallvec;

    fn intern_func_type(ctx: &mut IrContext, params: Vec<TypeRef>, result: TypeRef) -> TypeRef {
        wasm_dialect::func_sig(ctx, params, [result]).as_type_ref()
    }
    use trunk_ir::Span;
    use trunk_ir::dialect::wasm_gc;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::refs::PathRef;
    use trunk_ir::rewrite::LegalityCheck;

    #[test]
    fn adt_struct_get_converts_evidence_result_consistently() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Evidence = adt.array<adt.typeref<{name = "std::runtime::evidence::Marker"}>>
  !Closure = adt.struct<_closure(code: core.i32), {layout = "closure"}>
  !Frame = adt.struct<Frame()>
  !Env = adt.struct<Env(closure: !Closure, evidence: !Evidence, frame: !Frame)>
  wasm.func @worker(%evidence: !Evidence, %closure: !Closure, %frame: !Frame) {
    %env = adt.struct_new %closure, %evidence, %frame {type = !Env} : !Env
    %loaded = adt.struct_get %env {field = 1, type = !Env} : !Evidence
    wasm.return
  }
}"#,
        );

        let tc = wasm_type_converter(&mut ctx);
        PatternApplicator::new(tc)
            .add_pattern(WasmFuncSignatureConversionPattern)
            .apply_partial(&mut ctx, module);
        let tc = wasm_type_converter(&mut ctx);
        crate::wasm::descriptors::declare(&mut ctx, module);
        crate::wasm::adt_to_wasm::lower(&mut ctx, module, tc);
        crate::wasm::type_converter::convert_evidence_arrays(&mut ctx, module);

        let func = module.ops(&ctx)[0];
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let operations = ctx.block(block).ops.clone();
        let struct_new = operations
            .iter()
            .find_map(|&op| wasm_gc::StructNew::from_op(&ctx, op).ok())
            .expect("environment construction should lower to wasm_gc.struct_new");
        let struct_get = operations
            .iter()
            .find_map(|&op| wasm_gc::StructGet::from_op(&ctx, op).ok())
            .expect("environment read should lower to wasm_gc.struct_get");
        // Evidence is its GC array instead of erasing to arrayref.
        let evidence = crate::wasm::type_converter::evidence_wasm_type(&mut ctx);

        // The descriptor field precedes the source fields.
        assert_eq!(ctx.value_ty(struct_new.fields(&ctx)[2]), evidence);
        assert_eq!(struct_get.field_idx(&ctx), 2);
        assert_eq!(struct_get.result_ty(&ctx), evidence);
        assert_eq!(
            ctx.type_attrs(struct_get.r#type(&ctx)).get_str("name"),
            Some("Env")
        );

        finalize_wasm_gc_types(&mut ctx, module)
            .expect("the converted environment field should have one GC representation");
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
    fn wasm_start_exports_parameterless_main() {
        let mut ctx = IrContext::new();
        let module = empty_module_with_block(&mut ctx);
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let module_block = module.first_block(&ctx).expect("module body block");
        let mut lowerer = WasmLowerer::new();
        lowerer.saw_main = true;

        lowerer.append_extra_ops(&mut ctx, module_block, location);

        let exports: Vec<_> = module
            .ops(&ctx)
            .iter()
            .copied()
            .filter_map(|op| wasm_dialect::ExportFunc::from_op(&ctx, op).ok())
            .map(|export| (export.name(&ctx), export.func(&ctx).clone()))
            .collect();
        assert_eq!(exports, [("_start", SymbolPath::new(["main"]))]);
        assert!(
            module
                .ops(&ctx)
                .iter()
                .copied()
                .all(|op| wasm_dialect::Func::from_op(&ctx, op).is_err()),
            "no separate start function"
        );
    }

    #[test]
    fn module_lowerer_exports_declared_memory() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.memory {min = 1, max = 0, shared = false, memory64 = false}
  func.func @run() -> core.i32 {
    %value = arith.const {value = 1} : core.i32
    func.return %value
  }
}"#,
        );
        let mut lowerer = WasmLowerer::new();
        let module_block = module.first_block(&ctx).expect("module body block");
        let placeholder = ctx.block(module_block).ops[1];

        lowerer.lower_module(&mut ctx, module);
        trunk_ir::rewrite::erase_op(&mut ctx, placeholder);

        let location = Location::new(PathRef::from_u32(0), Span::default());
        let i32_ty = core::I32::type_ref(&mut ctx);
        let i64_ty = core::I64::type_ref(&mut ctx);
        let f32_ty = core::F32::type_ref(&mut ctx);
        let f64_ty = core::F64::type_ref(&mut ctx);
        let nil_ty = core::Nil::params().intern(&mut ctx).as_type_ref();
        let body_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let address = wasm_dialect::I32Const::operands()
            .value(0)
            .results(i32_ty)
            .build(&mut ctx, location);
        let address_result = address.result(&ctx);
        ctx.push_op(body_block, address.op_ref());
        let i64_load = wasm_dialect::I64Load::operands(address_result)
            .offset(0)
            .align(0)
            .memory(0)
            .results(i64_ty)
            .build(&mut ctx, location);
        let f32_load = wasm_dialect::F32Load::operands(address_result)
            .offset(0)
            .align(0)
            .memory(0)
            .results(f32_ty)
            .build(&mut ctx, location);
        let f64_load = wasm_dialect::F64Load::operands(address_result)
            .offset(0)
            .align(0)
            .memory(0)
            .results(f64_ty)
            .build(&mut ctx, location);
        let i64_result = i64_load.result(&ctx);
        let f32_result = f32_load.result(&ctx);
        let f64_result = f64_load.result(&ctx);
        for load in [i64_load.op_ref(), f32_load.op_ref(), f64_load.op_ref()] {
            ctx.push_op(body_block, load);
        }
        for store in [
            wasm_dialect::I64Store::operands(address_result, i64_result)
                .offset(0)
                .align(0)
                .memory(0)
                .build(&mut ctx, location)
                .op_ref(),
            wasm_dialect::F32Store::operands(address_result, f32_result)
                .offset(0)
                .align(0)
                .memory(0)
                .build(&mut ctx, location)
                .op_ref(),
            wasm_dialect::F64Store::operands(address_result, f64_result)
                .offset(0)
                .align(0)
                .memory(0)
                .build(&mut ctx, location)
                .op_ref(),
        ] {
            ctx.push_op(body_block, store);
        }
        let ret = wasm_dialect::Return::operands(vec![]).build(&mut ctx, location);
        ctx.push_op(body_block, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location,
            blocks: smallvec![body_block],
            parent_op: None,
        });
        let func_ty = intern_func_type(&mut ctx, vec![], nil_ty);
        let memory_func = wasm_dialect::Func::operands()
            .sym_name("memory_ops")
            .r#type(func_ty)
            .regions(body)
            .build(&mut ctx, location);
        ctx.push_op(module_block, memory_func.op_ref());

        let output = print_module(&ctx, module.op());
        assert_eq!(output.matches("wasm.memory").count(), 1, "{output}");
        assert_eq!(output.matches("wasm.export_memory").count(), 1, "{output}");

        let binary = trunk_ir_wasm_backend::emit_module_to_wasm(&mut ctx, module)
            .expect("lowered module requirements should emit");
        assert_eq!(&binary.bytes[..4], b"\0asm");
    }

    #[test]
    fn module_lowerer_recognizes_existing_memory_and_main_signature() {
        let mut ctx = IrContext::new();
        let module = empty_module_with_block(&mut ctx);
        let location = Location::new(PathRef::from_u32(0), Span::default());
        let module_block = module.first_block(&ctx).expect("module body block");
        let i32_ty = core::I32::type_ref(&mut ctx);
        let nil_ty = core::Nil::params().intern(&mut ctx).as_type_ref();

        let memory = wasm_dialect::Memory::operands()
            .min(1)
            .max(0)
            .shared(false)
            .memory64(false)
            .build(&mut ctx, location);
        let name = ctx.intern_str("memory");
        let export_memory = wasm_dialect::ExportMemory::operands()
            .name(name)
            .index(0)
            .build(&mut ctx, location);
        for op in [memory.op_ref(), export_memory.op_ref()] {
            ctx.push_op(module_block, op);
        }

        let main_block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret = wasm_dialect::Return::operands(vec![]).build(&mut ctx, location);
        ctx.push_op(main_block, ret.op_ref());
        let main_body = ctx.create_region(RegionData {
            location,
            blocks: smallvec![main_block],
            parent_op: None,
        });
        let main_ty = intern_func_type(&mut ctx, vec![i32_ty], nil_ty);
        let main = wasm_dialect::Func::operands()
            .sym_name("main")
            .r#type(main_ty)
            .regions(main_body)
            .build(&mut ctx, location);
        ctx.push_op(module_block, main.op_ref());

        let mut lowerer = WasmLowerer::new();
        lowerer.scan_module_ops(&ctx, module.body(&ctx).expect("module body"));

        assert!(lowerer.memory_plan.has_memory);
        assert!(lowerer.memory_plan.has_exported_memory);
        assert!(lowerer.saw_main);

        ctx.op_attrs_mut(main.op_ref()).remove("sym_name");
        lowerer.saw_main = false;
        lowerer.scan_wasm_func(&ctx, main.op_ref());
        assert!(!lowerer.saw_main);
    }

    #[test]
    fn module_lowerer_does_not_add_unused_globals_to_an_empty_module() {
        let mut ctx = IrContext::new();
        let module = empty_module_with_block(&mut ctx);
        let mut lowerer = WasmLowerer::new();

        lowerer.lower_module(&mut ctx, module);

        assert!(module.ops(&ctx).is_empty());
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
        let mut lowerer = WasmLowerer::new();

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
    %result = effect.dispatch_tail %ev, %payload {ability = @Console, op_name = "read"} : wasm.anyref
    func.return %result
  }
}"#,
        );

        let error = verify_wasm_backend_ready(&mut ctx, module)
            .expect_err("residual effect operation should fail wasm backend boundary");

        assert_eq!(error.boundary(), WASM_BACKEND_READY_BOUNDARY);
        assert_eq!(error.operations().len(), 1);
        assert_eq!(error.operations()[0].name, "effect.dispatch_tail");
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
    fn wasm_emission_ready_rejects_remaining_unrealized_cast() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main(%x: core.i32) -> core.i64 {
    %r = core.unrealized_conversion_cast %x : core.i64
    wasm.return %r
  }
}"#,
        );

        let error = finalize_wasm_gc_types(&mut ctx, module)
            .expect_err("a remaining cast should fail the wasm emission boundary");

        assert_eq!(error.operations().len(), 1);
        assert_eq!(
            error.operations()[0].name,
            "core.unrealized_conversion_cast"
        );
        assert_eq!(error.operations()[0].legality, LegalityCheck::Illegal);
    }

    #[test]
    fn lower_to_wasm_rejects_non_reference_string_constant_result() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !String = adt.enum<String { Leaf(core.bytes), Branch(tribute_rt.anyref, tribute_rt.anyref, core.i32) }>
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

        let error = lower_to_wasm(&mut ctx, module, &mut Default::default())
            .expect_err("non-reference string constant result must be rejected");

        assert!(
            error
                .to_string()
                .contains("adt.string_const must produce wasm.anyref"),
            "{error}"
        );
    }

    #[test]
    fn lower_to_wasm_rejects_used_never_before_io_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main(%bytes: core.bytes, %cond: core.i1) -> core.nil {
    %write = tribute_io.write %bytes, %cond : core.nil
    %never = scf.if %cond : core.never {
      func.unreachable
    } {
      func.unreachable
    }
    func.return %never
  }
}"#,
        );
        let before = print_module(&ctx, module.op());
        let error = lower_to_wasm(&mut ctx, module, &mut Default::default())
            .expect_err("used Never must reject");
        assert!(matches!(error, WasmLowerError::Conversion(_)), "{error}");
        assert!(error.to_string().contains("scf-to-wasm"), "{error}");
        assert!(
            error.to_string().contains("Never control requires"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn lower_to_wasm_rejects_nonlowerable_switch_before_io_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @main(%bytes: core.bytes, %newline: core.i1, %choice: core.i16) -> core.nil {
    %write = tribute_io.write %bytes, %newline : core.nil
    scf.switch %choice {
      scf.case {value = 0} { scf.yield }
      scf.default { scf.yield }
    }
    func.return
  }
}"#,
        );
        let before = print_module(&ctx, module.op());
        assert!(before.contains("tribute_io.write"), "{before}");

        let error = lower_to_wasm(&mut ctx, module, &mut Default::default())
            .expect_err("nonlowerable switch must fail the wasm lowering boundary");

        assert!(matches!(error, WasmLowerError::Conversion(_)), "{error}");
        assert!(error.to_string().contains("scf-to-wasm"), "{error}");
        assert!(
            error.to_string().contains(
                "unsupported discriminant type `core.i16`; expected `core.i32` or `core.i64`"
            ),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn lower_to_wasm_declares_each_module_resource_once() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%newline: core.i1) -> core.nil {
    %first = adt.bytes_const {value = b"first"} : core.bytes
    %second = adt.bytes_const {value = b"second"} : core.bytes
    %again = adt.bytes_const {value = b"first"} : core.bytes
    %write = tribute_io.write %first, %newline : core.nil
    %rewrite = tribute_io.write %again, %newline : core.nil
    func.return
  }
}"#,
        );

        lower_to_wasm(&mut ctx, module, &mut Default::default())
            .expect("module with output and constants should lower to wasm");

        let output = print_module(&ctx, module.op());
        assert_eq!(output.matches("wasm.import_func").count(), 1, "{output}");
        assert_eq!(output.matches("wasm.memory {").count(), 1, "{output}");
        assert_eq!(output.matches("wasm.export_memory").count(), 1, "{output}");
        assert_eq!(output.matches("wasm.data").count(), 2, "{output}");
        assert!(!output.contains("tribute_io.write"), "{output}");
    }

    #[test]
    fn wasm_lower_error_wraps_conversion_error() {
        let conversion = ConversionError::new(WASM_BACKEND_READY_BOUNDARY, vec![]);
        let error = WasmLowerError::from(conversion);

        assert!(error.to_string().contains(WASM_BACKEND_READY_BOUNDARY));
        assert!(std::error::Error::source(&error).is_some());
    }
}

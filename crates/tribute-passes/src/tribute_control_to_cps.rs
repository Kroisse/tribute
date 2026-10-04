//! Atomic legalization of verified `tribute_control` callable/control IR.
//!
//! The pass deliberately stops at the physical `func`/`closure` and logical
//! `ability` surface. It does not run closure extraction, evidence lowering,
//! or target-specific conversion.

use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;
use tribute_ir::continuation_frame;

use tribute_core::calling_convention::{
    cps_closure_function_type, cps_completion_type, cps_done_type, cps_resume_exact_type,
    cps_resume_type, physical_closure_function_type, physical_closure_type_with_environment_index,
};
use tribute_core::{
    CALLING_CONVENTION_ATTR, CallableAbi, CallingConvention, physical_closure_type,
    set_calling_convention,
};
use tribute_ir::dialect::adt;
use tribute_ir::dialect::{ability, closure, effect, tribute_control, tribute_rt};
use trunk_ir::OpList;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{arith, core, func, scf};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{ConversionMode, ConversionTarget, Module};
use trunk_ir::symbol_table::{SymbolTable, qualified_name};
use trunk_ir::types::{Attribute, AttributeMap, Location, StringRef, TypeDataBuilder};
use trunk_ir::{OperationDataBuilder, Symbol, SymbolPath};

mod boundary;
mod callable;
mod frame;
mod handle;
mod structured;
#[cfg(test)]
mod tests;
mod types;

pub use boundary::*;

/// Carry a source call's, resume's, or handle's evidence selection to the
/// operation that passes its evidence. The selection is copied unchanged.
fn carry_evidence_plan(ctx: &mut IrContext, source: OpRef, target: OpRef) {
    let plan = evidence_plan_of(ctx, source);
    set_evidence_plan(ctx, target, plan);
}

/// The `evidence_plan` of a source call, resume, or handle.
fn evidence_plan_of(ctx: &IrContext, source: OpRef) -> Option<Attribute> {
    ctx.op(source)
        .attributes
        .get(tribute_control::EVIDENCE_PLAN_ATTR)
        .cloned()
}

/// Put a selection on the operation that passes the evidence it selects.
fn set_evidence_plan(ctx: &mut IrContext, target: OpRef, plan: Option<Attribute>) {
    if let Some(plan) = plan {
        ctx.op_mut(target)
            .attributes
            .insert(tribute_control::EVIDENCE_PLAN_ATTR, plan);
    }
}

fn convert_convention(convention: tribute_control::CallingConvention) -> CallingConvention {
    match convention {
        tribute_control::CallingConvention::Direct => CallingConvention::Direct,
        tribute_control::CallingConvention::EvidenceDirect => CallingConvention::EvidenceDirect,
        tribute_control::CallingConvention::Cps => CallingConvention::Cps,
    }
}

#[derive(Clone)]
struct CallableInfo {
    symbol: SymbolPath,
    convention: CallingConvention,
    source_result: TypeRef,
    source_params: Vec<TypeRef>,
}

#[derive(Clone)]
struct HandlerArmInfo {
    op: OpRef,
    value: ValueRef,
    ability_ref: TypeRef,
    op_name: StringRef,
    general: bool,
    params: Vec<TypeRef>,
    has_resume_token: bool,
}

struct Converter<'a> {
    ctx: &'a mut IrContext,
    module_block: BlockRef,
    /// Callables by root-qualified name across the whole module tree.
    funcs: HashMap<SymbolPath, CallableInfo>,
    converted_types: HashMap<TypeRef, TypeRef>,
    frames: HashMap<TypeRef, FrameTypes>,
    frame_layout_aliases: Vec<(Symbol, TypeRef)>,
    helper_index: u32,
}

#[derive(Clone, Copy)]
struct FrameTypes {
    reference: TypeRef,
    layout: TypeRef,
    done: TypeRef,
    dispatch: TypeRef,
}

/// The static description of one `handle`, shared by every layer that
/// installs it: the first installation and each rebuilt continuation layer.
#[derive(Clone)]
struct HandleLayer {
    /// The source `handle`, which carries the installation's `evidence_plan`.
    source: OpRef,
    arms: Vec<HandlerArmInfo>,
    body_type: TypeRef,
    answer_type: TypeRef,
    /// Builds the dispatcher of an installed layer.
    dispatch_factory: Symbol,
    /// Builds the dispatcher of a layer resumed from a lambda, which keeps
    /// only the handle's completion.
    passthrough_factory: Symbol,
}

/// A call, resume, or structured suffix layer of a continuation.
#[derive(Clone)]
struct SuffixLayer {
    value_type: TypeRef,
    boundary: TypeRef,
    /// Builds the dispatcher that rebuilds this layer when it is resumed.
    dispatch_factory: Symbol,
    /// The `evidence_plan` that selects the evidence of the computation the
    /// layer continues.
    plan: Option<Attribute>,
}

/// The values one installed layer of a handle runs with.
#[derive(Clone)]
struct LayerValues {
    completion: ValueRef,
    prompt: ValueRef,
    /// The handler arm closures, in `HandleLayer::arms` order.
    arms: Vec<ValueRef>,
}

/// How a `resume` in the body of a handler arm reaches its continuation.
#[derive(Clone)]
struct ArmResume {
    /// The source resume token of the arm.
    token: ValueRef,
    /// The token that installs the arm's handle again on the evidence it is
    /// called with and resumes under it.
    installed: ValueRef,
    evidence: ArmEvidence,
}

/// Where the evidence of a handler arm is found at a point of its body.
#[derive(Clone)]
enum ArmEvidence {
    /// The flow's evidence.
    Flow,
    /// The flow's evidence beneath the top handler of each of these
    /// instances: inside handle bodies nested in the arm that installed them
    /// without a selection.
    Beneath(Vec<TypeRef>),
    /// The evidence a handle nested in the arm was installed on. A nested
    /// handle that masks an instance leaves no way to recover the arm's
    /// evidence from its body's.
    Captured(ValueRef),
}

/// The operations of a source block that follow the one being converted.
#[derive(Clone, Copy)]
struct Rest<'a> {
    ops: &'a [OpRef],
    start: usize,
}

#[derive(Clone)]
struct Flow {
    convention: CallingConvention,
    evidence: Option<ValueRef>,
    exit_k: Option<ValueRef>,
    root_exit_k: Option<ValueRef>,
    /// Zero-result structured control uses a private suffix closure whose
    /// arguments are `(Evidence, ContinuationFrame<R>)` rather than `Done<R>`.
    void_exit_k: Option<ValueRef>,
    answer_type: TypeRef,
    preserve_scf_yield: bool,
    /// Set in the body of a resumptive handler arm, outside its lambdas.
    arm: Option<ArmResume>,
}

impl<'a> Converter<'a> {
    fn new(
        ctx: &'a mut IrContext,
        module_block: BlockRef,
        funcs: HashMap<SymbolPath, CallableInfo>,
    ) -> Self {
        Self {
            ctx,
            module_block,
            funcs,
            converted_types: HashMap::default(),
            frames: HashMap::default(),
            frame_layout_aliases: Vec::new(),
            helper_index: 0,
        }
    }

    fn current_func(&self, symbol: &SymbolPath) -> Option<CallableInfo> {
        self.funcs.get(symbol).cloned()
    }

    fn malformed_source(
        &self,
        source: OpRef,
        message: impl Into<String>,
    ) -> TributeControlToCpsError {
        TributeControlToCpsError::one(
            PRE_CPS_BOUNDARY,
            Some(source),
            Some(self.ctx.op(source).location),
            message,
        )
    }

    fn fresh_helper(&mut self, prefix: &str) -> Symbol {
        let index = self.helper_index;
        self.helper_index += 1;
        Symbol::from_dynamic(&format!("__tribute_{prefix}_{index}"))
    }

    fn make_block(&mut self, location: Location, types: &[TypeRef]) -> BlockRef {
        self.ctx.create_block(BlockData {
            location,
            args: types
                .iter()
                .copied()
                .map(|ty| BlockArgData {
                    ty,
                    attrs: AttributeMap::new(),
                })
                .collect(),
            ops: Default::default(),
            parent_region: None,
        })
    }

    fn single_block_region(&mut self, location: Location, block: BlockRef) -> RegionRef {
        self.ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        })
    }

    fn current_evidence(
        &self,
        source: OpRef,
        flow: &Flow,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        flow.evidence.ok_or_else(|| {
            TributeControlToCpsError::post_op(
                source,
                self.ctx.op(source).location,
                "operation requires evidence but the enclosing callable convention is Direct",
            )
        })
    }

    fn convert_sequence(
        &mut self,
        source_ops: OpList,
        mut index: usize,
        block: BlockRef,
        mapping: &mut HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<(), TributeControlToCpsError> {
        while index < source_ops.len() {
            let source = source_ops[index];
            let rest = Rest {
                ops: &source_ops,
                start: index + 1,
            };
            let location = self.ctx.op(source).location;
            let dialect = self.ctx.op(source).dialect.clone();
            let name = self.ctx.op(source).name.clone();
            if dialect == Symbol::new("scf") && name == Symbol::new("yield") {
                if flow.preserve_scf_yield {
                    let cloned = self.clone_plain_op(source, mapping)?;
                    self.ctx.push_op(block, cloned);
                    return Ok(());
                }
                let values = self.ctx.op_operands(source);
                if values.is_empty() {
                    self.emit_void_exit(block, location, flow)?;
                    return Ok(());
                }
                if values.len() != 1 {
                    return Err(TributeControlToCpsError::post_op(
                        source,
                        location,
                        "CPS structured exit requires exactly one scf.yield value",
                    ));
                }
                let value = mapping.get(&values[0]).copied().unwrap_or(values[0]);
                self.emit_exit(block, location, value, flow)?;
                return Ok(());
            }
            if dialect == Symbol::new("scf")
                && name == Symbol::new("switch")
                && self.contains_tribute_control(source)
            {
                self.lower_structured_switch(source, rest, block, mapping, flow)?;
                return Ok(());
            }
            if dialect == Symbol::new("scf")
                && name == Symbol::new("if")
                && self.contains_tribute_control(source)
            {
                self.lower_structured_if(source, rest, block, mapping, flow)?;
                return Ok(());
            }
            if dialect != Symbol::new("tribute_control") {
                let cloned = self.clone_plain_op(source, mapping)?;
                self.ctx.push_op(block, cloned);
                index += 1;
                continue;
            }

            match name.with_str(|value| value.to_owned()).as_str() {
                "return" | "yield" => {
                    let value = self.ctx.op_operands(source)[0];
                    let value = mapping.get(&value).copied().unwrap_or(value);
                    self.emit_exit(block, location, value, flow)?;
                    return Ok(());
                }
                "lambda" => {
                    let lambda = self.lower_lambda(source, mapping)?;
                    self.ctx.push_op(block, lambda);
                    mapping.insert(self.ctx.op_result(source, 0), self.ctx.op_result(lambda, 0));
                    index += 1;
                }
                "func_ref" => {
                    let (ops, value) = self.lower_func_ref(source)?;
                    for op in ops {
                        self.ctx.push_op(block, op);
                    }
                    mapping.insert(self.ctx.op_result(source, 0), value);
                    index += 1;
                }
                "call" => {
                    if self
                        .lower_call(source, rest, block, mapping, flow)?
                        .is_break()
                    {
                        return Ok(());
                    }
                    index += 1;
                }
                "call_indirect" => {
                    if self
                        .lower_call_indirect(source, rest, block, mapping, flow)?
                        .is_break()
                    {
                        return Ok(());
                    }
                    index += 1;
                }
                "perform" => {
                    let kind = self
                        .ctx
                        .op(source)
                        .attributes
                        .get_str(self.ctx, "operation_kind")
                        .expect("pre-CPS validation checked operation kind");
                    if kind == "fn" {
                        self.lower_tail_perform(source, block, mapping, flow)?;
                        index += 1;
                    } else {
                        self.lower_general_perform(source, rest, block, mapping, flow)?;
                        return Ok(());
                    }
                }
                "resume" => {
                    self.lower_resume(source, rest, block, mapping, flow)?;
                    return Ok(());
                }
                "handle" => {
                    self.lower_handle(source, rest, block, mapping, flow)?;
                    return Ok(());
                }
                other => {
                    return Err(TributeControlToCpsError::post_op(
                        source,
                        location,
                        format!("unsupported tribute_control operation '{other}'"),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Every source callable, keyed by its root-qualified name.
fn collect_callable_graph(
    ctx: &IrContext,
    symbols: &SymbolTable,
) -> HashMap<SymbolPath, CallableInfo> {
    symbols
        .iter()
        .filter(|&(_, ops)| tribute_control::Func::matches(ctx, ops[0]))
        .map(|(symbol, ops)| {
            // Pre-CPS validation rejects duplicated qualified names.
            let op = ops[0];
            let logical_type = ctx
                .op(op)
                .attributes
                .get_type("type")
                .expect("pre-CPS validation checked function type");
            let callable = tribute_control::FuncSig::from_type_ref(ctx, logical_type)
                .expect("pre-CPS validation checked callable type");
            let convention = tribute_control::func_sig_convention(ctx, logical_type)
                .expect("pre-CPS validation checked callable convention");
            (
                symbol.clone(),
                CallableInfo {
                    symbol: symbol.clone(),
                    convention: convert_convention(convention),
                    source_result: callable.result(ctx),
                    source_params: callable.inputs(ctx).to_vec(),
                },
            )
        })
        .collect()
}

fn verify_candidate_or_restore_aliases(
    ctx: &mut IrContext,
    candidate: Module,
    source_aliases: &[(Symbol, TypeRef)],
    analyses: &mut AnalysisCache,
) -> Result<(), TributeControlToCpsError> {
    if let Err(error) = verify_tribute_control_post_cps(ctx, candidate, analyses) {
        for (name, ty) in source_aliases {
            ctx.register_type_alias(name.clone(), *ty);
        }
        ctx.remove_op(candidate.op());
        return Err(error);
    }
    Ok(())
}

/// Atomically convert the complete logical callable/control graph.
///
/// The existing module region is not detached until a separately built module
/// has passed the post-CPS operation and recursive type boundary.
///
/// Boundary verification and graph collection share the analyses cached in
/// `analyses` while they read unchanged IR.
pub fn tribute_control_to_cps(
    ctx: &mut IrContext,
    module: Module,
    declarations: &[tribute_control::OperationDeclaration],
    compiler_intrinsics: &[tribute_control::CompilerIntrinsicDeclaration],
    analyses: &mut AnalysisCache,
) -> Result<(), TributeControlToCpsError> {
    verify_tribute_control_pre_cps(ctx, module, declarations, compiler_intrinsics, analyses)?;
    let funcs = collect_callable_graph(ctx, &analyses.require::<SymbolTable>(ctx, module.op()));
    let source_region = module.body(ctx).ok_or_else(|| {
        TributeControlToCpsError::one(
            PRE_CPS_BOUNDARY,
            Some(module.op()),
            Some(ctx.op(module.op()).location),
            "core.module has no body region",
        )
    })?;
    let source_blocks = ctx.region(source_region).blocks.clone();
    if source_blocks.len() != 1 {
        return Err(TributeControlToCpsError::one(
            PRE_CPS_BOUNDARY,
            Some(module.op()),
            Some(ctx.op(module.op()).location),
            "tribute_control_to_cps currently requires a single module block",
        ));
    }
    let module_location = ctx.op(module.op()).location;
    let source_aliases = ctx.type_aliases().to_vec();
    let mut converted_aliases = Vec::with_capacity(source_aliases.len());
    let new_block = ctx.create_block(BlockData {
        location: ctx.block(source_blocks[0]).location,
        args: vec![],
        ops: Default::default(),
        parent_region: None,
    });
    {
        let mut converter = Converter::new(ctx, new_block, funcs);
        let mut mapping = HashMap::default();
        let source_ops = converter.ctx.block(source_blocks[0]).ops.clone();
        for source in source_ops {
            if tribute_control::Func::matches(converter.ctx, source) {
                let function = converter.convert_func(source)?;
                converter.ctx.push_op(new_block, function);
            } else if converter.ctx.op(source).dialect == Symbol::new("tribute_control") {
                return Err(TributeControlToCpsError::one(
                    PRE_CPS_BOUNDARY,
                    Some(source),
                    Some(converter.ctx.op(source).location),
                    "only tribute_control.func may appear directly in a module block",
                ));
            } else {
                let cloned = converter.clone_plain_op(source, &mut mapping)?;
                converter.ctx.push_op(new_block, cloned);
            }
        }
        converted_aliases.extend(
            source_aliases
                .iter()
                .map(|(name, ty)| (name.clone(), converter.convert_type(*ty))),
        );
        converted_aliases.extend(converter.frame_layout_aliases.iter().cloned());
    }
    let new_region = ctx.create_region(RegionData {
        location: ctx.region(source_region).location,
        blocks: trunk_ir::smallvec::smallvec![new_block],
        parent_op: None,
    });
    let temp_symbol = Symbol::new("__tribute_control_to_cps_candidate");
    let temp_module = core::Module::operands()
        .sym_name(temp_symbol)
        .regions(new_region)
        .build(ctx, module_location);
    let candidate: Module = temp_module.into();
    for (name, ty) in &converted_aliases {
        ctx.register_type_alias(name.clone(), *ty);
    }
    verify_candidate_or_restore_aliases(ctx, candidate, &source_aliases, analyses)?;

    ctx.detach_region(new_region);
    ctx.remove_op(candidate.op());
    ctx.detach_region(source_region);
    ctx.push_op_region(module.op(), new_region);
    if let Err(error) = verify_tribute_control_post_cps(ctx, module, analyses) {
        ctx.detach_region(new_region);
        ctx.push_op_region(module.op(), source_region);
        for (name, ty) in &source_aliases {
            ctx.register_type_alias(name.clone(), *ty);
        }
        return Err(error);
    }
    Ok(())
}

/// Pass-manager wrapper carrying the verified source operation declarations.
pub struct TributeControlToCps {
    declarations: Vec<tribute_control::OperationDeclaration>,
    compiler_intrinsics: Vec<tribute_control::CompilerIntrinsicDeclaration>,
}

impl TributeControlToCps {
    pub fn new(
        declarations: impl IntoIterator<Item = tribute_control::OperationDeclaration>,
    ) -> Self {
        Self {
            declarations: declarations.into_iter().collect(),
            compiler_intrinsics: Vec::new(),
        }
    }

    pub fn with_compiler_intrinsics(
        mut self,
        declarations: impl IntoIterator<Item = tribute_control::CompilerIntrinsicDeclaration>,
    ) -> Self {
        self.compiler_intrinsics = declarations.into_iter().collect();
        self
    }
}

impl Pass for TributeControlToCps {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "tribute-control-to-cps"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        tribute_control_to_cps(
            ctx,
            target.into(),
            &self.declarations,
            &self.compiler_intrinsics,
            analyses,
        )
        .map_err(|error| Box::new(error) as _)
    }
}

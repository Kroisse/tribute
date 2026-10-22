//! Atomic legalization of verified `tribute_control` callable/control IR.
//!
//! The pass deliberately stops at the physical `func`/`closure` and logical
//! `ability` surface. It does not run closure extraction, evidence lowering,
//! or target-specific conversion.

use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;

use tribute_ir::callable_abi::CallableAbi;
use tribute_ir::calling_convention::{
    CALLING_CONVENTION_ATTR, CallingConvention, cps_completion_type, cps_resume_exact_type,
    physical_closure_function_type, physical_closure_type,
    physical_closure_type_with_environment_index, set_calling_convention,
};
use tribute_ir::dialect::{ability, closure, effect, tribute_control};
use trunk_ir::OpList;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::{core, func, scf};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{ConversionMode, ConversionTarget, Module};
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::{Attribute, AttributeMap, Location, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};
use trunk_ir::{OperationDataBuilder, Symbol, SymbolPath};

mod boundary;
mod callable;
mod delimit;
mod frame;
mod handle;
mod structured;
#[cfg(test)]
mod tests;
mod types;

use crate::cps_builders::*;
pub use boundary::*;

/// Carry a source call's, resume's, or handle's evidence selection to the
/// operation that passes its evidence. The selection is copied unchanged.
fn carry_evidence_plan(ctx: &mut IrContext, source: OpRef, target: OpRef) {
    let plan = evidence_plan_of(ctx, source);
    set_evidence_plan(ctx, target, plan);
}

/// The `evidence_plan` of a source call, resume, or handle.
fn evidence_plan_of(ctx: &IrContext, source: OpRef) -> Option<Attribute> {
    ctx.op_attrs(source)
        .get(tribute_control::EVIDENCE_PLAN_ATTR)
        .cloned()
}

#[derive(Clone)]
struct CallableInfo {
    convention: CallingConvention,
    source_result: TypeRef,
    source_params: Vec<TypeRef>,
    /// Whether the callable keeps a platform `abi`, so no proper tail
    /// transfer may target it.
    platform: bool,
}

struct Converter<'a> {
    ctx: &'a mut IrContext,
    module_block: BlockRef,
    /// Every source callable, by its defining operation.
    funcs: HashMap<OpRef, CallableInfo>,
    /// The callable each source operation's callee or `func_ref` names,
    /// resolved before conversion moves anything.
    targets: HashMap<OpRef, OpRef>,
    /// Source operations whose references are looked up in a nested module.
    converted_types: HashMap<TypeRef, TypeRef>,
    helper_index: u32,
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
    /// The evidence the handles nested in the arm were installed on, read
    /// from the flow's evidence: one handled instance per nested handle,
    /// outermost first. Each handle's marker records the evidence it was
    /// installed on.
    Outside(Vec<trunk_ir::SymbolPath>),
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
    /// Set in the arms of a structured operation in tail position that a
    /// source `become` leaves.
    tail_join: Option<Box<TailJoin>>,
}

/// The continuation of a structured operation in tail position. Only casts
/// and the enclosing terminator follow such an operation, so each arm runs
/// them itself instead of yielding to a join: a `become` in an arm then
/// transfers the callable's own frame, and the operation ends its block.
#[derive(Clone)]
struct TailJoin {
    structured: OpRef,
    ops: OpList,
    start: usize,
    /// The join of the structured operation around this one, whose arm the
    /// continuation ends with.
    outer: Option<Box<TailJoin>>,
}

impl<'a> Converter<'a> {
    fn new(ctx: &'a mut IrContext, module_block: BlockRef, graph: CallableGraph) -> Self {
        Self {
            ctx,
            module_block,
            funcs: graph.funcs,
            targets: graph.targets,
            converted_types: HashMap::default(),
            helper_index: 0,
        }
    }

    /// The callable the callee or `func_ref` of the source operation `op`
    /// names.
    fn current_func(&self, op: OpRef) -> Option<CallableInfo> {
        self.targets
            .get(&op)
            .and_then(|target| self.funcs.get(target))
            .cloned()
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
            let name = self.ctx.op(source).name;
            if name == scf::Yield::NAME {
                if let Some(join) = &flow.tail_join
                    && self.ctx.op(source).parent_block.and_then(|block| {
                        self.ctx
                            .block(block)
                            .parent_region
                            .and_then(|region| self.ctx.region(region).parent_op)
                    }) == Some(join.structured)
                {
                    let join = join.clone();
                    let values = self.ctx.op_operands(source).to_vec();
                    let result = self.ctx.op_result(join.structured, 0);
                    let [value] = values.as_slice() else {
                        return Err(self.malformed_source(
                            source,
                            "a tail-position structured arm yields exactly one value",
                        ));
                    };
                    let value = mapping.get(value).copied().unwrap_or(*value);
                    mapping.insert(result, value);
                    // The arm still holds the callable's evidence and frame,
                    // possibly as the arguments of a suffix continuation.
                    let rest_flow = Flow {
                        tail_join: join.outer.clone(),
                        ..flow.clone()
                    };
                    return self.convert_sequence(join.ops, join.start, block, mapping, &rest_flow);
                }
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
            if name == scf::Switch::NAME && self.contains_tribute_control(source) {
                self.lower_structured_switch(source, rest, block, mapping, flow)?;
                return Ok(());
            }
            if name == scf::If::NAME && self.contains_tribute_control(source) {
                self.lower_structured_if(source, rest, block, mapping, flow)?;
                return Ok(());
            }
            if name.dialect(self.ctx) != tribute_control::DIALECT_NAME {
                let cloned = self.clone_plain_op(source, mapping)?;
                self.ctx.push_op(block, cloned);
                index += 1;
                continue;
            }

            if flow.convention != CallingConvention::Cps && self.needs_cps_control(source) {
                let answer = self.delimit(source, block, mapping, flow)?;
                mapping.insert(self.ctx.op_result(source, 0), answer);
                index += 1;
                continue;
            }

            if tribute_control::Return::matches(self.ctx, source)
                || tribute_control::Yield::matches(self.ctx, source)
            {
                let value = self.ctx.op_operands(source)[0];
                let value = mapping.get(&value).copied().unwrap_or(value);
                self.emit_exit(block, location, value, flow)?;
                return Ok(());
            } else if tribute_control::Lambda::matches(self.ctx, source) {
                let lambda = self.lower_lambda(source, mapping)?;
                self.ctx.push_op(block, lambda);
                mapping.insert(self.ctx.op_result(source, 0), self.ctx.op_result(lambda, 0));
                index += 1;
            } else if tribute_control::Instantiate::matches(self.ctx, source) {
                // Rows leave with legalization, so a use is its lambda.
                let lambda = self.ctx.op_operands(source)[0];
                let lambda = mapping.get(&lambda).copied().unwrap_or(lambda);
                mapping.insert(self.ctx.op_result(source, 0), lambda);
                index += 1;
            } else if tribute_control::FuncRef::matches(self.ctx, source) {
                let (ops, value) = self.lower_func_ref(source)?;
                for op in ops {
                    self.ctx.push_op(block, op);
                }
                mapping.insert(self.ctx.op_result(source, 0), value);
                index += 1;
            } else if tribute_control::Call::matches(self.ctx, source) {
                if self
                    .lower_call(source, rest, block, mapping, flow)?
                    .is_break()
                {
                    return Ok(());
                }
                index += 1;
            } else if tribute_control::CallIndirect::matches(self.ctx, source) {
                if self
                    .lower_call_indirect(source, rest, block, mapping, flow)?
                    .is_break()
                {
                    return Ok(());
                }
                index += 1;
            } else if tribute_control::TailCall::matches(self.ctx, source) {
                self.lower_tail_call(source, block, mapping, flow)?;
                return Ok(());
            } else if tribute_control::TailCallIndirect::matches(self.ctx, source) {
                self.lower_tail_call_indirect(source, block, mapping, flow)?;
                return Ok(());
            } else if tribute_control::Perform::matches(self.ctx, source) {
                let kind = self
                    .ctx
                    .op_attrs(source)
                    .get_str("operation_kind")
                    .expect("pre-CPS validation checked operation kind");
                if kind == "fn" {
                    self.lower_tail_perform(source, block, mapping, flow)?;
                    index += 1;
                } else {
                    self.lower_general_perform(source, rest, block, mapping, flow)?;
                    return Ok(());
                }
            } else if tribute_control::Resume::matches(self.ctx, source) {
                self.lower_resume(source, rest, block, mapping, flow)?;
                return Ok(());
            } else if tribute_control::Handle::matches(self.ctx, source) {
                self.lower_handle(source, rest, block, mapping, flow)?;
                return Ok(());
            } else {
                return Err(TributeControlToCpsError::post_op(
                    source,
                    location,
                    format!(
                        "unsupported tribute_control operation '{}'",
                        name.display(self.ctx)
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// The source callables of a module and what its references name.
#[derive(Default)]
struct CallableGraph {
    funcs: HashMap<OpRef, CallableInfo>,
    targets: HashMap<OpRef, OpRef>,
}

/// Every source callable by its defining operation, and the callable each
/// callee or `func_ref` in `module` resolves to.
fn collect_callable_graph(ctx: &IrContext, module: Module, symbols: &SymbolTable) -> CallableGraph {
    let funcs: HashMap<OpRef, CallableInfo> = symbols
        .symbols()
        .iter()
        .copied()
        .filter(|&op| tribute_control::Func::matches(ctx, op))
        .map(|op| {
            let logical_type = ctx
                .op_attrs(op)
                .get_type("type")
                .expect("pre-CPS validation checked function type");
            let callable = tribute_control::FuncSig::from_type_ref(ctx, logical_type)
                .expect("pre-CPS validation checked callable type");
            let convention = tribute_control::func_sig_convention(ctx, logical_type)
                .expect("pre-CPS validation checked callable convention");
            (
                op,
                CallableInfo {
                    convention,
                    source_result: callable.result(ctx),
                    source_params: callable.inputs(ctx).to_vec(),
                    platform: ctx.op_attrs(op).contains_key("abi"),
                },
            )
        })
        .collect();
    let mut targets = HashMap::default();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        let attributes = &ctx.op(op).attributes;
        if let Some(target) = attributes
            .view(ctx)
            .get_symbol_ref("callee")
            .or_else(|| attributes.view(ctx).get_symbol_ref("func_ref"))
            .and_then(|reference| symbols.resolve(ctx, op, reference))
            .filter(|target| funcs.contains_key(target))
        {
            targets.insert(op, target);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    CallableGraph { funcs, targets }
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
    compiler_intrinsics: &[tribute_control::CompilerIntrinsicDeclaration],
    analyses: &mut AnalysisCache,
) -> Result<(), TributeControlToCpsError> {
    verify_tribute_control_pre_cps(ctx, module, compiler_intrinsics, analyses)?;
    let graph = collect_callable_graph(
        ctx,
        module,
        &analyses.require::<SymbolTable>(ctx, module.op()),
    );
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
        let mut converter = Converter::new(ctx, new_block, graph);
        let mut mapping = HashMap::default();
        let source_ops = converter.ctx.block(source_blocks[0]).ops.clone();
        for source in source_ops {
            if tribute_control::Func::matches(converter.ctx, source) {
                let function = converter.convert_func(source)?;
                converter.ctx.push_op(new_block, function);
            } else if converter.ctx.op(source).name.dialect(converter.ctx)
                == tribute_control::DIALECT_NAME
            {
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
    }
    let new_region = ctx.create_region(RegionData {
        location: ctx.region(source_region).location,
        blocks: trunk_ir::smallvec::smallvec![new_block],
        parent_op: None,
    });
    let temp_module = core::Module::operands()
        .sym_name("__tribute_control_to_cps_candidate")
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

/// Pass-manager wrapper carrying the registered compiler intrinsics.
#[derive(Default)]
pub struct TributeControlToCps {
    compiler_intrinsics: Vec<tribute_control::CompilerIntrinsicDeclaration>,
}

impl TributeControlToCps {
    pub fn new() -> Self {
        Self::default()
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
        tribute_control_to_cps(ctx, target.into(), &self.compiler_intrinsics, analyses)
            .map_err(|error| Box::new(error) as _)
    }
}

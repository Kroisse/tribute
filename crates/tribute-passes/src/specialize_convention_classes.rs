//! Class specialization: the first pass of the CPS strategy
//! (`new-plans/cps-effects.md`, "Class 특수화 pass").
//!
//! The frontend's source-logical IR has rows but no calling conventions. This
//! pass selects the class variables of every definition, collects the class
//! instances root `main` and the references' `row_args` need, clones each
//! definition per instance, points every reference at its instance, and writes
//! a convention on every source signature.
//!
//! Analysis finishes on the unchanged module before anything changes, so every
//! instance is cloned from its original definition. A definition that ends up
//! unreferenced stays in place for the removal of unreferenced definitions that
//! follows the pass.

use std::collections::VecDeque;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::dialect::tribute_control::{self, CallingConvention, FuncSig, Row};
use trunk_ir::context::IrContext;
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;
use trunk_ir::{Symbol, SymbolPath};

use crate::convention_classes::{
    Function, class_suffix, requirements, row_class, settle_class_variables,
};

/// Why a module cannot be specialized.
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub enum ClassSpecializationError {
    /// Some or all source signatures already have a convention.
    #[display("class specialization takes source signatures without calling conventions")]
    ConventionsAssigned,
    /// A let-bound lambda quantifies rows, which this pass does not copy yet.
    #[display("class specialization does not copy let-bound lambdas yet (lambda at {location:?})")]
    QuantifyingLambda {
        #[error(not(source))]
        location: trunk_ir::Location,
    },
}

/// Specializes convention classes over a module without conventions.
pub struct SpecializeConventionClasses;

impl trunk_ir::pass::Pass for SpecializeConventionClasses {
    type Target = trunk_ir::dialect::core::Module;

    fn name(&self) -> &'static str {
        "specialize-convention-classes"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: trunk_ir::dialect::core::Module,
        _: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        specialize(ctx, target.into()).map_err(|error| Box::new(error) as _)
    }
}

/// Specialize the convention classes of `module`.
pub fn specialize(ctx: &mut IrContext, module: Module) -> Result<(), ClassSpecializationError> {
    if tribute_control::conventions_assigned(ctx, module) != Some(false)
        && has_source_signatures(ctx, module)
    {
        return Err(ClassSpecializationError::ConventionsAssigned);
    }
    let requirements = requirements(ctx, module);
    let mut functions: Vec<Function> = module
        .ops(ctx)
        .iter()
        .filter_map(|&op| Function::read(ctx, op))
        .collect();
    reject_quantifying_lambdas(ctx, &functions)?;
    let positions = settle_class_variables(&mut functions);
    let instances = collect_instances(&functions, &positions, &requirements);

    // Clone every instance from its original before rewriting any of them.
    let mut rewrites = Vec::new();
    for function in &functions {
        let vectors = &instances[&function.name];
        let mut before = next_op(ctx, function.op);
        for classes in &vectors[1..] {
            let clone = ctx.clone_op(function.op, &mut IrMapping::new());
            let name = format!("{}{}", function.name, class_suffix(classes));
            let name = ctx.intern_str(&name);
            ctx.op_attrs_mut(clone)
                .insert("sym_name", Attribute::String(name));
            insert_after(ctx, module, function.op, clone, before);
            before = next_op(ctx, clone);
            rewrites.push((clone, function, classes.clone()));
        }
        rewrites.push((function.op, function, vectors[0].clone()));
    }
    let names: HashSet<&str> = functions.iter().map(|f| f.name.as_str()).collect();
    for (op, function, classes) in rewrites {
        let params: HashMap<u32, CallingConvention> = positions
            .get(&function.base)
            .into_iter()
            .flatten()
            .copied()
            .zip(classes)
            .collect();
        let instance = Instance {
            function,
            requirements: &requirements,
            params,
            positions: &positions,
            names: &names,
        };
        instance.rewrite(ctx, op);
    }
    assign_module_level(ctx, module, &requirements);
    Ok(())
}

/// Whether `module` reaches a source signature at all. A module without
/// definitions has no conventions to assign, so it is already specialized.
fn has_source_signatures(ctx: &IrContext, module: Module) -> bool {
    module
        .ops(ctx)
        .iter()
        .any(|&op| tribute_control::Func::matches(ctx, op))
}

fn reject_quantifying_lambdas(
    ctx: &IrContext,
    functions: &[Function],
) -> Result<(), ClassSpecializationError> {
    for function in functions {
        let Some(body) = ctx.op_region(function.op, 0) else {
            continue;
        };
        let mut found = None;
        let _ = trunk_ir::walk::walk_region::<()>(ctx, body, &mut |op| {
            if found.is_none()
                && tribute_control::Lambda::matches(ctx, op)
                && tribute_control::quantified_rows(ctx, op).is_ok_and(|rows| !rows.is_empty())
            {
                found = Some(ctx.op(op).location);
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        if let Some(location) = found {
            return Err(ClassSpecializationError::QuantifyingLambda { location });
        }
    }
    Ok(())
}

/// The class vectors each definition is instantiated at, its primary instance
/// first: every class `Cps`, or every class `Direct` for root `main`, which
/// has no other instance.
fn collect_instances(
    functions: &[Function],
    positions: &HashMap<String, Vec<u32>>,
    requirements: &HashMap<SymbolPath, CallingConvention>,
) -> HashMap<String, Vec<Vec<CallingConvention>>> {
    let by_name: HashMap<&str, &Function> =
        functions.iter().map(|f| (f.name.as_str(), f)).collect();
    let variables = |name: &str| positions.get(name).map_or(&[][..], Vec::as_slice);
    let mut instances: HashMap<String, Vec<Vec<CallingConvention>>> = HashMap::default();
    let mut queue = VecDeque::new();
    for function in functions {
        let primary = if function.is_root_main() {
            CallingConvention::Direct
        } else {
            CallingConvention::Cps
        };
        let classes = vec![primary; variables(&function.name).len()];
        instances.insert(function.name.clone(), vec![classes.clone()]);
        queue.push_back((function.name.as_str(), classes));
    }
    while let Some((name, classes)) = queue.pop_front() {
        let function = by_name[name];
        let params: HashMap<u32, CallingConvention> =
            variables(name).iter().copied().zip(classes).collect();
        for reference in &function.references {
            let Some(callee) = by_name.get(reference.callee.as_str()) else {
                continue;
            };
            if callee.is_root_main() {
                continue;
            }
            let selected: Vec<_> = variables(&callee.name)
                .iter()
                .map(|&position| {
                    usize::try_from(position)
                        .ok()
                        .and_then(|position| reference.rows.get(position))
                        .map_or(CallingConvention::Cps, |row| {
                            row_class(
                                row,
                                &function.binders,
                                requirements,
                                &params,
                                &HashMap::default(),
                            )
                        })
                })
                .collect();
            let known = instances.entry(callee.name.clone()).or_default();
            if !known.contains(&selected) {
                known.push(selected.clone());
                queue.push_back((callee.name.as_str(), selected));
            }
        }
    }
    instances
}

/// The op after `op` in the module body, if any.
fn next_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    let block = ctx.op(op).parent_block?;
    let ops = &ctx.block(block).ops;
    let at = ops.iter().position(|&candidate| candidate == op)?;
    ops.get(at + 1).copied()
}

fn insert_after(
    ctx: &mut IrContext,
    module: Module,
    original: OpRef,
    clone: OpRef,
    before: Option<OpRef>,
) {
    let block = ctx
        .op(original)
        .parent_block
        .or_else(|| module.first_block(ctx))
        .expect("a definition lives in the module body");
    match before {
        Some(before) => ctx.insert_op_before(block, before, clone),
        None => ctx.push_op(block, clone),
    }
}

/// One instance being rewritten: its definition and the class of each class
/// variable.
struct Instance<'a> {
    function: &'a Function,
    requirements: &'a HashMap<SymbolPath, CallingConvention>,
    params: HashMap<u32, CallingConvention>,
    positions: &'a HashMap<String, Vec<u32>>,
    names: &'a HashSet<&'a str>,
}

impl Instance<'_> {
    fn class(&self, row: &Row) -> CallingConvention {
        row_class(
            row,
            &self.function.binders,
            self.requirements,
            &self.params,
            &HashMap::default(),
        )
    }

    fn rewrite(&self, ctx: &mut IrContext, op: OpRef) {
        let mut ops = vec![op];
        let mut blocks = Vec::new();
        if let Some(body) = ctx.op_region(op, 0) {
            let _ = trunk_ir::walk::walk_region::<()>(ctx, body, &mut |inner| {
                ops.push(inner);
                std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
            });
            let mut regions = vec![body];
            while let Some(region) = regions.pop() {
                for &block in &ctx.region(region).blocks {
                    blocks.push(block);
                    for &inner in &ctx.block(block).ops {
                        regions.extend(ctx.op_regions(inner));
                    }
                }
            }
        }
        for &inner in &ops[1..] {
            self.point_reference(ctx, inner);
        }
        for &inner in &ops {
            self.assign_op(ctx, inner);
        }
        for block in blocks {
            for index in 0..ctx.block_args(block).len() {
                let value = ctx.block_args(block)[index];
                let ty = self.assign(ctx, ctx.value_ty(value));
                ctx.set_block_arg_type(block, index as u32, ty);
            }
        }
        self.assign_own_convention(ctx, op);
    }

    /// Point a reference at the instance its rows select.
    fn point_reference(&self, ctx: &mut IrContext, op: OpRef) {
        let key = if tribute_control::FuncRef::matches(ctx, op) {
            "func_ref"
        } else if tribute_control::Call::matches(ctx, op)
            || tribute_control::TailCall::matches(ctx, op)
        {
            "callee"
        } else {
            return;
        };
        let callee = ctx
            .op_attrs(op)
            .get_symbol_ref(key)
            .and_then(|path| path.leaf())
            .expect("a reference names a definition")
            .to_string();
        if !self.names.contains(callee.as_str()) || callee == "main" {
            return;
        }
        let rows = tribute_control::row_args(ctx, op).unwrap_or_default();
        let classes: Vec<_> = self
            .positions
            .get(&callee)
            .into_iter()
            .flatten()
            .map(|&position| {
                usize::try_from(position)
                    .ok()
                    .and_then(|position| rows.get(position))
                    .map_or(CallingConvention::Cps, |row| self.class(row))
            })
            .collect();
        let suffix = class_suffix(&classes);
        if suffix.is_empty() {
            return;
        }
        let name = SymbolPath::new([Symbol::new(&format!("{callee}{suffix}"))]);
        ctx.op_attrs_mut(op).insert(key, Attribute::SymbolRef(name));
    }

    fn assign(&self, ctx: &mut IrContext, ty: TypeRef) -> TypeRef {
        tribute_control::assign_conventions(ctx, ty, |row| self.class(row))
    }

    /// Assign conventions to an operation's result types and the types its
    /// attributes hold.
    fn assign_op(&self, ctx: &mut IrContext, op: OpRef) {
        for index in 0..ctx.op_result_types(op).len() {
            let ty = self.assign(ctx, ctx.op_result_types(op)[index]);
            ctx.set_op_result_type(op, index as u32, ty);
        }
        let attributes: Vec<_> = ctx
            .op(op)
            .attributes
            .iter()
            .map(|(key, value)| (key, value.clone()))
            .collect();
        for (key, value) in attributes {
            let assigned = value.map_types(|ty| self.assign(ctx, ty));
            if assigned != value {
                ctx.op_attrs_mut(op).insert(key, assigned);
            }
        }
    }

    /// A definition with an inferred row reads its own tail only when that
    /// tail is a class variable.
    fn assign_own_convention(&self, ctx: &mut IrContext, op: OpRef) {
        if !self.function.inferred {
            return;
        }
        let ty = ctx
            .op_attrs(op)
            .get_type("type")
            .expect("a definition has a signature");
        let signature =
            FuncSig::from_type_ref(ctx, ty).expect("a definition's signature is a func_sig");
        let mut row = signature.row(ctx);
        if row.tail.is_none_or(|tail| self.params.contains_key(&tail)) {
            return;
        }
        row.tail = None;
        let convention = self.class(&row);
        let ty = tribute_control::func_sig_with_convention(ctx, ty, convention);
        ctx.op_attrs_mut(op).insert("type", Attribute::Type(ty));
    }
}

/// Assign conventions to the types of module-level operations outside
/// definitions, such as the signatures of ability operations. They have no
/// row binders, so only their abilities decide.
fn assign_module_level(
    ctx: &mut IrContext,
    module: Module,
    requirements: &HashMap<SymbolPath, CallingConvention>,
) {
    let class = |row: &Row| {
        row_class(
            row,
            &[],
            requirements,
            &HashMap::default(),
            &HashMap::default(),
        )
    };
    let mut ops = Vec::new();
    for &op in module.ops(ctx).iter() {
        if tribute_control::Func::matches(ctx, op) {
            continue;
        }
        ops.push(op);
        let _ = trunk_ir::walk::walk_op::<()>(ctx, op, &mut |inner| {
            if inner != op {
                ops.push(inner);
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
    }
    for op in ops {
        let attributes: Vec<_> = ctx
            .op(op)
            .attributes
            .iter()
            .map(|(key, value)| (key, value.clone()))
            .collect();
        for (key, value) in attributes {
            let assigned =
                value.map_types(|ty| tribute_control::assign_conventions(ctx, ty, class));
            if assigned != value {
                ctx.op_attrs_mut(op).insert(key, assigned);
            }
        }
    }
}

#[cfg(test)]
mod tests;

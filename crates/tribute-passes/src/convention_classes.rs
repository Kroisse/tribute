//! Convention classes as source-logical IR determines them: the class
//! variables of each definition, the class a row requires inside an
//! instance, and the instance names that record the classes.
//!
//! The rules are those of `new-plans/cps-effects.md` ("Row 변수의 convention
//! class"), read from the IR alone: row binders, `row_args` of references and
//! instantiations, and the operations each ability instance declares.

use std::ops::ControlFlow;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::calling_convention::CallingConvention;
use tribute_ir::dialect::tribute_control::{FuncSig, Row, RowBinder};
use tribute_ir::dialect::{ability, tribute_control, tribute_cps};
use trunk_ir::context::IrContext;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;
use trunk_ir::walk::{WalkAction, walk_region};
use trunk_ir::{Symbol, SymbolPath};

/// The suffix that names a class instance: `$9` and one letter per class
/// variable. Type arguments never start with a digit, so it is unambiguous.
pub(crate) const CLASS_MARKER: &str = "$9";

/// The convention each declared ability instance requires: `Cps` when it
/// declares an `op` operation, `EvidenceDirect` otherwise.
pub(crate) fn requirements(
    ctx: &IrContext,
    module: Module,
) -> HashMap<SymbolPath, CallingConvention> {
    let mut requirements = HashMap::default();
    for &op in module.ops(ctx).iter() {
        let Ok(decl) = ability::Decl::from_op(ctx, op) else {
            continue;
        };
        let mut convention = CallingConvention::EvidenceDirect;
        let _ = walk_region::<()>(ctx, decl.operations(ctx), &mut |operation| {
            if ability::Operation::from_op(ctx, operation)
                .is_ok_and(|operation| operation.kind(ctx) == "op")
            {
                convention = CallingConvention::Cps;
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        requirements.insert(
            SymbolPath::new([Symbol::new(decl.sym_name(ctx))]),
            convention,
        );
    }
    requirements
}

/// The base definition a function instance belongs to, and the classes its
/// name claims for the class variables.
pub(crate) fn split_name(name: &str) -> (&str, Option<Vec<CallingConvention>>) {
    let Some((base, letters)) = name.rsplit_once(CLASS_MARKER) else {
        return (name, None);
    };
    let classes: Option<Vec<_>> = letters
        .chars()
        .map(|letter| match letter {
            'D' => Some(CallingConvention::Direct),
            'E' => Some(CallingConvention::EvidenceDirect),
            'C' => Some(CallingConvention::Cps),
            _ => None,
        })
        .collect();
    match classes {
        Some(classes) if !classes.is_empty() => (base, Some(classes)),
        _ => (name, None),
    }
}

pub(crate) fn class_suffix(classes: &[CallingConvention]) -> String {
    if classes.iter().all(|class| *class == CallingConvention::Cps) {
        return String::new();
    }
    let letters: String = classes
        .iter()
        .map(|class| match class {
            CallingConvention::Direct => 'D',
            CallingConvention::EvidenceDirect => 'E',
            CallingConvention::Cps => 'C',
        })
        .collect();
    format!("{CLASS_MARKER}{letters}")
}

/// A reference to a named definition and the rows it passes.
pub(crate) struct Reference {
    pub(crate) op: OpRef,
    pub(crate) callee: String,
    pub(crate) rows: Vec<Row>,
}

/// What one function definition tells about its class variables.
pub(crate) struct Function {
    pub(crate) op: OpRef,
    pub(crate) name: String,
    pub(crate) base: String,
    pub(crate) claimed: Option<Vec<CallingConvention>>,
    pub(crate) binders: Vec<RowBinder>,
    pub(crate) ty: TypeRef,
    pub(crate) inferred: bool,
    /// Whether the root module names this function as its entry point.
    pub(crate) root_entry: bool,
    /// Tails that select a class variable without looking at callees.
    pub(crate) selecting: HashSet<u32>,
    pub(crate) references: Vec<Reference>,
    /// The selected class variables, by binder index.
    pub(crate) selected: Vec<u32>,
}

impl Function {
    pub(crate) fn read(ctx: &IrContext, op: OpRef) -> Option<Self> {
        if !tribute_control::is_func(ctx, op) {
            return None;
        }
        let name = ctx.op_attrs(op).get_str("sym_name")?.to_owned();
        let (base, claimed) = split_name(&name);
        let base = base.to_owned();
        let ty = ctx.op_attrs(op).get_type("type")?;
        let signature = FuncSig::from_type_ref(ctx, ty)?;
        let binders = tribute_control::row_binders(ctx, op).ok()?;
        let inferred = matches!(
            ctx.op_attrs(op).get(tribute_control::INFERRED_ROW_ATTR),
            Some(Attribute::Bool(true))
        );
        let root_entry = tribute_ir::metadata::is_root_entry(ctx, op);
        let mut selecting = HashSet::default();
        let mut tail_of = |ty: TypeRef| {
            tribute_control::visit_rows(ctx, ty, &mut |row| {
                if let Ok(Row {
                    tail: Some(tail), ..
                }) = row
                {
                    selecting.insert(tail);
                }
            });
        };
        let mut references = Vec::new();
        // A definition without a body is one external symbol with one ABI,
        // so it has no class variables.
        if let Some(body) = ctx.op_region(op, 0) {
            for &ty in signature.inputs(ctx).iter().chain(signature.results(ctx)) {
                tail_of(ty);
            }
            for ty in body_value_types(ctx, body) {
                tail_of(ty);
            }
            let _ = walk_region::<()>(ctx, body, &mut |inner| {
                let rows = tribute_control::row_args(ctx, inner).unwrap_or_default();
                if tribute_control::Instantiate::matches(ctx, inner) {
                    selecting.extend(rows.iter().filter_map(|row| row.tail));
                } else if let Some(callee) = reference_callee(ctx, inner) {
                    references.push(Reference {
                        op: inner,
                        callee,
                        rows,
                    });
                }
                ControlFlow::Continue(WalkAction::Advance)
            });
        }
        Some(Self {
            op,
            name,
            base,
            claimed,
            binders,
            ty,
            inferred,
            root_entry,
            selecting,
            references,
            selected: Vec::new(),
        })
    }

    /// Whether binder `index` may be a class variable.
    pub(crate) fn is_candidate(&self, index: u32) -> bool {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.binders.get(index))
            == Some(&RowBinder::Param)
    }
}

/// The name a `call`, `tail_call`, or `func_ref` refers to, through its
/// table for a `tribute_cps.func_ref`.
pub(crate) fn reference_callee(ctx: &IrContext, op: OpRef) -> Option<String> {
    if tribute_cps::FuncRef::matches(ctx, op) {
        // Every table class specialization makes has its instance as its
        // only target.
        let table = tribute_cps::func_ref_table(ctx, op)?;
        let (_, target) = tribute_cps::table_entries(ctx, table)?.into_iter().next()?;
        return Some(target.leaf()?.to_string());
    }
    let key = if tribute_control::is_func_ref(ctx, op) {
        "func_ref"
    } else if tribute_control::Call::matches(ctx, op) || tribute_control::TailCall::matches(ctx, op)
    {
        "callee"
    } else {
        return None;
    };
    let path = ctx.op_attrs(op).get_symbol_ref(key)?;
    Some(path.leaf()?.to_string())
}

/// The types of every value a body defines: operation results and block
/// arguments, nested regions included.
pub(crate) fn body_value_types(ctx: &IrContext, body: RegionRef) -> Vec<TypeRef> {
    let mut types = Vec::new();
    let mut regions = vec![body];
    while let Some(region) = regions.pop() {
        for &block in &ctx.region(region).blocks {
            types.extend(
                ctx.block_args(block)
                    .iter()
                    .map(|&value| ctx.value_ty(value)),
            );
            for &op in &ctx.block(block).ops {
                types.extend(ctx.op_result_types(op).iter().copied());
                regions.extend(ctx.op_regions(op));
            }
        }
    }
    types
}

/// Select the class variables of every definition: a `param` binder that is
/// the tail of a callable type in its signature or body, of a row it
/// instantiates a let-bound lambda with, or of a row it passes for a class
/// variable of a definition it references. Returns the class variable
/// positions of each base definition.
pub(crate) fn settle_class_variables(functions: &mut [Function]) -> HashMap<String, Vec<u32>> {
    for function in functions.iter_mut() {
        let mut selected: Vec<u32> = function
            .selecting
            .iter()
            .copied()
            .filter(|tail| function.is_candidate(*tail))
            .collect();
        selected.sort_unstable();
        function.selected = selected;
    }
    loop {
        let positions = class_positions(functions);
        let mut changed = false;
        for function in functions.iter_mut() {
            let mut added = Vec::new();
            for reference in &function.references {
                let (callee, _) = split_name(&reference.callee);
                for &position in positions.get(callee).into_iter().flatten() {
                    let tail = usize::try_from(position)
                        .ok()
                        .and_then(|position| reference.rows.get(position))
                        .and_then(|row| row.tail);
                    if let Some(tail) = tail
                        && function.is_candidate(tail)
                        && !function.selected.contains(&tail)
                    {
                        added.push(tail);
                    }
                }
            }
            if !added.is_empty() {
                function.selected.extend(added);
                function.selected.sort_unstable();
                function.selected.dedup();
                changed = true;
            }
        }
        if !changed {
            return class_positions(functions);
        }
    }
}

/// The class variable positions of each base definition, over all its
/// instances.
pub(crate) fn class_positions(functions: &[Function]) -> HashMap<String, Vec<u32>> {
    let mut positions: HashMap<String, Vec<u32>> = HashMap::default();
    for function in functions {
        let entry = positions.entry(function.base.clone()).or_default();
        entry.extend(&function.selected);
        entry.sort_unstable();
        entry.dedup();
    }
    positions
}

/// The class `row` requires inside an instance whose class variables have
/// `params` and whose local binders have the classes `local` reads. A local
/// binder not yet settled reads as `Direct`, the start of their least fixed
/// point.
pub(crate) fn row_class(
    row: &Row,
    binders: &[RowBinder],
    requirements: &HashMap<SymbolPath, CallingConvention>,
    params: &HashMap<u32, CallingConvention>,
    local: impl Fn(u32) -> Option<CallingConvention>,
) -> CallingConvention {
    let instances = row
        .instances
        .iter()
        .fold(CallingConvention::Direct, |class, instance| {
            class.max(
                requirements
                    .get(instance)
                    .copied()
                    .unwrap_or(CallingConvention::Cps),
            )
        });
    let Some(tail) = row.tail else {
        return instances;
    };
    let kind = usize::try_from(tail)
        .ok()
        .and_then(|index| binders.get(index));
    let tail = match kind {
        Some(RowBinder::Param) => params.get(&tail).copied().unwrap_or(CallingConvention::Cps),
        Some(RowBinder::Local) => local(tail).unwrap_or(CallingConvention::Direct),
        _ => CallingConvention::Cps,
    };
    instances.max(tail)
}

/// The lambda `op` instantiates, the binders it quantifies, and the rows `op`
/// passes for them.
pub(crate) fn instantiation(ctx: &IrContext, op: OpRef) -> Option<(OpRef, Vec<u32>, Vec<Row>)> {
    if !tribute_control::Instantiate::matches(ctx, op) {
        return None;
    }
    let operand = *ctx.op_operands(op).first()?;
    let trunk_ir::refs::ValueDef::OpResult(lambda, _) = ctx.value_def(operand) else {
        return None;
    };
    let binders = tribute_control::quantified_rows(ctx, lambda).ok()?;
    let rows = tribute_control::row_args(ctx, op).ok()?;
    Some((lambda, binders, rows))
}

/// The classes of the local binders of an instance's body. A binder has a
/// class in each body of the lambda that quantifies it: the join of the
/// classes its instantiations that select that body pass for it
/// (`new-plans/ir.md`, "Convention entry").
#[derive(Default, PartialEq, Eq)]
pub(crate) struct Locals {
    classes: HashMap<(u32, usize), CallingConvention>,
    owners: HashMap<u32, OpRef>,
}

impl Locals {
    /// The class of local `binder` in the body of its lambda that holds
    /// `at`, or in the lambda's first body when `at` is outside them.
    pub(crate) fn get(&self, ctx: &IrContext, binder: u32, at: OpRef) -> Option<CallingConvention> {
        let body = self
            .owners
            .get(&binder)
            .map_or(0, |&owner| body_index(ctx, owner, at));
        self.classes.get(&(binder, body)).copied()
    }
}

/// The index of the body of `owner` that holds `at`, or 0 when none does.
fn body_index(ctx: &IrContext, owner: OpRef, at: OpRef) -> usize {
    let mut current = at;
    while let Some(region) = ctx
        .op(current)
        .parent_block
        .and_then(|block| ctx.block(block).parent_region)
    {
        let Some(parent) = ctx.region(region).parent_op else {
            break;
        };
        if parent == owner {
            return ctx
                .op_regions(owner)
                .position(|body| body == region)
                .unwrap_or(0);
        }
        current = parent;
    }
    0
}

/// The class of each local binder of `body` in each body of its lambda.
///
/// An instantiation selects the body of the class of the row it makes. That
/// class can rise as the binders it reads settle and move the instantiation
/// to another body, so each round recomputes the classes from the previous
/// one rather than joining into them.
pub(crate) fn settle_locals(
    ctx: &IrContext,
    body: RegionRef,
    binders: &[RowBinder],
    requirements: &HashMap<SymbolPath, CallingConvention>,
    params: &HashMap<u32, CallingConvention>,
) -> Locals {
    let mut instantiations = Vec::new();
    let mut owners = HashMap::default();
    let _ = walk_region::<()>(ctx, body, &mut |op| {
        for binder in tribute_control::quantified_rows(ctx, op).unwrap_or_default() {
            owners.insert(binder, op);
        }
        if let Some((lambda, quantified, rows)) = instantiation(ctx, op) {
            let made = FuncSig::from_type_ref(ctx, ctx.op_result_types(op)[0])
                .map(|signature| signature.row(ctx));
            let bodies = tribute_cps::body_conventions(ctx, lambda).filter(|c| c.len() > 1);
            instantiations.push((op, quantified, rows, made, bodies));
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    let mut locals = Locals {
        classes: HashMap::default(),
        owners,
    };
    // While no instantiation changes body, a round only raises classes, each
    // binder at most twice in each of at most three bodies. An instantiation
    // that changes body can keep the classes from settling; the last round
    // then stands, and the checks report what it leaves.
    let rounds = 6 * locals.owners.len() + 2;
    for _ in 0..rounds {
        let mut classes: HashMap<(u32, usize), CallingConvention> = HashMap::default();
        for (op, quantified, rows, made, bodies) in &instantiations {
            let class = |row: &Row| {
                row_class(row, binders, requirements, params, |binder| {
                    locals.get(ctx, binder, *op)
                })
            };
            let body = match (made, bodies) {
                (Some(made), Some(bodies)) => {
                    let selected = class(made);
                    bodies
                        .iter()
                        .position(|&convention| convention == selected)
                        .unwrap_or(0)
                }
                _ => 0,
            };
            for (binder, row) in quantified.iter().zip(rows) {
                let class = class(row);
                classes
                    .entry((*binder, body))
                    .and_modify(|known| *known = (*known).max(class))
                    .or_insert(class);
            }
        }
        if classes == locals.classes {
            break;
        }
        locals.classes = classes;
    }
    locals
}

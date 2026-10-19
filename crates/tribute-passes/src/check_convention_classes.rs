//! Check that the convention classes the frontend specialized agree with the
//! classes the rows of source-logical IR imply.
//!
//! The rules are those of `new-plans/cps-effects.md` ("Row 변수의 convention
//! class"), read from the IR alone: row binders, `row_args` of references and
//! instantiations, and the operations each ability instance declares.

use std::ops::ControlFlow;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::dialect::tribute_control::{CallingConvention, FuncSig, Row, RowBinder};
use tribute_ir::dialect::{ability, tribute_control};
use trunk_ir::context::IrContext;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;
use trunk_ir::walk::{WalkAction, walk_region};
use trunk_ir::{Symbol, SymbolPath};

/// A place where the IR's rows imply another class than the frontend chose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Disagreement {
    pub op: OpRef,
    /// The rule that disagrees, such as `suffix` or `reference`.
    pub rule: &'static str,
    pub message: String,
}

/// The suffix that names a class instance: `$9` and one letter per class
/// variable. Type arguments never start with a digit, so it is unambiguous.
const CLASS_MARKER: &str = "$9";

/// Every disagreement in `module`.
pub fn check(ctx: &IrContext, module: Module) -> Vec<Disagreement> {
    let requirements = requirements(ctx, module);
    let mut functions: Vec<Function> = module
        .ops(ctx)
        .iter()
        .filter_map(|&op| Function::read(ctx, op))
        .collect();
    let positions = settle_class_variables(&mut functions);
    let mut disagreements = Vec::new();
    for function in &functions {
        Instance::new(ctx, function, &positions, &requirements, &mut disagreements)
            .check(&mut disagreements);
    }
    disagreements
}

/// The convention each declared ability instance requires: `Cps` when it
/// declares an `op` operation, `EvidenceDirect` otherwise.
fn requirements(ctx: &IrContext, module: Module) -> HashMap<SymbolPath, CallingConvention> {
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
fn split_name(name: &str) -> (&str, Option<Vec<CallingConvention>>) {
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

fn class_suffix(classes: &[CallingConvention]) -> String {
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
struct Reference {
    op: OpRef,
    callee: String,
    rows: Vec<Row>,
}

/// What one function definition tells about its class variables.
struct Function {
    op: OpRef,
    name: String,
    base: String,
    claimed: Option<Vec<CallingConvention>>,
    binders: Vec<RowBinder>,
    ty: TypeRef,
    inferred: bool,
    /// Tails that select a class variable without looking at callees.
    selecting: HashSet<u32>,
    references: Vec<Reference>,
    /// The selected class variables, by binder index.
    selected: Vec<u32>,
}

impl Function {
    fn read(ctx: &IrContext, op: OpRef) -> Option<Self> {
        let function = tribute_control::Func::from_op(ctx, op).ok()?;
        let name = function.sym_name(ctx).to_owned();
        let (base, claimed) = split_name(&name);
        let base = base.to_owned();
        let ty = function.r#type(ctx);
        let signature = FuncSig::from_type_ref(ctx, ty)?;
        let binders = tribute_control::row_binders(ctx, op).ok()?;
        let inferred = matches!(
            ctx.op_attrs(op).get(tribute_control::INFERRED_ROW_ATTR),
            Some(Attribute::Bool(true))
        );
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
        for &ty in signature.inputs(ctx).iter().chain(signature.results(ctx)) {
            tail_of(ty);
        }
        let mut references = Vec::new();
        if let Some(body) = ctx.op_region(op, 0) {
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
            selecting,
            references,
            selected: Vec::new(),
        })
    }

    /// Whether binder `index` may be a class variable.
    fn is_candidate(&self, index: u32) -> bool {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.binders.get(index))
            == Some(&RowBinder::Param)
    }

    fn is_root_main(&self) -> bool {
        self.name == "main"
    }
}

/// The name a `call`, `tail_call`, or `func_ref` refers to.
fn reference_callee(ctx: &IrContext, op: OpRef) -> Option<String> {
    let key = if tribute_control::FuncRef::matches(ctx, op) {
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
fn body_value_types(ctx: &IrContext, body: RegionRef) -> Vec<TypeRef> {
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
fn settle_class_variables(functions: &mut [Function]) -> HashMap<String, Vec<u32>> {
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
fn class_positions(functions: &[Function]) -> HashMap<String, Vec<u32>> {
    let mut positions: HashMap<String, Vec<u32>> = HashMap::default();
    for function in functions {
        let entry = positions.entry(function.base.clone()).or_default();
        entry.extend(&function.selected);
        entry.sort_unstable();
        entry.dedup();
    }
    positions
}

/// One function instance with the classes its name fixes.
struct Instance<'a> {
    ctx: &'a IrContext,
    function: &'a Function,
    positions: &'a HashMap<String, Vec<u32>>,
    requirements: &'a HashMap<SymbolPath, CallingConvention>,
    /// The class of each class variable.
    classes: HashMap<u32, CallingConvention>,
    /// The class of each local binder, and the lambda that quantifies it.
    locals: HashMap<u32, CallingConvention>,
}

impl<'a> Instance<'a> {
    fn new(
        ctx: &'a IrContext,
        function: &'a Function,
        positions: &'a HashMap<String, Vec<u32>>,
        requirements: &'a HashMap<SymbolPath, CallingConvention>,
        disagreements: &mut Vec<Disagreement>,
    ) -> Self {
        let variables = positions.get(&function.base).map_or(&[][..], Vec::as_slice);
        if function.selected != variables {
            disagreements.push(Disagreement {
                op: function.op,
                rule: "siblings",
                message: format!(
                    "{} selects class variables {:?}, other instances of {} select {:?}",
                    function.name, function.selected, function.base, variables
                ),
            });
        }
        let claimed = if function.is_root_main() {
            vec![CallingConvention::Direct; variables.len()]
        } else {
            match &function.claimed {
                Some(classes) => {
                    if classes.len() != variables.len() {
                        disagreements.push(Disagreement {
                            op: function.op,
                            rule: "suffix",
                            message: format!(
                                "{} names {} classes for {} class variables {:?}",
                                function.name,
                                classes.len(),
                                variables.len(),
                                variables
                            ),
                        });
                    }
                    classes.clone()
                }
                None => vec![CallingConvention::Cps; variables.len()],
            }
        };
        let classes = variables.iter().copied().zip(claimed).collect();
        let mut instance = Self {
            ctx,
            function,
            positions,
            requirements,
            classes,
            locals: HashMap::default(),
        };
        instance.settle_locals();
        instance
    }

    /// The class of each local binder: the join of the classes its
    /// instantiations pass for it, which may name other local binders.
    fn settle_locals(&mut self) {
        let Some(body) = self.ctx.op_region(self.function.op, 0) else {
            return;
        };
        let mut instantiations = Vec::new();
        let _ = walk_region::<()>(self.ctx, body, &mut |op| {
            if let Some((_, binders, rows)) = self.instantiation(op) {
                instantiations.push((binders, rows));
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        loop {
            let mut changed = false;
            for (binders, rows) in &instantiations {
                for (binder, row) in binders.iter().zip(rows) {
                    let class = self.class(row);
                    let joined = self
                        .locals
                        .get(binder)
                        .map_or(class, |known| (*known).max(class));
                    if self.locals.insert(*binder, joined) != Some(joined) {
                        changed = true;
                    }
                }
            }
            if !changed {
                return;
            }
        }
    }

    /// The lambda `op` instantiates, its quantified binders, and the rows.
    fn instantiation(&self, op: OpRef) -> Option<(OpRef, Vec<u32>, Vec<Row>)> {
        if !tribute_control::Instantiate::matches(self.ctx, op) {
            return None;
        }
        let operand = *self.ctx.op_operands(op).first()?;
        let trunk_ir::refs::ValueDef::OpResult(lambda, _) = self.ctx.value_def(operand) else {
            return None;
        };
        let binders = tribute_control::quantified_rows(self.ctx, lambda).ok()?;
        let rows = tribute_control::row_args(self.ctx, op).ok()?;
        Some((lambda, binders, rows))
    }

    /// The class a row requires inside this instance.
    fn class(&self, row: &Row) -> CallingConvention {
        let instances = row
            .instances
            .iter()
            .fold(CallingConvention::Direct, |class, instance| {
                class.max(
                    self.requirements
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
            .and_then(|index| self.function.binders.get(index));
        let tail = match kind {
            Some(RowBinder::Param) => self
                .classes
                .get(&tail)
                .copied()
                .unwrap_or(CallingConvention::Cps),
            // Local classes settle as a least fixed point from `Direct`.
            Some(RowBinder::Local) => self
                .locals
                .get(&tail)
                .copied()
                .unwrap_or(CallingConvention::Direct),
            _ => CallingConvention::Cps,
        };
        instances.max(tail)
    }

    fn check(&self, disagreements: &mut Vec<Disagreement>) {
        self.check_own_convention(disagreements);
        self.check_signature(disagreements);
        self.check_references(disagreements);
        self.check_body(disagreements);
    }

    /// A definition's own convention reads its row tail only when that tail
    /// is a class variable, if the row was inferred.
    fn check_own_convention(&self, disagreements: &mut Vec<Disagreement>) {
        let Some(signature) = FuncSig::from_type_ref(self.ctx, self.function.ty) else {
            return;
        };
        let mut row = signature.row(self.ctx);
        if self.function.inferred
            && row
                .tail
                .is_some_and(|tail| !self.classes.contains_key(&tail))
        {
            row.tail = None;
        }
        let expected = self.class(&row);
        let actual = signature.convention(self.ctx);
        if expected != actual {
            disagreements.push(Disagreement {
                op: self.function.op,
                rule: "own",
                message: format!(
                    "{} has convention {actual:?}, its row implies {expected:?}",
                    self.function.name
                ),
            });
        }
    }

    /// Callable types inside the signature have exactly their rows' class.
    fn check_signature(&self, disagreements: &mut Vec<Disagreement>) {
        let Some(signature) = FuncSig::from_type_ref(self.ctx, self.function.ty) else {
            return;
        };
        for &ty in signature
            .inputs(self.ctx)
            .iter()
            .chain(signature.results(self.ctx))
        {
            for callable in callable_types(self.ctx, ty) {
                self.compare(
                    self.function.op,
                    "signature",
                    callable,
                    |actual, expected| actual == expected,
                    disagreements,
                );
            }
        }
    }

    /// A reference names the instance of its callee for the classes of the
    /// rows it passes for the callee's class variables.
    fn check_references(&self, disagreements: &mut Vec<Disagreement>) {
        for reference in &self.function.references {
            let (callee, _) = split_name(&reference.callee);
            let classes: Vec<_> = self
                .positions
                .get(callee)
                .into_iter()
                .flatten()
                .map(|&position| {
                    usize::try_from(position)
                        .ok()
                        .and_then(|position| reference.rows.get(position))
                        .map_or(CallingConvention::Cps, |row| self.class(row))
                })
                .collect();
            let expected = format!("{callee}{}", class_suffix(&classes));
            if expected != reference.callee {
                disagreements.push(Disagreement {
                    op: reference.op,
                    rule: "reference",
                    message: format!(
                        "{} refers to {}, its rows select {expected}",
                        self.function.name, reference.callee
                    ),
                });
            }
        }
    }

    /// Lambdas have exactly their rows' class; every instantiation of one
    /// selects the same classes; other callable values are at least as
    /// strong as their rows.
    fn check_body(&self, disagreements: &mut Vec<Disagreement>) {
        let Some(body) = self.ctx.op_region(self.function.op, 0) else {
            return;
        };
        let mut selections: HashMap<OpRef, Vec<CallingConvention>> = HashMap::default();
        let _ = walk_region::<()>(self.ctx, body, &mut |op| {
            if tribute_control::Lambda::matches(self.ctx, op) {
                for &ty in self.ctx.op_result_types(op) {
                    self.compare(
                        op,
                        "lambda",
                        ty,
                        |actual, expected| actual == expected,
                        disagreements,
                    );
                }
            } else if let Some((lambda, _, rows)) = self.instantiation(op) {
                let classes: Vec<_> = rows.iter().map(|row| self.class(row)).collect();
                match selections.get(&lambda) {
                    Some(first) if *first != classes => disagreements.push(Disagreement {
                        op,
                        rule: "instantiate",
                        message: format!(
                            "{} instantiates one lambda at classes {first:?} and {classes:?}",
                            self.function.name
                        ),
                    }),
                    Some(_) => {}
                    None => {
                        selections.insert(lambda, classes);
                    }
                }
            } else if !tribute_control::FuncRef::matches(self.ctx, op) {
                // A `func_ref` keeps its target's convention under a wider
                // row, which reference validation checks.
                for &ty in self.ctx.op_result_types(op) {
                    for callable in callable_types(self.ctx, ty) {
                        self.compare(
                            op,
                            "value",
                            callable,
                            |actual, expected| actual >= expected,
                            disagreements,
                        );
                    }
                }
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }

    fn compare(
        &self,
        op: OpRef,
        rule: &'static str,
        callable: TypeRef,
        agrees: impl Fn(CallingConvention, CallingConvention) -> bool,
        disagreements: &mut Vec<Disagreement>,
    ) {
        let Some(signature) = FuncSig::from_type_ref(self.ctx, callable) else {
            return;
        };
        let expected = self.class(&signature.row(self.ctx));
        let actual = signature.convention(self.ctx);
        if !agrees(actual, expected) {
            disagreements.push(Disagreement {
                op,
                rule,
                message: format!(
                    "a callable in {} has convention {actual:?}, its row implies {expected:?}",
                    self.function.name
                ),
            });
        }
    }
}

/// Every source callable type in `ty`, itself included, through callable
/// parameters and results and tuple layout elements.
fn callable_types(ctx: &IrContext, ty: TypeRef) -> Vec<TypeRef> {
    let mut found = Vec::new();
    let mut seen = HashSet::default();
    let mut pending = vec![ty];
    while let Some(ty) = pending.pop() {
        if !seen.insert(ty) {
            continue;
        }
        if let Some(signature) = FuncSig::from_type_ref(ctx, ty) {
            found.push(ty);
            pending.extend(signature.inputs(ctx));
            pending.extend(signature.results(ctx));
        } else if let Some(elements) = tribute_control::logical_tuple_elements(ctx, ty) {
            pending.extend(elements);
        } else {
            pending.extend(ctx.get_type(ty).params.iter().copied());
        }
    }
    found
}

#[cfg(test)]
mod tests;

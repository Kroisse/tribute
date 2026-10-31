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

/// A convention class as class specialization reads a row: exact, or an
/// upper bound, which says the row may require any class no stronger than
/// it at run time (`new-plans/ir.md`, "Convention entry"). The upper bound
/// `Direct` is the exact `Direct`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Class {
    bound: CallingConvention,
    exact: bool,
}

impl Class {
    pub(crate) const DIRECT: Self = Self::exact(CallingConvention::Direct);
    pub(crate) const CPS: Self = Self::exact(CallingConvention::Cps);

    pub(crate) const fn exact(convention: CallingConvention) -> Self {
        Self {
            bound: convention,
            exact: true,
        }
    }

    pub(crate) fn upto(convention: CallingConvention) -> Self {
        Self {
            bound: convention,
            exact: convention == CallingConvention::Direct,
        }
    }

    /// The class itself when exact, or its upper bound: the convention a
    /// transfer and a definition of this class use.
    pub(crate) fn bound(self) -> CallingConvention {
        self.bound
    }

    /// The class of a row whose parts have the classes `self` and `other`:
    /// an upper bound when the stronger part is one and no exact part
    /// reaches it.
    pub(crate) fn join(self, other: Self) -> Self {
        let bound = self.bound.max(other.bound);
        let exact = [self, other]
            .iter()
            .any(|part| part.exact && part.bound == bound);
        if exact {
            Self::exact(bound)
        } else {
            Self::upto(bound)
        }
    }

    /// The class of a binder that uses pass `self` and `other` for: the one
    /// class they agree on, or else the upper bound of both.
    pub(crate) fn merge(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            Self::upto(self.bound.max(other.bound))
        }
    }

    /// The class read in a body of convention `convention`, which this class
    /// may require no more than: an upper bound above it narrows to it.
    pub(crate) fn narrow(self, convention: CallingConvention) -> Self {
        if self.bound <= convention {
            self
        } else {
            Self::upto(convention)
        }
    }

    /// The conventions a row of this class whose abilities require `lower`
    /// may require at run time, weakest first.
    pub(crate) fn conventions(self, lower: CallingConvention) -> Vec<CallingConvention> {
        if self.exact {
            return vec![self.bound];
        }
        CallingConvention::ALL
            .into_iter()
            .filter(|&convention| lower <= convention && convention <= self.bound)
            .collect()
    }

    /// The letter that names this class in an instance name.
    fn letter(self) -> char {
        match (self.bound, self.exact) {
            (CallingConvention::Direct, _) => 'D',
            (CallingConvention::EvidenceDirect, true) => 'E',
            (CallingConvention::EvidenceDirect, false) => 'e',
            (CallingConvention::Cps, true) => 'C',
            (CallingConvention::Cps, false) => 'c',
        }
    }

    fn from_letter(letter: char) -> Option<Self> {
        Some(match letter {
            'D' => Self::DIRECT,
            'E' => Self::exact(CallingConvention::EvidenceDirect),
            'e' => Self::upto(CallingConvention::EvidenceDirect),
            'C' => Self::CPS,
            'c' => Self::upto(CallingConvention::Cps),
            _ => return None,
        })
    }
}

impl std::fmt::Display for Class {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.exact {
            write!(f, "{}", self.bound)
        } else {
            write!(f, "<={}", self.bound)
        }
    }
}

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
pub(crate) fn split_name(name: &str) -> (&str, Option<Vec<Class>>) {
    let Some((base, letters)) = name.rsplit_once(CLASS_MARKER) else {
        return (name, None);
    };
    let classes: Option<Vec<_>> = letters.chars().map(Class::from_letter).collect();
    match classes {
        Some(classes) if !classes.is_empty() => (base, Some(classes)),
        _ => (name, None),
    }
}

/// The suffix of the instance with `classes`, empty when every class is the
/// exact `Cps`.
pub(crate) fn class_suffix(classes: &[Class]) -> String {
    if classes.iter().all(|&class| class == Class::CPS) {
        return String::new();
    }
    let letters: String = classes.iter().map(|class| class.letter()).collect();
    format!("{CLASS_MARKER}{letters}")
}

/// A reference to a named definition and the rows it passes.
pub(crate) struct Reference {
    pub(crate) op: OpRef,
    /// The instance its class arguments name, which a function reference's
    /// table is named after.
    pub(crate) callee: String,
    /// The instances it enters: its callee, or the targets of its table.
    pub(crate) targets: Vec<String>,
    pub(crate) rows: Vec<Row>,
}

/// What one function definition tells about its class variables.
pub(crate) struct Function {
    pub(crate) op: OpRef,
    pub(crate) name: String,
    pub(crate) base: String,
    pub(crate) claimed: Option<Vec<Class>>,
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
    pub(crate) fn read(
        ctx: &IrContext,
        op: OpRef,
        tables: &HashMap<String, OpRef>,
    ) -> Option<Self> {
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
                        targets: vec![callee.clone()],
                        callee,
                        rows,
                    });
                } else if let Some((callee, targets)) = table_callee(ctx, inner, tables) {
                    references.push(Reference {
                        op: inner,
                        callee,
                        targets,
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

/// The name a `call`, `tail_call`, or `tribute_control.func_ref` refers to.
pub(crate) fn reference_callee(ctx: &IrContext, op: OpRef) -> Option<String> {
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

/// The function tables of `module`, by name.
pub(crate) fn func_tables(ctx: &IrContext, module: Module) -> HashMap<String, OpRef> {
    module
        .ops(ctx)
        .iter()
        .filter(|&&op| tribute_cps::FuncTable::matches(ctx, op))
        .filter_map(|&op| Some((ctx.op_attrs(op).get_str("sym_name")?.to_owned(), op)))
        .collect()
}

/// The suffix of a function table's name after the name of the instance
/// its class arguments select (`new-plans/cps-effects.md`, "Class 특수화
/// pass").
pub(crate) const TABLE_SUFFIX: &str = "$8";

/// The instance a `tribute_cps.func_ref` names through its table in
/// `tables`, and the table's targets.
fn table_callee(
    ctx: &IrContext,
    op: OpRef,
    tables: &HashMap<String, OpRef>,
) -> Option<(String, Vec<String>)> {
    let reference = tribute_cps::FuncRef::from_op(ctx, op).ok()?;
    let name = reference.func_ref(ctx).as_simple()?.as_str();
    let table = *tables.get(name)?;
    let targets = tribute_cps::table_entries(ctx, table)?
        .into_iter()
        .map(|(_, target)| Some(target.leaf()?.to_string()))
        .collect::<Option<Vec<_>>>()?;
    Some((name.strip_suffix(TABLE_SUFFIX)?.to_owned(), targets))
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

/// How one instance reads the classes of rows: the classes of its class
/// variables, and of the local binders of its lambdas in each of their
/// bodies.
pub(crate) struct Reader<'a> {
    pub(crate) binders: Vec<RowBinder>,
    pub(crate) requirements: &'a HashMap<SymbolPath, CallingConvention>,
    pub(crate) params: HashMap<u32, Class>,
    pub(crate) locals: Locals,
}

impl Reader<'_> {
    /// The class the abilities of `row` require, without its tail: the
    /// row's lower bound.
    pub(crate) fn lower(&self, row: &Row) -> CallingConvention {
        row.instances
            .iter()
            .fold(CallingConvention::Direct, |class, instance| {
                class.max(
                    self.requirements
                        .get(instance)
                        .copied()
                        .unwrap_or(CallingConvention::Cps),
                )
            })
    }

    /// The class `row` requires at `at`.
    pub(crate) fn class(&self, ctx: &IrContext, row: &Row, at: OpRef) -> Class {
        let lower = Class::exact(self.lower(row));
        match row.tail {
            Some(tail) => lower.join(self.tail(ctx, tail, at)),
            None => lower,
        }
    }

    /// The conventions `row` may require at `at`, weakest first.
    pub(crate) fn conventions(
        &self,
        ctx: &IrContext,
        row: &Row,
        at: OpRef,
    ) -> Vec<CallingConvention> {
        self.class(ctx, row, at).conventions(self.lower(row))
    }

    /// The class of binder `tail` at `at`. A local binder not yet settled
    /// reads as `Direct`, the start of their least fixed point.
    fn tail(&self, ctx: &IrContext, tail: u32, at: OpRef) -> Class {
        let kind = usize::try_from(tail)
            .ok()
            .and_then(|index| self.binders.get(index));
        let class = match kind {
            Some(RowBinder::Param) => self.params.get(&tail).copied().unwrap_or(Class::CPS),
            Some(RowBinder::Local) => self.locals.get(ctx, tail, at).unwrap_or(Class::DIRECT),
            _ => Class::CPS,
        };
        narrowed(ctx, tail, at, class)
    }

    /// Settle the class of each local binder of `body` in each body of its
    /// lambda: the merge of the classes the instantiations that select that
    /// body pass for it (`new-plans/ir.md`, "Convention entry"). An
    /// instantiation selects every body its made row may require, so an
    /// instantiation whose row is an upper bound reaches the bodies from
    /// that row's lower bound up to the bound.
    ///
    /// That selection can rise as the binders it reads settle, so each
    /// round recomputes the classes from the previous one rather than
    /// merging into them.
    pub(crate) fn settle_locals(&mut self, ctx: &IrContext, body: RegionRef) {
        let mut instantiations = Vec::new();
        let mut owners = HashMap::default();
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            for binder in tribute_control::quantified_rows(ctx, op).unwrap_or_default() {
                owners.insert(binder, op);
            }
            if let Some((_, quantified, rows)) = instantiation(ctx, op)
                && let Some(made) = FuncSig::from_type_ref(ctx, ctx.op_result_types(op)[0])
            {
                instantiations.push((op, quantified, rows, made.row(ctx)));
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        self.locals = Locals {
            classes: HashMap::default(),
            owners,
        };
        // A class only rises while the selections stay, through at most five
        // classes for each binder in each of three bodies. A selection that
        // keeps changing can keep the classes from settling; the last round
        // then stands, and the checks report what it leaves.
        let rounds = 15 * self.locals.owners.len() + 2;
        for _ in 0..rounds {
            let mut classes: HashMap<(u32, CallingConvention), Class> = HashMap::default();
            for (op, quantified, rows, made) in &instantiations {
                let bodies = self.conventions(ctx, made, *op);
                for (binder, row) in quantified.iter().zip(rows) {
                    let class = self.class(ctx, row, *op);
                    for &body in &bodies {
                        classes
                            .entry((*binder, body))
                            .and_modify(|known| *known = known.merge(class))
                            .or_insert(class);
                    }
                }
            }
            if classes == self.locals.classes {
                break;
            }
            self.locals.classes = classes;
        }
    }
}

/// `class`, the class of binder `tail`, as read at `at`: each
/// `tribute_cps.lambda` body around `at` whose lambda's own row has that
/// tail reads it narrowed to the body's convention (`new-plans/ir.md`,
/// "Convention entry").
fn narrowed(ctx: &IrContext, tail: u32, at: OpRef, mut class: Class) -> Class {
    let mut current = at;
    while let Some(region) = ctx
        .op(current)
        .parent_block
        .and_then(|block| ctx.block(block).parent_region)
    {
        let Some(parent) = ctx.region(region).parent_op else {
            break;
        };
        if let Some(conventions) = tribute_cps::body_conventions(ctx, parent)
            && FuncSig::from_type_ref(ctx, ctx.op_result_types(parent)[0])
                .is_some_and(|signature| signature.row(ctx).tail == Some(tail))
            && let Some(&convention) = ctx
                .op_regions(parent)
                .position(|body| body == region)
                .and_then(|index| conventions.get(index))
        {
            class = class.narrow(convention);
        }
        current = parent;
    }
    class
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

/// The classes of the local binders of an instance's body, by binder and
/// the convention of the body of its lambda.
#[derive(Default, PartialEq, Eq)]
pub(crate) struct Locals {
    classes: HashMap<(u32, CallingConvention), Class>,
    owners: HashMap<u32, OpRef>,
}

impl Locals {
    /// The class of local `binder` in the body of its `tribute_cps.lambda`
    /// that holds `at`. Elsewhere, such as in a lambda whose bodies are not
    /// yet decided, the binder reads as the merge of its classes in every
    /// body.
    pub(crate) fn get(&self, ctx: &IrContext, binder: u32, at: OpRef) -> Option<Class> {
        let body = self.owners.get(&binder).and_then(|&owner| {
            tribute_cps::body_conventions(ctx, owner)?;
            tribute_cps::convention_at(ctx, owner, at)
        });
        match body {
            Some(body) => self.classes.get(&(binder, body)).copied(),
            None => self
                .classes
                .iter()
                .filter(|((known, _), _)| *known == binder)
                .map(|(_, &class)| class)
                .reduce(Class::merge),
        }
    }
}

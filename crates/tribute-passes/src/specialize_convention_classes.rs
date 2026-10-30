//! Class specialization: the first pass of the CPS strategy
//! (`new-plans/cps-effects.md`, "Class 특수화 pass").
//!
//! The frontend's source-logical IR has rows but no calling conventions. This
//! pass selects the class variables of every definition, collects the class
//! instances root `main` and the references' `row_args` need, clones each
//! definition per instance, points every reference at its instance, gives
//! each let-bound lambda of an instance a body per class its instantiations
//! select, and replaces every source callable and indirect transfer with its
//! `tribute_cps` operation, which declares its convention.
//!
//! A lambda is first copied once per class vector its instantiations select,
//! each copy with its own binders, so every copy reads one class for each
//! binder while the instance is rewritten. The lambda and its first copy of
//! each other class then fold into one lambda with a body per class, whose
//! bodies read its own binders again; a copy at a class the lambda already
//! has stays a separate value.
//!
//! Class variables are selected on the unchanged module. Instances are then
//! rewritten one at a time, and the instance each reference selects, read
//! with the local classes of its lambda copy, is cloned from an untouched copy
//! of its definition when first seen. A definition that ends up
//! unreferenced stays in place for the removal of unreferenced definitions that
//! follows the pass.

use std::collections::VecDeque;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::calling_convention::CallingConvention;
use tribute_ir::dialect::tribute_control::{self, FuncSig, Row, RowArgs, RowBinder};
use tribute_ir::dialect::tribute_cps;
use trunk_ir::context::IrContext;
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;
use trunk_ir::{Symbol, SymbolPath};

use crate::convention_classes::{
    Function, Locals, class_suffix, instantiation, requirements, row_class, settle_class_variables,
    settle_locals,
};

/// Why a module cannot be specialized.
#[derive(Debug, derive_more::Display, derive_more::Error)]
pub enum ClassSpecializationError {
    /// The module already holds `tribute_cps` operations.
    #[display("class specialization takes a module without tribute_cps operations")]
    ConventionsAssigned,
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
    if has_cps_operations(ctx, module) {
        return Err(ClassSpecializationError::ConventionsAssigned);
    }
    let requirements = requirements(ctx, module);
    let mut functions: Vec<Function> = module
        .ops(ctx)
        .iter()
        .filter_map(|&op| Function::read(ctx, op))
        .collect();
    let positions = settle_class_variables(&mut functions);
    let variables = |name: &str| positions.get(name).map_or(&[][..], Vec::as_slice);
    let by_name: HashMap<&str, &Function> =
        functions.iter().map(|f| (f.name.as_str(), f)).collect();

    // Instances are cloned from untouched copies, since a primary instance is
    // rewritten in place before every reference to its definition is known.
    let templates: HashMap<&str, OpRef> = functions
        .iter()
        .map(|f| (f.name.as_str(), ctx.clone_op(f.op, &mut IrMapping::new())))
        .collect();
    let mut known: HashMap<&str, Vec<Vec<CallingConvention>>> = HashMap::default();
    let mut anchors: HashMap<&str, OpRef> = HashMap::default();
    let mut queue = VecDeque::new();
    for function in &functions {
        let primary = if function.root_entry {
            CallingConvention::Direct
        } else {
            CallingConvention::Cps
        };
        let classes = vec![primary; variables(&function.name).len()];
        known.insert(&function.name, vec![classes.clone()]);
        anchors.insert(&function.name, function.op);
        queue.push_back((function, function.op, classes));
    }
    let mut instances = Vec::new();
    while let Some((function, op, classes)) = queue.pop_front() {
        let mut instance = Instance {
            function,
            requirements: &requirements,
            params: variables(&function.name)
                .iter()
                .copied()
                .zip(classes)
                .collect(),
            positions: &positions,
            by_name: &by_name,
            binders: function.binders.clone(),
            locals: Locals::default(),
            copies: Vec::new(),
        };
        let selected = instance.rewrite(ctx, op);
        instances.push((op, instance.own_convention(ctx, op)));
        for (callee, classes) in selected {
            let callee = by_name[callee.as_str()];
            let vectors = known.entry(&callee.name).or_default();
            if vectors.contains(&classes) {
                continue;
            }
            vectors.push(classes.clone());
            let clone = ctx.clone_op(templates[callee.name.as_str()], &mut IrMapping::new());
            let name = format!("{}{}", callee.name, class_suffix(&classes));
            let name = ctx.intern_str(&name);
            ctx.op_attrs_mut(clone)
                .insert("sym_name", Attribute::String(name));
            let anchor = anchors
                .get_mut(callee.name.as_str())
                .expect("every definition has an anchor");
            insert_after(ctx, *anchor, clone);
            *anchor = clone;
            queue.push_back((callee, clone, classes));
        }
    }
    for (_, template) in templates {
        ctx.remove_op(template);
    }
    for (op, convention) in instances {
        tribute_cps::replace_control_op(ctx, op, convention);
    }
    add_func_tables(ctx, module);
    Ok(())
}

/// Point every `tribute_cps.func_ref` at the table of the instance it names,
/// and add each table after its instance. An instance has one class vector,
/// so its table has the instance as its only target, at the instance's own
/// convention (`new-plans/ir.md`, "Convention entry", 함수 참조).
fn add_func_tables(ctx: &mut IrContext, module: Module) {
    let mut references = Vec::new();
    let _ = trunk_ir::walk::walk_op::<()>(ctx, module.op(), &mut |op| {
        if tribute_cps::FuncRef::matches(ctx, op) {
            references.push(op);
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    let instances: HashMap<String, OpRef> = module
        .ops(ctx)
        .iter()
        .filter(|&&op| tribute_cps::Func::matches(ctx, op))
        .filter_map(|&op| Some((ctx.op_attrs(op).get_str("sym_name")?.to_owned(), op)))
        .collect();
    let mut tables: HashSet<String> = HashSet::default();
    for reference in references {
        let target = tribute_cps::FuncRef::from_op(ctx, reference)
            .expect("collected a func_ref")
            .func_ref(ctx)
            .clone();
        let Some(&instance) = target
            .as_simple()
            .and_then(|name| instances.get(name.as_str()))
        else {
            continue;
        };
        let table = format!(
            "{}{TABLE_SUFFIX}",
            target.leaf().expect("a reference has a leaf")
        );
        ctx.op_attrs_mut(reference).insert(
            "func_ref",
            Attribute::SymbolRef(SymbolPath::new([Symbol::new(&table)])),
        );
        if !tables.insert(table.clone()) {
            continue;
        }
        let convention =
            tribute_cps::declared_convention(ctx, instance).expect("an instance has a convention");
        let location = ctx.op(instance).location;
        let op = tribute_cps::FuncTable::operands()
            .sym_name(table)
            .conventions([convention])
            .targets([target])
            .build(ctx, location);
        insert_after(ctx, instance, op.op_ref());
    }
}

/// The suffix of a function table's name after its instance's name
/// (`new-plans/cps-effects.md`, "Class 특수화 pass").
const TABLE_SUFFIX: &str = "$8";

/// Whether `module` holds a `tribute_cps` operation.
fn has_cps_operations(ctx: &IrContext, module: Module) -> bool {
    trunk_ir::walk::walk_op::<()>(ctx, module.op(), &mut |op| {
        if ctx.op(op).name.dialect(ctx) == tribute_cps::DIALECT_NAME {
            std::ops::ControlFlow::Break(())
        } else {
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        }
    })
    .is_break()
}

/// Insert `op` right after `anchor`.
fn insert_after(ctx: &mut IrContext, anchor: OpRef, op: OpRef) {
    let block = ctx
        .op(anchor)
        .parent_block
        .expect("an anchor lives in a block");
    let ops = &ctx.block(block).ops;
    let next = ops
        .iter()
        .position(|&candidate| candidate == anchor)
        .and_then(|at| ops.get(at + 1).copied());
    match next {
        Some(next) => ctx.insert_op_before(block, next, op),
        None => ctx.push_op(block, op),
    }
}

/// Keep the captures of every lambda between `instantiate` and the region
/// that defines `original` and `copy` equal to its bodies' external references
/// after `instantiate` moved from `original` to `copy`. The copy takes the
/// original's place in the capture order, next to it while the body still
/// uses both.
fn recapture(ctx: &mut IrContext, instantiate: OpRef, lambda: OpRef, copy: ValueRef) {
    let original = ctx.op_results(lambda)[0];
    let defining = region_of(ctx, lambda);
    let mut op = instantiate;
    while let Some(region) = region_of(ctx, op).filter(|&region| Some(region) != defining) {
        let parent = ctx
            .region(region)
            .parent_op
            .expect("a region inside the copied lambda's region belongs to an operation");
        if tribute_control::Lambda::matches(ctx, parent)
            || tribute_cps::Lambda::matches(ctx, parent)
        {
            let bodies: Vec<_> = ctx.op_regions(parent).collect();
            let used = |ctx: &IrContext, value| {
                ctx.uses(value)
                    .iter()
                    .any(|use_| bodies.iter().any(|&body| is_within(ctx, use_.user, body)))
            };
            let old = ctx.op_operands(parent).to_vec();
            let mut captures = old.clone();
            let at = captures.iter().position(|&value| value == original);
            if !captures.contains(&copy) && used(ctx, copy) {
                captures.insert(at.map_or(captures.len(), |at| at + 1), copy);
            }
            if !used(ctx, original) {
                captures.retain(|&value| value != original);
            }
            if captures != old {
                for index in (0..old.len()).rev() {
                    ctx.remove_op_operand(parent, index as u32);
                }
                for value in captures {
                    ctx.push_op_operand(parent, value);
                }
            }
        }
        op = parent;
    }
}

/// Rename the binders `old` to `new` in the types, attributes, quantified
/// binders, row unions, and row arguments of `ops`.
fn rename_binders(ctx: &mut IrContext, ops: &[OpRef], old: &[u32], new: &[u32]) {
    let binder = |binder: u32| {
        old.iter()
            .position(|&known| known == binder)
            .map_or(binder, |at| new[at])
    };
    let rows: Vec<Row> = new.iter().map(|&tail| Row::new([], Some(tail))).collect();
    let renamed = |row: &Row| {
        row.substitute(RowArgs::Binders(old, &rows))
            .expect("a row names binders of its definition")
    };
    let rename = |ctx: &mut IrContext, ty| {
        tribute_control::substitute_rows(ctx, ty, RowArgs::Binders(old, &rows))
            .expect("a type names binders of its definition")
    };
    for &op in ops {
        for index in 0..ctx.op_result_types(op).len() {
            let ty = rename(ctx, ctx.op_result_types(op)[index]);
            ctx.set_op_result_type(op, index as u32, ty);
        }
        for region in ctx.op_regions(op).collect::<Vec<_>>() {
            for block in ctx.region(region).blocks.clone() {
                for index in 0..ctx.block_args(block).len() {
                    let ty = rename(ctx, ctx.value_ty(ctx.block_args(block)[index]));
                    ctx.set_block_arg_type(block, index as u32, ty);
                }
            }
        }
        let attributes: Vec<_> = ctx
            .op(op)
            .attributes
            .iter()
            .map(|(key, value)| (key, value.clone()))
            .collect();
        for (key, value) in attributes {
            let renamed = value.map_types(|ty| rename(ctx, ty));
            if renamed != value {
                ctx.op_attrs_mut(op).insert(key, renamed);
            }
        }
        let quantified = tribute_control::quantified_rows(ctx, op).unwrap_or_default();
        if !quantified.is_empty() {
            let quantified = quantified
                .iter()
                .map(|&quantified| Attribute::Int(i128::from(binder(quantified))))
                .collect();
            ctx.op_attrs_mut(op).insert(
                tribute_control::QUANTIFIED_ROWS_ATTR,
                Attribute::List(quantified),
            );
        }
        let unions = tribute_control::row_unions(ctx, op).unwrap_or_default();
        if !unions.is_empty() {
            let unions = unions
                .into_iter()
                .map(|union| {
                    tribute_control::RowUnion {
                        result: binder(union.result),
                        sources: union.sources.into_iter().map(binder).collect(),
                    }
                    .to_attribute(ctx)
                })
                .collect();
            ctx.op_attrs_mut(op)
                .insert(tribute_control::ROW_UNIONS_ATTR, Attribute::List(unions));
        }
        let args = tribute_control::row_args(ctx, op).unwrap_or_default();
        if !args.is_empty() {
            let args = args
                .iter()
                .map(|row| renamed(row).to_attribute(ctx))
                .collect();
            ctx.op_attrs_mut(op)
                .insert(tribute_control::ROW_ARGS_ATTR, Attribute::List(args));
        }
    }
}

/// The operations nested in `op`, without `op` itself.
fn nested_ops(ctx: &IrContext, op: OpRef) -> Vec<OpRef> {
    let mut ops = Vec::new();
    let _ = trunk_ir::walk::walk_op::<()>(ctx, op, &mut |inner| {
        if inner != op {
            ops.push(inner);
        }
        std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
    });
    ops
}

/// The region whose block holds `op`.
fn region_of(ctx: &IrContext, op: OpRef) -> Option<RegionRef> {
    ctx.op(op)
        .parent_block
        .and_then(|block| ctx.block(block).parent_region)
}

/// Whether `op` is nested in `region`.
fn is_within(ctx: &IrContext, op: OpRef, region: RegionRef) -> bool {
    std::iter::successors(region_of(ctx, op), |&inner| {
        ctx.region(inner)
            .parent_op
            .and_then(|owner| region_of(ctx, owner))
    })
    .any(|inner| inner == region)
}

/// A let-bound lambda in one round of copying: its instantiations grouped by
/// the classes they select, in body order, and the tails of their rows.
struct Lambda {
    op: OpRef,
    groups: Vec<(Vec<CallingConvention>, Vec<OpRef>)>,
    tails: Vec<u32>,
}

/// One instance being rewritten: its definition, the class of each class
/// variable, and the row binders and local classes its lambda copies leave.
struct Instance<'a> {
    function: &'a Function,
    requirements: &'a HashMap<SymbolPath, CallingConvention>,
    params: HashMap<u32, CallingConvention>,
    positions: &'a HashMap<String, Vec<u32>>,
    /// The module's definitions, by name.
    by_name: &'a HashMap<&'a str, &'a Function>,
    binders: Vec<RowBinder>,
    locals: Locals,
    /// Each lambda split in this instance, with its copies in body order.
    copies: Vec<(OpRef, Vec<OpRef>)>,
}

impl Instance<'_> {
    /// The class `row` requires at `at`.
    fn class(&self, ctx: &IrContext, row: &Row, at: OpRef) -> CallingConvention {
        row_class(
            row,
            &self.binders,
            self.requirements,
            &self.params,
            |binder| self.locals.get(ctx, binder, at),
        )
    }

    /// Rewrite `op` as this instance, and return the instance each of its
    /// references selects.
    fn rewrite(&mut self, ctx: &mut IrContext, op: OpRef) -> Vec<(String, Vec<CallingConvention>)> {
        if let Some(body) = ctx.op_region(op, 0) {
            self.split_lambdas(ctx, body);
        }
        let mut ops = Vec::new();
        if let Some(body) = ctx.op_region(op, 0) {
            let _ = trunk_ir::walk::walk_region::<()>(ctx, body, &mut |inner| {
                ops.push(inner);
                std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
            });
        }
        let selected: Vec<_> = ops
            .iter()
            .filter_map(|&inner| self.point_reference(ctx, inner))
            .collect();
        let conventions: HashMap<OpRef, CallingConvention> = ops
            .iter()
            .filter_map(|&inner| Some((inner, self.transfer_convention(ctx, inner)?)))
            .collect();
        let (folded, unused) = self.fold_copies(ctx, &conventions);
        for &inner in &ops {
            if !folded.contains(&inner)
                && let Some(&convention) = conventions.get(&inner)
            {
                tribute_cps::replace_control_op(ctx, inner, convention);
            }
        }
        self.drop_binders(ctx, op, &unused);
        if self.binders.len() != self.function.binders.len() {
            let binders = self
                .binders
                .iter()
                .map(|binder| Attribute::String(ctx.intern_str(binder.keyword())))
                .collect();
            ctx.op_attrs_mut(op)
                .insert(tribute_control::ROW_BINDERS_ATTR, Attribute::List(binders));
        }
        selected
    }

    /// Copy each let-bound lambda once per class vector its instantiations
    /// select, until every lambda has one, and settle the local classes.
    ///
    /// A lambda's classes can depend on the local binders of other lambdas,
    /// so each round copies one lambda whose dependencies already have one
    /// class vector each; its classes are then final, and copying it never
    /// leaves two copies at one vector.
    fn split_lambdas(&mut self, ctx: &mut IrContext, body: RegionRef) {
        loop {
            self.locals = settle_locals(ctx, body, &self.binders, self.requirements, &self.params);
            let mut owners: HashMap<u32, OpRef> = HashMap::default();
            let mut lambdas: Vec<Lambda> = Vec::new();
            let _ = trunk_ir::walk::walk_region::<()>(ctx, body, &mut |op| {
                for binder in tribute_control::quantified_rows(ctx, op).unwrap_or_default() {
                    owners.insert(binder, op);
                }
                if let Some((lambda, _, rows)) = instantiation(ctx, op) {
                    let classes: Vec<_> = rows.iter().map(|row| self.class(ctx, row, op)).collect();
                    let at = match lambdas.iter().position(|known| known.op == lambda) {
                        Some(at) => at,
                        None => {
                            lambdas.push(Lambda {
                                op: lambda,
                                groups: Vec::new(),
                                tails: Vec::new(),
                            });
                            lambdas.len() - 1
                        }
                    };
                    let known = &mut lambdas[at];
                    known.tails.extend(rows.iter().filter_map(|row| row.tail));
                    match known.groups.iter_mut().find(|(known, _)| *known == classes) {
                        Some((_, uses)) => uses.push(op),
                        None => known.groups.push((classes, vec![op])),
                    }
                }
                std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
            });
            let by_op: HashMap<OpRef, &Lambda> = lambdas.iter().map(|l| (l.op, l)).collect();
            let waits = |lambda: &Lambda| {
                let mut seen = HashSet::default();
                let mut stack = vec![lambda];
                while let Some(lambda) = stack.pop() {
                    for owner in lambda.tails.iter().filter_map(|tail| owners.get(tail)) {
                        if let Some(&dependency) = by_op.get(owner)
                            && seen.insert(dependency.op)
                        {
                            if dependency.groups.len() > 1 {
                                return true;
                            }
                            stack.push(dependency);
                        }
                    }
                }
                false
            };
            let Some(lambda) = lambdas
                .iter()
                .find(|lambda| lambda.groups.len() > 1 && !waits(lambda))
            else {
                return;
            };
            let mut anchor = lambda.op;
            let mut copies = Vec::new();
            for (_, uses) in &lambda.groups[1..] {
                let copy = self.copy_lambda(ctx, lambda.op);
                insert_after(ctx, anchor, copy);
                anchor = copy;
                copies.push(copy);
                let result = ctx.op_results(copy)[0];
                for &instantiate in uses {
                    ctx.set_op_operand(instantiate, 0, result);
                    recapture(ctx, instantiate, lambda.op, result);
                }
            }
            match self
                .copies
                .iter_mut()
                .find(|(known, _)| *known == lambda.op)
            {
                Some((_, known)) => known.extend(copies),
                None => self.copies.push((lambda.op, copies)),
            }
        }
    }

    /// A detached copy of `lambda` whose quantifying lambdas, itself
    /// included, quantify new local binders.
    fn copy_lambda(&mut self, ctx: &mut IrContext, lambda: OpRef) -> OpRef {
        let mut mapping = IrMapping::new();
        let copy = ctx.clone_op(lambda, &mut mapping);
        // A lambda inside `lambda` that was already split is split in the
        // copy too, so the copy's lambdas fold the same way.
        let cloned = |ctx: &IrContext, op: OpRef| {
            mapping
                .lookup_value(ctx.op_results(op)[0])
                .and_then(|value| match ctx.value_def(value) {
                    trunk_ir::refs::ValueDef::OpResult(op, _) => Some(op),
                    _ => None,
                })
        };
        let nested: Vec<_> = self
            .copies
            .iter()
            .filter_map(|(inner, copies)| {
                let inner = cloned(ctx, *inner)?;
                let copies = copies
                    .iter()
                    .map(|&copy| cloned(ctx, copy))
                    .collect::<Option<Vec<_>>>()?;
                Some((inner, copies))
            })
            .collect();
        self.copies.extend(nested);
        let mut ops = vec![copy];
        ops.extend(nested_ops(ctx, copy));
        let mut old = Vec::new();
        for &op in &ops {
            old.extend(tribute_control::quantified_rows(ctx, op).unwrap_or_default());
        }
        let new: Vec<u32> = (0..old.len())
            .map(|offset| u32::try_from(self.binders.len() + offset).expect("binders fit in u32"))
            .collect();
        self.binders.extend(old.iter().map(|_| RowBinder::Local));
        rename_binders(ctx, &ops, &old, &new);
        copy
    }

    /// Fold each split lambda and its first copy of every other convention
    /// into one `tribute_cps.lambda` with a body per convention, and return
    /// the lambdas folded away. A copy at a convention the lambda already
    /// has stays its own value. The bodies of the copies read the lambda's
    /// own binders again, and the uses and captures of the copies move to
    /// the folded lambda, whose own binders they no longer quantify.
    fn fold_copies(
        &self,
        ctx: &mut IrContext,
        conventions: &HashMap<OpRef, CallingConvention>,
    ) -> (HashSet<OpRef>, Vec<u32>) {
        let mut folded = HashSet::default();
        let mut unused = Vec::new();
        for (lambda, copies) in &self.copies {
            if folded.contains(lambda) {
                continue;
            }
            let mut members = vec![(conventions[lambda], *lambda)];
            for &copy in copies.iter().filter(|copy| !folded.contains(*copy)) {
                let convention = conventions[&copy];
                if members.iter().all(|&(known, _)| known != convention) {
                    members.push((convention, copy));
                }
            }
            if members.len() == 1 {
                continue;
            }
            members.sort_by_key(|&(convention, _)| convention);
            let own = tribute_control::quantified_rows(ctx, *lambda).unwrap_or_default();
            let mut captures = ctx.op_operands(*lambda).to_vec();
            for &(_, member) in &members {
                for &capture in ctx.op_operands(member) {
                    if !captures.contains(&capture) {
                        captures.push(capture);
                    }
                }
                if member != *lambda {
                    let fresh = tribute_control::quantified_rows(ctx, member).unwrap_or_default();
                    let mut ops = vec![member];
                    ops.extend(nested_ops(ctx, member));
                    rename_binders(ctx, &ops, &fresh, &own);
                    unused.extend(fresh);
                }
            }
            let data = ctx.op(*lambda);
            let mut builder = trunk_ir::context::OperationDataBuilder::new(
                data.location,
                tribute_cps::Lambda::NAME,
            )
            .operands(captures)
            .results(ctx.op_result_types(*lambda).iter().copied());
            for (key, value) in data.attributes.iter() {
                builder = builder.attr(key, value.clone());
            }
            let mut written = Vec::new();
            for &(convention, member) in &members {
                written.push(<CallingConvention as trunk_ir::attr_kind::AttrKind>::write(
                    ctx, convention,
                ));
                let body = ctx.op_region(member, 0).expect("a lambda has a body");
                ctx.detach_region(body);
                builder = builder.region(body);
            }
            let data = builder
                .attr(tribute_cps::CONVENTIONS_ATTR, Attribute::List(written))
                .build(ctx);
            let merged = ctx.create_op(data);
            let block = ctx
                .op(*lambda)
                .parent_block
                .expect("a split lambda lives in a block");
            ctx.insert_op_before(block, *lambda, merged);
            let result = ctx.op_results(merged)[0];
            ctx.replace_all_uses(ctx.op_results(*lambda)[0], result);
            for &(_, member) in &members {
                if member == *lambda {
                    continue;
                }
                let uses: Vec<_> = ctx
                    .uses(ctx.op_results(member)[0])
                    .iter()
                    .map(|use_| use_.user)
                    .filter(|&user| tribute_control::Instantiate::matches(ctx, user))
                    .collect();
                for instantiate in uses {
                    ctx.set_op_operand(instantiate, 0, result);
                    recapture(ctx, instantiate, member, result);
                }
            }
            for &(_, member) in &members {
                debug_assert!(
                    ctx.uses(ctx.op_results(member)[0]).is_empty(),
                    "a folded lambda keeps no use"
                );
                ctx.remove_op_from_block(block, member);
                ctx.remove_op(member);
                folded.insert(member);
            }
        }
        (folded, unused)
    }

    /// Remove the local binders `unused` from this instance, `op`, and
    /// renumber the binders after them.
    fn drop_binders(&mut self, ctx: &mut IrContext, op: OpRef, unused: &[u32]) {
        if unused.is_empty() {
            return;
        }
        let kept: Vec<u32> = (0..self.binders.len())
            .map(|index| u32::try_from(index).expect("binders fit in u32"))
            .filter(|index| !unused.contains(index))
            .collect();
        let renumbered: Vec<u32> = (0..kept.len())
            .map(|index| u32::try_from(index).expect("binders fit in u32"))
            .collect();
        let mut ops = vec![op];
        ops.extend(nested_ops(ctx, op));
        rename_binders(ctx, &ops, &kept, &renumbered);
        let mut index = 0;
        self.binders.retain(|_| {
            index += 1;
            !unused.contains(&(index - 1))
        });
    }

    /// Point a reference at the instance its rows select, and return that
    /// instance.
    fn point_reference(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
    ) -> Option<(String, Vec<CallingConvention>)> {
        let key = if tribute_control::FuncRef::matches(ctx, op) {
            "func_ref"
        } else if tribute_control::Call::matches(ctx, op)
            || tribute_control::TailCall::matches(ctx, op)
        {
            "callee"
        } else {
            return None;
        };
        let callee = ctx
            .op_attrs(op)
            .get_symbol_ref(key)
            .and_then(|path| path.leaf())
            .expect("a reference names a definition")
            .to_string();
        if !self.by_name.contains_key(callee.as_str()) {
            return None;
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
                    .map_or(CallingConvention::Cps, |row| self.class(ctx, row, op))
            })
            .collect();
        let suffix = class_suffix(&classes);
        if !suffix.is_empty() {
            let name = SymbolPath::new([Symbol::new(&format!("{callee}{suffix}"))]);
            ctx.op_attrs_mut(op).insert(key, Attribute::SymbolRef(name));
        }
        Some((callee, classes))
    }

    /// The convention of a lambda, `func_ref`, or indirect transfer: the class
    /// of the row of the callable it makes or calls.
    fn transfer_convention(&self, ctx: &IrContext, op: OpRef) -> Option<CallingConvention> {
        let ty = if tribute_control::Lambda::matches(ctx, op)
            || tribute_control::FuncRef::matches(ctx, op)
        {
            ctx.op_result_types(op)[0]
        } else if tribute_control::CallIndirect::matches(ctx, op)
            || tribute_control::TailCallIndirect::matches(ctx, op)
        {
            ctx.value_ty(ctx.op_operands(op)[0])
        } else {
            return None;
        };
        let signature = FuncSig::from_type_ref(ctx, ty)?;
        Some(self.class(ctx, &signature.row(ctx), op))
    }

    /// The instance's own convention: the class of its row, where a
    /// definition with an inferred row reads its own tail only when that tail
    /// is a class variable.
    fn own_convention(&self, ctx: &IrContext, op: OpRef) -> CallingConvention {
        let ty = ctx
            .op_attrs(op)
            .get_type("type")
            .expect("a definition has a signature");
        let signature =
            FuncSig::from_type_ref(ctx, ty).expect("a definition's signature is a func_sig");
        let mut row = signature.row(ctx);
        if self.function.inferred
            && row
                .tail
                .is_some_and(|tail| !self.params.contains_key(&tail))
        {
            row.tail = None;
        }
        self.class(ctx, &row, op)
    }
}

#[cfg(test)]
mod tests;

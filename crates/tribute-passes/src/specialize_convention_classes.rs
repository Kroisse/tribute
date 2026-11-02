//! Class specialization: the first pass of the CPS strategy
//! (`new-plans/cps-effects.md`, "Class 특수화 pass").
//!
//! The frontend's source-logical IR has rows but no calling conventions. This
//! pass selects the class variables of every definition, collects the class
//! instances root `main` and the references' `row_args` need, clones each
//! definition per instance, points every reference at its instance or
//! function table, gives each lambda of an instance a body per class the
//! body rule selects, and replaces every source callable and indirect
//! transfer with its `tribute_cps` operation, which declares its convention.
//!
//! A class is exact or an upper bound (`new-plans/ir.md`, "Convention
//! entry"). A lambda stays one value: uses that select one body but pass
//! different classes for a binder make that binder an upper bound in that
//! body. Each body is a copy of the lambda's body, whose nested lambdas
//! quantify new local binders; the lambda's own binders are shared by its
//! bodies and read by the convention of the body that holds them.
//!
//! Class variables are selected on the unchanged module. Instances are then
//! rewritten one at a time, and the instance each reference selects, read
//! in the body that holds it, is cloned from an untouched copy of its
//! definition when first seen. A definition that ends up unreferenced stays
//! in place for the removal of unreferenced definitions that follows the
//! pass.

use std::collections::VecDeque;

use rustc_hash::FxHashMap as HashMap;
use tribute_ir::calling_convention::CallingConvention;
use tribute_ir::dialect::tribute_control::{self, FuncSig, Row, RowArgs, RowBinder};
use tribute_ir::dialect::tribute_cps;
use trunk_ir::context::IrContext;
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;
use trunk_ir::{Symbol, SymbolPath};

use crate::convention_classes::{
    Class, Function, Locals, Reader, TABLE_SUFFIX, class_suffix, instantiation, requirements,
    settle_class_variables,
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

/// The targets of each function table, by table name, in the order the
/// references name them.
type Tables = Vec<(String, Vec<(CallingConvention, String)>)>;

/// Specialize the convention classes of `module`.
pub fn specialize(ctx: &mut IrContext, module: Module) -> Result<(), ClassSpecializationError> {
    if has_cps_operations(ctx, module) {
        return Err(ClassSpecializationError::ConventionsAssigned);
    }
    let requirements = requirements(ctx, module);
    let mut functions: Vec<Function> = module
        .ops(ctx)
        .iter()
        // No table exists before specialization.
        .filter_map(|&op| Function::read(ctx, op, &HashMap::default()))
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
    let mut known: HashMap<&str, Vec<Vec<Class>>> = HashMap::default();
    let mut anchors: HashMap<&str, OpRef> = HashMap::default();
    let mut queue = VecDeque::new();
    for function in &functions {
        let primary = if function.root_entry {
            Class::DIRECT
        } else {
            Class::CPS
        };
        let classes = vec![primary; variables(&function.name).len()];
        known.insert(&function.name, vec![classes.clone()]);
        anchors.insert(&function.name, function.op);
        queue.push_back((function, function.op, classes));
    }
    let mut instances = Vec::new();
    let mut tables = Tables::new();
    while let Some((function, op, classes)) = queue.pop_front() {
        let mut instance = Instance {
            function,
            positions: &positions,
            by_name: &by_name,
            reader: Reader {
                binders: function.binders.clone(),
                requirements: &requirements,
                params: variables(&function.name)
                    .iter()
                    .copied()
                    .zip(classes)
                    .collect(),
                locals: Locals::default(),
            },
        };
        let selected = instance.rewrite(ctx, op, &mut tables);
        instances.push((op, instance.own_class(ctx, op).bound()));
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
    add_func_tables(ctx, module, tables);
    Ok(())
}

/// Add each function table after its first target.
fn add_func_tables(ctx: &mut IrContext, module: Module, tables: Tables) {
    let instances: HashMap<String, OpRef> = module
        .ops(ctx)
        .iter()
        .filter(|&&op| tribute_cps::Func::matches(ctx, op))
        .filter_map(|&op| Some((ctx.op_attrs(op).get_str("sym_name")?.to_owned(), op)))
        .collect();
    for (table, targets) in tables {
        let anchor = instances[&targets[0].1];
        let location = ctx.op(anchor).location;
        let entries = targets
            .into_iter()
            .map(|(convention, target)| {
                (
                    convention.keyword().into(),
                    SymbolPath::new([Symbol::new(&target)]),
                )
            })
            .collect();
        let op = tribute_cps::FuncTable::operands()
            .sym_name(table)
            .entries(entries)
            .build(ctx, location);
        insert_after(ctx, anchor, op.op_ref());
    }
}

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

/// The own row of definition `op`: its signature's row, where a definition
/// with an inferred row reads its own tail only when `is_class_variable`
/// says that tail is a class variable.
fn own_row(
    ctx: &IrContext,
    op: OpRef,
    inferred: bool,
    is_class_variable: impl Fn(u32) -> bool,
) -> Row {
    let ty = ctx
        .op_attrs(op)
        .get_type("type")
        .expect("a definition has a signature");
    let signature =
        FuncSig::from_type_ref(ctx, ty).expect("a definition's signature is a func_sig");
    let mut row = signature.row(ctx);
    if inferred && row.tail.is_some_and(|tail| !is_class_variable(tail)) {
        row.tail = None;
    }
    row
}

/// A lambda whose bodies are not yet the ones the body rule selects: the
/// conventions of those bodies, and the binders that selection reads.
struct Pending {
    op: OpRef,
    conventions: Vec<CallingConvention>,
    reads: Vec<u32>,
}

/// One instance being rewritten: its definition and its class reading.
struct Instance<'a> {
    function: &'a Function,
    positions: &'a HashMap<String, Vec<u32>>,
    /// The module's definitions, by name.
    by_name: &'a HashMap<&'a str, &'a Function>,
    reader: Reader<'a>,
}

impl Instance<'_> {
    /// Rewrite `op` as this instance, add the function tables its
    /// references name to `tables`, and return the instance each of its
    /// references selects.
    fn rewrite(
        &mut self,
        ctx: &mut IrContext,
        op: OpRef,
        tables: &mut Tables,
    ) -> Vec<(String, Vec<Class>)> {
        if let Some(body) = ctx.op_region(op, 0) {
            self.give_bodies(ctx, body);
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
            .flat_map(|&inner| self.point_reference(ctx, inner, tables))
            .collect();
        let conventions: Vec<_> = ops
            .iter()
            .filter_map(|&inner| Some((inner, self.transfer_convention(ctx, inner)?)))
            .collect();
        for (inner, convention) in conventions {
            tribute_cps::replace_control_op(ctx, inner, convention);
        }
        if self.reader.binders.len() != self.function.binders.len() {
            let binders = self
                .reader
                .binders
                .iter()
                .map(|binder| Attribute::String(ctx.intern_str(binder.keyword())))
                .collect();
            ctx.op_attrs_mut(op)
                .insert(tribute_control::ROW_BINDERS_ATTR, Attribute::List(binders));
        }
        selected
    }

    /// Give every lambda in `body` the bodies the body rule selects
    /// (`new-plans/ir.md`, "Convention entry"), one lambda at a time, and
    /// settle the local classes.
    ///
    /// The bodies a lambda needs depend on the binders its uses and its own
    /// row read, so a lambda waits for the lambdas that quantify those
    /// binders to have their bodies. A new body can select new bodies of
    /// another lambda, so the rounds repeat until no lambda needs one; since
    /// a lambda has at most three bodies, they end.
    fn give_bodies(&mut self, ctx: &mut IrContext, body: RegionRef) {
        loop {
            self.reader.settle_locals(ctx, body);
            let pending = self.pending(ctx, body);
            let owners: HashMap<u32, OpRef> = pending
                .iter()
                .flat_map(|lambda| {
                    tribute_control::quantified_rows(ctx, lambda.op)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|binder| (binder, lambda.op))
                })
                .collect();
            let ready = pending.iter().find(|lambda| {
                lambda
                    .reads
                    .iter()
                    .all(|binder| owners.get(binder).is_none_or(|&owner| owner == lambda.op))
            });
            let Some(lambda) = ready.or(pending.first()) else {
                return;
            };
            self.make_bodies(ctx, lambda.op, &lambda.conventions);
        }
    }

    /// The lambdas in `body` whose bodies differ from those the body rule
    /// selects, in walk order.
    fn pending(&self, ctx: &IrContext, body: RegionRef) -> Vec<Pending> {
        let mut lambdas: Vec<Pending> = Vec::new();
        let mut instantiated = Vec::new();
        let _ = trunk_ir::walk::walk_region::<()>(ctx, body, &mut |op| {
            if tribute_control::Lambda::matches(ctx, op) || tribute_cps::Lambda::matches(ctx, op) {
                lambdas.push(Pending {
                    op,
                    conventions: Vec::new(),
                    reads: Vec::new(),
                });
            } else if let Some((lambda, _, rows)) = instantiation(ctx, op)
                && let Some(made) = FuncSig::from_type_ref(ctx, ctx.op_result_types(op)[0])
            {
                let made = made.row(ctx);
                instantiated.push((lambda, self.reader.conventions(ctx, &made, op), {
                    let mut reads: Vec<u32> = rows.iter().filter_map(|row| row.tail).collect();
                    reads.extend(made.tail);
                    reads
                }));
            }
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        for (lambda, conventions, reads) in instantiated {
            if let Some(known) = lambdas.iter_mut().find(|known| known.op == lambda) {
                known.conventions.extend(conventions);
                known.reads.extend(reads);
            }
        }
        for lambda in &mut lambdas {
            if lambda.conventions.is_empty()
                && let Some(signature) =
                    FuncSig::from_type_ref(ctx, ctx.op_result_types(lambda.op)[0])
            {
                let row = signature.row(ctx);
                lambda.conventions = self.reader.conventions(ctx, &row, lambda.op);
                lambda.reads.extend(row.tail);
            }
            lambda.conventions.sort_unstable();
            lambda.conventions.dedup();
        }
        lambdas.retain(|lambda| {
            tribute_cps::body_conventions(ctx, lambda.op).is_none_or(|bodies| {
                !lambda
                    .conventions
                    .iter()
                    .all(|convention| bodies.contains(convention))
            })
        });
        lambdas
    }

    /// Replace `lambda` with a `tribute_cps.lambda` that has the bodies it
    /// has and one for each of `conventions` it lacks. A
    /// `tribute_control.lambda`'s one body becomes the first of them, and
    /// each new body is a copy of an existing one whose nested lambdas
    /// quantify new local binders.
    fn make_bodies(
        &mut self,
        ctx: &mut IrContext,
        lambda: OpRef,
        conventions: &[CallingConvention],
    ) {
        let mut bodies: Vec<(CallingConvention, RegionRef)> =
            match tribute_cps::body_conventions(ctx, lambda) {
                Some(known) => known.into_iter().zip(ctx.op_regions(lambda)).collect(),
                None => {
                    let body = ctx.op_region(lambda, 0).expect("a lambda has a body");
                    vec![(conventions[0], body)]
                }
            };
        let template = bodies[0].1;
        for &convention in conventions {
            if bodies.iter().all(|&(known, _)| known != convention) {
                let copy = ctx.clone_region(template, &mut IrMapping::new());
                self.renew_binders(ctx, copy);
                bodies.push((convention, copy));
            }
        }
        bodies.sort_by_key(|&(convention, _)| convention);
        let data = ctx.op(lambda);
        let mut builder =
            trunk_ir::context::OperationDataBuilder::new(data.location, tribute_cps::Lambda::NAME)
                .operands(ctx.op_operands(lambda).iter().copied())
                .results(ctx.op_result_types(lambda).iter().copied());
        for (key, value) in data.attributes.iter() {
            builder = builder.attr(key, value.clone());
        }
        let mut written = Vec::new();
        for &(convention, body) in &bodies {
            written.push(<CallingConvention as trunk_ir::attr_kind::AttrKind>::write(
                ctx, convention,
            ));
            if ctx.region(body).parent_op.is_some() {
                ctx.detach_region(body);
            }
            builder = builder.region(body);
        }
        let data = builder
            .attr(tribute_cps::CONVENTIONS_ATTR, Attribute::List(written))
            .build(ctx);
        let replacement = ctx.create_op(data);
        let block = ctx
            .op(lambda)
            .parent_block
            .expect("a lambda lives in a block");
        ctx.insert_op_before(block, lambda, replacement);
        ctx.replace_all_uses(ctx.op_results(lambda)[0], ctx.op_results(replacement)[0]);
        ctx.remove_op_from_block(block, lambda);
        ctx.remove_op(lambda);
    }

    /// Give the lambdas nested in `body`, a detached copy, new local
    /// binders for the binders they quantify.
    fn renew_binders(&mut self, ctx: &mut IrContext, body: RegionRef) {
        let mut ops = Vec::new();
        let _ = trunk_ir::walk::walk_region::<()>(ctx, body, &mut |op| {
            ops.push(op);
            std::ops::ControlFlow::Continue(trunk_ir::walk::WalkAction::Advance)
        });
        let mut old = Vec::new();
        for &op in &ops {
            old.extend(tribute_control::quantified_rows(ctx, op).unwrap_or_default());
        }
        let new: Vec<u32> = (0..old.len())
            .map(|offset| {
                u32::try_from(self.reader.binders.len() + offset).expect("binders fit in u32")
            })
            .collect();
        self.reader
            .binders
            .extend(old.iter().map(|_| RowBinder::Local));
        rename_binders(ctx, &ops, &old, &new);
    }

    /// Point a reference at the instance its rows select, or a function
    /// reference at the table of that instance, and return the instances it
    /// enters: its callee, or the table's targets.
    fn point_reference(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        tables: &mut Tables,
    ) -> Vec<(String, Vec<Class>)> {
        let is_func_ref = tribute_control::FuncRef::matches(ctx, op);
        let key = if is_func_ref {
            "func_ref"
        } else if tribute_control::Call::matches(ctx, op)
            || tribute_control::TailCall::matches(ctx, op)
        {
            "callee"
        } else {
            return Vec::new();
        };
        let callee = ctx
            .op_attrs(op)
            .get_symbol_ref(key)
            .and_then(|path| path.leaf())
            .expect("a reference names a definition")
            .to_string();
        let Some(&definition) = self.by_name.get(callee.as_str()) else {
            return Vec::new();
        };
        let rows = tribute_control::row_args(ctx, op).unwrap_or_default();
        let positions = self.positions.get(&callee).map_or(&[][..], Vec::as_slice);
        let classes: Vec<_> = positions
            .iter()
            .map(|&position| {
                usize::try_from(position)
                    .ok()
                    .and_then(|position| rows.get(position))
                    .map_or(Class::CPS, |row| self.reader.class(ctx, row, op))
            })
            .collect();
        let name = format!("{callee}{}", class_suffix(&classes));
        if !is_func_ref {
            if name != callee {
                ctx.op_attrs_mut(op).insert(
                    key,
                    Attribute::SymbolRef(SymbolPath::new([Symbol::new(&name)])),
                );
            }
            return vec![(callee, classes)];
        }
        let table = format!("{name}{TABLE_SUFFIX}");
        ctx.op_attrs_mut(op).insert(
            key,
            Attribute::SymbolRef(SymbolPath::new([Symbol::new(&table)])),
        );
        let targets = self.table_targets(ctx, definition, positions, &classes);
        if tables.iter().all(|(known, _)| *known != table) {
            let entries = targets
                .iter()
                .map(|(convention, classes)| {
                    (*convention, format!("{callee}{}", class_suffix(classes)))
                })
                .collect();
            tables.push((table, entries));
        }
        targets
            .into_iter()
            .map(|(_, classes)| (callee.clone(), classes))
            .collect()
    }

    /// The targets of the table of `definition` for `classes`, by
    /// convention: one per class its own row may require read with those
    /// classes, each the instance whose argument for the class variable that
    /// is that row's tail narrows to that class (`new-plans/ir.md`,
    /// "Convention entry", 함수 참조).
    fn table_targets(
        &self,
        ctx: &IrContext,
        definition: &Function,
        positions: &[u32],
        classes: &[Class],
    ) -> Vec<(CallingConvention, Vec<Class>)> {
        let params: HashMap<u32, Class> = positions
            .iter()
            .copied()
            .zip(classes.iter().copied())
            .collect();
        let row = own_row(ctx, definition.op, definition.inferred, |tail| {
            params.contains_key(&tail)
        });
        let reader = Reader {
            binders: definition.binders.clone(),
            requirements: self.reader.requirements,
            params,
            locals: Locals::default(),
        };
        let tail = row
            .tail
            .and_then(|tail| positions.iter().position(|&position| position == tail));
        reader
            .conventions(ctx, &row, definition.op)
            .into_iter()
            .map(|convention| {
                let mut classes = classes.to_vec();
                if let Some(tail) = tail {
                    classes[tail] = classes[tail].narrow(convention);
                }
                (convention, classes)
            })
            .collect()
    }

    /// The convention of a `func_ref` or indirect transfer: the class of the
    /// row of the callable it makes or calls, or its bound when that class is
    /// an upper bound.
    fn transfer_convention(&self, ctx: &IrContext, op: OpRef) -> Option<CallingConvention> {
        let ty = if tribute_control::FuncRef::matches(ctx, op) {
            ctx.op_result_types(op)[0]
        } else if tribute_control::CallIndirect::matches(ctx, op)
            || tribute_control::TailCallIndirect::matches(ctx, op)
        {
            ctx.value_ty(ctx.op_operands(op)[0])
        } else {
            return None;
        };
        let signature = FuncSig::from_type_ref(ctx, ty)?;
        Some(self.reader.class(ctx, &signature.row(ctx), op).bound())
    }

    /// The instance's own class: the class of its own row.
    fn own_class(&self, ctx: &IrContext, op: OpRef) -> Class {
        let row = own_row(ctx, op, self.function.inferred, |tail| {
            self.reader.params.contains_key(&tail)
        });
        self.reader.class(ctx, &row, op)
    }
}

#[cfg(test)]
mod tests;

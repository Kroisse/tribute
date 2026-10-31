//! Check that the convention classes class specialization chose agree with
//! the classes the rows of source-logical IR imply.
//!
//! The rules are those of `new-plans/cps-effects.md` ("Row 변수의 convention
//! class"), read from the IR alone: row binders, `row_args` of references and
//! instantiations, and the operations each ability instance declares.

use std::ops::ControlFlow;

use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use tribute_ir::calling_convention::CallingConvention;
use tribute_ir::dialect::tribute_control::{FuncSig, Row};
use tribute_ir::dialect::tribute_cps;
use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::walk::{WalkAction, walk_region};

use crate::convention_classes::{
    Class, Function, Locals, Reader, class_suffix, func_tables, instantiation, requirements,
    settle_class_variables, split_name,
};

/// A place where the IR's rows imply another class than specialization chose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Disagreement {
    pub op: OpRef,
    /// The rule that disagrees, such as `suffix` or `reference`.
    pub rule: &'static str,
    pub message: String,
}

/// The disagreements a module has, as a pass failure.
#[derive(Debug, derive_more::Display, derive_more::Error)]
#[display("convention classes disagree with rows: {}", disagreements.iter().format_with("; ", |d, f| f(&format_args!("({}) {}", d.rule, d.message))))]
pub struct ConventionClassError {
    #[error(not(source))]
    pub disagreements: Vec<Disagreement>,
}

/// Fails when a convention class specialization chose disagrees with
/// the rows of the module.
pub struct CheckConventionClasses;

impl trunk_ir::pass::Pass for CheckConventionClasses {
    type Target = trunk_ir::dialect::core::Module;

    fn name(&self) -> &'static str {
        "check-convention-classes"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: trunk_ir::dialect::core::Module,
        _: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        let disagreements = check(ctx, target.into());
        if disagreements.is_empty() {
            Ok(())
        } else {
            Err(Box::new(ConventionClassError { disagreements }))
        }
    }
}

/// Every disagreement in `module`.
pub fn check(ctx: &IrContext, module: Module) -> Vec<Disagreement> {
    let requirements = requirements(ctx, module);
    let tables = func_tables(ctx, module);
    let mut functions: Vec<Function> = module
        .ops(ctx)
        .iter()
        .filter_map(|&op| Function::read(ctx, op, &tables))
        .collect();
    let positions = settle_class_variables(&mut functions);
    let mut disagreements = Vec::new();
    for function in &functions {
        Instance::new(
            ctx,
            function,
            &positions,
            &requirements,
            &tables,
            &mut disagreements,
        )
        .check(&mut disagreements);
    }
    disagreements
}

/// One function instance with the classes its name fixes.
struct Instance<'a> {
    ctx: &'a IrContext,
    function: &'a Function,
    positions: &'a HashMap<String, Vec<u32>>,
    /// The module's function tables, by name.
    tables: &'a HashMap<String, OpRef>,
    reader: Reader<'a>,
}

impl<'a> Instance<'a> {
    fn new(
        ctx: &'a IrContext,
        function: &'a Function,
        positions: &'a HashMap<String, Vec<u32>>,
        requirements: &'a HashMap<SymbolPath, CallingConvention>,
        tables: &'a HashMap<String, OpRef>,
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
        let claimed = if function.root_entry {
            vec![Class::DIRECT; variables.len()]
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
                None => vec![Class::CPS; variables.len()],
            }
        };
        let mut reader = Reader {
            binders: function.binders.clone(),
            requirements,
            params: variables.iter().copied().zip(claimed).collect(),
            locals: Locals::default(),
        };
        if let Some(body) = ctx.op_region(function.op, 0) {
            reader.settle_locals(ctx, body);
        }
        Self {
            ctx,
            function,
            positions,
            tables,
            reader,
        }
    }

    /// The class a row requires at `at` inside this instance.
    fn class(&self, row: &Row, at: OpRef) -> Class {
        self.reader.class(self.ctx, row, at)
    }

    /// The conventions a row may require at `at` inside this instance.
    fn conventions(&self, row: &Row, at: OpRef) -> Vec<CallingConvention> {
        self.reader.conventions(self.ctx, row, at)
    }

    fn check(&self, disagreements: &mut Vec<Disagreement>) {
        self.check_own_convention(disagreements);
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
                .is_some_and(|tail| !self.reader.params.contains_key(&tail))
        {
            row.tail = None;
        }
        let expected = self.class(&row, self.function.op).bound();
        let actual = tribute_cps::declared_convention(self.ctx, self.function.op);
        if Some(expected) != actual {
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
                        .map_or(Class::CPS, |row| self.class(row, reference.op))
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

    /// A lambda has a body for exactly the conventions the rows of the
    /// callables its instantiations make may require, or, without
    /// instantiations, its own row; function references enter their table
    /// no stronger than their row, and indirect transfers have the class of
    /// their callee's row, or its bound.
    fn check_body(&self, disagreements: &mut Vec<Disagreement>) {
        let Some(body) = self.ctx.op_region(self.function.op, 0) else {
            return;
        };
        let mut lambdas = Vec::new();
        let mut selections: HashMap<OpRef, Vec<CallingConvention>> = HashMap::default();
        let _ = walk_region::<()>(self.ctx, body, &mut |op| {
            if tribute_cps::Lambda::matches(self.ctx, op) {
                lambdas.push(op);
            } else if tribute_cps::FuncRef::matches(self.ctx, op) {
                self.check_func_ref(op, disagreements);
            } else if tribute_cps::CallIndirect::matches(self.ctx, op)
                || tribute_cps::TailCallIndirect::matches(self.ctx, op)
            {
                let callee = self.ctx.value_ty(self.ctx.op_operands(op)[0]);
                self.compare(op, "transfer", callee, disagreements);
            } else if let Some((lambda, _, _)) = instantiation(self.ctx, op)
                && let Some(made) =
                    FuncSig::from_type_ref(self.ctx, self.ctx.op_result_types(op)[0])
            {
                selections
                    .entry(lambda)
                    .or_default()
                    .extend(self.conventions(&made.row(self.ctx), op));
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        for lambda in lambdas {
            let expected: Vec<_> = match selections.get(&lambda) {
                Some(selected) => selected.iter().copied().sorted().dedup().collect(),
                None => FuncSig::from_type_ref(self.ctx, self.ctx.op_result_types(lambda)[0])
                    .map(|signature| self.conventions(&signature.row(self.ctx), lambda))
                    .unwrap_or_default(),
            };
            let actual = tribute_cps::body_conventions(self.ctx, lambda);
            if actual.as_ref() != Some(&expected) {
                disagreements.push(Disagreement {
                    op: lambda,
                    rule: "lambda",
                    message: format!(
                        "a lambda in {} has bodies {actual:?}, its uses select {expected:?}",
                        self.function.name
                    ),
                });
            }
        }
    }

    /// Whether `op` declares the class of the row of `callable`.
    /// A function reference's table has an entry no stronger than the class
    /// of the row of the callable it makes, so every place the value reaches
    /// enters a target or an adapter to one.
    fn check_func_ref(&self, op: OpRef, disagreements: &mut Vec<Disagreement>) {
        let Some(signature) = FuncSig::from_type_ref(self.ctx, self.ctx.op_result_types(op)[0])
        else {
            return;
        };
        let expected = self.conventions(&signature.row(self.ctx), op)[0];
        let weakest = tribute_cps::FuncRef::from_op(self.ctx, op)
            .ok()
            .and_then(|reference| reference.func_ref(self.ctx).as_simple().cloned())
            .and_then(|name| self.tables.get(name.as_str()))
            .and_then(|&table| tribute_cps::table_entries(self.ctx, table))
            .and_then(|entries| entries.first().map(|&(convention, _)| convention));
        if weakest.is_none_or(|weakest| weakest > expected) {
            disagreements.push(Disagreement {
                op,
                rule: "func_ref",
                message: format!(
                    "a function reference in {} enters its table at {weakest:?}, its row implies {expected:?}",
                    self.function.name
                ),
            });
        }
    }

    fn compare(
        &self,
        op: OpRef,
        rule: &'static str,
        callable: TypeRef,
        disagreements: &mut Vec<Disagreement>,
    ) {
        let Some(signature) = FuncSig::from_type_ref(self.ctx, callable) else {
            return;
        };
        let expected = self.class(&signature.row(self.ctx), op).bound();
        let actual = tribute_cps::declared_convention(self.ctx, op);
        if actual != Some(expected) {
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

#[cfg(test)]
mod tests;

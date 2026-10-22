//! Check that the convention classes class specialization chose agree with
//! the classes the rows of source-logical IR imply.
//!
//! The rules are those of `new-plans/cps-effects.md` ("Row 변수의 convention
//! class"), read from the IR alone: row binders, `row_args` of references and
//! instantiations, and the operations each ability instance declares.

use std::ops::ControlFlow;

use itertools::Itertools;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::calling_convention::CallingConvention;
use tribute_ir::dialect::tribute_control;
use tribute_ir::dialect::tribute_control::{FuncSig, Row};
use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::walk::{WalkAction, walk_region};

use crate::convention_classes::{
    Function, class_suffix, instantiation, requirements, row_class, settle_class_variables,
    settle_locals, split_name,
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

    /// The class of each local binder.
    fn settle_locals(&mut self) {
        if let Some(body) = self.ctx.op_region(self.function.op, 0) {
            self.locals = settle_locals(
                self.ctx,
                body,
                &self.function.binders,
                self.requirements,
                &self.classes,
            );
        }
    }

    /// The lambda `op` instantiates, its quantified binders, and the rows.
    fn instantiation(&self, op: OpRef) -> Option<(OpRef, Vec<u32>, Vec<Row>)> {
        instantiation(self.ctx, op)
    }

    /// The class a row requires inside this instance.
    fn class(&self, row: &Row) -> CallingConvention {
        row_class(
            row,
            &self.function.binders,
            self.requirements,
            &self.classes,
            &self.locals,
        )
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

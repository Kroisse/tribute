//! Check that the convention classes class specialization chose agree with
//! the classes the rows of source-logical IR imply.
//!
//! The rules are those of `design/cps-effects.md` ("Row 변수의 convention
//! class"), read from the IR alone: row binders, `row_args` of references and
//! instantiations, and the operations each ability instance declares.

use std::ops::ControlFlow;

use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use tribute_ir::calling_convention::CallingConvention;
use tribute_ir::dialect::tribute_control::{self, FuncSig, Row};
use tribute_ir::dialect::tribute_cps;
use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::walk::{WalkAction, walk_region};

use crate::convention_classes::{
    Class, Entry, Function, Reader, func_tables, instance_key_name, instance_name, instantiation,
    key_references, requirements, settle_class_variables, split_name,
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

/// An instance declares `cps_entry` exactly when its `Cps` entry instance
/// exists, and names that instance (`tribute_cps.func`) as a definition of
/// its own module: by its name alone.
fn check_cps_entries(
    ctx: &IrContext,
    functions: &[Function],
    disagreements: &mut Vec<Disagreement>,
) {
    for function in functions {
        let expected = functions
            .iter()
            .find(|entry| {
                entry.entry == Entry::Cps
                    && instance_key_name(ctx, entry.op) == Some(function.name.as_str())
            })
            .filter(|_| function.entry == Entry::Row)
            .map(|entry| entry.name.clone());
        let declared = ctx
            .op_attrs(function.op)
            .get_symbol_ref(tribute_cps::CPS_ENTRY_ATTR);
        let agrees = match (declared, &expected) {
            (None, None) => true,
            (Some(declared), Some(expected)) => *declared == [expected.as_str()],
            _ => false,
        };
        if !agrees {
            disagreements.push(Disagreement {
                op: function.op,
                rule: "instance",
                message: format!(
                    "{} declares cps_entry {}, its Cps entry instance is {expected:?}",
                    function.name,
                    declared.map_or("none".to_owned(), ToString::to_string),
                ),
            });
        }
    }
}

/// Every disagreement in `module`.
pub fn check(ctx: &IrContext, module: Module) -> Vec<Disagreement> {
    let mut disagreements = Vec::new();
    let requirements = requirements(ctx, module);
    let tables = func_tables(ctx, module);
    let mut functions: Vec<Function> = module
        .ops(ctx)
        .iter()
        .filter_map(|&op| Function::read(ctx, op, &tables))
        .collect();
    key_references(&mut functions, ctx);
    let positions = settle_class_variables(&mut functions);
    let mut definitions: HashMap<&str, &Function> = HashMap::default();
    for function in &functions {
        definitions
            .entry(function.base.as_str())
            .or_insert(function);
    }
    let instances: HashMap<&str, OpRef> = functions
        .iter()
        .map(|function| (function.name.as_str(), function.op))
        .collect();
    check_cps_entries(ctx, &functions, &mut disagreements);
    let module = Definitions {
        positions: &positions,
        requirements: &requirements,
        tables: &tables,
        definitions: &definitions,
        instances: &instances,
        hands_frames: tribute_cps::hands_frames_to_values(ctx, module.op()),
    };
    for function in &functions {
        Instance::new(ctx, function, module, &mut disagreements).check(&mut disagreements);
    }
    check_cps_targets(ctx, &functions, module, &mut disagreements);
    disagreements
}

/// In a module that hands frames to callable values, a function table of a
/// definition with a body has a `cps` target, the `Cps` entry instance of
/// its strongest target when no other target is `Cps`
/// (`design/tribute-dialects.md`, "Convention entry", 함수 참조). Removing
/// unreferenced definitions after class specialization may remove the only
/// such transfer, so a module that hands none may still have one.
fn check_cps_targets(
    ctx: &IrContext,
    functions: &[Function],
    module: Definitions<'_>,
    disagreements: &mut Vec<Disagreement>,
) {
    if !module.hands_frames {
        return;
    }
    let by_name: HashMap<&str, &Function> = functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect();
    for (name, &table) in module
        .tables
        .iter()
        .sorted_by_key(|(name, _)| name.as_str())
    {
        let Some(entries) = tribute_cps::table_entries(ctx, table) else {
            continue;
        };
        let has_body = entries.iter().any(|(_, target)| {
            target
                .leaf()
                .and_then(|leaf| by_name.get(leaf.as_str()))
                .is_some_and(|function| ctx.op_has_regions(function.op))
        });
        let has_cps = entries
            .iter()
            .any(|&(convention, _)| convention == CallingConvention::Cps);
        if has_body && !has_cps {
            disagreements.push(Disagreement {
                op: table,
                rule: "table",
                message: format!(
                    "function table {name} of a definition with a body has no cps target"
                ),
            });
        }
    }
}

/// What the instances of a module share: the class variables and an
/// instance of each definition, the requirements of its abilities, and its
/// function tables.
#[derive(Clone, Copy)]
struct Definitions<'a> {
    positions: &'a HashMap<String, Vec<u32>>,
    requirements: &'a HashMap<SymbolPath, CallingConvention>,
    /// The module's function tables, by name.
    tables: &'a HashMap<String, OpRef>,
    /// An instance of each definition, by its base name.
    definitions: &'a HashMap<&'a str, &'a Function>,
    /// Every instance, by its name.
    instances: &'a HashMap<&'a str, OpRef>,
    /// Whether the module hands frames to callable values.
    hands_frames: bool,
}

/// One function instance with the classes its name fixes.
struct Instance<'a> {
    ctx: &'a IrContext,
    function: &'a Function,
    module: Definitions<'a>,
    /// The classes of its class variables.
    classes: Vec<Class>,
    reader: Reader<'a>,
}

impl<'a> Instance<'a> {
    fn new(
        ctx: &'a IrContext,
        function: &'a Function,
        module: Definitions<'a>,
        disagreements: &mut Vec<Disagreement>,
    ) -> Self {
        let positions = module.positions;
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
        let mut reader = function.reader(
            module.requirements,
            variables
                .iter()
                .copied()
                .zip(claimed.iter().copied())
                .collect(),
        );
        if let Some(body) = ctx.op_region(function.op, 0) {
            reader.settle_locals(ctx, body);
        }
        Self {
            ctx,
            function,
            module,
            classes: claimed,
            reader,
        }
    }

    /// The convention of the flow that runs `op` in this instance.
    fn flow(&self, op: OpRef) -> Option<CallingConvention> {
        tribute_cps::flow_convention(self.ctx, op)
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

    /// A definition's own convention is the class of its own row, and a
    /// `Cps` entry instance's is `Cps` while its row's is weaker
    /// (`design/cps-effects.md`, "`Cps` 진입 본문").
    fn check_own_convention(&self, disagreements: &mut Vec<Disagreement>) {
        let row = self.function.own_row(self.ctx, self.function.op, |tail| {
            self.reader.params.contains_key(&tail)
        });
        let implied = self.class(&row, self.function.op).bound();
        let actual = tribute_cps::declared_convention(self.ctx, self.function.op);
        let (expected, message) = match self.function.entry {
            Entry::Row => (implied, "its row implies"),
            Entry::Cps => (CallingConvention::Cps, "as a Cps entry instance it needs"),
        };
        if Some(expected) != actual {
            disagreements.push(Disagreement {
                op: self.function.op,
                rule: "own",
                message: format!(
                    "{} has convention {actual:?}, {message} {expected:?}",
                    self.function.name
                ),
            });
        }
        if self.function.entry == Entry::Cps {
            let name = instance_name(&self.function.base, &self.classes, Entry::Cps);
            if implied == CallingConvention::Cps || name != self.function.name {
                disagreements.push(Disagreement {
                    op: self.function.op,
                    rule: "entry",
                    message: format!(
                        "{} is a Cps entry instance named {name} whose row implies {implied:?}",
                        self.function.name
                    ),
                });
            }
        }
    }

    /// A reference names the instance of its callee for the classes of the
    /// rows it passes for the callee's class variables. A `become` in a `Cps`
    /// flow and a call that hands a handle body's frame on name the callee's
    /// `Cps` entry instance when the callee has a body and its row implies a
    /// weaker convention.
    fn check_references(&self, disagreements: &mut Vec<Disagreement>) {
        for reference in &self.function.references {
            let (callee, _) = split_name(&reference.key);
            let positions = self.module.positions.get(callee);
            let classes: Vec<_> = positions
                .into_iter()
                .flatten()
                .map(|&position| {
                    usize::try_from(position)
                        .ok()
                        .and_then(|position| reference.rows.get(position))
                        .map_or(Class::CPS, |row| self.class(row, reference.op))
                })
                .collect();
            let cps_flow = (tribute_control::TailCall::matches(self.ctx, reference.op)
                && self.flow(reference.op) == Some(CallingConvention::Cps))
                || tribute_cps::passes_body_frame_on(self.ctx, reference.op);
            // Whether the callee has a `Cps` entry instance: it has a body
            // and its row determines a weaker convention.
            let has_cps_entry = || match self.module.definitions.get(callee) {
                Some(definition) if self.ctx.op_has_regions(definition.op) => {
                    let params: HashMap<u32, Class> = positions
                        .into_iter()
                        .flatten()
                        .copied()
                        .zip(classes.iter().copied())
                        .collect();
                    let row = definition
                        .own_row(self.ctx, definition.op, |tail| params.contains_key(&tail));
                    let reader = definition.reader(self.module.requirements, params);
                    reader.class(self.ctx, &row, definition.op).bound() != CallingConvention::Cps
                }
                _ => false,
            };
            let entry = if cps_flow && has_cps_entry() {
                Entry::Cps
            } else {
                Entry::Row
            };
            let expected = instance_name(callee, &classes, entry);
            // A tail call in a body that also runs as its lambda's `Cps`
            // entry enters its callee's `cps_entry` there, so the callee
            // declares one.
            if !cps_flow
                && tribute_control::TailCall::matches(self.ctx, reference.op)
                && tribute_cps::in_cps_entry_flow(self.ctx, reference.op)
                && has_cps_entry()
                && self
                    .module
                    .instances
                    .get(expected.as_str())
                    .is_some_and(|&instance| {
                        !self
                            .ctx
                            .op_attrs(instance)
                            .contains_key(tribute_cps::CPS_ENTRY_ATTR)
                    })
            {
                disagreements.push(Disagreement {
                    op: reference.op,
                    rule: "reference",
                    message: format!(
                        "{} enters {expected} from a Cps entry, which declares no cps_entry",
                        self.function.name
                    ),
                });
            }
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
    /// instantiations, its own row, and declares `cps_entry` when its module
    /// hands frames to callable values and its own flow hands a frame on,
    /// which it may declare elsewhere too;
    /// function references enter their table
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
            let selected: Vec<_> = match selections.get(&lambda) {
                Some(selected) => selected.clone(),
                None => FuncSig::from_type_ref(self.ctx, self.ctx.op_result_types(lambda)[0])
                    .map(|signature| self.conventions(&signature.row(self.ctx), lambda))
                    .unwrap_or_default(),
            };
            let expected: Vec<_> = selected.into_iter().sorted().dedup().collect();
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
            // In a module that hands frames to callable values, a lambda
            // without a `Cps` body declares `cps_entry` exactly when the own
            // flow of its strongest body hands a frame on. Elsewhere such a
            // lambda may declare it: removing unreferenced definitions after
            // class specialization may remove the only such transfer.
            let hands_on = !expected.contains(&CallingConvention::Cps)
                && tribute_cps::strongest_body_hands_on_frame(self.ctx, lambda);
            let declared = tribute_cps::has_cps_entry(self.ctx, lambda);
            if declared != hands_on && (declared || self.module.hands_frames) {
                disagreements.push(Disagreement {
                    op: lambda,
                    rule: "lambda",
                    message: format!(
                        "a lambda in {} declares cps_entry = {declared}, its own flow gives {hands_on}",
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
            .and_then(|reference| reference.value(self.ctx).as_simple().cloned())
            .and_then(|name| self.module.tables.get(name.as_str()))
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
        // A `become` in a `Cps` flow and a call that hands a handle body's
        // frame on enter the value's `Cps` entry.
        let expected = if (tribute_cps::TailCallIndirect::matches(self.ctx, op)
            && self.flow(op) == Some(CallingConvention::Cps))
            || tribute_cps::passes_body_frame_on(self.ctx, op)
        {
            CallingConvention::Cps
        } else {
            self.class(&signature.row(self.ctx), op).bound()
        };
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

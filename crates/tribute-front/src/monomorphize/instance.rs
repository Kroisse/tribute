//! Identity of a specialized function instance: its type arguments.

use std::convert::Infallible;
use std::num::NonZero;
use std::ops::ControlFlow;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use smallvec::SmallVec;

use crate::ast::fold::{TypeVisitable, TypeVisitor};
use crate::ast::{
    AbilityRow, ExprKind, FuncDecl, NodeId, RowVar, Stmt, Type, TypedRef, collect_row_vars,
};
use crate::typeck::{FunctionInstance, LambdaSignature, LocalCallableInstance};

/// The arguments that select one instance of a function definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InstanceKey<'db> {
    pub(crate) type_args: SmallVec<[Type<'db>; 2]>,
}

impl<'db> InstanceKey<'db> {
    #[cfg(test)]
    pub(crate) fn of_types(type_args: impl IntoIterator<Item = Type<'db>>) -> Self {
        Self {
            type_args: type_args.into_iter().collect(),
        }
    }

    /// The key of the reference `instance`, or `None` when it selects the
    /// definition itself.
    pub(crate) fn of(instance: &FunctionInstance<'db>) -> Option<Self> {
        (!instance.type_arguments.is_empty()).then(|| Self {
            type_args: SmallVec::from_slice(&instance.type_arguments),
        })
    }

    /// The NodeId variant of this instance's clone.
    pub(crate) fn variant(&self) -> NonZero<u64> {
        super::specialize::type_args_variant(&self.type_args)
    }
}

/// One use of a let-bound lambda: the row it passes for each row variable
/// the lambda's type spells, in the order its scheme quantifies them.
pub(crate) struct LocalInstantiation<'db> {
    pub(crate) site: NodeId,
    pub(crate) binding: NodeId,
    pub(crate) rows: Vec<(RowVar, AbilityRow<'db>)>,
}

/// Each use of a let-bound lambda in `function`, in use order. A use whose
/// lambda does not spell every row its scheme quantifies passes none.
pub(crate) fn local_instantiations<'a, 'db: 'a>(
    db: &'db dyn salsa::Database,
    function: &FuncDecl<TypedRef<'db>>,
    local_instance: impl Fn(&NodeId) -> Option<&'a LocalCallableInstance<'db>>,
    lambda_signature: impl Fn(&NodeId) -> Option<&'a LambdaSignature<'db>>,
) -> Vec<LocalInstantiation<'db>> {
    // The lambda each `let` binds, by its pattern.
    let mut bound = HashMap::default();
    function.body.for_each(|expr| {
        let ExprKind::Block { stmts, .. } = &*expr.kind else {
            return;
        };
        for stmt in stmts {
            if let Stmt::Let { pattern, value, .. } = stmt
                && matches!(&*value.kind, ExprKind::Lambda { .. })
            {
                bound.insert(pattern.id, value.id);
            }
        }
    });
    let mut instantiations = Vec::new();
    function.body.for_each(|expr| {
        let Some(instance) = local_instance(&expr.id) else {
            return;
        };
        let Some(signature) = bound.get(&instance.binding).and_then(&lambda_signature) else {
            return;
        };
        // The lambda's own type spells the rows its scheme quantifies with
        // the variables its body uses.
        let quantified = collect_row_vars(db, instance.scheme.body(db));
        let spelled = collect_row_vars(db, signature.function_type);
        if quantified.len() != spelled.len() {
            return;
        }
        let variables = instance.scheme.effect_params(db);
        let rows = variables
            .iter()
            .zip(&instance.row_arguments)
            .filter_map(|(variable, row)| {
                let at = quantified.iter().position(|var| var == variable)?;
                Some((spelled[at], *row))
            })
            .collect();
        instantiations.push(LocalInstantiation {
            site: expr.id,
            binding: instance.binding,
            rows,
        });
    });
    instantiations
}

/// Collect the row variables that occur inside an ability argument of a row
/// anywhere in `ty`.
pub(crate) fn collect_ability_argument_tails<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    excluded: &mut HashSet<RowVar>,
) {
    struct ArgumentTails<'a, 'db> {
        db: &'db dyn salsa::Database,
        excluded: &'a mut HashSet<RowVar>,
    }

    impl<'db> TypeVisitor<'db> for ArgumentTails<'_, 'db> {
        type Break = Infallible;

        fn db(&self) -> &'db dyn salsa::Database {
            self.db
        }

        fn visit_row(&mut self, row: AbilityRow<'db>) -> ControlFlow<Infallible> {
            for ability in row.instances(self.db) {
                for arg in &ability.args {
                    self.excluded.extend(collect_row_vars(self.db, *arg));
                }
            }
            ControlFlow::Continue(())
        }
    }

    let ControlFlow::Continue(()) = ty.visit_with(&mut ArgumentTails { db, excluded });
}

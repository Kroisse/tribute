//! The row binders of a definition: every row variable its types name, in
//! the order the IR numbers them.

use rustc_hash::FxHashSet as HashSet;
use tribute_ir::dialect::tribute_control::RowBinder;

use crate::SortedMap;
use crate::ast::visit::Visit;
use crate::ast::{FuncDecl, NodeId, RowVar, Type, TypeScheme, TypedRef, collect_row_vars};
use crate::monomorphize::collect_ability_argument_tails;
use crate::typeck::{InstantiatedHandlerOperation, InstantiatedPerformOperation};

pub(crate) type RowBinders = Vec<(RowVar, RowBinder)>;

/// What a definition's body tells about its row variables.
pub(super) struct BodyTypes<'a, 'db> {
    pub node_types: &'a SortedMap<NodeId, Type<'db>>,
    pub handler_operations: &'a SortedMap<NodeId, InstantiatedHandlerOperation<'db>>,
    pub perform_operations: &'a SortedMap<NodeId, InstantiatedPerformOperation<'db>>,
}

/// The binders of a declaration without a body: those its scheme quantifies.
pub(super) fn of_scheme(db: &dyn salsa::Database, scheme: TypeScheme<'_>) -> RowBinders {
    let mut in_arguments = HashSet::default();
    collect_ability_argument_tails(db, scheme.body(db), &mut in_arguments);
    quantified(db, scheme, &in_arguments)
}

fn quantified(
    db: &dyn salsa::Database,
    scheme: TypeScheme<'_>,
    in_arguments: &HashSet<RowVar>,
) -> RowBinders {
    scheme
        .effect_params(db)
        .iter()
        .map(|variable| {
            let kind = if in_arguments.contains(variable) {
                RowBinder::AbilityArg
            } else {
                RowBinder::Param
            };
            (*variable, kind)
        })
        .collect()
}

/// The binders of a function definition: those its scheme quantifies, then
/// the other row variables of its body in the order they first occur. One
/// of `local`, the variables let-bound lambdas quantify, is local.
pub(super) fn of_definition<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    function: &FuncDecl<TypedRef<'db>>,
    body: &BodyTypes<'_, 'db>,
    local: &[RowVar],
) -> RowBinders {
    struct Collect<'a, 'b, 'db> {
        db: &'db dyn salsa::Database,
        body: &'a BodyTypes<'b, 'db>,
        in_arguments: HashSet<RowVar>,
        local: HashSet<RowVar>,
        seen: Vec<RowVar>,
    }
    impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for Collect<'_, '_, 'db> {
        fn visit_node_id(&mut self, id: NodeId) {
            let handler = self.body.handler_operations.get(&id);
            let perform = self.body.perform_operations.get(&id);
            for argument in handler
                .map(|operation| &operation.ability_args)
                .into_iter()
                .chain(perform.map(|operation| &operation.ability_args))
                .flatten()
            {
                self.in_arguments
                    .extend(collect_row_vars(self.db, *argument));
            }
            if let Some(ty) = self.body.node_types.get(&id) {
                collect_ability_argument_tails(self.db, *ty, &mut self.in_arguments);
                for variable in collect_row_vars(self.db, *ty) {
                    if !self.seen.contains(&variable) {
                        self.seen.push(variable);
                    }
                }
            }
        }
    }
    let mut collect = Collect {
        db,
        body,
        in_arguments: HashSet::default(),
        local: local.iter().copied().collect(),
        seen: collect_row_vars(db, scheme.body(db)),
    };
    collect.visit_func_decl(function);
    collect_ability_argument_tails(db, scheme.body(db), &mut collect.in_arguments);

    let mut binders = quantified(db, scheme, &collect.in_arguments);
    for &variable in collect.seen.iter().chain(local) {
        if binders.iter().any(|(bound, _)| *bound == variable) {
            continue;
        }
        let kind = if collect.local.contains(&variable) {
            RowBinder::Local
        } else {
            RowBinder::Body
        };
        binders.push((variable, kind));
    }
    binders
}

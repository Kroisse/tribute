//! Type traversal of the metadata type checking records.
//!
//! Each table's values implement [`TypeVisitable`] and [`TypeFoldable`], and
//! the metadata structs visit, fold, and merge every table. Each of these
//! destructures its value, so a new field or table does not compile until
//! both traversals handle it.

use std::ops::ControlFlow;

use crate::SortedMap;
use crate::ast::fold::{TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor};
use crate::ast::{AbilityInstance, NodeId};

use super::{
    ConstructorTypeMetadata, EvidenceStep, ExpressionTypeMetadata, FunctionInstance,
    InstantiatedHandlerOperation, InstantiatedPerformOperation, LambdaSignature,
    LocalCallableInstance,
};

/// The caller-side types of the instance. Its `scheme` is the callee's
/// declaration under the callee's own binders, so a visit or fold of the
/// caller's types leaves it.
impl<'db> TypeVisitable<'db> for FunctionInstance<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        let Self {
            origin: _,
            function: _,
            scheme: _,
            type_arguments,
            row_arguments,
            callable,
        } = self;
        callable.visit_with(visitor)?;
        type_arguments.visit_with(visitor)?;
        row_arguments.visit_with(visitor)
    }
}

impl<'db> TypeFoldable<'db> for FunctionInstance<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        let Self {
            origin,
            function,
            scheme,
            type_arguments,
            row_arguments,
            callable,
        } = self;
        Ok(Self {
            origin,
            function,
            scheme,
            callable: callable.try_fold_with(folder)?,
            type_arguments: type_arguments.try_fold_with(folder)?,
            row_arguments: row_arguments.try_fold_with(folder)?,
        })
    }
}

/// A local scheme is written under the enclosing function's binders, so it
/// is visited and folded with the instance's other types.
impl<'db> TypeVisitable<'db> for LocalCallableInstance<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        let Self {
            binding: _,
            local: _,
            scheme,
            row_arguments,
            callable,
        } = self;
        scheme.visit_with(visitor)?;
        callable.visit_with(visitor)?;
        row_arguments.visit_with(visitor)
    }
}

impl<'db> TypeFoldable<'db> for LocalCallableInstance<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        let Self {
            binding,
            local,
            scheme,
            row_arguments,
            callable,
        } = self;
        Ok(Self {
            binding,
            local,
            scheme: scheme.try_fold_with(folder)?,
            callable: callable.try_fold_with(folder)?,
            row_arguments: row_arguments.try_fold_with(folder)?,
        })
    }
}

impl<'db> TypeVisitable<'db> for InstantiatedHandlerOperation<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        let Self {
            ability: _,
            ability_args,
            kind: _,
            params,
            result,
        } = self;
        ability_args.visit_with(visitor)?;
        params.visit_with(visitor)?;
        result.visit_with(visitor)
    }
}

impl<'db> TypeFoldable<'db> for InstantiatedHandlerOperation<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        let Self {
            ability,
            ability_args,
            kind,
            params,
            result,
        } = self;
        Ok(Self {
            ability,
            ability_args: ability_args.try_fold_with(folder)?,
            kind,
            params: params.try_fold_with(folder)?,
            result: result.try_fold_with(folder)?,
        })
    }
}

impl<'db> TypeVisitable<'db> for InstantiatedPerformOperation<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        let Self {
            ability: _,
            ability_args,
            kind: _,
            params,
            result,
        } = self;
        ability_args.visit_with(visitor)?;
        params.visit_with(visitor)?;
        result.visit_with(visitor)
    }
}

impl<'db> TypeFoldable<'db> for InstantiatedPerformOperation<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        let Self {
            ability,
            ability_args,
            kind,
            params,
            result,
        } = self;
        Ok(Self {
            ability,
            ability_args: ability_args.try_fold_with(folder)?,
            kind,
            params: params.try_fold_with(folder)?,
            result: result.try_fold_with(folder)?,
        })
    }
}

impl<'db> TypeVisitable<'db> for LambdaSignature<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        let Self { function_type } = self;
        function_type.visit_with(visitor)
    }
}

impl<'db> TypeFoldable<'db> for LambdaSignature<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(Self {
            function_type: self.function_type.try_fold_with(folder)?,
        })
    }
}

/// The type arguments of each ability instance the step names.
impl<'db> TypeVisitable<'db> for EvidenceStep<'db> {
    fn visit_with<V: TypeVisitor<'db> + ?Sized>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        match self {
            Self::Mask(instance) | Self::Dup(instance) | Self::Push(instance) => {
                AbilityInstance::visit_with(instance, visitor)
            }
            Self::Select(_) => ControlFlow::Continue(()),
            Self::Tails(plans) => plans.visit_with(visitor),
        }
    }
}

impl<'db> TypeFoldable<'db> for EvidenceStep<'db> {
    fn try_fold_with<F: TypeFolder<'db> + ?Sized>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(match self {
            Self::Mask(instance) => Self::Mask(instance.try_fold_with(folder)?),
            Self::Dup(instance) => Self::Dup(instance.try_fold_with(folder)?),
            Self::Push(instance) => Self::Push(instance.try_fold_with(folder)?),
            Self::Select(index) => Self::Select(index),
            Self::Tails(plans) => Self::Tails(plans.try_fold_with(folder)?),
        })
    }
}

fn visit_values<'db, T: TypeVisitable<'db>, V: TypeVisitor<'db> + ?Sized>(
    table: &SortedMap<NodeId, T>,
    visitor: &mut V,
) -> ControlFlow<V::Break> {
    table
        .values()
        .try_for_each(|value| value.visit_with(visitor))
}

/// Fold each value of `table` in place, moving it out rather than copying
/// it. A failed fold leaves the table empty.
fn fold_values<'db, T: TypeFoldable<'db>, F: TypeFolder<'db> + ?Sized>(
    table: &mut SortedMap<NodeId, T>,
    folder: &mut F,
) -> Result<(), F::Error> {
    *table = std::mem::take(table).try_map_values(|value| value.try_fold_with(folder))?;
    Ok(())
}

impl<'db> ExpressionTypeMetadata<'db> {
    /// Visit the types of every table, table by table.
    pub fn visit_types<V: TypeVisitor<'db> + ?Sized>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::Break> {
        let Self {
            node_types,
            function_instances,
            local_instances,
            evidence_plans,
            handler_operations,
            perform_operations,
            lambda_signatures,
            exhaustive_cases: _,
        } = self;
        visit_values(node_types, visitor)?;
        visit_values(function_instances, visitor)?;
        visit_values(local_instances, visitor)?;
        visit_values(evidence_plans, visitor)?;
        visit_values(handler_operations, visitor)?;
        visit_values(perform_operations, visitor)?;
        visit_values(lambda_signatures, visitor)
    }

    /// Fold the types of every table in place, keeping the keys.
    pub fn try_fold_types<F: TypeFolder<'db> + ?Sized>(
        &mut self,
        folder: &mut F,
    ) -> Result<(), F::Error> {
        let Self {
            node_types,
            function_instances,
            local_instances,
            evidence_plans,
            handler_operations,
            perform_operations,
            lambda_signatures,
            exhaustive_cases: _,
        } = self;
        fold_values(node_types, folder)?;
        fold_values(function_instances, folder)?;
        fold_values(local_instances, folder)?;
        fold_values(evidence_plans, folder)?;
        fold_values(handler_operations, folder)?;
        fold_values(perform_operations, folder)?;
        fold_values(lambda_signatures, folder)
    }

    /// Add the entries of `others` to each table, one merge per table. An
    /// added entry replaces one at the same node.
    pub fn extend(&mut self, others: impl IntoIterator<Item = Self>) {
        let mut others: Vec<Self> = others.into_iter().collect();
        let Self {
            node_types,
            function_instances,
            local_instances,
            evidence_plans,
            handler_operations,
            perform_operations,
            lambda_signatures,
            exhaustive_cases,
        } = self;
        node_types.extend(take_entries(&mut others, |other| &mut other.node_types));
        function_instances.extend(take_entries(&mut others, |other| {
            &mut other.function_instances
        }));
        local_instances.extend(take_entries(&mut others, |other| {
            &mut other.local_instances
        }));
        evidence_plans.extend(take_entries(&mut others, |other| &mut other.evidence_plans));
        handler_operations.extend(take_entries(&mut others, |other| {
            &mut other.handler_operations
        }));
        perform_operations.extend(take_entries(&mut others, |other| {
            &mut other.perform_operations
        }));
        lambda_signatures.extend(take_entries(&mut others, |other| {
            &mut other.lambda_signatures
        }));
        let added: Vec<_> = others
            .iter_mut()
            .flat_map(|other| std::mem::take(&mut other.exhaustive_cases))
            .collect();
        if !added.is_empty() {
            exhaustive_cases.extend(added);
            exhaustive_cases.sort();
            exhaustive_cases.dedup();
        }
    }

    /// Keep only the entries of the nodes `keep` accepts, in every table.
    pub fn retain_nodes(&mut self, keep: impl Fn(&NodeId) -> bool) {
        let Self {
            node_types,
            function_instances,
            local_instances,
            evidence_plans,
            handler_operations,
            perform_operations,
            lambda_signatures,
            exhaustive_cases,
        } = self;
        node_types.retain(|node, _| keep(node));
        function_instances.retain(|node, _| keep(node));
        local_instances.retain(|node, _| keep(node));
        evidence_plans.retain(|node, _| keep(node));
        handler_operations.retain(|node, _| keep(node));
        perform_operations.retain(|node, _| keep(node));
        lambda_signatures.retain(|node, _| keep(node));
        exhaustive_cases.retain(|node| keep(node));
    }
}

/// Move the entries of one table out of each of `others`.
fn take_entries<'a, 'db: 'a, V: 'a>(
    others: &'a mut [ExpressionTypeMetadata<'db>],
    table: for<'b> fn(&'b mut ExpressionTypeMetadata<'db>) -> &'b mut SortedMap<NodeId, V>,
) -> impl Iterator<Item = (NodeId, V)> + 'a {
    others
        .iter_mut()
        .flat_map(move |other| std::mem::take(table(other)).into_vec())
}

impl<'db> ConstructorTypeMetadata<'db> {
    /// Visit the constructor schemes, then the specialized variant schemes.
    pub fn visit_types<V: TypeVisitor<'db> + ?Sized>(
        &self,
        visitor: &mut V,
    ) -> ControlFlow<V::Break> {
        let Self {
            schemes,
            specialized_enum_variants,
        } = self;
        schemes
            .iter()
            .try_for_each(|(_, scheme)| scheme.visit_with(visitor))?;
        visit_values(specialized_enum_variants, visitor)
    }

    /// Fold the constructor and specialized variant schemes in place.
    pub fn try_fold_types<F: TypeFolder<'db> + ?Sized>(
        &mut self,
        folder: &mut F,
    ) -> Result<(), F::Error> {
        let Self {
            schemes,
            specialized_enum_variants,
        } = self;
        for (_, scheme) in schemes {
            *scheme = scheme.try_fold_with(folder)?;
        }
        fold_values(specialized_enum_variants, folder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityRow, FuncDefId, Type, TypeKind, TypeScheme};
    use crate::typeck::FunctionInstanceOrigin;
    use crate::typeck::subst::BoundVarSubst;

    /// A fold substitutes the caller-side types of a function instance but
    /// leaves its scheme, which is written under the callee's binders.
    #[test]
    fn function_instance_folds_caller_types_but_not_the_callee_scheme() {
        let db = salsa::DatabaseImpl::new();
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let int = Type::new(&db, TypeKind::Int);
        let callable = |ty| {
            Type::new(
                &db,
                TypeKind::Func {
                    params: vec![ty],
                    result: ty,
                    effect: AbilityRow::pure(&db),
                },
            )
        };
        let scheme = TypeScheme::mono(&db, callable(bound));
        let node = NodeId::from_raw(1);
        let mut metadata = ExpressionTypeMetadata {
            function_instances: [(
                node,
                FunctionInstance {
                    origin: FunctionInstanceOrigin::Declaration,
                    function: FuncDefId::new(&db, trunk_ir::SymbolPath::new(["id"])),
                    scheme,
                    type_arguments: vec![bound],
                    row_arguments: vec![],
                    callable: callable(bound),
                },
            )]
            .into_iter()
            .collect(),
            ..ExpressionTypeMetadata::default()
        };

        let mut visited = Vec::new();
        let ControlFlow::Continue(()) = metadata.visit_types(&mut crate::ast::fold::ForEachType {
            db: &db,
            visit: |ty| visited.push(ty),
        });
        assert_eq!(visited, [callable(bound), bound]);

        metadata
            .try_fold_types(&mut BoundVarSubst::new(&db, &[int]))
            .unwrap();
        let instance = metadata.function_instances.get(&node).unwrap();
        assert_eq!(instance.callable, callable(int));
        assert_eq!(instance.type_arguments, [int]);
        assert_eq!(instance.scheme, scheme);
    }

    #[test]
    fn extend_adds_entries_and_keeps_exhaustive_cases_sorted() {
        let db = salsa::DatabaseImpl::new();
        let int = Type::new(&db, TypeKind::Int);
        let at = |raw, exhaustive: bool| ExpressionTypeMetadata {
            node_types: [(NodeId::from_raw(raw), int)].into_iter().collect(),
            exhaustive_cases: if exhaustive {
                vec![NodeId::from_raw(raw)]
            } else {
                vec![]
            },
            ..ExpressionTypeMetadata::default()
        };
        let mut metadata = at(2, true);
        metadata.extend([at(3, true), at(1, false), at(2, true)]);
        assert_eq!(
            metadata.node_types.keys().copied().collect::<Vec<_>>(),
            [1, 2, 3].map(NodeId::from_raw)
        );
        assert_eq!(metadata.exhaustive_cases, [2, 3].map(NodeId::from_raw));
    }
}

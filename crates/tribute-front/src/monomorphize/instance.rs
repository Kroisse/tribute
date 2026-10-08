//! Identity of a specialized function instance: its type arguments and the
//! convention class of each class variable.

use std::hash::{Hash, Hasher};
use std::num::NonZero;

use crate::ast::{CallingConvention, EffectVar, Type, TypeKind, TypeScheme};
use crate::typeck::FunctionInstance;

/// The arguments that select one instance of a function definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InstanceKey<'db> {
    pub(crate) type_args: Vec<Type<'db>>,
    /// One class per class variable of the definition, in binder order.
    pub(crate) class_args: Vec<CallingConvention>,
}

impl<'db> InstanceKey<'db> {
    /// The key of the reference `instance` to a definition of `scheme`, or
    /// `None` when it selects the definition itself.
    pub(crate) fn of(
        db: &'db dyn salsa::Database,
        scheme: TypeScheme<'db>,
        instance: &FunctionInstance<'db>,
    ) -> Option<Self> {
        let class_args = vec![CallingConvention::Cps; class_variables(db, scheme).len()];
        let key = Self {
            type_args: instance.type_arguments.clone(),
            class_args,
        };
        (!key.type_args.is_empty() || key.has_weaker_class()).then_some(key)
    }

    #[cfg(test)]
    pub(crate) fn of_types(type_args: Vec<Type<'db>>) -> Self {
        Self {
            type_args,
            class_args: Vec::new(),
        }
    }

    /// Whether some class is weaker than `Cps`, the class of a row variable
    /// that is not specialized.
    pub(crate) fn has_weaker_class(&self) -> bool {
        self.class_args
            .iter()
            .any(|class| *class != CallingConvention::Cps)
    }

    /// The NodeId variant of this instance's clone. Classes that are all
    /// `Cps` do not take part, as in the instance's name.
    pub(crate) fn variant(&self) -> NonZero<u64> {
        if !self.has_weaker_class() {
            return super::specialize::type_args_variant(&self.type_args);
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        NonZero::new(hasher.finish()).unwrap_or(NonZero::<u64>::MIN)
    }
}

/// The class variables of a definition: the row variables `scheme` quantifies
/// that occur as the row of a function type inside a parameter type.
pub(crate) fn class_variables<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
) -> Vec<EffectVar> {
    let TypeKind::Func { params, .. } = scheme.body(db).kind(db) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for param in params {
        collect_callable_tails(db, *param, &mut found);
    }
    scheme
        .effect_params(db)
        .iter()
        .copied()
        .filter(|var| found.contains(var))
        .collect()
}

fn collect_callable_tails<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    found: &mut Vec<EffectVar>,
) {
    match ty.kind(db) {
        TypeKind::Named { args, .. } | TypeKind::Tuple(args) | TypeKind::App { args, .. } => {
            for arg in args {
                collect_callable_tails(db, *arg, found);
            }
        }
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            for param in params {
                collect_callable_tails(db, *param, found);
            }
            collect_callable_tails(db, *result, found);
            found.extend(effect.rest(db));
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::EffectRow;

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    fn func<'db>(
        db: &'db TestDb,
        params: Vec<Type<'db>>,
        result: Type<'db>,
        tail: Option<u64>,
    ) -> Type<'db> {
        let effect = match tail {
            Some(id) => EffectRow::open(db, EffectVar { id }),
            None => EffectRow::pure(db),
        };
        Type::new(
            db,
            TypeKind::Func {
                params,
                result,
                effect,
                minimum_convention: CallingConvention::Direct,
            },
        )
    }

    fn scheme<'db>(db: &'db TestDb, row_vars: &[u64], body: Type<'db>) -> TypeScheme<'db> {
        let effect_params = row_vars.iter().map(|id| EffectVar { id: *id }).collect();
        TypeScheme::new(db, Vec::new(), effect_params, body)
    }

    #[test]
    fn a_tail_of_a_callable_parameter_is_a_class_variable() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let callback = func(&db, vec![int], int, Some(1));
        let map = scheme(&db, &[1], func(&db, vec![int, callback], int, Some(1)));

        assert_eq!(class_variables(&db, map), [EffectVar { id: 1 }]);
    }

    #[test]
    fn class_variables_follow_binder_order() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let first = func(&db, vec![], int, Some(2));
        let second = func(&db, vec![func(&db, vec![], int, Some(1))], int, None);
        let both = scheme(&db, &[1, 2], func(&db, vec![first, second], int, None));

        assert_eq!(
            class_variables(&db, both),
            [EffectVar { id: 1 }, EffectVar { id: 2 }]
        );
    }

    #[test]
    fn a_tail_outside_the_parameters_is_not_a_class_variable() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let own_row = scheme(&db, &[1], func(&db, vec![int], int, Some(1)));
        let returned = func(&db, vec![], int, Some(1));
        let result = scheme(&db, &[1], func(&db, vec![int], returned, None));

        assert!(class_variables(&db, own_row).is_empty());
        assert!(class_variables(&db, result).is_empty());
    }

    #[test]
    fn classes_that_are_all_cps_select_the_same_clone_as_no_classes() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let plain = InstanceKey::of_types(vec![int]);
        let cps = InstanceKey {
            type_args: vec![int],
            class_args: vec![CallingConvention::Cps],
        };
        let direct = InstanceKey {
            type_args: vec![int],
            class_args: vec![CallingConvention::Direct],
        };

        assert_eq!(plain.variant(), cps.variant());
        assert_ne!(plain.variant(), direct.variant());
    }
}

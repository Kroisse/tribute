//! Implicit constructors `.Name` (`new-plans/type-inference.md` § 암묵적 생성자).
//!
//! An implicit constructor names the constructor `Name` of the type its
//! context gives. Where that type or a single candidate decides it as it is
//! checked, it becomes the constructor reference a path would have written.
//! Otherwise its selection waits for the solved types like a deferred method
//! call, and finalization replaces it with the selected constructor.

use itertools::Itertools;
use trunk_ir::Symbol;

use super::TypeChecker;
use crate::ast::{AbilityRow, CtorId, LocalId, NodeId, ResolvedRef, Type, TypeKind};
use crate::typeck::constraint::{Constraint, ConstraintOrigin, ConstraintOriginKind};
use crate::typeck::func_context::{
    ConstructorShape, DeferredConstructor, FunctionInferenceContext, ImplicitConstructor,
};
use crate::typeck::solver::TypeSolver;

/// What the type of an implicit constructor's context says about it.
enum TypeConstructor<'db> {
    /// The type has the constructor.
    Found(CtorId<'db>),
    /// The type is not determined yet.
    Unknown,
    /// The type is a nominal type without a constructor of the name.
    Missing,
    /// The type is not one with constructors.
    NotNominal,
    /// The type has the constructor, but the module does not see it, for
    /// the reason given.
    Hidden(CtorId<'db>, String),
}

/// The constructor a deferred implicit constructor selects among its
/// candidates.
pub(super) enum ConstructorSelection<'db> {
    One(CtorId<'db>),
    /// Several fit, so it waits for more of its types.
    Ambiguous(Vec<CtorId<'db>>),
    /// None fits, or its type has none of the name.
    None,
}

impl<'db> TypeChecker<'db> {
    /// The constructor `resolved` names at the expression or pattern `node`.
    ///
    /// A path is itself. An implicit constructor is decided on its first
    /// visit: by the constructor `expected` has, else by its only candidate,
    /// else it waits, and stays the implicit constructor whose type
    /// [`deferred_constructor`](Self::deferred_constructor) gives.
    pub(super) fn decide_constructor(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        resolved: &ResolvedRef<'db>,
        expected: Option<Type<'db>>,
        shape: ConstructorShape,
        pattern: bool,
    ) -> ResolvedRef<'db> {
        let ResolvedRef::ImplicitConstructor {
            name,
            candidates,
            hidden,
        } = resolved
        else {
            return resolved.clone();
        };
        if ctx.implicit_constructor(node).is_some() {
            return self.constructor_reference(ctx, node, resolved);
        }
        let unresolved = || ResolvedRef::local(LocalId::UNRESOLVED, name.clone());
        let decided = match expected.map(|ty| (ty, self.type_constructor(ty, name, hidden))) {
            Some((_, TypeConstructor::Found(id))) => {
                ImplicitConstructor::Resolved(ResolvedRef::constructor(id, name.clone()))
            }
            Some((
                ty,
                missing @ (TypeConstructor::Missing
                | TypeConstructor::NotNominal
                | TypeConstructor::Hidden(..)),
            )) => {
                self.report_type_constructor(node, ty, name, &missing);
                ImplicitConstructor::Resolved(unresolved())
            }
            // Without candidates, only a type determined later can name it.
            None | Some((_, TypeConstructor::Unknown)) => match candidates.single() {
                Some(id) => {
                    ImplicitConstructor::Resolved(ResolvedRef::constructor(*id, name.clone()))
                }
                None => {
                    let (ctor_ty, result_ty, field_types) =
                        self.constructor_placeholder(ctx, &shape);
                    ImplicitConstructor::Deferred(DeferredConstructor {
                        node_id: node,
                        name: name.clone(),
                        candidates: candidates.clone(),
                        hidden: hidden.clone(),
                        shape,
                        pattern,
                        ctor_ty,
                        result_ty,
                        field_types,
                    })
                }
            },
        };
        ctx.record_implicit_constructor(node, decided);
        self.constructor_reference(ctx, node, resolved)
    }

    /// The constructor `resolved` names at `node` as it was decided there,
    /// or `resolved` itself while it waits or for a path.
    pub(super) fn constructor_reference(
        &self,
        ctx: &FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        resolved: &ResolvedRef<'db>,
    ) -> ResolvedRef<'db> {
        match ctx.implicit_constructor(node) {
            Some(ImplicitConstructor::Resolved(decided))
                if matches!(resolved, ResolvedRef::ImplicitConstructor { .. }) =>
            {
                decided.clone()
            }
            _ => resolved.clone(),
        }
    }

    /// Fix the instance of the implicit constructor `original` decided as
    /// `decided` at `node` to the type arguments of `expected`, the type it
    /// was decided by, so that the arguments and sub-patterns inside it see
    /// their field types where they are checked.
    pub(super) fn fix_implicit_constructor_instance(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        node: NodeId,
        original: &ResolvedRef<'db>,
        decided: &ResolvedRef<'db>,
        expected: Option<Type<'db>>,
    ) {
        let db = self.db();
        let (
            ResolvedRef::ImplicitConstructor { .. },
            ResolvedRef::Constructor { id, .. },
            Some(expected),
        ) = (original, decided, expected)
        else {
            return;
        };
        let TypeKind::Named {
            id: owner, args, ..
        } = expected.kind(db)
        else {
            return;
        };
        if ctx.get_constructor_reference_type(node).is_some() {
            return;
        }
        let Some(scheme) = self.env.lookup_constructor(*id) else {
            return;
        };
        let body = scheme.body(db);
        let result = match body.kind(db) {
            TypeKind::Func { result, .. } => *result,
            _ => body,
        };
        let constructs_owner = matches!(result.kind(db), TypeKind::Named { id, .. } if id == owner);
        if !constructs_owner
            || !scheme.effect_params(db).is_empty()
            || scheme.type_params(db).len() != args.len()
        {
            return;
        }
        if let crate::typeck::subst::SubstResult::Ok(instance) =
            crate::typeck::subst::substitute_bound_vars(db, body, args)
        {
            ctx.record_constructor_reference_type(node, instance);
        }
    }

    /// The implicit constructor at `node` that waits for its types.
    pub(super) fn deferred_constructor<'c>(
        &self,
        ctx: &'c FunctionInferenceContext<'_, 'db>,
        node: NodeId,
    ) -> Option<&'c DeferredConstructor<'db>> {
        match ctx.implicit_constructor(node)? {
            ImplicitConstructor::Deferred(deferred) => Some(deferred),
            ImplicitConstructor::Resolved(_) => None,
        }
    }

    /// The types that stand for a constructor written as `shape` until it
    /// is selected: the constructor's type, the type of its value, and the
    /// type of each argument, sub-pattern, or field.
    fn constructor_placeholder(
        &self,
        ctx: &mut FunctionInferenceContext<'_, 'db>,
        shape: &ConstructorShape,
    ) -> (Type<'db>, Type<'db>, Vec<Type<'db>>) {
        let result_ty = ctx.fresh_type_var();
        let count = match shape {
            ConstructorShape::Unit => 0,
            ConstructorShape::Positional(count) => *count,
            ConstructorShape::Fields { names, .. } => names.len(),
        };
        let field_types: Vec<_> = (0..count).map(|_| ctx.fresh_type_var()).collect();
        let ctor_ty = match shape {
            ConstructorShape::Unit => result_ty,
            _ => self
                .env
                .func_type(field_types.clone(), result_ty, AbilityRow::pure(self.db())),
        };
        (ctor_ty, result_ty, field_types)
    }

    /// The constructor `name` of `ty`, the type an implicit constructor
    /// constructs or matches, which is not one of the `hidden` ones.
    fn type_constructor(
        &self,
        ty: Type<'db>,
        name: &Symbol,
        hidden: &[(CtorId<'db>, String)],
    ) -> TypeConstructor<'db> {
        let db = self.db();
        match ty.kind(db) {
            TypeKind::Named { id, .. } => {
                let path = id.path(db);
                let variant = self
                    .env
                    .lookup_enum_variants(path)
                    .is_some_and(|variants| variants.contains(name))
                    .then(|| CtorId::new(db, path.child(name.clone())));
                let structure = (id.name(db) == *name)
                    .then(|| CtorId::new(db, path.clone()))
                    .filter(|ctor| self.env.lookup_constructor(*ctor).is_some());
                match variant.or(structure) {
                    Some(ctor) => match hidden.iter().find(|(id, _)| *id == ctor) {
                        Some((_, reason)) => TypeConstructor::Hidden(ctor, reason.clone()),
                        None => TypeConstructor::Found(ctor),
                    },
                    None => TypeConstructor::Missing,
                }
            }
            TypeKind::Int
            | TypeKind::Nat
            | TypeKind::Float
            | TypeKind::Bool
            | TypeKind::Bytes
            | TypeKind::Rune
            | TypeKind::Nil
            | TypeKind::Func { .. }
            | TypeKind::Tuple(_) => TypeConstructor::NotNominal,
            _ => TypeConstructor::Unknown,
        }
    }

    /// Report that `ty` has no constructor `name` for an implicit
    /// constructor at `node`.
    fn report_type_constructor(
        &self,
        node: NodeId,
        ty: Type<'db>,
        name: &Symbol,
        outcome: &TypeConstructor<'db>,
    ) {
        let message = match outcome {
            TypeConstructor::Missing => format!("type `{ty}` has no constructor `{name}`"),
            TypeConstructor::NotNominal => {
                format!("`.{name}` needs a type with constructors, found `{ty}`")
            }
            TypeConstructor::Hidden(ctor, reason) => format!(
                "`{}` is not visible here: {reason}",
                ctor.qualified(self.db())
            ),
            TypeConstructor::Found(_) | TypeConstructor::Unknown => return,
        };
        self.report_type_error(node, message);
    }

    // =========================================================================
    // Selection after solving
    // =========================================================================

    /// Select the constructor a deferred implicit constructor names with the
    /// types `solver` has inferred: the type it constructs or matches, else
    /// the candidate its arguments and that type fit, by rank. `settled` says
    /// whether those types are as determined as they will be.
    pub(super) fn select_deferred_constructor(
        &self,
        solver: &TypeSolver<'db>,
        deferred: &DeferredConstructor<'db>,
        settled: bool,
    ) -> ConstructorSelection<'db> {
        let db = self.db();
        let expected = self.deferred_expected_type(solver, deferred);
        if !matches!(
            expected.kind(db),
            TypeKind::UniVar { .. } | TypeKind::Never | TypeKind::Error
        ) {
            return match self.type_constructor(expected, &deferred.name, &deferred.hidden) {
                TypeConstructor::Found(id) => ConstructorSelection::One(id),
                _ => ConstructorSelection::None,
            };
        }
        let fields: Vec<_> = deferred
            .field_types
            .iter()
            .map(|ty| solver.type_subst().apply(db, *ty))
            .collect();
        let fits = |id: &CtorId<'db>| self.constructor_fits(*id, deferred, &fields);
        let ranks = deferred.candidates.ranks();
        for (index, rank) in ranks.iter().enumerate() {
            let matching: Vec<_> = rank.iter().copied().filter(fits).collect();
            match matching.as_slice() {
                [] => {}
                [one] => {
                    let lower: Vec<_> = ranks[index + 1..]
                        .iter()
                        .copied()
                        .flatten()
                        .copied()
                        .filter(fits)
                        .collect();
                    if settled || lower.is_empty() {
                        return ConstructorSelection::One(*one);
                    }
                    return ConstructorSelection::Ambiguous(
                        std::iter::once(*one).chain(lower).collect(),
                    );
                }
                _ => return ConstructorSelection::Ambiguous(matching),
            }
        }
        ConstructorSelection::None
    }

    /// The type that names the constructor of `deferred` once solved: the
    /// type of its value, or the result of the function a unit expression
    /// is used as.
    fn deferred_expected_type(
        &self,
        solver: &TypeSolver<'db>,
        deferred: &DeferredConstructor<'db>,
    ) -> Type<'db> {
        let expected = solver.expected_type(deferred.result_ty);
        match (expected.kind(self.db()), &deferred.shape) {
            (TypeKind::Func { result, .. }, ConstructorShape::Unit) if !deferred.pattern => {
                solver.type_subst().apply(self.db(), *result)
            }
            _ => expected,
        }
    }

    /// Whether the constructor `id` takes what `deferred` writes, whose
    /// fields have the types `fields` so far.
    fn constructor_fits(
        &self,
        id: CtorId<'db>,
        deferred: &DeferredConstructor<'db>,
        fields: &[Type<'db>],
    ) -> bool {
        let db = self.db();
        let Some(scheme) = self.env.lookup_constructor(id) else {
            return false;
        };
        let body = scheme.body(db);
        let params: &[Type<'db>] = match body.kind(db) {
            TypeKind::Func { params, .. } => params,
            _ => &[],
        };
        let matches = |declared: Type<'db>, actual: Type<'db>| {
            crate::typeck::parameter_type_matches(db, declared, actual)
        };
        match &deferred.shape {
            // A unit expression is the constructor as a value, of any arity.
            ConstructorShape::Unit => params.is_empty() || !deferred.pattern,
            ConstructorShape::Positional(count) => {
                params.len() == *count
                    && params
                        .iter()
                        .zip(fields)
                        .all(|(declared, actual)| matches(*declared, *actual))
            }
            ConstructorShape::Fields { names, .. } => {
                let Some(declared) = self.env.lookup_constructor_field_names(id) else {
                    return false;
                };
                names.iter().zip(fields).all(|(name, actual)| {
                    declared
                        .iter()
                        .position(|field| field == name)
                        .is_some_and(|index| matches(params[index], *actual))
                })
            }
        }
    }

    /// What selecting the constructor `id` for `deferred` requires: its
    /// fresh instance, and the constraints that relate the arguments,
    /// sub-patterns, or fields as written to that instance. Fields that do
    /// not fit the constructor are reported.
    pub(super) fn selected_constructor_constraints(
        &self,
        solver: &mut TypeSolver<'db>,
        deferred: &DeferredConstructor<'db>,
        id: CtorId<'db>,
    ) -> (Type<'db>, Vec<Constraint<'db>>) {
        let db = self.db();
        let scheme = self
            .env
            .lookup_constructor(id)
            .expect("a selected constructor has a scheme");
        let instance =
            crate::typeck::subst::instantiate_scheme_details_for_solver(db, scheme, solver).ty;
        let (params, result) = match instance.kind(db) {
            TypeKind::Func { params, result, .. } => (params.clone(), *result),
            _ => (Vec::new(), instance),
        };
        let origin = ConstraintOrigin {
            node_id: deferred.node_id,
            kind: ConstraintOriginKind::Expression,
        };
        let relate = |actual: Type<'db>, declared: Type<'db>| {
            if deferred.pattern {
                Constraint::TypeEqAt(actual, declared, origin)
            } else {
                Constraint::TypeCoerce(actual, declared, origin)
            }
        };
        let qualified = id.qualified(db);
        let mut constraints = Vec::new();
        match &deferred.shape {
            // A unit expression is the constructor itself, a function for
            // one with fields.
            ConstructorShape::Unit if !deferred.pattern => {
                constraints.push(Constraint::TypeEqAt(deferred.result_ty, instance, origin));
            }
            ConstructorShape::Unit | ConstructorShape::Positional(_) => {
                let written = deferred.field_types.len();
                if params.len() == written {
                    constraints.push(Constraint::TypeEqAt(deferred.ctor_ty, instance, origin));
                } else {
                    let plural = if params.len() == 1 { "" } else { "s" };
                    let site = if deferred.pattern {
                        "the pattern has"
                    } else {
                        "it is given"
                    };
                    self.report_type_error(
                        deferred.node_id,
                        format!(
                            "constructor `{qualified}` expects {} field{plural}, but {site} {written}",
                            params.len()
                        ),
                    );
                    constraints.push(Constraint::TypeEqAt(deferred.result_ty, result, origin));
                }
            }
            ConstructorShape::Fields { names, rest } => {
                constraints.push(Constraint::TypeEqAt(deferred.result_ty, result, origin));
                match self.env.lookup_constructor_field_names(id) {
                    Some(declared) => {
                        // The fields are checked as for a constructor
                        // written by its path.
                        let variant = matches!(
                            result.kind(db),
                            TypeKind::Named { id, .. }
                                if self.env.lookup_enum_variants(id.path(db)).is_some()
                        );
                        if variant && *rest && !deferred.pattern {
                            self.report_type_error(
                                deferred.node_id,
                                format!(
                                    "record spread is not allowed for variant `{qualified}`; write every field"
                                ),
                            );
                        }
                        let kind = if variant { "variant" } else { "struct" };
                        self.report_field_shape(
                            deferred.node_id,
                            format_args!("{kind} `{qualified}`"),
                            declared,
                            names.iter().cloned(),
                            *rest,
                        );
                        for (name, actual) in names.iter().zip(&deferred.field_types) {
                            if let Some(index) = declared.iter().position(|field| field == name) {
                                constraints.push(relate(*actual, params[index]));
                            }
                        }
                    }
                    None => {
                        let verb = if deferred.pattern {
                            "match"
                        } else {
                            "construct"
                        };
                        self.report_type_error(
                            deferred.node_id,
                            format!(
                                "`{qualified}` has positional fields; {verb} it with `.{}(...)`",
                                id.name(db)
                            ),
                        );
                    }
                }
            }
        }
        (instance, constraints)
    }

    /// Report a deferred implicit constructor that selects none of its
    /// candidates once its types are solved.
    pub(super) fn report_unselected_constructor(
        &self,
        solver: &TypeSolver<'db>,
        deferred: &DeferredConstructor<'db>,
    ) {
        let name = &deferred.name;
        let expected = self.deferred_expected_type(solver, deferred);
        let outcome = self.type_constructor(expected, name, &deferred.hidden);
        if matches!(
            outcome,
            TypeConstructor::Missing | TypeConstructor::NotNominal | TypeConstructor::Hidden(..)
        ) {
            self.report_type_constructor(deferred.node_id, expected, name, &outcome);
            return;
        }
        let names = |ctors: &[CtorId<'db>]| {
            ctors
                .iter()
                .format_with(", ", |ctor, f| {
                    f(&format_args!("`{}`", ctor.qualified(self.db())))
                })
                .to_string()
        };
        let message = match self.select_deferred_constructor(solver, deferred, true) {
            ConstructorSelection::Ambiguous(ctors) => format!(
                "ambiguous constructor `.{name}`: it may be {}; write its type",
                names(&ctors)
            ),
            ConstructorSelection::None if deferred.candidates.is_empty() => {
                format!("no type in scope has a constructor `{name}`")
            }
            ConstructorSelection::None => {
                let ctors: Vec<_> = deferred.candidates.iter().copied().collect();
                format!(
                    "no constructor `.{name}` in scope fits here: {}",
                    names(&ctors)
                )
            }
            ConstructorSelection::One(_) => return,
        };
        self.report_type_error(deferred.node_id, message);
    }
}

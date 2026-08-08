use std::collections::{HashMap, HashSet};

use crate::ast::{
    Decl, Expr, ExprKind, FuncDefId, Module, ResolvedRef, Stmt, Type, TypeDefId, TypeKind,
    TypeScheme, TypedRef,
};

/// Collect all generic function instantiations from a typed module.
///
/// Traverses the AST and records which concrete type argument combinations
/// each generic function is called with. The result maps each polymorphic
/// `FuncDefId` to the set of concrete type argument lists used.
pub fn collect_instantiations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    function_types: &[(trunk_ir::Symbol, TypeScheme<'db>)],
    call_callee_types: &HashMap<crate::ast::NodeId, Type<'db>>,
) -> HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let mut collector = InstantiationCollector::new(db, function_types, call_callee_types);
    collector.visit_module(module);
    collector.instantiations
}

/// Extract concrete type arguments by walking the scheme body and concrete type
/// in parallel. When the scheme has `BoundVar(i)`, the concrete type at that
/// position becomes `type_args[i]`.
///
/// Returns `None` if the scheme is monomorphic, or if extraction fails
/// (e.g., structural mismatch or inconsistent BoundVar mappings).
pub fn extract_type_args<'db>(
    db: &'db dyn salsa::Database,
    scheme: TypeScheme<'db>,
    concrete: Type<'db>,
) -> Option<Vec<Type<'db>>> {
    let num_params = scheme.type_params(db).len();
    if num_params == 0 {
        return None;
    }
    let mut type_args: Vec<Option<Type<'db>>> = vec![None; num_params];
    if !extract_recursive(db, scheme.body(db), concrete, &mut type_args) {
        return None;
    }
    type_args.into_iter().collect()
}

fn extract_recursive<'db>(
    db: &'db dyn salsa::Database,
    scheme_ty: Type<'db>,
    concrete_ty: Type<'db>,
    type_args: &mut [Option<Type<'db>>],
) -> bool {
    match (scheme_ty.kind(db), concrete_ty.kind(db)) {
        (TypeKind::BoundVar { index }, _) => {
            let i = *index as usize;
            if i >= type_args.len() {
                return false;
            }
            match type_args[i] {
                Some(existing) => existing == concrete_ty,
                None => {
                    type_args[i] = Some(concrete_ty);
                    true
                }
            }
        }
        (
            TypeKind::Func {
                params: sp,
                result: sr,
                effect: se,
                ..
            },
            TypeKind::Func {
                params: cp,
                result: cr,
                effect: ce,
                ..
            },
        ) => {
            if sp.len() != cp.len() {
                return false;
            }
            for (s, c) in sp.iter().zip(cp.iter()) {
                if !extract_recursive(db, *s, *c, type_args) {
                    return false;
                }
            }
            extract_recursive(db, *sr, *cr, type_args)
                && extract_effect_args(db, *se, *ce, type_args)
        }
        (
            TypeKind::Named {
                id: si, args: sa, ..
            },
            TypeKind::Named {
                id: ci, args: ca, ..
            },
        ) => {
            if si != ci || sa.len() != ca.len() {
                return false;
            }
            for (s, c) in sa.iter().zip(ca.iter()) {
                if !extract_recursive(db, *s, *c, type_args) {
                    return false;
                }
            }
            true
        }
        (TypeKind::Tuple(se), TypeKind::Tuple(ce)) => {
            if se.len() != ce.len() {
                return false;
            }
            for (s, c) in se.iter().zip(ce.iter()) {
                if !extract_recursive(db, *s, *c, type_args) {
                    return false;
                }
            }
            true
        }
        // Primitives and other identical types: interned equality
        _ => scheme_ty == concrete_ty,
    }
}

fn extract_effect_args<'db>(
    db: &'db dyn salsa::Database,
    scheme: crate::ast::EffectRow<'db>,
    concrete: crate::ast::EffectRow<'db>,
    type_args: &mut [Option<Type<'db>>],
) -> bool {
    let scheme_effects = scheme.effects(db);
    let concrete_effects = concrete.effects(db);
    if concrete_effects.len() < scheme_effects.len()
        || (scheme.rest(db).is_none()
            && (scheme_effects.len() != concrete_effects.len() || concrete.rest(db).is_some()))
    {
        return false;
    }
    scheme_effects.iter().zip(concrete_effects).all(|(s, c)| {
        s.ability_id == c.ability_id
            && s.args.len() == c.args.len()
            && s.args
                .iter()
                .zip(&c.args)
                .all(|(s, c)| extract_recursive(db, *s, *c, type_args))
    })
}

struct InstantiationCollector<'a, 'db> {
    db: &'db dyn salsa::Database,
    schemes: HashMap<FuncDefId<'db>, TypeScheme<'db>>,
    instantiations: HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>>,
    call_callee_types: &'a HashMap<crate::ast::NodeId, Type<'db>>,
}

impl<'a, 'db> InstantiationCollector<'a, 'db> {
    fn new(
        db: &'db dyn salsa::Database,
        function_types: &[(trunk_ir::Symbol, TypeScheme<'db>)],
        call_callee_types: &'a HashMap<crate::ast::NodeId, Type<'db>>,
    ) -> Self {
        let schemes = function_types
            .iter()
            .filter(|(_, scheme)| !scheme.is_mono(db))
            .map(|(sym, scheme)| (FuncDefId::new(db, *sym), *scheme))
            .collect();
        Self {
            db,
            schemes,
            instantiations: HashMap::new(),
            call_callee_types,
        }
    }

    fn try_record(&mut self, node: crate::ast::NodeId, typed_ref: &TypedRef<'db>) {
        let ResolvedRef::Function { id } = &typed_ref.resolved else {
            return;
        };
        let Some(scheme) = self.schemes.get(id) else {
            return;
        };
        let Some(concrete) = self.call_callee_types.get(&node).copied() else {
            return;
        };
        let Some(type_args) = extract_type_args(self.db, *scheme, concrete) else {
            return;
        };
        if !type_args
            .iter()
            .all(|type_arg| is_concrete_type(self.db, *type_arg))
        {
            return;
        }
        self.instantiations
            .entry(*id)
            .or_default()
            .insert(type_args);
    }

    fn visit_module(&mut self, module: &Module<TypedRef<'db>>) {
        for decl in &module.decls {
            self.visit_decl(decl);
        }
    }

    fn visit_decl(&mut self, decl: &Decl<TypedRef<'db>>) {
        match decl {
            Decl::Function(func) => self.visit_expr(&func.body),
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    for d in body {
                        self.visit_decl(d);
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr<TypedRef<'db>>) {
        match expr.kind.as_ref() {
            ExprKind::Var(typed_ref) => {
                self.try_record(expr.id, typed_ref);
            }
            ExprKind::Call { callee, args } => {
                self.visit_expr(callee);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::Block { stmts, value } => {
                for s in stmts {
                    self.visit_stmt(s);
                }
                self.visit_expr(value);
            }
            ExprKind::Case { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                }
            }
            ExprKind::Lambda { body, .. } => self.visit_expr(body),
            ExprKind::Handle { body, handlers } => {
                self.visit_expr(body);
                for h in handlers {
                    self.visit_expr(&h.body);
                }
            }
            ExprKind::Resume { arg, .. } => self.visit_expr(arg),
            ExprKind::Cons { args, .. } => {
                for a in args {
                    self.visit_expr(a);
                }
            }
            ExprKind::Record { fields, spread, .. } => {
                for (_, e) in fields {
                    self.visit_expr(e);
                }
                if let Some(s) = spread {
                    self.visit_expr(s);
                }
            }
            ExprKind::BinOp { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            ExprKind::Tuple(es) | ExprKind::List(es) => {
                for e in es {
                    self.visit_expr(e);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.visit_expr(receiver);
                for a in args {
                    self.visit_expr(a);
                }
            }
            // Leaf nodes — no sub-expressions to traverse
            ExprKind::NatLit(_)
            | ExprKind::IntLit(_)
            | ExprKind::FloatLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::BytesLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::RuneLit(_)
            | ExprKind::Nil
            | ExprKind::Error => {}
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt<TypedRef<'db>>) {
        match stmt {
            Stmt::Let { value, .. } => self.visit_expr(value),
            Stmt::Expr { expr, .. } => self.visit_expr(expr),
        }
    }
}

// ============================================================================
// Type instantiation collection (for generic struct/enum monomorphization)
// ============================================================================

/// Collect all generic type instantiations from a typed module.
///
/// Walks all types in the AST (recursively through Func, Tuple, Named, etc.)
/// and records which concrete type argument combinations each generic type
/// is used with. Only collects types whose names match generic struct/enum
/// declarations (those with non-empty `type_params`).
pub fn collect_type_instantiations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    type_definitions: &HashMap<trunk_ir::Symbol, TypeScheme<'db>>,
    node_types: &HashMap<crate::ast::NodeId, Type<'db>>,
    function_types: &[(trunk_ir::Symbol, TypeScheme<'db>)],
) -> HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let generic_types = collect_generic_type_ids(db, module, type_definitions);
    let mut result: HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>> = HashMap::new();
    for &ty in node_types.values() {
        collect_from_type(db, ty, &generic_types, &mut result);
    }
    for (_, scheme) in function_types {
        collect_from_type(db, scheme.body(db), &generic_types, &mut result);
    }

    let mut visitor = TypeInstantiationVisitor {
        db,
        generic_types: &generic_types,
        instantiations: &mut result,
    };
    visitor.visit_module(module);
    result
}

/// Collect declaration identities of all generic struct/enum declarations.
fn collect_generic_type_ids<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    type_definitions: &HashMap<trunk_ir::Symbol, TypeScheme<'db>>,
) -> HashSet<TypeDefId<'db>> {
    let mut ids = HashSet::new();
    let mut prefix = String::new();
    collect_generic_type_ids_inner(db, &module.decls, &mut prefix, type_definitions, &mut ids);
    ids
}

fn collect_generic_type_ids_inner<'db>(
    db: &'db dyn salsa::Database,
    decls: &[Decl<TypedRef<'db>>],
    prefix: &mut String,
    type_definitions: &HashMap<trunk_ir::Symbol, TypeScheme<'db>>,
    ids: &mut HashSet<TypeDefId<'db>>,
) {
    for decl in decls {
        match decl {
            Decl::Struct(s) if !s.type_params.is_empty() => {
                let qualified = crate::qualified_symbol(prefix, s.name);
                let scheme = type_definitions
                    .get(&qualified)
                    .expect("generic struct must have an exact nominal definition scheme");
                ids.insert(
                    nominal_result_id(db, scheme.body(db))
                        .expect("generic struct definition must contain a nominal type"),
                );
            }
            Decl::Enum(e) if !e.type_params.is_empty() => {
                let qualified = crate::qualified_symbol(prefix, e.name);
                let scheme = type_definitions
                    .get(&qualified)
                    .expect("generic enum must have an exact nominal definition scheme");
                ids.insert(
                    nominal_result_id(db, scheme.body(db))
                        .expect("generic enum definition must contain a nominal type"),
                );
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    let saved = crate::push_prefix(prefix, m.name);
                    collect_generic_type_ids_inner(db, body, prefix, type_definitions, ids);
                    prefix.truncate(saved);
                }
            }
            _ => {}
        }
    }
}

fn nominal_result_id<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> Option<TypeDefId<'db>> {
    let result = match ty.kind(db) {
        TypeKind::Func { result, .. } => *result,
        _ => ty,
    };
    match result.kind(db) {
        TypeKind::Named { id, .. } => Some(*id),
        _ => None,
    }
}

/// Recursively walk a Type and collect all declaration-backed generic types.
fn collect_from_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    generic_types: &HashSet<TypeDefId<'db>>,
    result: &mut HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
) {
    match ty.kind(db) {
        TypeKind::Named { id, args, .. } => {
            if !args.is_empty()
                && generic_types.contains(id)
                && args.iter().all(|arg| is_concrete_type(db, *arg))
            {
                result.entry(*id).or_default().insert(args.clone());
            }
            // Recurse into type arguments (e.g., List(Option(Int)) → collect Option(Int))
            for arg in args {
                collect_from_type(db, *arg, generic_types, result);
            }
        }
        TypeKind::Func {
            params,
            result: ret,
            effect,
            ..
        } => {
            for p in params {
                collect_from_type(db, *p, generic_types, result);
            }
            collect_from_type(db, *ret, generic_types, result);
            for ability in effect.effects(db) {
                for argument in &ability.args {
                    collect_from_type(db, *argument, generic_types, result);
                }
            }
        }
        TypeKind::Tuple(elems) => {
            for e in elems {
                collect_from_type(db, *e, generic_types, result);
            }
        }
        TypeKind::App { ctor, args } => {
            collect_from_type(db, *ctor, generic_types, result);
            for a in args {
                collect_from_type(db, *a, generic_types, result);
            }
        }
        TypeKind::Continuation {
            arg, result: ret, ..
        } => {
            collect_from_type(db, *arg, generic_types, result);
            collect_from_type(db, *ret, generic_types, result);
        }
        // Primitives and variables: no nested Named types
        _ => {}
    }
}

fn is_concrete_type(db: &dyn salsa::Database, ty: Type<'_>) -> bool {
    match ty.kind(db) {
        TypeKind::Named { args, .. } | TypeKind::Tuple(args) => {
            args.iter().all(|arg| is_concrete_type(db, *arg))
        }
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            params.iter().all(|param| is_concrete_type(db, *param))
                && is_concrete_type(db, *result)
                && effect
                    .effects(db)
                    .iter()
                    .all(|entry| entry.args.iter().all(|arg| is_concrete_type(db, *arg)))
        }
        TypeKind::App { .. }
        | TypeKind::Continuation { .. }
        | TypeKind::BoundVar { .. }
        | TypeKind::LocalBoundVar { .. }
        | TypeKind::UniVar { .. }
        | TypeKind::Error => false,
        TypeKind::Int
        | TypeKind::Nat
        | TypeKind::Float
        | TypeKind::Bool
        | TypeKind::Bytes
        | TypeKind::Rune
        | TypeKind::Nil
        | TypeKind::Never => true,
    }
}

struct TypeInstantiationVisitor<'a, 'db> {
    db: &'db dyn salsa::Database,
    generic_types: &'a HashSet<TypeDefId<'db>>,
    instantiations: &'a mut HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
}

impl<'a, 'db> TypeInstantiationVisitor<'a, 'db> {
    fn collect_type(&mut self, ty: Type<'db>) {
        collect_from_type(self.db, ty, self.generic_types, self.instantiations);
    }

    fn visit_typed_ref(&mut self, tr: &TypedRef<'db>) {
        self.collect_type(tr.ty);
    }

    fn visit_module(&mut self, module: &Module<TypedRef<'db>>) {
        for decl in &module.decls {
            self.visit_decl(decl);
        }
    }

    fn visit_decl(&mut self, decl: &Decl<TypedRef<'db>>) {
        match decl {
            Decl::Function(func) => self.visit_expr(&func.body),
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    for d in body {
                        self.visit_decl(d);
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr<TypedRef<'db>>) {
        match expr.kind.as_ref() {
            ExprKind::Var(tr) => self.visit_typed_ref(tr),
            ExprKind::Call { callee, args } => {
                self.visit_expr(callee);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::Block { stmts, value } => {
                for s in stmts {
                    self.visit_stmt(s);
                }
                self.visit_expr(value);
            }
            ExprKind::Case { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                }
            }
            ExprKind::Lambda { body, .. } => self.visit_expr(body),
            ExprKind::Handle { body, handlers } => {
                self.visit_expr(body);
                for h in handlers {
                    self.visit_expr(&h.body);
                }
            }
            ExprKind::Resume { arg, .. } => self.visit_expr(arg),
            ExprKind::Cons { ctor, args } => {
                self.visit_typed_ref(ctor);
                for a in args {
                    self.visit_expr(a);
                }
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
                ..
            } => {
                self.visit_typed_ref(type_name);
                for (_, e) in fields {
                    self.visit_expr(e);
                }
                if let Some(s) = spread {
                    self.visit_expr(s);
                }
            }
            ExprKind::BinOp { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            ExprKind::Tuple(es) | ExprKind::List(es) => {
                for e in es {
                    self.visit_expr(e);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.visit_expr(receiver);
                for a in args {
                    self.visit_expr(a);
                }
            }
            ExprKind::NatLit(_)
            | ExprKind::IntLit(_)
            | ExprKind::FloatLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::BytesLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::RuneLit(_)
            | ExprKind::Nil
            | ExprKind::Error => {}
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt<TypedRef<'db>>) {
        match stmt {
            Stmt::Let { value, .. } => self.visit_expr(value),
            Stmt::Expr { expr, .. } => self.visit_expr(expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{
        AbilityId, Effect, EffectRow, EffectVar, EnumDecl, NodeId, TypeParam, TypeParamDecl,
        TypeScheme,
    };

    use super::*;

    #[salsa::db]
    #[derive(Default)]
    struct TestDb {
        storage: salsa::Storage<Self>,
    }

    #[salsa::db]
    impl salsa::Database for TestDb {}

    fn make_scheme<'db>(
        db: &'db dyn salsa::Database,
        num_params: usize,
        body: Type<'db>,
    ) -> TypeScheme<'db> {
        let type_params: Vec<_> = (0..num_params).map(|_| TypeParam::anonymous()).collect();
        TypeScheme::new(db, type_params, Vec::new(), body)
    }

    fn pure_effect(db: &dyn salsa::Database) -> EffectRow<'_> {
        EffectRow::new(db, vec![], None)
    }

    // ========================================================================
    // extract_type_args tests
    // ========================================================================

    #[test]
    fn test_extract_single_param() {
        let db = TestDb::default();
        // ∀a. a → a
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let scheme_body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![bv0],
                result: bv0,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let scheme = make_scheme(&db, 1, scheme_body);

        let int = Type::new(&db, TypeKind::Int);
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let result = extract_type_args(&db, scheme, concrete);
        assert_eq!(result, Some(vec![int]));
    }

    #[test]
    fn test_extract_multiple_params() {
        let db = TestDb::default();
        // ∀a,b. (a, b) → a
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let bv1 = Type::new(&db, TypeKind::BoundVar { index: 1 });
        let scheme_body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![bv0, bv1],
                result: bv0,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let scheme = make_scheme(&db, 2, scheme_body);

        let int = Type::new(&db, TypeKind::Int);
        let float = Type::new(&db, TypeKind::Float);
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int, float],
                result: int,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let result = extract_type_args(&db, scheme, concrete);
        assert_eq!(result, Some(vec![int, float]));
    }

    #[test]
    fn test_extract_same_param_twice() {
        let db = TestDb::default();
        // ∀a. (a, a) → a
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let scheme_body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![bv0, bv0],
                result: bv0,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let scheme = make_scheme(&db, 1, scheme_body);

        let int = Type::new(&db, TypeKind::Int);
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int, int],
                result: int,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let result = extract_type_args(&db, scheme, concrete);
        assert_eq!(result, Some(vec![int]));
    }

    #[test]
    fn test_extract_param_present_only_in_ordered_effect_row() {
        let db = TestDb::default();
        let state = AbilityId::source(&db, trunk_ir::Symbol::new("State"));
        let audit = AbilityId::source(&db, trunk_ir::Symbol::new("Audit"));
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let nil = Type::new(&db, TypeKind::Nil);
        let scheme_effect = EffectRow::new(
            &db,
            vec![
                Effect {
                    ability_id: state,
                    args: vec![bv0],
                },
                Effect {
                    ability_id: audit,
                    args: vec![],
                },
            ],
            None,
        );
        let scheme = make_scheme(
            &db,
            1,
            Type::new(
                &db,
                TypeKind::Func {
                    params: vec![],
                    result: nil,
                    effect: scheme_effect,
                    minimum_convention: crate::ast::CallingConvention::Direct,
                },
            ),
        );

        let int = Type::new(&db, TypeKind::Int);
        let concrete_effect = EffectRow::new(
            &db,
            vec![
                Effect {
                    ability_id: state,
                    args: vec![int],
                },
                Effect {
                    ability_id: audit,
                    args: vec![],
                },
            ],
            None,
        );
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: concrete_effect,
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        assert_eq!(extract_type_args(&db, scheme, concrete), Some(vec![int]));

        let reversed = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: EffectRow::new(
                    &db,
                    vec![
                        Effect {
                            ability_id: audit,
                            args: vec![],
                        },
                        Effect {
                            ability_id: state,
                            args: vec![int],
                        },
                    ],
                    None,
                ),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        assert_eq!(extract_type_args(&db, scheme, reversed), None);
    }

    #[test]
    fn test_extract_open_effect_row_accepts_ordered_concrete_suffix() {
        let db = TestDb::default();
        let state = AbilityId::source(&db, trunk_ir::Symbol::new("State"));
        let audit = AbilityId::source(&db, trunk_ir::Symbol::new("Audit"));
        let trace = AbilityId::source(&db, trunk_ir::Symbol::new("Trace"));
        let scheme_tail = EffectVar { id: 10 };
        let concrete_tail = EffectVar { id: 20 };
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let nil = Type::new(&db, TypeKind::Nil);
        let scheme = TypeScheme::new(
            &db,
            vec![TypeParam::anonymous()],
            vec![scheme_tail],
            Type::new(
                &db,
                TypeKind::Func {
                    params: vec![],
                    result: nil,
                    effect: EffectRow::new(
                        &db,
                        vec![
                            Effect {
                                ability_id: state,
                                args: vec![bound],
                            },
                            Effect {
                                ability_id: audit,
                                args: vec![],
                            },
                        ],
                        Some(scheme_tail),
                    ),
                    minimum_convention: crate::ast::CallingConvention::Direct,
                },
            ),
        );
        let int = Type::new(&db, TypeKind::Int);
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: EffectRow::new(
                    &db,
                    vec![
                        Effect {
                            ability_id: state,
                            args: vec![int],
                        },
                        Effect {
                            ability_id: audit,
                            args: vec![],
                        },
                        Effect {
                            ability_id: trace,
                            args: vec![],
                        },
                    ],
                    Some(concrete_tail),
                ),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        assert_eq!(extract_type_args(&db, scheme, concrete), Some(vec![int]));
    }

    #[test]
    fn test_extract_open_effect_row_rejects_wrong_ordered_prefix() {
        let db = TestDb::default();
        let state = AbilityId::source(&db, trunk_ir::Symbol::new("State"));
        let audit = AbilityId::source(&db, trunk_ir::Symbol::new("Audit"));
        let scheme_tail = EffectVar { id: 30 };
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let nil = Type::new(&db, TypeKind::Nil);
        let scheme = TypeScheme::new(
            &db,
            vec![TypeParam::anonymous()],
            vec![scheme_tail],
            Type::new(
                &db,
                TypeKind::Func {
                    params: vec![],
                    result: nil,
                    effect: EffectRow::new(
                        &db,
                        vec![
                            Effect {
                                ability_id: state,
                                args: vec![bound],
                            },
                            Effect {
                                ability_id: audit,
                                args: vec![],
                            },
                        ],
                        Some(scheme_tail),
                    ),
                    minimum_convention: crate::ast::CallingConvention::Direct,
                },
            ),
        );
        let int = Type::new(&db, TypeKind::Int);
        let reordered = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: EffectRow::new(
                    &db,
                    vec![
                        Effect {
                            ability_id: audit,
                            args: vec![],
                        },
                        Effect {
                            ability_id: state,
                            args: vec![int],
                        },
                    ],
                    Some(EffectVar { id: 31 }),
                ),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        assert_eq!(extract_type_args(&db, scheme, reordered), None);
    }

    #[test]
    fn test_extract_closed_effect_row_rejects_extra_effects_and_open_tail() {
        let db = TestDb::default();
        let state = AbilityId::source(&db, trunk_ir::Symbol::new("State"));
        let audit = AbilityId::source(&db, trunk_ir::Symbol::new("Audit"));
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let nil = Type::new(&db, TypeKind::Nil);
        let scheme = make_scheme(
            &db,
            1,
            Type::new(
                &db,
                TypeKind::Func {
                    params: vec![],
                    result: nil,
                    effect: EffectRow::single(
                        &db,
                        Effect {
                            ability_id: state,
                            args: vec![bound],
                        },
                    ),
                    minimum_convention: crate::ast::CallingConvention::Direct,
                },
            ),
        );
        let int = Type::new(&db, TypeKind::Int);
        let extra_effect = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: EffectRow::new(
                    &db,
                    vec![
                        Effect {
                            ability_id: state,
                            args: vec![int],
                        },
                        Effect {
                            ability_id: audit,
                            args: vec![],
                        },
                    ],
                    None,
                ),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let open_tail = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: EffectRow::new(
                    &db,
                    vec![Effect {
                        ability_id: state,
                        args: vec![int],
                    }],
                    Some(EffectVar { id: 40 }),
                ),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        assert_eq!(extract_type_args(&db, scheme, extra_effect), None);
        assert_eq!(extract_type_args(&db, scheme, open_tail), None);
    }

    #[test]
    fn test_extract_consistency_mismatch() {
        let db = TestDb::default();
        // ∀a. (a, a) → a with (Int, Text) → Int — inconsistent
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let scheme_body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![bv0, bv0],
                result: bv0,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let scheme = make_scheme(&db, 1, scheme_body);

        let int = Type::new(&db, TypeKind::Int);
        let text = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Text")),
                name: trunk_ir::Symbol::new("Text"),
                args: vec![],
            },
        );
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int, text],
                result: int,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        assert_eq!(extract_type_args(&db, scheme, concrete), None);
    }

    #[test]
    fn test_extract_nested_named() {
        let db = TestDb::default();
        // ∀a. Option(a) → a
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let option_bv = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Option")),
                name: trunk_ir::Symbol::new("Option"),
                args: vec![bv0],
            },
        );
        let scheme_body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![option_bv],
                result: bv0,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let scheme = make_scheme(&db, 1, scheme_body);

        let int = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Option")),
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int],
            },
        );
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![option_int],
                result: int,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let result = extract_type_args(&db, scheme, concrete);
        assert_eq!(result, Some(vec![int]));
    }

    #[test]
    fn test_extract_monomorphic_returns_none() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let scheme = make_scheme(&db, 0, body);
        assert_eq!(extract_type_args(&db, scheme, body), None);
    }

    // ========================================================================
    // collect_type_instantiations tests
    // ========================================================================

    #[test]
    fn test_collect_type_from_named() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let option_id = crate::ast::TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Option"));
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: option_id,
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int],
            },
        );

        let mut generic_types = HashSet::new();
        generic_types.insert(option_id);

        let mut result = HashMap::new();
        collect_from_type(&db, option_int, &generic_types, &mut result);

        assert_eq!(result.len(), 1);
        let option_insts = result.get(&option_id).unwrap();
        assert!(option_insts.contains(&vec![int]));
    }

    #[test]
    fn test_collect_empty_generic_enum_uses_exact_definition_identity() {
        let db = TestDb::default();
        let declaration = NodeId::from_raw(1);
        let name = trunk_ir::Symbol::new("Empty");
        let id = TypeDefId::source(&db, name, declaration);
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let generic = Type::new(
            &db,
            TypeKind::Named {
                id,
                name,
                args: vec![bound],
            },
        );
        let module = Module::<TypedRef<'_>>::new(
            NodeId::from_raw(2),
            None,
            vec![Decl::Enum(EnumDecl {
                id: declaration,
                is_pub: false,
                name,
                type_params: vec![TypeParamDecl {
                    id: NodeId::from_raw(3),
                    name: trunk_ir::Symbol::new("a"),
                    bounds: vec![],
                }],
                variants: vec![],
            })],
        );
        let definitions = HashMap::from([(name, make_scheme(&db, 1, generic))]);
        let int = Type::new(&db, TypeKind::Int);
        let empty_int = Type::new(
            &db,
            TypeKind::Named {
                id,
                name,
                args: vec![int],
            },
        );
        let signature = Type::new(
            &db,
            TypeKind::Func {
                params: vec![empty_int],
                result: Type::new(&db, TypeKind::Nil),
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let function_types = vec![(
            trunk_ir::Symbol::new("consume_empty"),
            TypeScheme::mono(&db, signature),
        )];

        let instantiations = collect_type_instantiations(
            &db,
            &module,
            &definitions,
            &HashMap::new(),
            &function_types,
        );
        assert_eq!(instantiations.get(&id), Some(&HashSet::from([vec![int]])));
    }

    #[test]
    fn test_collect_nominal_instantiation_present_only_in_function_effect() {
        let db = TestDb::default();
        let declaration = NodeId::from_raw(10);
        let name = trunk_ir::Symbol::new("Token");
        let id = TypeDefId::source(&db, name, declaration);
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let generic = Type::new(
            &db,
            TypeKind::Named {
                id,
                name,
                args: vec![bound],
            },
        );
        let module = Module::<TypedRef<'_>>::new(
            NodeId::from_raw(11),
            None,
            vec![Decl::Enum(EnumDecl {
                id: declaration,
                is_pub: false,
                name,
                type_params: vec![TypeParamDecl {
                    id: NodeId::from_raw(12),
                    name: trunk_ir::Symbol::new("a"),
                    bounds: vec![],
                }],
                variants: vec![],
            })],
        );
        let definitions = HashMap::from([(name, make_scheme(&db, 1, generic))]);
        let int = Type::new(&db, TypeKind::Int);
        let token_int = Type::new(
            &db,
            TypeKind::Named {
                id,
                name,
                args: vec![int],
            },
        );
        let signature = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: Type::new(&db, TypeKind::Nil),
                effect: EffectRow::new(
                    &db,
                    vec![Effect {
                        ability_id: AbilityId::source(&db, trunk_ir::Symbol::new("Witness")),
                        args: vec![token_int],
                    }],
                    None,
                ),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let instantiations = collect_type_instantiations(
            &db,
            &module,
            &definitions,
            &HashMap::new(),
            &[(
                trunk_ir::Symbol::new("effect_only"),
                TypeScheme::mono(&db, signature),
            )],
        );
        assert_eq!(instantiations.get(&id), Some(&HashSet::from([vec![int]])));
    }

    #[test]
    fn test_collect_type_nested() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let option_id = crate::ast::TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Option"));
        let list_id = crate::ast::TypeDefId::builtin_list(&db);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: option_id,
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int],
            },
        );
        let list_option_int = Type::new(
            &db,
            TypeKind::Named {
                id: list_id,
                name: trunk_ir::Symbol::new("List"),
                args: vec![option_int],
            },
        );

        let mut generic_types = HashSet::new();
        generic_types.insert(option_id);
        generic_types.insert(list_id);

        let mut result = HashMap::new();
        collect_from_type(&db, list_option_int, &generic_types, &mut result);

        assert_eq!(result.len(), 2);
        assert!(result[&option_id].contains(&vec![int]));
        assert!(result[&list_id].contains(&vec![option_int]));
    }

    #[test]
    fn test_collect_type_in_func_params() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let pair_id = crate::ast::TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Pair"));
        let pair_int_int = Type::new(
            &db,
            TypeKind::Named {
                id: pair_id,
                name: trunk_ir::Symbol::new("Pair"),
                args: vec![int, int],
            },
        );
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![pair_int_int],
                result: int,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let mut generic_types = HashSet::new();
        generic_types.insert(pair_id);

        let mut result = HashMap::new();
        collect_from_type(&db, func_ty, &generic_types, &mut result);

        assert_eq!(result.len(), 1);
        assert!(result[&pair_id].contains(&vec![int, int]));
    }

    #[test]
    fn test_collect_type_ignores_non_generic() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        // Named type with args but NOT in generic_types set
        let unknown = Type::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Unknown")),
                name: trunk_ir::Symbol::new("Unknown"),
                args: vec![int],
            },
        );

        let generic_types = HashSet::new(); // empty — nothing is generic
        let mut result = HashMap::new();
        collect_from_type(&db, unknown, &generic_types, &mut result);

        assert!(result.is_empty());
    }

    // ========================================================================
    // extract_type_args tests (continued)
    // ========================================================================

    #[test]
    fn test_extract_func_type_arg() {
        let db = TestDb::default();
        // ∀a,b. fn(a) → b  (the whole thing is a function type with a function-typed param)
        let bv0 = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let bv1 = Type::new(&db, TypeKind::BoundVar { index: 1 });
        let fn_param = Type::new(
            &db,
            TypeKind::Func {
                params: vec![bv0],
                result: bv1,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let scheme_body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![fn_param, bv0],
                result: bv1,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let scheme = make_scheme(&db, 2, scheme_body);

        let int = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let fn_concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: bool_ty,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![fn_concrete, int],
                result: bool_ty,
                effect: pure_effect(&db),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let result = extract_type_args(&db, scheme, concrete);
        assert_eq!(result, Some(vec![int, bool_ty]));
    }
}

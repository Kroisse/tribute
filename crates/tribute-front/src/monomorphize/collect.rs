use super::nominal_index::NominalIndex;
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
    function_instances: &HashMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
) -> HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let mut collector = InstantiationCollector::new(db, function_types, function_instances);
    collector.visit_module(module);
    collector.instantiations
}

/// Extract concrete type arguments by walking the scheme body and concrete type
/// in parallel. When the scheme has `BoundVar(i)`, the concrete type at that
/// position becomes `type_args[i]`.
///
/// Returns `None` if the scheme is monomorphic, or if extraction fails
/// (e.g., structural mismatch or inconsistent BoundVar mappings).
#[cfg(test)]
fn extract_type_args<'db>(
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

#[cfg(test)]
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
                ..
            },
            TypeKind::Func {
                params: cp,
                result: cr,
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

struct InstantiationCollector<'a, 'db> {
    db: &'db dyn salsa::Database,
    schemes: HashMap<FuncDefId<'db>, TypeScheme<'db>>,
    function_instances: &'a HashMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
    instantiations: HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>>,
}

impl<'a, 'db> InstantiationCollector<'a, 'db> {
    fn new(
        db: &'db dyn salsa::Database,
        function_types: &[(trunk_ir::Symbol, TypeScheme<'db>)],
        function_instances: &'a HashMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
    ) -> Self {
        let schemes = function_types
            .iter()
            .filter(|(_, scheme)| !scheme.is_mono(db))
            .map(|(sym, scheme)| (FuncDefId::new(db, *sym), *scheme))
            .collect();
        Self {
            db,
            schemes,
            function_instances,
            instantiations: HashMap::new(),
        }
    }

    fn try_record(&mut self, node_id: crate::ast::NodeId, typed_ref: &TypedRef<'db>) {
        let ResolvedRef::Function { id } = &typed_ref.resolved else {
            return;
        };
        let Some(scheme) = self.schemes.get(id) else {
            return;
        };
        if scheme.type_params(self.db).is_empty() {
            return;
        }
        let Some(instance) = self.function_instances.get(&node_id) else {
            return;
        };
        if instance.function != *id {
            return;
        }
        let type_args = instance.type_arguments.clone();
        if !type_args.iter().all(|ty| is_concrete_type(self.db, *ty)) {
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

pub(crate) fn is_concrete_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    is_concrete_type_cached(db, ty, &mut HashMap::new())
}

fn is_concrete_type_cached<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    cache: &mut HashMap<Type<'db>, bool>,
) -> bool {
    if let Some(concrete) = cache.get(&ty) {
        return *concrete;
    }
    let concrete = match ty.kind(db) {
        TypeKind::Named { args, .. } => args
            .iter()
            .all(|arg| is_concrete_type_cached(db, *arg, cache)),
        TypeKind::Func {
            params,
            result,
            effect,
            ..
        } => {
            params
                .iter()
                .all(|param| is_concrete_type_cached(db, *param, cache))
                && is_concrete_type_cached(db, *result, cache)
                && is_concrete_effect_row_cached(db, *effect, cache)
        }
        TypeKind::Tuple(elements) => elements
            .iter()
            .all(|element| is_concrete_type_cached(db, *element, cache)),
        TypeKind::Int
        | TypeKind::Nat
        | TypeKind::Float
        | TypeKind::Bool
        | TypeKind::Bytes
        | TypeKind::Rune
        | TypeKind::Nil
        | TypeKind::Never => true,
        TypeKind::BoundVar { .. }
        | TypeKind::LocalBoundVar { .. }
        | TypeKind::UniVar { .. }
        | TypeKind::App { .. }
        | TypeKind::Error => false,
        TypeKind::Continuation {
            arg,
            result,
            effect,
        } => {
            is_concrete_type_cached(db, *arg, cache)
                && is_concrete_type_cached(db, *result, cache)
                && is_concrete_effect_row_cached(db, *effect, cache)
        }
    };
    cache.insert(ty, concrete);
    concrete
}

fn is_concrete_effect_row_cached<'db>(
    db: &'db dyn salsa::Database,
    row: crate::ast::EffectRow<'db>,
    cache: &mut HashMap<Type<'db>, bool>,
) -> bool {
    row.rest(db).is_none()
        && row.effects(db).iter().all(|effect| {
            effect
                .args
                .iter()
                .all(|arg| is_concrete_type_cached(db, *arg, cache))
        })
}

// ============================================================================
// Type instantiation collection (for generic struct/enum monomorphization)
// ============================================================================

/// Collect all generic type instantiations from a typed module.
///
/// Walks all types in the AST (recursively through Func, Tuple, Named, etc.)
/// and records which concrete type argument combinations each generic type
/// is used with. Only collects instances of indexed generic struct/enum
/// declarations (those with non-empty `type_params`), matched by TypeDefId.
pub fn collect_type_instantiations<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    extra_types: impl IntoIterator<Item = Type<'db>>,
) -> HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let index = NominalIndex::new(db, module);
    collect_type_instantiations_with_index(db, module, extra_types, &index)
}

pub(super) fn collect_type_instantiations_with_index<'db>(
    db: &'db dyn salsa::Database,
    module: &Module<TypedRef<'db>>,
    extra_types: impl IntoIterator<Item = Type<'db>>,
    index: &NominalIndex<'_, 'db>,
) -> HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let mut result: HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>> = HashMap::new();

    let mut visitor = TypeInstantiationVisitor {
        db,
        index,
        instantiations: &mut result,
    };
    visitor.visit_module(module);
    for ty in extra_types {
        collect_from_type(db, ty, index, &mut result);
    }
    result
}

/// Recursively walk a Type and collect all declaration-backed generic types.
pub(super) fn collect_from_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    index: &NominalIndex<'_, 'db>,
    result: &mut HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
) {
    collect_from_type_inner(db, ty, index, result, &mut HashSet::new());
}

// Interned types form a DAG. Visit shared arguments once, including during
// expanding nominal dependency discovery, so its round limit remains effective.
fn collect_from_type_inner<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    index: &NominalIndex<'_, 'db>,
    result: &mut HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
    seen: &mut HashSet<Type<'db>>,
) {
    if !seen.insert(ty) {
        return;
    }
    match ty.kind(db) {
        TypeKind::Named { id, args, .. } => {
            if !args.is_empty()
                && index.is_generic(*id)
                && args.iter().all(|arg| is_concrete_type(db, *arg))
            {
                result.entry(*id).or_default().insert(args.clone());
            }
            // Recurse into type arguments (e.g., List(Option(Int)) → collect Option(Int))
            for arg in args {
                collect_from_type_inner(db, *arg, index, result, seen);
            }
        }
        TypeKind::Func {
            params,
            result: ret,
            effect,
            ..
        } => {
            for p in params {
                collect_from_type_inner(db, *p, index, result, seen);
            }
            collect_from_type_inner(db, *ret, index, result, seen);
            for effect in effect.effects(db) {
                for arg in &effect.args {
                    collect_from_type_inner(db, *arg, index, result, seen);
                }
            }
        }
        TypeKind::Tuple(elems) => {
            for e in elems {
                collect_from_type_inner(db, *e, index, result, seen);
            }
        }
        TypeKind::App { ctor, args } => {
            collect_from_type_inner(db, *ctor, index, result, seen);
            for a in args {
                collect_from_type_inner(db, *a, index, result, seen);
            }
        }
        TypeKind::Continuation {
            arg,
            result: ret,
            effect,
        } => {
            collect_from_type_inner(db, *arg, index, result, seen);
            collect_from_type_inner(db, *ret, index, result, seen);
            for effect in effect.effects(db) {
                for arg in &effect.args {
                    collect_from_type_inner(db, *arg, index, result, seen);
                }
            }
        }
        // Primitives and variables: no nested Named types
        _ => {}
    }
}

struct TypeInstantiationVisitor<'a, 'ast, 'db> {
    db: &'db dyn salsa::Database,
    index: &'a NominalIndex<'ast, 'db>,
    instantiations: &'a mut HashMap<TypeDefId<'db>, HashSet<Vec<Type<'db>>>>,
}

impl<'a, 'ast, 'db> TypeInstantiationVisitor<'a, 'ast, 'db> {
    fn collect_type(&mut self, ty: Type<'db>) {
        collect_from_type(self.db, ty, self.index, self.instantiations);
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
    use crate::ast::{AbilityId, Effect, EffectRow, EffectVar, NodeId, TypeParam, TypeScheme};
    use trunk_ir::Symbol;

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

    fn direct_function_type<'db>(
        db: &'db dyn salsa::Database,
        effect: EffectRow<'db>,
    ) -> Type<'db> {
        let int = Type::new(db, TypeKind::Int);
        Type::new(
            db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect,
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        )
    }

    #[test]
    fn concrete_type_accepts_closed_effect_rows_with_concrete_ability_arguments() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let row = EffectRow::new(
            &db,
            vec![Effect {
                ability_id: AbilityId::source(&db, Symbol::new("State")),
                args: vec![int],
            }],
            None,
        );

        assert!(is_concrete_type(&db, direct_function_type(&db, row)));
    }

    #[test]
    fn concrete_type_rejects_open_effect_rows() {
        let db = TestDb::default();
        let open = EffectRow::open(&db, EffectVar { id: 0 });
        let int = Type::new(&db, TypeKind::Int);
        let continuation = Type::new(
            &db,
            TypeKind::Continuation {
                arg: int,
                result: int,
                effect: open,
            },
        );

        assert!(!is_concrete_type(&db, direct_function_type(&db, open)));
        assert!(!is_concrete_type(&db, continuation));
    }

    #[test]
    fn concrete_type_rejects_error_types_in_effect_arguments() {
        let db = TestDb::default();
        let error = Type::new(&db, TypeKind::Error);
        let row = EffectRow::new(
            &db,
            vec![Effect {
                ability_id: AbilityId::source(&db, Symbol::new("State")),
                args: vec![error],
            }],
            None,
        );

        assert!(!is_concrete_type(&db, error));
        assert!(!is_concrete_type(&db, direct_function_type(&db, row)));
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

    fn nominal_module<'db>() -> Module<TypedRef<'db>> {
        Module {
            id: NodeId::from_raw(0),
            name: None,
            decls: [
                ("Option", 1),
                ("Result", 2),
                ("List", 1),
                ("Pair", 2),
                ("Plain", 0),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, (name, arity))| {
                Decl::Struct(crate::ast::StructDecl {
                    id: NodeId::from_raw(i + 1),
                    is_pub: false,
                    name: Symbol::new(name),
                    type_params: (0..arity)
                        .map(|p| crate::ast::TypeParamDecl {
                            id: NodeId::from_raw(100 + i * 2 + p),
                            name: Symbol::from_dynamic(&format!("t{p}")),
                            bounds: vec![],
                        })
                        .collect(),
                    fields: vec![],
                })
            })
            .collect(),
        }
    }

    fn nominal_id<'db>(
        index: &NominalIndex<'_, 'db>,
        db: &'db dyn salsa::Database,
        name: &str,
    ) -> TypeDefId<'db> {
        *index
            .declarations
            .keys()
            .find(|id| id.qualified(db).with_str(|s| s == name))
            .unwrap()
    }

    #[test]
    fn test_collect_type_retains_concrete_specialization() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let int = Type::new(&db, TypeKind::Int);
        let option_id = nominal_id(&index, &db, "Option");
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: option_id,
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int],
            },
        );

        let mut result = HashMap::new();
        collect_from_type(&db, option_int, &index, &mut result);

        assert_eq!(result.len(), 1);
        let option_insts = result.get(&option_id).unwrap();
        assert!(option_insts.contains(&vec![int]));
    }

    #[test]
    fn test_collect_type_skips_bound_var_specialization() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let option_id = nominal_id(&index, &db, "Option");
        let option_bound = Type::new(
            &db,
            TypeKind::Named {
                id: option_id,
                name: trunk_ir::Symbol::new("Option"),
                args: vec![Type::new(&db, TypeKind::BoundVar { index: 0 })],
            },
        );

        let mut result = HashMap::new();
        collect_from_type(&db, option_bound, &index, &mut result);

        assert!(result.is_empty());
    }

    #[test]
    fn test_collect_type_skips_local_bound_var_specialization() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let option_id = nominal_id(&index, &db, "Option");
        let option_bound = Type::new(
            &db,
            TypeKind::Named {
                id: option_id,
                name: trunk_ir::Symbol::new("Option"),
                args: vec![Type::new(
                    &db,
                    TypeKind::LocalBoundVar {
                        scope: NodeId::from_raw(0),
                        index: 0,
                    },
                )],
            },
        );

        let mut result = HashMap::new();
        collect_from_type(&db, option_bound, &index, &mut result);

        assert!(result.is_empty());
    }

    #[test]
    fn test_collect_type_retains_concrete_child_below_skipped_parent() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let result_id = nominal_id(&index, &db, "Result");
        let option_id = nominal_id(&index, &db, "Option");
        let int = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: option_id,
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int],
            },
        );
        let result_bound_option_int = Type::new(
            &db,
            TypeKind::Named {
                id: result_id,
                name: trunk_ir::Symbol::new("Result"),
                args: vec![Type::new(&db, TypeKind::BoundVar { index: 0 }), option_int],
            },
        );

        let mut result = HashMap::new();
        collect_from_type(&db, result_bound_option_int, &index, &mut result);

        assert!(!result.contains_key(&result_id));
        assert_eq!(result[&option_id], HashSet::from([vec![int]]));
    }

    #[test]
    fn test_collect_type_nested() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let int = Type::new(&db, TypeKind::Int);
        let option_id = nominal_id(&index, &db, "Option");
        let list_id = nominal_id(&index, &db, "List");
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

        let mut result = HashMap::new();
        collect_from_type(&db, list_option_int, &index, &mut result);

        assert_eq!(result.len(), 2);
        assert!(result[&option_id].contains(&vec![int]));
        assert!(result[&list_id].contains(&vec![option_int]));
    }

    #[test]
    fn test_collect_type_in_func_params() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let int = Type::new(&db, TypeKind::Int);
        let pair_id = nominal_id(&index, &db, "Pair");
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

        let mut result = HashMap::new();
        collect_from_type(&db, func_ty, &index, &mut result);

        assert_eq!(result.len(), 1);
        assert!(result[&pair_id].contains(&vec![int, int]));
    }

    #[test]
    fn test_collect_type_ignores_non_generic() {
        let db = TestDb::default();
        let module = nominal_module();
        let index = NominalIndex::new(&db, &module);
        let int = Type::new(&db, TypeKind::Int);
        let mut result = HashMap::new();
        // Neither an unknown type nor an indexed non-generic declaration qualifies,
        // even if the type carries arguments.
        for id in [
            TypeDefId::synthetic(&db, Symbol::new("Unknown")),
            nominal_id(&index, &db, "Plain"),
        ] {
            assert!(!index.is_generic(id));
            let ty = Type::new(
                &db,
                TypeKind::Named {
                    id,
                    name: id.qualified(&db),
                    args: vec![int],
                },
            );
            collect_from_type(&db, ty, &index, &mut result);
        }

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

use std::collections::{HashMap, HashSet};

use trunk_ir::Symbol;

use crate::ast::{
    Decl, Expr, ExprKind, FuncDefId, Module, ResolvedRef, Stmt, Type, TypeKind, TypeScheme,
    TypedRef,
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
) -> HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>> {
    let mut collector = InstantiationCollector::new(db, function_types);
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
        (TypeKind::Named { name: sn, args: sa }, TypeKind::Named { name: cn, args: ca }) => {
            if sn != cn || sa.len() != ca.len() {
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

struct InstantiationCollector<'db> {
    db: &'db dyn salsa::Database,
    schemes: HashMap<FuncDefId<'db>, TypeScheme<'db>>,
    instantiations: HashMap<FuncDefId<'db>, HashSet<Vec<Type<'db>>>>,
}

impl<'db> InstantiationCollector<'db> {
    fn new(
        db: &'db dyn salsa::Database,
        function_types: &[(trunk_ir::Symbol, TypeScheme<'db>)],
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
        }
    }

    fn try_record(&mut self, typed_ref: &TypedRef<'db>) {
        let ResolvedRef::Function { id } = &typed_ref.resolved else {
            return;
        };
        let Some(scheme) = self.schemes.get(id) else {
            return;
        };
        let Some(type_args) = extract_type_args(self.db, *scheme, typed_ref.ty) else {
            return;
        };
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
                self.try_record(typed_ref);
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
) -> HashMap<Symbol, HashSet<Vec<Type<'db>>>> {
    let generic_types = collect_generic_type_names(module);
    let mut result: HashMap<Symbol, HashSet<Vec<Type<'db>>>> = HashMap::new();

    let mut visitor = TypeInstantiationVisitor {
        db,
        generic_types: &generic_types,
        instantiations: &mut result,
    };
    visitor.visit_module(module);
    result
}

/// Collect names of all struct/enum declarations that have type parameters.
fn collect_generic_type_names(module: &Module<TypedRef<'_>>) -> HashSet<Symbol> {
    let mut names = HashSet::new();
    collect_generic_type_names_inner(&module.decls, &mut names);
    names
}

fn collect_generic_type_names_inner(decls: &[Decl<TypedRef<'_>>], names: &mut HashSet<Symbol>) {
    for decl in decls {
        match decl {
            Decl::Struct(s) if !s.type_params.is_empty() => {
                names.insert(s.name);
            }
            Decl::Enum(e) if !e.type_params.is_empty() => {
                names.insert(e.name);
            }
            Decl::Module(m) => {
                if let Some(body) = &m.body {
                    collect_generic_type_names_inner(body, names);
                }
            }
            _ => {}
        }
    }
}

/// Recursively walk a Type and collect all `Named { name, args }` where
/// `args` is non-empty and `name` is a known generic type.
fn collect_from_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    generic_types: &HashSet<Symbol>,
    result: &mut HashMap<Symbol, HashSet<Vec<Type<'db>>>>,
) {
    match ty.kind(db) {
        TypeKind::Named { name, args } => {
            if !args.is_empty() && generic_types.contains(name) {
                result.entry(*name).or_default().insert(args.clone());
            }
            // Recurse into type arguments (e.g., List(Option(Int)) → collect Option(Int))
            for arg in args {
                collect_from_type(db, *arg, generic_types, result);
            }
        }
        TypeKind::Func {
            params,
            result: ret,
            ..
        } => {
            for p in params {
                collect_from_type(db, *p, generic_types, result);
            }
            collect_from_type(db, *ret, generic_types, result);
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

struct TypeInstantiationVisitor<'a, 'db> {
    db: &'db dyn salsa::Database,
    generic_types: &'a HashSet<Symbol>,
    instantiations: &'a mut HashMap<Symbol, HashSet<Vec<Type<'db>>>>,
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
    use crate::ast::{EffectRow, TypeParam, TypeScheme};

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
        TypeScheme::new(db, type_params, body)
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
            },
        );
        let scheme = make_scheme(&db, 1, scheme_body);

        let int = Type::new(&db, TypeKind::Int);
        let text = Type::new(
            &db,
            TypeKind::Named {
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
            },
        );
        let scheme = make_scheme(&db, 1, scheme_body);

        let int = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
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
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int],
            },
        );

        let mut generic_types = HashSet::new();
        generic_types.insert(trunk_ir::Symbol::new("Option"));

        let mut result = HashMap::new();
        collect_from_type(&db, option_int, &generic_types, &mut result);

        assert_eq!(result.len(), 1);
        let option_insts = result.get(&trunk_ir::Symbol::new("Option")).unwrap();
        assert!(option_insts.contains(&vec![int]));
    }

    #[test]
    fn test_collect_type_nested() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int],
            },
        );
        let list_option_int = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("List"),
                args: vec![option_int],
            },
        );

        let mut generic_types = HashSet::new();
        generic_types.insert(trunk_ir::Symbol::new("Option"));
        generic_types.insert(trunk_ir::Symbol::new("List"));

        let mut result = HashMap::new();
        collect_from_type(&db, list_option_int, &generic_types, &mut result);

        assert_eq!(result.len(), 2);
        assert!(result[&trunk_ir::Symbol::new("Option")].contains(&vec![int]));
        assert!(result[&trunk_ir::Symbol::new("List")].contains(&vec![option_int]));
    }

    #[test]
    fn test_collect_type_in_func_params() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        let pair_int_int = Type::new(
            &db,
            TypeKind::Named {
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
            },
        );

        let mut generic_types = HashSet::new();
        generic_types.insert(trunk_ir::Symbol::new("Pair"));

        let mut result = HashMap::new();
        collect_from_type(&db, func_ty, &generic_types, &mut result);

        assert_eq!(result.len(), 1);
        assert!(result[&trunk_ir::Symbol::new("Pair")].contains(&vec![int, int]));
    }

    #[test]
    fn test_collect_type_ignores_non_generic() {
        let db = TestDb::default();
        let int = Type::new(&db, TypeKind::Int);
        // Named type with args but NOT in generic_types set
        let unknown = Type::new(
            &db,
            TypeKind::Named {
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
            },
        );
        let scheme_body = Type::new(
            &db,
            TypeKind::Func {
                params: vec![fn_param, bv0],
                result: bv1,
                effect: pure_effect(&db),
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
            },
        );
        let concrete = Type::new(
            &db,
            TypeKind::Func {
                params: vec![fn_concrete, int],
                result: bool_ty,
                effect: pure_effect(&db),
            },
        );

        let result = extract_type_args(&db, scheme, concrete);
        assert_eq!(result, Some(vec![int, bool_ty]));
    }
}

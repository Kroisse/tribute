use std::collections::{HashMap, HashSet};

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

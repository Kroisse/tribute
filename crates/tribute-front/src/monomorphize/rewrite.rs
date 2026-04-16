//! Call site rewriting for monomorphization.
//!
//! Rewrites references to generic functions with their specialized versions
//! by matching the callee's concrete type against collected instantiations.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{
    Arm, Decl, Expr, ExprKind, FuncDefId, HandlerArm, Module, ModuleDecl, ResolvedRef, Stmt, Type,
    TypeScheme, TypedRef,
};

use super::collect::extract_type_args;

/// Rewrite map: original FuncDefId → list of (type_args, mangled_name) pairs.
pub type RewriteMap<'db> = HashMap<FuncDefId<'db>, Vec<(Vec<Type<'db>>, Symbol)>>;

/// Rewrite all generic function call sites in a module to use specialized versions.
pub fn rewrite_module<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
    function_types: &[(Symbol, TypeScheme<'db>)],
    rewrite_map: &RewriteMap<'db>,
) -> Module<TypedRef<'db>> {
    let scheme_map: HashMap<Symbol, TypeScheme<'db>> = function_types.iter().cloned().collect();
    let mut rewriter = CallSiteRewriter {
        db,
        scheme_map,
        rewrite_map,
    };
    let decls = module
        .decls
        .into_iter()
        .map(|d| rewriter.rewrite_decl(d))
        .collect();
    Module::new(module.id, module.name, decls)
}

struct CallSiteRewriter<'a, 'db> {
    db: &'db dyn salsa::Database,
    scheme_map: HashMap<Symbol, TypeScheme<'db>>,
    rewrite_map: &'a RewriteMap<'db>,
}

impl<'a, 'db> CallSiteRewriter<'a, 'db> {
    fn try_rewrite_ref(&self, typed_ref: &TypedRef<'db>) -> Option<TypedRef<'db>> {
        let ResolvedRef::Function { id } = &typed_ref.resolved else {
            return None;
        };
        let entries = self.rewrite_map.get(id)?;
        let qualified = id.qualified(self.db);
        let scheme = self.scheme_map.get(&qualified)?;
        let type_args = extract_type_args(self.db, *scheme, typed_ref.ty)?;

        // Find the matching mangled name
        let mangled = entries.iter().find_map(|(args, name)| {
            if args == &type_args {
                Some(*name)
            } else {
                None
            }
        })?;

        let specialized_id = FuncDefId::new(self.db, mangled);
        Some(TypedRef::new(
            ResolvedRef::Function { id: specialized_id },
            typed_ref.ty,
        ))
    }

    fn rewrite_typed_ref(&self, tr: TypedRef<'db>) -> TypedRef<'db> {
        self.try_rewrite_ref(&tr).unwrap_or(tr)
    }

    fn rewrite_decl(&mut self, decl: Decl<TypedRef<'db>>) -> Decl<TypedRef<'db>> {
        match decl {
            Decl::Function(mut func) => {
                func.body = self.rewrite_expr(func.body);
                Decl::Function(func)
            }
            Decl::Module(m) => {
                let body = m
                    .body
                    .map(|decls| decls.into_iter().map(|d| self.rewrite_decl(d)).collect());
                Decl::Module(ModuleDecl {
                    id: m.id,
                    name: m.name,
                    is_pub: m.is_pub,
                    body,
                })
            }
            other => other,
        }
    }

    fn rewrite_expr(&mut self, expr: Expr<TypedRef<'db>>) -> Expr<TypedRef<'db>> {
        let kind = match *expr.kind {
            ExprKind::Var(tr) => ExprKind::Var(self.rewrite_typed_ref(tr)),
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: self.rewrite_expr(callee),
                args: args.into_iter().map(|a| self.rewrite_expr(a)).collect(),
            },
            ExprKind::Block { stmts, value } => ExprKind::Block {
                stmts: stmts.into_iter().map(|s| self.rewrite_stmt(s)).collect(),
                value: self.rewrite_expr(value),
            },
            ExprKind::Case { scrutinee, arms } => ExprKind::Case {
                scrutinee: self.rewrite_expr(scrutinee),
                arms: arms.into_iter().map(|a| self.rewrite_arm(a)).collect(),
            },
            ExprKind::Lambda { params, body } => ExprKind::Lambda {
                params,
                body: self.rewrite_expr(body),
            },
            ExprKind::Handle { body, handlers } => ExprKind::Handle {
                body: self.rewrite_expr(body),
                handlers: handlers
                    .into_iter()
                    .map(|h| self.rewrite_handler_arm(h))
                    .collect(),
            },
            ExprKind::Resume { arg, local_id } => ExprKind::Resume {
                arg: self.rewrite_expr(arg),
                local_id,
            },
            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor,
                args: args.into_iter().map(|a| self.rewrite_expr(a)).collect(),
            },
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => ExprKind::Record {
                type_name,
                fields: fields
                    .into_iter()
                    .map(|(name, e)| (name, self.rewrite_expr(e)))
                    .collect(),
                spread: spread.map(|s| self.rewrite_expr(s)),
            },
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => ExprKind::MethodCall {
                receiver: self.rewrite_expr(receiver),
                method,
                args: args.into_iter().map(|a| self.rewrite_expr(a)).collect(),
            },
            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: self.rewrite_expr(lhs),
                rhs: self.rewrite_expr(rhs),
            },
            ExprKind::Tuple(es) => {
                ExprKind::Tuple(es.into_iter().map(|e| self.rewrite_expr(e)).collect())
            }
            ExprKind::List(es) => {
                ExprKind::List(es.into_iter().map(|e| self.rewrite_expr(e)).collect())
            }
            // Leaf nodes
            ExprKind::NatLit(v) => ExprKind::NatLit(v),
            ExprKind::IntLit(v) => ExprKind::IntLit(v),
            ExprKind::FloatLit(v) => ExprKind::FloatLit(v),
            ExprKind::StringLit(v) => ExprKind::StringLit(v),
            ExprKind::BytesLit(v) => ExprKind::BytesLit(v),
            ExprKind::BoolLit(v) => ExprKind::BoolLit(v),
            ExprKind::RuneLit(v) => ExprKind::RuneLit(v),
            ExprKind::Nil => ExprKind::Nil,
            ExprKind::Error => ExprKind::Error,
        };
        Expr::new(expr.id, kind)
    }

    fn rewrite_stmt(&mut self, stmt: Stmt<TypedRef<'db>>) -> Stmt<TypedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                ty,
                value,
            } => Stmt::Let {
                id,
                pattern,
                ty,
                value: self.rewrite_expr(value),
            },
            Stmt::Expr { id, expr } => Stmt::Expr {
                id,
                expr: self.rewrite_expr(expr),
            },
        }
    }

    fn rewrite_arm(&mut self, arm: Arm<TypedRef<'db>>) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: arm.pattern,
            guard: arm.guard.map(|g| self.rewrite_expr(g)),
            body: self.rewrite_expr(arm.body),
        }
    }

    fn rewrite_handler_arm(&mut self, arm: HandlerArm<TypedRef<'db>>) -> HandlerArm<TypedRef<'db>> {
        HandlerArm {
            id: arm.id,
            kind: arm.kind,
            body: self.rewrite_expr(arm.body),
        }
    }
}

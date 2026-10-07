//! Placement and operand rules for `become`.
//!
//! A `become` call transfers the enclosing callable's frame, so it must be in
//! that callable's tail position, outside any installed handler, and must call
//! a function whose frame can take over. These rules are syntactic apart from
//! the callee's declaration kind and are checked over the resolved body before
//! inference. The result-type rule is a typing constraint (see `expr.rs`).

use crate::ast::visit::{Visit, walk_expr};
use crate::ast::{Expr, ExprKind, FuncDefId, NodeId, ResolvedRef};

use super::TypeChecker;

impl<'db> TypeChecker<'db> {
    /// Report every `become` in a function body that is out of tail
    /// position, inside a handler, or applied to something other than a
    /// transferable function call.
    pub(crate) fn check_become_sites(&self, body: &Expr<ResolvedRef<'db>>) {
        let mut walker = BecomeSites {
            checker: self,
            tail: true,
            handler_depth: 0,
        };
        walker.visit_expr(body);
    }

    /// Report a `become` whose callee is an `extern` function.
    pub(crate) fn check_become_callee(&self, node: NodeId, callee: FuncDefId<'db>) {
        if self.env.is_extern_function(callee) {
            self.report_type_error(
                node,
                format!(
                    "`become` cannot call extern function `{}`, which uses a foreign calling convention",
                    callee.name(self.db())
                ),
            );
        }
    }
}

struct BecomeSites<'a, 'db> {
    checker: &'a TypeChecker<'db>,
    /// Whether the expression about to be visited is in tail position of
    /// the innermost callable.
    tail: bool,
    /// Handle expressions entered since the innermost callable.
    handler_depth: usize,
}

impl<'db> BecomeSites<'_, 'db> {
    fn check_operand(&self, become_id: NodeId, call: &Expr<ResolvedRef<'db>>) {
        let message = match &*call.kind {
            ExprKind::Call { callee, .. } => match &*callee.kind {
                ExprKind::Var(ResolvedRef::Function { id }) => {
                    self.checker.check_become_callee(become_id, *id);
                    None
                }
                ExprKind::Var(ResolvedRef::Constructor { .. }) => {
                    Some("`become` cannot call a constructor")
                }
                ExprKind::Var(ResolvedRef::AbilityOp { .. }) => {
                    Some("`become` cannot perform an ability operation")
                }
                _ => None,
            },
            // Method selection decides the callee; see the `become` arm of
            // expression checking.
            ExprKind::MethodCall { .. } | ExprKind::Error => None,
            ExprKind::Cons { .. } | ExprKind::Record { .. } => {
                Some("`become` cannot call a constructor")
            }
            ExprKind::Resume { .. } => Some("`become` cannot be applied to `resume`"),
            _ => Some("`become` requires a function call with an argument list"),
        };
        if let Some(message) = message {
            self.checker
                .report_type_error(become_id, message.to_owned());
        }
    }
}

impl<'ast, 'db: 'ast> Visit<'ast, ResolvedRef<'db>> for BecomeSites<'_, 'db> {
    fn visit_expr(&mut self, expr: &'ast Expr<ResolvedRef<'db>>) {
        let tail = std::mem::replace(&mut self.tail, false);
        match &*expr.kind {
            ExprKind::Become { call } => {
                if self.handler_depth > 0 {
                    self.checker.report_type_error(
                        expr.id,
                        "`become` cannot be used inside a `handle` body or handler arm, \
                         which run with the handler installed"
                            .to_owned(),
                    );
                } else if !tail {
                    self.checker.report_type_error(
                        expr.id,
                        "`become` must be in tail position of the enclosing function or lambda"
                            .to_owned(),
                    );
                }
                self.check_operand(expr.id, call);
                self.visit_expr(call);
            }
            ExprKind::Block { stmts, value } => {
                for stmt in stmts {
                    self.visit_stmt(stmt);
                }
                self.tail = tail;
                self.visit_expr(value);
            }
            ExprKind::Case { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.visit_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.tail = tail;
                    self.visit_expr(&arm.body);
                }
            }
            ExprKind::Lambda { body, .. } => {
                let handler_depth = std::mem::replace(&mut self.handler_depth, 0);
                self.tail = true;
                self.visit_expr(body);
                self.handler_depth = handler_depth;
            }
            ExprKind::Handle { .. } => {
                self.handler_depth += 1;
                walk_expr(self, expr);
                self.handler_depth -= 1;
            }
            _ => walk_expr(self, expr),
        }
        self.tail = false;
    }
}

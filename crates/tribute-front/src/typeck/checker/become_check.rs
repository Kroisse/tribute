//! Placement and operand rules for `become`.
//!
//! A `become` transfers the enclosing callable's frame, so it must be in
//! that callable's tail position and apply to a function call whose frame
//! can take over, a `resume`, or a `handle`. A `handle` in tail position
//! makes the results of its `do` and `op` arms tail positions; its body and
//! `fn` arms are not. These rules are syntactic apart from the callee's
//! declaration kind and are checked over the resolved body before inference.
//! The result-type rule is a typing constraint (see `expr.rs`).

use crate::ast::visit::{Visit, walk_expr};
use crate::ast::{Expr, ExprKind, FuncDefId, HandlerKind, NodeId, ResolvedRef};

use super::TypeChecker;

impl<'db> TypeChecker<'db> {
    /// Report every `become` in a function body that is out of tail
    /// position or applied to something other than a transferable function
    /// call, a `resume`, or a `handle`.
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
    /// `handle` bodies and `fn` arms entered since the innermost callable.
    handler_depth: usize,
}

impl<'db> BecomeSites<'_, 'db> {
    fn check_operand(&self, become_id: NodeId, operand: &Expr<ResolvedRef<'db>>) {
        let message = match &*operand.kind {
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
                // Selection among several functions decides the callee; see
                // the `become` arm of expression checking.
                _ => None,
            },
            ExprKind::Error | ExprKind::Resume { .. } | ExprKind::Handle { .. } => None,
            ExprKind::Cons { .. } | ExprKind::Record { .. } => {
                Some("`become` cannot call a constructor")
            }
            _ => Some(
                "`become` requires a function call with an argument list, a `resume`, or a `handle`",
            ),
        };
        if let Some(message) = message {
            self.checker
                .report_type_error(become_id, message.to_owned());
        }
    }

    /// Visit `expr` as a region that runs with a handler installed or
    /// returns to a `perform`, so nothing in it is a tail position.
    fn visit_in_handler(&mut self, expr: &Expr<ResolvedRef<'db>>) {
        self.handler_depth += 1;
        self.visit_expr(expr);
        self.handler_depth -= 1;
    }
}

impl<'ast, 'db: 'ast> Visit<'ast, ResolvedRef<'db>> for BecomeSites<'_, 'db> {
    fn visit_expr(&mut self, expr: &'ast Expr<ResolvedRef<'db>>) {
        let tail = std::mem::replace(&mut self.tail, false);
        match &*expr.kind {
            ExprKind::Become { call } => {
                if !tail {
                    let message = if self.handler_depth > 0 {
                        "`become` cannot be used in a `handle` body, which runs with the \
                         handler installed, or in an `fn` arm, which returns to its `perform`"
                    } else {
                        "`become` must be in tail position of the enclosing function or lambda"
                    };
                    self.checker.report_type_error(expr.id, message.to_owned());
                }
                self.check_operand(expr.id, call);
                // `become handle ..` keeps the handle in tail position.
                self.tail = tail;
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
            ExprKind::Handle { body, handlers } => {
                self.visit_in_handler(body);
                for arm in handlers {
                    match &arm.kind {
                        HandlerKind::Do { binding } => self.visit_pattern(binding),
                        HandlerKind::Fn { params, .. } | HandlerKind::Op { params, .. } => {
                            for param in params {
                                self.visit_pattern(param);
                            }
                        }
                    }
                    if matches!(arm.kind, HandlerKind::Fn { .. }) {
                        self.visit_in_handler(&arm.body);
                    } else {
                        self.tail = tail;
                        self.visit_expr(&arm.body);
                    }
                }
            }
            _ => walk_expr(self, expr),
        }
        self.tail = false;
    }
}

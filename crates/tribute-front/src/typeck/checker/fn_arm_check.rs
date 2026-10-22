//! The suspension rule for `fn` handler arms.
//!
//! A `fn` operation returns to its perform site without handing over the
//! continuation, so the arm that implements it cannot suspend or abort that
//! computation (see `new-plans/abilities.md`, "`fn` Operation Handler"). The
//! arm may not perform an `op` operation directly, and may not call a
//! callable whose ability row names an ability that declares one, unless a
//! `handle` inside the arm handles it. The rule is checked over the finalized
//! body, where calls are resolved and their types settled.

use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};

use crate::ast::visit::{Visit, walk_expr, walk_handler_arm};
use crate::ast::{
    AbilityId, Expr, ExprKind, HandlerArm, HandlerKind, NodeId, OpDeclKind, ResolvedRef, Type,
    TypeKind, TypedRef,
};

use super::TypeChecker;

impl<'db> TypeChecker<'db> {
    /// Report every `fn` handler arm in a finalized function body that
    /// performs an `op` operation or calls a callable that may.
    pub(crate) fn check_fn_arms(
        &self,
        body: &Expr<TypedRef<'db>>,
        node_types: &HashMap<NodeId, Type<'db>>,
    ) {
        FnArms {
            checker: self,
            node_types,
        }
        .visit_expr(body);
    }

    fn report_fn_arm(
        &self,
        arm: &HandlerArm<TypedRef<'db>>,
        node_types: &HashMap<NodeId, Type<'db>>,
    ) {
        let HandlerKind::Fn { ability, op, .. } = &arm.kind else {
            return;
        };
        let mut calls = ArmCalls {
            checker: self,
            node_types,
            handled: Vec::new(),
            suspending: Vec::new(),
        };
        calls.visit_expr(&arm.body);
        if calls.suspending.is_empty() {
            return;
        }
        let Some(ability) = self.extract_ability_id_from_ref(&ability.resolved) else {
            return;
        };
        let ability = ability.name(self.db());
        Diagnostic::new(
            format!(
                "`fn` handler arm for `{ability}::{op}` cannot perform effects that suspend or abort: {}",
                calls
                    .suspending
                    .iter()
                    .format_with(", ", |id, f| f(&id.name(self.db())))
            ),
            self.get_span(arm.id),
            DiagnosticSeverity::Error,
            CompilationPhase::TypeChecking,
        )
        .accumulate(self.db());
    }

    /// Whether an ability declares an `op` operation, which may suspend.
    fn declares_op_operation(&self, ability: AbilityId<'db>) -> bool {
        self.env.lookup_ability(ability).is_some_and(|info| {
            info.operations
                .values()
                .any(|operation| operation.kind == OpDeclKind::Op)
        })
    }
}

/// Finds the `fn` arms of a body, including those nested in other arms.
struct FnArms<'a, 'db> {
    checker: &'a TypeChecker<'db>,
    node_types: &'a HashMap<NodeId, Type<'db>>,
}

impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for FnArms<'_, 'db> {
    fn visit_handler_arm(&mut self, arm: &'ast HandlerArm<TypedRef<'db>>) {
        self.checker.report_fn_arm(arm, self.node_types);
        walk_handler_arm(self, arm);
    }
}

/// Collects the suspending abilities one `fn` arm body reaches.
struct ArmCalls<'a, 'db> {
    checker: &'a TypeChecker<'db>,
    node_types: &'a HashMap<NodeId, Type<'db>>,
    /// Abilities handled by the `handle` expressions entered inside the arm.
    handled: Vec<AbilityId<'db>>,
    /// Suspending abilities the arm reaches, in the order first reached.
    suspending: Vec<AbilityId<'db>>,
}

impl<'db> ArmCalls<'_, 'db> {
    fn reach(&mut self, ability: AbilityId<'db>) {
        if !self.handled.contains(&ability) && !self.suspending.contains(&ability) {
            self.suspending.push(ability);
        }
    }

    fn check_call(&mut self, callee: &Expr<TypedRef<'db>>) {
        if let ExprKind::Var(TypedRef {
            resolved: ResolvedRef::AbilityOp { ability, kind, .. },
            ..
        }) = &*callee.kind
        {
            if *kind == OpDeclKind::Op {
                self.reach(*ability);
            }
            return;
        }
        let callee_ty = match &*callee.kind {
            ExprKind::Var(reference) => Some(reference.ty),
            _ => self.node_types.get(&callee.id).copied(),
        };
        let Some(callee_ty) = callee_ty else {
            return;
        };
        let TypeKind::Func { effect, .. } = callee_ty.kind(self.checker.db()) else {
            return;
        };
        for instance in effect.instances(self.checker.db()) {
            if self.checker.declares_op_operation(instance.ability_id) {
                self.reach(instance.ability_id);
            }
        }
    }
}

impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for ArmCalls<'_, 'db> {
    fn visit_expr(&mut self, expr: &'ast Expr<TypedRef<'db>>) {
        match &*expr.kind {
            // A lambda's effects happen where it is called.
            ExprKind::Lambda { .. } => {}
            ExprKind::Handle { body, handlers } => {
                let depth = self.handled.len();
                for handler in handlers {
                    if let HandlerKind::Fn { ability, .. } | HandlerKind::Op { ability, .. } =
                        &handler.kind
                        && let Some(id) =
                            self.checker.extract_ability_id_from_ref(&ability.resolved)
                    {
                        self.handled.push(id);
                    }
                }
                self.visit_expr(body);
                self.handled.truncate(depth);
                // A nested `fn` arm is checked on its own; the other arms run
                // where this `handle` was installed, inside the outer arm.
                for handler in handlers {
                    if !matches!(handler.kind, HandlerKind::Fn { .. }) {
                        self.visit_expr(&handler.body);
                    }
                }
            }
            ExprKind::Call { callee, .. } => {
                self.check_call(callee);
                walk_expr(self, expr);
            }
            _ => walk_expr(self, expr),
        }
    }
}

//! Random AST generators for property-based testing.
//!
//! Provides `proptest` strategy functions that generate syntactically valid
//! Tribute AST nodes. The initial scope covers literals and arithmetic
//! expressions; more complex forms (variables, lambdas, case, etc.) will be
//! added in follow-up work.

use proptest::prelude::*;

use super::{BinOpKind, Decl, Expr, ExprKind, FloatBits, FuncDecl, Module, NodeId, UnresolvedName};
use trunk_ir::Symbol;

/// Dummy [`NodeId`] used by all generated nodes.
///
/// Property tests don't need real span information, and `NodeId::from_raw`
/// is `#[cfg(test)]`-gated, so this is fine.
fn dummy_id() -> NodeId {
    NodeId::from_raw(0)
}

// ============================================================================
// Leaf strategies
// ============================================================================

/// Strategy for arithmetic binary operators only.
pub fn arb_arith_op() -> impl Strategy<Value = BinOpKind> {
    prop_oneof![
        Just(BinOpKind::Add),
        Just(BinOpKind::Sub),
        Just(BinOpKind::Mul),
        Just(BinOpKind::Div),
        Just(BinOpKind::Mod),
    ]
}

/// Strategy for all `BinOpKind` variants.
pub fn arb_binop_kind() -> impl Strategy<Value = BinOpKind> {
    prop_oneof![
        Just(BinOpKind::Add),
        Just(BinOpKind::Sub),
        Just(BinOpKind::Mul),
        Just(BinOpKind::Div),
        Just(BinOpKind::Mod),
        Just(BinOpKind::Eq),
        Just(BinOpKind::Ne),
        Just(BinOpKind::Lt),
        Just(BinOpKind::Le),
        Just(BinOpKind::Gt),
        Just(BinOpKind::Ge),
        Just(BinOpKind::And),
        Just(BinOpKind::Or),
        Just(BinOpKind::Concat),
    ]
}

/// Strategy that produces literal expressions (leaf nodes).
///
/// Covers: `NatLit`, `IntLit`, `FloatLit`, `BoolLit`, `Nil`.
pub fn arb_literal_expr() -> impl Strategy<Value = Expr<UnresolvedName>> {
    prop_oneof![
        // NatLit: 0..=u32::MAX keeps values reasonable
        (0u64..=u32::MAX as u64).prop_map(|n| Expr::new(dummy_id(), ExprKind::NatLit(n))),
        // IntLit
        (i32::MIN as i64..=i32::MAX as i64)
            .prop_map(|n| Expr::new(dummy_id(), ExprKind::IntLit(n))),
        // FloatLit: finite values only (NaN/Inf testing is a follow-up)
        (-1e15f64..1e15f64)
            .prop_map(|f| Expr::new(dummy_id(), ExprKind::FloatLit(FloatBits::new(f)))),
        // BoolLit
        any::<bool>().prop_map(|b| Expr::new(dummy_id(), ExprKind::BoolLit(b))),
        // Nil
        Just(Expr::new(dummy_id(), ExprKind::Nil)),
    ]
}

// ============================================================================
// Recursive expression strategy
// ============================================================================

/// Strategy that produces arbitrary expressions up to `depth` levels deep.
///
/// Uses `prop_recursive` to build trees of binary operations on top of
/// literal leaves.
///
/// * `depth` – maximum nesting depth
/// * `desired_size` – target number of nodes
/// * `expected_branch_size` – expected children per recursive node
pub fn arb_expr(
    depth: u32,
    desired_size: u32,
    expected_branch_size: u32,
) -> impl Strategy<Value = Expr<UnresolvedName>> {
    arb_literal_expr().prop_recursive(depth, desired_size, expected_branch_size, |inner| {
        (arb_binop_kind(), inner.clone(), inner)
            .prop_map(|(op, lhs, rhs)| Expr::new(dummy_id(), ExprKind::BinOp { op, lhs, rhs }))
    })
}

/// Strategy with reasonable defaults: depth 8, size 64, branch 3.
pub fn arb_expr_default() -> impl Strategy<Value = Expr<UnresolvedName>> {
    arb_expr(8, 64, 3)
}

// ============================================================================
// Module-level strategy
// ============================================================================

/// Wraps an expression into a `fn main() { expr }` parsed module.
pub fn arb_parsed_module() -> impl Strategy<Value = Module<UnresolvedName>> {
    arb_expr_default().prop_map(|body| {
        let main_fn = FuncDecl {
            id: dummy_id(),
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![],
            return_ty: None,
            effects: None,
            body,
        };
        Module::new(dummy_id(), None, vec![Decl::Function(main_fn)])
    })
}

// ============================================================================
// Helper: expression depth measurement
// ============================================================================

/// Compute the depth of an expression tree (1 for leaves).
fn expr_depth(expr: &Expr<UnresolvedName>) -> usize {
    match expr.kind.as_ref() {
        ExprKind::BinOp { lhs, rhs, .. } => 1 + expr_depth(lhs).max(expr_depth(rhs)),
        _ => 1,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        /// Generated expressions implement Clone, Debug, PartialEq correctly.
        #[test]
        fn generated_expr_is_valid(expr in arb_expr_default()) {
            // Clone + PartialEq
            let cloned = expr.clone();
            prop_assert_eq!(&expr, &cloned);
            // Debug doesn't panic
            let _ = format!("{:?}", expr);
        }

        /// The depth of generated expressions is bounded by the configured maximum.
        ///
        /// `prop_recursive(depth=4, ...)` limits recursive nesting, so adding 1
        /// for the leaf level means total depth ≤ 5.
        #[test]
        fn depth_is_bounded(expr in arb_expr(4, 32, 3)) {
            let depth = expr_depth(&expr);
            prop_assert!(depth <= 5, "depth {} exceeded maximum 5", depth);
        }

        /// The literal-only strategy always produces depth-1 (leaf) expressions.
        #[test]
        fn literals_are_leaves(expr in arb_literal_expr()) {
            prop_assert_eq!(expr_depth(&expr), 1);
        }

        /// Wrapping into a module never panics.
        #[test]
        fn module_construction_does_not_panic(_module in arb_parsed_module()) {
            // reaching here without panic is the assertion
        }
    }
}

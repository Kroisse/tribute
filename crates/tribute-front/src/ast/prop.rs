//! Random AST generators for property-based testing.
//!
//! Provides `proptest` strategy functions that generate syntactically valid
//! Tribute AST nodes. Phase 1 covered literals and arithmetic expressions;
//! Phase 2 adds `Block`, `Let`, and `Var` with scope-correct variable
//! references via a parameterized `env_size` approach.

use proptest::prelude::*;

use super::{
    BinOpKind, Decl, Expr, ExprKind, FloatBits, FuncDecl, Module, NodeId, Pattern, PatternKind,
    Stmt, UnresolvedName,
};
use trunk_ir::Symbol;

/// Fixed pool of variable names. `env_size` indexes into this array:
/// `env_size=2` means `VAR_NAMES[0..2]` ("foo", "bar") are in scope.
const VAR_NAMES: &[&str] = &[
    "foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply",
];

/// Strategy for a random [`NodeId`], excluded from shrinking.
///
/// Uses `u32` for architecture-stable generation (consistent across 32/64-bit).
fn node_id() -> impl Strategy<Value = NodeId> {
    any::<u32>()
        .no_shrink()
        .prop_map(|n| NodeId::from_raw(n as usize))
}

// ============================================================================
// Leaf strategies
// ============================================================================

/// Strategy for arithmetic binary operators only.
pub fn arith_op() -> impl Strategy<Value = BinOpKind> {
    prop_oneof![
        Just(BinOpKind::Add),
        Just(BinOpKind::Sub),
        Just(BinOpKind::Mul),
        Just(BinOpKind::Div),
        Just(BinOpKind::Mod),
    ]
}

/// Strategy for all `BinOpKind` variants.
pub fn binop_kind() -> impl Strategy<Value = BinOpKind> {
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
pub fn literal_expr() -> impl Strategy<Value = Expr<UnresolvedName>> {
    prop_oneof![
        // NatLit: 0..=u32::MAX keeps values reasonable
        (node_id(), 0u64..=u32::MAX as u64).prop_map(|(id, n)| Expr::new(id, ExprKind::NatLit(n))),
        // IntLit
        (node_id(), i32::MIN as i64..=i32::MAX as i64)
            .prop_map(|(id, n)| Expr::new(id, ExprKind::IntLit(n))),
        // FloatLit: finite values only (NaN/Inf testing is a follow-up)
        (node_id(), -1e15f64..1e15f64)
            .prop_map(|(id, f)| Expr::new(id, ExprKind::FloatLit(FloatBits::new(f)))),
        // BoolLit
        (node_id(), any::<bool>()).prop_map(|(id, b)| Expr::new(id, ExprKind::BoolLit(b))),
        // Nil
        node_id().prop_map(|id| Expr::new(id, ExprKind::Nil)),
    ]
}

// ============================================================================
// Variable reference strategy
// ============================================================================

/// Strategy for a variable reference picking from `VAR_NAMES[0..env_size]`.
///
/// # Panics
/// Panics if `env_size == 0` (no names in scope).
fn var_expr(env_size: usize) -> BoxedStrategy<Expr<UnresolvedName>> {
    assert!(env_size > 0, "var_expr requires env_size > 0");
    (node_id(), node_id(), 0..env_size)
        .prop_map(|(expr_id, name_id, idx)| {
            let name = UnresolvedName::simple(Symbol::new(VAR_NAMES[idx]), name_id);
            Expr::new(expr_id, ExprKind::Var(name))
        })
        .boxed()
}

/// Leaf strategy that may include variable references when names are in scope.
fn leaf_expr(env_size: usize) -> BoxedStrategy<Expr<UnresolvedName>> {
    if env_size == 0 {
        literal_expr().boxed()
    } else {
        prop_oneof![
            8 => literal_expr(),
            2 => var_expr(env_size),
        ]
        .boxed()
    }
}

// ============================================================================
// Let statement helper
// ============================================================================

/// Strategy for a `Stmt::Let` binding with the given name and RHS strategy.
fn let_stmt(
    name: &'static str,
    rhs: BoxedStrategy<Expr<UnresolvedName>>,
) -> BoxedStrategy<Stmt<UnresolvedName>> {
    (node_id(), node_id(), rhs)
        .prop_map(move |(stmt_id, pat_id, value)| {
            let pattern = Pattern::new(
                pat_id,
                PatternKind::Bind {
                    name: Symbol::new(name),
                    local_id: None,
                },
            );
            Stmt::Let {
                id: stmt_id,
                pattern,
                ty: None,
                value,
            }
        })
        .boxed()
}

// ============================================================================
// Recursive expression strategy (parameterized by env_size)
// ============================================================================

/// Strategy for a BinOp node at the given env/depth.
fn binop_variant(env_size: usize, max_depth: usize) -> BoxedStrategy<Expr<UnresolvedName>> {
    let child = expr_with_env(env_size, max_depth - 1);
    (node_id(), binop_kind(), child.clone(), child)
        .prop_map(|(id, op, lhs, rhs)| Expr::new(id, ExprKind::BinOp { op, lhs, rhs }))
        .boxed()
}

/// Strategy for a Block expression with 1–3 let-stmts.
///
/// Each let-stmt binds `VAR_NAMES[env_size + i]`, extending the scope for
/// subsequent stmts and the final value expression.
fn block_variant(
    env_size: usize,
    max_depth: usize,
    remaining: usize,
) -> BoxedStrategy<Expr<UnresolvedName>> {
    let child_depth = max_depth - 1;

    // Build variants for 1, 2, and 3 let-stmts (capped by remaining pool).
    let mut variants: Vec<BoxedStrategy<Expr<UnresolvedName>>> = Vec::new();

    // --- 1 let-stmt ---
    if remaining >= 1 {
        let s0_name = VAR_NAMES[env_size];
        let s0_rhs = expr_with_env(env_size, child_depth);
        let s0 = let_stmt(s0_name, s0_rhs);
        let val = expr_with_env(env_size + 1, child_depth);
        variants.push(
            (node_id(), s0, val)
                .prop_map(|(block_id, stmt0, value)| {
                    Expr::new(
                        block_id,
                        ExprKind::Block {
                            stmts: vec![stmt0],
                            value,
                        },
                    )
                })
                .boxed(),
        );
    }

    // --- 2 let-stmts ---
    if remaining >= 2 {
        let s0_name = VAR_NAMES[env_size];
        let s1_name = VAR_NAMES[env_size + 1];
        let s0_rhs = expr_with_env(env_size, child_depth);
        let s1_rhs = expr_with_env(env_size + 1, child_depth);
        let s0 = let_stmt(s0_name, s0_rhs);
        let s1 = let_stmt(s1_name, s1_rhs);
        let val = expr_with_env(env_size + 2, child_depth);
        variants.push(
            (node_id(), s0, s1, val)
                .prop_map(|(block_id, stmt0, stmt1, value)| {
                    Expr::new(
                        block_id,
                        ExprKind::Block {
                            stmts: vec![stmt0, stmt1],
                            value,
                        },
                    )
                })
                .boxed(),
        );
    }

    // --- 3 let-stmts (use nested tuples to stay within proptest 10-element limit) ---
    if remaining >= 3 {
        let s0_name = VAR_NAMES[env_size];
        let s1_name = VAR_NAMES[env_size + 1];
        let s2_name = VAR_NAMES[env_size + 2];
        let s0_rhs = expr_with_env(env_size, child_depth);
        let s1_rhs = expr_with_env(env_size + 1, child_depth);
        let s2_rhs = expr_with_env(env_size + 2, child_depth);
        let s0 = let_stmt(s0_name, s0_rhs);
        let s1 = let_stmt(s1_name, s1_rhs);
        let s2 = let_stmt(s2_name, s2_rhs);
        let val = expr_with_env(env_size + 3, child_depth);
        variants.push(
            ((node_id(), s0, s1), (s2, val))
                .prop_map(|((block_id, stmt0, stmt1), (stmt2, value))| {
                    Expr::new(
                        block_id,
                        ExprKind::Block {
                            stmts: vec![stmt0, stmt1, stmt2],
                            value,
                        },
                    )
                })
                .boxed(),
        );
    }

    match variants.len() {
        1 => variants.pop().unwrap(),
        2 => {
            let [a, b] = <[_; 2]>::try_from(variants).unwrap();
            prop_oneof![a, b].boxed()
        }
        3 => {
            let [a, b, c] = <[_; 3]>::try_from(variants).unwrap();
            prop_oneof![a, b, c].boxed()
        }
        _ => unreachable!("block_variant called with remaining=0"),
    }
}

/// Core recursive strategy parameterized by environment size and nesting budget.
///
/// * `env_size` — number of names from `VAR_NAMES` currently in scope.
/// * `max_depth` — remaining nesting budget (0 → leaf only; the resulting
///   tree depth is at most `max_depth + 1`).
fn expr_with_env(env_size: usize, max_depth: usize) -> BoxedStrategy<Expr<UnresolvedName>> {
    let leaf = leaf_expr(env_size);
    if max_depth == 0 {
        return leaf;
    }

    let remaining = VAR_NAMES.len().saturating_sub(env_size);

    if remaining == 0 {
        // Pool exhausted: only leaf and binop variants.
        prop_oneof![
            4 => leaf,
            3 => binop_variant(env_size, max_depth),
        ]
        .boxed()
    } else {
        prop_oneof![
            4 => leaf,
            3 => binop_variant(env_size, max_depth),
            3 => block_variant(env_size, max_depth, remaining),
        ]
        .boxed()
    }
}

// ============================================================================
// Public expression strategies
// ============================================================================

/// Strategy that produces arbitrary expressions with the given nesting budget.
///
/// Uses manual recursion with `expr_with_env` to generate scope-correct
/// expressions including `Block`, `Let`, and `Var`.
///
/// * `depth` – nesting budget; the generated tree can be up to `depth + 1`
///   levels deep (a budget of 0 yields a leaf with depth 1).
/// * `_desired_size` – kept for API compatibility (unused with manual recursion)
/// * `_expected_branch_size` – kept for API compatibility (unused)
pub fn expr(
    depth: u32,
    _desired_size: u32,
    _expected_branch_size: u32,
) -> impl Strategy<Value = Expr<UnresolvedName>> {
    expr_with_env(0, depth as usize)
}

/// Strategy with reasonable defaults: budget 5 (max depth 6), empty initial scope.
///
/// Manual recursion builds the strategy tree eagerly, so the budget is kept
/// moderate to avoid exponential construction cost.
pub fn expr_default() -> impl Strategy<Value = Expr<UnresolvedName>> {
    expr_with_env(0, 5)
}

// ============================================================================
// Module-level strategy
// ============================================================================

/// Wraps an expression into a `fn main() { expr }` parsed module.
pub fn parsed_module() -> impl Strategy<Value = Module<UnresolvedName>> {
    (node_id(), node_id(), expr_default()).prop_map(|(module_id, func_id, body)| {
        let main_fn = FuncDecl {
            id: func_id,
            is_pub: false,
            name: Symbol::new("main"),
            type_params: vec![],
            params: vec![],
            return_ty: None,
            effects: None,
            body,
        };
        Module::new(module_id, None, vec![Decl::Function(main_fn)])
    })
}

// ============================================================================
// Helper: expression depth measurement
// ============================================================================

/// Compute the depth of an expression tree (1 for leaves).
fn expr_depth(expr: &Expr<UnresolvedName>) -> usize {
    match expr.kind.as_ref() {
        ExprKind::BinOp { lhs, rhs, .. } => 1 + expr_depth(lhs).max(expr_depth(rhs)),
        ExprKind::Block { stmts, value } => {
            let stmt_max = stmts
                .iter()
                .map(|s| match s {
                    Stmt::Let { value, .. } => expr_depth(value),
                    Stmt::Expr { expr, .. } => expr_depth(expr),
                })
                .max()
                .unwrap_or(0);
            1 + stmt_max.max(expr_depth(value))
        }
        _ => 1,
    }
}

// ============================================================================
// Helper: scope validation
// ============================================================================

/// Collect all variable names referenced in an expression and check that
/// each one is bound by an enclosing `let` in the same tree.
///
/// Returns `true` if every `Var` references a name from `bound`.
fn all_vars_bound(expr: &Expr<UnresolvedName>, bound: &[Symbol]) -> bool {
    match expr.kind.as_ref() {
        ExprKind::Var(name) => bound.contains(&name.name),
        ExprKind::BinOp { lhs, rhs, .. } => {
            all_vars_bound(lhs, bound) && all_vars_bound(rhs, bound)
        }
        ExprKind::Block { stmts, value } => {
            let mut scope = bound.to_vec();
            for stmt in stmts {
                match stmt {
                    Stmt::Let {
                        pattern,
                        value: rhs,
                        ..
                    } => {
                        // RHS is evaluated in the current scope (before this binding).
                        if !all_vars_bound(rhs, &scope) {
                            return false;
                        }
                        // After the let, the bound name enters scope.
                        if let PatternKind::Bind { name, .. } = pattern.kind.as_ref() {
                            scope.push(*name);
                        }
                    }
                    Stmt::Expr { expr, .. } => {
                        if !all_vars_bound(expr, &scope) {
                            return false;
                        }
                    }
                }
            }
            all_vars_bound(value, &scope)
        }
        // All other node kinds (literals, Nil, etc.) have no var references.
        _ => true,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::SpanMap;
    use crate::resolve;
    use crate::typeck;

    /// Salsa input wrapper so that generated modules can be passed into
    /// `#[salsa::tracked]` helper functions (which require Salsa struct params).
    #[salsa::input]
    struct PropTestInput {
        #[returns(ref)]
        module: Module<UnresolvedName>,
    }

    /// Tracked wrapper: resolve → typecheck pipeline.
    ///
    /// Running inside a tracked function provides the accumulator context
    /// that the type checker needs to report diagnostics.
    #[salsa::tracked]
    fn run_resolve_and_typecheck<'db>(db: &'db dyn salsa::Database, input: PropTestInput) {
        let module = input.module(db).clone();
        let span_map = SpanMap::default();
        let resolved = resolve::resolve_module(db, module, span_map.clone());
        let _output = typeck::typecheck_module(db, resolved, span_map);
    }

    proptest! {
        /// Generated expressions implement Clone, Debug, PartialEq correctly.
        #[test]
        fn generated_expr_is_valid(expr in expr_default()) {
            // Clone + PartialEq
            let cloned = expr.clone();
            prop_assert_eq!(&expr, &cloned);
            // Debug doesn't panic
            let _ = format!("{:?}", expr);
        }

        /// The depth of generated expressions is bounded by the configured maximum.
        #[test]
        fn depth_is_bounded(expr in expr(4, 32, 3)) {
            let depth = expr_depth(&expr);
            prop_assert!(depth <= 5, "depth {} exceeded maximum 5", depth);
        }

        /// The literal-only strategy always produces depth-1 (leaf) expressions.
        #[test]
        fn literals_are_leaves(expr in literal_expr()) {
            prop_assert_eq!(expr_depth(&expr), 1);
        }

        /// `var_expr(3)` always produces depth-1 (leaf) expressions.
        #[test]
        fn var_exprs_are_leaves(expr in var_expr(3)) {
            prop_assert_eq!(expr_depth(&expr), 1);
        }

        /// Block nodes always contain at least one statement.
        #[test]
        fn block_has_stmts(expr in block_variant(0, 3, 3)) {
            if let ExprKind::Block { stmts, .. } = expr.kind.as_ref() {
                prop_assert!(!stmts.is_empty(), "block must have ≥1 statement");
            } else {
                panic!("block_variant must produce Block");
            }
        }

        /// Every `Var` in a generated expression references a name that was
        /// bound by an enclosing `let` in the same tree.
        #[test]
        fn all_vars_in_scope(expr in expr_with_env(0, 6)) {
            prop_assert!(
                all_vars_bound(&expr, &[]),
                "found Var referencing unbound name"
            );
        }

        /// Wrapping into a module never panics.
        #[test]
        fn module_construction_does_not_panic(_module in parsed_module()) {
            // reaching here without panic is the assertion
        }
    }

    // Pipeline smoke tests: resolve and typecheck are heavier, so run fewer cases.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Name resolution does not panic on generated ASTs.
        ///
        /// All variables are bound by enclosing `let` bindings, so resolve
        /// should succeed without errors.
        #[test]
        fn resolve_does_not_panic(module in parsed_module()) {
            let db = salsa::DatabaseImpl::new();
            let _resolved = resolve::resolve_module(&db, module, SpanMap::default());
        }

        /// Type checking does not panic on generated ASTs.
        ///
        /// Type errors (e.g. `bool + int`) are expected and collected as
        /// diagnostics — the key property is that the pipeline never panics.
        /// Runs inside a `#[salsa::tracked]` wrapper to provide the
        /// accumulator context required by the type checker.
        #[test]
        fn typecheck_does_not_panic(module in parsed_module()) {
            let db = salsa::DatabaseImpl::new();
            let input = PropTestInput::new(&db, module);
            run_resolve_and_typecheck(&db, input);
        }
    }
}

//! Random AST generators for property-based testing.
//!
//! Provides `proptest` strategy functions that generate syntactically valid
//! Tribute AST nodes.
//!
//! - Phase 1: Literals and arithmetic expressions.
//! - Phase 2: `Block`, `Let`, and `Var` with scope-correct variable
//!   references via a parameterized `env_size` approach.
//! - Phase 3: `Lambda`, `Call`, `Tuple`, and multi-function modules.

use proptest::prelude::*;

use super::{
    BinOpKind, Decl, Expr, ExprKind, FloatBits, FuncDecl, Module, NodeId, Param, ParamDecl,
    Pattern, PatternKind, Stmt, UnresolvedName,
};
use trunk_ir::Symbol;

/// Fixed pool of variable names. `env_size` indexes into this array:
/// `env_size=2` means `VAR_NAMES[0..2]` ("foo", "bar") are in scope.
const VAR_NAMES: &[&str] = &[
    "foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply",
];

/// Fixed pool of helper function names for multi-function module generation.
const FUNC_NAMES: &[&str] = &["alpha", "beta"];

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

/// Strategy for a Tuple expression with 0–3 elements.
///
/// No scope change — all elements are generated in the same environment.
fn tuple_variant(env_size: usize, max_depth: usize) -> BoxedStrategy<Expr<UnresolvedName>> {
    let child_depth = max_depth - 1;
    prop_oneof![
        // 0-element (unit tuple)
        node_id().prop_map(|id| Expr::new(id, ExprKind::Tuple(vec![]))),
        // 1-element
        (node_id(), expr_with_env(env_size, child_depth))
            .prop_map(|(id, e)| Expr::new(id, ExprKind::Tuple(vec![e]))),
        // 2-element
        (
            node_id(),
            expr_with_env(env_size, child_depth),
            expr_with_env(env_size, child_depth),
        )
            .prop_map(|(id, a, b)| Expr::new(id, ExprKind::Tuple(vec![a, b]))),
        // 3-element
        (
            node_id(),
            expr_with_env(env_size, child_depth),
            expr_with_env(env_size, child_depth),
            expr_with_env(env_size, child_depth),
        )
            .prop_map(|(id, a, b, c)| Expr::new(id, ExprKind::Tuple(vec![a, b, c]))),
    ]
    .boxed()
}

/// Strategy for a Lambda expression with 1–3 parameters.
///
/// Parameters are drawn from `VAR_NAMES[env_size..]`, and the body is
/// generated in a scope extended by those parameter names.
///
/// # Panics
/// Panics if `remaining` (`VAR_NAMES.len() - env_size`) is 0.
fn lambda_variant(env_size: usize, max_depth: usize) -> BoxedStrategy<Expr<UnresolvedName>> {
    let remaining = VAR_NAMES.len().saturating_sub(env_size);
    assert!(remaining > 0, "lambda_variant requires remaining > 0");
    let child_depth = max_depth - 1;

    let mut variants: Vec<BoxedStrategy<Expr<UnresolvedName>>> = Vec::new();

    // --- 1 param ---
    {
        let p0 = VAR_NAMES[env_size];
        let body = expr_with_env(env_size + 1, child_depth);
        variants.push(
            (node_id(), node_id(), body)
                .prop_map(move |(expr_id, param_id, body)| {
                    Expr::new(
                        expr_id,
                        ExprKind::Lambda {
                            params: vec![Param {
                                id: param_id,
                                name: Symbol::new(p0),
                                ty: None,
                                local_id: None,
                            }],
                            body,
                        },
                    )
                })
                .boxed(),
        );
    }

    // --- 2 params ---
    if remaining >= 2 {
        let p0 = VAR_NAMES[env_size];
        let p1 = VAR_NAMES[env_size + 1];
        let body = expr_with_env(env_size + 2, child_depth);
        variants.push(
            (node_id(), node_id(), node_id(), body)
                .prop_map(move |(expr_id, pid0, pid1, body)| {
                    Expr::new(
                        expr_id,
                        ExprKind::Lambda {
                            params: vec![
                                Param {
                                    id: pid0,
                                    name: Symbol::new(p0),
                                    ty: None,
                                    local_id: None,
                                },
                                Param {
                                    id: pid1,
                                    name: Symbol::new(p1),
                                    ty: None,
                                    local_id: None,
                                },
                            ],
                            body,
                        },
                    )
                })
                .boxed(),
        );
    }

    // --- 3 params ---
    if remaining >= 3 {
        let p0 = VAR_NAMES[env_size];
        let p1 = VAR_NAMES[env_size + 1];
        let p2 = VAR_NAMES[env_size + 2];
        let body = expr_with_env(env_size + 3, child_depth);
        variants.push(
            ((node_id(), node_id(), node_id()), (node_id(), body))
                .prop_map(move |((expr_id, pid0, pid1), (pid2, body))| {
                    Expr::new(
                        expr_id,
                        ExprKind::Lambda {
                            params: vec![
                                Param {
                                    id: pid0,
                                    name: Symbol::new(p0),
                                    ty: None,
                                    local_id: None,
                                },
                                Param {
                                    id: pid1,
                                    name: Symbol::new(p1),
                                    ty: None,
                                    local_id: None,
                                },
                                Param {
                                    id: pid2,
                                    name: Symbol::new(p2),
                                    ty: None,
                                    local_id: None,
                                },
                            ],
                            body,
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
        _ => unreachable!(),
    }
}

/// Strategy for a Call expression with 0–2 arguments.
///
/// The callee is an arbitrary expression from the current scope; arguments
/// are also generated from the same scope.
fn call_variant(env_size: usize, max_depth: usize) -> BoxedStrategy<Expr<UnresolvedName>> {
    let child_depth = max_depth - 1;
    prop_oneof![
        // 0 args
        (node_id(), expr_with_env(env_size, child_depth)).prop_map(|(id, callee)| Expr::new(
            id,
            ExprKind::Call {
                callee,
                args: vec![]
            }
        )),
        // 1 arg
        (
            node_id(),
            expr_with_env(env_size, child_depth),
            expr_with_env(env_size, child_depth),
        )
            .prop_map(|(id, callee, a)| Expr::new(
                id,
                ExprKind::Call {
                    callee,
                    args: vec![a],
                },
            )),
        // 2 args
        (
            node_id(),
            expr_with_env(env_size, child_depth),
            expr_with_env(env_size, child_depth),
            expr_with_env(env_size, child_depth),
        )
            .prop_map(|(id, callee, a, b)| Expr::new(
                id,
                ExprKind::Call {
                    callee,
                    args: vec![a, b],
                },
            )),
    ]
    .boxed()
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
        // Pool exhausted: no block or lambda (both need fresh names).
        prop_oneof![
            4 => leaf,
            3 => binop_variant(env_size, max_depth),
            2 => tuple_variant(env_size, max_depth),
            2 => call_variant(env_size, max_depth),
        ]
        .boxed()
    } else {
        prop_oneof![
            4 => leaf,
            3 => binop_variant(env_size, max_depth),
            3 => block_variant(env_size, max_depth, remaining),
            2 => tuple_variant(env_size, max_depth),
            2 => lambda_variant(env_size, max_depth),
            2 => call_variant(env_size, max_depth),
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

/// Strategy with reasonable defaults: budget 3 (max depth 4), empty initial scope.
///
/// Manual recursion builds the strategy tree eagerly, so the budget is kept
/// moderate to avoid exponential construction cost. With 6 variant types
/// (leaf, binop, block, tuple, lambda, call) the branching factor is high.
pub fn expr_default() -> impl Strategy<Value = Expr<UnresolvedName>> {
    expr_with_env(0, 3)
}

// ============================================================================
// Module-level strategy
// ============================================================================

/// Strategy producing a module with 1–3 top-level functions.
///
/// Variants:
/// - `single_fn_module`: `fn main() { expr }`
/// - `two_fn_module`: `fn alpha(foo) { ... }` + `fn main() { alpha(expr) }`
/// - `three_fn_module`: `fn alpha(foo) { ... }` + `fn beta(foo) { ... }` + `fn main() { ... }`
pub fn parsed_module() -> impl Strategy<Value = Module<UnresolvedName>> {
    prop_oneof![
        4 => single_fn_module(),
        3 => two_fn_module(),
        3 => three_fn_module(),
    ]
}

/// Wraps an expression into a `fn main() { expr }` parsed module.
fn single_fn_module() -> BoxedStrategy<Module<UnresolvedName>> {
    (node_id(), node_id(), expr_default())
        .prop_map(|(module_id, func_id, body)| {
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
        .boxed()
}

/// Module with one helper function + main that calls it.
///
/// ```text
/// fn alpha(foo) { <body using foo> }
/// fn main() { alpha(<expr>) }
/// ```
fn two_fn_module() -> BoxedStrategy<Module<UnresolvedName>> {
    let helper_body = expr_with_env(1, 3); // env has "foo"
    let main_arg = expr_with_env(0, 3);
    (
        (node_id(), node_id(), node_id()),
        (node_id(), node_id(), node_id()),
        (helper_body, main_arg),
    )
        .prop_map(
            |((mod_id, helper_fid, helper_pid), (main_fid, call_id, callee_nid), (h_body, arg))| {
                let helper = FuncDecl {
                    id: helper_fid,
                    is_pub: false,
                    name: Symbol::new(FUNC_NAMES[0]),
                    type_params: vec![],
                    params: vec![ParamDecl {
                        id: helper_pid,
                        name: Symbol::new(VAR_NAMES[0]),
                        ty: None,
                        local_id: None,
                    }],
                    return_ty: None,
                    effects: None,
                    body: h_body,
                };
                let callee_name = UnresolvedName::simple(Symbol::new(FUNC_NAMES[0]), callee_nid);
                let callee_expr = Expr::new(call_id, ExprKind::Var(callee_name));
                let main_body = Expr::new(
                    call_id,
                    ExprKind::Call {
                        callee: callee_expr,
                        args: vec![arg],
                    },
                );
                let main_fn = FuncDecl {
                    id: main_fid,
                    is_pub: false,
                    name: Symbol::new("main"),
                    type_params: vec![],
                    params: vec![],
                    return_ty: None,
                    effects: None,
                    body: main_body,
                };
                Module::new(
                    mod_id,
                    None,
                    vec![Decl::Function(helper), Decl::Function(main_fn)],
                )
            },
        )
        .boxed()
}

/// Module with two helper functions + main.
///
/// ```text
/// fn alpha(foo) { <body> }
/// fn beta(foo) { <body> }
/// fn main() { alpha(beta(<expr>)) }
/// ```
fn three_fn_module() -> BoxedStrategy<Module<UnresolvedName>> {
    let alpha_body = expr_with_env(1, 3);
    let beta_body = expr_with_env(1, 3);
    let inner_arg = expr_with_env(0, 3);
    (
        (node_id(), node_id(), node_id(), node_id()),
        (node_id(), node_id(), node_id(), node_id()),
        (node_id(), alpha_body, beta_body, inner_arg),
    )
        .prop_map(
            |(
                (mod_id, alpha_fid, alpha_pid, beta_fid),
                (beta_pid, main_fid, call_outer_id, call_inner_id),
                (callee_nid, a_body, b_body, arg),
            )| {
                let alpha_fn = FuncDecl {
                    id: alpha_fid,
                    is_pub: false,
                    name: Symbol::new(FUNC_NAMES[0]),
                    type_params: vec![],
                    params: vec![ParamDecl {
                        id: alpha_pid,
                        name: Symbol::new(VAR_NAMES[0]),
                        ty: None,
                        local_id: None,
                    }],
                    return_ty: None,
                    effects: None,
                    body: a_body,
                };
                let beta_fn = FuncDecl {
                    id: beta_fid,
                    is_pub: false,
                    name: Symbol::new(FUNC_NAMES[1]),
                    type_params: vec![],
                    params: vec![ParamDecl {
                        id: beta_pid,
                        name: Symbol::new(VAR_NAMES[0]),
                        ty: None,
                        local_id: None,
                    }],
                    return_ty: None,
                    effects: None,
                    body: b_body,
                };
                // main body: alpha(beta(arg))
                let beta_callee = Expr::new(
                    callee_nid,
                    ExprKind::Var(UnresolvedName::simple(
                        Symbol::new(FUNC_NAMES[1]),
                        callee_nid,
                    )),
                );
                let inner_call = Expr::new(
                    call_inner_id,
                    ExprKind::Call {
                        callee: beta_callee,
                        args: vec![arg],
                    },
                );
                let alpha_callee = Expr::new(
                    callee_nid,
                    ExprKind::Var(UnresolvedName::simple(
                        Symbol::new(FUNC_NAMES[0]),
                        callee_nid,
                    )),
                );
                let outer_call = Expr::new(
                    call_outer_id,
                    ExprKind::Call {
                        callee: alpha_callee,
                        args: vec![inner_call],
                    },
                );
                let main_fn = FuncDecl {
                    id: main_fid,
                    is_pub: false,
                    name: Symbol::new("main"),
                    type_params: vec![],
                    params: vec![],
                    return_ty: None,
                    effects: None,
                    body: outer_call,
                };
                Module::new(
                    mod_id,
                    None,
                    vec![
                        Decl::Function(alpha_fn),
                        Decl::Function(beta_fn),
                        Decl::Function(main_fn),
                    ],
                )
            },
        )
        .boxed()
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
        ExprKind::Lambda { body, .. } => 1 + expr_depth(body),
        ExprKind::Call { callee, args } => {
            let arg_max = args.iter().map(expr_depth).max().unwrap_or(0);
            1 + expr_depth(callee).max(arg_max)
        }
        ExprKind::Tuple(elems) => 1 + elems.iter().map(expr_depth).max().unwrap_or(0),
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
        ExprKind::Lambda { params, body } => {
            let mut scope = bound.to_vec();
            scope.extend(params.iter().map(|p| p.name));
            all_vars_bound(body, &scope)
        }
        ExprKind::Call { callee, args } => {
            all_vars_bound(callee, bound) && args.iter().all(|a| all_vars_bound(a, bound))
        }
        ExprKind::Tuple(elems) => elems.iter().all(|e| all_vars_bound(e, bound)),
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
        fn depth_is_bounded(expr in expr(3, 32, 3)) {
            let depth = expr_depth(&expr);
            prop_assert!(depth <= 4, "depth {} exceeded maximum 4", depth);
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
        fn all_vars_in_scope(expr in expr_with_env(0, 4)) {
            prop_assert!(
                all_vars_bound(&expr, &[]),
                "found Var referencing unbound name"
            );
        }

        /// Lambda variants always have at least one parameter.
        #[test]
        fn lambda_has_params(expr in lambda_variant(0, 3)) {
            if let ExprKind::Lambda { params, .. } = expr.kind.as_ref() {
                prop_assert!(!params.is_empty(), "lambda must have ≥1 parameter");
            } else {
                panic!("lambda_variant must produce Lambda");
            }
        }

        /// Call variants always have a callee expression.
        #[test]
        fn call_callee_exists(expr in call_variant(0, 3)) {
            if let ExprKind::Call { args, .. } = expr.kind.as_ref() {
                prop_assert!(args.len() <= 2, "call must have ≤2 args");
            } else {
                panic!("call_variant must produce Call");
            }
        }

        /// Tuple element count is bounded by 3.
        #[test]
        fn tuple_elements_bounded(expr in tuple_variant(0, 3)) {
            if let ExprKind::Tuple(elems) = expr.kind.as_ref() {
                prop_assert!(elems.len() <= 3, "tuple must have ≤3 elements, got {}", elems.len());
            } else {
                panic!("tuple_variant must produce Tuple");
            }
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

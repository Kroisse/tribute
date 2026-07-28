//! Administrative A-normalization for CPS AST-to-IR entries.
//!
//! This is intentionally not a new frontend phase or HIR. It rewrites the
//! existing typed AST immediately before CPS lowering so strict children are
//! evaluated in source order and represented by local variables.

use trunk_ir::Symbol;

use crate::ast::{Expr, ExprKind, Pattern, PatternKind, ResolvedRef, Stmt, TypedRef};

use super::context::IrLoweringCtx;

pub(super) fn normalize_for_cps<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    expr: Expr<TypedRef<'db>>,
) -> Expr<TypedRef<'db>> {
    let normalized = Normalizer { ctx }.normalize_expr(expr);
    debug_assert!(validate_normal_form(&normalized).is_ok());
    normalized
}

/// The normal form consumed by CPS lowering. Administrative blocks may occur
/// only as the immediate root of a computation region. A boundary may own one
/// such root, but blocks may not nest through its final value or a strict child.
pub(super) fn validate_normal_form(expr: &Expr<TypedRef<'_>>) -> Result<(), &'static str> {
    validate_region_root(expr)
}

/// Validate a computation-region edge. Its normalized form may have one
/// administrative block shell that owns the region's prefix statements.
fn validate_region_root(expr: &Expr<TypedRef<'_>>) -> Result<(), &'static str> {
    match &*expr.kind {
        ExprKind::Block { stmts, value } => validate_block_contents(stmts, value),
        _ => validate_expr(expr),
    }
}

fn validate_block_contents(
    stmts: &[Stmt<TypedRef<'_>>],
    value: &Expr<TypedRef<'_>>,
) -> Result<(), &'static str> {
    for stmt in stmts {
        let value = match stmt {
            Stmt::Let { value, .. } => value,
            Stmt::Expr { expr, .. } => expr,
        };
        validate_expr(value)?;
    }
    validate_expr(value)
}

/// Validate an expression below a region edge. It must never contain an
/// administrative block directly; independent control-flow regions are
/// checked through `validate_region_root` instead.
fn validate_expr(expr: &Expr<TypedRef<'_>>) -> Result<(), &'static str> {
    let atom = |expr: &Expr<TypedRef<'_>>| {
        if is_atom(expr) {
            Ok(())
        } else {
            Err("strict child is not atomic")
        }
    };
    match &*expr.kind {
        ExprKind::Call { callee, args } => {
            atom(callee)?;
            args.iter().try_for_each(atom)
        }
        ExprKind::Cons { args, .. } | ExprKind::Tuple(args) | ExprKind::List(args) => {
            args.iter().try_for_each(atom)
        }
        ExprKind::Record { fields, spread, .. } => {
            if let Some(spread) = spread {
                atom(spread)?;
            }
            fields.iter().try_for_each(|(_, value)| atom(value))
        }
        ExprKind::BinOp { lhs, rhs, .. } => {
            atom(lhs)?;
            validate_region_root(rhs)
        }
        ExprKind::Block { .. } => Err("administrative block is not at a computation-region edge"),
        ExprKind::Case { scrutinee, arms } => {
            atom(scrutinee)?;
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    validate_region_root(guard)?;
                }
                validate_region_root(&arm.body)?;
            }
            Ok(())
        }
        ExprKind::Lambda { body, .. } => validate_region_root(body),
        ExprKind::Handle { body, handlers } => {
            validate_region_root(body)?;
            handlers
                .iter()
                .try_for_each(|handler| validate_region_root(&handler.body))
        }
        ExprKind::Resume { arg, .. } => atom(arg),
        ExprKind::MethodCall { receiver, args, .. } => {
            atom(receiver)?;
            args.iter().try_for_each(atom)
        }
        _ => Ok(()),
    }
}

struct Normalizer<'a, 'db> {
    ctx: &'a mut IrLoweringCtx<'db>,
}

impl<'db> Normalizer<'_, 'db> {
    fn normalize_expr(&mut self, expr: Expr<TypedRef<'db>>) -> Expr<TypedRef<'db>> {
        let id = expr.id;
        let kind = match *expr.kind {
            ExprKind::Call { callee, args } => {
                let (mut stmts, callee) = self.normalize_to_atom(callee);
                let mut atoms = Vec::with_capacity(args.len());
                for arg in args {
                    let (prefix, atom) = self.normalize_to_atom(arg);
                    stmts.extend(prefix);
                    atoms.push(atom);
                }
                return self.with_prefix(
                    stmts,
                    Expr::new(
                        id,
                        ExprKind::Call {
                            callee,
                            args: atoms,
                        },
                    ),
                );
            }
            ExprKind::Cons { ctor, args } => {
                let (stmts, args) = self.normalize_atoms(args);
                return self.with_prefix(stmts, Expr::new(id, ExprKind::Cons { ctor, args }));
            }
            ExprKind::Tuple(elements) => {
                let (stmts, elements) = self.normalize_atoms(elements);
                return self.with_prefix(stmts, Expr::new(id, ExprKind::Tuple(elements)));
            }
            ExprKind::List(elements) => {
                let (stmts, elements) = self.normalize_atoms(elements);
                return self.with_prefix(stmts, Expr::new(id, ExprKind::List(elements)));
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => {
                let mut stmts = Vec::new();
                let spread = spread.map(|spread| {
                    let (prefix, atom) = self.normalize_to_atom(spread);
                    stmts.extend(prefix);
                    atom
                });
                let mut normalized_fields = Vec::with_capacity(fields.len());
                for (name, field) in fields {
                    let (prefix, atom) = self.normalize_to_atom(field);
                    stmts.extend(prefix);
                    normalized_fields.push((name, atom));
                }
                return self.with_prefix(
                    stmts,
                    Expr::new(
                        id,
                        ExprKind::Record {
                            type_name,
                            fields: normalized_fields,
                            spread,
                        },
                    ),
                );
            }
            ExprKind::BinOp { op, lhs, rhs } => {
                let (stmts, lhs) = self.normalize_to_atom(lhs);
                // RHS stays in the binary expression: lowerers create its
                // selected region for &&/||, so prefix statements cannot
                // escape into the lhs/outer evaluation region.
                let rhs = self.normalize_expr(rhs);
                return self.with_prefix(stmts, Expr::new(id, ExprKind::BinOp { op, lhs, rhs }));
            }
            ExprKind::Block { stmts, value } => return self.normalize_block(id, stmts, value),
            ExprKind::Case { scrutinee, arms } => {
                let (stmts, scrutinee) = self.normalize_to_atom(scrutinee);
                let arms = arms
                    .into_iter()
                    .map(|mut arm| {
                        arm.guard = arm.guard.map(|guard| self.normalize_expr(guard));
                        arm.body = self.normalize_expr(arm.body);
                        arm
                    })
                    .collect();
                return self.with_prefix(stmts, Expr::new(id, ExprKind::Case { scrutinee, arms }));
            }
            ExprKind::Lambda { params, body } => ExprKind::Lambda {
                params,
                body: self.normalize_expr(body),
            },
            ExprKind::Handle { body, handlers } => ExprKind::Handle {
                body: self.normalize_expr(body),
                handlers: handlers
                    .into_iter()
                    .map(|mut handler| {
                        handler.body = self.normalize_expr(handler.body);
                        handler
                    })
                    .collect(),
            },
            ExprKind::Resume { arg, local_id } => {
                let (stmts, arg) = self.normalize_to_atom(arg);
                return self.with_prefix(stmts, Expr::new(id, ExprKind::Resume { arg, local_id }));
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let (mut stmts, receiver) = self.normalize_to_atom(receiver);
                let (arg_stmts, args) = self.normalize_atoms(args);
                stmts.extend(arg_stmts);
                return self.with_prefix(
                    stmts,
                    Expr::new(
                        id,
                        ExprKind::MethodCall {
                            receiver,
                            method,
                            args,
                        },
                    ),
                );
            }
            other => other,
        };
        Expr::new(id, kind)
    }

    fn normalize_block(
        &mut self,
        id: crate::ast::NodeId,
        stmts: Vec<Stmt<TypedRef<'db>>>,
        value: Expr<TypedRef<'db>>,
    ) -> Expr<TypedRef<'db>> {
        let mut normalized = Vec::new();
        for stmt in stmts {
            match stmt {
                Stmt::Let {
                    id,
                    pattern,
                    ty,
                    value,
                } => {
                    let value = self.normalize_expr(value);
                    match *value.kind {
                        ExprKind::Block { mut stmts, value } => {
                            normalized.append(&mut stmts);
                            normalized.push(Stmt::Let {
                                id,
                                pattern,
                                ty,
                                value,
                            });
                        }
                        _ => normalized.push(Stmt::Let {
                            id,
                            pattern,
                            ty,
                            value,
                        }),
                    }
                }
                Stmt::Expr { id, expr } => {
                    let expr = self.normalize_expr(expr);
                    match *expr.kind {
                        ExprKind::Block { mut stmts, value } => {
                            normalized.append(&mut stmts);
                            normalized.push(Stmt::Expr { id, expr: value });
                        }
                        _ => normalized.push(Stmt::Expr { id, expr }),
                    }
                }
            }
        }
        let value = self.normalize_expr(value);
        match *value.kind {
            ExprKind::Block { mut stmts, value } => {
                normalized.append(&mut stmts);
                Expr::new(
                    id,
                    ExprKind::Block {
                        stmts: normalized,
                        value,
                    },
                )
            }
            _ => Expr::new(
                id,
                ExprKind::Block {
                    stmts: normalized,
                    value,
                },
            ),
        }
    }

    fn normalize_atoms(
        &mut self,
        exprs: Vec<Expr<TypedRef<'db>>>,
    ) -> (Vec<Stmt<TypedRef<'db>>>, Vec<Expr<TypedRef<'db>>>) {
        let mut stmts = Vec::new();
        let mut atoms = Vec::with_capacity(exprs.len());
        for expr in exprs {
            let (prefix, atom) = self.normalize_to_atom(expr);
            stmts.extend(prefix);
            atoms.push(atom);
        }
        (stmts, atoms)
    }

    fn normalize_to_atom(
        &mut self,
        expr: Expr<TypedRef<'db>>,
    ) -> (Vec<Stmt<TypedRef<'db>>>, Expr<TypedRef<'db>>) {
        let expr = self.normalize_expr(expr);
        let (mut prefix, expr) = match *expr.kind {
            ExprKind::Block { stmts, value } => (stmts, value),
            _ => (vec![], expr),
        };
        if is_atom(&expr) {
            return (prefix, expr);
        }
        let id = expr.id;
        let ty = *self
            .ctx
            .get_node_type(id)
            .expect("typed AST expression must have a node type during normalization");
        let local_id = self.ctx.next_local_id();
        let name = Symbol::new("__cps_tmp");
        let pattern = Pattern::new(
            id,
            PatternKind::Bind {
                name,
                local_id: Some(local_id),
            },
        );
        let atom = Expr::new(
            id,
            ExprKind::Var(TypedRef::new(ResolvedRef::local(local_id, name), ty)),
        );
        prefix.push(Stmt::Let {
            id,
            pattern,
            ty: None,
            value: expr,
        });
        (prefix, atom)
    }

    fn with_prefix(
        &mut self,
        mut stmts: Vec<Stmt<TypedRef<'db>>>,
        expr: Expr<TypedRef<'db>>,
    ) -> Expr<TypedRef<'db>> {
        if stmts.is_empty() {
            expr
        } else {
            let id = expr.id;
            Expr::new(
                id,
                ExprKind::Block {
                    stmts: std::mem::take(&mut stmts),
                    value: expr,
                },
            )
        }
    }
}

fn is_atom(expr: &Expr<TypedRef<'_>>) -> bool {
    matches!(
        *expr.kind,
        ExprKind::Var(_)
            | ExprKind::NatLit(_)
            | ExprKind::IntLit(_)
            | ExprKind::FloatLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::BytesLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::Nil
            | ExprKind::RuneLit(_)
            | ExprKind::Lambda { .. }
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tree_sitter::Parser;
    use trunk_ir::context::IrContext;

    use super::*;
    use crate::ast::{NodeId, SpanMap, Type, TypeKind};

    fn test_id() -> NodeId {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse("fn main() { 0 }", None).unwrap();
        NodeId::from_cst(&tree.root_node(), 0)
    }

    fn test_ctx<'db>(db: &'db salsa::DatabaseImpl, id: NodeId) -> IrLoweringCtx<'db> {
        let mut ir = IrContext::new();
        let path = ir.paths.intern("normalize.trb".to_owned());
        let ty = Type::new(db, TypeKind::Error);
        IrLoweringCtx::new(
            db,
            path,
            SpanMap::default(),
            HashMap::new(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::from([(id, ty)]),
        )
    }

    fn local<'db>(id: NodeId, ty: Type<'db>, local: u32, name: &str) -> Expr<TypedRef<'db>> {
        let name = Symbol::from_dynamic(name);
        Expr::new(
            id,
            ExprKind::Var(TypedRef::new(
                ResolvedRef::local(crate::ast::LocalId::new(local), name),
                ty,
            )),
        )
    }

    fn direct_call_name(expr: &Expr<TypedRef<'_>>) -> String {
        let ExprKind::Call { callee, .. } = expr.kind.as_ref() else {
            panic!("expected an administrative binding for a call");
        };
        let ExprKind::Var(typed) = callee.kind.as_ref() else {
            panic!("expected the test call to have a local callee");
        };
        let ResolvedRef::Local { name, .. } = &typed.resolved else {
            panic!("expected the test call to have a local callee");
        };
        name.with_str(str::to_owned)
    }

    fn stmt_value<'a, 'db>(stmt: &'a Stmt<TypedRef<'db>>) -> &'a Expr<TypedRef<'db>> {
        match stmt {
            Stmt::Let { value, .. } => value,
            Stmt::Expr { expr, .. } => expr,
        }
    }

    #[test]
    fn flattens_callee_and_arguments_left_to_right() {
        let db = salsa::DatabaseImpl::new();
        let id = test_id();
        let ty = Type::new(&db, TypeKind::Error);
        let call = Expr::new(
            id,
            ExprKind::Call {
                callee: Expr::new(
                    id,
                    ExprKind::Call {
                        callee: local(id, ty, 1, "make"),
                        args: vec![],
                    },
                ),
                args: vec![
                    Expr::new(
                        id,
                        ExprKind::Call {
                            callee: local(id, ty, 2, "first"),
                            args: vec![],
                        },
                    ),
                    Expr::new(
                        id,
                        ExprKind::Call {
                            callee: local(id, ty, 3, "second"),
                            args: vec![],
                        },
                    ),
                ],
            },
        );
        let normalized = normalize_for_cps(&mut test_ctx(&db, id), call);
        let ExprKind::Block { stmts, value } = *normalized.kind else {
            panic!("strict prefixes must be flattened into a block");
        };
        assert_eq!(stmts.len(), 3);
        assert_eq!(
            stmts
                .iter()
                .map(stmt_value)
                .map(direct_call_name)
                .collect::<Vec<_>>(),
            ["make", "first", "second"],
            "computed callee and arguments must retain source evaluation order"
        );
        assert!(matches!(*value.kind, ExprKind::Call { .. }));
        assert!(validate_normal_form(&Expr::new(id, ExprKind::Block { stmts, value })).is_ok());
    }

    #[test]
    fn flattens_record_spread_and_fields_left_to_right() {
        let db = salsa::DatabaseImpl::new();
        let id = test_id();
        let ty = Type::new(&db, TypeKind::Error);
        let call = |name, local_id| {
            Expr::new(
                id,
                ExprKind::Call {
                    callee: local(id, ty, local_id, name),
                    args: vec![],
                },
            )
        };
        let record = Expr::new(
            id,
            ExprKind::Record {
                type_name: TypedRef::new(
                    ResolvedRef::local(crate::ast::LocalId::new(1), Symbol::new("Record")),
                    ty,
                ),
                fields: vec![
                    (Symbol::new("first"), call("first_field", 3)),
                    (Symbol::new("second"), call("second_field", 4)),
                ],
                spread: Some(call("spread", 2)),
            },
        );
        let normalized = normalize_for_cps(&mut test_ctx(&db, id), record);
        let ExprKind::Block { stmts, value } = *normalized.kind else {
            panic!("record strict prefixes must be flattened into a block");
        };
        assert_eq!(
            stmts
                .iter()
                .map(stmt_value)
                .map(direct_call_name)
                .collect::<Vec<_>>(),
            ["spread", "first_field", "second_field"],
            "record spread and fields must retain their defined source order"
        );
        assert!(matches!(*value.kind, ExprKind::Record { .. }));
        assert!(validate_normal_form(&Expr::new(id, ExprKind::Block { stmts, value })).is_ok());
    }

    #[test]
    fn preserves_short_circuit_rhs_boundary() {
        let db = salsa::DatabaseImpl::new();
        let id = test_id();
        let ty = Type::new(&db, TypeKind::Error);
        let expr = Expr::new(
            id,
            ExprKind::BinOp {
                op: crate::ast::BinOpKind::Or,
                lhs: local(id, ty, 1, "lhs"),
                rhs: Expr::new(
                    id,
                    ExprKind::Call {
                        callee: local(id, ty, 2, "effectful"),
                        args: vec![Expr::new(
                            id,
                            ExprKind::Call {
                                callee: local(id, ty, 3, "argument"),
                                args: vec![],
                            },
                        )],
                    },
                ),
            },
        );
        let normalized = normalize_for_cps(&mut test_ctx(&db, id), expr);
        let ExprKind::BinOp { rhs, .. } = *normalized.kind else {
            panic!("short-circuit form must remain structural");
        };
        assert!(matches!(*rhs.kind, ExprKind::Block { .. }));
        assert!(
            validate_normal_form(&Expr::new(
                id,
                ExprKind::BinOp {
                    op: crate::ast::BinOpKind::Or,
                    lhs: local(id, ty, 1, "lhs"),
                    rhs,
                },
            ))
            .is_ok()
        );
    }

    #[test]
    fn preserves_case_lambda_and_handle_boundaries() {
        let db = salsa::DatabaseImpl::new();
        let id = test_id();
        let ty = Type::new(&db, TypeKind::Error);
        let nested_call = || {
            Expr::new(
                id,
                ExprKind::Call {
                    callee: local(id, ty, 2, "consume"),
                    args: vec![Expr::new(
                        id,
                        ExprKind::Call {
                            callee: local(id, ty, 3, "produce"),
                            args: vec![],
                        },
                    )],
                },
            )
        };
        let wildcard = || Pattern::new(id, PatternKind::Wildcard);
        let lambda = Expr::new(
            id,
            ExprKind::Lambda {
                params: vec![],
                body: nested_call(),
            },
        );
        let expr = Expr::new(
            id,
            ExprKind::Case {
                scrutinee: local(id, ty, 1, "scrutinee"),
                arms: vec![crate::ast::Arm {
                    id,
                    pattern: wildcard(),
                    guard: Some(nested_call()),
                    body: Expr::new(
                        id,
                        ExprKind::Block {
                            stmts: vec![Stmt::Expr { id, expr: lambda }],
                            value: Expr::new(
                                id,
                                ExprKind::Handle {
                                    body: nested_call(),
                                    handlers: vec![crate::ast::HandlerArm {
                                        id,
                                        kind: crate::ast::HandlerKind::Do {
                                            binding: wildcard(),
                                        },
                                        body: nested_call(),
                                    }],
                                },
                            ),
                        },
                    ),
                }],
            },
        );
        let normalized = normalize_for_cps(&mut test_ctx(&db, id), expr);
        let ExprKind::Case { arms, .. } = *normalized.kind else {
            panic!("case must remain structural");
        };
        assert!(matches!(
            arms[0].guard.as_ref().unwrap().kind.as_ref(),
            ExprKind::Block { .. }
        ));
        let ExprKind::Block { stmts, value } = arms[0].body.kind.as_ref() else {
            panic!("case arm body must remain in its selected region");
        };
        let ExprKind::Lambda {
            body: lambda_body, ..
        } = stmt_value(&stmts[0]).kind.as_ref()
        else {
            panic!("lambda must remain inside the selected case arm");
        };
        assert!(matches!(lambda_body.kind.as_ref(), ExprKind::Block { .. }));
        let ExprKind::Handle { body, handlers } = value.kind.as_ref() else {
            panic!("handle must remain inside the selected case arm");
        };
        assert!(matches!(body.kind.as_ref(), ExprKind::Block { .. }));
        assert!(matches!(
            handlers[0].body.kind.as_ref(),
            ExprKind::Block { .. }
        ));
        assert!(
            validate_normal_form(&Expr::new(
                id,
                ExprKind::Case {
                    scrutinee: local(id, ty, 1, "scrutinee"),
                    arms,
                },
            ))
            .is_ok()
        );
    }

    #[test]
    fn normalization_is_idempotent_after_flattening() {
        let db = salsa::DatabaseImpl::new();
        let id = test_id();
        let ty = Type::new(&db, TypeKind::Error);
        let expr = Expr::new(
            id,
            ExprKind::Call {
                callee: Expr::new(
                    id,
                    ExprKind::Call {
                        callee: local(id, ty, 1, "make"),
                        args: vec![],
                    },
                ),
                args: vec![Expr::new(
                    id,
                    ExprKind::Call {
                        callee: local(id, ty, 2, "argument"),
                        args: vec![],
                    },
                )],
            },
        );
        let mut ctx = test_ctx(&db, id);
        let once = normalize_for_cps(&mut ctx, expr);
        let twice = normalize_for_cps(&mut ctx, once.clone());
        assert_eq!(
            twice, once,
            "a normalized computation must not allocate more temps"
        );
        assert!(validate_normal_form(&twice).is_ok());
    }

    #[test]
    fn rejects_nested_administrative_blocks_below_region_edges() {
        let db = salsa::DatabaseImpl::new();
        let id = test_id();
        let ty = Type::new(&db, TypeKind::Error);
        let nested_region = || {
            Expr::new(
                id,
                ExprKind::Block {
                    stmts: vec![],
                    value: Expr::new(
                        id,
                        ExprKind::Block {
                            stmts: vec![],
                            value: local(id, ty, 1, "value"),
                        },
                    ),
                },
            )
        };
        let wildcard = || Pattern::new(id, PatternKind::Wildcard);

        let short_circuit = Expr::new(
            id,
            ExprKind::BinOp {
                op: crate::ast::BinOpKind::Or,
                lhs: local(id, ty, 2, "lhs"),
                rhs: nested_region(),
            },
        );
        assert!(validate_normal_form(&short_circuit).is_err());

        let case_arm = Expr::new(
            id,
            ExprKind::Case {
                scrutinee: local(id, ty, 3, "scrutinee"),
                arms: vec![crate::ast::Arm {
                    id,
                    pattern: wildcard(),
                    guard: None,
                    body: nested_region(),
                }],
            },
        );
        assert!(validate_normal_form(&case_arm).is_err());

        let lambda = Expr::new(
            id,
            ExprKind::Lambda {
                params: vec![],
                body: nested_region(),
            },
        );
        assert!(validate_normal_form(&lambda).is_err());

        let handle_body = Expr::new(
            id,
            ExprKind::Handle {
                body: nested_region(),
                handlers: vec![],
            },
        );
        assert!(validate_normal_form(&handle_body).is_err());

        let handle_arm = Expr::new(
            id,
            ExprKind::Handle {
                body: local(id, ty, 4, "body"),
                handlers: vec![crate::ast::HandlerArm {
                    id,
                    kind: crate::ast::HandlerKind::Do {
                        binding: wildcard(),
                    },
                    body: nested_region(),
                }],
            },
        );
        assert!(validate_normal_form(&handle_arm).is_err());
    }
}

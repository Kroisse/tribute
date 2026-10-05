use trunk_ir::Symbol;

use super::*;
use crate::ast::{BinOpKind, FieldPattern, LiteralPattern, ModuleDecl, Param};

fn id(raw: usize) -> NodeId {
    NodeId::from_raw(raw)
}

fn expr(raw: usize, kind: ExprKind<u32>) -> Expr<u32> {
    Expr::new(id(raw), kind)
}

fn pattern(raw: usize, kind: PatternKind<u32>) -> Pattern<u32> {
    Pattern::new(id(raw), kind)
}

fn leaf(raw: usize) -> Expr<u32> {
    expr(raw, ExprKind::NatLit(0))
}

fn bind(raw: usize) -> Pattern<u32> {
    pattern(
        raw,
        PatternKind::Bind {
            name: Symbol::new("x"),
            local_id: None,
        },
    )
}

/// A module that uses every expression, statement, arm, and pattern shape,
/// with phase values `1..=7` and node ids `1..=50`.
fn sample() -> Module<u32> {
    let case_pattern = pattern(
        20,
        PatternKind::Variant {
            ctor: 4,
            fields: vec![
                pattern(
                    21,
                    PatternKind::Record {
                        type_name: 5,
                        fields: vec![
                            FieldPattern {
                                id: id(22),
                                name: Symbol::new("a"),
                                pattern: Some(bind(23)),
                            },
                            FieldPattern {
                                id: id(24),
                                name: Symbol::new("b"),
                                pattern: None,
                            },
                        ],
                        rest: true,
                    },
                ),
                pattern(
                    25,
                    PatternKind::Tuple(vec![pattern(26, PatternKind::Wildcard)]),
                ),
                pattern(
                    27,
                    PatternKind::List(vec![pattern(
                        28,
                        PatternKind::Literal(LiteralPattern::Nat(1)),
                    )]),
                ),
                pattern(
                    29,
                    PatternKind::ListRest {
                        head: vec![pattern(30, PatternKind::Error)],
                        rest: None,
                        rest_local_id: None,
                    },
                ),
                pattern(
                    31,
                    PatternKind::As {
                        pattern: bind(32),
                        name: Symbol::new("y"),
                        local_id: None,
                    },
                ),
            ],
        },
    );
    let body = expr(
        2,
        ExprKind::Block {
            stmts: vec![
                Stmt::Let {
                    id: id(3),
                    pattern: bind(4),
                    ty: None,
                    value: expr(
                        5,
                        ExprKind::Call {
                            callee: expr(6, ExprKind::Var(1)),
                            args: vec![expr(
                                7,
                                ExprKind::Cons {
                                    ctor: 2,
                                    args: vec![leaf(8)],
                                },
                            )],
                        },
                    ),
                },
                Stmt::Expr {
                    id: id(9),
                    expr: expr(
                        10,
                        ExprKind::Record {
                            type_name: 3,
                            fields: vec![(Symbol::new("a"), leaf(11))],
                            spread: Some(leaf(12)),
                        },
                    ),
                },
                Stmt::Expr {
                    id: id(13),
                    expr: expr(
                        14,
                        ExprKind::MethodCall {
                            receiver: leaf(15),
                            method: Symbol::new("m"),
                            args: vec![leaf(16)],
                        },
                    ),
                },
            ],
            value: expr(
                17,
                ExprKind::Case {
                    scrutinee: leaf(18),
                    arms: vec![Arm {
                        id: id(19),
                        pattern: case_pattern,
                        guard: Some(expr(
                            33,
                            ExprKind::BinOp {
                                op: BinOpKind::And,
                                lhs: leaf(34),
                                rhs: expr(35, ExprKind::BoolLit(true)),
                            },
                        )),
                        body: expr(
                            36,
                            ExprKind::Handle {
                                body: expr(
                                    37,
                                    ExprKind::Lambda {
                                        params: vec![Param {
                                            id: id(99),
                                            name: Symbol::new("p"),
                                            ty: None,
                                            local_id: None,
                                        }],
                                        body: expr(
                                            38,
                                            ExprKind::Tuple(vec![expr(
                                                39,
                                                ExprKind::List(vec![expr(40, ExprKind::Error)]),
                                            )]),
                                        ),
                                    },
                                ),
                                handlers: vec![
                                    HandlerArm {
                                        id: id(41),
                                        kind: HandlerKind::Do { binding: bind(42) },
                                        body: leaf(43),
                                    },
                                    HandlerArm {
                                        id: id(44),
                                        kind: HandlerKind::Fn {
                                            ability: 6,
                                            op: Symbol::new("f"),
                                            params: vec![bind(45)],
                                        },
                                        body: leaf(46),
                                    },
                                    HandlerArm {
                                        id: id(47),
                                        kind: HandlerKind::Op {
                                            ability: 7,
                                            op: Symbol::new("o"),
                                            params: vec![bind(48)],
                                            resume_local_id: None,
                                        },
                                        body: expr(
                                            49,
                                            ExprKind::Resume {
                                                arg: leaf(50),
                                                local_id: None,
                                            },
                                        ),
                                    },
                                ],
                            },
                        ),
                    }],
                },
            ),
        },
    );
    let func = FuncDecl {
        id: id(1),
        is_pub: false,
        name: Symbol::new("f"),
        type_params: vec![],
        params: vec![],
        return_ty: None,
        effects: None,
        body,
    };
    Module::new(
        id(0),
        None,
        vec![Decl::Module(ModuleDecl {
            id: id(98),
            name: Symbol::new("inner"),
            is_pub: false,
            body: Some(vec![Decl::Function(func)]),
        })],
    )
}

#[derive(Default)]
struct Record {
    refs: Vec<(RefSite, usize, u32)>,
    ids: Vec<usize>,
}

impl<'ast> Visit<'ast, u32> for Record {
    fn visit_ref(&mut self, site: RefSite, node: NodeId, value: &'ast u32) {
        self.refs.push((site, node.raw(), *value));
    }

    fn visit_node_id(&mut self, id: NodeId) {
        self.ids.push(id.raw());
    }
}

fn record(module: &Module<u32>) -> Record {
    let mut record = Record::default();
    walk_module(&mut record, module);
    record
}

#[test]
fn visits_every_phase_value_and_node_identity_in_source_order() {
    let record = record(&sample());
    assert_eq!(
        record.refs,
        [
            (RefSite::Var, 6, 1),
            (RefSite::ConsCtor, 7, 2),
            (RefSite::RecordType, 10, 3),
            (RefSite::PatternCtor, 20, 4),
            (RefSite::PatternRecordType, 21, 5),
            (RefSite::HandlerAbility, 44, 6),
            (RefSite::HandlerAbility, 47, 7),
        ]
    );
    // Every node from 1 to 50 in source order; parameters (99), the module
    // (0), and the inline module declaration (98) are not visited.
    assert_eq!(record.ids, (1..=50).collect::<Vec<_>>());
}

struct Shift;

impl VisitMut<u32> for Shift {
    fn visit_ref_mut(&mut self, _site: RefSite, node: NodeId, value: &mut u32) {
        // The identity is already rewritten when the phase value is visited.
        assert!(node.raw() >= 100);
        *value += 10;
    }

    fn visit_node_id_mut(&mut self, id: &mut NodeId) {
        *id = NodeId::from_raw(id.raw() + 100);
    }
}

#[test]
fn rewrites_every_phase_value_and_node_identity_in_place() {
    let mut module = sample();
    walk_module_mut(&mut Shift, &mut module);
    let original = record(&sample());
    let shifted = record(&module);
    assert_eq!(
        shifted.refs,
        original
            .refs
            .iter()
            .map(|(site, node, value)| (*site, node + 100, value + 10))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        shifted.ids,
        original.ids.iter().map(|id| id + 100).collect::<Vec<_>>()
    );
}

/// Records the inline modules enclosing each visited function.
#[derive(Default)]
struct Scopes {
    path: Vec<Symbol>,
    functions: Vec<Vec<Symbol>>,
}

impl<'ast> Visit<'ast, u32> for Scopes {
    fn visit_module_decl(&mut self, module: &'ast ModuleDecl<u32>) {
        self.path.push(module.name.clone());
        walk_module_decl(self, module);
        self.path.pop();
    }

    fn visit_func_decl(&mut self, func: &'ast FuncDecl<u32>) {
        self.functions.push(self.path.clone());
        walk_func_decl(self, func);
    }
}

#[test]
fn inline_modules_enclose_their_declarations() {
    let mut scopes = Scopes::default();
    walk_module(&mut scopes, &sample());
    assert_eq!(scopes.functions, [vec![Symbol::new("inner")]]);
    assert!(scopes.path.is_empty());
}

#[test]
fn closures_visit_phase_values_through_refs() {
    let mut module = sample();
    walk_module_mut(&mut Refs(|_, _, value: &mut u32| *value *= 2), &mut module);
    let mut values = Vec::new();
    walk_module(&mut Refs(|_, _, value: &u32| values.push(*value)), &module);
    assert_eq!(values, [2, 4, 6, 8, 10, 12, 14]);
}

#[test]
fn for_each_visits_expressions_in_pre_order() {
    let module = sample();
    let Decl::Module(ModuleDecl {
        body: Some(decls), ..
    }) = &module.decls[0]
    else {
        unreachable!()
    };
    let Decl::Function(func) = &decls[0] else {
        unreachable!()
    };
    let mut ids = Vec::new();
    func.body.for_each(|expr| ids.push(expr.id.raw()));
    let mut expected = record(&module).ids;
    // Only expressions: drop the function, statements, arms, and patterns.
    expected.retain(|id| {
        ![1, 3, 9, 13, 19, 41, 44, 47].contains(id)
            && !(20..=32).contains(id)
            && ![4, 42, 45, 48].contains(id)
    });
    assert_eq!(ids, expected);
}

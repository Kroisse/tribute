//! Constructor evidence agrees with the value and pattern specialization.
use salsa_test_macros::salsa_test;
use tribute::pipeline::{compile_frontend, parse_and_lower_ast, prepare_frontend_for_lowering};
use tribute_core::Diagnostic;
use tribute_front::{
    SourceCst,
    ast::{Decl, ExprKind, PatternKind, Stmt, Type, TypeKind},
};
use trunk_ir::Symbol;

#[salsa_test]
fn enum_constructor_instances_survive_conversion_and_specialization(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "constructor_instances.trb",
        r#"
extern "C" fn consume_int(value: Int) -> Nil
extern "C" fn consume_bool(value: Bool) -> Nil
enum Boxed(a) { Box(a), Empty }
fn int_box() -> Boxed(Int) { Box(+42) }
fn bool_box() -> Boxed(Bool) { Box(True) }
fn main() {
    case Box(+42) {
        Box(value) -> consume_int(value)
        Empty -> Nil
    }
    case Box(True) {
        Box(value) -> consume_bool(value)
        Empty -> Nil
    }
}
"#,
    );
    let typed = parse_and_lower_ast(db, source).unwrap();
    assert!(parse_and_lower_ast::accumulated::<Diagnostic>(db, source).is_empty());
    let prepared = prepare_frontend_for_lowering(db, typed, source).unwrap();
    for output in [typed, prepared] {
        for (name, payload) in [("int_box", TypeKind::Int), ("bool_box", TypeKind::Bool)] {
            let function = output
                .module(db)
                .decls
                .iter()
                .find_map(|decl| match decl {
                    Decl::Function(function) if function.name == Symbol::new(name) => {
                        Some(function)
                    }
                    _ => None,
                })
                .unwrap();
            let ExprKind::Block { value, .. } = &*function.body.kind else {
                panic!("function block")
            };
            let ExprKind::Cons { ctor, .. } = &*value.kind else {
                panic!("constructor expression")
            };
            let TypeKind::Func { params, result, .. } = ctor.ty.kind(db) else {
                panic!("constructor callable")
            };
            assert_eq!(
                params.as_slice(),
                &[Type::new(db, payload)],
                "{name}: constructor argument"
            );
            let node_ty = output
                .expression_types(db)
                .node_types
                .iter()
                .find_map(|(id, ty)| (*id == value.id).then_some(*ty))
                .unwrap();
            assert_eq!(
                *result, node_ty,
                "{name}: constructor and constructed value must agree"
            );
        }
    }
    let (ctx, module) = compile_frontend(db, source).expect("source-logical constructor IR");
    let ir = trunk_ir::printer::print_module(&ctx, module.op());
    for (name, representation) in [("Boxed$Int", "core.i32"), ("Boxed$Bool", "core.i1")] {
        let layout = format!("type = !\"{name}\"");
        assert!(
            ir.lines()
                .any(|line| line.contains("adt.variant_new") && line.contains(&layout)),
            "missing construction for {name}"
        );
        assert!(
            ir.lines().any(|line| line.contains("adt.variant_get")
                && line.contains(&layout)
                && line.ends_with(representation)),
            "missing concrete extraction for {name}"
        );
    }
}

#[salsa_test]
fn let_variant_pattern_preserves_its_constrained_constructor(db: &salsa::DatabaseImpl) {
    let source = SourceCst::from_source_str(
        db,
        "let_constructor.trb",
        "enum Wrapped(a) { Wrapped(a) }\nfn unpack() -> Int { let Wrapped(value) = Wrapped(+42)\nvalue }\nfn main() { Nil }",
    );
    let typed = parse_and_lower_ast(db, source).unwrap();
    assert!(parse_and_lower_ast::accumulated::<Diagnostic>(db, source).is_empty());
    let prepared = prepare_frontend_for_lowering(db, typed, source).unwrap();
    for output in [typed, prepared] {
        let function = output
            .module(db)
            .decls
            .iter()
            .find_map(|decl| match decl {
                Decl::Function(function) if function.name == Symbol::new("unpack") => {
                    Some(function)
                }
                _ => None,
            })
            .unwrap();
        let ExprKind::Block { stmts, .. } = &*function.body.kind else {
            panic!("block")
        };
        let Stmt::Let { pattern, value, .. } = &stmts[0] else {
            panic!("let")
        };
        let PatternKind::Variant {
            ctor: pattern_ctor, ..
        } = &*pattern.kind
        else {
            panic!("variant pattern")
        };
        let ExprKind::Cons {
            ctor: value_ctor, ..
        } = &*value.kind
        else {
            panic!("constructor value")
        };
        assert_eq!(pattern_ctor.ty, value_ctor.ty);
        let TypeKind::Func { params, .. } = pattern_ctor.ty.kind(db) else {
            panic!("constructor callable")
        };
        assert_eq!(params.as_slice(), &[Type::new(db, TypeKind::Int)]);
    }
}

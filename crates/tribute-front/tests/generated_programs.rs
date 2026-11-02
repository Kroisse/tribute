//! The frontend accepts every program the `tribute-prop` generator emits.
//!
//! The generator emits well-typed source with each check's type known by
//! construction. When the syntax or the prelude changes and the generator
//! goes stale, these properties fail first.

mod common;

use proptest::prelude::*;
use tribute_front::SourceCst;
use tribute_front::ast::{Decl, Type, TypeKind};
use tribute_prop::{Program, Ty, program};

use self::common::{analyze_with_prelude, ast_pipeline_diagnostics, prelude_string_type};

thread_local! {
    /// One database for every case, so the prelude is checked once.
    static DB: salsa::DatabaseImpl = salsa::DatabaseImpl::new();
}

/// Whether `ty` is the generator's `expected` type, with `String` the
/// prelude's.
fn has_type<'db>(
    db: &'db dyn salsa::Database,
    ty: Type<'db>,
    expected: Ty,
    string: Type<'db>,
) -> bool {
    match expected {
        Ty::Nat => matches!(ty.kind(db), TypeKind::Nat),
        Ty::Bool => matches!(ty.kind(db), TypeKind::Bool),
        Ty::String => ty == string,
    }
}

fn check_program(db: &dyn salsa::Database, program: &Program) -> Result<(), TestCaseError> {
    let text = program.source();
    let source = SourceCst::from_source_str(db, "generated.trb", &text);

    let diagnostics = ast_pipeline_diagnostics(db, source);
    prop_assert!(
        diagnostics.is_empty(),
        "unexpected diagnostics: {:?}\n{}",
        diagnostics,
        text
    );

    let checked = analyze_with_prelude(db, source);
    let node_types = &checked.expression_types(db).node_types;
    let string = prelude_string_type(db);
    for check in &program.checks {
        let body = checked
            .module(db)
            .decls()
            .find_map(|decl| match decl {
                Decl::Function(func) if func.name == check.name.as_str() => Some(&func.body),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{} is declared\n{text}", check.name));
        let ty = *node_types
            .get(&body.id)
            .unwrap_or_else(|| panic!("{}'s body has a type\n{text}", check.name));
        prop_assert!(
            has_type(db, ty, check.body.ty(), string),
            "{}'s body is {:?}, the generator assigned {}\n{}",
            check.name,
            ty.kind(db),
            check.body.ty(),
            text
        );
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Every generated program passes the frontend with no diagnostics, and
    /// the type inferred for each check's body is the one the generator
    /// assigned to it.
    #[test]
    fn generated_programs_type_check(program in program()) {
        DB.with(|db| check_program(db, &program))?;
    }
}

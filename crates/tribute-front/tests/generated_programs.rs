//! The frontend accepts every program the `tribute-prop` generator emits,
//! and analyzes an edited program as it analyzes the same text from
//! scratch.
//!
//! The generator emits well-typed source with each check's type known by
//! construction. When the syntax or the prelude changes and the generator
//! goes stale, these properties fail first.

mod common;

use std::cell::RefCell;
use std::ops::Range;

use itertools::Itertools;
use proptest::prelude::*;
use proptest::sample::Index;
use tribute_front::SourceCst;
use tribute_front::ast::visit::{Visit, walk_decl_table};
use tribute_front::ast::{Decl, NodeId, Type, TypeKind};
use tribute_front::resolve::{NameDisplay, PackageName};
use tribute_front::source_file::{apply_edit, new_parser};
use tribute_prop::env::{Env, Goal};
use tribute_prop::expr::fragment;
use tribute_prop::fragment::Fragment;
use tribute_prop::program::{Program, program};
use tribute_prop::value::{Ty, ty};

use self::common::{analyze_with_prelude, ast_pipeline_diagnostics, prelude_string_type};

const PATH: &str = "generated.trb";

thread_local! {
    /// One database for every case, so the prelude is checked once.
    static DB: salsa::DatabaseImpl = salsa::DatabaseImpl::new();
    /// One database that every case edits, so each edit meets what earlier
    /// cases left in it.
    static EDITED_DB: RefCell<salsa::DatabaseImpl> = RefCell::new(salsa::DatabaseImpl::new());
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
    let source = SourceCst::from_source_str(db, PATH, &text);

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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Analyzing a program after edits to its checks' bodies, in a database
    /// that analyzed it before each edit, gives the diagnostics and types
    /// that analyzing the edited text in a fresh database gives.
    #[test]
    fn edits_analyze_as_fresh_text((program, edits) in edited_program()) {
        check_edits(&program, &edits)?;
    }
}

/// Every node of a module, in source order.
#[derive(Default)]
struct AllNodes(Vec<NodeId>);

impl<'ast, V: 'ast> Visit<'ast, V> for AllNodes {
    fn visit_node_id(&mut self, id: NodeId) {
        self.0.push(id);
    }
}

/// The type of every node of `source`'s module, in source order, spelled
/// for a diagnostic.
fn node_types(db: &dyn salsa::Database, source: SourceCst) -> Vec<(NodeId, Option<String>)> {
    let checked = analyze_with_prelude(db, source);
    let types = &checked.expression_types(db).node_types;
    let mut nodes = AllNodes::default();
    walk_decl_table(&mut nodes, checked.module(db));
    let names = NameDisplay::new(&PackageName::executable_root(), []);
    nodes
        .0
        .into_iter()
        .map(|id| {
            (
                id,
                types.get(&id).map(|ty| ty.display(db, &names).to_string()),
            )
        })
        .collect()
}

/// Strategy for one edit: the index of the check whose body it replaces,
/// and the new body. A body of another type than its check declares makes
/// the program ill-typed, so the diagnostics compared are not all empty.
fn edit() -> impl Strategy<Value = (Index, Fragment)> {
    let body = ty().prop_flat_map(|ty| fragment(Env::default(), Goal::of(ty), 3));
    (any::<Index>(), body)
}

fn edited_program() -> impl Strategy<Value = (Program, Vec<(Index, Fragment)>)> {
    (program(), prop::collection::vec(edit(), 1..=3))
}

/// What analysis gives for a source: its diagnostics and the type of each
/// node, owned so that two databases' results compare.
#[derive(Debug, PartialEq, Eq)]
struct Analysis {
    diagnostics: Vec<String>,
    types: Vec<(NodeId, Option<String>)>,
}

fn analysis(db: &dyn salsa::Database, source: SourceCst) -> Analysis {
    Analysis {
        diagnostics: ast_pipeline_diagnostics(db, source)
            .iter()
            .map(|diagnostic| format!("{diagnostic:?}"))
            .collect(),
        types: node_types(db, source),
    }
}

/// The byte range of check `index`'s body in `text`, whose bodies are
/// `bodies`.
fn body_range(text: &str, program: &Program, bodies: &[String], index: usize) -> Range<usize> {
    let check = &program.checks[index];
    let header = format!("fn {}() -> {} {{\n    ", check.name, check.body.ty());
    let start = text.find(&header).expect("the check is declared") + header.len();
    start..start + bodies[index].len()
}

fn check_edits(program: &Program, edits: &[(Index, Fragment)]) -> Result<(), TestCaseError> {
    let mut text = program.source();
    let mut bodies = program
        .checks
        .iter()
        .map(|check| check.body.src.clone())
        .collect_vec();
    let edited = EDITED_DB.with_borrow_mut(|db| {
        let source = SourceCst::from_source_str(db, PATH, &text);
        analysis(db, source);
        let mut parser = new_parser();
        for (index, body) in edits {
            let index = index.index(program.checks.len());
            let range = body_range(&text, program, &bodies, index);
            apply_edit(db, &mut parser, source, range.clone(), &body.src);
            text.replace_range(range, &body.src);
            bodies[index].clone_from(&body.src);
            analysis(db, source);
        }
        assert_eq!(source.text(db).to_string(), text);
        analysis(db, source)
    });
    let fresh_db = salsa::DatabaseImpl::new();
    let fresh = analysis(
        &fresh_db,
        SourceCst::from_source_str(&fresh_db, PATH, &text),
    );
    prop_assert_eq!(edited, fresh, "\n{}", text);
    Ok(())
}

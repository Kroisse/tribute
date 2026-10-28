//! A program built from several library packages, without a file system:
//! `app` depends on `lib`, and both depend on `std`.

use salsa::Setter;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::SourceCst;
use tribute_front::analysis::{self, DependencyTable, Package};
use tribute_front::ast::SpanMaps;
use tribute_front::ast_to_ir::{self, LoweringEnvironment, LoweringInput};
use tribute_front::monomorphize::{
    SpecializationInput, SpecializationOutput, monomorphize_functions,
};
use tribute_front::resolve::PackageName;
use tribute_front::testing::{Executed, LoggingDb};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;

const PRELUDE: &str = include_str!("../../../lib/std/prelude.trb");

const LIB: &str = "\
pub struct Point { x: Nat }

pub fn origin() -> Point { Point { x: 1 } }

pub fn x_of(point: Point) -> Nat { point.x }
";

const APP: &str = "\
fn run() -> Nat { lib::x_of(lib::origin()) }
";

/// The packages of one build, with `app`'s source and the record of
/// analyzing it.
struct Build {
    db: LoggingDb,
    lib: Package,
    lib_source: SourceCst,
    app: SourceCst,
    environment: DependencyTable,
}

impl Build {
    fn new() -> Self {
        let db = LoggingDb::default();
        let std = Package::new(
            &db,
            PackageName::std(),
            db.source("prelude:///std/prelude", PRELUDE),
            DependencyTable::new(&db, Vec::new()),
        );
        let lib_source = db.source("lib:///lib/lib.trb", LIB);
        let lib = Package::new(
            &db,
            PackageName::library(Symbol::new("lib")),
            lib_source,
            DependencyTable::new(&db, vec![std]),
        );
        let app = db.source("app.trb", APP);
        let environment = DependencyTable::new(&db, vec![lib, std]);
        Self {
            db,
            lib,
            lib_source,
            app,
            environment,
        }
    }

    fn diagnostics(&self) -> Vec<String> {
        analysis::diagnostics(&self.db, self.app, self.environment)
            .into_iter()
            .map(|diagnostic: &Diagnostic| diagnostic.inner.message.clone())
            .collect()
    }

    fn analyze(&self) -> Executed {
        self.db.record(|db| {
            analysis::analyze(db, self.app, self.environment).expect("app parses");
        })
    }
}

#[test]
fn an_app_uses_a_library_that_uses_std() {
    let build = Build::new();
    assert_eq!(build.diagnostics(), Vec::<String>::new());
    let checked = analysis::checked_package(&build.db, build.lib).expect("lib checks");
    let functions = checked.exports.function_types(&build.db);
    assert!(
        functions
            .iter()
            .any(|(id, _)| id.path(&build.db) == ["lib", "origin"]),
        "lib exports `lib::origin`"
    );
}

#[test]
fn an_app_edit_reuses_the_library_analysis() {
    let mut build = Build::new();
    let baseline = build.analyze();
    build.db.replace(
        build.app,
        "lib::x_of(lib::origin())",
        "lib::x_of(lib::origin()) + 1",
    );
    build
        .analyze()
        .assert_reuse(&baseline, &["analyze"], &["package_env", "checked_package"]);
    assert_eq!(build.diagnostics(), Vec::<String>::new());
}

#[test]
fn a_library_edit_reruns_the_library_analysis() {
    let mut build = Build::new();
    let baseline = build.analyze();
    build
        .db
        .replace(build.lib_source, "Point { x: 1 }", "Point { x: 2 }");
    build
        .analyze()
        .assert_reuse(&baseline, &["checked_package"], &[]);
}

#[test]
fn a_package_that_depends_on_itself_is_rejected() {
    let mut build = Build::new();
    let table = build.lib.dependencies(&build.db);
    let mut packages = table.packages(&build.db).clone();
    packages.push(build.lib);
    table.set_packages(&mut build.db).to(packages);

    assert!(analysis::package_env(&build.db, build.lib).is_none());
    let messages: Vec<_> = analysis::package_env::accumulated::<Diagnostic>(&build.db, build.lib)
        .into_iter()
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect();
    assert_eq!(
        messages,
        ["package `lib` depends on itself through its dependency table"]
    );
}

/// `app` specialized with `lib`'s bodies merged in. Package `std`'s
/// declarations are looked up but its bodies are not specialized, as the
/// frontend test helpers do.
#[salsa::tracked]
fn specialized_with_library<'db>(
    db: &'db dyn salsa::Database,
    app: SourceCst,
    environment: DependencyTable,
    std: Package,
    lib: Package,
) -> SpecializationOutput<'db> {
    let checked = analysis::analyze(db, app, environment).expect("app parses");
    let std_checked = analysis::checked_package(db, std)
        .expect("std checks")
        .checked;
    let lib_checked = analysis::checked_package(db, lib)
        .expect("lib checks")
        .checked;
    let intrinsics = ast_to_ir::registered_compiler_intrinsics(std_checked.module(db))
        .expect("prelude intrinsic directives are supported");
    let input = SpecializationInput::new(
        db,
        checked,
        [std_checked, lib_checked],
        analysis::root_names(db, environment),
    )
    .merge_bodies(db, [lib_checked])
    .with_compiler_intrinsics(intrinsics);
    monomorphize_functions(db, input).expect("the program specializes")
}

#[test]
fn the_program_specializes_the_library_bodies() {
    let build = Build::new();
    let db = &build.db;
    let std = build.lib.dependencies(db).packages(db)[0];
    let specialized = *specialized_with_library(db, build.app, build.environment, std, build.lib);
    let app_spans = tribute_front::query::span_map(db, build.app).expect("app parses");
    let lib_spans = analysis::parsed_package(db, build.lib)
        .expect("lib parses")
        .span_map(db);
    let mut ir = IrContext::new();
    let module = LoweringInput::from_specialization(
        db,
        specialized,
        LoweringEnvironment {
            span_map: SpanMaps::new([app_spans, lib_spans]),
            ..Default::default()
        },
    )
    .lower_to_ir(db, &mut ir, build.app.uri(db).as_str());
    let validation = tribute_ir::dialect::tribute_control::validate(
        &ir,
        module.module,
        &module.compiler_intrinsics,
        &mut Default::default(),
    );
    assert!(validation.is_ok(), "the program validates: {validation}");
    let ir = trunk_ir::printer::print_module(&ir, module.module.op());
    for function in ["lib::origin", "lib::x_of", "pkg::run"] {
        assert!(
            ir.contains(&format!("tribute_control.func @\"{function}\"")),
            "missing `{function}`:\n{ir}"
        );
    }
}

/// A package reads the exports of a package it depends on only through
/// another: `app` matches on a value of an enum of `inner`, whose variants
/// it finds by the expected type, although its table names only `outer`.
#[test]
fn type_checking_reads_the_exports_of_indirect_dependencies() {
    let db = LoggingDb::default();
    let std = Package::new(
        &db,
        PackageName::std(),
        db.source("prelude:///std/prelude", PRELUDE),
        DependencyTable::new(&db, Vec::new()),
    );
    let inner = Package::new(
        &db,
        PackageName::library(Symbol::new("inner")),
        db.source("inner:///inner.trb", "pub enum Mode { Fast, Slow }\n"),
        DependencyTable::new(&db, vec![std]),
    );
    let outer = Package::new(
        &db,
        PackageName::library(Symbol::new("outer")),
        db.source(
            "outer:///outer.trb",
            "pub fn mode() -> inner::Mode { inner::Mode::Slow }\n",
        ),
        DependencyTable::new(&db, vec![inner, std]),
    );
    let app = db.source(
        "app.trb",
        "fn run() -> Nat {\n    case outer::mode() {\n        .Fast -> 1\n        .Slow -> 2\n    }\n}\n",
    );
    let environment = DependencyTable::new(&db, vec![outer, std]);
    let messages: Vec<_> = analysis::diagnostics(&db, app, environment)
        .into_iter()
        .map(|diagnostic| diagnostic.inner.message.clone())
        .collect();
    assert_eq!(messages, Vec::<String>::new());
}

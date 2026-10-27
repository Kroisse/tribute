//! A program built from several library packages, without a file system:
//! `app` depends on `lib`, and both depend on `std`.

use salsa::Setter;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::SourceCst;
use tribute_front::analysis::{self, DependencyTable, Environment, Package};
use tribute_front::resolve::PackageName;
use tribute_front::testing::{Executed, LoggingDb};
use trunk_ir::Symbol;

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
    environment: Environment,
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
        let environment = Environment::Packages(DependencyTable::new(&db, vec![lib, std]));
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

/// The logical IR of `app` with `lib`'s bodies merged in. Package `std`'s
/// declarations are looked up but its bodies are not lowered, as the
/// frontend test helpers do.
#[salsa::tracked(returns(clone))]
fn lowered_with_library(
    db: &dyn salsa::Database,
    app: SourceCst,
    environment: Environment,
    std: Package,
    lib: Package,
) -> String {
    use tribute_front::ast::SpanMaps;
    use tribute_front::ast_to_ir::{self, LoweringEnvironment, LoweringInput};
    use tribute_front::monomorphize::{SpecializationInput, monomorphize_functions};

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
    let specialized = monomorphize_functions(db, input).expect("the program specializes");
    let app_spans = tribute_front::query::span_map(db, app).expect("app parses");
    let lib_spans = analysis::parsed_package(db, lib)
        .expect("lib parses")
        .span_map(db);
    let mut ir = trunk_ir::context::IrContext::new();
    let module = LoweringInput::from_specialization(
        db,
        specialized,
        LoweringEnvironment {
            span_map: SpanMaps::new([app_spans, lib_spans]),
            ..Default::default()
        },
    )
    .lower_to_ir(db, &mut ir, app.uri(db).as_str());
    let validation = tribute_ir::dialect::tribute_control::validate(
        &ir,
        module.module,
        &module.compiler_intrinsics,
        &mut Default::default(),
    );
    assert!(validation.is_ok(), "the program validates: {validation}");
    trunk_ir::printer::print_module(&ir, module.module.op())
}

#[test]
fn the_program_specializes_the_library_bodies() {
    let build = Build::new();
    let std = build.lib.dependencies(&build.db).packages(&build.db)[0];
    let ir = lowered_with_library(&build.db, build.app, build.environment, std, build.lib);
    for function in ["lib::origin", "lib::x_of", "pkg::run"] {
        assert!(
            ir.contains(&format!("tribute_control.func @\"{function}\"")),
            "missing `{function}`:\n{ir}"
        );
    }
}

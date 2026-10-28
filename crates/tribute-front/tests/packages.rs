//! A program built from several library packages, without a file system:
//! `app` depends on `lib`, and both depend on `std`.

use rustc_hash::FxHashMap as HashMap;
use salsa::Setter;
use tribute_core::diagnostic::Diagnostic;
use tribute_front::SourceCst;
use tribute_front::analysis::{self, DependencyTable, Package};
use tribute_front::ast::SpanMaps;
use tribute_front::ast_to_ir::{self, LoweringEnvironment, LoweringInput};
use tribute_front::monomorphize::{
    SpecializationInput, SpecializationOutput, checked_dependencies, merge_specializations,
    monomorphize_functions, specialized_dependencies, specialized_package,
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
    let checked = *analysis::analyze(db, app, environment).expect("app parses");
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

/// A path through a re-export continues in the package the re-export
/// names: `outer` re-exports a module of `inner`, and `app`, whose table
/// names only `outer`, reaches a function nested below it.
#[test]
fn a_path_continues_through_a_re_export_of_an_indirect_dependency() {
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
        db.source(
            "inner:///inner.trb",
            "pub mod a {\n    pub mod b {\n        pub fn f() -> Nat { 1 }\n    }\n}\n",
        ),
        DependencyTable::new(&db, vec![std]),
    );
    let outer = Package::new(
        &db,
        PackageName::library(Symbol::new("outer")),
        db.source("outer:///outer.trb", "pub use inner::a\n"),
        DependencyTable::new(&db, vec![inner, std]),
    );
    let environment = DependencyTable::new(&db, vec![outer, std]);
    let messages = |text| -> Vec<String> {
        let app = db.source("app.trb", text);
        analysis::diagnostics(&db, app, environment)
            .into_iter()
            .map(|diagnostic| diagnostic.inner.message.clone())
            .collect()
    };
    assert_eq!(
        messages("fn run() -> Nat { outer::a::b::f() }\n"),
        Vec::<String>::new()
    );
    // `app`'s table does not name `inner`, so `app` cannot name it.
    assert_eq!(
        messages("fn run() -> Nat { inner::a::b::f() }\n"),
        ["unresolved name `inner::a::b::f`"]
    );
}

const BASE: &str = "\
pub struct Box(a) { value: a }

pub fn wrap(value: a) -> Box(a) { Box { value: value } }

pub fn unwrap(boxed: Box(a)) -> a { boxed.value }
";

const LEFT: &str = "\
pub fn left() -> Nat { base::unwrap(base::wrap(1)) }
";

const RIGHT: &str = "\
pub fn right() -> Nat { Option::unwrap_or(.Some(base::unwrap(base::wrap(2))), 0) }
";

const DIAMOND_APP: &str = "\
struct Local { n: Nat }

fn run() -> Nat {
    left::left() + right::right() + base::unwrap(base::wrap(Local { n: 3 })).n
}
";

/// A diamond: `left` and `right` each depend on `base`, and each first
/// instantiates `base`'s generics at `std`'s types alone. `app` depends on
/// all three.
struct Diamond {
    db: LoggingDb,
    base: Package,
    left: Package,
    right: Package,
    app: SourceCst,
    environment: DependencyTable,
}

impl Diamond {
    fn new() -> Self {
        let db = LoggingDb::default();
        let std = Package::new(
            &db,
            PackageName::std(),
            db.source("prelude:///std/prelude", PRELUDE),
            DependencyTable::new(&db, Vec::new()),
        );
        let library = |name: &str, text: &str, dependencies: Vec<Package>| {
            Package::new(
                &db,
                PackageName::library(Symbol::new(name)),
                db.source(&format!("{name}:///{name}.trb"), text),
                DependencyTable::new(&db, dependencies),
            )
        };
        let base = library("base", BASE, vec![std]);
        let left = library("left", LEFT, vec![base, std]);
        let right = library("right", RIGHT, vec![base, std]);
        let app = db.source("app.trb", DIAMOND_APP);
        let environment = DependencyTable::new(&db, vec![left, right, base, std]);
        Self {
            db,
            base,
            left,
            right,
            app,
            environment,
        }
    }

    fn specialize(&self) -> Executed {
        self.db.record(|db| {
            packaged_app(db, self.app, self.environment).expect("app specializes");
        })
    }
}

/// `app` specialized on its own and merged with the specializations of the
/// packages it depends on, as the root crate specializes a user package.
#[salsa::tracked]
fn packaged_app<'db>(
    db: &'db dyn salsa::Database,
    app: SourceCst,
    environment: DependencyTable,
) -> Option<SpecializationOutput<'db>> {
    let checked = *analysis::analyze(db, app, environment)?;
    let input = SpecializationInput::package(
        db,
        checked,
        checked_dependencies(db, environment),
        analysis::root_names(db, environment),
    );
    let intrinsics = ast_to_ir::registered_compiler_intrinsics(input.module()).ok()?;
    let own = monomorphize_functions(db, input.with_compiler_intrinsics(intrinsics)).ok()?;
    let dependencies = specialized_dependencies(db, environment)?;
    merge_specializations(db, dependencies.into_iter().chain([own])).ok()
}

/// `app` specialized with the bodies of every package it depends on merged
/// in at once.
#[salsa::tracked]
fn merged_app<'db>(
    db: &'db dyn salsa::Database,
    app: SourceCst,
    environment: DependencyTable,
) -> Option<SpecializationOutput<'db>> {
    let checked = *analysis::analyze(db, app, environment)?;
    let dependencies: Vec<_> = analysis::transitive_packages(db, environment)
        .into_iter()
        .map(|package| analysis::checked_package(db, package).map(|checked| checked.checked))
        .collect::<Option<_>>()?;
    let input = SpecializationInput::new(
        db,
        checked,
        dependencies.iter().copied(),
        analysis::root_names(db, environment),
    )
    .merge_bodies(db, dependencies);
    let intrinsics = ast_to_ir::registered_compiler_intrinsics(input.module()).ok()?;
    monomorphize_functions(db, input.with_compiler_intrinsics(intrinsics)).ok()
}

/// Each value of `items` and how many times it occurs.
fn counts<T: std::hash::Hash + Eq>(items: impl IntoIterator<Item = T>) -> HashMap<T, usize> {
    let mut counts = HashMap::default();
    for item in items {
        *counts.entry(item).or_default() += 1;
    }
    counts
}

/// Whether the function instances of `package`'s own specialization include
/// one of `base::wrap`.
fn instantiates_wrap(db: &LoggingDb, package: Package) -> bool {
    let specialized = specialized_package(db, package).expect("the library specializes");
    specialized.function_instances(db).iter().any(|path| {
        path.modules() == ["base"]
            && path
                .leaf()
                .is_some_and(|leaf| leaf.to_string().starts_with("wrap$"))
    })
}

/// The instance entries of `package`'s own specialization, spelled out.
fn own_instances(db: &LoggingDb, package: Package) -> String {
    let specialized = specialized_package(db, package).expect("the library specializes");
    format!(
        "{:?}",
        &specialized.module(db).entries[*specialized.source_entries(db)..]
    )
}

#[test]
fn merging_a_diamond_matches_specializing_every_body_at_once() {
    let diamond = Diamond::new();
    let db = &diamond.db;
    assert_eq!(
        analysis::diagnostics(db, diamond.app, diamond.environment)
            .into_iter()
            .map(|diagnostic: &Diagnostic| diagnostic.inner.message.clone())
            .collect::<Vec<_>>(),
        Vec::<String>::new()
    );
    // Both sides of the diamond have an instance of `base::wrap` of their
    // own, which the merged result keeps once.
    assert!(instantiates_wrap(db, diamond.left));
    assert!(instantiates_wrap(db, diamond.right));

    let merged = merged_app(db, diamond.app, diamond.environment).expect("merging specializes");
    let packaged =
        packaged_app(db, diamond.app, diamond.environment).expect("merging packages specializes");
    let sources = *merged.source_entries(db);
    assert_eq!(sources, *packaged.source_entries(db), "sources");
    let (merged_module, packaged_module) = (merged.module(db), packaged.module(db));
    assert_eq!(
        merged_module.entries[..sources],
        packaged_module.entries[..sources],
        "source entries"
    );
    let instances = counts(&packaged_module.entries[sources..]);
    assert_eq!(
        counts(&merged_module.entries[sources..]),
        instances,
        "instances"
    );
    assert!(
        instances.values().all(|&count| count == 1),
        "duplicate instance"
    );
    let (merged_decls, packaged_decls) = (merged.declarations(db), packaged.declarations(db));
    assert_eq!(
        counts(&merged_decls.function_types),
        counts(&packaged_decls.function_types),
        "function types"
    );
    assert_eq!(
        counts(&merged_decls.constructor_types.schemes),
        counts(&packaged_decls.constructor_types.schemes),
        "constructor schemes"
    );
    assert_eq!(
        merged_decls.constructor_types.specialized_enum_variants,
        packaged_decls.constructor_types.specialized_enum_variants,
        "enum variants"
    );
    assert_eq!(
        merged_decls.ability_definitions, packaged_decls.ability_definitions,
        "ability definitions"
    );
    assert_eq!(
        merged_decls.well_known_types, packaged_decls.well_known_types,
        "well-known types"
    );
    assert!(
        merged.expression_types(db) == packaged.expression_types(db),
        "expression metadata"
    );
    assert_eq!(
        merged.compiler_intrinsics(db),
        packaged.compiler_intrinsics(db),
        "compiler intrinsics"
    );
    assert_eq!(
        merged.function_instances(db),
        packaged.function_instances(db),
        "function instances"
    );
    assert_eq!(
        merged.nominal_instances(db),
        packaged.nominal_instances(db),
        "nominal instances"
    );
}

#[test]
fn an_app_edit_reuses_every_dependency_specialization() {
    let mut diamond = Diamond::new();
    let baseline = diamond.specialize();
    diamond
        .db
        .replace(diamond.app, "Local { n: 3 }", "Local { n: 4 }");
    diamond.specialize().assert_reuse(
        &baseline,
        &["packaged_app"],
        &["specialized_package", "checked_package"],
    );
}

/// Editing one side of the diamond respecializes that package alone: the
/// other side, the package both share, and `std` are reused.
#[test]
fn a_library_edit_respecializes_that_library_alone() {
    let mut diamond = Diamond::new();
    diamond.specialize();
    let left = *diamond.left.root(&diamond.db);
    diamond.db.replace(left, "base::wrap(1)", "base::wrap(5)");
    let executed = diamond.specialize();
    assert!(executed.contains("packaged_app"));
    assert_eq!(executed.count("specialized_package"), 1, "{executed:?}");
}

/// A package's specialization does not depend on the instances its
/// dependencies generate: when `base` starts to instantiate its own generic
/// at `std`'s types, `left` keeps its own instance of it.
#[test]
fn a_dependency_instance_leaves_a_dependent_specialization_unchanged() {
    let mut diamond = Diamond::new();
    let db = &diamond.db;
    let before = own_instances(db, diamond.left);
    assert!(instantiates_wrap(db, diamond.left));
    assert!(!instantiates_wrap(db, diamond.base));
    let base = *diamond.base.root(&diamond.db);
    diamond.db.replace(
        base,
        "pub fn unwrap(",
        "pub fn one() -> Nat { unwrap(wrap(1)) }\n\npub fn unwrap(",
    );
    let db = &diamond.db;
    assert!(instantiates_wrap(db, diamond.base));
    assert_eq!(own_instances(db, diamond.left), before);
    packaged_app(db, diamond.app, diamond.environment).expect("merging packages specializes");
}

//! Frontend analysis of a package source against the packages it depends on.
//!
//! The CLI, the language server, and tests analyze a source through these
//! queries, passing the dependency table it is analyzed against explicitly.
//! The caller owns where each library package's source comes from and builds
//! the [`Package`] inputs.
//!
//! [`analyze`] returns its result even when it reports errors, so editor
//! features can read a partial result. Whether a result may be lowered is the
//! caller's decision: the compilation route lowers only a source whose
//! analysis reported no error.

use salsa::Accumulator;
use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use trunk_ir::Symbol;

use crate::astgen::{self, ParsedAst};
use crate::query;
use crate::resolve::{self, Dependencies, NameDisplay, PackageEnv, PackageName};
use crate::source_file::SourceCst;
use crate::typeck::{PackageExports, TypeCheckOutput, TypeChecker};

/// A library package: its name, the source of its root module, and the
/// packages its dependency table names (`new-plans/implementation.md`).
/// Package `std` is one whose table is empty.
#[salsa::input(debug)]
pub struct Package {
    #[returns(ref)]
    pub name: PackageName,
    #[returns(copy)]
    pub root: SourceCst,
    #[returns(copy)]
    pub dependencies: DependencyTable,
}

/// A dependency table: the library packages a package depends on, each
/// named in it by its package name.
#[salsa::input(debug)]
pub struct DependencyTable {
    #[returns(ref)]
    pub packages: Vec<Package>,
}

/// A library package type checked on its own: the typed module, and the
/// exports a dependent package's type checking reads.
#[derive(Clone, Copy, PartialEq, Eq, salsa::SalsaValue)]
pub struct CheckedPackage<'db> {
    pub checked: TypeCheckOutput<'db>,
    pub exports: PackageExports<'db>,
}

/// Parse the root module of `package`. Its nodes are numbered under its
/// package name, so they never collide with another package's.
pub fn parsed_package(db: &dyn salsa::Database, package: Package) -> Option<ParsedAst<'_>> {
    let module = package
        .name(db)
        .name()
        .expect("a library package has a name")
        .clone();
    astgen::lower_source_to_parsed_ast_with_module_path(db, package.root(db), Some(module))
}

/// The environment of `package`, built from its declarations and its
/// dependency table. `None` if its root does not parse or it depends on
/// itself.
#[salsa::tracked(returns(as_ref))]
pub fn package_env<'db>(db: &'db dyn salsa::Database, package: Package) -> Option<PackageEnv<'db>> {
    if depends_on_itself(db, package) {
        Diagnostic::new(
            format!(
                "package `{}` depends on itself through its dependency table",
                package_name(db, package)
            ),
            trunk_ir::Span::new(0, 0),
            DiagnosticSeverity::Error,
            CompilationPhase::NameResolution,
        )
        .accumulate(db);
        return None;
    }
    let parsed = parsed_package(db, package)?;
    Some(resolve::build_package_env(
        db,
        parsed.module(db),
        package.name(db),
        &dependencies(db, package.dependencies(db)),
    ))
}

/// Resolve and type check `package` once, against the packages it depends
/// on.
#[salsa::tracked(returns(copy))]
pub fn checked_package<'db>(
    db: &'db dyn salsa::Database,
    package: Package,
) -> Option<CheckedPackage<'db>> {
    let env = package_env(db, package)?;
    let parsed = parsed_package(db, package)?;
    let span_map = parsed.span_map(db);
    let table = package.dependencies(db);
    let dependencies = dependencies(db, table);
    let resolved =
        resolve::resolve_library_with_env(db, parsed.module(db), env, &dependencies, span_map);
    let mut checker = TypeChecker::new(db, span_map, package_names(db, package));
    depend_on_table(db, &mut checker, table);
    let (result, exports) = checker.check_library(&resolved);
    Some(CheckedPackage {
        checked: TypeCheckOutput::from_check(db, result),
        exports,
    })
}

/// A dependency table of one package `std` whose root module is empty, so
/// that it declares only the compiler-owned names: for analyzing a source
/// without the prelude.
#[salsa::tracked(returns(copy))]
pub fn builtins(db: &dyn salsa::Database) -> DependencyTable {
    let text = ropey::Rope::new();
    let tree = crate::source_file::parse_rope(&text, None);
    let uri = fluent_uri::Uri::parse_from("builtins:///std".to_owned()).expect("valid URI");
    let std = Package::new(
        db,
        PackageName::std(),
        SourceCst::new(db, uri, text, tree),
        DependencyTable::new(db, Vec::new()),
    );
    DependencyTable::new(db, vec![std])
}

/// The packages of `table` with their environments, for name resolution.
pub fn dependencies(db: &dyn salsa::Database, table: DependencyTable) -> Dependencies<'_> {
    Dependencies::new(table.packages(db).iter().filter_map(|&package| {
        Some((package_name(db, package).clone(), package_env(db, package)?))
    }))
}

/// Let `checker` read the exports of every package `table` depends on,
/// transitively, each once and after the packages it depends on.
fn depend_on_table<'db>(
    db: &'db dyn salsa::Database,
    checker: &mut TypeChecker<'db>,
    table: DependencyTable,
) {
    for package in transitive_packages(db, table) {
        if let Some(checked) = checked_package(db, package) {
            checker.depend_on(&checked.exports);
        }
    }
}

/// The packages `table` depends on, transitively, each after the packages
/// it depends on.
fn transitive_packages(db: &dyn salsa::Database, table: DependencyTable) -> Vec<Package> {
    fn visit(db: &dyn salsa::Database, package: Package, order: &mut Vec<Package>) {
        if order.contains(&package) {
            return;
        }
        for &dependency in package.dependencies(db).packages(db) {
            // A package on a cycle is rejected by its environment, so the
            // visit stops at it.
            if !depends_on_itself(db, dependency) {
                visit(db, dependency, order);
            }
        }
        order.push(package);
    }
    let mut order = Vec::new();
    for &package in table.packages(db) {
        if !depends_on_itself(db, package) {
            visit(db, package, &mut order);
        }
    }
    order
}

/// Whether `package` reaches itself through dependency tables.
fn depends_on_itself(db: &dyn salsa::Database, package: Package) -> bool {
    let mut seen = Vec::new();
    let mut stack: Vec<Package> = package.dependencies(db).packages(db).clone();
    while let Some(next) = stack.pop() {
        if next == package {
            return true;
        }
        if !seen.contains(&next) {
            seen.push(next);
            stack.extend(next.dependencies(db).packages(db).iter().copied());
        }
    }
    false
}

/// The name a dependency table names `package` by.
fn package_name(db: &dyn salsa::Database, package: Package) -> &Symbol {
    package
        .name(db)
        .name()
        .expect("a library package has a name")
}

/// How diagnostics spell declarations while `package` is compiled.
pub fn package_names(db: &dyn salsa::Database, package: Package) -> NameDisplay {
    let table = package.dependencies(db).packages(db);
    NameDisplay::new(
        package.name(db),
        table.iter().map(|&dependency| package_name(db, dependency)),
    )
}

/// How diagnostics spell declarations while the executable root package is
/// compiled against `table`.
pub fn root_names(db: &dyn salsa::Database, table: DependencyTable) -> NameDisplay {
    NameDisplay::new(
        &PackageName::executable_root(),
        table
            .packages(db)
            .iter()
            .map(|&package| package_name(db, package)),
    )
}

/// Parse, resolve, and type check `source` against the packages `table`
/// names.
///
/// Returns `None` only if `source` does not parse. Errors are reported as
/// accumulated diagnostics, and the result is returned with them. The result
/// holds only `source`'s declarations; its dependencies' are read through
/// the dependency table.
#[salsa::tracked(returns(copy))]
pub fn analyze<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    table: DependencyTable,
) -> Option<TypeCheckOutput<'db>> {
    let parsed = query::parsed_ast(db, source)?;
    let module = parsed.module(db);
    let span_map = parsed.span_map(db);
    let scope = resolve::build_env(db, module, dependencies(db, table));
    let resolved = resolve::resolve_with_env(db, module, &scope, span_map);

    let mut checker = TypeChecker::new(db, span_map, root_names(db, table));
    depend_on_table(db, &mut checker, table);
    Some(TypeCheckOutput::from_check(
        db,
        checker.check_module(&resolved),
    ))
}

/// The diagnostics [`analyze`] reports for `source` against `table`,
/// including those of its dependencies' own analyses.
pub fn diagnostics(
    db: &dyn salsa::Database,
    source: SourceCst,
    table: DependencyTable,
) -> Vec<&Diagnostic> {
    analyze(db, source, table);
    analyze::accumulated::<Diagnostic>(db, source, table)
}

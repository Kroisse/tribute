//! Frontend analysis of a package source against the packages it depends on.
//!
//! The CLI, the language server, and tests analyze a source through these
//! queries, passing the environment it is analyzed in explicitly. The caller
//! owns where package `std`'s source comes from.
//!
//! [`analyze`] returns its result even when it reports errors, so editor
//! features can read a partial result. Whether a result may be lowered is the
//! caller's decision: the compilation route lowers only a source whose
//! analysis reported no error.

use tribute_core::diagnostic::Diagnostic;
use trunk_ir::Symbol;

use crate::astgen::ParsedAst;
use crate::query;
use crate::resolve::{self, Dependencies, PackageEnv, PackageName};
use crate::source_file::SourceCst;
use crate::typeck::{PreludeExports, TypeCheckOutput, TypeChecker};

/// The environment a source is analyzed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum Environment {
    /// A package `std` that declares only the compiler-owned names.
    Builtins,
    /// Package `std` whose root module is this source, the prelude. A source
    /// that does not parse declares nothing, not even the compiler-owned
    /// names.
    Std(SourceCst),
}

/// Package `std` type checked on its own: the typed module, and the exports
/// a dependent package's type checking reads through the dependency table.
#[derive(Clone, Copy, PartialEq, Eq, salsa::SalsaValue)]
pub struct CheckedStd<'db> {
    pub checked: TypeCheckOutput<'db>,
    pub exports: PreludeExports<'db>,
}

/// Parse the root module of package `std`.
#[salsa::tracked(returns(copy))]
pub fn parsed_std<'db>(db: &'db dyn salsa::Database, std: SourceCst) -> Option<ParsedAst<'db>> {
    query::parsed_ast_with_module_path(db, std, Symbol::new("prelude"))
}

/// The environment of package `std`, built from its declarations alone.
#[salsa::tracked(returns(as_ref))]
pub fn std_env<'db>(db: &'db dyn salsa::Database, std: SourceCst) -> Option<PackageEnv<'db>> {
    let parsed = parsed_std(db, std)?;
    Some(resolve::build_library_env(
        db,
        parsed.module(db),
        &PackageName::std(),
    ))
}

/// Resolve and type check package `std` once.
#[salsa::tracked(returns(copy))]
pub fn checked_std<'db>(db: &'db dyn salsa::Database, std: SourceCst) -> Option<CheckedStd<'db>> {
    let parsed = parsed_std(db, std)?;
    let span_map = parsed.span_map(db);
    let resolved =
        resolve::resolve_library_with_env(db, parsed.module(db), std_env(db, std)?, span_map);
    let (result, exports) = TypeChecker::new(db, span_map).check_prelude(&resolved);
    Some(CheckedStd {
        checked: TypeCheckOutput::from_check(db, result),
        exports,
    })
}

/// The packages a source analyzed in `environment` depends on.
pub fn dependencies(db: &dyn salsa::Database, environment: Environment) -> Dependencies<'_> {
    match environment {
        Environment::Builtins => Dependencies::std(resolve::builtin_library_env(db)),
        Environment::Std(std) => std_env(db, std).map(Dependencies::std).unwrap_or_default(),
    }
}

/// Parse, resolve, and type check `source` in `environment`.
///
/// Returns `None` only if `source` does not parse. Errors are reported as
/// accumulated diagnostics, and the result is returned with them. The result
/// holds only `source`'s declarations; package `std`'s are read through the
/// dependency table.
#[salsa::tracked(returns(copy))]
pub fn analyze<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
    environment: Environment,
) -> Option<TypeCheckOutput<'db>> {
    let parsed = query::parsed_ast(db, source)?;
    let module = parsed.module(db);
    let span_map = parsed.span_map(db);
    let scope = resolve::build_env(db, module, dependencies(db, environment));
    let resolved = resolve::resolve_with_env(db, module, &scope, span_map);

    let mut checker = TypeChecker::new(db, span_map);
    if let Environment::Std(std) = environment
        && let Some(std) = checked_std(db, std)
    {
        checker.depend_on(&std.exports);
    }
    Some(TypeCheckOutput::from_check(
        db,
        checker.check_module(&resolved),
    ))
}

/// The diagnostics [`analyze`] reports for `source` in `environment`,
/// including those of package `std`'s own analysis.
pub fn diagnostics(
    db: &dyn salsa::Database,
    source: SourceCst,
    environment: Environment,
) -> Vec<&Diagnostic> {
    analyze(db, source, environment);
    analyze::accumulated::<Diagnostic>(db, source, environment)
}

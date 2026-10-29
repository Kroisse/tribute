//! Frontend assembly for Tribute.
//!
//! Tracked queries take a user source through parsing, name resolution, and
//! type checking against package std, check compiler intrinsics, specialize
//! the user's package, merge it with std's own specialization, and lower the
//! result to source-logical arena IR. Std's specialization and runtime items depend on
//! std alone, so an edit of the user source reuses them. The stage chain in
//! [`crate::pipeline`] runs on that IR.

use crate::SourceCst;
use crate::prelude::root_environment;
use salsa::Accumulator;
use tribute_core::diagnostic::{Diagnostic, DiagnosticSeverity};
use tribute_front::ast::SpanMaps;
use tribute_front::ast_to_ir;
use tribute_front::monomorphize::{self, SpecializationInput, SpecializationOutput};
use tribute_front::resolve::NameDisplay;
use tribute_front::source_tree::source_tree_of;
use tribute_front::typeck as ast_typeck;
use tribute_front::{analysis, manifest};
use trunk_ir::{IrContext, Module};

/// How diagnostics and editor features spell declarations while a user
/// source is analyzed in the root crate's environment: as the root package
/// of an executable build, depending on the packages of its manifest and
/// package `std`.
pub fn diagnostic_names(db: &dyn salsa::Database, source: SourceCst) -> NameDisplay {
    analysis::root_names(db, root_environment(db, source))
}

/// Analyze a user source in the root crate's environment: the packages its
/// package's manifest names, and package `std` from the prelude source (see
/// [`tribute_front::analysis::analyze`] and
/// [`root_environment`](crate::prelude::root_environment)).
///
/// Returns the typed AST, even when the analysis reports errors; returns
/// `None` only if the source does not parse. Does NOT call `ast_to_ir` —
/// that is the caller's responsibility.
pub fn parse_and_lower_ast(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<ast_typeck::TypeCheckOutput<'_>> {
    analysis::analyze(db, source, root_environment(db, source)).copied()
}

/// The diagnostics [`parse_and_lower_ast`] reports for `source`, with those
/// of its package's manifests.
pub fn frontend_diagnostics(db: &dyn salsa::Database, source: SourceCst) -> Vec<&Diagnostic> {
    let implicit = *crate::prelude::frontend_environment(db);
    let mut diagnostics =
        manifest::source_dependencies::accumulated::<Diagnostic>(db, source, implicit);
    diagnostics.extend(analysis::diagnostics(
        db,
        source,
        root_environment(db, source),
    ));
    diagnostics
}

/// Check the library package `package` on its own, through
/// specialization, as a build that depends on it would. Returns whether it
/// checked; read its diagnostics with `check_library::accumulated`.
#[salsa::tracked(returns(copy))]
pub fn check_library(db: &dyn salsa::Database, package: analysis::Package) -> bool {
    analysis::checked_package(db, package).is_some()
        && monomorphize::specialized_package(db, package).is_some()
}

/// Merge prelude decls into user's typed AST and lower to arena IR.
///
/// This performs AST-level prelude merge (prepending prelude decls to user decls),
/// then runs `ast_to_ir` on the merged module.
///
/// Returns the logical frontend result, including the metadata required by the
/// shared CPS boundary.
pub(crate) fn merge_and_lower_to_ir<'db>(
    db: &'db dyn salsa::Database,
    typed: &ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> FrontendCompilation {
    let prepared = prepare_frontend_details(db, *typed, source)
        .expect("frontend instances must be checked before lowering");
    let mut context = IrContext::new();
    let frontend = ast_to_ir::LoweringInput::from_specialization(
        db,
        prepared.specialized,
        ast_to_ir::LoweringEnvironment {
            span_map: frontend_spans(db, source),
            runtime_types: prepared.runtime_types.clone(),
            merged_sources: merged_sources(db, source),
        },
    )
    .lower_to_ir(db, &mut context, source.uri(db).as_str());
    FrontendCompilation {
        context,
        module: frontend.module,
        compiler_intrinsics: frontend.compiler_intrinsics,
        names: diagnostic_names(db, source),
    }
}

/// Specialized frontend output, with the runtime types root-crate
/// preparation supplies to lowering.
#[derive(Clone, PartialEq, Eq, salsa::SalsaValue)]
struct PreparedFrontend<'db> {
    specialized: SpecializationOutput<'db>,
    runtime_types: ast_to_ir::runtime_items::RuntimeTypes,
}

/// Specialize the checked user module and merge it with the specializations
/// of the packages it depends on.
///
/// Returns `None` after reporting why it does not specialize; read those
/// diagnostics with [`prepare_frontend_diagnostics`].
pub fn prepare_frontend_for_lowering<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> Option<SpecializationOutput<'db>> {
    prepare_frontend_details(db, typed, source).map(|prepared| prepared.specialized)
}

/// The diagnostics [`prepare_frontend_for_lowering`] reports.
pub fn prepare_frontend_diagnostics<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> Vec<&'db Diagnostic> {
    prepare_frontend_details(db, typed, source);
    prepare_frontend_details::accumulated::<Diagnostic>(db, typed, source)
}

/// Merge and specialize inside a tracked query so specialization failures
/// become source diagnostics before either public IR lowering route runs.
#[salsa::tracked(returns(as_ref))]
fn prepare_frontend_details<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> Option<PreparedFrontend<'db>> {
    // Type checking reads package std's declarations without copying them.
    // The user's package is specialized on its own, cloning the
    // dependencies' generic bodies, and merged with each dependency's own
    // specialization; the result has the dependencies' bodies and
    // per-expression metadata ahead of the user's. Without a prelude there
    // are no runtime declarations to check.
    let runtime_types = match crate::prelude::prelude_module(db) {
        Some(_) => crate::prelude::prelude_runtime_types(db)?.clone(),
        None => Default::default(),
    };
    let environment = root_environment(db, source);
    let names = diagnostic_names(db, source);
    let input = SpecializationInput::package(
        db,
        typed,
        monomorphize::checked_dependencies(db, environment),
        names.clone(),
    );
    let span_map = frontend_spans(db, source);

    let compiler_intrinsics = match ast_to_ir::registered_compiler_intrinsics(input.module()) {
        Ok(registered) => registered,
        Err(unsupported) => {
            for directive in unsupported {
                directive.diagnostic(&span_map, &names).accumulate(db);
            }
            return None;
        }
    };
    let input = input.with_compiler_intrinsics(compiler_intrinsics);
    let specialized = monomorphize::monomorphize_functions(db, input).and_then(|own| {
        let dependencies = monomorphize::specialized_dependencies(db, environment);
        // A dependency that does not specialize has reported why.
        let Some(dependencies) = dependencies else {
            return Ok(None);
        };
        monomorphize::merge_specializations(db, dependencies.into_iter().chain([own])).map(Some)
    });
    let specialized = match specialized {
        Ok(specialized) => specialized?,
        Err(errors) => {
            for error in errors {
                error.diagnostic(&span_map).accumulate(db);
            }
            return None;
        }
    };
    Some(PreparedFrontend {
        specialized,
        runtime_types,
    })
}

/// The span maps of a user source's package, with its file modules, and of
/// the packages it depends on, which locate the nodes of the module merged
/// from all of them.
pub(crate) fn frontend_spans(db: &dyn salsa::Database, source: SourceCst) -> SpanMaps<'_> {
    let package = source_tree_of(db, source).map(|tree| tree.span_map(db));
    let dependencies = analysis::transitive_packages(db, root_environment(db, source))
        .into_iter()
        .filter_map(|package| analysis::package_tree(db, package))
        .map(|tree| tree.span_map(db));
    SpanMaps::new(package.into_iter().chain(dependencies))
}

/// The sources merged into a user source's module besides itself: the files
/// of its file modules, and those of the packages it depends on.
fn merged_sources(db: &dyn salsa::Database, source: SourceCst) -> Vec<String> {
    let own = source_tree_of(db, source).map_or(&[][..], |tree| &tree.files()[1..]);
    let dependencies = analysis::transitive_packages(db, root_environment(db, source))
        .into_iter()
        .filter_map(|package| analysis::package_tree(db, package));
    own.iter()
        .chain(dependencies.flat_map(|tree| tree.files()))
        .map(|file| file.source.uri(db).as_str().to_owned())
        .collect()
}

/// Arena IR together with the registered compiler intrinsics the shared CPS
/// conversion checks. They stay private; [`run_shared_middle_end`](crate::pipeline::run_shared_middle_end) is their
/// only consumer.
#[derive(Clone)]
pub struct FrontendCompilation {
    pub(crate) context: IrContext,
    pub(crate) module: Module,
    pub(crate) compiler_intrinsics:
        Vec<tribute_ir::dialect::tribute_control::CompilerIntrinsicDeclaration>,
    /// How diagnostics of the compiled package spell declarations.
    pub(crate) names: NameDisplay,
}

/// Run frontend (parse → typecheck) and lower to arena IR.
///
/// Returns `None` if parsing fails. Otherwise returns arena IR ready
/// for in-place passes, avoiding unnecessary Salsa↔Arena round-trips.
pub fn compile_frontend(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<(IrContext, Module)> {
    let compilation = compile_frontend_for_shared_route(db, source)?;
    Some((compilation.context, compilation.module))
}

/// Run the frontend for the shared middle-end.
///
/// Returns `None` if parsing fails or the frontend reports an error.
pub fn compile_frontend_for_shared_route(
    db: &dyn salsa::Database,
    source: SourceCst,
) -> Option<FrontendCompilation> {
    let typed = parse_and_lower_ast(db, source)?;
    let has_frontend_errors = frontend_diagnostics(db, source)
        .iter()
        .any(|diagnostic| diagnostic.inner.severity == DiagnosticSeverity::Error);
    if has_frontend_errors {
        return None;
    }
    prepare_frontend_for_lowering(db, typed, source)?;
    Some(merge_and_lower_to_ir(db, &typed, source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ropey::Rope;
    use rustc_hash::FxHashMap as HashMap;
    use salsa_test_macros::salsa_test;
    use tribute_front::monomorphize::monomorphize_functions;
    use tribute_front::source_file::parse_rope;

    /// Std's bodies merged into the user's and specialized with them.
    #[salsa::tracked(returns(copy))]
    fn merged_specialization<'db>(
        db: &'db dyn salsa::Database,
        source: SourceCst,
    ) -> Option<SpecializationOutput<'db>> {
        let typed = parse_and_lower_ast(db, source)?;
        let prelude = crate::prelude::prelude_module(db)?;
        let input =
            SpecializationInput::new(db, typed, Some(prelude), diagnostic_names(db, source))
                .merge_bodies(db, [prelude]);
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

    /// Assert that merging the per-package specializations gives the result
    /// of specializing the merged bodies at once: the same sources in order,
    /// and the same instances and metadata. Returns how many instances the
    /// merged result has beyond std's.
    fn assert_packages_merge_like_bodies(
        db: &dyn salsa::Database,
        name: &str,
        source: SourceCst,
    ) -> usize {
        let typed = parse_and_lower_ast(db, source)
            .unwrap_or_else(|| panic!("{name}: the source does not parse"));
        let merged = merged_specialization(db, source)
            .unwrap_or_else(|| panic!("{name}: the merged specialization fails"));
        let packaged = prepare_frontend_for_lowering(db, typed, source)
            .unwrap_or_else(|| panic!("{name}: the per-package specialization fails"));
        let sources = *merged.source_entries(db);
        assert_eq!(sources, *packaged.source_entries(db), "{name}: sources");
        let (merged_module, packaged_module) = (merged.module(db), packaged.module(db));
        assert_eq!(
            merged_module.entries[..sources],
            packaged_module.entries[..sources],
            "{name}: source entries"
        );
        assert_eq!(
            counts(&merged_module.entries[sources..]),
            counts(&packaged_module.entries[sources..]),
            "{name}: instances"
        );
        let (merged_decls, packaged_decls) = (merged.declarations(db), packaged.declarations(db));
        assert_eq!(
            counts(&merged_decls.function_types),
            counts(&packaged_decls.function_types),
            "{name}: function types"
        );
        assert_eq!(
            counts(&merged_decls.constructor_types.schemes),
            counts(&packaged_decls.constructor_types.schemes),
            "{name}: constructor schemes"
        );
        assert_eq!(
            merged_decls.constructor_types.specialized_enum_variants,
            packaged_decls.constructor_types.specialized_enum_variants,
            "{name}: enum variants"
        );
        assert_eq!(
            merged_decls.ability_definitions, packaged_decls.ability_definitions,
            "{name}: ability definitions"
        );
        assert_eq!(
            merged_decls.well_known_types, packaged_decls.well_known_types,
            "{name}: well-known types"
        );
        assert!(
            merged.expression_types(db) == packaged.expression_types(db),
            "{name}: expression metadata"
        );
        assert_eq!(
            merged.compiler_intrinsics(db),
            packaged.compiler_intrinsics(db),
            "{name}: compiler intrinsics"
        );
        assert_eq!(
            merged.function_instances(db),
            packaged.function_instances(db),
            "{name}: function instances"
        );
        // Instances are sorted by path, and those that share one stay in
        // hash order.
        assert_eq!(
            counts(merged.nominal_instances(db)),
            counts(packaged.nominal_instances(db)),
            "{name}: nominal instances"
        );
        let prelude = *monomorphize::specialized_package(db, *crate::prelude::std_package(db))
            .expect("std specializes");
        packaged_module.entries.len()
            - prelude.module(db).entries.len()
            - (sources - *prelude.source_entries(db))
    }

    #[salsa_test]
    fn per_package_specialization_matches_the_merged_one(db: &crate::TributeDatabaseImpl) {
        let source = SourceCst::from_source_str(
            db,
            "packaged.trb",
            r#"
use std::io::{Io, print_line}

struct Point { x: Int, y: Int }

enum Shape(a) {
    Circle(a)
    Square(a, a)
}

fn identity(x: a) -> a { x }

fn checked(point: Point) -> Result(Point, Int) { .Ok(point) }

fn first(shape: Shape(a)) -> Option(a) {
    case shape {
        .Circle(r) -> .Some(r)
        .Square(w, _) -> .Some(w)
    }
}

fn main() ->{Io} Nil {
    let point = Point { x: +1, y: +2 }
    let shapes = first(.Circle(point))
    let moved = shapes.map(fn(p) { Point { x: p.x, y: +3 } })
    let numbers = Option::map(.Some(+1), fn(n) { n })
    let kept = Option::unwrap_or(numbers, +0)
    let pair = identity(Result::map(checked(point), fn(p) { p.y }))
    print_line(Int::to_string(Result::unwrap_or(pair, kept)))
}
"#,
        );
        assert!(
            frontend_diagnostics(db, source).is_empty(),
            "{:?}",
            frontend_diagnostics(db, source)
        );
        assert!(assert_packages_merge_like_bodies(db, "packaged.trb", source) > 0);
    }

    #[salsa_test]
    fn per_package_specialization_matches_merging_on_examples(db: &crate::TributeDatabaseImpl) {
        fn examples(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).expect("lang-examples directory") {
                let path = entry.expect("directory entry").path();
                if path.is_dir() {
                    examples(&path, out);
                } else if path.extension().is_some_and(|extension| extension == "trb") {
                    out.push(path);
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lang-examples");
        let mut files = Vec::new();
        examples(&root, &mut files);
        files.sort();
        assert!(!files.is_empty());
        let mut compared = 0;
        for file in &files {
            let text = Rope::from_str(&std::fs::read_to_string(file).expect("example source"));
            let tree = parse_rope(&text, None);
            let source = SourceCst::from_path(db, file, text, tree);
            let has_errors = frontend_diagnostics(db, source)
                .iter()
                .any(|diagnostic| diagnostic.inner.severity == DiagnosticSeverity::Error);
            if !has_errors {
                assert_packages_merge_like_bodies(db, &file.display().to_string(), source);
                compared += 1;
            }
        }
        assert!(
            compared > files.len() / 2,
            "{compared} of {} compared",
            files.len()
        );
    }
}

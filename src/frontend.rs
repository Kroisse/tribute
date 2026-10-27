//! Frontend assembly for Tribute.
//!
//! Tracked queries take a user source through parsing, name resolution, and
//! type checking against package std, check compiler intrinsics, continue
//! std's own specialization with the user's bodies, and lower the result to
//! source-logical arena IR. Std's specialization and runtime items depend on
//! std alone, so an edit of the user source reuses them. The stage chain in
//! [`crate::pipeline`] runs on that IR.

use crate::SourceCst;
use salsa::Accumulator;
use tribute_core::diagnostic::{Diagnostic, DiagnosticSeverity};
use tribute_front::analysis;
use tribute_front::ast::SpanMaps;
use tribute_front::ast_to_ir;
use tribute_front::monomorphize::{SpecializationInput, SpecializationOutput};
use tribute_front::query as ast_query;
use tribute_front::typeck as ast_typeck;
use trunk_ir::{IrContext, Module};

/// Analyze a user source in the root crate's environment, package `std`
/// from the prelude source (see [`tribute_front::analysis::analyze`]).
///
/// Returns the typed AST, even when the analysis reports errors; returns
/// `None` only if the source does not parse. Does NOT call `ast_to_ir` —
/// that is the caller's responsibility.
#[salsa::tracked(returns(copy))]
pub fn parse_and_lower_ast<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<ast_typeck::TypeCheckOutput<'db>> {
    analysis::analyze(db, source, crate::prelude::frontend_environment(db))
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
            merged_sources: vec![crate::prelude::PRELUDE_URI.to_owned()],
        },
    )
    .lower_to_ir(db, &mut context, source.uri(db).as_str());
    FrontendCompilation {
        context,
        module: frontend.module,
        compiler_intrinsics: frontend.compiler_intrinsics,
    }
}

/// Specialized frontend output, with the runtime types root-crate
/// preparation supplies to lowering.
#[derive(Clone, PartialEq, Eq, salsa::SalsaValue)]
struct PreparedFrontend<'db> {
    specialized: SpecializationOutput<'db>,
    runtime_types: ast_to_ir::runtime_items::RuntimeTypes,
}

/// Merge and specialize inside a tracked query so specialization failures
/// become source diagnostics before either public IR lowering route runs.
#[salsa::tracked(returns(copy))]
pub fn prepare_frontend_for_lowering<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> Option<SpecializationOutput<'db>> {
    prepare_frontend_details(db, typed, source).map(|prepared| prepared.specialized)
}

#[salsa::tracked(returns(as_ref))]
fn prepare_frontend_details<'db>(
    db: &'db dyn salsa::Database,
    typed: ast_typeck::TypeCheckOutput<'db>,
    source: SourceCst,
) -> Option<PreparedFrontend<'db>> {
    // Type checking reads package std's declarations without copying them.
    // Specialization continues std's own, which depends on std alone, with
    // the user's bodies; the result has std's bodies and per-expression
    // metadata ahead of the user's.
    let prelude = match crate::prelude::prelude_module(db) {
        Some(_) => Some(crate::prelude::specialized_prelude(db)?),
        None => None,
    };
    let input = match prelude {
        Some(prelude) => {
            SpecializationInput::continuing(db, typed, prelude.checked, prelude.specialized)
        }
        None => SpecializationInput::new(db, typed, None),
    };
    let span_map = frontend_spans(db, source);

    let compiler_intrinsics = match ast_to_ir::registered_compiler_intrinsics(input.module()) {
        Ok(registered) => registered,
        Err(unsupported) => {
            for directive in unsupported {
                directive.diagnostic(&span_map).accumulate(db);
            }
            return None;
        }
    };
    // Without a prelude there are no runtime declarations to check.
    let runtime_types = prelude
        .map(|prelude| prelude.runtime_types.clone())
        .unwrap_or_default();

    let input = input.with_compiler_intrinsics(compiler_intrinsics);
    let specialized = match tribute_front::monomorphize::monomorphize_functions(db, input) {
        Ok(specialized) => specialized,
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

/// The span maps of a user source and of package std, which locate the
/// nodes of the module merged from both.
pub(crate) fn frontend_spans(db: &dyn salsa::Database, source: SourceCst) -> SpanMaps<'_> {
    let prelude = crate::prelude::prelude_span_map(db);
    SpanMaps::new(ast_query::span_map(db, source).into_iter().chain(prelude))
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
    let has_frontend_errors = parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
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
        let input = SpecializationInput::new(db, typed, Some(prelude)).merge_bodies(db, prelude);
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

    /// Assert that continuing std's specialization gives the merged result:
    /// the same sources in order, and the same instances and metadata.
    /// Returns how many instances the user module adds to std's.
    fn assert_continues_merged(db: &dyn salsa::Database, name: &str, source: SourceCst) -> usize {
        let typed = parse_and_lower_ast(db, source)
            .unwrap_or_else(|| panic!("{name}: the source does not parse"));
        let merged = merged_specialization(db, source)
            .unwrap_or_else(|| panic!("{name}: the merged specialization fails"));
        let continued = prepare_frontend_for_lowering(db, typed, source)
            .unwrap_or_else(|| panic!("{name}: the continued specialization fails"));
        let sources = *merged.source_entries(db);
        assert_eq!(sources, *continued.source_entries(db), "{name}: sources");
        let (merged_module, continued_module) = (merged.module(db), continued.module(db));
        assert_eq!(
            merged_module.entries[..sources],
            continued_module.entries[..sources],
            "{name}: source entries"
        );
        assert_eq!(
            counts(&merged_module.entries[sources..]),
            counts(&continued_module.entries[sources..]),
            "{name}: instances"
        );
        let (merged_decls, continued_decls) = (merged.declarations(db), continued.declarations(db));
        assert_eq!(
            counts(&merged_decls.function_types),
            counts(&continued_decls.function_types),
            "{name}: function types"
        );
        assert_eq!(
            counts(&merged_decls.constructor_types.schemes),
            counts(&continued_decls.constructor_types.schemes),
            "{name}: constructor schemes"
        );
        assert_eq!(
            merged_decls.constructor_types.specialized_enum_variants,
            continued_decls.constructor_types.specialized_enum_variants,
            "{name}: enum variants"
        );
        assert_eq!(
            merged_decls.ability_definitions, continued_decls.ability_definitions,
            "{name}: ability definitions"
        );
        assert_eq!(
            merged_decls.well_known_types, continued_decls.well_known_types,
            "{name}: well-known types"
        );
        assert!(
            merged.expression_types(db) == continued.expression_types(db),
            "{name}: expression metadata"
        );
        assert_eq!(
            merged.compiler_intrinsics(db),
            continued.compiler_intrinsics(db),
            "{name}: compiler intrinsics"
        );
        assert_eq!(
            merged.function_instances(db),
            continued.function_instances(db),
            "{name}: function instances"
        );
        assert_eq!(
            merged.nominal_instances(db),
            continued.nominal_instances(db),
            "{name}: nominal instances"
        );
        let prelude = crate::prelude::specialized_prelude(db)
            .as_ref()
            .expect("std specializes")
            .specialized;
        continued_module.entries.len()
            - prelude.module(db).entries.len()
            - (sources - *prelude.source_entries(db))
    }

    #[salsa_test]
    fn continuing_std_matches_the_merged_specialization(db: &crate::TributeDatabaseImpl) {
        let source = SourceCst::from_source_str(
            db,
            "continued.trb",
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
            parse_and_lower_ast::accumulated::<Diagnostic>(db, source).is_empty(),
            "{:?}",
            parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
        );
        assert!(assert_continues_merged(db, "continued.trb", source) > 0);
    }

    #[salsa_test]
    fn continuing_std_matches_merging_on_examples(db: &crate::TributeDatabaseImpl) {
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
            let has_errors = parse_and_lower_ast::accumulated::<Diagnostic>(db, source)
                .iter()
                .any(|diagnostic| diagnostic.inner.severity == DiagnosticSeverity::Error);
            if !has_errors {
                assert_continues_merged(db, &file.display().to_string(), source);
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

//! The prelude source: the root module of package `std`.

use ropey::Rope;
use rustc_hash::FxHashMap as HashMap;
use salsa::Accumulator;
use tree_sitter::Parser;
use tribute_front::analysis;
use tribute_front::ast::{SpanMap, SpanMaps};
use tribute_front::ast_to_ir;
use tribute_front::monomorphize::{self, SpecializationInput, SpecializationOutput};
use tribute_front::resolve as ast_resolve;
use tribute_front::source_file::parse_with_rope;
use tribute_front::typeck as ast_typeck;

/// The prelude source code, embedded at compile time.
const PRELUDE_SOURCE: &str = include_str!("../lib/std/prelude.trb");

/// The URI the prelude source is registered under.
pub(crate) const PRELUDE_URI: &str = "prelude:///std/prelude";

/// Source appended to the prelude and compiled as part of package `std`.
///
/// Tests use it to exercise package-internal items, such as the
/// compiler-owned runtime under `std::runtime`, that user packages cannot
/// name. Create it before the first query on a database: an absent input
/// records no dependency, so creating it later recompiles nothing. Its
/// declarations are prelude declarations and can shadow the prelude's own.
#[salsa::input(singleton)]
pub struct PreludeExtension {
    pub text: String,
}

/// The prelude source, with any [`PreludeExtension`] appended.
///
/// Tracked so that every query in a revision sees the same source input.
#[salsa::tracked(returns(copy))]
pub(crate) fn prelude_source(db: &dyn salsa::Database) -> crate::SourceCst {
    let uri = fluent_uri::Uri::parse_from(PRELUDE_URI.to_owned()).expect("valid prelude URI");
    let text: Rope = match PreludeExtension::try_get(db) {
        Some(extension) => format!("{PRELUDE_SOURCE}\n{}", extension.text(db)).into(),
        None => PRELUDE_SOURCE.into(),
    };
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");
    let tree = parse_with_rope(&mut parser, &text, None);
    crate::SourceCst::new(db, uri, text, tree)
}

/// The environment the root crate analyzes user sources in: package `std`
/// from the prelude source.
pub fn frontend_environment(db: &dyn salsa::Database) -> analysis::Environment {
    analysis::Environment::Std(prelude_source(db))
}

/// Package `std` specialized alone, with the runtime types its runtime
/// module declares. Both depend only on `std`'s checked result, so a user
/// package's specialization continues this result without repeating it.
///
/// Returns `None` without a prelude, or after reporting why `std` does not
/// specialize.
#[salsa::tracked(returns(as_ref))]
pub(crate) fn specialized_prelude<'db>(
    db: &'db dyn salsa::Database,
) -> Option<SpecializedPrelude<'db>> {
    let checked = prelude_module(db)?;
    let span_map = SpanMaps::new(prelude_span_map(db));
    let compiler_intrinsics = match ast_to_ir::registered_compiler_intrinsics(checked.module(db)) {
        Ok(registered) => registered,
        Err(unsupported) => {
            for directive in unsupported {
                directive.diagnostic(&span_map).accumulate(db);
            }
            return None;
        }
    };
    let function_types: HashMap<_, _> = checked
        .declarations(db)
        .function_types
        .iter()
        .cloned()
        .collect();
    let runtime_types = match ast_to_ir::runtime_items::check_runtime(
        db,
        &ast_to_ir::runtime_items::RuntimeInputs {
            module: checked.module(db),
            function_types: &function_types,
        },
        tribute_ir::runtime_items::RUNTIME_MODULE,
        tribute_ir::runtime_items::RUNTIME_ITEMS,
    ) {
        Ok(types) => types,
        Err(errors) => {
            for error in errors {
                error.diagnostic(&span_map).accumulate(db);
            }
            return None;
        }
    };
    let input =
        SpecializationInput::new(db, checked, None).with_compiler_intrinsics(compiler_intrinsics);
    match monomorphize::monomorphize_functions(db, input) {
        Ok(specialized) => Some(SpecializedPrelude {
            checked,
            specialized,
            runtime_types,
        }),
        Err(errors) => {
            for error in errors {
                error.diagnostic(&span_map).accumulate(db);
            }
            None
        }
    }
}

/// Package `std`'s checked and specialized results.
#[derive(Clone, PartialEq, Eq, salsa::SalsaValue)]
pub(crate) struct SpecializedPrelude<'db> {
    pub(crate) checked: ast_typeck::TypeCheckOutput<'db>,
    pub(crate) specialized: SpecializationOutput<'db>,
    pub(crate) runtime_types: ast_to_ir::runtime_items::RuntimeTypes,
}

/// The span map of the prelude's parse, which locates its nodes in a module
/// merged with it.
pub(crate) fn prelude_span_map(db: &dyn salsa::Database) -> Option<&SpanMap> {
    analysis::parsed_std(db, prelude_source(db)).map(|parsed| parsed.span_map(db))
}

/// The prelude's typed AST (parse → resolve → typecheck), without
/// lowering to TrunkIR. The caller is responsible for `ast_to_ir`.
pub(crate) fn prelude_module(db: &dyn salsa::Database) -> Option<ast_typeck::TypeCheckOutput<'_>> {
    analysis::checked_std(db, prelude_source(db)).map(|std| std.checked)
}

/// The packages a user package depends on: package `std`.
pub fn prelude_dependencies<'db>(db: &'db dyn salsa::Database) -> ast_resolve::Dependencies<'db> {
    analysis::dependencies(db, frontend_environment(db))
}

/// The prelude's type exports (TypeSchemes only, no UniVars), which user
/// module type checking reads through the dependency table.
#[cfg(test)]
pub(crate) fn prelude_exports(
    db: &dyn salsa::Database,
) -> Option<tribute_front::typeck::PreludeExports<'_>> {
    analysis::checked_std(db, prelude_source(db)).map(|std| std.exports)
}

//! Each library package's specialization, continuing those of the packages
//! it depends on (`new-plans/implementation.md`, "프론트엔드 단계 경계와 정보
//! 소유권").

use salsa::Accumulator;

use super::{SpecializationInput, SpecializationOutput, monomorphize_functions};
use crate::analysis::{self, DependencyTable, Package};
use crate::ast::SpanMaps;
use crate::ast_to_ir;
use crate::typeck::TypeCheckOutput;

/// The specialization of `package`, continuing those of the packages it
/// depends on. It depends only on their checked and specialized results, so
/// a package that depends on it continues it without repeating it.
///
/// Returns `None` if `package` does not check, or after reporting why it or
/// a package it depends on does not specialize.
#[salsa::tracked(returns(as_ref))]
pub fn specialized_package<'db>(
    db: &'db dyn salsa::Database,
    package: Package,
) -> Option<SpecializationOutput<'db>> {
    let checked = analysis::checked_package(db, package)?.checked;
    let table = *package.dependencies(db);
    let dependencies = specialized_dependencies(db, table)?;
    let names = analysis::package_names(db, package);
    let span_maps = SpanMaps::new(
        [package]
            .into_iter()
            .chain(analysis::transitive_packages(db, table))
            .filter_map(|package| analysis::parsed_package(db, package))
            .map(|parsed| parsed.span_map(db)),
    );
    let compiler_intrinsics = match ast_to_ir::registered_compiler_intrinsics(checked.module(db)) {
        Ok(registered) => registered,
        Err(unsupported) => {
            for directive in unsupported {
                directive.diagnostic(&span_maps, &names).accumulate(db);
            }
            return None;
        }
    };
    let input = SpecializationInput::continuing(db, checked, dependencies, names)
        .with_compiler_intrinsics(compiler_intrinsics);
    match monomorphize_functions(db, input) {
        Ok(specialized) => Some(specialized),
        Err(errors) => {
            for error in errors {
                error.diagnostic(&span_maps).accumulate(db);
            }
            None
        }
    }
}

/// The packages `table` depends on, transitively, each after the packages it
/// depends on, with each one's checked result and its specialization: what
/// [`SpecializationInput::continuing`] continues. A package that does not
/// check is left out. Returns `None` if one that checks does not specialize.
pub fn specialized_dependencies(
    db: &dyn salsa::Database,
    table: DependencyTable,
) -> Option<Vec<(TypeCheckOutput<'_>, SpecializationOutput<'_>)>> {
    analysis::transitive_packages(db, table)
        .into_iter()
        .filter_map(|package| {
            let checked = analysis::checked_package(db, package)?.checked;
            Some(specialized_package(db, package).map(|specialized| (checked, *specialized)))
        })
        .collect()
}

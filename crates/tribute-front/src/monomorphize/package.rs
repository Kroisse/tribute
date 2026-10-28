//! Each library package's specialization, which reads the packages it
//! depends on only through their checked results
//! (`new-plans/implementation.md`, "프론트엔드 단계 경계와 정보 소유권").

use salsa::Accumulator;

use super::{SpecializationInput, SpecializationOutput, monomorphize_functions};
use crate::analysis::{self, DependencyTable, Package};
use crate::ast::SpanMaps;
use crate::ast_to_ir;
use crate::typeck::TypeCheckOutput;

/// The specialization of `package`: its own sources and every instance
/// they reach. It depends only on the checked results of `package` and the
/// packages it depends on, so what another package's specialization
/// generates never changes it. A build merges it with the other packages'
/// results ([`merge_specializations`](super::merge_specializations)).
///
/// Returns `None` if `package` does not check, or after reporting why it
/// does not specialize.
#[salsa::tracked(returns(as_ref))]
pub fn specialized_package<'db>(
    db: &'db dyn salsa::Database,
    package: Package,
) -> Option<SpecializationOutput<'db>> {
    let checked = analysis::checked_package(db, package)?.checked;
    let table = *package.dependencies(db);
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
    let input = SpecializationInput::package(db, checked, checked_dependencies(db, table), names)
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

/// The checked results of the packages `table` depends on, transitively,
/// each after the packages it depends on. A package that does not check is
/// left out.
pub fn checked_dependencies(
    db: &dyn salsa::Database,
    table: DependencyTable,
) -> Vec<TypeCheckOutput<'_>> {
    analysis::transitive_packages(db, table)
        .into_iter()
        .filter_map(|package| Some(analysis::checked_package(db, package)?.checked))
        .collect()
}

/// The specializations of the packages `table` depends on, transitively,
/// each after the packages it depends on: what a build merges ahead of the
/// package it compiles. A package that does not check is left out. Returns
/// `None` if one that checks does not specialize.
pub fn specialized_dependencies(
    db: &dyn salsa::Database,
    table: DependencyTable,
) -> Option<Vec<SpecializationOutput<'_>>> {
    analysis::transitive_packages(db, table)
        .into_iter()
        .filter_map(|package| {
            analysis::checked_package(db, package)?;
            Some(specialized_package(db, package).copied())
        })
        .collect()
}

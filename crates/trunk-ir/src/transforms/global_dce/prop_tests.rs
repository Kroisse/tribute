//! Property test comparing global DCE with a naive reachability search on
//! random modules.
//!
//! The modules are random [`ModuleSpec`]s: functions in the root module and
//! a nested module, with `abi` definitions, bodyless `abi` imports, calls and
//! address references between functions and externals in reachable and
//! unreachable blocks, and exports. The oracle reads the roots from the pass contract and the
//! edges from the spec, and keeps exactly the functions a search from the
//! roots reaches, plus the functions the configuration does not analyze.

use proptest::prelude::*;

use super::*;
use crate::prop::{Built, ModuleSpec, built, module_spec, reachable_from};

fn config() -> impl Strategy<Value = (bool, Vec<bool>)> {
    (
        prop::bool::weighted(0.75),
        prop::collection::vec(prop::bool::weighted(0.15), 12),
    )
}

/// The indices of the functions the pass must remove.
fn expected_removed(spec: &ModuleSpec, recursive: bool, extra: &[bool]) -> HashSet<usize> {
    let candidate = |index: usize| recursive || !spec.functions[index].nested;
    let roots = spec
        .functions
        .iter()
        .enumerate()
        .filter(|&(index, function)| {
            let entry = !function.nested && (function.name == "main" || function.name == "_start");
            let exported = function.abi && function.body.is_some();
            !candidate(index) || entry || exported || extra[index]
        })
        .map(|(index, _)| index)
        .chain(spec.module_refs.iter().map(|reference| reference.target));
    let reachable = reachable_from(&spec.successors(false), roots, None);
    (0..spec.functions.len())
        .filter(|&index| candidate(index) && !reachable[index])
        .collect()
}

fn surviving_functions(ctx: &IrContext, module: Module) -> HashSet<SymbolPath> {
    SymbolTable::collect(ctx, module)
        .all_definitions()
        .filter(|&(_, op)| func::Func::matches(ctx, op))
        .map(|(name, _)| name.clone())
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn global_dce_keeps_exactly_the_functions_reachable_from_roots(
        built in built(module_spec(12)),
        (recursive, extra) in config(),
    ) {
        let Built { spec, mut ctx, module } = built;
        let paths = spec.paths();
        let extra = &extra[..spec.functions.len()];
        let config = GlobalDceConfig {
            extra_entry_points: paths
                .iter()
                .zip(extra)
                .filter(|&(_, &extra)| extra)
                .map(|(path, _)| path.clone())
                .collect(),
            recursive,
        };

        let result = eliminate_dead_functions_with_config(
            &mut ctx,
            module,
            config,
            &mut AnalysisCache::new(),
        );

        let removed = expected_removed(&spec, recursive, extra);
        let removed_paths: HashSet<SymbolPath> =
            removed.iter().map(|&index| paths[index].clone()).collect();
        prop_assert_eq!(result.removed_count, removed.len());
        prop_assert_eq!(
            result.removed_functions.iter().cloned().collect::<HashSet<_>>(),
            removed_paths.clone()
        );
        let survivors: HashSet<SymbolPath> = (0..spec.functions.len())
            .filter(|index| !removed.contains(index))
            .map(|index| paths[index].clone())
            .collect();
        prop_assert_eq!(surviving_functions(&ctx, module), survivors);
    }
}

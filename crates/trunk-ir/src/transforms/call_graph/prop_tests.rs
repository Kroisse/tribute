//! Property tests comparing the call graph and its SCCs with naive
//! reference implementations on random modules.
//!
//! Each case is a [`ModuleSpec`](crate::prop::ModuleSpec): functions in the root module and in one
//! nested module, with multi-block bodies whose blocks reference other
//! functions or undefined externals by direct calls, `func.constant`, and
//! symbol references on operations that are not calls, including blocks
//! unreachable from the entry. Module-level operations reference functions
//! as exports. The oracle reads the edges straight from the spec: SCCs are
//! the classes of mutual reachability, and a function is recursive when it
//! reaches itself.

use proptest::prelude::*;

use super::*;
use crate::prop::{Built, RefKind, built, module_spec, reachable_from};

/// Whether `from` reaches `to` along at least one edge.
fn reaches(successors: &[Vec<usize>], from: usize, to: usize) -> bool {
    reachable_from(successors, successors[from].iter().copied(), None)[to]
}

use crate::symbol::SymbolPath;
use crate::symbol_table::qualified_name;

fn path_set(paths: &[SymbolPath], indices: impl IntoIterator<Item = usize>) -> HashSet<SymbolPath> {
    indices
        .into_iter()
        .map(|index| paths[index].clone())
        .collect()
}

fn name(ctx: &IrContext, op: OpRef) -> SymbolPath {
    qualified_name(ctx, op).expect("symbol")
}

fn names<'a>(ctx: &IrContext, ops: impl IntoIterator<Item = &'a OpRef>) -> HashSet<SymbolPath> {
    ops.into_iter().map(|&op| name(ctx, op)).collect()
}

fn named_edges(ctx: &IrContext, edges: &Edges) -> HashMap<SymbolPath, HashSet<SymbolPath>> {
    edges
        .iter()
        .map(|(&caller, callees)| (name(ctx, caller), names(ctx, callees)))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn call_graph_records_every_reference(built in built(module_spec(10))) {
        let Built { spec, ctx, module } = &built;
        let module = *module;
        let graph = build_call_graph(ctx, module);
        let paths = spec.paths();
        let functions = spec.functions.len();

        prop_assert_eq!(names(ctx, &graph.functions), path_set(&paths, 0..functions));
        // A reference that names a defined function, by its path; any other
        // reference resolves to nothing and adds no edge.
        let resolved = |nested: bool, target: usize| {
            spec.resolved_target(nested, target)
                .filter(|&index| index < functions)
                .map(|index| paths[index].clone())
        };
        for calls_only in [false, true] {
            let expected: HashMap<SymbolPath, HashSet<SymbolPath>> = spec
                .functions
                .iter()
                .enumerate()
                .map(|(index, function)| {
                    let targets: HashSet<SymbolPath> = function
                        .refs
                        .iter()
                        .filter(|reference| !calls_only || reference.kind == RefKind::Call)
                        .filter_map(|reference| resolved(function.nested, reference.target))
                        .collect();
                    (paths[index].clone(), targets)
                })
                .filter(|(_, targets)| !targets.is_empty())
                .collect();
            let actual = named_edges(ctx, if calls_only { &graph.calls } else { &graph.edges });
            prop_assert_eq!(actual, expected, "calls_only = {}", calls_only);
        }

        let mut call_sites: HashMap<SymbolPath, usize> = HashMap::default();
        let mut address_taken = HashSet::default();
        for (function, reference) in spec
            .functions
            .iter()
            .flat_map(|function| function.refs.iter().map(move |reference| (function, reference)))
        {
            let Some(target) = resolved(function.nested, reference.target) else {
                continue;
            };
            if reference.kind == RefKind::Call {
                *call_sites.entry(target).or_default() += 1;
            } else {
                address_taken.insert(target);
            }
        }
        let module_references: HashSet<SymbolPath> = spec
            .module_refs
            .iter()
            .filter_map(|reference| resolved(reference.nested, reference.target))
            .collect();
        address_taken.extend(module_references.iter().cloned());
        let call_site_count: HashMap<SymbolPath, usize> = graph
            .call_site_count
            .iter()
            .map(|(&callee, &count)| (name(ctx, callee), count))
            .collect();
        prop_assert_eq!(call_site_count, call_sites);
        prop_assert_eq!(names(ctx, &graph.module_references), module_references);
        prop_assert_eq!(names(ctx, &graph.address_taken), address_taken);
    }

    #[test]
    fn sccs_are_the_classes_of_mutual_reachability(built in built(module_spec(10))) {
        let Built { spec, ctx, module } = &built;
        let module = *module;
        let graph = build_call_graph(ctx, module);
        let paths = spec.paths();
        let functions = spec.functions.len();
        let successors = spec.successors(false);

        let ids: HashMap<SymbolPath, u32> = tarjan_scc(&graph)
            .into_iter()
            .map(|(op, id)| (name(ctx, op), id))
            .collect();
        prop_assert_eq!(
            ids.keys().cloned().collect::<HashSet<_>>(),
            path_set(&paths, 0..functions)
        );
        for a in 0..functions {
            let from_a = reachable_from(&successors, [a], None);
            for b in 0..functions {
                let mutual = from_a[b] && reachable_from(&successors, [b], None)[a];
                prop_assert_eq!(
                    ids[&paths[a]] == ids[&paths[b]],
                    mutual,
                    "{} and {}",
                    paths[a],
                    paths[b]
                );
            }
        }

        for (calls_only, actual) in [
            (false, recursive_functions(&graph)),
            (true, directly_recursive_functions(&graph)),
        ] {
            let successors = spec.successors(calls_only);
            let expected = path_set(
                &paths,
                (0..functions).filter(|&index| reaches(&successors, index, index)),
            );
            prop_assert_eq!(names(ctx, &actual), expected, "calls_only = {}", calls_only);
        }
    }
}

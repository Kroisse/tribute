pub mod collect;
pub mod mangle;
mod rewrite;
pub mod specialize;

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{Decl, FuncDefId, Module, Type, TypeScheme, TypedRef};

/// Result of monomorphization: updated module + function types.
pub struct MonomorphizeResult<'db> {
    pub module: Module<TypedRef<'db>>,
    pub function_types: Vec<(Symbol, TypeScheme<'db>)>,
}

/// Run monomorphization on a typed module.
///
/// This is the main entry point that:
/// 1. Collects all generic function instantiations
/// 2. Generates specialized copies with concrete types
/// 3. Rewrites call sites to use the specialized versions
/// 4. Appends specialized functions to the module
pub fn monomorphize_functions<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
    function_types: HashMap<Symbol, TypeScheme<'db>>,
) -> MonomorphizeResult<'db> {
    let fn_types_vec: Vec<(Symbol, TypeScheme<'db>)> =
        function_types.iter().map(|(k, v)| (*k, *v)).collect();

    // Step 1: Collect instantiations
    let instantiations = collect::collect_instantiations(db, &module, &fn_types_vec);

    if instantiations.is_empty() {
        return MonomorphizeResult {
            module,
            function_types: fn_types_vec,
        };
    }

    // Step 2: Generate specialized functions
    let (specialized_decls, specialized_fn_types) =
        specialize::generate_specializations(db, &module, &instantiations, &fn_types_vec);

    // Build rewrite map: (original FuncDefId, type_args) → specialized FuncDefId
    let rewrite_map = build_rewrite_map(db, &instantiations, &fn_types_vec);

    // Step 3: Rewrite call sites in the module
    let rewritten_module = rewrite::rewrite_module(db, module, &fn_types_vec, &rewrite_map);

    // Step 4: Append specialized functions to module
    let mut decls = rewritten_module.decls;
    decls.extend(specialized_decls.into_iter().map(Decl::Function));

    let final_module = Module::new(rewritten_module.id, rewritten_module.name, decls);

    // Merge function types
    let mut all_fn_types = fn_types_vec;
    all_fn_types.extend(specialized_fn_types);

    MonomorphizeResult {
        module: final_module,
        function_types: all_fn_types,
    }
}

/// Build a map from (original FuncDefId, concrete callee type) → mangled Symbol
/// for use during call site rewriting.
fn build_rewrite_map<'db>(
    db: &'db dyn salsa::Database,
    instantiations: &HashMap<FuncDefId<'db>, std::collections::HashSet<Vec<Type<'db>>>>,
    function_types: &[(Symbol, TypeScheme<'db>)],
) -> HashMap<FuncDefId<'db>, Vec<(Vec<Type<'db>>, Symbol)>> {
    let scheme_map: HashMap<Symbol, TypeScheme<'db>> = function_types.iter().cloned().collect();
    let mut rewrite_map: HashMap<FuncDefId<'db>, Vec<(Vec<Type<'db>>, Symbol)>> = HashMap::new();

    for (func_id, type_arg_sets) in instantiations {
        let qualified = func_id.qualified(db);
        let Some(_scheme) = scheme_map.get(&qualified) else {
            continue;
        };

        let mut entries: Vec<(Vec<Type<'db>>, Symbol)> = type_arg_sets
            .iter()
            .map(|type_args| {
                let mangled = mangle::mangle_name(db, qualified, type_args);
                (type_args.clone(), mangled)
            })
            .collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1));
        rewrite_map.insert(*func_id, entries);
    }

    rewrite_map
}

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
/// 1. Collects all generic function/type instantiations
/// 2. Generates specialized copies with concrete types
/// 3. Rewrites call sites and type references to use specialized versions
/// 4. Appends specialized functions/types to the module
pub fn monomorphize_functions<'db>(
    db: &'db dyn salsa::Database,
    module: Module<TypedRef<'db>>,
    function_types: HashMap<Symbol, TypeScheme<'db>>,
) -> MonomorphizeResult<'db> {
    let fn_types_vec: Vec<(Symbol, TypeScheme<'db>)> =
        function_types.iter().map(|(k, v)| (*k, *v)).collect();

    // === Function monomorphization ===

    // Step 1: Collect function instantiations
    let func_instantiations = collect::collect_instantiations(db, &module, &fn_types_vec);

    let module = if !func_instantiations.is_empty() {
        // Step 2: Generate specialized functions
        let (specialized_decls, specialized_fn_types) =
            specialize::generate_specializations(db, &module, &func_instantiations, &fn_types_vec);

        // Build rewrite map
        let rewrite_map = build_rewrite_map(db, &func_instantiations, &fn_types_vec);

        // Step 3: Rewrite call sites in the module
        let rewritten_module = rewrite::rewrite_module(db, module, &fn_types_vec, &rewrite_map);

        // Step 4: Rewrite call sites inside specialized function bodies
        let specialized_decls: Vec<Decl<TypedRef<'db>>> =
            specialized_decls.into_iter().map(Decl::Function).collect();
        let rewritten_specialized =
            rewrite::rewrite_decls(db, specialized_decls, &fn_types_vec, &rewrite_map);

        // Step 5: Append specialized functions to module
        let mut decls = rewritten_module.decls;
        decls.extend(rewritten_specialized);

        // Update function types for downstream
        let mut all_fn_types = fn_types_vec;
        all_fn_types.extend(specialized_fn_types);

        // Return intermediate result with updated fn_types
        let module = Module::new(rewritten_module.id, rewritten_module.name, decls);
        (module, all_fn_types)
    } else {
        (module, fn_types_vec)
    };

    let (module, fn_types_vec) = module;

    // === Type monomorphization (struct/enum) ===

    let type_instantiations = collect::collect_type_instantiations(db, &module);

    let module = if !type_instantiations.is_empty() {
        // Generate specialized struct/enum declarations
        let specialized_structs =
            specialize::generate_struct_specializations(db, &module, &type_instantiations);
        let specialized_enums =
            specialize::generate_enum_specializations(db, &module, &type_instantiations);

        // Build type rewrite map and rewrite Named types throughout the module
        let type_rewrite_map = rewrite::build_type_rewrite_map(db, &type_instantiations);
        let rewritten_module = rewrite::rewrite_types_in_module(db, module, &type_rewrite_map);

        // Append specialized types to module
        let mut decls = rewritten_module.decls;
        decls.extend(specialized_structs.into_iter().map(Decl::Struct));
        decls.extend(specialized_enums.into_iter().map(Decl::Enum));

        Module::new(rewritten_module.id, rewritten_module.name, decls)
    } else {
        module
    };

    MonomorphizeResult {
        module,
        function_types: fn_types_vec,
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
        entries.sort_by_key(|e| e.1);
        rewrite_map.insert(*func_id, entries);
    }

    rewrite_map
}

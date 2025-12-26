//! Global Dead Code Elimination (DCE) pass.
//!
//! This pass removes function definitions that are not reachable from entry points.
//! Entry points include:
//! - Functions referenced by `wasm.export_func`
//! - Functions named "main" or "_start"
//!
//! The pass builds a call graph by analyzing:
//! - `func.call` operations (callee attribute)
//! - `func.tail_call` operations (callee attribute)
//! - `func.constant` operations (func_ref attribute)
//!
//! This pass should run BEFORE operation-level DCE for best results.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::dialect::core;
use crate::{Attribute, Block, IdVec, Operation, QualifiedName, Region, Symbol};

/// Configuration for global dead code elimination.
#[derive(Debug, Clone)]
pub struct GlobalDceConfig {
    /// Additional entry point function names (besides main/_start).
    pub extra_entry_points: Vec<String>,
    /// Whether to recursively process nested modules.
    /// Default: true
    pub recursive: bool,
}

impl Default for GlobalDceConfig {
    fn default() -> Self {
        Self {
            extra_entry_points: Vec::new(),
            recursive: true,
        }
    }
}

/// Result of running global DCE.
pub struct GlobalDceResult<'db> {
    /// The transformed module with dead functions removed.
    pub module: core::Module<'db>,
    /// Number of functions removed.
    pub removed_count: usize,
    /// Names of removed functions (for debugging).
    pub removed_functions: Vec<QualifiedName>,
}

/// Eliminate unreachable functions from a module.
pub fn eliminate_dead_functions<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
) -> GlobalDceResult<'db> {
    eliminate_dead_functions_with_config(db, module, GlobalDceConfig::default())
}

/// Eliminate unreachable functions with custom configuration.
pub fn eliminate_dead_functions_with_config<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    config: GlobalDceConfig,
) -> GlobalDceResult<'db> {
    GlobalDcePass::new(db, config).run(module)
}

/// Internal global DCE pass implementation.
struct GlobalDcePass<'db> {
    db: &'db dyn salsa::Database,
    config: GlobalDceConfig,
    /// All function definitions found in the module.
    /// Key: function's qualified name, Value: the func.func operation
    functions: HashMap<QualifiedName, Operation<'db>>,
    /// Call graph: caller -> set of callees
    call_graph: HashMap<QualifiedName, HashSet<QualifiedName>>,
    /// Entry points (roots for reachability analysis)
    entry_points: HashSet<QualifiedName>,
    // Cached symbols
    sym_func: Symbol,
    sym_call: Symbol,
    sym_tail_call: Symbol,
    sym_constant: Symbol,
    sym_export_func: Symbol,
    sym_module: Symbol,
    sym_sym_name: Symbol,
    sym_callee: Symbol,
    sym_func_ref: Symbol,
    sym_func_attr: Symbol,
    sym_name_attr: Symbol,
    sym_main: Symbol,
    sym_start: Symbol,
    dialect_func: Symbol,
    dialect_wasm: Symbol,
    dialect_core: Symbol,
}

impl<'db> GlobalDcePass<'db> {
    fn new(db: &'db dyn salsa::Database, config: GlobalDceConfig) -> Self {
        Self {
            db,
            config,
            functions: HashMap::new(),
            call_graph: HashMap::new(),
            entry_points: HashSet::new(),
            // Initialize cached symbols
            sym_func: Symbol::new("func"),
            sym_call: Symbol::new("call"),
            sym_tail_call: Symbol::new("tail_call"),
            sym_constant: Symbol::new("constant"),
            sym_export_func: Symbol::new("export_func"),
            sym_module: Symbol::new("module"),
            sym_sym_name: Symbol::new("sym_name"),
            sym_callee: Symbol::new("callee"),
            sym_func_ref: Symbol::new("func_ref"),
            sym_func_attr: Symbol::new("func"),
            sym_name_attr: Symbol::new("name"),
            sym_main: Symbol::new("main"),
            sym_start: Symbol::new("_start"),
            dialect_func: Symbol::new("func"),
            dialect_wasm: Symbol::new("wasm"),
            dialect_core: Symbol::new("core"),
        }
    }

    fn run(mut self, module: core::Module<'db>) -> GlobalDceResult<'db> {
        // Phase 1: Collect all function definitions and build call graph
        self.analyze_region(&module.body(self.db), &[]);

        // Phase 2: Compute reachable functions from entry points
        let reachable = self.compute_reachable();

        // Phase 3: Remove unreachable functions
        let (new_module, removed) = self.remove_dead_functions(module, &reachable);

        GlobalDceResult {
            module: new_module,
            removed_count: removed.len(),
            removed_functions: removed,
        }
    }

    /// Analyze a region to collect functions, call graph edges, and entry points.
    /// `module_path` is the path of nested modules we're currently inside.
    fn analyze_region(&mut self, region: &Region<'db>, module_path: &[Symbol]) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                let dialect = op.dialect(self.db);
                let name = op.name(self.db);

                // Handle nested core.module
                if dialect == self.dialect_core && name == self.sym_module {
                    if self.config.recursive {
                        let new_path = self.extend_module_path(op, module_path);
                        for nested_region in op.regions(self.db).iter() {
                            self.analyze_region(nested_region, &new_path);
                        }
                    }
                    continue;
                }

                // Collect func.func definitions
                if dialect == self.dialect_func
                    && name == self.sym_func
                    && let Some(func_name) = self.extract_func_name(op, module_path)
                {
                    self.functions.insert(func_name.clone(), *op);

                    // Check if this is an entry point by name
                    let simple_name = func_name.name();
                    if simple_name == self.sym_main || simple_name == self.sym_start {
                        self.entry_points.insert(func_name.clone());
                    }

                    // Check extra entry points from config
                    for extra in &self.config.extra_entry_points {
                        simple_name.with_str(|s| {
                            if s == extra {
                                self.entry_points.insert(func_name.clone());
                            }
                        });
                    }

                    // Analyze function body for calls
                    self.analyze_function_body(op, &func_name);
                }

                // Collect wasm.export_func as entry points
                if dialect == self.dialect_wasm
                    && name == self.sym_export_func
                    && let Some(func_ref) = self.extract_export_func_target(op)
                {
                    self.entry_points.insert(func_ref);
                }
            }
        }
    }

    /// Extract the qualified name of a func.func operation.
    fn extract_func_name(
        &self,
        op: &Operation<'db>,
        module_path: &[Symbol],
    ) -> Option<QualifiedName> {
        let sym_name = op.attributes(self.db).get(&self.sym_sym_name)?;
        if let Attribute::Symbol(name) = sym_name {
            Some(QualifiedName::new(module_path.to_vec(), *name))
        } else {
            None
        }
    }

    /// Extract a qualified name from an attribute by key.
    fn extract_qualified_name_attr(
        &self,
        op: &Operation<'db>,
        key: &Symbol,
    ) -> Option<QualifiedName> {
        let attr = op.attributes(self.db).get(key)?;
        match attr {
            Attribute::Symbol(s) => Some(QualifiedName::simple(*s)),
            Attribute::QualifiedName(qn) => Some(qn.clone()),
            _ => None,
        }
    }

    /// Extract the target function from wasm.export_func.
    fn extract_export_func_target(&self, op: &Operation<'db>) -> Option<QualifiedName> {
        self.extract_qualified_name_attr(op, &self.sym_func_attr)
    }

    /// Analyze a function body to find all callees.
    fn analyze_function_body(&mut self, func_op: &Operation<'db>, caller: &QualifiedName) {
        for region in func_op.regions(self.db).iter() {
            self.collect_calls_from_region(region, caller);
        }
    }

    /// Recursively collect call targets from a region.
    fn collect_calls_from_region(&mut self, region: &Region<'db>, caller: &QualifiedName) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                let dialect = op.dialect(self.db);
                let name = op.name(self.db);

                if dialect == self.dialect_func
                    && (name == self.sym_call || name == self.sym_tail_call)
                    && let Some(callee) = self.extract_callee(op)
                {
                    // func.call and func.tail_call have callee attribute
                    self.add_call_edge(caller.clone(), callee);
                } else if dialect == self.dialect_func
                    && name == self.sym_constant
                    && let Some(func_ref) = self.extract_func_ref(op)
                {
                    // func.constant has func_ref attribute
                    self.add_call_edge(caller.clone(), func_ref);
                }

                // Recurse into nested regions (e.g., scf.if, case.case)
                for nested_region in op.regions(self.db).iter() {
                    self.collect_calls_from_region(nested_region, caller);
                }
            }
        }
    }

    /// Extract callee from func.call or func.tail_call.
    fn extract_callee(&self, op: &Operation<'db>) -> Option<QualifiedName> {
        self.extract_qualified_name_attr(op, &self.sym_callee)
    }

    /// Extract func_ref from func.constant.
    fn extract_func_ref(&self, op: &Operation<'db>) -> Option<QualifiedName> {
        self.extract_qualified_name_attr(op, &self.sym_func_ref)
    }

    /// Extend the module path with the nested module's name.
    fn extend_module_path(&self, op: &Operation<'db>, current_path: &[Symbol]) -> Vec<Symbol> {
        let nested_name = op
            .attributes(self.db)
            .get(&self.sym_name_attr)
            .and_then(|attr| {
                if let Attribute::Symbol(s) = attr {
                    Some(*s)
                } else {
                    None
                }
            });

        if let Some(n) = nested_name {
            let mut p = current_path.to_vec();
            p.push(n);
            p
        } else {
            current_path.to_vec()
        }
    }

    /// Add a call edge to the call graph.
    fn add_call_edge(&mut self, caller: QualifiedName, callee: QualifiedName) {
        self.call_graph.entry(caller).or_default().insert(callee);
    }

    /// Compute the set of reachable functions from entry points using BFS.
    fn compute_reachable(&self) -> HashSet<QualifiedName> {
        let mut reachable = HashSet::new();
        let mut worklist: VecDeque<QualifiedName> = self.entry_points.iter().cloned().collect();

        while let Some(func) = worklist.pop_front() {
            if reachable.contains(&func) {
                continue;
            }
            reachable.insert(func.clone());

            // Add all callees to worklist
            if let Some(callees) = self.call_graph.get(&func) {
                for callee in callees {
                    if !reachable.contains(callee) {
                        worklist.push_back(callee.clone());
                    }
                }
            }
        }

        reachable
    }

    /// Remove unreachable functions from the module.
    fn remove_dead_functions(
        &self,
        module: core::Module<'db>,
        reachable: &HashSet<QualifiedName>,
    ) -> (core::Module<'db>, Vec<QualifiedName>) {
        let mut removed = Vec::new();

        let body = module.body(self.db);
        let (new_body, _) = self.filter_region(&body, reachable, &mut removed, &[]);

        let new_module = core::Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        );

        (new_module, removed)
    }

    /// Filter a region, removing unreachable func.func operations.
    fn filter_region(
        &self,
        region: &Region<'db>,
        reachable: &HashSet<QualifiedName>,
        removed: &mut Vec<QualifiedName>,
        module_path: &[Symbol],
    ) -> (Region<'db>, bool) {
        let mut changed = false;

        let new_blocks: IdVec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| {
                let mut new_ops: IdVec<Operation<'db>> = IdVec::new();

                for op in block.operations(self.db).iter() {
                    let dialect = op.dialect(self.db);
                    let name = op.name(self.db);

                    // Handle nested core.module
                    if dialect == self.dialect_core && name == self.sym_module {
                        if self.config.recursive {
                            let new_path = self.extend_module_path(op, module_path);

                            // Process nested module regions
                            let new_regions: IdVec<Region<'db>> = op
                                .regions(self.db)
                                .iter()
                                .map(|nested_region| {
                                    let (new_region, region_changed) = self.filter_region(
                                        nested_region,
                                        reachable,
                                        removed,
                                        &new_path,
                                    );
                                    if region_changed {
                                        changed = true;
                                    }
                                    new_region
                                })
                                .collect();

                            let new_op = op.modify(self.db).regions(new_regions).build();
                            new_ops.push(new_op);
                        } else {
                            new_ops.push(*op);
                        }
                        continue;
                    }

                    // Check if this is a func.func that should be removed
                    if dialect == self.dialect_func
                        && name == self.sym_func
                        && let Some(func_name) = self.extract_func_name(op, module_path)
                        && !reachable.contains(&func_name)
                    {
                        // Remove this function
                        removed.push(func_name);
                        changed = true;
                        continue;
                    }

                    new_ops.push(*op);
                }

                Block::new(
                    self.db,
                    block.id(self.db),
                    block.location(self.db),
                    block.args(self.db).clone(),
                    new_ops,
                )
            })
            .collect();

        let new_region = Region::new(self.db, region.location(self.db), new_blocks);
        (new_region, changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{arith, core, func, wasm};
    use crate::{DialectType, Location, PathId, Span, idvec};
    use salsa_test_macros::salsa_test;

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    // Helper tracked functions to build test modules (required by Salsa)

    #[salsa::tracked]
    fn build_main_and_unused<'db>(db: &'db dyn salsa::Database) -> core::Module<'db> {
        let loc = test_location(db);
        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func::Func::build(
                db,
                loc,
                "main",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
            top.op(func::Func::build(
                db,
                loc,
                "unused",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
        })
    }

    #[salsa::tracked]
    fn build_main_calls_helper<'db>(db: &'db dyn salsa::Database) -> core::Module<'db> {
        let loc = test_location(db);
        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func::Func::build(
                db,
                loc,
                "helper",
                idvec![],
                core::I32::new(db).as_type(),
                |entry| {
                    let c = entry.op(arith::Const::i32(db, loc, 42));
                    entry.op(func::Return::value(db, loc, c.result(db)));
                },
            ));
            top.op(func::Func::build(
                db,
                loc,
                "main",
                idvec![],
                core::I32::new(db).as_type(),
                |entry| {
                    let callee = QualifiedName::simple(Symbol::new("helper"));
                    let call = entry.op(func::call(
                        db,
                        loc,
                        std::iter::empty(),
                        core::I32::new(db).as_type(),
                        callee,
                    ));
                    entry.op(func::Return::value(db, loc, call.result(db)));
                },
            ));
        })
    }

    #[salsa::tracked]
    fn build_transitive_calls<'db>(db: &'db dyn salsa::Database) -> core::Module<'db> {
        let loc = test_location(db);
        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func::Func::build(
                db,
                loc,
                "leaf",
                idvec![],
                core::I32::new(db).as_type(),
                |entry| {
                    let c = entry.op(arith::Const::i32(db, loc, 1));
                    entry.op(func::Return::value(db, loc, c.result(db)));
                },
            ));
            top.op(func::Func::build(
                db,
                loc,
                "middle",
                idvec![],
                core::I32::new(db).as_type(),
                |entry| {
                    let callee = QualifiedName::simple(Symbol::new("leaf"));
                    let call = entry.op(func::call(
                        db,
                        loc,
                        std::iter::empty(),
                        core::I32::new(db).as_type(),
                        callee,
                    ));
                    entry.op(func::Return::value(db, loc, call.result(db)));
                },
            ));
            top.op(func::Func::build(
                db,
                loc,
                "main",
                idvec![],
                core::I32::new(db).as_type(),
                |entry| {
                    let callee = QualifiedName::simple(Symbol::new("middle"));
                    let call = entry.op(func::call(
                        db,
                        loc,
                        std::iter::empty(),
                        core::I32::new(db).as_type(),
                        callee,
                    ));
                    entry.op(func::Return::value(db, loc, call.result(db)));
                },
            ));
            top.op(func::Func::build(
                db,
                loc,
                "unreachable",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
        })
    }

    #[salsa::tracked]
    fn build_func_constant_ref<'db>(db: &'db dyn salsa::Database) -> core::Module<'db> {
        let loc = test_location(db);
        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func::Func::build(
                db,
                loc,
                "callback",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
            top.op(func::Func::build(
                db,
                loc,
                "main",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    let func_ref = QualifiedName::simple(Symbol::new("callback"));
                    let func_ty = core::Func::new(db, idvec![], core::Nil::new(db).as_type());
                    let _const_op = entry.op(func::constant(db, loc, func_ty.as_type(), func_ref));
                    entry.op(func::Return::empty(db, loc));
                },
            ));
        })
    }

    #[salsa::tracked]
    fn build_start_entry<'db>(db: &'db dyn salsa::Database) -> core::Module<'db> {
        let loc = test_location(db);
        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func::Func::build(
                db,
                loc,
                "_start",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
        })
    }

    #[salsa::tracked]
    fn build_custom_entry<'db>(db: &'db dyn salsa::Database) -> core::Module<'db> {
        let loc = test_location(db);
        core::Module::build(db, loc, "test".into(), |top| {
            top.op(func::Func::build(
                db,
                loc,
                "custom_init",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
        })
    }

    // Tracked wrapper functions that call eliminate_dead_functions
    // (required because DCE creates new tracked structs)

    #[salsa::tracked]
    fn run_dce_main_and_unused(db: &dyn salsa::Database) -> usize {
        let module = build_main_and_unused(db);
        let result = eliminate_dead_functions(db, module);
        result.removed_count
    }

    #[salsa::tracked]
    fn run_dce_main_calls_helper(db: &dyn salsa::Database) -> usize {
        let module = build_main_calls_helper(db);
        let result = eliminate_dead_functions(db, module);
        result.removed_count
    }

    #[salsa::tracked]
    fn run_dce_transitive_calls(db: &dyn salsa::Database) -> usize {
        let module = build_transitive_calls(db);
        let result = eliminate_dead_functions(db, module);
        result.removed_count
    }

    #[salsa::tracked]
    fn run_dce_func_constant_ref(db: &dyn salsa::Database) -> usize {
        let module = build_func_constant_ref(db);
        let result = eliminate_dead_functions(db, module);
        result.removed_count
    }

    #[salsa::tracked]
    fn run_dce_start_entry(db: &dyn salsa::Database) -> usize {
        let module = build_start_entry(db);
        let result = eliminate_dead_functions(db, module);
        result.removed_count
    }

    #[salsa::tracked]
    fn run_dce_custom_entry(db: &dyn salsa::Database) -> usize {
        let module = build_custom_entry(db);
        let config = GlobalDceConfig {
            extra_entry_points: vec!["custom_init".to_string()],
            recursive: true,
        };
        let result = eliminate_dead_functions_with_config(db, module, config);
        result.removed_count
    }

    #[salsa_test]
    fn removes_unreachable_function(db: &salsa::DatabaseImpl) {
        let removed_count = run_dce_main_and_unused(db);
        assert_eq!(removed_count, 1);
    }

    #[salsa_test]
    fn keeps_called_function(db: &salsa::DatabaseImpl) {
        let removed_count = run_dce_main_calls_helper(db);
        assert_eq!(removed_count, 0);
    }

    #[salsa_test]
    fn keeps_transitive_calls(db: &salsa::DatabaseImpl) {
        let removed_count = run_dce_transitive_calls(db);
        assert_eq!(removed_count, 1);
    }

    #[salsa_test]
    fn keeps_func_constant_reference(db: &salsa::DatabaseImpl) {
        let removed_count = run_dce_func_constant_ref(db);
        assert_eq!(removed_count, 0);
    }

    #[salsa_test]
    fn handles_start_entry_point(db: &salsa::DatabaseImpl) {
        let removed_count = run_dce_start_entry(db);
        assert_eq!(removed_count, 0);
    }

    #[salsa_test]
    fn extra_entry_points_config(db: &salsa::DatabaseImpl) {
        let removed_count = run_dce_custom_entry(db);
        assert_eq!(removed_count, 0);
    }

    // Additional tests for wasm.export_func and recursive config

    #[salsa::tracked]
    fn build_wasm_export<'db>(db: &'db dyn salsa::Database) -> core::Module<'db> {
        let loc = test_location(db);
        core::Module::build(db, loc, "test".into(), |top| {
            // exported_func is exported via wasm.export_func
            top.op(func::Func::build(
                db,
                loc,
                "exported_func",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
            // wasm.export_func marks exported_func as an entry point
            top.op(wasm::export_func(
                db,
                loc,
                Attribute::String("my_export".into()),
                Attribute::QualifiedName(QualifiedName::simple(Symbol::new("exported_func"))),
            ));
            // unused_func has no references
            top.op(func::Func::build(
                db,
                loc,
                "unused_func",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
        })
    }

    #[salsa::tracked]
    fn run_dce_wasm_export(db: &dyn salsa::Database) -> usize {
        let module = build_wasm_export(db);
        let result = eliminate_dead_functions(db, module);
        result.removed_count
    }

    #[salsa_test]
    fn keeps_wasm_exported_function(db: &salsa::DatabaseImpl) {
        // wasm.export_func should mark the function as an entry point
        let removed_count = run_dce_wasm_export(db);
        assert_eq!(removed_count, 1); // Only unused_func should be removed
    }

    #[salsa::tracked]
    fn build_nested_module<'db>(db: &'db dyn salsa::Database) -> core::Module<'db> {
        let loc = test_location(db);
        core::Module::build(db, loc, "test".into(), |top| {
            // Top-level main (entry point)
            top.op(func::Func::build(
                db,
                loc,
                "main",
                idvec![],
                core::Nil::new(db).as_type(),
                |entry| {
                    entry.op(func::Return::empty(db, loc));
                },
            ));
            // Nested module with its own main (entry point) and unused function
            top.op(core::Module::build(db, loc, "nested".into(), |nested| {
                nested.op(func::Func::build(
                    db,
                    loc,
                    "main",
                    idvec![],
                    core::Nil::new(db).as_type(),
                    |entry| {
                        entry.op(func::Return::empty(db, loc));
                    },
                ));
                nested.op(func::Func::build(
                    db,
                    loc,
                    "unused_in_nested",
                    idvec![],
                    core::Nil::new(db).as_type(),
                    |entry| {
                        entry.op(func::Return::empty(db, loc));
                    },
                ));
            }));
        })
    }

    #[salsa::tracked]
    fn run_dce_nested_recursive(db: &dyn salsa::Database) -> usize {
        let module = build_nested_module(db);
        let config = GlobalDceConfig {
            extra_entry_points: vec![],
            recursive: true,
        };
        let result = eliminate_dead_functions_with_config(db, module, config);
        result.removed_count
    }

    #[salsa::tracked]
    fn run_dce_nested_non_recursive(db: &dyn salsa::Database) -> usize {
        let module = build_nested_module(db);
        let config = GlobalDceConfig {
            extra_entry_points: vec![],
            recursive: false,
        };
        let result = eliminate_dead_functions_with_config(db, module, config);
        result.removed_count
    }

    #[salsa_test]
    fn recursive_removes_nested_unused(db: &salsa::DatabaseImpl) {
        // With recursive=true, unused_in_nested should be removed
        // nested::main is an entry point, so it stays
        let removed_count = run_dce_nested_recursive(db);
        assert_eq!(removed_count, 1); // unused_in_nested removed
    }

    #[salsa_test]
    fn non_recursive_keeps_nested(db: &salsa::DatabaseImpl) {
        // With recursive=false, nested modules are not processed
        let removed_count = run_dce_nested_non_recursive(db);
        assert_eq!(removed_count, 0); // Nothing removed (nested module not analyzed)
    }
}

//! Function-level Dead Code Elimination (DCE) pass.
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

/// Configuration for function-level dead code elimination.
#[derive(Debug, Clone)]
pub struct FuncDceConfig {
    /// Additional entry point function names (besides main/_start).
    pub extra_entry_points: Vec<String>,
    /// Whether to recursively process nested modules.
    /// Default: true
    pub recursive: bool,
}

impl Default for FuncDceConfig {
    fn default() -> Self {
        Self {
            extra_entry_points: Vec::new(),
            recursive: true,
        }
    }
}

/// Result of running function-level DCE.
pub struct FuncDceResult<'db> {
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
) -> FuncDceResult<'db> {
    eliminate_dead_functions_with_config(db, module, FuncDceConfig::default())
}

/// Eliminate unreachable functions with custom configuration.
pub fn eliminate_dead_functions_with_config<'db>(
    db: &'db dyn salsa::Database,
    module: core::Module<'db>,
    config: FuncDceConfig,
) -> FuncDceResult<'db> {
    FuncDcePass::new(db, config).run(module)
}

/// Internal function DCE pass implementation.
struct FuncDcePass<'db> {
    db: &'db dyn salsa::Database,
    config: FuncDceConfig,
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

impl<'db> FuncDcePass<'db> {
    fn new(db: &'db dyn salsa::Database, config: FuncDceConfig) -> Self {
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

    fn run(mut self, module: core::Module<'db>) -> FuncDceResult<'db> {
        // Phase 1: Collect all function definitions and build call graph
        self.analyze_region(&module.body(self.db), &[]);

        // Phase 2: Compute reachable functions from entry points
        let reachable = self.compute_reachable();

        // Phase 3: Remove unreachable functions
        let (new_module, removed) = self.remove_dead_functions(module, &reachable);

        FuncDceResult {
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
                        // Get module name to build path
                        let nested_name =
                            op.attributes(self.db)
                                .get(&self.sym_name_attr)
                                .and_then(|attr| {
                                    if let Attribute::Symbol(s) = attr {
                                        Some(*s)
                                    } else {
                                        None
                                    }
                                });

                        let new_path: Vec<Symbol> = if let Some(n) = nested_name {
                            let mut p = module_path.to_vec();
                            p.push(n);
                            p
                        } else {
                            module_path.to_vec()
                        };

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

    /// Extract the target function from wasm.export_func.
    fn extract_export_func_target(&self, op: &Operation<'db>) -> Option<QualifiedName> {
        let func_attr = op.attributes(self.db).get(&self.sym_func_attr)?;
        match func_attr {
            Attribute::Symbol(s) => Some(QualifiedName::simple(*s)),
            Attribute::QualifiedName(qn) => Some(qn.clone()),
            _ => None,
        }
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
        let callee_attr = op.attributes(self.db).get(&self.sym_callee)?;
        match callee_attr {
            Attribute::Symbol(s) => Some(QualifiedName::simple(*s)),
            Attribute::QualifiedName(qn) => Some(qn.clone()),
            _ => None,
        }
    }

    /// Extract func_ref from func.constant.
    fn extract_func_ref(&self, op: &Operation<'db>) -> Option<QualifiedName> {
        let func_ref_attr = op.attributes(self.db).get(&self.sym_func_ref)?;
        match func_ref_attr {
            Attribute::Symbol(s) => Some(QualifiedName::simple(*s)),
            Attribute::QualifiedName(qn) => Some(qn.clone()),
            _ => None,
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

                            let new_path: Vec<Symbol> = if let Some(n) = nested_name {
                                let mut p = module_path.to_vec();
                                p.push(n);
                                p
                            } else {
                                module_path.to_vec()
                            };

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

// TODO: Add unit tests for function-level DCE
// Tests should cover:
// - Removing unreachable functions
// - Keeping functions called from entry points
// - Keeping exported functions
// - Handling nested modules
// - Transitive call graph analysis

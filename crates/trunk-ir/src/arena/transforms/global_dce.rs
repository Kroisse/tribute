//! Global Dead Code Elimination (DCE) pass for arena IR.
//!
//! Removes function definitions that are not reachable from reachability roots.
//! Reachability roots include:
//! - Functions named "main" or "_start"
//! - Functions referenced by `wasm.export_func`
//! - Functions with `abi` attribute (externally callable)
//! - Custom entry points from configuration
//!
//! Builds a call graph by analyzing `func.call`, `func.tail_call`, and
//! `func.constant` operations, then removes unreachable functions via BFS.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::ControlFlow;

use crate::arena::context::IrContext;
use crate::arena::refs::OpRef;
use crate::arena::rewrite::ArenaModule;
use crate::arena::types::Attribute;
use crate::arena::walk::{WalkAction, walk_region};
use crate::ir::Symbol;

/// Configuration for global dead code elimination.
#[derive(Debug, Clone)]
pub struct GlobalDceConfig {
    /// Additional entry point function names (besides main/_start).
    pub extra_entry_points: Vec<String>,
    /// Whether to recursively process nested modules. Default: true.
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
pub struct GlobalDceResult {
    /// Number of functions removed.
    pub removed_count: usize,
    /// Names of removed functions (for debugging).
    pub removed_functions: Vec<Symbol>,
}

/// Eliminate unreachable functions from a module using default configuration.
pub fn eliminate_dead_functions(ctx: &mut IrContext, module: ArenaModule) -> GlobalDceResult {
    eliminate_dead_functions_with_config(ctx, module, GlobalDceConfig::default())
}

/// Eliminate unreachable functions with custom configuration.
pub fn eliminate_dead_functions_with_config(
    ctx: &mut IrContext,
    module: ArenaModule,
    config: GlobalDceConfig,
) -> GlobalDceResult {
    let mut pass = GlobalDcePass::new(config);
    pass.run(ctx, module)
}

/// Cached symbol constants used during analysis.
struct Syms {
    func: Symbol,
    call: Symbol,
    tail_call: Symbol,
    constant: Symbol,
    export_func: Symbol,
    module: Symbol,
    sym_name: Symbol,
    callee: Symbol,
    func_ref: Symbol,
    func_attr: Symbol,
    name_attr: Symbol,
    abi: Symbol,
    main: Symbol,
    start: Symbol,
    dialect_func: Symbol,
    dialect_wasm: Symbol,
    dialect_core: Symbol,
}

impl Syms {
    fn new() -> Self {
        Self {
            func: Symbol::new("func"),
            call: Symbol::new("call"),
            tail_call: Symbol::new("tail_call"),
            constant: Symbol::new("constant"),
            export_func: Symbol::new("export_func"),
            module: Symbol::new("module"),
            sym_name: Symbol::new("sym_name"),
            callee: Symbol::new("callee"),
            func_ref: Symbol::new("func_ref"),
            func_attr: Symbol::new("func"),
            name_attr: Symbol::new("name"),
            abi: Symbol::new("abi"),
            main: Symbol::new("main"),
            start: Symbol::new("_start"),
            dialect_func: Symbol::new("func"),
            dialect_wasm: Symbol::new("wasm"),
            dialect_core: Symbol::new("core"),
        }
    }
}

struct GlobalDcePass {
    config: GlobalDceConfig,
    syms: Syms,
    /// Function definitions: name → OpRef
    functions: HashMap<Symbol, OpRef>,
    /// Call graph: caller → set of callees
    call_graph: HashMap<Symbol, HashSet<Symbol>>,
    /// Roots for reachability analysis (main, exports, abi functions, etc.)
    reachability_roots: HashSet<Symbol>,
}

impl GlobalDcePass {
    fn new(config: GlobalDceConfig) -> Self {
        Self {
            config,
            syms: Syms::new(),
            functions: HashMap::new(),
            call_graph: HashMap::new(),
            reachability_roots: HashSet::new(),
        }
    }

    fn run(&mut self, ctx: &mut IrContext, module: ArenaModule) -> GlobalDceResult {
        // Phase 1: Analyze — collect functions, call graph, reachability roots
        if let Some(body) = module.body(ctx) {
            self.analyze_module_region(ctx, body, &[]);
        }

        // Phase 2: Compute reachable functions from roots
        let reachable = self.compute_reachable();

        // Phase 3: Remove unreachable functions
        self.remove_dead_functions(ctx, module, &reachable)
    }

    /// Analyze a module's body region to collect functions, call edges, and reachability roots.
    fn analyze_module_region(
        &mut self,
        ctx: &IrContext,
        region: crate::arena::refs::RegionRef,
        module_path: &[Symbol],
    ) {
        let blocks = ctx.region(region).blocks.to_vec();
        for block in blocks {
            let ops = ctx.block(block).ops.to_vec();
            for op in ops {
                let dialect = ctx.op(op).dialect;
                let name = ctx.op(op).name;

                // Handle nested core.module
                if dialect == self.syms.dialect_core && name == self.syms.module {
                    if self.config.recursive {
                        let new_path = self.extend_module_path(ctx, op, module_path);
                        for &region in &ctx.op(op).regions {
                            self.analyze_module_region(ctx, region, &new_path);
                        }
                    }
                    continue;
                }

                // Collect func.func definitions
                if dialect == self.syms.dialect_func
                    && name == self.syms.func
                    && let Some(func_name) = self.extract_func_name(ctx, op, module_path)
                {
                    self.functions.insert(func_name, op);

                    // Check if entry point by base name (unqualified sym_name)
                    let base_name = self
                        .extract_symbol_attr(ctx, op, &self.syms.sym_name)
                        .unwrap_or(func_name);
                    if base_name == self.syms.main || base_name == self.syms.start {
                        self.reachability_roots.insert(func_name);
                    }

                    // Treat abi functions as entry points so their callees
                    // are also considered reachable.
                    if ctx.op(op).attributes.contains_key(&self.syms.abi) {
                        self.reachability_roots.insert(func_name);
                    }

                    // Check extra entry points (match against both qualified and base name)
                    for extra in &self.config.extra_entry_points {
                        let mut matched = false;
                        func_name.with_str(|s| {
                            if s == extra {
                                matched = true;
                            }
                        });
                        if !matched {
                            base_name.with_str(|s| {
                                if s == extra {
                                    matched = true;
                                }
                            });
                        }
                        if matched {
                            self.reachability_roots.insert(func_name);
                        }
                    }

                    // Analyze function body for call edges
                    self.analyze_function_body(ctx, op, func_name);
                }

                // Collect wasm.export_func as entry points
                if dialect == self.syms.dialect_wasm
                    && name == self.syms.export_func
                    && let Some(func_ref) = self.extract_symbol_attr(ctx, op, &self.syms.func_attr)
                {
                    self.reachability_roots.insert(func_ref);
                }
            }
        }
    }

    /// Analyze a function body to find all callees.
    fn analyze_function_body(&mut self, ctx: &IrContext, func_op: OpRef, caller: Symbol) {
        let regions = ctx.op(func_op).regions.to_vec();
        for region in regions {
            self.collect_calls_from_region(ctx, region, caller);
        }
    }

    /// Recursively collect call targets from a region.
    fn collect_calls_from_region(
        &mut self,
        ctx: &IrContext,
        region: crate::arena::refs::RegionRef,
        caller: Symbol,
    ) {
        let _ = walk_region::<()>(ctx, region, &mut |op| {
            let dialect = ctx.op(op).dialect;
            let name = ctx.op(op).name;

            if dialect == self.syms.dialect_func {
                if (name == self.syms.call || name == self.syms.tail_call)
                    && let Some(callee) = self.extract_symbol_attr(ctx, op, &self.syms.callee)
                {
                    self.call_graph.entry(caller).or_default().insert(callee);
                } else if name == self.syms.constant
                    && let Some(func_ref) = self.extract_symbol_attr(ctx, op, &self.syms.func_ref)
                {
                    self.call_graph.entry(caller).or_default().insert(func_ref);
                }
            }

            ControlFlow::Continue(WalkAction::Advance)
        });
    }

    /// Extract a Symbol attribute from an op.
    fn extract_symbol_attr(&self, ctx: &IrContext, op: OpRef, key: &Symbol) -> Option<Symbol> {
        match ctx.op(op).attributes.get(key)? {
            Attribute::Symbol(s) => Some(*s),
            _ => None,
        }
    }

    /// Extract the qualified name of a func.func operation.
    fn extract_func_name(
        &self,
        ctx: &IrContext,
        op: OpRef,
        module_path: &[Symbol],
    ) -> Option<Symbol> {
        let sym_name = self.extract_symbol_attr(ctx, op, &self.syms.sym_name)?;
        if module_path.is_empty() {
            Some(sym_name)
        } else {
            let mut path = String::new();
            for (i, seg) in module_path.iter().enumerate() {
                if i > 0 {
                    path.push_str("::");
                }
                seg.with_str(|s| path.push_str(s));
            }
            path.push_str("::");
            sym_name.with_str(|s| path.push_str(s));
            Some(Symbol::from_dynamic(&path))
        }
    }

    /// Extend the module path with a nested module's name.
    fn extend_module_path(
        &self,
        ctx: &IrContext,
        op: OpRef,
        current_path: &[Symbol],
    ) -> Vec<Symbol> {
        let nested_name = self.extract_symbol_attr(ctx, op, &self.syms.name_attr);
        if let Some(n) = nested_name {
            let mut p = current_path.to_vec();
            p.push(n);
            p
        } else {
            current_path.to_vec()
        }
    }

    /// Compute reachable functions from reachability roots via BFS.
    fn compute_reachable(&self) -> HashSet<Symbol> {
        let mut reachable = HashSet::new();
        let mut worklist: VecDeque<Symbol> = self.reachability_roots.iter().copied().collect();

        while let Some(func) = worklist.pop_front() {
            if !reachable.insert(func) {
                continue;
            }
            if let Some(callees) = self.call_graph.get(&func) {
                for &callee in callees {
                    if !reachable.contains(&callee) {
                        worklist.push_back(callee);
                    }
                }
            }
        }

        reachable
    }

    /// Remove unreachable functions from the module.
    fn remove_dead_functions(
        &self,
        ctx: &mut IrContext,
        module: ArenaModule,
        reachable: &HashSet<Symbol>,
    ) -> GlobalDceResult {
        let mut removed = Vec::new();

        if let Some(body) = module.body(ctx) {
            self.filter_region(ctx, body, reachable, &mut removed, &[]);
        }

        GlobalDceResult {
            removed_count: removed.len(),
            removed_functions: removed,
        }
    }

    /// Filter a region, removing unreachable func.func operations.
    fn filter_region(
        &self,
        ctx: &mut IrContext,
        region: crate::arena::refs::RegionRef,
        reachable: &HashSet<Symbol>,
        removed: &mut Vec<Symbol>,
        module_path: &[Symbol],
    ) {
        let blocks = ctx.region(region).blocks.to_vec();
        for block in blocks {
            let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();
            for op in ops {
                let dialect = ctx.op(op).dialect;
                let name = ctx.op(op).name;

                // Handle nested core.module
                if dialect == self.syms.dialect_core && name == self.syms.module {
                    if self.config.recursive {
                        let new_path = self.extend_module_path(ctx, op, module_path);
                        let regions: Vec<_> = ctx.op(op).regions.to_vec();
                        for region in regions {
                            self.filter_region(ctx, region, reachable, removed, &new_path);
                        }
                    }
                    continue;
                }

                // Check if this is a func.func that should be removed
                if dialect == self.syms.dialect_func
                    && name == self.syms.func
                    && let Some(func_name) = self.extract_func_name(ctx, op, module_path)
                    && !reachable.contains(&func_name)
                {
                    removed.push(func_name);
                    ctx.remove_op_from_block(block, op);
                    // Note: we don't call ctx.remove_op because the func
                    // has regions/results that may have complex ownership.
                    // Detaching from the block is sufficient for DCE.
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::dialect::func;
    use crate::arena::*;
    use crate::ir::Symbol;
    use crate::location::Span;
    use smallvec::smallvec;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn fn_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).build())
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn build_simple_func(ctx: &mut IrContext, loc: Location, name: &str) -> OpRef {
        let fn_ty = fn_type(ctx);
        let sym_name = Symbol::from_dynamic(name);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret = func::r#return(ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::func(ctx, loc, sym_name, fn_ty, body).op_ref()
    }

    fn build_func_with_call(ctx: &mut IrContext, loc: Location, name: &str, callee: &str) -> OpRef {
        let fn_ty = fn_type(ctx);
        let i32_ty = i32_type(ctx);
        let sym_name = Symbol::from_dynamic(name);
        let sym_callee = Symbol::from_dynamic(callee);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let call = func::call(ctx, loc, std::iter::empty(), i32_ty, sym_callee);
        let call_result = call.result(ctx);
        ctx.push_op(entry, call.op_ref());
        let ret = func::r#return(ctx, loc, [call_result]);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::func(ctx, loc, sym_name, fn_ty, body).op_ref()
    }

    fn build_module(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> ArenaModule {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in ops {
            ctx.push_op(block, op);
        }
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("test")))
                .region(region)
                .build(ctx);
        let module_op = ctx.create_op(module_data);
        ArenaModule::new(ctx, module_op).unwrap()
    }

    fn count_funcs(ctx: &IrContext, module: ArenaModule) -> usize {
        module
            .ops(ctx)
            .iter()
            .filter(|&&op| {
                ctx.op(op).dialect == Symbol::new("func") && ctx.op(op).name == Symbol::new("func")
            })
            .count()
    }

    #[test]
    fn removes_unreachable_function() {
        let (mut ctx, loc) = test_ctx();
        let main = build_simple_func(&mut ctx, loc, "main");
        let unused = build_simple_func(&mut ctx, loc, "unused");
        let module = build_module(&mut ctx, loc, vec![main, unused]);

        let result = eliminate_dead_functions(&mut ctx, module);

        assert_eq!(result.removed_count, 1);
        assert_eq!(count_funcs(&ctx, module), 1);
    }

    #[test]
    fn keeps_called_function() {
        let (mut ctx, loc) = test_ctx();
        let helper = build_simple_func(&mut ctx, loc, "helper");
        let main = build_func_with_call(&mut ctx, loc, "main", "helper");
        let module = build_module(&mut ctx, loc, vec![helper, main]);

        let result = eliminate_dead_functions(&mut ctx, module);

        assert_eq!(result.removed_count, 0);
        assert_eq!(count_funcs(&ctx, module), 2);
    }

    #[test]
    fn keeps_transitive_calls() {
        let (mut ctx, loc) = test_ctx();
        let leaf = build_simple_func(&mut ctx, loc, "leaf");
        let middle = build_func_with_call(&mut ctx, loc, "middle", "leaf");
        let main = build_func_with_call(&mut ctx, loc, "main", "middle");
        let unreachable = build_simple_func(&mut ctx, loc, "unreachable");
        let module = build_module(&mut ctx, loc, vec![leaf, middle, main, unreachable]);

        let result = eliminate_dead_functions(&mut ctx, module);

        assert_eq!(result.removed_count, 1);
        assert_eq!(count_funcs(&ctx, module), 3);
    }

    #[test]
    fn keeps_func_constant_reference() {
        let (mut ctx, loc) = test_ctx();
        let fn_ty = fn_type(&mut ctx);

        let callback = build_simple_func(&mut ctx, loc, "callback");

        // Build main that references callback via func.constant
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let const_op = func::constant(&mut ctx, loc, fn_ty, Symbol::new("callback"));
        ctx.push_op(entry, const_op.op_ref());
        let ret = func::r#return(&mut ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let main = func::func(&mut ctx, loc, Symbol::new("main"), fn_ty, body).op_ref();

        let module = build_module(&mut ctx, loc, vec![callback, main]);

        let result = eliminate_dead_functions(&mut ctx, module);

        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn handles_start_entry_point() {
        let (mut ctx, loc) = test_ctx();
        let start = build_simple_func(&mut ctx, loc, "_start");
        let module = build_module(&mut ctx, loc, vec![start]);

        let result = eliminate_dead_functions(&mut ctx, module);

        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn extra_entry_points_config() {
        let (mut ctx, loc) = test_ctx();
        let custom = build_simple_func(&mut ctx, loc, "custom_init");
        let module = build_module(&mut ctx, loc, vec![custom]);

        let config = GlobalDceConfig {
            extra_entry_points: vec!["custom_init".to_string()],
            recursive: true,
        };
        let result = eliminate_dead_functions_with_config(&mut ctx, module, config);

        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn keeps_wasm_exported_function() {
        let (mut ctx, loc) = test_ctx();
        let exported = build_simple_func(&mut ctx, loc, "exported_func");
        let unused = build_simple_func(&mut ctx, loc, "unused_func");

        // Create wasm.export_func op
        let export_data =
            OperationDataBuilder::new(loc, Symbol::new("wasm"), Symbol::new("export_func"))
                .attr("name", Attribute::String("my_export".to_owned()))
                .attr("func", Attribute::Symbol(Symbol::new("exported_func")))
                .build(&mut ctx);
        let export_op = ctx.create_op(export_data);

        let module = build_module(&mut ctx, loc, vec![exported, export_op, unused]);

        let result = eliminate_dead_functions(&mut ctx, module);

        assert_eq!(result.removed_count, 1); // Only unused_func removed
    }

    #[test]
    fn preserves_extern_declarations() {
        let (mut ctx, loc) = test_ctx();
        let fn_ty = fn_type(&mut ctx);
        let main = build_simple_func(&mut ctx, loc, "main");

        // Build an unreachable extern func with abi attribute
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let extern_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("extern_fn")))
            .attr("type", Attribute::Type(fn_ty))
            .attr("abi", Attribute::String("C".to_owned()))
            .region(body)
            .build(&mut ctx);
        let extern_op = ctx.create_op(extern_data);

        let module = build_module(&mut ctx, loc, vec![main, extern_op]);

        let result = eliminate_dead_functions(&mut ctx, module);

        assert_eq!(result.removed_count, 0);
        assert_eq!(count_funcs(&ctx, module), 2);
    }

    #[test]
    fn abi_function_callees_are_reachable() {
        let (mut ctx, loc) = test_ctx();
        let fn_ty = fn_type(&mut ctx);

        let main = build_simple_func(&mut ctx, loc, "main");

        // helper is only called by extern_fn
        let helper = build_simple_func(&mut ctx, loc, "helper");

        // extern_fn (abi) calls helper
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let i32_ty = i32_type(&mut ctx);
        let call = func::call(
            &mut ctx,
            loc,
            std::iter::empty(),
            i32_ty,
            Symbol::new("helper"),
        );
        ctx.push_op(entry, call.op_ref());
        let ret = func::r#return(&mut ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let extern_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("extern_fn")))
            .attr("type", Attribute::Type(fn_ty))
            .attr("abi", Attribute::String("C".to_owned()))
            .region(body)
            .build(&mut ctx);
        let extern_op = ctx.create_op(extern_data);

        let module = build_module(&mut ctx, loc, vec![main, helper, extern_op]);

        let result = eliminate_dead_functions(&mut ctx, module);

        // extern_fn is preserved (abi) and helper is reachable from extern_fn
        assert_eq!(result.removed_count, 0);
        assert_eq!(count_funcs(&ctx, module), 3);
    }

    #[test]
    fn nested_module_recursive() {
        let (mut ctx, loc) = test_ctx();

        let top_main = build_simple_func(&mut ctx, loc, "main");

        // Build nested module with its own main and an unused func
        let nested_main = build_simple_func(&mut ctx, loc, "main");
        let nested_unused = build_simple_func(&mut ctx, loc, "unused_in_nested");

        let nested_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(nested_block, nested_main);
        ctx.push_op(nested_block, nested_unused);
        let nested_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![nested_block],
            parent_op: None,
        });
        let nested_module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("nested")))
                .attr("name", Attribute::Symbol(Symbol::new("nested")))
                .region(nested_region)
                .build(&mut ctx);
        let nested_module_op = ctx.create_op(nested_module_data);

        let module = build_module(&mut ctx, loc, vec![top_main, nested_module_op]);

        let config = GlobalDceConfig {
            extra_entry_points: vec![],
            recursive: true,
        };
        let result = eliminate_dead_functions_with_config(&mut ctx, module, config);

        assert_eq!(result.removed_count, 1); // nested::unused_in_nested removed
    }

    #[test]
    fn non_recursive_keeps_nested() {
        let (mut ctx, loc) = test_ctx();

        let top_main = build_simple_func(&mut ctx, loc, "main");

        // Same nested module setup
        let nested_unused = build_simple_func(&mut ctx, loc, "unused_in_nested");
        let nested_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        ctx.push_op(nested_block, nested_unused);
        let nested_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![nested_block],
            parent_op: None,
        });
        let nested_module_data =
            OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
                .attr("sym_name", Attribute::Symbol(Symbol::new("nested")))
                .attr("name", Attribute::Symbol(Symbol::new("nested")))
                .region(nested_region)
                .build(&mut ctx);
        let nested_module_op = ctx.create_op(nested_module_data);

        let module = build_module(&mut ctx, loc, vec![top_main, nested_module_op]);

        let config = GlobalDceConfig {
            extra_entry_points: vec![],
            recursive: false,
        };
        let result = eliminate_dead_functions_with_config(&mut ctx, module, config);

        // With recursive=false, nested module is not analyzed
        assert_eq!(result.removed_count, 0);
    }
}

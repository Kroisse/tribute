//! Call graph analysis for TrunkIR.
//!
//! Builds a directed graph of function call relationships from `func.call`,
//! `func.tail_call`, and `func.constant` operations. Provides Tarjan's SCC
//! for detecting recursive functions (direct self-recursion and mutual
//! recursion), which is a prerequisite for inlining (`inline.rs`) and other
//! interprocedural transforms.
//!
//! Recurses into nested `core.module` operations, qualifying function names
//! with their module path (e.g. `nested::helper`).

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

use crate::context::IrContext;
use crate::refs::{OpRef, RegionRef};
use crate::rewrite::Module;
use crate::symbol::Symbol;
use crate::types::Attribute;
use crate::walk::{WalkAction, walk_region};

/// A function call graph over a module.
///
/// Edges include direct calls (`func.call`), tail calls (`func.tail_call`),
/// and reference-as-value (`func.constant`). Reference edges are conservative:
/// they represent "function X escapes as a value, so could be called from
/// anywhere later" and should be treated as a call edge for recursion
/// detection.
#[derive(Debug, Default)]
pub struct CallGraph {
    /// Caller → set of callees (includes direct calls and `func.constant` refs).
    pub edges: HashMap<Symbol, HashSet<Symbol>>,
    /// Function name (possibly qualified) → its defining `func.func` op.
    pub func_ops: HashMap<Symbol, OpRef>,
    /// Functions referenced by at least one `func.constant` anywhere in the module.
    pub has_constant_ref: HashSet<Symbol>,
    /// Callee → number of static direct-call sites across the whole module.
    /// `func.constant` references are *not* counted here (they are tracked in
    /// `has_constant_ref`).
    pub call_site_count: HashMap<Symbol, usize>,
}

/// Build a call graph for `module`, recursing into nested `core.module` ops.
pub fn build_call_graph(ctx: &IrContext, module: Module) -> CallGraph {
    let mut builder = Builder::new();
    if let Some(body) = module.body(ctx) {
        builder.analyze_region(ctx, body, &[]);
    }
    builder.into_call_graph()
}

/// Compute the SCC id for every function in the call graph using Tarjan's algorithm.
///
/// Only functions defined in `graph.func_ops` are assigned an id. External
/// callees (callees that appear in edges but have no corresponding `func.func`)
/// are skipped.
pub fn tarjan_scc(graph: &CallGraph) -> HashMap<Symbol, u32> {
    let mut state = TarjanState::default();
    for &v in graph.func_ops.keys() {
        if !state.index.contains_key(&v) {
            strongconnect(v, &mut state, graph);
        }
    }
    state.scc_id
}

/// Functions that participate in a recursive cycle (direct or mutual).
///
/// A function is "recursive" if it belongs to an SCC of size > 1 **or** its
/// singleton SCC contains a self-edge (direct self-recursion).
pub fn recursive_functions(graph: &CallGraph) -> HashSet<Symbol> {
    let scc_ids = tarjan_scc(graph);
    let mut by_scc: HashMap<u32, Vec<Symbol>> = HashMap::new();
    for (&v, &id) in &scc_ids {
        by_scc.entry(id).or_default().push(v);
    }
    let mut result = HashSet::new();
    for members in by_scc.into_values() {
        if members.len() > 1 {
            result.extend(members);
        } else {
            let v = members[0];
            if graph.edges.get(&v).is_some_and(|s| s.contains(&v)) {
                result.insert(v);
            }
        }
    }
    result
}

// =========================================================================
// Builder
// =========================================================================

struct Builder {
    syms: Syms,
    graph: CallGraph,
}

impl Builder {
    fn new() -> Self {
        Self {
            syms: Syms::new(),
            graph: CallGraph::default(),
        }
    }

    fn into_call_graph(self) -> CallGraph {
        self.graph
    }

    fn analyze_region(&mut self, ctx: &IrContext, region: RegionRef, module_path: &[Symbol]) {
        let blocks = ctx.region(region).blocks.to_vec();
        for block in blocks {
            let ops = ctx.block(block).ops.to_vec();
            for op in ops {
                let dialect = ctx.op(op).dialect;
                let name = ctx.op(op).name;

                // Recurse into nested core.module
                if dialect == self.syms.dialect_core && name == self.syms.module {
                    let new_path = self.extend_module_path(ctx, op, module_path);
                    for &region in &ctx.op(op).regions {
                        self.analyze_region(ctx, region, &new_path);
                    }
                    continue;
                }

                // Record func.func definitions and walk their bodies
                if dialect == self.syms.dialect_func
                    && name == self.syms.func
                    && let Some(func_name) = self.extract_func_name(ctx, op, module_path)
                {
                    self.graph.func_ops.insert(func_name, op);
                    // Iterate over a snapshot: collect_calls_from_region may mutate
                    // self.graph, but ctx remains immutably borrowed.
                    let regions: Vec<RegionRef> = ctx.op(op).regions.iter().copied().collect();
                    for region in regions {
                        self.collect_calls_from_region(ctx, region, func_name);
                    }
                }
            }
        }
    }

    fn collect_calls_from_region(&mut self, ctx: &IrContext, region: RegionRef, caller: Symbol) {
        let _ = walk_region::<()>(ctx, region, &mut |op| {
            let dialect = ctx.op(op).dialect;
            let name = ctx.op(op).name;

            if dialect == self.syms.dialect_func {
                if (name == self.syms.call || name == self.syms.tail_call)
                    && let Some(callee) = self.extract_symbol_attr(ctx, op, &self.syms.callee)
                {
                    self.graph.edges.entry(caller).or_default().insert(callee);
                    *self.graph.call_site_count.entry(callee).or_insert(0) += 1;
                } else if name == self.syms.constant
                    && let Some(func_ref) = self.extract_symbol_attr(ctx, op, &self.syms.func_ref)
                {
                    self.graph.edges.entry(caller).or_default().insert(func_ref);
                    self.graph.has_constant_ref.insert(func_ref);
                }
            }

            ControlFlow::Continue(WalkAction::Advance)
        });
    }

    fn extract_symbol_attr(&self, ctx: &IrContext, op: OpRef, key: &Symbol) -> Option<Symbol> {
        match ctx.op(op).attributes.get(key)? {
            Attribute::Symbol(s) => Some(*s),
            _ => None,
        }
    }

    fn extract_func_name(
        &self,
        ctx: &IrContext,
        op: OpRef,
        module_path: &[Symbol],
    ) -> Option<Symbol> {
        let sym_name = self.extract_symbol_attr(ctx, op, &self.syms.sym_name)?;
        if module_path.is_empty() {
            return Some(sym_name);
        }
        use itertools::Itertools;
        let qualified = module_path
            .iter()
            .chain(std::iter::once(&sym_name))
            .join("::");
        Some(Symbol::from_dynamic(&qualified))
    }

    fn extend_module_path(
        &self,
        ctx: &IrContext,
        op: OpRef,
        current_path: &[Symbol],
    ) -> Vec<Symbol> {
        let nested_name = self.extract_symbol_attr(ctx, op, &self.syms.sym_name);
        if let Some(n) = nested_name {
            let mut p = current_path.to_vec();
            p.push(n);
            p
        } else {
            current_path.to_vec()
        }
    }
}

// =========================================================================
// Tarjan's SCC
// =========================================================================

#[derive(Default)]
struct TarjanState {
    next_index: u32,
    stack: Vec<Symbol>,
    on_stack: HashSet<Symbol>,
    index: HashMap<Symbol, u32>,
    lowlink: HashMap<Symbol, u32>,
    scc_id: HashMap<Symbol, u32>,
    next_scc: u32,
}

fn strongconnect(v: Symbol, state: &mut TarjanState, graph: &CallGraph) {
    let v_index = state.next_index;
    state.next_index += 1;
    state.index.insert(v, v_index);
    state.lowlink.insert(v, v_index);
    state.stack.push(v);
    state.on_stack.insert(v);

    if let Some(successors) = graph.edges.get(&v) {
        let successors: Vec<Symbol> = successors.iter().copied().collect();
        for w in successors {
            // Skip external callees (not defined in this module).
            if !graph.func_ops.contains_key(&w) {
                continue;
            }
            if !state.index.contains_key(&w) {
                strongconnect(w, state, graph);
                let w_low = state.lowlink[&w];
                let v_low = state.lowlink[&v];
                state.lowlink.insert(v, v_low.min(w_low));
            } else if state.on_stack.contains(&w) {
                let w_idx = state.index[&w];
                let v_low = state.lowlink[&v];
                state.lowlink.insert(v, v_low.min(w_idx));
            }
        }
    }

    if state.lowlink[&v] == state.index[&v] {
        let scc_id = state.next_scc;
        state.next_scc += 1;
        loop {
            let w = state
                .stack
                .pop()
                .expect("stack non-empty while popping SCC");
            state.on_stack.remove(&w);
            state.scc_id.insert(w, scc_id);
            if w == v {
                break;
            }
        }
    }
}

// =========================================================================
// Cached symbols
// =========================================================================

struct Syms {
    func: Symbol,
    call: Symbol,
    tail_call: Symbol,
    constant: Symbol,
    module: Symbol,
    sym_name: Symbol,
    callee: Symbol,
    func_ref: Symbol,
    dialect_func: Symbol,
    dialect_core: Symbol,
}

impl Syms {
    fn new() -> Self {
        Self {
            func: Symbol::new("func"),
            call: Symbol::new("call"),
            tail_call: Symbol::new("tail_call"),
            constant: Symbol::new("constant"),
            module: Symbol::new("module"),
            sym_name: Symbol::new("sym_name"),
            callee: Symbol::new("callee"),
            func_ref: Symbol::new("func_ref"),
            dialect_func: Symbol::new("func"),
            dialect_core: Symbol::new("core"),
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::func;
    use crate::location::Span;
    use crate::*;
    use smallvec::smallvec;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn fn_type(ctx: &mut IrContext) -> TypeRef {
        let nil_ty = crate::dialect::core::nil(ctx).as_type_ref();
        crate::dialect::core::func(ctx, nil_ty, []).as_type_ref()
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn simple_func(ctx: &mut IrContext, loc: Location, name: &str) -> OpRef {
        let fn_ty = fn_type(ctx);
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
        func::func(ctx, loc, Symbol::from_dynamic(name), fn_ty, body).op_ref()
    }

    fn func_that_calls(ctx: &mut IrContext, loc: Location, name: &str, callees: &[&str]) -> OpRef {
        let fn_ty = fn_type(ctx);
        let i32_ty = i32_type(ctx);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for callee in callees {
            let call = func::call(
                ctx,
                loc,
                std::iter::empty(),
                i32_ty,
                Symbol::from_dynamic(callee),
            );
            ctx.push_op(entry, call.op_ref());
        }
        let ret = func::r#return(ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::func(ctx, loc, Symbol::from_dynamic(name), fn_ty, body).op_ref()
    }

    fn func_that_takes_constant_of(
        ctx: &mut IrContext,
        loc: Location,
        name: &str,
        target: &str,
    ) -> OpRef {
        let fn_ty = fn_type(ctx);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c = func::constant(ctx, loc, fn_ty, Symbol::from_dynamic(target));
        ctx.push_op(entry, c.op_ref());
        let ret = func::r#return(ctx, loc, std::iter::empty());
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::func(ctx, loc, Symbol::from_dynamic(name), fn_ty, body).op_ref()
    }

    fn build_module(ctx: &mut IrContext, loc: Location, ops: Vec<OpRef>) -> Module {
        let module_op = build_module_op(ctx, loc, "test", ops);
        Module::new(ctx, module_op).unwrap()
    }

    fn build_module_op(ctx: &mut IrContext, loc: Location, name: &str, ops: Vec<OpRef>) -> OpRef {
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
                .attr("sym_name", Attribute::Symbol(Symbol::from_dynamic(name)))
                .region(region)
                .build(ctx);
        ctx.create_op(module_data)
    }

    #[test]
    fn call_graph_records_direct_calls() {
        let (mut ctx, loc) = test_ctx();
        let leaf = simple_func(&mut ctx, loc, "leaf");
        let mid = func_that_calls(&mut ctx, loc, "mid", &["leaf"]);
        let main = func_that_calls(&mut ctx, loc, "main", &["mid"]);
        let module = build_module(&mut ctx, loc, vec![leaf, mid, main]);

        let g = build_call_graph(&ctx, module);
        assert!(
            g.edges
                .get(&Symbol::new("main"))
                .unwrap()
                .contains(&Symbol::new("mid"))
        );
        assert!(
            g.edges
                .get(&Symbol::new("mid"))
                .unwrap()
                .contains(&Symbol::new("leaf"))
        );
        assert!(g.func_ops.contains_key(&Symbol::new("leaf")));
        assert!(g.func_ops.contains_key(&Symbol::new("mid")));
        assert!(g.func_ops.contains_key(&Symbol::new("main")));
    }

    #[test]
    fn call_graph_records_func_constant_as_edge() {
        let (mut ctx, loc) = test_ctx();
        let target = simple_func(&mut ctx, loc, "target");
        let holder = func_that_takes_constant_of(&mut ctx, loc, "holder", "target");
        let module = build_module(&mut ctx, loc, vec![target, holder]);

        let g = build_call_graph(&ctx, module);
        assert!(
            g.edges
                .get(&Symbol::new("holder"))
                .unwrap()
                .contains(&Symbol::new("target"))
        );
        assert!(g.has_constant_ref.contains(&Symbol::new("target")));
        // func.constant should NOT count toward call_site_count
        assert_eq!(g.call_site_count.get(&Symbol::new("target")).copied(), None);
    }

    #[test]
    fn call_graph_counts_static_call_sites() {
        let (mut ctx, loc) = test_ctx();
        let leaf = simple_func(&mut ctx, loc, "leaf");
        // main calls leaf twice in its body
        let main = func_that_calls(&mut ctx, loc, "main", &["leaf", "leaf"]);
        let module = build_module(&mut ctx, loc, vec![leaf, main]);

        let g = build_call_graph(&ctx, module);
        assert_eq!(g.call_site_count[&Symbol::new("leaf")], 2);
    }

    #[test]
    fn tarjan_detects_self_recursion() {
        let (mut ctx, loc) = test_ctx();
        let f = func_that_calls(&mut ctx, loc, "f", &["f"]);
        let module = build_module(&mut ctx, loc, vec![f]);

        let g = build_call_graph(&ctx, module);
        let rec = recursive_functions(&g);
        assert!(rec.contains(&Symbol::new("f")));
    }

    #[test]
    fn tarjan_detects_mutual_recursion() {
        let (mut ctx, loc) = test_ctx();
        let a = func_that_calls(&mut ctx, loc, "a", &["b"]);
        let b = func_that_calls(&mut ctx, loc, "b", &["a"]);
        let module = build_module(&mut ctx, loc, vec![a, b]);

        let g = build_call_graph(&ctx, module);
        let rec = recursive_functions(&g);
        assert!(rec.contains(&Symbol::new("a")));
        assert!(rec.contains(&Symbol::new("b")));
    }

    #[test]
    fn tarjan_trivial_scc_not_flagged() {
        let (mut ctx, loc) = test_ctx();
        let leaf = simple_func(&mut ctx, loc, "leaf");
        let main = func_that_calls(&mut ctx, loc, "main", &["leaf"]);
        let module = build_module(&mut ctx, loc, vec![leaf, main]);

        let g = build_call_graph(&ctx, module);
        let rec = recursive_functions(&g);
        assert!(rec.is_empty());
    }

    #[test]
    fn nested_module_qualifies_func_names() {
        let (mut ctx, loc) = test_ctx();
        // Outer module contains a nested `core.module` named "inner"
        // with a single `func.func` named "foo". The call graph should
        // record the function as `inner::foo`, not just `foo`.
        let foo = simple_func(&mut ctx, loc, "foo");
        let inner = build_module_op(&mut ctx, loc, "inner", vec![foo]);
        let top = simple_func(&mut ctx, loc, "top");
        let module = build_module(&mut ctx, loc, vec![inner, top]);

        let g = build_call_graph(&ctx, module);
        assert!(
            g.func_ops.contains_key(&Symbol::from_dynamic("inner::foo")),
            "expected qualified symbol `inner::foo`, got: {:?}",
            g.func_ops.keys().collect::<Vec<_>>()
        );
        // Unqualified symbol should not leak through.
        assert!(!g.func_ops.contains_key(&Symbol::new("foo")));
        // Top-level function stays unqualified.
        assert!(g.func_ops.contains_key(&Symbol::new("top")));
    }

    #[test]
    fn tarjan_assigns_scc_ids_to_all_functions() {
        let (mut ctx, loc) = test_ctx();
        let a = simple_func(&mut ctx, loc, "a");
        let b = simple_func(&mut ctx, loc, "b");
        let module = build_module(&mut ctx, loc, vec![a, b]);

        let g = build_call_graph(&ctx, module);
        let ids = tarjan_scc(&g);
        assert_eq!(ids.len(), 2);
        // Two non-recursive functions → two distinct SCCs
        assert_ne!(ids[&Symbol::new("a")], ids[&Symbol::new("b")]);
    }
}

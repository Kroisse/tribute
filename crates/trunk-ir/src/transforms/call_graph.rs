//! Call graph analysis for TrunkIR.
//!
//! Builds a directed graph of function call relationships from `func.call`,
//! `func.tail_call`, and `func.constant` operations. Provides Tarjan's SCC
//! for detecting recursive functions (direct self-recursion and mutual
//! recursion), which is a prerequisite for inlining (`inline.rs`) and other
//! interprocedural transforms.
//!
//! Functions and references use root-qualified names (e.g. `nested::helper`),
//! as resolved by [`SymbolTable`].

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

use crate::analysis::{Analysis, AnalysisContext, AnalysisError, InfallibleAnalysis};
use crate::context::IrContext;
use crate::dialect::func;
use crate::ops::DialectOp;
use crate::refs::{OpRef, RegionRef};
use crate::rewrite::Module;
use crate::symbol::Symbol;
use crate::symbol_table::SymbolTable;
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
///
/// Functions are named by their root-qualified path. A duplicated qualified
/// name has no entry in `func_ops`, but calls in each of its bodies are still
/// recorded.
pub fn build_call_graph(ctx: &IrContext, module: Module) -> CallGraph {
    call_graph_over(ctx, &SymbolTable::collect(ctx, module))
}

fn call_graph_over(ctx: &IrContext, symbols: &SymbolTable) -> CallGraph {
    let mut graph = CallGraph::default();
    for (name, ops) in symbols.iter() {
        if let &[op] = ops
            && func::Func::matches(ctx, op)
        {
            graph.func_ops.insert(name, op);
        }
        for &op in ops.iter().filter(|&&op| func::Func::matches(ctx, op)) {
            for region in ctx.op_regions(op) {
                collect_calls(ctx, region, name, &mut graph);
            }
        }
    }
    graph
}

/// Record the calls and references in `region` as edges from `caller`.
/// Nested function definitions record their own edges.
fn collect_calls(ctx: &IrContext, region: RegionRef, caller: Symbol, graph: &mut CallGraph) {
    let _ = walk_region::<()>(ctx, region, &mut |op| {
        if func::Func::matches(ctx, op) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        let attributes = &ctx.op(op).attributes;
        if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
            if let Some(callee) = attributes.get_symbol_ref("callee") {
                record_call(graph, caller, callee);
            }
        } else if func::Constant::matches(ctx, op)
            && let Some(func_ref) = attributes.get_symbol_ref("func_ref")
        {
            graph.edges.entry(caller).or_default().insert(func_ref);
            graph.has_constant_ref.insert(func_ref);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
}

fn record_call(graph: &mut CallGraph, caller: Symbol, callee: Symbol) {
    graph.edges.entry(caller).or_default().insert(callee);
    *graph.call_site_count.entry(callee).or_insert(0) += 1;
}

/// `CallGraph` as an [`Analysis`] over the [`SymbolTable`] of `target`, which
/// must be a `core.module` op.
impl Analysis for CallGraph {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let symbols = ctx.get::<SymbolTable>(target)?;
        Ok(call_graph_over(ctx.ir(), &symbols))
    }
}

/// Depends only on the infallible [`SymbolTable`].
impl InfallibleAnalysis for CallGraph {}

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
        let path = ctx.intern_path("test.trb");
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    fn fn_type(ctx: &mut IrContext) -> TypeRef {
        let nil_ty = crate::dialect::core::nil(ctx).as_type_ref();
        crate::dialect::func::func_sig(ctx, [], [nil_ty]).as_type_ref()
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.intern_type(TypeDataBuilder::new("core", "i32").build())
    }

    fn simple_func(ctx: &mut IrContext, loc: Location, name: &str) -> OpRef {
        let fn_ty = fn_type(ctx);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let ret = func::Return::operands(std::iter::empty()).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::Func::operands()
            .sym_name(Symbol::from_dynamic(name))
            .r#type(fn_ty)
            .regions(body)
            .build(ctx, loc)
            .op_ref()
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
            let call = func::Call::operands(std::iter::empty())
                .callee(Symbol::from_dynamic(callee))
                .results([i32_ty])
                .build(ctx, loc);
            ctx.push_op(entry, call.op_ref());
        }
        let ret = func::Return::operands(std::iter::empty()).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::Func::operands()
            .sym_name(Symbol::from_dynamic(name))
            .r#type(fn_ty)
            .regions(body)
            .build(ctx, loc)
            .op_ref()
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
        let c = func::Constant::operands()
            .func_ref(Symbol::from_dynamic(target))
            .results(fn_ty)
            .build(ctx, loc);
        ctx.push_op(entry, c.op_ref());
        let ret = func::Return::operands(std::iter::empty()).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        func::Func::operands()
            .sym_name(Symbol::from_dynamic(name))
            .r#type(fn_ty)
            .regions(body)
            .build(ctx, loc)
            .op_ref()
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
                .attr(
                    "sym_name",
                    Attribute::String(ctx.intern_symbol_text(Symbol::from_dynamic(name))),
                )
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
    fn cached_call_graph_recomputes_after_call_target_changes() {
        use crate::analysis::AnalysisCache;

        let (mut ctx, loc) = test_ctx();
        let leaf = simple_func(&mut ctx, loc, "leaf");
        let other = simple_func(&mut ctx, loc, "other");
        let caller = func_that_calls(&mut ctx, loc, "caller", &["leaf"]);
        let module = build_module(&mut ctx, loc, vec![leaf, other, caller]);
        let mut analyses = AnalysisCache::new();
        let old = analyses.require::<CallGraph>(&ctx, module.op());
        assert_eq!(old.call_site_count.get(&Symbol::new("leaf")), Some(&1));

        let body = ctx.op_region(caller, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let call = ctx.block(block).ops[0];
        ctx.op_mut(call).attributes.insert(
            Symbol::new("callee"),
            Attribute::SymbolRef(Symbol::new("other")),
        );
        assert!(
            analyses
                .get_cached::<CallGraph>(&ctx, module.op())
                .is_none()
        );
        let fresh = analyses.require::<CallGraph>(&ctx, module.op());
        assert_eq!(fresh.call_site_count.get(&Symbol::new("leaf")), None);
        assert_eq!(fresh.call_site_count.get(&Symbol::new("other")), Some(&1));
        assert_eq!(old.call_site_count.get(&Symbol::new("leaf")), Some(&1));
        assert!(!std::sync::Arc::ptr_eq(&old, &fresh));
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
    fn call_graph_as_analysis_matches_direct_build() {
        use crate::analysis::AnalysisCache;

        let input = r#"core.module @test {
  func.func @leaf() -> core.i32 {
    %0 = arith.const {value = 0} : core.i32
    func.return %0
  }
  func.func @main() -> core.i32 {
    %0 = func.call {callee = @leaf} : core.i32
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);

        let direct = build_call_graph(&ctx, module);

        let mut am = AnalysisCache::new();
        let cached = am.require::<CallGraph>(&ctx, module.op());

        assert_eq!(direct.func_ops.len(), cached.func_ops.len());
        assert_eq!(direct.edges.len(), cached.edges.len());
        for (caller, callees) in &direct.edges {
            assert_eq!(Some(callees), cached.edges.get(caller));
        }
        assert!(cached.func_ops.contains_key(&Symbol::new("leaf")));
        assert!(cached.edges.contains_key(&Symbol::new("main")));

        // Second call should hit the cache.
        let cached2 = am.require::<CallGraph>(&ctx, module.op());
        assert!(std::sync::Arc::ptr_eq(&cached, &cached2));
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

    #[test]
    fn edges_use_root_qualified_callees_across_nested_modules() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @main() {
    func.call {callee = @"outer::same"}
    func.return
  }
  core.module @outer {
    func.func @same() {
      func.call {callee = @"outer::inner::same"}
      func.return
    }
    core.module @inner {
      func.func @same() {
        func.call {callee = @"outer::inner::same"}
        func.return
      }
    }
  }
}"#,
        );
        let g = build_call_graph(&ctx, module);
        let outer = Symbol::from_dynamic("outer::same");
        let inner = Symbol::from_dynamic("outer::inner::same");
        assert!(g.edges[&Symbol::new("main")].contains(&outer));
        assert!(g.edges[&outer].contains(&inner));
        assert_eq!(g.call_site_count[&inner], 2);
        // Only the innermost function calls itself.
        assert_eq!(recursive_functions(&g), HashSet::from([inner]));
    }

    #[test]
    fn duplicated_qualified_names_have_no_definition() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @twice() {
    func.call {callee = @leaf}
    func.return
  }
  func.func @twice() {
    func.return
  }
  func.func @leaf() {
    func.return
  }
}"#,
        );
        let g = build_call_graph(&ctx, module);
        assert!(!g.func_ops.contains_key(&Symbol::new("twice")));
        // Calls in each duplicate body are still recorded.
        assert_eq!(g.call_site_count[&Symbol::new("leaf")], 1);
    }
}

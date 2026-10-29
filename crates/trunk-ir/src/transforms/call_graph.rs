//! Call graph analysis for TrunkIR.
//!
//! Builds a directed graph of function call relationships from every symbol
//! reference in a function body. A [`CallLike`](crate::op_interface::CallLike) callee is a call; any
//! other reference takes its target's address. Provides Tarjan's SCC
//! for detecting recursive functions (direct self-recursion and mutual
//! recursion), which is a prerequisite for inlining (`inline.rs`) and other
//! interprocedural transforms.
//!
//! Functions are identified by their defining operations. A reference is
//! resolved by [`SymbolTable`] from the operation that holds it; one that
//! names no unique symbol adds no edge.
//!
//! A pass that rewrites the calls a graph describes can keep its own copy
//! current with [`CallGraph::inline_call`] instead of rebuilding it.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::ops::ControlFlow;

use crate::analysis::{Analysis, AnalysisContext, AnalysisError, InfallibleAnalysis};
use crate::context::IrContext;
use crate::dialect::func;
use crate::op_interface::DynCallLike;
use crate::ops::DialectOp;
use crate::refs::{OpRef, RegionRef};
use crate::rewrite::Module;
use crate::symbol_table::SymbolTable;
use crate::walk::{WalkAction, walk_op, walk_region};

type Edges = HashMap<OpRef, HashSet<OpRef>>;

/// A function call graph over a module.
///
/// Edges include direct calls and address references such as `func.constant`.
/// Reference edges are conservative: they represent "function X escapes as a
/// value, so could be called from anywhere later" and should be treated as a
/// call edge for recursion detection.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CallGraph {
    /// Caller → set of callees (includes direct calls and address references).
    pub edges: Edges,
    /// Caller → callees of its direct calls only; a subset of `edges`.
    pub calls: Edges,
    /// Every function that is a symbol of a module in the tree, in traversal
    /// order, including each definition of a duplicated name.
    pub functions: Vec<OpRef>,
    /// Symbols whose address is taken: referenced other than as the callee
    /// of a direct call in a function body. Includes `module_references`.
    pub address_taken: HashSet<OpRef>,
    /// Symbols referenced by an operation outside every function definition,
    /// such as an export.
    pub module_references: HashSet<OpRef>,
    /// Callee → number of static direct-call sites across the whole module.
    /// Address references are *not* counted here (they are tracked in
    /// `address_taken`).
    pub call_site_count: HashMap<OpRef, usize>,
    /// Caller → the references in its body, for callers with any.
    bodies: HashMap<OpRef, BodyReferences>,
    /// Symbol → number of function bodies that take its address.
    address_referrers: HashMap<OpRef, usize>,
}

/// The references in one function body, by target.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct BodyReferences {
    /// Callee → number of direct-call sites.
    calls: HashMap<OpRef, usize>,
    /// Symbols referenced other than as the callee of a direct call.
    addresses: HashSet<OpRef>,
}

impl BodyReferences {
    fn is_empty(&self) -> bool {
        self.calls.is_empty() && self.addresses.is_empty()
    }
}

impl CallGraph {
    /// Update the graph after the direct call from `caller` to `callee` was
    /// replaced by a copy of `callee`'s body, as the inliner does.
    ///
    /// The graph is then the one a rebuild gives, provided `caller` and
    /// `callee` are in the same symbol table, so the copied references
    /// resolve to the same symbols, and `callee` is not `caller`.
    ///
    /// # Panics
    ///
    /// If `caller` has no direct call to `callee`.
    pub fn inline_call(&mut self, caller: OpRef, callee: OpRef) {
        assert_ne!(caller, callee, "a function is not inlined into itself");
        let mut body = self.remove_body(caller);
        let sites = body
            .calls
            .get_mut(&callee)
            .expect("the inlined call is a direct call from the caller");
        *sites -= 1;
        if *sites == 0 {
            body.calls.remove(&callee);
        }
        if let Some(inlined) = self.bodies.get(&callee) {
            for (&target, &sites) in &inlined.calls {
                *body.calls.entry(target).or_insert(0) += sites;
            }
            body.addresses.extend(inlined.addresses.iter().copied());
        }
        self.add_body(caller, body);
    }

    fn add_body(&mut self, caller: OpRef, body: BodyReferences) {
        if body.is_empty() {
            return;
        }
        for (&callee, &sites) in &body.calls {
            self.edges.entry(caller).or_default().insert(callee);
            self.calls.entry(caller).or_default().insert(callee);
            *self.call_site_count.entry(callee).or_insert(0) += sites;
        }
        for &target in &body.addresses {
            self.edges.entry(caller).or_default().insert(target);
            self.address_taken.insert(target);
            *self.address_referrers.entry(target).or_insert(0) += 1;
        }
        self.bodies.insert(caller, body);
    }

    fn remove_body(&mut self, caller: OpRef) -> BodyReferences {
        let body = self.bodies.remove(&caller).unwrap_or_default();
        self.edges.remove(&caller);
        self.calls.remove(&caller);
        for (callee, &sites) in &body.calls {
            let count = self
                .call_site_count
                .get_mut(callee)
                .expect("every recorded call site is counted");
            *count -= sites;
            if *count == 0 {
                self.call_site_count.remove(callee);
            }
        }
        for target in &body.addresses {
            let referrers = self
                .address_referrers
                .get_mut(target)
                .expect("every recorded address reference is counted");
            *referrers -= 1;
            if *referrers == 0 {
                self.address_referrers.remove(target);
                if !self.module_references.contains(target) {
                    self.address_taken.remove(target);
                }
            }
        }
        body
    }
}

/// Build a call graph for `module`, recursing into nested `core.module` ops.
///
/// A reference to a duplicated name resolves to nothing, but calls in each
/// of its bodies are still recorded.
pub fn build_call_graph(ctx: &IrContext, module: Module) -> CallGraph {
    call_graph_over(
        ctx,
        module.op(),
        &SymbolTable::collect(ctx, module),
        func::Func::matches,
    )
}

/// Whether an operation is a function definition of the dialect a call graph
/// is built over.
pub type FunctionDefinition = fn(&IrContext, OpRef) -> bool;

/// Build the call graph of the functions `is_function` identifies, which
/// need not be `func.func`: a dialect with its own function definition
/// operation names its functions and references the same way.
pub fn call_graph_over(
    ctx: &IrContext,
    module: OpRef,
    symbols: &SymbolTable,
    is_function: FunctionDefinition,
) -> CallGraph {
    let mut graph = CallGraph::default();
    let _ = walk_op::<()>(ctx, module, &mut |op| {
        if is_function(ctx, op) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        ctx.op(op).attributes.visit_symbol_refs(&mut |reference| {
            if let Some(target) = symbols.resolve(ctx, op, reference) {
                graph.module_references.insert(target);
                graph.address_taken.insert(target);
            }
        });
        ControlFlow::Continue(WalkAction::Advance)
    });
    graph.functions = symbols
        .symbols()
        .iter()
        .copied()
        .filter(|&op| is_function(ctx, op))
        .collect();
    for caller in graph.functions.clone() {
        let mut body = BodyReferences::default();
        for region in ctx.op_regions(caller) {
            collect_references(ctx, region, symbols, &mut body, is_function);
        }
        graph.add_body(caller, body);
    }
    graph
}

/// Record the calls and references in `region` into `body`. Nested function
/// definitions record their own.
fn collect_references(
    ctx: &IrContext,
    region: RegionRef,
    symbols: &SymbolTable,
    body: &mut BodyReferences,
    is_function: FunctionDefinition,
) {
    let _ = walk_region::<()>(ctx, region, &mut |op| {
        if is_function(ctx, op) {
            return ControlFlow::Continue(WalkAction::Skip);
        }
        let mut callee = DynCallLike::get(ctx, op).map(|call| call.direct_callee(ctx));
        ctx.op(op).attributes.visit_symbol_refs(&mut |reference| {
            let is_callee = callee == Some(reference);
            if is_callee {
                callee = None;
            }
            let Some(target) = symbols.resolve(ctx, op, reference) else {
                return;
            };
            if is_callee {
                *body.calls.entry(target).or_insert(0) += 1;
            } else {
                body.addresses.insert(target);
            }
        });
        ControlFlow::Continue(WalkAction::Advance)
    });
}

/// `CallGraph` as an [`Analysis`] over the [`SymbolTable`] of `target`, which
/// must be a `core.module` op.
impl Analysis for CallGraph {
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
        let symbols = ctx.get::<SymbolTable>(target)?;
        Ok(call_graph_over(
            ctx.ir(),
            target,
            &symbols,
            func::Func::matches,
        ))
    }
}

/// Depends only on the infallible [`SymbolTable`].
impl InfallibleAnalysis for CallGraph {}

/// Compute the SCC id for every function in the call graph using Tarjan's algorithm.
///
/// Only functions in `graph.functions` are assigned an id; edges to any other
/// symbol are skipped.
pub fn tarjan_scc(graph: &CallGraph) -> HashMap<OpRef, u32> {
    scc_over(graph, &graph.edges)
}

fn scc_over(graph: &CallGraph, edges: &Edges) -> HashMap<OpRef, u32> {
    let functions: HashSet<OpRef> = graph.functions.iter().copied().collect();
    let mut state = TarjanState::default();
    for &v in &graph.functions {
        if !state.index.contains_key(&v) {
            strongconnect(v, &mut state, &functions, edges);
        }
    }
    state.scc_id
}

/// Functions that participate in a recursive cycle (direct or mutual).
///
/// A function is "recursive" if it belongs to an SCC of size > 1 **or** its
/// singleton SCC contains a self-edge (direct self-recursion).
pub fn recursive_functions(graph: &CallGraph) -> HashSet<OpRef> {
    cyclic_functions(graph, &graph.edges)
}

/// Functions on a cycle of direct calls.
///
/// Address references are ignored: a reference copied into a caller does not
/// add a call site, so only these functions can be instantiated without
/// bound by inlining.
pub fn directly_recursive_functions(graph: &CallGraph) -> HashSet<OpRef> {
    cyclic_functions(graph, &graph.calls)
}

fn cyclic_functions(graph: &CallGraph, edges: &Edges) -> HashSet<OpRef> {
    let scc_ids = scc_over(graph, edges);
    let mut by_scc: HashMap<u32, Vec<OpRef>> = HashMap::default();
    for (&v, &id) in &scc_ids {
        by_scc.entry(id).or_default().push(v);
    }
    let mut result = HashSet::default();
    for members in by_scc.into_values() {
        if members.len() > 1 {
            result.extend(members);
        } else {
            let v = members[0];
            if edges.get(&v).is_some_and(|s| s.contains(&v)) {
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
    stack: Vec<OpRef>,
    on_stack: HashSet<OpRef>,
    index: HashMap<OpRef, u32>,
    lowlink: HashMap<OpRef, u32>,
    scc_id: HashMap<OpRef, u32>,
    next_scc: u32,
}

fn strongconnect(v: OpRef, state: &mut TarjanState, functions: &HashSet<OpRef>, edges: &Edges) {
    let v_index = state.next_index;
    state.next_index += 1;
    state.index.insert(v, v_index);
    state.lowlink.insert(v, v_index);
    state.stack.push(v);
    state.on_stack.insert(v);

    if let Some(successors) = edges.get(&v) {
        let successors: Vec<OpRef> = successors.iter().copied().collect();
        for w in successors {
            // Skip symbols that are not functions of this graph.
            if !functions.contains(&w) {
                continue;
            }
            if !state.index.contains_key(&w) {
                strongconnect(w, state, functions, edges);
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
mod prop_tests;

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
        let nil_ty = crate::dialect::core::Nil::params()
            .intern(ctx)
            .as_type_ref();
        crate::dialect::func::func_sig(ctx, [], [nil_ty]).as_type_ref()
    }

    fn i32_type(ctx: &mut IrContext) -> TypeRef {
        TypeDataBuilder::new("core.i32").intern(ctx)
    }

    /// A call graph with each symbol spelled by its root-qualified path.
    struct Named {
        functions: HashSet<SymbolPath>,
        edges: HashMap<SymbolPath, HashSet<SymbolPath>>,
        address_taken: HashSet<SymbolPath>,
        call_site_count: HashMap<SymbolPath, usize>,
        recursive: HashSet<SymbolPath>,
    }

    fn named(ctx: &IrContext, graph: &CallGraph) -> Named {
        let name = |op: &OpRef| crate::symbol_table::qualified_name(ctx, *op).expect("symbol");
        let set = |ops: &HashSet<OpRef>| ops.iter().map(name).collect::<HashSet<_>>();
        Named {
            functions: graph.functions.iter().map(name).collect(),
            edges: graph
                .edges
                .iter()
                .map(|(caller, callees)| (name(caller), set(callees)))
                .collect(),
            address_taken: set(&graph.address_taken),
            call_site_count: graph
                .call_site_count
                .iter()
                .map(|(callee, &count)| (name(callee), count))
                .collect(),
            recursive: set(&recursive_functions(graph)),
        }
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
            .sym_name(ctx.intern_str(name))
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
                .callee(SymbolPath::new([*callee]))
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
            .sym_name(ctx.intern_str(name))
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
        let module_data = OperationDataBuilder::new(loc, "core.module")
            .attr("sym_name", Attribute::String(ctx.intern_str(name)))
            .region(region)
            .build(ctx);
        ctx.create_op(module_data)
    }

    #[test]
    fn a_reference_other_than_a_direct_callee_takes_the_address() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @called() {
    func.return
  }
  func.func @captured() {
    func.return
  }
  func.func @listed() {
    func.return
  }
  func.func @holder() {
    func.call {callee = @called}
    test.make_closure {func_ref = @captured}
    test.table {entries = [{target = @listed}], callee = @called}
    func.return
  }
}"#,
        );

        let g = named(&ctx, &build_call_graph(&ctx, module));
        let [called, captured, listed] =
            ["called", "captured", "listed"].map(|name| SymbolPath::new([name]));
        assert_eq!(
            g.edges[&SymbolPath::new(["holder"])],
            [called.clone(), captured.clone(), listed.clone()]
                .into_iter()
                .collect::<HashSet<_>>()
        );
        // `callee` is a call only on an operation registered as a direct call.
        assert_eq!(
            g.address_taken,
            [called.clone(), captured, listed]
                .into_iter()
                .collect::<HashSet<_>>()
        );
        assert_eq!(
            g.call_site_count,
            [(called, 1)].into_iter().collect::<HashMap<_, _>>()
        );
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
        assert_eq!(old.call_site_count.get(&leaf), Some(&1));

        let body = ctx.op_region(caller, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let call = ctx.block(block).ops[0];
        ctx.op_attrs_mut(call)
            .insert("callee", Attribute::SymbolRef(SymbolPath::new(["other"])));
        assert!(
            analyses
                .get_cached::<CallGraph>(&ctx, module.op())
                .is_none()
        );
        let fresh = analyses.require::<CallGraph>(&ctx, module.op());
        assert_eq!(fresh.call_site_count.get(&leaf), None);
        assert_eq!(fresh.call_site_count.get(&other), Some(&1));
        assert_eq!(old.call_site_count.get(&leaf), Some(&1));
        assert!(!std::sync::Arc::ptr_eq(&old, &fresh));
    }

    /// The call in `caller`'s body to `callee`, the first one if several.
    fn call_to(ctx: &IrContext, caller: OpRef, callee: &str) -> OpRef {
        let body = ctx.op_region(caller, 0).unwrap();
        let mut found = None;
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            if found.is_none()
                && func::Call::from_op(ctx, op).is_ok_and(|call| call.callee(ctx) == [callee])
            {
                found = Some(op);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        found.expect("caller calls callee")
    }

    fn root_symbol(ctx: &IrContext, module: Module, name: &str) -> OpRef {
        SymbolTable::collect(ctx, module)
            .resolve_in(ctx, module.op(), &SymbolPath::new([name]))
            .unwrap()
    }

    const INLINE_FIXTURE: &str = r#"core.module @test {
  func.func @leaf() -> core.i32 {
    %0 = arith.const {value = 0} : core.i32
    func.return %0
  }
  func.func @other() -> core.i32 {
    %0 = arith.const {value = 1} : core.i32
    func.return %0
  }
  func.func @helper() -> core.i32 {
    %0 = func.call {callee = @leaf} : core.i32
    %1 = func.constant {func_ref = @other} : core.ptr
    func.return %0
  }
  func.func @main() -> core.i32 {
    %0 = func.call {callee = @helper} : core.i32
    %1 = func.call {callee = @helper} : core.i32
    func.return %1
  }
}"#;

    #[test]
    fn inline_call_matches_a_rebuild() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, INLINE_FIXTURE);
        let [main, helper, leaf, other] =
            ["main", "helper", "leaf", "other"].map(|name| root_symbol(&ctx, module, name));
        let mut graph = build_call_graph(&ctx, module);

        // One of two calls: `main` still calls `helper` and now also
        // calls `leaf` and takes the address of `other`.
        let call = call_to(&ctx, main, "helper");
        crate::transforms::inline_single_call(&mut ctx, call, helper).unwrap();
        graph.inline_call(main, helper);
        assert_eq!(graph, build_call_graph(&ctx, module));
        assert_eq!(graph.call_site_count.get(&helper), Some(&1));
        assert_eq!(graph.call_site_count.get(&leaf), Some(&2));

        // The last call: `main` no longer calls `helper`, and the address
        // of `other` is taken once by each body.
        let call = call_to(&ctx, main, "helper");
        crate::transforms::inline_single_call(&mut ctx, call, helper).unwrap();
        graph.inline_call(main, helper);
        assert_eq!(graph, build_call_graph(&ctx, module));
        assert_eq!(graph.call_site_count.get(&helper), None);
        assert!(!graph.calls[&main].contains(&helper));
        assert!(graph.address_taken.contains(&other));
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

        let g = named(&ctx, &build_call_graph(&ctx, module));
        assert!(
            g.functions.contains(&SymbolPath::new(["inner", "foo"])),
            "expected qualified symbol `inner::foo`, got: {:?}",
            g.functions
        );
        // Unqualified symbol should not leak through.
        assert!(!g.functions.contains(&SymbolPath::new(["foo"])));
        // Top-level function stays unqualified.
        assert!(g.functions.contains(&SymbolPath::new(["top"])));
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

        assert_eq!(direct.functions, cached.functions);
        assert_eq!(direct.edges, cached.edges);
        let names = named(&ctx, &cached);
        assert!(names.functions.contains(&SymbolPath::new(["leaf"])));
        assert!(names.edges.contains_key(&SymbolPath::new(["main"])));

        // Second call should hit the cache.
        let cached2 = am.require::<CallGraph>(&ctx, module.op());
        assert!(std::sync::Arc::ptr_eq(&cached, &cached2));
    }

    #[test]
    fn edges_resolve_callees_in_the_nearest_module() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @root {
  func.func @main() {
    func.call {callee = @outer::@same}
    func.return
  }
  core.module @outer {
    func.func @same() {
      func.call {callee = @inner::@same}
      func.call {callee = @main}
      func.return
    }
    core.module @inner {
      func.func @same() {
        func.call {callee = @same}
        func.return
      }
    }
  }
}"#,
        );
        let g = named(&ctx, &build_call_graph(&ctx, module));
        let outer = SymbolPath::new(["outer", "same"]);
        let inner = SymbolPath::new(["outer", "inner", "same"]);
        assert!(g.edges[&SymbolPath::new(["main"])].contains(&outer));
        // `@main` inside `outer` is looked up in `outer`, which has none, so
        // it adds no edge to the root's `main`.
        assert_eq!(g.edges[&outer], [inner.clone()].into_iter().collect());
        assert_eq!(g.call_site_count[&inner], 2);
        // Only the innermost function calls itself.
        assert_eq!(g.recursive, [inner].into_iter().collect::<HashSet<_>>());
    }

    #[test]
    fn no_reference_reaches_a_duplicated_name() {
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
        let graph = build_call_graph(&ctx, module);
        // Both definitions of `twice` are functions; calls in each body are
        // recorded.
        assert_eq!(graph.functions.len(), 3);
        let g = named(&ctx, &graph);
        assert_eq!(g.call_site_count[&SymbolPath::new(["leaf"])], 1);
    }
}

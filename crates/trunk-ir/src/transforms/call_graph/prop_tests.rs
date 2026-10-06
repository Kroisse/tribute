//! Property tests comparing the call graph and its SCCs with naive
//! reference implementations on random modules.
//!
//! A [`ModuleSpec`] describes functions in the root module and in one nested
//! module, each referencing other functions or undefined externals by direct
//! calls, `func.constant`, and symbol references on operations that are not
//! calls. Module-level operations reference functions as exports. The
//! oracle reads the edges straight from the spec: SCCs are the classes of
//! mutual reachability, and a function is recursive when it reaches itself.
//! Global DCE's property tests build their modules from the same spec.

use proptest::prelude::*;
use smallvec::smallvec;

use super::*;
use crate::dialect::{core, func, wasm};
use crate::location::Span;
use crate::{
    Attribute, BlockData, BlockRef, Location, OperationDataBuilder, RegionData, Symbol, TypeRef,
};

/// How a function body refers to its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefKind {
    /// `func.call {callee = @target}`.
    Call,
    /// `func.constant {func_ref = @target}`.
    Constant,
    /// `test.ref {callee = @target}`: `callee` on an operation that is not a
    /// registered call.
    NonCallCallee,
    /// `test.table {entries = [@target]}`.
    ListEntry,
}

#[derive(Clone, Debug)]
pub(crate) struct Reference {
    pub kind: RefKind,
    /// An index into [`ModuleSpec::paths`]: a function, or an undefined
    /// external past the functions.
    pub target: usize,
    /// Whether the referencing operation is nested in a region of another
    /// operation in the body.
    pub in_region: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct FunctionSpec {
    pub name: String,
    /// Whether the function is defined in the nested module `m`.
    pub nested: bool,
    /// Whether the function has an `abi` attribute.
    pub abi: bool,
    /// Whether the function has no body; only `abi` functions are bodyless.
    pub bodyless: bool,
    pub refs: Vec<Reference>,
}

/// A module-level reference, outside every function.
#[derive(Clone, Debug)]
pub(crate) struct ModuleReference {
    pub target: usize,
    /// `wasm.export_func` when false, `test.table` when true.
    pub table: bool,
    /// Whether the operation is in the nested module.
    pub nested: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ModuleSpec {
    pub functions: Vec<FunctionSpec>,
    pub module_refs: Vec<ModuleReference>,
}

/// The number of undefined external names references may target.
const EXTERNALS: usize = 2;

pub(crate) fn module_spec(max_functions: usize) -> impl Strategy<Value = ModuleSpec> {
    (1..=max_functions).prop_flat_map(|count| {
        let targets = count + EXTERNALS;
        let kind = prop_oneof![
            4 => Just(RefKind::Call),
            1 => Just(RefKind::Constant),
            1 => Just(RefKind::NonCallCallee),
            1 => Just(RefKind::ListEntry),
        ];
        let reference =
            (kind, 0..targets, prop::bool::weighted(0.2)).prop_map(|(kind, target, in_region)| {
                Reference {
                    kind,
                    target,
                    in_region,
                }
            });
        let function = (
            prop::bool::weighted(0.25),
            prop::bool::weighted(0.2),
            prop::bool::weighted(0.3),
            prop::collection::vec(reference, 0..4),
        );
        let module_ref = (0..targets, any::<bool>(), prop::bool::weighted(0.25)).prop_map(
            |(target, table, nested)| ModuleReference {
                target,
                table,
                nested,
            },
        );
        (
            prop::collection::vec(function, count),
            // Function 0 may be `main` and function 1 `_start`.
            any::<bool>(),
            any::<bool>(),
            prop::collection::vec(module_ref, 0..3),
        )
            .prop_map(|(functions, main, start, module_refs)| {
                let functions = functions
                    .into_iter()
                    .enumerate()
                    .map(|(index, (nested, abi, bodyless, refs))| {
                        let name = match index {
                            0 if main => "main".to_owned(),
                            1 if start => "_start".to_owned(),
                            _ => format!("f{index}"),
                        };
                        let bodyless = abi && bodyless;
                        FunctionSpec {
                            name,
                            nested,
                            abi,
                            bodyless,
                            refs: if bodyless { Vec::new() } else { refs },
                        }
                    })
                    .collect();
                ModuleSpec {
                    functions,
                    module_refs,
                }
            })
    })
}

impl ModuleSpec {
    /// The root-qualified path of every function, then of every external.
    pub fn paths(&self) -> Vec<SymbolPath> {
        self.functions
            .iter()
            .map(|function| {
                if function.nested {
                    SymbolPath::new(["m", function.name.as_str()])
                } else {
                    SymbolPath::from(function.name.as_str())
                }
            })
            .chain((0..EXTERNALS).map(|index| SymbolPath::from(format!("ext{index}").as_str())))
            .collect()
    }

    /// The targets of each function's references, by index, optionally only
    /// its direct calls.
    pub fn successors(&self, calls_only: bool) -> Vec<Vec<usize>> {
        let mut successors = vec![Vec::new(); self.functions.len() + EXTERNALS];
        for (index, function) in self.functions.iter().enumerate() {
            successors[index] = function
                .refs
                .iter()
                .filter(|reference| !calls_only || reference.kind == RefKind::Call)
                .map(|reference| reference.target)
                .collect();
        }
        successors
    }

    /// Build the module this spec describes.
    pub fn build(&self, ctx: &mut IrContext) -> Module {
        let loc = location(ctx);
        let paths = self.paths();
        let nil = core::nil(ctx).as_type_ref();
        let sig = func::func_sig(ctx, [], [nil]).as_type_ref();
        let root_block = empty_block(ctx, loc);
        let nested_block = empty_block(ctx, loc);
        for function in &self.functions {
            let op = build_function(ctx, loc, sig, function, &paths);
            let block = if function.nested {
                nested_block
            } else {
                root_block
            };
            ctx.push_op(block, op);
        }
        for reference in &self.module_refs {
            let target = paths[reference.target].clone();
            let op = if reference.table {
                table(ctx, loc, target)
            } else {
                wasm::ExportFunc::operands()
                    .name("export")
                    .func(target)
                    .build(ctx, loc)
                    .op_ref()
            };
            let block = if reference.nested {
                nested_block
            } else {
                root_block
            };
            ctx.push_op(block, op);
        }
        let nested = module_op(ctx, loc, "m", nested_block);
        ctx.push_op(root_block, nested);
        let root = module_op(ctx, loc, "root", root_block);
        Module::new(ctx, root).expect("core.module")
    }
}

fn location(ctx: &mut IrContext) -> Location {
    let path = ctx.intern_path("file:///prop.trb");
    Location::new(path, Span::new(0, 0))
}

fn empty_block(ctx: &mut IrContext, loc: Location) -> BlockRef {
    ctx.create_block(BlockData {
        location: loc,
        args: vec![],
        ops: smallvec![],
        parent_region: None,
    })
}

fn single_block_region(ctx: &mut IrContext, loc: Location, block: BlockRef) -> RegionRef {
    ctx.create_region(RegionData {
        location: loc,
        blocks: smallvec![block],
        parent_op: None,
    })
}

fn module_op(ctx: &mut IrContext, loc: Location, name: &'static str, block: BlockRef) -> OpRef {
    let region = single_block_region(ctx, loc, block);
    core::Module::operands()
        .sym_name(name)
        .regions(region)
        .build(ctx, loc)
        .op_ref()
}

fn table(ctx: &mut IrContext, loc: Location, target: SymbolPath) -> OpRef {
    let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("table"))
        .attr(
            "entries",
            Attribute::List(vec![Attribute::SymbolRef(target)]),
        )
        .build(ctx);
    ctx.create_op(data)
}

fn build_function(
    ctx: &mut IrContext,
    loc: Location,
    sig: TypeRef,
    function: &FunctionSpec,
    paths: &[SymbolPath],
) -> OpRef {
    // The generated builder requires a body, so build the operation
    // generically to allow a bodyless declaration.
    let mut builder = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
        .attr(
            "sym_name",
            Attribute::String(ctx.intern_str(&function.name)),
        )
        .attr("type", Attribute::Type(sig));
    if function.abi {
        builder = builder.attr("abi", ctx.string_attr("C"));
    }
    if !function.bodyless {
        let entry = empty_block(ctx, loc);
        for reference in &function.refs {
            let target = paths[reference.target].clone();
            let op = match reference.kind {
                RefKind::Call => func::Call::operands(std::iter::empty())
                    .callee(target)
                    .results([])
                    .build(ctx, loc)
                    .op_ref(),
                RefKind::Constant => func::Constant::operands()
                    .func_ref(target)
                    .results(sig)
                    .build(ctx, loc)
                    .op_ref(),
                RefKind::NonCallCallee => {
                    let data =
                        OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("ref"))
                            .attr("callee", Attribute::SymbolRef(target))
                            .build(ctx);
                    ctx.create_op(data)
                }
                RefKind::ListEntry => table(ctx, loc, target),
            };
            let op = if reference.in_region {
                let inner = empty_block(ctx, loc);
                ctx.push_op(inner, op);
                let region = single_block_region(ctx, loc, inner);
                let data =
                    OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("scope"))
                        .region(region)
                        .build(ctx);
                ctx.create_op(data)
            } else {
                op
            };
            ctx.push_op(entry, op);
        }
        let ret = func::Return::operands(std::iter::empty()).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        builder = builder.region(single_block_region(ctx, loc, entry));
    }
    let data = builder.build(ctx);
    ctx.create_op(data)
}

/// The nodes reachable from `starts` along at least zero edges.
pub(crate) fn reachable_from(
    successors: &[Vec<usize>],
    starts: impl IntoIterator<Item = usize>,
) -> Vec<bool> {
    let mut seen = vec![false; successors.len()];
    let mut pending: Vec<usize> = starts.into_iter().collect();
    while let Some(node) = pending.pop() {
        if !std::mem::replace(&mut seen[node], true) {
            pending.extend(successors[node].iter().copied());
        }
    }
    seen
}

/// Whether `from` reaches `to` along at least one edge.
fn reaches(successors: &[Vec<usize>], from: usize, to: usize) -> bool {
    reachable_from(successors, successors[from].iter().copied())[to]
}

fn path_set(paths: &[SymbolPath], indices: impl IntoIterator<Item = usize>) -> HashSet<SymbolPath> {
    indices
        .into_iter()
        .map(|index| paths[index].clone())
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn call_graph_records_every_reference(spec in module_spec(10)) {
        let mut ctx = IrContext::new();
        let module = spec.build(&mut ctx);
        let graph = build_call_graph(&ctx, module);
        let paths = spec.paths();
        let functions = spec.functions.len();

        prop_assert_eq!(
            graph.func_ops.keys().cloned().collect::<HashSet<_>>(),
            path_set(&paths, 0..functions)
        );
        for calls_only in [false, true] {
            let expected: HashMap<SymbolPath, HashSet<SymbolPath>> = spec
                .successors(calls_only)
                .iter()
                .enumerate()
                .filter(|(_, targets)| !targets.is_empty())
                .map(|(index, targets)| {
                    (paths[index].clone(), path_set(&paths, targets.iter().copied()))
                })
                .collect();
            let actual = if calls_only { &graph.calls } else { &graph.edges };
            prop_assert_eq!(actual, &expected, "calls_only = {}", calls_only);
        }

        let mut call_sites: HashMap<SymbolPath, usize> = HashMap::default();
        let mut address_taken = HashSet::default();
        for reference in spec.functions.iter().flat_map(|function| &function.refs) {
            let target = paths[reference.target].clone();
            if reference.kind == RefKind::Call {
                *call_sites.entry(target).or_default() += 1;
            } else {
                address_taken.insert(target);
            }
        }
        let module_references = path_set(
            &paths,
            spec.module_refs.iter().map(|reference| reference.target),
        );
        address_taken.extend(module_references.iter().cloned());
        prop_assert_eq!(&graph.call_site_count, &call_sites);
        prop_assert_eq!(&graph.module_references, &module_references);
        prop_assert_eq!(&graph.address_taken, &address_taken);
    }

    #[test]
    fn sccs_are_the_classes_of_mutual_reachability(spec in module_spec(10)) {
        let mut ctx = IrContext::new();
        let module = spec.build(&mut ctx);
        let graph = build_call_graph(&ctx, module);
        let paths = spec.paths();
        let functions = spec.functions.len();
        let successors = spec.successors(false);

        let ids = tarjan_scc(&graph);
        prop_assert_eq!(
            ids.keys().cloned().collect::<HashSet<_>>(),
            path_set(&paths, 0..functions)
        );
        for a in 0..functions {
            let from_a = reachable_from(&successors, [a]);
            for b in 0..functions {
                let mutual = from_a[b] && reachable_from(&successors, [b])[a];
                prop_assert_eq!(
                    ids[&paths[a]] == ids[&paths[b]],
                    mutual,
                    "{} and {} in {:?}",
                    paths[a],
                    paths[b],
                    spec
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
            prop_assert_eq!(actual, expected, "calls_only = {}", calls_only);
        }
    }
}

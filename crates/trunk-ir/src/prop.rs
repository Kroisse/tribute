//! Random IR for property tests.
//!
//! A spec is a plain description of IR: a [`CfgSpec`] gives the successor
//! list of each block of one function body, and a [`ModuleSpec`] gives
//! functions in the root module and in one nested module `m`, each with such
//! a body and with references to other functions or undefined externals
//! placed in its blocks. A spec builds its IR through [`BuildIr`], and
//! [`built`] turns a spec strategy into one yielding [`Built`] values:
//! proptest shrinks the spec, rebuilds the IR from each shrunk spec, and
//! reports a failing case with the spec and its IR text.
//!
//! Oracles read the expected answers from the spec, never from the built IR,
//! so they do not depend on the code under test.

use std::fmt;
use std::ops::RangeInclusive;

use proptest::prelude::*;
use smallvec::smallvec;

use crate::dialect::{arith, cf, core, func, wasm};
use crate::location::Span;
use crate::printer::print_module;
use crate::rewrite::Module;
use crate::symbol::SymbolPath;
use crate::{
    Attribute, BlockData, BlockRef, IrContext, Location, OpRef, OperationDataBuilder, RegionData,
    RegionRef, Symbol, TypeRef,
};

/// A spec that can build the IR it describes.
pub(crate) trait BuildIr {
    /// Build the IR into `ctx` and return its root module.
    fn build(&self, ctx: &mut IrContext) -> Module;
}

/// A spec together with the IR built from it.
pub(crate) struct Built<S> {
    pub spec: S,
    pub ctx: IrContext,
    pub module: Module,
}

impl<S: fmt::Debug> fmt::Debug for Built<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:?}", self.spec)?;
        f.write_str(&print_module(&self.ctx, self.module.op()))
    }
}

/// Build each generated spec into a fresh context.
pub(crate) fn built<S: BuildIr + fmt::Debug>(
    specs: impl Strategy<Value = S>,
) -> impl Strategy<Value = Built<S>> {
    specs.prop_map(|spec| {
        let mut ctx = IrContext::new();
        let module = spec.build(&mut ctx);
        Built { spec, ctx, module }
    })
}

// ============================================================================
// Control-flow graphs
// ============================================================================

/// The successor list of each block of a function body, by block index;
/// block 0 is the entry.
///
/// Blocks end in real terminators: `func.return` without successors,
/// `cf.br` with one, `cf.cond_br` with two, and `cf.switch` with more.
#[derive(Clone, Debug)]
pub(crate) struct CfgSpec {
    pub successors: Vec<Vec<usize>>,
}

/// Control-flow graphs with a block count in `blocks`, including loops,
/// self-loops, repeated edges, and blocks unreachable from the entry.
pub(crate) fn cfg_spec(blocks: RangeInclusive<usize>) -> impl Strategy<Value = CfgSpec> {
    blocks.prop_flat_map(|blocks| {
        // Successor counts favor jumps and two-way branches but include
        // returns and multi-way switches.
        let successors = prop_oneof![
            2 => Just(0usize),
            4 => Just(1usize),
            4 => Just(2usize),
            1 => 3usize..=4,
        ]
        .prop_flat_map(move |count| prop::collection::vec(0..blocks.max(1), count));
        prop::collection::vec(successors, blocks).prop_map(|successors| CfgSpec { successors })
    })
}

impl BuildIr for CfgSpec {
    /// A root module holding one function `f` with this body.
    fn build(&self, ctx: &mut IrContext) -> Module {
        let loc = location(ctx);
        let sig = unit_sig(ctx);
        let body = build_body(ctx, loc, self, |_, _| Vec::new());
        let function = func::Func::operands()
            .sym_name("f")
            .r#type(sig)
            .regions(body)
            .build(ctx, loc)
            .op_ref();
        let block = empty_block(ctx, loc);
        ctx.push_op(block, function);
        let root = module_op(ctx, loc, "root", block);
        Module::new(ctx, root).expect("core.module")
    }
}

/// Build a body region with one block per entry of `cfg`. `ops` returns the
/// operations of each block, which precede its terminator.
fn build_body(
    ctx: &mut IrContext,
    loc: Location,
    cfg: &CfgSpec,
    mut ops: impl FnMut(&mut IrContext, usize) -> Vec<OpRef>,
) -> RegionRef {
    let blocks: Vec<BlockRef> = cfg
        .successors
        .iter()
        .map(|_| empty_block(ctx, loc))
        .collect();
    let i1 = core::I1::type_ref(ctx);
    let i32_ty = core::I32::type_ref(ctx);
    for (index, (&block, successors)) in blocks.iter().zip(&cfg.successors).enumerate() {
        for op in ops(ctx, index) {
            ctx.push_op(block, op);
        }
        let targets: Vec<BlockRef> = successors.iter().map(|&i| blocks[i]).collect();
        let terminator = match targets.as_slice() {
            [] => func::Return::operands(std::iter::empty())
                .build(ctx, loc)
                .op_ref(),
            &[dest] => cf::Br::operands(std::iter::empty::<crate::ValueRef>())
                .successors(dest)
                .build(ctx, loc)
                .op_ref(),
            &[then_dest, else_dest] => {
                let cond = arith::Const::operands()
                    .value(Attribute::Int(1))
                    .results(i1)
                    .build(ctx, loc);
                ctx.push_op(block, cond.op_ref());
                let cond = cond.result(ctx);
                cf::CondBr::operands(cond)
                    .successors(then_dest, else_dest)
                    .build(ctx, loc)
                    .op_ref()
            }
            [default, cases @ ..] => {
                let discriminant = arith::Const::operands()
                    .value(Attribute::Int(0))
                    .results(i32_ty)
                    .build(ctx, loc);
                ctx.push_op(block, discriminant.op_ref());
                let discriminant = discriminant.result(ctx);
                cf::Switch::operands(discriminant)
                    .cases(0..cases.len() as i64)
                    .successors(*default, cases.iter().copied())
                    .build(ctx, loc)
                    .op_ref()
            }
        };
        ctx.push_op(block, terminator);
    }
    ctx.create_region(RegionData {
        location: loc,
        blocks: blocks.into_iter().collect(),
        parent_op: None,
    })
}

// ============================================================================
// Modules
// ============================================================================

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
    /// The body block holding the reference; the block may be unreachable.
    pub block: usize,
    /// Whether the referencing operation is nested in a region of another
    /// operation in the block.
    pub in_region: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct FunctionSpec {
    pub name: String,
    /// Whether the function is defined in the nested module `m`.
    pub nested: bool,
    /// Whether the function has an `abi` attribute.
    pub abi: bool,
    /// The body, or `None` for a bodyless declaration; only `abi` functions
    /// are bodyless.
    pub body: Option<CfgSpec>,
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

/// Modules with one to `max_functions` functions, each with a body of up to
/// four blocks.
pub(crate) fn module_spec(max_functions: usize) -> impl Strategy<Value = ModuleSpec> {
    (1..=max_functions).prop_flat_map(|count| {
        let targets = count + EXTERNALS;
        let function = (
            prop::bool::weighted(0.25),
            prop::bool::weighted(0.2),
            prop::bool::weighted(0.3),
            cfg_spec(1..=4),
        )
            .prop_flat_map(move |(nested, abi, bodyless, body)| {
                let kind = prop_oneof![
                    4 => Just(RefKind::Call),
                    1 => Just(RefKind::Constant),
                    1 => Just(RefKind::NonCallCallee),
                    1 => Just(RefKind::ListEntry),
                ];
                let reference = (
                    kind,
                    0..targets,
                    0..body.successors.len(),
                    prop::bool::weighted(0.2),
                )
                    .prop_map(|(kind, target, block, in_region)| Reference {
                        kind,
                        target,
                        block,
                        in_region,
                    });
                let bodyless = abi && bodyless;
                (
                    Just((nested, abi, (!bodyless).then_some(body))),
                    prop::collection::vec(reference, if bodyless { 0..1 } else { 0..4 }),
                )
            });
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
                    .map(|(index, ((nested, abi, body), refs))| {
                        let name = match index {
                            0 if main => "main".to_owned(),
                            1 if start => "_start".to_owned(),
                            _ => format!("f{index}"),
                        };
                        FunctionSpec {
                            name,
                            nested,
                            abi,
                            body,
                            refs,
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
}

impl BuildIr for ModuleSpec {
    fn build(&self, ctx: &mut IrContext) -> Module {
        let loc = location(ctx);
        let paths = self.paths();
        let sig = unit_sig(ctx);
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
    if let Some(body) = &function.body {
        let body = build_body(ctx, loc, body, |ctx, block| {
            function
                .refs
                .iter()
                .filter(|reference| reference.block == block)
                .map(|reference| build_reference(ctx, loc, sig, reference, paths))
                .collect()
        });
        builder = builder.region(body);
    }
    let data = builder.build(ctx);
    ctx.create_op(data)
}

fn build_reference(
    ctx: &mut IrContext,
    loc: Location,
    sig: TypeRef,
    reference: &Reference,
    paths: &[SymbolPath],
) -> OpRef {
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
            let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("ref"))
                .attr("callee", Attribute::SymbolRef(target))
                .build(ctx);
            ctx.create_op(data)
        }
        RefKind::ListEntry => table(ctx, loc, target),
    };
    if !reference.in_region {
        return op;
    }
    let inner = empty_block(ctx, loc);
    ctx.push_op(inner, op);
    let region = single_block_region(ctx, loc, inner);
    let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("scope"))
        .region(region)
        .build(ctx);
    ctx.create_op(data)
}

// ============================================================================
// Oracles and helpers
// ============================================================================

/// The nodes reachable from `starts` along at least zero edges, never
/// entering `avoiding`.
pub(crate) fn reachable_from(
    successors: &[Vec<usize>],
    starts: impl IntoIterator<Item = usize>,
    avoiding: Option<usize>,
) -> Vec<bool> {
    let mut seen = vec![false; successors.len()];
    let mut pending: Vec<usize> = starts
        .into_iter()
        .filter(|&node| Some(node) != avoiding)
        .collect();
    while let Some(node) = pending.pop() {
        if !std::mem::replace(&mut seen[node], true) {
            pending.extend(
                successors[node]
                    .iter()
                    .copied()
                    .filter(|&successor| Some(successor) != avoiding),
            );
        }
    }
    seen
}

fn location(ctx: &mut IrContext) -> Location {
    let path = ctx.intern_path("file:///prop.trb");
    Location::new(path, Span::new(0, 0))
}

fn unit_sig(ctx: &mut IrContext) -> TypeRef {
    let nil = core::nil(ctx).as_type_ref();
    func::func_sig(ctx, [], [nil]).as_type_ref()
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

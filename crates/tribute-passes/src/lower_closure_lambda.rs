//! Lower `closure.lambda` operations to `func.func` + `closure.new`.
//!
//! This pass performs closure conversion on high-level `closure.lambda` ops:
//!
//! ```text
//! // Before:
//! %k = closure.lambda [%x, %y] {
//!   ^bb0(%param: anyref):
//!     %r = arith.add %x, %param
//!     func.return %r
//! }
//!
//! // After (inside func.func @foo):
//! func.func @foo::__clam_0(%__env: !k, %param: anyref) -> anyref {
//!   %x, %y = closure.unpack %__env
//!   %r = arith.add %x, %param
//!   func.return %r
//! }
//! %k = closure.new @foo::__clam_0, %x, %y
//! ```
//!
//! The code function's environment is the closure itself, typed with the
//! closure's exact type, and it reads its captures with `closure.unpack`.

use crate::collections::HashMap;

use tribute_ir::calling_convention::get_physical_closure_convention;
use tribute_ir::calling_convention::{
    CLOSURE_ENVIRONMENT_INDEX_ATTR, ConventionBearing, DynConventionBearing,
    get_physical_closure_environment_index,
};
use trunk_ir::Symbol;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::dialect::{core, func};
use trunk_ir::ir_mapping::IrMapping;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{Module, erase_op};
use trunk_ir::types::{Attribute, AttributeMap};

use tribute_ir::dialect::closure;

/// Lower all `closure.lambda` ops in the module to `func.func` + `closure.new`.
pub(crate) fn lower_closure_lambda(ctx: &mut IrContext, module: Module) {
    let module_block = match module.first_block(ctx) {
        Some(b) => b,
        None => return,
    };

    let mut namer = LambdaNamer::new();

    // Iterate until no more closure.lambda ops remain.
    // Multiple iterations handle nested lambdas: the first pass converts
    // outermost lambdas, which may reveal inner closure.lambda ops that were
    // inside the cloned body regions.
    loop {
        let lambdas = collect_closure_lambdas(ctx, module);
        if lambdas.is_empty() {
            break;
        }

        let mut made_progress = false;
        for lambda_ref in lambdas {
            made_progress |= lower_single_lambda(ctx, module_block, lambda_ref, &mut namer);
        }
        if !made_progress {
            break;
        }
    }
}

/// Converts each `closure.lambda` into a lifted `func.func` and a `closure.new`.
pub struct LowerClosureLambda;

impl Pass for LowerClosureLambda {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-closure-lambda"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower_closure_lambda(ctx, target.into());
        Ok(())
    }
}

// ============================================================================
// Collection
// ============================================================================

/// Collect all `closure.lambda` ops in the module (DFS, outermost first).
fn collect_closure_lambdas(ctx: &IrContext, module: Module) -> Vec<OpRef> {
    let mut result = Vec::new();
    for &op in module.ops(ctx) {
        collect_lambdas_in_op(ctx, op, &mut result);
    }
    result
}

fn collect_lambdas_in_op(ctx: &IrContext, op: OpRef, result: &mut Vec<OpRef>) {
    if closure::Lambda::matches(ctx, op) {
        // Don't recurse into the lambda body — nested lambdas will be
        // found on the next iteration after this one is lowered.
        result.push(op);
        return;
    }
    for region in ctx.op_regions(op) {
        for &block in ctx.region(region).blocks.iter() {
            for &child_op in ctx.block(block).ops.iter() {
                collect_lambdas_in_op(ctx, child_op, result);
            }
        }
    }
}

// ============================================================================
// Single lambda lowering
// ============================================================================

fn lower_single_lambda(
    ctx: &mut IrContext,
    module_block: BlockRef,
    lambda_ref: OpRef,
    namer: &mut LambdaNamer,
) -> bool {
    let Ok(lambda_op) = closure::Lambda::from_op(ctx, lambda_ref) else {
        return false;
    };

    let location = ctx.op(lambda_ref).location;
    let result_ty = ctx.op_result_types(lambda_ref)[0];

    // Gather captures and their types.
    let captures: Vec<ValueRef> = ctx.op_operands(lambda_ref).to_vec();
    let capture_types: Vec<TypeRef> = captures.iter().map(|&v| ctx.value_ty(v)).collect();

    let body_region = lambda_op.body(ctx);

    // Generate unique lifted name derived from the enclosing function.
    let parent_name = find_enclosing_func_name(ctx, lambda_ref);
    let lifted_name = namer.next_name(&parent_name);

    // Original entry block args = lambda formal parameters.
    let orig_entry = ctx.region(body_region).blocks[0];
    let orig_param_count = ctx.block(orig_entry).args.len();
    let orig_param_types: Vec<TypeRef> = (0..orig_param_count)
        .map(|i| ctx.block(orig_entry).args[i].ty)
        .collect();

    // Convention-proven physical closures define the only legal environment
    // insertion slot. Do not recover hidden operands from a lambda's shape.
    let convention = DynConventionBearing::of(ctx, lambda_ref);
    let environment_index = match (convention, get_physical_closure_convention(ctx, result_ty)) {
        (Some(convention), Some(provenance)) if provenance == convention => {
            let Some(index) = get_physical_closure_environment_index(ctx, result_ty) else {
                return false;
            };
            index
        }
        _ => return false,
    };
    let Some(function_ty) =
        closure::Closure::from_type_ref(ctx, result_ty).map(|closure| closure.func_type(ctx))
    else {
        return false;
    };
    let Some(callable) = func::FuncSig::from_type_ref(ctx, function_ty) else {
        return false;
    };
    if callable.single_result(ctx).is_none() {
        return false;
    }
    if callable.inputs(ctx) != orig_param_types.as_slice() {
        return false;
    }
    if environment_index > orig_param_count {
        return false;
    }
    // === Build the lifted function ===
    let func_body_region = build_lifted_body(
        ctx,
        location,
        body_region,
        &LiftBodyParams {
            captures: &captures,
            capture_types: &capture_types,
            orig_param_types: &orig_param_types,
            environment_index,
            closure_ty: result_ty,
        },
    );

    let func_ty = callable
        .rebuild(ctx, |inputs, _| {
            inputs.insert(environment_index, (result_ty, AttributeMap::new()));
        })
        .as_type_ref();

    let func_op = func::Func::operands()
        .sym_name(ctx.intern_symbol_text(&lifted_name))
        .r#type(func_ty)
        .regions(func_body_region)
        .build(ctx, location);
    if let Some(convention) = convention {
        func_op.set_calling_convention(ctx, convention);
        ctx.op_attrs_mut(func_op.op_ref()).insert(
            CLOSURE_ENVIRONMENT_INDEX_ATTR,
            Attribute::Int(environment_index as i128),
        );
    }
    ctx.push_op(module_block, func_op.op_ref());

    // === Replace closure.lambda at the original site ===
    let parent_block = ctx
        .op(lambda_ref)
        .parent_block
        .expect("closure.lambda should be in a block");

    // Create closure.new holding the captures, replacing the lambda.
    let closure_new_op = closure::New::operands(captures.clone())
        .entries([trunk_ir::SymbolPath::new([lifted_name])])
        .results(result_ty)
        .build(ctx, location);
    if let Some(convention) = convention {
        closure_new_op.set_calling_convention(ctx, convention);
    }
    ctx.insert_op_before(parent_block, lambda_ref, closure_new_op.op_ref());

    // RAUW: lambda result → closure.new result.
    let old_result = ctx.op_result(lambda_ref, 0);
    let new_result = closure_new_op.result(ctx);
    ctx.replace_all_uses(old_result, new_result);

    // Erase the original closure.lambda op. `erase_op` clears the operand
    // use-chains of the lambda and its (now-cloned) body subtree, not just
    // detaches it — see #710.
    erase_op(ctx, lambda_ref);
    true
}

// ============================================================================
// Body construction
// ============================================================================

/// Parameters for building a lifted function body.
struct LiftBodyParams<'a> {
    captures: &'a [ValueRef],
    capture_types: &'a [TypeRef],
    orig_param_types: &'a [TypeRef],
    environment_index: usize,
    /// The exact closure type, which is the environment parameter's type.
    closure_ty: TypeRef,
}

/// Build the lifted function body region by cloning the lambda body
/// with capture values remapped to the closure's unpacked captures.
fn build_lifted_body(
    ctx: &mut IrContext,
    location: trunk_ir::types::Location,
    orig_body: trunk_ir::refs::RegionRef,
    params: &LiftBodyParams<'_>,
) -> trunk_ir::refs::RegionRef {
    let captures = params.captures;
    let capture_types = params.capture_types;
    let orig_param_types = params.orig_param_types;
    let environment_index = params.environment_index;
    let orig_blocks = ctx.region(orig_body).blocks.clone();
    let orig_entry = orig_blocks[0];
    let orig_param_count = orig_param_types.len();

    let mut mapping = IrMapping::new();

    // --- Pass 1: Create new entry block with the physical closure ABI. ---
    let mut new_entry_args: Vec<_> = orig_param_types
        .iter()
        .enumerate()
        .map(|(index, &ty)| BlockArgData {
            ty,
            attrs: ctx.block(orig_entry).args[index].attrs.clone(),
        })
        .collect();
    let environment_attrs = make_bind_name_attrs(ctx, "__env");
    new_entry_args.insert(
        environment_index,
        BlockArgData {
            ty: params.closure_ty,
            attrs: environment_attrs,
        },
    );
    let new_entry = ctx.create_block(BlockData {
        location,
        args: new_entry_args,
        ops: Default::default(),
        parent_region: None,
    });

    mapping.map_block(orig_entry, new_entry);

    // Map logical closure args around the inserted environment parameter.
    for i in 0..orig_param_count {
        let old_arg = ctx.block_arg(orig_entry, i as u32);
        let new_index = i + usize::from(i >= environment_index);
        let new_arg = ctx.block_arg(new_entry, new_index as u32);
        mapping.map_value(old_arg, new_arg);
    }

    // Unpack the captures at the start of the new entry block.
    let environment = ctx.block_arg(new_entry, environment_index as u32);
    if !captures.is_empty() {
        let unpack = closure::Unpack::operands(environment)
            .results(capture_types.to_vec())
            .build(ctx, location);
        ctx.push_op(new_entry, unpack.op_ref());
        for (&capture, &unpacked) in captures.iter().zip(unpack.results(ctx)) {
            mapping.map_value(capture, unpacked);
        }
    }

    // --- Pass 1b: Create remaining blocks (non-entry) ---
    let mut new_blocks = vec![new_entry];
    for &orig_block in &orig_blocks[1..] {
        let arg_data: Vec<BlockArgData> = ctx
            .block(orig_block)
            .args
            .iter()
            .map(|a| BlockArgData {
                ty: a.ty,
                attrs: a.attrs.clone(),
            })
            .collect();
        let block_loc = ctx.block(orig_block).location;
        let new_block = ctx.create_block(BlockData {
            location: block_loc,
            args: arg_data,
            ops: Default::default(),
            parent_region: None,
        });
        mapping.map_block(orig_block, new_block);

        let arg_count = ctx.block(orig_block).args.len();
        for j in 0..arg_count {
            let old_arg = ctx.block_arg(orig_block, j as u32);
            let new_arg = ctx.block_arg(new_block, j as u32);
            mapping.map_value(old_arg, new_arg);
        }
        new_blocks.push(new_block);
    }

    // --- Pass 2: Clone ops from original blocks to new blocks ---
    for (idx, &orig_block) in orig_blocks.iter().enumerate() {
        let new_block = new_blocks[idx];
        let ops = ctx.block(orig_block).ops.clone();
        for op in ops {
            ctx.clone_op_into_block(new_block, op, &mut mapping);
        }
    }

    // Create the function body region.
    ctx.create_region(RegionData {
        location,
        blocks: new_blocks.into(),
        parent_op: None,
    })
}

// ============================================================================
// Helpers
// ============================================================================

/// Walk up the parent chain to find the enclosing `func.func` and return its `sym_name`.
fn find_enclosing_func_name(ctx: &IrContext, op: OpRef) -> String {
    let mut current_op = op;
    while let Some(block) = ctx.op(current_op).parent_block {
        let Some(region) = ctx.block(block).parent_region else {
            break;
        };
        let Some(parent) = ctx.region(region).parent_op else {
            break;
        };
        if let Ok(f) = func::Func::from_op(ctx, parent) {
            return f.sym_name(ctx).to_owned();
        }
        current_op = parent;
    }
    "<anon>".to_string()
}

/// Generates unique lifted lambda names, scoped by parent function.
struct LambdaNamer {
    counters: HashMap<String, u32>,
}

impl LambdaNamer {
    fn new() -> Self {
        Self {
            counters: HashMap::default(),
        }
    }

    fn next_name(&mut self, parent: &str) -> Symbol {
        let count = self.counters.entry_ref(parent).or_insert(0);
        let name = format!("{parent}::__clam_{count}");
        *count += 1;
        Symbol::new(&name)
    }
}

/// Create a `bind_name` attribute map for a block argument.
fn make_bind_name_attrs(ctx: &mut IrContext, name: &str) -> AttributeMap {
    let mut attrs = AttributeMap::new();
    let bind_name = ctx.string_attr(name);
    attrs.view_mut(ctx).insert("bind_name", bind_name);
    attrs
}

#[cfg(test)]
mod tests {
    use super::*;
    use tribute_ir::calling_convention::CallingConvention;
    use tribute_ir::dialect::tribute_rt;
    use trunk_ir::SymbolPath;
    use trunk_ir::context::RegionData;
    use trunk_ir::dialect::{arith, core};
    use trunk_ir::printer::print_module;
    use trunk_ir::refs::PathRef;
    use trunk_ir::types::Location;
    use trunk_ir::types::TypeDataBuilder;
    use trunk_ir::{Attribute, IrContext, OperationDataBuilder, Span};

    fn test_ctx() -> (IrContext, Location) {
        let ctx = IrContext::new();
        let loc = Location::new(PathRef::from_u32(0), Span::default());
        (ctx, loc)
    }

    fn make_module(ctx: &mut IrContext, loc: Location) -> (Module, BlockRef) {
        let module_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let module_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![module_block],
            parent_op: None,
        });
        let module_op = OperationDataBuilder::new(loc, "core.module")
            .attr("sym_name", Attribute::String(ctx.intern_str("test")))
            .region(module_region)
            .build(ctx);
        let module_ref = ctx.create_op(module_op);
        (Module::new(ctx, module_ref).unwrap(), module_block)
    }

    fn make_anyref_ty(ctx: &mut IrContext) -> TypeRef {
        tribute_rt::Anyref::params().intern(ctx).as_type_ref()
    }

    fn make_i32_ty(ctx: &mut IrContext) -> TypeRef {
        TypeDataBuilder::new("core.i32").intern(ctx)
    }

    #[test]
    fn test_lower_lambda_no_captures() {
        let (mut ctx, loc) = test_ctx();
        let (module, module_block) = make_module(&mut ctx, loc);
        let anyref_ty = make_anyref_ty(&mut ctx);
        let i32_ty = make_i32_ty(&mut ctx);

        // Build a func.func containing a closure.lambda with no captures:
        //   func.func @test_fn() {
        //     %k = closure.lambda [] {
        //       ^bb0(%x: i32):
        //         func.return %x
        //     }
        //   }

        // Lambda body: ^bb0(%x: i32): func.return %x
        let x_attrs = make_bind_name_attrs(&mut ctx, "x");
        let lambda_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: x_attrs,
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let x_val = ctx.block_arg(lambda_entry, 0);
        let ret_op = func::Return::operands([x_val]).build(&mut ctx, loc);
        ctx.push_op(lambda_entry, ret_op.op_ref());

        let lambda_body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![lambda_entry],
            parent_op: None,
        });

        // closure type: closure.closure<func.func_sig<i32, i32>>, whose input
        // carries a parameter attribute.
        let marked = AttributeMap::from_entries(
            &mut ctx,
            [("k", Attribute::SymbolRef(SymbolPath::new(["v"])))],
        );
        let func_ty = func::func_sig_with_param_attrs(
            &mut ctx,
            [(i32_ty, marked.clone())],
            [(i32_ty, AttributeMap::new())],
            AttributeMap::new(),
        )
        .as_type_ref();
        let closure_ty = tribute_ir::calling_convention::physical_closure_type(
            &mut ctx,
            func_ty,
            CallingConvention::Direct,
        );

        // closure.lambda [] { ... } -> closure_ty
        let lambda_op = closure::Lambda::operands(Vec::<ValueRef>::new())
            .results(closure_ty)
            .regions(lambda_body_region)
            .build(&mut ctx, loc);
        lambda_op.set_calling_convention(&mut ctx, CallingConvention::Direct);

        // Wrap in a func.func
        let outer_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        ctx.push_op(outer_entry, lambda_op.op_ref());

        let outer_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![outer_entry],
            parent_op: None,
        });
        let outer_func_ty =
            func::func_sig(&mut ctx, std::iter::empty::<TypeRef>(), [anyref_ty]).as_type_ref();
        let outer_func = func::Func::operands()
            .sym_name("test_fn")
            .r#type(outer_func_ty)
            .regions(outer_body)
            .build(&mut ctx, loc);
        ctx.push_op(module_block, outer_func.op_ref());

        // Run the pass.
        lower_closure_lambda(&mut ctx, module);

        // Verify: module should now have 2 functions (test_fn + test_fn::__clam_0).
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 2, "expected 2 top-level ops after lowering");

        // The original function should contain closure.new instead of closure.lambda.
        let test_fn = func::Func::from_op(&ctx, ops[0]).unwrap();
        let test_fn_body = test_fn.body(&ctx);
        let test_fn_entry = ctx.region(test_fn_body).blocks[0];
        let test_fn_ops = &ctx.block(test_fn_entry).ops;

        // A capture-free closure.new is the only operation.
        let [closure_op] = test_fn_ops[..] else {
            panic!("expected only closure.new in test_fn");
        };
        let closure_new = closure::New::from_op(&ctx, closure_op).expect("closure.new");
        assert!(closure_new.captures(&ctx).is_empty());
        let closure_ty = ctx.value_ty(closure_new.result(&ctx));

        // The lifted function should exist.
        let lifted = func::Func::from_op(&ctx, ops[1]).unwrap();
        assert_eq!(lifted.sym_name(&ctx), "test_fn::__clam_0");

        // Direct lifted function has only the physical environment and source arg.
        let lifted_ty = lifted.r#type(&ctx);
        let lifted_type = func::FuncSig::from_type_ref(&ctx, lifted_ty).unwrap();
        assert_eq!(lifted_type.inputs(&ctx).len(), 2); // environment + x
        assert!(lifted_type.single_result(&ctx).is_some());
        // The inserted environment is the closure itself and has no
        // attributes; `x` keeps its own.
        itertools::assert_equal(
            lifted_type.inputs_with_attrs(&ctx),
            [(closure_ty, &AttributeMap::new()), (i32_ty, &marked)],
        );
    }

    #[test]
    fn incomplete_or_mismatched_physical_contract_leaves_lambda_unchanged() {
        for (provenance, environment_index) in [
            (Some(CallingConvention::Cps), None),
            (None, Some(0)),
            (Some(CallingConvention::Direct), Some(0)),
        ] {
            let (mut ctx, loc) = test_ctx();
            let (module, module_block) = make_module(&mut ctx, loc);
            let anyref_ty = make_anyref_ty(&mut ctx);
            let evidence_ty = tribute_ir::dialect::ability::evidence_adt_type_ref(&mut ctx);

            // Incomplete or conflicting provenance cannot determine hidden
            // evidence operands or the environment slot.
            let evidence_attrs = make_bind_name_attrs(&mut ctx, "evidence");
            let outer_entry = ctx.create_block(BlockData {
                location: loc,
                args: vec![BlockArgData {
                    ty: evidence_ty,
                    attrs: evidence_attrs,
                }],
                ops: Default::default(),
                parent_region: None,
            });
            let outer_evidence = ctx.block_arg(outer_entry, 0);

            let lambda_entry = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: Default::default(),
                parent_region: None,
            });
            let cast = core::UnrealizedConversionCast::operands(outer_evidence)
                .results(anyref_ty)
                .build(&mut ctx, loc);
            ctx.push_op(lambda_entry, cast.op_ref());
            let cast_result = cast.result(&ctx);
            let ret = func::Return::operands([cast_result]).build(&mut ctx, loc);
            ctx.push_op(lambda_entry, ret.op_ref());
            let lambda_body = ctx.create_region(RegionData {
                location: loc,
                blocks: trunk_ir::smallvec::smallvec![lambda_entry],
                parent_op: None,
            });

            let lambda_func_ty =
                func::func_sig(&mut ctx, std::iter::empty::<TypeRef>(), [anyref_ty]);
            let mut closure_type =
                TypeDataBuilder::new("closure.closure").param(lambda_func_ty.as_type_ref());
            if let Some(provenance) = provenance {
                closure_type = closure_type.attr(
                    tribute_ir::calling_convention::CALLING_CONVENTION_ATTR,
                    Attribute::Int(provenance as i128),
                );
            }
            if let Some(index) = environment_index {
                closure_type =
                    closure_type.attr(CLOSURE_ENVIRONMENT_INDEX_ATTR, Attribute::Int(index));
            }
            let lambda_ty = closure_type.intern(&mut ctx);
            let lambda = closure::Lambda::operands(Vec::<ValueRef>::new())
                .results(lambda_ty)
                .regions(lambda_body)
                .build(&mut ctx, loc);
            lambda.set_calling_convention(&mut ctx, CallingConvention::Cps);
            ctx.push_op(outer_entry, lambda.op_ref());

            let outer_body = ctx.create_region(RegionData {
                location: loc,
                blocks: trunk_ir::smallvec::smallvec![outer_entry],
                parent_op: None,
            });
            let outer_ty = func::func_sig(&mut ctx, [evidence_ty], [anyref_ty]).as_type_ref();
            let outer = func::Func::operands()
                .sym_name("test_fn")
                .r#type(outer_ty)
                .regions(outer_body)
                .build(&mut ctx, loc);
            ctx.push_op(module_block, outer.op_ref());

            let before = print_module(&ctx, module.op());
            lower_closure_lambda(&mut ctx, module);
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn test_lower_lambda_with_captures() {
        let (mut ctx, loc) = test_ctx();
        let (module, module_block) = make_module(&mut ctx, loc);
        let anyref_ty = make_anyref_ty(&mut ctx);
        let i32_ty = make_i32_ty(&mut ctx);

        // Build:
        //   func.func @test_fn(%a: i32) {
        //     %k = closure.lambda [%a] {
        //       ^bb0(%x: i32):
        //         %r = arith.add %a, %x
        //         func.return %r
        //     }
        //   }

        // Outer function entry block with one param %a
        let a_attrs = make_bind_name_attrs(&mut ctx, "a");
        let outer_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: a_attrs,
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let a_val = ctx.block_arg(outer_entry, 0);

        // Lambda body: ^bb0(%x: i32): %r = arith.add %a, %x; func.return %r
        let x_attrs = make_bind_name_attrs(&mut ctx, "x");
        let lambda_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: x_attrs,
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let x_val = ctx.block_arg(lambda_entry, 0);

        let add_op = arith::Addi::operands(a_val, x_val).build(&mut ctx, loc);
        ctx.push_op(lambda_entry, add_op.op_ref());
        let add_result = add_op.result(&ctx);

        let ret_op = func::Return::operands([add_result]).build(&mut ctx, loc);
        ctx.push_op(lambda_entry, ret_op.op_ref());

        let lambda_body_region = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![lambda_entry],
            parent_op: None,
        });

        let func_ty = func::func_sig(&mut ctx, [i32_ty], [i32_ty]).as_type_ref();
        let closure_ty = tribute_ir::calling_convention::physical_closure_type(
            &mut ctx,
            func_ty,
            CallingConvention::Direct,
        );

        // closure.lambda [%a] { ... }
        let lambda_op = closure::Lambda::operands(vec![a_val])
            .results(closure_ty)
            .regions(lambda_body_region)
            .build(&mut ctx, loc);
        lambda_op.set_calling_convention(&mut ctx, CallingConvention::Direct);
        ctx.push_op(outer_entry, lambda_op.op_ref());

        let outer_body = ctx.create_region(RegionData {
            location: loc,
            blocks: trunk_ir::smallvec::smallvec![outer_entry],
            parent_op: None,
        });
        let outer_func_ty = func::func_sig(&mut ctx, [i32_ty], [anyref_ty]).as_type_ref();
        let outer_func = func::Func::operands()
            .sym_name("test_fn")
            .r#type(outer_func_ty)
            .regions(outer_body)
            .build(&mut ctx, loc);
        ctx.push_op(module_block, outer_func.op_ref());

        // Run the pass.
        lower_closure_lambda(&mut ctx, module);

        // Verify: 2 functions.
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 2);

        // The original function holds the capture in its closure.new.
        let test_fn = func::Func::from_op(&ctx, ops[0]).unwrap();
        let test_fn_body = test_fn.body(&ctx);
        let test_fn_entry = ctx.region(test_fn_body).blocks[0];
        let closure_new = ctx
            .block(test_fn_entry)
            .ops
            .iter()
            .find_map(|&op| closure::New::from_op(&ctx, op).ok())
            .expect("should have closure.new");
        let capture = ctx.block_arg(test_fn_entry, 0);
        assert_eq!(closure_new.captures(&ctx), [capture]);
        let closure_ty = ctx.value_ty(closure_new.result(&ctx));

        // The lifted function receives the closure itself and unpacks it.
        let lifted = func::Func::from_op(&ctx, ops[1]).unwrap();
        let lifted_body = lifted.body(&ctx);
        let lifted_entry = ctx.region(lifted_body).blocks[0];
        let environment = ctx.block_arg(lifted_entry, 0);
        assert_eq!(ctx.value_ty(environment), closure_ty);
        let unpack = closure::Unpack::from_op(&ctx, ctx.block(lifted_entry).ops[0])
            .expect("first op should be closure.unpack");
        assert_eq!(unpack.closure(&ctx), environment);
        assert_eq!(unpack.results(&ctx).len(), 1);
        assert_eq!(ctx.value_ty(unpack.results(&ctx)[0]), i32_ty);
    }
}

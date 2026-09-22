//! Evidence runtime lowering for the native backend.
//!
//! Target evidence lowering declares the native runtime ABI and lowers
//! `effect.extend`, `effect.dispatch_tail`, and `effect.dispatch_cps` to runtime
//! calls and closure transfers. Empty evidence arrays become calls to
//! `__tribute_evidence_empty`.
//!
use std::ops::ControlFlow;

use tribute_core::{get_physical_closure_convention, set_calling_convention};
use tribute_ir::dialect::ability::{self, compute_op_idx, evidence_abi, evidence_runtime_symbols};
use tribute_ir::dialect::{effect, tribute_rt};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::dialect::{adt, arith, core};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunError, PassRunResult};
use trunk_ir::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use trunk_ir::rewrite::helpers::erase_op;
use trunk_ir::rewrite::{
    ConversionError, ConversionTarget, Module, PatternApplicator, PatternRewriter, RewritePattern,
    TypeConverter,
};
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};
use trunk_ir::walk::{WalkAction, walk_op};

fn attach_exact_indirect_signature(ctx: &mut IrContext, call: OpRef) {
    let result = ctx
        .op_result_types(call)
        .first()
        .copied()
        .unwrap_or_else(|| core::nil(ctx).as_type_ref());
    let parameters = ctx.op_operands(call)[1..]
        .iter()
        .map(|&value| ctx.value_ty(value))
        .collect::<Vec<_>>();
    let signature = func::func_sig(ctx, parameters, [result]).as_type_ref();
    let _ = func::set_indirect_call_signature(ctx, call, signature);
}

/// Lower evidence operations for the native backend.
///
/// Must run AFTER effect lowering passes and BEFORE DCE.
pub fn lower_evidence_to_native(ctx: &mut IrContext, module: Module) {
    prepare_native_evidence_runtime(ctx, module);
    rewrite_evidence_ops_in_module(ctx, module);
}

/// Prepare native evidence runtime declarations at module scope.
pub fn prepare_native_evidence_runtime(ctx: &mut IrContext, module: Module) {
    declare_evidence_runtime(ctx, module);
}

/// Lower evidence operations inside one function for the native backend.
pub fn lower_evidence_to_native_func(ctx: &mut IrContext, func_op: func::Func) {
    try_lower_evidence_to_native_func(ctx, func_op).expect("native evidence lowering failed");
}

fn try_lower_evidence_to_native_func(ctx: &mut IrContext, func_op: func::Func) -> PassRunResult {
    if is_evidence_runtime_fn(func_op.sym_name(ctx)) {
        return Ok(());
    }
    lower_effect_abi_to_native(ctx, func_op)?;
    let Some(body) = ctx.op(func_op.op_ref()).regions.first().copied() else {
        return Ok(());
    };
    rewrite_evidence_ops_in_region(ctx, body)?;
    Ok(())
}

/// PassManager-friendly native evidence lowering pass.
pub struct LowerEvidenceToNative;

impl Pass for LowerEvidenceToNative {
    type Target = func::Func;

    fn name(&self) -> &'static str {
        "lower-evidence-to-native"
    }

    fn run(&mut self, ctx: &mut IrContext, target: func::Func) -> PassRunResult {
        try_lower_evidence_to_native_func(ctx, target)
    }
}

// =============================================================================
// Native runtime declarations
// =============================================================================

fn declare_evidence_runtime(ctx: &mut IrContext, module: Module) {
    let Some(block) = module.first_block(ctx) else {
        return;
    };
    let loc = ctx.op(module.op()).location;
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    let i32_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build());
    for (name, params, result) in [
        (evidence_abi::EMPTY, &[][..], ptr_ty),
        (evidence_abi::LOOKUP, &[ptr_ty, i32_ty][..], i32_ty),
        (
            evidence_abi::EXTEND,
            &[ptr_ty, i32_ty, i32_ty, ptr_ty, ptr_ty][..],
            ptr_ty,
        ),
        (evidence_abi::LOOKUP_TR, &[ptr_ty, i32_ty][..], ptr_ty),
        (evidence_abi::LOOKUP_HANDLER, &[ptr_ty, i32_ty][..], ptr_ty),
    ] {
        if module.ops(ctx).into_iter().any(|op| {
            func::Func::from_op(ctx, op)
                .is_ok_and(|function| function.sym_name(ctx) == Symbol::new(name))
        }) {
            continue;
        }
        let declaration = super::build_extern_func(ctx, loc, name, params, result);
        if let Some(&first) = ctx.block(block).ops.first() {
            ctx.insert_op_before(block, first, declaration);
        } else {
            ctx.push_op(block, declaration);
        }
    }
}

// =============================================================================
// Phase 2: Rewrite evidence ops inside function bodies
// =============================================================================

fn is_evidence_runtime_fn(name: Symbol) -> bool {
    evidence_runtime_symbols().contains(&name)
}

fn rewrite_evidence_ops_in_module(ctx: &mut IrContext, module: Module) {
    let mut funcs = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        if let Ok(func_op) = func::Func::from_op(ctx, op) {
            funcs.push(func_op);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });

    for func_op in funcs {
        lower_evidence_to_native_func(ctx, func_op);
    }
}

fn native_effect_abi_target() -> ConversionTarget {
    ConversionTarget::new()
        .legal_op("func", "func")
        .recursive_legal_op("func", "func")
        .illegal_op("effect", "extend")
        .illegal_op("effect", "dispatch_tail")
        .illegal_op("effect", "dispatch_cps")
}

fn lower_effect_abi_to_native(
    ctx: &mut IrContext,
    func_op: func::Func,
) -> Result<(), ConversionError> {
    PatternApplicator::new(TypeConverter::new())
        .with_target(native_effect_abi_target())
        .add_pattern(LowerEffectExtendToNative)
        .add_pattern(LowerEffectDispatchTailToNative)
        .add_pattern(LowerEffectDispatchCpsToNative)
        .apply_partial_conversion(ctx, func_op, "native-evidence-effect-abi")?;
    Ok(())
}

#[derive(Debug)]
struct NativeEvidenceRewriteError {
    op: OpRef,
    loc: Location,
    message: String,
}

impl std::fmt::Display for NativeEvidenceRewriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "native evidence rewrite failed at op {:?} (loc {:?}): {}",
            self.op, self.loc, self.message
        )
    }
}

impl std::error::Error for NativeEvidenceRewriteError {}

fn native_evidence_rewrite_error(
    op: OpRef,
    loc: Location,
    message: impl Into<String>,
) -> PassRunError {
    Box::new(NativeEvidenceRewriteError {
        op,
        loc,
        message: message.into(),
    })
}

fn rewrite_evidence_ops_in_region(ctx: &mut IrContext, region: RegionRef) -> PassRunResult {
    let blocks: Vec<BlockRef> = ctx.region(region).blocks.to_vec();
    for block in blocks {
        rewrite_evidence_ops_in_block(ctx, block)?;
    }
    Ok(())
}

/// Check if a type is an evidence type in arena (adt.array<ability.evidence>).
fn is_evidence_type(ctx: &IrContext, ty: TypeRef) -> bool {
    tribute_ir::dialect::ability::is_evidence_type_ref(ctx, ty)
}

fn op_idx_const(
    ctx: &mut IrContext,
    loc: Location,
    i32_ty: TypeRef,
    ability_ref: TypeRef,
    op_name: Symbol,
) -> arith::Const {
    let op_idx = compute_op_idx(ability::ability_name(ctx, ability_ref), Some(op_name));
    arith::r#const(ctx, loc, i32_ty, Attribute::Int(op_idx as i128))
}

fn core_ptr_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build())
}

fn core_i32_type(ctx: &mut IrContext) -> TypeRef {
    ctx.types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
}

fn lower_evidence_dispatch_operand(
    ctx: &mut IrContext,
    loc: Location,
    value: ValueRef,
    ptr_ty: TypeRef,
) -> OpRef {
    if crate::closure_lower::is_closure_struct_type_ref(ctx, ctx.value_ty(value))
        || get_physical_closure_convention(ctx, ctx.value_ty(value)).is_some()
    {
        tribute_rt::into_raw(ctx, loc, value, ptr_ty).op_ref()
    } else {
        core::unrealized_conversion_cast(ctx, loc, value, ptr_ty).op_ref()
    }
}

struct LowerEffectExtendToNative;

impl RewritePattern for LowerEffectExtendToNative {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(extend_op) = effect::Extend::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ptr_ty = core_ptr_type(ctx);
        let i32_ty = core_i32_type(ctx);

        let ability_id_op = ability::ability_id_const(ctx, loc, i32_ty, extend_op.ability_ref(ctx));
        let ability_id_val = ability_id_op.result(ctx);
        rewriter.insert_op(ability_id_op.op_ref());

        let tr_dispatch_ptr =
            lower_evidence_dispatch_operand(ctx, loc, extend_op.tr_dispatch_fn(ctx), ptr_ty);
        let tr_dispatch = ctx.op_result(tr_dispatch_ptr, 0);
        rewriter.insert_op(tr_dispatch_ptr);

        let handler_dispatch_ptr =
            lower_evidence_dispatch_operand(ctx, loc, extend_op.handler_dispatch(ctx), ptr_ty);
        let handler_dispatch = ctx.op_result(handler_dispatch_ptr, 0);
        rewriter.insert_op(handler_dispatch_ptr);

        let mut operands = vec![
            extend_op.evidence(ctx),
            ability_id_val,
            extend_op.prompt_tag(ctx),
        ];
        operands.push(tr_dispatch);
        operands.push(handler_dispatch);

        let extend_call = func::call(
            ctx,
            loc,
            operands,
            [ptr_ty],
            Symbol::new(evidence_abi::EXTEND),
        );
        let new_result = extend_call.result(ctx);
        rewriter.insert_op(extend_call.op_ref());
        rewriter.erase_op(vec![new_result]);
        true
    }
}

struct LowerEffectDispatchTailToNative;

impl RewritePattern for LowerEffectDispatchTailToNative {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchTail::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        let ptr_ty = core_ptr_type(ctx);
        let i32_ty = core_i32_type(ctx);
        let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
        let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
        let ability_ref = dispatch_op.ability_ref(ctx);

        let ability_id_op = ability::ability_id_const(ctx, loc, i32_ty, ability_ref);
        let ability_id_val = ability_id_op.result(ctx);
        rewriter.insert_op(ability_id_op.op_ref());

        let dispatch_closure = func::call(
            ctx,
            loc,
            [dispatch_op.evidence(ctx), ability_id_val],
            [ptr_ty],
            Symbol::new(evidence_abi::LOOKUP_TR),
        );
        let dispatch_val = dispatch_closure.result(ctx);
        rewriter.insert_op(dispatch_closure.op_ref());

        let op_idx_op = op_idx_const(ctx, loc, i32_ty, ability_ref, dispatch_op.op_name(ctx));
        let op_idx_val = op_idx_op.result(ctx);
        rewriter.insert_op(op_idx_op.op_ref());

        let fn_ptr_get = adt::struct_get(ctx, loc, dispatch_val, i32_ty, closure_ty, 0);
        let fn_ptr = fn_ptr_get.result(ctx);
        rewriter.insert_op(fn_ptr_get.op_ref());

        let env_get = adt::struct_get(ctx, loc, dispatch_val, anyref_ty, closure_ty, 1);
        let env_val = env_get.result(ctx);
        rewriter.insert_op(env_get.op_ref());

        let result_ty = ctx.op_result_types(op)[0];
        let call = func::call_indirect(
            ctx,
            loc,
            fn_ptr,
            [
                dispatch_op.evidence(ctx),
                env_val,
                op_idx_val,
                dispatch_op.payload(ctx),
            ],
            [result_ty],
            None,
        );
        attach_exact_indirect_signature(ctx, call.op_ref());
        let new_result = call.result(ctx);
        rewriter.insert_op(call.op_ref());
        rewriter.erase_op(vec![new_result]);
        true
    }
}

struct LowerEffectDispatchCpsToNative;

impl RewritePattern for LowerEffectDispatchCpsToNative {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(dispatch_op) = effect::DispatchCps::from_op(ctx, op) else {
            return false;
        };

        let loc = ctx.op(op).location;
        // A result-bearing final dispatch is malformed. Reject it before any
        // helper op is inserted so partial conversion leaves the IR unchanged.
        if !ctx.op_result_types(op).is_empty()
            || ctx.op(op).attributes.get_type("answer_type").is_none()
            || ctx.op_operands(op).len() != 4
        {
            return false;
        }
        let i32_ty = core_i32_type(ctx);
        let anyref_ty = tribute_rt::anyref(ctx).as_type_ref();
        let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
        let ability_ref = dispatch_op.ability_ref(ctx);
        let evidence_ty = ability::evidence_adt_type_ref(ctx);
        let signature = func::func_sig(
            ctx,
            [
                evidence_ty,
                anyref_ty,
                closure_ty,
                i32_ty,
                i32_ty,
                i32_ty,
                anyref_ty,
            ],
            [],
        )
        .as_type_ref();
        let (converter, _) = super::type_converter::native_type_converter(ctx);
        let expected = [evidence_ty, closure_ty, closure_ty, anyref_ty];
        for (value, expected) in ctx.op_operands(op).to_vec().into_iter().zip(expected) {
            if converter.convert_type_or_identity(ctx, ctx.value_ty(value))
                != converter.convert_type_or_identity(ctx, expected)
            {
                return false;
            }
        }

        let ability_id_op = ability::ability_id_const(ctx, loc, i32_ty, ability_ref);
        let ability_id_val = ability_id_op.result(ctx);
        rewriter.insert_op(ability_id_op.op_ref());

        let prompt = func::call(
            ctx,
            loc,
            [dispatch_op.evidence(ctx), ability_id_val],
            [i32_ty],
            Symbol::new(evidence_abi::LOOKUP),
        );
        let prompt_val = prompt.result(ctx);
        rewriter.insert_op(prompt.op_ref());

        let op_idx_op = op_idx_const(ctx, loc, i32_ty, ability_ref, dispatch_op.op_name(ctx));
        let op_idx_val = op_idx_op.result(ctx);
        rewriter.insert_op(op_idx_op.op_ref());

        let fn_ptr_get =
            adt::struct_get(ctx, loc, dispatch_op.dispatch(ctx), i32_ty, closure_ty, 0);
        let fn_ptr = fn_ptr_get.result(ctx);
        rewriter.insert_op(fn_ptr_get.op_ref());

        let env_get = adt::struct_get(
            ctx,
            loc,
            dispatch_op.dispatch(ctx),
            anyref_ty,
            closure_ty,
            1,
        );
        let env_val = env_get.result(ctx);
        rewriter.insert_op(env_get.op_ref());

        let tail = func::tail_call_indirect(
            ctx,
            loc,
            fn_ptr,
            [
                dispatch_op.evidence(ctx),
                env_val,
                dispatch_op.resume(ctx),
                prompt_val,
                ability_id_val,
                op_idx_val,
                dispatch_op.payload(ctx),
            ],
            Some(signature),
        );
        set_calling_convention(ctx, tail.op_ref(), tribute_core::CallingConvention::Cps);
        rewriter.replace_op(tail.op_ref());
        true
    }
}

fn rewrite_evidence_ops_in_block(ctx: &mut IrContext, block: BlockRef) -> PassRunResult {
    let ptr_ty = ctx
        .types
        .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ptr")).build());
    // Ops to erase after processing
    let mut ops_to_erase: Vec<OpRef> = Vec::new();

    let ops: Vec<OpRef> = ctx.block(block).ops.to_vec();

    for op in ops {
        let op_data = ctx.op(op);
        let dialect = op_data.dialect;
        let name = op_data.name;
        let loc = op_data.location;

        // --- adt.ref_null with evidence type → func.call @__tribute_evidence_empty ---
        // The root CPS bridge creates `adt.ref_null {type = evidence}` for empty evidence.
        // Without this, the null ptr gets unboxed via `clif.load` which dereferences null.
        if dialect == Symbol::new("adt") && name == Symbol::new("ref_null") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_evidence_type(ctx, result_types[0]) {
                let old_result = ctx.op_result(op, 0);
                let call = func::call(ctx, loc, [], [ptr_ty], Symbol::new(evidence_abi::EMPTY));
                let new_result = call.result(ctx);
                ctx.insert_op_before(block, op, call.op_ref());
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- adt.array_new with evidence type → func.call @__tribute_evidence_empty ---
        if dialect == Symbol::new("adt") && name == Symbol::new("array_new") {
            let result_types = ctx.op_result_types(op).to_vec();
            if !result_types.is_empty() && is_evidence_type(ctx, result_types[0]) {
                // Evidence array_new must represent an empty evidence vector.
                // The only operand should be the size hint (arith.const 0).
                let operand_count = ctx.op_operands(op).len();
                if operand_count > 1 {
                    return Err(native_evidence_rewrite_error(
                        op,
                        loc,
                        format!(
                            "adt.array_new with evidence type has {operand_count} operands; \
                             expected at most 1 (the size hint). Non-empty evidence arrays \
                             should not reach this pass."
                        ),
                    ));
                }
                let old_result = ctx.op_result(op, 0);
                let call = func::call(ctx, loc, [], [ptr_ty], Symbol::new(evidence_abi::EMPTY));
                let new_result = call.result(ctx);
                ctx.insert_op_before(block, op, call.op_ref());
                ctx.replace_all_uses(old_result, new_result);
                ops_to_erase.push(op);
                continue;
            }
        }

        // --- Recurse into nested regions, but leave nested functions for their
        // own function-scoped pass invocation.
        if func::Func::from_op(ctx, op).is_ok() {
            continue;
        }
        let regions: Vec<RegionRef> = ctx.op(op).regions.to_vec();
        for region in regions {
            rewrite_evidence_ops_in_region(ctx, region)?;
        }
    }

    // Erase dead ops (in reverse to handle dependencies)
    for op in ops_to_erase.into_iter().rev() {
        erase_op(ctx, op);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::ControlFlow;
    use trunk_ir::op_interface::IndirectCallLikeModel;
    use trunk_ir::ops::DialectType;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::walk::{WalkAction, walk_op};

    fn dispatch_module() -> &'static str {
        r#"core.module @test {
  func.func @selected(%ev: core.ptr, %payload: tribute_rt.anyref) -> core.ptr {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : core.ptr
    func.return %result
  }
  func.func @untouched(%ev: core.ptr, %payload: tribute_rt.anyref) -> core.ptr {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @print} : core.ptr
    func.return %result
  }
}"#
    }

    fn func_by_name_recursive(ctx: &IrContext, module: Module, name: &'static str) -> func::Func {
        let name = Symbol::new(name);
        let mut found = None;
        let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
            if let Ok(func_op) = func::Func::from_op(ctx, op)
                && func_op.sym_name(ctx) == name
            {
                found = Some(func_op);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        found.expect("test function should exist")
    }

    fn call_indirect_operands_in_func(ctx: &IrContext, func_op: func::Func) -> Vec<Vec<ValueRef>> {
        let mut calls = Vec::new();
        for &block in &ctx.region(func_op.body(ctx)).blocks {
            for &op in &ctx.block(block).ops {
                if func::CallIndirect::from_op(ctx, op).is_ok() {
                    calls.push(ctx.op_operands(op).to_vec());
                }
            }
        }
        calls
    }

    fn entry_arg(ctx: &IrContext, func_op: func::Func, index: usize) -> ValueRef {
        let entry = ctx.region(func_op.body(ctx)).blocks[0];
        ctx.block_args(entry)[index]
    }

    #[test]
    fn runtime_declarations_are_created_without_shared_stubs_and_are_idempotent() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            "core.module @test { func.func @user() -> core.i32 }",
        );
        prepare_native_evidence_runtime(&mut ctx, module);
        assert_eq!(module.ops(&ctx).len(), 6);
        for (name, params, result) in [
            (evidence_abi::EMPTY, &[][..], "core.ptr"),
            (
                evidence_abi::LOOKUP,
                &["core.ptr", "core.i32"][..],
                "core.i32",
            ),
            (
                evidence_abi::EXTEND,
                &["core.ptr", "core.i32", "core.i32", "core.ptr", "core.ptr"][..],
                "core.ptr",
            ),
            (
                evidence_abi::LOOKUP_TR,
                &["core.ptr", "core.i32"][..],
                "core.ptr",
            ),
            (
                evidence_abi::LOOKUP_HANDLER,
                &["core.ptr", "core.i32"][..],
                "core.ptr",
            ),
        ] {
            let function = func_by_name_recursive(&ctx, module, name);
            assert!(ctx.op(function.op_ref()).regions.is_empty());
            assert_eq!(
                ctx.op(function.op_ref()).attributes.get_str("abi"),
                Some("C")
            );
            let signature = func::FuncSig::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
            let parameter_types: Vec<_> = signature
                .inputs(&ctx)
                .iter()
                .map(|&ty| trunk_ir::printer::print_type(&ctx, ty))
                .collect();
            assert_eq!(parameter_types, params, "{name}");
            assert_eq!(
                trunk_ir::printer::print_type(&ctx, signature.single_result(&ctx).unwrap()),
                result,
                "{name}"
            );
        }
        let before = print_module(&ctx, module.op());
        let ops = module.ops(&ctx);
        prepare_native_evidence_runtime(&mut ctx, module);
        assert_eq!(print_module(&ctx, module.op()), before);
        assert_eq!(module.ops(&ctx), ops);
    }

    #[test]
    fn function_scope_rewrites_only_selected_function() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, dispatch_module());
        let selected = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| func::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a selected function");

        lower_evidence_to_native_func(&mut ctx, selected);

        let ir_text = print_module(&ctx, module.op());
        assert_eq!(ir_text.matches("effect.dispatch_tail").count(), 1);
        assert!(ir_text.contains("func.func @untouched"));
        assert!(ir_text.contains("op_name = @print"));
        assert!(ir_text.contains("__tribute_evidence_lookup_tr"));
    }

    #[test]
    fn function_scope_preserves_bodyless_declaration_and_rewrites_body() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !marker = adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker}
  !evidence = core.array(!marker)
  func.func @external(%ev: !evidence) -> !marker
  func.func @selected(%ev: core.ptr, %payload: tribute_rt.anyref) -> core.ptr {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : core.ptr
    func.return %result
  }
}"#,
        );
        let external = func_by_name_recursive(&ctx, module, "external");
        assert!(ctx.op(external.op_ref()).regions.is_empty());
        let before = print_module(&ctx, module.op());

        lower_evidence_to_native_func(&mut ctx, external);

        let external_after = func_by_name_recursive(&ctx, module, "external");
        assert!(ctx.op(external_after.op_ref()).regions.is_empty());
        assert_eq!(print_module(&ctx, module.op()), before);

        let selected = func_by_name_recursive(&ctx, module, "selected");
        lower_evidence_to_native_func(&mut ctx, selected);

        let after = print_module(&ctx, module.op());
        assert!(after.contains("func.func @external(%arg0: !evidence) -> !marker\n"));
        assert!(
            !after.contains("effect.dispatch_tail"),
            "body-bearing function was not transformed:\n{after}"
        );
        assert!(after.contains("__tribute_evidence_lookup_tr"));
    }

    #[test]
    fn module_entrypoint_prepares_runtime_and_rewrites_all_functions() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, dispatch_module());

        lower_evidence_to_native(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert!(!ir_text.contains("effect.dispatch_tail"));
        assert!(ir_text.contains("__tribute_evidence_empty"));
        assert!(ir_text.contains("__tribute_evidence_lookup_tr"));
        assert!(ir_text.contains("__tribute_evidence_lookup_handler"));
    }

    #[test]
    fn pass_adapter_runs_function_lowering() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, dispatch_module());
        let selected = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| func::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a selected function");
        let mut pass = LowerEvidenceToNative;

        assert_eq!(pass.name(), "lower-evidence-to-native");
        pass.run(&mut ctx, selected).unwrap();

        let ir_text = print_module(&ctx, module.op());
        assert_eq!(ir_text.matches("effect.dispatch_tail").count(), 1);
        assert!(ir_text.contains("__tribute_evidence_lookup"));
    }

    #[test]
    fn function_scope_skips_evidence_runtime_functions() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @__tribute_evidence_empty(%ev: core.ptr, %payload: tribute_rt.anyref) -> core.ptr {
    %result = effect.dispatch_tail %ev, %payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : core.ptr
    func.return %result
  }
}"#,
        );
        let runtime_func = module
            .ops(&ctx)
            .into_iter()
            .filter_map(|op| func::Func::from_op(&ctx, op).ok())
            .next()
            .expect("test module should contain a runtime function");

        lower_evidence_to_native_func(&mut ctx, runtime_func);

        let ir_text = print_module(&ctx, module.op());
        assert!(ir_text.contains("func.func @__tribute_evidence_empty"));
        assert!(ir_text.contains("effect.dispatch_tail"));
    }

    #[test]
    fn function_scope_leaves_nested_func_for_own_lowering() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @outer(%outer_ev: core.ptr, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
    func.func @inner(%inner_ev: core.ptr, %inner_payload: tribute_rt.anyref) -> core.ptr {
      %result = effect.dispatch_tail %inner_ev, %inner_payload {ability_ref = core.ability_ref() {name = @Console}, op_name = @read} : core.ptr
      func.return %result
    }
    func.return %payload
  }
}"#,
        );
        let outer = func_by_name_recursive(&ctx, module, "outer");

        lower_evidence_to_native_func(&mut ctx, outer);

        let inner = func_by_name_recursive(&ctx, module, "inner");
        let inner_after_outer = print_module(&ctx, inner.op_ref());
        assert!(
            inner_after_outer.contains("effect.dispatch_tail"),
            "outer function pass should not lower nested function body:\n{inner_after_outer}"
        );

        lower_evidence_to_native_func(&mut ctx, inner);

        let inner_calls = call_indirect_operands_in_func(&ctx, inner);
        assert_eq!(inner_calls.len(), 1);
        assert_eq!(
            inner_calls[0][1],
            entry_arg(&ctx, inner, 0),
            "nested native evidence lowering should pass the nested function's own evidence"
        );
        let inner_call = ctx
            .region(inner.body(&ctx))
            .blocks
            .iter()
            .flat_map(|&block| ctx.block(block).ops.iter().copied())
            .find(|&op| func::CallIndirect::matches(&ctx, op))
            .expect("lowered indirect dispatch");
        assert!(
            trunk_ir::op_interface::IndirectCallLikeOps::exact_signature(&ctx, inner_call)
                .is_some(),
            "native-generated indirect calls must carry an exact signature"
        );
    }

    #[test]
    fn result_bearing_final_dispatch_fails_before_native_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%ev: core.ptr, %dispatch: tribute_rt.anyref, %resume: tribute_rt.anyref, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
    %result = effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref() {name = @State}, op_name = @get} : tribute_rt.anyref
    func.return %result
  }
}"#,
        );
        let run = func_by_name_recursive(&ctx, module, "run");
        let before = print_module(&ctx, module.op());

        let error = try_lower_evidence_to_native_func(&mut ctx, run)
            .expect_err("result-bearing final dispatch must remain illegal");

        assert!(error.to_string().contains("effect.dispatch_cps"), "{error}");
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn final_dispatch_uses_one_canonical_signature_for_distinct_answers() {
        let mut ctx = IrContext::new();
        let source = r#"core.module @test {
          func.func @first(%ev: core.ptr, %dispatch: tribute_rt.anyref, %resume: tribute_rt.anyref, %payload: tribute_rt.anyref) {
            effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref() {name = @State}, op_name = @get, answer_type = core.i32}
          }
          func.func @second(%ev: core.ptr, %dispatch: tribute_rt.anyref, %resume: tribute_rt.anyref, %payload: tribute_rt.anyref) {
            effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref() {name = @State}, op_name = @get, answer_type = core.i64}
          }
        }"#;
        let module = parse_test_module(&mut ctx, source);
        let mut signatures = Vec::new();
        for name in ["first", "second"] {
            let function = func_by_name_recursive(&ctx, module, name);
            lower_evidence_to_native_func(&mut ctx, function);
            let block = ctx.region(function.body(&ctx)).blocks[0];
            let tail = *ctx.block(block).ops.last().unwrap();
            assert!(!ctx.op(tail).attributes.contains_key("answer_type"));
            signatures.push(
                func::TailCallIndirect::from_op(&ctx, tail)
                    .unwrap()
                    .exact_signature(&ctx)
                    .unwrap(),
            );
        }
        assert_eq!(signatures[0], signatures[1]);
        let signature = func::FuncSig::from_type_ref(&ctx, signatures[0]).unwrap();
        assert!(signature.results(&ctx).is_empty());
        let (converter, _) = super::super::type_converter::native_type_converter(&mut ctx);
        let actual: Vec<_> = signature
            .inputs(&ctx)
            .iter()
            .map(|ty| converter.convert_type_or_identity(&ctx, *ty))
            .collect();
        let ptr = core::ptr(&mut ctx).as_type_ref();
        let i32_ty = core_i32_type(&mut ctx);
        assert_eq!(actual, [ptr, ptr, ptr, i32_ty, i32_ty, i32_ty, ptr]);
        for index in 0..4 {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, source);
            let function = func_by_name_recursive(&ctx, module, "first");
            let entry = ctx.region(function.body(&ctx)).blocks[0];
            let wrong = core_i32_type(&mut ctx);
            ctx.set_block_arg_type(entry, index, wrong);
            let before = print_module(&ctx, module.op());
            assert!(try_lower_evidence_to_native_func(&mut ctx, function).is_err());
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn resultless_final_dispatch_lowers_to_a_native_proper_tail() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%ev: core.ptr, %dispatch: tribute_rt.anyref, %resume: tribute_rt.anyref, %payload: tribute_rt.anyref) -> core.never {
    effect.dispatch_cps %ev, %dispatch, %resume, %payload {ability_ref = core.ability_ref() {name = @State}, op_name = @get, answer_type = core.i32}
  }
}"#,
        );
        let run = func_by_name_recursive(&ctx, module, "run");

        lower_evidence_to_native_func(&mut ctx, run);

        let output = print_module(&ctx, module.op());
        assert!(!output.contains("effect.dispatch_cps"), "{output}");
        assert!(
            output.contains("adt.struct_get %1"),
            "the native tail must decompose the explicit dispatch operand: {output}"
        );
        assert!(
            !output.contains("core.unrealized_conversion_cast"),
            "the native tail must not cast the explicit dispatch operand: {output}"
        );
        assert!(output.contains("func.tail_call_indirect"), "{output}");
        assert!(output.contains("signature"), "{output}");
        assert!(
            output.contains("tribute.calling_convention = 2"),
            "{output}"
        );
    }
}

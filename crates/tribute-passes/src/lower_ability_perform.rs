//! Lower `ability.perform` operations to handler_dispatch calls.
//!
//! In CPS-based effect handling, `ability.perform` carries an explicit
//! continuation closure. This pass converts it to:
//!
//! ```text
//! // Input:
//! %yr = ability.perform %continuation, [%args...]
//!   { ability_ref: @State, op_name: @get }
//!
//! // Output:
//! %marker = ability.evidence_lookup %evidence { ability_ref: @State }
//! %handler = adt.struct_get %marker, 3       // handler_dispatch closure
//! %fn_ptr = adt.struct_get %handler, 0       // closure function pointer
//! %env = adt.struct_get %handler, 1          // closure environment
//! %op_idx = arith.const <hash(ability, op)>
//! %shift_value = cast %args to anyref (or null)
//! %cont = cast %continuation to anyref
//! %result = func.call_indirect %fn_ptr, (%ev, %env, %cont, %op_idx, %shift_value)
//! func.return %result
//! ```
//!
//! Uses `PatternApplicator` for declarative op-level rewriting.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{adt, arith, core, func};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::Attribute;

use tribute_ir::dialect::ability;
use tribute_ir::dialect::ability::compute_op_idx;
use tribute_ir::dialect::tribute_rt;

/// Cached common type references used by the perform lowering pattern.
#[derive(Clone, Copy)]
struct CommonTypes {
    anyref: TypeRef,
    i32: TypeRef,
}

impl CommonTypes {
    fn new(ctx: &mut IrContext) -> Self {
        Self {
            anyref: tribute_rt::anyref(ctx).as_type_ref(),
            i32: ctx.types.intern(
                trunk_ir::types::TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32"))
                    .build(),
            ),
        }
    }
}

/// Lower all `ability.perform` ops in the module.
pub fn lower_ability_perform(ctx: &mut IrContext, module: Module) {
    let types = CommonTypes::new(ctx);
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(LowerPerformPattern { types });
    applicator.apply_partial(ctx, module);
}

/// Pattern: `ability.perform` → evidence lookup + handler_dispatch call + return.
struct LowerPerformPattern {
    types: CommonTypes,
}

impl RewritePattern for LowerPerformPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(perform_op) = ability::Perform::from_op(ctx, op) else {
            return false;
        };

        let location = ctx.op(op).location;
        let ability_ref_type = perform_op.ability_ref(ctx);
        let op_name_sym = perform_op.op_name(ctx);

        // Operands: [continuation, ...values]
        let operands: Vec<ValueRef> = ctx.op_operands(op).to_vec();
        let continuation_val = operands[0];
        let value_operands = &operands[1..];

        let t = &self.types;

        // Find evidence parameter from enclosing func's entry block.
        let evidence_val = find_evidence_from_op(ctx, op);

        // === 1. Evidence lookup → marker ===
        let marker_ty = ability::marker_adt_type_ref(ctx);
        let evidence_val = evidence_val.expect(
            "ability.perform requires evidence parameter; \
             enclosing function is missing evidence (was add_evidence_params skipped?)",
        );
        let lookup =
            ability::evidence_lookup(ctx, location, evidence_val, marker_ty, ability_ref_type);
        rewriter.insert_op(lookup.op_ref());
        let marker_val = lookup.result(ctx);

        // === 2. Extract handler_dispatch from marker (field index 3) ===
        // The Marker ADT declares this field as core.ptr, but we read it with
        // closure_ty because evidence_to_native replaces this struct_get entirely
        // with a __tribute_evidence_lookup_handler call before type conversion.
        let closure_ty = crate::closure_lower::closure_struct_type_ref(ctx);
        let handler_get = adt::struct_get(ctx, location, marker_val, closure_ty, marker_ty, 3);
        rewriter.insert_op(handler_get.op_ref());
        let handler_dispatch_val = handler_get.result(ctx);

        // === 3. Compute op_idx ===
        let ability_data = ctx.types.get(ability_ref_type);
        let ability_name = match ability_data.attrs.get(&Symbol::new("name")) {
            Some(Attribute::Symbol(s)) => Some(*s),
            _ => None,
        };
        let op_idx = compute_op_idx(ability_name, Some(op_name_sym));
        let op_idx_const = arith::r#const(ctx, location, t.i32, Attribute::Int(op_idx as i128));
        rewriter.insert_op(op_idx_const.op_ref());

        // === 4. Build shift value (pack args or null) ===
        assert!(
            value_operands.len() <= 1,
            "ability.perform expects at most 1 value operand (multi-arg should be tuple-packed), \
             got {}",
            value_operands.len()
        );
        let shift_value_val = if let Some(&sv) = value_operands.first() {
            let cast = core::unrealized_conversion_cast(ctx, location, sv, t.anyref);
            rewriter.insert_op(cast.op_ref());
            cast.result(ctx)
        } else {
            let null_op = adt::ref_null(ctx, location, t.anyref, t.anyref);
            rewriter.insert_op(null_op.op_ref());
            null_op.result(ctx)
        };

        // === 5. Cast continuation closure to anyref ===
        let cont_anyref =
            core::unrealized_conversion_cast(ctx, location, continuation_val, t.anyref);
        rewriter.insert_op(cont_anyref.op_ref());

        // === 6. Decompose handler_dispatch closure and call ===
        let fn_ptr_get = adt::struct_get(ctx, location, handler_dispatch_val, t.i32, closure_ty, 0);
        rewriter.insert_op(fn_ptr_get.op_ref());
        let fn_ptr = fn_ptr_get.result(ctx);

        let env_get = adt::struct_get(ctx, location, handler_dispatch_val, t.anyref, closure_ty, 1);
        rewriter.insert_op(env_get.op_ref());
        let env_val = env_get.result(ctx);

        // Placeholder evidence for resolve_evidence to replace later.
        let placeholder_ev = evidence_val;

        let call_op = func::call_indirect(
            ctx,
            location,
            fn_ptr,
            vec![
                placeholder_ev,
                env_val,
                cont_anyref.result(ctx),
                op_idx_const.result(ctx),
                shift_value_val,
            ],
            t.anyref,
        );
        rewriter.insert_op(call_op.op_ref());

        // === 7. Return the handler's result (cast to function return type) ===
        let block = ctx.op(op).parent_block.unwrap();
        if is_in_func_body(ctx, block) {
            let result_val = call_op.result(ctx);
            let func_ret_ty = find_enclosing_func_return_type(ctx, block);
            let ret_op = if let Some(ret_ty) = func_ret_ty {
                if is_nil_type(ctx, ret_ty) {
                    // Nil return type = void in Cranelift; return with no args.
                    func::r#return(ctx, location, std::iter::empty::<ValueRef>())
                } else if ret_ty != t.anyref {
                    let cast = core::unrealized_conversion_cast(ctx, location, result_val, ret_ty);
                    rewriter.insert_op(cast.op_ref());
                    func::r#return(ctx, location, [cast.result(ctx)])
                } else {
                    func::r#return(ctx, location, [result_val])
                }
            } else {
                func::r#return(ctx, location, [result_val])
            };
            rewriter.insert_op(ret_op.op_ref());
        }

        // Erase perform op, mapping its result to the call_indirect result.
        rewriter.erase_op(vec![call_op.result(ctx)]);

        // Remove dead code after the perform op in the same block.
        // Only do this when we inserted a func.return terminator (i.e., in
        // func body); in other regions (scf.if, etc.) the existing terminator
        // (scf.yield) must be preserved.
        if is_in_func_body(ctx, block) {
            let dead_ops: Vec<OpRef> = {
                let ops = &ctx.block(block).ops;
                let Some(idx) = ops.iter().position(|&o| o == op) else {
                    return true;
                };
                ops[idx + 1..].to_vec()
            };
            for dead_op in dead_ops {
                ctx.remove_op_from_block(block, dead_op);
            }
        }

        true
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Check if a TypeRef is `core.nil` (void return type in Cranelift).
fn is_nil_type(ctx: &IrContext, ty: trunk_ir::TypeRef) -> bool {
    let td = ctx.types.get(ty);
    td.dialect == Symbol::new("core") && td.name == Symbol::new("nil")
}

/// Find the return type of the enclosing `func.func`.
fn find_enclosing_func_return_type(ctx: &IrContext, block: BlockRef) -> Option<trunk_ir::TypeRef> {
    let region = ctx.block(block).parent_region?;
    let parent_op = ctx.region(region).parent_op?;
    let func_op = func::Func::from_op(ctx, parent_op).ok()?;
    let func_ty = func_op.r#type(ctx);
    let td = ctx.types.get(func_ty);
    if td.dialect == Symbol::new("core") && td.name == Symbol::new("func") {
        td.params.first().copied()
    } else {
        None
    }
}

/// Check if a block belongs directly to a `func.func` body region.
fn is_in_func_body(ctx: &IrContext, block: BlockRef) -> bool {
    let Some(region) = ctx.block(block).parent_region else {
        return false;
    };
    let Some(parent_op) = ctx.region(region).parent_op else {
        return false;
    };
    func::Func::matches(ctx, parent_op)
}

/// Find the evidence parameter by walking up from the op to its enclosing func.
fn find_evidence_from_op(ctx: &IrContext, op: OpRef) -> Option<ValueRef> {
    let mut current_op = op;
    loop {
        let block = ctx.op(current_op).parent_block?;
        let region = ctx.block(block).parent_region?;
        let parent = ctx.region(region).parent_op?;
        if func::Func::matches(ctx, parent) {
            // Found the enclosing func — check entry block args.
            let func_body = func::Func::from_op(ctx, parent).ok()?.body(ctx);
            let entry = ctx.region(func_body).blocks[0];
            return ctx
                .block_args(entry)
                .iter()
                .find(|&&arg| ability::is_evidence_type_ref(ctx, ctx.value_ty(arg)))
                .copied();
        }
        current_op = parent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use insta::assert_snapshot;
    use trunk_ir::context::IrContext;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    /// Initialize common types used by the pass.
    fn init_common_types(ctx: &mut IrContext) {
        CommonTypes::new(ctx);
    }

    /// Build the canonical evidence type string for use in test IR.
    fn evidence_type_str() -> &'static str {
        "core.array(adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker})"
    }

    #[test]
    fn test_lower_perform_basic() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();

        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @test_fn(%ev: {ev_ty}) -> tribute_rt.anyref {{
    %k = arith.const {{value = 0}} : tribute_rt.anyref
    %yr = ability.perform %k {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @get}} : tribute_rt.anyref
    func.return %yr
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert_snapshot!(ir_text);
    }

    #[test]
    fn test_lower_perform_with_args() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);
        let ev_ty = evidence_type_str();

        let module = parse_test_module(
            &mut ctx,
            &format!(
                r#"core.module @test {{
  func.func @test_fn(%ev: {ev_ty}) -> tribute_rt.anyref {{
    %val = arith.const {{value = 42}} : core.i32
    %k = arith.const {{value = 0}} : tribute_rt.anyref
    %yr = ability.perform %k, %val {{ability_ref = core.ability_ref() {{name = @State}}, op_name = @set}} : tribute_rt.anyref
    func.return %yr
  }}
}}"#
            ),
        );

        lower_ability_perform(&mut ctx, module);

        let ir_text = print_module(&ctx, module.op());
        assert_snapshot!(ir_text);
    }

    #[test]
    #[should_panic(expected = "ability.perform requires evidence parameter")]
    fn test_lower_perform_no_evidence() {
        let mut ctx = IrContext::new();
        init_common_types(&mut ctx);

        // Function without evidence parameter — should panic.
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test_fn() -> tribute_rt.anyref {
    %k = arith.const {value = 0} : tribute_rt.anyref
    %yr = ability.perform %k {ability_ref = core.ability_ref() {name = @State}, op_name = @get} : tribute_rt.anyref
    func.return %yr
  }
}"#,
        );

        lower_ability_perform(&mut ctx, module);
    }
}

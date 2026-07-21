//! Resolve typed `wasm_gc` operations to indexed `wasm` instructions.

use std::collections::HashMap;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{wasm, wasm_gc};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::Attribute;

use crate::gc_types::{
    BOXED_F64_IDX, BYTES_ARRAY_IDX, BYTES_STRUCT_IDX, CLOSURE_STRUCT_IDX, CONTINUATION_IDX,
    EVIDENCE_IDX, FIRST_USER_TYPE_IDX, MARKER_IDX, RESUME_WRAPPER_IDX, STEP_IDX,
};

fn named_adt(ctx: &IrContext, ty: TypeRef, expected: &'static str) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("adt")
        && matches!(
            data.attrs.get("name"),
            Some(Attribute::Symbol(name)) if *name == Symbol::new(expected)
        )
}

fn is_bytes_array(ctx: &IrContext, ty: TypeRef) -> bool {
    let reference = ctx.types.get(ty);
    if reference.dialect != Symbol::new("core")
        || reference.name != Symbol::new("ref")
        || reference.params.len() != 1
    {
        return false;
    }
    let array = ctx.types.get(reference.params[0]);
    array.dialect == Symbol::new("core")
        && array.name == Symbol::new("array")
        && array.params.len() == 1
        && {
            let element = ctx.types.get(array.params[0]);
            element.dialect == Symbol::new("core") && element.name == Symbol::new("i8")
        }
}

fn is_evidence_array(ctx: &IrContext, ty: TypeRef) -> bool {
    let array = ctx.types.get(ty);
    array.dialect == Symbol::new("core")
        && array.name == Symbol::new("array")
        && array.params.len() == 1
        && named_adt(ctx, array.params[0], "_Marker")
}

fn builtin_type_idx(ctx: &IrContext, ty: TypeRef) -> Option<u32> {
    let data = ctx.types.get(ty);
    if data.dialect == Symbol::new("core") && data.name == Symbol::new("bytes") {
        Some(BYTES_STRUCT_IDX)
    } else if is_bytes_array(ctx, ty) {
        Some(BYTES_ARRAY_IDX)
    } else if named_adt(ctx, ty, "_BoxedF64") {
        Some(BOXED_F64_IDX)
    } else if named_adt(ctx, ty, "_Step") {
        Some(STEP_IDX)
    } else if named_adt(ctx, ty, "_closure") {
        Some(CLOSURE_STRUCT_IDX)
    } else if named_adt(ctx, ty, "_Marker") {
        Some(MARKER_IDX)
    } else if is_evidence_array(ctx, ty) {
        Some(EVIDENCE_IDX)
    } else if named_adt(ctx, ty, "_Continuation") {
        Some(CONTINUATION_IDX)
    } else if named_adt(ctx, ty, "_ResumeWrapper") {
        Some(RESUME_WRAPPER_IDX)
    } else {
        None
    }
}

fn is_abstract_heap_type(ctx: &IrContext, ty: TypeRef) -> bool {
    let data = ctx.types.get(ty);
    data.dialect == Symbol::new("wasm")
        && [
            "anyref",
            "eqref",
            "i31ref",
            "structref",
            "arrayref",
            "funcref",
            "externref",
        ]
        .into_iter()
        .any(|name| data.name == Symbol::new(name))
}

fn collect_typed_ops(ctx: &IrContext, region: RegionRef, types: &mut Vec<TypeRef>) {
    for &block in &ctx.region(region).blocks {
        for &op in &ctx.block(block).ops {
            let mut push = |ty| {
                if !types.contains(&ty) && !is_abstract_heap_type(ctx, ty) {
                    types.push(ty);
                }
            };
            if let Ok(op) = wasm_gc::StructNew::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::StructGet::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::StructSet::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::ArrayNew::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::ArrayNewDefault::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::ArrayNewData::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::ArrayGet::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::ArrayGetS::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::ArrayGetU::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::ArraySet::from_op(ctx, op) {
                push(op.r#type(ctx));
            } else if let Ok(op) = wasm_gc::ArrayCopy::from_op(ctx, op) {
                push(op.dst_type(ctx));
                push(op.src_type(ctx));
            } else if let Ok(op) = wasm_gc::RefNull::from_op(ctx, op) {
                push(op.target_type(ctx));
            } else if let Ok(op) = wasm_gc::RefCast::from_op(ctx, op) {
                push(op.target_type(ctx));
            } else if let Ok(op) = wasm_gc::RefTest::from_op(ctx, op) {
                push(op.target_type(ctx));
            }
            for &nested in &ctx.op(op).regions {
                collect_typed_ops(ctx, nested, types);
            }
        }
    }
}

struct LowerTypedGcPattern {
    indices: HashMap<TypeRef, u32>,
}

impl LowerTypedGcPattern {
    fn index(&self, ty: TypeRef) -> Option<u32> {
        self.indices.get(&ty).copied()
    }
}

impl RewritePattern for LowerTypedGcPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let loc = ctx.op(op).location;
        if let Ok(old) = wasm_gc::StructNew::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::struct_new(ctx, loc, old.fields(ctx).to_vec(), old.result_ty(ctx), idx);
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::StructGet::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::struct_get(
                ctx,
                loc,
                old.r#ref(ctx),
                old.result_ty(ctx),
                idx,
                old.field_idx(ctx),
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::StructSet::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::struct_set(
                ctx,
                loc,
                old.r#ref(ctx),
                old.value(ctx),
                idx,
                old.field_idx(ctx),
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::ArrayNew::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::array_new(
                ctx,
                loc,
                old.size(ctx),
                old.init(ctx),
                old.result_ty(ctx),
                idx,
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::ArrayNewDefault::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::array_new_default(ctx, loc, old.size(ctx), old.result_ty(ctx), idx);
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::ArrayNewData::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::array_new_data(
                ctx,
                loc,
                old.offset(ctx),
                old.size(ctx),
                old.result_ty(ctx),
                idx,
                old.data_idx(ctx),
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::ArrayGet::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::array_get(
                ctx,
                loc,
                old.r#ref(ctx),
                old.index(ctx),
                old.result_ty(ctx),
                idx,
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::ArrayGetS::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::array_get_s(
                ctx,
                loc,
                old.r#ref(ctx),
                old.index(ctx),
                old.result_ty(ctx),
                idx,
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::ArrayGetU::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::array_get_u(
                ctx,
                loc,
                old.r#ref(ctx),
                old.index(ctx),
                old.result_ty(ctx),
                idx,
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::ArraySet::from_op(ctx, op) {
            let Some(idx) = self.index(old.r#type(ctx)) else {
                return false;
            };
            let new = wasm::array_set(
                ctx,
                loc,
                old.r#ref(ctx),
                old.index(ctx),
                old.value(ctx),
                idx,
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::ArrayCopy::from_op(ctx, op) {
            let (Some(dst_idx), Some(src_idx)) =
                (self.index(old.dst_type(ctx)), self.index(old.src_type(ctx)))
            else {
                return false;
            };
            let new = wasm::array_copy(
                ctx,
                loc,
                old.dst(ctx),
                old.dst_offset(ctx),
                old.src(ctx),
                old.src_offset(ctx),
                old.len(ctx),
                dst_idx,
                src_idx,
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::RefNull::from_op(ctx, op) {
            let target = old.target_type(ctx);
            let new = wasm::ref_null(
                ctx,
                loc,
                old.result_ty(ctx),
                ctx.types.get(target).name,
                self.index(target),
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::RefCast::from_op(ctx, op) {
            let target = old.target_type(ctx);
            let new = wasm::ref_cast(
                ctx,
                loc,
                old.r#ref(ctx),
                old.result_ty(ctx),
                target,
                self.index(target),
            );
            rewriter.replace_op(new.op_ref());
        } else if let Ok(old) = wasm_gc::RefTest::from_op(ctx, op) {
            let target = old.target_type(ctx);
            let new = wasm::ref_test(
                ctx,
                loc,
                old.r#ref(ctx),
                old.result_ty(ctx),
                target,
                self.index(target),
            );
            rewriter.replace_op(new.op_ref());
        } else {
            return false;
        }
        true
    }
}

/// Assign module-local GC type indices and fully lower typed GC operations.
pub fn lower(ctx: &mut IrContext, module: Module) {
    let mut types = Vec::new();
    if let Some(body) = module.body(ctx) {
        collect_typed_ops(ctx, body, &mut types);
    }

    let mut next = FIRST_USER_TYPE_IDX;
    let mut indices = HashMap::new();
    for ty in types {
        let idx = builtin_type_idx(ctx, ty).unwrap_or_else(|| {
            let idx = next;
            next += 1;
            idx
        });
        indices.insert(ty, idx);
    }

    PatternApplicator::new(TypeConverter::new())
        .add_pattern(LowerTypedGcPattern { indices })
        .apply_partial(ctx, module);
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    #[test]
    fn nominal_types_with_equal_layout_receive_distinct_indices() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !A = adt.struct() {fields = [[@value, core.i32]], name = @A}
  !B = adt.struct() {fields = [[@value, core.i32]], name = @B}

  wasm.func @main() -> core.nil {
    %zero = wasm.i32_const {value = 0} : core.i32
    %a = wasm_gc.struct_new %zero {type = !A} : !A
    %b = wasm_gc.struct_new %zero {type = !B} : !B
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module);

        let func = module.ops(&ctx)[0];
        let body = ctx.op(func).regions[0];
        let block = ctx.region(body).blocks[0];
        let indices: Vec<u32> = ctx
            .block(block)
            .ops
            .iter()
            .filter_map(|&op| wasm::StructNew::from_op(&ctx, op).ok())
            .map(|op| op.type_idx(&ctx))
            .collect();
        assert_eq!(indices, vec![FIRST_USER_TYPE_IDX, FIRST_USER_TYPE_IDX + 1]);
        assert!(
            ctx.block(block)
                .ops
                .iter()
                .all(|&op| ctx.op(op).dialect != wasm_gc::DIALECT_NAME())
        );
    }

    #[test]
    fn builtin_semantic_type_receives_reserved_index() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    %zero = wasm.i32_const {value = 0} : core.i32
    %bytes = wasm_gc.struct_new %zero {type = core.bytes} : core.bytes
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module);

        let func = module.ops(&ctx)[0];
        let body = ctx.op(func).regions[0];
        let block = ctx.region(body).blocks[0];
        let op = ctx
            .block(block)
            .ops
            .iter()
            .find_map(|&op| wasm::StructNew::from_op(&ctx, op).ok())
            .expect("typed struct.new should be lowered");
        assert_eq!(op.type_idx(&ctx), BYTES_STRUCT_IDX);
    }

    #[test]
    fn only_concrete_heap_types_receive_indices() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !A = adt.struct() {fields = [], name = @A}

  wasm.func @main() -> core.nil {
    %null = wasm.ref_null {heap_type = @anyref} : wasm.anyref
    %concrete = wasm_gc.ref_cast %null {target_type = !A} : !A
    %abstract = wasm_gc.ref_cast %null {target_type = wasm.anyref} : wasm.anyref
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module);

        let func = module.ops(&ctx)[0];
        let body = ctx.op(func).regions[0];
        let block = ctx.region(body).blocks[0];
        let indices: Vec<Option<u32>> = ctx
            .block(block)
            .ops
            .iter()
            .filter_map(|&op| wasm::RefCast::from_op(&ctx, op).ok())
            .map(|op| op.type_idx(&ctx))
            .collect();
        assert_eq!(indices, vec![Some(FIRST_USER_TYPE_IDX), None]);
    }

    #[test]
    fn lowers_all_typed_gc_operations_and_preserves_operands_and_attributes() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !S = adt.struct() {fields = [[@value, core.i32]], name = @S}
  !A = core.array(core.i32)
  !B = core.array(core.i32)

  wasm.func @main() -> core.nil {
    %zero = wasm.i32_const {value = 0} : core.i32
    %one = wasm.i32_const {value = 1} : core.i32
    %struct = wasm_gc.struct_new %one {type = !S} : !S
    %field = wasm_gc.struct_get %struct {type = !S, field_idx = 3} : core.i32
    wasm_gc.struct_set %struct, %field {type = !S, field_idx = 4}
    %array = wasm_gc.array_new %one, %zero {type = !A} : !A
    %default = wasm_gc.array_new_default %one {type = !A} : !A
    %data = wasm_gc.array_new_data %zero, %one {type = !A, data_idx = 5} : !A
    %element = wasm_gc.array_get %array, %zero {type = !A} : core.i32
    %signed = wasm_gc.array_get_s %default, %zero {type = !A} : core.i32
    %unsigned = wasm_gc.array_get_u %data, %zero {type = !A} : core.i32
    wasm_gc.array_set %array, %zero, %element {type = !A}
    wasm_gc.array_copy %array, %zero, %default, %one, %one {dst_type = !A, src_type = !B}
    %null = wasm_gc.ref_null {target_type = !S} : !S
    %cast = wasm_gc.ref_cast %null {target_type = !S} : !S
    %tested = wasm_gc.ref_test %cast {target_type = !S} : core.i32
    wasm.return
  }
}"#,
        );

        lower(&mut ctx, module);

        let func = module.ops(&ctx)[0];
        let body = ctx.op(func).regions[0];
        let block = ctx.region(body).blocks[0];
        let ops = &ctx.block(block).ops;
        assert!(
            ops.iter()
                .all(|&op| ctx.op(op).dialect != wasm_gc::DIALECT_NAME())
        );

        let struct_new = ops
            .iter()
            .find_map(|&op| wasm::StructNew::from_op(&ctx, op).ok())
            .unwrap();
        let struct_get = ops
            .iter()
            .find_map(|&op| wasm::StructGet::from_op(&ctx, op).ok())
            .unwrap();
        let struct_set = ops
            .iter()
            .find_map(|&op| wasm::StructSet::from_op(&ctx, op).ok())
            .unwrap();
        assert_eq!(struct_new.type_idx(&ctx), FIRST_USER_TYPE_IDX);
        assert_eq!(struct_get.r#ref(&ctx), struct_new.result(&ctx));
        assert_eq!(struct_get.field_idx(&ctx), 3);
        assert_eq!(struct_set.r#ref(&ctx), struct_new.result(&ctx));
        assert_eq!(struct_set.value(&ctx), struct_get.result(&ctx));
        assert_eq!(struct_set.field_idx(&ctx), 4);

        let array_new = ops
            .iter()
            .find_map(|&op| wasm::ArrayNew::from_op(&ctx, op).ok())
            .unwrap();
        let array_default = ops
            .iter()
            .find_map(|&op| wasm::ArrayNewDefault::from_op(&ctx, op).ok())
            .unwrap();
        let array_data = ops
            .iter()
            .find_map(|&op| wasm::ArrayNewData::from_op(&ctx, op).ok())
            .unwrap();
        let array_get = ops
            .iter()
            .find_map(|&op| wasm::ArrayGet::from_op(&ctx, op).ok())
            .unwrap();
        let array_get_s = ops
            .iter()
            .find_map(|&op| wasm::ArrayGetS::from_op(&ctx, op).ok())
            .unwrap();
        let array_get_u = ops
            .iter()
            .find_map(|&op| wasm::ArrayGetU::from_op(&ctx, op).ok())
            .unwrap();
        let array_set = ops
            .iter()
            .find_map(|&op| wasm::ArraySet::from_op(&ctx, op).ok())
            .unwrap();
        let array_copy = ops
            .iter()
            .find_map(|&op| wasm::ArrayCopy::from_op(&ctx, op).ok())
            .unwrap();
        let array_idx = FIRST_USER_TYPE_IDX + 1;
        assert_eq!(array_new.type_idx(&ctx), array_idx);
        assert_eq!(array_default.type_idx(&ctx), array_idx);
        assert_eq!(array_data.type_idx(&ctx), array_idx);
        assert_eq!(array_data.data_idx(&ctx), 5);
        assert_eq!(array_get.r#ref(&ctx), array_new.result(&ctx));
        assert_eq!(array_get.type_idx(&ctx), array_idx);
        assert_eq!(array_get_s.r#ref(&ctx), array_default.result(&ctx));
        assert_eq!(array_get_s.type_idx(&ctx), array_idx);
        assert_eq!(array_get_u.r#ref(&ctx), array_data.result(&ctx));
        assert_eq!(array_get_u.type_idx(&ctx), array_idx);
        assert_eq!(array_set.r#ref(&ctx), array_new.result(&ctx));
        assert_eq!(array_set.value(&ctx), array_get.result(&ctx));
        assert_eq!(array_set.type_idx(&ctx), array_idx);
        assert_eq!(array_copy.dst(&ctx), array_new.result(&ctx));
        assert_eq!(array_copy.src(&ctx), array_default.result(&ctx));
        assert_eq!(array_copy.dst_type_idx(&ctx), array_idx);
        assert_eq!(array_copy.src_type_idx(&ctx), array_idx);

        let ref_null = ops
            .iter()
            .find_map(|&op| wasm::RefNull::from_op(&ctx, op).ok())
            .unwrap();
        let ref_cast = ops
            .iter()
            .find_map(|&op| wasm::RefCast::from_op(&ctx, op).ok())
            .unwrap();
        let ref_test = ops
            .iter()
            .find_map(|&op| wasm::RefTest::from_op(&ctx, op).ok())
            .unwrap();
        assert_eq!(ref_null.type_idx(&ctx), Some(FIRST_USER_TYPE_IDX));
        assert_eq!(ref_cast.r#ref(&ctx), ref_null.result(&ctx));
        assert_eq!(ref_cast.type_idx(&ctx), Some(FIRST_USER_TYPE_IDX));
        assert_eq!(ref_test.r#ref(&ctx), ref_cast.result(&ctx));
        assert_eq!(ref_test.type_idx(&ctx), Some(FIRST_USER_TYPE_IDX));
    }
}

//! Lower the nominal layouts of struct and variant field accesses to
//! `mem.struct`, and variant tests to descriptor comparisons.
//!
//! Allocation is the last use of a nominal layout: `adt_rc_header` finds an
//! allocation's descriptor by its `adt.struct` type. A field access needs only
//! the representation and order of the fields. This pass replaces the `type`
//! of each `adt.struct_get` and `adt.struct_set` with the nameless
//! `mem.struct` of its target field representations, so `adt_to_clif`
//! computes offsets without a type converter.
//!
//! A field the allocation releases stays `tribute_rt.anyref`, and every other
//! field takes its native representation, so an unmanaged pointer is
//! `core.ptr`. Both are pointer-sized; they differ in how they are released.
//! Whether a field is released is read from the layout's `tribute_rtti.layout`
//! declaration, or from the ownership plan for a layout the module never
//! allocates. The pass therefore runs after closure layout adaptation
//! settles the declarations and before `adt_rc_header` erases them.
//!
//! A variant object holds only its own fields, so a variant field access is
//! a struct field access: each `adt.variant_get` becomes an `adt.struct_get`
//! of the variant's `mem.struct`.
//!
//! A variant is told apart by its runtime type descriptor. The same
//! declarations number the descriptors, so this pass replaces each
//! `adt.variant_is` with a comparison of the value's
//! `tribute_rtti.descriptor` with the variant's number. A variant the module
//! never allocates has no number, and its test
//! is constant false without reading the reference. A variant test of a null
//! reference is undefined, so that path need not fault as a header read does.

use rustc_hash::FxHashMap as HashMap;
use std::ops::ControlFlow;

use tribute_ir::dialect::adt;
use tribute_ir::dialect::adt::layout::get_struct_fields;
use tribute_ir::dialect::{tribute_rt, tribute_rtti};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{arith, core, mem};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::{Attribute, StringRef};
use trunk_ir::walk::{WalkAction, walk_op};

use super::ownership_plan::NativeOwnershipPlan;

/// PassManager-friendly `struct_to_mem`. It takes over the ownership plan,
/// whose last use is the release judgment of a layout without a declaration.
pub struct StructToMem {
    plan: NativeOwnershipPlan,
}

impl StructToMem {
    pub fn new(plan: NativeOwnershipPlan) -> Self {
        Self { plan }
    }
}

impl Pass for StructToMem {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "struct-to-mem"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let (type_converter, _) = super::type_converter::native_type_converter(ctx);
        lower(ctx, target.into(), &self.plan, &type_converter);
        Ok(())
    }
}

/// Replace the `adt.struct` layout of every struct field access in `module`
/// with its `mem.struct`, and every variant test with a comparison of the
/// variant's descriptor number.
pub fn lower(
    ctx: &mut IrContext,
    module: Module,
    plan: &NativeOwnershipPlan,
    type_converter: &TypeConverter,
) {
    // Whether each field of a declared descriptor is released: a struct
    // layout, or an enum layout and one of its variant tags.
    let released: HashMap<(TypeRef, Option<StringRef>), Vec<bool>> =
        tribute_rtti::Layout::declared(ctx, module)
            .iter()
            .map(|layout| {
                let fields = layout.field_kinds(ctx);
                (
                    (layout.r#type(ctx), layout.tag_ref(ctx)),
                    fields.into_iter().map(|kind| kind.is_released()).collect(),
                )
            })
            .collect();

    let mut accesses: Vec<(OpRef, TypeRef)> = Vec::new();
    let mut variants: Vec<(TypeRef, StringRef)> = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        let layout = if let Ok(get) = adt::StructGet::from_op(ctx, op) {
            Some(get.r#type(ctx))
        } else {
            adt::StructSet::from_op(ctx, op)
                .ok()
                .map(|set| set.r#type(ctx))
        };
        if let Some(layout) = layout {
            accesses.push((op, layout));
        }
        if let Ok(get) = adt::VariantGet::from_op(ctx, op) {
            variants.push((get.r#type(ctx), get.tag_ref(ctx)));
        }
        ControlFlow::Continue(WalkAction::Advance)
    });

    let mut converted: HashMap<TypeRef, TypeRef> = HashMap::default();
    for (op, layout) in accesses {
        let structural = match converted.get(&layout) {
            Some(&structural) => structural,
            None => {
                let Some(fields) = get_struct_fields(ctx, layout) else {
                    continue;
                };
                let structural = structural_layout(
                    ctx,
                    fields.into_iter().map(|(_name, field_ty)| field_ty),
                    released.get(&(layout, None)).map(Vec::as_slice),
                    plan,
                    type_converter,
                );
                converted.insert(layout, structural);
                structural
            }
        };
        ctx.op_mut(op)
            .attributes
            .insert("type", Attribute::Type(structural));
    }

    // A variant holds only its own fields, so its layout is a struct's.
    let mut variant_layouts: HashMap<(TypeRef, StringRef), TypeRef> = HashMap::default();
    for (enum_ty, tag) in variants {
        if variant_layouts.contains_key(&(enum_ty, tag)) {
            continue;
        }
        let Some(fields) = adt::Enum::from_type_ref(ctx, enum_ty)
            .and_then(|adt_enum| adt_enum.variant_fields(ctx, tag))
            .map(<[TypeRef]>::to_vec)
        else {
            continue;
        };
        let structural = structural_layout(
            ctx,
            fields,
            released.get(&(enum_ty, Some(tag))).map(Vec::as_slice),
            plan,
            type_converter,
        );
        variant_layouts.insert((enum_ty, tag), structural);
    }

    let numbers = tribute_rtti::Layout::declared_indices(ctx, module);
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(VariantIsPattern { numbers })
        .add_pattern(VariantGetPattern {
            layouts: variant_layouts,
        })
        .apply_partial(ctx, module);
}

/// `adt.variant_get` -> `adt.struct_get` of the variant's `mem.struct`.
struct VariantGetPattern {
    layouts: HashMap<(TypeRef, StringRef), TypeRef>,
}

impl RewritePattern for VariantGetPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_get) = adt::VariantGet::from_op(ctx, op) else {
            return false;
        };
        let variant = (variant_get.r#type(ctx), variant_get.tag_ref(ctx));
        let Some(&structural) = self.layouts.get(&variant) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let field_get = adt::StructGet::operands(variant_get.r#ref(ctx))
            .r#type(structural)
            .field(variant_get.field(ctx))
            .results(variant_get.result_ty(ctx))
            .build(ctx, loc);
        rewriter.replace_op(field_get.op_ref());
        true
    }
}

/// `adt.variant_is` -> `arith.cmpi eq` of the value's
/// `tribute_rtti.descriptor` and the variant's declared descriptor number, or
/// constant false for a variant without one.
struct VariantIsPattern {
    numbers: HashMap<(TypeRef, Option<StringRef>), u32>,
}

impl RewritePattern for VariantIsPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(variant_is) = adt::VariantIs::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let result_ty = variant_is.result_ty(ctx);
        let descriptor = (variant_is.r#type(ctx), Some(variant_is.tag_ref(ctx)));
        let Some(&number) = self.numbers.get(&descriptor) else {
            let never = arith::Const::operands()
                .value(Attribute::Bool(false))
                .results(result_ty)
                .build(ctx, loc);
            rewriter.replace_op(never.op_ref());
            return true;
        };
        // The comparison yields `core.i1`, the type a variant test has.
        if !core::I1::matches(ctx, result_ty) {
            return false;
        }
        let actual = tribute_rtti::Descriptor::operands(variant_is.r#ref(ctx)).build(ctx, loc);
        let i32_ty = ctx.value_ty(actual.result(ctx));
        let expected = arith::Const::operands()
            .value(Attribute::Int(i128::from(number)))
            .results(i32_ty)
            .build(ctx, loc);
        let matches = arith::Cmpi::operands(actual.result(ctx), expected.result(ctx))
            .predicate("eq")
            .build(ctx, loc);
        rewriter.insert_op(actual.op_ref());
        rewriter.insert_op(expected.op_ref());
        rewriter.replace_op(matches.op_ref());
        true
    }
}

/// The `mem.struct` of a struct or variant with the source field types
/// `fields`. `released` is the declared release of each field.
fn structural_layout(
    ctx: &mut IrContext,
    fields: impl IntoIterator<Item = TypeRef>,
    released: Option<&[bool]>,
    plan: &NativeOwnershipPlan,
    type_converter: &TypeConverter,
) -> TypeRef {
    let anyref = tribute_rt::anyref(ctx).as_type_ref();
    let fields: Vec<TypeRef> = fields
        .into_iter()
        .enumerate()
        .map(|(index, field_ty)| {
            let is_released = match released {
                Some(released) => released[index],
                None => plan.is_managed_type(ctx, field_ty),
            };
            if is_released {
                anyref
            } else {
                type_converter.convert_type_or_identity(ctx, field_ty)
            }
        })
        .collect();
    mem::r#struct(ctx, fields).as_type_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::ownership_plan::{NativeOwnershipPlanOptions, build_native_ownership_plan};
    use crate::native::type_converter::native_type_converter;
    use tribute_ir::dialect::adt::layout::{compute_mem_struct_layout, compute_struct_layout};
    use trunk_ir::dialect::core;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::types::TypeDataBuilder;

    fn type_alias(ctx: &IrContext, name: &str) -> TypeRef {
        ctx.type_aliases()
            .iter()
            .find_map(|(alias, ty)| (*alias == name).then_some(*ty))
            .unwrap_or_else(|| panic!("missing type alias !{name}"))
    }

    /// Parse `ir`, lower it, and return each struct access's layout in
    /// operation order.
    fn lower_module(ir: &str) -> (IrContext, Module, Vec<TypeRef>) {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        let plan = build_native_ownership_plan(
            &ctx,
            module,
            NativeOwnershipPlanOptions::production(),
            &mut Default::default(),
        )
        .expect("typed ownership plan");
        let (type_converter, _) = native_type_converter(&mut ctx);

        lower(&mut ctx, module, &plan, &type_converter);

        let mut layouts = Vec::new();
        let _ = walk_op::<()>(&ctx, module.op(), &mut |op| {
            if let Ok(get) = adt::StructGet::from_op(&ctx, op) {
                layouts.push(get.r#type(&ctx));
            } else if let Ok(set) = adt::StructSet::from_op(&ctx, op) {
                layouts.push(set.r#type(&ctx));
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
        (ctx, module, layouts)
    }

    #[test]
    fn structural_layout_keeps_the_offsets_and_size_of_the_nominal_layout() {
        let (mut ctx, _module, layouts) = lower_module(
            r#"core.module @test {
  !Mixed = adt.struct<Mixed(flag: core.i8, count: tribute_rt.int, any: tribute_rt.anyref, byte: core.i8, raw: core.ptr, ratio: tribute_rt.float, small: core.i16)>
  func.func @read(%mixed: adt.typeref<{name = "Mixed"}>) -> core.i8 {
    %flag = adt.struct_get %mixed {field = 0, type = !Mixed} : core.i8
    func.return %flag
  }
}"#,
        );
        let nominal = type_alias(&ctx, "Mixed");
        let [structural] = layouts[..] else {
            panic!("one struct access")
        };

        let scalar = |ctx: &mut IrContext, name: &'static str| {
            ctx.intern_type(TypeDataBuilder::new("core", name).build())
        };
        let i8_ty = scalar(&mut ctx, "i8");
        let i16_ty = scalar(&mut ctx, "i16");
        let i32_ty = scalar(&mut ctx, "i32");
        let f64_ty = scalar(&mut ctx, "f64");
        let ptr_ty = core::ptr(&mut ctx).as_type_ref();
        let anyref = tribute_rt::anyref(&mut ctx).as_type_ref();
        assert_eq!(
            mem::Struct::from_type_ref(&ctx, structural)
                .expect("mem.struct")
                .fields(&ctx),
            [i8_ty, i32_ty, anyref, i8_ty, ptr_ty, f64_ty, i16_ty]
        );

        let (type_converter, _) = native_type_converter(&mut ctx);
        let expected = compute_struct_layout(&ctx, nominal, &type_converter).expect("adt.struct");
        let actual = compute_mem_struct_layout(&ctx, structural).expect("mem.struct");
        assert_eq!(actual.field_offsets, expected.field_offsets);
        assert_eq!(actual.field_offsets, [0, 4, 8, 16, 24, 32, 40]);
        assert_eq!(actual.total_size, expected.total_size);
        assert_eq!(actual.alignment, expected.alignment);
    }

    #[test]
    fn layouts_of_equal_shape_share_one_structural_layout() {
        let (ctx, module, layouts) = lower_module(
            r#"core.module @test {
  !Point = adt.struct<Point(x: core.i32, y: core.i32)>
  !Size = adt.struct<Size(width: core.i32, height: core.i32)>
  func.func @read(%point: !Point, %size: !Size) -> core.i32 {
    %x = adt.struct_get %point {field = 0, type = !Point} : core.i32
    adt.struct_set %size, %x {field = 1, type = !Size}
    func.return %x
  }
}"#,
        );

        let [point, size] = layouts[..] else {
            panic!("two struct accesses")
        };
        assert_eq!(point, size);
        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("type = mem.struct<core.i32, core.i32>"),
            "{printed}"
        );
    }

    #[test]
    fn declared_release_decides_a_field() {
        let (ctx, module, layouts) = lower_module(
            r#"core.module @test {
  !Pair = adt.struct<Pair(code: core.i64, env: core.ptr)>
  tribute_rtti.layout {fields = ["raw", "dynamic"], index = 5, type = !Pair}
  func.func @read(%pair: core.ptr) -> core.ptr {
    %env = adt.struct_get %pair {field = 1, type = !Pair} : core.ptr
    func.return %env
  }
}"#,
        );

        let [structural] = layouts[..] else {
            panic!("one struct access")
        };
        let printed = print_module(&ctx, module.op());
        assert!(mem::Struct::matches(&ctx, structural));
        assert!(
            printed.contains("type = mem.struct<core.i64, tribute_rt.anyref>"),
            "{printed}"
        );
    }

    #[test]
    fn variant_tests_compare_the_declared_descriptor_number() {
        let (ctx, module, _layouts) = lower_module(
            r#"core.module @test {
  !ChoiceRef = adt.typeref<{name = "Choice"}>
  !Choice = adt.enum<Choice { None(), Some(core.i32), Other() }>
  tribute_rtti.layout {fields = [], index = 5, type = !Choice, tag = "None"}
  tribute_rtti.layout {fields = ["u32"], index = 6, type = !Choice, tag = "Some"}
  func.func @test(%choice: !ChoiceRef) -> core.i1 {
    %is_some = adt.variant_is %choice {tag = "Some", type = !Choice} : core.i1
    %is_other = adt.variant_is %choice {tag = "Other", type = !Choice} : core.i1
    func.return %is_some
  }
}"#,
        );

        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("adt.variant_is"), "{printed}");
        assert!(
            printed.contains("tribute_rtti.descriptor %0 : core.i32"),
            "{printed}"
        );
        assert!(
            printed.contains("arith.const {value = 6} : core.i32"),
            "{printed}"
        );
        assert!(printed.contains("arith.cmpi"), "{printed}");
        // `Other` is never allocated, so no object has its descriptor.
        assert!(
            printed.contains("arith.const {value = false} : core.i1"),
            "{printed}"
        );
    }

    #[test]
    fn variant_field_accesses_use_the_variants_own_structural_layout() {
        let (ctx, module, layouts) = lower_module(
            r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !ChoiceRef = adt.typeref<{name = "Choice"}>
  !Choice = adt.enum<Choice { None(), Pair(core.i8, !ChildRef), Raw(core.ptr, core.i64) }>
  tribute_rtti.layout {fields = ["u8", "managed"], index = 6, type = !Choice, tag = "Pair"}
  func.func @test(%choice: !ChoiceRef) -> !ChildRef {
    %child = adt.variant_get %choice {tag = "Pair", field = 1, type = !Choice} : !ChildRef
    %raw = adt.variant_get %choice {tag = "Raw", field = 1, type = !Choice} : core.i64
    func.return %child
  }
}"#,
        );

        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("adt.variant_get"), "{printed}");
        // `Pair` is declared, `Raw` reads its fields from the ownership plan.
        let [pair, raw] = layouts[..] else {
            panic!("two variant accesses: {printed}")
        };
        assert!(
            printed.contains("{field = 1, type = mem.struct<core.i8, tribute_rt.anyref>}"),
            "{printed}"
        );
        assert!(
            printed.contains("{field = 1, type = mem.struct<core.ptr, core.i64>}"),
            "{printed}"
        );
        // Fields start at the payload, with no tag before them.
        assert_eq!(
            compute_mem_struct_layout(&ctx, pair).unwrap().field_offsets,
            [0, 8]
        );
        assert_eq!(
            compute_mem_struct_layout(&ctx, raw).unwrap().field_offsets,
            [0, 8]
        );
    }

    #[test]
    fn undeclared_layout_reads_managed_fields_from_the_ownership_plan() {
        let (ctx, module, _layouts) = lower_module(
            r#"core.module @test {
  !Child = adt.struct<Child(value: core.i32)>
  !ChildRef = adt.typeref<{name = "Child"}>
  !Parent = adt.struct<Parent(child: !ChildRef, raw: core.ptr, bytes: core.bytes)>
  func.func @read(%parent: adt.typeref<{name = "Parent"}>) -> !ChildRef {
    %child = adt.struct_get %parent {field = 0, type = !Parent} : !ChildRef
    func.return %child
  }
}"#,
        );

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("type = mem.struct<tribute_rt.anyref, core.ptr, core.ptr>"),
            "{printed}"
        );
    }
}

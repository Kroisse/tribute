//! Lower adt.string_const and adt.bytes_const to wasm data segments.
//!
//! This pass uses a two-phase approach:
//! 1. Analysis: Collect the distinct string/bytes payloads of the input IR
//! 2. Transform: Declare a passive `wasm.data` segment for each payload and
//!    replace const operations with wasm ops that reference it
//!
//! A data index is the position of its segment among the module's `wasm.data`
//! operations, so later steps read the segments from the IR itself.

use hashbrown::hash_map::EntryRef;
use std::borrow::Cow;

use crate::collections::{HashMap, HashSet};
use std::fmt;

use crate::string_literal::{ExpandBytesConst, ExpandStringConst, LiteralLayouts};
use tribute_ir::dialect::adt;
use tribute_ir::metadata::StringLeaf;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::dialect::wasm_gc;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef};
use trunk_ir::rewrite::{
    Module, OpRewritePattern, PatternApplicator, PatternRewriter, TypeConverter,
};
use trunk_ir::types::Attribute;

#[derive(Debug, PartialEq, Eq)]
pub enum ConstValidationError {
    MissingCanonicalStringType,
    InvalidStringResultType { actual: Cow<'static, str> },
}

impl fmt::Display for ConstValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCanonicalStringType => {
                f.write_str("adt.string_const requires the canonical prelude String type")
            }
            Self::InvalidStringResultType { actual } => write!(
                f,
                "adt.string_const must produce a value of the canonical String type, found {actual}"
            ),
        }
    }
}

impl std::error::Error for ConstValidationError {}

/// Result of constant analysis.
pub struct ConstAnalysis {
    /// Distinct string/bytes payloads, in first-occurrence order.
    pub contents: Vec<Vec<u8>>,
    /// The layouts the byte and string constants build.
    pub(crate) layouts: LiteralLayouts,
}

/// Context for collecting const payloads during analysis.
struct ConstCollector {
    contents: Vec<Vec<u8>>,
    seen: HashSet<Vec<u8>>,
    has_string_consts: bool,
}

impl ConstCollector {
    fn new() -> Self {
        Self {
            contents: Vec::new(),
            seen: HashSet::default(),
            has_string_consts: false,
        }
    }

    fn collect_content(&mut self, bytes: &[u8]) {
        self.seen.get_or_insert_with(bytes, |bytes| {
            self.contents.push(bytes.to_vec());
            bytes.to_vec()
        });
    }

    fn visit_op(&mut self, ctx: &IrContext, op: OpRef) {
        let data = ctx.op(op);

        if data.name.dialect(ctx) == adt::DIALECT_NAME {
            if adt::StringConst::matches(ctx, op) {
                if let Some(s) = ctx.op_attrs(op).get_str("value") {
                    self.has_string_consts = true;
                    self.collect_content(s.as_bytes());
                }
            } else if (adt::BytesConst::matches(ctx, op) || adt::ArrayConst::matches(ctx, op))
                && let Some(Attribute::Bytes(b)) = ctx.op_attrs(op).get("value")
            {
                self.collect_content(b);
            }
        }
    }
}

/// Walk all operations in a region recursively.
fn walk_ops_in_region(
    ctx: &IrContext,
    region: RegionRef,
    callback: &mut impl FnMut(&IrContext, OpRef),
) {
    for &block in ctx.region(region).blocks.iter() {
        for &op in ctx.block(block).ops.iter() {
            callback(ctx, op);
            for nested in ctx.op_regions(op) {
                walk_ops_in_region(ctx, nested, callback);
            }
        }
    }
}

/// Analyze a module to collect the distinct string/bytes constant payloads.
pub(crate) fn analyze_consts(ctx: &IrContext, module: Module) -> ConstAnalysis {
    let mut collector = ConstCollector::new();

    // Walk all operations in module body (recursively into nested regions)
    if let Some(body) = module.body(ctx) {
        walk_ops_in_region(ctx, body, &mut |ctx, op| {
            collector.visit_op(ctx, op);
        });
    }

    ConstAnalysis {
        contents: collector.contents,
        layouts: LiteralLayouts::collect(ctx, module, collector.has_string_consts),
    }
}

/// Validate the representation expected at the Wasm constant-lowering boundary:
/// a string constant is a value of the canonical String type, which its
/// lowering builds as that enum's leaf.
pub(crate) fn validate_for_wasm(
    ctx: &IrContext,
    module: Module,
    analysis: &ConstAnalysis,
) -> Result<(), ConstValidationError> {
    let Some(body) = module.body(ctx) else {
        return Ok(());
    };

    let mut result = Ok(());
    walk_ops_in_region(ctx, body, &mut |ctx, op| {
        if result.is_err() || adt::StringConst::from_op(ctx, op).is_err() {
            return;
        }
        let Some(StringLeaf {
            string: string_ty, ..
        }) = analysis.layouts.string_leaf()
        else {
            result = Err(ConstValidationError::MissingCanonicalStringType);
            return;
        };

        let Some(&result_ty) = ctx.op_result_types(op).first() else {
            result = Err(ConstValidationError::InvalidStringResultType {
                actual: "<missing>".into(),
            });
            return;
        };
        if adt::layout_of(ctx, result_ty) != Some(string_ty) {
            result = Err(ConstValidationError::InvalidStringResultType {
                actual: ctx.get_type(result_ty).name.to_text(ctx),
            });
        }
    });
    result
}

/// Lower const operations using an analysis of the same input IR.
///
/// Declares a passive `wasm.data` segment for every payload in `analysis`
/// that the module does not already carry, then lowers the constants to
/// reference those segments.
pub(crate) fn lower(ctx: &mut IrContext, module: Module, analysis: &ConstAnalysis) {
    let segments = declare_data_segments(ctx, module, &analysis.contents);

    let nat = |ctx: &mut IrContext, ty, value: i64, location| {
        let constant = wasm_dialect::I32Const::operands()
            .value(i32::try_from(value).expect("a constant fits Wasm memory"))
            .results(ty)
            .build(ctx, location);
        (constant.op_ref(), constant.result(ctx))
    };
    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(ExpandStringConst {
            layouts: &analysis.layouts,
            nat: &nat,
            // The string's value is the leaf itself, of the String enum's type.
            cast_to_result: false,
        })
        .add_pattern(ExpandBytesConst {
            layouts: &analysis.layouts,
            nat: &nat,
        })
        .add_pattern(ArrayConstPattern::new(&segments));
    applicator.apply_partial(ctx, module);
}

/// Passive data segment index by payload.
type DataSegments = HashMap<Vec<u8>, u32>;

/// Map each payload to a passive `wasm.data` segment of the module.
///
/// Reuses an existing passive segment with the same bytes and appends a new
/// one at the end of the module for every other payload. A segment's data
/// index is its position among the module's `wasm.data` operations.
fn declare_data_segments(
    ctx: &mut IrContext,
    module: Module,
    contents: &[Vec<u8>],
) -> DataSegments {
    let mut segments = DataSegments::default();
    let Some(module_block) = module.first_block(ctx) else {
        return segments;
    };

    let mut next_idx = 0;
    for &op in ctx.block(module_block).ops.iter() {
        let Ok(data) = wasm_dialect::Data::from_op(ctx, op) else {
            continue;
        };
        if data.passive(ctx) {
            segments
                .entry_ref(data.bytes(ctx).as_slice())
                .or_insert(next_idx);
        }
        next_idx += 1;
    }

    let location = ctx.op(module.op()).location;
    for content in contents {
        let EntryRef::Vacant(entry) = segments.entry_ref(content.as_slice()) else {
            continue;
        };
        let op = wasm_dialect::Data::operands()
            .offset(0)
            .bytes(content.as_slice().into())
            .passive(true)
            .build(ctx, location);
        ctx.push_op(module_block, op.op_ref());
        entry.insert(next_idx);
        next_idx += 1;
    }
    segments
}

/// Look up the data index and length of the segment holding `content`.
fn lookup_segment(segments: &DataSegments, content: &[u8]) -> Option<(u32, u32)> {
    segments
        .get(content)
        .map(|&data_idx| (data_idx, content.len() as u32))
}

/// Pattern for `adt.array_const` -> `wasm_gc.array_new_data` of its passive
/// data segment, a new array on each evaluation.
struct ArrayConstPattern<'a> {
    segments: &'a DataSegments,
}

impl<'a> ArrayConstPattern<'a> {
    fn new(segments: &'a DataSegments) -> Self {
        Self { segments }
    }
}

impl OpRewritePattern for ArrayConstPattern<'_> {
    type Op = adt::ArrayConst;

    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        array_const: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> Option<()> {
        let op = array_const.op_ref();
        let content: Vec<u8> = array_const.value(ctx).to_vec();

        let (data_idx, len) = lookup_segment(self.segments, &content)?;

        let location = ctx.op(op).location;
        let result_ty = ctx.op_result_types(op)[0];
        let array_ty = super::type_converter::gc_array_type(ctx, result_ty);
        let i32_ty = trunk_ir::dialect::core::I32::type_ref(ctx);
        let constant = |ctx: &mut IrContext, value: u32| {
            wasm_dialect::I32Const::operands()
                .value(value as i32)
                .results(i32_ty)
                .build(ctx, location)
        };
        let offset = constant(ctx, 0);
        let size = constant(ctx, len);
        let array = wasm_gc::ArrayNewData::operands(offset.result(ctx), size.result(ctx))
            .r#type(array_ty)
            .data_idx(data_idx)
            .build(ctx, location);
        // The byte array keeps its `adt.array` type until `adt` lowering is
        // done, which then makes this cast an identity.
        let cast = trunk_ir::dialect::core::UnrealizedConversionCast::operands(array.result(ctx))
            .results(result_ty)
            .build(ctx, location);
        for inserted in [offset.op_ref(), size.op_ref(), array.op_ref()] {
            rewriter.insert_op(inserted);
        }
        rewriter.replace_op(cast.op_ref());
        Some(())
    }

    fn name(&self) -> &'static str {
        "ArrayConstPattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    fn type_alias(ctx: &IrContext, name: &str) -> trunk_ir::TypeRef {
        ctx.type_aliases()
            .iter()
            .find_map(|(alias, ty)| (*alias == name).then_some(*ty))
            .unwrap_or_else(|| panic!("missing type alias !{name}"))
    }

    fn attach_string_type(ctx: &mut IrContext, module: Module, string: trunk_ir::TypeRef) {
        tribute_ir::metadata::WellKnownTypes {
            string: Some(string),
        }
        .attach(ctx, module.op());
    }

    fn string_module(ctx: &mut IrContext, values: &[&str]) -> Module {
        let constants = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                format!("    %value{index} = adt.string_const \"{value}\" : tribute_rt.anyref")
            })
            .collect::<Vec<_>>()
            .join("\n");
        parse_test_module(
            ctx,
            &format!(
                "core.module @test {{\n  wasm.func @main() -> core.nil {{\n{constants}\n    wasm.return\n  }}\n}}"
            ),
        )
    }

    fn string_const_count(ctx: &IrContext, module: Module) -> usize {
        let func = module.ops(ctx)[0];
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        ctx.block(block)
            .ops
            .iter()
            .filter(|&&op| adt::StringConst::from_op(ctx, op).is_ok())
            .count()
    }

    #[test]
    fn analysis_deduplicates_and_looks_up_passive_data() {
        let mut ctx = IrContext::new();
        let module = string_module(&mut ctx, &["first", "second", "first", "", ""]);

        let analysis = analyze_consts(&ctx, module);

        assert_eq!(
            analysis.contents,
            vec![b"first".to_vec(), b"second".to_vec(), Vec::new()]
        );
    }

    #[test]
    fn analysis_shares_passive_data_between_string_and_bytes_literals() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    %string = adt.string_const "shared" : tribute_rt.anyref
    %bytes = adt.array_const dense<"0x736861726564"> {type = adt.array<core.i8>} : adt.array<core.i8>
    wasm.return
  }
}"#,
        );

        let analysis = analyze_consts(&ctx, module);

        assert_eq!(analysis.contents, vec![b"shared".to_vec()]);
    }

    #[test]
    fn string_lowering_builds_a_canonical_leaf() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Bytes = adt.struct<"std::Bytes", (adt.array<core.i8>, core.i32, core.i32)>
  !String = adt.enum<"String", [(wasm.anyref, wasm.anyref, core.i32), (!Bytes)]>
  adt.struct_decl "std::Bytes" !Bytes ["data", "start", "len"]
  adt.enum_decl "String" !String ["Branch", "Leaf"]
  wasm.func @main() -> core.nil {
    %string = adt.string_const "hello" : !String
    wasm.return
  }
}"#,
        );
        let string_ty = type_alias(&ctx, "String");
        attach_string_type(&mut ctx, module, string_ty);
        let analysis = analyze_consts(&ctx, module);

        validate_for_wasm(&ctx, module, &analysis).expect("canonical String result type");
        lower(&mut ctx, module, &analysis);

        let output = module.op().display(&ctx).to_string();
        assert!(!output.contains("adt.string_const"), "{output}");
        assert!(output.contains("wasm_gc.array_new_data"), "{output}");
        assert!(output.contains("adt.struct_new"), "{output}");
        assert!(output.contains("adt.variant_new"), "{output}");
        // The declaration, not a fixed position, locates `Leaf`.
        assert!(output.contains("adt.variant_new 1("), "{output}");
    }

    #[test]
    fn validation_rejects_string_constants_without_the_canonical_type() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    %string = adt.string_const "hello" : wasm.anyref
    wasm.return
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        assert_eq!(
            validate_for_wasm(&ctx, module, &analysis),
            Err(ConstValidationError::MissingCanonicalStringType)
        );
    }

    #[test]
    fn user_string_lookalike_cannot_substitute_for_metadata_identity() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Bytes = adt.struct<"std::Bytes", (adt.array<core.i8>, core.i32, core.i32)>
  !PreludeString = adt.enum<"String", [(!Bytes), (wasm.anyref, wasm.anyref, core.i32)]>
  !UserString = adt.enum<"user::String", [(!Bytes), (wasm.anyref, wasm.anyref, core.i32)]>
  adt.struct_decl "std::Bytes" !Bytes ["data", "start", "len"]
  adt.enum_decl "String" !PreludeString ["Leaf", "Branch"]
  adt.enum_decl "user::String" !UserString ["Leaf", "Branch"]
  wasm.func @main() -> core.nil {
    %string = adt.string_const "hello" : wasm.anyref
    wasm.return
  }
}"#,
        );
        let prelude_string = type_alias(&ctx, "PreludeString");
        attach_string_type(&mut ctx, module, prelude_string);

        let analysis = analyze_consts(&ctx, module);

        let leaf = analysis
            .layouts
            .string_leaf()
            .expect("the prelude String's leaf");
        assert_eq!((leaf.string, leaf.leaf), (prelude_string, 0));
    }

    #[test]
    fn textual_ir_round_trip_preserves_the_canonical_string_type() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Bytes = adt.struct<"std::Bytes", (adt.array<core.i8>, core.i32, core.i32)>
  !String = adt.enum<"String", [(!Bytes), (wasm.anyref, wasm.anyref, core.i32)]>
  adt.struct_decl "std::Bytes" !Bytes ["data", "start", "len"]
  adt.enum_decl "String" !String ["Leaf", "Branch"]
  wasm.func @main() -> core.nil {
    %string = adt.string_const "hello" : !String
    wasm.return
  }
}"#,
        );
        let string_ty = type_alias(&ctx, "String");
        attach_string_type(&mut ctx, module, string_ty);

        let printed = module.op().display(&ctx).to_string();
        let mut reparsed_ctx = IrContext::new();
        let reparsed = parse_test_module(&mut reparsed_ctx, &printed);
        let analysis = analyze_consts(&reparsed_ctx, reparsed);

        assert_eq!(
            analysis
                .layouts
                .string_leaf()
                .map(|leaf| (leaf.string, leaf.leaf)),
            Some((type_alias(&reparsed_ctx, "String"), 0))
        );
        assert_eq!(
            validate_for_wasm(&reparsed_ctx, reparsed, &analysis),
            Ok(())
        );
    }

    #[test]
    fn string_lowering_preserves_constants_when_analysis_is_incomplete() {
        let mut missing_type_ctx = IrContext::new();
        let missing_type_module = string_module(&mut missing_type_ctx, &["hello"]);
        lower(
            &mut missing_type_ctx,
            missing_type_module,
            &ConstAnalysis {
                contents: vec![b"hello".to_vec()],
                layouts: LiteralLayouts::default(),
            },
        );
        assert_eq!(
            string_const_count(&missing_type_ctx, missing_type_module),
            1
        );

        assert_eq!(
            trunk_ir::rewrite::RewritePattern::name(&ArrayConstPattern::new(
                &DataSegments::default()
            )),
            "ArrayConstPattern"
        );
    }

    #[test]
    fn bytes_lowering_builds_its_representation_by_field_names() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Repr = adt.struct<"std::Bytes::impl::Repr", (core.i32, adt.array<core.i8>, core.i32)>
  adt.struct_decl "std::Bytes::impl::Repr" !Repr ["len", "data", "start"]
  wasm.func @main() -> core.nil {
    %bytes = adt.bytes_const b"hello" : !Repr
    wasm.return
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis);

        let output = module.op().display(&ctx).to_string();
        assert!(!output.contains("adt.bytes_const"), "{output}");
        assert!(!output.contains("adt.array_const"), "{output}");
        assert_eq!(bytes_from_data(&ctx, module), [(0, 5)], "{output}");
        // `len`, `data`, and `start` are placed by their declared names.
        let constant = |value| {
            let trunk_ir::refs::ValueDef::OpResult(def, _) = ctx.value_def(value) else {
                return None;
            };
            wasm_dialect::I32Const::from_op(&ctx, def)
                .ok()
                .map(|constant| constant.value(&ctx))
        };
        let fields = module_body_ops(&ctx, module)
            .into_iter()
            .find_map(|op| adt::StructNew::from_op(&ctx, op).ok())
            .map(|record| ctx.op_operands(record.op_ref()).to_vec())
            .expect("the representation is built");
        assert_eq!(
            [
                constant(fields[0]),
                constant(fields[1]),
                constant(fields[2])
            ],
            [Some(5), None, Some(0)],
            "{output}"
        );
    }

    fn module_body_ops(ctx: &IrContext, module: Module) -> Vec<OpRef> {
        let func = module
            .ops(ctx)
            .iter()
            .copied()
            .find(|&op| wasm_dialect::Func::from_op(ctx, op).is_ok())
            .expect("wasm.func");
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        ctx.block(block).ops.to_vec()
    }
    #[test]
    fn lowering_declares_one_passive_segment_per_payload() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    %first = adt.array_const dense<"0x6669727374"> {type = adt.array<core.i8>} : adt.array<core.i8>
    %second = adt.array_const dense<"0x7365636F6E64"> {type = adt.array<core.i8>} : adt.array<core.i8>
    %again = adt.array_const dense<"0x6669727374"> {type = adt.array<core.i8>} : adt.array<core.i8>
    wasm.return
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis);

        let output = module.op().display(&ctx).to_string();
        assert_eq!(
            data_segments(&ctx, module),
            [b"first".to_vec(), b"second".to_vec()]
        );
        assert_eq!(
            bytes_from_data(&ctx, module),
            [(0, 5), (1, 6), (0, 5)],
            "{output}"
        );
    }

    #[test]
    fn lowering_reuses_existing_passive_segments() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.data(b"active") offset 0
  wasm.data(b"kept") offset 0 passive
  wasm.func @main() -> core.nil {
    %new = adt.array_const dense<"0x6E6577"> {type = adt.array<core.i8>} : adt.array<core.i8>
    %kept = adt.array_const dense<"0x6B657074"> {type = adt.array<core.i8>} : adt.array<core.i8>
    %active = adt.array_const dense<"0x616374697665"> {type = adt.array<core.i8>} : adt.array<core.i8>
    wasm.return
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis);

        let output = module.op().display(&ctx).to_string();
        assert_eq!(
            data_segments(&ctx, module),
            [
                b"active".to_vec(),
                b"kept".to_vec(),
                b"new".to_vec(),
                b"active".to_vec()
            ],
            "{output}"
        );
        assert_eq!(
            bytes_from_data(&ctx, module),
            [(2, 3), (1, 4), (3, 6)],
            "{output}"
        );
    }

    fn data_segments(ctx: &IrContext, module: Module) -> Vec<Vec<u8>> {
        module
            .ops(ctx)
            .iter()
            .copied()
            .filter_map(|op| wasm_dialect::Data::from_op(ctx, op).ok())
            .map(|data| data.bytes(ctx).to_vec())
            .collect()
    }

    fn bytes_from_data(ctx: &IrContext, module: Module) -> Vec<(u32, u32)> {
        let func = module
            .ops(ctx)
            .iter()
            .copied()
            .find(|&op| wasm_dialect::Func::from_op(ctx, op).is_ok())
            .expect("wasm.func");
        let body = ctx.op_region(func, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        ctx.block(block)
            .ops
            .iter()
            .filter_map(|&op| wasm_gc::ArrayNewData::from_op(ctx, op).ok())
            .map(|array| {
                let trunk_ir::refs::ValueDef::OpResult(def, _) = ctx.value_def(array.size(ctx))
                else {
                    panic!("the size is a constant");
                };
                let size =
                    wasm_dialect::I32Const::from_op(ctx, def).expect("the size is a constant");
                (array.data_idx(ctx), size.value(ctx) as u32)
            })
            .collect()
    }
}

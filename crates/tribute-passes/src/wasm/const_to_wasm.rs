//! Lower adt.string_const and adt.bytes_const to wasm data segments.
//!
//! This pass uses a two-phase approach:
//! 1. Analysis: Collect the distinct string/bytes payloads of the input IR
//! 2. Transform: Declare a passive `wasm.data` segment for each payload and
//!    replace const operations with wasm ops that reference it
//!
//! A data index is the position of its segment among the module's `wasm.data`
//! operations, so later steps read the segments from the IR itself.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::Attribute;

#[derive(Debug, PartialEq, Eq)]
pub enum ConstValidationError {
    MissingCanonicalStringType,
    InvalidStringResultType { actual: String },
}

impl fmt::Display for ConstValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCanonicalStringType => {
                f.write_str("adt.string_const requires the canonical prelude String type")
            }
            Self::InvalidStringResultType { actual } => write!(
                f,
                "adt.string_const must produce wasm.anyref before canonical String lowering, found {actual}"
            ),
        }
    }
}

impl std::error::Error for ConstValidationError {}

/// Result of constant analysis.
pub struct ConstAnalysis {
    /// Distinct string/bytes payloads, in first-occurrence order.
    pub contents: Vec<Vec<u8>>,
    /// Canonical prelude String enum type, when string constants are present.
    pub(crate) string_enum_ty: Option<trunk_ir::TypeRef>,
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
            seen: HashSet::new(),
            has_string_consts: false,
        }
    }

    fn collect_content(&mut self, bytes: Vec<u8>) {
        if !self.seen.contains(&bytes) {
            self.seen.insert(bytes.clone());
            self.contents.push(bytes);
        }
    }

    fn visit_op(&mut self, ctx: &IrContext, op: OpRef) {
        let data = ctx.op(op);

        if data.dialect == adt::DIALECT_NAME() {
            if data.name == Symbol::new("string_const") {
                if let Some(s) = data.attributes.get_str(ctx, "value") {
                    self.has_string_consts = true;
                    self.collect_content(s.as_bytes().to_vec());
                }
            } else if data.name == Symbol::new("bytes_const")
                && let Some(Attribute::Bytes(b)) = data.attributes.get("value")
            {
                self.collect_content(b.to_vec());
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
pub fn analyze_consts(ctx: &IrContext, module: Module) -> ConstAnalysis {
    let mut collector = ConstCollector::new();

    // Walk all operations in module body (recursively into nested regions)
    if let Some(body) = module.body(ctx) {
        walk_ops_in_region(ctx, body, &mut |ctx, op| {
            collector.visit_op(ctx, op);
        });
    }

    ConstAnalysis {
        contents: collector.contents,
        string_enum_ty: collector
            .has_string_consts
            .then(|| tribute_ir::metadata::WellKnownTypes::from_module(ctx, module.op()).string)
            .flatten(),
    }
}

/// Validate the representation expected at the Wasm constant-lowering boundary.
///
/// This runs after primitive type normalization, so source-level
/// `tribute_rt.anyref` results must already be represented as `wasm.anyref`.
pub fn validate_for_wasm(
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
        if analysis.string_enum_ty.is_none() {
            result = Err(ConstValidationError::MissingCanonicalStringType);
            return;
        }

        let Some(&result_ty) = ctx.op_result_types(op).first() else {
            result = Err(ConstValidationError::InvalidStringResultType {
                actual: "<missing>".to_owned(),
            });
            return;
        };
        let ty = ctx.get_type(result_ty);
        if ty.dialect != wasm_dialect::DIALECT_NAME() || ty.name != Symbol::new("anyref") {
            result = Err(ConstValidationError::InvalidStringResultType {
                actual: format!("{}.{}", ty.dialect, ty.name),
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
pub fn lower(ctx: &mut IrContext, module: Module, analysis: &ConstAnalysis) {
    let segments = Rc::new(declare_data_segments(ctx, module, &analysis.contents));

    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(StringConstPattern::new(
            segments.clone(),
            analysis.string_enum_ty,
        ))
        .add_pattern(BytesConstPattern::new(segments));
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
    let mut segments = DataSegments::new();
    let Some(module_block) = module.first_block(ctx) else {
        return segments;
    };

    let mut next_idx = 0;
    for &op in ctx.block(module_block).ops.iter() {
        let Ok(data) = wasm_dialect::Data::from_op(ctx, op) else {
            continue;
        };
        if data.passive(ctx) {
            segments.entry(data.bytes(ctx).to_vec()).or_insert(next_idx);
        }
        next_idx += 1;
    }

    let location = ctx.op(module.op()).location;
    for content in contents {
        let Entry::Vacant(entry) = segments.entry(content.clone()) else {
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

/// Pattern for `adt.string_const` -> `String::Leaf(wasm.bytes_from_data)`.
struct StringConstPattern {
    segments: Rc<DataSegments>,
    string_enum_ty: Option<trunk_ir::TypeRef>,
}

impl StringConstPattern {
    fn new(segments: Rc<DataSegments>, string_enum_ty: Option<trunk_ir::TypeRef>) -> Self {
        Self {
            segments,
            string_enum_ty,
        }
    }
}

impl RewritePattern for StringConstPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(string_const) = adt::StringConst::from_op(ctx, op) else {
            return false;
        };

        let value_str = string_const.value(ctx);
        let content = value_str.as_bytes().to_vec();

        let Some((data_idx, len)) = lookup_segment(&self.segments, &content) else {
            return false;
        };
        let Some(string_enum_ty) = self.string_enum_ty else {
            tracing::warn!("const_to_wasm: canonical String enum type not found");
            return false;
        };

        let location = ctx.op(op).location;
        let bytes_ty = super::bytes::bytes_struct_type(ctx);
        let bytes = wasm_dialect::BytesFromData::operands()
            .data_idx(data_idx)
            .offset(0)
            .len(len)
            .results(bytes_ty)
            .build(ctx, location);
        let result_ty = ctx.op_result_types(op)[0];
        let leaf = adt::VariantNew::operands([bytes.result(ctx)])
            .r#type(string_enum_ty)
            .tag(Symbol::new("Leaf"))
            .results(result_ty)
            .build(ctx, location);

        rewriter.insert_op(bytes.op_ref());
        rewriter.replace_op(leaf.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "StringConstPattern"
    }
}

/// Pattern for `adt.bytes_const` -> `wasm.bytes_from_data`
struct BytesConstPattern {
    segments: Rc<DataSegments>,
}

impl BytesConstPattern {
    fn new(segments: Rc<DataSegments>) -> Self {
        Self { segments }
    }
}

impl RewritePattern for BytesConstPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(bytes_const) = adt::BytesConst::from_op(ctx, op) else {
            return false;
        };

        let b = bytes_const.value(ctx);
        let content: Vec<u8> = b.to_vec();

        let Some((data_idx, len)) = lookup_segment(&self.segments, &content) else {
            return false;
        };

        let location = ctx.op(op).location;
        let bytes_ty = super::bytes::bytes_struct_type(ctx);

        // Create wasm.bytes_from_data operation
        let new_op = wasm_dialect::BytesFromData::operands()
            .data_idx(data_idx)
            .offset(0)
            .len(len)
            .results(bytes_ty)
            .build(ctx, location);

        rewriter.replace_op(new_op.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "BytesConstPattern"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::types::TypeDataBuilder;

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
                format!(
                    "    %value{index} = adt.string_const {{value = \"{value}\"}} : tribute_rt.anyref"
                )
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
        let module = string_module(&mut ctx, &["hello", "hello"]);

        let analysis = analyze_consts(&ctx, module);

        assert_eq!(analysis.contents, vec![b"hello".to_vec()]);
    }

    #[test]
    fn analysis_shares_passive_data_between_string_and_bytes_literals() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    %string = adt.string_const {value = "shared"} : tribute_rt.anyref
    %bytes = adt.bytes_const {value = b"shared"} : core.bytes
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
  !String = adt.enum<{name = @String, variants = [[@Leaf, [core.bytes]], [@Branch, [wasm.anyref, wasm.anyref, core.i32]]]}>
  wasm.func @main() -> core.nil {
    %string = adt.string_const {value = "hello"} : wasm.anyref
    wasm.return
  }
}"#,
        );
        let string_ty = type_alias(&ctx, "String");
        attach_string_type(&mut ctx, module, string_ty);
        let analysis = analyze_consts(&ctx, module);

        validate_for_wasm(&ctx, module, &analysis).expect("canonical String result type");
        lower(&mut ctx, module, &analysis);

        let output = trunk_ir::printer::print_module(&ctx, module.op());
        assert!(!output.contains("adt.string_const"), "{output}");
        assert!(output.contains("wasm.bytes_from_data"), "{output}");
        assert!(output.contains("adt.variant_new"), "{output}");
        assert!(output.contains("tag = @Leaf"), "{output}");
    }

    #[test]
    fn validation_rejects_string_constants_without_the_canonical_type() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    %string = adt.string_const {value = "hello"} : wasm.anyref
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
  !PreludeString = adt.enum<{name = @String, variants = [[@Leaf, [core.bytes]], [@Branch, [wasm.anyref, wasm.anyref, core.i32]]]}>
  !UserString = adt.enum<{name = @"user::String", variants = [[@Leaf, [core.bytes]], [@Branch, [wasm.anyref, wasm.anyref, core.i32]]]}>
  wasm.func @main() -> core.nil {
    %string = adt.string_const {value = "hello"} : wasm.anyref
    wasm.return
  }
}"#,
        );
        let prelude_string = type_alias(&ctx, "PreludeString");
        let user_string = type_alias(&ctx, "UserString");
        attach_string_type(&mut ctx, module, prelude_string);

        let analysis = analyze_consts(&ctx, module);

        assert_eq!(analysis.string_enum_ty, Some(prelude_string));
        assert_ne!(analysis.string_enum_ty, Some(user_string));
    }

    #[test]
    fn textual_ir_round_trip_preserves_the_canonical_string_type() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !String = adt.enum<{name = @String, variants = [[@Leaf, [core.bytes]], [@Branch, [wasm.anyref, wasm.anyref, core.i32]]]}>
  wasm.func @main() -> core.nil {
    %string = adt.string_const {value = "hello"} : wasm.anyref
    wasm.return
  }
}"#,
        );
        let string_ty = type_alias(&ctx, "String");
        attach_string_type(&mut ctx, module, string_ty);

        let printed = trunk_ir::printer::print_module(&ctx, module.op());
        let mut reparsed_ctx = IrContext::new();
        let reparsed = parse_test_module(&mut reparsed_ctx, &printed);
        let analysis = analyze_consts(&reparsed_ctx, reparsed);

        assert_eq!(
            analysis.string_enum_ty,
            Some(type_alias(&reparsed_ctx, "String"))
        );
        assert_eq!(
            validate_for_wasm(&reparsed_ctx, reparsed, &analysis),
            Ok(())
        );
    }

    #[test]
    fn string_lowering_preserves_constants_when_analysis_is_incomplete() {
        let mut missing_data_ctx = IrContext::new();
        let missing_data_module = string_module(&mut missing_data_ctx, &["hello"]);
        let placeholder_ty =
            missing_data_ctx.intern_type(TypeDataBuilder::new("adt", "enum").build());
        lower(
            &mut missing_data_ctx,
            missing_data_module,
            &ConstAnalysis {
                contents: Vec::new(),
                string_enum_ty: Some(placeholder_ty),
            },
        );
        assert_eq!(
            string_const_count(&missing_data_ctx, missing_data_module),
            1
        );

        let mut missing_type_ctx = IrContext::new();
        let missing_type_module = string_module(&mut missing_type_ctx, &["hello"]);
        lower(
            &mut missing_type_ctx,
            missing_type_module,
            &ConstAnalysis {
                contents: vec![b"hello".to_vec()],
                string_enum_ty: None,
            },
        );
        assert_eq!(
            string_const_count(&missing_type_ctx, missing_type_module),
            1
        );

        assert_eq!(
            StringConstPattern::new(Rc::default(), None).name(),
            "StringConstPattern"
        );
        assert_eq!(
            BytesConstPattern::new(Rc::default()).name(),
            "BytesConstPattern"
        );
    }
    #[test]
    fn lowering_declares_one_passive_segment_per_payload() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    %first = adt.bytes_const {value = b"first"} : core.bytes
    %second = adt.bytes_const {value = b"second"} : core.bytes
    %again = adt.bytes_const {value = b"first"} : core.bytes
    wasm.return
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis);

        let output = trunk_ir::printer::print_module(&ctx, module.op());
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
  wasm.data {offset = 0, bytes = b"active", passive = false}
  wasm.data {offset = 0, bytes = b"kept", passive = true}
  wasm.func @main() -> core.nil {
    %new = adt.bytes_const {value = b"new"} : core.bytes
    %kept = adt.bytes_const {value = b"kept"} : core.bytes
    %active = adt.bytes_const {value = b"active"} : core.bytes
    wasm.return
  }
}"#,
        );
        let analysis = analyze_consts(&ctx, module);

        lower(&mut ctx, module, &analysis);

        let output = trunk_ir::printer::print_module(&ctx, module.op());
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
            .filter_map(|&op| wasm_dialect::BytesFromData::from_op(ctx, op).ok())
            .map(|bytes| (bytes.data_idx(ctx), bytes.len(ctx)))
            .collect()
    }
}

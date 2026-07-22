//! Lower adt.string_const and adt.bytes_const to wasm data segments.
//!
//! This pass uses a two-phase approach:
//! 1. Analysis: Collect all string/bytes constants and allocate data segment offsets
//! 2. Transform: Replace const operations with wasm ops
//!
//! The analysis produces a plain struct (no Salsa tracking).

use std::collections::HashMap;
use std::fmt;

use trunk_ir::Symbol;
use trunk_ir::adt_layout::get_enum_variants;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::adt;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef};
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};
use trunk_ir::types::Attribute;
use trunk_ir::types::TypeDataBuilder;

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
    /// Shared passive data allocations: (content, data_idx, length).
    pub allocations: Vec<(Vec<u8>, u32, u32)>,
    /// Canonical prelude String enum type, when string constants are present.
    pub(crate) string_enum_ty: Option<trunk_ir::TypeRef>,
}

impl ConstAnalysis {
    /// Passive data segments do not reserve linear memory.
    pub fn total_size(&self) -> u32 {
        0
    }

    /// Look up the passive data segment for the given content.
    pub fn data_info_for(&self, content: &[u8]) -> Option<(u32, u32)> {
        self.allocations
            .iter()
            .find(|(data, _, _)| data.as_slice() == content)
            .map(|(_, data_idx, len)| (*data_idx, *len))
    }
}

/// Context for collecting const allocations during analysis.
struct ConstCollector {
    allocations: Vec<(Vec<u8>, u32, u32)>,
    seen: HashMap<Vec<u8>, usize>,
    has_string_consts: bool,
}

impl ConstCollector {
    fn new() -> Self {
        Self {
            allocations: Vec::new(),
            seen: HashMap::new(),
            has_string_consts: false,
        }
    }

    fn collect_content(&mut self, bytes: Vec<u8>) {
        if self.seen.contains_key(&bytes) {
            return;
        }
        let data_idx = self.allocations.len() as u32;
        let len = bytes.len() as u32;
        self.seen.insert(bytes.clone(), self.allocations.len());
        self.allocations.push((bytes, data_idx, len));
    }

    fn visit_op(&mut self, ctx: &IrContext, op: OpRef) {
        let data = ctx.op(op);

        if data.dialect == adt::DIALECT_NAME() {
            if data.name == Symbol::new("string_const") {
                if let Some(s) = data.attributes.get_str("value") {
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

/// Find the canonical prelude String enum type.
fn find_string_enum_type(ctx: &IrContext) -> Option<trunk_ir::TypeRef> {
    ctx.types.iter().find_map(|(ty_ref, data)| {
        if data.dialect != "adt" || data.name != "enum" {
            return None;
        }
        if data.attrs.get_symbol("name") != Some(Symbol::new("String")) {
            return None;
        }
        // TODO(#790): Consume the canonical prelude String TypeRef carried from
        // the frontend instead of identifying it by name and structural layout.
        let variants = get_enum_variants(ctx, ty_ref)?;
        let [(leaf_tag, leaf_fields), (branch_tag, branch_fields)] = variants.as_slice() else {
            return None;
        };
        let [leaf_ty] = leaf_fields.as_slice() else {
            return None;
        };
        let [left_ty, right_ty, length_ty] = branch_fields.as_slice() else {
            return None;
        };

        let is_type = |ty, dialect, name| {
            let field = ctx.types.get(ty);
            field.dialect == dialect && field.name == name
        };
        let is_anyref = |ty| is_type(ty, "tribute_rt", "anyref") || is_type(ty, "wasm", "anyref");

        (leaf_tag == "Leaf"
            && branch_tag == "Branch"
            && is_type(*leaf_ty, "core", "bytes")
            && is_anyref(*left_ty)
            && is_anyref(*right_ty)
            && is_type(*length_ty, "core", "i32"))
        .then_some(ty_ref)
    })
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
            for &nested in ctx.op(op).regions.iter() {
                walk_ops_in_region(ctx, nested, callback);
            }
        }
    }
}

/// Analyze a module to collect all string/bytes constants and allocate offsets.
pub fn analyze_consts(ctx: &IrContext, module: Module) -> ConstAnalysis {
    let mut collector = ConstCollector::new();

    // Walk all operations in module body (recursively into nested regions)
    if let Some(body) = module.body(ctx) {
        walk_ops_in_region(ctx, body, &mut |ctx, op| {
            collector.visit_op(ctx, op);
        });
    }

    ConstAnalysis {
        allocations: collector.allocations,
        string_enum_ty: collector
            .has_string_consts
            .then(|| find_string_enum_type(ctx))
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
        let ty = ctx.types.get(result_ty);
        if ty.dialect != wasm_dialect::DIALECT_NAME() || ty.name != Symbol::new("anyref") {
            result = Err(ConstValidationError::InvalidStringResultType {
                actual: format!("{}.{}", ty.dialect, ty.name),
            });
        }
    });
    result
}

/// Lower const operations using pre-computed analysis.
pub fn lower(ctx: &mut IrContext, module: Module, analysis: &ConstAnalysis) {
    let allocations = analysis.allocations.clone();

    let applicator = PatternApplicator::new(TypeConverter::new())
        .add_pattern(StringConstPattern::new(
            allocations.clone(),
            analysis.string_enum_ty,
        ))
        .add_pattern(BytesConstPattern::new(allocations));
    applicator.apply_partial(ctx, module);
}

/// Allocation data: (content, data index, length).
type Allocations = Vec<(Vec<u8>, u32, u32)>;

/// Look up data index and length for given content.
fn lookup_offset(allocations: &Allocations, content: &[u8]) -> Option<(u32, u32)> {
    allocations
        .iter()
        .find(|(data, _, _)| data.as_slice() == content)
        .map(|(_, data_idx, len)| (*data_idx, *len))
}

/// Pattern for `adt.string_const` -> `String::Leaf(wasm.bytes_from_data)`.
struct StringConstPattern {
    allocations: Allocations,
    string_enum_ty: Option<trunk_ir::TypeRef>,
}

impl StringConstPattern {
    fn new(allocations: Allocations, string_enum_ty: Option<trunk_ir::TypeRef>) -> Self {
        Self {
            allocations,
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
        let content = value_str.into_bytes();

        let Some((data_idx, len)) = lookup_offset(&self.allocations, &content) else {
            return false;
        };
        let Some(string_enum_ty) = self.string_enum_ty else {
            tracing::warn!("const_to_wasm: canonical String enum type not found");
            return false;
        };

        let location = ctx.op(op).location;
        let bytes_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("bytes")).build());
        let bytes = wasm_dialect::bytes_from_data(ctx, location, bytes_ty, data_idx, 0, len);
        let result_ty = ctx.op_result_types(op)[0];
        let leaf = adt::variant_new(
            ctx,
            location,
            [bytes.result(ctx)],
            result_ty,
            string_enum_ty,
            Symbol::new("Leaf"),
        );

        rewriter.insert_op(bytes.op_ref());
        rewriter.replace_op(leaf.op_ref());
        true
    }

    fn name(&self) -> &'static str {
        "StringConstPattern"
    }
}

/// Bytes allocation data: (content, data_idx, length).
type BytesAllocations = Vec<(Vec<u8>, u32, u32)>;

/// Look up data_idx and length for given bytes content.
fn lookup_bytes_info(allocations: &BytesAllocations, content: &[u8]) -> Option<(u32, u32)> {
    allocations
        .iter()
        .find(|(data, _, _)| data.as_slice() == content)
        .map(|(_, data_idx, len)| (*data_idx, *len))
}

/// Pattern for `adt.bytes_const` -> `wasm.bytes_from_data`
struct BytesConstPattern {
    allocations: BytesAllocations,
}

impl BytesConstPattern {
    fn new(allocations: BytesAllocations) -> Self {
        Self { allocations }
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

        let Some((data_idx, len)) = lookup_bytes_info(&self.allocations, &content) else {
            return false;
        };

        let location = ctx.op(op).location;
        let bytes_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("bytes")).build());

        // Create wasm.bytes_from_data operation
        let new_op = wasm_dialect::bytes_from_data(ctx, location, bytes_ty, data_idx, 0, len);

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
        let body = ctx.op(func).regions[0];
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

        assert_eq!(analysis.allocations, vec![(b"hello".to_vec(), 0, 5)]);
        assert_eq!(analysis.data_info_for(b"hello"), Some((0, 5)));
        assert_eq!(analysis.data_info_for(b"missing"), None);
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

        assert_eq!(analysis.allocations, vec![(b"shared".to_vec(), 0, 6)]);
        assert_eq!(analysis.data_info_for(b"shared"), Some((0, 6)));
    }

    #[test]
    fn string_lowering_builds_a_canonical_leaf() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !String = adt.enum() {name = @String, variants = [[@Leaf, [core.bytes]], [@Branch, [wasm.anyref, wasm.anyref, core.i32]]]}
  wasm.func @main() -> core.nil {
    %string = adt.string_const {value = "hello"} : wasm.anyref
    wasm.return
  }
}"#,
        );
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
    fn validation_rejects_malformed_string_layouts() {
        for variants in [
            "[[@Leaf, [core.bytes]], [@Branch, [wasm.anyref, core.i32, core.i32]]]",
            "[[@Leaf, [core.bytes]], [@Branch, [wasm.anyref, wasm.anyref, core.i32]], [@Extra, []]]",
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  !String = adt.enum() {{name = @String, variants = {variants}}}
  wasm.func @main() -> core.nil {{
    %string = adt.string_const {{value = "hello"}} : wasm.anyref
    wasm.return
  }}
}}"#
                ),
            );
            let analysis = analyze_consts(&ctx, module);

            assert_eq!(
                validate_for_wasm(&ctx, module, &analysis),
                Err(ConstValidationError::MissingCanonicalStringType),
                "accepted malformed variants: {variants}"
            );
        }
    }

    #[test]
    fn string_lowering_preserves_constants_when_analysis_is_incomplete() {
        let mut missing_data_ctx = IrContext::new();
        let missing_data_module = string_module(&mut missing_data_ctx, &["hello"]);
        let placeholder_ty = missing_data_ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("enum")).build());
        lower(
            &mut missing_data_ctx,
            missing_data_module,
            &ConstAnalysis {
                allocations: Vec::new(),
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
                allocations: vec![(b"hello".to_vec(), 0, 5)],
                string_enum_ty: None,
            },
        );
        assert_eq!(
            string_const_count(&missing_type_ctx, missing_type_module),
            1
        );

        assert_eq!(
            StringConstPattern::new(Vec::new(), None).name(),
            "StringConstPattern"
        );
        assert_eq!(
            BytesConstPattern::new(Vec::new()).name(),
            "BytesConstPattern"
        );
    }
}

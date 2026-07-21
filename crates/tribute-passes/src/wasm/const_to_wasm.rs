//! Lower adt.string_const and adt.bytes_const to wasm data segments.
//!
//! This pass uses a two-phase approach:
//! 1. Analysis: Collect all string/bytes constants and allocate data segment offsets
//! 2. Transform: Replace const operations with wasm ops
//!
//! The analysis produces a plain struct (no Salsa tracking).

use std::collections::HashMap;

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
                if let Some(Attribute::String(s)) = data.attributes.get(&Symbol::new("value")) {
                    self.has_string_consts = true;
                    self.collect_content(s.clone().into_bytes());
                }
            } else if data.name == Symbol::new("bytes_const")
                && let Some(Attribute::Bytes(b)) = data.attributes.get(&Symbol::new("value"))
            {
                self.collect_content(b.to_vec());
            }
        }
    }
}

/// Find the canonical prelude String enum type.
fn find_string_enum_type(ctx: &IrContext) -> Option<trunk_ir::TypeRef> {
    ctx.types.iter().find_map(|(ty_ref, data)| {
        if data.dialect != Symbol::new("adt") || data.name != Symbol::new("enum") {
            return None;
        }
        if !matches!(
            data.attrs.get(&Symbol::new("name")),
            Some(Attribute::Symbol(name)) if *name == Symbol::new("String")
        ) {
            return None;
        }
        let variants = get_enum_variants(ctx, ty_ref)?;
        let has_bytes_leaf = variants.iter().any(|(tag, fields)| {
            *tag == Symbol::new("Leaf") && fields.len() == 1 && {
                let field = ctx.types.get(fields[0]);
                field.dialect == Symbol::new("core") && field.name == Symbol::new("bytes")
            }
        });
        let has_branch = variants
            .iter()
            .any(|(tag, fields)| *tag == Symbol::new("Branch") && fields.len() == 3);
        (has_bytes_leaf && has_branch).then_some(ty_ref)
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

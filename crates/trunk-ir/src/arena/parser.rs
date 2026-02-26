//! Arena IR text format parser.
//!
//! Parses the textual IR format produced by [`super::printer`] into
//! arena-based IR structures. Uses shared winnow combinators from
//! [`crate::parser::raw`].
//!
//! # Two-stage parsing
//!
//! 1. **Raw parse**: winnow combinators parse text into `Raw*` structures.
//! 2. **IR build**: `ArenaIrBuilder` converts `Raw*` → arena `OpRef`,
//!    `BlockRef`, `RegionRef`, etc.

use std::collections::{BTreeMap, HashMap};

use smallvec::smallvec;
use winnow::prelude::*;

use super::context::{IrContext, OperationDataBuilder};
use super::refs::*;
use super::rewrite::ArenaModule;
use super::types::*;
use super::{BlockArgData, BlockData, RegionData};
use crate::Symbol;
use crate::parser::raw::{self, ParseError, RawAttribute, RawOperation, RawRegion, RawType};

// ============================================================================
// ArenaIrBuilder (Raw -> Arena IR)
// ============================================================================

struct ArenaIrBuilder<'a> {
    ctx: &'a mut IrContext,
    location: Location,
    /// Maps value name (without %) -> ValueRef
    value_map: HashMap<String, ValueRef>,
    /// Maps block label (without ^) -> BlockRef
    block_map: HashMap<String, BlockRef>,
}

impl<'a> ArenaIrBuilder<'a> {
    fn new(ctx: &'a mut IrContext) -> Self {
        let path = ctx.paths.intern("textual-ir".to_owned());
        let location = Location::new(path, crate::location::Span::new(0, 0));
        Self {
            ctx,
            location,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
        }
    }

    // ----------------------------------------------------------------
    // Type / Attribute conversion
    // ----------------------------------------------------------------

    fn build_type(&mut self, raw: &RawType<'_>) -> TypeRef {
        let dialect = Symbol::from_dynamic(raw.dialect);
        let name = Symbol::from_dynamic(raw.name);
        let params: Vec<TypeRef> = raw.params.iter().map(|p| self.build_type(p)).collect();
        let attrs: BTreeMap<Symbol, Attribute> = raw
            .attrs
            .iter()
            .map(|(k, v)| (Symbol::from_dynamic(k), self.build_attribute(v)))
            .collect();

        let mut builder = TypeDataBuilder::new(dialect, name);
        for p in params {
            builder = builder.param(p);
        }
        for (k, v) in attrs {
            builder = builder.attr(k, v);
        }
        self.ctx.types.intern(builder.build())
    }

    fn build_attribute(&mut self, raw: &RawAttribute<'_>) -> Attribute {
        match raw {
            RawAttribute::Bool(b) => Attribute::Bool(*b),
            RawAttribute::Int(n) => Attribute::IntBits(*n),
            RawAttribute::Float(f) => Attribute::FloatBits(f.to_bits()),
            RawAttribute::String(s) => Attribute::String(s.clone()),
            RawAttribute::Symbol(s) => Attribute::Symbol(Symbol::from_dynamic(s.as_str())),
            RawAttribute::Type(t) => Attribute::Type(self.build_type(t)),
            RawAttribute::List(items) => {
                let list: Vec<Attribute> = items.iter().map(|a| self.build_attribute(a)).collect();
                Attribute::List(list)
            }
            RawAttribute::Unit => Attribute::Unit,
            RawAttribute::Location(path, start, end) => {
                let path_ref = self.ctx.paths.intern(path.clone());
                Attribute::Location(Location::new(
                    path_ref,
                    crate::location::Span::new(*start, *end),
                ))
            }
            RawAttribute::Bytes(bytes) => Attribute::Bytes(bytes.iter().copied().collect()),
        }
    }

    // ----------------------------------------------------------------
    // Value / block resolution
    // ----------------------------------------------------------------

    fn resolve_value(&self, name: &str) -> Option<ValueRef> {
        self.value_map.get(name).copied()
    }

    fn save_scopes(&self) -> (HashMap<String, ValueRef>, HashMap<String, BlockRef>) {
        (self.value_map.clone(), self.block_map.clone())
    }

    fn restore_scopes(&mut self, saved: (HashMap<String, ValueRef>, HashMap<String, BlockRef>)) {
        self.value_map = saved.0;
        self.block_map = saved.1;
    }

    // ----------------------------------------------------------------
    // Region / Block building
    // ----------------------------------------------------------------

    /// Build a region from raw data.
    ///
    /// `extra_entry_args` are injected as the entry block's arguments
    /// (e.g. from a func-style signature).
    fn build_region(
        &mut self,
        raw: &RawRegion<'_>,
        extra_entry_args: &[(&str, RawType<'_>)],
    ) -> Result<RegionRef, ParseError> {
        let saved = self.save_scopes();
        let result = self.build_region_inner(raw, extra_entry_args);
        self.restore_scopes(saved);
        result
    }

    fn build_region_inner(
        &mut self,
        raw: &RawRegion<'_>,
        extra_entry_args: &[(&str, RawType<'_>)],
    ) -> Result<RegionRef, ParseError> {
        // --- Pass 1: Pre-create all blocks (with args) to get BlockRefs ---
        let mut seen_labels = std::collections::HashSet::new();
        let mut block_refs = Vec::with_capacity(raw.blocks.len());

        for (i, raw_block) in raw.blocks.iter().enumerate() {
            let label = raw_block.label.to_string();
            if !seen_labels.insert(label.clone()) {
                return Err(ParseError {
                    message: format!("duplicate block label '^{}'", label),
                    offset: 0,
                });
            }

            // Determine effective args (merge func_params for entry block)
            let all_args = self.resolve_block_args(i, raw_block, extra_entry_args)?;

            // Build BlockArgData
            let mut seen_names = std::collections::HashSet::new();
            let mut block_arg_data = Vec::with_capacity(all_args.len());
            let mut arg_names = Vec::with_capacity(all_args.len());

            for (j, (name, raw_ty)) in all_args.iter().enumerate() {
                if !seen_names.insert(name.to_string()) {
                    return Err(ParseError {
                        message: format!("duplicate block argument name '{}' at index {}", name, j),
                        offset: 0,
                    });
                }

                let ty = self.build_type(raw_ty);
                let is_default_name = name
                    .strip_prefix("arg")
                    .and_then(|rest| rest.parse::<usize>().ok())
                    .is_some_and(|n| n == j);

                let mut attrs = BTreeMap::new();
                if !is_default_name {
                    attrs.insert(
                        Symbol::new("bind_name"),
                        Attribute::Symbol(Symbol::from_dynamic(name)),
                    );
                }
                block_arg_data.push(BlockArgData { ty, attrs });
                arg_names.push(name.to_string());
            }

            let block_ref = self.ctx.create_block(BlockData {
                location: self.location,
                args: block_arg_data,
                ops: smallvec![],
                parent_region: None,
            });

            self.block_map.insert(label, block_ref);

            // Register arg values
            for (j, name) in arg_names.iter().enumerate() {
                let value = self.ctx.block_arg(block_ref, j as u32);
                self.value_map.insert(name.clone(), value);
            }

            block_refs.push(block_ref);
        }

        // --- Pass 2: Build operations for each block ---
        for (raw_block, &block_ref) in raw.blocks.iter().zip(block_refs.iter()) {
            for raw_op in &raw_block.ops {
                let op_ref = self.build_operation(raw_op)?;
                self.ctx.push_op(block_ref, op_ref);
            }
        }

        // Create region
        let region = self.ctx.create_region(RegionData {
            location: self.location,
            blocks: block_refs.into_iter().collect(),
            parent_op: None,
        });

        Ok(region)
    }

    /// Determine effective block args by merging func_params for the entry block.
    fn resolve_block_args<'b>(
        &mut self,
        block_index: usize,
        raw_block: &raw::RawBlock<'b>,
        extra_entry_args: &[(&'b str, RawType<'b>)],
    ) -> Result<Vec<(&'b str, RawType<'b>)>, ParseError> {
        if block_index != 0 || extra_entry_args.is_empty() {
            return Ok(raw_block.args.clone());
        }

        if raw_block.args.is_empty() {
            // No explicit args — use func_params
            return Ok(extra_entry_args.to_vec());
        }

        // Both present — validate arity
        if raw_block.args.len() != extra_entry_args.len() {
            return Err(ParseError {
                message: format!(
                    "entry block has {} args but function signature has {} params",
                    raw_block.args.len(),
                    extra_entry_args.len()
                ),
                offset: 0,
            });
        }

        // Validate types match
        for (j, ((_, block_ty), (_, param_ty))) in raw_block
            .args
            .iter()
            .zip(extra_entry_args.iter())
            .enumerate()
        {
            let bt = self.build_type(block_ty);
            let pt = self.build_type(param_ty);
            if bt != pt {
                let (bd, bn) = {
                    let d = self.ctx.types.get(bt);
                    (d.dialect, d.name)
                };
                let (pd, pn) = {
                    let d = self.ctx.types.get(pt);
                    (d.dialect, d.name)
                };
                return Err(ParseError {
                    message: format!(
                        "entry block arg {} type mismatch: block has {}.{} but function param has {}.{}",
                        j, bd, bn, pd, pn,
                    ),
                    offset: 0,
                });
            }
        }

        Ok(raw_block.args.clone())
    }

    // ----------------------------------------------------------------
    // Operation building
    // ----------------------------------------------------------------

    fn build_operation(&mut self, raw: &RawOperation<'_>) -> Result<OpRef, ParseError> {
        let dialect = Symbol::from_dynamic(raw.dialect);
        let op_name = Symbol::from_dynamic(raw.op_name);

        // Resolve operands
        let operands: Vec<ValueRef> = raw
            .operands
            .iter()
            .map(|name| {
                self.resolve_value(name).ok_or_else(|| ParseError {
                    message: format!(
                        "undefined value '%{}' in operation '{}.{}'",
                        name, raw.dialect, raw.op_name
                    ),
                    offset: 0,
                })
            })
            .collect::<Result<_, _>>()?;

        // Build result types
        let result_types: Vec<TypeRef> = raw
            .result_types
            .iter()
            .map(|t| self.build_type(t))
            .collect();

        // Build attributes from explicit attr dict
        let mut attributes: BTreeMap<Symbol, Attribute> = raw
            .attributes
            .iter()
            .map(|(k, v)| (Symbol::from_dynamic(k), self.build_attribute(v)))
            .collect();

        // Add sym_name if present
        if let Some(ref name) = raw.sym_name {
            attributes.insert(
                Symbol::new("sym_name"),
                Attribute::Symbol(Symbol::from_dynamic(name.as_str())),
            );
        }

        // Handle func-style signature → core.func type
        let has_func_signature =
            raw.return_type.is_some() || !raw.func_params.is_empty() || raw.effect_type.is_some();
        if has_func_signature {
            let return_ty = raw
                .return_type
                .as_ref()
                .map(|t| self.build_type(t))
                .unwrap_or_else(|| {
                    self.ctx.types.intern(
                        TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build(),
                    )
                });

            let mut func_type_builder =
                TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).param(return_ty);
            for (_, raw_ty) in &raw.func_params {
                let param_ty = self.build_type(raw_ty);
                func_type_builder = func_type_builder.param(param_ty);
            }
            if let Some(effect_raw) = &raw.effect_type {
                let effect_ty = self.build_type(effect_raw);
                func_type_builder =
                    func_type_builder.attr(Symbol::new("effect"), Attribute::Type(effect_ty));
            }

            let func_ty = self.ctx.types.intern(func_type_builder.build());
            attributes.insert(Symbol::new("type"), Attribute::Type(func_ty));
        }

        // Resolve successors
        let successors: Vec<BlockRef> = raw
            .successors
            .iter()
            .map(|label| {
                self.block_map
                    .get(*label)
                    .copied()
                    .ok_or_else(|| ParseError {
                        message: format!(
                            "undefined block '^{}' in successor list of '{}.{}'",
                            label, raw.dialect, raw.op_name
                        ),
                        offset: 0,
                    })
            })
            .collect::<Result<_, _>>()?;

        // Build regions (inject func_params as entry block args for the first region)
        let mut regions = Vec::with_capacity(raw.regions.len());
        for (i, r) in raw.regions.iter().enumerate() {
            let extra_args = if i == 0 && !raw.func_params.is_empty() {
                &raw.func_params[..]
            } else {
                &[]
            };
            let region = self.build_region(r, extra_args)?;
            regions.push(region);
        }

        // Validate result name count vs result type count
        if !raw.results.is_empty() && raw.results.len() != result_types.len() {
            return Err(ParseError {
                message: format!(
                    "operation '{}.{}' declares {} result names but {} result types",
                    raw.dialect,
                    raw.op_name,
                    raw.results.len(),
                    result_types.len()
                ),
                offset: 0,
            });
        }

        // Assemble OperationData via builder
        let mut builder = OperationDataBuilder::new(self.location, dialect, op_name);
        for v in operands {
            builder = builder.operand(v);
        }
        for ty in result_types {
            builder = builder.result(ty);
        }
        for (k, v) in attributes {
            builder = builder.attr(k, v);
        }
        for r in regions {
            builder = builder.region(r);
        }
        for b in successors {
            builder = builder.successor(b);
        }
        let data = builder.build(self.ctx);
        let op_ref = self.ctx.create_op(data);

        // Register result values, rejecting duplicates
        for (i, name) in raw.results.iter().enumerate() {
            if self.value_map.contains_key(*name) {
                return Err(ParseError {
                    message: format!(
                        "duplicate SSA name '{}' in operation '{}.{}' result index {}",
                        name, raw.dialect, raw.op_name, i
                    ),
                    offset: 0,
                });
            }
            let value = self.ctx.op_result(op_ref, i as u32);
            self.value_map.insert(name.to_string(), value);
        }

        Ok(op_ref)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Parse a TrunkIR operation from its textual representation into arena IR.
pub fn parse_module(ctx: &mut IrContext, input: &str) -> Result<OpRef, ParseError> {
    let mut remaining = input;
    raw::ws.parse_next(&mut remaining).map_err(|e| ParseError {
        message: format!("lexer error: {}", e),
        offset: input.len() - remaining.len(),
    })?;

    let raw_op = raw::raw_operation
        .parse_next(&mut remaining)
        .map_err(|e| ParseError {
            message: format!("parse error: {}", e),
            offset: input.len() - remaining.len(),
        })?;

    // Reject trailing input
    raw::ws.parse_next(&mut remaining).map_err(|e| ParseError {
        message: format!("lexer error: {}", e),
        offset: input.len() - remaining.len(),
    })?;
    if !remaining.is_empty() {
        return Err(ParseError {
            message: "trailing input after top-level operation".to_string(),
            offset: input.len() - remaining.len(),
        });
    }

    let mut builder = ArenaIrBuilder::new(ctx);
    builder.build_operation(&raw_op)
}

/// Parse textual IR into an [`ArenaModule`], panicking on failure.
///
/// Convenience wrapper around [`parse_module`] for tests.
pub fn parse_test_module(ctx: &mut IrContext, input: &str) -> ArenaModule {
    let op = parse_module(ctx, input).unwrap_or_else(|e| {
        panic!(
            "Failed to parse test IR at offset {}:\n  {}\n\nInput:\n{}",
            e.offset, e.message, input
        );
    });
    ArenaModule::new(ctx, op).unwrap_or_else(|| {
        panic!(
            "Parsed operation is not a core.module.\n\nInput:\n{}",
            input
        );
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::dialect::{arith, core, func};
    use crate::arena::printer::print_module;
    use crate::arena::validation;

    /// Build IR → print → parse → print, verify texts match.
    fn assert_roundtrip(ctx: &IrContext, module_op: OpRef) {
        let printed = print_module(ctx, module_op);
        let mut ctx2 = IrContext::new();
        let parsed_op = parse_module(&mut ctx2, &printed).unwrap_or_else(|e| {
            panic!(
                "Round-trip parse failed at offset {}:\n  {}\n\nPrinted IR:\n{}",
                e.offset, e.message, printed
            );
        });
        let reprinted = print_module(&ctx2, parsed_op);
        assert_eq!(printed, reprinted, "Round-trip mismatch");
    }

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.paths.intern("test.trb".to_owned());
        Location::new(path, crate::location::Span::new(0, 0))
    }

    fn make_i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    fn make_nil_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("nil")).build())
    }

    fn make_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(ret)
                .params(params.iter().copied())
                .build(),
        )
    }

    fn make_func_type_with_effect(
        ctx: &mut IrContext,
        params: &[TypeRef],
        ret: TypeRef,
        effect: TypeRef,
    ) -> TypeRef {
        ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func"))
                .param(ret)
                .params(params.iter().copied())
                .attr(Symbol::new("effect"), Attribute::Type(effect))
                .build(),
        )
    }

    fn wrap_in_module(ctx: &mut IrContext, loc: Location, func_ops: Vec<OpRef>) -> OpRef {
        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for op in func_ops {
            ctx.push_op(mod_block, op);
        }
        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        core::module(ctx, loc, Symbol::new("test"), mod_region).op_ref()
    }

    // ================================================================
    // Round-trip tests
    // ================================================================

    #[test]
    fn test_roundtrip_simple_module() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
        ctx.push_op(entry, c.op_ref());
        let c_val = c.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [c_val]);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let f = func::func(&mut ctx, loc, Symbol::new("main"), func_ty, body);
        let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_with_params() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let func_ty = make_func_type(&mut ctx, &[i32_ty, i32_ty], i32_ty);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: i32_ty,
                    attrs: Default::default(),
                },
                BlockArgData {
                    ty: i32_ty,
                    attrs: Default::default(),
                },
            ],
            ops: smallvec![],
            parent_region: None,
        });
        let x = ctx.block_arg(entry, 0);
        let y = ctx.block_arg(entry, 1);
        let add = arith::add(&mut ctx, loc, x, y, i32_ty);
        ctx.push_op(entry, add.op_ref());
        let add_val = add.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [add_val]);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let f = func::func(&mut ctx, loc, Symbol::new("add"), func_ty, body);
        let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_nested_regions() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let i1_ty = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i1")).build());
        let func_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);

        // Entry block with param
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let param = ctx.block_arg(entry, 0);

        // Condition
        let cond = arith::r#const(&mut ctx, loc, i1_ty, Attribute::IntBits(1));
        ctx.push_op(entry, cond.op_ref());
        let cond_val = cond.result(&ctx);

        // Then region: scf.yield %param
        let then_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let yield_then = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("yield"))
            .operand(param)
            .build(&mut ctx);
        let yield_then = ctx.create_op(yield_then);
        ctx.push_op(then_block, yield_then);
        let then_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![then_block],
            parent_op: None,
        });

        // Else region: const 1, add param+1, yield
        let else_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        ctx.push_op(else_block, c1.op_ref());
        let c1_val = c1.result(&ctx);
        let sum = arith::add(&mut ctx, loc, param, c1_val, i32_ty);
        ctx.push_op(else_block, sum.op_ref());
        let sum_val = sum.result(&ctx);
        let yield_else = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("yield"))
            .operand(sum_val)
            .build(&mut ctx);
        let yield_else = ctx.create_op(yield_else);
        ctx.push_op(else_block, yield_else);
        let else_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![else_block],
            parent_op: None,
        });

        // scf.if
        let if_op = OperationDataBuilder::new(loc, Symbol::new("scf"), Symbol::new("if"))
            .operand(cond_val)
            .result(i32_ty)
            .region(then_region)
            .region(else_region)
            .build(&mut ctx);
        let if_op = ctx.create_op(if_op);
        ctx.push_op(entry, if_op);
        let if_result = ctx.op_result(if_op, 0);

        let ret = func::r#return(&mut ctx, loc, [if_result]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let f = func::func(&mut ctx, loc, Symbol::new("choose"), func_ty, body);
        let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_effects() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let nil_ty = make_nil_type(&mut ctx);
        let func_ty = make_func_type_with_effect(&mut ctx, &[], i32_ty, nil_ty);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(7));
        ctx.push_op(entry, c.op_ref());
        let c_val = c.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [c_val]);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let f = func::func(&mut ctx, loc, Symbol::new("pure"), func_ty, body);
        let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_multiple_functions() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);

        let mut funcs = vec![];
        for (name, val) in &[("foo", 1u64), ("bar", 2)] {
            let entry = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(*val));
            ctx.push_op(entry, c.op_ref());
            let c_val = c.result(&ctx);
            let ret = func::r#return(&mut ctx, loc, [c_val]);
            ctx.push_op(entry, ret.op_ref());
            let body = ctx.create_region(RegionData {
                location: loc,
                blocks: smallvec![entry],
                parent_op: None,
            });
            let f = func::func(&mut ctx, loc, Symbol::new(name), func_ty, body);
            funcs.push(f.op_ref());
        }
        let module_op = wrap_in_module(&mut ctx, loc, funcs);

        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_func_call() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // fn callee() -> i32 { return 42 }
        let callee_ty = make_func_type(&mut ctx, &[], i32_ty);
        let entry1 = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
        ctx.push_op(entry1, c.op_ref());
        let c_val = c.result(&ctx);
        let ret1 = func::r#return(&mut ctx, loc, [c_val]);
        ctx.push_op(entry1, ret1.op_ref());
        let body1 = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry1],
            parent_op: None,
        });
        let callee = func::func(&mut ctx, loc, Symbol::new("callee"), callee_ty, body1);

        // fn main() -> i32 { call @callee, return result }
        let main_ty = make_func_type(&mut ctx, &[], i32_ty);
        let entry2 = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let call = func::call(&mut ctx, loc, [], i32_ty, Symbol::new("callee"));
        ctx.push_op(entry2, call.op_ref());
        let call_val = call.result(&ctx);
        let ret2 = func::r#return(&mut ctx, loc, [call_val]);
        ctx.push_op(entry2, ret2.op_ref());
        let body2 = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry2],
            parent_op: None,
        });
        let main_fn = func::func(&mut ctx, loc, Symbol::new("main"), main_ty, body2);

        let module_op = wrap_in_module(&mut ctx, loc, vec![callee.op_ref(), main_fn.op_ref()]);

        assert_roundtrip(&ctx, module_op);
    }

    // ================================================================
    // Error detection tests
    // ================================================================

    #[test]
    fn test_parse_undefined_operand() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    func.return %missing
  }
}"#;
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input).unwrap_err();
        assert!(
            err.message.contains("undefined value '%missing'"),
            "Expected undefined value error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_duplicate_block_label() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    ^bb0:
      %0 = arith.const {value = 1} : core.i32
      func.return %0
    ^bb0:
      %1 = arith.const {value = 2} : core.i32
      func.return %1
  }
}"#;
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input).unwrap_err();
        assert!(
            err.message.contains("duplicate block label"),
            "Expected duplicate block label error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_entry_block_arity_mismatch() {
        let input = r#"core.module @test {
  func.func @f(%0: core.i32, %1: core.i32) -> core.i32 {
    ^bb0(%2: core.i32):
      func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input).unwrap_err();
        assert!(
            err.message
                .contains("entry block has 1 args but function signature has 2 params"),
            "Expected arity mismatch error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_result_count_mismatch() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %0, %1 = arith.const {value = 42} : core.i32
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input).unwrap_err();
        assert!(
            err.message
                .contains("declares 2 result names but 1 result types"),
            "Expected result count mismatch error, got: {}",
            err.message
        );
    }

    // ================================================================
    // Arena-specific tests (validation after parse)
    // ================================================================

    #[test]
    fn test_parse_validates_value_integrity() {
        let input = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
    ^bb0:
      %1 = arith.const {value = 1} : core.i32
      %2 = arith.add %0, %1 : core.i32
      func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let result = validation::validate_value_integrity(&ctx, module);
        assert!(
            result.is_ok(),
            "Parsed module should pass value integrity: {}",
            result
        );
    }

    #[test]
    fn test_parse_validates_use_chains() {
        let input = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
    ^bb0:
      %1 = arith.const {value = 1} : core.i32
      %2 = arith.add %0, %1 : core.i32
      func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let result = validation::validate_use_chains(&ctx, module);
        assert!(
            result.is_ok(),
            "Parsed module should pass use-chain validation: {}",
            result
        );
    }

    #[test]
    fn test_parse_then_validate_all() {
        let input = r#"core.module @test {
  func.func @add(%0: core.i32, %1: core.i32) -> core.i32 {
    ^bb0:
      %2 = arith.add %0, %1 : core.i32
      func.return %2
  }
  func.func @main() -> core.i32 {
    %0 = arith.const {value = 40} : core.i32
    %1 = arith.const {value = 2} : core.i32
    %2 = func.call %0, %1 {callee = @add} : core.i32
    func.return %2
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let result = validation::validate_all(&ctx, module);
        assert!(result.is_ok(), "Full validation should pass: {}", result);
    }

    #[test]
    fn test_parse_test_module_convenience() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    %0 = arith.const {value = 99} : core.i32
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        // Verify it's a valid module
        let ops = module.ops(&ctx);
        assert_eq!(ops.len(), 1);
        let func_data = ctx.op(ops[0]);
        assert_eq!(func_data.dialect, Symbol::new("func"));
        assert_eq!(func_data.name, Symbol::new("func"));
    }
}

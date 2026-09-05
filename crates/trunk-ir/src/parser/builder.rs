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

use std::collections::HashMap;

use smallvec::smallvec;
use winnow::prelude::*;

use super::raw::{self, ParseError, RawAttribute, RawOperation, RawRegion, RawType};
use crate::Symbol;
use crate::context::{IrContext, OperationDataBuilder};
use crate::ops::DialectType;
use crate::refs::*;
use crate::rewrite::Module;
use crate::types::*;
use crate::{BlockArgData, BlockData, RegionData};

// ============================================================================
// ArenaIrBuilder (Raw -> Arena IR)
// ============================================================================

/// Scoped name resolution state, saved/restored per region.
#[derive(Clone, Default)]
struct BuilderScope {
    /// Maps value name (without %) -> ValueRef
    value_map: HashMap<String, ValueRef>,
    /// Maps block label (without ^) -> BlockRef
    block_map: HashMap<String, BlockRef>,
    /// Maps type alias name -> TypeRef
    type_alias_map: HashMap<String, TypeRef>,
}

struct ArenaIrBuilder<'a> {
    ctx: &'a mut IrContext,
    location: Location,
    scope: BuilderScope,
}

impl<'a> ArenaIrBuilder<'a> {
    fn new(ctx: &'a mut IrContext) -> Self {
        let path = ctx.paths.intern("textual-ir".to_owned());
        let location = Location::new(path, crate::location::Span::new(0, 0));
        Self {
            ctx,
            location,
            scope: BuilderScope::default(),
        }
    }

    // ----------------------------------------------------------------
    // Type / Attribute conversion
    // ----------------------------------------------------------------

    fn build_type(&mut self, raw: &RawType<'_>) -> Result<TypeRef, ParseError> {
        match raw {
            RawType::Alias(name) => self
                .scope
                .type_alias_map
                .get(name.as_str())
                .copied()
                .ok_or_else(|| ParseError {
                    message: format!("undefined type alias '!{name}'"),
                    offset: 0,
                }),
            RawType::Concrete {
                dialect,
                name,
                params,
                attrs,
            } => {
                if *dialect == "core" && *name == "func" {
                    let Some((result, inputs)) = params.split_first() else {
                        return Err(ParseError {
                            message: "legacy core.func requires a result type".to_string(),
                            offset: 0,
                        });
                    };
                    return self.build_function_type(
                        "func",
                        "func_sig",
                        inputs,
                        std::slice::from_ref(result),
                        attrs,
                    );
                }
                let dialect = Symbol::from_dynamic(dialect);
                let name = Symbol::from_dynamic(name);
                let params: Vec<TypeRef> = params
                    .iter()
                    .map(|p| self.build_type(p))
                    .collect::<Result<_, _>>()?;
                let attrs: AttributeMap = attrs
                    .iter()
                    .map(|(k, v)| Ok((Symbol::from_dynamic(k), self.build_attribute(v)?)))
                    .collect::<Result<_, ParseError>>()?;

                let mut builder = TypeDataBuilder::new(dialect, name);
                for p in params {
                    builder = builder.param(p);
                }
                for (k, v) in attrs {
                    builder = builder.attr(k, v);
                }
                Ok(self.ctx.types.intern(builder.build()))
            }
            RawType::Function {
                dialect,
                name,
                inputs,
                results,
                attrs,
            } => self.build_function_type(dialect, name, inputs, results, attrs),
        }
    }

    fn build_function_type(
        &mut self,
        dialect: &str,
        name: &str,
        inputs: &[RawType<'_>],
        results: &[RawType<'_>],
        attrs: &[(&str, RawAttribute<'_>)],
    ) -> Result<TypeRef, ParseError> {
        if (dialect == "func" && name == "func_sig") || (dialect == "core" && name == "func") {
            return self.build_shared_function_type(inputs, results, attrs);
        }
        self.build_foreign_function_type(dialect, name, inputs, results, attrs)
    }

    /// Build the existing shared signature through its owning validated API.
    fn build_shared_function_type(
        &mut self,
        inputs: &[RawType<'_>],
        results: &[RawType<'_>],
        attrs: &[(&str, RawAttribute<'_>)],
    ) -> Result<TypeRef, ParseError> {
        if results.len() > 1 {
            return Err(ParseError {
                message: format!(
                    "func.func_sig currently does not support multiple results (found {})",
                    results.len()
                ),
                offset: 0,
            });
        }
        if let Some((name, _)) = attrs.iter().find(|(name, _)| {
            matches!(
                *name,
                crate::dialect::func::NUM_INPUTS_ATTR | crate::dialect::func::NUM_RESULTS_ATTR
            )
        }) {
            return Err(ParseError {
                message: format!("`{name}` is reserved by func.func_sig"),
                offset: 0,
            });
        }

        let inputs = inputs
            .iter()
            .map(|ty| self.build_type(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let results = results
            .iter()
            .map(|ty| self.build_type(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let attrs = attrs
            .iter()
            .map(|(name, value)| Ok((Symbol::from_dynamic(name), self.build_attribute(value)?)))
            .collect::<Result<AttributeMap, ParseError>>()?;
        Ok(
            crate::dialect::func::func_sig_with_attrs(self.ctx, inputs, results, attrs)
                .as_type_ref(),
        )
    }

    /// Build a non-shared qualified `*.func_sig` as opaque dialect-owned storage.
    /// Semantic result cardinality remains the owning dialect's responsibility.
    fn build_foreign_function_type(
        &mut self,
        dialect: &str,
        name: &str,
        inputs: &[RawType<'_>],
        results: &[RawType<'_>],
        attrs: &[(&str, RawAttribute<'_>)],
    ) -> Result<TypeRef, ParseError> {
        if let Some((reserved, _)) = attrs.iter().find(|(key, _)| {
            matches!(
                *key,
                crate::dialect::func::NUM_INPUTS_ATTR | crate::dialect::func::NUM_RESULTS_ATTR
            )
        }) {
            return Err(ParseError {
                message: format!("`{reserved}` is reserved by {dialect}.{name}"),
                offset: 0,
            });
        }
        let inputs = inputs
            .iter()
            .map(|ty| self.build_type(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let results = results
            .iter()
            .map(|ty| self.build_type(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let attrs = attrs
            .iter()
            .map(|(key, value)| Ok((Symbol::from_dynamic(key), self.build_attribute(value)?)))
            .collect::<Result<AttributeMap, ParseError>>()?;
        let num_inputs = u32::try_from(inputs.len()).map_err(|_| ParseError {
            message: format!("{dialect}.{name} input count exceeds u32"),
            offset: 0,
        })?;
        let num_results = u32::try_from(results.len()).map_err(|_| ParseError {
            message: format!("{dialect}.{name} result count exceeds u32"),
            offset: 0,
        })?;
        let mut builder =
            TypeDataBuilder::new(Symbol::from_dynamic(dialect), Symbol::from_dynamic(name))
                .params(inputs)
                .params(results);
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        Ok(self.ctx.types.intern(
            builder
                .attr(
                    crate::dialect::func::NUM_INPUTS_ATTR,
                    Attribute::from(num_inputs),
                )
                .attr(
                    crate::dialect::func::NUM_RESULTS_ATTR,
                    Attribute::from(num_results),
                )
                .build(),
        ))
    }

    fn build_attribute(&mut self, raw: &RawAttribute<'_>) -> Result<Attribute, ParseError> {
        Ok(match raw {
            RawAttribute::Bool(b) => Attribute::Bool(*b),
            RawAttribute::Int(n) => Attribute::Int(*n),
            RawAttribute::Float(f) => Attribute::FloatBits(f.to_bits()),
            RawAttribute::String(s) => Attribute::String(s.clone()),
            RawAttribute::Symbol(s) => Attribute::Symbol(Symbol::from_dynamic(s.as_str())),
            RawAttribute::Type(t) => Attribute::Type(self.build_type(t)?),
            RawAttribute::List(items) => {
                let list: Vec<Attribute> = items
                    .iter()
                    .map(|a| self.build_attribute(a))
                    .collect::<Result<_, _>>()?;
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
        })
    }

    // ----------------------------------------------------------------
    // Value / block resolution
    // ----------------------------------------------------------------

    fn resolve_value(&self, name: &str) -> Option<ValueRef> {
        self.scope.value_map.get(name).copied()
    }

    fn save_scope(&self) -> BuilderScope {
        self.scope.clone()
    }

    fn restore_scope(&mut self, saved: BuilderScope) {
        self.scope = saved;
    }

    // ----------------------------------------------------------------
    // Region / Block building
    // ----------------------------------------------------------------

    /// Build a region from raw data.
    ///
    /// `extra_entry_args` are injected as the entry block's arguments
    /// (e.g. from a func-style signature).
    /// `is_module_region` controls whether type alias definitions are allowed.
    fn build_region(
        &mut self,
        raw: &RawRegion<'_>,
        extra_entry_args: &[(&str, RawType<'_>)],
        is_module_region: bool,
    ) -> Result<RegionRef, ParseError> {
        let saved = self.save_scope();
        self.scope.block_map.clear(); // isolate block scope per region
        let result = self.build_region_inner(raw, extra_entry_args, is_module_region);
        self.restore_scope(saved);
        result
    }

    fn build_region_inner(
        &mut self,
        raw: &RawRegion<'_>,
        extra_entry_args: &[(&str, RawType<'_>)],
        is_module_region: bool,
    ) -> Result<RegionRef, ParseError> {
        // --- Process type alias definitions ---
        if !raw.type_aliases.is_empty() && !is_module_region {
            return Err(ParseError {
                message: "type aliases are only allowed in module regions".to_string(),
                offset: 0,
            });
        }
        for (name, raw_ty) in &raw.type_aliases {
            let ty = self.build_type(raw_ty)?;
            self.scope.type_alias_map.insert(name.to_string(), ty);
            self.ctx.register_type_alias(Symbol::from_dynamic(name), ty);
        }

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

                let ty = self.build_type(raw_ty)?;
                let is_default_name = name
                    .strip_prefix("arg")
                    .and_then(|rest| rest.parse::<usize>().ok())
                    .is_some_and(|n| n == j);

                let mut attrs = AttributeMap::new();
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

            self.scope.block_map.insert(label, block_ref);

            // Register arg values, rejecting cross-block duplicates
            for (j, name) in arg_names.iter().enumerate() {
                if self.scope.value_map.contains_key(name) {
                    return Err(ParseError {
                        message: format!("duplicate SSA name '%{}' in block argument", name),
                        offset: 0,
                    });
                }
                let value = self.ctx.block_arg(block_ref, j as u32);
                self.scope.value_map.insert(name.clone(), value);
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
            let bt = self.build_type(block_ty)?;
            let pt = self.build_type(param_ty)?;
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
            .collect::<Result<_, _>>()?;

        // Build attributes from explicit attr dict
        let mut attributes: AttributeMap = raw
            .attributes
            .iter()
            .map(|(k, v)| Ok((Symbol::from_dynamic(k), self.build_attribute(v)?)))
            .collect::<Result<_, ParseError>>()?;

        // Add sym_name if present
        if let Some(ref name) = raw.sym_name {
            attributes.insert(
                Symbol::new("sym_name"),
                Attribute::Symbol(Symbol::from_dynamic(name.as_str())),
            );
        }

        // Handle func-style signature → func.func_sig type
        let has_func_signature =
            raw.has_func_signature || raw.return_type.is_some() || !raw.func_params.is_empty();
        if has_func_signature {
            let results = match raw.return_type.as_ref() {
                Some(t) => vec![self.build_type(t)?],
                None => vec![],
            };

            let param_types: Vec<TypeRef> = raw
                .func_params
                .iter()
                .map(|(_, raw_ty)| self.build_type(raw_ty))
                .collect::<Result<_, _>>()?;

            let func_ty =
                crate::dialect::func::func_sig(self.ctx, param_types, results).as_type_ref();
            if attributes.contains_key("type") && attributes.get_type("type").is_none() {
                return Err(ParseError {
                    message: "explicit function type attribute must be a type".into(),
                    offset: 0,
                });
            }
            if let Some(explicit) = attributes.get_type("type") {
                let signature = crate::dialect::func::FuncSig::from_type_ref(self.ctx, explicit);
                let synthesized =
                    crate::dialect::func::FuncSig::from_type_ref(self.ctx, func_ty).unwrap();
                if signature.is_none_or(|signature| {
                    signature.inputs(self.ctx) != synthesized.inputs(self.ctx)
                        || signature.results(self.ctx) != synthesized.results(self.ctx)
                }) {
                    return Err(ParseError {
                        message: "explicit function type does not match custom signature".into(),
                        offset: 0,
                    });
                }
            } else {
                attributes.insert(Symbol::new("type"), Attribute::Type(func_ty));
            }
        }

        // Resolve successors
        let successors: Vec<BlockRef> = raw
            .successors
            .iter()
            .map(|label| {
                self.scope
                    .block_map
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
        let is_module = dialect == Symbol::new("core") && op_name == Symbol::new("module");
        let mut regions = Vec::with_capacity(raw.regions.len());
        for (i, r) in raw.regions.iter().enumerate() {
            let extra_args = if i == 0 && !raw.func_params.is_empty() {
                &raw.func_params[..]
            } else {
                &[]
            };
            let region = self.build_region(r, extra_args, is_module)?;
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
            if self.scope.value_map.contains_key(*name) {
                return Err(ParseError {
                    message: format!(
                        "duplicate SSA name '{}' in operation '{}.{}' result index {}",
                        name, raw.dialect, raw.op_name, i
                    ),
                    offset: 0,
                });
            }
            let value = self.ctx.op_result(op_ref, i as u32);
            self.scope.value_map.insert(name.to_string(), value);
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

/// Parse textual IR into an [`Module`], panicking on failure.
///
/// Convenience wrapper around [`parse_module`] for tests.
pub fn parse_test_module(ctx: &mut IrContext, input: &str) -> Module {
    let op = parse_module(ctx, input).unwrap_or_else(|e| {
        panic!(
            "Failed to parse test IR at offset {}:\n  {}\n\nInput:\n{}",
            e.offset, e.message, input
        );
    });
    Module::new(ctx, op).unwrap_or_else(|| {
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
    use crate::dialect::{arith, core, func};
    use crate::ops::{DialectOp, DialectType};
    use crate::printer::print_module;
    use crate::validation;

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

    fn make_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
        func::func_sig(ctx, params.iter().copied(), [ret]).as_type_ref()
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
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
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
    fn test_roundtrip_namespaced_type_attribute_key() {
        let input = r#"
core.module @test {
  %0 = test.make : test.value() {tribute.calling_convention = 2}
}
"#;
        let mut ctx = IrContext::new();
        let root = parse_module(&mut ctx, input).expect("namespaced attribute key should parse");
        let printed = print_module(&ctx, root);
        assert!(printed.contains("tribute.calling_convention = 2"));
        assert_roundtrip(&ctx, root);
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
        let add = arith::addi(&mut ctx, loc, x, y, i32_ty);
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
        let cond = arith::r#const(&mut ctx, loc, i1_ty, Attribute::Int(1));
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
        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        ctx.push_op(else_block, c1.op_ref());
        let c1_val = c1.result(&ctx);
        let sum = arith::addi(&mut ctx, loc, param, c1_val, i32_ty);
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
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(7));
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
        for (name, val) in &[("foo", 1i128), ("bar", 2)] {
            let entry = ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            });
            let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(*val));
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
        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
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
        let call = func::call(&mut ctx, loc, [], [i32_ty], Symbol::new("callee"));
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

    #[test]
    fn test_roundtrip_bodyless_func_decl() {
        // Body-less func.func should still emit its full signature
        let input = r#"core.module @test {
  func.func @extern_add(%arg0: core.i32, %arg1: core.i32) -> core.i32
}"#;
        let mut ctx = IrContext::new();
        let module_op = parse_module(&mut ctx, input).unwrap();
        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_non_ascii_symbol() {
        // Non-ASCII symbol names must be quoted to survive round-trip
        // (parser only accepts ASCII alphanumeric + '_' for bare symbols)
        let input = r#"core.module @test {
  func.func @"π"() -> core.i32 {
    %0 = arith.const {value = 42} : core.i32
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module_op = parse_module(&mut ctx, input).unwrap();
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
      %2 = arith.addi %0, %1 : core.i32
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
      %2 = arith.addi %0, %1 : core.i32
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
      %2 = arith.addi %0, %1 : core.i32
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

    // ================================================================
    // Scoping error tests
    // ================================================================

    #[test]
    fn test_parse_cross_block_duplicate_arg_name() {
        // Two blocks both define %x as a block argument — should error
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    ^entry:
      %cond = arith.const {value = 1} : core.i1
      scf.br [^left]
    ^left(%x: core.i32):
      scf.br [^right]
    ^right(%x: core.i32):
      func.return %x
  }
}"#;
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input).unwrap_err();
        assert!(
            err.message.contains("duplicate SSA name '%x'"),
            "Expected duplicate SSA name error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_nested_region_block_isolation() {
        // Inner region must not resolve outer block labels as successors
        let input = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
    ^entry:
      %cond = arith.const {value = 1} : core.i1
      %r = scf.if %cond : core.i32 {
        scf.br [^entry]
      } {
        scf.yield %0
      }
      func.return %r
  }
}"#;
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input).unwrap_err();
        assert!(
            err.message.contains("undefined block '^entry'"),
            "Expected undefined block error for outer label in nested region, got: {}",
            err.message
        );
    }

    // ================================================================
    // Type alias tests
    // ================================================================

    #[test]
    fn review_function_result_type_preserves_empty_body() {
        for attrs in ["", " {effect = core.nil}"] {
            let input = format!(
                "core.module @test {{ func.func @f() -> func.func_sig<() -> ()>{attrs} {{}} }}"
            );
            let mut ctx = IrContext::new();
            let module = parse_module(&mut ctx, &input).unwrap();
            let function = ctx
                .block(ctx.region(ctx.op(module).regions[0]).blocks[0])
                .ops[0];
            assert_eq!(
                ctx.op(function).regions.len(),
                1,
                "empty body must not become a declaration"
            );
            assert_roundtrip(&ctx, module);
            let printed = print_module(&ctx, module);
            let mut reparsed = IrContext::new();
            let module2 = parse_module(&mut reparsed, &printed).unwrap();
            let function2 = reparsed
                .block(reparsed.region(reparsed.op(module2).regions[0]).blocks[0])
                .ops[0];
            assert_eq!(reparsed.op(function2).regions.len(), 1);
            assert_eq!(printed.contains("effect = core.nil"), !attrs.is_empty());
        }
    }

    #[test]
    fn test_canonical_func_type_roundtrips_all_supported_arities() {
        let input = r#"core.module @test {
  !zero_zero = func.func_sig<() -> ()>
  !many_zero = func.func_sig<(core.i32, core.i64) -> ()>
  !zero_one = func.func_sig<() -> core.i32>
  !many_one = func.func_sig<(core.i32, core.i64) -> core.i32>
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("canonical function types should parse");

        for (name, input_count, result_count) in [
            ("zero_zero", 0, 0),
            ("many_zero", 2, 0),
            ("zero_one", 0, 1),
            ("many_one", 2, 1),
        ] {
            let ty = ctx
                .type_alias_by_name(Symbol::from_dynamic(name))
                .expect("function alias");
            let function = func::FuncSig::from_type_ref(&ctx, ty).expect("validated func.func_sig");
            assert_eq!(function.inputs(&ctx).len(), input_count);
            assert_eq!(function.results(&ctx).len(), result_count);
        }
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_legacy_func_type_normalizes_to_canonical_storage_and_text() {
        let input = r#"core.module @test {
  !legacy = core.func(core.i64, core.i32)
  !legacy_canonical = core.func<(core.i32) -> core.i64>
  !canonical = func.func_sig<(core.i32) -> core.i64>
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("legacy function type should parse");
        let printed = print_module(&ctx, module);
        assert!(
            printed.contains("!legacy = func.func_sig<(core.i32) -> core.i64>"),
            "{printed}"
        );
        assert!(!printed.contains("core.func(core.i64"), "{printed}");

        let ty = ctx
            .type_alias_by_name(Symbol::new("legacy"))
            .expect("legacy alias");
        assert_eq!(
            Some(ty),
            ctx.type_alias_by_name(Symbol::new("legacy_canonical")),
            "legacy core.func canonical syntax must intern as func.func_sig"
        );
        assert_eq!(
            Some(ty),
            ctx.type_alias_by_name(Symbol::new("canonical")),
            "all accepted legacy spellings must normalize to one identity"
        );
        let data = ctx.types.get(ty);
        assert_eq!(data.params.len(), 2);
        assert_eq!(data.attrs.get_u32(func::NUM_INPUTS_ATTR), Ok(Some(1)));
        assert_eq!(data.attrs.get_u32(func::NUM_RESULTS_ATTR), Ok(Some(1)));
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_func_type_preserves_non_reserved_attributes() {
        let input = r#"core.module @test {
  !with_attr = func.func_sig<(core.i32) -> core.i64> {effect = core.nil}
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("function type attribute should parse");
        let printed = print_module(&ctx, module);
        assert!(
            printed
                .contains("!with_attr = func.func_sig<(core.i32) -> core.i64> {effect = core.nil}"),
            "{printed}"
        );
        assert!(!printed.contains("num_inputs"), "{printed}");
        assert!(!printed.contains("num_results"), "{printed}");
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_func_type_rejects_reserved_textual_attributes() {
        for reserved in [func::NUM_INPUTS_ATTR, func::NUM_RESULTS_ATTR] {
            for form in [
                "core.func(core.nil)",
                "core.func<() -> core.nil>",
                "func.func_sig<() -> core.nil>",
            ] {
                let input = format!("core.module @test {{ !bad = {form} {{{reserved} = 0}} }}");
                let mut ctx = IrContext::new();
                let error =
                    parse_module(&mut ctx, &input).expect_err("reserved key must be rejected");
                assert!(
                    error.message.contains("reserved by func.func_sig"),
                    "{error}"
                );
            }
        }
    }

    #[test]
    fn test_func_type_rejects_multiple_results_after_recognizing_syntax() {
        let input = "core.module @test { !bad = func.func_sig<() -> (core.i32, core.i64)> }";
        let mut ctx = IrContext::new();
        let error = parse_module(&mut ctx, input).expect_err("multi-result must be rejected");
        assert!(error.message.contains("multiple results"), "{error}");
    }

    #[test]
    fn foreign_func_sig_roundtrips_without_changing_shared_contracts() {
        let input = r#"core.module @test {
  !foreign = foreign.func_sig<(core.i32, core.i64) -> (core.i1, core.i32)> {nested = core.array(core.i32)}
  !unrelated = foreign.record(core.i32) {num_inputs = 1, num_results = 0}
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("foreign signature should parse");
        let printed = print_module(&ctx, module);
        assert!(printed.contains("!foreign = foreign.func_sig<(core.i32, core.i64) -> (core.i1, core.i32)> {nested = core.array(core.i32)}"), "{printed}");
        assert!(
            printed.contains(
                "!unrelated = foreign.record(core.i32) {num_inputs = 1, num_results = 0}"
            ),
            "{printed}"
        );
        assert_roundtrip(&ctx, module);

        for reserved in [func::NUM_INPUTS_ATTR, func::NUM_RESULTS_ATTR] {
            let mut rejected = IrContext::new();
            let error = parse_module(
                &mut rejected,
                &format!("core.module @test {{ !bad = foreign.func_sig<() -> core.i32> {{{reserved} = 0}} }}"),
            )
            .expect_err("foreign signature delimiters remain reserved");
            assert!(
                error.message.contains("reserved by foreign.func_sig"),
                "{error}"
            );
        }
    }

    #[test]
    fn malformed_func_sig_storage_roundtrips_as_concrete_type() {
        let input = r#"core.module @test {
  !foreign = foreign.func_sig(core.i32) {num_inputs = 2, num_results = 1}
  !shared = func.func_sig(core.i32, core.i32, core.i32) {num_inputs = 1, num_results = 2}
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("raw storage should remain parseable");
        let printed = print_module(&ctx, module);
        assert!(
            printed.contains(
                "!foreign = foreign.func_sig(core.i32) {num_inputs = 2, num_results = 1}"
            ),
            "{printed}"
        );
        assert!(
            printed.contains(
                "!shared = func.func_sig(core.i32, core.i32, core.i32) {num_inputs = 1, num_results = 2}"
            ),
            "{printed}"
        );
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_no_arrow_func_op_preserves_zero_results() {
        let input = r#"core.module @test {
  func.func @consume(%value: core.i32) {
    func.return
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_module(&mut ctx, input).expect("Unit function should parse");
        let printed = print_module(&ctx, module);
        assert!(
            printed.contains("func.func @consume(%0: core.i32) {"),
            "{printed}"
        );

        let module_view = crate::rewrite::Module::new(&ctx, module).unwrap();
        let function = func::Func::from_op(&ctx, module_view.ops(&ctx)[0]).unwrap();
        let signature = func::FuncSig::from_type_ref(&ctx, function.r#type(&ctx)).unwrap();
        assert_eq!(signature.inputs(&ctx).len(), 1);
        assert!(signature.is_resultless(&ctx));
        assert_roundtrip(&ctx, module);
    }

    #[test]
    fn test_roundtrip_type_alias() {
        let input = r#"core.module @test {
  !marker = adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32]], name = @_Marker}

  func.func @foo(%0: core.array(!marker)) -> core.array(!marker) {
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module_op = parse_module(&mut ctx, input).unwrap_or_else(|e| {
            panic!(
                "Parse failed at offset {}: {}\n\nInput:\n{}",
                e.offset, e.message, input
            );
        });
        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_programmatic_type_alias() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let tuple_ty = core::tuple(&mut ctx, [i32_ty, i32_ty]).as_type_ref();

        // Register alias
        ctx.register_type_alias(Symbol::new("point"), tuple_ty);

        // Build a function using that type
        let func_ty = make_func_type(&mut ctx, &[tuple_ty], tuple_ty);
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: tuple_ty,
                attrs: Default::default(),
            }],
            ops: smallvec![],
            parent_region: None,
        });
        let x = ctx.block_arg(entry, 0);
        let ret = func::r#return(&mut ctx, loc, [x]);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let f = func::func(&mut ctx, loc, Symbol::new("identity"), func_ty, body);
        let module_op = wrap_in_module(&mut ctx, loc, vec![f.op_ref()]);

        let printed = print_module(&ctx, module_op);
        // Alias should appear in output
        assert!(
            printed.contains("!point = core.tuple(core.i32, core.i32)"),
            "Expected alias definition in output:\n{printed}",
        );
        assert!(
            printed.contains("!point)"),
            "Expected alias usage in output:\n{printed}",
        );
    }

    #[test]
    fn test_nested_type_alias() {
        let input = r#"core.module @test {
  !inner = core.tuple(core.i32, core.i32)
  !outer = func.func_sig<() -> !inner>

  func.func @foo(%0: !inner) -> !inner {
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module_op = parse_module(&mut ctx, input).unwrap_or_else(|e| {
            panic!(
                "Parse failed at offset {}: {}\n\nInput:\n{}",
                e.offset, e.message, input
            );
        });
        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_undefined_type_alias_error() {
        let input = r#"core.module @test {
  func.func @f(%0: !unknown) -> core.i32 {
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input).unwrap_err();
        assert!(
            err.message.contains("undefined type alias '!unknown'"),
            "Expected undefined type alias error, got: {}",
            err.message,
        );
    }

    #[test]
    fn test_type_alias_no_alias_backward_compat() {
        // Existing IR without aliases should still round-trip
        let input = r#"core.module @test {
  func.func @f(%0: core.i32) -> core.i32 {
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module_op = parse_module(&mut ctx, input).unwrap();
        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_quoted_type_alias_roundtrip() {
        let input = r#"core.module @test {
  !"test::MyStruct" = adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @"test::MyStruct"}

  func.func @foo(%0: !"test::MyStruct") -> !"test::MyStruct" {
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module_op = parse_module(&mut ctx, input).unwrap_or_else(|e| {
            panic!(
                "Parse failed at offset {}: {}\n\nInput:\n{}",
                e.offset, e.message, input
            );
        });
        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_bytes_const() {
        let input = r#"core.module @test {
  func.func @f() -> core.bytes {
    %0 = adt.bytes_const {value = b"hello"} : core.bytes
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module_op = parse_module(&mut ctx, input).expect("should parse bytes_const");
        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_bytes_const_with_escapes() {
        let input = r#"core.module @test {
  func.func @f() -> core.bytes {
    %0 = adt.bytes_const {value = b"a\nb\t\0\\\""} : core.bytes
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module_op =
            parse_module(&mut ctx, input).expect("should parse bytes_const with escapes");
        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_roundtrip_bytes_const_with_non_ascii() {
        let input = r#"core.module @test {
  func.func @f() -> core.bytes {
    %0 = adt.bytes_const {value = b"\x80\xff\x00"} : core.bytes
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let module_op =
            parse_module(&mut ctx, input).expect("should parse bytes_const with non-ASCII hex");
        assert_roundtrip(&ctx, module_op);
    }

    #[test]
    fn test_type_alias_rejected_in_non_module_region() {
        let input = r#"core.module @test {
  func.func @f() -> core.i32 {
    !bad = core.i32
    %0 = arith.const {value = 42} : !bad
    func.return %0
  }
}"#;
        let mut ctx = IrContext::new();
        let err = parse_module(&mut ctx, input).unwrap_err();
        assert!(
            err.message
                .contains("type aliases are only allowed in module regions"),
            "Expected module-only error, got: {}",
            err.message,
        );
    }
}

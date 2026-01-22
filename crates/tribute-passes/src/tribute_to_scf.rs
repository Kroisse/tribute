//! Lower tribute dialect operations to scf dialect.
//!
//! This pass transforms:
//! - `tribute.case` → `scf.if` chains (pattern matching)
//! - `tribute.block` → `scf.if` with always-true condition (block expressions)
//! - `tribute.yield` → `scf.yield`
//!
//! Pattern matching currently supports a minimal subset:
//! - wildcard (`_`)
//! - bind (`x`)
//! - literal int/bool
//!
//! Unsupported patterns or non-exhaustive cases emit diagnostics.

use std::collections::HashMap;

use salsa::Accumulator;
use tribute_ir::ModulePathExt;
use tribute_ir::dialect::{tribute, tribute_pat};
use trunk_ir::dialect::adt;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{arith, core, scf};
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, DialectType, IdVec, Location, Operation, Region,
    SymbolVec, Type,
};
use trunk_ir::{Symbol, Value, ValueDef};

use crate::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};

#[derive(Debug, Clone)]
enum ArmPattern<'db> {
    Wildcard,
    Bind,
    Literal(Attribute<'db>),
    Variant(Symbol),
}

#[derive(Debug, Clone)]
struct ArmInfo<'db> {
    pattern: ArmPattern<'db>,
    pattern_region: Option<Region<'db>>,
    body: Region<'db>,
}

pub fn lower_tribute_to_scf<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    // Sanity check: verify all operand references point to operations in the same module
    #[cfg(debug_assertions)]
    verify_operand_references(db, module);
    CaseLowerer::new(db).lower_module(module)
}

#[cfg(debug_assertions)]
fn verify_operand_references<'db>(db: &'db dyn salsa::Database, module: Module<'db>) {
    use std::collections::HashSet;

    // Collect all operations in the module
    let mut all_ops: HashSet<trunk_ir::Operation<'db>> = HashSet::new();
    collect_ops_in_region(db, module.body(db), &mut all_ops);

    // Verify all operand references point to operations in the set
    verify_refs_in_region(db, module.body(db), &all_ops);
}

#[cfg(debug_assertions)]
fn collect_ops_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    ops: &mut std::collections::HashSet<trunk_ir::Operation<'db>>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter().copied() {
            ops.insert(op);
            for nested in op.regions(db).iter().copied() {
                collect_ops_in_region(db, nested, ops);
            }
        }
    }
}

#[cfg(debug_assertions)]
fn verify_refs_in_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    all_ops: &std::collections::HashSet<trunk_ir::Operation<'db>>,
) {
    use trunk_ir::ValueDef;
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter().copied() {
            for operand in op.operands(db).iter() {
                if let ValueDef::OpResult(ref_op) = operand.def(db)
                    && !all_ops.contains(&ref_op)
                {
                    tracing::warn!(
                        "STALE REFERENCE DETECTED in input to tribute_to_scf!\n  \
                         Operation {}.{} references {}.{} which is NOT in the module",
                        op.dialect(db),
                        op.name(db),
                        ref_op.dialect(db),
                        ref_op.name(db)
                    );
                }
            }
            for nested in op.regions(db).iter().copied() {
                verify_refs_in_region(db, nested, all_ops);
            }
        }
    }
}

struct CaseLowerer<'db> {
    db: &'db dyn salsa::Database,
    /// Rewrite context for value mapping.
    ctx: RewriteContext<'db>,
    variant_tags: HashMap<Symbol, u32>,
    variant_owner: HashMap<Symbol, Symbol>,
    enum_variants: HashMap<Symbol, SymbolVec>,
    /// Block argument types indexed by BlockId
    block_arg_types: HashMap<BlockId, IdVec<Type<'db>>>,
}

impl<'db> CaseLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            ctx: RewriteContext::new(),
            variant_tags: HashMap::new(),
            variant_owner: HashMap::new(),
            enum_variants: HashMap::new(),
            block_arg_types: HashMap::new(),
        }
    }

    fn lower_module(&mut self, module: Module<'db>) -> Module<'db> {
        self.collect_variant_tags(module);
        let body = module.body(self.db);
        let new_body = self.lower_region(body);
        Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        )
    }

    fn lower_region(&mut self, region: Region<'db>) -> Region<'db> {
        let blocks = region
            .blocks(self.db)
            .iter()
            .map(|block| self.lower_block(*block))
            .collect::<Vec<_>>();
        Region::new(self.db, region.location(self.db), IdVec::from(blocks))
    }

    fn lower_block(&mut self, block: Block<'db>) -> Block<'db> {
        // Register block arg types for value_type lookups
        let arg_types = block.arg_types(self.db);
        self.block_arg_types.insert(block.id(self.db), arg_types);

        let mut new_ops = IdVec::new();
        for op in block.operations(self.db).iter().copied() {
            let rewritten = self.lower_op(op);
            new_ops.extend(rewritten);
        }
        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    fn lower_op(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let remapped_op = self.ctx.remap_operands(self.db, &op);
        let remapped_operands = remapped_op.operands(self.db).clone();

        if op.dialect(self.db) == tribute::DIALECT_NAME() && op.name(self.db) == tribute::CASE() {
            return self.lower_case(op, remapped_operands);
        }

        // Note: tribute.handle is now handled by tribute_to_cont pass

        if op.dialect(self.db) == tribute::DIALECT_NAME() && op.name(self.db) == tribute::YIELD() {
            let new_op = scf::r#yield(self.db, op.location(self.db), remapped_operands);
            return vec![new_op.as_operation()];
        }

        // Lower tribute.block to scf.if with always-true condition
        // This converts block expressions { ... } into structured control flow
        if op.dialect(self.db) == tribute::DIALECT_NAME() && op.name(self.db) == tribute::BLOCK() {
            return self.lower_block_expr(op);
        }

        // Handle tribute.let: extract values from pattern and produce results
        if op.dialect(self.db) == tribute::DIALECT_NAME() && op.name(self.db) == tribute::LET() {
            return self.lower_let(op, remapped_operands);
        }

        // Note: tribute.var for pattern bindings is now handled via block args.
        // tirgen registers bindings to block args, so tribute.var is not emitted for pattern bindings.

        let new_regions = op
            .regions(self.db)
            .iter()
            .copied()
            .map(|region| self.lower_region(region))
            .collect::<Vec<_>>();

        let new_op = op
            .modify(self.db)
            .operands(remapped_operands)
            .regions(IdVec::from(new_regions))
            .build();
        self.ctx.map_results(self.db, &op, &new_op);
        vec![new_op]
    }

    fn lower_case(
        &mut self,
        op: Operation<'db>,
        remapped_operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let Some(result_type) = op.results(self.db).first().copied() else {
            self.emit_error(location, "case expression must produce a result");
            return vec![op];
        };

        let Some(scrutinee) = remapped_operands.first().copied() else {
            return vec![op];
        };

        let Some(body_region) = op.regions(self.db).first().copied() else {
            return vec![op];
        };

        let (arms, supported) = self.collect_arms(body_region);
        if !supported {
            return vec![self.rebuild_case(op, remapped_operands)];
        }

        // Note: handler patterns are now handled by tribute_to_cont pass

        if arms.is_empty() {
            self.emit_error(location, "case expression has no arms");
            return vec![self.rebuild_case(op, remapped_operands)];
        }

        if !self.is_exhaustive(&arms) {
            self.emit_error(location, "non-exhaustive case expression");
            return vec![self.rebuild_case(op, remapped_operands)];
        }

        if arms.len() == 1 && !self.is_irrefutable(&arms[0].pattern) {
            self.emit_error(location, "single-arm case requires irrefutable pattern");
            return vec![self.rebuild_case(op, remapped_operands)];
        }

        let (ops, _final_value) = self.build_arm_chain(location, scrutinee, result_type, &arms);
        if let Some(last_op) = ops.last() {
            self.ctx.map_results(self.db, &op, last_op);
        }
        ops
    }

    /// Lower `tribute.block` to `scf.if` with always-true condition.
    ///
    /// Block expressions `{ statements; result }` are converted to:
    /// ```text
    /// %true = arith.const true
    /// %result = scf.if %true {
    ///     ... statements ...
    ///     scf.yield %result
    /// } else {
    ///     scf.yield %result  // unreachable but required for scf.if
    /// }
    /// ```
    fn lower_block_expr(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let Some(result_type) = op.results(self.db).first().copied() else {
            // No result - just lower the body operations inline
            return self.inline_block_body(op);
        };

        let Some(body_region) = op.regions(self.db).first().copied() else {
            return vec![op];
        };

        // Lower the body region (this will convert tribute.yield to scf.yield)
        let lowered_body = self.lower_region(body_region);

        // Create always-true condition
        let bool_ty = core::I1::new(self.db).as_type();
        let true_const = arith::r#const(self.db, location, bool_ty, true.into()).as_operation();
        let cond = true_const.result(self.db, 0);

        // Create scf.if with the lowered body as both then and else branches
        // (else is required but unreachable with true condition)
        let if_op = scf::r#if(
            self.db,
            location,
            cond,
            result_type,
            lowered_body,
            lowered_body,
        );

        // Map the block result to the if result
        self.ctx.map_results(self.db, &op, &if_op.as_operation());

        vec![true_const, if_op.as_operation()]
    }

    /// Inline block body operations when the block has no result.
    fn inline_block_body(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let Some(body_region) = op.regions(self.db).first().copied() else {
            return vec![];
        };

        let lowered_body = self.lower_region(body_region);
        let mut ops = Vec::new();

        // Extract all operations from the lowered body, skipping scf.yield
        for block in lowered_body.blocks(self.db).iter() {
            for body_op in block.operations(self.db).iter().copied() {
                // Skip scf.yield since we're inlining (no value to yield)
                if body_op.dialect(self.db) == Symbol::new("scf")
                    && body_op.name(self.db) == Symbol::new("yield")
                {
                    continue;
                }
                ops.push(body_op);
            }
        }

        ops
    }

    /// Lower `tribute.let` by extracting values from the pattern and producing results.
    ///
    /// `tribute.let` now returns multiple results (one per binding).
    /// This function:
    /// 1. Extracts values from the input based on the pattern structure
    /// 2. Maps the let operation's results to the extracted values
    /// 3. Returns extraction operations (the let itself is erased)
    fn lower_let(
        &mut self,
        op: Operation<'db>,
        remapped_operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let Some(value) = remapped_operands.first().copied() else {
            return vec![op];
        };
        let Some(pattern_region) = op.regions(self.db).first().copied() else {
            return vec![op];
        };

        // Collect expected types from the let operation's results
        let expected_types: Vec<_> = op.results(self.db).iter().copied().collect();

        // Extract values based on pattern structure
        let (extracted, extraction_ops) =
            self.extract_values_from_pattern(location, pattern_region, value, &expected_types);

        // Map each result of tribute.let to the corresponding extracted value
        for (i, &extracted_value) in extracted.iter().enumerate() {
            let let_result = op.result(self.db, i);
            self.ctx.map_value(let_result, extracted_value);
        }

        // Return the extraction operations (tribute.let is erased)
        extraction_ops
    }

    fn rebuild_case(
        &mut self,
        op: Operation<'db>,
        remapped_operands: IdVec<Value<'db>>,
    ) -> Operation<'db> {
        let new_op = op.modify(self.db).operands(remapped_operands).build();
        self.ctx.map_results(self.db, &op, &new_op);
        new_op
    }

    fn collect_arms(&mut self, body_region: Region<'db>) -> (Vec<ArmInfo<'db>>, bool) {
        let mut supported = true;
        let mut arms = Vec::new();
        for block in body_region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter().copied() {
                if op.dialect(self.db) != tribute::DIALECT_NAME()
                    || op.name(self.db) != tribute::ARM()
                {
                    continue;
                }
                let arm_location = op.location(self.db);
                let pattern_region = op.regions(self.db).first().copied();
                let body_region = op.regions(self.db).get(1).copied();
                let (pattern, ok) = pattern_region
                    .and_then(|region| self.parse_pattern(region))
                    .unwrap_or((ArmPattern::Wildcard, false));
                supported &= ok;
                let body =
                    body_region.unwrap_or_else(|| Region::new(self.db, arm_location, IdVec::new()));
                arms.push(ArmInfo {
                    pattern,
                    pattern_region,
                    body,
                });
            }
        }
        if !supported {
            self.emit_error(
                body_region.location(self.db),
                "unsupported pattern in case expression",
            );
        }
        (arms, supported)
    }

    fn parse_pattern(&self, region: Region<'db>) -> Option<(ArmPattern<'db>, bool)> {
        let block = region.blocks(self.db).first().copied()?;
        let mut ops = block.operations(self.db).iter().copied();
        let op = ops.next()?;
        if ops.next().is_some() {
            return Some((ArmPattern::Wildcard, false));
        }

        if op.dialect(self.db) != tribute_pat::DIALECT_NAME() {
            return Some((ArmPattern::Wildcard, false));
        }

        match op.name(self.db) {
            name if name == tribute_pat::WILDCARD() => Some((ArmPattern::Wildcard, true)),
            name if name == tribute_pat::BIND() => Some((ArmPattern::Bind, true)),
            name if name == tribute_pat::LITERAL() => {
                let attr = op.attributes(self.db).get(&Symbol::new("value"))?.clone();
                match attr {
                    Attribute::IntBits(_) | Attribute::Bool(_) => {
                        Some((ArmPattern::Literal(attr), true))
                    }
                    _ => Some((ArmPattern::Wildcard, false)),
                }
            }
            name if name == tribute_pat::VARIANT() => {
                let attr = op.attributes(self.db).get(&Symbol::new("variant"))?;
                let variant_path = match attr {
                    Attribute::Symbol(path) => *path,
                    _ => return Some((ArmPattern::Wildcard, false)),
                };
                let name = variant_path.last_segment();
                if !self.variant_tags.contains_key(&name) {
                    return Some((ArmPattern::Wildcard, false));
                }
                let fields_region = op.regions(self.db).first().copied();
                if fields_region
                    .is_some_and(|fields| !self.pattern_region_is_bind_or_wildcard(fields))
                {
                    return Some((ArmPattern::Wildcard, false));
                }
                Some((ArmPattern::Variant(name), true))
            }
            // Note: handler_done and handler_suspend patterns are now handled by tribute_to_cont pass
            _ => Some((ArmPattern::Wildcard, false)),
        }
    }

    fn pattern_region_is_bind_or_wildcard(&self, region: Region<'db>) -> bool {
        let Some(block) = region.blocks(self.db).first() else {
            return true;
        };
        for op in block.operations(self.db).iter().copied() {
            if op.dialect(self.db) != tribute_pat::DIALECT_NAME() {
                return false;
            }
            let ok = matches!(op.name(self.db), name if name == tribute_pat::BIND() || name == tribute_pat::WILDCARD());
            if !ok {
                return false;
            }
        }
        true
    }

    fn is_exhaustive(&self, arms: &[ArmInfo<'db>]) -> bool {
        if matches!(
            arms.last().map(|arm| &arm.pattern),
            Some(ArmPattern::Wildcard) | Some(ArmPattern::Bind)
        ) {
            return true;
        }

        let mut owner: Option<Symbol> = None;
        let mut seen_variants = SymbolVec::new();

        for arm in arms {
            let ArmPattern::Variant(variant) = arm.pattern else {
                return false;
            };
            let Some(variant_owner) = self.variant_owner.get(&variant).copied() else {
                return false;
            };
            if let Some(existing) = owner {
                if existing != variant_owner {
                    return false;
                }
            } else {
                owner = Some(variant_owner);
            }
            if !seen_variants.contains(&variant) {
                seen_variants.push(variant);
            }
        }

        let Some(owner) = owner else {
            return false;
        };
        let Some(all_variants) = self.enum_variants.get(&owner) else {
            return false;
        };
        seen_variants.len() == all_variants.len()
    }

    fn is_irrefutable(&self, pattern: &ArmPattern<'db>) -> bool {
        match pattern {
            ArmPattern::Wildcard | ArmPattern::Bind => true,
            ArmPattern::Literal(_) => false,
            ArmPattern::Variant(name) => {
                let Some(owner) = self.variant_owner.get(name).copied() else {
                    return false;
                };
                let Some(all_variants) = self.enum_variants.get(&owner) else {
                    return false;
                };
                all_variants.len() == 1
            }
        }
    }

    /// Lower an arm body with pattern bindings set up.
    ///
    /// The body block has block arguments corresponding to pattern bindings.
    /// This function:
    /// 1. Extracts values from the scrutinee based on the pattern structure
    /// 2. Maps body block arguments to the extracted values
    /// 3. Lowers the body with block arg references resolved
    fn lower_arm_body(&mut self, scrutinee: Value<'db>, arm: &ArmInfo<'db>) -> Region<'db> {
        let location = arm.body.location(self.db);

        // For variant patterns, create a cast operation to get variant-specific reference
        let (cast_op, cast_result) = if let ArmPattern::Variant(variant_name) = &arm.pattern {
            if let Some(enum_type) = self.value_type(scrutinee) {
                let cast = adt::variant_cast(
                    self.db,
                    location,
                    scrutinee,
                    enum_type,
                    enum_type,
                    *variant_name,
                )
                .as_operation();
                let cast_result = cast.result(self.db, 0);
                (Some(cast), Some(cast_result))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Get entry block and its arguments
        let entry_block = arm.body.blocks(self.db).first().copied();

        // Collect operations to prepend (cast + extraction ops)
        let mut ops_to_prepend: Vec<Operation<'db>> = Vec::new();
        if let Some(cast) = cast_op {
            ops_to_prepend.push(cast);
        }

        // Extract values from pattern and map to block arguments
        if let (Some(pattern_region), Some(block)) = (arm.pattern_region, entry_block) {
            let block_args = block.args(self.db);
            let base_value = cast_result.unwrap_or(scrutinee);

            // Collect expected types from block args
            let expected_types: Vec<_> = block_args.iter().map(|arg| arg.ty(self.db)).collect();
            tracing::debug!(
                "lower_arm_body: expected_types from block_args: {:?}",
                expected_types
                    .iter()
                    .map(|t| format!("{}.{}", t.dialect(self.db), t.name(self.db)))
                    .collect::<Vec<_>>()
            );

            // Extract values based on pattern structure, using expected types
            let (extracted_values, extraction_ops) = self.extract_values_from_pattern(
                location,
                pattern_region,
                base_value,
                &expected_types,
            );
            ops_to_prepend.extend(extraction_ops);

            // Map each block argument to its extracted value
            for (i, &extracted) in extracted_values.iter().enumerate() {
                if i < block_args.len() {
                    let block_arg_value =
                        Value::new(self.db, ValueDef::BlockArg(block.id(self.db)), i);
                    self.ctx.map_value(block_arg_value, extracted);
                }
            }
        }

        // Lower the body (block arg references will be resolved via ctx)
        let lowered = self.lower_region(arm.body);

        // Prepend cast and extraction operations to the first block
        if !ops_to_prepend.is_empty() {
            self.prepend_ops_to_region(lowered, ops_to_prepend)
        } else {
            lowered
        }
    }

    /// Extract values from a pattern, generating extraction operations as needed.
    ///
    /// Returns a vector of values corresponding to each binding in the pattern.
    /// The order matches the order of `tribute_pat.bind` operations in the pattern region.
    ///
    /// `expected_types` provides the types for each binding (from block args),
    /// which are used when creating extraction operations.
    fn extract_values_from_pattern(
        &mut self,
        location: Location<'db>,
        pattern_region: Region<'db>,
        base_value: Value<'db>,
        expected_types: &[Type<'db>],
    ) -> (Vec<Value<'db>>, Vec<Operation<'db>>) {
        let mut values = Vec::new();
        let mut ops = Vec::new();
        let mut type_idx = 0;
        self.extract_values_recursive(
            location,
            pattern_region,
            base_value,
            expected_types,
            &mut type_idx,
            &mut values,
            &mut ops,
        );
        (values, ops)
    }

    /// Recursively extract values from a pattern region.
    ///
    /// `type_idx` tracks the current index into `expected_types` for binding types.
    /// `ops` collects the extraction operations that need to be added to the IR.
    fn extract_values_recursive(
        &mut self,
        location: Location<'db>,
        region: Region<'db>,
        current_value: Value<'db>,
        expected_types: &[Type<'db>],
        type_idx: &mut usize,
        values: &mut Vec<Value<'db>>,
        ops: &mut Vec<Operation<'db>>,
    ) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter().copied() {
                if op.dialect(self.db) != tribute_pat::DIALECT_NAME() {
                    continue;
                }

                let op_name = op.name(self.db);

                if op_name == tribute_pat::BIND() {
                    // Bind: add current value to extracted values
                    values.push(current_value);
                    *type_idx += 1;
                } else if op_name == tribute_pat::WILDCARD() {
                    // Wildcard: no binding, nothing to extract
                } else if op_name == tribute_pat::VARIANT() {
                    // Variant: recurse into fields with field extraction
                    if let Some(fields_region) = op.regions(self.db).first().copied() {
                        // Get the variant tag from the pattern
                        let variant_tag = op
                            .attributes(self.db)
                            .get(&Symbol::new("variant"))
                            .and_then(|attr| match attr {
                                Attribute::Symbol(s) => Some(*s),
                                _ => None,
                            });

                        // Get field types from the enum type definition
                        let enum_ty = self.value_type(current_value);
                        tracing::debug!(
                            "extract_values_recursive: VARIANT {:?}, enum_ty={:?}",
                            variant_tag,
                            enum_ty.map(|ty| format!(
                                "{}.{}",
                                ty.dialect(self.db),
                                ty.name(self.db)
                            ))
                        );
                        let variant_field_types: Option<Vec<Type<'db>>> =
                            enum_ty.and_then(|ty| {
                                if let Some(tag) = variant_tag {
                                    let variants = adt::get_enum_variants(self.db, ty);
                                    tracing::debug!(
                                        "extract_values_recursive: get_enum_variants returned {:?}",
                                        variants.as_ref().map(|v| v.len())
                                    );
                                    variants.and_then(|variants| {
                                        variants
                                            .into_iter()
                                            .find(|(name, _)| *name == tag)
                                            .map(|(_, fields)| {
                                                tracing::debug!(
                                                    "extract_values_recursive: variant {:?} fields: {:?}",
                                                    tag,
                                                    fields.iter().map(|t| format!("{}.{}", t.dialect(self.db), t.name(self.db))).collect::<Vec<_>>()
                                                );
                                                fields
                                            })
                                    })
                                } else {
                                    None
                                }
                            });

                        // Extract field values using variant_get for each nested pattern
                        let mut field_idx = 0u64;
                        for field_block in fields_region.blocks(self.db).iter() {
                            for field_op in field_block.operations(self.db).iter().copied() {
                                // Use the actual field type from enum definition if available,
                                // otherwise fall back to expected_types (binding types)
                                let field_ty = variant_field_types
                                    .as_ref()
                                    .and_then(|types| types.get(field_idx as usize).copied())
                                    .or_else(|| expected_types.get(*type_idx).copied())
                                    .unwrap_or_else(|| {
                                        tribute::new_type_var(
                                            self.db,
                                            std::collections::BTreeMap::new(),
                                        )
                                    });
                                let variant_get_op = adt::variant_get(
                                    self.db,
                                    location,
                                    current_value,
                                    field_ty,
                                    field_idx.into(),
                                );
                                let field_val = variant_get_op.result(self.db);
                                ops.push(variant_get_op.as_operation());

                                // Recurse into the field pattern
                                let field_region = Region::new(
                                    self.db,
                                    location,
                                    IdVec::from(vec![Block::new(
                                        self.db,
                                        BlockId::fresh(),
                                        location,
                                        IdVec::new(),
                                        IdVec::from(vec![field_op]),
                                    )]),
                                );
                                self.extract_values_recursive(
                                    location,
                                    field_region,
                                    field_val,
                                    expected_types,
                                    type_idx,
                                    values,
                                    ops,
                                );
                                field_idx += 1;
                            }
                        }
                    }
                } else if op_name == tribute_pat::TUPLE() {
                    // Tuple: similar to variant, extract each element
                    if let Some(elements_region) = op.regions(self.db).first().copied() {
                        let mut elem_idx = 0u64;
                        for elem_block in elements_region.blocks(self.db).iter() {
                            for elem_op in elem_block.operations(self.db).iter().copied() {
                                // Create tuple element extraction
                                let tuple_get_name =
                                    Symbol::from_dynamic(&format!("tuple_get_{}", elem_idx));
                                // Use the expected type from block args if available
                                let elem_ty =
                                    expected_types.get(*type_idx).copied().unwrap_or_else(|| {
                                        tribute::new_type_var(
                                            self.db,
                                            std::collections::BTreeMap::new(),
                                        )
                                    });
                                let tuple_call_op = tribute::call(
                                    self.db,
                                    location,
                                    vec![current_value],
                                    elem_ty,
                                    tuple_get_name,
                                );
                                let elem_val = tuple_call_op.result(self.db);
                                ops.push(tuple_call_op.as_operation());

                                // Recurse into the element pattern
                                let elem_region = Region::new(
                                    self.db,
                                    location,
                                    IdVec::from(vec![Block::new(
                                        self.db,
                                        BlockId::fresh(),
                                        location,
                                        IdVec::new(),
                                        IdVec::from(vec![elem_op]),
                                    )]),
                                );
                                self.extract_values_recursive(
                                    location,
                                    elem_region,
                                    elem_val,
                                    expected_types,
                                    type_idx,
                                    values,
                                    ops,
                                );
                                elem_idx += 1;
                            }
                        }
                    }
                } else if op_name == tribute_pat::AS_PAT() {
                    // As pattern: bind the value, then recurse into inner pattern
                    values.push(current_value);
                    *type_idx += 1;
                    if let Some(inner_region) = op.regions(self.db).first().copied() {
                        self.extract_values_recursive(
                            location,
                            inner_region,
                            current_value,
                            expected_types,
                            type_idx,
                            values,
                            ops,
                        );
                    }
                } else if op_name == tribute_pat::LITERAL() {
                    // Literal patterns: no binding
                }
                // Other patterns: recurse into nested regions
                else {
                    for nested_region in op.regions(self.db).iter().copied() {
                        self.extract_values_recursive(
                            location,
                            nested_region,
                            current_value,
                            expected_types,
                            type_idx,
                            values,
                            ops,
                        );
                    }
                }
            }
        }
    }

    /// Prepend multiple operations to the first block of a region.
    fn prepend_ops_to_region(
        &self,
        region: Region<'db>,
        ops_to_prepend: Vec<Operation<'db>>,
    ) -> Region<'db> {
        let blocks = region.blocks(self.db);
        if blocks.is_empty() || ops_to_prepend.is_empty() {
            return region;
        }

        let first_block = blocks[0];
        let mut new_ops = IdVec::from(ops_to_prepend);
        new_ops.extend(first_block.operations(self.db).iter().copied());

        let new_first_block = Block::new(
            self.db,
            first_block.id(self.db),
            first_block.location(self.db),
            first_block.args(self.db).clone(),
            new_ops,
        );

        let mut new_blocks: Vec<Block<'db>> = vec![new_first_block];
        new_blocks.extend(blocks.iter().skip(1).copied());
        Region::new(self.db, region.location(self.db), IdVec::from(new_blocks))
    }

    fn build_arm_chain(
        &mut self,
        location: Location<'db>,
        scrutinee: Value<'db>,
        result_type: Type<'db>,
        arms: &[ArmInfo<'db>],
    ) -> (Vec<Operation<'db>>, Value<'db>) {
        if arms.len() == 1 {
            let (cond_ops, cond) = self.build_condition_ops(location, scrutinee, &arms[0].pattern);
            let body_region = self.lower_arm_body(scrutinee, &arms[0]);
            let then_region = body_region;
            // Single-arm cases are irrefutable; reuse the body for the else branch.
            let else_region = body_region;
            let if_op = scf::r#if(
                self.db,
                location,
                cond,
                result_type,
                then_region,
                else_region,
            );
            let mut ops = Vec::new();
            ops.extend(cond_ops);
            ops.push(if_op.as_operation());
            return (ops, if_op.as_operation().result(self.db, 0));
        }

        let (cond_ops, cond) = self.build_condition_ops(location, scrutinee, &arms[0].pattern);
        let then_region = self.lower_arm_body(scrutinee, &arms[0]);
        let else_region = self.build_else_region(location, scrutinee, result_type, &arms[1..]);

        let if_op = scf::r#if(
            self.db,
            location,
            cond,
            result_type,
            then_region,
            else_region,
        );

        let mut ops = Vec::new();
        ops.extend(cond_ops);
        ops.push(if_op.as_operation());
        (ops, if_op.as_operation().result(self.db, 0))
    }

    fn build_else_region(
        &mut self,
        location: Location<'db>,
        scrutinee: Value<'db>,
        result_type: Type<'db>,
        arms: &[ArmInfo<'db>],
    ) -> Region<'db> {
        if arms.len() == 1 {
            return self.lower_arm_body(scrutinee, &arms[0]);
        }

        let (mut ops, value) = self.build_arm_chain(location, scrutinee, result_type, arms);
        let yield_op = scf::r#yield(self.db, location, IdVec::from(vec![value]));
        ops.push(yield_op.as_operation());

        let block = Block::new(
            self.db,
            BlockId::fresh(),
            location,
            IdVec::new(),
            IdVec::from(ops),
        );
        Region::new(self.db, location, IdVec::from(vec![block]))
    }

    fn build_condition_ops(
        &self,
        location: Location<'db>,
        scrutinee: Value<'db>,
        pattern: &ArmPattern<'db>,
    ) -> (Vec<Operation<'db>>, Value<'db>) {
        let bool_ty = core::I1::new(self.db).as_type();
        match pattern {
            ArmPattern::Wildcard | ArmPattern::Bind => {
                let op = arith::r#const(self.db, location, bool_ty, true.into()).as_operation();
                let value = op.result(self.db, 0);
                (vec![op], value)
            }
            ArmPattern::Literal(attr) => {
                let Some(scrutinee_ty) = self.value_type(scrutinee) else {
                    let op = arith::r#const(self.db, location, bool_ty, true.into()).as_operation();
                    let value = op.result(self.db, 0);
                    return (vec![op], value);
                };
                let lit_op =
                    arith::r#const(self.db, location, scrutinee_ty, attr.clone()).as_operation();
                let lit_value = lit_op.result(self.db, 0);
                let cmp_op =
                    arith::cmp_eq(self.db, location, scrutinee, lit_value, bool_ty).as_operation();
                let cmp_value = cmp_op.result(self.db, 0);
                (vec![lit_op, cmp_op], cmp_value)
            }
            ArmPattern::Variant(name) => {
                // Get the enum type from scrutinee
                let Some(enum_type) = self.value_type(scrutinee) else {
                    let op = arith::r#const(self.db, location, bool_ty, true.into()).as_operation();
                    let value = op.result(self.db, 0);
                    return (vec![op], value);
                };

                // Use adt.variant_is which tests if scrutinee is of specific variant type
                let variant_is_op =
                    adt::variant_is(self.db, location, scrutinee, bool_ty, enum_type, *name)
                        .as_operation();
                let result = variant_is_op.result(self.db, 0);
                (vec![variant_is_op], result)
            }
        }
    }

    fn value_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        match value.def(self.db) {
            ValueDef::OpResult(op) => op.results(self.db).get(value.index(self.db)).copied(),
            ValueDef::BlockArg(block_id) => self
                .block_arg_types
                .get(&block_id)
                .and_then(|args| args.get(value.index(self.db)).copied()),
        }
    }

    fn emit_error(&self, location: Location<'db>, message: &str) {
        Diagnostic {
            message: message.to_string(),
            span: location.span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::Optimization,
        }
        .accumulate(self.db);
    }

    #[allow(dead_code)]
    fn emit_warning(&self, location: Location<'db>, message: &str) {
        Diagnostic {
            message: message.to_string(),
            span: location.span,
            severity: DiagnosticSeverity::Warning,
            phase: CompilationPhase::Optimization,
        }
        .accumulate(self.db);
    }

    fn collect_variant_tags(&mut self, module: Module<'db>) {
        let body = module.body(self.db);
        self.collect_variant_tags_in_region(body);
    }

    fn collect_variant_tags_in_region(&mut self, region: Region<'db>) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter().copied() {
                if op.dialect(self.db) == tribute::DIALECT_NAME()
                    && op.name(self.db) == tribute::ENUM_DEF()
                {
                    self.collect_variant_tags_from_enum(op);
                }
                for nested in op.regions(self.db).iter().copied() {
                    self.collect_variant_tags_in_region(nested);
                }
            }
        }
    }

    fn collect_variant_tags_from_enum(&mut self, op: Operation<'db>) {
        // Parse as tribute.enum_def with region-based variants
        let Ok(enum_def) = tribute::EnumDef::from_operation(self.db, op) else {
            return;
        };

        let type_name = enum_def.sym_name(self.db);

        let variants_region = enum_def.variants(self.db);
        let mut variant_names = SymbolVec::new();
        let mut idx = 0u32;

        for block in variants_region.blocks(self.db).iter() {
            for variant_op in block.operations(self.db).iter().copied() {
                // Check if this is a tribute.variant_def
                let Ok(variant_def) = tribute::VariantDef::from_operation(self.db, variant_op)
                else {
                    continue;
                };

                let name = variant_def.sym_name(self.db);
                let tag = idx;
                idx += 1;

                if self
                    .variant_tags
                    .insert(name, tag)
                    .is_some_and(|existing| existing != tag)
                {
                    self.emit_error(
                        op.location(self.db),
                        "ambiguous variant name in enum declarations",
                    );
                }
                if self
                    .variant_owner
                    .insert(name, type_name)
                    .is_some_and(|existing| existing != type_name)
                {
                    self.emit_error(
                        op.location(self.db),
                        "ambiguous variant name in enum declarations",
                    );
                }
                variant_names.push(name);
            }
        }
        self.enum_variants.insert(type_name, variant_names);
    }
}

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
use tribute_ir::dialect::{adt, tribute, tribute_pat};
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{arith, cont, core, scf};
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, DialectType, IdVec, Location, Operation, Region,
    SymbolVec, Type,
};
use trunk_ir::{Symbol, Value, ValueDef};

use crate::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};

/// Compute a deterministic hash-based index from an operation name.
/// This must match the implementation in cont_to_wasm.rs.
fn compute_op_idx_hash(name: &str) -> u64 {
    name.bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
        & 0xFFFF // Keep lower 16 bits
}

#[derive(Debug, Clone)]
enum ArmPattern<'db> {
    Wildcard,
    Bind,
    Literal(Attribute<'db>),
    Variant(Symbol),
    /// Handler done pattern: `{ result }` - matches Request::Done
    HandlerDone,
    /// Handler suspend pattern: `{ Op(args) -> k }` - matches Request::Suspend
    #[allow(dead_code)]
    HandlerSuspend {
        op_name: Symbol,
    },
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
    /// Current arm's pattern bindings: binding name -> bound value (scrutinee)
    current_arm_bindings: HashMap<Symbol, Value<'db>>,
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
            current_arm_bindings: HashMap::new(),
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

        if op.dialect(self.db) == tribute::DIALECT_NAME() && op.name(self.db) == tribute::YIELD() {
            let new_op = scf::r#yield(self.db, op.location(self.db), remapped_operands);
            return vec![new_op.as_operation()];
        }

        // Lower tribute.block to scf.if with always-true condition
        // This converts block expressions { ... } into structured control flow
        if op.dialect(self.db) == tribute::DIALECT_NAME() && op.name(self.db) == tribute::BLOCK() {
            return self.lower_block_expr(op);
        }

        // Handle tribute.var for case pattern bindings: replace with the bound value
        // This is used when tribute.var in arm body references a tribute_pat.bind from pattern region
        if op.dialect(self.db) == tribute::DIALECT_NAME()
            && op.name(self.db) == tribute::VAR()
            && let Some(Attribute::Symbol(name)) = op.attributes(self.db).get(&Symbol::new("name"))
            && let Some(&bound_value) = self.current_arm_bindings.get(name)
        {
            // Look up the current mapping for the bound value (handles remapping from lowering)
            let current_bound_value = self.ctx.lookup(bound_value);
            // Map tribute.var result to the bound value (scrutinee or destructured value)
            let var_result = op.result(self.db, 0);
            self.ctx.map_value(var_result, current_bound_value);
            // Erase the tribute.var operation - value is remapped
            return vec![];
        }
        // If binding not found in pattern bindings, keep the operation (regular local variable)

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

        // Check if any arm has a handler pattern - these are lowered to cont.handler_dispatch
        let has_handler_patterns = arms.iter().any(|arm| {
            matches!(
                arm.pattern,
                ArmPattern::HandlerDone | ArmPattern::HandlerSuspend { .. }
            )
        });
        if has_handler_patterns {
            return self.lower_handler_case(op, scrutinee, result_type, &arms);
        }

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

    /// Lower handler expressions to `cont.handler_dispatch`.
    ///
    /// Handler arms come from `handle expr { ... }` and contain:
    /// - `{ result }` (HandlerDone) - matches normal completion
    /// - `{ Op(args) -> k }` (HandlerSuspend) - matches effect operations
    ///
    /// This lowers to `cont.handler_dispatch` which the WASM backend
    /// will convert to yield-checking dispatch logic.
    fn lower_handler_case(
        &mut self,
        op: Operation<'db>,
        scrutinee: Value<'db>,
        result_type: Type<'db>,
        arms: &[ArmInfo<'db>],
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        // Find the done arm and suspend arms
        let mut done_arm: Option<&ArmInfo<'db>> = None;
        let mut suspend_arms: Vec<&ArmInfo<'db>> = Vec::new();

        for arm in arms {
            match &arm.pattern {
                ArmPattern::HandlerDone => {
                    done_arm = Some(arm);
                }
                ArmPattern::HandlerSuspend { .. } => {
                    suspend_arms.push(arm);
                }
                _ => {
                    // Other patterns in handler case - emit warning and skip
                    self.emit_error(location, "unexpected pattern in handler case expression");
                }
            }
        }

        // Process done_body - lower the body region
        let done_body = if let Some(done_arm) = done_arm {
            // Set up bindings for the result value
            // The done arm's pattern region may have a bind pattern for the result
            if let Some(pattern_region) = done_arm.pattern_region {
                let bindings = self.extract_bindings_from_pattern(pattern_region);
                for name in bindings.iter() {
                    // In handler_done, the result is the scrutinee (push_prompt result)
                    self.current_arm_bindings.insert(*name, scrutinee);
                }
            }

            let lowered_body = self.lower_region(done_arm.body);

            // Clear bindings after processing
            self.current_arm_bindings.clear();

            lowered_body
        } else {
            // No done arm - create empty region (unreachable case)
            Region::new(self.db, location, IdVec::new())
        };

        // Process suspend arms - build dispatch info for each operation
        // Each suspend arm has an op_name, and we compute a hash-based index for dispatch
        let mut suspend_bodies: Vec<(u64, Region<'db>)> = Vec::new();

        for suspend_arm in &suspend_arms {
            // Get the operation name from the pattern
            let op_name = match &suspend_arm.pattern {
                ArmPattern::HandlerSuspend { op_name } => *op_name,
                _ => continue,
            };

            // Compute hash-based index (same algorithm as in cont_to_wasm.rs)
            let op_idx: u64 = op_name.with_str(compute_op_idx_hash);

            // Set up bindings for continuation and args
            // Generate extraction operations that will be lowered by WASM backend
            let mut extraction_ops: Vec<Operation<'db>> = Vec::new();

            if let Some(pattern_region) = suspend_arm.pattern_region {
                let bindings = self.extract_bindings_from_pattern(pattern_region);
                // The last binding is typically the continuation (k)
                // Other bindings are operation arguments (shift_value)

                // Use core::Ptr as a placeholder type for opaque references
                // The WASM backend will determine the actual types during lowering
                let ptr_ty = core::Ptr::new(self.db).as_type();

                for (i, name) in bindings.iter().enumerate() {
                    if i == bindings.len() - 1 {
                        // Last binding is the continuation (k)
                        let get_cont_op =
                            cont::get_continuation(self.db, location, ptr_ty).as_operation();
                        let cont_val = get_cont_op.result(self.db, 0);
                        extraction_ops.push(get_cont_op);
                        self.current_arm_bindings.insert(*name, cont_val);
                    } else {
                        // Other bindings are shift_value arguments
                        let get_shift_op =
                            cont::get_shift_value(self.db, location, ptr_ty).as_operation();
                        let shift_val = get_shift_op.result(self.db, 0);
                        extraction_ops.push(get_shift_op);
                        self.current_arm_bindings.insert(*name, shift_val);
                    }
                }
            }

            let lowered_body = self.lower_region(suspend_arm.body);

            // Prepend extraction ops to the lowered body
            let body_with_extractions = if !extraction_ops.is_empty() {
                let body_blocks = lowered_body.blocks(self.db);
                if let Some(first_block) = body_blocks.first().copied() {
                    // Prepend extraction ops to the first block
                    let mut new_ops = IdVec::from(extraction_ops);
                    new_ops.extend(first_block.operations(self.db).iter().copied());
                    let new_block = Block::new(
                        self.db,
                        first_block.id(self.db),
                        first_block.location(self.db),
                        first_block.args(self.db).clone(),
                        new_ops,
                    );

                    // Rebuild the region with the new first block
                    let mut new_blocks: Vec<Block<'db>> = vec![new_block];
                    new_blocks.extend(body_blocks.iter().skip(1).copied());
                    Region::new(
                        self.db,
                        lowered_body.location(self.db),
                        IdVec::from(new_blocks),
                    )
                } else {
                    // No blocks - create a block with just the extraction ops
                    let block = Block::new(
                        self.db,
                        BlockId::fresh(),
                        location,
                        IdVec::new(),
                        IdVec::from(extraction_ops),
                    );
                    Region::new(self.db, location, IdVec::from(vec![block]))
                }
            } else {
                lowered_body
            };

            // Clear bindings after processing
            self.current_arm_bindings.clear();

            suspend_bodies.push((op_idx, body_with_extractions));
        }

        // Note: Each suspend body is stored as a separate region in handler_dispatch.
        // The WASM backend will read num_suspend_arms and the op_idx_N attributes
        // to generate proper dispatch logic.

        // Build the handler_dispatch operation with all suspend arms as regions
        // Region 0: done_body
        // Region 1+: suspend_bodies (one per handler arm)
        // Note: Using Operation::of_name here because we need to add dynamic regions
        // and attributes (op_idx_N, num_suspend_arms) that the typed helper doesn't support
        let mut builder = Operation::of_name(self.db, location, "cont.handler_dispatch")
            .operand(scrutinee)
            .result(result_type)
            .region(done_body);

        // Add all suspend body regions
        for (i, (op_idx, body)) in suspend_bodies.iter().enumerate() {
            // Store each suspend arm as a separate region with its op_idx
            let attr_name = format!("op_idx_{}", i);
            builder = builder.region(*body).attr(
                Symbol::from_dynamic(&attr_name),
                Attribute::IntBits(*op_idx),
            );
        }

        // Store the number of suspend arms for the WASM backend
        builder = builder.attr(
            "num_suspend_arms",
            Attribute::IntBits(suspend_bodies.len() as u64),
        );

        let dispatch_op = builder.build();

        // Map original case result to dispatch result
        self.ctx.map_results(self.db, &op, &dispatch_op);

        vec![dispatch_op]
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
            name if name == tribute_pat::HANDLER_DONE() => Some((ArmPattern::HandlerDone, true)),
            name if name == tribute_pat::HANDLER_SUSPEND() => {
                let op_name = op
                    .attributes(self.db)
                    .get(&Symbol::new("op"))
                    .and_then(|attr| match attr {
                        Attribute::Symbol(s) => Some(*s),
                        _ => None,
                    })
                    .unwrap_or_else(|| Symbol::new("?"));
                Some((ArmPattern::HandlerSuspend { op_name }, true))
            }
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

    /// Extract binding names from a pattern region.
    /// For simple `tribute_pat.bind("x")` patterns, returns the binding name.
    /// For variant patterns with bindings, returns all nested binding names.
    fn extract_bindings_from_pattern(&self, region: Region<'db>) -> SymbolVec {
        let mut bindings = SymbolVec::new();
        self.collect_bindings_recursive(region, &mut bindings);
        bindings
    }

    fn collect_bindings_recursive(&self, region: Region<'db>, bindings: &mut SymbolVec) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter().copied() {
                if op.dialect(self.db) == tribute_pat::DIALECT_NAME()
                    && op.name(self.db) == tribute_pat::BIND()
                    && let Some(Attribute::Symbol(name)) =
                        op.attributes(self.db).get(&Symbol::new("name"))
                {
                    bindings.push(*name);
                }
                // Recurse into nested regions (for variant fields, etc.)
                for nested_region in op.regions(self.db).iter().copied() {
                    self.collect_bindings_recursive(nested_region, bindings);
                }
            }
        }
    }

    /// Collect `adt.variant_get` operations from an arm body.
    /// Returns a map from field index to the result value.
    fn collect_variant_get_ops(
        &self,
        body: Region<'db>,
        scrutinee: Value<'db>,
    ) -> HashMap<u64, Value<'db>> {
        let mut field_extractions = HashMap::new();
        let expected_ref = self.ctx.lookup(scrutinee);

        for block in body.blocks(self.db).iter() {
            for op in block.operations(self.db).iter().copied() {
                if let Ok(vget) = adt::VariantGet::from_operation(self.db, op) {
                    // Check if this variant_get operates on our scrutinee
                    let operands = op.operands(self.db);
                    if !operands.is_empty() {
                        let ref_operand = operands[0];
                        // The ref operand may be the original scrutinee
                        // (tirgen passes scrutinee to variant_get)
                        if ref_operand == scrutinee
                            || ref_operand == expected_ref
                            || self.ctx.lookup(ref_operand) == expected_ref
                        {
                            // Get field index from the operation
                            if let Attribute::IntBits(idx) = vget.field(self.db) {
                                let result = op.result(self.db, 0);
                                field_extractions.insert(idx, result);
                            }
                        }
                    }
                }
            }
        }

        field_extractions
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
            // Handler patterns are always refutable (need all cases for exhaustiveness)
            ArmPattern::HandlerDone | ArmPattern::HandlerSuspend { .. } => false,
        }
    }

    /// Lower an arm body with pattern bindings set up.
    /// For simple bind patterns, the scrutinee IS the bound value.
    /// For variant patterns, we insert a `variant_cast` to convert the scrutinee
    /// to the variant-specific type before field access.
    fn lower_arm_body(&mut self, scrutinee: Value<'db>, arm: &ArmInfo<'db>) -> Region<'db> {
        let location = arm.body.location(self.db);

        // For variant patterns, create a cast operation to get variant-specific reference
        let cast_info = if let ArmPattern::Variant(variant_name) = &arm.pattern {
            // Get the enum type from scrutinee
            if let Some(enum_type) = self.value_type(scrutinee) {
                // Create variant_cast: casts scrutinee to variant-specific type
                // The result type uses the same enum type (will be refined to variant type in lowering)
                let cast = adt::variant_cast(
                    self.db,
                    location,
                    scrutinee,
                    enum_type, // result type (will become ref $Enum$Variant)
                    enum_type, // enum type for the cast
                    *variant_name,
                )
                .as_operation();

                let cast_result = cast.result(self.db, 0);

                // Temporarily map scrutinee -> cast result for this arm only
                self.ctx.map_value(scrutinee, cast_result);

                Some((cast, cast_result))
            } else {
                None
            }
        } else {
            None
        };

        // Extract bindings from pattern and map them to extracted field values
        if let Some(pattern_region) = arm.pattern_region {
            let bindings = self.extract_bindings_from_pattern(pattern_region);

            if matches!(arm.pattern, ArmPattern::Variant(_)) {
                // For variant patterns, find adt.variant_get ops in the body
                // and map bindings to their results (by field index order)
                let field_extractions = self.collect_variant_get_ops(arm.body, scrutinee);

                for (i, name) in bindings.iter().enumerate() {
                    if let Some(&field_value) = field_extractions.get(&(i as u64)) {
                        self.current_arm_bindings.insert(*name, field_value);
                    } else {
                        // Fallback: use cast result if no variant_get found
                        let bound_value = self.ctx.lookup(scrutinee);
                        self.current_arm_bindings.insert(*name, bound_value);
                    }
                }
            } else {
                // For simple patterns (wildcard, bind, literal), use scrutinee directly
                let bound_value = self.ctx.lookup(scrutinee);
                for name in bindings {
                    self.current_arm_bindings.insert(name, bound_value);
                }
            }
        }

        // Lower the body with bindings in scope
        let lowered = self.lower_region(arm.body);

        // Clear bindings after processing this arm
        self.current_arm_bindings.clear();

        // Restore the scrutinee mapping (remove the temporary cast mapping)
        // This is important so that other arms don't see the cast mapping
        if cast_info.is_some() {
            // Reset scrutinee mapping to itself (effectively removing the cast mapping)
            self.ctx.map_value(scrutinee, scrutinee);
        }

        // If we created a cast op, prepend it to the first block
        if let Some((cast_op, _)) = cast_info {
            self.prepend_op_to_region(lowered, cast_op)
        } else {
            lowered
        }
    }

    /// Prepend an operation to the first block of a region.
    fn prepend_op_to_region(&self, region: Region<'db>, op: Operation<'db>) -> Region<'db> {
        let blocks = region.blocks(self.db);
        if blocks.is_empty() {
            return region;
        }

        let first_block = blocks[0];
        let mut new_ops = IdVec::from(vec![op]);
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
            // Handler patterns are skipped in lower_case, so these should never be reached
            ArmPattern::HandlerDone | ArmPattern::HandlerSuspend { .. } => {
                unreachable!("Handler patterns should be skipped in lower_case")
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

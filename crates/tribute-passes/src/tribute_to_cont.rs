//! Lower tribute handler operations to cont dialect.
//!
//! This pass transforms handler-related operations:
//! - `tribute.handle` → `cont.push_prompt` + `cont.handler_dispatch`
//!
//! Handler arms are converted to:
//! - `{ result }` (done pattern) → done_body in handler_dispatch
//! - `{ Op(args) -> k }` (suspend pattern) → suspend arms in handler_dispatch
//!
//! This pass should run BEFORE `tribute_to_scf` as it handles handler-specific
//! patterns that would otherwise be passed through to scf lowering.
//!
//! Uses `RewritePattern` + `PatternApplicator` for declarative transformation.

use std::collections::HashMap;

use salsa::Accumulator;
use std::collections::BTreeMap;
use tribute_ir::dialect::{tribute, tribute_pat};
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{cont, core, scf};
use trunk_ir::rewrite::{
    ConversionTarget, OpAdaptor, PatternApplicator, RewriteContext, RewritePattern, RewriteResult,
    TypeConverter,
};
use trunk_ir::{
    Attribute, Block, BlockArg, BlockId, DialectOp, DialectType, IdVec, Location, Operation,
    Region, SymbolVec, Type,
};
use trunk_ir::{Symbol, Value, ValueDef};

use crate::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};

/// Handler pattern types for matching in handler arms.
///
/// Handler arms must contain exactly one of these patterns.
/// Other patterns (bind, wildcard, etc.) should not appear in handler arms
/// and indicate a frontend bug if encountered.
#[derive(Debug, Clone)]
enum HandlerPattern<'db> {
    /// Handler done pattern: `{ result }` - matches normal completion
    Done,
    /// Handler suspend pattern: `{ Op(args) -> k }` - matches effect operations
    Suspend {
        /// The ability type (for distinguishing same-named operations across abilities)
        ability_ref: Type<'db>,
        /// The operation name within the ability
        op_name: Symbol,
    },
}

/// Information about a handler arm.
#[derive(Debug, Clone)]
struct HandlerArmInfo<'db> {
    pattern: HandlerPattern<'db>,
    pattern_region: Option<Region<'db>>,
    body: Region<'db>,
}

/// Lower tribute handler operations to cont dialect.
///
/// Uses `PatternApplicator` for declarative transformation.
pub fn lower_tribute_to_cont<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    let applicator =
        PatternApplicator::new(TypeConverter::new()).add_pattern(LowerTributeHandlePattern);
    let target = ConversionTarget::new();
    applicator.apply_partial(db, module, target).module
}

// =============================================================================
// Pattern Implementation
// =============================================================================

/// Pattern to lower `tribute.handle` to `cont.push_prompt` + `cont.handler_dispatch`.
struct LowerTributeHandlePattern;

impl<'db> RewritePattern<'db> for LowerTributeHandlePattern {
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        _adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db> {
        // Match: tribute.handle
        if tribute::Handle::from_operation(db, *op).is_err() {
            return RewriteResult::Unchanged;
        };

        // Use HandlerLowerer for the complex transformation
        let mut lowerer = HandlerLowerer::new(db);
        let ops = lowerer.lower_handle(*op);

        if ops.len() == 1 && ops[0] == *op {
            return RewriteResult::Unchanged;
        }

        RewriteResult::expand(ops)
    }
}

// =============================================================================
// Handler Lowerer (Internal Helper)
// =============================================================================

/// Lowerer for handler expressions to continuation operations.
struct HandlerLowerer<'db> {
    db: &'db dyn salsa::Database,
    ctx: RewriteContext<'db>,
    /// Bindings for pattern variables in the current arm being processed.
    current_arm_bindings: HashMap<Symbol, Value<'db>>,
}

impl<'db> HandlerLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            ctx: RewriteContext::new(),
            current_arm_bindings: HashMap::new(),
        }
    }

    fn lower_region(&mut self, region: Region<'db>) -> Region<'db> {
        let blocks: Vec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|b| self.lower_block(*b))
            .collect();
        Region::new(self.db, region.location(self.db), IdVec::from(blocks))
    }

    fn lower_block(&mut self, block: Block<'db>) -> Block<'db> {
        let mut new_ops = Vec::new();
        for op in block.operations(self.db).iter() {
            let lowered = self.lower_op(*op);
            new_ops.extend(lowered);
        }
        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            block.args(self.db).clone(),
            IdVec::from(new_ops),
        )
    }

    fn lower_op(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);

        // Remap operands through the context
        let remapped_operands: IdVec<Value<'db>> = op
            .operands(self.db)
            .iter()
            .map(|v| self.remap_value(*v))
            .collect();

        // Handle tribute.handle operations
        if dialect == tribute::DIALECT_NAME() && name == tribute::HANDLE() {
            return self.lower_handle(op);
        }

        // For other operations, recursively lower nested regions
        let new_regions: IdVec<Region<'db>> = op
            .regions(self.db)
            .iter()
            .map(|r| self.lower_region(*r))
            .collect();

        // Rebuild the operation with remapped operands and lowered regions
        let new_op = op
            .modify(self.db)
            .operands(remapped_operands)
            .regions(new_regions)
            .build();

        self.ctx.map_results(self.db, &op, &new_op);
        vec![new_op]
    }

    /// Lower `tribute.handle` to `cont.push_prompt` + `cont.handler_dispatch`.
    fn lower_handle(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        // Generate deterministic tag based on operation's location
        let location = op.location(self.db);
        let tag = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            location.hash(&mut hasher);
            (hasher.finish() & 0xFFFFFFFF) as u32
        };

        let result_type = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Nil::new(self.db).as_type());

        // Get body and arms regions from tribute.handle
        let Ok(handle_op) = tribute::Handle::from_operation(self.db, op) else {
            return vec![op];
        };
        let body_region = handle_op.body(self.db);
        let arms_region = handle_op.arms(self.db);

        // Lower the body region
        let lowered_body = self.lower_region(body_region);

        // Create empty handlers region for push_prompt
        let empty_handlers = Region::new(
            self.db,
            location,
            IdVec::from(vec![Block::new(
                self.db,
                BlockId::fresh(),
                location,
                IdVec::new(),
                IdVec::new(),
            )]),
        );

        // Create cont.push_prompt with the body
        let push_prompt = cont::push_prompt(
            self.db,
            location,
            result_type,
            tag,
            lowered_body,
            empty_handlers,
        );
        let push_prompt_result = push_prompt.as_operation().result(self.db, 0);

        // Collect arms from the arms region
        let arms = self.collect_handler_arms(arms_region);

        // Process done arm and suspend arms
        let mut done_arm: Option<&HandlerArmInfo<'db>> = None;
        let mut suspend_arms: Vec<&HandlerArmInfo<'db>> = Vec::new();

        for arm in &arms {
            match &arm.pattern {
                HandlerPattern::Done => {
                    done_arm = Some(arm);
                }
                HandlerPattern::Suspend { .. } => {
                    suspend_arms.push(arm);
                }
            }
        }

        // Process done_body
        let get_done_value_op =
            cont::get_done_value(self.db, location, push_prompt_result, result_type);
        let done_value = get_done_value_op.as_operation().result(self.db, 0);

        let done_body = if let Some(done_arm) = done_arm {
            // Bind result variable to the extracted done value
            if let Some(pattern_region) = done_arm.pattern_region {
                let bindings = self.extract_bindings_from_pattern(pattern_region);
                for name in bindings.iter() {
                    self.current_arm_bindings.insert(*name, done_value);
                }
            }

            let lowered_body = self.lower_region(done_arm.body);
            self.current_arm_bindings.clear();

            // Prepend the get_done_value operation to the body
            let body_with_extraction =
                self.prepend_op_to_region(lowered_body, get_done_value_op.as_operation());

            // Ensure the done_body ends with a yield operation
            self.ensure_region_yields(body_with_extraction, done_value, location)
        } else {
            // No done arm - extract value and return it directly
            let yield_op = scf::r#yield(self.db, location, vec![done_value]);
            let block = Block::new(
                self.db,
                BlockId::fresh(),
                location,
                IdVec::new(),
                IdVec::from(vec![
                    get_done_value_op.as_operation(),
                    yield_op.as_operation(),
                ]),
            );
            Region::new(self.db, location, IdVec::from(vec![block]))
        };

        // Process suspend arms: (ability_ref, op_name, body)
        let mut suspend_bodies: Vec<(Type<'db>, Symbol, Region<'db>)> = Vec::new();

        for suspend_arm in &suspend_arms {
            let (ability_ref, op_name) = match &suspend_arm.pattern {
                HandlerPattern::Suspend {
                    ability_ref,
                    op_name,
                } => (*ability_ref, *op_name),
                _ => continue,
            };

            // Set up bindings for continuation and args
            let mut extraction_ops: Vec<Operation<'db>> = Vec::new();
            let ptr_ty = core::Ptr::new(self.db).as_type();

            if let Some(pattern_region) = suspend_arm.pattern_region {
                let bindings = self.extract_bindings_from_pattern(pattern_region);

                // Extract continuation type from handler_suspend operation
                // (set by tirgen, constrained by typeck, resolved by TypeSubst)
                let continuation_ty = self
                    .extract_continuation_type(pattern_region)
                    .unwrap_or(ptr_ty);

                for (i, name) in bindings.iter().enumerate() {
                    if i == bindings.len() - 1 {
                        // Last binding is continuation (k) - use the resolved continuation type
                        let get_cont_op =
                            cont::get_continuation(self.db, location, continuation_ty)
                                .as_operation();
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

            // Prepend extraction ops
            let body_with_extractions = if !extraction_ops.is_empty() {
                self.prepend_ops_to_region(lowered_body, &extraction_ops)
            } else {
                lowered_body
            };

            self.current_arm_bindings.clear();
            suspend_bodies.push((ability_ref, op_name, body_with_extractions));
        }

        // Build body region with multiple blocks
        let body_region = build_handler_body_region(self.db, location, done_body, &suspend_bodies);

        // Build cont.handler_dispatch using typed helper
        // Order: operand, output_type, tag_attr, result_type_attr, region
        // output_type and result_type_attr are the same here (both are the handler's result type)
        let dispatch_op = cont::handler_dispatch(
            self.db,
            location,
            push_prompt_result,
            result_type,
            tag,
            result_type,
            body_region,
        );

        self.ctx
            .map_results(self.db, &op, &dispatch_op.as_operation());

        vec![push_prompt.as_operation(), dispatch_op.as_operation()]
    }

    /// Collect handler arms from a tribute.handle's arms region.
    fn collect_handler_arms(&self, arms_region: Region<'db>) -> Vec<HandlerArmInfo<'db>> {
        let mut arms = Vec::new();

        for block in arms_region.blocks(self.db).iter() {
            for arm_op in block.operations(self.db).iter() {
                if let Ok(arm) = tribute::Arm::from_operation(self.db, *arm_op) {
                    let pattern_region = arm.pattern(self.db);
                    let body = arm.body(self.db);
                    let Some(pattern) = self.analyze_handler_pattern(pattern_region) else {
                        // Invalid pattern in handler arm - emit diagnostic and skip
                        self.emit_error(
                            arm_op.location(self.db),
                            "handler arm must contain handler_done or handler_suspend pattern",
                        );
                        continue;
                    };

                    arms.push(HandlerArmInfo {
                        pattern,
                        pattern_region: Some(pattern_region),
                        body,
                    });
                }
            }
        }

        arms
    }

    /// Analyze a handler pattern region to determine if it's done or suspend.
    ///
    /// Returns `None` if the pattern region does not contain a handler pattern,
    /// which indicates a frontend bug - handler arms must contain handler patterns.
    fn analyze_handler_pattern(&self, pattern_region: Region<'db>) -> Option<HandlerPattern<'db>> {
        for block in pattern_region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                if tribute_pat::HandlerDone::from_operation(self.db, *op).is_ok() {
                    return Some(HandlerPattern::Done);
                }
                if let Ok(suspend) = tribute_pat::HandlerSuspend::from_operation(self.db, *op) {
                    let ability_ref = suspend.ability_ref(self.db);
                    let op_name = suspend.op(self.db);
                    return Some(HandlerPattern::Suspend {
                        ability_ref,
                        op_name,
                    });
                }
            }
        }

        None
    }

    /// Remap a value through the rewrite context.
    fn remap_value(&self, value: Value<'db>) -> Value<'db> {
        // Check if this is a tribute.var referencing a bound pattern variable
        if let ValueDef::OpResult(def_op) = value.def(self.db)
            && def_op.dialect(self.db) == tribute::DIALECT_NAME()
            && def_op.name(self.db) == tribute::VAR()
            && let Some(Attribute::Symbol(name)) =
                def_op.attributes(self.db).get(&Symbol::new("name"))
            && let Some(&bound_value) = self.current_arm_bindings.get(name)
        {
            return bound_value;
        }
        self.ctx.lookup(value)
    }

    /// Extract binding names from a pattern region.
    fn extract_bindings_from_pattern(&self, region: Region<'db>) -> SymbolVec {
        let mut bindings = SymbolVec::new();
        self.collect_bindings_recursive(region, &mut bindings);
        bindings
    }

    fn collect_bindings_recursive(&self, region: Region<'db>, bindings: &mut SymbolVec) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                // Check for tribute_pat.bind
                if let Ok(bind) = tribute_pat::Bind::from_operation(self.db, *op) {
                    bindings.push(bind.name(self.db));
                }
                // Recurse into nested regions
                for nested in op.regions(self.db).iter() {
                    self.collect_bindings_recursive(*nested, bindings);
                }
            }
        }
    }

    /// Extract the continuation type from a handler_suspend pattern region.
    ///
    /// The continuation type is stored as an attribute on the handler_suspend operation,
    /// set by tirgen as a type variable and resolved by TypeSubst.
    fn extract_continuation_type(&self, pattern_region: Region<'db>) -> Option<Type<'db>> {
        use tribute_pat::handler_suspend_attrs::CONTINUATION_TYPE;

        for block in pattern_region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                if let Ok(_suspend) = tribute_pat::HandlerSuspend::from_operation(self.db, *op) {
                    // Check for continuation_type attribute
                    if let Some(Attribute::Type(cont_ty)) =
                        op.attributes(self.db).get(&CONTINUATION_TYPE())
                    {
                        return Some(*cont_ty);
                    }
                }
            }
        }
        None
    }

    /// Ensure a region ends with a yield operation.
    fn ensure_region_yields(
        &self,
        region: Region<'db>,
        yield_value: Value<'db>,
        location: Location<'db>,
    ) -> Region<'db> {
        let blocks = region.blocks(self.db);
        if blocks.is_empty() {
            // Create a block with just a yield
            let yield_op = scf::r#yield(self.db, location, vec![yield_value]);
            let block = Block::new(
                self.db,
                BlockId::fresh(),
                location,
                IdVec::new(),
                IdVec::from(vec![yield_op.as_operation()]),
            );
            return Region::new(self.db, location, IdVec::from(vec![block]));
        }

        let last_block = blocks.last().copied().unwrap();
        let ops = last_block.operations(self.db);

        // Check if last operation is already a yield
        if let Some(last_op) = ops.last() {
            let dialect = last_op.dialect(self.db);
            let name = last_op.name(self.db);
            if (dialect == scf::DIALECT_NAME() && name == scf::YIELD())
                || (dialect == tribute::DIALECT_NAME() && name == tribute::YIELD())
            {
                return region;
            }
        }

        // Add yield at the end
        let yield_op = scf::r#yield(self.db, location, vec![yield_value]);
        let mut new_ops: IdVec<Operation<'db>> = ops.clone();
        new_ops.push(yield_op.as_operation());

        let new_block = Block::new(
            self.db,
            last_block.id(self.db),
            last_block.location(self.db),
            last_block.args(self.db).clone(),
            new_ops,
        );

        let mut new_blocks: Vec<Block<'db>> = blocks.iter().copied().collect();
        new_blocks.pop();
        new_blocks.push(new_block);

        Region::new(self.db, region.location(self.db), IdVec::from(new_blocks))
    }

    /// Prepend an operation to the first block of a region.
    fn prepend_op_to_region(&self, region: Region<'db>, op: Operation<'db>) -> Region<'db> {
        self.prepend_ops_to_region(region, &[op])
    }

    /// Prepend operations to the first block of a region.
    fn prepend_ops_to_region(&self, region: Region<'db>, ops: &[Operation<'db>]) -> Region<'db> {
        let blocks = region.blocks(self.db);
        if let Some(first_block) = blocks.first().copied() {
            let mut new_ops = IdVec::from(ops.to_vec());
            new_ops.extend(first_block.operations(self.db).iter().copied());
            let new_block = Block::new(
                self.db,
                first_block.id(self.db),
                first_block.location(self.db),
                first_block.args(self.db).clone(),
                new_ops,
            );
            let mut new_blocks: Vec<Block<'db>> = vec![new_block];
            new_blocks.extend(blocks.iter().skip(1).copied());
            Region::new(self.db, region.location(self.db), IdVec::from(new_blocks))
        } else {
            // No blocks - create one with the ops
            let block = Block::new(
                self.db,
                BlockId::fresh(),
                region.location(self.db),
                IdVec::new(),
                IdVec::from(ops.to_vec()),
            );
            Region::new(self.db, region.location(self.db), IdVec::from(vec![block]))
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
}

/// Build a handler body region with multiple blocks for handler_dispatch.
///
/// The resulting region has:
/// - Block 0: done case (from done_body)
/// - Block 1+: suspend cases, each with a marker block arg containing ability_ref + op_name
fn build_handler_body_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    done_body: Region<'db>,
    suspend_arms: &[(Type<'db>, Symbol, Region<'db>)],
) -> Region<'db> {
    let mut all_blocks = Vec::new();

    // Block 0: done case (use blocks from done_body)
    all_blocks.extend(done_body.blocks(db).iter().cloned());

    // Block 1+: suspend cases
    for (ability_ref, op_name, suspend_body) in suspend_arms {
        for block in suspend_body.blocks(db).iter() {
            // Create marker block arg with ability_ref and op_name attributes
            let nil_ty = core::Nil::new(db).as_type();
            let marker_arg = BlockArg::new(
                db,
                nil_ty,
                BTreeMap::from([
                    (Symbol::new("ability_ref"), Attribute::Type(*ability_ref)),
                    (Symbol::new("op_name"), Attribute::Symbol(*op_name)),
                ]),
            );

            // Prepend marker arg to existing block args
            let mut new_args = IdVec::from(vec![marker_arg]);
            new_args.extend(block.args(db).iter().cloned());

            all_blocks.push(Block::new(
                db,
                block.id(db),
                block.location(db),
                new_args,
                block.operations(db).clone(),
            ));
        }
    }

    Region::new(db, location, IdVec::from(all_blocks))
}

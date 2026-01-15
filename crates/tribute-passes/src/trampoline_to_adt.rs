//! Lower trampoline dialect operations to ADT operations.
//!
//! This pass transforms trampoline operations to ADT struct operations:
//! - `trampoline.build_continuation` → `adt.struct_new` (Continuation type)
//! - `trampoline.step_done` → `adt.struct_new` (Step type with tag=0)
//! - `trampoline.step_shift` → `adt.struct_new` (Step type with tag=1)
//! - `trampoline.step_get` → `adt.struct_get`
//! - `trampoline.continuation_get` → `adt.struct_get`
//! - `trampoline.build_state` → `adt.struct_new` (custom state type)
//! - `trampoline.build_resume_wrapper` → `adt.struct_new` (ResumeWrapper type)
//! - `trampoline.resume_wrapper_get` → `adt.struct_get`
//!
//! This pass should run after `cont_to_trampoline` and before backend-specific lowering.

use tribute_ir::dialect::{adt, tribute_rt};
use trunk_ir::dialect::arith;
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::trampoline::{self, STEP_TAG_DONE, STEP_TAG_SHIFT};
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Location, Operation, Region, Symbol, Type,
    Value,
};

/// Lower trampoline dialect operations to ADT operations.
pub fn lower_trampoline_to_adt<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    TrampolineToAdtLowerer::new(db).lower_module(module)
}

/// Lowerer for trampoline operations to ADT operations.
struct TrampolineToAdtLowerer<'db> {
    db: &'db dyn salsa::Database,
    ctx: RewriteContext<'db>,
    /// Cached Step ADT type
    step_type: Type<'db>,
    /// Cached Continuation ADT type
    continuation_type: Type<'db>,
    /// Cached ResumeWrapper ADT type
    resume_wrapper_type: Type<'db>,
}

impl<'db> TrampolineToAdtLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        let ptr_ty = core::Ptr::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();

        // Step type: ADT struct with layout (tag: i32, value: any, prompt: i32, op_idx: i32)
        // This is lowered to a GC struct by adt_to_wasm
        // Using tribute_rt.any for value field (type-erased reference)
        let any_ty = tribute_rt::Any::new(db).as_type();
        let step_type = adt::struct_type(
            db,
            "_Step",
            vec![
                (Symbol::new("tag"), i32_ty),
                (Symbol::new("value"), any_ty),
                (Symbol::new("prompt"), i32_ty),
                (Symbol::new("op_idx"), i32_ty),
            ],
        );

        // Continuation type: (resume_fn: ptr, state: any, tag: i32, shift_value: any)
        // state is type-erased because each shift point captures different locals
        // shift_value is type-erased value passed to effect operation
        let continuation_type = adt::struct_type(
            db,
            "_Continuation",
            vec![
                (Symbol::new("resume_fn"), ptr_ty),
                (Symbol::new("state"), any_ty),
                (Symbol::new("tag"), i32_ty),
                (Symbol::new("shift_value"), any_ty),
            ],
        );

        // ResumeWrapper type: (state: any, resume_value: any)
        // state is type-erased because each shift point captures different locals
        // resume_value is type-erased value passed when resuming
        let resume_wrapper_type = adt::struct_type(
            db,
            "_ResumeWrapper",
            vec![
                (Symbol::new("state"), any_ty),
                (Symbol::new("resume_value"), any_ty),
            ],
        );

        Self {
            db,
            ctx: RewriteContext::new(),
            step_type,
            continuation_type,
            resume_wrapper_type,
        }
    }

    fn lower_module(&mut self, module: Module<'db>) -> Module<'db> {
        let body = module.body(self.db);
        let lowered = self.lower_region(body);
        Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            lowered,
        )
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
            .map(|v| self.ctx.lookup(*v))
            .collect();

        // Handle trampoline dialect operations
        if dialect == trampoline::DIALECT_NAME() {
            if name == trampoline::BUILD_CONTINUATION() {
                return self.lower_build_continuation(op, &remapped_operands);
            }
            if name == trampoline::STEP_DONE() {
                return self.lower_step_done(op, &remapped_operands);
            }
            if name == trampoline::STEP_SHIFT() {
                return self.lower_step_shift(op, &remapped_operands);
            }
            if name == trampoline::STEP_GET() {
                return self.lower_step_get(op, &remapped_operands);
            }
            if name == trampoline::CONTINUATION_GET() {
                return self.lower_continuation_get(op, &remapped_operands);
            }
            if name == trampoline::BUILD_STATE() {
                return self.lower_build_state(op, &remapped_operands);
            }
            if name == trampoline::BUILD_RESUME_WRAPPER() {
                return self.lower_build_resume_wrapper(op, &remapped_operands);
            }
            if name == trampoline::RESUME_WRAPPER_GET() {
                return self.lower_resume_wrapper_get(op, &remapped_operands);
            }
            if name == trampoline::STATE_GET() {
                return self.lower_state_get(op, &remapped_operands);
            }
            // Pass through other trampoline operations (check_yield, set_yield_state, etc.)
            // These will be handled by backend-specific lowering
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

    /// Lower `trampoline.build_continuation` → `adt.struct_new`
    fn lower_build_continuation(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let any_ty = tribute_rt::Any::new(self.db).as_type();

        // Get tag from attribute
        let tag = op
            .attributes(self.db)
            .get(&Symbol::new("tag"))
            .and_then(|a| match a {
                Attribute::IntBits(n) => Some(*n as i32),
                _ => None,
            })
            .unwrap_or(0);

        // operands: (resume_fn, state, shift_value)
        let resume_fn = operands.first().copied();
        let state = operands.get(1).copied();
        let shift_value = operands.get(2).copied();

        // Create tag constant
        let tag_const = self.create_i32_const(location, tag);
        let tag_value = tag_const.result(self.db, 0);

        let mut ops = vec![tag_const];

        // Build struct fields: (resume_fn, state, tag, shift_value)
        // All fields must be present - use ref.null for missing values
        let resume_fn_field = if let Some(v) = resume_fn {
            v
        } else {
            let null_op = adt::ref_null(self.db, location, any_ty, any_ty);
            let null_val = null_op.as_operation().result(self.db, 0);
            ops.push(null_op.as_operation());
            null_val
        };

        // State is type-erased (any) - different shift points capture different locals
        // Cast state to any type to ensure uniform field type in Continuation struct
        let state_field = if let Some(v) = state {
            let cast_op = adt::ref_cast(self.db, location, v, any_ty, any_ty);
            let cast_val = cast_op.as_operation().result(self.db, 0);
            ops.push(cast_op.as_operation());
            cast_val
        } else {
            let null_op = adt::ref_null(self.db, location, any_ty, any_ty);
            let null_val = null_op.as_operation().result(self.db, 0);
            ops.push(null_op.as_operation());
            null_val
        };

        // Shift value is also type-erased - cast to any
        let shift_value_field = if let Some(v) = shift_value {
            let cast_op = adt::ref_cast(self.db, location, v, any_ty, any_ty);
            let cast_val = cast_op.as_operation().result(self.db, 0);
            ops.push(cast_op.as_operation());
            cast_val
        } else {
            let null_op = adt::ref_null(self.db, location, any_ty, any_ty);
            let null_val = null_op.as_operation().result(self.db, 0);
            ops.push(null_op.as_operation());
            null_val
        };

        let fields = vec![resume_fn_field, state_field, tag_value, shift_value_field];

        let struct_new = adt::struct_new(
            self.db,
            location,
            fields,
            self.continuation_type,
            self.continuation_type,
        );
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        ops.push(struct_new);
        ops
    }

    /// Lower `trampoline.step_done` → `adt.struct_new` with tag=0
    fn lower_step_done(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        // operands: (value,)
        let value = operands.first().copied();

        // Create constants for tag=0, prompt=0, op_idx=0
        let tag_const = self.create_i32_const(location, STEP_TAG_DONE);
        let prompt_const = self.create_i32_const(location, 0);
        let op_idx_const = self.create_i32_const(location, 0);

        let tag_value = tag_const.result(self.db, 0);
        let prompt_value = prompt_const.result(self.db, 0);
        let op_idx_value = op_idx_const.result(self.db, 0);

        let mut ops = vec![tag_const, prompt_const, op_idx_const];

        // Build struct fields: (tag, value, prompt, op_idx)
        // Value field is type-erased (any) - cast value to any
        let any_ty = tribute_rt::Any::new(self.db).as_type();
        let value_field = if let Some(v) = value {
            let cast_op = adt::ref_cast(self.db, location, v, any_ty, any_ty);
            let cast_val = cast_op.as_operation().result(self.db, 0);
            ops.push(cast_op.as_operation());
            cast_val
        } else {
            let null_op = adt::ref_null(self.db, location, any_ty, any_ty);
            let null_val = null_op.as_operation().result(self.db, 0);
            ops.push(null_op.as_operation());
            null_val
        };

        let fields = vec![tag_value, value_field, prompt_value, op_idx_value];
        let struct_new = adt::struct_new(self.db, location, fields, self.step_type, self.step_type);
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        ops.push(struct_new);
        ops
    }

    /// Lower `trampoline.step_shift` → `adt.struct_new` with tag=1
    fn lower_step_shift(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        // Get prompt and op_idx from attributes
        let prompt = op
            .attributes(self.db)
            .get(&Symbol::new("prompt"))
            .and_then(|a| match a {
                Attribute::IntBits(n) => Some(*n as i32),
                _ => None,
            })
            .unwrap_or(0);

        let op_idx = op
            .attributes(self.db)
            .get(&Symbol::new("op_idx"))
            .and_then(|a| match a {
                Attribute::IntBits(n) => Some(*n as i32),
                _ => None,
            })
            .unwrap_or(0);

        // operands: (continuation,)
        let continuation = operands.first().copied();

        // Create constants
        let tag_const = self.create_i32_const(location, STEP_TAG_SHIFT);
        let prompt_const = self.create_i32_const(location, prompt);
        let op_idx_const = self.create_i32_const(location, op_idx);

        let tag_value = tag_const.result(self.db, 0);
        let prompt_value = prompt_const.result(self.db, 0);
        let op_idx_value = op_idx_const.result(self.db, 0);

        let mut ops = vec![tag_const, prompt_const, op_idx_const];

        // Build struct fields: (tag, value (continuation as any), prompt, op_idx)
        // Value field is type-erased (any) - cast continuation to any
        let any_ty = tribute_rt::Any::new(self.db).as_type();
        let value_field = if let Some(v) = continuation {
            let cast_op = adt::ref_cast(self.db, location, v, any_ty, any_ty);
            let cast_val = cast_op.as_operation().result(self.db, 0);
            ops.push(cast_op.as_operation());
            cast_val
        } else {
            let null_op = adt::ref_null(self.db, location, any_ty, any_ty);
            let null_val = null_op.as_operation().result(self.db, 0);
            ops.push(null_op.as_operation());
            null_val
        };

        let fields = vec![tag_value, value_field, prompt_value, op_idx_value];
        let struct_new = adt::struct_new(self.db, location, fields, self.step_type, self.step_type);
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        ops.push(struct_new);
        ops
    }

    /// Lower `trampoline.step_get` → `adt.struct_get` (+ cast for any fields)
    ///
    /// Step fields: tag=0 (i32), value=1 (any), prompt=2 (i32), op_idx=3 (i32)
    fn lower_step_get(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let any_ty = tribute_rt::Any::new(self.db).as_type();

        let field_name = op
            .attributes(self.db)
            .get(&Symbol::new("field"))
            .and_then(|a| match a {
                Attribute::Symbol(s) => Some(*s),
                _ => None,
            })
            .unwrap_or_else(|| Symbol::new("value"));

        // Convert field name to index
        // Step fields: tag=0, value=1, prompt=2, op_idx=3
        let field_idx = self.step_field_index(field_name);

        let expected_result_type = op.results(self.db).first().copied().unwrap_or(any_ty);

        let step_value = operands
            .first()
            .copied()
            .expect("step_get requires operand");

        // Check if this field is type-erased (any)
        // field 1 (value) is any type
        let is_any_field = field_idx == 1;

        if is_any_field {
            let mut ops = Vec::new();

            // Extract field as any type
            let struct_get = adt::struct_get(
                self.db,
                location,
                step_value,
                any_ty,
                self.step_type,
                Attribute::IntBits(field_idx),
            );
            let struct_get = struct_get.as_operation();
            let any_value = struct_get.result(self.db, 0);
            ops.push(struct_get);

            // Cast from any to expected result type
            let cast_op = adt::ref_cast(
                self.db,
                location,
                any_value,
                expected_result_type,
                expected_result_type,
            );
            let cast_op = cast_op.as_operation();
            ops.push(cast_op);

            self.ctx.map_results(self.db, &op, &cast_op);
            ops
        } else {
            // Non-any field: extract directly with expected type
            let struct_get = adt::struct_get(
                self.db,
                location,
                step_value,
                expected_result_type,
                self.step_type,
                Attribute::IntBits(field_idx),
            );
            let struct_get = struct_get.as_operation();

            self.ctx.map_results(self.db, &op, &struct_get);
            vec![struct_get]
        }
    }

    /// Lower `trampoline.continuation_get` → `adt.struct_get` (+ cast for any fields)
    ///
    /// Continuation fields: resume_fn=0 (ptr), state=1 (any), tag=2 (i32), shift_value=3 (any)
    fn lower_continuation_get(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let any_ty = tribute_rt::Any::new(self.db).as_type();

        let field_name = op
            .attributes(self.db)
            .get(&Symbol::new("field"))
            .and_then(|a| match a {
                Attribute::Symbol(s) => Some(*s),
                _ => None,
            })
            .unwrap_or_else(|| Symbol::new("value"));

        // Convert field name to index
        // Continuation fields: resume_fn=0, state=1, tag=2, shift_value=3
        let field_idx = self.continuation_field_index(field_name);

        let expected_result_type = op.results(self.db).first().copied().unwrap_or(any_ty);

        let cont_value = operands
            .first()
            .copied()
            .expect("continuation_get requires operand");

        // Check if this field is type-erased (any)
        // field 1 (state) and field 3 (shift_value) are any type
        let is_any_field = field_idx == 1 || field_idx == 3;

        if is_any_field {
            let mut ops = Vec::new();

            // Extract field as any type
            let struct_get = adt::struct_get(
                self.db,
                location,
                cont_value,
                any_ty,
                self.continuation_type,
                Attribute::IntBits(field_idx),
            );
            let struct_get = struct_get.as_operation();
            let any_value = struct_get.result(self.db, 0);
            ops.push(struct_get);

            // Cast from any to expected result type
            let cast_op = adt::ref_cast(
                self.db,
                location,
                any_value,
                expected_result_type,
                expected_result_type,
            );
            let cast_op = cast_op.as_operation();
            ops.push(cast_op);

            self.ctx.map_results(self.db, &op, &cast_op);
            ops
        } else {
            // Non-any field: extract directly with expected type
            let struct_get = adt::struct_get(
                self.db,
                location,
                cont_value,
                expected_result_type,
                self.continuation_type,
                Attribute::IntBits(field_idx),
            );
            let struct_get = struct_get.as_operation();

            self.ctx.map_results(self.db, &op, &struct_get);
            vec![struct_get]
        }
    }

    /// Lower `trampoline.build_state` → `adt.struct_new`
    fn lower_build_state(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        // Get state_type from attribute, or create a generic one
        let state_type = op
            .attributes(self.db)
            .get(&Symbol::new("state_type"))
            .and_then(|a| match a {
                Attribute::Type(t) => Some(*t),
                _ => None,
            })
            .unwrap_or_else(|| {
                // Create a generic state struct type with ptr fields
                let ptr_ty = core::Ptr::new(self.db).as_type();
                let fields: Vec<(Symbol, Type<'db>)> = operands
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (Symbol::from_dynamic(&format!("field{}", i)), ptr_ty))
                    .collect();
                adt::struct_type(self.db, "_State", fields)
            });

        let fields: Vec<Value<'db>> = operands.iter().copied().collect();
        let struct_new = adt::struct_new(self.db, location, fields, state_type, state_type);
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        vec![struct_new]
    }

    /// Lower `trampoline.build_resume_wrapper` → `adt.struct_new`
    fn lower_build_resume_wrapper(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let any_ty = tribute_rt::Any::new(self.db).as_type();

        let mut ops = Vec::new();

        // operands: (state, resume_value)
        // Both fields are type-erased (any) - cast to ensure uniform field types
        let state = operands.first().copied();
        let resume_value = operands.get(1).copied();

        // Cast state to any type
        let state_field = if let Some(v) = state {
            let cast_op = adt::ref_cast(self.db, location, v, any_ty, any_ty);
            let cast_val = cast_op.as_operation().result(self.db, 0);
            ops.push(cast_op.as_operation());
            cast_val
        } else {
            let null_op = adt::ref_null(self.db, location, any_ty, any_ty);
            let null_val = null_op.as_operation().result(self.db, 0);
            ops.push(null_op.as_operation());
            null_val
        };

        // Cast resume_value to any type
        let resume_value_field = if let Some(v) = resume_value {
            let cast_op = adt::ref_cast(self.db, location, v, any_ty, any_ty);
            let cast_val = cast_op.as_operation().result(self.db, 0);
            ops.push(cast_op.as_operation());
            cast_val
        } else {
            let null_op = adt::ref_null(self.db, location, any_ty, any_ty);
            let null_val = null_op.as_operation().result(self.db, 0);
            ops.push(null_op.as_operation());
            null_val
        };

        let fields = vec![state_field, resume_value_field];
        let struct_new = adt::struct_new(
            self.db,
            location,
            fields,
            self.resume_wrapper_type,
            self.resume_wrapper_type,
        );
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        ops.push(struct_new);
        ops
    }

    /// Lower `trampoline.resume_wrapper_get` → `adt.struct_get` + `adt.ref_cast`
    ///
    /// ResumeWrapper fields are type-erased (any), so we extract as any then cast
    /// to the expected result type.
    fn lower_resume_wrapper_get(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let any_ty = tribute_rt::Any::new(self.db).as_type();

        let field_name = op
            .attributes(self.db)
            .get(&Symbol::new("field"))
            .and_then(|a| match a {
                Attribute::Symbol(s) => Some(*s),
                _ => None,
            })
            .unwrap_or_else(|| Symbol::new("state"));

        // Convert field name to index
        // ResumeWrapper fields: state=0, resume_value=1
        let field_idx = self.resume_wrapper_field_index(field_name);

        let expected_result_type = op.results(self.db).first().copied().unwrap_or(any_ty);

        let wrapper_value = operands
            .first()
            .copied()
            .expect("resume_wrapper_get requires operand");

        let mut ops = Vec::new();

        // Extract field as any type (actual storage type)
        let struct_get = adt::struct_get(
            self.db,
            location,
            wrapper_value,
            any_ty, // Field type in ResumeWrapper is any
            self.resume_wrapper_type,
            Attribute::IntBits(field_idx),
        );
        let struct_get = struct_get.as_operation();
        let any_value = struct_get.result(self.db, 0);
        ops.push(struct_get);

        // Cast from any to expected result type
        let cast_op = adt::ref_cast(
            self.db,
            location,
            any_value,
            expected_result_type,
            expected_result_type,
        );
        let cast_op = cast_op.as_operation();
        ops.push(cast_op);

        self.ctx.map_results(self.db, &op, &cast_op);
        ops
    }

    /// Lower `trampoline.state_get` → `adt.struct_get`
    ///
    /// State struct fields are dynamically named: field0, field1, ...
    fn lower_state_get(
        &mut self,
        op: Operation<'db>,
        operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        let field_name = op
            .attributes(self.db)
            .get(&Symbol::new("field"))
            .and_then(|a| match a {
                Attribute::Symbol(s) => Some(*s),
                _ => None,
            })
            .unwrap_or_else(|| Symbol::new("field0"));

        // Parse field index from "fieldN" pattern
        let field_idx = self.state_field_index(field_name);

        let result_type = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Ptr::new(self.db).as_type());

        let state_value = operands
            .first()
            .copied()
            .expect("state_get requires operand");

        // Use tribute_rt.any as placeholder type since state structs vary per shift point.
        // The actual type_idx will be inferred from the operand at emit time.
        let any_ty = tribute_rt::Any::new(self.db).as_type();

        let struct_get = adt::struct_get(
            self.db,
            location,
            state_value,
            result_type,
            any_ty,
            Attribute::IntBits(field_idx),
        );
        let struct_get = struct_get.as_operation();

        self.ctx.map_results(self.db, &op, &struct_get);
        vec![struct_get]
    }

    /// Helper: Create an i32 constant operation using arith.const
    fn create_i32_const(&self, location: Location<'db>, value: i32) -> Operation<'db> {
        arith::Const::i32(self.db, location, value).as_operation()
    }

    /// Helper: Convert Step field name to index
    /// Step fields: tag=0, value=1, prompt=2, op_idx=3
    fn step_field_index(&self, field_name: Symbol) -> u64 {
        if field_name == "tag" {
            0
        } else if field_name == "value" {
            1
        } else if field_name == "prompt" {
            2
        } else if field_name == "op_idx" {
            3
        } else {
            1 // default to value
        }
    }

    /// Helper: Convert Continuation field name to index
    /// Continuation fields: resume_fn=0, state=1, tag=2, shift_value=3
    fn continuation_field_index(&self, field_name: Symbol) -> u64 {
        if field_name == "resume_fn" {
            0
        } else if field_name == "state" {
            1
        } else if field_name == "tag" {
            2
        } else if field_name == "shift_value" {
            3
        } else {
            0 // default to resume_fn
        }
    }

    /// Helper: Convert ResumeWrapper field name to index
    /// ResumeWrapper fields: state=0, resume_value=1
    fn resume_wrapper_field_index(&self, field_name: Symbol) -> u64 {
        if field_name == "state" {
            0
        } else if field_name == "resume_value" {
            1
        } else {
            0 // default to state
        }
    }

    /// Helper: Convert State field name to index
    /// State fields: field0=0, field1=1, field2=2, ...
    fn state_field_index(&self, field_name: Symbol) -> u64 {
        let name_str = field_name.to_string();
        if let Some(suffix) = name_str.strip_prefix("field") {
            suffix.parse::<u64>().unwrap_or(0)
        } else {
            0 // default to field0
        }
    }
}

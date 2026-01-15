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

use tribute_ir::dialect::adt;
use trunk_ir::dialect::arith::Const;
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

        // Step type: (tag: i32, value: ptr, prompt: i32, op_idx: i32)
        let step_type = adt::struct_type(
            db,
            "_Step",
            vec![
                (Symbol::new("tag"), i32_ty),
                (Symbol::new("value"), ptr_ty),
                (Symbol::new("prompt"), i32_ty),
                (Symbol::new("op_idx"), i32_ty),
            ],
        );

        // Continuation type: (resume_fn: ptr, state: ptr, tag: i32, shift_value: ptr)
        let continuation_type = adt::struct_type(
            db,
            "_Continuation",
            vec![
                (Symbol::new("resume_fn"), ptr_ty),
                (Symbol::new("state"), ptr_ty),
                (Symbol::new("tag"), i32_ty),
                (Symbol::new("shift_value"), ptr_ty),
            ],
        );

        // ResumeWrapper type: (state: ptr, resume_value: ptr)
        let resume_wrapper_type = adt::struct_type(
            db,
            "_ResumeWrapper",
            vec![
                (Symbol::new("state"), ptr_ty),
                (Symbol::new("resume_value"), ptr_ty),
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

        // Build struct fields: (resume_fn, state, tag, shift_value)
        let mut fields: Vec<Value<'db>> = Vec::new();
        if let Some(v) = resume_fn {
            fields.push(v);
        }
        if let Some(v) = state {
            fields.push(v);
        }
        fields.push(tag_value);
        if let Some(v) = shift_value {
            fields.push(v);
        }

        let struct_new = adt::struct_new(
            self.db,
            location,
            fields,
            self.continuation_type,
            self.continuation_type,
        );
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        vec![tag_const, struct_new]
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

        // Build struct fields: (tag, value, prompt, op_idx)
        let mut fields = vec![tag_value];
        if let Some(v) = value {
            fields.push(v);
        }
        fields.push(prompt_value);
        fields.push(op_idx_value);

        let struct_new = adt::struct_new(self.db, location, fields, self.step_type, self.step_type);
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        vec![tag_const, prompt_const, op_idx_const, struct_new]
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

        // Build struct fields: (tag, value (continuation as anyref), prompt, op_idx)
        let mut fields = vec![tag_value];
        if let Some(v) = continuation {
            fields.push(v);
        }
        fields.push(prompt_value);
        fields.push(op_idx_value);

        let struct_new = adt::struct_new(self.db, location, fields, self.step_type, self.step_type);
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        vec![tag_const, prompt_const, op_idx_const, struct_new]
    }

    /// Lower `trampoline.step_get` → `adt.struct_get`
    fn lower_step_get(
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
            .unwrap_or_else(|| Symbol::new("value"));

        let result_type = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Ptr::new(self.db).as_type());

        let step_value = operands
            .first()
            .copied()
            .expect("step_get requires operand");

        let struct_get = adt::struct_get(
            self.db,
            location,
            step_value,
            result_type,
            self.step_type,
            Attribute::Symbol(field_name),
        );
        let struct_get = struct_get.as_operation();

        self.ctx.map_results(self.db, &op, &struct_get);
        vec![struct_get]
    }

    /// Lower `trampoline.continuation_get` → `adt.struct_get`
    fn lower_continuation_get(
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
            .unwrap_or_else(|| Symbol::new("value"));

        let result_type = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Ptr::new(self.db).as_type());

        let cont_value = operands
            .first()
            .copied()
            .expect("continuation_get requires operand");

        let struct_get = adt::struct_get(
            self.db,
            location,
            cont_value,
            result_type,
            self.continuation_type,
            Attribute::Symbol(field_name),
        );
        let struct_get = struct_get.as_operation();

        self.ctx.map_results(self.db, &op, &struct_get);
        vec![struct_get]
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

        // operands: (state, resume_value)
        let fields: Vec<Value<'db>> = operands.iter().copied().collect();
        let struct_new = adt::struct_new(
            self.db,
            location,
            fields,
            self.resume_wrapper_type,
            self.resume_wrapper_type,
        );
        let struct_new = struct_new.as_operation();

        self.ctx.map_results(self.db, &op, &struct_new);
        vec![struct_new]
    }

    /// Lower `trampoline.resume_wrapper_get` → `adt.struct_get`
    fn lower_resume_wrapper_get(
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
            .unwrap_or_else(|| Symbol::new("state"));

        let result_type = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| core::Ptr::new(self.db).as_type());

        let wrapper_value = operands
            .first()
            .copied()
            .expect("resume_wrapper_get requires operand");

        let struct_get = adt::struct_get(
            self.db,
            location,
            wrapper_value,
            result_type,
            self.resume_wrapper_type,
            Attribute::Symbol(field_name),
        );
        let struct_get = struct_get.as_operation();

        self.ctx.map_results(self.db, &op, &struct_get);
        vec![struct_get]
    }

    /// Helper: Create an i32 constant operation using arith.const
    fn create_i32_const(&self, location: Location<'db>, value: i32) -> Operation<'db> {
        Const::i32(self.db, location, value).as_operation()
    }
}

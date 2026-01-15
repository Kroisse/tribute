//! Lower trampoline and cont dialect operations to WASM dialect.
//!
//! This pass converts:
//! - `trampoline.set_yield_state` → wasm.global_set (multiple)
//! - `trampoline.reset_yield_state` → wasm.global_set
//! - `trampoline.get_yield_continuation` → wasm.global_get + wasm.ref_cast
//! - `trampoline.get_yield_shift_value` → wasm.global_get + wasm.struct_get
//! - `cont.push_prompt` → yield-checking wasm block
//! - `cont.handler_dispatch` → yield-checking dispatch
//!
//! This pass should run AFTER trampoline_to_adt (which handles struct ops) and
//! BEFORE the final wasm emit.

use tribute_ir::dialect::{adt, tribute_rt};
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::trampoline;
use trunk_ir::dialect::wasm;
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type, Value,
    ValueDef,
};

/// Global variable indices for yield state
const YIELD_STATE_IDX: u32 = 0;
const YIELD_TAG_IDX: u32 = 1;
const YIELD_CONT_IDX: u32 = 2;
const YIELD_OP_IDX: u32 = 3;

/// Lower trampoline global state operations to WASM.
pub fn lower<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    TrampolineToWasmLowerer::new(db).lower_module(module)
}

/// Lowerer for trampoline global ops to WASM.
struct TrampolineToWasmLowerer<'db> {
    db: &'db dyn salsa::Database,
    ctx: RewriteContext<'db>,
    /// Cached Continuation ADT type (must match trampoline_to_adt)
    continuation_type: Type<'db>,
}

impl<'db> TrampolineToWasmLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        let ptr_ty = core::Ptr::new(db).as_type();
        let i32_ty = core::I32::new(db).as_type();
        let any_ty = tribute_rt::Any::new(db).as_type();

        // Continuation type: (resume_fn: ptr, state: any, tag: i32, shift_value: any)
        // This MUST match the definition in trampoline_to_adt.rs
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

        Self {
            db,
            ctx: RewriteContext::new(),
            continuation_type,
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
            if name == trampoline::SET_YIELD_STATE() {
                return self.lower_set_yield_state(op, &remapped_operands);
            }
            if name == trampoline::RESET_YIELD_STATE() {
                return self.lower_reset_yield_state(op);
            }
            if name == trampoline::GET_YIELD_CONTINUATION() {
                return self.lower_get_yield_continuation(op);
            }
            if name == trampoline::GET_YIELD_SHIFT_VALUE() {
                return self.lower_get_yield_shift_value(op);
            }
            if name == trampoline::CHECK_YIELD() {
                return self.lower_check_yield(op);
            }
            // Other trampoline ops (struct ops) should have been lowered by trampoline_to_adt
        }

        // Note: cont.push_prompt and cont.handler_dispatch are handled by
        // cont_to_trampoline pass and should not appear here.

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

    /// Lower `trampoline.set_yield_state` → wasm.global_set (multiple)
    fn lower_set_yield_state(
        &mut self,
        op: Operation<'db>,
        remapped_operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let i32_ty = core::I32::new(self.db).as_type();

        // Get attributes
        let attrs = op.attributes(self.db);
        let tag = attrs
            .get(&Symbol::new("tag"))
            .and_then(|a| match a {
                Attribute::IntBits(v) => Some(*v as i32),
                _ => None,
            })
            .unwrap_or(0);
        let op_idx = attrs
            .get(&Symbol::new("op_idx"))
            .and_then(|a| match a {
                Attribute::IntBits(v) => Some(*v as i32),
                _ => None,
            })
            .unwrap_or(0);

        // Get continuation value
        let cont_val = remapped_operands
            .first()
            .copied()
            .expect("set_yield_state requires continuation operand");

        let mut ops = Vec::new();

        // Set $yield_state = 1 (yielding)
        let const_1 = wasm::i32_const(self.db, location, i32_ty, 1);
        let const_1_val = const_1.as_operation().result(self.db, 0);
        ops.push(const_1.as_operation());
        ops.push(wasm::global_set(self.db, location, const_1_val, YIELD_STATE_IDX).as_operation());

        // Set $yield_tag = tag
        let tag_const = wasm::i32_const(self.db, location, i32_ty, tag);
        let tag_val = tag_const.as_operation().result(self.db, 0);
        ops.push(tag_const.as_operation());
        ops.push(wasm::global_set(self.db, location, tag_val, YIELD_TAG_IDX).as_operation());

        // Set $yield_cont = continuation
        ops.push(wasm::global_set(self.db, location, cont_val, YIELD_CONT_IDX).as_operation());

        // Set $yield_op_idx = op_idx
        let op_idx_const = wasm::i32_const(self.db, location, i32_ty, op_idx);
        let op_idx_val = op_idx_const.as_operation().result(self.db, 0);
        ops.push(op_idx_const.as_operation());
        ops.push(wasm::global_set(self.db, location, op_idx_val, YIELD_OP_IDX).as_operation());

        ops
    }

    /// Lower `trampoline.reset_yield_state` → wasm.global_set ($yield_state = 0)
    fn lower_reset_yield_state(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let i32_ty = core::I32::new(self.db).as_type();

        let mut ops = Vec::new();

        // Set $yield_state = 0 (not yielding)
        let const_0 = wasm::i32_const(self.db, location, i32_ty, 0);
        let const_0_val = const_0.as_operation().result(self.db, 0);
        ops.push(const_0.as_operation());
        ops.push(wasm::global_set(self.db, location, const_0_val, YIELD_STATE_IDX).as_operation());

        ops
    }

    /// Lower `trampoline.get_yield_continuation` → wasm.global_get + wasm.ref_cast
    fn lower_get_yield_continuation(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let anyref_ty = wasm::Anyref::new(self.db).as_type();

        let mut ops = Vec::new();

        // Load continuation from $yield_cont global
        let get_cont = wasm::global_get(self.db, location, anyref_ty, YIELD_CONT_IDX);
        let cont_anyref = get_cont.as_operation().result(self.db, 0);
        ops.push(get_cont.as_operation());

        // Cast anyref to continuation type (uses concrete _Continuation ADT type)
        let cont_cast_op = wasm::ref_cast(
            self.db,
            location,
            cont_anyref,
            self.continuation_type,
            self.continuation_type,
            None,
        )
        .as_operation()
        .modify(self.db)
        .attr("type", Attribute::Type(self.continuation_type))
        .build();
        ops.push(cont_cast_op);

        self.ctx.map_results(self.db, &op, &cont_cast_op);
        ops
    }

    /// Lower `trampoline.get_yield_shift_value` → wasm.global_get + adt.struct_get
    ///
    /// Note: We use adt.struct_get here (not wasm.struct_get) because adt_to_wasm
    /// runs after trampoline_to_wasm and will handle the conversion consistently.
    fn lower_get_yield_shift_value(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let anyref_ty = wasm::Anyref::new(self.db).as_type();

        let mut ops = Vec::new();

        // Load continuation from $yield_cont global
        let get_cont = wasm::global_get(self.db, location, anyref_ty, YIELD_CONT_IDX);
        let cont_anyref = get_cont.as_operation().result(self.db, 0);
        ops.push(get_cont.as_operation());

        // Cast anyref to continuation type (uses concrete _Continuation ADT type)
        let cont_cast = wasm::ref_cast(
            self.db,
            location,
            cont_anyref,
            self.continuation_type,
            self.continuation_type,
            None,
        )
        .as_operation()
        .modify(self.db)
        .attr("type", Attribute::Type(self.continuation_type))
        .build();
        let cont_ref = Value::new(self.db, ValueDef::OpResult(cont_cast), 0);
        ops.push(cont_cast);

        // Extract shift_value from continuation using adt.struct_get
        // field 3 = shift_value in _Continuation struct
        let get_shift_value = adt::struct_get(
            self.db,
            location,
            cont_ref,
            anyref_ty, // shift_value field type is tribute_rt.any
            self.continuation_type,
            Attribute::IntBits(3), // field index
        );
        let get_shift_value = get_shift_value.as_operation();
        ops.push(get_shift_value);

        self.ctx.map_results(self.db, &op, &get_shift_value);
        ops
    }

    /// Lower `trampoline.check_yield` → wasm.global_get (yield_state)
    ///
    /// Returns i32: 0 = not yielding, 1 = yielding
    fn lower_check_yield(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let i32_ty = core::I32::new(self.db).as_type();

        // Get yield state from global
        let get_yield = wasm::global_get(self.db, location, i32_ty, YIELD_STATE_IDX);
        let get_yield = get_yield.as_operation();

        self.ctx.map_results(self.db, &op, &get_yield);
        vec![get_yield]
    }
}

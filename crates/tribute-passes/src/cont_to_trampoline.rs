//! Lower cont dialect operations to trampoline dialect.
//!
//! This pass transforms continuation operations to trampoline operations:
//! - `cont.shift` → `trampoline.build_continuation` + `trampoline.set_yield_state` + `trampoline.step_shift`
//! - `cont.resume` → `trampoline.reset_yield_state` + `trampoline.continuation_get` + call
//! - `cont.push_prompt` → trampoline loop with yield check
//! - `cont.handler_dispatch` → yield check + dispatch logic
//! - `cont.get_continuation` → `trampoline.continuation_get`
//! - `cont.get_shift_value` → `trampoline.continuation_get(field="shift_value")`
//! - `cont.get_done_value` → `trampoline.step_get(field="value")`
//!
//! This pass should run after `tribute_to_cont` and before `trampoline_to_adt`.

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{cont, trampoline};
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{Block, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Value};

/// Lower cont dialect operations to trampoline dialect.
pub fn lower_cont_to_trampoline<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> Module<'db> {
    ContToTrampolineLowerer::new(db).lower_module(module)
}

/// Lowerer for cont operations to trampoline operations.
struct ContToTrampolineLowerer<'db> {
    db: &'db dyn salsa::Database,
    ctx: RewriteContext<'db>,
}

impl<'db> ContToTrampolineLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            ctx: RewriteContext::new(),
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

        // Handle cont dialect operations
        if dialect == cont::DIALECT_NAME() {
            if name == cont::GET_CONTINUATION() {
                return self.lower_get_continuation(op);
            }
            if name == cont::GET_SHIFT_VALUE() {
                return self.lower_get_shift_value(op);
            }
            if name == cont::GET_DONE_VALUE() {
                return self.lower_get_done_value(op, &remapped_operands);
            }
            // TODO: Lower other cont operations
            // - cont.shift
            // - cont.resume
            // - cont.push_prompt
            // - cont.handler_dispatch
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

    /// Lower `cont.get_continuation` → `trampoline.get_yield_continuation`
    ///
    /// The cont.get_continuation operation retrieves the continuation from global
    /// yield state. We convert this to trampoline.get_yield_continuation.
    fn lower_get_continuation(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let result_type = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| trampoline::Continuation::new(self.db).as_type());

        let trampoline_op = trampoline::get_yield_continuation(self.db, location, result_type);
        let trampoline_op = trampoline_op.as_operation();

        self.ctx.map_results(self.db, &op, &trampoline_op);
        vec![trampoline_op]
    }

    /// Lower `cont.get_shift_value` → `trampoline.get_yield_shift_value`
    fn lower_get_shift_value(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let result_type = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| trampoline::Continuation::new(self.db).as_type());

        let trampoline_op = trampoline::get_yield_shift_value(self.db, location, result_type);
        let trampoline_op = trampoline_op.as_operation();

        self.ctx.map_results(self.db, &op, &trampoline_op);
        vec![trampoline_op]
    }

    /// Lower `cont.get_done_value` → `trampoline.step_get(field="value")`
    fn lower_get_done_value(
        &mut self,
        op: Operation<'db>,
        remapped_operands: &IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let result_type = op
            .results(self.db)
            .first()
            .copied()
            .unwrap_or_else(|| trampoline::Step::new(self.db).as_type());

        let step_value = remapped_operands
            .first()
            .copied()
            .expect("get_done_value requires step operand");

        let trampoline_op = trampoline::step_get(
            self.db,
            location,
            step_value,
            result_type,
            Symbol::new("value"),
        );
        let trampoline_op = trampoline_op.as_operation();

        self.ctx.map_results(self.db, &op, &trampoline_op);
        vec![trampoline_op]
    }
}

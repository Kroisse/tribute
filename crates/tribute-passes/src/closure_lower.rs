//! Lower closure operations in indirect calls.
//!
//! This pass transforms `func.call_indirect` operations when the callee
//! is a closure (result of `closure.new`):
//!
//! Before:
//! ```text
//! %closure = closure.new @lifted_func, %env
//! %result = func.call_indirect %closure, %args...
//! ```
//!
//! After:
//! ```text
//! %closure = closure.new @lifted_func, %env
//! %funcref = closure.func %closure
//! %env = closure.env %closure
//! %result = func.call_indirect %funcref, %env, %args...
//! ```

use std::collections::{HashMap, HashSet};

use trunk_ir::dialect::closure;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::func;
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{Block, BlockId, DialectOp, IdVec, Operation, Region, Type, Value, ValueDef};

/// Lower closure operations in the module.
#[salsa::tracked]
pub fn lower_closures<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    ClosureLowerer::new(db).lower_module(module)
}

struct ClosureLowerer<'db> {
    db: &'db dyn salsa::Database,
    ctx: RewriteContext<'db>,
    /// Set of values that are results of closure.new operations
    closure_values: HashSet<Value<'db>>,
    /// Block argument types indexed by BlockId
    block_arg_types: HashMap<BlockId, IdVec<Type<'db>>>,
}

impl<'db> ClosureLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            ctx: RewriteContext::new(),
            closure_values: HashSet::new(),
            block_arg_types: HashMap::new(),
        }
    }

    fn lower_module(&mut self, module: Module<'db>) -> Module<'db> {
        // First pass: collect all closure values
        self.collect_closure_values(module.body(self.db));

        // Second pass: transform call_indirect operations
        let new_body = self.transform_region(module.body(self.db));
        Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        )
    }

    /// Collect all values that are results of closure.new operations.
    fn collect_closure_values(&mut self, region: Region<'db>) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter() {
                // Check if this is a closure.new operation
                if closure::New::from_operation(self.db, *op).is_ok() {
                    let result = op.result(self.db, 0);
                    self.closure_values.insert(result);
                }

                // Recurse into nested regions
                for nested in op.regions(self.db).iter() {
                    self.collect_closure_values(*nested);
                }
            }
        }
    }

    fn transform_region(&mut self, region: Region<'db>) -> Region<'db> {
        let blocks: Vec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| self.transform_block(block))
            .collect();

        Region::new(self.db, region.location(self.db), IdVec::from(blocks))
    }

    fn transform_block(&mut self, block: &Block<'db>) -> Block<'db> {
        // Register block arg types for value_type lookups
        let args = block.args(self.db);
        self.block_arg_types.insert(block.id(self.db), args.clone());

        // Transform operations
        let mut new_ops = IdVec::new();
        for op in block.operations(self.db).iter() {
            let transformed = self.transform_op(op);
            new_ops.extend(transformed);
        }

        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    /// Transform an operation, potentially expanding it into multiple operations.
    fn transform_op(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // First, remap operands using the context
        let remapped_op = self.ctx.remap_operands(self.db, op);

        // Check if this is a func.call_indirect operation
        if let Ok(call_indirect) = func::CallIndirect::from_operation(self.db, remapped_op) {
            let callee = call_indirect.callee(self.db);

            // Check if the callee is a closure value
            if self.is_closure_value(callee) {
                return self.transform_closure_call(call_indirect);
            }
        }

        // For other operations, just transform nested regions and map results
        let new_op = self.transform_op_regions(&remapped_op);

        // Map results
        for (i, _) in op.results(self.db).iter().enumerate() {
            let old_result = op.result(self.db, i);
            let new_result = new_op.result(self.db, i);
            self.ctx.map_value(old_result, new_result);
        }

        vec![new_op]
    }

    /// Check if a value is a closure (result of closure.new or mapped from one).
    fn is_closure_value(&self, value: Value<'db>) -> bool {
        // Check if this value is directly in our set
        if self.closure_values.contains(&value) {
            return true;
        }

        // Check if the original (unmapped) value was a closure
        match value.def(self.db) {
            ValueDef::OpResult(op) => closure::New::from_operation(self.db, op).is_ok(),
            ValueDef::BlockArg(_) => {
                // Block arguments could be closures passed as parameters
                // Check if the type looks like a function type
                if let Some(ty) = self.value_type(value) {
                    return self.could_be_closure_type(ty);
                }
                false
            }
        }
    }

    /// Check if a type could be a closure type (function type).
    fn could_be_closure_type(&self, ty: Type<'db>) -> bool {
        // Function types in Tribute use func.fn or core.func dialect
        let dialect = ty.dialect(self.db);
        let name = ty.name(self.db);
        (dialect == "func" && name == "fn") || (dialect == "core" && name == "func")
    }

    /// Transform a call_indirect on a closure value.
    fn transform_closure_call(&mut self, call: func::CallIndirect<'db>) -> Vec<Operation<'db>> {
        let location = call.as_operation().location(self.db);
        let callee = call.callee(self.db);
        let args = call.args(self.db);
        let result_ty = call
            .as_operation()
            .results(self.db)
            .first()
            .copied()
            .expect("call_indirect should have a result");

        // Get the funcref type and env type from the closure
        let funcref_ty = self.value_type(callee).unwrap_or(result_ty);
        let env_ty = self.get_env_type_from_closure(callee);

        // Generate: %funcref = closure.func %closure
        let funcref_op = closure::func(self.db, location, callee, funcref_ty);
        let funcref = funcref_op.as_operation().result(self.db, 0);

        // Generate: %env = closure.env %closure
        let env_op = closure::env(self.db, location, callee, env_ty);
        let env = env_op.as_operation().result(self.db, 0);

        // Generate: %result = func.call_indirect %funcref, %env, %args...
        let mut new_args: Vec<Value<'db>> = vec![env];
        new_args.extend(args.iter().copied());

        let new_call = func::call_indirect(self.db, location, funcref, new_args, result_ty);
        let new_call_op = new_call.as_operation();

        // Map old result to new result
        let old_result = call.as_operation().result(self.db, 0);
        let new_result = new_call_op.result(self.db, 0);
        self.ctx.map_value(old_result, new_result);

        vec![
            funcref_op.as_operation(),
            env_op.as_operation(),
            new_call_op,
        ]
    }

    /// Get the env type from a closure value.
    fn get_env_type_from_closure(&self, closure_value: Value<'db>) -> Type<'db> {
        // Try to find the closure.new operation and get the env operand's type
        match closure_value.def(self.db) {
            ValueDef::OpResult(op) => {
                if let Ok(closure_new) = closure::New::from_operation(self.db, op) {
                    // Get the env operand's type
                    let env_value = closure_new.env(self.db);
                    if let Some(env_ty) = self.value_type(env_value) {
                        return env_ty;
                    }
                }
            }
            ValueDef::BlockArg(_) => {
                // For block arguments, we don't know the env type
                // Use nil as fallback
            }
        }

        // Fallback: return nil type
        *trunk_ir::dialect::core::Nil::new(self.db)
    }

    /// Get the type of a value.
    fn value_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        match value.def(self.db) {
            ValueDef::OpResult(op) => op.results(self.db).get(value.index(self.db)).copied(),
            ValueDef::BlockArg(block_id) => self
                .block_arg_types
                .get(&block_id)
                .and_then(|args| args.get(value.index(self.db)).copied()),
        }
    }

    fn transform_op_regions(&mut self, op: &Operation<'db>) -> Operation<'db> {
        let regions = op.regions(self.db);
        if regions.is_empty() {
            return *op;
        }

        let new_regions: IdVec<Region<'db>> =
            regions.iter().map(|r| self.transform_region(*r)).collect();

        op.modify(self.db).regions(new_regions).build()
    }
}

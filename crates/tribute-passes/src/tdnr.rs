//! Type-Directed Name Resolution (TDNR) pass.
//!
//! This pass resolves remaining `src.call` operations that couldn't be resolved
//! during initial name resolution because they require type information.
//!
//! ## UFCS Resolution
//!
//! UFCS (Uniform Function Call Syntax) transforms `x.method(y)` into `method(x, y)`,
//! then finds the function `method` in the current namespace where the first
//! parameter type matches the receiver's type.
//!
//! ```text
//! x.len()           // src.call(x, "len") → len(x) where first param matches x's type
//! list.map(f)       // src.call(list, f, "map") → map(list, f)
//! ```
//!
//! ## Pipeline Position
//!
//! TDNR runs after type inference, when concrete types are available:
//! ```text
//! stage_resolve → stage_typecheck → stage_tdnr
//! ```

use std::collections::HashMap;

use crate::resolve::{Binding, ModuleEnv, build_env};
use trunk_ir::dialect::core::{self, Module};
use trunk_ir::dialect::func;
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type,
    Value,
};

// =============================================================================
// TDNR Resolver
// =============================================================================

/// TDNR resolver context.
pub struct TdnrResolver<'db> {
    db: &'db dyn salsa::Database,
    /// Module environment for function lookups.
    env: ModuleEnv<'db>,
    /// Rewrite context for value mapping.
    ctx: RewriteContext<'db>,
    /// Block argument types indexed by BlockId.
    block_arg_types: HashMap<BlockId, IdVec<Type<'db>>>,
}

impl<'db> TdnrResolver<'db> {
    /// Create a new TDNR resolver with the given module environment.
    pub fn new(db: &'db dyn salsa::Database, env: ModuleEnv<'db>) -> Self {
        Self {
            db,
            env,
            ctx: RewriteContext::new(),
            block_arg_types: HashMap::new(),
        }
    }

    /// Resolve a module with TDNR.
    pub fn resolve_module(&mut self, module: &Module<'db>) -> Module<'db> {
        let body = module.body(self.db);
        let new_body = self.resolve_region(&body);

        Module::create(
            self.db,
            module.location(self.db),
            module.name(self.db),
            new_body,
        )
    }

    /// Resolve a region.
    fn resolve_region(&mut self, region: &Region<'db>) -> Region<'db> {
        let new_blocks: IdVec<Block<'db>> = region
            .blocks(self.db)
            .iter()
            .map(|block| self.resolve_block(block))
            .collect();

        Region::new(self.db, region.location(self.db), new_blocks)
    }

    /// Resolve a block.
    fn resolve_block(&mut self, block: &Block<'db>) -> Block<'db> {
        // Register block arg types for get_value_type lookups
        self.block_arg_types
            .insert(block.id(self.db), block.args(self.db).clone());

        let new_ops: IdVec<Operation<'db>> = block
            .operations(self.db)
            .iter()
            .flat_map(|op| self.resolve_operation(op))
            .collect();

        Block::new(
            self.db,
            block.id(self.db),
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    /// Resolve a single operation.
    fn resolve_operation(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // First, remap operands
        let remapped_op = self.ctx.remap_operands(self.db, op);

        // If operands were remapped, map old results to new results
        if remapped_op != *op {
            self.ctx.map_results(self.db, op, &remapped_op);
        }

        let dialect = remapped_op.dialect(self.db);
        let op_name = remapped_op.name(self.db);

        if dialect == "src" && op_name == "call" {
            if let Some(resolved) = self.try_resolve_method_call(&remapped_op) {
                self.ctx.map_results(self.db, &remapped_op, &resolved);
                vec![resolved]
            } else {
                // Still unresolved - keep as is (will be an error later)
                let final_op = self.resolve_op_regions(&remapped_op);
                if final_op != remapped_op {
                    self.ctx.map_results(self.db, &remapped_op, &final_op);
                }
                vec![final_op]
            }
        } else {
            // Recursively process regions
            let final_op = self.resolve_op_regions(&remapped_op);
            if final_op != remapped_op {
                self.ctx.map_results(self.db, &remapped_op, &final_op);
            }
            vec![final_op]
        }
    }

    /// Recursively resolve regions within an operation.
    fn resolve_op_regions(&mut self, op: &Operation<'db>) -> Operation<'db> {
        let regions = op.regions(self.db);
        if regions.is_empty() {
            return *op;
        }

        let new_regions: IdVec<Region<'db>> = regions
            .iter()
            .map(|region| self.resolve_region(region))
            .collect();

        op.modify(self.db).regions(new_regions).build()
    }

    /// Try to resolve a `src.call` as a UFCS method call.
    ///
    /// For `x.method(y)` (represented as `src.call` with receiver `x` and name `method`):
    /// 1. Look up `method` in the module environment
    /// 2. If it's a function and its first parameter matches `x`'s type, resolve it
    /// 3. Transform to `func.call(method, x, y, ...)`
    ///
    /// Also supports qualified names: `x.foo::bar(y)` → `foo::bar(x, y)`
    fn try_resolve_method_call(&mut self, op: &Operation<'db>) -> Option<Operation<'db>> {
        let operands = op.operands(self.db);
        if operands.is_empty() {
            return None; // No receiver
        }

        // Get method name from attributes
        let attrs = op.attributes(self.db);
        let Attribute::QualifiedName(qual_name) = attrs.get(&Symbol::new("name"))? else {
            return None;
        };

        let receiver = operands[0];

        // Get the receiver's type
        let receiver_type = self.get_value_type(receiver)?;

        // Look up the function - handle both simple and qualified names
        let (func_path, func_ty) = if qual_name.is_simple() {
            // Simple name: look up directly
            let binding = self.env.lookup(qual_name.name())?;
            let Binding::Function { path, ty } = binding else {
                return None;
            };
            (path.clone(), *ty)
        } else {
            // Qualified name: look up by full path, fall back to namespace lookup
            let binding = self.env.lookup_path(qual_name).or_else(|| {
                // Fall back to namespace lookup for enum variants, etc.
                let namespace = *qual_name.as_parent().last()?;
                self.env.lookup_qualified(namespace, qual_name.name())
            })?;
            let Binding::Function { ty, .. } = binding else {
                return None;
            };
            // Use the full qualified name for the call
            (qual_name.clone(), *ty)
        };

        // Check if the first parameter type matches the receiver type
        let func_type = core::Func::from_type(self.db, func_ty)?;
        let params = func_type.params(self.db);
        let first_param = params.first()?;

        // Match receiver type with first parameter type
        if !self.types_compatible(receiver_type, *first_param) {
            return None;
        }

        // Create func.call with the resolved function
        let location = op.location(self.db);
        let result_ty = op.results(self.db).first().copied()?;
        let args: Vec<Value<'db>> = operands.iter().copied().collect();

        let new_op = func::call(self.db, location, args, result_ty, func_path);
        let new_operation = new_op.as_operation();

        // Map old result to new result
        let old_result = op.result(self.db, 0);
        let new_result = new_operation.result(self.db, 0);
        self.ctx.map_value(old_result, new_result);

        Some(new_operation)
    }

    /// Check if two types are compatible for UFCS resolution.
    ///
    /// For now, this uses simple equality. A more sophisticated implementation
    /// could consider type variables and subtyping.
    fn types_compatible(&self, actual: Type<'db>, expected: Type<'db>) -> bool {
        // Simple equality check - both types should be concrete after typecheck
        actual == expected
    }

    /// Get the type of a value.
    ///
    /// For operation results, returns the result type.
    /// For block arguments, returns the block argument type.
    fn get_value_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        use trunk_ir::ValueDef;

        match value.def(self.db) {
            ValueDef::OpResult(op) => {
                let results = op.results(self.db);
                let index = value.index(self.db);
                results.get(index).copied()
            }
            ValueDef::BlockArg(block_id) => self
                .block_arg_types
                .get(&block_id)
                .and_then(|args| args.get(value.index(self.db)).copied()),
        }
    }
}

// =============================================================================
// Pipeline Integration
// =============================================================================

/// Run TDNR on a module.
///
/// This builds a `ModuleEnv` from the module and uses it for UFCS resolution.
/// For `x.method(y)`, it looks up `method` in the environment and checks
/// if the first parameter type matches `x`'s type.
pub fn resolve_tdnr<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let env = build_env(db, &module);
    let mut resolver = TdnrResolver::new(db, env);
    resolver.resolve_module(&module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_env_lookup() {
        // Basic smoke test - ModuleEnv is used for UFCS resolution
        let env: ModuleEnv<'_> = ModuleEnv::new();
        assert!(env.lookup(Symbol::new("nonexistent")).is_none());
    }
}

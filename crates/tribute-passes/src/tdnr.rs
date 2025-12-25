//! Type-Directed Name Resolution (TDNR) pass.
//!
//! This pass resolves remaining `src.call` operations that couldn't be resolved
//! during initial name resolution because they require type information.
//!
//! ## Examples
//!
//! ```text
//! x.len()           // src.call(x, "len") → List::len(x) based on x's type
//! list.map(f)       // src.call(list, f, "map") → List::map(list, f)
//! ```
//!
//! ## Pipeline Position
//!
//! TDNR runs after type inference, when concrete types are available:
//! ```text
//! stage_resolve → stage_typecheck → stage_tdnr
//! ```

use std::collections::HashMap;

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::func;
use trunk_ir::rewrite::RewriteContext;
use trunk_ir::{
    Attribute, Block, BlockId, DialectOp, IdVec, Operation, QualifiedName, Region, Symbol, Type,
    Value,
};

// =============================================================================
// Method Registry
// =============================================================================

/// Information about a method that can be called on a type.
#[derive(Clone, Debug)]
pub struct MethodInfo<'db> {
    /// The type this method belongs to (e.g., `List(a)`)
    pub receiver_type: Type<'db>,
    /// The method name (e.g., "len", "map")
    pub name: Symbol,
    /// The full function path to call
    pub func_path: QualifiedName,
    /// The function type
    pub func_type: Type<'db>,
}

/// Registry of methods available for types.
///
/// Maps (type_dialect, type_name) → method_name → MethodInfo
#[derive(Debug, Default)]
pub struct MethodRegistry<'db> {
    /// Methods indexed by (type_dialect, type_name, method_name)
    methods: HashMap<(Symbol, Symbol, Symbol), MethodInfo<'db>>,
}

impl<'db> MethodRegistry<'db> {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a method for a type.
    pub fn register(&mut self, db: &'db dyn salsa::Database, info: MethodInfo<'db>) {
        let key = (
            info.receiver_type.dialect(db),
            info.receiver_type.name(db),
            info.name,
        );
        self.methods.insert(key, info);
    }

    /// Look up a method by receiver type and name.
    pub fn lookup(
        &self,
        db: &'db dyn salsa::Database,
        receiver_type: Type<'db>,
        method_name: Symbol,
    ) -> Option<&MethodInfo<'db>> {
        let dialect = receiver_type.dialect(db);
        let type_name = receiver_type.name(db);
        let key = (dialect, type_name, method_name);
        self.methods.get(&key)
    }
}

// =============================================================================
// Built-in Methods
// =============================================================================

/// Build a registry with built-in methods for core types.
pub fn builtin_methods<'db>(_db: &'db dyn salsa::Database) -> MethodRegistry<'db> {
    // TODO: Register built-in methods for core types
    // For now, return empty registry
    //
    // Example registrations would look like:
    // let mut registry = MethodRegistry::new();
    // registry.register(db, MethodInfo {
    //     receiver_type: list_type,
    //     name: Symbol::new("len"),
    //     func_path: idvec![Symbol::new("List"), Symbol::new("len")],
    //     func_type: ...,
    // });
    // registry

    MethodRegistry::new()
}

// =============================================================================
// TDNR Resolver
// =============================================================================

/// TDNR resolver context.
pub struct TdnrResolver<'db> {
    db: &'db dyn salsa::Database,
    registry: MethodRegistry<'db>,
    /// Rewrite context for value mapping.
    ctx: RewriteContext<'db>,
    /// Block argument types indexed by BlockId
    block_arg_types: HashMap<BlockId, IdVec<Type<'db>>>,
}

impl<'db> TdnrResolver<'db> {
    /// Create a new TDNR resolver.
    pub fn new(db: &'db dyn salsa::Database, registry: MethodRegistry<'db>) -> Self {
        Self {
            db,
            registry,
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
        self.block_arg_types.insert(block.id(self.db), block.args(self.db).clone());

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

    /// Try to resolve a `src.call` as a method call using TDNR.
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

        // Single-segment name means it's a method call needing TDNR
        if !qual_name.is_simple() {
            return None; // Already qualified, shouldn't be here
        }

        let method_name = qual_name.name();
        let receiver = operands[0];

        // Get the receiver's type
        let receiver_type = self.get_value_type(receiver)?;

        // Look up the method
        let method_info = self.registry.lookup(self.db, receiver_type, method_name)?;

        // Create func.call with the resolved function
        let location = op.location(self.db);
        let result_ty = op.results(self.db).first().copied()?;
        let args: Vec<Value<'db>> = operands.iter().copied().collect();

        let new_op = func::call(
            self.db,
            location,
            args,
            result_ty,
            method_info.func_path.clone(),
        );
        let new_operation = new_op.as_operation();

        // Map old result to new result
        let old_result = op.result(self.db, 0);
        let new_result = new_operation.result(self.db, 0);
        self.ctx.map_value(old_result, new_result);

        Some(new_operation)
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
pub fn resolve_tdnr<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let registry = builtin_methods(db);
    let mut resolver = TdnrResolver::new(db, registry);
    resolver.resolve_module(&module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_registry() {
        // Basic smoke test
        let registry: MethodRegistry<'_> = MethodRegistry::new();
        assert!(registry.methods.is_empty());
    }
}

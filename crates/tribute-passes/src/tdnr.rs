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
use trunk_ir::{Attribute, Block, DialectOp, IdVec, Operation, Region, Symbol, Type, Value};

// =============================================================================
// Method Registry
// =============================================================================

/// Information about a method that can be called on a type.
#[derive(Clone, Debug)]
pub struct MethodInfo<'db> {
    /// The type this method belongs to (e.g., `List(a)`)
    pub receiver_type: Type<'db>,
    /// The method name (e.g., "len", "map")
    pub name: Symbol<'db>,
    /// The full function path to call
    pub func_path: IdVec<Symbol<'db>>,
    /// The function type
    pub func_type: Type<'db>,
}

/// Registry of methods available for types.
///
/// Maps (type_dialect, type_name) → method_name → MethodInfo
#[derive(Debug, Default)]
pub struct MethodRegistry<'db> {
    /// Methods indexed by (type_dialect, type_name, method_name)
    methods: HashMap<(String, String, String), MethodInfo<'db>>,
}

impl<'db> MethodRegistry<'db> {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a method for a type.
    pub fn register(&mut self, db: &'db dyn salsa::Database, info: MethodInfo<'db>) {
        let key = (
            info.receiver_type.dialect(db).text(db).to_string(),
            info.receiver_type.name(db).text(db).to_string(),
            info.name.text(db).to_string(),
        );
        self.methods.insert(key, info);
    }

    /// Look up a method by receiver type and name.
    pub fn lookup(
        &self,
        db: &'db dyn salsa::Database,
        receiver_type: Type<'db>,
        method_name: &str,
    ) -> Option<&MethodInfo<'db>> {
        let dialect = receiver_type.dialect(db).text(db);
        let type_name = receiver_type.name(db).text(db);
        let key = (
            dialect.to_string(),
            type_name.to_string(),
            method_name.to_string(),
        );
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
    //     name: Symbol::new(db, "len"),
    //     func_path: idvec![Symbol::new(db, "List"), Symbol::new(db, "len")],
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
    /// Maps old values to their replacements.
    value_map: HashMap<Value<'db>, Value<'db>>,
}

impl<'db> TdnrResolver<'db> {
    /// Create a new TDNR resolver.
    pub fn new(db: &'db dyn salsa::Database, registry: MethodRegistry<'db>) -> Self {
        Self {
            db,
            registry,
            value_map: HashMap::new(),
        }
    }

    /// Look up a mapped value, or return the original.
    fn lookup_value(&self, old: Value<'db>) -> Value<'db> {
        self.value_map.get(&old).copied().unwrap_or(old)
    }

    /// Map a value from old to new.
    fn map_value(&mut self, old: Value<'db>, new: Value<'db>) {
        self.value_map.insert(old, new);
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
        let new_ops: IdVec<Operation<'db>> = block
            .operations(self.db)
            .iter()
            .flat_map(|op| self.resolve_operation(op))
            .collect();

        Block::new(
            self.db,
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    /// Resolve a single operation.
    fn resolve_operation(&mut self, op: &Operation<'db>) -> Vec<Operation<'db>> {
        // First, remap operands
        let remapped_op = self.remap_operands(op);

        let dialect = remapped_op.dialect(self.db).text(self.db);
        let op_name = remapped_op.name(self.db).text(self.db);

        match (dialect, op_name) {
            ("src", "call") => {
                if let Some(resolved) = self.try_resolve_method_call(&remapped_op) {
                    vec![resolved]
                } else {
                    // Still unresolved - keep as is (will be an error later)
                    vec![self.resolve_op_regions(&remapped_op)]
                }
            }
            _ => {
                // Recursively process regions
                vec![self.resolve_op_regions(&remapped_op)]
            }
        }
    }

    /// Remap operands using the value map.
    fn remap_operands(&self, op: &Operation<'db>) -> Operation<'db> {
        let operands = op.operands(self.db);
        let mut new_operands: IdVec<Value<'db>> = IdVec::new();
        let mut changed = false;

        for &operand in operands.iter() {
            let mapped = self.lookup_value(operand);
            new_operands.push(mapped);
            if mapped != operand {
                changed = true;
            }
        }

        if !changed {
            return *op;
        }

        op.modify(self.db).operands(new_operands).build()
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
        let name_key = Symbol::new(self.db, "name");
        let Attribute::SymbolRef(name_segments) = attrs.get(&name_key)? else {
            return None;
        };

        if name_segments.is_empty() {
            return None;
        }

        // Single-segment name means it's a method call needing TDNR
        if name_segments.len() != 1 {
            return None; // Already qualified, shouldn't be here
        }

        let method_name = name_segments[0].text(self.db);
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
        self.map_value(old_result, new_result);

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
            ValueDef::BlockArg(block) => {
                let args = block.args(self.db);
                let index = value.index(self.db);
                args.get(index).copied()
            }
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

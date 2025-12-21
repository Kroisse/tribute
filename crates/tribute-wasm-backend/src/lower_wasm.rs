//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This pass transforms mid-level IR operations (func, arith, scf, adt) to
//! wasm dialect operations. Phase 1 handles basic arithmetic and function calls.

use std::collections::HashMap;

use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{arith, func};
use trunk_ir::{Attribute, Block, IdVec, Operation, Region, Symbol, Type, Value};

/// Entry point for lowering mid-level IR to wasm dialect.
#[salsa::tracked]
pub fn lower_to_wasm<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    let mut lowerer = WasmLowerer::new(db);
    let lowered = lowerer.lower_module(module);

    // Debug: Verify all operations are now in wasm dialect
    if cfg!(debug_assertions) {
        check_all_wasm_dialect(db, &lowered);
    }

    lowered
}

/// Debug helper to check if all operations in function bodies are in wasm dialect
fn check_all_wasm_dialect<'db>(db: &'db dyn salsa::Database, module: &Module<'db>) {
    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            let dialect = op.dialect(db);
            let name = op.name(db);

            // Check func.func operations and their bodies
            if dialect == Symbol::new("func") && name == Symbol::new("func") {
                check_function_body(db, op);
            }
        }
    }
}

/// Check if all operations in a function body are in wasm dialect
fn check_function_body<'db>(db: &'db dyn salsa::Database, func_op: &Operation<'db>) {
    if let Some(body_region) = func_op.regions(db).first() {
        for block in body_region.blocks(db).iter() {
            for op in block.operations(db).iter() {
                let dialect = op.dialect(db);
                if dialect != Symbol::new("wasm") {
                    eprintln!(
                        "ERROR: Found non-wasm operation in function body: {}.{}",
                        dialect,
                        op.name(db)
                    );
                }
            }
        }
    }
}

/// Lowers mid-level IR to wasm dialect operations.
struct WasmLowerer<'db> {
    db: &'db dyn salsa::Database,
    value_map: HashMap<Value<'db>, Value<'db>>,
}

impl<'db> WasmLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            value_map: HashMap::new(),
        }
    }

    fn lower_module(&mut self, module: Module<'db>) -> Module<'db> {
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
        let mut new_ops = IdVec::new();
        for op in block.operations(self.db).iter().copied() {
            let rewritten = self.lower_op(op);
            new_ops.extend(rewritten);
        }
        Block::new(
            self.db,
            block.location(self.db),
            block.args(self.db).clone(),
            new_ops,
        )
    }

    fn lower_op(&mut self, op: Operation<'db>) -> Vec<Operation<'db>> {
        let remapped_operands = self.remap_operands(op);
        let dialect = op.dialect(self.db);
        let name = op.name(self.db);

        // Transform operations based on dialect
        if dialect == arith::DIALECT_NAME() {
            return self.lower_arith_op(op, name, remapped_operands);
        }

        if dialect == func::DIALECT_NAME() {
            return self.lower_func_op(op, name, remapped_operands);
        }

        // Operations we're not handling should not appear in function bodies
        // They should have been processed by earlier pipeline stages
        if cfg!(debug_assertions) {
            let dialect_str = dialect.to_string();
            if !["wasm", "core", "type", "ty"].contains(&dialect_str.as_str()) {
                eprintln!(
                    "WARNING: Unhandled operation in lowering: {}.{} (this may cause emit errors)",
                    dialect, name
                );
            }
        }

        // Keep other operations as-is (recursively lower regions)
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
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn lower_arith_op(
        &mut self,
        op: Operation<'db>,
        name: Symbol,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let result_type = op.results(self.db).first().copied();

        let wasm_name = if name == arith::CONST() {
            self.arith_const_to_wasm_name(result_type)
        } else {
            self.arith_op_to_wasm_name(name, result_type)
        };

        let mut new_op_builder =
            Operation::of_name(self.db, location, wasm_name).operands(operands.clone());

        // For const operations, copy the value attribute
        if name == arith::CONST()
            && let Some(attr @ (Attribute::IntBits(_) | Attribute::FloatBits(_))) =
                op.attributes(self.db).get(&Symbol::new("value"))
        {
            new_op_builder = new_op_builder.attr("value", attr.clone());
        }

        // Set result types for arithmetic operations
        let new_op = new_op_builder.results(op.results(self.db).clone()).build();

        self.map_results(op, new_op);
        vec![new_op]
    }

    fn arith_const_to_wasm_name(&self, ty: Option<Type<'db>>) -> &'static str {
        match ty {
            Some(t) => {
                let name = t.name(self.db);
                if name == Symbol::new("i32") {
                    "wasm.i32_const"
                } else if name == Symbol::new("i64") {
                    "wasm.i64_const"
                } else if name == Symbol::new("f32") {
                    "wasm.f32_const"
                } else if name == Symbol::new("f64") {
                    "wasm.f64_const"
                } else {
                    // Default to i32 for unknown types
                    "wasm.i32_const"
                }
            }
            None => "wasm.i32_const", // Default to i32
        }
    }

    fn arith_op_to_wasm_name(&self, name: Symbol, ty: Option<Type<'db>>) -> &'static str {
        let type_suffix = self.type_suffix(ty);

        if name == arith::ADD() {
            match type_suffix {
                "i32" => "wasm.i32_add",
                "i64" => "wasm.i64_add",
                "f32" => "wasm.f32_add",
                "f64" => "wasm.f64_add",
                _ => "wasm.i32_add",
            }
        } else if name == arith::SUB() {
            match type_suffix {
                "i32" => "wasm.i32_sub",
                "i64" => "wasm.i64_sub",
                "f32" => "wasm.f32_sub",
                "f64" => "wasm.f64_sub",
                _ => "wasm.i32_sub",
            }
        } else if name == arith::MUL() {
            match type_suffix {
                "i32" => "wasm.i32_mul",
                "i64" => "wasm.i64_mul",
                "f32" => "wasm.f32_mul",
                "f64" => "wasm.f64_mul",
                _ => "wasm.i32_mul",
            }
        } else if name == arith::DIV() {
            match type_suffix {
                "i32" => "wasm.i32_div_s",
                "i64" => "wasm.i64_div_s",
                "f32" => "wasm.f32_div",
                "f64" => "wasm.f64_div",
                _ => "wasm.i32_div_s",
            }
        } else if name == arith::CMP_EQ() {
            match type_suffix {
                "i32" => "wasm.i32_eq",
                "i64" => "wasm.i64_eq",
                "f32" => "wasm.f32_eq",
                "f64" => "wasm.f64_eq",
                _ => "wasm.i32_eq",
            }
        } else if name == arith::CMP_NE() {
            match type_suffix {
                "i32" => "wasm.i32_ne",
                "i64" => "wasm.i64_ne",
                "f32" => "wasm.f32_ne",
                "f64" => "wasm.f64_ne",
                _ => "wasm.i32_ne",
            }
        } else if name == arith::CMP_LT() {
            match type_suffix {
                "i32" => "wasm.i32_lt_s",
                "i64" => "wasm.i64_lt_s",
                "f32" => "wasm.f32_lt",
                "f64" => "wasm.f64_lt",
                _ => "wasm.i32_lt_s",
            }
        } else if name == arith::CMP_LE() {
            match type_suffix {
                "i32" => "wasm.i32_le_s",
                "i64" => "wasm.i64_le_s",
                "f32" => "wasm.f32_le",
                "f64" => "wasm.f64_le",
                _ => "wasm.i32_le_s",
            }
        } else if name == arith::CMP_GT() {
            match type_suffix {
                "i32" => "wasm.i32_gt_s",
                "i64" => "wasm.i64_gt_s",
                "f32" => "wasm.f32_gt",
                "f64" => "wasm.f64_gt",
                _ => "wasm.i32_gt_s",
            }
        } else if name == arith::CMP_GE() {
            match type_suffix {
                "i32" => "wasm.i32_ge_s",
                "i64" => "wasm.i64_ge_s",
                "f32" => "wasm.f32_ge",
                "f64" => "wasm.f64_ge",
                _ => "wasm.i32_ge_s",
            }
        } else {
            // Unsupported arith operation; keep as-is
            "wasm.i32_add" // placeholder; should not reach here
        }
    }

    fn type_suffix(&self, ty: Option<Type<'db>>) -> &'static str {
        match ty {
            Some(t) => {
                let name = t.name(self.db);
                if name == Symbol::new("i32") {
                    "i32"
                } else if name == Symbol::new("i64") {
                    "i64"
                } else if name == Symbol::new("f32") {
                    "f32"
                } else if name == Symbol::new("f64") {
                    "f64"
                } else {
                    "i32" // Default to i32
                }
            }
            None => "i32", // Default to i32
        }
    }

    fn lower_func_op(
        &mut self,
        op: Operation<'db>,
        name: Symbol,
        operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);

        // func.call -> wasm.call
        if name == func::CALL() {
            let mut new_op_builder = Operation::of_name(self.db, location, "wasm.call")
                .operands(operands)
                .results(op.results(self.db).clone());

            // Copy callee attribute
            if let Some(callee) = op.attributes(self.db).get(&Symbol::new("callee")) {
                new_op_builder = new_op_builder.attr("callee", callee.clone());
            }

            let new_op = new_op_builder.build();
            self.map_results(op, new_op);
            return vec![new_op];
        }

        // func.return -> wasm.return
        if name == func::RETURN() {
            let new_op = Operation::of_name(self.db, location, "wasm.return")
                .operands(operands)
                .build();
            return vec![new_op];
        }

        // func.func - keep as-is but lower regions
        // (emit_wasm handles func.func directly)
        let new_regions = op
            .regions(self.db)
            .iter()
            .copied()
            .map(|region| self.lower_region(region))
            .collect::<Vec<_>>();

        let new_op = op
            .modify(self.db)
            .operands(operands)
            .regions(IdVec::from(new_regions))
            .build();
        self.map_results(op, new_op);
        vec![new_op]
    }

    fn remap_operands(&self, op: Operation<'db>) -> IdVec<Value<'db>> {
        let mut operands = IdVec::new();
        for &operand in op.operands(self.db).iter() {
            let mapped = self.value_map.get(&operand).copied().unwrap_or(operand);
            operands.push(mapped);
        }
        operands
    }

    fn map_results(&mut self, old_op: Operation<'db>, new_op: Operation<'db>) {
        let old_results = old_op.results(self.db);
        let new_results = new_op.results(self.db);
        let count = old_results.len().min(new_results.len());
        for i in 0..count {
            let old_val = old_op.result(self.db, i);
            let new_val = new_op.result(self.db, i);
            self.value_map.insert(old_val, new_val);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lower_arith_const() {
        // Basic test placeholder
    }

    #[test]
    fn test_lower_arith_add() {
        // Basic test placeholder
    }
}

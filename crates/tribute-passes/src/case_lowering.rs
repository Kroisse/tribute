//! Lower `case.case` pattern matching to `scf.if` chains.
//!
//! This pass currently supports only a minimal pattern subset:
//! - wildcard (`_`)
//! - bind (`x`)
//! - literal int/bool
//!
//! Unsupported patterns or non-exhaustive cases emit diagnostics.

use std::collections::HashMap;

use salsa::Accumulator;
use trunk_ir::dialect::core::Module;
use trunk_ir::dialect::{adt, arith, case, core, pat, ty};
use trunk_ir::{
    Attribute, Block, DialectOp, DialectType, IdVec, Location, Operation, Region, SymbolVec, Type
};
use trunk_ir::{Symbol, Value, ValueDef};

use crate::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};

#[derive(Debug, Clone)]
enum ArmPattern<'db> {
    Wildcard,
    Bind,
    Literal(Attribute<'db>),
    Variant(Symbol),
}

#[derive(Debug, Clone)]
struct ArmInfo<'db> {
    pattern: ArmPattern<'db>,
    pattern_region: Option<Region<'db>>,
    body: Region<'db>,
}

pub fn lower_case_to_scf<'db>(db: &'db dyn salsa::Database, module: Module<'db>) -> Module<'db> {
    CaseLowerer::new(db).lower_module(module)
}

struct CaseLowerer<'db> {
    db: &'db dyn salsa::Database,
    value_map: HashMap<Value<'db>, Value<'db>>,
    variant_tags: HashMap<Symbol, u32>,
    variant_owner: HashMap<Symbol, Symbol>,
    enum_variants: HashMap<Symbol, SymbolVec>,
    /// Current arm's pattern bindings: binding name -> bound value (scrutinee)
    current_arm_bindings: HashMap<Symbol, Value<'db>>,
}

impl<'db> CaseLowerer<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            value_map: HashMap::new(),
            variant_tags: HashMap::new(),
            variant_owner: HashMap::new(),
            enum_variants: HashMap::new(),
            current_arm_bindings: HashMap::new(),
        }
    }

    fn lower_module(&mut self, module: Module<'db>) -> Module<'db> {
        self.collect_variant_tags(module);
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

        if op.dialect(self.db) == case::DIALECT_NAME() && op.name(self.db) == case::CASE() {
            return self.lower_case(op, remapped_operands);
        }

        if op.dialect(self.db) == case::DIALECT_NAME() && op.name(self.db) == case::YIELD() {
            let new_op = Operation::of_name(self.db, op.location(self.db), "scf.yield")
                .operands(remapped_operands)
                .build();
            return vec![new_op];
        }

        // Handle case.bind: replace with the bound value from pattern matching
        if op.dialect(self.db) == case::DIALECT_NAME() && op.name(self.db) == case::BIND() {
            if let Some(Attribute::Symbol(name)) = op.attributes(self.db).get(&Symbol::new("name"))
            {
                if let Some(&bound_value) = self.current_arm_bindings.get(name) {
                    // Map case.bind result to the bound value (scrutinee or destructured value)
                    let bind_result = op.result(self.db, 0);
                    self.value_map.insert(bind_result, bound_value);
                    // Erase the case.bind operation - value is remapped
                    return vec![];
                }
            }
            // If binding not found, keep the operation (shouldn't happen in well-formed IR)
        }

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

    fn lower_case(
        &mut self,
        op: Operation<'db>,
        remapped_operands: IdVec<Value<'db>>,
    ) -> Vec<Operation<'db>> {
        let location = op.location(self.db);
        let Some(result_type) = op.results(self.db).first().copied() else {
            self.emit_error(location, "case expression must produce a result");
            return vec![op];
        };

        let Some(scrutinee) = remapped_operands.first().copied() else {
            return vec![op];
        };

        let Some(body_region) = op.regions(self.db).first().copied() else {
            return vec![op];
        };

        let (arms, supported) = self.collect_arms(body_region);
        if !supported {
            return vec![self.rebuild_case(op, remapped_operands)];
        }

        if arms.is_empty() {
            self.emit_error(location, "case expression has no arms");
            return vec![self.rebuild_case(op, remapped_operands)];
        }

        if !self.is_exhaustive(&arms) {
            self.emit_error(location, "non-exhaustive case expression");
            return vec![self.rebuild_case(op, remapped_operands)];
        }

        if arms.len() == 1 && !self.is_irrefutable(&arms[0].pattern) {
            self.emit_error(location, "single-arm case requires irrefutable pattern");
            return vec![self.rebuild_case(op, remapped_operands)];
        }

        let (ops, _final_value) = self.build_arm_chain(location, scrutinee, result_type, &arms);
        if let Some(last_op) = ops.last() {
            self.map_results(op, *last_op);
        }
        ops
    }

    fn rebuild_case(
        &mut self,
        op: Operation<'db>,
        remapped_operands: IdVec<Value<'db>>,
    ) -> Operation<'db> {
        let new_op = op.modify(self.db).operands(remapped_operands).build();
        self.map_results(op, new_op);
        new_op
    }

    fn collect_arms(&mut self, body_region: Region<'db>) -> (Vec<ArmInfo<'db>>, bool) {
        let mut supported = true;
        let mut arms = Vec::new();
        for block in body_region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter().copied() {
                if op.dialect(self.db) != case::DIALECT_NAME() || op.name(self.db) != case::ARM() {
                    continue;
                }
                let arm_location = op.location(self.db);
                let pattern_region = op.regions(self.db).first().copied();
                let body_region = op.regions(self.db).get(1).copied();
                let (pattern, ok) = pattern_region
                    .and_then(|region| self.parse_pattern(region))
                    .unwrap_or_else(|| (ArmPattern::Wildcard, false));
                supported &= ok;
                let body =
                    body_region.unwrap_or_else(|| Region::new(self.db, arm_location, IdVec::new()));
                arms.push(ArmInfo { pattern, pattern_region, body });
            }
        }
        if !supported {
            self.emit_error(
                body_region.location(self.db),
                "unsupported pattern in case expression",
            );
        }
        (arms, supported)
    }

    fn parse_pattern(&self, region: Region<'db>) -> Option<(ArmPattern<'db>, bool)> {
        let block = region.blocks(self.db).first().copied()?;
        let mut ops = block.operations(self.db).iter().copied();
        let op = ops.next()?;
        if ops.next().is_some() {
            return Some((ArmPattern::Wildcard, false));
        }

        if op.dialect(self.db) != pat::DIALECT_NAME() {
            return Some((ArmPattern::Wildcard, false));
        }

        match op.name(self.db) {
            name if name == pat::WILDCARD() => Some((ArmPattern::Wildcard, true)),
            name if name == pat::BIND() => Some((ArmPattern::Bind, true)),
            name if name == pat::LITERAL() => {
                let attr = op.attributes(self.db).get(&Symbol::new("value"))?.clone();
                match attr {
                    Attribute::IntBits(_) | Attribute::Bool(_) => {
                        Some((ArmPattern::Literal(attr), true))
                    }
                    _ => Some((ArmPattern::Wildcard, false)),
                }
            }
            name if name == pat::VARIANT() => {
                let attr = op.attributes(self.db).get(&Symbol::new("variant"))?;
                let variant_path = match attr {
                    Attribute::QualifiedName(path) => path,
                    _ => return Some((ArmPattern::Wildcard, false)),
                };
                let name = variant_path.name();
                if !self.variant_tags.contains_key(&name) {
                    return Some((ArmPattern::Wildcard, false));
                }
                let fields_region = op.regions(self.db).first().copied();
                if fields_region
                    .is_some_and(|fields| !self.pattern_region_is_bind_or_wildcard(fields))
                {
                    return Some((ArmPattern::Wildcard, false));
                }
                Some((ArmPattern::Variant(name), true))
            }
            _ => Some((ArmPattern::Wildcard, false)),
        }
    }

    fn pattern_region_is_bind_or_wildcard(&self, region: Region<'db>) -> bool {
        let Some(block) = region.blocks(self.db).first() else {
            return true;
        };
        for op in block.operations(self.db).iter().copied() {
            if op.dialect(self.db) != pat::DIALECT_NAME() {
                return false;
            }
            let ok =
                matches!(op.name(self.db), name if name == pat::BIND() || name == pat::WILDCARD());
            if !ok {
                return false;
            }
        }
        true
    }

    /// Extract binding names from a pattern region.
    /// For simple `pat.bind("x")` patterns, returns the binding name.
    /// For variant patterns with bindings, returns all nested binding names.
    fn extract_bindings_from_pattern(&self, region: Region<'db>) -> SymbolVec {
        let mut bindings = SymbolVec::new();
        self.collect_bindings_recursive(region, &mut bindings);
        bindings
    }

    fn collect_bindings_recursive(&self, region: Region<'db>, bindings: &mut SymbolVec) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter().copied() {
                if op.dialect(self.db) == pat::DIALECT_NAME() && op.name(self.db) == pat::BIND() {
                    if let Some(Attribute::Symbol(name)) =
                        op.attributes(self.db).get(&Symbol::new("name"))
                    {
                        bindings.push(*name);
                    }
                }
                // Recurse into nested regions (for variant fields, etc.)
                for nested_region in op.regions(self.db).iter().copied() {
                    self.collect_bindings_recursive(nested_region, bindings);
                }
            }
        }
    }

    fn is_exhaustive(&self, arms: &[ArmInfo<'db>]) -> bool {
        if matches!(
            arms.last().map(|arm| &arm.pattern),
            Some(ArmPattern::Wildcard) | Some(ArmPattern::Bind)
        ) {
            return true;
        }

        let mut owner: Option<Symbol> = None;
        let mut seen_variants = SymbolVec::new();

        for arm in arms {
            let ArmPattern::Variant(variant) = arm.pattern else {
                return false;
            };
            let Some(variant_owner) = self.variant_owner.get(&variant).copied() else {
                return false;
            };
            if let Some(existing) = owner {
                if existing != variant_owner {
                    return false;
                }
            } else {
                owner = Some(variant_owner);
            }
            if !seen_variants.contains(&variant) {
                seen_variants.push(variant);
            }
        }

        let Some(owner) = owner else {
            return false;
        };
        let Some(all_variants) = self.enum_variants.get(&owner) else {
            return false;
        };
        seen_variants.len() == all_variants.len()
    }

    fn is_irrefutable(&self, pattern: &ArmPattern<'db>) -> bool {
        match pattern {
            ArmPattern::Wildcard | ArmPattern::Bind => true,
            ArmPattern::Literal(_) => false,
            ArmPattern::Variant(name) => {
                let Some(owner) = self.variant_owner.get(name).copied() else {
                    return false;
                };
                let Some(all_variants) = self.enum_variants.get(&owner) else {
                    return false;
                };
                all_variants.len() == 1
            }
        }
    }

    /// Lower an arm body with pattern bindings set up.
    /// For simple bind patterns, the scrutinee IS the bound value.
    /// For variant patterns with nested bindings, we use the scrutinee for now
    /// (full destructuring support would require more complex handling).
    fn lower_arm_body(
        &mut self,
        scrutinee: Value<'db>,
        arm: &ArmInfo<'db>,
    ) -> Region<'db> {
        // Extract bindings from pattern and map them to scrutinee
        // For simple `x` or `Some(x)` patterns, x gets the scrutinee value
        // (More sophisticated handling would destruct variants first)
        if let Some(pattern_region) = arm.pattern_region {
            let bindings = self.extract_bindings_from_pattern(pattern_region);
            for name in bindings {
                self.current_arm_bindings.insert(name, scrutinee);
            }
        }

        // Lower the body with bindings in scope
        let result = self.lower_region(arm.body);

        // Clear bindings after processing this arm
        self.current_arm_bindings.clear();

        result
    }

    fn build_arm_chain(
        &mut self,
        location: Location<'db>,
        scrutinee: Value<'db>,
        result_type: Type<'db>,
        arms: &[ArmInfo<'db>],
    ) -> (Vec<Operation<'db>>, Value<'db>) {
        if arms.len() == 1 {
            let (cond_ops, cond) = self.build_condition_ops(location, scrutinee, &arms[0].pattern);
            let body_region = self.lower_arm_body(scrutinee, &arms[0]);
            let then_region = body_region;
            // Single-arm cases are irrefutable; reuse the body for the else branch.
            let else_region = body_region;
            let if_op = Operation::of_name(self.db, location, "scf.if")
                .operands(IdVec::from(vec![cond]))
                .results(IdVec::from(vec![result_type]))
                .regions(IdVec::from(vec![then_region, else_region]))
                .build();
            let mut ops = Vec::new();
            ops.extend(cond_ops);
            ops.push(if_op);
            return (ops, if_op.result(self.db, 0));
        }

        let (cond_ops, cond) = self.build_condition_ops(location, scrutinee, &arms[0].pattern);
        let then_region = self.lower_arm_body(scrutinee, &arms[0]);
        let else_region = self.build_else_region(location, scrutinee, result_type, &arms[1..]);

        let if_op = Operation::of_name(self.db, location, "scf.if")
            .operands(IdVec::from(vec![cond]))
            .results(IdVec::from(vec![result_type]))
            .regions(IdVec::from(vec![then_region, else_region]))
            .build();

        let mut ops = Vec::new();
        ops.extend(cond_ops);
        ops.push(if_op);
        (ops, if_op.result(self.db, 0))
    }

    fn build_else_region(
        &mut self,
        location: Location<'db>,
        scrutinee: Value<'db>,
        result_type: Type<'db>,
        arms: &[ArmInfo<'db>],
    ) -> Region<'db> {
        if arms.len() == 1 {
            return self.lower_arm_body(scrutinee, &arms[0]);
        }

        let (mut ops, value) = self.build_arm_chain(location, scrutinee, result_type, arms);
        let yield_op = Operation::of_name(self.db, location, "scf.yield")
            .operands(IdVec::from(vec![value]))
            .build();
        ops.push(yield_op);

        let block = Block::new(self.db, location, IdVec::new(), IdVec::from(ops));
        Region::new(self.db, location, IdVec::from(vec![block]))
    }

    fn build_condition_ops(
        &self,
        location: Location<'db>,
        scrutinee: Value<'db>,
        pattern: &ArmPattern<'db>,
    ) -> (Vec<Operation<'db>>, Value<'db>) {
        let bool_ty = core::I1::new(self.db).as_type();
        match pattern {
            ArmPattern::Wildcard | ArmPattern::Bind => {
                let op = arith::r#const(self.db, location, bool_ty, true.into()).as_operation();
                let value = op.result(self.db, 0);
                (vec![op], value)
            }
            ArmPattern::Literal(attr) => {
                let Some(scrutinee_ty) = self.value_type(scrutinee) else {
                    let op = arith::r#const(self.db, location, bool_ty, true.into()).as_operation();
                    let value = op.result(self.db, 0);
                    return (vec![op], value);
                };
                let lit_op =
                    arith::r#const(self.db, location, scrutinee_ty, attr.clone()).as_operation();
                let lit_value = lit_op.result(self.db, 0);
                let cmp_op =
                    arith::cmp_eq(self.db, location, scrutinee, lit_value, bool_ty).as_operation();
                let cmp_value = cmp_op.result(self.db, 0);
                (vec![lit_op, cmp_op], cmp_value)
            }
            ArmPattern::Variant(name) => {
                let Some(tag) = self.variant_tags.get(name).copied() else {
                    let op = arith::r#const(self.db, location, bool_ty, true.into()).as_operation();
                    let value = op.result(self.db, 0);
                    return (vec![op], value);
                };
                let tag_ty = core::I32::new(self.db).as_type();
                let tag_op = adt::variant_tag(self.db, location, scrutinee, tag_ty).as_operation();
                let tag_value = tag_op.result(self.db, 0);
                let tag_const =
                    arith::r#const(self.db, location, tag_ty, Attribute::IntBits(tag as u64))
                        .as_operation();
                let tag_const_value = tag_const.result(self.db, 0);
                let cmp_op = arith::cmp_eq(self.db, location, tag_value, tag_const_value, bool_ty)
                    .as_operation();
                let cmp_value = cmp_op.result(self.db, 0);
                (vec![tag_op, tag_const, cmp_op], cmp_value)
            }
        }
    }

    fn value_type(&self, value: Value<'db>) -> Option<Type<'db>> {
        match value.def(self.db) {
            ValueDef::OpResult(op) => op.results(self.db).get(value.index(self.db)).copied(),
            ValueDef::BlockArg(block) => block.args(self.db).get(value.index(self.db)).copied(),
        }
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

    fn emit_error(&self, location: Location<'db>, message: &str) {
        Diagnostic {
            message: message.to_string(),
            span: location.span,
            severity: DiagnosticSeverity::Error,
            phase: CompilationPhase::Optimization,
        }
        .accumulate(self.db);
    }

    fn collect_variant_tags(&mut self, module: Module<'db>) {
        let body = module.body(self.db);
        self.collect_variant_tags_in_region(body);
    }

    fn collect_variant_tags_in_region(&mut self, region: Region<'db>) {
        for block in region.blocks(self.db).iter() {
            for op in block.operations(self.db).iter().copied() {
                if op.dialect(self.db) == ty::DIALECT_NAME() && op.name(self.db) == ty::ENUM() {
                    self.collect_variant_tags_from_enum(op);
                }
                for nested in op.regions(self.db).iter().copied() {
                    self.collect_variant_tags_in_region(nested);
                }
            }
        }
    }

    fn collect_variant_tags_from_enum(&mut self, op: Operation<'db>) {
        let Some(Attribute::List(variants)) = op.attributes(self.db).get(&Symbol::new("variants"))
        else {
            return;
        };
        let type_name = op
            .attributes(self.db)
            .get(&Symbol::new("sym_name"))
            .and_then(|attr| match attr {
                Attribute::Symbol(sym) => Some(*sym),
                _ => None,
            })
            .unwrap_or_else(|| Symbol::new("_"));

        let mut variant_names = SymbolVec::new();
        for (idx, variant) in variants.iter().enumerate() {
            let Attribute::List(parts) = variant else {
                continue;
            };
            let Some(Attribute::Symbol(name)) = parts.first() else {
                continue;
            };
            let tag = idx as u32;
            if self
                .variant_tags
                .insert(*name, tag)
                .is_some_and(|existing| existing != tag)
            {
                self.emit_error(
                    op.location(self.db),
                    "ambiguous variant name in enum declarations",
                );
            }
            if self
                .variant_owner
                .insert(*name, type_name)
                .is_some_and(|existing| existing != type_name)
            {
                self.emit_error(
                    op.location(self.db),
                    "ambiguous variant name in enum declarations",
                );
            }
            variant_names.push(*name);
        }
        self.enum_variants.insert(type_name, variant_names);
    }
}

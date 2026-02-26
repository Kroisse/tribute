//! TrunkIR text format parser.
//!
//! Parses the textual IR format produced by [`super::printer`] back into
//! Salsa-tracked TrunkIR structures. Uses winnow for parsing.
//!
//! # Two-stage parsing
//!
//! 1. **Raw parse**: winnow combinators parse text into `Raw*` structures
//!    (strings, not Salsa-tracked).
//! 2. **IR build**: `Raw*` structures are converted to Salsa `Operation`,
//!    `Block`, `Region`, etc., resolving SSA value names and block labels.

pub(crate) mod raw;

use std::collections::BTreeMap;

use winnow::prelude::*;

pub use raw::ParseError;
use raw::*;

use crate::{
    Attribute, Block, BlockArg, BlockId, DialectOp, IdVec, Location, Operation, PathId, Region,
    Span, Symbol, Type, Value, ValueDef,
};

// ============================================================================
// IR Builder (Raw -> Salsa-tracked IR)
// ============================================================================

struct IrBuilder<'db> {
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    /// Maps value name (without %) -> Value
    value_map: std::collections::HashMap<String, Value<'db>>,
    /// Maps block label (without ^) -> BlockId
    block_id_map: std::collections::HashMap<String, BlockId>,
}

impl<'db> IrBuilder<'db> {
    fn new(db: &'db dyn salsa::Database) -> Self {
        let path = PathId::new(db, "textual-ir".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        Self {
            db,
            location,
            value_map: std::collections::HashMap::new(),
            block_id_map: std::collections::HashMap::new(),
        }
    }

    fn build_type(&self, raw: &RawType<'_>) -> Type<'db> {
        let dialect = Symbol::from_dynamic(raw.dialect);
        let name = Symbol::from_dynamic(raw.name);
        let params: IdVec<Type<'db>> = raw.params.iter().map(|p| self.build_type(p)).collect();
        let attrs: BTreeMap<Symbol, Attribute<'db>> = raw
            .attrs
            .iter()
            .map(|(k, v)| (Symbol::from_dynamic(k), self.build_attribute(v)))
            .collect();
        Type::new(self.db, dialect, name, params, attrs)
    }

    fn build_attribute(&self, raw: &RawAttribute<'_>) -> Attribute<'db> {
        match raw {
            RawAttribute::Bool(b) => Attribute::Bool(*b),
            RawAttribute::Int(n) => Attribute::IntBits(*n),
            RawAttribute::Float(f) => Attribute::FloatBits(f.to_bits()),
            RawAttribute::String(s) => Attribute::String(s.clone()),
            RawAttribute::Symbol(s) => Attribute::Symbol(Symbol::from_dynamic(s.as_str())),
            RawAttribute::Type(t) => Attribute::Type(self.build_type(t)),
            RawAttribute::List(items) => {
                Attribute::List(items.iter().map(|a| self.build_attribute(a)).collect())
            }
            RawAttribute::Unit => Attribute::Unit,
            RawAttribute::Location(path, start, end) => {
                let path_id = PathId::new(self.db, path.clone());
                Attribute::Location(Location::new(path_id, Span::new(*start, *end)))
            }
            RawAttribute::Bytes(bytes) => Attribute::Bytes(bytes.clone()),
        }
    }

    fn resolve_value(&self, name: &str) -> Option<Value<'db>> {
        self.value_map.get(name).copied()
    }

    /// Pre-assign BlockIds for all blocks in a region.
    ///
    /// Returns an error if the same label appears more than once *within the
    /// same region*. Labels in different (possibly nested) regions are
    /// independent and may legitimately share the same name (e.g. `^bb0`).
    fn pre_assign_blocks(&mut self, raw_region: &RawRegion<'_>) -> Result<(), ParseError> {
        let mut seen_in_region = std::collections::HashSet::new();
        for block in &raw_region.blocks {
            let label = block.label.to_string();
            if !seen_in_region.insert(label.clone()) {
                return Err(ParseError {
                    message: format!("duplicate block label '^{}'", label),
                    offset: 0,
                });
            }
            let block_id = BlockId::fresh();
            self.block_id_map.insert(label, block_id);
        }
        Ok(())
    }

    /// Save the current `block_id_map` entries for the given labels, so they
    /// can be restored after an inner region is built.  Returns the entries
    /// that existed before (if any) for each label.
    fn save_block_labels(&self, raw_region: &RawRegion<'_>) -> Vec<(String, Option<BlockId>)> {
        raw_region
            .blocks
            .iter()
            .map(|b| {
                let label = b.label.to_string();
                let prev = self.block_id_map.get(&label).copied();
                (label, prev)
            })
            .collect()
    }

    /// Restore previously saved `block_id_map` entries, undoing the insertions
    /// made by `pre_assign_blocks` for an inner region.
    fn restore_block_labels(&mut self, saved: Vec<(String, Option<BlockId>)>) {
        for (label, prev) in saved {
            match prev {
                Some(id) => {
                    self.block_id_map.insert(label, id);
                }
                None => {
                    self.block_id_map.remove(&label);
                }
            }
        }
    }

    /// Snapshot the entire `value_map` so it can be restored after leaving a
    /// region.  Inner SSA names (block args, op results) are region-local and
    /// must not leak into the enclosing scope.
    fn save_value_map(&self) -> std::collections::HashMap<String, Value<'db>> {
        self.value_map.clone()
    }

    /// Restore the `value_map` to a previously saved snapshot, discarding any
    /// names introduced inside the region.
    fn restore_value_map(&mut self, saved: std::collections::HashMap<String, Value<'db>>) {
        self.value_map = saved;
    }

    fn build_region(&mut self, raw: &RawRegion<'_>) -> Result<Region<'db>, ParseError> {
        let saved_blocks = self.save_block_labels(raw);
        let saved_values = self.save_value_map();
        self.pre_assign_blocks(raw)?;
        let blocks: IdVec<Block<'db>> = raw
            .blocks
            .iter()
            .map(|b| self.build_block(b))
            .collect::<Result<_, _>>()?;
        let region = Region::new(self.db, self.location, blocks);
        self.restore_value_map(saved_values);
        self.restore_block_labels(saved_blocks);
        Ok(region)
    }

    fn build_block_with_extra_args(
        &mut self,
        raw: &RawBlock<'_>,
        extra_args: &[(&str, RawType<'_>)],
    ) -> Result<Block<'db>, ParseError> {
        let block_id = self
            .block_id_map
            .get(raw.label)
            .copied()
            .unwrap_or_else(BlockId::fresh);

        // Build block args: merge extra_args (from func params) with block's own args.
        // If the block already has args that match the extra_args, use the block's args.
        // When both are present, validate that they agree in count and types.
        let all_args = if raw.args.is_empty() && !extra_args.is_empty() {
            extra_args.to_vec()
        } else if !raw.args.is_empty() && !extra_args.is_empty() {
            // Both present — validate arity and types match
            if raw.args.len() != extra_args.len() {
                return Err(ParseError {
                    message: format!(
                        "entry block has {} args but function signature has {} params",
                        raw.args.len(),
                        extra_args.len()
                    ),
                    offset: 0,
                });
            }
            for (i, ((_, block_ty), (_, param_ty))) in
                raw.args.iter().zip(extra_args.iter()).enumerate()
            {
                let bt = self.build_type(block_ty);
                let pt = self.build_type(param_ty);
                if bt != pt {
                    return Err(ParseError {
                        message: format!(
                            "entry block arg {} type mismatch: block has {}.{} but function param has {}.{}",
                            i,
                            bt.dialect(self.db),
                            bt.name(self.db),
                            pt.dialect(self.db),
                            pt.name(self.db),
                        ),
                        offset: 0,
                    });
                }
            }
            raw.args.clone()
        } else {
            raw.args.clone()
        };

        let mut block_args = IdVec::new();
        let mut seen_names = std::collections::HashSet::new();
        for (i, (name, raw_ty)) in all_args.iter().enumerate() {
            // Detect duplicate argument names
            if !seen_names.insert(name.to_string()) {
                return Err(ParseError {
                    message: format!("duplicate block argument name '{}' at index {}", name, i),
                    offset: 0,
                });
            }

            let ty = self.build_type(raw_ty);
            let is_default_name = name
                .strip_prefix("arg")
                .and_then(|rest| rest.parse::<usize>().ok())
                .is_some_and(|n| n == i);
            let arg = if !is_default_name {
                let mut attrs = BTreeMap::new();
                attrs.insert(
                    Symbol::new("bind_name"),
                    Attribute::Symbol(Symbol::from_dynamic(name)),
                );
                BlockArg::new(self.db, ty, attrs)
            } else {
                BlockArg::of_type(self.db, ty)
            };
            block_args.push(arg);

            let value = Value::new(self.db, ValueDef::BlockArg(block_id), i);
            self.value_map.insert(name.to_string(), value);
        }

        // Build operations
        let ops: IdVec<Operation<'db>> = raw
            .ops
            .iter()
            .map(|op| self.build_operation(op))
            .collect::<Result<_, _>>()?;

        Ok(Block::new(
            self.db,
            block_id,
            self.location,
            block_args,
            ops,
        ))
    }

    fn build_block(&mut self, raw: &RawBlock<'_>) -> Result<Block<'db>, ParseError> {
        self.build_block_with_extra_args(raw, &[])
    }

    fn build_operation(&mut self, raw: &RawOperation<'_>) -> Result<Operation<'db>, ParseError> {
        let dialect = Symbol::from_dynamic(raw.dialect);
        let op_name = Symbol::from_dynamic(raw.op_name);

        // Resolve operands
        let operands: IdVec<Value<'db>> = raw
            .operands
            .iter()
            .map(|name| {
                self.resolve_value(name).ok_or_else(|| ParseError {
                    message: format!(
                        "undefined value '%{}' in operation '{}.{}'",
                        name, raw.dialect, raw.op_name
                    ),
                    offset: 0,
                })
            })
            .collect::<Result<_, _>>()?;

        // Build result types
        let results: IdVec<Type<'db>> = raw
            .result_types
            .iter()
            .map(|t| self.build_type(t))
            .collect();

        // Build attributes from explicit attr dict
        let mut attributes: BTreeMap<Symbol, Attribute<'db>> = raw
            .attributes
            .iter()
            .map(|(k, v)| (Symbol::from_dynamic(k), self.build_attribute(v)))
            .collect();

        // Add sym_name attribute if present
        if let Some(ref name) = raw.sym_name {
            attributes.insert(
                Symbol::new("sym_name"),
                Attribute::Symbol(Symbol::from_dynamic(name.as_str())),
            );
        }

        // Handle func-style signature: build function type and inject into attributes.
        // Include effect_type so effect-only signatures are not silently dropped.
        let has_func_signature =
            raw.return_type.is_some() || !raw.func_params.is_empty() || raw.effect_type.is_some();
        if has_func_signature {
            let return_ty = raw
                .return_type
                .as_ref()
                .map(|t| self.build_type(t))
                .unwrap_or_else(|| {
                    Type::new(
                        self.db,
                        Symbol::new("core"),
                        Symbol::new("nil"),
                        IdVec::new(),
                        BTreeMap::new(),
                    )
                });

            // Build function type: core.func(return_type, param_types...)
            let mut func_params: IdVec<Type<'db>> = IdVec::new();
            func_params.push(return_ty);
            for (_, raw_ty) in &raw.func_params {
                func_params.push(self.build_type(raw_ty));
            }

            let mut func_attrs = BTreeMap::new();
            if let Some(effect_raw) = &raw.effect_type {
                let effect_ty = self.build_type(effect_raw);
                func_attrs.insert(Symbol::new("effect"), Attribute::Type(effect_ty));
            }

            let func_ty = Type::new(
                self.db,
                Symbol::new("core"),
                Symbol::new("func"),
                func_params,
                func_attrs,
            );
            attributes.insert(Symbol::new("type"), Attribute::Type(func_ty));
        }

        // Build regions, injecting func params as entry block args if needed
        let regions: IdVec<Region<'db>> = raw
            .regions
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i == 0 && !raw.func_params.is_empty() {
                    // First region: inject func_params as entry block args.
                    // Save/restore both block_id_map and value_map so inner
                    // names don't clobber outer mappings.
                    let saved_blocks = self.save_block_labels(r);
                    let saved_values = self.save_value_map();
                    self.pre_assign_blocks(r)?;
                    let blocks: IdVec<Block<'db>> = r
                        .blocks
                        .iter()
                        .enumerate()
                        .map(|(j, b)| {
                            if j == 0 {
                                self.build_block_with_extra_args(b, &raw.func_params)
                            } else {
                                self.build_block(b)
                            }
                        })
                        .collect::<Result<_, _>>()?;
                    let region = Region::new(self.db, self.location, blocks);
                    self.restore_value_map(saved_values);
                    self.restore_block_labels(saved_blocks);
                    Ok(region)
                } else {
                    self.build_region(r)
                }
            })
            .collect::<Result<_, _>>()?;

        // Validate result name count matches declared result types.
        // This catches both mismatches (2 names vs 1 type) and names-without-
        // types (1 name vs 0 types), preventing semantically invalid Values.
        if !raw.results.is_empty() && raw.results.len() != results.len() {
            return Err(ParseError {
                message: format!(
                    "operation '{}.{}' declares {} result names but {} result types",
                    raw.dialect,
                    raw.op_name,
                    raw.results.len(),
                    results.len()
                ),
                offset: 0,
            });
        }

        let op = Operation::of(self.db, self.location, dialect, op_name)
            .operands(operands)
            .results(results)
            .attrs(attributes)
            .regions(regions)
            .build();

        // Register result values, rejecting duplicates within the region
        for (i, name) in raw.results.iter().enumerate() {
            if self.value_map.contains_key(*name) {
                return Err(ParseError {
                    message: format!(
                        "duplicate SSA name '{}' in operation '{}.{}' result index {}",
                        name, raw.dialect, raw.op_name, i
                    ),
                    offset: 0,
                });
            }
            let value = op.result(self.db, i);
            self.value_map.insert(name.to_string(), value);
        }

        Ok(op)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Parse a TrunkIR module from its textual representation.
pub fn parse_module<'db>(
    db: &'db dyn salsa::Database,
    input: &str,
) -> Result<Operation<'db>, ParseError> {
    let mut remaining = input;
    ws.parse_next(&mut remaining).map_err(|e| ParseError {
        message: format!("lexer error: {}", e),
        offset: input.len() - remaining.len(),
    })?;

    // Parse the top-level operation
    let raw_op = raw_operation
        .parse_next(&mut remaining)
        .map_err(|e| ParseError {
            message: format!("parse error: {}", e),
            offset: input.len() - remaining.len(),
        })?;

    // Reject trailing input
    ws.parse_next(&mut remaining).map_err(|e| ParseError {
        message: format!("lexer error: {}", e),
        offset: input.len() - remaining.len(),
    })?;
    if !remaining.is_empty() {
        return Err(ParseError {
            message: "trailing input after top-level operation".to_string(),
            offset: input.len() - remaining.len(),
        });
    }

    // Build IR
    let mut builder = IrBuilder::new(db);
    builder.build_operation(&raw_op)
}

/// Parse textual IR into a [`core::Module`], panicking on failure.
///
/// This is a convenience wrapper around [`parse_module`] for tests that
/// construct IR from textual format.  Parse errors and non-module top-level
/// operations cause a panic with a descriptive message.
///
/// # Salsa context
///
/// This function creates Salsa tracked structs internally, so it **must** be
/// called from within a `#[salsa::tracked]` function context.
pub fn parse_test_module<'db>(
    db: &'db dyn salsa::Database,
    input: &str,
) -> crate::dialect::core::Module<'db> {
    let op = parse_module(db, input).unwrap_or_else(|e| {
        panic!(
            "Failed to parse test IR at offset {}:\n  {}\n\nInput:\n{}",
            e.offset, e.message, input
        );
    });
    crate::dialect::core::Module::from_operation(db, op).unwrap_or_else(|_| {
        panic!(
            "Parsed operation is not a core.module.\n\nInput:\n{}",
            input
        );
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DialectOp, dialect::core, printer::print_op};
    use salsa::Database;
    use salsa_test_macros::salsa_test;

    // Salsa input to hold text for tracked function parsing
    #[salsa::input]
    struct TextInput {
        #[returns(ref)]
        text: String,
    }

    // Tracked wrapper: Salsa requires tracked struct creation inside tracked functions
    #[salsa::tracked]
    fn do_parse(db: &dyn salsa::Database, input: TextInput) -> Operation<'_> {
        parse_module(db, input.text(db)).expect("should parse")
    }

    /// Tracked wrapper that preserves the error message, for testing parse
    /// failures that occur after tracked structs have already been created.
    #[salsa::tracked]
    fn try_parse_err(db: &dyn salsa::Database, input: TextInput) -> Option<String> {
        parse_module(db, input.text(db)).err().map(|e| e.message)
    }

    #[salsa_test]
    fn test_parse_simple_module(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            r#"core.module @test {
  func.func @main() -> core.i32 {
    ^bb0:
      %0 = arith.const {value = 40} : core.i32
      %1 = arith.const {value = 2} : core.i32
      %2 = arith.add %0, %1 : core.i32
      func.return %2
  }
}"#
            .to_string(),
        );

        let op = do_parse(db, input);
        let module = core::Module::from_operation(db, op).expect("should be a module");
        assert_eq!(module.name(db).to_string(), "test");

        // Verify round-trip: print -> re-parse -> print -> compare
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "round-trip failed");
    }

    #[salsa_test]
    fn test_roundtrip_with_params(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            r#"core.module @test {
  func.func @add(%x: core.i32, %y: core.i32) -> core.i32 {
    ^bb0(%x: core.i32, %y: core.i32):
      %0 = arith.add %x, %y : core.i32
      func.return %0
  }
}"#
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "round-trip failed");
    }

    #[test]
    fn test_parse_trailing_input() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.nil {\n",
                    "    func.return\n",
                    "  }\n",
                    "} garbage",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on trailing input");
        assert!(
            err.message.contains("trailing input"),
            "unexpected error: {}",
            err.message
        );
    }

    #[salsa_test]
    fn test_parse_trailing_whitespace_ok(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil {\n",
                "    func.return\n",
                "  }\n",
                "}  \n  \n",
            )
            .to_string(),
        );
        // Should succeed — do_parse calls .expect() internally
        let _ = do_parse(db, input);
    }

    #[salsa_test]
    fn test_string_escape_roundtrip(db: &salsa::DatabaseImpl) {
        // Build an operation with a string attribute containing special chars
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil {\n",
                "    %0 = test.op {msg = \"line1\\nline2\\ttab\\r\\0end\"} : core.nil\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );
        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "string escape round-trip failed");
    }

    #[salsa_test]
    fn test_symbol_escape_roundtrip(db: &salsa::DatabaseImpl) {
        // Module with a function whose name contains special characters
        // that need escaping in quoted symbols
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @\"has\\\\slash\"() -> core.nil {\n",
                "    func.return\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );
        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "symbol escape round-trip failed");
    }

    #[test]
    fn test_parse_undefined_operand() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.i32 {\n",
                    "    ^bb0:\n",
                    "      %0 = arith.add %x, %y : core.i32\n",
                    "      func.return %0\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on undefined operand");
        assert!(
            err.message.contains("%x"),
            "error should mention the undefined value: {}",
            err.message
        );
        assert!(
            err.message.contains("arith.add"),
            "error should mention the operation: {}",
            err.message
        );
    }

    // ---- Fix 4: type attributes round-trip ----

    #[salsa_test]
    fn test_type_attrs_roundtrip(db: &salsa::DatabaseImpl) {
        // A function type with an effect attribute: core.func(...) {effect = ...}
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.i32 effects core.nil {\n",
                "    %0 = arith.const {value = 1} : core.i32\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );
        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "type attrs round-trip failed");
    }

    // ---- Fix 2: duplicate block labels ----

    #[test]
    fn test_parse_duplicate_block_label() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.i32 {\n",
                    "    ^bb0:\n",
                    "      %0 = arith.const {value = 1} : core.i32\n",
                    "      func.return %0\n",
                    "    ^bb0:\n",
                    "      %1 = arith.const {value = 2} : core.i32\n",
                    "      func.return %1\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on duplicate block label");
        assert!(
            err.message.contains("duplicate block label"),
            "unexpected error: {}",
            err.message
        );
    }

    // ---- Fix 1: entry-block arg validation ----

    #[test]
    fn test_parse_entry_block_arity_mismatch() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                // func declares 2 params but entry block has 1 arg
                concat!(
                    "core.module @test {\n",
                    "  func.func @f(%x: core.i32, %y: core.i32) -> core.i32 {\n",
                    "    ^bb0(%x: core.i32):\n",
                    "      func.return %x\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on arity mismatch");
        assert!(
            err.message.contains("args") && err.message.contains("params"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_entry_block_type_mismatch() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                // func param is i32, block arg is f64
                concat!(
                    "core.module @test {\n",
                    "  func.func @f(%x: core.i32) -> core.i32 {\n",
                    "    ^bb0(%x: core.f64):\n",
                    "      func.return %x\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on type mismatch");
        assert!(
            err.message.contains("type mismatch"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn test_parse_duplicate_block_arg_name() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                concat!(
                    "core.module @test {\n",
                    "  func.func @f(%x: core.i32, %x: core.i32) -> core.i32 {\n",
                    "    func.return %x\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on duplicate arg name");
        assert!(
            err.message.contains("duplicate block argument name"),
            "unexpected error: {}",
            err.message
        );
    }

    // ---- Fix 3: result count arity ----

    #[test]
    fn test_parse_result_count_mismatch() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                // 2 result names but only 1 type
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.nil {\n",
                    "    %0, %1 = test.op : core.i32\n",
                    "    func.return %0\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail on result count mismatch");
        assert!(
            err.message.contains("result names") && err.message.contains("result types"),
            "unexpected error: {}",
            err.message
        );
    }

    // ---- Fix: result names without types ----

    #[test]
    fn test_parse_result_names_without_types() {
        let db = salsa::DatabaseImpl::default();
        let result: Result<(), ParseError> = db.attach(|db| {
            parse_module(
                db,
                // Result name but no `: type` annotation
                concat!(
                    "core.module @test {\n",
                    "  func.func @f() -> core.nil {\n",
                    "    %0 = test.op\n",
                    "    func.return\n",
                    "  }\n",
                    "}",
                ),
            )
            .map(|_| ())
        });
        let err = result.expect_err("should fail: result name without type");
        assert!(
            err.message.contains("result names") && err.message.contains("result types"),
            "unexpected error: {}",
            err.message
        );
    }

    // ---- Fix: nested regions with shared block labels ----

    #[salsa_test]
    fn test_nested_regions_shared_block_labels(db: &salsa::DatabaseImpl) {
        // Two sibling functions inside a module, each with ^bb0.
        // After parsing the first func, ^bb0 must NOT be clobbered for the
        // second func (or vice versa).  Round-trip verifies correctness.
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.i32 {\n",
                "    ^bb0:\n",
                "      %0 = arith.const {value = 1} : core.i32\n",
                "      func.return %0\n",
                "  }\n",
                "  func.func @g() -> core.i32 {\n",
                "    ^bb0:\n",
                "      %0 = arith.const {value = 2} : core.i32\n",
                "      func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "nested-region round-trip failed");
    }

    // ---- Fix: FloatBits round-trip (exponent forms) ----

    #[salsa_test]
    fn test_float_bits_roundtrip(db: &salsa::DatabaseImpl) {
        // Use a value that Rust's Display formats with exponent notation.
        // The printer must ensure a decimal point so the parser accepts it.
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil {\n",
                "    %0 = test.op {big = 1.0e300, small = 1.0e-300} : core.nil\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        // The printed text must be re-parseable
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "float-bits round-trip failed");
    }

    // ---- Fix: type with attrs + empty parens round-trip ----

    #[salsa_test]
    fn test_type_empty_parens_attrs_roundtrip(db: &salsa::DatabaseImpl) {
        // Type with attrs but no params: printer emits `custom.t() {k = v}`.
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil {\n",
                "    %0 = test.op {ty = custom.t() {size = 4}} : core.nil\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(
            printed, printed2,
            "type empty-parens attrs round-trip failed"
        );
    }

    // ---- Fix: is_default_name index comparison ----

    #[salsa_test]
    fn test_arg_default_name_preserves_mismatched_index(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f(%arg5: core.i32) -> core.i32 {\n",
                "    func.return %arg5\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        assert!(
            printed.contains("%arg5"),
            "explicit %arg5 at param index 0 should be preserved, got:\n{}",
            printed,
        );

        // Full round-trip
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "arg5 round-trip failed");
    }

    #[salsa_test]
    fn test_arg_default_name_strips_matching_index(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f(%arg0: core.i32) -> core.i32 {\n",
                "    func.return %arg0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        assert!(
            printed.contains("%arg0"),
            "default %arg0 at index 0 should still appear, got:\n{}",
            printed,
        );

        // Round-trip
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "arg0 round-trip failed");
    }

    // ---- Fix: value_map scoping in nested regions ----

    #[salsa_test]
    fn test_value_map_scoped_across_sibling_functions(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f(%arg0: core.i32) -> core.i32 {\n",
                "    %0 = arith.const {value = 10} : core.i32\n",
                "    %1 = arith.add %arg0, %0 : core.i32\n",
                "    func.return %1\n",
                "  }\n",
                "  func.func @g(%arg0: core.i32) -> core.i32 {\n",
                "    %0 = arith.const {value = 20} : core.i32\n",
                "    %1 = arith.add %arg0, %0 : core.i32\n",
                "    func.return %1\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "value_map scoping round-trip failed");
    }

    // ---- Fix: duplicate SSA name in op results ----

    #[salsa_test]
    fn test_parse_duplicate_result_name(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            // Two operations produce the same result name %0
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.i32 {\n",
                "    %0 = arith.const {value = 1} : core.i32\n",
                "    %0 = arith.const {value = 2} : core.i32\n",
                "    func.return %0\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );
        let err_msg = try_parse_err(db, input);
        let err_msg = err_msg.as_ref().expect("should fail on duplicate SSA name");
        assert!(
            err_msg.contains("duplicate SSA name"),
            "unexpected error: {}",
            err_msg
        );
    }

    // ---- Fix: effect-only func signature ----

    #[salsa_test]
    fn test_effect_only_signature_roundtrip(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            concat!(
                "core.module @test {\n",
                "  func.func @f() -> core.nil effects core.nil {\n",
                "    func.return\n",
                "  }\n",
                "}",
            )
            .to_string(),
        );

        let op = do_parse(db, input);
        let printed = print_op(db, op);
        assert!(
            printed.contains("effects"),
            "effect should be preserved in printed output:\n{}",
            printed,
        );

        let input2 = TextInput::new(db, printed.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);
        assert_eq!(printed, printed2, "effect-only signature round-trip failed");
    }

    // ========================================================================
    // Round-trip verification tests
    // ========================================================================

    /// Helper: parse textual IR, print it, re-parse, re-print, and assert
    /// the two printed forms are identical.  Returns the canonical printed
    /// form for snapshot testing.
    fn assert_roundtrip(db: &salsa::DatabaseImpl, input: &str) -> String {
        let input1 = TextInput::new(db, input.to_string());
        let op1 = do_parse(db, input1);
        let printed1 = print_op(db, op1);

        let input2 = TextInput::new(db, printed1.clone());
        let op2 = do_parse(db, input2);
        let printed2 = print_op(db, op2);

        assert_eq!(printed1, printed2, "round-trip mismatch");
        printed1
    }

    #[salsa_test]
    fn test_roundtrip_scf_if(db: &salsa::DatabaseImpl) {
        let printed = assert_roundtrip(
            db,
            r#"core.module @test {
  func.func @f(%cond: core.i1) -> core.i32 {
    %0 = scf.if %cond : core.i32 {
      %1 = arith.const {value = 1} : core.i32
      scf.yield %1
    } {
      %2 = arith.const {value = 0} : core.i32
      scf.yield %2
    }
    func.return %0
  }
}"#,
        );
        insta::assert_snapshot!(printed);
    }

    #[salsa_test]
    fn test_roundtrip_scf_loop(db: &salsa::DatabaseImpl) {
        let printed = assert_roundtrip(
            db,
            r#"core.module @test {
  func.func @f() -> core.i32 {
    %init = arith.const {value = 0} : core.i32
    %0 = scf.loop %init : core.i32 {
      ^bb0(%acc: core.i32):
        %limit = arith.const {value = 10} : core.i32
        %done = arith.cmp_ge %acc, %limit : core.i1
        %1 = scf.if %done : core.i32 {
          scf.break %acc
        } {
          %one = arith.const {value = 1} : core.i32
          %next = arith.add %acc, %one : core.i32
          scf.continue %next
        }
        scf.yield %1
    }
    func.return %0
  }
}"#,
        );
        insta::assert_snapshot!(printed);
    }

    #[salsa_test]
    fn test_roundtrip_scf_switch(db: &salsa::DatabaseImpl) {
        let printed = assert_roundtrip(
            db,
            r#"core.module @test {
  func.func @f(%disc: core.i32) -> core.i32 {
    scf.switch %disc {
      scf.case {value = 0} {
        %0 = arith.const {value = 100} : core.i32
        scf.yield %0
      }
      scf.case {value = 1} {
        %1 = arith.const {value = 200} : core.i32
        scf.yield %1
      }
      scf.default {
        %2 = arith.const {value = 0} : core.i32
        scf.yield %2
      }
    }
    func.return %disc
  }
}"#,
        );
        insta::assert_snapshot!(printed);
    }

    #[salsa_test]
    fn test_roundtrip_func_call(db: &salsa::DatabaseImpl) {
        let printed = assert_roundtrip(
            db,
            r#"core.module @test {
  func.func @add(%x: core.i32, %y: core.i32) -> core.i32 {
    %0 = arith.add %x, %y : core.i32
    func.return %0
  }
  func.func @main() -> core.i32 {
    %a = arith.const {value = 3} : core.i32
    %b = arith.const {value = 4} : core.i32
    %r = func.call %a, %b {callee = @add} : core.i32
    func.return %r
  }
}"#,
        );
        insta::assert_snapshot!(printed);
    }

    #[salsa_test]
    fn test_roundtrip_adt_struct(db: &salsa::DatabaseImpl) {
        let printed = assert_roundtrip(
            db,
            r#"core.module @test {
  func.func @f() -> core.i32 {
    %x = arith.const {value = 10} : core.i32
    %y = arith.const {value = 20} : core.i32
    %s = adt.struct_new %x, %y {type = adt.struct() {name = @Point, fields = [@x, @y]}} : adt.struct() {name = @Point, fields = [@x, @y]}
    %0 = adt.struct_get %s {type = adt.struct() {name = @Point, fields = [@x, @y]}, field = 0} : core.i32
    func.return %0
  }
}"#,
        );
        insta::assert_snapshot!(printed);
    }

    #[salsa_test]
    fn test_roundtrip_arith_ops(db: &salsa::DatabaseImpl) {
        let printed = assert_roundtrip(
            db,
            r#"core.module @test {
  func.func @f(%a: core.i32, %b: core.i32) -> core.i1 {
    %sum = arith.add %a, %b : core.i32
    %diff = arith.sub %a, %b : core.i32
    %prod = arith.mul %a, %b : core.i32
    %quot = arith.div %a, %b : core.i32
    %neg = arith.neg %a : core.i32
    %eq = arith.cmp_eq %a, %b : core.i1
    %lt = arith.cmp_lt %a, %b : core.i1
    func.return %eq
  }
}"#,
        );
        insta::assert_snapshot!(printed);
    }

    #[salsa_test]
    fn test_roundtrip_func_effects(db: &salsa::DatabaseImpl) {
        let printed = assert_roundtrip(
            db,
            r#"core.module @test {
  func.func @f() -> core.i32 effects core.nil {
    %0 = arith.const {value = 42} : core.i32
    func.return %0
  }
}"#,
        );
        insta::assert_snapshot!(printed);
    }

    /// Verify that `parse_test_module` returns a proper `Module` wrapper.
    #[salsa::tracked]
    fn do_parse_test_module<'db>(
        db: &'db dyn salsa::Database,
        input: TextInput,
    ) -> core::Module<'db> {
        parse_test_module(db, input.text(db))
    }

    #[salsa_test]
    fn test_parse_test_module_helper(db: &salsa::DatabaseImpl) {
        let input = TextInput::new(
            db,
            r#"core.module @mymod {
  func.func @f() -> core.nil {
    func.return
  }
}"#
            .to_string(),
        );
        let module = do_parse_test_module(db, input);
        assert_eq!(module.name(db).to_string(), "mymod");
    }

    // ========================================================================
    // Property-based tests (proptest)
    // ========================================================================

    mod proptest_fuzz {
        use super::{TextInput, parse_module};
        use crate::parser::raw::{raw_attr_value, raw_type};
        use crate::printer::print_op;
        use proptest::prelude::*;
        use salsa::Database;
        use winnow::prelude::*;

        /// Tracked wrapper: attempts parse and returns whether it succeeded.
        /// Panics from Salsa tracked struct creation are avoided because
        /// this function IS a tracked function.
        #[salsa::tracked]
        fn try_parse<'db>(
            db: &'db dyn salsa::Database,
            input: TextInput,
        ) -> Option<crate::Operation<'db>> {
            parse_module(db, input.text(db)).ok()
        }

        /// Parse and print, returning the printed text (or None on parse failure).
        #[salsa::tracked]
        fn parse_and_print(db: &dyn salsa::Database, input: TextInput) -> Option<String> {
            let op = parse_module(db, input.text(db)).ok()?;
            Some(print_op(db, op))
        }

        /// Valid IR texts used as seed corpus for mutation.
        fn seed_corpus() -> Vec<&'static str> {
            vec![
                // Simple module with no-arg function
                concat!(
                    "core.module @test {\n",
                    "  func.func @main() -> core.i32 {\n",
                    "    %0 = arith.const {value = 40} : core.i32\n",
                    "    %1 = arith.const {value = 2} : core.i32\n",
                    "    %2 = arith.add %0, %1 : core.i32\n",
                    "    func.return %2\n",
                    "  }\n",
                    "}",
                ),
                // Function with named params
                concat!(
                    "core.module @test {\n",
                    "  func.func @add(%x: core.i32, %y: core.i32) -> core.i32 {\n",
                    "    %0 = arith.add %x, %y : core.i32\n",
                    "    func.return %0\n",
                    "  }\n",
                    "}",
                ),
                // Multiple functions
                concat!(
                    "core.module @test {\n",
                    "  func.func @foo() -> core.nil {\n",
                    "    %0 = arith.const {value = 42} : core.i32\n",
                    "    func.return %0\n",
                    "  }\n",
                    "  func.func @bar(%a: core.f64) -> core.f64 {\n",
                    "    func.return %a\n",
                    "  }\n",
                    "}",
                ),
                // Explicit block labels
                concat!(
                    "core.module @m {\n",
                    "  func.func @f() -> core.i32 {\n",
                    "    ^bb0:\n",
                    "      %0 = arith.const {value = 1} : core.i32\n",
                    "      func.return %0\n",
                    "  }\n",
                    "}",
                ),
                // Various attribute types
                concat!(
                    "core.module @test {\n",
                    "  func.func @g() -> core.nil {\n",
                    "    %0 = test.op {a = true, b = 99, c = \"hello\", d = @sym, e = unit} : core.nil\n",
                    "    func.return %0\n",
                    "  }\n",
                    "}",
                ),
            ]
        }

        /// Strategy: pick a seed and apply a random mutation.
        fn mutated_ir() -> impl Strategy<Value = String> {
            let seeds = seed_corpus();
            let n = seeds.len();
            (0..n, 0..1000usize, 0..5u8, proptest::num::u8::ANY).prop_map(
                move |(seed_idx, pos_raw, mutation_kind, random_byte)| {
                    let text = seeds[seed_idx];
                    let mut bytes = text.as_bytes().to_vec();
                    if bytes.is_empty() {
                        return String::new();
                    }
                    let pos = pos_raw % bytes.len();

                    match mutation_kind {
                        0 => {
                            // Replace byte
                            bytes[pos] = random_byte;
                        }
                        1 => {
                            // Delete byte
                            bytes.remove(pos);
                        }
                        2 => {
                            // Insert byte
                            bytes.insert(pos, random_byte);
                        }
                        3 => {
                            // Delete a chunk (up to 8 bytes)
                            let end = (pos + 8).min(bytes.len());
                            bytes.drain(pos..end);
                        }
                        _ => {
                            // Duplicate a chunk
                            let end = (pos + 8).min(bytes.len());
                            let chunk: Vec<u8> = bytes[pos..end].to_vec();
                            bytes.splice(pos..pos, chunk);
                        }
                    }

                    String::from_utf8(bytes).unwrap_or_default()
                },
            )
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(2000))]

            /// Parser must never panic on arbitrary mutated input.
            #[test]
            fn parser_never_panics(input in mutated_ir()) {
                let db = salsa::DatabaseImpl::default();
                db.attach(|db| {
                    let ti = TextInput::new(db, input.clone());
                    let _ = try_parse(db, ti);
                });
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]

            /// Completely random strings must not panic the parser.
            #[test]
            fn parser_handles_random_strings(input in "\\PC{0,200}") {
                let db = salsa::DatabaseImpl::default();
                db.attach(|db| {
                    let ti = TextInput::new(db, input.clone());
                    let _ = try_parse(db, ti);
                });
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]

            /// Valid seed texts must round-trip: print(parse(text)) → parse → print = same.
            #[test]
            fn seed_corpus_round_trips(seed_idx in 0..3usize) {
                let seeds = seed_corpus();
                let text = seeds[seed_idx].to_string();
                let db = salsa::DatabaseImpl::default();
                db.attach(|db| {
                    let ti = TextInput::new(db, text.clone());
                    if let Some(printed) = parse_and_print(db, ti) {
                        let ti2 = TextInput::new(db, printed.clone());
                        if let Some(printed2) = parse_and_print(db, ti2) {
                            prop_assert_eq!(printed, printed2, "round-trip mismatch");
                        }
                    }
                    Ok(())
                })?;
            }
        }

        // Individual combinator fuzzing: type parser.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(1000))]

            #[test]
            fn type_parser_never_panics(input in "[a-z_.()0-9, ]{0,80}") {
                let mut s = input.as_str();
                let _ = raw_type.parse_next(&mut s);
            }
        }

        // Individual combinator fuzzing: attribute value parser.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(1000))]

            #[test]
            fn attr_parser_never_panics(input in "[a-z0-9_.@\"\\[\\]{}, =truefalsnui]{0,60}") {
                let mut s = input.as_str();
                let _ = raw_attr_value.parse_next(&mut s);
            }
        }
    }
}

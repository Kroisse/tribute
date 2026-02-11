//! TrunkIR text format printer.
//!
//! Prints TrunkIR operations in an MLIR-inspired textual format with
//! dialect-qualified types. This format is designed for round-trip
//! fidelity with the parser in [`super::parser`].
//!
//! # Example output
//!
//! ```text
//! core.module @main {
//!   func.func @add(%arg0: core.i32, %arg1: core.i32) -> core.i32 {
//!   ^bb0(%arg0: core.i32, %arg1: core.i32):
//!     %0 = arith.const {value = 40} : core.i32
//!     %1 = arith.const {value = 2} : core.i32
//!     %2 = arith.add %0, %1 : core.i32
//!     func.return %2
//!   }
//! }
//! ```

use std::collections::HashMap;
use std::fmt::{self, Write};

use crate::{Attribute, Block, BlockId, Operation, Region, Symbol, Type, Value};

// ============================================================================
// Op printer registry (inventory-based)
// ============================================================================

/// Function signature for custom operation printers.
pub type OpPrintFn =
    for<'db> fn(&'db dyn salsa::Database, Operation<'db>, &mut PrintState<'db>) -> fmt::Result;

/// Registration entry for custom operation printers.
pub struct OpPrintRegistration {
    pub dialect: &'static str,
    pub op_name: &'static str,
    pub print: OpPrintFn,
}

inventory::collect!(OpPrintRegistration);

struct OpPrintRegistry {
    printers: HashMap<(Symbol, Symbol), OpPrintFn>,
}

impl OpPrintRegistry {
    fn new() -> Self {
        Self {
            printers: HashMap::new(),
        }
    }

    fn lookup(&self, dialect: Symbol, op_name: Symbol) -> Option<OpPrintFn> {
        self.printers.get(&(dialect, op_name)).copied()
    }
}

static PRINT_REGISTRY: std::sync::LazyLock<OpPrintRegistry> = std::sync::LazyLock::new(|| {
    let mut registry = OpPrintRegistry::new();
    for reg in inventory::iter::<OpPrintRegistration> {
        let dialect = Symbol::from_dynamic(reg.dialect);
        let op_name = Symbol::from_dynamic(reg.op_name);
        registry.printers.insert((dialect, op_name), reg.print);
    }
    registry
});

// ============================================================================
// Printer State
// ============================================================================

/// IR printer state, managing SSA value numbering and block labeling.
///
/// Uses a single lifetime since all Salsa data shares the same DB lifetime.
pub struct PrintState<'db> {
    db: &'db dyn salsa::Database,
    /// Maps Value -> printed name (e.g., "%0", "%arg0", "%x")
    value_names: HashMap<Value<'db>, String>,
    /// Maps BlockId -> block label (e.g., "^bb0")
    block_labels: HashMap<BlockId, String>,
    /// Next sequential value number (within current scope)
    next_value_num: usize,
    /// Next sequential block number (within current scope)
    next_block_num: usize,
    /// Output buffer
    pub output: String,
    /// Current indentation level (in spaces, 2-space indent)
    pub indent: usize,
}

impl<'db> PrintState<'db> {
    /// Create a new printer.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            value_names: HashMap::new(),
            block_labels: HashMap::new(),
            next_value_num: 0,
            next_block_num: 0,
            output: String::new(),
            indent: 0,
        }
    }

    /// Get the database reference.
    pub fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    /// Get the accumulated output.
    pub fn finish(self) -> String {
        self.output
    }

    // ---- Value naming ----

    /// Assign a name to a value (operation result).
    pub fn assign_result_name(&mut self, value: Value<'db>) -> String {
        let name = format!("%{}", self.next_value_num);
        self.next_value_num += 1;
        self.value_names.insert(value, name.clone());
        name
    }

    /// Assign a name to a block argument.
    pub fn assign_block_arg_name(
        &mut self,
        block: Block<'db>,
        index: usize,
        bind_name: Option<Symbol>,
    ) -> String {
        let value = block.arg(self.db, index);
        let name = if let Some(sym) = bind_name {
            let name_str = sym.to_string();
            let candidate = format!("%{name_str}");
            if self.value_names.values().any(|v| *v == candidate) {
                format!("%arg{}", index)
            } else {
                candidate
            }
        } else {
            format!("%arg{}", index)
        };
        self.value_names.insert(value, name.clone());
        name
    }

    /// Look up the name for a value (returns owned String to avoid borrow issues).
    pub fn get_value_name(&self, value: Value<'db>) -> String {
        self.value_names
            .get(&value)
            .cloned()
            .unwrap_or_else(|| "%?".to_string())
    }

    // ---- Block labeling ----

    pub fn assign_block_label(&mut self, block: Block<'db>) -> String {
        let block_id = block.id(self.db);
        let label = format!("^bb{}", self.next_block_num);
        self.next_block_num += 1;
        self.block_labels.insert(block_id, label.clone());
        label
    }

    pub fn get_block_label(&self, block: Block<'db>) -> String {
        let block_id = block.id(self.db);
        self.block_labels
            .get(&block_id)
            .cloned()
            .unwrap_or_else(|| "^bb?".to_string())
    }

    // ---- Indentation ----

    pub fn write_indent(&mut self) {
        for _ in 0..self.indent {
            self.output.push(' ');
        }
    }

    /// Reset value numbering and block labeling (for isolated regions).
    pub fn reset_numbering(&mut self) {
        self.next_value_num = 0;
        self.next_block_num = 0;
    }

    /// Save current numbering state.
    pub fn save_numbering(&self) -> (usize, usize) {
        (self.next_value_num, self.next_block_num)
    }

    /// Restore numbering state.
    pub fn restore_numbering(&mut self, state: (usize, usize)) {
        self.next_value_num = state.0;
        self.next_block_num = state.1;
    }
}

// ============================================================================
// Printing functions
// ============================================================================

/// Print a top-level operation (module or standalone).
fn print_top_level<'db>(state: &mut PrintState<'db>, op: Operation<'db>) -> fmt::Result {
    print_operation(state, op)?;
    state.output.push('\n');
    Ok(())
}

/// Print a single operation (dispatches to custom or generic).
fn print_operation<'db>(state: &mut PrintState<'db>, op: Operation<'db>) -> fmt::Result {
    let db = state.db;
    let dialect = op.dialect(db);
    let op_name = op.name(db);

    if let Some(custom_print) = PRINT_REGISTRY.lookup(dialect, op_name) {
        return custom_print(db, op, state);
    }

    print_operation_generic(state, op)
}

/// Print an operation in generic format.
fn print_operation_generic<'db>(state: &mut PrintState<'db>, op: Operation<'db>) -> fmt::Result {
    let db = state.db;
    let results = op.results(db);
    let operands = op.operands(db);
    let attributes = op.attributes(db);
    let regions = op.regions(db);

    // Results: %0 = or %0, %1 =
    if !results.is_empty() {
        let names: Vec<String> = (0..results.len())
            .map(|i| {
                let value = op.result(db, i);
                state.assign_result_name(value)
            })
            .collect();
        write!(state.output, "{} = ", names.join(", "))?;
    }

    // dialect.op
    write!(state.output, "{}.{}", op.dialect(db), op.name(db))?;

    // Operands: %a, %b
    if !operands.is_empty() {
        state.output.push(' ');
        for (i, &operand) in operands.iter().enumerate() {
            if i > 0 {
                state.output.push_str(", ");
            }
            let name = state.get_value_name(operand);
            state.output.push_str(&name);
        }
    }

    // Attributes: {key = value, ...}
    // Filter out "location" attribute since it's metadata
    let visible_attrs: Vec<_> = attributes
        .iter()
        .filter(|(k, _)| !k.with_str(|s| s == "location"))
        .collect();
    if !visible_attrs.is_empty() {
        state.output.push_str(" {");
        for (i, (key, value)) in visible_attrs.iter().enumerate() {
            if i > 0 {
                state.output.push_str(", ");
            }
            write!(state.output, "{} = ", key)?;
            print_attribute(state, value)?;
        }
        state.output.push('}');
    }

    // Type annotation: : type1, type2
    if !results.is_empty() {
        state.output.push_str(" : ");
        for (i, result_ty) in results.iter().enumerate() {
            if i > 0 {
                state.output.push_str(", ");
            }
            print_type(state, *result_ty)?;
        }
    }

    // Regions
    for region in regions.iter() {
        state.output.push(' ');
        print_region(state, *region)?;
    }

    Ok(())
}

/// Print a region with blocks.
fn print_region<'db>(state: &mut PrintState<'db>, region: Region<'db>) -> fmt::Result {
    let db = state.db;
    let blocks = region.blocks(db);

    state.output.push_str("{\n");

    // First pass: assign block labels and block arg names
    for block in blocks.iter() {
        state.assign_block_label(*block);
        assign_block_arg_names(state, *block);
    }

    // Second pass: print blocks
    let single_block_no_args = blocks.len() == 1 && blocks[0].args(db).is_empty();
    for block in blocks.iter() {
        print_block(state, *block, single_block_no_args)?;
    }

    state.write_indent();
    state.output.push('}');

    Ok(())
}

/// Assign names to all arguments of a block.
fn assign_block_arg_names<'db>(state: &mut PrintState<'db>, block: Block<'db>) {
    let db = state.db;
    let args = block.args(db);
    for (i, arg) in args.iter().enumerate() {
        let bind_name = arg
            .get_attr(db, Symbol::new("bind_name"))
            .and_then(|a| match a {
                Attribute::Symbol(s) => Some(*s),
                _ => None,
            });
        state.assign_block_arg_name(block, i, bind_name);
    }
}

/// Print a block with label, arguments, and operations.
///
/// If `elide_label` is true, the block label line is omitted entirely.
/// Callers decide when to elide (e.g., single-block regions, func bodies).
fn print_block<'db>(
    state: &mut PrintState<'db>,
    block: Block<'db>,
    elide_label: bool,
) -> fmt::Result {
    let db = state.db;
    let args = block.args(db);

    // Block label (may be elided for single-block regions or func bodies)
    if !elide_label {
        state.write_indent();
        let label = state.get_block_label(block);
        state.output.push_str(&label);

        if !args.is_empty() {
            state.output.push('(');
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    state.output.push_str(", ");
                }
                let value = block.arg(db, i);
                let name = state.get_value_name(value);
                state.output.push_str(&name);
                state.output.push_str(": ");
                print_type(state, arg.ty(db))?;
            }
            state.output.push(')');
        }
        state.output.push_str(":\n");
    }

    // Operations
    state.indent += 2;
    for op in block.operations(db).iter() {
        state.write_indent();
        print_operation(state, *op)?;
        state.output.push('\n');
    }
    state.indent -= 2;

    Ok(())
}

/// Print a type in dialect-qualified format.
pub fn print_type<'db>(state: &mut PrintState<'db>, ty: Type<'db>) -> fmt::Result {
    let db = state.db;
    let dialect = ty.dialect(db);
    let name = ty.name(db);
    let params = ty.params(db);
    let attrs = ty.attrs(db);

    write!(state.output, "{}.{}", dialect, name)?;

    if !params.is_empty() {
        state.output.push('(');
        for (i, &p) in params.iter().enumerate() {
            if i > 0 {
                state.output.push_str(", ");
            }
            print_type(state, p)?;
        }
        state.output.push(')');
    }

    if !attrs.is_empty() {
        state.output.push_str(" {");
        for (i, (key, value)) in attrs.iter().enumerate() {
            if i > 0 {
                state.output.push_str(", ");
            }
            write!(state.output, "{} = ", key)?;
            print_attribute(state, value)?;
        }
        state.output.push('}');
    }

    Ok(())
}

/// Print an attribute value.
pub fn print_attribute<'db>(state: &mut PrintState<'db>, attr: &Attribute<'db>) -> fmt::Result {
    match attr {
        Attribute::Unit => state.output.push_str("unit"),
        Attribute::Bool(b) => write!(state.output, "{}", b)?,
        Attribute::IntBits(n) => write!(state.output, "{}", n)?,
        Attribute::FloatBits(bits) => {
            let f = f64::from_bits(*bits);
            if f.fract() == 0.0 && f.is_finite() {
                write!(state.output, "{:.1}", f)?;
            } else {
                write!(state.output, "{}", f)?;
            }
        }
        Attribute::String(s) => {
            write!(
                state.output,
                "\"{}\"",
                s.replace('\\', "\\\\").replace('"', "\\\"")
            )?;
        }
        Attribute::Bytes(bytes) => {
            state.output.push_str("bytes(");
            for (i, b) in bytes.iter().enumerate() {
                if i > 0 {
                    state.output.push_str(", ");
                }
                write!(state.output, "{}", b)?;
            }
            state.output.push(')');
        }
        Attribute::Type(ty) => print_type(state, *ty)?,
        Attribute::Symbol(sym) => print_symbol(state, *sym)?,
        Attribute::List(items) => {
            state.output.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    state.output.push_str(", ");
                }
                print_attribute(state, item)?;
            }
            state.output.push(']');
        }
        Attribute::Location(loc) => {
            write!(
                state.output,
                "loc(\"{}\" {}:{})",
                loc.path.uri(state.db),
                loc.span.start,
                loc.span.end
            )?;
        }
    }
    Ok(())
}

/// Print a symbol reference (@ prefix).
pub fn print_symbol(state: &mut PrintState<'_>, sym: Symbol) -> fmt::Result {
    // Note: no lifetime constraint needed since Symbol is Copy and 'static-like
    sym.with_str(|s| {
        let needs_quotes = s.contains("::") || s.contains(' ') || s.contains('.');
        if needs_quotes {
            write!(state.output, "@\"{}\"", s)
        } else {
            write!(state.output, "@{}", s)
        }
    })
}

// ============================================================================
// Public API
// ============================================================================

/// Print a TrunkIR operation to a string in textual format.
pub fn print_op(db: &dyn salsa::Database, op: Operation<'_>) -> String {
    let mut state = PrintState::new(db);
    print_top_level(&mut state, op).expect("printing should not fail");
    state.finish()
}

// ============================================================================
// Custom printers for core operations
// ============================================================================

// core.module custom printer
inventory::submit! {
    OpPrintRegistration {
        dialect: "core",
        op_name: "module",
        print: print_core_module,
    }
}

fn print_core_module<'db>(
    db: &'db dyn salsa::Database,
    op: Operation<'db>,
    state: &mut PrintState<'db>,
) -> fmt::Result {
    let attrs = op.attributes(db);
    let name = attrs.get(&Symbol::new("sym_name")).and_then(|a| match a {
        Attribute::Symbol(s) => Some(*s),
        _ => None,
    });

    state.output.push_str("core.module");
    if let Some(sym) = name {
        state.output.push(' ');
        print_symbol(state, sym)?;
    }

    // Save and reset numbering for isolated region
    let saved = state.save_numbering();
    state.reset_numbering();

    let regions = op.regions(db);
    if let Some(region) = regions.first() {
        state.output.push(' ');
        print_region(state, *region)?;
    }

    state.restore_numbering(saved);
    Ok(())
}

// func.func custom printer
inventory::submit! {
    OpPrintRegistration {
        dialect: "func",
        op_name: "func",
        print: print_func_func,
    }
}

fn print_func_func<'db>(
    db: &'db dyn salsa::Database,
    op: Operation<'db>,
    state: &mut PrintState<'db>,
) -> fmt::Result {
    use crate::dialect::{core as core_dialect, func};
    use crate::{DialectOp, DialectType};

    let func_op = func::Func::from_operation(db, op).map_err(|_| fmt::Error)?;
    let name = func_op.sym_name(db);
    let func_ty = func_op.r#type(db);

    state.output.push_str("func.func ");
    print_symbol(state, name)?;

    // Save and reset numbering for isolated region
    let saved = state.save_numbering();
    state.reset_numbering();

    // Get function type info
    let func_type_wrapper = core_dialect::Func::from_type(db, func_ty);
    let result_ty = func_type_wrapper
        .as_ref()
        .map(|ft| ft.result(db))
        .unwrap_or(func_ty);
    let effect = func_type_wrapper.as_ref().and_then(|ft| ft.effect(db));

    // Parameters: extract names from block arguments
    let body = func_op.body(db);
    let entry_block = body.blocks(db).first().copied();

    state.output.push('(');
    if let Some(block) = entry_block {
        state.assign_block_label(block);

        let args = block.args(db);
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                state.output.push_str(", ");
            }
            let bind_name = arg
                .get_attr(db, Symbol::new("bind_name"))
                .and_then(|a| match a {
                    Attribute::Symbol(s) => Some(*s),
                    _ => None,
                });
            let value_name = state.assign_block_arg_name(block, i, bind_name);
            state.output.push_str(&value_name);
            state.output.push_str(": ");
            print_type(state, arg.ty(db))?;
        }
    }
    state.output.push(')');

    // Return type
    state.output.push_str(" -> ");
    print_type(state, result_ty)?;

    // Effect
    if let Some(eff) = effect {
        state.output.push_str(" effects ");
        print_type(state, eff)?;
    }

    // Print extra attributes (excluding sym_name, type, name_location, location)
    let extra_attrs: Vec<_> = op
        .attributes(db)
        .iter()
        .filter(|(k, _)| {
            !k.with_str(|s| matches!(s, "sym_name" | "type" | "name_location" | "location"))
        })
        .collect();
    if !extra_attrs.is_empty() {
        state.output.push_str(" {");
        for (i, (key, value)) in extra_attrs.iter().enumerate() {
            if i > 0 {
                state.output.push_str(", ");
            }
            write!(state.output, "{} = ", key)?;
            print_attribute(state, value)?;
        }
        state.output.push('}');
    }

    // Body region
    state.output.push_str(" {\n");
    let blocks = body.blocks(db);

    // Assign remaining block labels and args (for blocks after entry)
    for block in blocks.iter().skip(1) {
        state.assign_block_label(*block);
        assign_block_arg_names(state, *block);
    }

    // Elide entry block label when function body has a single block
    // (block args are already shown in the function signature)
    let single_block = blocks.len() == 1;
    state.indent += 2;
    for block in blocks.iter() {
        print_block(state, *block, single_block)?;
    }
    state.indent -= 2;

    state.write_indent();
    state.output.push('}');

    state.restore_numbering(saved);
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DialectOp, DialectType, Location, PathId, Span,
        dialect::{arith, core, func},
        idvec,
    };
    use salsa_test_macros::salsa_test;

    #[salsa::tracked]
    fn build_simple_module(db: &dyn salsa::Database) -> Operation<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));

        let main_func = func::Func::build(
            db,
            location,
            "main",
            idvec![],
            core::I32::new(db).as_type(),
            |entry| {
                let c0 = entry.op(arith::Const::i32(db, location, 40));
                let c1 = entry.op(arith::Const::i32(db, location, 2));
                let add = entry.op(arith::add(
                    db,
                    location,
                    c0.result(db),
                    c1.result(db),
                    core::I32::new(db).as_type(),
                ));
                entry.op(func::Return::value(db, location, add.result(db)));
            },
        );

        core::Module::build(db, location, "test".into(), |top| {
            top.op(main_func);
        })
        .as_operation()
    }

    #[salsa_test]
    fn test_print_simple_module(db: &salsa::DatabaseImpl) {
        let op = build_simple_module(db);
        let text = print_op(db, op);
        insta::assert_snapshot!(text);
    }

    #[salsa::tracked]
    fn build_module_with_params(db: &dyn salsa::Database) -> Operation<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let i32_ty = core::I32::new(db).as_type();

        let add_func = func::Func::build_with_named_params(
            db,
            location,
            "add",
            None,
            vec![
                (i32_ty, Some(Symbol::new("x"))),
                (i32_ty, Some(Symbol::new("y"))),
            ],
            i32_ty,
            None,
            |entry, args| {
                let add = entry.op(arith::add(db, location, args[0], args[1], i32_ty));
                entry.op(func::Return::value(db, location, add.result(db)));
            },
        );

        core::Module::build(db, location, "test".into(), |top| {
            top.op(add_func);
        })
        .as_operation()
    }

    #[salsa_test]
    fn test_print_module_with_params(db: &salsa::DatabaseImpl) {
        let op = build_module_with_params(db);
        let text = print_op(db, op);
        insta::assert_snapshot!(text);
    }
}

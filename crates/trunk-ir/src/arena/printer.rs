//! Text format printer for arena-based IR.
//!
//! Prints IR in a format similar to the Salsa-based printer:
//!
//! ```text
//! core.module @name {
//!   func.func @main(%arg0: core.i32) -> core.i32 {
//!     %0 = arith.const {value = 42} : core.i32
//!     func.return %0
//!   }
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write;

use super::context::IrContext;
use super::refs::*;
use super::types::*;

/// Print state for value numbering and block labeling.
struct PrintState<'a> {
    ctx: &'a IrContext,
    value_names: HashMap<ValueRef, String>,
    block_labels: HashMap<BlockRef, String>,
    next_value_num: usize,
    next_block_num: usize,
}

impl<'a> PrintState<'a> {
    fn new(ctx: &'a IrContext) -> Self {
        Self {
            ctx,
            value_names: HashMap::new(),
            block_labels: HashMap::new(),
            next_value_num: 0,
            next_block_num: 0,
        }
    }

    fn assign_value_name(&mut self, v: ValueRef) -> String {
        let name = format!("%{}", self.next_value_num);
        self.next_value_num += 1;
        self.value_names.insert(v, name.clone());
        name
    }

    fn get_value_name(&self, v: ValueRef) -> &str {
        self.value_names.get(&v).map(|s| s.as_str()).unwrap_or("%?")
    }

    fn assign_block_label(&mut self, b: BlockRef) -> String {
        let label = format!("^bb{}", self.next_block_num);
        self.next_block_num += 1;
        self.block_labels.insert(b, label.clone());
        label
    }

    fn get_block_label(&self, b: BlockRef) -> &str {
        self.block_labels
            .get(&b)
            .map(|s| s.as_str())
            .unwrap_or("^bb?")
    }

    /// Save only counters (not maps). Use with `restore_counters`.
    fn save_counters(&self) -> (usize, usize) {
        (self.next_value_num, self.next_block_num)
    }

    fn reset_numbering(&mut self) {
        self.next_value_num = 0;
        self.next_block_num = 0;
        self.value_names.clear();
        self.block_labels.clear();
    }

    /// Restore only counters saved by `save_counters`.
    fn restore_counters(&mut self, state: (usize, usize)) {
        self.next_value_num = state.0;
        self.next_block_num = state.1;
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Print an operation as IR text.
pub fn print_op(ctx: &IrContext, op: OpRef) -> String {
    let mut state = PrintState::new(ctx);
    let mut out = String::new();
    print_operation(&mut state, &mut out, op, 0).expect("fmt::Write to String never fails");
    out
}

/// Print a type as IR text.
pub fn print_type(ctx: &IrContext, ty: TypeRef) -> String {
    let mut out = String::new();
    write_type(ctx, &mut out, ty).expect("fmt::Write to String never fails");
    out
}

/// Print a module (root operation with nested functions) as IR text.
pub fn print_module(ctx: &IrContext, root: OpRef) -> String {
    let mut state = PrintState::new(ctx);
    let mut out = String::new();
    print_module_op(&mut state, &mut out, root).expect("fmt::Write to String never fails");
    out
}

// ============================================================================
// Type printing
// ============================================================================

fn write_type(ctx: &IrContext, f: &mut impl Write, ty: TypeRef) -> fmt::Result {
    let data = ctx.types.get(ty);
    write!(f, "{}.{}", data.dialect, data.name)?;
    if !data.params.is_empty() {
        f.write_char('(')?;
        for (i, &param) in data.params.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write_type(ctx, f, param)?;
        }
        f.write_char(')')?;
    } else if !data.attrs.is_empty() {
        // Empty parens signal that attrs follow
        f.write_str("()")?;
    }
    if !data.attrs.is_empty() {
        f.write_str(" {")?;
        for (i, (key, val)) in data.attrs.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{} = ", key)?;
            write_attribute(ctx, f, val)?;
        }
        f.write_char('}')?;
    }
    Ok(())
}

// ============================================================================
// Attribute printing
// ============================================================================

fn write_attribute(ctx: &IrContext, f: &mut impl Write, attr: &Attribute) -> fmt::Result {
    match attr {
        Attribute::Unit => f.write_str("unit"),
        Attribute::Bool(b) => write!(f, "{b}"),
        Attribute::IntBits(v) => write!(f, "{v}"),
        Attribute::FloatBits(bits) => {
            let v = f64::from_bits(*bits);
            let s = format!("{v}");
            f.write_str(&s)?;
            // Ensure decimal point for finite whole numbers (don't corrupt inf/NaN)
            if v.is_finite() && !s.contains('.') && !s.contains('e') && !s.contains('E') {
                f.write_str(".0")?;
            }
            Ok(())
        }
        Attribute::String(s) => {
            f.write_char('"')?;
            write_escaped_string(f, s)?;
            f.write_char('"')
        }
        Attribute::Bytes(bytes) => {
            f.write_str("bytes(")?;
            for (i, b) in bytes.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{b}")?;
            }
            f.write_char(')')
        }
        Attribute::Symbol(sym) => write_symbol(f, *sym),
        Attribute::Type(ty) => write_type(ctx, f, *ty),
        Attribute::List(list) => {
            f.write_char('[')?;
            for (i, item) in list.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write_attribute(ctx, f, item)?;
            }
            f.write_char(']')
        }
        Attribute::Location(loc) => {
            let path_str = ctx.paths.get(loc.path);
            f.write_str("loc(\"")?;
            write_escaped_string(f, path_str)?;
            write!(f, "\" {}:{})", loc.span.start, loc.span.end)
        }
    }
}

fn write_escaped_string(f: &mut impl Write, s: &str) -> fmt::Result {
    for ch in s.chars() {
        match ch {
            '\\' => f.write_str("\\\\")?,
            '"' => f.write_str("\\\"")?,
            '\n' => f.write_str("\\n")?,
            '\t' => f.write_str("\\t")?,
            '\r' => f.write_str("\\r")?,
            '\0' => f.write_str("\\0")?,
            c if c.is_control() => write!(f, "\\x{:02x}", c as u32)?,
            c => f.write_char(c)?,
        }
    }
    Ok(())
}

fn write_symbol(f: &mut impl Write, sym: crate::ir::Symbol) -> fmt::Result {
    sym.with_str(|s| {
        let needs_quoting = s.is_empty() || !s.chars().all(|c| c.is_alphanumeric() || c == '_');
        if needs_quoting {
            f.write_str("@\"")?;
            write_escaped_string(f, s)?;
            f.write_char('"')
        } else {
            write!(f, "@{s}")
        }
    })
}

// ============================================================================
// Operation printing
// ============================================================================

fn print_operation(
    state: &mut PrintState<'_>,
    f: &mut impl Write,
    op: OpRef,
    indent: usize,
) -> fmt::Result {
    let data = state.ctx.op(op);
    let dialect = data.dialect;
    let name = data.name;

    // Check for special ops
    let is_func = dialect == crate::Symbol::new("func") && name == crate::Symbol::new("func");
    let is_module = dialect == crate::Symbol::new("core") && name == crate::Symbol::new("module");

    if is_module {
        return print_module_op(state, f, op);
    }
    if is_func {
        return print_func_op(state, f, op, indent);
    }

    print_generic_op(state, f, op, indent)
}

fn print_generic_op(
    state: &mut PrintState<'_>,
    f: &mut impl Write,
    op: OpRef,
    indent: usize,
) -> fmt::Result {
    let indent_str = " ".repeat(indent);
    write!(f, "{indent_str}")?;

    // Results
    let results = state.ctx.op_results(op);
    if results.len() == 1 {
        let name = state.assign_value_name(results[0]);
        write!(f, "{name} = ")?;
    } else if results.len() > 1 {
        for (i, &v) in results.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            let name = state.assign_value_name(v);
            f.write_str(&name)?;
        }
        f.write_str(" = ")?;
    }

    // Dialect.op
    let data = state.ctx.op(op);
    write!(f, "{}.{}", data.dialect, data.name)?;

    // Operands
    let operands = state.ctx.op_operands(op);
    if !operands.is_empty() {
        f.write_char(' ')?;
        for (i, &v) in operands.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(state.get_value_name(v))?;
        }
    }

    // Successors
    let successors = &state.ctx.op(op).successors;
    if !successors.is_empty() {
        f.write_str(" [")?;
        for (i, &b) in successors.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(state.get_block_label(b))?;
        }
        f.write_char(']')?;
    }

    // Attributes
    let attrs = &state.ctx.op(op).attributes;
    if !attrs.is_empty() {
        f.write_str(" {")?;
        for (i, (key, val)) in attrs.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{key} = ")?;
            write_attribute(state.ctx, f, val)?;
        }
        f.write_char('}')?;
    }

    // Result types
    let result_types = state.ctx.op_result_types(op);
    if !result_types.is_empty() {
        f.write_str(" : ")?;
        for (i, &ty) in result_types.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write_type(state.ctx, f, ty)?;
        }
    }

    // Regions
    let regions = &state.ctx.op(op).regions;
    for &region in regions.iter() {
        f.write_str(" {\n")?;
        print_region(state, f, region, indent + 2)?;
        write!(f, "{indent_str}}}")?;
    }

    f.write_char('\n')
}

// ============================================================================
// Region / Block printing
// ============================================================================

fn print_region(
    state: &mut PrintState<'_>,
    f: &mut impl Write,
    region: RegionRef,
    indent: usize,
) -> fmt::Result {
    let region_data = state.ctx.region(region);
    let blocks: Vec<_> = region_data.blocks.iter().copied().collect();

    // Pre-assign block labels
    for &block in &blocks {
        state.assign_block_label(block);
    }

    let can_elide_label = blocks.len() == 1 && state.ctx.block_args(blocks[0]).is_empty();

    for (i, &block) in blocks.iter().enumerate() {
        if !can_elide_label {
            let indent_str = " ".repeat(indent);
            let label = state.get_block_label(block).to_owned();
            write!(f, "{indent_str}{label}")?;
            let args = state.ctx.block_args(block);
            if !args.is_empty() {
                f.write_char('(')?;
                for (j, &arg) in args.iter().enumerate() {
                    if j > 0 {
                        f.write_str(", ")?;
                    }
                    let arg_name = state.assign_value_name(arg);
                    let ty = state.ctx.value_ty(arg);
                    write!(f, "{arg_name}: ")?;
                    write_type(state.ctx, f, ty)?;
                }
                f.write_char(')')?;
            }
            f.write_str(":\n")?;
        }

        // Print ops in this block
        let block_data = state.ctx.block(block);
        let ops: Vec<_> = block_data.ops.iter().copied().collect();
        for &op in &ops {
            print_operation(state, f, op, indent + 2)?;
        }
        if i + 1 < blocks.len() {
            // Add blank line between blocks for readability
            f.write_char('\n')?;
        }
    }

    Ok(())
}

// ============================================================================
// Special operation printers
// ============================================================================

fn print_module_op(state: &mut PrintState<'_>, f: &mut impl Write, op: OpRef) -> fmt::Result {
    let data = state.ctx.op(op);
    write!(f, "core.module")?;

    // Module name
    if let Some(Attribute::Symbol(name)) = data.attributes.get(&crate::Symbol::new("sym_name")) {
        f.write_char(' ')?;
        write_symbol(f, *name)?;
    }

    let regions = &data.regions;
    if let Some(&region) = regions.first() {
        f.write_str(" {\n")?;

        // Print each top-level op with reset numbering
        let region_data = state.ctx.region(region);
        let blocks: Vec<_> = region_data.blocks.iter().copied().collect();
        for &block in &blocks {
            let block_data = state.ctx.block(block);
            let ops: Vec<_> = block_data.ops.iter().copied().collect();
            for &child_op in &ops {
                let saved = state.save_counters();
                state.reset_numbering();
                print_operation(state, f, child_op, 2)?;
                state.reset_numbering();
                state.restore_counters(saved);
            }
        }

        f.write_str("}\n")?;
    } else {
        f.write_char('\n')?;
    }

    Ok(())
}

fn print_func_op(
    state: &mut PrintState<'_>,
    f: &mut impl Write,
    op: OpRef,
    indent: usize,
) -> fmt::Result {
    let indent_str = " ".repeat(indent);
    let data = state.ctx.op(op);

    write!(f, "{indent_str}func.func")?;

    // Function name
    if let Some(Attribute::Symbol(name)) = data.attributes.get(&crate::Symbol::new("sym_name")) {
        f.write_char(' ')?;
        write_symbol(f, *name)?;
    }

    // Reset numbering for function body
    state.reset_numbering();

    let regions = &data.regions;
    if let Some(&region) = regions.first() {
        let region_data = state.ctx.region(region);
        let blocks: Vec<_> = region_data.blocks.iter().copied().collect();

        // Print entry block args as function signature
        if let Some(&entry_block) = blocks.first() {
            let args = state.ctx.block_args(entry_block);
            f.write_char('(')?;
            for (i, &arg) in args.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                let name = state.assign_value_name(arg);
                let ty = state.ctx.value_ty(arg);
                write!(f, "{name}: ")?;
                write_type(state.ctx, f, ty)?;
            }
            f.write_char(')')?;
        }

        // Return type from func type attribute
        if let Some(Attribute::Type(func_ty)) = data.attributes.get(&crate::Symbol::new("type")) {
            f.write_str(" -> ")?;
            write_type(state.ctx, f, *func_ty)?;
        }

        // Body
        f.write_str(" {\n")?;

        // Pre-assign block labels for all blocks
        for &block in &blocks {
            state.assign_block_label(block);
        }

        let single_block_no_args = blocks.len() == 1 && state.ctx.block_args(blocks[0]).is_empty();

        for (i, &block) in blocks.iter().enumerate() {
            // For entry block, args already printed in signature
            if i == 0 && single_block_no_args {
                // Elide entry block label
            } else if i > 0 || !single_block_no_args {
                let label = state.get_block_label(block).to_owned();
                write!(f, "{}  {label}", indent_str)?;
                if i > 0 {
                    // Non-entry blocks show args
                    let args = state.ctx.block_args(block);
                    if !args.is_empty() {
                        f.write_char('(')?;
                        for (j, &arg) in args.iter().enumerate() {
                            if j > 0 {
                                f.write_str(", ")?;
                            }
                            let name = state.assign_value_name(arg);
                            let ty = state.ctx.value_ty(arg);
                            write!(f, "{name}: ")?;
                            write_type(state.ctx, f, ty)?;
                        }
                        f.write_char(')')?;
                    }
                }
                f.write_str(":\n")?;
            }

            let block_data = state.ctx.block(block);
            let ops: Vec<_> = block_data.ops.iter().copied().collect();
            for &child_op in &ops {
                print_operation(state, f, child_op, indent + 4)?;
            }
        }

        writeln!(f, "{indent_str}}}")?;
    } else {
        f.write_char('\n')?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Symbol;
    use crate::arena::dialect::{arith, core, func};
    use crate::arena::{BlockArgData, BlockData, RegionData};
    use smallvec::smallvec;

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.paths.intern("test.trb".to_owned());
        Location::new(path, crate::Span::new(0, 0))
    }

    fn make_i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("i32"),
            params: Default::default(),
            attrs: Default::default(),
        })
    }

    fn make_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
        let mut p = smallvec::SmallVec::new();
        p.push(ret);
        p.extend_from_slice(params);
        ctx.types.intern(TypeData {
            dialect: Symbol::new("func"),
            name: Symbol::new("fn"),
            params: p,
            attrs: Default::default(),
        })
    }

    #[test]
    fn test_print_type_simple() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        assert_eq!(print_type(&ctx, i32_ty), "core.i32");
    }

    #[test]
    fn test_print_type_with_params() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let tuple_ty = ctx.types.intern(TypeData {
            dialect: Symbol::new("core"),
            name: Symbol::new("tuple"),
            params: smallvec::smallvec![i32_ty, i32_ty],
            attrs: Default::default(),
        });
        assert_eq!(print_type(&ctx, tuple_ty), "core.tuple(core.i32, core.i32)");
    }

    #[test]
    fn test_print_simple_op() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
        let output = print_op(&ctx, c.op_ref());
        assert_eq!(output, "%0 = arith.const {value = 42} : core.i32\n");
    }

    #[test]
    fn test_print_binary_op() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(1));
        let c2 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(2));
        let v1 = c1.result(&ctx);
        let v2 = c2.result(&ctx);

        let add = arith::add(&mut ctx, loc, v1, v2, i32_ty);
        // Printer assigns names based on what it sees - standalone op print
        // only numbers the result of THIS op since it can't see c1/c2
        let output = print_op(&ctx, add.op_ref());
        // %0 = arith.add %?, %? : core.i32 (operands unknown since not in scope)
        assert!(output.contains("arith.add"));
        assert!(output.contains("core.i32"));
    }

    #[test]
    fn test_print_simple_function() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let func_ty = make_func_type(&mut ctx, &[i32_ty, i32_ty], i32_ty);

        // Build: fn add(x: i32, y: i32) -> i32 { return x + y; }
        let entry_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![
                BlockArgData {
                    ty: i32_ty,
                    attrs: Default::default(),
                },
                BlockArgData {
                    ty: i32_ty,
                    attrs: Default::default(),
                },
            ],
            ops: Default::default(),
            parent_region: None,
        });

        // x + y
        let x = ctx.block_arg(entry_block, 0);
        let y = ctx.block_arg(entry_block, 1);
        let add = arith::add(&mut ctx, loc, x, y, i32_ty);
        ctx.push_op(entry_block, add.op_ref());

        // return result
        let result = add.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [result]);
        ctx.push_op(entry_block, ret.op_ref());

        // Region
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });

        // Function
        let f = func::func(&mut ctx, loc, Symbol::new("add"), func_ty, body);

        let output = print_op(&ctx, f.op_ref());
        assert!(output.contains("func.func @add"));
        assert!(output.contains("%0: core.i32"));
        assert!(output.contains("%1: core.i32"));
        assert!(output.contains("arith.add %0, %1"));
        assert!(output.contains("func.return %2"));
    }

    #[test]
    fn test_print_module() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let func_ty = make_func_type(&mut ctx, &[], i32_ty);

        // Build: fn main() -> i32 { return 42; }
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });

        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
        ctx.push_op(entry, c.op_ref());

        let result = c.result(&ctx);
        let ret = func::r#return(&mut ctx, loc, [result]);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let f = func::func(&mut ctx, loc, Symbol::new("main"), func_ty, body);

        // Module
        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        ctx.push_op(mod_block, f.op_ref());

        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        let module = core::module(&mut ctx, loc, Symbol::new("test"), mod_region);

        let output = print_module(&ctx, module.op_ref());
        assert!(output.contains("core.module @test"));
        assert!(output.contains("func.func @main"));
        assert!(output.contains("arith.const {value = 42}"));
        assert!(output.contains("func.return %0"));
    }

    #[test]
    fn test_print_attribute_types() {
        let ctx = IrContext::new();
        let mut out = String::new();

        // Bool
        write_attribute(&ctx, &mut out, &Attribute::Bool(true)).unwrap();
        assert_eq!(out, "true");

        // Float
        out.clear();
        write_attribute(&ctx, &mut out, &Attribute::FloatBits(2.78f64.to_bits())).unwrap();
        assert_eq!(out, "2.78");

        // Float whole number
        out.clear();
        write_attribute(&ctx, &mut out, &Attribute::FloatBits(42.0f64.to_bits())).unwrap();
        assert_eq!(out, "42.0");

        // String
        out.clear();
        write_attribute(
            &ctx,
            &mut out,
            &Attribute::String("hello\nworld".to_owned()),
        )
        .unwrap();
        assert_eq!(out, r#""hello\nworld""#);

        // Symbol
        out.clear();
        write_attribute(&ctx, &mut out, &Attribute::Symbol(Symbol::new("foo"))).unwrap();
        assert_eq!(out, "@foo");

        // Symbol with path (needs quoting)
        out.clear();
        write_attribute(
            &ctx,
            &mut out,
            &Attribute::Symbol(Symbol::from_dynamic("std::List::map")),
        )
        .unwrap();
        assert_eq!(out, r#"@"std::List::map""#);

        // Empty symbol (should quote)
        out.clear();
        write_symbol(&mut out, Symbol::from_dynamic("")).unwrap();
        assert_eq!(out, r#"@"""#);

        // Float infinity (should not append .0)
        out.clear();
        write_attribute(
            &ctx,
            &mut out,
            &Attribute::FloatBits(f64::INFINITY.to_bits()),
        )
        .unwrap();
        assert_eq!(out, "inf");

        // Float NaN (should not append .0)
        out.clear();
        write_attribute(&ctx, &mut out, &Attribute::FloatBits(f64::NAN.to_bits())).unwrap();
        assert_eq!(out, "NaN");

        // Float negative infinity
        out.clear();
        write_attribute(
            &ctx,
            &mut out,
            &Attribute::FloatBits(f64::NEG_INFINITY.to_bits()),
        )
        .unwrap();
        assert_eq!(out, "-inf");
    }

    #[test]
    fn test_rauw_reflected_in_print() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Create: %0 = const 42; %1 = const 99; add(%0, %0) → replace %0 with %1
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });

        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(42));
        ctx.push_op(entry, c1.op_ref());
        let v1 = c1.result(&ctx);

        let c2 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::IntBits(99));
        ctx.push_op(entry, c2.op_ref());
        let v2 = c2.result(&ctx);

        let add = arith::add(&mut ctx, loc, v1, v1, i32_ty);
        ctx.push_op(entry, add.op_ref());

        // RAUW: replace v1 with v2
        ctx.replace_all_uses(v1, v2);

        // Verify operands were updated
        let operands = ctx.op_operands(add.op_ref());
        assert_eq!(operands[0], v2);
        assert_eq!(operands[1], v2);
    }
}

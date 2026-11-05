//! Text format printer for arena-based IR.
//!
//! Prints IR in a format similar to the Salsa-based printer:
//!
//! ```text
//! core.module @name {
//!   func.func @main(%arg0: core.i32) -> core.i32 {
//!     %0 = arith.constant 42 : core.i32
//!     func.return %0
//!   }
//! }
//! ```
//!
//! Printing is split as MLIR splits it. A pre-pass computes an immutable
//! [`AsmState`]: the name of every value and block label the printer
//! writes, in the order it writes them ([`AsmNamer`]), and the type aliases
//! of each module. Its leaves are [`Display`](fmt::Display) values, such as
//! [`AsmState::ty`] and [`AsmState::value`]. The mutable [`AsmPrinter`] and
//! [`OpAsmPrinter`] write them, with `write!`, to any [`fmt::Write`], and own
//! the indentation.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::fmt;
use std::fmt::Write;
use std::ops::{ControlFlow, Deref, DerefMut};
use std::rc::Rc;

use itertools::Itertools;

use super::context::IrContext;
use super::ops::DialectOp;
use super::refs::*;
use super::types::*;
use super::walk::{WalkAction, walk_region};
use crate::dialect::core;

// ============================================================================
// AsmState — names and aliases computed before printing
// ============================================================================

/// What printing reads and never changes: the context, the name of each
/// value and block label, and the type aliases in scope, as MLIR's
/// `AsmState` holds them.
///
/// [`AsmState::new`] runs the naming pre-pass for an operation. The
/// leaves of the printed form are [`Display`](fmt::Display) values a
/// format writes with `write!`:
///
/// ```ignore
/// let s = p.state();
/// write!(p, "{} : {}", s.value(value), s.ty(ty))?;
/// ```
pub struct AsmState<'a> {
    ctx: &'a IrContext,
    names: Rc<AsmNames>,
    /// The alias of each aliased type in scope.
    aliases: HashMap<TypeRef, String>,
    /// The state of each module printed inside this scope.
    modules: HashMap<OpRef, ModuleState<'a>>,
}

/// A module's type alias definitions and the state of its body.
struct ModuleState<'a> {
    /// The context's aliases, which every module defines, in the order it
    /// writes them, each with the state its definition is written in when
    /// it differs from the enclosing scope's.
    manual: Vec<ManualAlias<'a>>,
    /// The aliases the printer generates for this module's body, in the
    /// order it writes them.
    auto: Vec<(String, TypeRef)>,
    body: AsmState<'a>,
}

/// A manual alias definition of a module.
struct ManualAlias<'a> {
    name: String,
    ty: TypeRef,
    /// Each definition makes its alias the one later lines use, so when
    /// two aliases name one type, a definition after the first is written
    /// with that alias.
    state: Option<AsmState<'a>>,
}

impl<'a> AsmState<'a> {
    /// Name every value and block `op` prints, and compute the type aliases
    /// of each module it holds.
    pub fn new(ctx: &'a IrContext, op: OpRef) -> Self {
        let mut namer = AsmNamer::new(ctx);
        namer.op(op);
        Self::with_names(ctx, namer.finish(), op, core::Module::matches(ctx, op))
    }

    /// A state that names nothing and writes every type in full, for a type
    /// or an attribute printed on its own.
    pub fn without_aliases(ctx: &'a IrContext) -> Self {
        Self {
            ctx,
            names: Rc::default(),
            aliases: HashMap::default(),
            modules: HashMap::default(),
        }
    }

    fn with_names(ctx: &'a IrContext, names: AsmNames, root: OpRef, root_is_module: bool) -> Self {
        let mut state = Self {
            ctx,
            names: Rc::new(names),
            aliases: ctx
                .type_aliases()
                .iter()
                .map(|(name, ty)| (*ty, name.to_string()))
                .collect(),
            modules: HashMap::default(),
        };
        if root_is_module {
            state.add_module(root);
        } else {
            for region in ctx.op_regions(root) {
                state.add_modules_in(region);
            }
        }
        state
    }

    /// Add the state of each module `region` holds, at any depth outside
    /// another module.
    fn add_modules_in(&mut self, region: RegionRef) {
        let ctx = self.ctx;
        let _ = walk_region::<()>(ctx, region, &mut |op| {
            if core::Module::matches(ctx, op) {
                self.add_module(op);
                ControlFlow::Continue(WalkAction::Skip)
            } else {
                ControlFlow::Continue(WalkAction::Advance)
            }
        });
    }

    /// Add the state of `module`: it defines the context's aliases again,
    /// then aliases of its own, which only its body uses.
    fn add_module(&mut self, module: OpRef) {
        let ctx = self.ctx;
        let mut manual: Vec<_> = ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (name.to_string(), *ty))
            .collect();
        manual.sort_by(|a, b| a.0.cmp(&b.0));
        topological_sort_aliases(ctx, &mut manual);
        let mut aliases = self.aliases.clone();
        let manual: Vec<_> = manual
            .into_iter()
            .map(|(name, ty)| {
                let state = (aliases != self.aliases).then(|| Self {
                    ctx,
                    names: self.names.clone(),
                    aliases: aliases.clone(),
                    modules: HashMap::default(),
                });
                aliases.insert(ty, name.clone());
                ManualAlias { name, ty, state }
            })
            .collect();
        let region = ctx.op_region(module, 0);
        let auto = region
            .map(|region| generate_auto_aliases(ctx, region, &aliases))
            .unwrap_or_default();
        for (name, ty) in &auto {
            aliases.insert(*ty, name.clone());
        }
        let mut body = Self {
            ctx,
            names: self.names.clone(),
            aliases,
            modules: HashMap::default(),
        };
        if let Some(region) = region {
            body.add_modules_in(region);
        }
        self.modules
            .insert(module, ModuleState { manual, auto, body });
    }

    /// The IR context.
    pub fn ctx(&self) -> &'a IrContext {
        self.ctx
    }

    /// A type, written as its alias when it has one in scope.
    pub fn ty(&self, ty: TypeRef) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| AsmPrinter::inline(self, f).print_type(ty))
    }

    /// A type written in full even when it has an alias, as its alias
    /// definition writes it.
    fn ty_expanded(&self, ty: TypeRef) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| AsmPrinter::inline(self, f).print_type_expanded(ty))
    }

    /// An attribute value.
    pub fn attr<'s>(&'s self, attr: &'s Attribute) -> impl fmt::Display + 's {
        fmt::from_fn(move |f| AsmPrinter::inline(self, f).print_attribute(attr))
    }

    /// `{key = value, ..}`. Like [`Itertools::format`], it can be written
    /// only once.
    pub fn attr_dict<'s>(
        &'s self,
        attrs: impl IntoIterator<Item = (StringRef, &'s Attribute)> + 's,
    ) -> impl fmt::Display + 's {
        let ctx = self.ctx;
        let entries = attrs.into_iter().format_with(", ", move |(key, value), f| {
            f(&format_args!(
                "{} = {}",
                display_attribute_key(ctx.str(key)),
                self.attr(value)
            ))
        });
        fmt::from_fn(move |f| write!(f, "{{{entries}}}"))
    }

    /// A value's name, `%N`, or `%?` for a value the printed operation does
    /// not name.
    pub fn value(&self, value: ValueRef) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| match self.names.value(value) {
            Some(number) => write!(f, "%{number}"),
            None => f.write_str("%?"),
        })
    }

    /// The names of `values`, separated by commas.
    pub fn values<'s>(&'s self, values: &'s [ValueRef]) -> impl fmt::Display + 's {
        fmt::from_fn(move |f| {
            write!(
                f,
                "{}",
                values.iter().map(|&value| self.value(value)).format(", ")
            )
        })
    }

    /// A block's label, `^bbN`, or `^bb?` for a block the printed operation
    /// does not label.
    pub fn block(&self, block: BlockRef) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| match self.names.block(block) {
            Some(number) => write!(f, "^bb{number}"),
            None => f.write_str("^bb?"),
        })
    }

    /// The entry block arguments of `region` as `(%a: A, ..)`, for a format
    /// that then prints the region with
    /// [`OpAsmPrinter::print_region_without_entry`]; [`AsmNamer::entry_args`]
    /// names them. [`parse_entry_args`](crate::asm_format::parse_entry_args)
    /// and [`attach_entry_args`](crate::asm_format::attach_entry_args) read
    /// them back.
    pub fn entry_args(&self, region: RegionRef) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| write!(f, "({})", self.typed_values(entry_args(self.ctx, region))))
    }

    /// `%a: A, %b: B`.
    fn typed_values<'s>(&'s self, values: &'s [ValueRef]) -> impl fmt::Display + 's {
        fmt::from_fn(move |f| {
            let entries = values.iter().format_with(", ", |&value, f| {
                f(&format_args!(
                    "{}: {}",
                    self.value(value),
                    self.ty(self.ctx.value_ty(value))
                ))
            });
            write!(f, "{entries}")
        })
    }
}

/// The type attributes printed in a type's attribute dictionary: all of them
/// except, when they are printed inline after each parameter, the parameter
/// attributes. Malformed parameter attributes stay in the dictionary so the
/// printed form shows them.
fn visible_type_attrs<'b>(
    ctx: &'b IrContext,
    data: &'b TypeData,
    inline_param_attrs: bool,
) -> impl Iterator<Item = (StringRef, &'b Attribute)> + Clone {
    data.attrs
        .iter()
        .filter(move |&(key, _)| !(inline_param_attrs && ctx.str(key) == PARAM_ATTRS_ATTR))
}

// ============================================================================
// AsmPrinter — writes types, attributes, and symbols
// ============================================================================

/// Writes the printed form to a [`fmt::Write`], as MLIR's `AsmPrinter`
/// does: types, attributes, and symbols, and through `write!` any
/// [`Display`](fmt::Display) of its [`AsmState`].
///
/// It owns the indentation: it writes the current indentation before the
/// first character of each line, so text written through it, `Display`
/// output included, is indented, and an empty line stays empty.
/// [`TypeAsmFormat`](crate::asm_format::TypeAsmFormat) and
/// [`AttrKind::print_text`](crate::attr_kind::AttrKind::print_text) take
/// it.
pub struct AsmPrinter<'a> {
    state: &'a AsmState<'a>,
    out: &'a mut (dyn Write + 'a),
    /// The indentation of the current line.
    indent: usize,
    /// Whether nothing is written on the current line yet, so the next
    /// character writes the indentation first.
    at_line_start: bool,
}

impl<'a> AsmPrinter<'a> {
    /// A printer of `state` that writes to `out`, at the start of a line.
    pub fn new(state: &'a AsmState<'a>, out: &'a mut (dyn Write + 'a)) -> Self {
        Self {
            state,
            out,
            indent: 0,
            at_line_start: true,
        }
    }

    /// A printer inside a line another printer indents, for a `Display`
    /// value written through it.
    fn inline(state: &'a AsmState<'a>, out: &'a mut (dyn Write + 'a)) -> Self {
        Self {
            state,
            out,
            indent: 0,
            at_line_start: false,
        }
    }

    /// The state the printer reads, for its `Display` values:
    /// `let s = p.state(); write!(p, "{}", s.ty(ty))`.
    pub fn state(&self) -> &'a AsmState<'a> {
        self.state
    }

    /// The IR context.
    pub fn ctx(&self) -> &'a IrContext {
        self.state.ctx
    }

    /// Write a type, as its alias when it has one in scope.
    pub fn print_type(&mut self, ty: TypeRef) -> fmt::Result {
        if let Some(alias) = self.state.aliases.get(&ty) {
            return write!(self, "{}", display_alias(alias));
        }
        self.print_type_expanded(ty)
    }

    fn print_type_expanded(&mut self, ty: TypeRef) -> fmt::Result {
        let state = self.state;
        let ctx = state.ctx;
        let data = ctx.get_type(ty);
        if let Some(hooks) = data.name.def().and_then(|def| def.asm_format)
            && let Some(result) = (hooks.print)(self, ty)
        {
            return result;
        }
        write!(self, "{}", data.name.display(ctx))?;
        let inline_param_attrs = data.validate_param_attrs(ctx).is_ok();
        let attrs = visible_type_attrs(ctx, data, inline_param_attrs);
        let has_attrs = attrs.clone().next().is_some();
        if data.params.is_empty() && !has_attrs {
            return Ok(());
        }
        let params = data
            .params
            .iter()
            .enumerate()
            .format_with(", ", |(index, &param), f| {
                let param_attrs = data.param_attrs(ctx, index);
                if inline_param_attrs && !param_attrs.is_empty() {
                    f(&format_args!(
                        "{} {}",
                        state.ty(param),
                        state.attr_dict(param_attrs.iter())
                    ))
                } else {
                    f(&state.ty(param))
                }
            });
        write!(self, "<{params}")?;
        if has_attrs {
            if !data.params.is_empty() {
                self.write_str(", ")?;
            }
            write!(self, "{}", state.attr_dict(attrs))?;
        }
        self.write_char('>')
    }

    /// Write an attribute value.
    pub fn print_attribute(&mut self, attr: &Attribute) -> fmt::Result {
        let state = self.state;
        match attr {
            Attribute::Unit => self.write_str("unit"),
            Attribute::Bool(b) => write!(self, "{b}"),
            Attribute::Int(v) => write!(self, "{v}"),
            Attribute::FloatBits(bits) => {
                let v = f64::from_bits(*bits);
                let s = format!("{v}");
                self.write_str(&s)?;
                if v.is_finite() && !s.contains('.') && !s.contains('e') && !s.contains('E') {
                    self.write_str(".0")?;
                }
                Ok(())
            }
            Attribute::String(s) => {
                self.write_char('"')?;
                write_escaped_string(self, state.ctx.str(*s))?;
                self.write_char('"')
            }
            Attribute::Bytes(bytes) => {
                self.write_str("b\"")?;
                write_escaped_bytes(self, bytes)?;
                self.write_char('"')
            }
            Attribute::SymbolRef(path) => write_symbol_path(self, path),
            Attribute::Type(ty) => self.print_type(*ty),
            Attribute::List(list) => write!(
                self,
                "[{}]",
                list.iter().map(|item| state.attr(item)).format(", ")
            ),
            Attribute::Dict(dict) => write!(self, "{}", state.attr_dict(dict.iter())),
            Attribute::Location(loc) => {
                let path_str = state.ctx.paths().get(loc.path);
                self.write_str("loc(\"")?;
                write_escaped_string(self, path_str)?;
                write!(self, "\" {}:{})", loc.span.start, loc.span.end)
            }
        }
    }
}

impl Write for AsmPrinter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for (index, line) in s.split('\n').enumerate() {
            if index > 0 {
                self.out.write_char('\n')?;
                self.at_line_start = true;
            }
            if line.is_empty() {
                continue;
            }
            if self.at_line_start {
                for _ in 0..self.indent {
                    self.out.write_char(' ')?;
                }
                self.at_line_start = false;
            }
            self.out.write_str(line)?;
        }
        Ok(())
    }
}

// ============================================================================
// OpAsmPrinter — writes operations
// ============================================================================

/// Writes operations, as MLIR's `OpAsmPrinter` does: what an
/// [`AsmPrinter`] writes, which it dereferences to, and the structure of
/// operations, such as regions and new lines. Value names come from its
/// [`AsmState`]: `write!(p, "{}", p.state().value(v))`.
///
/// [`OpAsmFormat`](crate::asm_format::OpAsmFormat) and
/// [`OpAsmDirective`](crate::asm_format::OpAsmDirective) take it.
pub struct OpAsmPrinter<'a> {
    printer: AsmPrinter<'a>,
}

impl<'a> Deref for OpAsmPrinter<'a> {
    type Target = AsmPrinter<'a>;

    fn deref(&self) -> &AsmPrinter<'a> {
        &self.printer
    }
}

impl DerefMut for OpAsmPrinter<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.printer
    }
}

impl Write for OpAsmPrinter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.printer.write_str(s)
    }
}

impl<'a> OpAsmPrinter<'a> {
    /// A printer of `state` that writes to `out`, at the start of a line.
    pub fn new(state: &'a AsmState<'a>, out: &'a mut (dyn Write + 'a)) -> Self {
        Self {
            printer: AsmPrinter::new(state, out),
        }
    }

    /// Print `op` on its own line, at the current indentation, followed by
    /// a newline.
    pub fn print_operation(&mut self, op: OpRef) -> fmt::Result {
        let ctx = self.ctx();
        if core::Module::matches(ctx, op) {
            return self.print_module_op(op);
        }
        let def = crate::OpDef::of(ctx, op);
        match def.and_then(|def| Some((def, def.asm_format?))) {
            Some((_, crate::asm_format::OpAsmFormatDef::Hooks(hooks))) => (hooks.print)(self, op)?,
            Some((def, crate::asm_format::OpAsmFormatDef::Declarative(format))) => {
                crate::asm_format::declarative::print(self, op, def, format)?
            }
            None => self.print_generic(op)?,
        }
        self.write_char('\n')
    }

    /// Print the operation with generic assembly, after its line's
    /// indentation and before its newline.
    ///
    /// Custom formats use this when their concise form cannot represent the
    /// complete operation without loss.
    pub fn print_generic(&mut self, op: OpRef) -> fmt::Result {
        let state = self.state();
        let ctx = state.ctx;

        let results = ctx.op_results(op);
        if !results.is_empty() {
            write!(self, "{} = ", state.values(results))?;
        }
        write!(
            self,
            "\"{}\"({})",
            ctx.op(op).name.display(ctx),
            state.values(ctx.op_operands(op))
        )?;

        let successors = ctx.op_successors(op);
        if successors.len() != 0 {
            write!(
                self,
                "[{}]",
                successors.map(|block| state.block(block)).format(", ")
            )?;
        }

        // Regions, before the attributes so that a brace there is always the
        // dictionary
        if ctx.op_has_regions(op) {
            self.write_str(" (")?;
            for (index, region) in ctx.op_regions(op).enumerate() {
                if index > 0 {
                    self.write_str(", ")?;
                }
                self.print_region(region)?;
            }
            self.write_char(')')?;
        }

        // Attributes, without the declared defaults the canonical form leaves
        // absent
        let schema = crate::op_def::OpDef::of(ctx, op).map(|def| &def.schema);
        let mut attrs = ctx
            .op(op)
            .attributes
            .iter()
            .filter(|&(key, val)| {
                !schema.is_some_and(|schema| schema.holds_default(ctx, ctx.str(key), val))
            })
            .peekable();
        if attrs.peek().is_some() {
            let entries = attrs.format_with(", ", |(key, val), f| {
                f(&format_args!("{} = {}", ctx.str(key), state.attr(val)))
            });
            write!(self, " {{{entries}}}")?;
        }

        let result_types = ctx.op_result_types(op);
        if !result_types.is_empty() {
            write!(
                self,
                " : {}",
                result_types.iter().map(|&ty| state.ty(ty)).format(", ")
            )?;
        }
        Ok(())
    }

    /// Print `{ .. }` holding a region in generic form, block labels
    /// included.
    pub fn print_region(&mut self, region: RegionRef) -> fmt::Result {
        self.write_str("{\n")?;
        self.print_blocks(region, false)?;
        self.write_char('}')
    }

    /// Print `{ .. }` holding a region whose entry block arguments the
    /// operation's signature wrote, eliding the entry block as
    /// [`Self::print_body`] does.
    pub fn print_region_without_entry(&mut self, region: RegionRef) -> fmt::Result {
        self.write_str("{\n")?;
        self.print_blocks(region, true)?;
        self.write_char('}')
    }

    /// Print ` { .. }` holding a region with the entry block label elided.
    ///
    /// Entry block args are assumed to have been printed externally (e.g., as
    /// function parameters in the operation signature), so the entry label is
    /// omitted, and the other blocks are printed normally.
    pub fn print_body(&mut self, region: RegionRef) -> fmt::Result {
        self.write_char(' ')?;
        self.print_region_without_entry(region)
    }

    /// Print ` {`, a `label { .. }` line for each region one level in, and
    /// `}`, for an operation whose regions are named. Each region's entry
    /// block label is elided as [`Self::print_body`] elides it.
    pub fn print_labelled_bodies<'r>(
        &mut self,
        bodies: impl IntoIterator<Item = (&'r str, RegionRef)>,
    ) -> fmt::Result {
        self.write_str(" {\n")?;
        self.indent += 2;
        for (label, region) in bodies {
            writeln!(self, "{label} {{")?;
            self.print_blocks(region, true)?;
            self.write_str("}\n")?;
        }
        self.indent -= 2;
        self.write_char('}')
    }

    /// Start a new line at the operation's indentation, as MLIR's
    /// `printNewline` does.
    pub fn newline(&mut self) -> fmt::Result {
        self.write_char('\n')
    }

    /// Print the blocks of `region` one level in, and their operations two.
    /// With `entry_args_written`, the operation wrote the entry block
    /// arguments, so the entry block writes none and elides its label when
    /// it can; otherwise it elides its label only without arguments, and a
    /// blank line separates the blocks.
    fn print_blocks(&mut self, region: RegionRef, entry_args_written: bool) -> fmt::Result {
        let state = self.state();
        let ctx = state.ctx;
        let blocks = &ctx.region(region).blocks;
        let elide_entry = blocks.first().is_some_and(|&entry| {
            (entry_args_written || ctx.block_args(entry).is_empty())
                && can_elide_entry_label(ctx, blocks, entry)
        });

        self.indent += 2;
        for (index, &block) in blocks.iter().enumerate() {
            if !(index == 0 && elide_entry) {
                write!(self, "{}", state.block(block))?;
                let args = ctx.block_args(block);
                if !(index == 0 && entry_args_written) && !args.is_empty() {
                    write!(self, "({})", state.typed_values(args))?;
                }
                self.write_str(":\n")?;
            }
            self.indent += 2;
            for &op in &ctx.block(block).ops {
                self.print_operation(op)?;
            }
            self.indent -= 2;
            if !entry_args_written && index + 1 < blocks.len() {
                // A blank line between blocks for readability
                self.write_char('\n')?;
            }
        }
        self.indent -= 2;
        Ok(())
    }

    fn print_module_op(&mut self, op: OpRef) -> fmt::Result {
        let state = self.state();
        let ctx = state.ctx;
        self.write_str("core.module")?;

        if let Some(name) = ctx.op_attrs(op).get_str("sym_name") {
            write!(self, " {}", display_symbol_text(name))?;
        }

        // Type aliases are declared inside the body, so these attributes spell
        // their types out.
        let mut attrs = ctx
            .op(op)
            .attributes
            .iter()
            .filter(|&(key, _)| ctx.str(key) != "sym_name")
            .peekable();
        if attrs.peek().is_some() {
            let expanded = AsmState::without_aliases(ctx);
            write!(self, " {}", expanded.attr_dict(attrs))?;
        }

        let regions = ctx.op_region_count(op);
        assert!(
            regions <= 1,
            "print_module_op: expected at most one region, found {regions}",
        );
        let Some(region) = ctx.op_region(op, 0) else {
            return self.write_char('\n');
        };
        let module = state
            .modules
            .get(&op)
            .expect("the naming pre-pass visited every printed module");
        self.write_str(" {\n")?;
        self.indent += 2;
        for alias in &module.manual {
            let scope = alias.state.as_ref().unwrap_or(state);
            writeln!(
                self,
                "{} = {}",
                display_alias(&alias.name),
                scope.ty_expanded(alias.ty)
            )?;
        }
        for (name, ty) in &module.auto {
            writeln!(
                self,
                "{} = {}",
                display_alias(name),
                module.body.ty_expanded(*ty)
            )?;
        }
        if !module.manual.is_empty() || !module.auto.is_empty() {
            // A blank line after the alias definitions
            self.write_char('\n')?;
        }

        let mut body = OpAsmPrinter {
            printer: AsmPrinter {
                state: &module.body,
                out: &mut *self.printer.out,
                indent: self.printer.indent,
                at_line_start: self.printer.at_line_start,
            },
        };
        for &block in &ctx.region(region).blocks {
            for &child in &ctx.block(block).ops {
                body.print_operation(child)?;
            }
        }
        let at_line_start = body.printer.at_line_start;
        self.printer.at_line_start = at_line_start;
        self.indent -= 2;
        self.write_str("}\n")
    }
}

// ============================================================================
// Public API
// ============================================================================

impl OpRef {
    /// The operation as IR text, on its own line, as
    /// [`Path::display`](std::path::Path::display) writes a path: a module
    /// with its type aliases, anything else as its format writes it. Each
    /// write runs the naming pre-pass for it.
    pub fn display(self, ctx: &IrContext) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| {
            let state = AsmState::new(ctx, self);
            OpAsmPrinter::new(&state, f).print_operation(self)
        })
    }
}

impl TypeRef {
    /// The type as IR text, written in full without aliases.
    pub fn display(self, ctx: &IrContext) -> impl fmt::Display + '_ {
        fmt::from_fn(move |f| write!(f, "{}", AsmState::without_aliases(ctx).ty(self)))
    }
}

#[cfg(test)]
fn write_attribute(ctx: &IrContext, f: &mut impl Write, attr: &Attribute) -> fmt::Result {
    write!(f, "{}", AsmState::without_aliases(ctx).attr(attr))
}

pub(crate) fn write_escaped_bytes(f: &mut dyn Write, bytes: &[u8]) -> fmt::Result {
    for &b in bytes {
        match b {
            b'\\' => f.write_str("\\\\")?,
            b'"' => f.write_str("\\\"")?,
            b'\n' => f.write_str("\\n")?,
            b'\t' => f.write_str("\\t")?,
            b'\r' => f.write_str("\\r")?,
            b'\0' => f.write_str("\\0")?,
            0x20..=0x7e => f.write_char(b as char)?,
            _ => write!(f, "\\x{b:02x}")?,
        }
    }
    Ok(())
}

pub(crate) fn write_escaped_string(f: &mut dyn Write, s: &str) -> fmt::Result {
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

/// A type alias name with its `!` prefix, quoted when it is not an
/// identifier.
fn display_alias(name: &str) -> impl fmt::Display + '_ {
    fmt::from_fn(move |f| {
        let needs_quoting =
            name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if needs_quoting {
            f.write_str("!\"")?;
            write_escaped_string(f, name)?;
            f.write_char('"')
        } else {
            write!(f, "!{name}")
        }
    })
}

/// A dictionary key, bare when the reader accepts it, otherwise quoted.
fn display_attribute_key(key: &str) -> impl fmt::Display + '_ {
    fmt::from_fn(move |f| {
        if crate::parser::raw::is_bare_attribute_key(key) {
            f.write_str(key)
        } else {
            f.write_char('"')?;
            write_escaped_string(f, key)?;
            f.write_char('"')
        }
    })
}

/// `name` bare when it is an identifier (`[A-Za-z_][A-Za-z0-9_]*`), and as
/// a quoted string otherwise.
pub fn display_name(name: &str) -> impl fmt::Display + '_ {
    fmt::from_fn(move |f| {
        let mut chars = name.chars();
        let is_ident = chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
        if is_ident {
            f.write_str(name)
        } else {
            f.write_char('"')?;
            write_escaped_string(f, name)?;
            f.write_char('"')
        }
    })
}

/// Write a symbol path as `@outer::@name`, each component quoted when
/// needed.
fn write_symbol_path(f: &mut dyn Write, path: &crate::symbol::SymbolPath) -> fmt::Result {
    for (index, component) in path.iter().enumerate() {
        if index != 0 {
            f.write_str(crate::symbol::SymbolPath::SEPARATOR)?;
        }
        write_symbol_text(f, component.as_str())?;
    }
    Ok(())
}

/// A symbol reference as the printer writes it, `@outer::@name`, for a
/// format that writes it among other text.
pub fn display_symbol_ref(path: &crate::symbol::SymbolPath) -> impl fmt::Display + '_ {
    fmt::from_fn(move |f| write_symbol_path(f, path))
}

/// `@name` for symbol text as the printer writes it, quoted when needed.
pub fn display_symbol_text(name: &str) -> impl fmt::Display + '_ {
    fmt::from_fn(move |f| write_symbol_text(f, name))
}

/// Write `@name` for symbol text, quoting it when needed.
fn write_symbol_text(f: &mut dyn Write, s: &str) -> fmt::Result {
    let needs_quoting = s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if needs_quoting {
        f.write_str("@\"")?;
        write_escaped_string(f, s)?;
        f.write_char('"')
    } else {
        write!(f, "@{s}")
    }
}

// ============================================================================
// Auto alias generation
// ============================================================================

/// Minimum character complexity for a type to be alias-eligible.
const MIN_ALIAS_COMPLEXITY: usize = 40;

/// Minimum use count for a type to be alias-eligible.
const MIN_ALIAS_USES: usize = 2;

/// Collect all TypeRefs used in a module region and count occurrences.
///
/// Only counts direct usage sites (result types, block args, attributes).
/// Does not recurse into type params — nested types become aliased naturally
/// when their parent is aliased.
fn collect_module_types(ctx: &IrContext, region: RegionRef) -> HashMap<TypeRef, usize> {
    let mut counts: HashMap<TypeRef, usize> = HashMap::default();

    let _ = walk_region::<()>(ctx, region, &mut |op| {
        let data = ctx.op(op);

        // Skip nested modules — they collect their own aliases.
        if crate::dialect::core::Module::matches(ctx, op) {
            return ControlFlow::Continue(WalkAction::Skip);
        }

        // Result types
        for &ty in ctx.op_result_types(op) {
            *counts.entry(ty).or_default() += 1;
        }

        // Attributes containing types, except a signature its format writes
        // in the function form, where an alias would go unused.
        let signature_attrs = crate::op_def::OpDef::of(ctx, op)
            .map(crate::asm_format::declarative::signature_attributes)
            .unwrap_or_default();
        for (key, attr) in data.attributes.iter() {
            if !signature_attrs.contains(&ctx.str(key)) {
                count_attr_types(&mut counts, attr);
            }
        }

        // Block args in regions
        for region in ctx.op_regions(op) {
            for &block in &ctx.region(region).blocks {
                for arg in &ctx.block(block).args {
                    *counts.entry(arg.ty).or_default() += 1;
                }
            }
        }

        ControlFlow::Continue(WalkAction::Advance)
    });

    counts
}

fn count_attr_types(counts: &mut HashMap<TypeRef, usize>, attr: &Attribute) {
    attr.visit_types(&mut |ty| *counts.entry(ty).or_default() += 1);
}

/// Generate auto aliases for types that are used frequently and are complex enough.
fn generate_auto_aliases(
    ctx: &IrContext,
    region: RegionRef,
    existing: &HashMap<TypeRef, String>,
) -> Vec<(String, TypeRef)> {
    let counts = collect_module_types(ctx, region);
    let mut candidates: Vec<(TypeRef, usize, usize, String)> = Vec::new();

    for (&ty, &count) in &counts {
        if existing.contains_key(&ty) {
            continue;
        }
        let complexity = ctx.types().complexity(ty, ctx.strings());
        let has_hint = crate::asm_format::suggest_type_alias_name(ctx, ty).is_some();
        // Types with a dialect-provided name hint (e.g. named structs) are
        // alias-eligible when used often enough. Types without a hint need
        // sufficient complexity to justify a fallback name like t0, t1.
        let eligible = if has_hint {
            count >= MIN_ALIAS_USES || complexity >= MIN_ALIAS_COMPLEXITY
        } else {
            count >= MIN_ALIAS_USES && complexity >= MIN_ALIAS_COMPLEXITY
        };
        if eligible {
            // `TypeRef` is an allocation detail. The alias-free type printer
            // gives equivalent types the same stable structural ordering.
            candidates.push((ty, count, complexity, ty.display(ctx).to_string()));
        }
    }

    // Sort by (count desc, complexity desc, structural type key asc).
    // This order also makes name-conflict assignment and the stable
    // topological extraction deterministic for independent candidates.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.3.cmp(&b.3)));

    // Name assignment
    let mut used_names: HashSet<String> = existing.values().cloned().collect();
    let mut next_num = 0usize;
    let mut result = Vec::new();

    for (ty, _, _, _) in &candidates {
        let name = choose_alias_name(ctx, *ty, &used_names, &mut next_num);
        used_names.insert(name.clone());
        result.push((name, *ty));
    }

    // Topological sort: if type A references type B, B must come first
    topological_sort_aliases(ctx, &mut result);

    result
}

/// Choose an alias name for a type.
///
/// 1. Try dialect-provided hint (e.g., the `name` attribute of a named type)
/// 2. On conflict, add suffix: `Point`, `Point_1`, `Point_2`, ...
/// 3. Fallback: `t0`, `t1`, `t2`, ...
fn choose_alias_name(
    ctx: &IrContext,
    ty: TypeRef,
    used_names: &HashSet<String>,
    next_num: &mut usize,
) -> String {
    if let Some(base) = crate::asm_format::suggest_type_alias_name(ctx, ty) {
        let base = base.to_owned();
        if !used_names.contains(&base) {
            return base;
        }
        // Try with suffix
        for i in 1.. {
            let candidate = format!("{base}_{i}");
            if !used_names.contains(&candidate) {
                return candidate;
            }
        }
    }
    // Fallback: t0, t1, ...
    loop {
        let name = format!("t{next_num}");
        *next_num += 1;
        if !used_names.contains(&name) {
            return name;
        }
    }
}

/// Topological sort: ensure that if type A references type B, B's alias appears first.
fn topological_sort_aliases(ctx: &IrContext, aliases: &mut Vec<(String, TypeRef)>) {
    // Build a set of aliased types for quick lookup
    let alias_set: HashSet<TypeRef> = aliases.iter().map(|(_, ty)| *ty).collect();

    // For each alias, compute the set of aliased types it depends on
    let deps: Vec<HashSet<TypeRef>> = aliases
        .iter()
        .map(|(_, ty)| {
            let mut deps = HashSet::default();
            collect_type_deps(ctx, *ty, &alias_set, &mut deps);
            deps
        })
        .collect();

    // Simple stable topological sort via repeated extraction of dependency-free items
    let n = aliases.len();
    let mut sorted: Vec<(String, TypeRef)> = Vec::with_capacity(n);
    let mut placed: HashSet<TypeRef> = HashSet::default();
    let mut remaining: Vec<bool> = vec![true; n];

    for _ in 0..n {
        for i in 0..n {
            if !remaining[i] {
                continue;
            }
            // Check if all deps are placed
            if deps[i].iter().all(|d| placed.contains(d)) {
                remaining[i] = false;
                placed.insert(aliases[i].1);
                sorted.push(aliases[i].clone());
                break;
            }
        }
    }

    // If we placed everything, use sorted order; otherwise keep original (cycle)
    if sorted.len() == n {
        *aliases = sorted;
    }
}

/// Collect all TypeRefs within `ty` that are in `alias_set` (direct type params only).
fn collect_type_deps(
    ctx: &IrContext,
    ty: TypeRef,
    alias_set: &HashSet<TypeRef>,
    deps: &mut HashSet<TypeRef>,
) {
    let data = ctx.get_type(ty);
    for &param in &data.params {
        if alias_set.contains(&param) {
            deps.insert(param);
        }
        collect_type_deps(ctx, param, alias_set, deps);
    }
    // Also check types embedded in attributes
    for attr in data.attrs.values() {
        collect_attr_type_deps(ctx, attr, alias_set, deps);
    }
}

fn collect_attr_type_deps(
    ctx: &IrContext,
    attr: &Attribute,
    alias_set: &HashSet<TypeRef>,
    deps: &mut HashSet<TypeRef>,
) {
    attr.visit_types(&mut |ty| {
        if alias_set.contains(&ty) {
            deps.insert(ty);
        }
        collect_type_deps(ctx, ty, alias_set, deps);
    });
}

// ============================================================================
// Region / Block printing
// ============================================================================

/// Whether `entry` reads back without its label: it has operations, or the
/// region would read back without it, and no branch in the region names it.
fn can_elide_entry_label(ctx: &IrContext, blocks: &[BlockRef], entry: BlockRef) -> bool {
    !ctx.block(entry).ops.is_empty()
        && !blocks.iter().any(|&block| {
            ctx.block(block)
                .ops
                .iter()
                .any(|&op| ctx.op_successors(op).any(|successor| successor == entry))
        })
}

mod names;
#[cfg(test)]
mod prop_tests;

use names::AsmNames;
pub use names::{AsmNamer, entry_args};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Symbol;
    use crate::SymbolPath;
    use crate::dialect::{arith, core, func};
    use crate::{BlockArgData, BlockData, RegionData, TypeDataBuilder};
    use smallvec::smallvec;

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.intern_path("test.trb");
        Location::new(path, crate::Span::new(0, 0))
    }

    fn make_i32_type(ctx: &mut IrContext) -> TypeRef {
        TypeDataBuilder::new("core.i32").intern(ctx)
    }

    /// Create a one-result `func.func_sig` type.
    fn make_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
        crate::dialect::func::func_sig(ctx, params.iter().copied(), [ret]).as_type_ref()
    }

    #[test]
    fn printer_indents_each_line_and_leaves_empty_lines_empty() {
        let ctx = IrContext::new();
        let state = AsmState::without_aliases(&ctx);
        let mut out = String::new();
        let mut p = AsmPrinter::new(&state, &mut out);
        p.indent = 4;
        let nested = fmt::from_fn(|f| f.write_str("b\nc"));
        write!(p, "a\n\n{nested}\n").unwrap();
        assert_eq!(out, "    a\n\n    b\n    c\n");
    }

    #[test]
    fn test_print_type_simple() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        assert_eq!(i32_ty.display(&ctx).to_string(), "core.i32");
    }

    #[test]
    fn test_print_type_with_params() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let tuple_ty = crate::dialect::core::Tuple::params([i32_ty, i32_ty])
            .intern(&mut ctx)
            .as_type_ref();
        assert_eq!(
            tuple_ty.display(&ctx).to_string(),
            "core.tuple<core.i32, core.i32>"
        );
    }

    #[test]
    fn standalone_function_printing_expands_aliases_in_lists_and_attributes() {
        for result in ["()", "!scalar"] {
            let mut ctx = IrContext::new();
            let input = format!(
                "core.module @m {{
                !scalar = core.i32
                !callable = func.func_sig<(!scalar) -> {result}, {{nested = [!scalar]}}>
                func.func @f(%x: !callable) {{ func.return }}
            }}"
            );
            let module = crate::parser::parse_module(&mut ctx, &input).unwrap();
            let callable = ctx.type_alias_by_text("callable").unwrap();
            let expanded_result = if result == "()" { "()" } else { "core.i32" };
            assert_eq!(
                callable.display(&ctx).to_string(),
                format!("func.func_sig<(core.i32) -> {expanded_result}, {{nested = [core.i32]}}>")
            );
            let printed = module.display(&ctx).to_string();
            assert!(
                printed.contains(&format!(
                    "func.func_sig<(!scalar) -> {result}, {{nested = [!scalar]}}>"
                )),
                "{printed}"
            );
            let mut reparsed = IrContext::new();
            let module = crate::parser::parse_module(&mut reparsed, &printed).unwrap();
            assert_eq!(module.display(&reparsed).to_string(), printed);
        }
    }

    #[test]
    fn test_print_func_type_uses_canonical_result_list_syntax() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let zero_zero = func::func_sig(&mut ctx, [], []).as_type_ref();
        let many_zero = func::func_sig(&mut ctx, [i32_ty, i32_ty], []).as_type_ref();
        let zero_one = func::func_sig(&mut ctx, [], [i32_ty]).as_type_ref();
        let many_one = func::func_sig(&mut ctx, [i32_ty, i32_ty], [i32_ty]).as_type_ref();

        assert_eq!(
            zero_zero.display(&ctx).to_string(),
            "func.func_sig<() -> ()>"
        );
        assert_eq!(
            many_zero.display(&ctx).to_string(),
            "func.func_sig<(core.i32, core.i32) -> ()>"
        );
        assert_eq!(
            zero_one.display(&ctx).to_string(),
            "func.func_sig<() -> core.i32>"
        );
        assert_eq!(
            many_one.display(&ctx).to_string(),
            "func.func_sig<(core.i32, core.i32) -> core.i32>"
        );
    }

    #[test]
    fn malformed_func_sig_storage_prints_as_concrete_type() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let foreign = TypeDataBuilder::new("foreign.func_sig")
            .param(i32_ty)
            .attr(func::NUM_INPUTS_ATTR, Attribute::Int(2))
            .attr(func::NUM_RESULTS_ATTR, Attribute::Int(1))
            .intern(&mut ctx);
        assert_eq!(
            foreign.display(&ctx).to_string(),
            "foreign.func_sig<core.i32, {num_inputs = 2, num_results = 1}>"
        );

        // Shared signatures have a stricter one-result contract, so raw
        // multi-result storage must not be printed in arrow syntax either.
        let shared = TypeDataBuilder::new("func.func_sig")
            .params([i32_ty, i32_ty, i32_ty])
            .attr(func::NUM_INPUTS_ATTR, Attribute::Int(1))
            .attr(func::NUM_RESULTS_ATTR, Attribute::Int(2))
            .intern(&mut ctx);
        assert_eq!(
            shared.display(&ctx).to_string(),
            "func.func_sig<core.i32, core.i32, core.i32, {num_inputs = 1, num_results = 2}>"
        );
    }

    #[test]
    fn test_print_simple_op() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let c = arith::Constant::operands()
            .value(Attribute::Int(42))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let output = c.op_ref().display(&ctx).to_string();
        assert_eq!(output, "%0 = arith.constant 42 : core.i32\n");
    }

    #[test]
    fn test_print_binary_op() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let c1 = arith::Constant::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let c2 = arith::Constant::operands()
            .value(Attribute::Int(2))
            .results(i32_ty)
            .build(&mut ctx, loc);
        let v1 = c1.result(&ctx);
        let v2 = c2.result(&ctx);

        let add = arith::Addi::operands(v1, v2).build(&mut ctx, loc);
        // Printer assigns names based on what it sees - standalone op print
        // only numbers the result of THIS op since it can't see c1/c2
        let output = add.op_ref().display(&ctx).to_string();
        // %0 = arith.addi %?, %? : core.i32 (operands unknown since not in scope)
        assert!(output.contains("arith.addi"));
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
        let add = arith::Addi::operands(x, y).build(&mut ctx, loc);
        ctx.push_op(entry_block, add.op_ref());

        // return result
        let result = add.result(&ctx);
        let ret = func::Return::operands([result]).build(&mut ctx, loc);
        ctx.push_op(entry_block, ret.op_ref());

        // Region
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry_block],
            parent_op: None,
        });

        // Function
        let f = func::Func::operands()
            .sym_name("add")
            .r#type(func_ty)
            .regions(body)
            .build(&mut ctx, loc);

        let output = f.op_ref().display(&ctx).to_string();
        assert!(output.contains("func.func @add"));
        assert!(output.contains("%0: core.i32"));
        assert!(output.contains("%1: core.i32"));
        assert!(output.contains("arith.addi %0, %1"));
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

        let c = arith::Constant::operands()
            .value(Attribute::Int(42))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(entry, c.op_ref());

        let result = c.result(&ctx);
        let ret = func::Return::operands([result]).build(&mut ctx, loc);
        ctx.push_op(entry, ret.op_ref());

        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let f = func::Func::operands()
            .sym_name("main")
            .r#type(func_ty)
            .regions(body)
            .build(&mut ctx, loc);

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
        let module = core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(&mut ctx, loc);

        let output = module.op_ref().display(&ctx).to_string();
        assert!(output.contains("core.module @test"));
        assert!(output.contains("func.func @main"));
        assert!(output.contains("arith.constant 42"));
        assert!(output.contains("func.return %0"));
    }

    #[test]
    fn test_print_nested_module() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);
        let func_ty = make_func_type(&mut ctx, &[i32_ty], i32_ty);

        // Inner function: fn get_x(%0: i32) -> i32 { return %0; }
        let inner_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let x = ctx.block_arg(inner_entry, 0);
        let ret_inner = func::Return::operands([x]).build(&mut ctx, loc);
        ctx.push_op(inner_entry, ret_inner.op_ref());

        let inner_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![inner_entry],
            parent_op: None,
        });
        let inner_func = func::Func::operands()
            .sym_name("get_x")
            .r#type(func_ty)
            .regions(inner_body)
            .build(&mut ctx, loc);

        // Inner module: core.module @Point { func.func @get_x ... }
        let inner_mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        ctx.push_op(inner_mod_block, inner_func.op_ref());

        let inner_mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![inner_mod_block],
            parent_op: None,
        });
        let inner_module = core::Module::operands()
            .sym_name("Point")
            .regions(inner_mod_region)
            .build(&mut ctx, loc);

        // Outer function: fn make() -> i32 { return 1; }
        let outer_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let one = arith::Constant::operands()
            .value(Attribute::Int(1))
            .results(i32_ty)
            .build(&mut ctx, loc);
        ctx.push_op(outer_entry, one.op_ref());
        let one_val = one.result(&ctx);
        let ret_outer = func::Return::operands([one_val]).build(&mut ctx, loc);
        ctx.push_op(outer_entry, ret_outer.op_ref());

        let outer_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![outer_entry],
            parent_op: None,
        });
        let make_func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let outer_func = func::Func::operands()
            .sym_name("make")
            .r#type(make_func_ty)
            .regions(outer_body)
            .build(&mut ctx, loc);

        // Outer module: core.module @test { core.module @Point { ... }  func.func @make ... }
        let outer_mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        ctx.push_op(outer_mod_block, inner_module.op_ref());
        ctx.push_op(outer_mod_block, outer_func.op_ref());

        let outer_mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![outer_mod_block],
            parent_op: None,
        });
        let outer_module = core::Module::operands()
            .sym_name("test")
            .regions(outer_mod_region)
            .build(&mut ctx, loc);

        let output = outer_module.op_ref().display(&ctx).to_string();
        insta::assert_snapshot!(output);
    }

    #[test]
    fn test_print_attribute_types() {
        let mut ctx = IrContext::new();
        let hello = ctx.string_attr("hello\nworld");
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
        write_attribute(&ctx, &mut out, &hello).unwrap();
        assert_eq!(out, r#""hello\nworld""#);

        // Symbol
        out.clear();
        write_attribute(
            &ctx,
            &mut out,
            &Attribute::SymbolRef(SymbolPath::new(["foo"])),
        )
        .unwrap();
        assert_eq!(out, "@foo");

        // Symbol with path (needs quoting)
        out.clear();
        write_attribute(
            &ctx,
            &mut out,
            &Attribute::SymbolRef(SymbolPath::new(["std::List::map"])),
        )
        .unwrap();
        assert_eq!(out, r#"@"std::List::map""#);

        // Nested modules: one `@name` per component
        out.clear();
        let nested = SymbolPath::new(["outer", "odd::name", "leaf"]);
        write_attribute(&ctx, &mut out, &Attribute::SymbolRef(nested.clone())).unwrap();
        assert_eq!(out, r#"@outer::@"odd::name"::@leaf"#);
        let mut input = out.as_str();
        let parsed = crate::parser::raw::symbol_path(&mut input).expect("a symbol path");
        assert!(input.is_empty());
        assert!(
            parsed
                .iter()
                .map(String::as_str)
                .eq(nested.iter().map(Symbol::as_str))
        );

        // Empty symbol (should quote)
        assert_eq!(
            display_symbol_text(Symbol::new("").as_str()).to_string(),
            r#"@"""#
        );

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

    // ====================================================================
    // Auto alias tests
    // ====================================================================

    #[crate::dialect]
    mod alias_test {
        #[alias_hint]
        struct Record<#[rest] Fields> {
            name: Attr<String>,
        }
    }

    impl crate::asm_format::TypeAliasHint for Record {
        fn alias_hint<'a>(&self, ctx: &'a IrContext) -> Option<&'a str> {
            Some(self.name(ctx))
        }
    }

    /// Helper: build a named `alias_test.record` type, which suggests its
    /// `name` as an alias, with the given named fields.
    fn make_named_type(ctx: &mut IrContext, name: &str, fields: &[(&str, TypeRef)]) -> TypeRef {
        let mut builder = TypeDataBuilder::new("alias_test.record");
        for (field, ty) in fields {
            let mut attrs = AttributeMap::new();
            let field = ctx.string_attr(field);
            attrs.view_mut(ctx).insert("name", field);
            builder = builder.param_with_attrs(*ty, attrs);
        }
        let name = ctx.string_attr(name);
        builder.attr("name", name).intern(ctx)
    }

    /// Helper: build a module with given functions.
    fn make_module_with_funcs(ctx: &mut IrContext, loc: Location, funcs: Vec<OpRef>) -> OpRef {
        let mod_block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        for func_op in funcs {
            ctx.push_op(mod_block, func_op);
        }
        let mod_region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![mod_block],
            parent_op: None,
        });
        core::Module::operands()
            .sym_name("test")
            .regions(mod_region)
            .build(ctx, loc)
            .op_ref()
    }

    /// Helper: build a function that takes a param and returns it.
    fn make_identity_func(
        ctx: &mut IrContext,
        loc: Location,
        name: &str,
        param_ty: TypeRef,
        ret_ty: TypeRef,
    ) -> OpRef {
        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![BlockArgData {
                ty: param_ty,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let arg = ctx.block_arg(entry, 0);
        let ret = func::Return::operands([arg]).build(ctx, loc);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(ctx, &[param_ty], ret_ty);
        func::Func::operands()
            .sym_name(ctx.intern_str(name))
            .r#type(func_ty)
            .regions(body)
            .build(ctx, loc)
            .op_ref()
    }

    #[test]
    fn test_auto_alias_repeated_type() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Create a complex struct type
        let struct_ty = make_named_type(&mut ctx, "Point", &[("x", i32_ty), ("y", i32_ty)]);

        // Use it in 3 functions
        let f1 = make_identity_func(&mut ctx, loc, "f1", struct_ty, struct_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", struct_ty, struct_ty);
        let f3 = make_identity_func(&mut ctx, loc, "f3", struct_ty, struct_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2, f3]);
        let output = module.display(&ctx).to_string();

        // The struct type should be auto-aliased with its name
        assert!(
            output.contains("!Point = alias_test.record<"),
            "Expected auto alias !Point in:\n{output}"
        );
        // The functions should reference the alias
        assert!(
            output.contains("!Point)"),
            "Expected !Point reference in:\n{output}"
        );
    }

    #[test]
    fn test_auto_alias_simple_type_skipped() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Use core.i32 many times - should NOT get aliased (it's a leaf type)
        let f1 = make_identity_func(&mut ctx, loc, "f1", i32_ty, i32_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", i32_ty, i32_ty);
        let f3 = make_identity_func(&mut ctx, loc, "f3", i32_ty, i32_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2, f3]);
        let output = module.display(&ctx).to_string();

        // No auto aliases should be generated — core.i32 is a leaf type and
        // func.func_sig has no dialect-provided name hint
        assert!(
            !output.contains('!'),
            "No types should be aliased:\n{output}"
        );
    }

    #[test]
    fn test_auto_alias_manual_priority() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let struct_ty = make_named_type(&mut ctx, "Point", &[("x", i32_ty), ("y", i32_ty)]);

        // Manually register this type as an alias
        ctx.register_type_alias(Symbol::new("my_point"), struct_ty);

        let f1 = make_identity_func(&mut ctx, loc, "f1", struct_ty, struct_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", struct_ty, struct_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2]);
        let output = module.display(&ctx).to_string();

        // Should use the manual alias, not auto-generate one
        assert!(
            output.contains("!my_point = alias_test.record<"),
            "Expected manual alias:\n{output}"
        );
        assert!(
            !output.contains("!Point"),
            "Should not auto-alias when manual exists:\n{output}"
        );
    }

    #[test]
    fn test_manual_alias_order_is_independent_of_registration_order() {
        fn print_with_aliases(reverse_registration: bool) -> String {
            let mut ctx = IrContext::new();
            let loc = test_location(&mut ctx);
            let i32_ty = make_i32_type(&mut ctx);
            let alpha_ty = make_named_type(&mut ctx, "Alpha", &[("value", i32_ty)]);
            let inner_ty = make_named_type(&mut ctx, "Inner", &[("value", i32_ty)]);
            let outer_ty = make_named_type(&mut ctx, "Outer", &[("inner", inner_ty)]);
            let zebra_ty = make_named_type(&mut ctx, "Zebra", &[("value", i32_ty)]);
            let aliases = [
                (Symbol::new("alpha"), alpha_ty),
                (Symbol::new("a_outer"), outer_ty),
                (Symbol::new("z_inner"), inner_ty),
                (Symbol::new("zebra"), zebra_ty),
            ];

            if reverse_registration {
                for (name, ty) in aliases.iter().cloned().rev() {
                    ctx.register_type_alias(name, ty);
                }
            } else {
                for (name, ty) in aliases {
                    ctx.register_type_alias(name, ty);
                }
            }

            let module = make_module_with_funcs(&mut ctx, loc, vec![]);
            module.display(&ctx).to_string()
        }

        let output = print_with_aliases(false);
        assert_eq!(output, print_with_aliases(true));

        let alpha_pos = output.find("!alpha =").expect("missing !alpha alias");
        let inner_pos = output.find("!z_inner =").expect("missing !z_inner alias");
        let outer_pos = output.find("!a_outer =").expect("missing !a_outer alias");
        let zebra_pos = output.find("!zebra =").expect("missing !zebra alias");
        assert!(alpha_pos < inner_pos && inner_pos < outer_pos && outer_pos < zebra_pos);
        let outer_line = output
            .lines()
            .find(|line| line.contains("!a_outer ="))
            .unwrap();
        assert!(outer_line.contains("!z_inner"));

        let mut reparsed_ctx = IrContext::new();
        let reparsed =
            crate::parser::parse_module(&mut reparsed_ctx, &output).expect("parse failed");
        assert_eq!(output, reparsed.display(&reparsed_ctx).to_string());
    }

    #[test]
    fn test_auto_alias_roundtrip() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let struct_ty = make_named_type(&mut ctx, "Point", &[("x", i32_ty), ("y", i32_ty)]);

        let f1 = make_identity_func(&mut ctx, loc, "f1", struct_ty, struct_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", struct_ty, struct_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2]);
        let output1 = module.display(&ctx).to_string();

        // Parse the output back
        let mut ctx2 = IrContext::new();
        let root2 = crate::parser::parse_module(&mut ctx2, &output1).expect("parse failed");
        let output2 = root2.display(&ctx2).to_string();

        assert_eq!(output1, output2, "Round-trip mismatch");
    }

    #[test]
    fn test_auto_alias_topological() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Type B: a simple struct
        let b_ty = make_named_type(&mut ctx, "Inner", &[("val", i32_ty)]);
        // Type A: references B
        let a_ty = make_named_type(&mut ctx, "Outer", &[("inner", b_ty), ("extra", i32_ty)]);

        // Use both types multiple times
        let f1 = make_identity_func(&mut ctx, loc, "f1", a_ty, a_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", a_ty, b_ty);
        let f3 = make_identity_func(&mut ctx, loc, "f3", b_ty, b_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2, f3]);
        let output = module.display(&ctx).to_string();

        // B (Inner) should appear before A (Outer) in alias definitions
        let inner_pos = output.find("!Inner").expect("Expected !Inner alias");
        let outer_pos = output.find("!Outer").expect("Expected !Outer alias");
        assert!(
            inner_pos < outer_pos,
            "!Inner should come before !Outer for topological ordering:\n{output}"
        );
        // Outer's definition should reference !Inner
        let outer_line = output.lines().find(|l| l.contains("!Outer =")).unwrap();
        assert!(
            outer_line.contains("!Inner"),
            "Outer should reference !Inner:\n{outer_line}"
        );
    }

    #[test]
    fn test_auto_alias_nested_module_isolation() {
        let input = "\
core.module @test {
  core.module @inner {
    func.func @f1(%0: alias_test.record<core.i32 {name = \"a\"}, core.i32 {name = \"b\"}, {name = \"InnerOnly\"}>) -> alias_test.record<core.i32 {name = \"a\"}, core.i32 {name = \"b\"}, {name = \"InnerOnly\"}> {
    ^bb0:
      func.return %0
    }
    func.func @f2(%0: alias_test.record<core.i32 {name = \"a\"}, core.i32 {name = \"b\"}, {name = \"InnerOnly\"}>) -> alias_test.record<core.i32 {name = \"a\"}, core.i32 {name = \"b\"}, {name = \"InnerOnly\"}> {
    ^bb0:
      func.return %0
    }
  }
  func.func @g1(%0: alias_test.record<core.i32 {name = \"x\"}, core.i32 {name = \"y\"}, {name = \"OuterOnly\"}>) -> alias_test.record<core.i32 {name = \"x\"}, core.i32 {name = \"y\"}, {name = \"OuterOnly\"}> {
  ^bb0:
    func.return %0
  }
  func.func @g2(%0: alias_test.record<core.i32 {name = \"x\"}, core.i32 {name = \"y\"}, {name = \"OuterOnly\"}>) -> alias_test.record<core.i32 {name = \"x\"}, core.i32 {name = \"y\"}, {name = \"OuterOnly\"}> {
  ^bb0:
    func.return %0
  }
}
";
        let mut ctx = IrContext::new();
        let root = crate::parser::parse_module(&mut ctx, input).expect("parse failed");
        let output = root.display(&ctx).to_string();
        insta::assert_snapshot!(output);
    }

    #[test]
    fn an_empty_block_survives_a_round_trip() {
        let input = r#"core.module @m {
  "test.op"() ({
  ^bb0:
  }, {
  })
}
"#;
        let mut ctx = IrContext::new();
        let root = crate::parser::parse_module(&mut ctx, input).expect("parse failed");
        let op = ctx
            .block(ctx.region(ctx.op_region(root, 0).unwrap()).blocks[0])
            .ops[0];
        let counts = |ctx: &IrContext| {
            ctx.op_regions(op)
                .map(|region| ctx.region(region).blocks.len())
                .collect::<Vec<_>>()
        };
        assert_eq!(counts(&ctx), [1, 0]);

        let printed = root.display(&ctx).to_string();
        let mut reparsed = IrContext::new();
        let root = crate::parser::parse_module(&mut reparsed, &printed).expect("parse failed");
        let op = reparsed
            .block(reparsed.region(reparsed.op_region(root, 0).unwrap()).blocks[0])
            .ops[0];
        let regions: Vec<_> = reparsed
            .op_regions(op)
            .map(|region| reparsed.region(region).blocks.len())
            .collect();
        assert_eq!(regions, [1, 0], "{printed}");
    }

    #[test]
    fn test_auto_alias_name_conflict() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Two different struct types with the same name attribute
        let s1_ty = make_named_type(&mut ctx, "Point", &[("x", i32_ty)]);
        let s2_ty = make_named_type(&mut ctx, "Point", &[("x", i32_ty), ("y", i32_ty)]);

        let f1 = make_identity_func(&mut ctx, loc, "f1", s1_ty, s1_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", s2_ty, s2_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2]);
        let output = module.display(&ctx).to_string();

        // Both should exist, one as !Point and other as !Point_1
        assert!(
            output.contains("!Point ="),
            "Expected !Point alias:\n{output}"
        );
        assert!(
            output.contains("!Point_1 ="),
            "Expected !Point_1 alias for conflict:\n{output}"
        );
    }

    #[test]
    fn test_auto_alias_order_is_independent_of_type_interning_order() {
        fn print_module_with_point_types(reverse_interning: bool) -> String {
            let mut ctx = IrContext::new();
            let loc = test_location(&mut ctx);
            let i32_ty = make_i32_type(&mut ctx);
            let make_point =
                |ctx: &mut IrContext, field| make_named_type(ctx, "Point", &[(field, i32_ty)]);
            let (alpha_point, zebra_point) = if reverse_interning {
                let zebra = make_point(&mut ctx, "zebra");
                let alpha = make_point(&mut ctx, "alpha");
                (alpha, zebra)
            } else {
                let alpha = make_point(&mut ctx, "alpha");
                let zebra = make_point(&mut ctx, "zebra");
                (alpha, zebra)
            };
            let funcs = vec![
                make_identity_func(&mut ctx, loc, "alpha_1", alpha_point, alpha_point),
                make_identity_func(&mut ctx, loc, "alpha_2", alpha_point, alpha_point),
                make_identity_func(&mut ctx, loc, "zebra_1", zebra_point, zebra_point),
                make_identity_func(&mut ctx, loc, "zebra_2", zebra_point, zebra_point),
            ];
            let module = make_module_with_funcs(&mut ctx, loc, funcs);
            module.display(&ctx).to_string()
        }

        let output = print_module_with_point_types(false);
        assert_eq!(output, print_module_with_point_types(true));
        assert!(output.contains(
            r#"!Point = alias_test.record<core.i32 {name = "alpha"}, {name = "Point"}>"#
        ));
        assert!(output.contains(
            r#"!Point_1 = alias_test.record<core.i32 {name = "zebra"}, {name = "Point"}>"#
        ));

        let mut reparsed_ctx = IrContext::new();
        let reparsed =
            crate::parser::parse_module(&mut reparsed_ctx, &output).expect("parse failed");
        assert_eq!(output, reparsed.display(&reparsed_ctx).to_string());
    }

    /// Documents the escape spellings; `prop_tests` checks that they round-trip.
    #[test]
    fn test_write_escaped_bytes_escapes() {
        let mut buf = String::new();
        write_escaped_bytes(&mut buf, b"a\n\t\r\0\\\"b\x7f\x80\xff").unwrap();
        assert_eq!(buf, r#"a\n\t\r\0\\\"b\x7f\x80\xff"#);
    }
}

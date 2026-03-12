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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Write;
use std::ops::ControlFlow;

use super::context::IrContext;
use super::refs::*;
use super::types::*;
use super::walk::{WalkAction, walk_region};

/// Print state for value numbering and block labeling.
struct PrintState<'a> {
    ctx: &'a IrContext,
    value_names: HashMap<ValueRef, String>,
    block_labels: HashMap<BlockRef, String>,
    next_value_num: usize,
    next_block_num: usize,
    /// Reverse map: TypeRef → alias name for substitution during printing.
    type_alias_names: HashMap<TypeRef, String>,
}

impl<'a> PrintState<'a> {
    fn new(ctx: &'a IrContext) -> Self {
        let type_alias_names = ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (*ty, name.to_string()))
            .collect();
        Self {
            ctx,
            value_names: HashMap::new(),
            block_labels: HashMap::new(),
            next_value_num: 0,
            next_block_num: 0,
            type_alias_names,
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

    // ====================================================================
    // Type / Attribute writing (with alias support)
    // ====================================================================

    fn write_type(&self, f: &mut impl Write, ty: TypeRef) -> fmt::Result {
        // Check alias map first
        if let Some(alias_name) = self.type_alias_names.get(&ty) {
            return write_type_alias_name(f, alias_name);
        }
        let data = self.ctx.types.get(ty);
        write!(f, "{}.{}", data.dialect, data.name)?;
        if !data.params.is_empty() {
            f.write_char('(')?;
            for (i, &param) in data.params.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                self.write_type(f, param)?;
            }
            f.write_char(')')?;
        } else if !data.attrs.is_empty() {
            f.write_str("()")?;
        }
        if !data.attrs.is_empty() {
            f.write_str(" {")?;
            for (i, (key, val)) in data.attrs.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{} = ", key)?;
                self.write_attribute(f, val)?;
            }
            f.write_char('}')?;
        }
        Ok(())
    }

    fn write_attribute(&self, f: &mut impl Write, attr: &Attribute) -> fmt::Result {
        match attr {
            Attribute::Unit => f.write_str("unit"),
            Attribute::Bool(b) => write!(f, "{b}"),
            Attribute::Int(v) => write!(f, "{v}"),
            Attribute::FloatBits(bits) => {
                let v = f64::from_bits(*bits);
                let s = format!("{v}");
                f.write_str(&s)?;
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
            Attribute::Type(ty) => self.write_type(f, *ty),
            Attribute::List(list) => {
                f.write_char('[')?;
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    self.write_attribute(f, item)?;
                }
                f.write_char(']')
            }
            Attribute::Location(loc) => {
                let path_str = self.ctx.paths.get(loc.path);
                f.write_str("loc(\"")?;
                write_escaped_string(f, path_str)?;
                write!(f, "\" {}:{})", loc.span.start, loc.span.end)
            }
        }
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
    print_module_op(&mut state, &mut out, root, 0).expect("fmt::Write to String never fails");
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
        Attribute::Int(v) => write!(f, "{v}"),
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

/// Write a type alias name with `!` prefix. Quotes if name contains non-ident chars.
fn write_type_alias_name(f: &mut impl Write, name: &str) -> fmt::Result {
    let needs_quoting = name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_');
    if needs_quoting {
        f.write_str("!\"")?;
        write_escaped_string(f, name)?;
        f.write_char('"')
    } else {
        write!(f, "!{name}")
    }
}

fn write_symbol(f: &mut impl Write, sym: crate::symbol::Symbol) -> fmt::Result {
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
    let mut counts: HashMap<TypeRef, usize> = HashMap::new();

    let _ = walk_region::<()>(ctx, region, &mut |op| {
        // Result types
        for &ty in ctx.op_result_types(op) {
            *counts.entry(ty).or_default() += 1;
        }

        let data = ctx.op(op);

        // Attributes containing types
        for attr in data.attributes.values() {
            count_attr_types(&mut counts, attr);
        }

        // Block args in regions
        for &region in &data.regions {
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
    match attr {
        Attribute::Type(ty) => *counts.entry(*ty).or_default() += 1,
        Attribute::List(list) => {
            for item in list {
                count_attr_types(counts, item);
            }
        }
        _ => {}
    }
}

/// Estimate the printed character length of a type.
fn type_complexity(ctx: &IrContext, ty: TypeRef) -> usize {
    let data = ctx.types.get(ty);
    let mut size = data.dialect.with_str(|s| s.len()) + 1 + data.name.with_str(|s| s.len());
    for &param in &data.params {
        size += type_complexity(ctx, param) + 2; // ", " separator
    }
    for (key, val) in &data.attrs {
        size += key.with_str(|s| s.len()) + 3; // "key = "
        size += attr_complexity(val);
    }
    size
}

/// Estimate the printed character length of an attribute.
fn attr_complexity(attr: &Attribute) -> usize {
    match attr {
        Attribute::Unit => 4,
        Attribute::Bool(_) => 5,
        Attribute::Int(v) => format!("{v}").len(),
        Attribute::FloatBits(_) => 8,
        Attribute::String(s) => s.len() + 2,
        Attribute::Bytes(b) => b.len() * 4 + 7,
        Attribute::Symbol(sym) => sym.with_str(|s| s.len()) + 1,
        Attribute::Type(_) => 10, // rough estimate; actual depends on type
        Attribute::List(list) => list.iter().map(attr_complexity).sum::<usize>() + list.len() * 2,
        Attribute::Location(_) => 20,
    }
}

/// Generate auto aliases for types that are used frequently and are complex enough.
fn generate_auto_aliases(
    ctx: &IrContext,
    region: RegionRef,
    existing: &HashMap<TypeRef, String>,
) -> Vec<(String, TypeRef)> {
    let counts = collect_module_types(ctx, region);
    let mut candidates: Vec<(TypeRef, usize, usize)> = Vec::new();

    for (&ty, &count) in &counts {
        if existing.contains_key(&ty) {
            continue;
        }
        let data = ctx.types.get(ty);
        // Skip leaf types (no params, no attrs) — they're already concise
        if data.params.is_empty() && data.attrs.is_empty() {
            continue;
        }
        let complexity = type_complexity(ctx, ty);
        if count >= MIN_ALIAS_USES || complexity >= MIN_ALIAS_COMPLEXITY {
            candidates.push((ty, count, complexity));
        }
    }

    // Sort by (count desc, complexity desc, TypeRef asc) for deterministic output
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));

    // Name assignment
    let mut used_names: HashSet<String> = existing.values().cloned().collect();
    let mut next_num = 0usize;
    let mut result = Vec::new();

    for &(ty, _, _) in &candidates {
        let name = choose_alias_name(ctx, ty, &used_names, &mut next_num);
        used_names.insert(name.clone());
        result.push((name, ty));
    }

    // Topological sort: if type A references type B, B must come first
    topological_sort_aliases(ctx, &mut result);

    result
}

/// Choose an alias name for a type.
///
/// 1. Try dialect-provided hint (e.g., `name` attribute on adt.struct)
/// 2. On conflict, add suffix: `Point`, `Point_1`, `Point_2`, ...
/// 3. Fallback: `t0`, `t1`, `t2`, ...
fn choose_alias_name(
    ctx: &IrContext,
    ty: TypeRef,
    used_names: &HashSet<String>,
    next_num: &mut usize,
) -> String {
    if let Some(sym) = crate::op_interface::suggest_type_alias_name(ctx, ty) {
        let base = sym.with_str(|s| s.to_string());
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
            let mut deps = HashSet::new();
            collect_type_deps(ctx, *ty, &alias_set, &mut deps);
            deps
        })
        .collect();

    // Simple stable topological sort via repeated extraction of dependency-free items
    let n = aliases.len();
    let mut sorted: Vec<(String, TypeRef)> = Vec::with_capacity(n);
    let mut placed: HashSet<TypeRef> = HashSet::new();
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
    let data = ctx.types.get(ty);
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
    match attr {
        Attribute::Type(ty) => {
            if alias_set.contains(ty) {
                deps.insert(*ty);
            }
            collect_type_deps(ctx, *ty, alias_set, deps);
        }
        Attribute::List(list) => {
            for item in list {
                collect_attr_type_deps(ctx, item, alias_set, deps);
            }
        }
        _ => {}
    }
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
        return print_module_op(state, f, op, indent);
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
            state.write_attribute(f, val)?;
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
            state.write_type(f, ty)?;
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
                    state.write_type(f, ty)?;
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

fn print_module_op(
    state: &mut PrintState<'_>,
    f: &mut impl Write,
    op: OpRef,
    indent: usize,
) -> fmt::Result {
    let indent_str = " ".repeat(indent);
    let data = state.ctx.op(op);
    write!(f, "{indent_str}core.module")?;

    // Module name
    if let Some(Attribute::Symbol(name)) = data.attributes.get(&crate::Symbol::new("sym_name")) {
        f.write_char(' ')?;
        write_symbol(f, *name)?;
    }

    let regions = &data.regions;
    assert!(
        regions.len() <= 1,
        "print_module_op: expected at most one region, found {}",
        regions.len(),
    );
    if let Some(&region) = regions.first() {
        f.write_str(" {\n")?;

        let inner_indent = format!("{}  ", indent_str);

        // 1. Emit manual type alias definitions
        let manual_aliases: Vec<_> = state.ctx.type_aliases().to_vec();
        for (name, ty) in &manual_aliases {
            write!(f, "{inner_indent}")?;
            write_type_alias_name(f, &name.to_string())?;
            f.write_str(" = ")?;
            // Temporarily remove this alias from the map so we print the
            // full type definition, while earlier aliases can still be used.
            state.type_alias_names.remove(ty);
            state.write_type(f, *ty)?;
            // Re-insert so subsequent aliases and ops can reference it
            state.type_alias_names.insert(*ty, name.to_string());
            f.write_char('\n')?;
        }

        // 2. Generate and emit auto aliases
        let auto_aliases = generate_auto_aliases(state.ctx, region, &state.type_alias_names);
        for (name, ty) in &auto_aliases {
            write!(f, "{inner_indent}")?;
            write_type_alias_name(f, name)?;
            f.write_str(" = ")?;
            state.write_type(f, *ty)?;
            // Register so subsequent aliases and ops can reference it
            state.type_alias_names.insert(*ty, name.clone());
            f.write_char('\n')?;
        }

        // Blank line after all alias definitions
        if !manual_aliases.is_empty() || !auto_aliases.is_empty() {
            f.write_char('\n')?;
        }

        // Print each top-level op with reset numbering
        let region_data = state.ctx.region(region);
        let blocks: Vec<_> = region_data.blocks.iter().copied().collect();
        for &block in &blocks {
            let block_data = state.ctx.block(block);
            let ops: Vec<_> = block_data.ops.iter().copied().collect();
            for &child_op in &ops {
                let saved = state.save_counters();
                state.reset_numbering();
                print_operation(state, f, child_op, indent + 2)?;
                state.reset_numbering();
                state.restore_counters(saved);
            }
        }

        writeln!(f, "{indent_str}}}")?;
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
    assert!(
        regions.len() <= 1,
        "print_func_op: expected at most one region, found {}",
        regions.len(),
    );
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
                state.write_type(f, ty)?;
            }
            f.write_char(')')?;
        }

        // Return type and effects from func type attribute
        if let Some(Attribute::Type(func_ty)) = data.attributes.get(&crate::Symbol::new("type")) {
            let ty_data = state.ctx.types.get(*func_ty);
            let is_core_func = ty_data.dialect == crate::Symbol::new("core")
                && ty_data.name == crate::Symbol::new("func");
            if is_core_func && !ty_data.params.is_empty() {
                // Decompose core.func: params[0] = return type, attrs["effect"] = effect
                let result_ty = ty_data.params[0];
                let effect_ty = ty_data
                    .attrs
                    .get(&crate::Symbol::new("effect"))
                    .and_then(|a| match a {
                        Attribute::Type(t) => Some(*t),
                        _ => None,
                    });
                f.write_str(" -> ")?;
                state.write_type(f, result_ty)?;
                if let Some(eff) = effect_ty {
                    f.write_str(" effects ")?;
                    state.write_type(f, eff)?;
                }
            } else {
                f.write_str(" -> ")?;
                state.write_type(f, *func_ty)?;
            }
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
                            state.write_type(f, ty)?;
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
    use crate::dialect::{arith, core, func};
    use crate::{BlockArgData, BlockData, RegionData, TypeDataBuilder};
    use smallvec::smallvec;

    fn test_location(ctx: &mut IrContext) -> Location {
        let path = ctx.paths.intern("test.trb".to_owned());
        Location::new(path, crate::Span::new(0, 0))
    }

    fn make_i32_type(ctx: &mut IrContext) -> TypeRef {
        ctx.types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("i32")).build())
    }

    /// Create a `core.func` type. Parameters are laid out as `[ret, ...params]`
    /// in `TypeData.params`, matching the convention used by `core::Func`.
    fn make_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
        crate::dialect::core::func(ctx, ret, params.iter().copied(), None).as_type_ref()
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
        let tuple_ty = crate::dialect::core::tuple(&mut ctx, [i32_ty, i32_ty]).as_type_ref();
        assert_eq!(print_type(&ctx, tuple_ty), "core.tuple(core.i32, core.i32)");
    }

    #[test]
    fn test_print_simple_op() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        let output = print_op(&ctx, c.op_ref());
        assert_eq!(output, "%0 = arith.const {value = 42} : core.i32\n");
    }

    #[test]
    fn test_print_binary_op() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        let c2 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(2));
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

        let c = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
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
        let ret_inner = func::r#return(&mut ctx, loc, [x]);
        ctx.push_op(inner_entry, ret_inner.op_ref());

        let inner_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![inner_entry],
            parent_op: None,
        });
        let inner_func = func::func(&mut ctx, loc, Symbol::new("get_x"), func_ty, inner_body);

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
        let inner_module = core::module(&mut ctx, loc, Symbol::new("Point"), inner_mod_region);

        // Outer function: fn make() -> i32 { return 1; }
        let outer_entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        let one = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(1));
        ctx.push_op(outer_entry, one.op_ref());
        let one_val = one.result(&ctx);
        let ret_outer = func::r#return(&mut ctx, loc, [one_val]);
        ctx.push_op(outer_entry, ret_outer.op_ref());

        let outer_body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![outer_entry],
            parent_op: None,
        });
        let make_func_ty = make_func_type(&mut ctx, &[], i32_ty);
        let outer_func = func::func(&mut ctx, loc, Symbol::new("make"), make_func_ty, outer_body);

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
        let outer_module = core::module(&mut ctx, loc, Symbol::new("test"), outer_mod_region);

        let output = print_module(&ctx, outer_module.op_ref());
        insta::assert_snapshot!(output);
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

    // ====================================================================
    // Auto alias tests
    // ====================================================================

    /// Helper: build an `adt.struct` type with given field list and name.
    fn make_adt_struct(ctx: &mut IrContext, name: &str, fields: &[(&str, TypeRef)]) -> TypeRef {
        let field_list: Vec<Attribute> = fields
            .iter()
            .map(|(fname, fty)| {
                Attribute::List(vec![
                    Attribute::Symbol(Symbol::from_dynamic(fname)),
                    Attribute::Type(*fty),
                ])
            })
            .collect();
        let data = TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
            .attr("fields", Attribute::List(field_list))
            .attr("name", Attribute::Symbol(Symbol::from_dynamic(name)))
            .build();
        ctx.types.intern(data)
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
        core::module(ctx, loc, Symbol::new("test"), mod_region).op_ref()
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
        let ret = func::r#return(ctx, loc, [arg]);
        ctx.push_op(entry, ret.op_ref());
        let body = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![entry],
            parent_op: None,
        });
        let func_ty = make_func_type(ctx, &[param_ty], ret_ty);
        func::func(ctx, loc, Symbol::from_dynamic(name), func_ty, body).op_ref()
    }

    #[test]
    fn test_auto_alias_repeated_type() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Create a complex struct type
        let struct_ty = make_adt_struct(&mut ctx, "Point", &[("x", i32_ty), ("y", i32_ty)]);

        // Use it in 3 functions
        let f1 = make_identity_func(&mut ctx, loc, "f1", struct_ty, struct_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", struct_ty, struct_ty);
        let f3 = make_identity_func(&mut ctx, loc, "f3", struct_ty, struct_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2, f3]);
        let output = print_module(&ctx, module);

        // The struct type should be auto-aliased with its name
        assert!(
            output.contains("!Point = adt.struct()"),
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
        let output = print_module(&ctx, module);

        // core.i32 should NOT be aliased (it's a leaf type with no params/attrs)
        // The func type may be aliased, but core.i32 itself should appear inline
        assert!(
            !output.contains("= core.i32\n"),
            "core.i32 should not be aliased:\n{output}"
        );
    }

    #[test]
    fn test_auto_alias_named_struct() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Create a named struct
        let marker_ty = make_adt_struct(
            &mut ctx,
            "_Marker",
            &[("ability_id", i32_ty), ("prompt_tag", i32_ty)],
        );

        let f1 = make_identity_func(&mut ctx, loc, "f1", marker_ty, marker_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", marker_ty, marker_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2]);
        let output = print_module(&ctx, module);

        // Should use the name from the `name` attribute
        assert!(
            output.contains("!_Marker = adt.struct()"),
            "Expected !_Marker alias:\n{output}"
        );
    }

    #[test]
    fn test_auto_alias_manual_priority() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let struct_ty = make_adt_struct(&mut ctx, "Point", &[("x", i32_ty), ("y", i32_ty)]);

        // Manually register this type as an alias
        ctx.register_type_alias(Symbol::from_dynamic("my_point"), struct_ty);

        let f1 = make_identity_func(&mut ctx, loc, "f1", struct_ty, struct_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", struct_ty, struct_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2]);
        let output = print_module(&ctx, module);

        // Should use the manual alias, not auto-generate one
        assert!(
            output.contains("!my_point = adt.struct()"),
            "Expected manual alias:\n{output}"
        );
        assert!(
            !output.contains("!Point"),
            "Should not auto-alias when manual exists:\n{output}"
        );
    }

    #[test]
    fn test_auto_alias_roundtrip() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        let struct_ty = make_adt_struct(&mut ctx, "Point", &[("x", i32_ty), ("y", i32_ty)]);

        let f1 = make_identity_func(&mut ctx, loc, "f1", struct_ty, struct_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", struct_ty, struct_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2]);
        let output1 = print_module(&ctx, module);

        // Parse the output back
        let mut ctx2 = IrContext::new();
        let root2 = crate::parser::parse_module(&mut ctx2, &output1).expect("parse failed");
        let output2 = print_module(&ctx2, root2);

        assert_eq!(output1, output2, "Round-trip mismatch");
    }

    #[test]
    fn test_auto_alias_topological() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Type B: a simple struct
        let b_ty = make_adt_struct(&mut ctx, "Inner", &[("val", i32_ty)]);
        // Type A: references B
        let a_ty = make_adt_struct(&mut ctx, "Outer", &[("inner", b_ty), ("extra", i32_ty)]);

        // Use both types multiple times
        let f1 = make_identity_func(&mut ctx, loc, "f1", a_ty, a_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", a_ty, b_ty);
        let f3 = make_identity_func(&mut ctx, loc, "f3", b_ty, b_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2, f3]);
        let output = print_module(&ctx, module);

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
    fn test_auto_alias_name_conflict() {
        let mut ctx = IrContext::new();
        let loc = test_location(&mut ctx);
        let i32_ty = make_i32_type(&mut ctx);

        // Two different struct types with the same name attribute
        let s1_ty = make_adt_struct(&mut ctx, "Point", &[("x", i32_ty)]);
        let s2_ty = make_adt_struct(&mut ctx, "Point", &[("x", i32_ty), ("y", i32_ty)]);

        let f1 = make_identity_func(&mut ctx, loc, "f1", s1_ty, s1_ty);
        let f2 = make_identity_func(&mut ctx, loc, "f2", s2_ty, s2_ty);

        let module = make_module_with_funcs(&mut ctx, loc, vec![f1, f2]);
        let output = print_module(&ctx, module);

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
    fn test_rauw_updates_operands() {
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

        let c1 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(42));
        ctx.push_op(entry, c1.op_ref());
        let v1 = c1.result(&ctx);

        let c2 = arith::r#const(&mut ctx, loc, i32_ty, Attribute::Int(99));
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

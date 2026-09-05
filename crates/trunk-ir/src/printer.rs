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
use super::ops::DialectType;
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
        let mut state = Self::without_aliases(ctx);
        state.type_alias_names = ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (*ty, name.to_string()))
            .collect();
        state
    }

    fn without_aliases(ctx: &'a IrContext) -> Self {
        Self {
            ctx,
            value_names: HashMap::new(),
            block_labels: HashMap::new(),
            next_value_num: 0,
            next_block_num: 0,
            type_alias_names: HashMap::new(),
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

    fn write_type(&self, f: &mut dyn Write, ty: TypeRef) -> fmt::Result {
        // Check alias map first
        if let Some(alias_name) = self.type_alias_names.get(&ty) {
            return write_type_alias_name(f, alias_name);
        }
        if let Some((inputs, results)) = func_sig_parts(self.ctx, ty) {
            return self.write_func_sig_type(f, ty, inputs, results);
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

    fn write_func_sig_type(
        &self,
        f: &mut dyn Write,
        ty: TypeRef,
        inputs: &[TypeRef],
        results: &[TypeRef],
    ) -> fmt::Result {
        let data = self.ctx.types.get(ty);
        write!(f, "{}.{}<(", data.dialect, data.name)?;
        for (index, &input) in inputs.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            self.write_type(f, input)?;
        }
        f.write_str(") -> ")?;
        if results.len() == 1 {
            self.write_type(f, results[0])?;
        } else {
            f.write_char('(')?;
            for (index, &result) in results.iter().enumerate() {
                if index > 0 {
                    f.write_str(", ")?;
                }
                self.write_type(f, result)?;
            }
            f.write_char(')')?;
        }
        f.write_char('>')?;

        let mut visible = data.attrs.iter().filter(|(key, _)| {
            **key != crate::Symbol::new(crate::dialect::func::NUM_INPUTS_ATTR)
                && **key != crate::Symbol::new(crate::dialect::func::NUM_RESULTS_ATTR)
        });
        if let Some((key, value)) = visible.next() {
            write!(f, " {{{key} = ")?;
            self.write_attribute(f, value)?;
            for (key, value) in visible {
                write!(f, ", {key} = ")?;
                self.write_attribute(f, value)?;
            }
            f.write_char('}')?;
        }
        Ok(())
    }

    fn write_attribute(&self, f: &mut dyn Write, attr: &Attribute) -> fmt::Result {
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
                f.write_str("b\"")?;
                write_escaped_bytes(f, bytes)?;
                f.write_char('"')
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

/// Return the delimiter-sliced storage only for a complete `*.func_sig` shape.
/// Other types, including malformed count-shaped data, retain concrete printing.
fn func_sig_parts(ctx: &IrContext, ty: TypeRef) -> Option<(&[TypeRef], &[TypeRef])> {
    let data = ctx.types.get(ty);
    if data.name != crate::Symbol::new("func_sig") {
        return None;
    }
    if data.dialect == crate::Symbol::new("func")
        && crate::dialect::func::FuncSig::from_type_ref(ctx, ty).is_none()
    {
        return None;
    }
    let num_inputs = match data.attrs.get(crate::dialect::func::NUM_INPUTS_ATTR) {
        Some(Attribute::Int(value)) => usize::try_from(u32::try_from(*value).ok()?).ok()?,
        _ => return None,
    };
    let num_results = match data.attrs.get(crate::dialect::func::NUM_RESULTS_ATTR) {
        Some(Attribute::Int(value)) => usize::try_from(u32::try_from(*value).ok()?).ok()?,
        _ => return None,
    };
    let end = num_inputs.checked_add(num_results)?;
    if end != data.params.len() {
        return None;
    }
    Some((&data.params[..num_inputs], &data.params[num_inputs..]))
}

// ============================================================================
// OpPrintHelper — public wrapper for custom assembly format printers
// ============================================================================

/// Public wrapper around printer state for custom `OpAsmFormat` implementations.
///
/// Provides access to IR context, value naming, type/attribute printing, and
/// region printing without exposing `PrintState` internals.
///
/// Implements `fmt::Write` so `write!(helper, ...)` can be used directly.
pub struct OpPrintHelper<'a, 'ctx> {
    state: &'a mut PrintState<'ctx>,
    f: &'a mut dyn Write,
}

impl<'a, 'ctx> OpPrintHelper<'a, 'ctx> {
    /// Access the IR context.
    pub fn ctx(&self) -> &IrContext {
        self.state.ctx
    }

    /// Assign a sequential name (%0, %1, ...) to a value.
    pub fn assign_value_name(&mut self, v: ValueRef) -> String {
        self.state.assign_value_name(v)
    }

    /// Get the previously assigned name of a value.
    pub fn get_value_name(&self, v: ValueRef) -> &str {
        self.state.get_value_name(v)
    }

    /// Write a type using the current alias map.
    pub fn write_type(&mut self, ty: TypeRef) -> fmt::Result {
        self.state.write_type(&mut *self.f, ty)
    }

    /// Write an attribute value.
    pub fn write_attribute(&mut self, attr: &Attribute) -> fmt::Result {
        self.state.write_attribute(&mut *self.f, attr)
    }

    /// Reset value and block numbering (for func.func — each function restarts at %0).
    pub fn reset_numbering(&mut self) {
        self.state.reset_numbering();
    }

    /// Print a region with the entry block label elided.
    ///
    /// Entry block args are assumed to have been printed externally (e.g., as
    /// function parameters in the operation signature). For single-block regions,
    /// the entry label is omitted entirely. For multi-block regions, the entry
    /// block label is printed without args, and non-entry blocks are printed
    /// normally.
    pub fn print_region_eliding_entry(&mut self, region: RegionRef, indent: usize) -> fmt::Result {
        let blocks: Vec<BlockRef> = self
            .state
            .ctx
            .region(region)
            .blocks
            .iter()
            .copied()
            .collect();

        // Pre-assign block labels for all blocks
        for &block in &blocks {
            self.state.assign_block_label(block);
        }

        let is_single_block = blocks.len() == 1;

        for (i, &block) in blocks.iter().enumerate() {
            if i == 0 && is_single_block {
                // Single block: elide entry block label entirely
            } else {
                // Multi-block: print label
                let indent_str = " ".repeat(indent);
                let label = self.state.get_block_label(block).to_owned();
                write!(self.f, "{indent_str}{label}")?;
                if i > 0 {
                    // Non-entry blocks: print args
                    let args = self.state.ctx.block_args(block);
                    if !args.is_empty() {
                        self.f.write_char('(')?;
                        for (j, &arg) in args.iter().enumerate() {
                            if j > 0 {
                                self.f.write_str(", ")?;
                            }
                            let name = self.state.assign_value_name(arg);
                            let ty = self.state.ctx.value_ty(arg);
                            write!(self.f, "{name}: ")?;
                            self.state.write_type(&mut *self.f, ty)?;
                        }
                        self.f.write_char(')')?;
                    }
                }
                self.f.write_str(":\n")?;
            }

            // Print ops in block
            let block_data = self.state.ctx.block(block);
            let ops: Vec<_> = block_data.ops.iter().copied().collect();
            for &op in &ops {
                print_operation(self.state, &mut *self.f, op, indent + 2)?;
            }
        }

        Ok(())
    }
}

impl fmt::Write for OpPrintHelper<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.f.write_str(s)
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
    PrintState::without_aliases(ctx)
        .write_type(&mut out, ty)
        .expect("fmt::Write to String never fails");
    out
}

/// Print a module (root operation with nested functions) as IR text.
pub fn print_module(ctx: &IrContext, root: OpRef) -> String {
    let mut state = PrintState::new(ctx);
    let mut out = String::new();
    print_module_op(&mut state, &mut out, root, 0).expect("fmt::Write to String never fails");
    out
}

#[cfg(test)]
fn write_attribute(ctx: &IrContext, f: &mut impl Write, attr: &Attribute) -> fmt::Result {
    PrintState::without_aliases(ctx).write_attribute(f, attr)
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

/// Write a type alias name with `!` prefix. Quotes if name contains non-ident chars.
fn write_type_alias_name(f: &mut dyn Write, name: &str) -> fmt::Result {
    let needs_quoting =
        name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if needs_quoting {
        f.write_str("!\"")?;
        write_escaped_string(f, name)?;
        f.write_char('"')
    } else {
        write!(f, "!{name}")
    }
}

fn write_symbol(f: &mut dyn Write, sym: crate::symbol::Symbol) -> fmt::Result {
    sym.with_str(|s| {
        let needs_quoting =
            s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
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
        let data = ctx.op(op);

        // Skip nested modules — they collect their own aliases.
        if data.dialect == crate::Symbol::new("core") && data.name == crate::Symbol::new("module") {
            return ControlFlow::Continue(WalkAction::Skip);
        }

        // Result types
        for &ty in ctx.op_result_types(op) {
            *counts.entry(ty).or_default() += 1;
        }

        // Attributes containing types.
        // Skip func.func — its `type` attribute holds the function signature
        // (func.func_sig), which would otherwise dominate alias candidates.
        let is_func_decl =
            data.dialect == crate::Symbol::new("func") && data.name == crate::Symbol::new("func");
        if !is_func_decl {
            for attr in data.attributes.values() {
                count_attr_types(&mut counts, attr);
            }
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
        let complexity = ctx.types.complexity(ty);
        let has_hint = crate::op_interface::suggest_type_alias_name(ctx, ty).is_some();
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
            candidates.push((ty, count, complexity, print_type(ctx, ty)));
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
    f: &mut dyn Write,
    op: OpRef,
    indent: usize,
) -> fmt::Result {
    let data = state.ctx.op(op);
    let dialect = data.dialect;
    let name = data.name;

    // Check for special ops
    let is_module = dialect == crate::Symbol::new("core") && name == crate::Symbol::new("module");
    if is_module {
        return print_module_op(state, f, op, indent);
    }

    // Check custom assembly format registry
    if let Some(fmt) = crate::op_interface::lookup_asm_format(dialect, name) {
        let mut helper = OpPrintHelper { state, f };
        return (fmt.print_fn)(&mut helper, op, indent);
    }

    print_generic_op(state, f, op, indent)
}

fn print_generic_op(
    state: &mut PrintState<'_>,
    f: &mut dyn Write,
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
    f: &mut dyn Write,
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
    f: &mut dyn Write,
    op: OpRef,
    indent: usize,
) -> fmt::Result {
    let indent_str = " ".repeat(indent);
    let data = state.ctx.op(op);
    write!(f, "{indent_str}core.module")?;

    // Module name
    if let Some(name) = data.attributes.get_symbol("sym_name") {
        f.write_char(' ')?;
        write_symbol(f, name)?;
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

        // Snapshot alias state so auto aliases are module-local.
        let saved_aliases = state.type_alias_names.clone();

        // 1. Emit manual type alias definitions
        let mut manual_aliases: Vec<_> = state
            .ctx
            .type_aliases()
            .iter()
            .map(|(name, ty)| (name.to_string(), *ty))
            .collect();
        manual_aliases.sort_by(|a, b| a.0.cmp(&b.0));
        topological_sort_aliases(state.ctx, &mut manual_aliases);
        for (name, ty) in &manual_aliases {
            write!(f, "{inner_indent}")?;
            write_type_alias_name(f, name)?;
            f.write_str(" = ")?;
            // Temporarily remove this alias from the map so we print the
            // full type definition, while earlier aliases can still be used.
            state.type_alias_names.remove(ty);
            state.write_type(f, *ty)?;
            // Re-insert so subsequent aliases and ops can reference it
            state.type_alias_names.insert(*ty, name.clone());
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

        // Restore alias state — auto aliases are scoped to this module.
        state.type_alias_names = saved_aliases;

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

    /// Create a one-result `func.func_sig` type.
    fn make_func_type(ctx: &mut IrContext, params: &[TypeRef], ret: TypeRef) -> TypeRef {
        crate::dialect::func::func_sig(ctx, params.iter().copied(), [ret]).as_type_ref()
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
    fn standalone_function_printing_expands_aliases_in_lists_and_attributes() {
        for result in ["()", "!scalar"] {
            let mut ctx = IrContext::new();
            let input = format!(
                "core.module @m {{
                !scalar = core.i32
                !callable = func.func_sig<(!scalar) -> {result}> {{nested = [!scalar]}}
                func.func @f(%x: !callable) {{ func.return }}
            }}"
            );
            let module = crate::parser::parse_module(&mut ctx, &input).unwrap();
            let callable = ctx.type_alias_by_name(Symbol::new("callable")).unwrap();
            let expanded_result = if result == "()" { "()" } else { "core.i32" };
            assert_eq!(
                print_type(&ctx, callable),
                format!("func.func_sig<(core.i32) -> {expanded_result}> {{nested = [core.i32]}}")
            );
            let printed = print_module(&ctx, module);
            assert!(
                printed.contains(&format!(
                    "func.func_sig<(!scalar) -> {result}> {{nested = [!scalar]}}"
                )),
                "{printed}"
            );
            let mut reparsed = IrContext::new();
            let module = crate::parser::parse_module(&mut reparsed, &printed).unwrap();
            assert_eq!(print_module(&reparsed, module), printed);
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

        assert_eq!(print_type(&ctx, zero_zero), "func.func_sig<() -> ()>");
        assert_eq!(
            print_type(&ctx, many_zero),
            "func.func_sig<(core.i32, core.i32) -> ()>"
        );
        assert_eq!(print_type(&ctx, zero_one), "func.func_sig<() -> core.i32>");
        assert_eq!(
            print_type(&ctx, many_one),
            "func.func_sig<(core.i32, core.i32) -> core.i32>"
        );
    }

    #[test]
    fn malformed_func_sig_storage_prints_as_concrete_type() {
        let mut ctx = IrContext::new();
        let i32_ty = make_i32_type(&mut ctx);
        let foreign = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("foreign"), Symbol::new("func_sig"))
                .param(i32_ty)
                .attr(func::NUM_INPUTS_ATTR, Attribute::Int(2))
                .attr(func::NUM_RESULTS_ATTR, Attribute::Int(1))
                .build(),
        );
        assert_eq!(
            print_type(&ctx, foreign),
            "foreign.func_sig(core.i32) {num_inputs = 2, num_results = 1}"
        );

        // Shared signatures have a stricter one-result contract, so raw
        // multi-result storage must not be printed in arrow syntax either.
        let shared = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("func"), Symbol::new("func_sig"))
                .params([i32_ty, i32_ty, i32_ty])
                .attr(func::NUM_INPUTS_ATTR, Attribute::Int(1))
                .attr(func::NUM_RESULTS_ATTR, Attribute::Int(2))
                .build(),
        );
        assert_eq!(
            print_type(&ctx, shared),
            "func.func_sig(core.i32, core.i32, core.i32) {num_inputs = 1, num_results = 2}"
        );
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

        let add = arith::addi(&mut ctx, loc, v1, v2, i32_ty);
        // Printer assigns names based on what it sees - standalone op print
        // only numbers the result of THIS op since it can't see c1/c2
        let output = print_op(&ctx, add.op_ref());
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
        let add = arith::addi(&mut ctx, loc, x, y, i32_ty);
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

        // No auto aliases should be generated — core.i32 is a leaf type and
        // func.func_sig has no dialect-provided name hint
        assert!(
            !output.contains('!'),
            "No types should be aliased:\n{output}"
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
    fn test_manual_alias_order_is_independent_of_registration_order() {
        fn print_with_aliases(reverse_registration: bool) -> String {
            let mut ctx = IrContext::new();
            let loc = test_location(&mut ctx);
            let i32_ty = make_i32_type(&mut ctx);
            let alpha_ty = make_adt_struct(&mut ctx, "Alpha", &[("value", i32_ty)]);
            let inner_ty = make_adt_struct(&mut ctx, "Inner", &[("value", i32_ty)]);
            let outer_ty = make_adt_struct(&mut ctx, "Outer", &[("inner", inner_ty)]);
            let zebra_ty = make_adt_struct(&mut ctx, "Zebra", &[("value", i32_ty)]);
            let aliases = [
                (Symbol::new("alpha"), alpha_ty),
                (Symbol::new("a_outer"), outer_ty),
                (Symbol::new("z_inner"), inner_ty),
                (Symbol::new("zebra"), zebra_ty),
            ];

            if reverse_registration {
                for &(name, ty) in aliases.iter().rev() {
                    ctx.register_type_alias(name, ty);
                }
            } else {
                for (name, ty) in aliases {
                    ctx.register_type_alias(name, ty);
                }
            }

            let module = make_module_with_funcs(&mut ctx, loc, vec![]);
            print_module(&ctx, module)
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
        assert_eq!(output, print_module(&reparsed_ctx, reparsed));
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
    fn test_auto_alias_nested_module_isolation() {
        let input = "\
core.module @test {
  core.module @inner {
    func.func @f1(%0: adt.struct() {fields = [[@a, core.i32], [@b, core.i32]], name = @InnerOnly}) -> adt.struct() {fields = [[@a, core.i32], [@b, core.i32]], name = @InnerOnly} {
    ^bb0:
      func.return %0
    }
    func.func @f2(%0: adt.struct() {fields = [[@a, core.i32], [@b, core.i32]], name = @InnerOnly}) -> adt.struct() {fields = [[@a, core.i32], [@b, core.i32]], name = @InnerOnly} {
    ^bb0:
      func.return %0
    }
  }
  func.func @g1(%0: adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @OuterOnly}) -> adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @OuterOnly} {
  ^bb0:
    func.return %0
  }
  func.func @g2(%0: adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @OuterOnly}) -> adt.struct() {fields = [[@x, core.i32], [@y, core.i32]], name = @OuterOnly} {
  ^bb0:
    func.return %0
  }
}
";
        let mut ctx = IrContext::new();
        let root = crate::parser::parse_module(&mut ctx, input).expect("parse failed");
        let output = print_module(&ctx, root);
        insta::assert_snapshot!(output);
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
    fn test_auto_alias_order_is_independent_of_type_interning_order() {
        fn print_module_with_point_types(reverse_interning: bool) -> String {
            let mut ctx = IrContext::new();
            let loc = test_location(&mut ctx);
            let i32_ty = make_i32_type(&mut ctx);
            let make_point =
                |ctx: &mut IrContext, field| make_adt_struct(ctx, "Point", &[(field, i32_ty)]);
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
            print_module(&ctx, module)
        }

        let output = print_module_with_point_types(false);
        assert_eq!(output, print_module_with_point_types(true));
        assert!(
            output.contains("!Point = adt.struct() {fields = [[@alpha, core.i32]], name = @Point}")
        );
        assert!(
            output
                .contains("!Point_1 = adt.struct() {fields = [[@zebra, core.i32]], name = @Point}")
        );

        let mut reparsed_ctx = IrContext::new();
        let reparsed =
            crate::parser::parse_module(&mut reparsed_ctx, &output).expect("parse failed");
        assert_eq!(output, print_module(&reparsed_ctx, reparsed));
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

        let add = arith::addi(&mut ctx, loc, v1, v1, i32_ty);
        ctx.push_op(entry, add.op_ref());

        // RAUW: replace v1 with v2
        ctx.replace_all_uses(v1, v2);

        // Verify operands were updated
        let operands = ctx.op_operands(add.op_ref());
        assert_eq!(operands[0], v2);
        assert_eq!(operands[1], v2);
    }

    #[test]
    fn test_write_escaped_bytes_ascii() {
        let mut buf = String::new();
        write_escaped_bytes(&mut buf, b"hello").unwrap();
        assert_eq!(buf, "hello");
    }

    #[test]
    fn test_write_escaped_bytes_escapes() {
        let mut buf = String::new();
        write_escaped_bytes(&mut buf, b"a\n\t\r\0\\\"b").unwrap();
        assert_eq!(buf, r#"a\n\t\r\0\\\"b"#);
    }

    #[test]
    fn test_write_escaped_bytes_non_ascii() {
        let mut buf = String::new();
        write_escaped_bytes(&mut buf, &[0x00, 0x7f, 0x80, 0xff]).unwrap();
        assert_eq!(buf, r"\0\x7f\x80\xff");
    }

    #[test]
    fn test_print_bytes_attribute() {
        let ctx = IrContext::new();
        let attr = Attribute::Bytes(smallvec::smallvec![104, 101, 108, 108, 111]);
        let mut buf = String::new();
        write_attribute(&ctx, &mut buf, &attr).unwrap();
        assert_eq!(buf, r#"b"hello""#);
    }
}

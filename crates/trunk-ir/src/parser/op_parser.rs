//! [`OpAsmParser`]: the reader of operations, regions, and blocks.

use std::ops::{Deref, DerefMut};

use rustc_hash::FxHashMap as HashMap;

use smallvec::smallvec;
use winnow::Parser as _;

use super::asm_parser::AsmParser;
use super::lexer;
use super::{ParseError, ParseResult};
use crate::context::{IrContext, OperationDataBuilder};
use crate::ops::DialectOp;
use crate::refs::{BlockRef, OpRef, RegionRef, TypeRef, ValueRef};
use crate::types::{Attribute, AttributeMap, Location, StringArg, TypeDataBuilder};
use crate::{BlockArgData, BlockData, BlockList, OpName, RegionData};

/// What an operation's format reads, from which the parser builds the
/// operation, as MLIR's `OperationState` holds it.
///
/// A custom format fills it from [`OpAsmParser`]'s pieces; the parser then
/// creates the operation, reads a declared default as absent, and names the
/// results the text wrote before `=`.
#[derive(Debug)]
pub struct OperationState {
    pub name: OpName,
    pub location: Location,
    pub operands: Vec<ValueRef>,
    pub result_types: Vec<TypeRef>,
    pub attributes: AttributeMap,
    pub regions: Vec<RegionRef>,
    pub successors: Vec<BlockRef>,
}

impl OperationState {
    pub fn new(name: OpName, location: Location) -> Self {
        Self {
            name,
            location,
            operands: Vec::new(),
            result_types: Vec::new(),
            attributes: AttributeMap::new(),
            regions: Vec::new(),
            successors: Vec::new(),
        }
    }

    /// Set the attribute `key`.
    pub fn add_attribute(
        &mut self,
        ctx: &mut IrContext,
        key: impl Into<StringArg>,
        value: impl Into<Attribute>,
    ) {
        let value = value.into();
        self.attributes.view_mut(ctx).insert(key, value);
    }

    /// Set every attribute of `attributes`.
    pub fn add_attributes(&mut self, ctx: &mut IrContext, attributes: AttributeMap) {
        for (key, value) in attributes {
            self.attributes.view_mut(ctx).insert(key, value);
        }
    }

    /// Whether the attribute `key` is set.
    pub fn has_attribute(&self, ctx: &IrContext, key: &str) -> bool {
        self.attributes.view(ctx).contains_key(key)
    }
}

/// A block argument the text names, `%name: T`, as an operation's header
/// writes it before the region whose entry block takes it.
#[derive(Clone, Debug)]
pub struct Argument {
    pub name: String,
    pub ty: TypeRef,
    /// Where the text writes the name.
    pub offset: usize,
}

/// An operand as the text writes it: its value and its name.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Operand<'a> {
    pub value: ValueRef,
    pub name: &'a str,
    pub offset: usize,
}

/// How a region's entry block takes the arguments a header wrote.
#[derive(Clone, Copy)]
enum EntryArgs<'x> {
    None,
    /// The entry block, if the region has one, takes them.
    IfPresent(&'x [Argument]),
    /// The region has an entry block, which takes them.
    Always(&'x [Argument]),
}

/// The names one region defines and refers to.
#[derive(Default)]
struct Scope {
    values: HashMap<String, ValueRef>,
    /// Placeholders for names used before their definition.
    forward: HashMap<String, ValueRef>,
    blocks: HashMap<String, BlockEntry>,
}

struct BlockEntry {
    block: BlockRef,
    defined: bool,
    /// Where the text first refers to it, and in which operation.
    first_use: Option<(usize, String)>,
}

/// A value used before its definition.
struct Placeholder {
    op: OpRef,
    name: String,
    offset: usize,
    op_name: String,
    /// Types the text writes for it, checked at its definition.
    expected: Vec<(TypeRef, usize, String)>,
}

/// The operation being parsed.
struct Frame {
    name: String,
    results: usize,
}

/// Reads operations, regions, and blocks into IR, as MLIR's `OpAsmParser`
/// does, and through [`AsmParser`] types and attributes. It is the parsing
/// counterpart of [`OpAsmPrinter`](crate::printer::OpAsmPrinter): each
/// piece reads what the printer's piece of the same name writes and returns
/// IR.
///
/// Names resolve as they are read. A value name refers to the definition
/// visible in this or an enclosing region; a name used before its
/// definition, such as a later block's argument, gets a placeholder that
/// the definition replaces, and a name never defined is an error at its
/// first use. A block label refers to a block of the same region, created
/// at its first mention and filled where the label is defined.
pub struct OpAsmParser<'a> {
    parser: AsmParser<'a>,
    scopes: Vec<Scope>,
    placeholders: HashMap<ValueRef, Placeholder>,
    frames: Vec<Frame>,
}

impl<'a> Deref for OpAsmParser<'a> {
    type Target = AsmParser<'a>;

    fn deref(&self) -> &Self::Target {
        &self.parser
    }
}

impl DerefMut for OpAsmParser<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.parser
    }
}

impl<'a> OpAsmParser<'a> {
    pub(crate) fn new(ctx: &'a mut IrContext, source: &'a str) -> Self {
        Self {
            parser: AsmParser::new(ctx, source),
            scopes: vec![Scope::default()],
            placeholders: HashMap::default(),
            frames: Vec::new(),
        }
    }

    /// How many result names the text writes before the operation's `=`.
    pub fn num_results(&self) -> usize {
        self.frames.last().map_or(0, |frame| frame.results)
    }

    fn op_name(&self) -> String {
        self.frames
            .last()
            .map(|frame| frame.name.clone())
            .unwrap_or_default()
    }

    // ------------------------------------------------------------------
    // Values
    // ------------------------------------------------------------------

    /// Read an operand, `%name`.
    pub fn parse_operand(&mut self) -> ParseResult<ValueRef> {
        Ok(self.parse_operand_ref()?.value)
    }

    /// Read an operand if one is next.
    pub fn parse_optional_operand(&mut self) -> ParseResult<Option<ValueRef>> {
        Ok(self
            .parse_optional_operand_ref()?
            .map(|operand| operand.value))
    }

    /// Read an operand whose type must be `ty`.
    pub fn parse_operand_of_type(&mut self, ty: TypeRef) -> ParseResult<ValueRef> {
        let operand = self.parse_operand_ref()?;
        self.check_operand_type(operand, ty, operand.offset)?;
        Ok(operand.value)
    }

    /// Read operands separated by commas on the current line, as
    /// [`AsmState::values`](crate::printer::AsmState::values) writes them:
    /// none when the next one would follow a line break, which begins the
    /// next operation.
    pub fn parse_operand_list(&mut self) -> ParseResult<Vec<ValueRef>> {
        Ok(self
            .parse_operand_refs()?
            .into_iter()
            .map(|operand| operand.value)
            .collect())
    }

    pub(crate) fn parse_operand_ref(&mut self) -> ParseResult<Operand<'a>> {
        self.skip_ws();
        let offset = self.offset();
        let name = self.parse_token("an operand", lexer::value_ref)?;
        let value = self.resolve_value(name, offset);
        Ok(Operand {
            value,
            name,
            offset,
        })
    }

    fn parse_optional_operand_ref(&mut self) -> ParseResult<Option<Operand<'a>>> {
        let checkpoint = self.checkpoint();
        if self.peek().starts_with('%') {
            return self.parse_operand_ref().map(Some);
        }
        self.reset(checkpoint);
        Ok(None)
    }

    pub(crate) fn parse_operand_refs(&mut self) -> ParseResult<Vec<Operand<'a>>> {
        let checkpoint = self.checkpoint();
        if self.skip_ws() || !self.peek().starts_with('%') {
            self.reset(checkpoint);
            return Ok(Vec::new());
        }
        let mut operands = vec![self.parse_operand_ref()?];
        loop {
            let checkpoint = self.checkpoint();
            if self.parse_optional_punct(",") && self.peek().starts_with('%') {
                operands.push(self.parse_operand_ref()?);
            } else {
                self.reset(checkpoint);
                return Ok(operands);
            }
        }
    }

    /// Check that the text writes `ty` for `operand`'s type, at `offset`.
    /// A value not yet defined is checked at its definition.
    pub(crate) fn check_operand_type(
        &mut self,
        operand: Operand<'_>,
        ty: TypeRef,
        offset: usize,
    ) -> ParseResult<()> {
        let op_name = self.op_name();
        if let Some(placeholder) = self.placeholders.get_mut(&operand.value) {
            placeholder.expected.push((ty, offset, op_name));
            return Ok(());
        }
        let actual = self.ctx().value_ty(operand.value);
        if actual == ty {
            return Ok(());
        }
        Err(self.operand_type_error(operand.name, &op_name, ty, actual, offset))
    }

    fn operand_type_error(
        &self,
        name: &str,
        op_name: &str,
        written: TypeRef,
        actual: TypeRef,
        offset: usize,
    ) -> ParseError {
        self.error_at(
            offset,
            format!(
                "operand '%{name}' of '{op_name}' is written as {}, but its type is {}",
                crate::printer::print_type(self.ctx(), written),
                crate::printer::print_type(self.ctx(), actual),
            ),
        )
    }

    /// The value `name` refers to, or a placeholder for it.
    fn resolve_value(&mut self, name: &str, offset: usize) -> ValueRef {
        if let Some(value) = self
            .scopes
            .iter()
            .rev()
            .find_map(|scope| scope.values.get(name).copied())
        {
            return value;
        }
        let scope = self.scopes.last().expect("a scope is open");
        if let Some(&placeholder) = scope.forward.get(name) {
            return placeholder;
        }
        let location = self.location();
        let ty = TypeDataBuilder::new("parser.forward_ref").intern(self.ctx_mut());
        let data = OperationDataBuilder::new(location, "parser.forward_ref")
            .result(ty)
            .build(self.ctx_mut());
        let op = self.ctx_mut().create_op(data);
        let value = self.ctx().op_result(op, 0);
        let op_name = self.op_name();
        self.placeholders.insert(
            value,
            Placeholder {
                op,
                name: name.to_owned(),
                offset,
                op_name,
                expected: Vec::new(),
            },
        );
        self.scopes
            .last_mut()
            .expect("a scope is open")
            .forward
            .insert(name.to_owned(), value);
        value
    }

    /// Define `name` as `value` in the current region, replacing a
    /// placeholder its earlier uses got. `duplicate` describes the error
    /// when an enclosing region already defines it.
    fn define_value(
        &mut self,
        name: &str,
        value: ValueRef,
        offset: usize,
        duplicate: impl FnOnce() -> String,
    ) -> ParseResult<()> {
        if self
            .scopes
            .iter()
            .any(|scope| scope.values.contains_key(name))
        {
            return Err(self.error_at(offset, duplicate()));
        }
        let scope = self.scopes.last_mut().expect("a scope is open");
        let placeholder = scope.forward.remove(name);
        scope.values.insert(name.to_owned(), value);
        if let Some(placeholder) = placeholder {
            self.replace_placeholder(placeholder, value)?;
        }
        Ok(())
    }

    fn replace_placeholder(&mut self, placeholder: ValueRef, value: ValueRef) -> ParseResult<()> {
        let entry = self
            .placeholders
            .remove(&placeholder)
            .expect("a forward reference has a placeholder");
        if let Some(other) = self.placeholders.get_mut(&value) {
            // Two regions' forward references to one name meet.
            other.expected.extend(entry.expected);
        } else {
            let actual = self.ctx().value_ty(value);
            if let Some((written, offset, op_name)) = entry
                .expected
                .iter()
                .find(|(written, ..)| *written != actual)
            {
                return Err(self.operand_type_error(
                    &entry.name,
                    op_name,
                    *written,
                    actual,
                    *offset,
                ));
            }
        }
        self.ctx_mut().replace_all_uses(placeholder, value);
        self.ctx_mut().remove_op(entry.op);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Blocks
    // ------------------------------------------------------------------

    /// Read a successor, `^label`, a block of the current region.
    pub fn parse_successor(&mut self) -> ParseResult<BlockRef> {
        self.skip_ws();
        let offset = self.offset();
        let label = self.parse_token("a successor", lexer::block_label)?;
        Ok(self.refer_block(label, offset))
    }

    /// Read a successor if one is next.
    pub fn parse_optional_successor(&mut self) -> ParseResult<Option<BlockRef>> {
        let checkpoint = self.checkpoint();
        if self.peek().starts_with('^') {
            return self.parse_successor().map(Some);
        }
        self.reset(checkpoint);
        Ok(None)
    }

    fn refer_block(&mut self, label: &str, offset: usize) -> BlockRef {
        let op_name = self.op_name();
        let location = self.location();
        let scope = self.scopes.last_mut().expect("a scope is open");
        if let Some(entry) = scope.blocks.get_mut(label) {
            if !entry.defined && entry.first_use.is_none() {
                entry.first_use = Some((offset, op_name));
            }
            return entry.block;
        }
        let block = self.parser.ctx_mut().create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        self.scopes
            .last_mut()
            .expect("a scope is open")
            .blocks
            .insert(
                label.to_owned(),
                BlockEntry {
                    block,
                    defined: false,
                    first_use: Some((offset, op_name)),
                },
            );
        block
    }

    /// The block `label` defines, with `args` as its arguments.
    fn define_block(
        &mut self,
        label: &str,
        offset: usize,
        args: &[Argument],
    ) -> ParseResult<BlockRef> {
        let scope = self.scopes.last_mut().expect("a scope is open");
        let known = scope
            .blocks
            .get_mut(label)
            .map(|entry| (entry.block, std::mem::replace(&mut entry.defined, true)));
        let block = match known {
            Some((_, true)) => {
                return Err(self.error_at(offset, format!("duplicate block label '^{label}'")));
            }
            Some((block, false)) => {
                self.add_block_args(block, args)?;
                return Ok(block);
            }
            None => self.new_block(args)?,
        };
        self.scopes
            .last_mut()
            .expect("a scope is open")
            .blocks
            .insert(
                label.to_owned(),
                BlockEntry {
                    block,
                    defined: true,
                    first_use: None,
                },
            );
        Ok(block)
    }

    /// A new block with `args` as its arguments.
    fn new_block(&mut self, args: &[Argument]) -> ParseResult<BlockRef> {
        let data = self.block_args(args)?;
        let location = self.location();
        let block = self.ctx_mut().create_block(BlockData {
            location,
            args: data,
            ops: smallvec![],
            parent_region: None,
        });
        self.define_block_args(block, args)?;
        Ok(block)
    }

    fn add_block_args(&mut self, block: BlockRef, args: &[Argument]) -> ParseResult<()> {
        for data in self.block_args(args)? {
            self.ctx_mut().add_block_arg(block, data);
        }
        self.define_block_args(block, args)
    }

    /// The arguments `args` declare. An argument whose name is not its
    /// position's default, `%argN`, keeps its name as `bind_name`.
    fn block_args(&mut self, args: &[Argument]) -> ParseResult<Vec<BlockArgData>> {
        let mut data = Vec::with_capacity(args.len());
        for (index, arg) in args.iter().enumerate() {
            if args[..index].iter().any(|other| other.name == arg.name) {
                return Err(self.error_at(
                    arg.offset,
                    format!(
                        "duplicate block argument name '{}' at index {index}",
                        arg.name
                    ),
                ));
            }
            let is_default_name = arg
                .name
                .strip_prefix("arg")
                .and_then(|rest| rest.parse::<usize>().ok())
                .is_some_and(|n| n == index);
            let mut attrs = AttributeMap::new();
            if !is_default_name {
                let name = self.ctx_mut().string_attr(&arg.name);
                attrs.view_mut(self.ctx_mut()).insert("bind_name", name);
            }
            data.push(BlockArgData { ty: arg.ty, attrs });
        }
        Ok(data)
    }

    fn define_block_args(&mut self, block: BlockRef, args: &[Argument]) -> ParseResult<()> {
        for (index, arg) in args.iter().enumerate() {
            let value = self.ctx().block_arg(block, index as u32);
            self.define_value(&arg.name, value, arg.offset, || {
                format!("duplicate SSA name '%{}' in block argument", arg.name)
            })?;
        }
        Ok(())
    }

    /// Read a block argument, `%name: T`.
    pub fn parse_argument(&mut self) -> ParseResult<Argument> {
        Self::parse_argument_of(&mut self.parser)
    }

    /// Read the entry block arguments
    /// [`AsmState::entry_args`](crate::printer::AsmState::entry_args)
    /// writes, `(%a: A, ..)`, for a region read after them with
    /// [`Self::parse_region_with_args`] or [`Self::parse_region_with_entry`].
    pub fn parse_entry_args(&mut self) -> ParseResult<Vec<Argument>> {
        self.parse_delimited("(", ")", Self::parse_argument_of)
    }

    fn parse_argument_of(p: &mut AsmParser<'a>) -> ParseResult<Argument> {
        p.skip_ws();
        let offset = p.offset();
        let name = p.parse_token("an argument `%name`", lexer::value_ref)?;
        p.parse_punct(":")?;
        let ty = p.parse_type()?;
        Ok(Argument {
            name: name.to_owned(),
            ty,
            offset,
        })
    }

    // ------------------------------------------------------------------
    // Regions
    // ------------------------------------------------------------------

    /// Read a region in generic form, `{ .. }`, block labels included, as
    /// [`OpAsmPrinter::print_region`](crate::printer::OpAsmPrinter::print_region)
    /// writes it.
    pub fn parse_region(&mut self) -> ParseResult<RegionRef> {
        self.region(EntryArgs::None, false)
    }

    /// Read a region if one is next.
    pub fn parse_optional_region(&mut self) -> ParseResult<Option<RegionRef>> {
        let checkpoint = self.checkpoint();
        if self.peek().starts_with('{') {
            return self.parse_region().map(Some);
        }
        self.reset(checkpoint);
        Ok(None)
    }

    /// Read a region whose entry block takes `args`, which a header wrote
    /// before it, as
    /// [`OpAsmPrinter::print_region_without_entry`](crate::printer::OpAsmPrinter::print_region_without_entry)
    /// writes it. A region without blocks, `{}`, stays without them. An
    /// entry block that writes its own arguments must write as many, of the
    /// same types; it keeps its own names.
    pub fn parse_region_with_args(&mut self, args: &[Argument]) -> ParseResult<RegionRef> {
        self.region(EntryArgs::IfPresent(args), false)
    }

    /// [`Self::parse_region_with_args`] for a region that always has an
    /// entry block: `{}` reads as an entry block without operations.
    pub fn parse_region_with_entry(&mut self, args: &[Argument]) -> ParseResult<RegionRef> {
        self.region(EntryArgs::Always(args), false)
    }

    /// Read a module's body region, which begins with its type alias
    /// definitions.
    fn parse_module_region(&mut self) -> ParseResult<RegionRef> {
        self.region(EntryArgs::None, true)
    }

    fn region(&mut self, entry: EntryArgs<'_>, module: bool) -> ParseResult<RegionRef> {
        self.parse_punct("{")?;
        self.scopes.push(Scope::default());
        self.push_alias_scope();
        let result = self.region_body(entry, module);
        self.pop_alias_scope();
        let scope = self.scopes.pop().expect("the region's scope");
        let region = result?;
        self.close_scope(scope)?;
        Ok(region)
    }

    fn region_body(&mut self, entry: EntryArgs<'_>, module: bool) -> ParseResult<RegionRef> {
        self.skip_ws();
        while lexer::is_type_alias_def(self.peek()) {
            if !module {
                return Err(self.error("type aliases are only allowed in module regions"));
            }
            self.parse_alias_def()?;
        }

        let given = match entry {
            EntryArgs::None => &[][..],
            EntryArgs::IfPresent(args) | EntryArgs::Always(args) => args,
        };
        let mut blocks = BlockList::new();
        let rest = self.peek();
        if !rest.starts_with('^') && !rest.starts_with('}') && !rest.is_empty() {
            // An entry block without arguments of its own may omit its
            // label, as in MLIR.
            let block = self.new_block(given)?;
            blocks.push(block);
            self.parse_block_ops(block)?;
        } else if rest.starts_with('}') && matches!(entry, EntryArgs::Always(_)) {
            blocks.push(self.new_block(given)?);
        }
        while self.peek().starts_with('^') {
            self.skip_ws();
            let offset = self.offset();
            let label = self.parse_token("a block label", lexer::block_label)?;
            let written = if self.peek().starts_with('(') {
                self.parse_delimited("(", ")", Self::parse_argument_of)?
            } else {
                Vec::new()
            };
            self.parse_punct(":")?;
            let args = if blocks.is_empty() {
                self.entry_block_args(written, given)?
            } else {
                written
            };
            let block = self.define_block(label, offset, &args)?;
            blocks.push(block);
            self.parse_block_ops(block)?;
        }
        self.parse_punct("}")?;

        let location = self.location();
        Ok(self.ctx_mut().create_region(RegionData {
            location,
            blocks,
            parent_op: None,
        }))
    }

    /// The arguments of an entry block that writes `written` in a region
    /// whose header wrote `given`.
    fn entry_block_args(
        &self,
        written: Vec<Argument>,
        given: &[Argument],
    ) -> ParseResult<Vec<Argument>> {
        if given.is_empty() {
            return Ok(written);
        }
        if written.is_empty() {
            return Ok(given.to_vec());
        }
        if written.len() != given.len() {
            return Err(self.error(format!(
                "entry block has {} args but function signature has {} params",
                written.len(),
                given.len()
            )));
        }
        for (index, (block, param)) in written.iter().zip(given).enumerate() {
            if block.ty != param.ty {
                let ctx = self.ctx();
                return Err(self.error_at(
                    block.offset,
                    format!(
                        "entry block arg {index} type mismatch: block has {} but function param has {}",
                        ctx.get_type(block.ty).name.display(ctx),
                        ctx.get_type(param.ty).name.display(ctx),
                    ),
                ));
            }
        }
        Ok(written)
    }

    fn parse_block_ops(&mut self, block: BlockRef) -> ParseResult<()> {
        loop {
            let rest = self.peek();
            if rest.starts_with('^') || rest.starts_with('}') || rest.is_empty() {
                return Ok(());
            }
            let op = self.parse_operation()?;
            self.ctx_mut().push_op(block, op);
        }
    }

    /// Check what a closed region left: every block it refers to is
    /// defined, and its forward references go to the enclosing region,
    /// whose later definitions may still resolve them.
    fn close_scope(&mut self, scope: Scope) -> ParseResult<()> {
        if let Some((label, (offset, op_name))) = scope
            .blocks
            .iter()
            .filter(|(_, entry)| !entry.defined)
            .filter_map(|(label, entry)| Some((label, entry.first_use.clone()?)))
            .min_by_key(|(_, (offset, _))| *offset)
        {
            return Err(self.error_at(
                offset,
                format!("undefined block '^{label}' in successor list of '{op_name}'"),
            ));
        }
        let mut forward: Vec<_> = scope.forward.into_iter().collect();
        forward.sort_by_key(|(_, value)| self.placeholders[value].offset);
        for (name, placeholder) in forward {
            let Some(parent) = self.scopes.last_mut() else {
                let entry = &self.placeholders[&placeholder];
                return Err(self.error_at(
                    entry.offset,
                    format!(
                        "undefined value '%{}' in operation '{}'",
                        entry.name, entry.op_name
                    ),
                ));
            };
            match parent.forward.get(&name) {
                Some(&other) => self.replace_placeholder(placeholder, other)?,
                None => {
                    parent.forward.insert(name, placeholder);
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Operations
    // ------------------------------------------------------------------

    /// Read an operation: the generic form after a quoted name, or the
    /// operation's custom syntax after a bare one.
    ///
    /// ```text
    /// [results =] "dialect.op"(operands)[successors] (regions) {attrs} : types
    /// [results =] dialect.op custom-syntax
    /// ```
    pub(crate) fn parse_operation(&mut self) -> ParseResult<OpRef> {
        self.skip_ws();
        let start = self.offset();
        let source_len = self.offset() + self.checkpoint().len();
        let results: Vec<(&'a str, usize)> = self
            .parse_optional_token("result names", lexer::result_list)?
            .unwrap_or_default()
            .into_iter()
            .map(|(name, remaining)| (name, source_len - remaining))
            .collect();

        self.skip_ws();
        let name_offset = self.offset();
        let quoted = self.peek().starts_with('"');
        let name = if quoted {
            self.parse_token("a quoted operation name", |input| {
                winnow::combinator::delimited('"', lexer::qualified_name, '"').parse_next(input)
            })?
        } else {
            self.parse_token("an operation name", lexer::qualified_name)?
        };
        self.frames.push(Frame {
            name: name.to_owned(),
            results: results.len(),
        });
        let result = self.parse_operation_body(name, quoted, name_offset);
        self.frames.pop();
        let state = result?;

        if !results.is_empty() && results.len() != state.result_types.len() {
            return Err(self.error_at(
                start,
                format!(
                    "operation '{name}' declares {} result names but {} result types",
                    results.len(),
                    state.result_types.len()
                ),
            ));
        }
        let op = self.create_operation(state);
        for (index, (result, offset)) in results.into_iter().enumerate() {
            let value = self.ctx().op_result(op, index as u32);
            self.define_value(result, value, offset, || {
                format!("duplicate SSA name '{result}' in operation '{name}' result index {index}")
            })?;
        }
        Ok(op)
    }

    fn parse_operation_body(
        &mut self,
        name: &'a str,
        quoted: bool,
        name_offset: usize,
    ) -> ParseResult<OperationState> {
        let op_name = OpName::intern(self.ctx_mut().strings_mut(), name);
        let mut state = OperationState::new(op_name, self.location());
        if quoted {
            self.parse_generic_body(&mut state)?;
            return Ok(state);
        }
        if op_name == crate::dialect::core::Module::NAME {
            self.parse_module_body(&mut state)?;
            return Ok(state);
        }
        let format = crate::OpDef::lookup(name).and_then(|def| Some((def, def.asm_format?)));
        match format {
            Some((def, crate::asm_format::OpAsmFormatDef::Declarative(format))) => {
                crate::asm_format::declarative::parse(self, &mut state, def, format)?;
            }
            Some((_, crate::asm_format::OpAsmFormatDef::Hooks(hooks))) => {
                (hooks.parse)(self, &mut state)?;
            }
            None => {
                return Err(self.error_at(
                    name_offset,
                    "expected a quoted name: an operation without a custom format is written in generic form",
                ));
            }
        }
        Ok(state)
    }

    /// The generic form after the name:
    /// `(operands)[successors] (regions) {attrs} : types`.
    fn parse_generic_body(&mut self, state: &mut OperationState) -> ParseResult<()> {
        self.parse_punct("(")?;
        let operands = self.parse_list(")", |p| {
            p.skip_ws();
            let offset = p.offset();
            let name = p.parse_token("an operand", lexer::value_ref)?;
            Ok((name, offset))
        })?;
        self.parse_punct(")")?;
        for (name, offset) in operands {
            let value = self.resolve_value(name, offset);
            state.operands.push(value);
        }
        // A successor list follows the operands directly.
        if self.checkpoint().starts_with('[') {
            self.parse_punct("[")?;
            loop {
                let block = self.parse_successor()?;
                state.successors.push(block);
                if !self.parse_optional_punct(",") {
                    break;
                }
            }
            self.parse_punct("]")?;
        }
        if self.parse_optional_punct("(") {
            loop {
                let region = self.parse_region()?;
                state.regions.push(region);
                if !self.parse_optional_punct(",") {
                    break;
                }
            }
            self.parse_punct(")")?;
        }
        if let Some(attributes) = self.parse_optional_attr_dict()? {
            state.attributes = attributes;
        }
        if self.parse_optional_punct(":") {
            loop {
                let ty = self.parse_type()?;
                state.result_types.push(ty);
                if !self.parse_optional_punct(",") {
                    break;
                }
            }
        }
        Ok(())
    }

    /// `core.module` after its name: `[@name] [{attrs}] {body}`.
    fn parse_module_body(&mut self, state: &mut OperationState) -> ParseResult<()> {
        let sym_name = self.parse_optional_symbol_name()?;
        // A dictionary before the body holds the module's attributes; the
        // body itself does not read as one.
        let checkpoint = self.checkpoint();
        match self.parse_optional_attr_dict() {
            Ok(Some(attributes)) => state.attributes = attributes,
            Ok(None) => {}
            Err(_) => self.reset(checkpoint),
        }
        if let Some(name) = sym_name {
            let name = self.ctx_mut().string_attr(&name);
            state.add_attribute(self.ctx_mut(), "sym_name", name);
        }
        while self.peek().starts_with('{') {
            let region = self.parse_module_region()?;
            state.regions.push(region);
        }
        Ok(())
    }

    fn create_operation(&mut self, state: OperationState) -> OpRef {
        let OperationState {
            name,
            location,
            operands,
            result_types,
            attributes,
            regions,
            successors,
        } = state;
        let mut builder = OperationDataBuilder::new(location, name)
            .operands(operands)
            .results(result_types);
        for (key, value) in attributes {
            builder = builder.attr(key, value);
        }
        for region in regions {
            builder = builder.region(region);
        }
        for block in successors {
            builder = builder.successor(block);
        }
        let data = builder.build(self.ctx_mut());
        let op = self.ctx_mut().create_op(data);
        // A declared default is read in its canonical form, absent.
        if let Some(def) = crate::op_def::OpDef::of(self.ctx(), op) {
            def.schema.remove_defaults(self.ctx_mut(), op);
        }
        op
    }

    /// Read the whole text as one operation.
    pub(crate) fn parse_top_level(mut self) -> ParseResult<OpRef> {
        let op = self.parse_operation()?;
        if !self.at_end() {
            return Err(self.error("trailing input after top-level operation"));
        }
        let scope = self.scopes.pop().expect("the top-level scope");
        self.close_scope(scope)?;
        Ok(op)
    }
}

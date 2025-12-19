//! CST to TrunkIR lowering pass.
//!
//! This pass converts Tree-sitter CST directly to TrunkIR operations,
//! bypassing the AST intermediate representation.
//! At this stage, names are unresolved (using `src` dialect ops).
//!
//! ## Pipeline
//!
//! The lowering is split into two Salsa-tracked stages:
//! 1. `parse_cst` - Parse source to CST (cached by Salsa)
//! 2. `lower_cst` - Lower CST to TrunkIR module
//!
//! This allows Salsa to cache the CST independently from the TrunkIR output.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use tree_sitter::{Node, Parser, Tree};
use tribute_core::{Location, PathId, SourceFile, Span};
use tribute_trunk_ir::{
    Attribute, Block, BlockBuilder, DialectOp, DialectType, IdVec, Operation, Region, Symbol, Type,
    Value,
    dialect::{ability, adt, arith, case, core, func, list, pat, src, ty},
    idvec,
};

// =============================================================================
// Parsed CST (Salsa-cacheable)
// =============================================================================

/// A parsed CST tree, wrapped for Salsa caching.
///
/// Tree-sitter's `Tree` is internally reference-counted (`ts_tree_copy` is O(1)),
/// so cloning is cheap and we can use it directly without additional wrapping.
#[derive(Clone, Debug)]
pub struct ParsedCst(Tree);

impl ParsedCst {
    /// Create a new ParsedCst from a tree-sitter Tree.
    pub fn new(tree: Tree) -> Self {
        Self(tree)
    }

    /// Get a reference to the underlying tree.
    pub fn tree(&self) -> &Tree {
        &self.0
    }

    /// Get the root node of the CST.
    pub fn root_node(&self) -> Node<'_> {
        self.0.root_node()
    }
}

impl PartialEq for ParsedCst {
    fn eq(&self, other: &Self) -> bool {
        // Trees from the same parse are equal if they have the same root node id
        self.0.root_node().id() == other.0.root_node().id()
    }
}

impl Eq for ParsedCst {}

impl Hash for ParsedCst {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by root node id
        self.0.root_node().id().hash(state);
    }
}

/// Create a symbol from a string.
fn sym<'db>(db: &'db dyn salsa::Database, name: &str) -> Symbol<'db> {
    Symbol::new(db, name)
}

/// Create a symbol reference (path) from a single name.
fn sym_ref<'db>(db: &'db dyn salsa::Database, name: &str) -> IdVec<Symbol<'db>> {
    idvec![Symbol::new(db, name)]
}

// =============================================================================
// CST Navigation Helpers
// =============================================================================

/// Check if a node is a comment that should be skipped.
fn is_comment(kind: &str) -> bool {
    matches!(
        kind,
        "line_comment" | "block_comment" | "line_doc_comment" | "block_doc_comment"
    )
}

/// Get text from a node.
fn node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    node.utf8_text(source.as_bytes()).unwrap_or("")
}

/// Create a Span from a tree-sitter Node.
fn span_from_node(node: &Node) -> Span {
    Span::new(node.start_byte(), node.end_byte())
}

// =============================================================================
// Lowering Context
// =============================================================================

/// Context for lowering, tracking local variable bindings and type variable generation.
struct CstLoweringCtx<'db, 'src> {
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    source: &'src str,
    /// Map from variable names to their SSA values.
    bindings: HashMap<String, Value<'db>>,
    /// Map from type variable names to their Type representations.
    type_var_bindings: HashMap<String, Type<'db>>,
    /// Counter for generating unique type variable IDs.
    next_type_var_id: u64,
}

impl<'db, 'src> CstLoweringCtx<'db, 'src> {
    fn new(db: &'db dyn salsa::Database, path: PathId<'db>, source: &'src str) -> Self {
        Self {
            db,
            path,
            source,
            bindings: HashMap::new(),
            type_var_bindings: HashMap::new(),
            next_type_var_id: 0,
        }
    }

    /// Generate a fresh type variable with a unique ID.
    fn fresh_type_var(&mut self) -> Type<'db> {
        let id = self.next_type_var_id;
        self.next_type_var_id += 1;
        ty::var_with_id(self.db, id)
    }

    /// Get or create a named type variable.
    /// Same name always returns the same type variable within a scope.
    fn named_type_var(&mut self, name: &str) -> Type<'db> {
        if let Some(&ty) = self.type_var_bindings.get(name) {
            ty
        } else {
            let ty = self.fresh_type_var();
            self.type_var_bindings.insert(name.to_string(), ty);
            ty
        }
    }

    /// Resolve a type node to an IR Type.
    fn resolve_type_node(&mut self, node: Node) -> Type<'db> {
        let mut cursor = node.walk();
        match node.kind() {
            "type_identifier" => {
                // Concrete named type
                let name = node_text(&node, self.source);
                src::unresolved_type(self.db, name, idvec![])
            }
            "type_variable" => {
                // Type variable (lowercase)
                let name = node_text(&node, self.source);
                self.named_type_var(name)
            }
            "generic_type" => {
                // Generic type: List(a), Option(b)
                let mut name = None;
                let mut args = Vec::new();

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "type_identifier" if name.is_none() => {
                            name = Some(node_text(&child, self.source));
                        }
                        "type_variable" | "type_identifier" | "generic_type" => {
                            args.push(self.resolve_type_node(child));
                        }
                        _ => {}
                    }
                }

                let name = name.unwrap_or("Unknown");
                let params: IdVec<Type<'db>> = args.into_iter().collect();
                src::unresolved_type(self.db, name, params)
            }
            _ => {
                // Fallback to fresh type var
                self.fresh_type_var()
            }
        }
    }

    /// Bind a name to a value.
    fn bind(&mut self, name: String, value: Value<'db>) {
        self.bindings.insert(name, value);
    }

    /// Look up a binding by name.
    fn lookup(&self, name: &str) -> Option<Value<'db>> {
        self.bindings.get(name).copied()
    }

    /// Execute a closure in a new scope. Bindings created inside are discarded after.
    fn scoped<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let saved_bindings = self.bindings.clone();
        let saved_type_vars = self.type_var_bindings.clone();
        let result = f(self);
        self.bindings = saved_bindings;
        self.type_var_bindings = saved_type_vars;
        result
    }

    /// Create a Location from a node.
    fn location(&self, node: &Node) -> Location<'db> {
        Location::new(self.path, span_from_node(node))
    }
}

// =============================================================================
// Use Declaration Lowering
// =============================================================================

#[derive(Debug)]
struct UseImport {
    path: Vec<String>,
    alias: Option<String>,
}

fn lower_use_decl<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
    block: &mut BlockBuilder<'db>,
) {
    let location = ctx.location(&node);
    let is_pub = node
        .named_children(&mut node.walk())
        .any(|child| child.kind() == "visibility_marker");

    let Some(tree_node) = node.child_by_field_name("tree") else {
        return;
    };

    let mut imports = Vec::new();
    collect_use_imports(ctx, tree_node, &mut Vec::new(), &mut imports);

    for import in imports {
        let path: IdVec<Symbol<'db>> = import
            .path
            .iter()
            .map(|segment| sym(ctx.db, segment))
            .collect();

        if path.is_empty() {
            continue;
        }

        let alias_sym = import
            .alias
            .as_ref()
            .map(|alias| sym(ctx.db, alias))
            .unwrap_or_else(|| sym(ctx.db, ""));

        block.op(src::r#use(ctx.db, location, path, alias_sym, is_pub));
    }
}

fn collect_use_imports<'db, 'src>(
    ctx: &CstLoweringCtx<'db, 'src>,
    node: Node,
    base: &mut Vec<String>,
    out: &mut Vec<UseImport>,
) {
    match node.kind() {
        "use_group" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if child.kind() == "use_tree" {
                    collect_use_imports(ctx, child, base, out);
                }
            }
        }
        "use_tree" => {
            let alias_node = node.child_by_field_name("alias");
            let alias_id = alias_node.as_ref().map(|n| n.id());
            let alias = alias_node.map(|n| node_text(&n, ctx.source).to_string());

            let mut cursor = node.walk();
            let mut head = None;
            let mut group_node = None;
            let mut tail_node = None;

            for child in node.named_children(&mut cursor) {
                if alias_id == Some(child.id()) {
                    continue;
                }
                match child.kind() {
                    "identifier" | "type_identifier" | "path_keyword" if head.is_none() => {
                        head = Some(node_text(&child, ctx.source).to_string());
                    }
                    "use_group" => group_node = Some(child),
                    "use_tree" => tail_node = Some(child),
                    _ => {}
                }
            }

            let Some(head) = head else {
                return;
            };

            if let Some(group) = group_node {
                let mut new_base = base.clone();
                new_base.push(head);
                collect_use_imports(ctx, group, &mut new_base, out);
                return;
            }

            if let Some(tail) = tail_node {
                let mut new_base = base.clone();
                new_base.push(head);
                collect_use_imports(ctx, tail, &mut new_base, out);
                return;
            }

            if head == "self" && !base.is_empty() {
                out.push(UseImport {
                    path: base.clone(),
                    alias,
                });
            } else {
                let mut path = base.clone();
                path.push(head);
                out.push(UseImport { path, alias });
            }
        }
        _ => {}
    }
}

// =============================================================================
// Entry Points (Salsa-tracked)
// =============================================================================

/// Parse a source file to CST.
///
/// This is the first stage of the compilation pipeline. The resulting
/// `ParsedCst` is cached by Salsa and will only be recomputed when
/// the source file changes.
#[salsa::tracked]
pub fn parse_cst(db: &dyn salsa::Database, source: SourceFile) -> Option<ParsedCst> {
    let text = source.text(db);

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_tribute::LANGUAGE.into())
        .expect("Failed to set language");

    parser.parse(text, None).map(ParsedCst::new)
}

/// Lower a parsed CST to TrunkIR module.
///
/// This is the second stage of the compilation pipeline. It takes
/// the parsed CST and source file (for text extraction) and produces
/// a TrunkIR module.
#[salsa::tracked]
pub fn lower_cst<'db>(
    db: &'db dyn salsa::Database,
    source: SourceFile,
    cst: ParsedCst,
) -> core::Module<'db> {
    let path = PathId::new(db, source.path(db));
    let text = source.text(db);
    let root = cst.root_node();
    let location = Location::new(path, span_from_node(&root));

    lower_cst_impl(db, path, text, root, location)
}

/// Lower a source file directly from CST to TrunkIR module.
///
/// This is a convenience function that combines `parse_cst` and `lower_cst`.
/// For fine-grained caching control, use the two functions separately.
#[salsa::tracked]
pub fn lower_source_file<'db>(
    db: &'db dyn salsa::Database,
    source: SourceFile,
) -> core::Module<'db> {
    let path = PathId::new(db, source.path(db));

    match parse_cst(db, source) {
        Some(cst) => lower_cst(db, source, cst),
        None => {
            // Return empty module on parse failure
            let location = Location::new(path, Span::new(0, 0));
            core::Module::build(db, location, "main", |_| {})
        }
    }
}

/// Internal implementation of CST lowering.
fn lower_cst_impl<'db>(
    db: &'db dyn salsa::Database,
    path: PathId<'db>,
    text: &str,
    root: Node<'_>,
    location: Location<'db>,
) -> core::Module<'db> {
    core::Module::build(db, location, "main", |top| {
        let mut cursor = root.walk();
        let mut ctx = CstLoweringCtx::new(db, path, text);

        for child in root.named_children(&mut cursor) {
            if is_comment(child.kind()) {
                continue;
            }
            match child.kind() {
                "function_definition" => {
                    if let Some(func) = lower_function(&mut ctx, child) {
                        top.op(func);
                    }
                }
                "struct_declaration" => {
                    if let Some(struct_op) = lower_struct_decl(&mut ctx, child) {
                        top.op(struct_op);
                    }
                }
                "enum_declaration" => {
                    if let Some(enum_op) = lower_enum_decl(&mut ctx, child) {
                        top.op(enum_op);
                    }
                }
                "const_declaration" => {
                    if let Some(const_op) = lower_const_decl(&mut ctx, top, child) {
                        top.op(const_op);
                    }
                }
                "ability_declaration" => {
                    if let Some(ability_op) = lower_ability_decl(&mut ctx, child) {
                        top.op(ability_op);
                    }
                }
                "mod_declaration" => {
                    if let Some(mod_op) = lower_mod_decl(&mut ctx, child) {
                        top.op(mod_op);
                    }
                }
                "use_declaration" => {
                    lower_use_decl(&mut ctx, child, top);
                }
                _ => {}
            }
        }
    })
}

// =============================================================================
// Function Lowering
// =============================================================================

/// Lower a function definition to a func.func operation.
fn lower_function<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Option<func::Func<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);

    let mut name = None;
    let mut name_span = None;
    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut return_type = None;
    let mut body_node = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "identifier" if name.is_none() => {
                name = Some(node_text(&child, ctx.source).to_string());
                name_span = Some(Span {
                    start: child.start_byte(),
                    end: child.end_byte(),
                });
            }
            "parameter_list" => {
                let (names, types) = parse_parameter_list(ctx, child);
                param_names = names;
                param_types = types;
            }
            "return_type_annotation" => {
                return_type = parse_return_type(ctx, child);
            }
            "block" => {
                body_node = Some(child);
            }
            _ => {}
        }
    }

    let name = name?;
    let body_node = body_node?;

    // Resolve parameter types
    let params: IdVec<Type> = param_types.into_iter().collect();

    // Resolve return type or create fresh type var
    let result = return_type.unwrap_or_else(|| ctx.fresh_type_var());

    Some(func::Func::build_with_name_span(
        ctx.db,
        location,
        &name,
        name_span,
        params.clone(),
        result,
        |entry| {
            // Bind parameters
            for (i, param_name) in param_names.iter().enumerate() {
                let infer_ty = ctx.fresh_type_var();
                let param_value = entry.op(src::var(
                    ctx.db,
                    location,
                    infer_ty,
                    sym(ctx.db, param_name),
                ));
                ctx.bind(param_name.clone(), param_value.result(ctx.db));
                let _ = i;
            }

            // Lower body statements
            let last_value = lower_block_body(ctx, entry, body_node);

            // Return
            if let Some(value) = last_value {
                entry.op(func::Return::value(ctx.db, location, value));
            } else {
                entry.op(func::Return::empty(ctx.db, location));
            }
        },
    ))
}

// =============================================================================
// Type Declaration Lowering
// =============================================================================

/// Lower a struct declaration to type.struct.
fn lower_struct_decl<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Option<ty::Struct<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut name = None;
    let mut fields = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "type_identifier" if name.is_none() => {
                name = Some(node_text(&child, ctx.source).to_string());
            }
            "struct_body" | "record_fields" => {
                fields = parse_struct_fields(ctx, child);
            }
            _ => {}
        }
    }

    let name = name?;
    let fields_attr = Attribute::List(
        fields
            .into_iter()
            .map(|(field_name, field_type)| {
                Attribute::List(vec![
                    Attribute::Symbol(sym(ctx.db, &field_name)),
                    Attribute::Symbol(sym(ctx.db, &format!("{:?}", field_type))),
                ])
            })
            .collect(),
    );

    Some(ty::r#struct(
        ctx.db,
        location,
        infer_ty,
        Attribute::Symbol(sym(ctx.db, &name)),
        fields_attr,
    ))
}

/// Parse struct fields from struct_body or record_fields.
fn parse_struct_fields<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Vec<(String, Type<'db>)> {
    let mut cursor = node.walk();
    let mut fields = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if child.kind() == "struct_field" || child.kind() == "record_field" {
            let mut field_cursor = child.walk();
            let mut field_name = None;
            let mut field_type = None;

            for field_child in child.named_children(&mut field_cursor) {
                if is_comment(field_child.kind()) {
                    continue;
                }
                match field_child.kind() {
                    "identifier" if field_name.is_none() => {
                        field_name = Some(node_text(&field_child, ctx.source).to_string());
                    }
                    "type_identifier" | "type_variable" | "generic_type" => {
                        field_type = Some(ctx.resolve_type_node(field_child));
                    }
                    _ => {}
                }
            }

            if let Some(name) = field_name {
                fields.push((name, field_type.unwrap_or_else(|| ctx.fresh_type_var())));
            }
        }
    }

    fields
}

/// Lower an enum declaration to type.enum.
fn lower_enum_decl<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Option<ty::Enum<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut name = None;
    let mut variants = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "type_identifier" if name.is_none() => {
                name = Some(node_text(&child, ctx.source).to_string());
            }
            "enum_body" => {
                variants = parse_enum_variants(ctx, child);
            }
            _ => {}
        }
    }

    let name = name?;
    let variants_attr = Attribute::List(
        variants
            .into_iter()
            .map(|(variant_name, variant_fields)| {
                Attribute::List(vec![
                    Attribute::Symbol(sym(ctx.db, &variant_name)),
                    Attribute::List(
                        variant_fields
                            .into_iter()
                            .map(|(f_name, f_type)| {
                                Attribute::List(vec![
                                    Attribute::Symbol(sym(ctx.db, &f_name)),
                                    Attribute::Symbol(sym(ctx.db, &format!("{:?}", f_type))),
                                ])
                            })
                            .collect(),
                    ),
                ])
            })
            .collect(),
    );

    Some(ty::r#enum(
        ctx.db,
        location,
        infer_ty,
        Attribute::Symbol(sym(ctx.db, &name)),
        variants_attr,
    ))
}

/// Parse enum variants from enum_body.
fn parse_enum_variants<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Vec<(String, Vec<(String, Type<'db>)>)> {
    let mut cursor = node.walk();
    let mut variants = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if child.kind() == "enum_variant" {
            let mut variant_cursor = child.walk();
            let mut variant_name = None;
            let mut variant_fields = Vec::new();

            for variant_child in child.named_children(&mut variant_cursor) {
                if is_comment(variant_child.kind()) {
                    continue;
                }
                match variant_child.kind() {
                    "type_identifier" if variant_name.is_none() => {
                        variant_name = Some(node_text(&variant_child, ctx.source).to_string());
                    }
                    "tuple_fields" => {
                        // Positional fields: Variant(Int, String)
                        let mut field_cursor = variant_child.walk();
                        let mut idx = 0;
                        for field_child in variant_child.named_children(&mut field_cursor) {
                            if !is_comment(field_child.kind()) {
                                let field_type = ctx.resolve_type_node(field_child);
                                variant_fields.push((format!("_{}", idx), field_type));
                                idx += 1;
                            }
                        }
                    }
                    "struct_body" | "record_fields" => {
                        // Named fields: Variant { x: Int, y: String }
                        variant_fields = parse_struct_fields(ctx, variant_child);
                    }
                    _ => {}
                }
            }

            if let Some(name) = variant_name {
                variants.push((name, variant_fields));
            }
        }
    }

    variants
}

/// Lower a const declaration to src.const.
fn lower_const_decl<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    _block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<src::Const<'db>> {
    let location = ctx.location(&node);

    // Use field access to get the named children
    let name_node = node.child_by_field_name("name")?;
    let value_node = node.child_by_field_name("value")?;
    let type_node = node.child_by_field_name("type");

    let name = node_text(&name_node, ctx.source).to_string();
    let result_type = type_node
        .map(|n| ctx.resolve_type_node(n))
        .unwrap_or_else(|| ctx.fresh_type_var());

    // Extract literal value directly as an Attribute (no arith.const generated)
    let value_attr = literal_to_attribute(ctx, value_node)?;

    Some(src::r#const(
        ctx.db,
        location,
        result_type,
        sym(ctx.db, &name),
        value_attr,
    ))
}

/// Convert a literal CST node to an Attribute.
fn literal_to_attribute<'db, 'src>(
    ctx: &CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Option<Attribute<'db>> {
    // Unwrap expression wrapper nodes to get the actual literal
    let actual_node = unwrap_expression_node(node, ctx.source)?;
    let text = node_text(&actual_node, ctx.source);

    match actual_node.kind() {
        "nat_literal" => {
            let n = parse_nat_literal(text)?;
            Some(Attribute::IntBits(n))
        }
        "int_literal" => {
            let n = parse_int_literal(text)?;
            Some(Attribute::IntBits(n as u64))
        }
        "float_literal" => {
            let n = parse_float_literal(text)?;
            Some(Attribute::FloatBits(n.to_bits()))
        }
        "rune" => {
            let ch = parse_rune_literal(text)?;
            Some(Attribute::IntBits(ch as u64))
        }
        "string" | "raw_string" | "multiline_string" => {
            let s = parse_string_literal(actual_node, ctx.source);
            Some(Attribute::String(s))
        }
        "true" => Some(Attribute::Bool(true)),
        "false" => Some(Attribute::Bool(false)),
        _ => None, // Non-literal expressions not supported in const
    }
}

/// Unwrap expression wrapper nodes to get the actual literal node.
fn unwrap_expression_node<'tree>(node: Node<'tree>, _source: &str) -> Option<Node<'tree>> {
    match node.kind() {
        "primary_expression" | "expression" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return unwrap_expression_node(child, _source);
                }
            }
            None
        }
        _ => Some(node),
    }
}

/// Lower an ability declaration to type.ability.
fn lower_ability_decl<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Option<ty::Ability<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut name = None;
    let mut operations = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "type_identifier" if name.is_none() => {
                name = Some(node_text(&child, ctx.source).to_string());
            }
            "ability_body" => {
                operations = parse_ability_operations(ctx, child);
            }
            _ => {}
        }
    }

    let name = name?;
    let operations_attr = Attribute::List(
        operations
            .into_iter()
            .map(|(op_name, param_types, return_type)| {
                Attribute::List(vec![
                    Attribute::Symbol(sym(ctx.db, &op_name)),
                    Attribute::List(
                        param_types
                            .iter()
                            .map(|t| Attribute::Symbol(sym(ctx.db, &format!("{:?}", t))))
                            .collect(),
                    ),
                    Attribute::Symbol(sym(ctx.db, &format!("{:?}", return_type))),
                ])
            })
            .collect(),
    );

    Some(ty::ability(
        ctx.db,
        location,
        infer_ty,
        Attribute::Symbol(sym(ctx.db, &name)),
        operations_attr,
    ))
}

/// Parse ability operations from ability_body.
fn parse_ability_operations<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Vec<(String, Vec<Type<'db>>, Type<'db>)> {
    let mut cursor = node.walk();
    let mut operations = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if child.kind() == "ability_operation" {
            let mut op_cursor = child.walk();
            let mut op_name = None;
            let mut param_types = Vec::new();
            let mut return_type = None;

            for op_child in child.named_children(&mut op_cursor) {
                if is_comment(op_child.kind()) {
                    continue;
                }
                match op_child.kind() {
                    "identifier" if op_name.is_none() => {
                        op_name = Some(node_text(&op_child, ctx.source).to_string());
                    }
                    "parameter_list" => {
                        let (_, types) = parse_parameter_list(ctx, op_child);
                        param_types = types;
                    }
                    "return_type_annotation" => {
                        return_type = parse_return_type(ctx, op_child);
                    }
                    _ => {}
                }
            }

            if let Some(name) = op_name {
                operations.push((
                    name,
                    param_types,
                    return_type.unwrap_or_else(|| ctx.fresh_type_var()),
                ));
            }
        }
    }

    operations
}

// =============================================================================
// Module Lowering
// =============================================================================

/// Lower a mod_declaration to a core.module operation.
///
/// Handles both inline modules (`mod foo { ... }`) and file-based module
/// declarations (`mod foo`). Currently, only inline modules are fully lowered.
fn lower_mod_decl<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Option<core::Module<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);

    let mut name = None;
    let mut body_node = None;
    let mut _is_pub = false;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "visibility_marker" => {
                // visibility_marker contains keyword_pub and optional (pkg) or (super)
                _is_pub = true;
                // TODO: Parse visibility modifier (pub, pub(pkg), pub(super))
            }
            "identifier" | "type_identifier" if name.is_none() => {
                name = Some(node_text(&child, ctx.source).to_string());
            }
            "mod_body" => {
                body_node = Some(child);
            }
            _ => {}
        }
    }

    let name = name?;

    // Build the module with its body
    let module = core::Module::build(ctx.db, location, &name, |mod_builder| {
        if let Some(body) = body_node {
            lower_mod_body(ctx, body, mod_builder);
        }
        // File-based modules (no body) will be handled later in the pipeline
        // when we have package/file loading infrastructure
    });

    // TODO: Track visibility (_is_pub) for name resolution
    Some(module)
}

/// Lower items within a mod_body into the module's block.
fn lower_mod_body<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
    builder: &mut BlockBuilder<'db>,
) {
    let mut cursor = node.walk();

    #[cfg(test)]
    eprintln!("mod_body node kind: {}, children:", node.kind());
    for child in node.named_children(&mut cursor) {
        #[cfg(test)]
        eprintln!("  child kind: {}", child.kind());
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "function_definition" => {
                if let Some(func) = lower_function(ctx, child) {
                    builder.op(func);
                }
            }
            "struct_declaration" => {
                if let Some(struct_op) = lower_struct_decl(ctx, child) {
                    builder.op(struct_op);
                }
            }
            "enum_declaration" => {
                if let Some(enum_op) = lower_enum_decl(ctx, child) {
                    builder.op(enum_op);
                }
            }
            "mod_declaration" => {
                // Nested modules
                if let Some(mod_op) = lower_mod_decl(ctx, child) {
                    builder.op(mod_op);
                }
            }
            "use_declaration" => {
                lower_use_decl(ctx, child, builder);
            }
            _ => {}
        }
    }
}

/// Parse a parameter list node, returning (names, types).
fn parse_parameter_list<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> (Vec<String>, Vec<Type<'db>>) {
    let mut cursor = node.walk();
    let mut names = Vec::new();
    let mut types = Vec::new();

    for child in node.named_children(&mut cursor) {
        if child.kind() == "parameter" {
            let mut param_cursor = child.walk();
            let mut param_name = None;
            let mut param_type = None;

            for param_child in child.named_children(&mut param_cursor) {
                match param_child.kind() {
                    "identifier" if param_name.is_none() => {
                        param_name = Some(node_text(&param_child, ctx.source).to_string());
                    }
                    "type_identifier" | "type_variable" | "generic_type" => {
                        param_type = Some(ctx.resolve_type_node(param_child));
                    }
                    _ => {}
                }
            }

            if let Some(name) = param_name {
                names.push(name);
                types.push(param_type.unwrap_or_else(|| ctx.fresh_type_var()));
            }
        }
    }

    (names, types)
}

/// Parse a return type annotation.
fn parse_return_type<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Option<Type<'db>> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "type_identifier" | "type_variable" | "generic_type" => {
                return Some(ctx.resolve_type_node(child));
            }
            _ => {}
        }
    }
    None
}

// =============================================================================
// Block and Statement Lowering
// =============================================================================

/// Lower block body statements, returning the last expression value.
fn lower_block_body<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let mut last_value = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "let_statement" => {
                lower_let_statement(ctx, block, child);
                last_value = None; // Let doesn't produce a value
            }
            _ => {
                // Try to lower as expression
                if let Some(value) = lower_expr(ctx, block, child) {
                    last_value = Some(value);
                }
            }
        }
    }

    last_value
}

/// Lower a let statement.
fn lower_let_statement<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) {
    let mut cursor = node.walk();
    let mut pattern_node = None;
    let mut value_node = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            // Simple identifier pattern
            "identifier" | "identifier_pattern" if pattern_node.is_none() => {
                pattern_node = Some(child);
            }
            // Other patterns
            "wildcard_pattern"
            | "constructor_pattern"
            | "tuple_pattern"
            | "list_pattern"
            | "as_pattern"
                if pattern_node.is_none() =>
            {
                pattern_node = Some(child);
            }
            // The value expression (anything else after pattern)
            _ if pattern_node.is_some() && value_node.is_none() => {
                value_node = Some(child);
            }
            _ => {}
        }
    }

    if let (Some(pattern), Some(value)) = (pattern_node, value_node)
        && let Some(value) = lower_expr(ctx, block, value)
    {
        bind_pattern(ctx, block, pattern, value);
    }
}

/// Bind a pattern to a value, emitting extraction operations as needed.
fn bind_pattern<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    pattern: Node,
    value: Value<'db>,
) {
    let location = ctx.location(&pattern);
    let infer_ty = ctx.fresh_type_var();

    match pattern.kind() {
        "identifier" | "identifier_pattern" => {
            let name = node_text(&pattern, ctx.source).to_string();
            ctx.bind(name, value);
        }
        "wildcard_pattern" => {
            // Discard - no binding
        }
        "as_pattern" => {
            // Bind the whole value to the name, then recurse on inner pattern
            let mut cursor = pattern.walk();
            let mut inner_pattern = None;
            let mut binding_name = None;

            for child in pattern.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "identifier" => {
                        binding_name = Some(node_text(&child, ctx.source).to_string());
                    }
                    _ if inner_pattern.is_none() => {
                        inner_pattern = Some(child);
                    }
                    _ => {}
                }
            }

            if let Some(name) = binding_name {
                ctx.bind(name, value);
            }
            if let Some(inner) = inner_pattern {
                bind_pattern(ctx, block, inner, value);
            }
        }
        "literal_pattern" => {
            // Literal patterns in let bindings don't introduce bindings
            // They just assert the value matches - no action needed
        }
        "constructor_pattern" => {
            // Destructure variant: let Some(x) = opt
            let mut cursor = pattern.walk();
            let mut idx = 0;

            for child in pattern.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "type_identifier" => {
                        // Skip the constructor name
                    }
                    "pattern_list" => {
                        // Positional fields: Some(x, y)
                        let mut list_cursor = child.walk();
                        for pat_child in child.named_children(&mut list_cursor) {
                            if is_comment(pat_child.kind()) {
                                continue;
                            }
                            let field_value = block
                                .op(adt::variant_get(
                                    ctx.db,
                                    location,
                                    value,
                                    infer_ty,
                                    (idx as u64).into(),
                                ))
                                .result(ctx.db);
                            bind_pattern(ctx, block, pat_child, field_value);
                            idx += 1;
                        }
                    }
                    "pattern_fields" => {
                        // Named fields: Point { x, y }
                        let mut fields_cursor = child.walk();
                        for field_child in child.named_children(&mut fields_cursor) {
                            if field_child.kind() == "pattern_field" {
                                let field_value = block
                                    .op(adt::variant_get(
                                        ctx.db,
                                        location,
                                        value,
                                        infer_ty,
                                        (idx as u64).into(),
                                    ))
                                    .result(ctx.db);

                                // Get the pattern inside the field
                                let mut field_cursor = field_child.walk();
                                for pat in field_child.named_children(&mut field_cursor) {
                                    if !is_comment(pat.kind()) && pat.kind() != "identifier" {
                                        bind_pattern(ctx, block, pat, field_value);
                                        break;
                                    } else if pat.kind() == "identifier" {
                                        // Shorthand: { name } means { name: name }
                                        let field_name = node_text(&pat, ctx.source).to_string();
                                        ctx.bind(field_name, field_value);
                                        break;
                                    }
                                }
                                idx += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "tuple_pattern" => {
            // Destructure tuple: let #(a, b, c) = tuple
            let mut cursor = pattern.walk();

            for child in pattern.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                if child.kind() == "pattern_list" {
                    let mut list_cursor = child.walk();
                    let mut idx = 0;
                    for pat_child in child.named_children(&mut list_cursor) {
                        if is_comment(pat_child.kind()) {
                            continue;
                        }
                        let elem_value = block
                            .op(src::call(
                                ctx.db,
                                location,
                                vec![value],
                                infer_ty,
                                sym_ref(ctx.db, &format!("tuple_get_{}", idx)),
                            ))
                            .result(ctx.db);
                        bind_pattern(ctx, block, pat_child, elem_value);
                        idx += 1;
                    }
                }
            }
        }
        "list_pattern" => {
            // Destructure list: let [a, b, ..rest] = list
            let mut cursor = pattern.walk();
            let mut idx = 0;
            let mut rest_pattern = None;

            for child in pattern.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                if child.kind() == "rest_pattern" {
                    rest_pattern = Some(child);
                } else {
                    // Regular element pattern
                    let index_value = block
                        .op(arith::Const::i64(ctx.db, location, idx))
                        .result(ctx.db);
                    let elem_ty = ctx.fresh_type_var();
                    let elem_value = block
                        .op(list::get(
                            ctx.db,
                            location,
                            value,
                            index_value,
                            infer_ty,
                            elem_ty,
                        ))
                        .result(ctx.db);
                    bind_pattern(ctx, block, child, elem_value);
                    idx += 1;
                }
            }

            // Handle rest pattern: ..rest binds to list[n..]
            if let Some(rest_node) = rest_pattern {
                let mut rest_cursor = rest_node.walk();
                for rest_child in rest_node.named_children(&mut rest_cursor) {
                    if rest_child.kind() == "identifier" {
                        let rest_name = node_text(&rest_child, ctx.source).to_string();
                        let start_value = block
                            .op(arith::Const::i64(ctx.db, location, idx))
                            .result(ctx.db);
                        let len_value = block
                            .op(list::len(ctx.db, location, value, infer_ty))
                            .result(ctx.db);
                        let elem_ty = ctx.fresh_type_var();
                        let rest_value = block
                            .op(list::slice(
                                ctx.db,
                                location,
                                value,
                                start_value,
                                len_value,
                                infer_ty,
                                elem_ty,
                            ))
                            .result(ctx.db);
                        ctx.bind(rest_name, rest_value);
                        break;
                    }
                }
            }
        }
        "handler_pattern" => {
            // Handler patterns are for ability effect handling in match expressions
            // They don't make sense in let bindings - handled in case arms
        }
        _ => {
            // Unknown pattern - try to handle child patterns
            let mut cursor = pattern.walk();
            for child in pattern.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    bind_pattern(ctx, block, child, value);
                    break;
                }
            }
        }
    }
}

// =============================================================================
// Expression Lowering
// =============================================================================

/// Lower an expression node to TrunkIR operations.
fn lower_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();
    let unit_ty = core::Nil::new(ctx.db).as_type();

    match node.kind() {
        // === Literals ===
        "nat_literal" => {
            let value = parse_nat_literal(node_text(&node, ctx.source))?;
            let op = block.op(arith::Const::u64(ctx.db, location, value));
            Some(op.result(ctx.db))
        }
        "int_literal" => {
            let value = parse_int_literal(node_text(&node, ctx.source))?;
            let op = block.op(arith::Const::i64(ctx.db, location, value));
            Some(op.result(ctx.db))
        }
        "float_literal" => {
            let value = parse_float_literal(node_text(&node, ctx.source))?;
            let op = block.op(arith::Const::f64(ctx.db, location, value));
            Some(op.result(ctx.db))
        }
        "true" => {
            let op = block.op(arith::r#const(
                ctx.db,
                location,
                core::I1::new(ctx.db).as_type(),
                true.into(),
            ));
            Some(op.result(ctx.db))
        }
        "false" => {
            let op = block.op(arith::r#const(
                ctx.db,
                location,
                core::I1::new(ctx.db).as_type(),
                false.into(),
            ));
            Some(op.result(ctx.db))
        }
        "nil" => {
            let op = block.op(arith::r#const(ctx.db, location, unit_ty, Attribute::Unit));
            Some(op.result(ctx.db))
        }
        "rune" => {
            let c = parse_rune_literal(node_text(&node, ctx.source))?;
            let op = block.op(arith::r#const(
                ctx.db,
                location,
                core::I32::new(ctx.db).as_type(),
                u64::from(u32::from(c)).into(),
            ));
            Some(op.result(ctx.db))
        }

        // === String literals ===
        "string" | "raw_string" | "multiline_string" => {
            let s = parse_string_literal(node, ctx.source);
            let string_ty = core::String::new(ctx.db).as_type();
            let op = block.op(adt::string_const(ctx.db, location, string_ty, s));
            Some(op.result(ctx.db))
        }

        // === Bytes literals ===
        "bytes_string" | "raw_bytes" | "multiline_bytes" => {
            let bytes = parse_bytes_literal(node, ctx.source);
            let bytes_ty = core::Bytes::new(ctx.db).as_type();
            let op = block.op(adt::bytes_const(
                ctx.db,
                location,
                bytes_ty,
                Attribute::Bytes(bytes),
            ));
            Some(op.result(ctx.db))
        }

        // === Identifiers and paths ===
        "identifier" => {
            // Always create src.var for variable references, even for local bindings.
            // This preserves the source span for hover. Resolution will transform
            // local references to identity operations with the correct type.
            let name = node_text(&node, ctx.source);
            let op = block.op(src::var(ctx.db, location, infer_ty, sym(ctx.db, name)));
            Some(op.result(ctx.db))
        }
        "path_expression" => {
            let mut cursor = node.walk();
            let segments: Vec<Symbol<'db>> = node
                .named_children(&mut cursor)
                .filter(|n| n.kind() == "identifier" || n.kind() == "type_identifier")
                .map(|n| sym(ctx.db, node_text(&n, ctx.source)))
                .collect();

            if segments.is_empty() {
                return None;
            }

            let path: IdVec<_> = segments.into_iter().collect();
            let op = block.op(src::path(ctx.db, location, infer_ty, path));
            Some(op.result(ctx.db))
        }

        // === Binary expressions ===
        "binary_expression" => lower_binary_expr(ctx, block, node),

        // === Call expressions ===
        "call_expression" => lower_call_expr(ctx, block, node),

        // === Method call ===
        "method_call_expression" => lower_method_call_expr(ctx, block, node),

        // === Lambda ===
        "lambda_expression" => lower_lambda_expr(ctx, block, node),

        // === Match/Case ===
        "case_expression" => lower_case_expr(ctx, block, node),

        // === Block ===
        "block" => lower_block_expr(ctx, block, node),

        // === List ===
        "list_expression" => lower_list_expr(ctx, block, node),

        // === Tuple ===
        "tuple_expression" => lower_tuple_expr(ctx, block, node),

        // === Record ===
        "record_expression" => lower_record_expr(ctx, block, node),

        // === Parenthesized ===
        "parenthesized_expression" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return lower_expr(ctx, block, child);
                }
            }
            None
        }

        // === Primary expression (wrapper for literals, paths, etc.) ===
        "primary_expression" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return lower_expr(ctx, block, child);
                }
            }
            None
        }

        // === Expression statement (expression as statement in a block) ===
        "expression_statement" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return lower_expr(ctx, block, child);
                }
            }
            None
        }

        // === Handle expression (ability handling) ===
        "handle_expression" => lower_handle_expr(ctx, block, node),

        // === Interpolated strings ===
        "string_interpolation" => lower_string_interpolation(ctx, block, node),

        // === Interpolated bytes ===
        "bytes_interpolation" => lower_bytes_interpolation(ctx, block, node),

        // Unknown expression type - return None
        _ => None,
    }
}

// =============================================================================
// Literal Parsing
// =============================================================================

/// Parse a natural number literal (unsigned).
fn parse_nat_literal(text: &str) -> Option<u64> {
    let text = text.replace('_', "");
    if text.starts_with("0x") || text.starts_with("0X") {
        u64::from_str_radix(&text[2..], 16).ok()
    } else if text.starts_with("0b") || text.starts_with("0B") {
        u64::from_str_radix(&text[2..], 2).ok()
    } else if text.starts_with("0o") || text.starts_with("0O") {
        u64::from_str_radix(&text[2..], 8).ok()
    } else {
        text.parse().ok()
    }
}

/// Parse an integer literal (signed).
fn parse_int_literal(text: &str) -> Option<i64> {
    let text = text.replace('_', "");
    // Handle explicit + or - prefix
    if let Some(rest) = text.strip_prefix('+') {
        parse_nat_literal(rest).map(|n| n as i64)
    } else if let Some(rest) = text.strip_prefix('-') {
        parse_nat_literal(rest).map(|n| -(n as i64))
    } else {
        text.parse().ok()
    }
}

/// Parse a float literal.
fn parse_float_literal(text: &str) -> Option<f64> {
    let text = text.replace('_', "");
    text.parse().ok()
}

/// Parse a rune (character) literal.
fn parse_rune_literal(text: &str) -> Option<char> {
    // Format: ?c, ?\n, ?\xHH, ?\uHHHH
    let text = text.strip_prefix('?')?;

    if let Some(escape) = text.strip_prefix('\\') {
        match escape.chars().next()? {
            'n' => Some('\n'),
            'r' => Some('\r'),
            't' => Some('\t'),
            '\\' => Some('\\'),
            '0' => Some('\0'),
            'x' => {
                let hex = &escape[1..];
                u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
            }
            'u' => {
                let hex = &escape[1..];
                u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
            }
            _ => None,
        }
    } else {
        text.chars().next()
    }
}

/// Parse a string literal (handling escapes).
fn parse_string_literal(node: Node, source: &str) -> String {
    let text = node_text(&node, source);

    match node.kind() {
        "raw_string" => {
            // r"..." or r#"..."#
            extract_raw_string_content(text)
        }
        "multiline_string" => {
            // #"..."#
            extract_multiline_string_content(text)
        }
        "string" => {
            // Regular "..." with escapes
            extract_string_content(text)
        }
        _ => text.to_string(),
    }
}

/// Parse a bytes literal.
fn parse_bytes_literal(node: Node, source: &str) -> Vec<u8> {
    let text = node_text(&node, source);

    match node.kind() {
        "raw_bytes" => {
            // rb"..." or rb#"..."#
            if let Some(content) = text.strip_prefix("rb") {
                extract_raw_string_content(content).into_bytes()
            } else {
                Vec::new()
            }
        }
        "multiline_bytes" => {
            // b#"..."#
            if let Some(content) = text.strip_prefix("b") {
                extract_multiline_string_content(content).into_bytes()
            } else {
                Vec::new()
            }
        }
        "bytes_string" => {
            // b"..."
            if let Some(content) = text.strip_prefix("b") {
                extract_string_content(content).into_bytes()
            } else {
                Vec::new()
            }
        }
        _ => text.as_bytes().to_vec(),
    }
}

/// Extract content from a regular string literal.
fn extract_string_content(text: &str) -> String {
    // Remove quotes
    let inner = text
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(text);

    // Process escapes
    let mut result = String::new();
    let mut chars = inner.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('0') => result.push('\0'),
                Some('x') => {
                    // \xHH
                    let hex: String = chars.by_ref().take(2).collect();
                    if let Ok(code) = u8::from_str_radix(&hex, 16) {
                        result.push(code as char);
                    }
                }
                Some('u') => {
                    // \uHHHH
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(code) = u32::from_str_radix(&hex, 16)
                        && let Some(c) = char::from_u32(code)
                    {
                        result.push(c);
                    }
                }
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Extract content from a raw string literal.
fn extract_raw_string_content(text: &str) -> String {
    // r"..." or r#"..."#
    if text.starts_with('#') {
        // Count opening hashes
        let hash_count = text.chars().take_while(|&c| c == '#').count();
        let delimiter = "#".repeat(hash_count);
        let open = format!("{}\"", delimiter);
        let close = format!("\"{}", delimiter);

        if let Some(start) = text.find(&open)
            && let Some(end) = text.rfind(&close)
        {
            return text[start + open.len()..end].to_string();
        }
        text.to_string()
    } else {
        // r"..."
        text.strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(text)
            .to_string()
    }
}

/// Extract content from a multiline string literal.
fn extract_multiline_string_content(text: &str) -> String {
    // #"..."#
    let hash_count = text.chars().take_while(|&c| c == '#').count();
    let open_delim = format!("{}\"", "#".repeat(hash_count));
    let close_delim = format!("\"{}", "#".repeat(hash_count));

    if let Some(start) = text.find(&open_delim)
        && let Some(end) = text.rfind(&close_delim)
    {
        return text[start + open_delim.len()..end].to_string();
    }
    text.to_string()
}

// =============================================================================
// Complex Expression Lowering
// =============================================================================

/// Lower a binary expression.
fn lower_binary_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();
    let bool_ty = core::I1::new(ctx.db).as_type();

    let mut lhs_node = None;
    let mut operator = None;
    let mut rhs_node = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if lhs_node.is_none() {
            lhs_node = Some(child);
        } else if operator.is_none() && is_operator_node(&child) {
            operator = Some(node_text(&child, ctx.source).to_string());
        } else if rhs_node.is_none() {
            rhs_node = Some(child);
        }
    }

    // Handle the case where operator is an unnamed token
    if operator.is_none() {
        // Look for operator in unnamed children
        let mut child_cursor = node.walk();
        if child_cursor.goto_first_child() {
            loop {
                let child = child_cursor.node();
                if !child.is_named() && is_operator_text(node_text(&child, ctx.source)) {
                    operator = Some(node_text(&child, ctx.source).to_string());
                    break;
                }
                if !child_cursor.goto_next_sibling() {
                    break;
                }
            }
        }
    }

    let lhs_node = lhs_node?;
    let operator = operator?;
    let rhs_node = rhs_node?;

    let lhs = lower_expr(ctx, block, lhs_node)?;
    let rhs = lower_expr(ctx, block, rhs_node)?;

    // Map operator to IR operation
    let result = match operator.as_str() {
        "+" => block
            .op(arith::add(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "-" => block
            .op(arith::sub(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "*" => block
            .op(arith::mul(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "/" => block
            .op(arith::div(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "%" => block
            .op(arith::rem(ctx.db, location, lhs, rhs, infer_ty))
            .result(ctx.db),
        "==" => block
            .op(arith::cmp_eq(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "!=" => block
            .op(arith::cmp_ne(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "<" => block
            .op(arith::cmp_lt(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "<=" => block
            .op(arith::cmp_le(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        ">" => block
            .op(arith::cmp_gt(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        ">=" => block
            .op(arith::cmp_ge(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "&&" => block
            .op(arith::and(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "||" => block
            .op(arith::or(ctx.db, location, lhs, rhs, bool_ty))
            .result(ctx.db),
        "<>" => {
            // String concatenation - use src.binop
            block
                .op(src::binop(
                    ctx.db,
                    location,
                    lhs,
                    rhs,
                    infer_ty,
                    sym(ctx.db, "concat"),
                ))
                .result(ctx.db)
        }
        _ => {
            // Unknown operator - emit as src.binop
            block
                .op(src::binop(
                    ctx.db,
                    location,
                    lhs,
                    rhs,
                    infer_ty,
                    sym(ctx.db, &operator),
                ))
                .result(ctx.db)
        }
    };

    Some(result)
}

fn is_operator_node(node: &Node) -> bool {
    matches!(node.kind(), "operator" | "binary_operator")
}

fn is_operator_text(text: &str) -> bool {
    matches!(
        text,
        "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||" | "<>"
    )
}

/// Lower a call expression.
fn lower_call_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut func_path: Option<IdVec<Symbol<'db>>> = None;
    let mut args = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "identifier" if func_path.is_none() => {
                func_path = Some(sym_ref(ctx.db, node_text(&child, ctx.source)));
            }
            "path_expression" if func_path.is_none() => {
                let mut path_cursor = child.walk();
                let segments: Vec<Symbol<'db>> = child
                    .named_children(&mut path_cursor)
                    .filter(|n| n.kind() == "identifier" || n.kind() == "type_identifier")
                    .map(|n| sym(ctx.db, node_text(&n, ctx.source)))
                    .collect();
                if !segments.is_empty() {
                    func_path = Some(segments.into_iter().collect());
                }
            }
            "argument_list" => {
                let mut arg_cursor = child.walk();
                for arg_child in child.named_children(&mut arg_cursor) {
                    if !is_comment(arg_child.kind())
                        && let Some(value) = lower_expr(ctx, block, arg_child)
                    {
                        args.push(value);
                    }
                }
            }
            _ => {}
        }
    }

    let func_path = func_path?;
    let op = block.op(src::call(
        ctx.db,
        location,
        args,
        infer_ty,
        func_path,
    ));
    Some(op.result(ctx.db))
}

/// Lower a method call expression (UFCS).
fn lower_method_call_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut receiver_node = None;
    let mut method_name = None;
    let mut args = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            // First expression is the receiver
            _ if receiver_node.is_none()
                && !matches!(child.kind(), "identifier" | "argument_list") =>
            {
                receiver_node = Some(child);
            }
            "identifier" if method_name.is_none() => {
                method_name = Some(node_text(&child, ctx.source).to_string());
            }
            "argument_list" => {
                let mut arg_cursor = child.walk();
                for arg_child in child.named_children(&mut arg_cursor) {
                    if !is_comment(arg_child.kind())
                        && let Some(value) = lower_expr(ctx, block, arg_child)
                    {
                        args.push(value);
                    }
                }
            }
            _ => {
                // Could be receiver
                if receiver_node.is_none() {
                    receiver_node = Some(child);
                }
            }
        }
    }

    let receiver_node = receiver_node?;
    let method_name = method_name?;

    // Lower receiver first
    let receiver = lower_expr(ctx, block, receiver_node)?;

    // UFCS: x.f(y, z) → f(x, y, z)
    let mut all_args = vec![receiver];
    all_args.extend(args);

    let op = block.op(src::call(
        ctx.db,
        location,
        all_args,
        infer_ty,
        sym_ref(ctx.db, &method_name),
    ));
    Some(op.result(ctx.db))
}

/// Lower a lambda expression.
fn lower_lambda_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut param_names = Vec::new();
    let mut body_node = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "parameter_list" => {
                let (names, _types) = parse_parameter_list(ctx, child);
                param_names = names;
            }
            "identifier" if param_names.is_empty() => {
                // Single parameter without parens: fn x x + 1
                param_names.push(node_text(&child, ctx.source).to_string());
            }
            _ if body_node.is_none() => {
                body_node = Some(child);
            }
            _ => {}
        }
    }

    let body_node = body_node?;

    // Build lambda body
    let param_types: IdVec<Type<'_>> = std::iter::repeat_n(infer_ty, param_names.len()).collect();
    let result_type = infer_ty;
    let mut body_block = BlockBuilder::new(ctx.db, location).args(param_types.clone());

    let result_value = ctx.scoped(|ctx| {
        // Bind parameters
        for param_name in &param_names {
            let param_value = body_block.op(src::var(
                ctx.db,
                location,
                infer_ty,
                sym(ctx.db, param_name),
            ));
            ctx.bind(param_name.clone(), param_value.result(ctx.db));
        }

        // Lower body
        lower_expr(ctx, &mut body_block, body_node)
    });

    let result_value = result_value?;
    body_block.op(src::r#yield(ctx.db, location, result_value));

    let func_type = core::Func::new(ctx.db, param_types, result_type).as_type();
    let region = Region::new(ctx.db, location, idvec![body_block.build()]);
    let lambda_op = block.op(src::lambda(ctx.db, location, infer_ty, func_type, region));
    Some(lambda_op.result(ctx.db))
}

/// Lower a case/match expression.
fn lower_case_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut scrutinee_node = None;
    let mut arms = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "case_arm" => {
                arms.push(child);
            }
            _ if scrutinee_node.is_none() => {
                scrutinee_node = Some(child);
            }
            _ => {}
        }
    }

    let scrutinee_node = scrutinee_node?;
    let scrutinee = lower_expr(ctx, block, scrutinee_node)?;

    // Build the body region containing case.arm operations
    let mut body_block = BlockBuilder::new(ctx.db, location);

    for arm in arms {
        if let Some(arm_op) = lower_case_arm(ctx, arm, scrutinee) {
            body_block.op(arm_op);
        }
    }

    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    let op = block.op(case::r#case(
        ctx.db,
        location,
        scrutinee,
        infer_ty,
        body_region,
    ));
    Some(op.result(ctx.db))
}

/// Lower a single case arm.
fn lower_case_arm<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
    scrutinee: Value<'db>,
) -> Option<case::Arm<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);

    let mut pattern_node = None;
    let mut body_node = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if pattern_node.is_none() {
            pattern_node = Some(child);
        } else if body_node.is_none() {
            body_node = Some(child);
        }
    }

    let pattern_node = pattern_node?;
    let body_node = body_node?;

    // Create arm body
    let mut body_block = BlockBuilder::new(ctx.db, location);

    let result_value = ctx.scoped(|ctx| {
        // Bind pattern
        bind_pattern(ctx, &mut body_block, pattern_node, scrutinee);

        // Lower body
        lower_expr(ctx, &mut body_block, body_node)
    });

    let result_value = result_value?;
    body_block.op(case::r#yield(ctx.db, location, result_value));

    let pattern_region = pattern_to_region(ctx, pattern_node);
    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    Some(case::arm(ctx.db, location, pattern_region, body_region))
}

/// Convert a pattern node to a pattern region for case arms.
///
/// Creates a region containing pattern operations from the `pat` dialect.
fn pattern_to_region<'db, 'src>(ctx: &CstLoweringCtx<'db, 'src>, node: Node) -> Region<'db> {
    let location = ctx.location(&node);

    match node.kind() {
        "identifier" | "identifier_pattern" => {
            let name = node_text(&node, ctx.source);
            pat::helpers::bind_region(ctx.db, location, name)
        }
        "wildcard_pattern" => pat::helpers::wildcard_region(ctx.db, location),
        "literal_pattern" => {
            // Get the literal value
            let mut cursor = node.walk();
            if let Some(child) = node.named_children(&mut cursor).next() {
                pattern_to_region(ctx, child)
            } else {
                pat::helpers::wildcard_region(ctx.db, location)
            }
        }
        "nat_literal" | "int_literal" => {
            if let Some(n) = parse_int_literal(node_text(&node, ctx.source)) {
                pat::helpers::int_region(ctx.db, location, n)
            } else {
                pat::helpers::wildcard_region(ctx.db, location)
            }
        }
        "true" => pat::helpers::bool_region(ctx.db, location, true),
        "false" => pat::helpers::bool_region(ctx.db, location, false),
        "string" | "raw_string" | "multiline_string" => {
            let s = parse_string_literal(node, ctx.source);
            pat::helpers::string_region(ctx.db, location, &s)
        }
        "constructor_pattern" => {
            let mut cursor = node.walk();
            let mut ctor_name = None;
            let mut field_ops: Vec<Operation<'db>> = Vec::new();

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "type_identifier" => {
                        ctor_name = Some(node_text(&child, ctx.source));
                    }
                    "pattern_list" => {
                        let mut list_cursor = child.walk();
                        for pat_child in child.named_children(&mut list_cursor) {
                            if !is_comment(pat_child.kind()) {
                                let pat_region = pattern_to_region(ctx, pat_child);
                                // Extract the pattern operation from the region
                                if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                                    field_ops.push(op);
                                }
                            }
                        }
                    }
                    "pattern_fields" => {
                        // Named fields - treat as positional for pattern matching
                        let mut fields_cursor = child.walk();
                        for field_child in child.named_children(&mut fields_cursor) {
                            if field_child.kind() == "pattern_field" {
                                let mut field_cursor = field_child.walk();
                                for pat in field_child.named_children(&mut field_cursor) {
                                    if !is_comment(pat.kind()) {
                                        let pat_region = pattern_to_region(ctx, pat);
                                        if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                                            field_ops.push(op);
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            let name = ctor_name.unwrap_or("_");
            let variant_path = idvec![Symbol::new(ctx.db, name)];
            let fields_region = ops_to_region(ctx.db, location, field_ops);
            pat::helpers::variant_region(ctx.db, location, variant_path, fields_region)
        }
        "tuple_pattern" => {
            let mut cursor = node.walk();
            let mut elem_ops: Vec<Operation<'db>> = Vec::new();

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                if child.kind() == "pattern_list" {
                    let mut list_cursor = child.walk();
                    for pat_child in child.named_children(&mut list_cursor) {
                        if !is_comment(pat_child.kind()) {
                            let pat_region = pattern_to_region(ctx, pat_child);
                            if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                                elem_ops.push(op);
                            }
                        }
                    }
                }
            }

            let elements_region = ops_to_region(ctx.db, location, elem_ops);
            pat::helpers::tuple_region(ctx.db, location, elements_region)
        }
        "list_pattern" => {
            let mut cursor = node.walk();
            let mut elem_ops: Vec<Operation<'db>> = Vec::new();
            let mut rest_name: Option<&str> = None;

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                if child.kind() == "rest_pattern" {
                    let mut rest_cursor = child.walk();
                    for rest_child in child.named_children(&mut rest_cursor) {
                        if rest_child.kind() == "identifier" {
                            rest_name = Some(node_text(&rest_child, ctx.source));
                            break;
                        }
                    }
                } else {
                    let pat_region = pattern_to_region(ctx, child);
                    if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                        elem_ops.push(op);
                    }
                }
            }

            if let Some(name) = rest_name {
                let head_region = ops_to_region(ctx.db, location, elem_ops);
                pat::helpers::list_rest_region(
                    ctx.db,
                    location,
                    Symbol::new(ctx.db, name),
                    head_region,
                )
            } else {
                let elements_region = ops_to_region(ctx.db, location, elem_ops);
                pat::helpers::list_region(ctx.db, location, elements_region)
            }
        }
        "as_pattern" => {
            let mut cursor = node.walk();
            let mut inner_region = None;
            let mut binding_name = None;

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "identifier" => {
                        binding_name = Some(node_text(&child, ctx.source));
                    }
                    _ if inner_region.is_none() => {
                        inner_region = Some(pattern_to_region(ctx, child));
                    }
                    _ => {}
                }
            }

            let inner =
                inner_region.unwrap_or_else(|| pat::helpers::wildcard_region(ctx.db, location));
            let name = binding_name.unwrap_or("_");
            // Create as_pat operation with inner region
            let as_op = pat::as_pat(ctx.db, location, Symbol::new(ctx.db, name), inner);
            pat::helpers::single_op_region(ctx.db, location, as_op.as_operation())
        }
        _ => pat::helpers::wildcard_region(ctx.db, location),
    }
}

/// Extract the first pattern operation from a region.
fn extract_pattern_op<'db>(
    db: &'db dyn salsa::Database,
    region: &Region<'db>,
) -> Option<Operation<'db>> {
    let blocks = region.blocks(db);
    let block = blocks.first()?;
    let ops = block.operations(db);
    ops.first().copied()
}

/// Create a region from a list of operations.
fn ops_to_region<'db>(
    db: &'db dyn salsa::Database,
    location: Location<'db>,
    ops: Vec<Operation<'db>>,
) -> Region<'db> {
    let block = Block::new(db, location, IdVec::new(), IdVec::from(ops));
    Region::new(db, location, IdVec::from(vec![block]))
}

/// Lower a block expression.
fn lower_block_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut body_block = BlockBuilder::new(ctx.db, location);
    let result_value = ctx.scoped(|ctx| lower_block_body(ctx, &mut body_block, node));

    let result_value = result_value?;
    body_block.op(src::r#yield(ctx.db, location, result_value));

    let region = Region::new(ctx.db, location, idvec![body_block.build()]);
    let block_op = block.op(src::block(ctx.db, location, infer_ty, region));
    Some(block_op.result(ctx.db))
}

/// Lower a list expression.
fn lower_list_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();
    let elem_ty = ctx.fresh_type_var();

    let mut elements = Vec::new();
    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind())
            && let Some(value) = lower_expr(ctx, block, child)
        {
            elements.push(value);
        }
    }

    let op = block.op(list::new(ctx.db, location, elements, infer_ty, elem_ty));
    Some(op.result(ctx.db))
}

/// Lower a tuple expression.
fn lower_tuple_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut elements = Vec::new();
    for child in node.named_children(&mut cursor) {
        if !is_comment(child.kind())
            && let Some(value) = lower_expr(ctx, block, child)
        {
            elements.push(value);
        }
    }

    if elements.is_empty() {
        return None;
    }

    let tuple_op = block.op(src::tuple(ctx.db, location, elements, infer_ty));
    Some(tuple_op.result(ctx.db))
}

/// Lower a record expression.
fn lower_record_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    let mut type_name = None;
    let mut field_values = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "type_identifier" if type_name.is_none() => {
                type_name = Some(node_text(&child, ctx.source).to_string());
            }
            "record_field" => {
                // Get field value
                let mut field_cursor = child.walk();
                for field_child in child.named_children(&mut field_cursor) {
                    if is_comment(field_child.kind()) {
                        continue;
                    }
                    if field_child.kind() != "identifier" {
                        if let Some(value) = lower_expr(ctx, block, field_child) {
                            field_values.push(value);
                        }
                    } else {
                        // Shorthand: { name } means { name: name }
                        let field_name = node_text(&field_child, ctx.source);
                        if let Some(value) = ctx.lookup(field_name) {
                            field_values.push(value);
                        } else {
                            let var_op = block.op(src::var(
                                ctx.db,
                                location,
                                infer_ty,
                                sym(ctx.db, field_name),
                            ));
                            field_values.push(var_op.result(ctx.db));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let type_name = type_name?;
    let op = block.op(src::call(
        ctx.db,
        location,
        field_values,
        infer_ty,
        sym_ref(ctx.db, &type_name),
    ));
    Some(op.result(ctx.db))
}

/// Lower a handle expression (ability handling).
///
/// Source: `case handle expr { arms... }`
/// Lowers to:
/// ```text
/// %request = ability.prompt { expr }
/// case.case(%request) { arms... }
/// ```
fn lower_handle_expr<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();
    let request_ty = ctx.fresh_type_var(); // Type for Request value

    let mut expr_node = None;
    let mut handler_arms = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) || child.kind() == "keyword_handle" {
            continue;
        }
        // First expression is the expression to handle
        if expr_node.is_none() {
            expr_node = Some(child);
        } else if child.kind() == "case_arm" {
            handler_arms.push(child);
        }
    }

    let expr_node = expr_node?;

    // Build body region with the expression
    let mut body_block = BlockBuilder::new(ctx.db, location);
    let result_value = ctx.scoped(|ctx| lower_expr(ctx, &mut body_block, expr_node));

    if let Some(value) = result_value {
        body_block.op(src::r#yield(ctx.db, location, value));
    }

    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    // Create ability.prompt to run body and get Request
    let prompt_op = block.op(ability::prompt(ctx.db, location, request_ty, body_region));
    let request_value = prompt_op.request(ctx.db);

    // Build handler arms as case.case body
    let case_body_region = if handler_arms.is_empty() {
        // No handlers - just a single wildcard arm
        let mut arm_block = BlockBuilder::new(ctx.db, location);
        arm_block.op(case::r#yield(ctx.db, location, request_value));
        Region::new(ctx.db, location, idvec![arm_block.build()])
    } else {
        // Build case.arm operations for each handler
        let mut arms_block = BlockBuilder::new(ctx.db, location);

        for arm_node in &handler_arms {
            // Lower handler arm (similar to case arm but with request as scrutinee)
            if let Some(arm_op) = ctx.scoped(|ctx| lower_handler_arm(ctx, *arm_node, request_value))
            {
                arms_block.op(arm_op);
            }
        }

        Region::new(ctx.db, location, idvec![arms_block.build()])
    };

    // Create case.case to pattern match on the Request
    let case_op = block.op(case::r#case(
        ctx.db,
        location,
        request_value,
        infer_ty,
        case_body_region,
    ));
    Some(case_op.result(ctx.db))
}

/// Lower a handler arm (for ability handling).
fn lower_handler_arm<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    node: Node,
    request: Value<'db>,
) -> Option<case::Arm<'db>> {
    let mut cursor = node.walk();
    let location = ctx.location(&node);

    let mut pattern_node = None;
    let mut body_node = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if pattern_node.is_none() {
            pattern_node = Some(child);
        } else if body_node.is_none() {
            body_node = Some(child);
        }
    }

    let pattern_node = pattern_node?;
    let body_node = body_node?;

    // Create arm body
    let mut body_block = BlockBuilder::new(ctx.db, location);

    let result_value = ctx.scoped(|ctx| {
        // Bind handler pattern (value binding, continuation binding, etc.)
        bind_handler_pattern(ctx, &mut body_block, pattern_node, request);

        // Lower body
        lower_expr(ctx, &mut body_block, body_node)
    });

    let result_value = result_value?;
    body_block.op(case::r#yield(ctx.db, location, result_value));

    let pattern_region = handler_pattern_to_region(ctx, pattern_node);
    let body_region = Region::new(ctx.db, location, idvec![body_block.build()]);

    Some(case::arm(ctx.db, location, pattern_region, body_region))
}

/// Bind handler pattern variables.
fn bind_handler_pattern<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
    request: Value<'db>,
) {
    let location = ctx.location(&node);
    let mut cursor = node.walk();

    match node.kind() {
        "handler_pattern" => {
            // Handler patterns: { value } or { Op(args) -> k }
            let mut value_name = None;
            let mut op_name = None;
            let mut continuation_name = None;

            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                match child.kind() {
                    "identifier" if op_name.is_none() && value_name.is_none() => {
                        // Could be value binding or operation name
                        let name = node_text(&child, ctx.source);
                        // Check if this is followed by -> (continuation)
                        // For now, treat single identifier as value binding
                        value_name = Some(name);
                    }
                    "type_identifier" => {
                        // This is an ability operation name
                        op_name = Some(node_text(&child, ctx.source));
                    }
                    "identifier" if op_name.is_some() => {
                        // This is the continuation binding
                        continuation_name = Some(node_text(&child, ctx.source));
                    }
                    _ => {}
                }
            }

            // Bind the value (for { result } pattern)
            if let Some(name) = value_name
                && op_name.is_none()
            {
                // Simple value binding: { result }
                // The request's Done payload is bound to name
                let bind_op = block.op(case::bind(
                    ctx.db,
                    location,
                    ctx.fresh_type_var(),
                    Symbol::new(ctx.db, name),
                ));
                ctx.bind(name.to_string(), bind_op.result(ctx.db));
            }

            // Bind continuation (for { Op(args) -> k } pattern)
            if let Some(cont_name) = continuation_name {
                let cont_bind = block.op(case::bind(
                    ctx.db,
                    location,
                    ctx.fresh_type_var(),
                    Symbol::new(ctx.db, cont_name),
                ));
                ctx.bind(cont_name.to_string(), cont_bind.result(ctx.db));
            }

            // TODO: Bind operation arguments
            let _ = request; // Will be used for extracting payload
        }
        _ => {
            // Unknown pattern type
        }
    }
}

/// Convert a handler pattern to a pattern region.
///
/// Handler patterns are for ability effect handling:
/// - `{ result }` - Done pattern, binds the result value
/// - `{ Path::op(args) -> k }` - Suspend pattern, binds args and continuation
fn handler_pattern_to_region<'db, 'src>(
    ctx: &CstLoweringCtx<'db, 'src>,
    node: Node,
) -> Region<'db> {
    let location = ctx.location(&node);
    let mut cursor = node.walk();

    let mut result_name: Option<&str> = None;
    let mut operation_path: Option<Node> = None;
    let mut args_node: Option<Node> = None;
    let mut continuation_name: Option<&str> = None;

    // Parse handler pattern children
    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        match child.kind() {
            "identifier" => {
                // Could be result binding or continuation
                if operation_path.is_some() {
                    // After operation, this is the continuation
                    continuation_name = Some(node_text(&child, ctx.source));
                } else if result_name.is_none() {
                    // First identifier without operation is result binding
                    result_name = Some(node_text(&child, ctx.source));
                }
            }
            "path_expression" => {
                // This is the operation path (e.g., State::get)
                operation_path = Some(child);
            }
            "pattern_list" => {
                // Arguments to the operation
                args_node = Some(child);
            }
            _ => {}
        }
    }

    // Determine pattern type: Done or Suspend
    if let Some(op_path) = operation_path {
        // Suspend pattern: { Path::op(args) -> k }
        let (ability_ref, op_name) = parse_operation_path(ctx, op_path);

        // Build args pattern region
        let args_region = if let Some(args) = args_node {
            let mut ops = Vec::new();
            let mut args_cursor = args.walk();
            for arg_child in args.named_children(&mut args_cursor) {
                if is_comment(arg_child.kind()) {
                    continue;
                }
                let pat_region = pattern_to_region(ctx, arg_child);
                if let Some(op) = extract_pattern_op(ctx.db, &pat_region) {
                    ops.push(op);
                }
            }
            ops_to_region(ctx.db, location, ops)
        } else {
            pat::helpers::empty_region(ctx.db, location)
        };

        // Continuation name (empty Symbol for wildcard/discard)
        let cont_symbol = Symbol::new(ctx.db, continuation_name.unwrap_or("_"));

        pat::helpers::handler_suspend_region(
            ctx.db,
            location,
            ability_ref,
            op_name,
            args_region,
            cont_symbol,
        )
    } else {
        // Done pattern: { result }
        let result_region = match result_name {
            Some(name) => pat::helpers::bind_region(ctx.db, location, name),
            None => pat::helpers::wildcard_region(ctx.db, location),
        };

        pat::helpers::handler_done_region(ctx.db, location, result_region)
    }
}

/// Parse an operation path like `State::get` into (ability_ref, op_name).
fn parse_operation_path<'db, 'src>(
    ctx: &CstLoweringCtx<'db, 'src>,
    node: Node,
) -> (IdVec<Symbol<'db>>, Symbol<'db>) {
    let mut path_parts = Vec::new();
    let mut cursor = node.walk();

    // Collect all path components
    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if child.kind() == "identifier" || child.kind() == "type_identifier" {
            path_parts.push(node_text(&child, ctx.source));
        }
    }

    // If it's a single identifier, treat it as ability::op where ability is inferred
    // Otherwise, the last part is the operation name, rest is the ability path
    if path_parts.len() <= 1 {
        let op_name = path_parts.first().copied().unwrap_or("unknown");
        (
            IdVec::new(), // Empty ability ref (to be inferred)
            Symbol::new(ctx.db, op_name),
        )
    } else {
        let op_name = path_parts.pop().unwrap();
        let ability_ref: Vec<Symbol<'db>> = path_parts
            .into_iter()
            .map(|s| Symbol::new(ctx.db, s))
            .collect();
        (IdVec::from(ability_ref), Symbol::new(ctx.db, op_name))
    }
}

/// Lower a string interpolation expression.
fn lower_string_interpolation<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let string_ty = core::String::new(ctx.db).as_type();
    let infer_ty = ctx.fresh_type_var();

    let mut cursor = node.walk();
    let mut result: Option<Value<'db>> = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        let value = match child.kind() {
            "string_segment" | "multiline_string_segment" => {
                // Regular string content
                let content = node_text(&child, ctx.source);
                let op = block.op(adt::string_const(
                    ctx.db,
                    location,
                    string_ty,
                    content.to_string(),
                ));
                Some(op.result(ctx.db))
            }
            "interpolation" | "multiline_interpolation" => {
                // Interpolated expression ${expr}
                let mut interp_cursor = child.walk();
                let mut expr_value = None;
                for interp_child in child.named_children(&mut interp_cursor) {
                    if !is_comment(interp_child.kind()) {
                        expr_value = lower_expr(ctx, block, interp_child);
                        break;
                    }
                }

                // Convert to string using to_string
                expr_value.map(|v| {
                    block
                        .op(src::call(
                            ctx.db,
                            location,
                            vec![v],
                            string_ty,
                            sym_ref(ctx.db, "to_string"),
                        ))
                        .result(ctx.db)
                })
            }
            _ => lower_expr(ctx, block, child),
        };

        if let Some(v) = value {
            result = Some(match result {
                None => v,
                Some(r) => block
                    .op(src::binop(
                        ctx.db,
                        location,
                        r,
                        v,
                        infer_ty,
                        sym(ctx.db, "concat"),
                    ))
                    .result(ctx.db),
            });
        }
    }

    result.or_else(|| {
        // Empty string
        let op = block.op(adt::string_const(
            ctx.db,
            location,
            string_ty,
            String::new(),
        ));
        Some(op.result(ctx.db))
    })
}

/// Lower a bytes interpolation expression.
fn lower_bytes_interpolation<'db, 'src>(
    ctx: &mut CstLoweringCtx<'db, 'src>,
    block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<Value<'db>> {
    let location = ctx.location(&node);
    let bytes_ty = core::Bytes::new(ctx.db).as_type();
    let infer_ty = ctx.fresh_type_var();

    let mut cursor = node.walk();
    let mut result: Option<Value<'db>> = None;

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        let value = match child.kind() {
            "bytes_segment" | "multiline_bytes_segment" => {
                // Regular bytes content
                let content = node_text(&child, ctx.source);
                let op = block.op(adt::bytes_const(
                    ctx.db,
                    location,
                    bytes_ty,
                    Attribute::Bytes(content.as_bytes().to_vec()),
                ));
                Some(op.result(ctx.db))
            }
            "bytes_interpolation" | "multiline_bytes_interpolation" => {
                // Interpolated expression ${expr}
                let mut interp_cursor = child.walk();
                let mut expr_value = None;
                for interp_child in child.named_children(&mut interp_cursor) {
                    if !is_comment(interp_child.kind()) {
                        expr_value = lower_expr(ctx, block, interp_child);
                        break;
                    }
                }

                // Convert to bytes using to_bytes
                expr_value.map(|v| {
                    block
                        .op(src::call(
                            ctx.db,
                            location,
                            vec![v],
                            bytes_ty,
                            sym_ref(ctx.db, "to_bytes"),
                        ))
                        .result(ctx.db)
                })
            }
            _ => lower_expr(ctx, block, child),
        };

        if let Some(v) = value {
            result = Some(match result {
                None => v,
                Some(r) => block
                    .op(src::binop(
                        ctx.db,
                        location,
                        r,
                        v,
                        infer_ty,
                        sym(ctx.db, "concat"),
                    ))
                    .result(ctx.db),
            });
        }
    }

    result.or_else(|| {
        // Empty bytes
        let op = block.op(adt::bytes_const(
            ctx.db,
            location,
            bytes_ty,
            Attribute::Bytes(vec![]),
        ));
        Some(op.result(ctx.db))
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tribute_core::TributeDatabaseImpl;
    use tribute_trunk_ir::DialectOp;
    use tribute_trunk_ir::dialect::func;

    fn lower_and_get_module<'db>(db: &'db TributeDatabaseImpl, source: &str) -> core::Module<'db> {
        let file = SourceFile::new(db, std::path::PathBuf::from("test.tr"), source.to_string());
        lower_source_file(db, file)
    }

    #[test]
    fn test_simple_function() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { 42 }";
        let module = lower_and_get_module(&db, source);

        // Check module has a function named "main"
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty(), "Module should have at least one block");

        let ops = blocks[0].operations(&db);
        assert!(!ops.is_empty(), "Block should have at least one operation");

        // Check first op is a function
        let func_op = func::Func::from_operation(&db, ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "main");
    }

    #[test]
    fn test_nat_literal() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { 123 }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_binary_expression() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { 1 + 2 }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_let_binding() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { let x = 10; x }";
        let module = lower_and_get_module(&db, source);

        // Verify module was created successfully
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_tuple_pattern() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { let #(a, b) = #(1, 2); a + b }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_list_expression() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { [1, 2, 3] }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_case_expression() {
        let db = TributeDatabaseImpl::default();
        let source = r#"
            fn main() {
                let x = 1;
                case x {
                    0 { "zero" }
                    1 { "one" }
                    _ { "other" }
                }
            }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_lambda_expression() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { fn(x) { x + 1 } }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_method_call() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { [1, 2, 3].len() }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_string_literal() {
        let db = TributeDatabaseImpl::default();
        let source = r#"fn main() { "hello" }"#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_wildcard_pattern() {
        let db = TributeDatabaseImpl::default();
        let source = "fn main() { let _ = 42; 0 }";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_struct_declaration() {
        let db = TributeDatabaseImpl::default();
        let source = r#"
            struct Point {
                x: Int,
                y: Int,
            }
            fn main() { 0 }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        // Should have at least 2 operations (struct and function)
        let ops = blocks[0].operations(&db);
        assert!(ops.len() >= 2, "Should have struct and function");
    }

    #[test]
    fn test_enum_declaration() {
        let db = TributeDatabaseImpl::default();
        let source = r#"
            enum Option(a) {
                Some(a),
                None,
            }
            fn main() { 0 }
        "#;
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        let ops = blocks[0].operations(&db);
        assert!(ops.len() >= 2, "Should have enum and function");
    }

    #[test]
    fn test_const_declaration() {
        let db = TributeDatabaseImpl::default();
        // Test const declaration lowered to src.const
        // Note: uppercase identifiers like PI are parsed as type_identifier by the grammar
        // so we use lowercase for const names
        let source = "const pi = 42";
        let module = lower_and_get_module(&db, source);

        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        let ops = blocks[0].operations(&db);
        // Now only src.const is generated (no separate arith.const for the value)
        assert!(!ops.is_empty(), "Should have at least one operation");

        // The first op should be src.const
        let const_op = src::Const::from_operation(&db, ops[0]).expect("Should be a src.const");
        assert_eq!(const_op.name(&db).text(&db), "pi");
    }

    #[test]
    fn test_inline_module() {
        let db = TributeDatabaseImpl::default();
        let source = r#"
            pub mod math {
                pub fn add(x: Int, y: Int) -> Int {
                    x + y
                }
            }
        "#;
        let module = lower_and_get_module(&db, source);

        // Top-level module should contain a nested module
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        assert!(!blocks.is_empty());

        let ops = blocks[0].operations(&db);
        assert!(!ops.is_empty(), "Should have at least one operation");

        // The first op should be core.module (the nested "math" module)
        let nested_module =
            core::Module::from_operation(&db, ops[0]).expect("Should be a core.module");
        assert_eq!(nested_module.name(&db), "math");

        // The nested module should contain the "add" function
        let nested_body = nested_module.body(&db);
        let nested_blocks = nested_body.blocks(&db);
        assert!(!nested_blocks.is_empty());

        let nested_ops = nested_blocks[0].operations(&db);
        assert!(!nested_ops.is_empty(), "Nested module should have operations");

        let func_op =
            func::Func::from_operation(&db, nested_ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "add");
    }

    #[test]
    fn test_nested_modules() {
        let db = TributeDatabaseImpl::default();
        let source = r#"
            pub mod outer {
                pub mod inner {
                    pub fn value() -> Int { 42 }
                }
            }
        "#;
        let module = lower_and_get_module(&db, source);

        // Get the outer module
        let body_region = module.body(&db);
        let blocks = body_region.blocks(&db);
        let ops = blocks[0].operations(&db);
        let outer_module =
            core::Module::from_operation(&db, ops[0]).expect("Should be a core.module");
        assert_eq!(outer_module.name(&db), "outer");

        // Get the inner module
        let outer_body = outer_module.body(&db);
        let outer_blocks = outer_body.blocks(&db);
        let outer_ops = outer_blocks[0].operations(&db);
        let inner_module =
            core::Module::from_operation(&db, outer_ops[0]).expect("Should be a core.module");
        assert_eq!(inner_module.name(&db), "inner");

        // Check the function inside inner
        let inner_body = inner_module.body(&db);
        let inner_blocks = inner_body.blocks(&db);
        let inner_ops = inner_blocks[0].operations(&db);
        let func_op =
            func::Func::from_operation(&db, inner_ops[0]).expect("Should be a func.func");
        assert_eq!(func_op.name(&db), "value");
    }
}

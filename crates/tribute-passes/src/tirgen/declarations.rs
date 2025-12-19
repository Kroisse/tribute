//! Declaration lowering for functions, types, and modules.

use tree_sitter::Node;
use tribute_core::Span;
use tribute_trunk_ir::{
    Attribute, BlockBuilder, IdVec, Symbol, Type,
    dialect::{core, func, src, ty},
};

use super::context::CstLoweringCtx;
use super::helpers::{is_comment, node_text, sym};
use super::literals::{parse_int_literal, parse_nat_literal, parse_float_literal, parse_rune_literal, parse_string_literal};
use super::statements::lower_block_body;

// =============================================================================
// Use Declaration Lowering
// =============================================================================

#[derive(Debug)]
struct UseImport {
    path: Vec<String>,
    alias: Option<String>,
}

pub fn lower_use_decl<'db, 'src>(
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
// Function Lowering
// =============================================================================

/// Lower a function definition to a func.func operation.
pub fn lower_function<'db, 'src>(
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
pub fn lower_struct_decl<'db, 'src>(
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
pub fn lower_enum_decl<'db, 'src>(
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
pub fn lower_const_decl<'db, 'src>(
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
fn unwrap_expression_node<'tree>(node: Node<'tree>, source: &str) -> Option<Node<'tree>> {
    match node.kind() {
        "primary_expression" | "expression" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return unwrap_expression_node(child, source);
                }
            }
            None
        }
        _ => Some(node),
    }
}

/// Lower an ability declaration to type.ability.
pub fn lower_ability_decl<'db, 'src>(
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
pub fn lower_mod_decl<'db, 'src>(
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
pub fn lower_mod_body<'db, 'src>(
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
pub fn parse_parameter_list<'db, 'src>(
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
pub fn parse_return_type<'db, 'src>(
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

//! CST to AST lowering for declarations.

use tree_sitter::Node;
use trunk_ir::Symbol;

use crate::ast::{
    AbilityDecl, Decl, EnumDecl, FieldDecl, FuncDecl, Module, ModuleDecl, OpDecl, ParamDecl,
    StructDecl, TypeAnnotation, TypeAnnotationKind, UnresolvedName, UseDecl, VariantDecl,
};

use super::context::AstLoweringCtx;
use super::expressions::lower_expr;
use super::helpers::is_comment;

/// Lower a CST source file to an AST Module.
///
/// The `module_name` parameter is the name derived from the source file path.
/// If `None`, a later phase (or the IR lowering) can assign a default.
pub fn lower_module(
    ctx: &mut AstLoweringCtx,
    root: Node,
    module_name: Option<Symbol>,
) -> Module<UnresolvedName> {
    let id = ctx.fresh_id_with_span(&root);
    let name = module_name;
    let mut decls = Vec::new();

    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        decls.extend(lower_decl(ctx, child));
    }

    Module { id, name, decls }
}

/// Lower a CST declaration node to an AST Decl.
/// Returns a Vec because use declarations can expand to multiple items.
fn lower_decl(ctx: &mut AstLoweringCtx, node: Node) -> Vec<Decl<UnresolvedName>> {
    match node.kind() {
        "function_definition" => lower_function(ctx, node).into_iter().collect(),
        "struct_declaration" => lower_struct(ctx, node)
            .map(Decl::Struct)
            .into_iter()
            .collect(),
        "enum_declaration" => lower_enum(ctx, node).map(Decl::Enum).into_iter().collect(),
        "ability_declaration" => lower_ability(ctx, node)
            .map(Decl::Ability)
            .into_iter()
            .collect(),
        "mod_declaration" => lower_mod(ctx, node).map(Decl::Module).into_iter().collect(),
        "use_declaration" => lower_use(ctx, node).into_iter().map(Decl::Use).collect(),
        _ => Vec::new(),
    }
}

/// Lower a function definition.
///
/// Returns either `Decl::Function` or `Decl::ExternFunction` depending on whether
/// the function is `regular_function` or `extern_function`.
fn lower_function(ctx: &mut AstLoweringCtx, node: Node) -> Option<Decl<UnresolvedName>> {
    // function_definition always contains regular_function or extern_function
    let func_node = node
        .named_child(0)
        .filter(|c| c.kind() == "regular_function" || c.kind() == "extern_function")
        .unwrap_or_else(|| {
            unreachable!(
                "function_definition must contain regular_function or extern_function, got: {:?}",
                node.named_child(0).map(|c| c.kind())
            )
        });

    let id = ctx.fresh_id_with_span(&func_node);

    let is_extern = func_node.kind() == "extern_function";

    let name_node = func_node
        .child_by_field_name("name")
        .expect("function must have a name");
    let body_node = func_node.child_by_field_name("body");

    // Regular functions must have bodies
    if !is_extern && body_node.is_none() {
        let span = trunk_ir::Span::new(func_node.start_byte(), func_node.end_byte());
        ctx.error(span, "function is missing a body");
        return None;
    }

    let name = extract_function_name(ctx, name_node);
    let params = func_node
        .child_by_field_name("params")
        .map(|n| lower_param_list(ctx, n))
        .unwrap_or_default();

    let return_ty = func_node
        .child_by_field_name("return_type")
        .and_then(|n| lower_type_annotation(ctx, n));

    if is_extern {
        // For extern functions, return type defaults to Nil when omitted.
        let extern_return_ty = return_ty.clone().unwrap_or_else(|| TypeAnnotation {
            id,
            kind: TypeAnnotationKind::Named(Symbol::new("Nil")),
        });
        // ABI is a field of the `extern_marker` child, not `extern_function` directly.
        // Grammar: extern_marker -> seq(keyword_extern, optional(field("abi", $.string)))
        let abi = func_node
            .children(&mut func_node.walk())
            .find(|c| c.kind() == "extern_marker")
            .and_then(|marker| marker.child_by_field_name("abi"))
            .map(|n| {
                let text = ctx.node_text(&n);
                // Strip surrounding quotes: "intrinsic" -> intrinsic
                Symbol::from_dynamic(text.trim_matches('"'))
            })
            .unwrap_or_else(|| Symbol::new("C"));

        return Some(Decl::ExternFunction(crate::ast::ExternFuncDecl {
            id,
            is_pub: false, // TODO: parse visibility
            name,
            abi,
            params,
            return_ty: extern_return_ty,
        }));
    }

    let body = lower_expr(ctx, body_node.unwrap());

    Some(Decl::Function(FuncDecl {
        id,
        is_pub: false, // TODO: parse visibility
        name,
        type_params: Vec::new(), // TODO: parse type params
        params,
        return_ty,
        effects: None, // TODO: parse effects
        body,
    }))
}

/// Extract function name, handling operator names like `(<>)`.
fn extract_function_name(ctx: &AstLoweringCtx, name_node: Node) -> Symbol {
    if name_node.kind() == "operator_name" {
        let text = ctx.node_text(&name_node);
        // Strip surrounding parentheses: "(<>)" -> "<>"
        let stripped = text
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or(&text);
        Symbol::from_dynamic(stripped)
    } else {
        ctx.node_symbol(&name_node)
    }
}

/// Lower a parameter list.
fn lower_param_list(ctx: &mut AstLoweringCtx, node: Node) -> Vec<ParamDecl> {
    let mut params = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if child.kind() == "parameter"
            && let Some(name_node) = child.child_by_field_name("name")
        {
            let id = ctx.fresh_id_with_span(&child);
            let name = ctx.node_symbol(&name_node);
            let ty = child
                .child_by_field_name("type")
                .and_then(|n| lower_type_annotation(ctx, n));
            params.push(ParamDecl {
                id,
                name,
                ty,
                local_id: None,
            });
        }
    }

    params
}

/// Lower a type annotation.
fn lower_type_annotation(ctx: &mut AstLoweringCtx, node: Node) -> Option<TypeAnnotation> {
    // Check if the node itself is already a type node
    if matches!(
        node.kind(),
        "type_identifier" | "type_variable" | "generic_type" | "function_type" | "tuple_type"
    ) {
        return lower_type_node(ctx, node);
    }

    // Otherwise, search children for the actual type node
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "type_identifier" | "type_variable" | "generic_type" | "function_type"
            | "tuple_type" => {
                return lower_type_node(ctx, child);
            }
            _ => {}
        }
    }

    None
}

/// Lower a type node directly (for variant tuple fields).
fn lower_type_node(ctx: &mut AstLoweringCtx, node: Node) -> Option<TypeAnnotation> {
    // Handle various type node kinds
    match node.kind() {
        "type_identifier" | "type_variable" => {
            let id = ctx.fresh_id_with_span(&node);
            let name = ctx.node_symbol(&node);
            Some(TypeAnnotation {
                id,
                kind: TypeAnnotationKind::Named(name),
            })
        }
        "generic_type" => {
            // Generic type like List(a) or Map(k, v)
            let id = ctx.fresh_id_with_span(&node);

            // Get the constructor type (first child is the type identifier)
            let mut cursor = node.walk();
            let children: Vec<_> = node.named_children(&mut cursor).collect();

            if let Some(name_node) = children.first().filter(|n| n.kind() == "type_identifier") {
                let ctor = Box::new(TypeAnnotation {
                    id: ctx.fresh_id_with_span(name_node),
                    kind: TypeAnnotationKind::Named(ctx.node_symbol(name_node)),
                });

                // Collect type arguments (remaining children)
                let args: Vec<TypeAnnotation> = children
                    .iter()
                    .skip(1)
                    .filter_map(|child| lower_type_node(ctx, *child))
                    .collect();

                return Some(TypeAnnotation {
                    id,
                    kind: TypeAnnotationKind::App { ctor, args },
                });
            }
            None
        }
        "function_type" => {
            // Function type: fn(Int, Int) -> Int or fn(a) ->{State(s)} b
            let id = ctx.fresh_id_with_span(&node);

            // Get parameter types from "params" field (type_list)
            let params: Vec<TypeAnnotation> = node
                .child_by_field_name("params")
                .map(|params_node| {
                    let mut cursor = params_node.walk();
                    params_node
                        .named_children(&mut cursor)
                        .filter_map(|child| lower_type_node(ctx, child))
                        .collect()
                })
                .unwrap_or_default();

            // Get return type from "return_type" field
            let result = node
                .child_by_field_name("return_type")
                .and_then(|rt| lower_type_node(ctx, rt))
                .map(Box::new)?;

            // No ability row → effect polymorphic (single Infer as row variable)
            // Empty ability row `{}` → pure (empty vec)
            // Explicit abilities → listed in vec
            let abilities = match node.child_by_field_name("abilities") {
                Some(abilities_node) => lower_ability_row(ctx, abilities_node),
                None => vec![TypeAnnotation {
                    id: ctx.fresh_id_with_span(&node),
                    kind: TypeAnnotationKind::Infer,
                }],
            };

            Some(TypeAnnotation {
                id,
                kind: TypeAnnotationKind::Func {
                    params,
                    result,
                    abilities,
                },
            })
        }
        "tuple_type" => {
            // Tuple type: #(Int, String)
            let id = ctx.fresh_id_with_span(&node);
            let mut cursor = node.walk();
            let elements: Vec<TypeAnnotation> = node
                .named_children(&mut cursor)
                .filter_map(|child| lower_type_node(ctx, child))
                .collect();
            Some(TypeAnnotation {
                id,
                kind: TypeAnnotationKind::Tuple(elements),
            })
        }
        _ => {
            // Try to find type within the node
            lower_type_annotation(ctx, node)
        }
    }
}

/// Lower an ability_row CST node into a list of effect type annotations.
///
/// Grammar: `ability_row = "{" ability_list? "}"`
/// where `ability_list` contains `ability_item` (e.g. `State(Int)`) and
/// `ability_tail` (row variable, e.g. `e`).
fn lower_ability_row(ctx: &mut AstLoweringCtx, node: Node) -> Vec<TypeAnnotation> {
    let mut effects = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "ability_list" {
            let mut inner_cursor = child.walk();
            for item in child.named_children(&mut inner_cursor) {
                match item.kind() {
                    "ability_item" => {
                        if let Some(ann) = lower_ability_item(ctx, item) {
                            effects.push(ann);
                        }
                    }
                    "ability_tail" => {
                        // Row variable (e.g. `e`)
                        let id = ctx.fresh_id_with_span(&item);
                        let name = ctx.node_symbol(&item);
                        effects.push(TypeAnnotation {
                            id,
                            kind: TypeAnnotationKind::Named(name),
                        });
                    }
                    _ => {}
                }
            }
        }
    }
    effects
}

/// Lower an ability_item into a TypeAnnotation.
///
/// Grammar: `ability_item = type_identifier optional(type_arguments)`
/// e.g. `Console` or `State(Int)`
fn lower_ability_item(ctx: &mut AstLoweringCtx, node: Node) -> Option<TypeAnnotation> {
    let mut cursor = node.walk();
    let children: Vec<_> = node.named_children(&mut cursor).collect();

    let name_node = children.first().filter(|n| n.kind() == "type_identifier")?;
    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(name_node);

    // Check for type_arguments (e.g. State(Int))
    let type_args_node = children.iter().find(|n| n.kind() == "type_arguments");
    if let Some(args_node) = type_args_node {
        let ctor = Box::new(TypeAnnotation {
            id: ctx.fresh_id_with_span(name_node),
            kind: TypeAnnotationKind::Named(name),
        });
        let mut args_cursor = args_node.walk();
        let args: Vec<TypeAnnotation> = args_node
            .named_children(&mut args_cursor)
            .filter_map(|child| lower_type_node(ctx, child))
            .collect();
        Some(TypeAnnotation {
            id,
            kind: TypeAnnotationKind::App { ctor, args },
        })
    } else {
        Some(TypeAnnotation {
            id,
            kind: TypeAnnotationKind::Named(name),
        })
    }
}

/// Lower a struct declaration.
fn lower_struct(ctx: &mut AstLoweringCtx, node: Node) -> Option<StructDecl> {
    let name_node = node.child_by_field_name("name")?;
    let body_node = node.child_by_field_name("body")?;

    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(&name_node);
    let fields = lower_struct_fields(ctx, body_node);

    Some(StructDecl {
        id,
        is_pub: false, // TODO: parse visibility
        name,
        type_params: Vec::new(), // TODO: parse type params
        fields,
    })
}

/// Lower struct fields from struct body.
fn lower_struct_fields(ctx: &mut AstLoweringCtx, node: Node) -> Vec<FieldDecl> {
    let mut fields = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        // Handle struct_fields wrapper
        if child.kind() == "struct_fields" {
            let mut inner_cursor = child.walk();
            for field_child in child.named_children(&mut inner_cursor) {
                if field_child.kind() == "struct_field"
                    && let Some(field) = lower_struct_field(ctx, field_child)
                {
                    fields.push(field);
                }
            }
            continue;
        }

        if child.kind() == "struct_field"
            && let Some(field) = lower_struct_field(ctx, child)
        {
            fields.push(field);
        }
    }

    fields
}

/// Lower a single struct field.
fn lower_struct_field(ctx: &mut AstLoweringCtx, node: Node) -> Option<FieldDecl> {
    let name_node = node.child_by_field_name("name")?;
    let type_node = node.child_by_field_name("type")?;

    let id = ctx.fresh_id_with_span(&node);
    let name = Some(ctx.node_symbol(&name_node));
    let ty = lower_type_annotation(ctx, type_node)?;

    Some(FieldDecl {
        id,
        is_pub: false, // TODO: parse visibility
        name,
        ty,
    })
}

/// Lower an enum declaration.
fn lower_enum(ctx: &mut AstLoweringCtx, node: Node) -> Option<EnumDecl> {
    let name_node = node.child_by_field_name("name")?;
    let body_node = node.child_by_field_name("body")?;

    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(&name_node);
    let variants = lower_enum_variants(ctx, body_node);

    Some(EnumDecl {
        id,
        is_pub: false, // TODO: parse visibility
        name,
        type_params: Vec::new(), // TODO: parse type params
        variants,
    })
}

/// Lower enum variants from enum body.
fn lower_enum_variants(ctx: &mut AstLoweringCtx, node: Node) -> Vec<VariantDecl> {
    let mut variants = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        if child.kind() == "enum_variants" {
            let mut variant_cursor = child.walk();
            for variant_node in child.named_children(&mut variant_cursor) {
                if variant_node.kind() == "enum_variant"
                    && let Some(variant) = lower_enum_variant(ctx, variant_node)
                {
                    variants.push(variant);
                }
            }
        }
    }

    variants
}

/// Lower a single enum variant.
fn lower_enum_variant(ctx: &mut AstLoweringCtx, node: Node) -> Option<VariantDecl> {
    let name_node = node.child_by_field_name("name")?;
    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(&name_node);

    // Parse variant fields (tuple or record style)
    let fields = if let Some(fields_node) = node.child_by_field_name("fields") {
        lower_variant_fields(ctx, fields_node)
    } else {
        Vec::new()
    };

    Some(VariantDecl { id, name, fields })
}

/// Lower variant fields (tuple or struct style).
fn lower_variant_fields(ctx: &mut AstLoweringCtx, node: Node) -> Vec<FieldDecl> {
    let mut fields = Vec::new();

    match node.kind() {
        // Tuple fields: (Type1, Type2, ...)
        "tuple_fields" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if is_comment(child.kind()) {
                    continue;
                }
                // Each child is a type - create anonymous field
                if let Some(ty) = lower_type_node(ctx, child) {
                    let id = ctx.fresh_id_with_span(&child);
                    fields.push(FieldDecl {
                        id,
                        is_pub: false,
                        name: None, // Tuple fields are anonymous
                        ty,
                    });
                }
            }
        }
        // Struct fields: { name: Type, ... }
        "struct_fields_block" => {
            fields = lower_struct_fields(ctx, node);
        }
        _ => {
            // Unknown variant field type - skip
        }
    }

    fields
}

/// Lower an ability declaration.
fn lower_ability(ctx: &mut AstLoweringCtx, node: Node) -> Option<AbilityDecl> {
    let name_node = node.child_by_field_name("name")?;
    let body_node = node.child_by_field_name("body")?;

    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(&name_node);
    let operations = lower_ability_operations(ctx, body_node);

    Some(AbilityDecl {
        id,
        is_pub: false, // TODO: parse visibility
        name,
        type_params: Vec::new(), // TODO: parse type params
        operations,
    })
}

/// Lower ability operations from ability body.
fn lower_ability_operations(ctx: &mut AstLoweringCtx, node: Node) -> Vec<OpDecl> {
    let mut operations = Vec::new();
    let mut cursor = node.walk();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        // Handle ability_operations wrapper (like enum_variants, struct_fields)
        if child.kind() == "ability_operations" {
            let mut inner_cursor = child.walk();
            for op_node in child.named_children(&mut inner_cursor) {
                if op_node.kind() == "ability_operation"
                    && let Some(op) = lower_ability_operation(ctx, op_node)
                {
                    operations.push(op);
                }
            }
            continue;
        }

        // Also handle direct ability_operation children (for compatibility)
        if child.kind() == "ability_operation"
            && let Some(op) = lower_ability_operation(ctx, child)
        {
            operations.push(op);
        }
    }

    operations
}

/// Lower a single ability operation.
fn lower_ability_operation(ctx: &mut AstLoweringCtx, node: Node) -> Option<OpDecl> {
    let name_node = node.child_by_field_name("name")?;
    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(&name_node);

    let params = find_child_by_kind(node, "parameter_list")
        .map(|n| lower_param_list(ctx, n))
        .unwrap_or_default();

    let return_ty = node
        .child_by_field_name("return_type")
        .and_then(|n| lower_type_annotation(ctx, n))
        .unwrap_or_else(|| {
            let id = ctx.fresh_id_with_span(&node);
            TypeAnnotation {
                id,
                kind: TypeAnnotationKind::Named(Symbol::new("()")),
            }
        });

    Some(OpDecl {
        id,
        name,
        params,
        return_ty,
    })
}

/// Lower a use declaration.
/// Returns a Vec because `use std::{io, fmt}` expands to multiple UseDecls.
fn lower_use(ctx: &mut AstLoweringCtx, node: Node) -> Vec<UseDecl> {
    let Some(tree_node) = node.child_by_field_name("tree") else {
        return Vec::new();
    };

    let is_pub = node
        .named_children(&mut node.walk())
        .any(|child| child.kind() == "visibility_marker");

    // Expand the use tree into multiple (path, alias) pairs
    let expanded = expand_use_tree(ctx, tree_node, &[]);

    expanded
        .into_iter()
        .map(|(path, alias)| UseDecl {
            id: ctx.fresh_id_with_span(&node),
            is_pub,
            path,
            alias,
        })
        .collect()
}

/// Expand a use tree into a list of (path, alias) pairs.
/// Handles nested groups like `use std::{io, fmt, collections::{List, Map}}`.
fn expand_use_tree(
    ctx: &AstLoweringCtx,
    node: Node,
    prefix: &[Symbol],
) -> Vec<(Vec<Symbol>, Option<Symbol>)> {
    let mut results = Vec::new();

    match node.kind() {
        "use_tree" => {
            // Check for alias
            let alias = node
                .child_by_field_name("alias")
                .map(|n| ctx.node_symbol(&n));

            // Collect path segments and look for continuation
            let mut path: Vec<Symbol> = prefix.to_vec();
            let mut cursor = node.walk();

            for child in node.named_children(&mut cursor) {
                // Skip alias node
                if node.child_by_field_name("alias").map(|n| n.id()) == Some(child.id()) {
                    continue;
                }

                match child.kind() {
                    "identifier" | "type_identifier" | "path_keyword" => {
                        path.push(ctx.node_symbol(&child));
                    }
                    "use_tree" => {
                        // Nested use_tree: recurse with current path as prefix
                        results.extend(expand_use_tree(ctx, child, &path));
                    }
                    "use_group" => {
                        // Group: {A, B, C} - expand each child with current prefix
                        let mut group_cursor = child.walk();
                        for group_child in child.named_children(&mut group_cursor) {
                            if group_child.kind() == "use_tree" {
                                results.extend(expand_use_tree(ctx, group_child, &path));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // If we didn't recurse into a group/nested tree, this is a terminal path
            if results.is_empty() && !path.is_empty() {
                results.push((path, alias));
            }
        }
        "identifier" | "type_identifier" | "path_keyword" => {
            let mut path = prefix.to_vec();
            path.push(ctx.node_symbol(&node));
            results.push((path, None));
        }
        _ => {}
    }

    results
}

/// Lower a module declaration.
fn lower_mod(ctx: &mut AstLoweringCtx, node: Node) -> Option<ModuleDecl<UnresolvedName>> {
    let name_node = node.child_by_field_name("name")?;

    let id = ctx.fresh_id_with_span(&node);
    let name = ctx.node_symbol(&name_node);
    let is_pub = node
        .named_children(&mut node.walk())
        .any(|child| child.kind() == "visibility_marker");

    // Check for inline module body
    let body = if let Some(body_node) = node.child_by_field_name("body") {
        let mut decls = Vec::new();
        let mut cursor = body_node.walk();
        for child in body_node.named_children(&mut cursor) {
            if is_comment(child.kind()) {
                continue;
            }
            decls.extend(lower_decl(ctx, child));
        }
        Some(decls)
    } else {
        // External module (file-based) - no inline body
        None
    };

    Some(ModuleDecl {
        id,
        name,
        is_pub,
        body,
    })
}

/// Find a child node by kind.
fn find_child_by_kind<'tree>(node: Node<'tree>, kind: &str) -> Option<Node<'tree>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .find(|child| child.kind() == kind)
}

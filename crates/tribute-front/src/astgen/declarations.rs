//! CST to AST lowering for declarations.

use tree_sitter::Node;
use trunk_ir::Symbol;

use crate::ast::{
    AbilityDecl, Decl, EnumDecl, FieldDecl, FuncDecl, Module, OpDecl, ParamDecl, StructDecl,
    TypeAnnotation, TypeAnnotationKind, UnresolvedName, UseDecl, VariantDecl,
};

use super::context::AstLoweringCtx;
use super::expressions::lower_expr;
use super::helpers::is_comment;

/// Lower a CST source file to an AST Module.
pub fn lower_module(ctx: &mut AstLoweringCtx, root: Node) -> Module<UnresolvedName> {
    let id = ctx.fresh_id_with_span(&root);
    let name = Some(Symbol::new("main"));
    let mut decls = Vec::new();

    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }

        if let Some(decl) = lower_decl(ctx, child) {
            decls.push(decl);
        }
    }

    Module { id, name, decls }
}

/// Lower a CST declaration node to an AST Decl.
fn lower_decl(ctx: &mut AstLoweringCtx, node: Node) -> Option<Decl<UnresolvedName>> {
    match node.kind() {
        "function_definition" => lower_function(ctx, node).map(Decl::Function),
        "struct_declaration" => lower_struct(ctx, node).map(Decl::Struct),
        "enum_declaration" => lower_enum(ctx, node).map(Decl::Enum),
        "ability_declaration" => lower_ability(ctx, node).map(Decl::Ability),
        "mod_declaration" => {
            lower_mod(ctx, node).map(|m| Decl::Function(dummy_func_for_module(ctx, m)))
        }
        "use_declaration" => lower_use(ctx, node).map(Decl::Use),
        "const_declaration" => {
            // TODO: implement const declarations
            None
        }
        _ => None,
    }
}

/// Create a dummy function to represent a nested module (temporary workaround).
/// TODO: Add proper Module variant to Decl<V>
fn dummy_func_for_module(
    _ctx: &mut AstLoweringCtx,
    _module: Module<UnresolvedName>,
) -> FuncDecl<UnresolvedName> {
    // For now, we skip nested modules - they need proper Decl::Module support
    // This is a placeholder that should be replaced when Decl<V> is updated
    unreachable!("Nested modules not yet supported in AST")
}

/// Lower a function definition.
fn lower_function(ctx: &mut AstLoweringCtx, node: Node) -> Option<FuncDecl<UnresolvedName>> {
    // function_definition contains either regular_function or extern_function
    let func_node = node
        .named_child(0)
        .filter(|c| c.kind() == "regular_function" || c.kind() == "extern_function")?;

    let id = ctx.fresh_id_with_span(&func_node);

    let is_extern = func_node.kind() == "extern_function";

    let name_node = func_node.child_by_field_name("name")?;
    let body_node = func_node.child_by_field_name("body");

    // Extern functions don't have bodies
    if !is_extern && body_node.is_none() {
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

    // For extern functions, create a unit expression as placeholder
    let body = if is_extern {
        let unit_id = ctx.fresh_id_with_span(&func_node);
        crate::ast::Expr::new(unit_id, crate::ast::ExprKind::UnitLit)
    } else {
        lower_expr(ctx, body_node?)
    };

    Some(FuncDecl {
        id,
        is_pub: false, // TODO: parse visibility
        name,
        type_params: Vec::new(), // TODO: parse type params
        params,
        return_ty,
        effects: None, // TODO: parse effects
        body,
    })
}

/// Extract function name, handling operator names like `(<>)`.
fn extract_function_name(ctx: &AstLoweringCtx, name_node: Node) -> Symbol {
    if name_node.kind() == "operator_name" {
        let text = ctx.node_text(&name_node);
        // Strip surrounding parentheses: "(<>)" -> "<>"
        let stripped = text
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or(text);
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
            params.push(ParamDecl { id, name, ty });
        }
    }

    params
}

/// Lower a type annotation.
fn lower_type_annotation(ctx: &mut AstLoweringCtx, node: Node) -> Option<TypeAnnotation> {
    // Find the actual type node within the annotation
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "type_identifier" | "type_variable" | "generic_type" | "function_type"
            | "tuple_type" => {
                let id = ctx.fresh_id_with_span(&child);
                let name = ctx.node_symbol(&child);
                return Some(TypeAnnotation {
                    id,
                    kind: TypeAnnotationKind::Named(name),
                });
            }
            _ => {}
        }
    }

    // Fallback: use the node itself if it's a type
    if matches!(
        node.kind(),
        "type_identifier" | "type_variable" | "generic_type"
    ) {
        let id = ctx.fresh_id_with_span(&node);
        let name = ctx.node_symbol(&node);
        return Some(TypeAnnotation {
            id,
            kind: TypeAnnotationKind::Named(name),
        });
    }

    None
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

    // TODO: parse variant fields (tuple or record style)
    let fields = Vec::new();

    Some(VariantDecl { id, name, fields })
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

/// Lower a mod declaration.
fn lower_mod(ctx: &mut AstLoweringCtx, node: Node) -> Option<Module<UnresolvedName>> {
    let name_node = node.child_by_field_name("name")?;
    let body_node = node.child_by_field_name("body");

    let id = ctx.fresh_id_with_span(&node);
    let name = Some(ctx.node_symbol(&name_node));

    let mut decls = Vec::new();

    if let Some(body) = body_node {
        let mut cursor = body.walk();
        for child in body.named_children(&mut cursor) {
            if is_comment(child.kind()) {
                continue;
            }
            if let Some(decl) = lower_decl(ctx, child) {
                decls.push(decl);
            }
        }
    }

    Some(Module { id, name, decls })
}

/// Lower a use declaration.
fn lower_use(ctx: &mut AstLoweringCtx, node: Node) -> Option<UseDecl> {
    let tree_node = node.child_by_field_name("tree")?;

    let id = ctx.fresh_id_with_span(&node);
    let is_pub = node
        .named_children(&mut node.walk())
        .any(|child| child.kind() == "visibility_marker");

    let (path, alias) = collect_use_path(ctx, tree_node);

    Some(UseDecl {
        id,
        is_pub,
        path,
        alias,
    })
}

/// Collect use path from a use tree node.
fn collect_use_path(ctx: &AstLoweringCtx, node: Node) -> (Vec<Symbol>, Option<Symbol>) {
    let mut path = Vec::new();
    let mut alias = None;
    collect_use_path_impl(ctx, node, &mut path, &mut alias);
    (path, alias)
}

fn collect_use_path_impl(
    ctx: &AstLoweringCtx,
    node: Node,
    path: &mut Vec<Symbol>,
    alias: &mut Option<Symbol>,
) {
    match node.kind() {
        "use_tree" => {
            if let Some(alias_node) = node.child_by_field_name("alias") {
                *alias = Some(ctx.node_symbol(&alias_node));
            }

            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if node.child_by_field_name("alias").map(|n| n.id()) == Some(child.id()) {
                    continue;
                }
                match child.kind() {
                    "identifier" | "type_identifier" | "path_keyword" => {
                        path.push(ctx.node_symbol(&child));
                    }
                    "use_tree" => {
                        collect_use_path_impl(ctx, child, path, alias);
                    }
                    _ => {}
                }
            }
        }
        "identifier" | "type_identifier" | "path_keyword" => {
            path.push(ctx.node_symbol(&node));
        }
        _ => {}
    }
}

/// Find a child node by kind.
fn find_child_by_kind<'tree>(node: Node<'tree>, kind: &str) -> Option<Node<'tree>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .find(|child| child.kind() == kind)
}

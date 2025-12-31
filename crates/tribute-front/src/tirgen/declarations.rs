//! Declaration lowering for functions, types, and modules.

use tree_sitter::Node;
use tribute_ir::dialect::{ability, adt, src, ty};
use trunk_ir::{
    Attribute, BlockBuilder, DialectType, IdVec, QualifiedName, Region, Span, Symbol, SymbolVec,
    Type,
    dialect::{core, func},
    idvec,
};

use super::context::CstLoweringCtx;
use super::helpers::{is_comment, node_text, sym};
use super::literals::{
    parse_float_literal, parse_int_literal, parse_nat_literal, parse_rune_literal,
    parse_string_literal,
};
use super::statements::lower_block_body;

// =============================================================================
// Use Declaration Lowering
// =============================================================================

#[derive(Debug)]
struct UseImport {
    path: QualifiedName,
    alias: Option<Symbol>,
}

pub fn lower_use_decl<'db>(
    ctx: &mut CstLoweringCtx<'db>,
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
    collect_use_imports(ctx, tree_node, &mut SymbolVec::new(), &mut imports);

    for import in imports {
        let alias_sym = import.alias.unwrap_or_else(|| sym(""));

        block.op(src::r#use(ctx.db, location, import.path, alias_sym, is_pub));
    }
}

fn collect_use_imports<'db>(
    ctx: &CstLoweringCtx<'db>,
    node: Node,
    base: &mut SymbolVec,
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
            let alias = alias_node.map(|n| Symbol::from(node_text(&n, &ctx.source)));

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
                        head = Some(Symbol::from(node_text(&child, &ctx.source)));
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

            let path = match base.split_last() {
                Some((name, parent)) if head == "self" => QualifiedName::new(parent, *name),
                _ => QualifiedName::new(&base[..], head),
            };
            out.push(UseImport { path, alias });
        }
        _ => {}
    }
}

// =============================================================================
// Function Lowering
// =============================================================================

/// Lower a function definition to a func.func operation.
pub fn lower_function<'db>(ctx: &mut CstLoweringCtx<'db>, node: Node) -> Option<func::Func<'db>> {
    // function_definition contains either regular_function or extern_function
    let func_node = node
        .named_child(0)
        .filter(|c| c.kind() == "regular_function" || c.kind() == "extern_function")?;

    let location = ctx.location(&func_node);

    // extern_function has no body - skip for now (TODO: handle extern functions)
    if func_node.kind() == "extern_function" {
        return None;
    }

    // Use field-based access for cleaner extraction
    let name_node = func_node.child_by_field_name("name")?;
    let body_node = func_node.child_by_field_name("body")?;

    let name_str = node_text(&name_node, &ctx.source);
    let name_sym = Symbol::from_dynamic(&name_str);
    let qualified_name = ctx.qualified_name(name_sym);
    let name_span = Some(Span {
        start: name_node.start_byte(),
        end: name_node.end_byte(),
    });

    // Optional fields
    let (param_names, param_types) = func_node
        .child_by_field_name("params")
        .map(|params| parse_parameter_list(ctx, params))
        .unwrap_or_default();

    let return_type = func_node
        .child_by_field_name("return_type")
        .and_then(|rt| parse_return_type(ctx, rt));

    // Resolve parameter types
    let params: IdVec<Type> = param_types.into_iter().collect();

    // Resolve return type or create fresh type var
    let result = return_type.unwrap_or_else(|| ctx.fresh_type_var());

    let effect_type = ctx.fresh_effect_row_type();
    Some(func::Func::build_with_name_span_and_effect(
        ctx.db,
        location,
        qualified_name,
        name_span,
        params.clone(),
        result,
        Some(effect_type),
        |entry| {
            // Bind parameters
            for (i, param_name) in param_names.into_iter().enumerate() {
                let infer_ty = ctx.fresh_type_var();
                let param_value = entry.op(src::var(ctx.db, location, infer_ty, param_name));
                ctx.bind(param_name, param_value.result(ctx.db));
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

/// Lower a struct declaration to type.struct and a module containing field accessors.
///
/// For a struct like `struct User { name: String, age: Int }`, this generates:
/// 1. The type.struct definition
/// 2. A module named `User` containing:
///    - `fn name(self: User) -> String` (getter)
///    - `mod name { fn set(...), fn modify(...) }`
///    - `fn age(self: User) -> Int` (getter)
///    - `mod age { fn set(...), fn modify(...) }`
pub fn lower_struct_decl<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    node: Node,
) -> Option<(ty::Struct<'db>, core::Module<'db>)> {
    let location = ctx.location(&node);

    // Use field-based access
    let name_node = node.child_by_field_name("name")?;
    let body_node = node.child_by_field_name("body")?;

    let name = node_text(&name_node, &ctx.source).to_string();
    let type_name = sym(&name);

    // Create adt.typeref as the result type - this provides a proper ADT type
    // instead of type.var, allowing correct type flow through the pipeline
    let struct_ty = adt::typeref(ctx.db, QualifiedName::simple(type_name));
    let fields = parse_struct_fields(ctx, body_node);

    // Build fields attribute for the struct definition
    let fields_attr = Attribute::List(
        fields
            .iter()
            .map(|(field_name, field_type)| {
                Attribute::List(vec![
                    Attribute::Symbol(sym(field_name)),
                    Attribute::Type(*field_type),
                ])
            })
            .collect(),
    );

    // Create the struct definition operation
    let struct_op = ty::r#struct(
        ctx.db,
        location,
        struct_ty,
        Attribute::Symbol(type_name),
        fields_attr,
    );

    // Create a module with the same name as the struct, containing field accessors
    let fields_clone = fields.clone();
    let accessors_module = core::Module::build(ctx.db, location, type_name, |module_builder| {
        for (idx, (field_name, field_type)) in fields_clone.iter().enumerate() {
            let field_sym = sym(field_name);

            // Generate getter: fn field_name(self: StructType) -> FieldType
            let getter =
                generate_field_getter(ctx, location, struct_ty, field_sym, *field_type, idx);
            module_builder.op(getter);

            // Generate field module containing set and modify
            let field_module = generate_field_module(
                ctx,
                location,
                struct_ty,
                field_sym,
                *field_type,
                idx,
                &fields_clone,
            );
            module_builder.op(field_module);
        }
    });

    Some((struct_op, accessors_module))
}

/// Generate a field getter function.
///
/// Creates: `fn field_name(self: StructType) -> FieldType { adt.struct_get(self, field_index) }`
fn generate_field_getter<'db>(
    ctx: &CstLoweringCtx<'db>,
    location: trunk_ir::Location<'db>,
    struct_ty: Type<'db>,
    field_name: Symbol,
    field_type: Type<'db>,
    field_index: usize,
) -> func::Func<'db> {
    func::Func::build(
        ctx.db,
        location,
        QualifiedName::simple(field_name),
        idvec![struct_ty],
        field_type,
        |entry| {
            let self_value = entry.block_arg(ctx.db, 0);
            let field_value = entry.op(adt::struct_get(
                ctx.db,
                location,
                self_value,
                field_type,
                struct_ty,
                Attribute::IntBits(field_index as u64),
            ));
            entry.op(func::Return::value(
                ctx.db,
                location,
                field_value.result(ctx.db),
            ));
        },
    )
}

/// Generate a field module containing set and modify functions.
///
/// Creates:
/// ```text
/// mod field_name {
///     fn set(self: StructType, value: FieldType) -> StructType { ... }
///     fn modify(self: StructType, f: fn(FieldType) -> FieldType) -> StructType { ... }
/// }
/// ```
fn generate_field_module<'db>(
    ctx: &CstLoweringCtx<'db>,
    location: trunk_ir::Location<'db>,
    struct_ty: Type<'db>,
    field_name: Symbol,
    field_type: Type<'db>,
    field_index: usize,
    all_fields: &[(String, Type<'db>)],
) -> core::Module<'db> {
    let all_fields = all_fields.to_vec();

    core::Module::build(ctx.db, location, field_name, |module_builder| {
        // Generate set function
        let set_fn = generate_field_set(
            ctx,
            location,
            struct_ty,
            field_type,
            field_index,
            &all_fields,
        );
        module_builder.op(set_fn);

        // Generate modify function
        let modify_fn = generate_field_modify(
            ctx,
            location,
            struct_ty,
            field_type,
            field_index,
            &all_fields,
        );
        module_builder.op(modify_fn);
    })
}

/// Generate a field setter function.
///
/// Creates: `fn set(self: StructType, value: FieldType) -> StructType`
/// Implementation: Create a new struct with all fields copied except the target field.
fn generate_field_set<'db>(
    ctx: &CstLoweringCtx<'db>,
    location: trunk_ir::Location<'db>,
    struct_ty: Type<'db>,
    field_type: Type<'db>,
    field_index: usize,
    all_fields: &[(String, Type<'db>)],
) -> func::Func<'db> {
    let all_fields = all_fields.to_vec();

    func::Func::build(
        ctx.db,
        location,
        QualifiedName::simple(sym("set")),
        idvec![struct_ty, field_type],
        struct_ty,
        |entry| {
            let self_value = entry.block_arg(ctx.db, 0);
            let new_value = entry.block_arg(ctx.db, 1);

            // Build new struct with updated field
            let mut field_values = Vec::new();
            for (i, (_, fty)) in all_fields.iter().enumerate() {
                if i == field_index {
                    field_values.push(new_value);
                } else {
                    let extracted = entry.op(adt::struct_get(
                        ctx.db,
                        location,
                        self_value,
                        *fty,
                        struct_ty,
                        Attribute::IntBits(i as u64),
                    ));
                    field_values.push(extracted.result(ctx.db));
                }
            }

            let new_struct = entry.op(adt::struct_new(
                ctx.db,
                location,
                field_values,
                struct_ty,
                struct_ty,
            ));
            entry.op(func::Return::value(
                ctx.db,
                location,
                new_struct.result(ctx.db),
            ));
        },
    )
}

/// Generate a field modify function.
///
/// Creates: `fn modify(self: StructType, f: fn(FieldType) -> FieldType) -> StructType`
/// Implementation: Get current value, apply f, then set.
fn generate_field_modify<'db>(
    ctx: &CstLoweringCtx<'db>,
    location: trunk_ir::Location<'db>,
    struct_ty: Type<'db>,
    field_type: Type<'db>,
    field_index: usize,
    all_fields: &[(String, Type<'db>)],
) -> func::Func<'db> {
    let all_fields = all_fields.to_vec();
    let fn_type = core::Func::new(ctx.db, idvec![field_type], field_type).as_type();

    func::Func::build(
        ctx.db,
        location,
        QualifiedName::simple(sym("modify")),
        idvec![struct_ty, fn_type],
        struct_ty,
        |entry| {
            let self_value = entry.block_arg(ctx.db, 0);
            let f_value = entry.block_arg(ctx.db, 1);

            // Get current field value
            let current = entry.op(adt::struct_get(
                ctx.db,
                location,
                self_value,
                field_type,
                struct_ty,
                Attribute::IntBits(field_index as u64),
            ));

            // Apply f to get new value
            let new_field_value = entry.op(func::call_indirect(
                ctx.db,
                location,
                f_value,
                vec![current.result(ctx.db)],
                field_type,
            ));

            // Build new struct with updated field
            let mut field_values = Vec::new();
            for (i, (_, fty)) in all_fields.iter().enumerate() {
                if i == field_index {
                    field_values.push(new_field_value.result(ctx.db));
                } else {
                    let extracted = entry.op(adt::struct_get(
                        ctx.db,
                        location,
                        self_value,
                        *fty,
                        struct_ty,
                        Attribute::IntBits(i as u64),
                    ));
                    field_values.push(extracted.result(ctx.db));
                }
            }

            let new_struct = entry.op(adt::struct_new(
                ctx.db,
                location,
                field_values,
                struct_ty,
                struct_ty,
            ));
            entry.op(func::Return::value(
                ctx.db,
                location,
                new_struct.result(ctx.db),
            ));
        },
    )
}

/// Parse struct fields from struct_body or record_fields.
fn parse_struct_fields<'db>(ctx: &mut CstLoweringCtx<'db>, node: Node) -> Vec<(String, Type<'db>)> {
    let mut cursor = node.walk();
    let mut fields = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        // Handle struct_fields wrapper (struct_body -> struct_fields -> struct_field)
        if child.kind() == "struct_fields" {
            let mut inner_cursor = child.walk();
            for field_child in child.named_children(&mut inner_cursor) {
                if field_child.kind() == "struct_field"
                    && let Some(name_node) = field_child.child_by_field_name("name")
                {
                    let field_name = node_text(&name_node, &ctx.source).to_string();
                    let field_type = field_child
                        .child_by_field_name("type")
                        .map(|t| ctx.resolve_type_node(t))
                        .unwrap_or_else(|| ctx.fresh_type_var());
                    fields.push((field_name, field_type));
                }
            }
            continue;
        }
        if child.kind() == "struct_field" || child.kind() == "record_field" {
            // Use field-based access for struct_field
            if let Some(name_node) = child.child_by_field_name("name") {
                let field_name = node_text(&name_node, &ctx.source).to_string();
                let field_type = child
                    .child_by_field_name("type")
                    .map(|t| ctx.resolve_type_node(t))
                    .unwrap_or_else(|| ctx.fresh_type_var());
                fields.push((field_name, field_type));
            }
        }
    }

    fields
}

/// Lower an enum declaration to type.enum.
pub fn lower_enum_decl<'db>(ctx: &mut CstLoweringCtx<'db>, node: Node) -> Option<ty::Enum<'db>> {
    let location = ctx.location(&node);

    // Use field-based access
    let name_node = node.child_by_field_name("name")?;
    let body_node = node.child_by_field_name("body")?;

    let name = node_text(&name_node, &ctx.source).to_string();

    // Parse variants from AST
    let parsed_variants = parse_enum_variants(ctx, body_node);

    // Convert to adt::enum_type format: Vec<(Symbol, Vec<Type>)>
    let enum_variants: Vec<(Symbol, Vec<Type<'db>>)> = parsed_variants
        .iter()
        .map(|(variant_name, fields)| {
            let field_types: Vec<Type<'db>> = fields.iter().map(|(_, ty)| *ty).collect();
            (sym(variant_name), field_types)
        })
        .collect();

    // Create adt.enum as the result type - this provides a self-descriptive type
    // with variant information accessible via adt::get_enum_variants
    let result_ty = adt::enum_type(ctx.db, QualifiedName::simple(sym(&name)), enum_variants);

    // Also build the attribute for ty.enum operation (for backwards compatibility)
    let variants_attr = Attribute::List(
        parsed_variants
            .into_iter()
            .map(|(variant_name, variant_fields)| {
                Attribute::List(vec![
                    Attribute::Symbol(sym(&variant_name)),
                    Attribute::List(
                        variant_fields
                            .into_iter()
                            .map(|(_f_name, f_type)| Attribute::Type(f_type))
                            .collect(),
                    ),
                ])
            })
            .collect(),
    );

    Some(ty::r#enum(
        ctx.db,
        location,
        result_ty,
        Attribute::Symbol(sym(&name)),
        variants_attr,
    ))
}

/// Parse enum variants from enum_body.
fn parse_enum_variants<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    node: Node,
) -> Vec<(String, Vec<(String, Type<'db>)>)> {
    let mut cursor = node.walk();
    let mut variants = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if child.kind() == "enum_variants" {
            let mut variant_cursor = child.walk();
            for variant_node in child.named_children(&mut variant_cursor) {
                if is_comment(variant_node.kind()) {
                    continue;
                }
                if variant_node.kind() != "enum_variant" {
                    continue;
                }
                if let Some(name_node) = variant_node.child_by_field_name("name") {
                    let variant_name = node_text(&name_node, &ctx.source).to_string();
                    let variant_fields = variant_node
                        .child_by_field_name("fields")
                        .map(|fields_node| parse_variant_fields(ctx, fields_node))
                        .unwrap_or_default();
                    variants.push((variant_name, variant_fields));
                }
            }
        }
    }

    variants
}

/// Parse variant fields (tuple or record style).
fn parse_variant_fields<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    node: Node,
) -> Vec<(String, Type<'db>)> {
    match node.kind() {
        "variant_fields" => {
            // variant_fields wraps tuple_fields or struct_body
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                let result = parse_variant_fields(ctx, child);
                if !result.is_empty() {
                    return result;
                }
            }
            Vec::new()
        }
        "tuple_fields" => {
            // Positional fields: Variant(Int, String)
            let mut cursor = node.walk();
            node.named_children(&mut cursor)
                .filter(|child| !is_comment(child.kind()))
                .enumerate()
                .map(|(idx, child)| (format!("_{}", idx), ctx.resolve_type_node(child)))
                .collect()
        }
        "struct_body" | "record_fields" => {
            // Named fields: Variant { x: Int, y: String }
            parse_struct_fields(ctx, node)
        }
        _ => Vec::new(),
    }
}

/// Lower a const declaration to src.const.
pub fn lower_const_decl<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    _block: &mut BlockBuilder<'db>,
    node: Node,
) -> Option<src::Const<'db>> {
    let location = ctx.location(&node);

    // Use field access to get the named children
    let name_node = node.child_by_field_name("name")?;
    let value_node = node.child_by_field_name("value")?;
    let type_node = node.child_by_field_name("type");

    let name = node_text(&name_node, &ctx.source).to_string();
    let result_type = type_node
        .map(|n| ctx.resolve_type_node(n))
        .unwrap_or_else(|| ctx.fresh_type_var());

    // Extract literal value directly as an Attribute (no arith.const generated)
    let value_attr = literal_to_attribute(ctx, value_node)?;

    Some(src::r#const(
        ctx.db,
        location,
        result_type,
        sym(&name),
        value_attr,
    ))
}

/// Convert a literal CST node to an Attribute.
fn literal_to_attribute<'db>(ctx: &CstLoweringCtx<'db>, node: Node) -> Option<Attribute<'db>> {
    // Unwrap expression wrapper nodes to get the actual literal
    let actual_node = unwrap_expression_node(node)?;
    let text = node_text(&actual_node, &ctx.source);

    match actual_node.kind() {
        "nat_literal" => {
            let n = parse_nat_literal(&text)?;
            Some(Attribute::IntBits(n))
        }
        "int_literal" => {
            let n = parse_int_literal(&text)?;
            Some(Attribute::IntBits(n as u64))
        }
        "float_literal" => {
            let n = parse_float_literal(&text)?;
            Some(Attribute::FloatBits(n.to_bits()))
        }
        "rune" => {
            let ch = parse_rune_literal(&text)?;
            Some(Attribute::IntBits(ch as u64))
        }
        "string" | "raw_string" | "multiline_string" => {
            let s = parse_string_literal(actual_node, &ctx.source);
            Some(Attribute::String(s))
        }
        "raw_interpolated_string" => {
            if has_interpolation(actual_node) {
                None
            } else {
                let s = parse_string_literal(actual_node, &ctx.source);
                Some(Attribute::String(s))
            }
        }
        "true" => Some(Attribute::Bool(true)),
        "false" => Some(Attribute::Bool(false)),
        _ => None, // Non-literal expressions not supported in const
    }
}

fn has_interpolation(node: Node) -> bool {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "interpolation"
            | "multiline_interpolation"
            | "bytes_interpolation"
            | "multiline_bytes_interpolation" => {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Unwrap expression wrapper nodes to get the actual literal node.
fn unwrap_expression_node<'tree>(node: Node<'tree>) -> Option<Node<'tree>> {
    match node.kind() {
        "primary_expression" | "expression" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if !is_comment(child.kind()) {
                    return unwrap_expression_node(child);
                }
            }
            None
        }
        _ => Some(node),
    }
}

/// Lower an ability declaration to type.ability.
pub fn lower_ability_decl<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    node: Node,
) -> Option<ty::Ability<'db>> {
    let location = ctx.location(&node);
    let infer_ty = ctx.fresh_type_var();

    // Use field-based access
    let name_node = node.child_by_field_name("name")?;
    let body_node = node.child_by_field_name("body")?;

    let name = node_text(&name_node, &ctx.source).to_string();
    let operations = parse_ability_operations(ctx, body_node);

    // Build operations region containing ability.op operations
    let mut ops_block = BlockBuilder::new(ctx.db, location);
    for (op_name, param_types, return_type) in operations {
        let op_type = core::Func::new(ctx.db, param_types.into(), return_type).as_type();
        ops_block.op(ability::op(ctx.db, location, sym(&op_name), op_type));
    }
    let operations_region = Region::new(ctx.db, location, idvec![ops_block.build()]);

    Some(ty::ability(
        ctx.db,
        location,
        infer_ty,
        Attribute::Symbol(sym(&name)),
        operations_region,
    ))
}

/// Parse ability operations from ability_body.
fn parse_ability_operations<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    node: Node,
) -> Vec<(String, Vec<Type<'db>>, Type<'db>)> {
    let mut cursor = node.walk();
    let mut operations = Vec::new();

    for child in node.named_children(&mut cursor) {
        if is_comment(child.kind()) {
            continue;
        }
        if child.kind() == "ability_operation" {
            // Use field-based access for ability_operation
            if let Some(name_node) = child.child_by_field_name("name") {
                let op_name = node_text(&name_node, &ctx.source).to_string();

                // Parameters need manual iteration (not a field in grammar)
                let param_types = find_child_by_kind(child, "parameter_list")
                    .map(|params| parse_parameter_list(ctx, params).1)
                    .unwrap_or_default();

                let return_type = child
                    .child_by_field_name("return_type")
                    .map(|rt| ctx.resolve_type_node(rt))
                    .unwrap_or_else(|| ctx.fresh_type_var());

                operations.push((op_name, param_types, return_type));
            }
        }
    }

    operations
}

/// Find a child node by kind (helper for nodes without field names).
fn find_child_by_kind<'tree>(node: Node<'tree>, kind: &str) -> Option<Node<'tree>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .find(|child| child.kind() == kind)
}

// =============================================================================
// Module Lowering
// =============================================================================

/// Lower a mod_declaration to a core.module operation.
///
/// Handles both inline modules (`mod foo { ... }`) and file-based module
/// declarations (`mod foo`). Currently, only inline modules are fully lowered.
pub fn lower_mod_decl<'db>(ctx: &mut CstLoweringCtx<'db>, node: Node) -> Option<core::Module<'db>> {
    let location = ctx.location(&node);

    // Use field-based access
    let name_node = node.child_by_field_name("name")?;
    let body_node = node.child_by_field_name("body");

    let name = Symbol::from(node_text(&name_node, &ctx.source));

    // Check for visibility marker (not a field, so use helper)
    let _is_pub = find_child_by_kind(node, "visibility_marker").is_some();
    // TODO: Parse visibility modifier (pub, pub(pkg), pub(super))

    // Enter module scope for qualified name generation
    ctx.enter_module(name);

    // Build the module with its body
    let module = core::Module::build(ctx.db, location, name, |mod_builder| {
        if let Some(body) = body_node {
            lower_mod_body(ctx, body, mod_builder);
        }
        // File-based modules (no body) will be handled later in the pipeline
        // when we have package/file loading infrastructure
    });

    // Exit module scope
    ctx.exit_module();

    // TODO: Track visibility (_is_pub) for name resolution
    Some(module)
}

/// Lower items within a mod_body into the module's block.
pub fn lower_mod_body<'db>(
    ctx: &mut CstLoweringCtx<'db>,
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
                if let Some((struct_op, accessors_module)) = lower_struct_decl(ctx, child) {
                    builder.op(struct_op);
                    builder.op(accessors_module);
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
pub fn parse_parameter_list<'db>(
    ctx: &mut CstLoweringCtx<'db>,
    node: Node,
) -> (SymbolVec, Vec<Type<'db>>) {
    let mut cursor = node.walk();
    let mut names = SymbolVec::new();
    let mut types = Vec::new();

    for child in node.named_children(&mut cursor) {
        if child.kind() == "parameter" {
            // Use field-based access for parameter
            if let Some(name_node) = child.child_by_field_name("name") {
                let param_name = node_text(&name_node, &ctx.source).into();
                let param_type = child
                    .child_by_field_name("type")
                    .map(|t| ctx.resolve_type_node(t))
                    .unwrap_or_else(|| ctx.fresh_type_var());
                names.push(param_name);
                types.push(param_type);
            }
        }
    }

    (names, types)
}

/// Parse a return type annotation.
pub fn parse_return_type<'db>(ctx: &mut CstLoweringCtx<'db>, node: Node) -> Option<Type<'db>> {
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

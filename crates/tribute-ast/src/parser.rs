use crate::ast::*;
use tree_sitter::{Node, Parser};
use tribute_core::SourceFile;

pub struct TributeParser {
    parser: Parser,
}

impl TributeParser {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut parser = Parser::new();
        let language = tree_sitter_tribute::language();
        parser.set_language(&language)?;
        Ok(TributeParser { parser })
    }

    pub fn parse_internal<'db>(
        &mut self,
        db: &'db dyn salsa::Database,
        source: &'db str,
    ) -> Result<Program<'db>, Box<dyn std::error::Error>> {
        let tree = self.parser.parse(source, None).ok_or("Failed to parse")?;

        let root_node = tree.root_node();
        let mut items = Vec::new();
        let mut cursor = root_node.walk();

        for child in root_node.named_children(&mut cursor) {
            // Skip comments
            if child.kind() == "line_comment" || child.kind() == "block_comment" {
                continue;
            }
            items.push(self.node_to_item(db, child, source)?);
        }

        Ok(Program::new(db, items))
    }
}

#[salsa::tracked]
pub fn parse_source_file<'db>(db: &'db dyn salsa::Database, source: SourceFile) -> Program<'db> {
    let mut parser = match TributeParser::new() {
        Ok(p) => p,
        Err(_) => return Program::new(db, Vec::new()),
    };

    parser
        .parse_internal(db, source.text(db))
        .unwrap_or_else(|_| Program::new(db, Vec::new()))
}

impl TributeParser {
    fn node_to_item<'db>(
        &self,
        db: &'db dyn salsa::Database,
        node: Node,
        source: &'db str,
    ) -> Result<Item<'db>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        match node.kind() {
            "function_definition" => {
                let mut name = None;
                let mut parameters = Vec::new();
                let mut return_type = None;
                let mut body = None;

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "identifier" => {
                            if name.is_none() {
                                name = Some(child.utf8_text(source.as_bytes())?.to_string());
                            }
                        }
                        "parameter_list" => {
                            parameters = self.parse_parameter_list(child, source)?;
                        }
                        "return_type_annotation" => {
                            return_type = Some(self.parse_return_type_annotation(child, source)?);
                        }
                        "block" => {
                            body = Some(self.parse_block(child, source)?);
                        }
                        _ => {}
                    }
                }

                let name = name.ok_or("Missing function name")?;
                let body = body.ok_or("Missing function body")?;
                let span = Span::new(node.start_byte(), node.end_byte());

                Ok(Item::new(
                    db,
                    ItemKind::Function(FunctionDefinition::new(
                        db,
                        name,
                        parameters,
                        return_type,
                        body,
                        span,
                    )),
                    span,
                ))
            }
            "struct_declaration" => {
                let mut name = None;
                let mut type_params = Vec::new();
                let mut fields = Vec::new();
                let mut is_pub = false;

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "keyword_pub" => is_pub = true,
                        "type_identifier" if name.is_none() => {
                            name = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                        "type_parameters" => {
                            type_params = self.parse_type_parameters(child, source)?;
                        }
                        "struct_body" => {
                            fields = self.parse_struct_body(child, source)?;
                        }
                        _ => {}
                    }
                }

                let name = name.ok_or("Missing struct name")?;
                let span = Span::new(node.start_byte(), node.end_byte());

                Ok(Item::new(
                    db,
                    ItemKind::Struct(StructDefinition::new(
                        db,
                        name,
                        type_params,
                        fields,
                        is_pub,
                        span,
                    )),
                    span,
                ))
            }
            "enum_declaration" => {
                let mut name = None;
                let mut type_params = Vec::new();
                let mut variants = Vec::new();
                let mut is_pub = false;

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "keyword_pub" => is_pub = true,
                        "type_identifier" if name.is_none() => {
                            name = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                        "type_parameters" => {
                            type_params = self.parse_type_parameters(child, source)?;
                        }
                        "enum_body" => {
                            variants = self.parse_enum_body(child, source)?;
                        }
                        _ => {}
                    }
                }

                let name = name.ok_or("Missing enum name")?;
                let span = Span::new(node.start_byte(), node.end_byte());

                Ok(Item::new(
                    db,
                    ItemKind::Enum(EnumDefinition::new(
                        db,
                        name,
                        type_params,
                        variants,
                        is_pub,
                        span,
                    )),
                    span,
                ))
            }
            "const_declaration" => {
                let mut name = None;
                let mut ty = None;
                let mut value = None;
                let mut is_pub = false;

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "keyword_pub" => is_pub = true,
                        "identifier" if name.is_none() => {
                            name = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                        "type_identifier" | "type_variable" | "generic_type" => {
                            ty = Some(self.parse_type_ref(child, source)?);
                        }
                        _ => {
                            // Try to parse as value expression
                            if value.is_none()
                                && let Ok(expr) = self.node_to_expr_with_span(child, source)
                            {
                                value = Some(expr);
                            }
                        }
                    }
                }

                let name = name.ok_or("Missing const name")?;
                let value = value.ok_or("Missing const value")?;
                let span = Span::new(node.start_byte(), node.end_byte());

                Ok(Item::new(
                    db,
                    ItemKind::Const(ConstDefinition::new(db, name, ty, value, is_pub, span)),
                    span,
                ))
            }
            "use_declaration" => {
                let mut path = None;
                let mut is_pub = false;

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "keyword_pub" => is_pub = true,
                        "use_path" => {
                            path = Some(self.parse_use_path(child, source)?);
                        }
                        _ => {}
                    }
                }

                let path = path.ok_or("Missing use path")?;
                let span = Span::new(node.start_byte(), node.end_byte());

                Ok(Item::new(
                    db,
                    ItemKind::Use(UseDeclaration::new(db, path, is_pub, span)),
                    span,
                ))
            }
            "mod_declaration" => {
                let mut name = None;
                let mut items = None;
                let mut is_pub = false;

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "keyword_pub" => is_pub = true,
                        "identifier" if name.is_none() => {
                            name = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                        "mod_body" => {
                            items = Some(self.parse_mod_body(db, child, source)?);
                        }
                        _ => {}
                    }
                }

                let name = name.ok_or("Missing mod name")?;
                let span = Span::new(node.start_byte(), node.end_byte());

                Ok(Item::new(
                    db,
                    ItemKind::Mod(ModDeclaration::new(db, name, items, is_pub, span)),
                    span,
                ))
            }
            _ => Err(format!("Unknown item kind: {}", node.kind()).into()),
        }
    }

    fn parse_parameter_list(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Parameter>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut parameters = Vec::new();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "parameter" {
                parameters.push(self.parse_parameter(child, source)?);
            }
        }

        Ok(parameters)
    }

    fn parse_parameter(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Parameter, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut name = None;
        let mut ty = None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "identifier" if name.is_none() => {
                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                }
                "type_identifier" | "type_variable" | "generic_type" => {
                    ty = Some(self.parse_type_ref(child, source)?);
                }
                _ => {}
            }
        }

        let name = name.ok_or("Missing parameter name")?;

        Ok(Parameter { name, ty })
    }

    fn parse_use_path(
        &self,
        node: Node,
        source: &str,
    ) -> Result<UsePath, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut segments = Vec::new();
        let mut group = None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "identifier" => {
                    segments.push(child.utf8_text(source.as_bytes())?.to_string());
                }
                "use_path_segment" => {
                    // use_path_segment contains an identifier
                    if let Some(id_child) = child.child(0) {
                        segments.push(id_child.utf8_text(source.as_bytes())?.to_string());
                    }
                }
                "use_group" => {
                    group = Some(self.parse_use_group(child, source)?);
                }
                _ => {}
            }
        }

        Ok(UsePath { segments, group })
    }

    fn parse_use_group(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Identifier>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut items = Vec::new();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "identifier" {
                items.push(child.utf8_text(source.as_bytes())?.to_string());
            }
        }

        Ok(items)
    }

    fn parse_mod_body<'db>(
        &self,
        db: &'db dyn salsa::Database,
        node: Node,
        source: &'db str,
    ) -> Result<Vec<Item<'db>>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut items = Vec::new();

        for child in node.named_children(&mut cursor) {
            // Skip comments
            if child.kind() == "line_comment" || child.kind() == "block_comment" {
                continue;
            }
            items.push(self.node_to_item(db, child, source)?);
        }

        Ok(items)
    }

    fn parse_type_parameters(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Identifier>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut params = Vec::new();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "identifier" {
                params.push(child.utf8_text(source.as_bytes())?.to_string());
            }
        }

        Ok(params)
    }

    fn parse_struct_body(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<StructField>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "struct_fields" {
                return self.parse_struct_fields(child, source);
            }
        }

        Ok(Vec::new())
    }

    fn parse_struct_fields(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<StructField>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut fields = Vec::new();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "struct_field" {
                fields.push(self.parse_struct_field(child, source)?);
            }
        }

        Ok(fields)
    }

    fn parse_struct_field(
        &self,
        node: Node,
        source: &str,
    ) -> Result<StructField, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut name = None;
        let mut ty = None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "identifier" if name.is_none() => {
                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                }
                "type_identifier" | "type_variable" | "generic_type" => {
                    ty = Some(self.parse_type_ref(child, source)?);
                }
                _ => {}
            }
        }

        let name = name.ok_or("Missing field name")?;
        let ty = ty.ok_or("Missing field type")?;

        Ok(StructField { name, ty })
    }

    fn parse_enum_body(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<EnumVariant>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "enum_variants" {
                return self.parse_enum_variants(child, source);
            }
        }

        Ok(Vec::new())
    }

    fn parse_enum_variants(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<EnumVariant>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut variants = Vec::new();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "enum_variant" {
                variants.push(self.parse_enum_variant(child, source)?);
            }
        }

        Ok(variants)
    }

    fn parse_enum_variant(
        &self,
        node: Node,
        source: &str,
    ) -> Result<EnumVariant, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut name = None;
        let mut fields = None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "type_identifier" if name.is_none() => {
                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                }
                "variant_fields" => {
                    fields = Some(self.parse_variant_fields(child, source)?);
                }
                _ => {}
            }
        }

        let name = name.ok_or("Missing variant name")?;

        Ok(EnumVariant { name, fields })
    }

    fn parse_variant_fields(
        &self,
        node: Node,
        source: &str,
    ) -> Result<VariantFields, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "tuple_fields" => return self.parse_tuple_fields(child, source),
                "struct_fields_block" => return self.parse_struct_fields_block(child, source),
                _ => {}
            }
        }

        Err("Empty variant fields".into())
    }

    fn parse_tuple_fields(
        &self,
        node: Node,
        source: &str,
    ) -> Result<VariantFields, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut types = Vec::new();

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "type_identifier" | "type_variable" | "generic_type" => {
                    types.push(self.parse_type_ref(child, source)?);
                }
                _ => {}
            }
        }

        Ok(VariantFields::Tuple(types))
    }

    fn parse_struct_fields_block(
        &self,
        node: Node,
        source: &str,
    ) -> Result<VariantFields, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "struct_fields" {
                let fields = self.parse_struct_fields(child, source)?;
                return Ok(VariantFields::Struct(fields));
            }
        }

        Ok(VariantFields::Struct(Vec::new()))
    }

    /// Parse return type annotation: -> Type
    fn parse_return_type_annotation(
        &self,
        node: Node,
        source: &str,
    ) -> Result<TypeRef, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        // return_type_annotation contains a single _type child
        if let Some(child) = node.named_children(&mut cursor).next() {
            return self.parse_type_ref(child, source);
        }
        Err("Missing type in return type annotation".into())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn parse_type_ref(
        &self,
        node: Node,
        source: &str,
    ) -> Result<TypeRef, Box<dyn std::error::Error>> {
        match node.kind() {
            "type_identifier" => {
                let name = node.utf8_text(source.as_bytes())?.to_string();
                Ok(TypeRef::Named(name))
            }
            "type_variable" => {
                // type_variable contains an identifier child
                if let Some(child) = node.child(0) {
                    let name = child.utf8_text(source.as_bytes())?.to_string();
                    Ok(TypeRef::Variable(name))
                } else {
                    Err("Empty type variable".into())
                }
            }
            "generic_type" => {
                let mut cursor = node.walk();
                let mut name = None;
                let mut args = Vec::new();

                for child in node.named_children(&mut cursor) {
                    match child.kind() {
                        "type_identifier" => {
                            if name.is_none() {
                                name = Some(child.utf8_text(source.as_bytes())?.to_string());
                            } else {
                                // Nested type_identifier as arg
                                args.push(TypeRef::Named(
                                    child.utf8_text(source.as_bytes())?.to_string(),
                                ));
                            }
                        }
                        "type_variable" | "generic_type" => {
                            args.push(self.parse_type_ref(child, source)?);
                        }
                        _ => {}
                    }
                }

                let name = name.ok_or("Missing generic type name")?;
                Ok(TypeRef::Generic { name, args })
            }
            _ => Err(format!("Unknown type reference kind: {}", node.kind()).into()),
        }
    }

    fn parse_block(&self, node: Node, source: &str) -> Result<Block, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut statements = Vec::new();

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "let_statement" => {
                    statements.push(Statement::Let(self.parse_let_statement(child, source)?));
                }
                "expression_statement" => {
                    statements.push(Statement::Expression(
                        self.parse_expression_statement(child, source)?,
                    ));
                }
                "line_comment" | "block_comment" => {
                    // Skip comments
                }
                _ => {
                    // Try to parse as expression statement
                    if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                        statements.push(Statement::Expression(expr));
                    }
                }
            }
        }

        Ok(Block { statements })
    }

    fn parse_let_statement(
        &self,
        node: Node,
        source: &str,
    ) -> Result<LetStatement, Box<dyn std::error::Error>> {
        let pattern_node = node
            .child_by_field_name("pattern")
            .ok_or("Missing let pattern")?;
        let value_node = node
            .child_by_field_name("value")
            .ok_or("Missing let value")?;

        let pattern = self.parse_pattern(pattern_node, source)?;
        let value = self.node_to_expr_with_span(value_node, source)?;

        Ok(LetStatement { pattern, value })
    }

    fn parse_expression_statement(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Spanned<Expr>, Box<dyn std::error::Error>> {
        // expression_statement should have one child which is the expression
        if let Some(child) = node.child(0) {
            self.node_to_expr_with_span(child, source)
        } else {
            Err("Empty expression statement".into())
        }
    }

    fn node_to_expr_with_span(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Spanned<Expr>, Box<dyn std::error::Error>> {
        let span = Span::new(node.start_byte(), node.end_byte());
        let expr = self.node_to_expr(node, source)?;
        Ok((expr, span))
    }

    fn node_to_expr(&self, node: Node, source: &str) -> Result<Expr, Box<dyn std::error::Error>> {
        match node.kind() {
            "nat_literal" => {
                let text = node.utf8_text(source.as_bytes())?;
                let num = parse_nat_literal(text)?;
                Ok(Expr::Nat(num))
            }
            "int_literal" => {
                let text = node.utf8_text(source.as_bytes())?;
                let num = parse_int_literal(text)?;
                Ok(Expr::Int(num))
            }
            "float_literal" => {
                let text = node.utf8_text(source.as_bytes())?;
                let num = text.parse::<f64>()?;
                Ok(Expr::Float(num))
            }
            "rune" => {
                let text = node.utf8_text(source.as_bytes())?;
                let ch = parse_rune_literal(text)?;
                Ok(Expr::Rune(ch))
            }
            "string" => {
                // All strings are now StringInterpolation, even simple ones
                self.parse_interpolated_string(node, source)
            }
            "raw_string" => {
                // Raw strings: no escape processing, just literal content
                self.parse_raw_string(node, source)
            }
            "bytes_string" => {
                // Bytes literals with escape sequences and interpolation
                self.parse_bytes_string(node, source)
            }
            "raw_bytes" => {
                // Raw bytes: no escape processing
                self.parse_raw_bytes(node, source)
            }
            "multiline_string" => {
                // Multiline strings: #"..."#, ##"..."##
                self.parse_multiline_string(node, source)
            }
            "multiline_bytes" => {
                // Multiline bytes: b#"..."#, b##"..."##
                self.parse_multiline_bytes(node, source)
            }
            "identifier" => {
                let text = node.utf8_text(source.as_bytes())?;
                Ok(Expr::Identifier(text.to_string()))
            }
            "path_expression" => {
                // Qualified path: foo::bar::baz
                let mut cursor = node.walk();
                let mut segments = Vec::new();
                for child in node.named_children(&mut cursor) {
                    if child.kind() != "path_segment" {
                        continue;
                    }
                    if let Some(inner) = child.child(0) {
                        segments.push(inner.utf8_text(source.as_bytes())?.to_string());
                    }
                }
                Ok(Expr::Path(segments))
            }
            "keyword_true" => Ok(Expr::Bool(true)),
            "keyword_false" => Ok(Expr::Bool(false)),
            "keyword_nil" => Ok(Expr::Nil),
            "binary_expression" => self.parse_binary_expression(node, source),
            "call_expression" => self.parse_call_expression(node, source),
            "method_call_expression" => self.parse_method_call_expression(node, source),
            "case_expression" => self.parse_case_expression(node, source),
            "lambda_expression" => self.parse_lambda_expression(node, source),
            "list_expression" => self.parse_list_expression(node, source),
            "tuple_expression" => self.parse_tuple_expression(node, source),
            "record_expression" => self.parse_record_expression(node, source),
            "block" => self.parse_block_as_expr(node, source),
            "operator_fn" => {
                // operator_fn has a field "operator" containing either:
                // - a simple operator token: (+), (<>)
                // - a qualified_operator: (Int::+), (String::<>)
                if let Some(op_node) = node.child_by_field_name("operator") {
                    if op_node.kind() == "qualified_operator" {
                        // Qualified operator: (Int::+)
                        let (qualifier, _bin_op) =
                            self.parse_qualified_operator(op_node, source)?;
                        let inner_op_node = op_node
                            .child_by_field_name("operator")
                            .ok_or("Missing operator in qualified operator")?;
                        let op = inner_op_node.utf8_text(source.as_bytes())?.to_string();
                        Ok(Expr::OperatorFn(OperatorFnExpression {
                            op,
                            qualifier: Some(qualifier),
                        }))
                    } else {
                        // Simple operator: (+)
                        let op = op_node.utf8_text(source.as_bytes())?.to_string();
                        Ok(Expr::OperatorFn(OperatorFnExpression {
                            op,
                            qualifier: None,
                        }))
                    }
                } else {
                    Err("Missing operator in operator_fn".into())
                }
            }
            "primary_expression" => {
                // primary_expression should have one child
                if let Some(child) = node.child(0) {
                    self.node_to_expr(child, source)
                } else {
                    Err("Empty primary expression".into())
                }
            }
            _ => Err(format!("Unknown expression kind: {}", node.kind()).into()),
        }
    }

    fn parse_binary_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut left = None;
        let mut operator = None;
        let mut qualifier = None;
        let mut right = None;

        for child in node.children(&mut cursor) {
            // Check for qualified_operator first
            if child.kind() == "qualified_operator" {
                let (q, op) = self.parse_qualified_operator(child, source)?;
                qualifier = Some(q);
                operator = Some(op);
            } else if let Some(op) = parse_operator_token(child.kind()) {
                operator = Some(op);
            } else {
                // Try to parse as expression
                if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                    if left.is_none() {
                        left = Some(Box::new(expr));
                    } else if right.is_none() {
                        right = Some(Box::new(expr));
                    }
                }
            }
        }

        let left = left.ok_or("Missing left operand")?;
        let operator = operator.ok_or("Missing operator")?;
        let right = right.ok_or("Missing right operand")?;

        Ok(Expr::Binary(BinaryExpression {
            left,
            operator,
            qualifier,
            right,
        }))
    }

    /// Parse a qualified operator node like `Int::+` or `List::<>`
    fn parse_qualified_operator(
        &self,
        node: Node,
        source: &str,
    ) -> Result<(Identifier, BinaryOperator), Box<dyn std::error::Error>> {
        let type_node = node
            .child_by_field_name("type")
            .ok_or("Missing type in qualified operator")?;
        let qualifier = type_node.utf8_text(source.as_bytes())?.to_string();

        let op_node = node
            .child_by_field_name("operator")
            .ok_or("Missing operator in qualified operator")?;
        let op_str = op_node.utf8_text(source.as_bytes())?;
        let operator =
            parse_operator_token(op_str).ok_or_else(|| format!("Unknown operator: {}", op_str))?;

        Ok((qualifier, operator))
    }

    fn parse_call_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut function = None;
        let mut arguments = Vec::new();

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "identifier" if function.is_none() => {
                    function = Some(child.utf8_text(source.as_bytes())?.to_string());
                }
                "argument_list" => {
                    arguments = self.parse_argument_list(child, source)?;
                }
                _ => {}
            }
        }

        let function = function.ok_or("Missing function name")?;

        Ok(Expr::Call(CallExpression {
            function,
            arguments,
        }))
    }

    /// Parse UFCS method call: x.f(y) or x.f
    fn parse_method_call_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut receiver = None;
        let mut method = None;
        let mut arguments = Vec::new();

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "identifier" => {
                    // Method name comes after receiver
                    if receiver.is_some() && method.is_none() {
                        method = Some(child.utf8_text(source.as_bytes())?.to_string());
                    }
                }
                "argument_list" => {
                    arguments = self.parse_argument_list(child, source)?;
                }
                _ => {
                    // Try to parse as receiver expression
                    if receiver.is_none()
                        && let Ok(expr) = self.node_to_expr_with_span(child, source)
                    {
                        receiver = Some(Box::new(expr));
                    }
                }
            }
        }

        let receiver = receiver.ok_or("Missing method call receiver")?;
        let method = method.ok_or("Missing method name")?;

        Ok(Expr::MethodCall(MethodCallExpression {
            receiver,
            method,
            arguments,
        }))
    }

    fn parse_argument_list(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Spanned<Expr>>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut arguments = Vec::new();

        for child in node.named_children(&mut cursor) {
            if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                arguments.push(expr);
            }
        }

        Ok(arguments)
    }

    fn parse_case_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut value = None;
        let mut arms = Vec::new();

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "case_arm" => {
                    arms.push(self.parse_case_arm(child, source)?);
                }
                "keyword_case" => {
                    // Skip keyword
                }
                _ => {
                    // Try to parse as the value expression
                    if value.is_none()
                        && let Ok(expr) = self.node_to_expr_with_span(child, source)
                    {
                        value = Some(Box::new(expr));
                    }
                }
            }
        }

        let value = value.ok_or("Missing case value")?;

        Ok(Expr::Match(MatchExpression { value, arms }))
    }

    fn parse_case_arm(
        &self,
        node: Node,
        source: &str,
    ) -> Result<MatchArm, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut pattern = None;
        let mut branches = Vec::new();
        let mut simple_value = None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "pattern" => {
                    pattern = Some(self.parse_pattern(child, source)?);
                }
                "guarded_branch" => {
                    branches.push(self.parse_guarded_branch(child, source)?);
                }
                _ => {
                    // Try to parse as value expression (simple case without guard)
                    if simple_value.is_none()
                        && let Ok(expr) = self.node_to_expr_with_span(child, source)
                    {
                        simple_value = Some(expr);
                    }
                }
            }
        }

        let pattern = pattern.ok_or("Missing pattern")?;

        // If we have guarded branches, use them; otherwise use the simple value
        let branches = if !branches.is_empty() {
            branches
        } else {
            let value = simple_value.ok_or("Missing case arm value")?;
            vec![GuardedBranch { guard: None, value }]
        };

        Ok(MatchArm { pattern, branches })
    }

    fn parse_guarded_branch(
        &self,
        node: Node,
        source: &str,
    ) -> Result<GuardedBranch, Box<dyn std::error::Error>> {
        let guard = node
            .child_by_field_name("guard")
            .ok_or("Missing guard expression")?;
        let value = node
            .child_by_field_name("value")
            .ok_or("Missing guarded branch value")?;

        Ok(GuardedBranch {
            guard: Some(self.node_to_expr_with_span(guard, source)?),
            value: self.node_to_expr_with_span(value, source)?,
        })
    }

    fn parse_list_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut elements = Vec::new();

        for child in node.named_children(&mut cursor) {
            if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                elements.push(expr);
            }
        }

        Ok(Expr::List(elements))
    }

    fn parse_tuple_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut elements = Vec::new();

        for child in node.named_children(&mut cursor) {
            if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                elements.push(expr);
            }
        }

        if elements.is_empty() {
            return Err("Tuple must have at least one element".into());
        }

        let mut iter = elements.into_iter();
        let first = iter.next().unwrap();
        let rest: Vec<_> = iter.collect();

        Ok(Expr::Tuple(Box::new(first), rest))
    }

    /// Parse record expression: User { name: "Alice", age: 30 }
    fn parse_record_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let type_node = node
            .child_by_field_name("type")
            .ok_or("Missing type in record expression")?;
        let type_name = type_node.utf8_text(source.as_bytes())?.to_string();

        let mut fields = Vec::new();
        if let Some(fields_node) = node.child_by_field_name("fields") {
            let mut cursor = fields_node.walk();
            for child in fields_node.named_children(&mut cursor) {
                if child.kind() == "record_field" {
                    fields.push(self.parse_record_field(child, source)?);
                }
            }
        }

        Ok(Expr::Record(RecordExpression { type_name, fields }))
    }

    /// Parse a single record field: name: value, name, or ..expr
    fn parse_record_field(
        &self,
        node: Node,
        source: &str,
    ) -> Result<RecordField, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();

        // Check if it's a spread field
        for child in node.named_children(&mut cursor) {
            if child.kind() == "spread" {
                // Spread: ..expr
                if let Some(value_node) = node.child_by_field_name("value") {
                    let value = self.node_to_expr_with_span(value_node, source)?;
                    return Ok(RecordField::Spread(value));
                }
                return Err("Missing value in spread field".into());
            }
        }

        // Check for name field
        if let Some(name_node) = node.child_by_field_name("name") {
            let name = name_node.utf8_text(source.as_bytes())?.to_string();

            // Check if it has a value (full form) or is shorthand
            if let Some(value_node) = node.child_by_field_name("value") {
                let value = self.node_to_expr_with_span(value_node, source)?;
                Ok(RecordField::Field { name, value })
            } else {
                Ok(RecordField::Shorthand(name))
            }
        } else {
            Err("Invalid record field".into())
        }
    }

    /// Parse lambda expression: fn(x) x + 1, fn(x) -> Int x + 1
    fn parse_lambda_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut parameters = Vec::new();
        let mut return_type = None;
        let mut body = None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "parameter_list" => {
                    parameters = self.parse_parameter_list(child, source)?;
                }
                "return_type_annotation" => {
                    return_type = Some(self.parse_return_type_annotation(child, source)?);
                }
                _ => {
                    // Try to parse as body expression
                    if body.is_none()
                        && let Ok(expr) = self.node_to_expr_with_span(child, source)
                    {
                        body = Some(Box::new(expr));
                    }
                }
            }
        }

        let body = body.ok_or("Missing lambda body")?;

        Ok(Expr::Lambda(LambdaExpression {
            parameters,
            return_type,
            body,
        }))
    }

    /// Parse a block as an expression (e.g., { let x = 1; x + 1 })
    fn parse_block_as_expr(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let block = self.parse_block(node, source)?;
        Ok(Expr::Block(block.statements))
    }

    fn parse_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Pattern, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "simple_pattern" => {
                    return self.parse_simple_pattern_node(child, source);
                }
                "as_pattern" => {
                    return self.parse_as_pattern(child, source);
                }
                "handler_pattern" => {
                    return Ok(Pattern::Handler(self.parse_handler_pattern(child, source)?));
                }
                _ => {}
            }
        }
        Err("Invalid pattern".into())
    }

    fn parse_simple_pattern_node(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Pattern, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "literal_pattern" => {
                    return Ok(Pattern::Literal(self.parse_literal_pattern(child, source)?));
                }
                "wildcard_pattern" => {
                    return Ok(Pattern::Wildcard);
                }
                "constructor_pattern" => {
                    return Ok(Pattern::Constructor(
                        self.parse_constructor_pattern(child, source)?,
                    ));
                }
                "tuple_pattern" => {
                    let (first, rest) = self.parse_tuple_pattern(child, source)?;
                    return Ok(Pattern::Tuple(Box::new(first), rest));
                }
                "list_pattern" => {
                    return Ok(Pattern::List(self.parse_list_pattern(child, source)?));
                }
                "identifier_pattern" => {
                    if let Some(id_child) = child.child(0) {
                        let text = id_child.utf8_text(source.as_bytes())?;
                        return Ok(Pattern::Identifier(text.to_string()));
                    }
                }
                _ => {}
            }
        }
        Err("Invalid simple pattern".into())
    }

    fn parse_tuple_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<(Pattern, Vec<Pattern>), Box<dyn std::error::Error>> {
        let mut cursor = node.walk();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "pattern_list" {
                let patterns = self.parse_pattern_list(child, source)?;
                if patterns.is_empty() {
                    return Err("Tuple pattern must have at least one element".into());
                }
                let mut iter = patterns.into_iter();
                let first = iter.next().unwrap();
                let rest: Vec<_> = iter.collect();
                return Ok((first, rest));
            }
        }

        Err("Missing elements in tuple pattern".into())
    }

    fn parse_constructor_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<ConstructorPattern, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut name = None;
        let mut args = ConstructorArgs::None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "type_identifier" if name.is_none() => {
                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                }
                "pattern_list" => {
                    args = ConstructorArgs::Positional(self.parse_pattern_list(child, source)?);
                }
                "pattern_fields" => {
                    let (fields, rest) = self.parse_pattern_fields(child, source)?;
                    args = ConstructorArgs::Named { fields, rest };
                }
                _ => {}
            }
        }

        let name = name.ok_or("Missing constructor name")?;

        Ok(ConstructorPattern { name, args })
    }

    fn parse_pattern_list(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Pattern>, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut patterns = Vec::new();

        for child in node.named_children(&mut cursor) {
            if child.kind() == "pattern" {
                patterns.push(self.parse_pattern(child, source)?);
            }
        }

        Ok(patterns)
    }

    fn parse_pattern_fields(
        &self,
        node: Node,
        source: &str,
    ) -> Result<(Vec<PatternField>, bool), Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut fields = Vec::new();
        let mut has_rest = false;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "pattern_field" => {
                    fields.push(self.parse_pattern_field(child, source)?);
                }
                "spread" => {
                    has_rest = true;
                }
                _ => {}
            }
        }

        Ok((fields, has_rest))
    }

    fn parse_pattern_field(
        &self,
        node: Node,
        source: &str,
    ) -> Result<PatternField, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut name = None;
        let mut pattern = None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "identifier" if name.is_none() => {
                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                }
                "pattern" => {
                    pattern = Some(self.parse_pattern(child, source)?);
                }
                _ => {}
            }
        }

        let name = name.ok_or("Missing pattern field name")?;
        // For shorthand { name }, use Pattern::Identifier(name)
        let pattern = pattern.unwrap_or_else(|| Pattern::Identifier(name.clone()));

        Ok(PatternField { name, pattern })
    }

    fn parse_list_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<ListPattern, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut elements = Vec::new();
        let mut rest: Option<Option<Identifier>> = None;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "pattern" => {
                    elements.push(self.parse_pattern(child, source)?);
                }
                "rest_pattern" => {
                    // rest_pattern contains optional "name" field
                    let name = child
                        .child_by_field_name("name")
                        .map(|n| n.utf8_text(source.as_bytes()).map(|s| s.to_string()))
                        .transpose()?;
                    rest = Some(name);
                }
                _ => {}
            }
        }

        Ok(ListPattern { elements, rest })
    }

    fn parse_as_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Pattern, Box<dyn std::error::Error>> {
        let pattern_node = node
            .child_by_field_name("pattern")
            .ok_or("Missing pattern in as pattern")?;
        let binding_node = node
            .child_by_field_name("binding")
            .ok_or("Missing binding in as pattern")?;

        let inner = self.parse_simple_pattern_node(pattern_node, source)?;
        let binding = binding_node.utf8_text(source.as_bytes())?.to_string();

        Ok(Pattern::As(Box::new(inner), binding))
    }

    fn parse_handler_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<HandlerPattern, Box<dyn std::error::Error>> {
        // Check for completion pattern: { result }
        if let Some(result) = node.child_by_field_name("result") {
            let name = result.utf8_text(source.as_bytes())?.to_string();
            return Ok(HandlerPattern::Done(name));
        }

        // Suspend pattern: { operation(args) -> continuation }
        let operation_node = node
            .child_by_field_name("operation")
            .ok_or("Missing operation in handler pattern")?;

        let operation = self.parse_path_or_identifier(operation_node, source)?;

        let args = if let Some(args_node) = node.child_by_field_name("args") {
            self.parse_pattern_list(args_node, source)?
        } else {
            Vec::new()
        };

        let continuation_node = node
            .child_by_field_name("continuation")
            .ok_or("Missing continuation in handler pattern")?;
        let continuation = continuation_node.utf8_text(source.as_bytes())?.to_string();

        Ok(HandlerPattern::Suspend {
            operation,
            args,
            continuation,
        })
    }

    fn parse_path_or_identifier(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Identifier>, Box<dyn std::error::Error>> {
        match node.kind() {
            "identifier" => Ok(vec![node.utf8_text(source.as_bytes())?.to_string()]),
            "path_expression" => {
                let mut cursor = node.walk();
                let mut segments = Vec::new();
                for child in node.named_children(&mut cursor) {
                    if child.kind() != "path_segment" {
                        continue;
                    }
                    if let Some(inner) = child.child(0) {
                        segments.push(inner.utf8_text(source.as_bytes())?.to_string());
                    }
                }
                Ok(segments)
            }
            _ => Err(format!("Unknown operation kind: {}", node.kind()).into()),
        }
    }

    /// Parse raw string: r"...", r#"..."#, r##"..."##, etc.
    /// External scanner handles the full token, we extract content from it
    fn parse_raw_string(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        // raw_string contains raw_string_literal from external scanner
        // The literal includes r, optional #s, quotes, and content
        // We need to extract just the content
        if let Some(literal_node) = node.child(0) {
            let text = literal_node.utf8_text(source.as_bytes())?;
            let content = extract_raw_string_content(text)?;
            Ok(Expr::StringInterpolation(StringInterpolation {
                leading: content,
                segments: Vec::new(),
            }))
        } else {
            // Empty raw string
            Ok(Expr::StringInterpolation(StringInterpolation {
                leading: String::new(),
                segments: Vec::new(),
            }))
        }
    }

    /// Parse bytes string: b"hello", b"\x00\x01"
    fn parse_bytes_string(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut segments: Vec<BytesSegment> = Vec::new();
        let mut leading = Vec::new();
        let mut current_bytes = Vec::new();
        let mut expecting_bytes = true;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "bytes_segment" => {
                    let text = child.utf8_text(source.as_bytes())?;
                    let processed = process_bytes_escape_sequences(text)?;
                    if expecting_bytes {
                        leading.extend(processed);
                        expecting_bytes = false;
                    } else {
                        current_bytes.extend(processed);
                    }
                }
                "bytes_interpolation" => {
                    if let Some(expr_node) = child.child_by_field_name("expression") {
                        let expr = self.node_to_expr(expr_node, source)?;
                        let span = Span {
                            start: expr_node.start_byte(),
                            end: expr_node.end_byte(),
                        };
                        if let Some(last) = segments.last_mut() {
                            last.trailing = std::mem::take(&mut current_bytes);
                        }
                        segments.push(BytesSegment {
                            interpolation: Box::new((expr, span)),
                            trailing: Vec::new(),
                        });
                    }
                }
                _ => {}
            }
        }

        // Handle trailing bytes after last interpolation
        if let Some(last) = segments.last_mut() {
            last.trailing = current_bytes;
        }

        Ok(Expr::BytesInterpolation(BytesInterpolation {
            leading,
            segments,
        }))
    }

    /// Parse raw bytes: rb"hello", rb#"data"#
    fn parse_raw_bytes(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        // raw_bytes contains raw_bytes_literal from external scanner
        if let Some(literal_node) = node.child(0) {
            let text = literal_node.utf8_text(source.as_bytes())?;
            let content = extract_raw_bytes_content(text)?;
            Ok(Expr::BytesInterpolation(BytesInterpolation {
                leading: content.into_bytes(),
                segments: Vec::new(),
            }))
        } else {
            Ok(Expr::BytesInterpolation(BytesInterpolation {
                leading: Vec::new(),
                segments: Vec::new(),
            }))
        }
    }

    /// Parse multiline string: #"hello"#, ##"contains "# inside"##
    /// With optional interpolation: #"hello \{name}"#
    fn parse_multiline_string(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut segments: Vec<StringSegment> = Vec::new();
        let mut leading = String::new();
        let mut current_text = String::new();
        let mut expecting_text = true;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "multiline_string_segment" => {
                    // Content segment - no escape processing for multiline strings
                    let text = child.utf8_text(source.as_bytes())?;
                    current_text.push_str(text);
                }
                "multiline_interpolation" => {
                    if let Some(expr_node) = child.child_by_field_name("expression") {
                        let expr = self.node_to_expr_with_span(expr_node, source)?;

                        if expecting_text {
                            leading = std::mem::take(&mut current_text);
                            expecting_text = false;
                        } else if let Some(last_segment) = segments.last_mut() {
                            last_segment.trailing = std::mem::take(&mut current_text);
                        }

                        segments.push(StringSegment {
                            interpolation: Box::new(expr),
                            trailing: String::new(),
                        });
                    }
                }
                _ => {}
            }
        }

        if segments.is_empty() {
            // Simple multiline string without interpolation
            Ok(Expr::StringInterpolation(StringInterpolation {
                leading: current_text,
                segments: Vec::new(),
            }))
        } else {
            // Set trailing text for the last segment
            if let Some(last_segment) = segments.last_mut() {
                last_segment.trailing = current_text;
            }

            Ok(Expr::StringInterpolation(StringInterpolation {
                leading,
                segments,
            }))
        }
    }

    /// Parse multiline bytes: b#"hello"#, b##"contains "# inside"##
    /// With optional interpolation: b#"hello \{name}"#
    fn parse_multiline_bytes(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut segments: Vec<BytesSegment> = Vec::new();
        let mut leading = Vec::new();
        let mut current_bytes = Vec::new();
        let mut expecting_bytes = true;

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "multiline_bytes_segment" => {
                    // Content segment - no escape processing for multiline bytes
                    let text = child.utf8_text(source.as_bytes())?;
                    current_bytes.extend_from_slice(text.as_bytes());
                }
                "multiline_bytes_interpolation" => {
                    if let Some(expr_node) = child.child_by_field_name("expression") {
                        let expr = self.node_to_expr(expr_node, source)?;
                        let span = Span {
                            start: expr_node.start_byte(),
                            end: expr_node.end_byte(),
                        };

                        if expecting_bytes {
                            leading = std::mem::take(&mut current_bytes);
                            expecting_bytes = false;
                        } else if let Some(last_segment) = segments.last_mut() {
                            last_segment.trailing = std::mem::take(&mut current_bytes);
                        }

                        segments.push(BytesSegment {
                            interpolation: Box::new((expr, span)),
                            trailing: Vec::new(),
                        });
                    }
                }
                _ => {}
            }
        }

        if segments.is_empty() {
            // Simple multiline bytes without interpolation
            Ok(Expr::BytesInterpolation(BytesInterpolation {
                leading: current_bytes,
                segments: Vec::new(),
            }))
        } else {
            // Set trailing bytes for the last segment
            if let Some(last_segment) = segments.last_mut() {
                last_segment.trailing = current_bytes;
            }

            Ok(Expr::BytesInterpolation(BytesInterpolation {
                leading,
                segments,
            }))
        }
    }

    fn parse_interpolated_string(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        let mut segments: Vec<StringSegment> = Vec::new();
        let mut leading = String::new();
        let mut current_text = String::new();
        let mut expecting_text = true; // Start expecting text (leading)

        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "string_segment" => {
                    let text = child.utf8_text(source.as_bytes())?;
                    let processed = process_escape_sequences(text)?;
                    current_text.push_str(&processed);
                }
                "interpolation" => {
                    if let Some(expr_node) = child.child_by_field_name("expression") {
                        let expr = self.node_to_expr_with_span(expr_node, source)?;

                        if expecting_text {
                            // This is the first interpolation, current_text is leading
                            leading = std::mem::take(&mut current_text);
                            expecting_text = false;
                        } else {
                            // Update the trailing of the previous segment
                            if let Some(last_segment) = segments.last_mut() {
                                last_segment.trailing = std::mem::take(&mut current_text);
                            }
                        }

                        // Create new segment with empty trailing (will be filled later)
                        segments.push(StringSegment {
                            interpolation: Box::new(expr),
                            trailing: String::new(),
                        });
                    } else {
                        return Err("Interpolation missing expression".into());
                    }
                }
                _ => {}
            }
        }

        if segments.is_empty() {
            // Simple string without interpolation
            Ok(Expr::StringInterpolation(StringInterpolation {
                leading: current_text,
                segments: Vec::new(),
            }))
        } else {
            // Set trailing text for the last segment
            if let Some(last_segment) = segments.last_mut() {
                last_segment.trailing = current_text;
            }

            Ok(Expr::StringInterpolation(StringInterpolation {
                leading,
                segments,
            }))
        }
    }

    fn parse_literal_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<LiteralPattern, Box<dyn std::error::Error>> {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "nat_literal" => {
                    let text = child.utf8_text(source.as_bytes())?;
                    let num = parse_nat_literal(text)?;
                    return Ok(LiteralPattern::Nat(num));
                }
                "int_literal" => {
                    let text = child.utf8_text(source.as_bytes())?;
                    let num = parse_int_literal(text)?;
                    return Ok(LiteralPattern::Int(num));
                }
                "float_literal" => {
                    let text = child.utf8_text(source.as_bytes())?;
                    let num = text.parse::<f64>()?;
                    return Ok(LiteralPattern::Float(num));
                }
                "rune" => {
                    let text = child.utf8_text(source.as_bytes())?;
                    let ch = parse_rune_literal(text)?;
                    return Ok(LiteralPattern::Rune(ch));
                }
                "string" => {
                    // Parse as StringInterpolation and convert to appropriate pattern
                    if let Ok(Expr::StringInterpolation(interp)) =
                        self.parse_interpolated_string(child, source)
                    {
                        if interp.segments.is_empty() {
                            // Simple string without interpolation
                            return Ok(LiteralPattern::String(interp.leading));
                        } else {
                            // String with interpolation
                            return Ok(LiteralPattern::StringInterpolation(interp));
                        }
                    }
                }
                "raw_string" => {
                    // Raw string pattern - no escape processing
                    if let Ok(Expr::StringInterpolation(interp)) =
                        self.parse_raw_string(child, source)
                    {
                        return Ok(LiteralPattern::String(interp.leading));
                    }
                }
                "bytes_string" => {
                    // Bytes pattern with possible interpolation
                    if let Ok(Expr::BytesInterpolation(interp)) =
                        self.parse_bytes_string(child, source)
                    {
                        if interp.segments.is_empty() {
                            return Ok(LiteralPattern::Bytes(interp.leading));
                        } else {
                            return Ok(LiteralPattern::BytesInterpolation(interp));
                        }
                    }
                }
                "raw_bytes" => {
                    // Raw bytes pattern - no escape processing
                    if let Ok(Expr::BytesInterpolation(interp)) =
                        self.parse_raw_bytes(child, source)
                    {
                        return Ok(LiteralPattern::Bytes(interp.leading));
                    }
                }
                "keyword_true" => return Ok(LiteralPattern::Bool(true)),
                "keyword_false" => return Ok(LiteralPattern::Bool(false)),
                "keyword_nil" => return Ok(LiteralPattern::Nil),
                _ => {}
            }
        }
        Err("Invalid literal pattern".into())
    }
}

/// Error type for string literal processing
#[derive(Debug, Clone, PartialEq, derive_more::Display, derive_more::Error)]
pub enum StringLiteralError {
    #[display("Invalid UTF-8 sequence in processed string")]
    InvalidUtf8Sequence,
    #[display("Incomplete hex escape sequence")]
    IncompleteHexEscape,
    #[display("Invalid hex digits in escape sequence: {hex_str}")]
    InvalidHexDigits { hex_str: String },
    #[display("Unknown escape sequence: \\{char}")]
    UnknownEscapeSequence { char: char },
    #[display("Trailing backslash in string literal")]
    TrailingBackslash,
}

/// Process a hex escape sequence (\xHH) and return the byte value
fn process_hex_escape(chars: &mut std::str::Chars) -> Result<u8, StringLiteralError> {
    let hex1 = chars
        .next()
        .ok_or(StringLiteralError::IncompleteHexEscape)?;
    let hex2 = chars
        .next()
        .ok_or(StringLiteralError::IncompleteHexEscape)?;

    let hex_str = format!("{}{}", hex1, hex2);
    u8::from_str_radix(&hex_str, 16).map_err(|_| StringLiteralError::InvalidHexDigits {
        hex_str: hex_str.clone(),
    })
}

/// Process escape sequences in a string literal
fn process_escape_sequences(input: &str) -> Result<String, StringLiteralError> {
    let mut result = Vec::<u8>::new();
    let mut chars = input.chars();

    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => result.extend_from_slice("\"".as_bytes()),
                Some('\\') => result.extend_from_slice("\\".as_bytes()),
                Some('n') | Some('N') => result.extend_from_slice("\n".as_bytes()),
                Some('t') | Some('T') => result.extend_from_slice("\t".as_bytes()),
                Some('r') | Some('R') => result.extend_from_slice("\r".as_bytes()),
                Some('0') => result.push(0),
                Some('x') | Some('X') => {
                    // Hex escape sequence: \xHH
                    let byte_value = process_hex_escape(&mut chars)?;
                    // Push the raw byte - this might create invalid UTF-8
                    result.push(byte_value);
                }
                Some(other) => {
                    // Unknown escape sequences are now errors
                    return Err(StringLiteralError::UnknownEscapeSequence { char: other });
                }
                None => {
                    // Trailing backslash is now an error
                    return Err(StringLiteralError::TrailingBackslash);
                }
            }
        } else {
            let mut buffer = [0; 4];
            let encoded = ch.encode_utf8(&mut buffer);
            result.extend_from_slice(encoded.as_bytes());
        }
    }

    // Validate that the final result is valid UTF-8
    match String::from_utf8(result) {
        Ok(valid_string) => Ok(valid_string),
        Err(_) => Err(StringLiteralError::InvalidUtf8Sequence),
    }
}

/// Parse an operator token string to a BinaryOperator
fn parse_operator_token(token: &str) -> Option<BinaryOperator> {
    match token {
        // Arithmetic
        "+" => Some(BinaryOperator::Add),
        "-" => Some(BinaryOperator::Subtract),
        "*" => Some(BinaryOperator::Multiply),
        "/" => Some(BinaryOperator::Divide),
        "%" => Some(BinaryOperator::Modulo),
        // Comparison
        "==" => Some(BinaryOperator::Equal),
        "!=" => Some(BinaryOperator::NotEqual),
        "<" => Some(BinaryOperator::LessThan),
        ">" => Some(BinaryOperator::GreaterThan),
        "<=" => Some(BinaryOperator::LessEqual),
        ">=" => Some(BinaryOperator::GreaterEqual),
        // Logical
        "&&" => Some(BinaryOperator::And),
        "||" => Some(BinaryOperator::Or),
        // Concatenation
        "<>" => Some(BinaryOperator::Concat),
        _ => None,
    }
}

/// Extract content from a raw string literal like r"hello" or r#"hello"#
/// Returns the content without the r, #s, and quotes
fn extract_raw_string_content(text: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Must start with 'r'
    let rest = text
        .strip_prefix('r')
        .ok_or("Raw string must start with 'r'")?;

    // Count and skip opening hashes
    let hash_count = rest.chars().take_while(|&c| c == '#').count();
    let after_hashes = &rest[hash_count..];

    // Must have opening quote
    let after_open_quote = after_hashes
        .strip_prefix('"')
        .ok_or("Raw string must have opening quote")?;

    // Remove closing quote and hashes
    // The content ends at the closing " followed by the same number of #s
    let closing = format!("\"{}", "#".repeat(hash_count));
    let content = after_open_quote
        .strip_suffix(&closing)
        .ok_or("Raw string must have matching closing delimiter")?;

    Ok(content.to_string())
}

/// Extract content from a raw bytes literal like rb"hello" or rb#"hello"#
fn extract_raw_bytes_content(text: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Must start with 'rb'
    let rest = text
        .strip_prefix("rb")
        .ok_or("Raw bytes must start with 'rb'")?;

    // Count and skip opening hashes
    let hash_count = rest.chars().take_while(|&c| c == '#').count();
    let after_hashes = &rest[hash_count..];

    // Must have opening quote
    let after_open_quote = after_hashes
        .strip_prefix('"')
        .ok_or("Raw bytes must have opening quote")?;

    // Remove closing quote and hashes
    let closing = format!("\"{}", "#".repeat(hash_count));
    let content = after_open_quote
        .strip_suffix(&closing)
        .ok_or("Raw bytes must have matching closing delimiter")?;

    Ok(content.to_string())
}

/// Process escape sequences in bytes literals
/// Similar to process_escape_sequences but returns Vec<u8>
fn process_bytes_escape_sequences(text: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push(b'\n'),
                Some('r') => result.push(b'\r'),
                Some('t') => result.push(b'\t'),
                Some('0') => result.push(0),
                Some('"') => result.push(b'"'),
                Some('\\') => result.push(b'\\'),
                Some('x') => {
                    // \xNN hex escape
                    let hex: String = chars.by_ref().take(2).collect();
                    if hex.len() == 2 {
                        if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                            result.push(byte);
                        } else {
                            return Err(format!("Invalid hex escape: \\x{}", hex).into());
                        }
                    } else {
                        return Err("Incomplete hex escape".into());
                    }
                }
                Some(other) => {
                    // Unknown escape sequence is an error (consistent with string handling)
                    return Err(format!("Unknown escape sequence: \\{}", other).into());
                }
                None => {
                    // Trailing backslash is an error
                    return Err("Trailing backslash in bytes literal".into());
                }
            }
        } else if c.is_ascii() {
            result.push(c as u8);
        } else {
            // Non-ASCII in bytes literal - encode as UTF-8
            let mut buf = [0u8; 4];
            let encoded = c.encode_utf8(&mut buf);
            result.extend_from_slice(encoded.as_bytes());
        }
    }

    Ok(result)
}

/// Parse a rune literal like ?a, ?\n, ?\x41, ?\u0041
fn parse_rune_literal(text: &str) -> Result<char, Box<dyn std::error::Error>> {
    // Text should start with '?'
    if !text.starts_with('?') {
        return Err("Rune literal must start with '?'".into());
    }

    let rest = &text[1..]; // Skip the '?'

    if let Some(escape) = rest.strip_prefix('\\') {
        // Escape sequence
        match escape.chars().next() {
            Some('n') => Ok('\n'),
            Some('r') => Ok('\r'),
            Some('t') => Ok('\t'),
            Some('0') => Ok('\0'),
            Some('\\') => Ok('\\'),
            Some('x') => {
                // \xHH - two hex digits
                let hex = &escape[1..];
                if hex.len() != 2 {
                    return Err("Invalid hex escape in rune".into());
                }
                let code = u8::from_str_radix(hex, 16)?;
                Ok(code as char)
            }
            Some('u') => {
                // \uHHHH - four hex digits
                let hex = &escape[1..];
                if hex.len() != 4 {
                    return Err("Invalid unicode escape in rune".into());
                }
                let code = u32::from_str_radix(hex, 16)?;
                char::from_u32(code).ok_or_else(|| "Invalid unicode codepoint".into())
            }
            _ => Err(format!("Unknown escape sequence in rune: \\{}", escape).into()),
        }
    } else {
        // Simple character
        let mut chars = rest.chars();
        match chars.next() {
            Some(ch) => {
                if chars.next().is_some() {
                    Err("Rune literal must be a single character".into())
                } else {
                    Ok(ch)
                }
            }
            None => Err("Empty rune literal".into()),
        }
    }
}

/// Parse a Nat literal: decimal, binary, octal, or hexadecimal
/// Examples: 42, 0b1010, 0o777, 0xc0ffee
fn parse_nat_literal(text: &str) -> Result<u64, Box<dyn std::error::Error>> {
    if let Some(bin) = text.strip_prefix("0b").or_else(|| text.strip_prefix("0B")) {
        Ok(u64::from_str_radix(bin, 2)?)
    } else if let Some(oct) = text.strip_prefix("0o").or_else(|| text.strip_prefix("0O")) {
        Ok(u64::from_str_radix(oct, 8)?)
    } else if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        Ok(u64::from_str_radix(hex, 16)?)
    } else {
        Ok(text.parse::<u64>()?)
    }
}

/// Parse an Int literal: signed decimal, binary, octal, or hexadecimal
/// Examples: +1, -1, +0b1010, -0xff
fn parse_int_literal(text: &str) -> Result<i64, Box<dyn std::error::Error>> {
    let (sign, rest) = if let Some(rest) = text.strip_prefix('+') {
        (1i64, rest)
    } else if let Some(rest) = text.strip_prefix('-') {
        (-1i64, rest)
    } else {
        return Err("Int literal must have explicit sign".into());
    };

    let abs_value = if let Some(bin) = rest.strip_prefix("0b").or_else(|| rest.strip_prefix("0B")) {
        u64::from_str_radix(bin, 2)?
    } else if let Some(oct) = rest.strip_prefix("0o").or_else(|| rest.strip_prefix("0O")) {
        u64::from_str_radix(oct, 8)?
    } else if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16)?
    } else {
        rest.parse::<u64>()?
    };

    // Handle potential overflow, especially for i64::MIN
    // i64::MIN = -9223372036854775808, but i64::MAX = 9223372036854775807
    // So |i64::MIN| = i64::MAX + 1
    if sign == 1 {
        if abs_value > i64::MAX as u64 {
            return Err("Integer literal out of range for i64".into());
        }
        Ok(abs_value as i64)
    } else {
        // For negative numbers, we can represent up to i64::MAX + 1
        if abs_value > (i64::MAX as u64) + 1 {
            return Err("Integer literal out of range for i64".into());
        }
        if abs_value == (i64::MAX as u64) + 1 {
            // Special case: i64::MIN
            Ok(i64::MIN)
        } else {
            Ok(-(abs_value as i64))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let db = tribute_core::TributeDatabaseImpl::default();
        use salsa::Database;

        db.attach(|db| {
            let source_file = tribute_core::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn main() {
    print_line("Hello, world!")
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);

            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db) else {
                panic!("Expected function")
            };
            assert_eq!(func.name(db), "main");
            assert_eq!(func.parameters(db).len(), 0);
            assert_eq!(func.body(db).statements.len(), 1);
        });
    }

    #[test]
    fn test_function_with_parameters() {
        let db = tribute_core::TributeDatabaseImpl::default();
        use salsa::Database;

        db.attach(|db| {
            let source_file = tribute_core::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn add(a, b) {
    a + b
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);

            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db) else {
                panic!("Expected function")
            };
            assert_eq!(func.name(db), "add");
            let params = func.parameters(db);
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "a");
            assert_eq!(params[0].ty, None);
            assert_eq!(params[1].name, "b");
            assert_eq!(params[1].ty, None);
        });
    }

    #[test]
    fn test_function_with_typed_parameters() {
        let db = tribute_core::TributeDatabaseImpl::default();
        use salsa::Database;

        db.attach(|db| {
            let source_file = tribute_core::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn add(x: Int, y: Int) -> Int {
    x + y
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);

            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db) else {
                panic!("Expected function")
            };
            assert_eq!(func.name(db), "add");
            let params = func.parameters(db);
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "x");
            assert_eq!(params[0].ty, Some(TypeRef::Named("Int".to_string())));
            assert_eq!(params[1].name, "y");
            assert_eq!(params[1].ty, Some(TypeRef::Named("Int".to_string())));
            assert_eq!(
                func.return_type(db),
                Some(TypeRef::Named("Int".to_string()))
            );
        });
    }

    #[test]
    fn test_function_with_mixed_parameters() {
        let db = tribute_core::TributeDatabaseImpl::default();
        use salsa::Database;

        db.attach(|db| {
            let source_file = tribute_core::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn bar(x: a, y) -> a {
    x
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);

            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db) else {
                panic!("Expected function")
            };
            assert_eq!(func.name(db), "bar");
            let params = func.parameters(db);
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "x");
            assert_eq!(params[0].ty, Some(TypeRef::Variable("a".to_string())));
            assert_eq!(params[1].name, "y");
            assert_eq!(params[1].ty, None);
            assert_eq!(
                func.return_type(db),
                Some(TypeRef::Variable("a".to_string()))
            );
        });
    }

    #[test]
    fn test_case_expression() {
        let db = tribute_core::TributeDatabaseImpl::default();
        use salsa::Database;

        db.attach(|db| {
            let source_file = tribute_core::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn test(n) {
    case n {
        0 -> "zero",
        1 -> "one",
        _ -> "other"
    }
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);

            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db) else {
                panic!("Expected function")
            };
            if let Statement::Expression((Expr::Match(_), _)) = &func.body(db).statements[0] {
                // Case expression parsed successfully
            } else {
                panic!("Expected case expression");
            }
        });
    }

    #[test]
    fn test_process_escape_sequences_basic() {
        assert_eq!(process_escape_sequences("hello"), Ok("hello".to_string()));
        assert_eq!(process_escape_sequences(""), Ok("".to_string()));
    }

    #[test]
    fn test_process_escape_sequences_quotes() {
        assert_eq!(
            process_escape_sequences(r#"Hello \"World\""#),
            Ok(r#"Hello "World""#.to_string())
        );
        assert_eq!(process_escape_sequences(r#"\""#), Ok(r#"""#.to_string()));
    }

    #[test]
    fn test_process_escape_sequences_backslash() {
        assert_eq!(
            process_escape_sequences(r"C:\\Users\\name"),
            Ok(r"C:\Users\name".to_string())
        );
        assert_eq!(process_escape_sequences(r"\\"), Ok(r"\".to_string()));
    }

    #[test]
    fn test_process_escape_sequences_whitespace() {
        assert_eq!(
            process_escape_sequences(r"Line 1\nLine 2"),
            Ok("Line 1\nLine 2".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Tab\there"),
            Ok("Tab\there".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Carriage\rReturn"),
            Ok("Carriage\rReturn".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_case_insensitive() {
        // Test uppercase escape sequences
        assert_eq!(
            process_escape_sequences(r"Line 1\NLine 2"),
            Ok("Line 1\nLine 2".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Tab\There"),
            Ok("Tab\there".to_string())
        );
        assert_eq!(
            process_escape_sequences(r"Carriage\RReturn"),
            Ok("Carriage\rReturn".to_string())
        );

        // Test mixed case
        assert_eq!(
            process_escape_sequences(r"Mixed\n\T\r\N"),
            Ok("Mixed\n\t\r\n".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_null() {
        assert_eq!(
            process_escape_sequences(r"Null\0char"),
            Ok("Null\0char".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_unknown() {
        // Unknown escape sequences should now be errors
        assert_eq!(
            process_escape_sequences(r"Unknown\z escape"),
            Err(StringLiteralError::UnknownEscapeSequence { char: 'z' })
        );
        assert_eq!(
            process_escape_sequences(r"\z"),
            Err(StringLiteralError::UnknownEscapeSequence { char: 'z' })
        );
    }

    #[test]
    fn test_process_escape_sequences_trailing_backslash() {
        assert_eq!(
            process_escape_sequences(r"trailing\"),
            Err(StringLiteralError::TrailingBackslash)
        );
    }

    #[test]
    fn test_process_escape_sequences_mixed() {
        assert_eq!(
            process_escape_sequences(r#"Mixed: \"quote\" and \\backslash\n"#),
            Ok("Mixed: \"quote\" and \\backslash\n".to_string())
        );
    }

    #[test]
    fn test_process_escape_sequences_hex() {
        // Valid hex escapes
        assert_eq!(process_escape_sequences(r"\x41"), Ok("A".to_string()));
        assert_eq!(
            process_escape_sequences(r"Hello\x20World"),
            Ok("Hello World".to_string())
        );

        // Invalid hex escapes should return errors
        assert_eq!(
            process_escape_sequences(r"\x4"),
            Err(StringLiteralError::IncompleteHexEscape)
        );
        assert_eq!(
            process_escape_sequences(r"\xGG"),
            Err(StringLiteralError::InvalidHexDigits {
                hex_str: "GG".to_string()
            })
        );
        assert_eq!(
            process_escape_sequences(r"\x"),
            Err(StringLiteralError::IncompleteHexEscape)
        );
    }

    #[test]
    fn test_process_escape_sequences_invalid_utf8() {
        // Create invalid UTF-8 sequences using hex escapes
        // 0xFF is not valid UTF-8
        assert_eq!(
            process_escape_sequences(r"\xFF"),
            Err(StringLiteralError::InvalidUtf8Sequence)
        );

        // 0x80 without proper leading byte is invalid UTF-8
        assert_eq!(
            process_escape_sequences(r"\x80"),
            Err(StringLiteralError::InvalidUtf8Sequence)
        );

        // Valid UTF-8 should work
        assert_eq!(
            process_escape_sequences(r"\x41\x42\x43"),
            Ok("ABC".to_string())
        );
    }

    #[test]
    fn test_string_parsing_with_escape_sequences() {
        let db = tribute_core::TributeDatabaseImpl::default();
        use salsa::Database;

        db.attach(|db| {
            // Test basic quote escaping
            let source_file = tribute_core::SourceFile::new(
                db,
                std::path::PathBuf::from("test.trb"),
                r#"
fn test() {
    "Hello \"World\""
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file);
            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db) else {
                panic!("Expected function")
            };
            if let Statement::Expression((Expr::StringInterpolation(interp), _)) =
                &func.body(db).statements[0]
            {
                assert_eq!(interp.leading, "Hello \"World\"");
            } else {
                panic!("Expected string expression");
            }

            // Test mixed escape sequences
            let source_file2 = SourceFile::new(
                db,
                std::path::PathBuf::from("test2.trb"),
                r#"
fn test() {
    "Line1\nTab\tQuote\""
}
"#
                .to_string(),
            );
            let result = parse_source_file(db, source_file2);
            assert_eq!(result.items(db).len(), 1);
            let ItemKind::Function(func) = result.items(db)[0].kind(db) else {
                panic!("Expected function")
            };
            if let Statement::Expression((Expr::StringInterpolation(interp), _)) =
                &func.body(db).statements[0]
            {
                assert_eq!(interp.leading, "Line1\nTab\tQuote\"");
            } else {
                panic!("Expected string expression");
            }
        });
    }

    #[test]
    fn test_process_bytes_escape_sequences() {
        // Valid escapes
        assert_eq!(
            process_bytes_escape_sequences(r"hello").unwrap(),
            b"hello".to_vec()
        );
        assert_eq!(
            process_bytes_escape_sequences(r"line\nbreak").unwrap(),
            b"line\nbreak".to_vec()
        );
        assert_eq!(
            process_bytes_escape_sequences(r"\t\r\n").unwrap(),
            b"\t\r\n".to_vec()
        );
        assert_eq!(process_bytes_escape_sequences(r"\0").unwrap(), vec![0]);
        assert_eq!(
            process_bytes_escape_sequences(r#"\""#).unwrap(),
            b"\"".to_vec()
        );
        assert_eq!(
            process_bytes_escape_sequences(r"\\").unwrap(),
            b"\\".to_vec()
        );

        // Hex escapes
        assert_eq!(
            process_bytes_escape_sequences(r"\x41\x42\x43").unwrap(),
            b"ABC".to_vec()
        );
        assert_eq!(
            process_bytes_escape_sequences(r"\x00\xff").unwrap(),
            vec![0x00, 0xff]
        );

        // Unknown escape is an error (consistent with strings)
        assert!(process_bytes_escape_sequences(r"\z").is_err());
        assert!(process_bytes_escape_sequences(r"test\q").is_err());

        // Trailing backslash is an error
        assert!(process_bytes_escape_sequences(r"trailing\").is_err());

        // Invalid hex escapes
        assert!(process_bytes_escape_sequences(r"\x").is_err());
        assert!(process_bytes_escape_sequences(r"\x4").is_err());
        assert!(process_bytes_escape_sequences(r"\xGG").is_err());
    }

    #[test]
    fn test_parse_int_literal() {
        // Basic positive and negative numbers
        assert_eq!(parse_int_literal("+42").unwrap(), 42);
        assert_eq!(parse_int_literal("-42").unwrap(), -42);
        assert_eq!(parse_int_literal("+0").unwrap(), 0);
        assert_eq!(parse_int_literal("-0").unwrap(), 0);

        // i64::MAX and i64::MIN
        assert_eq!(parse_int_literal("+9223372036854775807").unwrap(), i64::MAX);
        assert_eq!(parse_int_literal("-9223372036854775808").unwrap(), i64::MIN);

        // Overflow cases
        assert!(parse_int_literal("+9223372036854775808").is_err()); // i64::MAX + 1
        assert!(parse_int_literal("-9223372036854775809").is_err()); // i64::MIN - 1

        // Binary, octal, hex
        assert_eq!(parse_int_literal("+0b1010").unwrap(), 10);
        assert_eq!(parse_int_literal("-0xff").unwrap(), -255);
        assert_eq!(parse_int_literal("+0o777").unwrap(), 511);

        // Missing sign is an error
        assert!(parse_int_literal("42").is_err());
    }
}

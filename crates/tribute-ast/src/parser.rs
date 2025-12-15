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

        for i in 0..root_node.child_count() {
            if let Some(child) = root_node.child(i) {
                // Skip comments and whitespace
                if child.kind() == "line_comment"
                    || child.kind() == "block_comment"
                    || child.kind() == "ERROR"
                {
                    continue;
                }
                match self.node_to_item(db, child, source) {
                    Ok(item) => items.push(item),
                    Err(e) => return Err(e),
                }
            }
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
        match node.kind() {
            "function_definition" => {
                let mut name = None;
                let mut parameters = Vec::new();
                let mut body = None;

                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        match child.kind() {
                            "identifier" => {
                                if name.is_none() {
                                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                                }
                            }
                            "parameter_list" => {
                                parameters = self.parse_parameter_list(child, source)?;
                            }
                            "block" => {
                                body = Some(self.parse_block(child, source)?);
                            }
                            _ => {} // Skip other tokens like 'fn', '(', ')'
                        }
                    }
                }

                let name = name.ok_or("Missing function name")?;
                let body = body.ok_or("Missing function body")?;
                let span = Span::new(node.start_byte(), node.end_byte());

                Ok(Item::new(
                    db,
                    ItemKind::Function(FunctionDefinition::new(db, name, parameters, body, span)),
                    span,
                ))
            }
            "struct_declaration" => {
                let mut name = None;
                let mut type_params = Vec::new();
                let mut fields = Vec::new();
                let mut is_pub = false;

                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        match child.kind() {
                            "keyword_pub" => {
                                is_pub = true;
                            }
                            "type_identifier" => {
                                if name.is_none() {
                                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                                }
                            }
                            "type_parameters" => {
                                type_params = self.parse_type_parameters(child, source)?;
                            }
                            "struct_body" => {
                                fields = self.parse_struct_body(child, source)?;
                            }
                            _ => {} // Skip other tokens
                        }
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

                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
                        match child.kind() {
                            "keyword_pub" => {
                                is_pub = true;
                            }
                            "type_identifier" => {
                                if name.is_none() {
                                    name = Some(child.utf8_text(source.as_bytes())?.to_string());
                                }
                            }
                            "type_parameters" => {
                                type_params = self.parse_type_parameters(child, source)?;
                            }
                            "enum_body" => {
                                variants = self.parse_enum_body(child, source)?;
                            }
                            _ => {} // Skip other tokens
                        }
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
            _ => Err(format!("Unknown item kind: {}", node.kind()).into()),
        }
    }

    fn parse_parameter_list(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Identifier>, Box<dyn std::error::Error>> {
        let mut parameters = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i)
                && child.kind() == "identifier"
            {
                parameters.push(child.utf8_text(source.as_bytes())?.to_string());
            }
        }

        Ok(parameters)
    }

    fn parse_type_parameters(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<Identifier>, Box<dyn std::error::Error>> {
        let mut params = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i)
                && child.kind() == "identifier"
            {
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
        let mut fields = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i)
                && child.kind() == "struct_fields"
            {
                fields = self.parse_struct_fields(child, source)?;
            }
        }

        Ok(fields)
    }

    fn parse_struct_fields(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<StructField>, Box<dyn std::error::Error>> {
        let mut fields = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i)
                && child.kind() == "struct_field"
            {
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
        let mut name = None;
        let mut ty = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => {
                        if name.is_none() {
                            name = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                    }
                    "type_identifier" | "type_variable" | "generic_type" => {
                        ty = Some(self.parse_type_ref(child, source)?);
                    }
                    _ => {} // Skip colon
                }
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
        let mut variants = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i)
                && child.kind() == "enum_variants"
            {
                variants = self.parse_enum_variants(child, source)?;
            }
        }

        Ok(variants)
    }

    fn parse_enum_variants(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Vec<EnumVariant>, Box<dyn std::error::Error>> {
        let mut variants = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i)
                && child.kind() == "enum_variant"
            {
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
        let mut name = None;
        let mut fields = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "type_identifier" => {
                        if name.is_none() {
                            name = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                    }
                    "variant_fields" => {
                        fields = Some(self.parse_variant_fields(child, source)?);
                    }
                    _ => {} // Skip other tokens
                }
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
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "tuple_fields" => {
                        return self.parse_tuple_fields(child, source);
                    }
                    "struct_fields_block" => {
                        return self.parse_struct_fields_block(child, source);
                    }
                    _ => {}
                }
            }
        }

        Err("Empty variant fields".into())
    }

    fn parse_tuple_fields(
        &self,
        node: Node,
        source: &str,
    ) -> Result<VariantFields, Box<dyn std::error::Error>> {
        let mut types = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "type_identifier" | "type_variable" | "generic_type" => {
                        types.push(self.parse_type_ref(child, source)?);
                    }
                    _ => {} // Skip parens and commas
                }
            }
        }

        Ok(VariantFields::Tuple(types))
    }

    fn parse_struct_fields_block(
        &self,
        node: Node,
        source: &str,
    ) -> Result<VariantFields, Box<dyn std::error::Error>> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i)
                && child.kind() == "struct_fields"
            {
                let fields = self.parse_struct_fields(child, source)?;
                return Ok(VariantFields::Struct(fields));
            }
        }

        Ok(VariantFields::Struct(Vec::new()))
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
                let mut name = None;
                let mut args = Vec::new();

                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i) {
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
                            _ => {} // Skip parens and commas
                        }
                    }
                }

                let name = name.ok_or("Missing generic type name")?;
                Ok(TypeRef::Generic { name, args })
            }
            _ => Err(format!("Unknown type reference kind: {}", node.kind()).into()),
        }
    }

    fn parse_block(&self, node: Node, source: &str) -> Result<Block, Box<dyn std::error::Error>> {
        let mut statements = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "let_statement" => {
                        statements.push(Statement::Let(self.parse_let_statement(child, source)?));
                    }
                    "expression_statement" => {
                        statements.push(Statement::Expression(
                            self.parse_expression_statement(child, source)?,
                        ));
                    }
                    "line_comment" | "block_comment" | "{" | "}" => {
                        // Skip comments and block delimiters
                    }
                    _ => {
                        // Try to parse as expression statement
                        if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                            statements.push(Statement::Expression(expr));
                        }
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
        let mut name = None;
        let mut value = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => {
                        if name.is_none() {
                            name = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                    }
                    _ => {
                        // Try to parse as expression
                        if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                            value = Some(expr);
                        }
                    }
                }
            }
        }

        let name = name.ok_or("Missing let variable name")?;
        let value = value.ok_or("Missing let value")?;

        Ok(LetStatement { name, value })
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
            "number" => {
                let text = node.utf8_text(source.as_bytes())?;
                let num = text.parse::<i64>()?;
                Ok(Expr::Number(num))
            }
            "string" => {
                // All strings are now StringInterpolation, even simple ones
                self.parse_interpolated_string(node, source)
            }
            "identifier" => {
                let text = node.utf8_text(source.as_bytes())?;
                Ok(Expr::Identifier(text.to_string()))
            }
            "binary_expression" => self.parse_binary_expression(node, source),
            "call_expression" => self.parse_call_expression(node, source),
            "method_call_expression" => self.parse_method_call_expression(node, source),
            "case_expression" => self.parse_case_expression(node, source),
            "list_expression" => self.parse_list_expression(node, source),
            "tuple_expression" => self.parse_tuple_expression(node, source),
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
        let mut left = None;
        let mut operator = None;
        let mut right = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    // Arithmetic
                    "+" => operator = Some(BinaryOperator::Add),
                    "-" => operator = Some(BinaryOperator::Subtract),
                    "*" => operator = Some(BinaryOperator::Multiply),
                    "/" => operator = Some(BinaryOperator::Divide),
                    "%" => operator = Some(BinaryOperator::Modulo),
                    // Comparison
                    "==" => operator = Some(BinaryOperator::Equal),
                    "!=" => operator = Some(BinaryOperator::NotEqual),
                    "<" => operator = Some(BinaryOperator::LessThan),
                    ">" => operator = Some(BinaryOperator::GreaterThan),
                    "<=" => operator = Some(BinaryOperator::LessEqual),
                    ">=" => operator = Some(BinaryOperator::GreaterEqual),
                    // Logical
                    "&&" => operator = Some(BinaryOperator::And),
                    "||" => operator = Some(BinaryOperator::Or),
                    // Concatenation
                    "<>" => operator = Some(BinaryOperator::Concat),
                    _ => {
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
            }
        }

        let left = left.ok_or("Missing left operand")?;
        let operator = operator.ok_or("Missing operator")?;
        let right = right.ok_or("Missing right operand")?;

        Ok(Expr::Binary(BinaryExpression {
            left,
            operator,
            right,
        }))
    }

    fn parse_call_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut function = None;
        let mut arguments = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "identifier" => {
                        if function.is_none() {
                            function = Some(child.utf8_text(source.as_bytes())?.to_string());
                        }
                    }
                    "argument_list" => {
                        arguments = self.parse_argument_list(child, source)?;
                    }
                    _ => {} // Skip other tokens like '(', ')'
                }
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
        let mut receiver = None;
        let mut method = None;
        let mut arguments = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
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
                    "." | "(" | ")" => {
                        // Skip tokens
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
        let mut arguments = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i)
                && child.kind() != ","
                && let Ok(expr) = self.node_to_expr_with_span(child, source)
            {
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
        let mut value = None;
        let mut arms = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "case_arm" => {
                        arms.push(self.parse_case_arm(child, source)?);
                    }
                    "keyword_case" | "{" | "}" => {
                        // Skip keywords and delimiters
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
        }

        let value = value.ok_or("Missing case value")?;

        Ok(Expr::Match(MatchExpression { value, arms }))
    }

    fn parse_case_arm(
        &self,
        node: Node,
        source: &str,
    ) -> Result<MatchArm, Box<dyn std::error::Error>> {
        let mut pattern = None;
        let mut value = None;

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "pattern" => {
                        pattern = Some(self.parse_pattern(child, source)?);
                    }
                    "->" | "," => {
                        // Skip tokens
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
        }

        let pattern = pattern.ok_or("Missing pattern")?;
        let value = value.ok_or("Missing case arm value")?;

        Ok(MatchArm { pattern, value })
    }

    fn parse_list_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut elements = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "[" | "]" | "," => {
                        // Skip brackets and commas
                    }
                    _ => {
                        if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                            elements.push(expr);
                        }
                    }
                }
            }
        }

        Ok(Expr::List(elements))
    }

    fn parse_tuple_expression(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut elements = Vec::new();

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "#" | "(" | ")" | "," => {
                        // Skip hash, parentheses and commas
                    }
                    _ => {
                        if let Ok(expr) = self.node_to_expr_with_span(child, source) {
                            elements.push(expr);
                        }
                    }
                }
            }
        }

        Ok(Expr::Tuple(elements))
    }

    fn parse_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Pattern, Box<dyn std::error::Error>> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "literal_pattern" => {
                        return Ok(Pattern::Literal(self.parse_literal_pattern(child, source)?));
                    }
                    "wildcard_pattern" => {
                        return Ok(Pattern::Wildcard);
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
        }
        Err("Invalid pattern".into())
    }

    fn parse_interpolated_string(
        &self,
        node: Node,
        source: &str,
    ) -> Result<Expr, Box<dyn std::error::Error>> {
        let mut segments: Vec<StringSegment> = Vec::new();
        let mut leading_text = String::new();
        let mut current_text = String::new();
        let mut expecting_text = true; // Start expecting text (leading_text)

        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
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
                                // This is the first interpolation, current_text is leading_text
                                leading_text = std::mem::take(&mut current_text);
                                expecting_text = false;
                            } else {
                                // Update the trailing_text of the previous segment
                                if let Some(last_segment) = segments.last_mut() {
                                    last_segment.trailing_text = std::mem::take(&mut current_text);
                                }
                            }

                            // Create new segment with empty trailing_text (will be filled later)
                            segments.push(StringSegment {
                                interpolation: Box::new(expr),
                                trailing_text: String::new(),
                            });
                        } else {
                            return Err("Interpolation missing expression".into());
                        }
                    }
                    "\"" => {
                        // Skip quote delimiters
                    }
                    _ => {}
                }
            }
        }

        if segments.is_empty() {
            // Simple string without interpolation
            Ok(Expr::StringInterpolation(StringInterpolation {
                leading_text: current_text,
                segments: Vec::new(),
            }))
        } else {
            // Set trailing text for the last segment
            if let Some(last_segment) = segments.last_mut() {
                last_segment.trailing_text = current_text;
            }

            Ok(Expr::StringInterpolation(StringInterpolation {
                leading_text,
                segments,
            }))
        }
    }

    fn parse_literal_pattern(
        &self,
        node: Node,
        source: &str,
    ) -> Result<LiteralPattern, Box<dyn std::error::Error>> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                match child.kind() {
                    "number" => {
                        let text = child.utf8_text(source.as_bytes())?;
                        let num = text.parse::<i64>()?;
                        return Ok(LiteralPattern::Number(num));
                    }
                    "string" => {
                        // Parse as StringInterpolation and convert to appropriate pattern
                        if let Ok(Expr::StringInterpolation(interp)) =
                            self.parse_interpolated_string(child, source)
                        {
                            if interp.segments.is_empty() {
                                // Simple string without interpolation
                                return Ok(LiteralPattern::String(interp.leading_text));
                            } else {
                                // String with interpolation
                                return Ok(LiteralPattern::StringInterpolation(interp));
                            }
                        }
                    }
                    _ => {}
                }
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
            assert_eq!(func.parameters(db), vec!["a".to_string(), "b".to_string()]);
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
                assert_eq!(interp.leading_text, "Hello \"World\"");
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
                assert_eq!(interp.leading_text, "Line1\nTab\tQuote\"");
            } else {
                panic!("Expected string expression");
            }
        });
    }
}

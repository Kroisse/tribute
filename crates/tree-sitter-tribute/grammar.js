/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
  name: 'tribute',

  rules: {
    source_file: $ => repeat($._item),

    _item: $ => choice(
      $.function_definition,
      $.struct_declaration,
      $.enum_declaration
    ),

    // struct User { name: String, age: Nat }
    // struct Box(a) { value: a }
    // pub struct Point { x: Int, y: Int }
    struct_declaration: $ => seq(
      optional($.keyword_pub),
      $.keyword_struct,
      field('name', $.type_identifier),
      optional(field('type_params', $.type_parameters)),
      field('body', $.struct_body)
    ),

    type_parameters: $ => seq(
      '(',
      $.identifier,
      repeat(seq(',', $.identifier)),
      optional(','),
      ')'
    ),

    struct_body: $ => seq(
      '{',
      optional($.struct_fields),
      '}'
    ),

    struct_fields: $ => seq(
      $.struct_field,
      repeat(seq(optional(','), $.struct_field)),
      optional(',')
    ),

    struct_field: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type)
    ),

    // Type reference (simple for now)
    // Can be: String (type_identifier), a (type_variable), List(a) (generic_type)
    _type: $ => choice(
      $.type_identifier,
      $.type_variable,
      $.generic_type
    ),

    // Type variables are lowercase: a, b, elem
    type_variable: $ => $.identifier,

    // Generic type like List(a) or Option(String)
    generic_type: $ => seq(
      $.type_identifier,
      '(',
      $._type,
      repeat(seq(',', $._type)),
      optional(','),
      ')'
    ),

    // Type names start with uppercase: User, String, List
    type_identifier: $ => /[A-Z][a-zA-Z0-9_]*/,

    // enum Option(a) { None, Some(a) }
    // enum Result(a, e) { Ok { value: a }, Err { error: e } }
    // pub enum Status { Active, Inactive }
    enum_declaration: $ => seq(
      optional($.keyword_pub),
      $.keyword_enum,
      field('name', $.type_identifier),
      optional(field('type_params', $.type_parameters)),
      field('body', $.enum_body)
    ),

    enum_body: $ => seq(
      '{',
      optional($.enum_variants),
      '}'
    ),

    enum_variants: $ => seq(
      $.enum_variant,
      repeat(seq(optional(','), $.enum_variant)),
      optional(',')
    ),

    // Variant: None, Some(a), Ok { value: a }
    enum_variant: $ => seq(
      field('name', $.type_identifier),
      optional(field('fields', $.variant_fields))
    ),

    // Variant fields: tuple (a, b) or struct { name: Type }
    variant_fields: $ => choice(
      $.tuple_fields,
      $.struct_fields_block
    ),

    // Tuple variant fields: (Int, String)
    tuple_fields: $ => seq(
      '(',
      $._type,
      repeat(seq(',', $._type)),
      optional(','),
      ')'
    ),

    // Struct variant fields: { name: Type, ... }
    struct_fields_block: $ => seq(
      '{',
      optional($.struct_fields),
      '}'
    ),

    function_definition: $ => seq(
      $.keyword_fn,
      field('name', $.identifier),
      '(',
      optional($.parameter_list),
      ')',
      field('body', $.block)
    ),

    parameter_list: $ => seq(
      $.identifier,
      repeat(seq(',', $.identifier))
    ),

    block: $ => seq(
      '{',
      repeat(choice(
        $.let_statement,
        $.expression_statement
      )),
      '}'
    ),

    let_statement: $ => seq(
      $.keyword_let,
      field('name', $.identifier),
      '=',
      field('value', $._expression)
    ),

    expression_statement: $ => $._expression,


    _expression: $ => choice(
      $.binary_expression,
      $.call_expression,
      $.method_call_expression,
      $.case_expression,
      $.primary_expression
    ),

    binary_expression: $ => choice(
      // Precedence (higher number = binds tighter)
      // 5: * / %
      prec.left(5, seq($._expression, '*', $._expression)),
      prec.left(5, seq($._expression, '/', $._expression)),
      prec.left(5, seq($._expression, '%', $._expression)),
      // 4: + - <>
      prec.left(4, seq($._expression, '+', $._expression)),
      prec.left(4, seq($._expression, '-', $._expression)),
      prec.left(4, seq($._expression, '<>', $._expression)),
      // 3: == != < > <= >=
      prec.left(3, seq($._expression, '==', $._expression)),
      prec.left(3, seq($._expression, '!=', $._expression)),
      prec.left(3, seq($._expression, '<', $._expression)),
      prec.left(3, seq($._expression, '>', $._expression)),
      prec.left(3, seq($._expression, '<=', $._expression)),
      prec.left(3, seq($._expression, '>=', $._expression)),
      // 2: &&
      prec.left(2, seq($._expression, '&&', $._expression)),
      // 1: ||
      prec.left(1, seq($._expression, '||', $._expression))
    ),

    call_expression: $ => prec(10, seq(
      field('function', $.identifier),
      '(',
      optional($.argument_list),
      ')'
    )),

    // UFCS: x.f(y) -> f(x, y), x.f -> f(x)
    method_call_expression: $ => prec.left(10, seq(
      field('receiver', $._expression),
      '.',
      field('method', $.identifier),
      optional(seq('(', optional($.argument_list), ')'))
    )),

    case_expression: $ => seq(
      $.keyword_case,
      field('value', $._expression),
      '{',
      repeat($.case_arm),
      '}'
    ),

    case_arm: $ => seq(
      field('pattern', $.pattern),
      '->',
      field('value', $._expression),
      optional(',')
    ),

    pattern: $ => choice(
      $.literal_pattern,
      $.wildcard_pattern,
      $.identifier_pattern
    ),

    literal_pattern: $ => choice(
      $.number,
      $.string
    ),

    wildcard_pattern: $ => '_',

    identifier_pattern: $ => $.identifier,

    argument_list: $ => seq(
      $._expression,
      repeat(seq(',', $._expression))
    ),

    primary_expression: $ => choice(
      $.number,
      $.string,
      $.identifier,
      $.list_expression,
      $.tuple_expression,
      $.block  // { expr } for grouping
    ),

    // [1, 2, 3]
    list_expression: $ => seq(
      '[',
      optional(seq(
        $._expression,
        repeat(seq(',', $._expression)),
        optional(',')  // trailing comma
      )),
      ']'
    ),

    // #(1, "hello", 3.14)
    tuple_expression: $ => seq(
      '#',
      '(',
      optional(seq(
        $._expression,
        repeat(seq(',', $._expression)),
        optional(',')  // trailing comma
      )),
      ')'
    ),

    number: $ => /-?\d+(\.\d+)?/,

    string: $ => seq(
      '"',
      $.string_segment,
      optional(repeat1(seq(
        $.interpolation,
        $.string_segment
      ))),
      '"'
    ),

    string_segment: $ => prec(-1, /([^"\\]|\\[nrtN0"\\]|\\x[0-9a-fA-F]{2})*/),

    interpolation: $ => seq(
      '\\',
      '{',
      field('expression', $._expression),
      '}'
    ),

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    // Keywords
    keyword_fn: $ => 'fn',
    keyword_let: $ => 'let',
    keyword_case: $ => 'case',
    keyword_struct: $ => 'struct',
    keyword_enum: $ => 'enum',
    keyword_pub: $ => 'pub',

    // Comments
    line_comment: $ => token(seq(
      '//',
      /.*/
    )),
  },

  extras: $ => [
    /\s/,
    $.line_comment,
  ],

  word: $ => $.identifier,

  inline: $ => [],
});

module.exports = grammar({
  name: 'tribute',

  rules: {
    source_file: $ => repeat($._item),

    _item: $ => $.function_definition,

    function_definition: $ => seq(
      'fn',
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
      'let',
      field('name', $.identifier),
      '=',
      field('value', $._expression)
    ),

    expression_statement: $ => $._expression,


    _expression: $ => choice(
      $.binary_expression,
      $.call_expression,
      $.match_expression,
      $.primary_expression
    ),

    binary_expression: $ => choice(
      prec.left(1, seq($._expression, '+', $._expression)),
      prec.left(1, seq($._expression, '-', $._expression)),
      prec.left(2, seq($._expression, '*', $._expression)),
      prec.left(2, seq($._expression, '/', $._expression))
    ),

    call_expression: $ => prec(10, seq(
      field('function', $.identifier),
      '(',
      optional($.argument_list),
      ')'
    )),

    match_expression: $ => seq(
      'match',
      field('value', $._expression),
      '{',
      repeat($.match_arm),
      '}'
    ),

    match_arm: $ => seq(
      field('pattern', $.pattern),
      '=>',
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
      $.parenthesized_expression
    ),

    parenthesized_expression: $ => seq(
      '(',
      $._expression,
      ')'
    ),

    number: $ => /-?\d+(\.\d+)?/,

    string: $ => /"([^"\\]|\\.)*"/,

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    // Comments
    line_comment: $ => token(seq(
      '//',
      /.*/
    )),

    block_comment: $ => token(seq(
      '/*',
      /[^*]*\*+([^/*][^*]*\*+)*/,
      '/'
    ))
  },

  extras: $ => [
    /\s/,
    $.line_comment,
    $.block_comment
  ]
});
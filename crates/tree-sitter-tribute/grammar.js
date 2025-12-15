/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
  name: 'tribute',

  rules: {
    source_file: $ => repeat($._item),

    _item: $ => $.function_definition,

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
      $.case_expression,
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
      $.block  // { expr } for grouping
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

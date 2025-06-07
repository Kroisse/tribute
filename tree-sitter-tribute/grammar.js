module.exports = grammar({
  name: 'tribute',

  rules: {
    source_file: $ => repeat($._expression),

    _expression: $ => choice(
      $.number,
      $.string,
      $.identifier,
      $.list
    ),

    number: $ => /-?\d+/,

    string: $ => /"([^"\\]|\\.)*"/,

    identifier: $ => /[a-zA-Z_+\-*\/][a-zA-Z0-9_+\-*\/]*/,

    list: $ => seq(
      '(',
      repeat($._expression),
      ')'
    ),

    // Comments
    comment: $ => token(seq(
      '//',
      /.*/
    ))
  },

  extras: $ => [
    /\s/,
    $.comment
  ]
});
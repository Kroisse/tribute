; highlights.scm for Tribute language

; Keywords
[
  (keyword_fn)
  (keyword_let)
  (keyword_case)
  (keyword_struct)
  (keyword_enum)
  (keyword_ability)
  (keyword_const)
  (keyword_pub)
  (keyword_use)
  (keyword_mod)
  (keyword_if)
  (keyword_handle)
  (keyword_as)
] @keyword

; Boolean and nil literals (keyword-like constants)
[
  (keyword_true)
  (keyword_false)
  (keyword_nil)
] @constant.builtin

; Types (PascalCase)
(type_identifier) @type

; Type variables (lowercase in type position)
(type_variable) @type

; Function types - fn keyword in type position
(function_type
  (keyword_fn) @type.builtin)

; Ability items in function types
(ability_item
  (type_identifier) @type)

; Function definitions
(function_definition
  name: (identifier) @function)

; Ability operations
(ability_operation
  name: (identifier) @function)

; Function calls
(call_expression
  function: (identifier) @function.call)

(call_expression
  function: (path_expression) @function.call)

; Method calls (UFCS)
(method_call_expression
  method: (method_path) @function.method)

; Parameters
(parameter
  name: (identifier) @variable.parameter)

(typed_parameter
  name: (identifier) @variable.parameter)

; Struct fields
(struct_field
  name: (identifier) @property)

; Record fields
(record_field
  name: (identifier) @property)

; Pattern fields
(pattern_field
  name: (identifier) @property)

; Enum variants (types)
(enum_variant
  name: (type_identifier) @constructor)

; Constructor patterns
(constructor_pattern
  name: (type_identifier) @constructor)

; Numbers
(nat_literal) @number
(int_literal) @number
(float_literal) @number

; Strings
(string) @string
(raw_string) @string
(raw_interpolated_string) @string
(multiline_string) @string
(bytes_string) @string
(raw_bytes) @string
(raw_interpolated_bytes) @string
(multiline_bytes) @string
(string_segment) @string
(bytes_segment) @string
(raw_interpolated_string_segment) @string
(raw_interpolated_bytes_segment) @string
(multiline_string_segment) @string
(multiline_bytes_segment) @string

; String interpolation
(interpolation
  "{" @punctuation.special
  "}" @punctuation.special)

(multiline_interpolation
  "{" @punctuation.special
  "}" @punctuation.special)

(bytes_interpolation
  "{" @punctuation.special
  "}" @punctuation.special)

(multiline_bytes_interpolation
  "{" @punctuation.special
  "}" @punctuation.special)

; Runes (character literals)
(rune) @character

; Comments
(line_comment) @comment
(block_comment) @comment

; Doc comments
(line_doc_comment) @comment.doc
(block_doc_comment) @comment.doc

; Operators
[
  "+"
  "-"
  "*"
  "/"
  "%"
  "<>"
  "=="
  "!="
  "<"
  ">"
  "<="
  ">="
  "&&"
  "||"
  "="
  "->"
] @operator

(operator) @operator
(qualified_operator) @operator

; Spread operator
(spread) @operator

; Punctuation - brackets
[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

; Punctuation - delimiters
[
  ","
  ":"
  "::"
  "."
  "#"
] @punctuation.delimiter

; Identifiers (variables)
(identifier) @variable

; Path segments
(path_segment) @variable

; Wildcards
(wildcard_pattern) @variable.builtin

; Spread operator
(spread) @operator

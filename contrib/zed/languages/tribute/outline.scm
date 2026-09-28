; Modules
(mod_declaration
  (visibility_marker)? @context
  (keyword_mod) @context
  name: (_) @name) @item

; Functions
(regular_function
  (visibility_marker)? @context
  (keyword_fn) @context
  name: (_) @name) @item

(extern_function
  (visibility_marker)? @context
  (extern_marker) @context
  (keyword_fn) @context
  name: (_) @name) @item

; Constants
(const_declaration
  (visibility_marker)? @context
  (keyword_const) @context
  name: (identifier) @name) @item

; Structs
(struct_declaration
  (visibility_marker)? @context
  (keyword_struct) @context
  name: (type_identifier) @name) @item

; Enums
(enum_declaration
  (visibility_marker)? @context
  (keyword_enum) @context
  name: (type_identifier) @name) @item

; Abilities
(ability_declaration
  (visibility_marker)? @context
  (keyword_ability) @context
  name: (type_identifier) @name) @item

; Ability operations
(ability_operation
  kind: (_) @context
  name: (identifier) @name) @item

; Struct fields
(struct_field
  name: (identifier) @name) @item

; Enum variants
(enum_variant
  name: (type_identifier) @name) @item

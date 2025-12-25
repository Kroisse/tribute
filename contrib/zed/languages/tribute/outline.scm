; Modules
(mod_declaration
  (visibility_marker)? @context
  (keyword_mod) @context
  name: (_) @name) @item

; Functions
(function_definition
  (visibility_marker)? @context
  (keyword_fn) @context
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
  (keyword_fn) @context
  name: (identifier) @name) @item

; Struct fields
(struct_field
  name: (identifier) @name) @item

; Enum variants
(enum_variant
  name: (type_identifier) @name) @item

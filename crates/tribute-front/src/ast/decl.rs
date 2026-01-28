//! Declaration types for the AST.
//!
//! Declarations are top-level definitions in a module:
//! - Functions: `fn foo(x: Int) -> Int { ... }`
//! - Structs: `struct Point { x: Int, y: Int }`
//! - Enums: `enum Option(a) { Some(a), None }`
//! - Abilities: `ability State(s) { get() -> s, put(s) -> () }`
//! - Imports: `use std::collections::List`
//! - Modules: `mod math { ... }`

use trunk_ir::Symbol;

use super::expr::Expr;
use super::node_id::NodeId;
use super::types::TypeAnnotation;

/// A module containing declarations.
///
/// This is the top-level AST node for a source file.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct Module<V>
where
    V: salsa::Update,
{
    /// Node ID for span lookup.
    pub id: NodeId,
    /// The module name (derived from file path or explicit module declaration).
    pub name: Option<Symbol>,
    /// The declarations in the module.
    pub decls: Vec<Decl<V>>,
}

impl<V> Module<V>
where
    V: salsa::Update,
{
    /// Create a new module.
    pub fn new(id: NodeId, name: Option<Symbol>, decls: Vec<Decl<V>>) -> Self {
        Self { id, name, decls }
    }
}

/// A declaration in a module.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum Decl<V>
where
    V: salsa::Update,
{
    /// Function declaration.
    Function(FuncDecl<V>),
    /// Extern function declaration (no body).
    ExternFunction(ExternFuncDecl),
    /// Struct declaration.
    Struct(StructDecl),
    /// Enum declaration.
    Enum(EnumDecl),
    /// Ability declaration.
    Ability(AbilityDecl),
    /// Import declaration.
    Use(UseDecl),
    /// Inline module declaration.
    Module(ModuleDecl<V>),
}

/// Extern function declaration: `extern "abi" fn name(params) -> ReturnType`
///
/// Unlike `FuncDecl`, this has no body — the implementation is provided externally.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct ExternFuncDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Whether this function is public.
    pub is_pub: bool,
    /// Function name.
    pub name: Symbol,
    /// ABI identifier (e.g., `intrinsic`). Defaults to `C`.
    pub abi: Symbol,
    /// Function parameters.
    pub params: Vec<ParamDecl>,
    /// Return type annotation (optional).
    pub return_ty: Option<TypeAnnotation>,
}

/// Function declaration: `fn name(params) -> ReturnType { body }`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct FuncDecl<V>
where
    V: salsa::Update,
{
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Whether this function is public.
    pub is_pub: bool,
    /// Function name.
    pub name: Symbol,
    /// Type parameters for generic functions.
    pub type_params: Vec<TypeParamDecl>,
    /// Function parameters.
    pub params: Vec<ParamDecl>,
    /// Return type annotation (optional, can be inferred).
    pub return_ty: Option<TypeAnnotation>,
    /// Effect annotation (optional).
    pub effects: Option<Vec<TypeAnnotation>>,
    /// Function body.
    pub body: Expr<V>,
}

/// Type parameter declaration: `a`, `T: Eq`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct TypeParamDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Parameter name.
    pub name: Symbol,
    /// Optional bounds (trait/ability constraints).
    pub bounds: Vec<TypeAnnotation>,
}

/// Parameter declaration in a function.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct ParamDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Parameter name.
    pub name: Symbol,
    /// Type annotation (optional).
    pub ty: Option<TypeAnnotation>,
    /// Local ID assigned during name resolution (None before resolution).
    pub local_id: Option<super::phases::LocalId>,
}

/// Struct declaration: `struct Name { fields }`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct StructDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Whether this struct is public.
    pub is_pub: bool,
    /// Struct name.
    pub name: Symbol,
    /// Type parameters.
    pub type_params: Vec<TypeParamDecl>,
    /// Struct fields.
    pub fields: Vec<FieldDecl>,
}

/// Field declaration in a struct or enum variant.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct FieldDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Whether this field is public (for struct fields).
    pub is_pub: bool,
    /// Field name (None for positional fields in tuple-like variants).
    pub name: Option<Symbol>,
    /// Field type.
    pub ty: TypeAnnotation,
}

/// Enum declaration: `enum Name { Variants }`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct EnumDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Whether this enum is public.
    pub is_pub: bool,
    /// Enum name.
    pub name: Symbol,
    /// Type parameters.
    pub type_params: Vec<TypeParamDecl>,
    /// Enum variants.
    pub variants: Vec<VariantDecl>,
}

/// Variant declaration in an enum.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct VariantDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Variant name.
    pub name: Symbol,
    /// Variant fields (empty for unit variants like `None`).
    pub fields: Vec<FieldDecl>,
}

/// Ability declaration: `ability Name { operations }`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct AbilityDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Whether this ability is public.
    pub is_pub: bool,
    /// Ability name.
    pub name: Symbol,
    /// Type parameters.
    pub type_params: Vec<TypeParamDecl>,
    /// Ability operations.
    pub operations: Vec<OpDecl>,
}

/// Operation declaration in an ability.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct OpDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Operation name.
    pub name: Symbol,
    /// Operation parameters.
    pub params: Vec<ParamDecl>,
    /// Return type.
    pub return_ty: TypeAnnotation,
}

/// Import declaration: `use path::to::item` or `use path::to::item as alias`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct UseDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Whether this import is public (re-export).
    pub is_pub: bool,
    /// The import path.
    pub path: Vec<Symbol>,
    /// Optional alias.
    pub alias: Option<Symbol>,
}

/// Inline module declaration: `mod name { ... }`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct ModuleDecl<V>
where
    V: salsa::Update,
{
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Module name.
    pub name: Symbol,
    /// Whether this module is public.
    pub is_pub: bool,
    /// Module body (declarations). None for external modules (file-based).
    pub body: Option<Vec<Decl<V>>>,
}

// ============================================================================
// Phase-specific type aliases
// ============================================================================

use super::phases::{ResolvedRef, TypedRef, UnresolvedName};

/// Module after parsing.
pub type ParsedModule = Module<UnresolvedName>;

/// Module after name resolution.
pub type ResolvedModule<'db> = Module<ResolvedRef<'db>>;

/// Module after type checking.
pub type TypedModule<'db> = Module<TypedRef<'db>>;

/// Declaration after parsing.
pub type ParsedDecl = Decl<UnresolvedName>;

/// Declaration after name resolution.
pub type ResolvedDecl<'db> = Decl<ResolvedRef<'db>>;

/// Declaration after type checking.
pub type TypedDecl<'db> = Decl<TypedRef<'db>>;

/// Function declaration after parsing.
pub type ParsedFuncDecl = FuncDecl<UnresolvedName>;

/// Function declaration after name resolution.
pub type ResolvedFuncDecl<'db> = FuncDecl<ResolvedRef<'db>>;

/// Function declaration after type checking.
pub type TypedFuncDecl<'db> = FuncDecl<TypedRef<'db>>;

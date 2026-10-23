//! Declaration types for the AST.
//!
//! Declarations are top-level definitions in a module:
//! - Functions: `fn foo(x: Int) -> Int { ... }`
//! - Structs: `struct Point { x: Int, y: Int }`
//! - Enums: `enum Option(a) { Some(a), None }`
//! - Abilities: `ability State(s) { get() -> s, put(s) -> () }`
//! - Imports: `use std::collections::List`
//! - Modules: `mod math { ... }`

use trunk_ir::{Symbol, SymbolPath};

use super::expr::Expr;
use super::node_id::NodeId;
use super::types::TypeAnnotation;

/// A module containing declarations.
///
/// This is the top-level AST node for a source file.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct Module<V> {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// The module name (derived from file path or explicit module declaration).
    pub name: Option<Symbol>,
    /// The items in the module.
    pub decls: Vec<ModuleItem<V>>,
}

impl<V> Module<V> {
    /// Create a new module.
    pub fn new(
        id: NodeId,
        name: Option<Symbol>,
        decls: impl IntoIterator<Item = impl Into<ModuleItem<V>>>,
    ) -> Self {
        Self {
            id,
            name,
            decls: decls.into_iter().map(Into::into).collect(),
        }
    }
}

/// An item of a module body: a declaration or a nested module.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum ModuleItem<V> {
    /// A declaration.
    Decl(Decl<V>),
    /// Inline module declaration.
    Module(ModuleDecl<V>),
}

impl<V> From<Decl<V>> for ModuleItem<V> {
    fn from(decl: Decl<V>) -> Self {
        ModuleItem::Decl(decl)
    }
}

impl<V> From<ModuleDecl<V>> for ModuleItem<V> {
    fn from(module: ModuleDecl<V>) -> Self {
        ModuleItem::Module(module)
    }
}

/// A declaration. Modules are not declarations; they group them in the
/// source tree (see [`ModuleItem`]) and give them their path.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub enum Decl<V> {
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
}

/// Where a declaration is visible, from its visibility marker.
///
/// A declaration is visible in a module and that module's descendants: the
/// module that declares it, a module around that one that its marker names,
/// or, when public, every module of every package.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// No marker: the declaring module.
    #[default]
    Private,
    /// `pub(path)`: the module `path` names, as written. The path starts
    /// with a path keyword, as in `pub(pkg)`, `pub(super)`, `pub(self)`, and
    /// `pub(pkg::a::b)`.
    Restricted(Vec<Symbol>),
    /// `pub`: every package.
    Public,
}

impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Visibility::Private => f.write_str("private"),
            Visibility::Restricted(path) => {
                write!(
                    f,
                    "pub({})",
                    itertools::Itertools::format(path.iter(), "::")
                )
            }
            Visibility::Public => f.write_str("pub"),
        }
    }
}

/// Extern function declaration: `extern "abi" fn name(params) -> ReturnType`
///
/// Unlike `FuncDecl`, this has no body — the implementation is provided externally.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternFuncDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Where the declaration is visible.
    pub visibility: Visibility,
    /// Function name.
    pub name: Symbol,
    /// ABI identifier (e.g., `intrinsic`). Defaults to `C`.
    pub abi: Symbol,
    /// Function parameters.
    pub params: Vec<ParamDecl>,
    /// Return type annotation. Defaults to `Nil` when omitted.
    pub return_ty: TypeAnnotation,
}

/// Function declaration: `fn name(params) -> ReturnType { body }`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct FuncDecl<V> {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Where the declaration is visible.
    pub visibility: Visibility,
    /// Function name.
    pub name: Symbol,
    /// Type parameters for generic functions.
    pub type_params: Vec<TypeParamDecl>,
    /// Function parameters.
    pub params: Vec<ParamDecl>,
    /// Return type annotation (optional, can be inferred).
    pub return_ty: Option<TypeAnnotation>,
    /// Ability annotation (optional).
    pub effects: Option<Vec<TypeAnnotation>>,
    /// Function body.
    pub body: Expr<V>,
}

/// Type parameter declaration: `a`, `T: Eq`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeParamDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Parameter name.
    pub name: Symbol,
    /// Optional bounds (trait/ability constraints).
    pub bounds: Vec<TypeAnnotation>,
}

/// Parameter declaration in a function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Where the declaration is visible.
    pub visibility: Visibility,
    /// Struct name.
    pub name: Symbol,
    /// How many type declarations of the same name precede this one in its
    /// module. Zero unless the module redeclares the name.
    pub disambiguator: u32,
    /// Type parameters.
    pub type_params: Vec<TypeParamDecl>,
    /// Struct fields.
    pub fields: Vec<FieldDecl>,
}

/// The functions each named struct field `f` of `T` has in the namespace
/// `T::f`: the setter and the modifier.
pub const FIELD_LENS_FUNCTIONS: [&str; 2] = ["set", "modify"];

/// Field declaration in a struct or enum variant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Node ID whose span is the field name (same as `id` for positional fields).
    pub name_id: NodeId,
    /// Whether this field is public (for struct fields).
    pub is_pub: bool,
    /// Field name (None for positional fields in tuple-like variants).
    pub name: Option<Symbol>,
    /// Field type.
    pub ty: TypeAnnotation,
}

/// Enum declaration: `enum Name { Variants }`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnumDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Where the declaration is visible.
    pub visibility: Visibility,
    /// Enum name.
    pub name: Symbol,
    /// How many type declarations of the same name precede this one in its
    /// module. Zero unless the module redeclares the name.
    pub disambiguator: u32,
    /// Type parameters.
    pub type_params: Vec<TypeParamDecl>,
    /// Enum variants.
    pub variants: Vec<VariantDecl>,
}

/// Variant declaration in an enum.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VariantDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Variant name.
    pub name: Symbol,
    /// Variant fields (empty for unit variants like `None`).
    pub fields: Vec<FieldDecl>,
}

/// Ability declaration: `ability Name { operations }`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbilityDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Where the declaration is visible.
    pub visibility: Visibility,
    /// Ability name.
    pub name: Symbol,
    /// Type parameters.
    pub type_params: Vec<TypeParamDecl>,
    /// Ability operations.
    pub operations: Vec<OpDecl>,
}

/// Operation declaration in an ability.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Whether this is a `fn` (tail-resumptive) or `op` (general) operation.
    pub kind: OpDeclKind,
    /// Operation name.
    pub name: Symbol,
    /// Operation parameters.
    pub params: Vec<ParamDecl>,
    /// Return type.
    pub return_ty: TypeAnnotation,
}

/// The kind of ability operation declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpDeclKind {
    /// Tail-resumptive: handler returns a value that is used to resume.
    /// No continuation capture.
    Fn,
    /// General: handler uses `resume` keyword explicitly.
    /// Continuation is captured (unless `-> Never`).
    Op,
}

/// Import declaration: `use path::to::item` or `use path::to::item as alias`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UseDecl {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Where the imported name is visible; wider than private re-exports it.
    pub visibility: Visibility,
    /// The import path as written.
    pub path: Vec<Symbol>,
    /// Optional alias.
    pub alias: Option<Symbol>,
    /// The package-root path of what the import names, recorded by name
    /// resolution. `None` before resolution, or when the path names nothing.
    ///
    /// A path may be written relative to the enclosing inline module; later
    /// phases read this instead of reinterpreting `path`.
    pub target: Option<Vec<Symbol>>,
}

/// Inline module declaration: `mod name { ... }`
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct ModuleDecl<V> {
    /// Node ID for span lookup.
    pub id: NodeId,
    /// Module name.
    pub name: Symbol,
    /// Where the declaration is visible.
    pub visibility: Visibility,
    /// Module body (declarations). None for external modules (file-based).
    // Inline modules own their nested declarations; V is the only phase-dependent value.
    #[salsa_value(unsafe(prove(V: salsa::SalsaValue)))]
    pub body: Option<Vec<ModuleItem<V>>>,
}

/// A declaration with the path of the module that encloses it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct DeclEntry<V> {
    /// The enclosing module: the path of the declaration without its own
    /// name. A top-level declaration is in the root module.
    pub module: SymbolPath,
    /// The declaration.
    pub decl: Decl<V>,
}

/// The declarations of a module tree, each with its module path.
///
/// Entries are in a fixed order: the source declarations in source order,
/// as a preorder walk of the tree visits them, followed by any entries a
/// later phase appends.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct DeclTable<V> {
    /// Node ID of the module the table was built from.
    pub id: NodeId,
    /// The name of that module (derived from the file path).
    pub name: Option<Symbol>,
    /// The entries, in table order.
    pub entries: Vec<DeclEntry<V>>,
}

impl<V> DeclTable<V> {
    /// Flatten a module tree. A file-based module without an inline body
    /// contributes no entries.
    pub fn from_module(module: Module<V>) -> Self {
        fn flatten<V>(
            items: Vec<ModuleItem<V>>,
            module: &mut SymbolPath,
            entries: &mut Vec<DeclEntry<V>>,
        ) {
            for item in items {
                match item {
                    ModuleItem::Decl(decl) => entries.push(DeclEntry {
                        module: module.clone(),
                        decl,
                    }),
                    ModuleItem::Module(inner) => {
                        let Some(body) = inner.body else { continue };
                        module.push(inner.name);
                        flatten(body, module, entries);
                        module.pop();
                    }
                }
            }
        }

        let mut entries = Vec::new();
        flatten(module.decls, &mut SymbolPath::root(), &mut entries);
        Self {
            id: module.id,
            name: module.name,
            entries,
        }
    }

    /// The declarations, in table order.
    pub fn decls(&self) -> impl Iterator<Item = &Decl<V>> {
        self.entries.iter().map(|entry| &entry.decl)
    }

    /// Append `decl` to the table, in `module`.
    pub fn push(&mut self, module: SymbolPath, decl: Decl<V>) {
        self.entries.push(DeclEntry { module, decl });
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn structure(id: usize, name: &str) -> ModuleItem<u32> {
        Decl::Struct(StructDecl {
            id: NodeId::from_raw(id),
            visibility: Visibility::Private,
            name: Symbol::new(name),
            disambiguator: 0,
            type_params: vec![],
            fields: vec![],
        })
        .into()
    }

    fn module(id: usize, name: &str, body: Option<Vec<ModuleItem<u32>>>) -> ModuleItem<u32> {
        ModuleDecl {
            id: NodeId::from_raw(id),
            name: Symbol::new(name),
            visibility: Visibility::Private,
            body,
        }
        .into()
    }

    #[test]
    fn a_table_lists_declarations_in_source_order_with_their_modules() {
        let tree = Module::new(
            NodeId::from_raw(0),
            None,
            [
                structure(1, "A"),
                module(
                    2,
                    "outer",
                    Some(vec![
                        structure(3, "B"),
                        module(4, "inner", Some(vec![structure(5, "C")])),
                        structure(6, "D"),
                    ]),
                ),
                module(7, "file", None),
                structure(8, "E"),
            ],
        );

        let table = DeclTable::from_module(tree);

        assert_eq!(table.id, NodeId::from_raw(0));
        itertools::assert_equal(
            table.entries.iter().map(|entry| {
                let Decl::Struct(declaration) = &entry.decl else {
                    panic!("only structs were declared");
                };
                (entry.module.clone(), declaration.name.clone())
            }),
            [
                (SymbolPath::root(), Symbol::new("A")),
                (SymbolPath::new(["outer"]), Symbol::new("B")),
                (SymbolPath::new(["outer", "inner"]), Symbol::new("C")),
                (SymbolPath::new(["outer"]), Symbol::new("D")),
                (SymbolPath::root(), Symbol::new("E")),
            ],
        );
    }
}

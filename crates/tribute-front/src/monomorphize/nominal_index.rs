//! Source nominal declarations shared by collection and generation in one preparation.
use rustc_hash::FxHashMap as HashMap;
use trunk_ir::SymbolPath;

use crate::ast::{CtorId, Decl, DeclTable, EnumDecl, NodeId, StructDecl, TypeDefId, TypedRef};

pub(super) struct Constructor<'db> {
    pub node: NodeId,
    pub id: CtorId<'db>,
    pub fields: usize,
}

pub(super) enum NominalDeclaration<'ast> {
    Struct(&'ast StructDecl),
    Enum(&'ast EnumDecl),
}

pub(super) struct Declaration<'ast, 'db> {
    pub source: NominalDeclaration<'ast>,
    pub constructors: Vec<Constructor<'db>>,
}

impl Declaration<'_, '_> {
    pub fn params(&self) -> usize {
        match self.source {
            NominalDeclaration::Struct(s) => s.type_params.len(),
            NominalDeclaration::Enum(e) => e.type_params.len(),
        }
    }

    pub fn is_enum(&self) -> bool {
        matches!(self.source, NominalDeclaration::Enum(_))
    }
}

pub(super) struct NominalIndex<'ast, 'db> {
    pub module: NodeId,
    pub declarations: HashMap<TypeDefId<'db>, Declaration<'ast, 'db>>,
}

impl<'ast, 'db> NominalIndex<'ast, 'db> {
    /// The nominal declarations of `module` and of `dependencies`.
    pub fn new(
        db: &'db dyn salsa::Database,
        module: &'ast DeclTable<TypedRef<'db>>,
        dependencies: &[&'ast DeclTable<TypedRef<'db>>],
    ) -> Self {
        let mut index = Self {
            module: module.id,
            declarations: HashMap::default(),
        };
        for entry in dependencies
            .iter()
            .copied()
            .chain([module])
            .flat_map(|table| &table.entries)
        {
            index.collect(db, &entry.module, &entry.decl);
        }
        index
    }

    pub fn is_generic(&self, id: TypeDefId<'db>) -> bool {
        self.declarations
            .get(&id)
            .is_some_and(|declaration| declaration.params() != 0)
    }

    fn collect(
        &mut self,
        db: &'db dyn salsa::Database,
        prefix: &SymbolPath,
        decl: &'ast Decl<TypedRef<'db>>,
    ) {
        let (name, disambiguator, declaration) = match decl {
            Decl::Struct(s) => (
                s.name.clone(),
                s.disambiguator,
                Declaration {
                    source: NominalDeclaration::Struct(s),
                    constructors: vec![Constructor {
                        node: s.id,
                        id: CtorId::new(db, prefix.child(s.name.clone())),
                        fields: s.fields.len(),
                    }],
                },
            ),
            Decl::Enum(e) => (
                e.name.clone(),
                e.disambiguator,
                Declaration {
                    source: NominalDeclaration::Enum(e),
                    constructors: e
                        .variants
                        .iter()
                        .map(|v| Constructor {
                            node: v.id,
                            id: CtorId::new(db, prefix.child(e.name.clone()).child(v.name.clone())),
                            fields: v.fields.len(),
                        })
                        .collect(),
                },
            ),
            _ => return,
        };
        let id = TypeDefId::declared(db, prefix.child(name.clone()), disambiguator);
        self.declarations.insert(id, declaration);
    }
}

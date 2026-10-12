//! Source nominal declarations shared by collection and generation in one preparation.
use rustc_hash::FxHashMap as HashMap;
use trunk_ir::SymbolPath;

use crate::ast::{CtorId, Decl, EnumDecl, Module, NodeId, StructDecl, TypeDefId, TypedRef};

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
    /// The module that declares it.
    pub module: SymbolPath,
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
    pub fn new(db: &'db dyn salsa::Database, module: &'ast Module<TypedRef<'db>>) -> Self {
        let mut index = Self {
            module: module.id,
            declarations: HashMap::default(),
        };
        index.collect(db, &module.decls, &mut SymbolPath::root());
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
        decls: &'ast [Decl<TypedRef<'db>>],
        prefix: &mut SymbolPath,
    ) {
        for decl in decls {
            let (name, node, declaration) = match decl {
                Decl::Struct(s) => (
                    s.name.clone(),
                    s.id,
                    Declaration {
                        source: NominalDeclaration::Struct(s),
                        module: prefix.clone(),
                        constructors: vec![Constructor {
                            node: s.id,
                            id: CtorId::new(db, crate::qualified_symbol(prefix, &s.name)),
                            fields: s.fields.len(),
                        }],
                    },
                ),
                Decl::Enum(e) => (
                    e.name.clone(),
                    e.id,
                    Declaration {
                        source: NominalDeclaration::Enum(e),
                        module: prefix.clone(),
                        constructors: e
                            .variants
                            .iter()
                            .map(|v| Constructor {
                                node: v.id,
                                id: CtorId::new(db, crate::qualified_symbol(prefix, &v.name)),
                                fields: v.fields.len(),
                            })
                            .collect(),
                    },
                ),
                Decl::Module(m) => {
                    if let Some(body) = &m.body {
                        prefix.push(m.name.clone());
                        self.collect(db, body, prefix);
                        prefix.pop();
                    }
                    continue;
                }
                _ => continue,
            };
            let id = TypeDefId::source(db, crate::qualified_symbol(prefix, &name), node);
            self.declarations.insert(id, declaration);
        }
    }
}

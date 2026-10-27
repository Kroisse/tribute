//! Exclusion of generic templates from the specialization output
//! (`new-plans/generics.md`, "특수화 결과").
//!
//! A function or extern function with type parameters is a template. Its
//! nodes, its metadata, and its scheme leave the output with it. Struct and
//! enum templates stay: an instance whose arguments hold an open ability row
//! is not specialized and is represented by its template.

use rustc_hash::FxHashSet as HashSet;

use trunk_ir::SymbolPath;

use crate::ast::visit::Visit;
use crate::ast::{Decl, DeclTable, NodeId, TypeScheme, TypedRef};
use crate::typeck::ExpressionTypeMetadata;

/// Remove the function and extern function templates, with their schemes
/// and the expression metadata of their nodes, returning how many were
/// removed. Runs before nominal collection, so a nominal instance only a
/// template mentions is not generated. A generic intrinsic keeps its
/// identity: a specialization that continues this one instantiates it at its
/// own types.
pub(super) fn remove_function_templates<'db>(
    db: &'db dyn salsa::Database,
    module: &mut DeclTable<TypedRef<'db>>,
    function_types: &mut Vec<(SymbolPath, TypeScheme<'db>)>,
    expressions: &mut ExpressionTypeMetadata<'db>,
) -> usize {
    let generic: HashSet<SymbolPath> = function_types
        .iter()
        .filter(|(_, scheme)| !scheme.type_params(db).is_empty())
        .map(|(path, _)| path.clone())
        .collect();
    let mut removed_paths = HashSet::default();
    let mut removed = DeclNodes::default();
    module.entries.retain(|entry| {
        let name = match &entry.decl {
            Decl::Function(function) => &function.name,
            Decl::ExternFunction(function) => &function.name,
            _ => return true,
        };
        let path = entry.module.child(name.clone());
        if !generic.contains(&path) {
            return true;
        }
        removed.visit_decl(&entry.decl);
        removed_paths.insert(path);
        false
    });
    let removed = removed.0;
    function_types.retain(|(path, _)| !removed_paths.contains(path));
    expressions.retain_nodes(|node| !removed.contains(node));
    removed_paths.len()
}

/// Every node of the declarations it visits, including those of bodiless
/// and nominal declarations.
#[derive(Default)]
pub(super) struct DeclNodes(pub(super) HashSet<NodeId>);

impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for DeclNodes {
    fn visit_decl(&mut self, decl: &'ast Decl<TypedRef<'db>>) {
        match decl {
            Decl::ExternFunction(function) => {
                self.0.insert(function.id);
            }
            Decl::Struct(declaration) => {
                self.0.insert(declaration.id);
                self.0
                    .extend(declaration.fields.iter().map(|field| field.id));
            }
            Decl::Enum(declaration) => {
                self.0.insert(declaration.id);
                for variant in &declaration.variants {
                    self.0.insert(variant.id);
                    self.0.extend(variant.fields.iter().map(|field| field.id));
                }
            }
            _ => {}
        }
        crate::ast::visit::walk_decl(self, decl);
    }

    fn visit_node_id(&mut self, id: NodeId) {
        self.0.insert(id);
    }
}

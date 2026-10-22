//! The package a module tree is resolved as.

use trunk_ir::{Symbol, SymbolPath};

use super::LIBRARY_PACKAGE;

/// The package whose declarations are being resolved, and where its root
/// is in their module paths.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Package {
    /// The package's name, for a library package another one depends on.
    /// `None` for the package being compiled.
    name: Option<Symbol>,
    /// How many leading segments of a module path name the package root.
    root_depth: usize,
}

impl Package {
    /// The package being compiled. Its module paths start at its root.
    pub fn compiled() -> Self {
        Self {
            name: None,
            root_depth: 0,
        }
    }

    /// The library package `name`, whose module tree is one module of that
    /// name; its module paths start with the name.
    pub fn library(name: Symbol) -> Self {
        Self {
            name: Some(name),
            root_depth: 1,
        }
    }

    /// The library package the prelude is the root of.
    pub fn std() -> Self {
        Self::library(Symbol::new(LIBRARY_PACKAGE))
    }

    /// Whether this is a library package.
    pub fn is_library(&self) -> bool {
        self.name.is_some()
    }

    /// The package root's path, the leading segments of `module`.
    pub fn root<'a>(&self, module: &'a [Symbol]) -> &'a [Symbol] {
        &module[..self.root_depth.min(module.len())]
    }

    /// Whether `module` is the package root.
    pub fn is_root(&self, module: &[Symbol]) -> bool {
        module.len() <= self.root_depth
    }

    /// The identity path of the module at `module`, a path of this
    /// package's module tree.
    ///
    /// A library package's paths start with its name. The package being
    /// compiled omits its own, unless the path would then start with a
    /// library package's name: such a path starts with `pkg`, which names no
    /// module.
    pub fn identity_path(&self, module: &[Symbol]) -> SymbolPath {
        let qualified =
            !self.is_library() && module.first().is_some_and(|first| first == LIBRARY_PACKAGE);
        SymbolPath::new(
            qualified
                .then(|| Symbol::new("pkg"))
                .into_iter()
                .chain(module.iter().cloned()),
        )
    }
}

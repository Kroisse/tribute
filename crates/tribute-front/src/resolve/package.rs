//! The package a module tree is resolved as.

use trunk_ir::{Symbol, SymbolPath};

use super::LIBRARY_PACKAGE;

/// A package: the module tree of one root module. Name resolution reads
/// the tree's module paths from that root.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct Package {
    /// The package's name, for a library package another one depends on.
    /// `None` for the package being compiled.
    name: Option<Symbol>,
}

impl Package {
    /// The package being compiled, also the default.
    pub fn compiled() -> Self {
        Self { name: None }
    }

    /// The library package `name`.
    pub fn library(name: Symbol) -> Self {
        Self { name: Some(name) }
    }

    /// The library package the prelude is the root of.
    pub fn std() -> Self {
        Self::library(Symbol::new(LIBRARY_PACKAGE))
    }

    /// Whether this is a library package.
    pub fn is_library(&self) -> bool {
        self.name.is_some()
    }

    /// The identity path of `module`, a path of this package's module tree
    /// from its root.
    ///
    /// A library package's paths start with its name. The package being
    /// compiled omits its own, unless the path would then start with a
    /// library package's name: such a path starts with `pkg`, which names no
    /// module.
    pub fn identity_path(&self, module: &[Symbol]) -> SymbolPath {
        let package = match &self.name {
            Some(name) => Some(name.clone()),
            None => module
                .first()
                .is_some_and(|first| first == LIBRARY_PACKAGE)
                .then(|| Symbol::new("pkg")),
        };
        SymbolPath::new(package.into_iter().chain(module.iter().cloned()))
    }

    /// The module path, from this package's root, of the identity path
    /// `identity`, if it is one of this package's paths.
    pub fn module_path<'a>(&self, identity: &'a [Symbol]) -> Option<&'a [Symbol]> {
        match (&self.name, identity.split_first()) {
            (Some(name), Some((first, rest))) if first == name => Some(rest),
            (Some(_), _) => None,
            (None, Some((first, rest)))
                if first == "pkg" && rest.first().is_some_and(|f| f == LIBRARY_PACKAGE) =>
            {
                Some(rest)
            }
            (None, _) => Some(identity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(text: &str) -> Vec<Symbol> {
        text.split("::").map(Symbol::new).collect()
    }

    #[test]
    fn identity_paths_start_with_the_library_name() {
        let std = Package::std();
        assert_eq!(
            std.identity_path(&path("io::print")),
            path("std::io::print")[..]
        );
        assert_eq!(
            std.module_path(&path("std::io::print")),
            Some(&path("io::print")[..])
        );
        assert_eq!(std.module_path(&path("main")), None);
    }

    #[test]
    fn the_compiled_package_omits_its_name() {
        let compiled = Package::compiled();
        assert_eq!(
            compiled.identity_path(&path("geometry::area")),
            path("geometry::area")[..]
        );
        assert_eq!(
            compiled.identity_path(&path("std::f")),
            path("pkg::std::f")[..]
        );
        assert_eq!(
            compiled.module_path(&path("pkg::std::f")),
            Some(&path("std::f")[..])
        );
        assert_eq!(
            compiled.module_path(&path("geometry::area")),
            Some(&path("geometry::area")[..])
        );
    }
}

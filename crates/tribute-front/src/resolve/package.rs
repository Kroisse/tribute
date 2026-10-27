//! The name of the package a module tree belongs to, and the two ways a
//! path names a place in a package: by identity, and from its root.

use trunk_ir::{Symbol, SymbolPath};

use super::LIBRARY_PACKAGE;

/// Which package a module tree is: a library package, by its name, or the
/// root package of an executable build. Every module path of the tree is
/// read from its root; this name spells its identity paths.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct PackageName {
    /// The library package's name. `None` for the root package of an
    /// executable build, whose identity paths omit it.
    name: Option<Symbol>,
}

impl PackageName {
    /// The root package of an executable build, also the default.
    pub fn executable_root() -> Self {
        Self { name: None }
    }

    /// The library package `name`, whether it is compiled alone or as a
    /// dependency.
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

    /// Whether this is package `std`.
    pub fn is_std(&self) -> bool {
        self.name
            .as_ref()
            .is_some_and(|name| name == LIBRARY_PACKAGE)
    }

    /// The identity path of `module`, a path of this package's module tree
    /// from its root.
    ///
    /// A library package's paths start with its name, wherever it is
    /// compiled. The root package of an executable build omits its own,
    /// unless the path would then start with a library package's name: such
    /// a path starts with `pkg`, which names no module.
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

    /// The identity path of the item at `module`, a path of this package's
    /// module tree from its root, spelled as text.
    pub fn identity_path_of(&self, module: &[&str]) -> SymbolPath {
        let module: Vec<Symbol> = module.iter().copied().map(Symbol::new).collect();
        self.identity_path(&module)
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
            // A library package's paths start with its name.
            (None, Some((first, _))) if first == LIBRARY_PACKAGE => None,
            (None, _) => Some(identity),
        }
    }

    /// The location of `path`, a path from this package's root.
    pub fn at(&self, path: SymbolPath) -> Location {
        Location {
            package: self.clone(),
            path,
        }
    }
}

/// A place in a package's module tree: the package, and the path from its
/// root. A path from the root names a place only together with its
/// package, since a module of one package may share its name with another
/// package.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct Location {
    pub package: PackageName,
    pub path: SymbolPath,
}

impl Location {
    /// The location `name` names in the module or namespace here.
    pub fn child(&self, name: Symbol) -> Location {
        self.package.at(self.path.child(name))
    }

    /// The location `path` continues to from here.
    pub fn join<'a>(&self, path: impl IntoIterator<Item = &'a Symbol>) -> Location {
        let mut joined = self.path.clone();
        for segment in path {
            joined.push(segment.clone());
        }
        self.package.at(joined)
    }

    /// The namespace that holds the entry here, and the entry's name; `None`
    /// at a package root.
    pub fn split_last(&self) -> Option<(&Symbol, Location)> {
        let (last, namespace) = self.path.split_last()?;
        Some((
            last,
            self.package.at(SymbolPath::new(namespace.iter().cloned())),
        ))
    }

    /// The identity path of the place.
    pub fn identity(&self) -> IdentityPath {
        IdentityPath(self.package.identity_path(&self.path))
    }
}

/// The path that identifies a declaration or module across packages: a
/// library package's paths start with its name, an executable root
/// package's omit theirs (`std::Option`, `geometry::area`, `pkg::std::f`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, salsa::SalsaValue)]
pub struct IdentityPath(SymbolPath);

impl IdentityPath {
    pub fn new(path: SymbolPath) -> Self {
        Self(path)
    }
}

impl std::ops::Deref for IdentityPath {
    type Target = [Symbol];

    fn deref(&self) -> &[Symbol] {
        &self.0
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
        let std = PackageName::std();
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
    fn any_library_keeps_its_name() {
        let lib = PackageName::library(Symbol::new("lib"));
        assert!(!lib.is_std());
        assert_eq!(
            lib.identity_path(&path("geometry::area")),
            path("lib::geometry::area")[..]
        );
        // A library's own module `std` needs no `pkg` prefix.
        assert_eq!(lib.identity_path(&path("std::f")), path("lib::std::f")[..]);
        assert_eq!(
            lib.module_path(&path("lib::geometry::area")),
            Some(&path("geometry::area")[..])
        );
        assert_eq!(lib.module_path(&path("geometry::area")), None);
        assert_eq!(lib.module_path(&path("std::io::print")), None);
    }

    #[test]
    fn an_executable_root_omits_its_name() {
        let root = PackageName::executable_root();
        assert_eq!(
            root.identity_path(&path("geometry::area")),
            path("geometry::area")[..]
        );
        assert_eq!(root.identity_path(&path("std::f")), path("pkg::std::f")[..]);
        assert_eq!(
            root.module_path(&path("pkg::std::f")),
            Some(&path("std::f")[..])
        );
        assert_eq!(
            root.module_path(&path("geometry::area")),
            Some(&path("geometry::area")[..])
        );
        assert_eq!(root.module_path(&path("std::io::print")), None);
    }
}

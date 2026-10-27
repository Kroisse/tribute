//! The name of the package a module tree belongs to, and the two ways a
//! path names a place in a package: by identity, and from its root.

use itertools::Itertools;
use trunk_ir::{Symbol, SymbolPath};

use super::STD_PACKAGE;

/// Which package a module tree is: a library package, by its name, or the
/// root package of an executable build. Every module path of the tree is
/// read from its root; this name spells its identity paths.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, salsa::SalsaValue)]
pub struct PackageName {
    /// The library package's name. `None` for the root package of an
    /// executable build, whose identity paths start with `pkg` instead.
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
        Self::library(Symbol::new(STD_PACKAGE))
    }

    /// Whether this is a library package.
    pub fn is_library(&self) -> bool {
        self.name.is_some()
    }

    /// Whether this is package `std`.
    pub fn is_std(&self) -> bool {
        self.name.as_ref().is_some_and(|name| name == STD_PACKAGE)
    }

    /// The identity path of `module`, a path of this package's module tree
    /// from its root.
    ///
    /// A library package's paths start with its name, wherever it is
    /// compiled. The root package of an executable build has no name a
    /// dependent could use, and its paths start with `pkg`, which names no
    /// package or module.
    pub fn identity_path(&self, module: &[Symbol]) -> SymbolPath {
        let package = self.name.clone().unwrap_or_else(|| Symbol::new("pkg"));
        SymbolPath::new(std::iter::once(package).chain(module.iter().cloned()))
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
        let (first, rest) = identity.split_first()?;
        let matches = match &self.name {
            Some(name) => first == name,
            None => first == "pkg",
        };
        matches.then_some(rest)
    }

    /// The location of `path`, a path from this package's root.
    pub fn at(&self, path: SymbolPath) -> Location {
        Location {
            package: self.clone(),
            path,
        }
    }
}

/// How diagnostics spell identity paths while one package is compiled: its
/// own declarations without its package component, unless the rest would
/// then start with a dependency's name (`new-plans/modules.md`, "진단의 이름
/// 표기").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameDisplay {
    own: Option<Symbol>,
    dependencies: Vec<Symbol>,
}

impl NameDisplay {
    /// The spelling while compiling `package`, which depends on the packages
    /// named `dependencies`.
    pub fn new<'a>(
        package: &PackageName,
        dependencies: impl IntoIterator<Item = &'a Symbol>,
    ) -> Self {
        let own = package
            .identity_path(&[])
            .first()
            .expect("a package's root has its package component")
            .clone();
        Self {
            own: Some(own),
            dependencies: dependencies.into_iter().cloned().collect(),
        }
    }

    /// The spelling that keeps every identity path whole, for comparing
    /// declarations by what a diagnostic would print.
    pub fn identity() -> Self {
        Self {
            own: None,
            dependencies: Vec::new(),
        }
    }

    /// The spelling of `name`, the text of an identity path.
    pub fn text<'a>(&self, name: &'a str) -> &'a str {
        let Some(rest) = self.own.as_ref().and_then(|own| {
            own.with_str(|own| {
                name.strip_prefix(own)
                    .and_then(|rest| rest.strip_prefix("::"))
            })
        }) else {
            return name;
        };
        let first = rest.split("::").next().unwrap_or(rest);
        if self
            .dependencies
            .iter()
            .any(|dependency| *dependency == *first)
        {
            name
        } else {
            rest
        }
    }

    /// The spelling of the identity path whose joined text is `name`.
    pub fn symbol(&self, name: &Symbol) -> String {
        name.with_str(|text| self.text(text).to_owned())
    }

    /// The spelling of the identity path `path`.
    pub fn path(&self, path: &[Symbol]) -> String {
        let text = path.iter().join("::");
        self.text(&text).to_owned()
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
/// package's start with `pkg` (`std::Option`,
/// `pkg::geometry::area`, `pkg::std::f`).
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
    fn diagnostics_spell_the_compiled_package_without_its_component() {
        let std = Symbol::new(STD_PACKAGE);
        let root = NameDisplay::new(&PackageName::executable_root(), [&std]);
        assert_eq!(root.text("pkg::geometry::area"), "geometry::area");
        assert_eq!(root.text("pkg::stdlib::f"), "stdlib::f");
        assert_eq!(root.text("pkg::std::f"), "pkg::std::f");
        assert_eq!(root.text("std::Option"), "std::Option");
        assert_eq!(root.path(&path("pkg::main")), "main");

        let lib = NameDisplay::new(&PackageName::library(Symbol::new("lib")), [&std]);
        assert_eq!(lib.text("lib::area"), "area");
        assert_eq!(lib.text("library::area"), "library::area");
        assert_eq!(lib.text("pkg::area"), "pkg::area");
        assert_eq!(lib.text("std::Option"), "std::Option");

        assert_eq!(NameDisplay::identity().text("pkg::a"), "pkg::a");

        let std_itself = NameDisplay::new(&PackageName::std(), []);
        assert_eq!(std_itself.text("std::Option"), "Option");
    }

    #[test]
    fn an_executable_root_starts_with_pkg() {
        let root = PackageName::executable_root();
        assert_eq!(root.identity_path(&[]), path("pkg")[..]);
        assert_eq!(
            root.identity_path(&path("geometry::area")),
            path("pkg::geometry::area")[..]
        );
        assert_eq!(root.identity_path(&path("std::f")), path("pkg::std::f")[..]);
        assert_eq!(
            root.module_path(&path("pkg::std::f")),
            Some(&path("std::f")[..])
        );
        assert_eq!(
            root.module_path(&path("pkg::geometry::area")),
            Some(&path("geometry::area")[..])
        );
        assert_eq!(root.module_path(&path("geometry::area")), None);
        assert_eq!(root.module_path(&path("std::io::print")), None);
    }
}

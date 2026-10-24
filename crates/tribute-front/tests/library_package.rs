//! The prelude is the module `std::prelude::v0`: every module sees its items
//! and their namespaces under short names, and a user declaration or import
//! of the same name takes precedence.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "library.trb", text))
}

#[salsa_test]
fn builtin_list_namespace_is_a_path_start_everywhere(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    pub fn one() -> List(Nat) { List::prepend(1, []) }
}

fn main() -> Nil {
    let _ = List::prepend(2, m::one())
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn imports_may_start_from_a_library_namespace(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
use Option::None as Nothing

mod m {
    use Option::Some as Just

    pub fn one() -> Option(Nat) { Just(1) }
}

fn main() -> Nil {
    let _ = case m::one() {
        .Some(n) -> n
        Nothing -> 0
    }
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// A package-root import of a library name hides the library item, in
/// annotations as well as expressions.
#[salsa_test]
fn root_import_shadows_a_library_name(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod foo {
    pub enum Option { Nothing }
}

use foo::Option

fn f(o: Option) -> Nat {
    case o {
        foo::Option::Nothing -> 1
    }
}

fn main() -> Nil {
    let _ = f(foo::Option::Nothing)
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// An empty package-root module hides the library item of the same name,
/// although it defines no namespace entries.
#[salsa_test]
fn empty_root_module_shadows_a_library_name(db: &salsa::DatabaseImpl) {
    let call = |annotation: &str| {
        format!(
            "mod Option {{}}\n\nfn f(o: {annotation}) -> Nat {{ 0 }}\n\n\
             fn main() -> Nil {{\n    let _ = f(std::Option::Some(1))\n}}\n"
        )
    };
    assert_eq!(
        errors(db, &call("Option(Nat)")),
        ["type error at call site in function 'main': \
             expected `Option(Nat)`, found `std::Option(Nat)`"]
    );
    assert_eq!(
        errors(db, &call("Option::Option(Nat)")),
        ["unresolved name `Option::Option`"]
    );
    assert_eq!(
        errors(
            db,
            "mod Option {}\n\nfn main() -> Nil {\n    let _ = Option::Some(1)\n}\n"
        ),
        ["unresolved name `Option::Some`"]
    );
    assert_eq!(
        errors(db, "mod Option {}\n\nuse Option::Some\n"),
        ["unresolved import `Option::Some`"]
    );
}

/// Package `std`'s `pub(pkg)` items stay hidden behind the dependency path.
#[salsa_test]
fn a_package_internal_std_item_is_hidden(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn slots(s: std::runtime::Slots(Nat)) -> Nat { 0 }

fn main() -> Nil {}
"#,
    );
    assert_eq!(
        errors,
        ["`std::runtime::Slots` is not visible here: `std::runtime::Slots` is declared `pub(pkg)`"]
    );
}

/// A package-root path names the package's own items, not the prelude's.
#[salsa_test]
fn a_package_root_path_does_not_name_a_prelude_item(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn main() -> Nil {
    let _ = pkg::Option::Some(1)
}
"#,
    );
    assert_eq!(errors, ["unresolved name `pkg::Option::Some`"]);
}

/// A module the package root declares as `std` hides package `std` from the
/// paths of the root: `std::…` names only the module, in expressions and
/// annotations alike, while the prelude's names and namespaces still reach
/// the package.
#[salsa_test]
fn a_root_module_named_std_hides_the_package(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod std {
    pub fn answer() -> Nat { 42 }
}

fn mine() -> Nat { std::answer() }

fn theirs() -> Nat { std::String::len("a") }

fn annotated(o: std::Option(Nat)) -> Nat { 0 }

fn prelude() -> Option(List(Nat)) { Option::Some(List::prepend(1, [])) }
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved name `std::String::len`",
            "unresolved name `std::Option`",
        ]
    );
}

/// A module named `std` hides the package only where it is in scope: an
/// inline module does not see the package root's modules.
#[salsa_test]
fn a_root_module_named_std_is_not_in_scope_in_other_modules(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod std {
    pub fn answer() -> Nat { 42 }
}

mod m {
    use std::Option::Some

    pub fn wrapped(o: std::Option(Nat)) -> std::Option(Nat) { Some(std::String::len("a")) }
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// A `use` whose first segment names both a module of the package and
/// package `std` is ambiguous; `self::`/`pkg::` and `::` tell them apart.
#[salsa_test]
fn an_import_through_a_module_named_std_is_ambiguous(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod std {
    pub fn answer() -> Nat { 42 }
}

use std::answer
use pkg::std::answer as mine
use ::std::String::len

mod m {
    mod std {
        pub fn answer() -> Nat { 42 }
    }

    use std::answer
    use self::std::answer as mine
}

fn main() -> Nil { let _ = mine() + len("a") }
"#,
    );
    let ambiguous = |path: &str| {
        format!(
            "`use {path}` is ambiguous: `std` names both a name of this module and package \
             `std`; write `self::std` or `pkg::std` for the module's name, or `::std` for \
             the package"
        )
    };
    assert_eq!(errors, [ambiguous("std::answer"), ambiguous("std::answer")]);
}

/// A path with a leading `::` names a dependency package even where a
/// module of the same name hides it: in values, patterns, annotations,
/// ability rows, handler arms, and imports.
#[salsa_test]
fn a_leading_separator_reaches_a_shadowed_package(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
use ::std::io::{Io, print_line}

mod std {
    pub fn answer() -> Nat { 42 }
}

mod m {
    use ::std::Option::Some as Just

    pub fn one() -> ::std::Option(Nat) { Just(1) }
}

fn unwrap(o: ::std::Option(Nat)) -> Nat {
    case o {
        ::std::Option::Some(n) -> n
        ::std::Option::None -> std::answer()
    }
}

fn say() ->{::std::io::Io} Nil { ::std::io::print_line("hi") }

fn fail() ->{::std::abilities::Abort} Nat { ::std::abilities::Abort::abort() }

fn main() ->{Io} Nil {
    let _ = unwrap(::std::Option::Some(::std::String::len("a")))
    let _ = unwrap(m::one())
    let _ = handle fail() {
        do result { result }
        op ::std::abilities::Abort::abort() { 0 }
    }
    say()
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// A leading `::` must name a package the package depends on.
#[salsa_test]
fn a_leading_separator_names_only_dependencies(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
use ::foo::bar

fn f(x: ::nope::T) -> Nat { ::missing::g() }

fn g(x: ::std::DoesNotExist, y: pkg::Missing) -> Nat { 0 }
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved package `foo`: `::foo` names no package this package depends on",
            "unresolved package `nope`: `::nope` names no package this package depends on",
            "unresolved package `missing`: `::missing` names no package this package depends on",
            "unresolved name `::std::DoesNotExist`",
            "unresolved name `pkg::Missing`",
        ]
    );
}

/// An inline module re-exports through a leading `::`, and a path with a
/// fixed start must name a type: not a module, and not nothing.
#[salsa_test]
fn rooted_paths_re_export_and_name_types(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod std {
    pub fn answer() -> Nat { 42 }
}

mod m {
    pub use ::std::Option::Some as Just

    fn f(x: nope::T) -> Nat { 0 }
}

mod Geo {
    pub fn area() -> Nat { 1 }
}

fn g(x: pkg::Geo) -> Nat { 0 }

fn main() -> Nil {
    let _ = case m::Just(1) {
        .Some(n) -> n
        .None -> std::answer()
    }
}
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved name `nope::T`",
            "`pkg::Geo` names a module, not a type",
        ]
    );
}

/// Inline modules re-export through the prelude's namespaces, a dependency,
/// and their own modules; a name an inline module has keeps its paths in
/// the module, through an import too.
#[salsa_test]
fn inline_module_paths_start_from_what_the_module_names(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod std {
    pub fn twice(n: Nat) -> Nat { n + n }
}

pub mod a {
    pub mod b {
        pub mod c {
            pub fn one() -> Nat { 1 }
        }
    }
}

use a::b

mod m {
    pub use Option::Some as Just
    pub use std::String::len as size
    pub use std as library
    pub use super::a::b::c as deeper

    fn hidden(x: nope::T) -> Nat { 0 }
}

mod k {
    use super::std

    pub fn doubled() -> Nat { 1.std::twice() }
}

mod n {
    mod std {
        pub fn f() -> Nat { 1 }
    }

    use std::String::len
}

fn main() -> Nil {
    let _ = case m::Just(m::size("a") + m::deeper::one() + b::c::one() + k::doubled()) {
        .Some(n) -> n
        .None -> 0
    }
    let _ = m::library::Option::Some(1)
}
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved name `nope::T`",
            "`use std::String::len` is ambiguous: `std` names both a name of this module and \
             package `std`; write `self::std` or `pkg::std` for the module's name, or `::std` \
             for the package",
        ]
    );
}

/// A `use` may name a dependency's root, and the name it imports reaches
/// that package's items.
#[salsa_test]
fn a_root_import_may_name_a_dependency_root(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
use ::std as library

mod std {
    pub fn answer() -> Nat { 42 }
}

fn main() -> Nil {
    let _ = case library::Option::Some(std::answer()) {
        library::Option::Some(n) -> n
        library::Option::None -> 0
    }
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// An annotation path names only what is at the location it reads: one
/// that names nothing is reported at the package root as in a module.
#[salsa_test]
fn an_annotation_path_must_name_a_type(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod geo {
    pub struct Point { x: Nat }

    pub mod shapes {
        pub enum Shape { Dot }
    }
}

use geo::shapes

fn own(p: geo::Point, s: shapes::Shape) -> Nat { 0 }

fn dependency(o: std::Option(Nat), l: std::collections::List(Nat)) -> Nat { 0 }

fn missing(a: nope::T, b: geo::Nope, c: std::Nope) -> Nat { 0 }

mod m {
    fn missing(a: nope::T, b: std::Nope) -> Nat { 0 }
}
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved name `nope::T`",
            "unresolved name `geo::Nope`",
            "unresolved name `std::Nope`",
            "unresolved name `nope::T`",
            "unresolved name `std::Nope`",
        ]
    );
}

/// A path continues through a re-export into the re-exported item's
/// namespace, unless the re-export is hidden from where the path is.
#[salsa_test]
fn paths_continue_through_a_re_export(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod a {
    pub mod m {
        pub use std as library
        pub(super) use std as internal
    }

    fn inside() -> Option(Nat) { m::internal::Option::Some(1) }
}

fn variant() -> Option(Nat) { a::m::library::Option::Some(1) }
fn function() -> Nat { a::m::library::Option::unwrap_or(Option::Some(1), 0) }
fn rooted() -> Option(Nat) { pkg::a::m::library::Option::None }
fn hidden() -> Option(Nat) { a::m::internal::Option::Some(1) }
"#,
    );
    assert_eq!(errors, ["unresolved name `a::m::internal::Option::Some`"]);
}

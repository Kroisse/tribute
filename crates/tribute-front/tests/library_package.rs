//! The prelude is the root of package `std`: every module sees its root items
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
        Some(n) -> n
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
             fn main() -> Nil {{\n    let _ = f(Some(1))\n}}\n"
        )
    };
    assert_eq!(
        errors(db, &call("Option(Nat)")),
        ["type error at call site in function 'main': \
             expected `Option(Nat)`, found `std::Option(Nat)`"]
    );
    assert_eq!(
        errors(db, &call("Option::Option(Nat)")),
        ["type error at call site in function 'main': \
             expected `Option::Option(Nat)`, found `std::Option(Nat)`"]
    );
    assert_eq!(
        errors(
            db,
            "mod Option {}\n\nfn main() -> Nil {\n    let _ = Option::Some(1)\n}\n"
        ),
        ["unresolved name `Option::Some`; did you mean `Some`?"]
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
    let _ = pkg::Some(1)
}
"#,
    );
    assert_eq!(
        errors,
        ["unresolved name `pkg::Some`; did you mean `Some`?"]
    );
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

fn call() -> Nat { annotated(Some(1)) }

fn prelude() -> Option(List(Nat)) { Option::Some(List::prepend(1, [])) }
"#,
    );
    assert_eq!(
        errors,
        [
            "unresolved name `std::String::len`",
            "type error at call site in function 'call': \
             expected `pkg::std::Option(Nat)`, found `std::Option(Nat)`",
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

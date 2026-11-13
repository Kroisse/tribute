//! Item visibility: a declaration is visible in the module that declares it
//! and its descendants, widened by `pub(super)`, `pub(pkg)`, and `pub`, and
//! only where every module around it is visible too.

mod common;

use self::common::ast_pipeline_error_messages;
use salsa_test_macros::salsa_test;
use tribute_front::SourceCst;

fn errors(db: &dyn salsa::Database, text: &str) -> Vec<String> {
    ast_pipeline_error_messages(db, SourceCst::from_source_str(db, "visibility.trb", text))
}

fn assert_hidden(errors: &[String], expected: &str) {
    assert!(
        errors.iter().any(|error| error == expected),
        "expected `{expected}` in {errors:#?}"
    );
}

#[salsa_test]
fn a_private_item_is_visible_in_its_module_and_below(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn helper() -> Nat { 1 }

mod outer {
    use super::helper

    fn own() -> Nat { helper() }

    pub mod inner {
        use super::own

        pub fn twice() -> Nat { own() + pkg::outer::own() }
    }

    pub fn run() -> Nat { inner::twice() }
}

fn main() -> Nil {
    let _ = outer::run() + helper()
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[salsa_test]
fn a_private_item_is_hidden_outside_its_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    fn hidden() -> Nat { 1 }
    struct Secret { value: Nat }
    enum Mode { Quiet }
}

fn read(secret: m::Secret, mode: m::Mode) -> Nat {
    case mode {
        m::Mode::Quiet -> m::hidden()
    }
}

fn main() -> Nil {
    let _ = m::Mode::Quiet
}
"#,
    );
    assert_hidden(
        &errors,
        "`m::hidden` is not visible here: `m::hidden` is private",
    );
    assert_hidden(
        &errors,
        "`m::Secret` is not visible here: `m::Secret` is private",
    );
    assert_hidden(
        &errors,
        "`m::Mode` is not visible here: `m::Mode` is private",
    );
    assert_hidden(
        &errors,
        "`m::Mode::Quiet` is not visible here: `m::Mode` is private",
    );
    assert_hidden(
        &errors,
        "`m::Mode::Quiet` is not visible here: `m::Mode` is private",
    );
}

#[salsa_test]
fn a_hidden_module_hides_its_public_items(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod outer {
    mod inner {
        pub fn shown() -> Nat { 1 }
    }
}

use outer::inner::shown

fn main() -> Nil {
    let _ = outer::inner::shown()
}
"#,
    );
    assert_hidden(
        &errors,
        "`outer::inner::shown` is not visible here: `outer::inner` is private",
    );
}

#[salsa_test]
fn restricted_visibility_widens_to_the_parent_and_the_package(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub(pkg) mod outer {
    pub mod inner {
        pub(super) fn parent() -> Nat { 1 }
        pub(pkg) fn package() -> Nat { 2 }
    }

    pub fn run() -> Nat { inner::parent() }
}

fn main() -> Nil {
    let _ = outer::run() + outer::inner::package() + outer::inner::parent()
}
"#,
    );
    assert_eq!(
        errors,
        [
            "`outer::inner::parent` is not visible here: `outer::inner::parent` is declared `pub(super)`"
        ]
    );
}

#[salsa_test]
fn a_module_path_restricts_to_a_module_around_the_declaration(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod outer {
    pub mod inner {
        pub(pkg::outer) fn in_outer() -> Nat { 1 }
        pub(self) fn in_inner() -> Nat { 2 }
        pub fn both() -> Nat { in_outer() + in_inner() }
    }

    pub fn run() -> Nat { inner::in_outer() + inner::both() }
}

fn main() -> Nil {
    let _ = outer::run() + outer::inner::in_outer() + outer::inner::in_inner()
}
"#,
    );
    assert_eq!(
        errors,
        [
            "`outer::inner::in_outer` is not visible here: `outer::inner::in_outer` is declared `pub(pkg::outer)`",
            "`outer::inner::in_inner` is not visible here: `outer::inner::in_inner` is declared `pub(self)`",
        ]
    );
}

#[salsa_test]
fn a_visibility_path_must_name_a_module_around_the_declaration(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub(super) fn f() -> Nat { 1 }

pub mod a {
    pub(super::b) fn g() -> Nat { 2 }
    pub(pkg::nowhere) fn h() -> Nat { 3 }
}

pub mod b {}

fn main() -> Nil {}
"#,
    );
    assert_eq!(
        errors,
        [
            "`pub(super)`: `super` at the package root has no parent module",
            "`pub(super::b)` names no module around this declaration: a restricted visibility names the declaring module or a module around it",
            "`pub(pkg::nowhere)` names no module around this declaration: a restricted visibility names the declaring module or a module around it",
        ]
    );
}

/// A type's own visibility decides where it is visible, even when its
/// companion module is less visible.
#[salsa_test]
fn a_hidden_companion_does_not_hide_its_type(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod api {
    pub struct T { pub n: Nat }
    mod T {
        pub fn hidden(t: T) -> Nat { t.n }
    }
}

use api::T

fn read(t: api::T) -> T { t }

fn main() -> Nil {
    let _ = api::T::hidden(api::T { n: 1 })
}
"#,
    );
    assert_eq!(
        errors,
        ["`api::T::hidden` is not visible here: `api::T` is private"]
    );
}

#[salsa_test]
fn a_hidden_ability_is_hidden_in_annotations_and_handler_arms(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod fx {
    ability Tick {
        op tick() -> Nat
    }
}

fn count() ->{fx::Tick} Nat { fx::Tick::tick() }

fn main() -> Nil {
    let _ = handle count() {
        do result { result }
        op fx::Tick::tick() { resume 1 }
    }
}
"#,
    );
    assert_hidden(
        &errors,
        "`fx::Tick` is not visible here: `fx::Tick` is private",
    );
    assert_hidden(
        &errors,
        "`fx::Tick::tick` is not visible here: `fx::Tick` is private",
    );
}

#[salsa_test]
fn a_hidden_method_is_no_ufcs_candidate(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
struct Counter { n: Nat }

pub mod Counter {
    fn bump(counter: Counter) -> Nat { counter.n + 1 }
    pub fn read(counter: Counter) -> Nat { counter.n }
}

fn main() -> Nil {
    let counter = Counter { n: 1 }
    let _ = counter.read() + counter.bump()
}
"#,
    );
    assert_eq!(
        errors,
        ["`bump` is not visible here: `Counter::bump` is private"]
    );
}

#[salsa_test]
fn a_re_export_cannot_widen_its_target(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod outer {
    pub(pkg) fn internal() -> Nat { 1 }
}

pub use outer::internal
pub(pkg) use outer::internal as same

fn main() -> Nil {}
"#,
    );
    assert_eq!(
        errors,
        [
            "`outer::internal` cannot be re-exported as `pub`: `outer::internal` is declared `pub(pkg)`"
        ]
    );
}

#[salsa_test]
fn a_re_export_joins_its_module_namespace(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod source {
    pub fn f() -> Nat { 1 }
    pub struct P { pub n: Nat }
    pub ability Tick {
        op tick() -> Nat
    }
}

pub mod facade {
    pub use pkg::source::f
    pub use super::source::P as Point
    pub use pkg::source::Tick
    pub(pkg) use pkg::source::f as internal
    use pkg::source as origin
    pub use origin::f as again
}

pub mod outer {
    mod hidden {
        pub use pkg::source::f
    }
}

use facade::Point

fn count() ->{facade::Tick} Nat { facade::Tick::tick() }

fn read(p: facade::Point) -> Nat { p.n }

fn main() -> Nil {
    let _ = facade::f() + facade::internal() + facade::again() + read(Point { n: 1 }) + outer::hidden::f()
}
"#,
    );
    assert_eq!(
        errors,
        ["`outer::hidden::f` is not visible here: `outer::hidden` is private"]
    );
}

/// The prelude is the root of package `std`, so a user package sees only its
/// `pub` items, on every resolution path.
#[salsa_test]
fn the_prelude_shows_only_its_public_items(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
use std::io::ReadLineResult

fn raw() -> Nil {
    let _ = __tribute_byte_array_range_equal
    let _ = std::__tribute_byte_array_range_equal
}

fn cursor(c: String::Cursor) -> Nat { 0 }

fn same(left: String, right: String) -> Bool { left.rope_equal(right) }

fn main() -> Nil {
    let _ = Option::Some(1).is_some()
    let _ = Int::to_string(+1)
}
"#,
    );
    for expected in [
        "`std::io::ReadLineResult` is not visible here: `std::io::ReadLineResult` is private",
        // Not an item of `std::prelude::v0`, so not a short name.
        "unresolved name `__tribute_byte_array_range_equal`",
        "`std::__tribute_byte_array_range_equal` is not visible here: `std::__tribute_byte_array_range_equal` is private",
        "`std::String::Cursor` is not visible here: `std::String::Cursor` is private",
        "`rope_equal` is not visible here: `std::String::rope_equal` is private",
    ] {
        assert_hidden(&errors, expected);
    }
    assert_eq!(errors.len(), 5, "{errors:#?}");
}

/// `Bytes` is a primitive type: outside `std` its representation is hidden,
/// and it has no fields or constructor.
#[salsa_test]
fn bytes_shows_no_representation(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
fn reads(bytes: Bytes) -> Nat {
    let _ = Bytes::repr(bytes)
    bytes.len()
}

fn builds(r: Bytes::Repr) -> Bytes { Bytes::from_repr(r) }

fn spreads(bytes: Bytes) -> Bytes { Bytes { ..bytes } }
"#,
    );
    assert_eq!(
        errors,
        [
            "`Bytes::repr` is not visible here: `std::Bytes::repr` is declared `pub(pkg)`",
            "`std::Bytes::Repr` is not visible here: `std::Bytes::Repr` is declared `pub(pkg)`",
            "`Bytes::from_repr` is not visible here: `std::Bytes::from_repr` is private",
            "expected a constructor, found module `std::Bytes`",
        ]
    );
}

/// An implicit constructor does not reach a hidden constructor through the
/// type its context gives.
#[salsa_test]
fn an_implicit_constructor_does_not_name_a_hidden_variant(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    enum Mode { Quiet, Loud }
    pub fn take(mode: Mode) -> Nat { 0 }
    pub fn inside() -> Nat { take(.Quiet) }
}

fn outside() -> Nat { m::take(.Quiet) }

fn later() -> Nat {
    let mode = .Loud
    m::take(mode)
}
"#,
    );
    assert_eq!(
        errors,
        [
            "`m::Mode::Quiet` is not visible here: `m::Mode` is private",
            "`m::Mode::Loud` is not visible here: `m::Mode` is private",
        ]
    );
}

/// The entries a re-export shows under its name are visible where the
/// re-export is, so a path or annotation through them does not get around
/// a hidden re-export.
#[salsa_test]
fn entries_under_a_hidden_re_export_are_hidden(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod a {
    pub mod m {
        pub(super) use std as internal
        pub(super) use std::Option as Opt
    }

    fn inside(o: m::internal::Option(Nat)) -> Option(Nat) { m::Opt::Some(1) }
}

fn annotation(o: a::m::internal::Option(Nat)) -> Nat { 0 }
fn value() -> Option(Nat) { a::m::Opt::Some(1) }
"#,
    );
    assert_eq!(
        errors,
        [
            "`a::m::internal::Option` is not visible here: `a::m::internal` is declared `pub(super)`",
            "`a::m::Opt::Some` is not visible here: `a::m::Opt` is declared `pub(super)`",
        ]
    );
}

/// A restricted re-export does not widen the visibility of a member of what
/// it names.
#[salsa_test]
fn a_restricted_re_export_keeps_its_members_private(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod source {
    fn secret() -> Nat { 1 }
    pub fn open() -> Nat { 2 }
}

pub mod facade {
    pub(pkg) use pkg::source as alias
}

fn hidden() -> Nat { facade::alias::secret() }
fn visible() -> Nat { facade::alias::open() }
"#,
    );
    assert_eq!(
        errors,
        ["`facade::alias::secret` is not visible here: `source::secret` is private"]
    );
}

/// An entry under a re-export is visible only where both it and the
/// re-export are, even where the two are visible in unrelated modules.
#[salsa_test]
fn an_entry_under_a_re_export_needs_both_visible(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod source {
    pub mod inner {
        fn secret() -> Nat { 1 }
        fn inside() -> Nat { pkg::facade::alias::secret() }
    }
}

pub mod facade {
    pub(self) use pkg::source::inner as alias
    fn here() -> Nat { alias::secret() }
}
"#,
    );
    assert_eq!(
        errors,
        [
            "`pkg::facade::alias::secret` is not visible here: `facade::alias` is declared `pub(self)`",
            "`alias::secret` is not visible here: `source::inner::secret` is private",
        ]
    );
}

/// A struct field is private unless marked, even on a public struct. The
/// struct's module sees every field; elsewhere a spread record and a record
/// pattern may leave hidden fields out, while a public field is read and
/// changed as usual.
#[salsa_test]
fn a_public_field_of_a_struct_with_hidden_ones(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod m {
    pub struct Window { pub label: String, data: Nat, pub(pkg) start: Nat }

    pub fn new(label: String) -> Window { Window { label, data: 0, start: 0 } }

    fn data(window: Window) -> Nat {
        let Window { label: _, data, start: _ } = window
        window.data + data + Window::data(window)
    }
}

use m::Window

fn main() -> Nil {
    let window = m::new("a")
    let renamed = Window { ..window, label: "b", start: 1 }
    let relabeled = window.label::set("c").start::modify(fn(n) n + 1)
    let Window { label, start, .. } = renamed
    let Window { .. } = relabeled
    let _ = label <> relabeled.label
    let _ = start + window.start
}
"#,
    );
    assert_eq!(errors, Vec::<String>::new());
}

/// Outside its module a hidden field has no field functions, and a struct
/// with one cannot be built from scratch or matched in full.
#[salsa_test]
fn a_hidden_field_is_hidden_outside_its_module(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod m {
    pub struct Window { pub label: String, data: Nat }

    pub fn new(label: String) -> Window { Window { label, data: 0 } }
}

use m::Window

fn main() -> Nil {
    let window = m::new("a")
    let _ = window.data
    let _ = Window::data(window)
    let _ = Window::data::set(window, 1)
    let _ = Window { label: "b", data: 1 }
    let _ = Window("b", 1)
    let make = Window
    let _ = Window { ..window, data: 1 }
    let Window { data, .. } = window
    let Window(label, data) = window
}

fn implicit() -> Window { .Window { label: "b", data: 1 } }

fn respread(window: Window) -> Window { .Window { ..window, label: "c" } }
"#,
    );
    let hidden = "`m::Window::data` is private";
    assert_eq!(
        errors,
        [
            format!("`data` is not visible here: {hidden}"),
            format!("`Window::data` is not visible here: {hidden}"),
            format!("`Window::data::set` is not visible here: {hidden}"),
            format!("`Window` is not visible here: {hidden}"),
            format!("`Window` is not visible here: {hidden}"),
            format!("`Window` is not visible here: {hidden}"),
            format!("field `data` of `Window` is not visible here: {hidden}"),
            format!("field `data` of `Window` is not visible here: {hidden}"),
            format!("`Window` is not visible here: {hidden}"),
            format!("`m::Window` is not visible here: {hidden}"),
        ]
    );
}

/// An implicit record form and a restricted re-export cannot reach a hidden
/// field either.
#[salsa_test]
fn other_spellings_cannot_reach_a_hidden_field(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod m {
    pub struct Window { pub label: String, data: Nat }
}

mod facade {
    pub(pkg) use pkg::m::Window
}

use m::Window

fn respread(window: Window) -> Window { .Window { ..window, data: 1 } }

fn relabel(window: Window) -> Window { .Window { ..window, label: "b" } }

fn unannotated(window: Window) -> Nil {
    let updated = .Window { ..window, data: 1 }
    let relabeled = .Window { ..window, label: "b" }
}

mod aliased {
    use pkg::m::Window as Pane

    fn update(window: pkg::m::Window) -> Nil {
        let changed = .Pane { ..window, data: 1 }
        let relabeled = .Pane { ..window, label: "b" }
    }
}

fn matched(window: Window) -> Nat {
    case window {
        .Window { data, .. } -> data
    }
}

fn labeled(window: Window) -> String {
    case window {
        .Window { label, .. } -> label
    }
}

fn reexported() -> Window { facade::Window("b", 1) }
"#,
    );
    let hidden = "`m::Window::data` is private";
    assert_eq!(
        errors,
        [
            format!("`facade::Window` is not visible here: {hidden}"),
            format!("`m::Window` is not visible here: {hidden}"),
            format!("`m::Window` is not visible here: {hidden}"),
            // The alias selects no constructor once its candidate is hidden.
            "type `m::Window` has no constructor `Pane`".to_owned(),
            format!("`m::Window` is not visible here: {hidden}"),
        ]
    );
}

/// A field's `pub(path)` names the struct's module or a module around it,
/// and an enum variant's fields take no marker.
#[salsa_test]
fn field_visibility_markers_are_checked(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod m {
    pub struct Window { pub(pkg::other) data: Nat, pub(super) start: Nat }
    pub enum Shape { Dot { pub x: Nat } }
}

pub mod other {}

fn main() -> Nil {}
"#,
    );
    assert_eq!(
        errors,
        [
            "`pub(pkg::other)` names no module around this declaration: \
             a restricted visibility names the declaring module or a module around it",
            "`pub` on an enum variant field: a variant's fields follow the enum's visibility",
        ]
    );
}

/// A field is visible no wider than its struct, and `pub(path)` widens it
/// to the modules below that path.
#[salsa_test]
fn a_field_is_visible_no_wider_than_its_struct(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
pub mod a {
    pub mod b {
        pub(super) struct Inner { pub value: Nat, pub(super) shared: Nat, own: Nat }

        pub(super) fn make() -> Inner { Inner { value: 1, shared: 2, own: 3 } }
    }

    fn read(inner: b::Inner) -> Nat { b::Inner::value(inner) + b::Inner::shared(inner) }
    fn own(inner: b::Inner) -> Nat { b::Inner::own(inner) }
}

fn main() -> Nil {
    let _ = a::b::Inner::value
}
"#,
    );
    assert_eq!(
        errors,
        [
            "`b::Inner::own` is not visible here: `a::b::Inner::own` is private",
            "`a::b::Inner::value` is not visible here: `a::b::Inner` is declared `pub(super)`",
        ]
    );
}

/// A private `use` imports a name into its module's body alone, not into
/// the module's namespace. A path through that module stops at the name,
/// and says the module imports it privately.
#[salsa_test]
fn a_path_through_a_private_use_is_not_visible(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    mod inner {
        pub enum Mode { Quiet }
        pub fn hello() -> Nat { 1 }
        pub ability Ping { fn ping() -> Nil }
    }
    use inner::Mode
    use inner::hello
    use inner::Ping
}

fn take(mode: m::Mode) -> Nat { 0 }

fn quiet(mode: Nat) -> Nat {
    case mode {
        m::Mode::Quiet -> 0
        _ -> 1
    }
}

fn ping() ->{m::Ping} Nil {}

fn main() -> Nil {
    let _ = m::hello()
    let _ = m::Mode::Quiet
}
"#,
    );
    for expected in [
        "`m::Mode` is not visible here: `m` imports `Mode` privately",
        "`m::Mode::Quiet` is not visible here: `m` imports `Mode` privately",
        "`m::Ping` is not visible here: `m` imports `Ping` privately",
        "`m::hello` is not visible here: `m` imports `hello` privately",
    ] {
        assert_hidden(&errors, expected);
    }
    assert!(
        errors.iter().all(|error| !error.starts_with("unresolved")),
        "{errors:#?}"
    );
}

/// A path that names nothing is unresolved, whatever the module it stops
/// in imports privately.
#[salsa_test]
fn a_path_to_nothing_beside_a_private_use_is_unresolved(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    mod inner {
        pub fn hello() -> Nat { 1 }
    }
    use inner::hello
    use inner::nope
}

fn main() -> Nil {
    let _ = m::missing()
    let _ = m::nope()
}
"#,
    );
    assert!(
        errors.contains(&"unresolved name `m::missing`".to_owned())
            && errors.contains(&"unresolved name `m::nope`".to_owned())
            && errors.iter().all(|error| !error.contains("privately")),
        "{errors:#?}"
    );
}

/// A `use` outside the module stops at a private `use` the same way.
#[salsa_test]
fn a_use_through_a_private_use_is_not_visible(db: &salsa::DatabaseImpl) {
    let errors = errors(
        db,
        r#"
mod m {
    mod inner {
        pub fn hello() -> Nat { 1 }
    }
    use inner::hello
}

use m::hello
use m::missing

fn main() -> Nil {}
"#,
    );
    assert_eq!(
        errors,
        [
            "`m::hello` is not visible here: `m` imports `hello` privately",
            "unresolved import `m::missing`",
        ]
    );
}

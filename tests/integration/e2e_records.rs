//! Record expressions.

use crate::common::assert_output_on_both_targets;

#[test]
fn test_record_field_shorthand() {
    // `User { name, age }` initializes each field from the variable of the
    // same name (`new-plans/syntax.md`).
    assert_output_on_both_targets(
        "record_field_shorthand.trb",
        r#"
use std::io::{Io, print_line}

struct User { name: String, age: Int, role: String }

fn main() ->{Io} Nil {
    let name = "John"
    let age = +30
    let user = User { name, age, role: "admin" }
    print_line(user.name)
    print_line(Int::to_string(user.age))
    print_line(user.role)
}
"#,
        "John\n30\nadmin",
    );
}

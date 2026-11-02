//! Implicit constructors `.Name` (`new-plans/type-inference.md` § 암묵적 생성자).

use crate::common::assert_output_on_both_targets;

#[test]
fn test_implicit_constructors_take_the_expected_type() {
    // The return type, the parameter of a call, and the scrutinee give the
    // type whose constructor `.Name` names.
    assert_output_on_both_targets(
        "implicit_constructors_expected.trb",
        r#"
use std::io::{Io, print_line}

enum Tree { Leaf(Int), Node(Tree, Tree) }

fn sum(t: Tree) -> Int {
    case t {
        .Leaf(n) -> n
        .Node(l, r) -> sum(l) + sum(r)
    }
}

fn positive(n: Int) -> Option(Int) {
    case n > +0 {
        True -> .Some(n)
        False -> .None
    }
}

fn describe(o: Option(Int)) -> String {
    case o {
        .Some(n) -> Int::to_string(n)
        .None -> "none"
    }
}

fn main() ->{Io} Nil {
    print_line(Int::to_string(sum(.Node(.Leaf(+1), .Node(.Leaf(+2), .Leaf(+3))))))
    print_line(describe(positive(+5)))
    print_line(describe(positive(-5)))
}
"#,
        "6\n5\nnone",
    );
}

#[test]
fn test_implicit_constructors_wait_for_their_use() {
    // Two types in scope have a `Leaf`: each binding takes the one its later
    // use requires, and a lambda's pattern takes the one `Empty` decides.
    assert_output_on_both_targets(
        "implicit_constructors_deferred.trb",
        r#"
use std::io::{Io, print_line}

enum Tree { Leaf(Int), Node(Tree, Tree) }
enum Heap { Leaf(Int), Empty }

fn tree_value(t: Tree) -> Int {
    case t {
        .Leaf(n) -> n
        .Node(_, _) -> +0
    }
}

fn heap_value(h: Heap) -> Int {
    case h {
        .Leaf(n) -> n + +100
        .Empty -> +0
    }
}

fn main() ->{Io} Nil {
    let t = .Leaf(+1)
    print_line(Int::to_string(tree_value(t)))
    let h = .Leaf(+2)
    print_line(Int::to_string(heap_value(h)))
    let pick = fn(x) {
        case x {
            .Leaf(n) -> n
            .Empty -> +5
        }
    }
    print_line(Int::to_string(pick(.Empty)))
}
"#,
        "1\n102\n5",
    );
}

#[test]
fn test_implicit_constructor_records() {
    assert_output_on_both_targets(
        "implicit_constructors_records.trb",
        r#"
use std::io::{Io, print_line}

struct Point { x: Int, y: Int }
enum Shape { Circle { radius: Int }, Rect { width: Int, height: Int } }

fn area(s: Shape) -> Int {
    case s {
        .Circle { radius } -> radius * radius * +3
        .Rect { width, height } -> width * height
    }
}

fn origin() -> Point { .Point { x: +0, y: +0 } }

fn main() ->{Io} Nil {
    print_line(Int::to_string(area(.Rect { width: +2, height: +3 })))
    print_line(Int::to_string(area(.Circle { radius: +1 })))
    let .Point { x, y } = origin()
    print_line(Int::to_string(x + y))
}
"#,
        "6\n3\n0",
    );
}

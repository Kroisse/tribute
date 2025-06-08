use tribute::TributeDatabaseImpl;
use salsa::Database;

#[test]
fn test_basic_string_interpolation() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = r#"
            fn main() {
                let name = "Alice"
                "Hello, \{name}!"
            }
        "#;
        match tribute::eval_str(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "Hello, Alice!");
            }
            Ok(other) => panic!("Expected String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate string interpolation: {}", e),
        }
    });
}

#[test]
fn test_multiple_interpolations() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = r#"
            fn main() {
                let x = 10
                let y = 20
                "\{x} + \{y} = \{x + y}"
            }
        "#;
        match tribute::eval_str(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "10 + 20 = 30");
            }
            Ok(other) => panic!("Expected String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate multiple interpolations: {}", e),
        }
    });
}

#[test]
fn test_mixed_text_and_interpolation() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = r#"
            fn main() {
                let items = 3
                let price = 5
                "You have \{items} items costing \{items * price} dollars."
            }
        "#;
        match tribute::eval_str(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "You have 3 items costing 15 dollars.");
            }
            Ok(other) => panic!("Expected String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate mixed interpolation: {}", e),
        }
    });
}

#[test]
fn test_empty_interpolation() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = r#"
            fn main() {
                let empty = ""
                "Start\{empty}End"
            }
        "#;
        match tribute::eval_str(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "StartEnd");
            }
            Ok(other) => panic!("Expected String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate empty interpolation: {}", e),
        }
    });
}

#[test]
fn test_function_call_in_interpolation() {
    TributeDatabaseImpl::default().attach(|db| {
        let source = r#"
            fn double(x) { x * 2 }
            fn main() {
                let n = 5
                "Double of \{n} is \{double(n)}"
            }
        "#;
        match tribute::eval_str(db, "test.trb", source) {
            Ok(tribute::eval::Value::String(s)) => {
                assert_eq!(s, "Double of 5 is 10");
            }
            Ok(other) => panic!("Expected String, got {:?}", other),
            Err(e) => panic!("Failed to evaluate function call interpolation: {}", e),
        }
    });
}
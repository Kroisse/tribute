//! Built-in functions for the Tribute programming language.
//!
//! This module contains all standard library functions that are available
//! in Tribute programs without requiring imports. These functions provide
//! basic functionality for I/O, arithmetic, string manipulation, and list
//! operations.
//!
//! All functions follow a consistent pattern:
//! - Take a slice of `Value` arguments
//! - Return `Result<Value, Error>`
//! - Validate argument count and types
//! - Provide clear error messages

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::eval::{BuiltinFn, Value};

type Error = Box<dyn std::error::Error + 'static>;

/// Prints all arguments to stdout followed by a newline.
///
/// # Arguments
/// * `args` - Values to print. Each value is printed using its Display implementation.
///
/// # Returns
/// Always returns `Value::Unit`.
///
/// # Example
/// ```tribute
/// (print_line "Hello" " " "world!")  ; prints "Hello world!"
/// ```
fn print_line(args: &[Value]) -> Result<Value, Error> {
    for arg in args {
        print!("{}", arg);
    }
    println!();
    Ok(Value::Unit)
}

/// Reads a line from stdin including the trailing newline.
///
/// # Arguments
/// This function takes no arguments.
///
/// # Returns
/// Returns the input line as a `Value::String` including the newline character.
///
/// # Errors
/// Returns an error if reading from stdin fails.
///
/// # Example
/// ```tribute
/// (let line (input_line))  ; reads a line from the user
/// ```
fn input_line(_args: &[Value]) -> Result<Value, Error> {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(Value::String(input))
}

/// Adds two numbers together.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments, both numbers.
///
/// # Returns
/// Returns the sum as a `Value::Number`.
///
/// # Errors
/// * Returns an error if not exactly 2 arguments are provided.
/// * Returns an error if any argument is not a number.
///
/// # Example
/// ```tribute
/// (+ 5 3)  ; returns 8
/// ```
fn add(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("+ requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
        _ => Err("+ requires two numbers".into()),
    }
}

/// Subtracts the second number from the first.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments, both numbers.
///
/// # Returns
/// Returns the difference as a `Value::Number`.
///
/// # Errors
/// * Returns an error if not exactly 2 arguments are provided.
/// * Returns an error if any argument is not a number.
///
/// # Example
/// ```tribute
/// (- 10 4)  ; returns 6
/// ```
fn subtract(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("- requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a - b)),
        _ => Err("- requires two numbers".into()),
    }
}

/// Multiplies two numbers.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments, both numbers.
///
/// # Returns
/// Returns the product as a `Value::Number`.
///
/// # Errors
/// * Returns an error if not exactly 2 arguments are provided.
/// * Returns an error if any argument is not a number.
///
/// # Example
/// ```tribute
/// (* 6 7)  ; returns 42
/// ```
fn multiply(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("* requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a * b)),
        _ => Err("* requires two numbers".into()),
    }
}

/// Divides the first number by the second (integer division).
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments, both numbers.
///
/// # Returns
/// Returns the quotient as a `Value::Number`.
///
/// # Errors
/// * Returns an error if not exactly 2 arguments are provided.
/// * Returns an error if any argument is not a number.
/// * Returns an error if attempting to divide by zero.
///
/// # Example
/// ```tribute
/// (/ 15 3)  ; returns 5
/// (/ 7 2)   ; returns 3 (integer division)
/// ```
fn divide(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("/ requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => {
            if *b == 0 {
                Err("division by zero".into())
            } else {
                Ok(Value::Number(a / b))
            }
        }
        _ => Err("/ requires two numbers".into()),
    }
}

/// Returns the remainder of integer division.
fn modulo(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("% requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => {
            if *b == 0 {
                Err("modulo by zero".into())
            } else {
                Ok(Value::Number(a % b))
            }
        }
        _ => Err("% requires two numbers".into()),
    }
}

/// Checks if two values are equal.
fn equal(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("== requires exactly 2 arguments".into());
    }
    let result = match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    };
    Ok(Value::Number(if result { 1 } else { 0 }))
}

/// Checks if two values are not equal.
fn not_equal(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("!= requires exactly 2 arguments".into());
    }
    let result = match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => a != b,
        (Value::String(a), Value::String(b)) => a != b,
        (Value::Unit, Value::Unit) => false,
        _ => true,
    };
    Ok(Value::Number(if result { 1 } else { 0 }))
}

/// Checks if the first number is less than the second.
fn less_than(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("< requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(if a < b { 1 } else { 0 })),
        _ => Err("< requires two numbers".into()),
    }
}

/// Checks if the first number is greater than the second.
fn greater_than(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("> requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(if a > b { 1 } else { 0 })),
        _ => Err("> requires two numbers".into()),
    }
}

/// Checks if the first number is less than or equal to the second.
fn less_equal(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("<= requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(if a <= b { 1 } else { 0 })),
        _ => Err("<= requires two numbers".into()),
    }
}

/// Checks if the first number is greater than or equal to the second.
fn greater_equal(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err(">= requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(a), Value::Number(b)) => Ok(Value::Number(if a >= b { 1 } else { 0 })),
        _ => Err(">= requires two numbers".into()),
    }
}

/// Logical AND - returns 1 if both values are truthy (non-zero), 0 otherwise.
fn logical_and(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("&& requires exactly 2 arguments".into());
    }
    let is_truthy = |v: &Value| -> bool {
        match v {
            Value::Number(n) => *n != 0,
            Value::String(s) => !s.is_empty(),
            Value::List(items) => !items.is_empty(),
            Value::Unit => false,
            Value::Fn(_, _, _) => true,
            Value::BuiltinFn(_, _) => true,
        }
    };
    let result = is_truthy(&args[0]) && is_truthy(&args[1]);
    Ok(Value::Number(if result { 1 } else { 0 }))
}

/// Logical OR - returns 1 if either value is truthy (non-zero), 0 otherwise.
fn logical_or(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("|| requires exactly 2 arguments".into());
    }
    let is_truthy = |v: &Value| -> bool {
        match v {
            Value::Number(n) => *n != 0,
            Value::String(s) => !s.is_empty(),
            Value::List(items) => !items.is_empty(),
            Value::Unit => false,
            Value::Fn(_, _, _) => true,
            Value::BuiltinFn(_, _) => true,
        }
    };
    let result = is_truthy(&args[0]) || is_truthy(&args[1]);
    Ok(Value::Number(if result { 1 } else { 0 }))
}

/// Splits a string into a list of substrings using the given delimiter.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments:
///   - First: delimiter string
///   - Second: text to split
///
/// # Returns
/// Returns a `Value::List` containing the split strings.
///
/// # Errors
/// * Returns an error if not exactly 2 arguments are provided.
/// * Returns an error if any argument is not a string.
///
/// # Example
/// ```tribute
/// (split " " "hello world")  ; returns ["hello", "world"]
/// (split "," "a,b,c")        ; returns ["a", "b", "c"]
/// ```
fn split(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("split requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::String(delimiter), Value::String(text)) => {
            let parts: Vec<Value> = text
                .split(delimiter)
                .map(|s| Value::String(s.to_string()))
                .collect();
            Ok(Value::List(parts))
        }
        _ => Err("split requires two strings".into()),
    }
}

/// Removes trailing whitespace from a string.
///
/// # Arguments
/// * `args` - Must be exactly 1 argument, a string.
///
/// # Returns
/// Returns the trimmed string as a `Value::String`.
///
/// # Errors
/// * Returns an error if not exactly 1 argument is provided.
/// * Returns an error if the argument is not a string.
///
/// # Example
/// ```tribute
/// (trim_right "hello   ")  ; returns "hello"
/// (trim_right "test\n")    ; returns "test"
/// ```
fn trim_right(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err("trim_right requires exactly 1 argument".into());
    }
    match &args[0] {
        Value::String(s) => Ok(Value::String(s.trim_end().to_string())),
        _ => Err("trim_right requires a string".into()),
    }
}

/// Converts a string to a number.
///
/// # Arguments
/// * `args` - Must be exactly 1 argument, a string containing a valid integer.
///
/// # Returns
/// Returns the parsed number as a `Value::Number`.
///
/// # Errors
/// * Returns an error if not exactly 1 argument is provided.
/// * Returns an error if the argument is not a string.
/// * Returns an error if the string cannot be parsed as an integer.
///
/// # Example
/// ```tribute
/// (to_number "42")   ; returns 42
/// (to_number "-10")  ; returns -10
/// ```
fn to_number(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err("to_number requires exactly 1 argument".into());
    }
    match &args[0] {
        Value::String(s) => s
            .parse::<i64>()
            .map(Value::Number)
            .map_err(|_| "failed to parse string as number".into()),
        _ => Err("to_number requires a string".into()),
    }
}

/// Gets an element from a list by index.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments:
///   - First: index (number) - 0-based index
///   - Second: list to get from
///
/// # Returns
/// Returns the element at the given index.
///
/// # Errors
/// * Returns an error if not exactly 2 arguments are provided.
/// * Returns an error if the first argument is not a number or the second is not a list.
/// * Returns an error if the index is out of bounds.
///
/// # Example
/// ```tribute
/// (get 0 ["a" "b" "c"])  ; returns "a"
/// (get 2 ["x" "y" "z"])  ; returns "z"
/// ```
fn get(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("get requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Number(index), Value::List(items)) => {
            let idx = *index as usize;
            if idx < items.len() {
                Ok(items[idx].clone())
            } else {
                Err("index out of bounds".into())
            }
        }
        _ => Err("get requires a number and a list".into()),
    }
}

/// Global registry of built-in functions available in the Tribute language.
///
/// This HashMap is lazily initialized and contains all standard library functions
/// that are available in the top-level environment. Each function is stored with
/// its name as the key and wrapped in a `Value::BuiltinFn`.
///
/// # Available Functions
///
/// ## I/O Functions
/// - `print_line`: Print values to stdout with newline
/// - `input_line`: Read a line from stdin
///
/// ## Arithmetic Operations
/// - `+`: Addition
/// - `-`: Subtraction
/// - `*`: Multiplication
/// - `/`: Division (integer)
///
/// ## String Operations
/// - `split`: Split string by delimiter
/// - `trim_right`: Remove trailing whitespace
/// - `to_number`: Parse string to number
///
/// ## List Operations
/// - `get`: Get element by index
pub static BUILTINS: LazyLock<HashMap<String, Value>> = LazyLock::new(|| {
    let temp: &[(&str, BuiltinFn)] = &[
        // I/O
        ("print_line", print_line),
        ("input_line", input_line),
        // Arithmetic
        ("+", add),
        ("-", subtract),
        ("*", multiply),
        ("/", divide),
        ("%", modulo),
        // Comparison
        ("==", equal),
        ("!=", not_equal),
        ("<", less_than),
        (">", greater_than),
        ("<=", less_equal),
        (">=", greater_equal),
        // Logical
        ("&&", logical_and),
        ("||", logical_or),
        // String
        ("split", split),
        ("trim_right", trim_right),
        ("to_number", to_number),
        // List
        ("get", get),
    ];
    temp.iter()
        .map(|(name, f)| (name.to_string(), Value::BuiltinFn(name, *f)))
        .collect()
});

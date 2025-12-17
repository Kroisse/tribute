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
/// Returns the sum. Type promotion: Float > Int > Nat
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
        // Float operations
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + (*b as f64))),
        (Value::Float(a), Value::Nat(b)) => Ok(Value::Float(a + (*b as f64))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64) + b)),
        (Value::Nat(a), Value::Float(b)) => Ok(Value::Float((*a as f64) + b)),
        // Int operations
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (Value::Int(a), Value::Nat(b)) => Ok(Value::Int(a + (*b as i64))),
        (Value::Nat(a), Value::Int(b)) => Ok(Value::Int((*a as i64) + b)),
        // Nat operations
        (Value::Nat(a), Value::Nat(b)) => Ok(Value::Nat(a + b)),
        _ => Err("+ requires two numbers".into()),
    }
}

/// Subtracts the second number from the first.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments, both numbers.
///
/// # Returns
/// Returns the difference. Type promotion: Float > Int > Nat
/// Note: Nat - Nat always returns Int since result may be negative
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
        // Float operations
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - (*b as f64))),
        (Value::Float(a), Value::Nat(b)) => Ok(Value::Float(a - (*b as f64))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64) - b)),
        (Value::Nat(a), Value::Float(b)) => Ok(Value::Float((*a as f64) - b)),
        // Int operations
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
        (Value::Int(a), Value::Nat(b)) => Ok(Value::Int(a - (*b as i64))),
        (Value::Nat(a), Value::Int(b)) => Ok(Value::Int((*a as i64) - b)),
        // Nat - Nat returns Int since result may be negative
        (Value::Nat(a), Value::Nat(b)) => Ok(Value::Int((*a as i64) - (*b as i64))),
        _ => Err("- requires two numbers".into()),
    }
}

/// Multiplies two numbers.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments, both numbers.
///
/// # Returns
/// Returns the product. Type promotion: Float > Int > Nat
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
        // Float operations
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * (*b as f64))),
        (Value::Float(a), Value::Nat(b)) => Ok(Value::Float(a * (*b as f64))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64) * b)),
        (Value::Nat(a), Value::Float(b)) => Ok(Value::Float((*a as f64) * b)),
        // Int operations
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
        (Value::Int(a), Value::Nat(b)) => Ok(Value::Int(a * (*b as i64))),
        (Value::Nat(a), Value::Int(b)) => Ok(Value::Int((*a as i64) * b)),
        // Nat operations
        (Value::Nat(a), Value::Nat(b)) => Ok(Value::Nat(a * b)),
        _ => Err("* requires two numbers".into()),
    }
}

/// Divides the first number by the second.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments, both numbers.
///
/// # Returns
/// Returns the quotient. Type promotion: Float > Int > Nat
/// For integers, performs integer division.
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
        // Float operations (no division by zero check needed, returns infinity or NaN)
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / (*b as f64))),
        (Value::Float(a), Value::Nat(b)) => Ok(Value::Float(a / (*b as f64))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64) / b)),
        (Value::Nat(a), Value::Float(b)) => Ok(Value::Float((*a as f64) / b)),
        // Int operations
        (Value::Int(a), Value::Int(b)) => {
            if *b == 0 {
                Err("division by zero".into())
            } else {
                Ok(Value::Int(a / b))
            }
        }
        (Value::Int(a), Value::Nat(b)) => {
            if *b == 0 {
                Err("division by zero".into())
            } else {
                Ok(Value::Int(a / (*b as i64)))
            }
        }
        (Value::Nat(a), Value::Int(b)) => {
            if *b == 0 {
                Err("division by zero".into())
            } else {
                Ok(Value::Int((*a as i64) / b))
            }
        }
        // Nat operations
        (Value::Nat(a), Value::Nat(b)) => {
            if *b == 0 {
                Err("division by zero".into())
            } else {
                Ok(Value::Nat(a / b))
            }
        }
        _ => Err("/ requires two numbers".into()),
    }
}

/// Returns the remainder of division.
fn modulo(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("% requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        // Float operations
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a % (*b as f64))),
        (Value::Float(a), Value::Nat(b)) => Ok(Value::Float(a % (*b as f64))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64) % b)),
        (Value::Nat(a), Value::Float(b)) => Ok(Value::Float((*a as f64) % b)),
        // Int operations
        (Value::Int(a), Value::Int(b)) => {
            if *b == 0 {
                Err("modulo by zero".into())
            } else {
                Ok(Value::Int(a % b))
            }
        }
        (Value::Int(a), Value::Nat(b)) => {
            if *b == 0 {
                Err("modulo by zero".into())
            } else {
                Ok(Value::Int(a % (*b as i64)))
            }
        }
        (Value::Nat(a), Value::Int(b)) => {
            if *b == 0 {
                Err("modulo by zero".into())
            } else {
                Ok(Value::Int((*a as i64) % b))
            }
        }
        // Nat operations
        (Value::Nat(a), Value::Nat(b)) => {
            if *b == 0 {
                Err("modulo by zero".into())
            } else {
                Ok(Value::Nat(a % b))
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
        // Numeric comparisons with type coercion
        (Value::Nat(a), Value::Nat(b)) => *a == *b,
        (Value::Int(a), Value::Int(b)) => *a == *b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Nat(a), Value::Int(b)) => (*a as i64) == *b,
        (Value::Int(a), Value::Nat(b)) => *a == (*b as i64),
        (Value::Nat(a), Value::Float(b)) => ((*a as f64) - b).abs() < f64::EPSILON,
        (Value::Float(a), Value::Nat(b)) => (a - (*b as f64)).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) => ((*a as f64) - b).abs() < f64::EPSILON,
        (Value::Float(a), Value::Int(b)) => (a - (*b as f64)).abs() < f64::EPSILON,
        // Other types
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Rune(a), Value::Rune(b)) => a == b,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    };
    Ok(Value::Bool(result))
}

/// Checks if two values are not equal.
fn not_equal(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("!= requires exactly 2 arguments".into());
    }
    let result = match (&args[0], &args[1]) {
        // Numeric comparisons with type coercion
        (Value::Nat(a), Value::Nat(b)) => *a != *b,
        (Value::Int(a), Value::Int(b)) => *a != *b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() >= f64::EPSILON,
        (Value::Nat(a), Value::Int(b)) => (*a as i64) != *b,
        (Value::Int(a), Value::Nat(b)) => *a != (*b as i64),
        (Value::Nat(a), Value::Float(b)) => ((*a as f64) - b).abs() >= f64::EPSILON,
        (Value::Float(a), Value::Nat(b)) => (a - (*b as f64)).abs() >= f64::EPSILON,
        (Value::Int(a), Value::Float(b)) => ((*a as f64) - b).abs() >= f64::EPSILON,
        (Value::Float(a), Value::Int(b)) => (a - (*b as f64)).abs() >= f64::EPSILON,
        // Other types
        (Value::String(a), Value::String(b)) => a != b,
        (Value::Bool(a), Value::Bool(b)) => a != b,
        (Value::Rune(a), Value::Rune(b)) => a != b,
        (Value::Unit, Value::Unit) => false,
        _ => true,
    };
    Ok(Value::Bool(result))
}

/// Checks if the first number is less than the second.
fn less_than(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("< requires exactly 2 arguments".into());
    }
    let result = match (&args[0], &args[1]) {
        (Value::Nat(a), Value::Nat(b)) => *a < *b,
        (Value::Int(a), Value::Int(b)) => *a < *b,
        (Value::Float(a), Value::Float(b)) => *a < *b,
        (Value::Nat(a), Value::Int(b)) => (*a as i64) < *b,
        (Value::Int(a), Value::Nat(b)) => *a < (*b as i64),
        (Value::Nat(a), Value::Float(b)) => (*a as f64) < *b,
        (Value::Float(a), Value::Nat(b)) => *a < (*b as f64),
        (Value::Int(a), Value::Float(b)) => (*a as f64) < *b,
        (Value::Float(a), Value::Int(b)) => *a < (*b as f64),
        _ => return Err("< requires two numbers".into()),
    };
    Ok(Value::Bool(result))
}

/// Checks if the first number is greater than the second.
fn greater_than(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("> requires exactly 2 arguments".into());
    }
    let result = match (&args[0], &args[1]) {
        (Value::Nat(a), Value::Nat(b)) => *a > *b,
        (Value::Int(a), Value::Int(b)) => *a > *b,
        (Value::Float(a), Value::Float(b)) => *a > *b,
        (Value::Nat(a), Value::Int(b)) => (*a as i64) > *b,
        (Value::Int(a), Value::Nat(b)) => *a > (*b as i64),
        (Value::Nat(a), Value::Float(b)) => (*a as f64) > *b,
        (Value::Float(a), Value::Nat(b)) => *a > (*b as f64),
        (Value::Int(a), Value::Float(b)) => (*a as f64) > *b,
        (Value::Float(a), Value::Int(b)) => *a > (*b as f64),
        _ => return Err("> requires two numbers".into()),
    };
    Ok(Value::Bool(result))
}

/// Checks if the first number is less than or equal to the second.
fn less_equal(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("<= requires exactly 2 arguments".into());
    }
    let result = match (&args[0], &args[1]) {
        (Value::Nat(a), Value::Nat(b)) => *a <= *b,
        (Value::Int(a), Value::Int(b)) => *a <= *b,
        (Value::Float(a), Value::Float(b)) => *a <= *b,
        (Value::Nat(a), Value::Int(b)) => (*a as i64) <= *b,
        (Value::Int(a), Value::Nat(b)) => *a <= (*b as i64),
        (Value::Nat(a), Value::Float(b)) => (*a as f64) <= *b,
        (Value::Float(a), Value::Nat(b)) => *a <= (*b as f64),
        (Value::Int(a), Value::Float(b)) => (*a as f64) <= *b,
        (Value::Float(a), Value::Int(b)) => *a <= (*b as f64),
        _ => return Err("<= requires two numbers".into()),
    };
    Ok(Value::Bool(result))
}

/// Checks if the first number is greater than or equal to the second.
fn greater_equal(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err(">= requires exactly 2 arguments".into());
    }
    let result = match (&args[0], &args[1]) {
        (Value::Nat(a), Value::Nat(b)) => *a >= *b,
        (Value::Int(a), Value::Int(b)) => *a >= *b,
        (Value::Float(a), Value::Float(b)) => *a >= *b,
        (Value::Nat(a), Value::Int(b)) => (*a as i64) >= *b,
        (Value::Int(a), Value::Nat(b)) => *a >= (*b as i64),
        (Value::Nat(a), Value::Float(b)) => (*a as f64) >= *b,
        (Value::Float(a), Value::Nat(b)) => *a >= (*b as f64),
        (Value::Int(a), Value::Float(b)) => (*a as f64) >= *b,
        (Value::Float(a), Value::Int(b)) => *a >= (*b as f64),
        _ => return Err(">= requires two numbers".into()),
    };
    Ok(Value::Bool(result))
}

/// Logical AND - returns true if both values are truthy, false otherwise.
fn logical_and(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("&& requires exactly 2 arguments".into());
    }
    let is_truthy = |v: &Value| -> bool {
        match v {
            Value::Nat(n) => *n != 0,
            Value::Int(n) => *n != 0,
            Value::Float(n) => *n != 0.0,
            Value::Rune(_) => true, // Runes are always truthy
            Value::Bool(b) => *b,
            Value::String(s) => !s.is_empty(),
            Value::Bytes(bytes) => !bytes.is_empty(),
            Value::List(items) => !items.is_empty(),
            Value::Tuple(items) => !items.is_empty(),
            Value::Record(_, fields) => !fields.is_empty(),
            Value::Unit => false,
            Value::Fn(_, _, _) => true,
            Value::Lambda(_, _) => true,
            Value::BuiltinFn(_, _) => true,
        }
    };
    let result = is_truthy(&args[0]) && is_truthy(&args[1]);
    Ok(Value::Bool(result))
}

/// Logical OR - returns true if either value is truthy, false otherwise.
fn logical_or(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("|| requires exactly 2 arguments".into());
    }
    let is_truthy = |v: &Value| -> bool {
        match v {
            Value::Nat(n) => *n != 0,
            Value::Int(n) => *n != 0,
            Value::Float(n) => *n != 0.0,
            Value::Rune(_) => true, // Runes are always truthy
            Value::Bool(b) => *b,
            Value::String(s) => !s.is_empty(),
            Value::Bytes(bytes) => !bytes.is_empty(),
            Value::List(items) => !items.is_empty(),
            Value::Tuple(items) => !items.is_empty(),
            Value::Record(_, fields) => !fields.is_empty(),
            Value::Unit => false,
            Value::Fn(_, _, _) => true,
            Value::Lambda(_, _) => true,
            Value::BuiltinFn(_, _) => true,
        }
    };
    let result = is_truthy(&args[0]) || is_truthy(&args[1]);
    Ok(Value::Bool(result))
}

/// Concatenates two values (String or List).
fn concat(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("<> requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
        (Value::List(a), Value::List(b)) => {
            let mut result = a.clone();
            result.extend(b.iter().cloned());
            Ok(Value::List(result))
        }
        _ => Err("<> requires two strings or two lists".into()),
    }
}

/// Converts a value to its string representation.
///
/// # Arguments
/// * `args` - Must be exactly 1 argument: the value to convert
///
/// # Returns
/// Returns a `Value::String` with the string representation.
///
/// # Errors
/// * Returns an error if not exactly 1 argument is provided.
fn to_string_builtin(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err("to_string requires exactly 1 argument".into());
    }
    let s = match &args[0] {
        Value::Nat(n) => n.to_string(),
        Value::Int(n) => n.to_string(),
        Value::Float(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Unit => "unit".to_string(),
        Value::List(items) => {
            let elements: Vec<String> = items.iter().map(|v| format!("{:?}", v)).collect();
            format!("[{}]", elements.join(", "))
        }
        Value::Tuple(elements) => {
            let element_strs: Vec<String> = elements.iter().map(|v| format!("{:?}", v)).collect();
            format!("#({})", element_strs.join(", "))
        }
        Value::Bytes(b) => format!("b{:?}", String::from_utf8_lossy(b)),
        Value::Rune(c) => c.to_string(),
        Value::BuiltinFn(name, _) => format!("<builtin: {}>", name),
        Value::Lambda(params, _) => format!("<lambda({})>", params.join(", ")),
        Value::Fn(name, _, _) => format!("<function: {}>", name),
        Value::Record(name, fields) => {
            let field_strs: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("{}: {:?}", k, v))
                .collect();
            format!("{} {{ {} }}", name, field_strs.join(", "))
        }
    };
    Ok(Value::String(s))
}

/// Converts a value to its bytes representation.
///
/// # Arguments
/// * `args` - Must be exactly 1 argument: the value to convert
///
/// # Returns
/// Returns a `Value::Bytes` with the bytes representation.
///
/// # Errors
/// * Returns an error if not exactly 1 argument is provided.
fn to_bytes_builtin(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 1 {
        return Err("to_bytes requires exactly 1 argument".into());
    }
    let bytes = match &args[0] {
        Value::Bytes(b) => b.clone(),
        Value::String(s) => s.as_bytes().to_vec(),
        Value::Nat(n) => n.to_string().into_bytes(),
        Value::Int(n) => n.to_string().into_bytes(),
        Value::Float(n) => n.to_string().into_bytes(),
        Value::Bool(b) => b.to_string().into_bytes(),
        Value::Unit => b"unit".to_vec(),
        Value::Rune(c) => {
            let mut buf = [0; 4];
            c.encode_utf8(&mut buf).as_bytes().to_vec()
        }
        _ => return Err("to_bytes: cannot convert this value type to bytes".into()),
    };
    Ok(Value::Bytes(bytes))
}

/// Concatenates two bytes values.
///
/// # Arguments
/// * `args` - Must be exactly 2 arguments: the byte slices to concatenate
///
/// # Returns
/// Returns a `Value::Bytes` with the concatenated bytes.
///
/// # Errors
/// * Returns an error if not exactly 2 arguments are provided.
/// * Returns an error if arguments are not bytes.
fn bytes_concat(args: &[Value]) -> Result<Value, Error> {
    if args.len() != 2 {
        return Err("Bytes::<> requires exactly 2 arguments".into());
    }
    match (&args[0], &args[1]) {
        (Value::Bytes(a), Value::Bytes(b)) => {
            let mut result = a.clone();
            result.extend(b);
            Ok(Value::Bytes(result))
        }
        _ => Err("Bytes::<> requires two bytes values".into()),
    }
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

/// Converts a string to an integer.
///
/// # Arguments
/// * `args` - Must be exactly 1 argument, a string containing a valid integer.
///
/// # Returns
/// Returns the parsed number as a `Value::Int`.
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
            .map(Value::Int)
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
/// * Returns an error if the index is out of bounds or negative.
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
    let index = match &args[0] {
        Value::Nat(n) => *n as usize,
        Value::Int(n) => {
            if *n < 0 {
                return Err("index cannot be negative".into());
            }
            *n as usize
        }
        Value::Float(n) => {
            if *n < 0.0 {
                return Err("index cannot be negative".into());
            }
            *n as usize
        }
        _ => return Err("get requires a number and a list".into()),
    };
    match &args[1] {
        Value::List(items) => {
            if index < items.len() {
                Ok(items[index].clone())
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
        // Concatenation
        ("<>", concat),
        // String
        ("split", split),
        ("trim_right", trim_right),
        ("to_number", to_number),
        ("to_string", to_string_builtin),
        ("to_bytes", to_bytes_builtin),
        ("Bytes::<>", bytes_concat),
        // List
        ("get", get),
    ];
    temp.iter()
        .map(|(name, f)| (name.to_string(), Value::BuiltinFn(name, *f)))
        .collect()
});

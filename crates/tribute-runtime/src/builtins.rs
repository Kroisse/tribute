//! Built-in functions for Tribute programs
//!
//! This module implements the built-in functions that are available
//! to all Tribute programs, such as print_line and input_line.
//! Uses handle-based API for GC compatibility.

use crate::value::{TrHandle, TrValue, allocation_table};

/// Print a value followed by a newline
#[unsafe(no_mangle)]
pub extern "C" fn tr_builtin_print_line(handle: TrHandle) {
    if handle.is_null() {
        print_str("(null)\n");
        return;
    }

    unsafe {
        let val = handle.deref();
        let output = match val {
            TrValue::String(s) => {
                let str_val = s.as_str();
                format!("{}\n", str_val)
            }
            TrValue::Number(n) => {
                format!("{}\n", n)
            }
            TrValue::Unit => "()\n".to_owned(),
        };

        print_str(&output);
    }
}

/// Print a value without a newline
#[unsafe(no_mangle)]
pub extern "C" fn tr_builtin_print(handle: TrHandle) {
    if handle.is_null() {
        print_str("(null)");
        return;
    }

    unsafe {
        let val = handle.deref();
        let output = match val {
            TrValue::String(s) => {
                let str_val = s.as_str();
                str_val.to_owned()
            }
            TrValue::Number(n) => {
                format!("{}", n)
            }
            TrValue::Unit => "()".to_owned(),
        };

        print_str(&output);
    }
}

/// Read a line of input from the user
#[unsafe(no_mangle)]
pub extern "C" fn tr_builtin_input_line() -> TrHandle {
    // For now, return a placeholder string
    // In a full implementation, this would read from stdin
    let placeholder = "input_placeholder";
    allocation_table().allocate(TrValue::string_static(placeholder))
}

/// Read input with a prompt
#[unsafe(no_mangle)]
pub extern "C" fn tr_builtin_input_with_prompt(prompt: TrHandle) -> TrHandle {
    if !prompt.is_null() {
        tr_builtin_print(prompt);
    }
    tr_builtin_input_line()
}

/// Platform-specific print function
fn print_str(s: &str) {
    print!("{}", s);
}

/// Convert a number to string (utility function)
#[unsafe(no_mangle)]
pub extern "C" fn tr_builtin_number_to_string(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::string_static("0"));
    }

    unsafe {
        let val = handle.deref();
        let result = match val {
            TrValue::Number(n) => {
                let s = format!("{}", n);
                TrValue::string(s)
            }
            TrValue::String(s) => {
                // Already a string, clone it
                let str_val = s.as_str();
                TrValue::string(str_val.to_owned())
            }
            TrValue::Unit => TrValue::string_static("()"),
        };

        allocation_table().allocate(result)
    }
}

/// Try to parse a string as a number
#[unsafe(no_mangle)]
pub extern "C" fn tr_builtin_string_to_number(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }

    unsafe {
        let val = handle.deref();
        let result = match val {
            TrValue::String(s) => {
                let str_val = s.as_str();
                match str_val.parse::<f64>() {
                    Ok(num) => TrValue::number(num),
                    Err(_) => TrValue::number(0.0),
                }
            }
            TrValue::Number(n) => {
                // Already a number, clone it
                TrValue::number(*n)
            }
            TrValue::Unit => TrValue::number(0.0),
        };

        allocation_table().allocate(result)
    }
}

/// Check if a value is truthy (for conditional expressions)
#[unsafe(no_mangle)]
pub extern "C" fn tr_builtin_is_truthy(handle: TrHandle) -> bool {
    if handle.is_null() {
        return false;
    }

    unsafe {
        let val = handle.deref();
        match val {
            TrValue::Number(n) => *n != 0.0 && !n.is_nan(),
            TrValue::String(s) => !s.is_empty(),
            TrValue::Unit => false,
        }
    }
}

/// Get the type name of a value (for debugging)
#[unsafe(no_mangle)]
pub extern "C" fn tr_builtin_type_of(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::string_static("null"));
    }

    unsafe {
        let val = handle.deref();
        let type_name = match val {
            TrValue::Number(_) => "number",
            TrValue::String(_) => "string",
            TrValue::Unit => "unit",
        };

        allocation_table().allocate(TrValue::string_static(type_name))
    }
}

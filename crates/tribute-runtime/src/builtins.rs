//! Built-in functions for Tribute programs
//!
//! This module implements the built-in functions that are available
//! to all Tribute programs, such as print_line and input_line.
//! Uses handle-based API for GC compatibility.

use crate::value::{TrValue, TrHandle, ValueTag, allocation_table};

/// Print a value followed by a newline
#[no_mangle]
pub extern "C" fn tr_builtin_print_line(handle: TrHandle) {
    if handle.is_null() {
        print_str("(null)\n");
        return;
    }
    
    unsafe {
        let val = handle.deref();
        let output = match val.tag {
            ValueTag::String => {
                let s = val.data.string.as_str();
                format!("{}\n", s)
            },
            ValueTag::Number => {
                format!("{}\n", val.data.number)
            },
            ValueTag::Unit => "()\n".to_owned(),
        };
        
        print_str(&output);
    }
}

/// Print a value without a newline
#[no_mangle]
pub extern "C" fn tr_builtin_print(handle: TrHandle) {
    if handle.is_null() {
        print_str("(null)");
        return;
    }
    
    unsafe {
        let val = handle.deref();
        let output = match val.tag {
            ValueTag::String => {
                let s = val.data.string.as_str();
                s.to_owned()
            },
            ValueTag::Number => {
                format!("{}", val.data.number)
            },
            ValueTag::Unit => "()".to_owned(),
        };
        
        print_str(&output);
    }
}

/// Read a line of input from the user
#[no_mangle]
pub extern "C" fn tr_builtin_input_line() -> TrHandle {
    // For now, return a placeholder string
    // In a full implementation, this would read from stdin
    let placeholder = "input_placeholder";
    allocation_table().allocate(TrValue::string_static(placeholder))
}

/// Read input with a prompt
#[no_mangle]
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
#[no_mangle]
pub extern "C" fn tr_builtin_number_to_string(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::string_static("0"));
    }
    
    unsafe {
        let val = handle.deref();
        let result = match val.tag {
            ValueTag::Number => {
                let s = format!("{}", val.data.number);
                TrValue::string(s)
            },
            ValueTag::String => {
                // Already a string, clone it
                let s = val.data.string.as_str();
                TrValue::string(s.to_owned())
            },
            ValueTag::Unit => {
                TrValue::string_static("()")
            }
        };
        
        allocation_table().allocate(result)
    }
}

/// Try to parse a string as a number
#[no_mangle]
pub extern "C" fn tr_builtin_string_to_number(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::number(0.0));
    }
    
    unsafe {
        let val = handle.deref();
        let result = match val.tag {
            ValueTag::String => {
                let s = val.data.string.as_str();
                match s.parse::<f64>() {
                    Ok(num) => TrValue::number(num),
                    Err(_) => TrValue::number(0.0),
                }
            },
            ValueTag::Number => {
                // Already a number, clone it
                TrValue::number(val.data.number)
            },
            ValueTag::Unit => {
                TrValue::number(0.0)
            }
        };
        
        allocation_table().allocate(result)
    }
}

/// Check if a value is truthy (for conditional expressions)
#[no_mangle]
pub extern "C" fn tr_builtin_is_truthy(handle: TrHandle) -> bool {
    if handle.is_null() {
        return false;
    }
    
    unsafe {
        let val = handle.deref();
        match val.tag {
            ValueTag::Number => val.data.number != 0.0 && !val.data.number.is_nan(),
            ValueTag::String => val.data.string.len > 0,
            ValueTag::Unit => false,
        }
    }
}

/// Get the type name of a value (for debugging)
#[no_mangle]
pub extern "C" fn tr_builtin_type_of(handle: TrHandle) -> TrHandle {
    if handle.is_null() {
        return allocation_table().allocate(TrValue::string_static("null"));
    }
    
    unsafe {
        let val = handle.deref();
        let type_name = match val.tag {
            ValueTag::Number => "number",
            ValueTag::String => "string", 
            ValueTag::Unit => "unit",
        };
        
        allocation_table().allocate(TrValue::string_static(type_name))
    }
}
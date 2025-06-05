//! Tribute Runtime Library
//!
//! This library provides the runtime support for Tribute programs,
//! including garbage collection, boxing/unboxing, and builtin functions.

use std::{
    alloc::{Layout, alloc, dealloc},
    mem,
    ptr::{self, NonNull},
    sync::atomic::{AtomicU32, Ordering},
};

/// Type tags for boxed values
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TributeType {
    Number = 0,
    String = 1,
    Boolean = 2,
    Function = 3,
    List = 4,
    Nil = 5,
}

/// Boxed value structure with reference counting
#[repr(C)]
pub struct TributeBoxed {
    /// Type tag
    pub type_tag: TributeType,
    /// Reference count (atomic for thread safety)
    pub ref_count: AtomicU32,
    /// Value payload
    pub value: TributeValue,
}

/// Value union for different types
#[repr(C)]
pub union TributeValue {
    pub number: i64,
    pub boolean: bool,
    pub string: TributeString,
    pub function: TributeFunction,
    pub list: TributeList,
}

/// Generic array type for both strings and lists
#[repr(C)]
#[derive(Copy, Clone)]
pub struct TributeArray<T> {
    pub data: *mut T,
    pub length: usize,
    pub capacity: usize,
}

pub type TributeString = TributeArray<u8>;
pub type TributeList = TributeArray<*mut TributeBoxed>;

impl<T> TributeArray<T> {
    /// Create a new empty array with given capacity
    pub unsafe fn new_with_capacity(capacity: usize) -> Self {
        let capacity = if capacity == 0 { 4 } else { capacity };

        let data_layout =
            Layout::from_size_align(capacity * mem::size_of::<T>(), mem::align_of::<T>()).unwrap();
        let data = unsafe { alloc(data_layout) as *mut T };

        if data.is_null() {
            panic!("Failed to allocate memory for array data");
        }

        Self {
            data,
            length: 0,
            capacity,
        }
    }

    /// Get the current length
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the current capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Deallocate the array data
    pub unsafe fn deallocate(&self) {
        if !self.data.is_null() {
            unsafe {
                let layout = Layout::from_size_align_unchecked(
                    self.capacity * mem::size_of::<T>(),
                    mem::align_of::<T>(),
                );
                dealloc(self.data as *mut u8, layout);
            }
        }
    }

    /// Get element at index (unsafe)
    pub unsafe fn get_unchecked(&self, index: usize) -> T
    where
        T: Copy,
    {
        unsafe { *self.data.add(index) }
    }

    /// Set element at index (unsafe)
    pub unsafe fn set_unchecked(&mut self, index: usize, value: T) {
        unsafe {
            *self.data.add(index) = value;
        }
    }

    /// Resize the array to new capacity
    pub unsafe fn resize(&mut self, new_capacity: usize) {
        let new_layout =
            Layout::from_size_align(new_capacity * mem::size_of::<T>(), mem::align_of::<T>())
                .unwrap();
        let new_data = unsafe { alloc(new_layout) as *mut T };

        if new_data.is_null() {
            panic!("Failed to allocate memory for array resize");
        }

        // Copy existing elements
        if !self.data.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(self.data, new_data, self.length);
            }

            // Deallocate old data
            unsafe {
                let old_layout = Layout::from_size_align_unchecked(
                    self.capacity * mem::size_of::<T>(),
                    mem::align_of::<T>(),
                );
                dealloc(self.data as *mut u8, old_layout);
            }
        }

        self.data = new_data;
        self.capacity = new_capacity;
    }
}

impl TributeList {
    /// Initialize all pointers to null (for list arrays)
    pub unsafe fn init_null_pointers(&mut self) {
        for i in 0..self.capacity {
            unsafe {
                *self.data.add(i) = ptr::null_mut();
            }
        }
    }

    /// Push an element to the list with automatic resizing
    pub unsafe fn push(&mut self, value: *mut TributeBoxed) {
        // Check if we need to resize
        if self.length >= self.capacity {
            let new_capacity = self.capacity * 2;
            unsafe {
                self.resize(new_capacity);
            }

            // Initialize new slots to null
            for i in self.length..self.capacity {
                unsafe {
                    *self.data.add(i) = ptr::null_mut();
                }
            }
        }

        // Add the new element
        unsafe {
            *self.data.add(self.length) = value;
        }
        self.length += 1;
    }

    /// Pop the last element from the list
    pub unsafe fn pop(&mut self) -> *mut TributeBoxed {
        if self.length == 0 {
            return ptr::null_mut();
        }

        self.length -= 1;
        let element = unsafe { *self.data.add(self.length) };
        unsafe {
            *self.data.add(self.length) = ptr::null_mut();
        }
        element
    }

    /// Release all elements in the list
    pub unsafe fn release_all_elements(&self) {
        for i in 0..self.length {
            let element = unsafe { *self.data.add(i) };
            if !element.is_null() {
                unsafe {
                    (*element).release();
                }
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TributeFunction {
    pub code_ptr: *mut u8,
    pub env_ptr: *mut TributeBoxed,
}

impl TributeBoxed {
    /// Create a new boxed value with ref_count = 1
    pub fn new(type_tag: TributeType, value: TributeValue) -> NonNull<Self> {
        let layout = Layout::new::<Self>();
        let ptr = unsafe { alloc(layout) as *mut Self };

        if ptr.is_null() {
            panic!("Failed to allocate memory for TributeBoxed");
        }

        unsafe {
            ptr::write(
                ptr,
                Self {
                    type_tag,
                    ref_count: AtomicU32::new(1),
                    value,
                },
            );
        }

        unsafe { NonNull::new_unchecked(ptr) }
    }

    /// Increment reference count
    pub fn retain(&self) -> u32 {
        self.ref_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement reference count and deallocate if it reaches 0
    pub fn release(&self) -> bool {
        let old_count = self.ref_count.fetch_sub(1, Ordering::AcqRel);

        if old_count == 1 {
            // Last reference, need to deallocate
            unsafe { self.deallocate() };
            true
        } else {
            false
        }
    }

    /// Deallocate the boxed value and its contents
    unsafe fn deallocate(&self) {
        unsafe {
            // First, deallocate the contents based on type
            match self.type_tag {
                TributeType::String => {
                    self.value.string.deallocate();
                }
                TributeType::Function => {
                    let func = self.value.function;
                    // Release environment if it exists
                    if !func.env_ptr.is_null() {
                        (*func.env_ptr).release();
                    }
                }
                TributeType::List => {
                    let list_data = self.value.list;
                    list_data.release_all_elements();
                    list_data.deallocate();
                }
                _ => {
                    // Number, Boolean, Nil don't need special cleanup
                }
            }

            // Finally, deallocate the boxed value itself
            let layout = Layout::new::<Self>();
            dealloc(self as *const Self as *mut u8, layout);
        }
    }
}

/// C-compatible functions for MLIR integration
/// Box a number value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_number(value: i64) -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(TributeType::Number, TributeValue { number: value });
    boxed.as_ptr()
}

/// Box a string value (takes ownership of the string data)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_string(data: *mut u8, length: usize) -> *mut TributeBoxed {
    let string = TributeString {
        data,
        length,
        capacity: length,
    };

    let boxed = TributeBoxed::new(TributeType::String, TributeValue { string });
    boxed.as_ptr()
}

/// Box a boolean value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_boolean(value: bool) -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(TributeType::Boolean, TributeValue { boolean: value });
    boxed.as_ptr()
}

/// Box a nil value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_nil() -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(
        TributeType::Nil,
        TributeValue { number: 0 }, // Nil doesn't use the value
    );
    boxed.as_ptr()
}

/// Box a function value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_function(
    code_ptr: *mut u8,
    env_ptr: *mut TributeBoxed,
) -> *mut TributeBoxed {
    let function = TributeFunction { code_ptr, env_ptr };

    // Retain the environment if it exists
    if !env_ptr.is_null() {
        unsafe {
            (*env_ptr).retain();
        }
    }

    let boxed = TributeBoxed::new(TributeType::Function, TributeValue { function });
    boxed.as_ptr()
}

/// Create an empty list with initial capacity
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_list_empty(initial_capacity: usize) -> *mut TributeBoxed {
    let mut list = unsafe { TributeList::new_with_capacity(initial_capacity) };
    unsafe {
        list.init_null_pointers();
    }

    let boxed = TributeBoxed::new(TributeType::List, TributeValue { list });
    boxed.as_ptr()
}

/// Create a list from an array of boxed values
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid arrays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_box_list_from_array(
    elements: *mut *mut TributeBoxed,
    count: usize,
) -> *mut TributeBoxed {
    unsafe {
        let list_boxed = tribute_box_list_empty(count);

        if count == 0 {
            return list_boxed;
        }

        let list_data = &mut (*list_boxed).value.list;

        // Copy elements and retain them
        for i in 0..count {
            let element = *elements.add(i);
            if !element.is_null() {
                (*element).retain(); // Increment ref count
                *list_data.data.add(i) = element;
            }
        }

        list_data.length = count;

        list_boxed
    }
}

/// Unbox a number (with type checking)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_number(boxed: *mut TributeBoxed) -> i64 {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        if (*boxed).type_tag != TributeType::Number {
            panic!("Type error: expected Number, got {:?}", (*boxed).type_tag);
        }
        (*boxed).value.number
    }
}

/// Unbox a string (returns pointer to string data and length)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_string(
    boxed: *mut TributeBoxed,
    length_out: *mut usize,
) -> *mut u8 {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        if (*boxed).type_tag != TributeType::String {
            panic!("Type error: expected String, got {:?}", (*boxed).type_tag);
        }
        let string = (*boxed).value.string;
        if !length_out.is_null() {
            *length_out = string.length;
        }
        string.data
    }
}

/// Unbox a boolean
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_boolean(boxed: *mut TributeBoxed) -> bool {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        if (*boxed).type_tag != TributeType::Boolean {
            panic!("Type error: expected Boolean, got {:?}", (*boxed).type_tag);
        }
        (*boxed).value.boolean
    }
}

/// Unbox a function (returns code pointer and environment pointer)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_unbox_function(
    boxed: *mut TributeBoxed,
    env_out: *mut *mut TributeBoxed,
) -> *mut u8 {
    unsafe {
        if boxed.is_null() {
            panic!("Attempted to unbox null pointer");
        }

        if (*boxed).type_tag != TributeType::Function {
            panic!("Type error: expected Function, got {:?}", (*boxed).type_tag);
        }

        let function = (*boxed).value.function;

        if !env_out.is_null() {
            *env_out = function.env_ptr;
            // Retain the environment for the caller
            if !function.env_ptr.is_null() {
                (*function.env_ptr).retain();
            }
        }

        function.code_ptr
    }
}

/// Increment reference count
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_retain(boxed: *mut TributeBoxed) -> *mut TributeBoxed {
    unsafe {
        if !boxed.is_null() {
            (*boxed).retain();
        }
        boxed
    }
}

/// Decrement reference count
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_release(boxed: *mut TributeBoxed) {
    unsafe {
        if !boxed.is_null() {
            (*boxed).release();
        }
    }
}

/// Get current reference count (for debugging)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_get_ref_count(boxed: *mut TributeBoxed) -> u32 {
    unsafe {
        if boxed.is_null() {
            return 0;
        }
        (*boxed).ref_count.load(Ordering::Acquire)
    }
}

/// Check type of boxed value
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_get_type(boxed: *mut TributeBoxed) -> u32 {
    unsafe {
        if boxed.is_null() {
            return TributeType::Nil as u32;
        }
        (*boxed).type_tag as u32
    }
}

/// Arithmetic operations on boxed values
/// Add two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_add_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val + right_val;

        // Release input arguments (consumed by the operation)
        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Subtract two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_sub_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val - right_val;

        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Multiply two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_mul_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val * right_val;

        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Divide two boxed numbers
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_div_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);

        if right_val == 0 {
            panic!("Division by zero");
        }

        let result = left_val / right_val;

        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_number(result)
    }
}

/// Compare two boxed numbers for equality
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_eq_boxed(
    lhs: *mut TributeBoxed,
    rhs: *mut TributeBoxed,
) -> *mut TributeBoxed {
    unsafe {
        let left_val = tribute_unbox_number(lhs);
        let right_val = tribute_unbox_number(rhs);
        let result = left_val == right_val;

        tribute_release(lhs);
        tribute_release(rhs);

        tribute_box_boolean(result)
    }
}

/// Get list length - O(1)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_list_length(list_boxed: *mut TributeBoxed) -> usize {
    unsafe {
        if list_boxed.is_null() {
            return 0;
        }

        if (*list_boxed).type_tag != TributeType::List {
            panic!(
                "Type error: expected List, got {:?}",
                (*list_boxed).type_tag
            );
        }

        (*list_boxed).value.list.len()
    }
}

/// Get element at index - O(1)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_list_get(
    list_boxed: *mut TributeBoxed,
    index: usize,
) -> *mut TributeBoxed {
    unsafe {
        if list_boxed.is_null() {
            panic!("Attempted to access null list");
        }

        if (*list_boxed).type_tag != TributeType::List {
            panic!(
                "Type error: expected List, got {:?}",
                (*list_boxed).type_tag
            );
        }

        let list_data = &(*list_boxed).value.list;
        if index >= list_data.len() {
            panic!(
                "Index {} out of bounds for list of length {}",
                index,
                list_data.len()
            );
        }

        if list_data.data.is_null() {
            panic!("List data is null");
        }

        let element = list_data.get_unchecked(index);
        if !element.is_null() {
            (*element).retain(); // Caller owns a reference
        }
        element
    }
}

/// Set element at index - O(1)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_list_set(
    list_boxed: *mut TributeBoxed,
    index: usize,
    value: *mut TributeBoxed,
) {
    unsafe {
        if list_boxed.is_null() {
            panic!("Attempted to modify null list");
        }

        if (*list_boxed).type_tag != TributeType::List {
            panic!(
                "Type error: expected List, got {:?}",
                (*list_boxed).type_tag
            );
        }

        let list_data = &mut (*list_boxed).value.list;
        if index >= list_data.len() {
            panic!(
                "Index {} out of bounds for list of length {}",
                index,
                list_data.len()
            );
        }

        if list_data.data.is_null() {
            panic!("List data is null");
        }

        // Release old value
        let old_element = list_data.get_unchecked(index);
        if !old_element.is_null() {
            (*old_element).release();
        }

        // Set new value and retain it
        if !value.is_null() {
            (*value).retain();
        }
        list_data.set_unchecked(index, value);
    }
}

/// Append element to list - Amortized O(1)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_list_push(
    list_boxed: *mut TributeBoxed,
    value: *mut TributeBoxed,
) {
    unsafe {
        if list_boxed.is_null() {
            panic!("Attempted to push to null list");
        }

        if (*list_boxed).type_tag != TributeType::List {
            panic!(
                "Type error: expected List, got {:?}",
                (*list_boxed).type_tag
            );
        }

        let list_data = &mut (*list_boxed).value.list;

        // Retain the value before pushing
        if !value.is_null() {
            (*value).retain();
        }

        list_data.push(value);
    }
}

/// Pop last element from list - O(1)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_list_pop(list_boxed: *mut TributeBoxed) -> *mut TributeBoxed {
    unsafe {
        if list_boxed.is_null() {
            panic!("Attempted to pop from null list");
        }

        if (*list_boxed).type_tag != TributeType::List {
            panic!(
                "Type error: expected List, got {:?}",
                (*list_boxed).type_tag
            );
        }

        let list_data = &mut (*list_boxed).value.list;
        list_data.pop()
    }
}

/// Print a boxed value (for debugging)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_print_boxed(boxed: *mut TributeBoxed) {
    unsafe {
        if boxed.is_null() {
            println!("nil");
            return;
        }

        match (*boxed).type_tag {
            TributeType::Number => {
                println!("{}", (*boxed).value.number);
            }
            TributeType::Boolean => {
                println!("{}", (*boxed).value.boolean);
            }
            TributeType::String => {
                let string = (*boxed).value.string;
                if !string.data.is_null() {
                    let slice = core::slice::from_raw_parts(string.data, string.length);
                    if let Ok(_s) = core::str::from_utf8(slice) {
                        println!("[string content]");
                    } else {
                        println!("[invalid UTF-8 string]");
                    }
                } else {
                    println!("[null string]");
                }
            }
            TributeType::List => {
                let list_data = &(*boxed).value.list;
                if list_data.data.is_null() {
                    println!("[]");
                } else {
                    print!("[");
                    for i in 0..list_data.len() {
                        if i > 0 {
                            print!(", ");
                        }
                        let element = list_data.get_unchecked(i);
                        if element.is_null() {
                            print!("nil");
                        } else {
                            // Recursively print element (without releasing it)
                            match (*element).type_tag {
                                TributeType::Number => print!("{}", (*element).value.number),
                                TributeType::Boolean => print!("{}", (*element).value.boolean),
                                TributeType::String => print!("[string]"),
                                TributeType::Nil => print!("nil"),
                                _ => print!("[nested]"),
                            }
                        }
                    }
                    println!("]");
                }
            }
            TributeType::Nil => {
                println!("nil");
            }
            TributeType::Function => {
                println!("[function]");
            }
        }
    }
}

// Uses standard library println! for output

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_box_unbox_string() {
        unsafe {
            // Create a test string
            let test_str = "Hello, World!";
            let data = test_str.as_bytes().as_ptr() as *mut u8;
            let length = test_str.len();

            // Box the string
            let boxed = tribute_box_string(data, length);
            assert!(!boxed.is_null());

            // Check type
            assert_eq!(tribute_get_type(boxed), TributeType::String as u32);

            // Unbox the string
            let mut out_length = 0;
            let unboxed_data = tribute_unbox_string(boxed, &mut out_length);
            assert_eq!(out_length, length);
            assert!(!unboxed_data.is_null());

            // Verify content
            let slice = std::slice::from_raw_parts(unboxed_data, out_length);
            let unboxed_str = std::str::from_utf8(slice).unwrap();
            assert_eq!(unboxed_str, test_str);

            // Clean up
            tribute_release(boxed);
        }
    }

    #[test]
    fn test_box_unbox_number() {
        unsafe {
            let value = 42i64;

            // Box the number
            let boxed = tribute_box_number(value);
            assert!(!boxed.is_null());

            // Check type
            assert_eq!(tribute_get_type(boxed), TributeType::Number as u32);

            // Unbox the number
            let unboxed = tribute_unbox_number(boxed);
            assert_eq!(unboxed, value);

            // Clean up
            tribute_release(boxed);
        }
    }

    #[test]
    fn test_list_operations() {
        unsafe {
            // Create an empty list
            let list = tribute_box_list_empty(10);
            assert!(!list.is_null());

            // Check type
            assert_eq!(tribute_get_type(list), TributeType::List as u32);

            // Check initial length
            assert_eq!(tribute_list_length(list), 0);

            // Push some numbers
            let num1 = tribute_box_number(10);
            let num2 = tribute_box_number(20);
            let num3 = tribute_box_number(30);

            tribute_list_push(list, num1);
            tribute_list_push(list, num2);
            tribute_list_push(list, num3);

            assert_eq!(tribute_list_length(list), 3);

            // Get elements
            let elem0 = tribute_list_get(list, 0);
            let elem1 = tribute_list_get(list, 1);
            let elem2 = tribute_list_get(list, 2);

            assert_eq!(tribute_unbox_number(elem0), 10);
            assert_eq!(tribute_unbox_number(elem1), 20);
            assert_eq!(tribute_unbox_number(elem2), 30);

            // Clean up
            tribute_release(elem0);
            tribute_release(elem1);
            tribute_release(elem2);
            tribute_release(list);
        }
    }
}

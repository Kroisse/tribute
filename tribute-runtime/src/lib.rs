//! Tribute Runtime Library
//! 
//! This library provides the runtime support for Tribute programs,
//! including garbage collection, boxing/unboxing, and builtin functions.

use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::{self, NonNull},
    sync::atomic::{AtomicU32, Ordering},
    mem,
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
    pub function: *mut TributeFunction,
    pub list: *mut TributeList,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TributeString {
    pub data: *mut u8,
    pub length: usize,
    pub capacity: usize,
}

#[repr(C)]
pub struct TributeFunction {
    pub code_ptr: *mut u8,
    pub env_ptr: *mut TributeBoxed,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TributeList {
    /// Pointer to array of TributeBoxed pointers
    pub data: *mut *mut TributeBoxed,
    /// Number of elements currently in the list
    pub length: usize,
    /// Total capacity of the allocated array
    pub capacity: usize,
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
            ptr::write(ptr, Self {
                type_tag,
                ref_count: AtomicU32::new(1),
                value,
            });
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
                    let string_data = self.value.string;
                    if !string_data.data.is_null() {
                        let layout = Layout::from_size_align_unchecked(
                            string_data.capacity, 
                            mem::align_of::<u8>()
                        );
                        dealloc(string_data.data, layout);
                    }
                }
                TributeType::Function => {
                    let func = self.value.function;
                    if !func.is_null() {
                        // Release environment if it exists
                        if !(*func).env_ptr.is_null() {
                            (*(*func).env_ptr).release();
                        }
                        // Deallocate function struct
                        let layout = Layout::new::<TributeFunction>();
                        dealloc(func as *mut u8, layout);
                    }
                }
                TributeType::List => {
                    let list = self.value.list;
                    if !list.is_null() {
                        let list_data = *list;
                        
                        // Release all elements in the array
                        if !list_data.data.is_null() {
                            for i in 0..list_data.length {
                                let element = *list_data.data.add(i);
                                if !element.is_null() {
                                    (*element).release();
                                }
                            }
                            
                            // Deallocate the data array
                            let data_layout = Layout::from_size_align_unchecked(
                                list_data.capacity * mem::size_of::<*mut TributeBoxed>(),
                                mem::align_of::<*mut TributeBoxed>()
                            );
                            dealloc(list_data.data as *mut u8, data_layout);
                        }
                        
                        // Deallocate the list structure itself
                        let layout = Layout::new::<TributeList>();
                        dealloc(list as *mut u8, layout);
                    }
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
    let boxed = TributeBoxed::new(
        TributeType::Number,
        TributeValue { number: value }
    );
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
    
    let boxed = TributeBoxed::new(
        TributeType::String,
        TributeValue { string }
    );
    boxed.as_ptr()
}

/// Box a boolean value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_boolean(value: bool) -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(
        TributeType::Boolean,
        TributeValue { boolean: value }
    );
    boxed.as_ptr()
}

/// Box a nil value
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_nil() -> *mut TributeBoxed {
    let boxed = TributeBoxed::new(
        TributeType::Nil,
        TributeValue { number: 0 } // Nil doesn't use the value
    );
    boxed.as_ptr()
}

/// Create an empty list with initial capacity
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_list_empty(initial_capacity: usize) -> *mut TributeBoxed {
    let capacity = if initial_capacity == 0 { 4 } else { initial_capacity };
    
    // Allocate the data array
    let data_layout = Layout::from_size_align(
        capacity * mem::size_of::<*mut TributeBoxed>(),
        mem::align_of::<*mut TributeBoxed>()
    ).unwrap();
    let data = unsafe { alloc(data_layout) as *mut *mut TributeBoxed };
    
    if data.is_null() {
        panic!("Failed to allocate memory for list data");
    }
    
    // Initialize all pointers to null
    unsafe {
        for i in 0..capacity {
            *data.add(i) = ptr::null_mut();
        }
    }
    
    // Allocate the list structure
    let list_layout = Layout::new::<TributeList>();
    let list_ptr = unsafe { alloc(list_layout) as *mut TributeList };
    
    if list_ptr.is_null() {
        unsafe { dealloc(data as *mut u8, data_layout); }
        panic!("Failed to allocate memory for list structure");
    }
    
    unsafe {
        ptr::write(list_ptr, TributeList {
            data,
            length: 0,
            capacity,
        });
    }
    
    let boxed = TributeBoxed::new(
        TributeType::List,
        TributeValue { list: list_ptr }
    );
    boxed.as_ptr()
}

/// Create a list from an array of boxed values
/// 
/// # Safety
/// This function dereferences raw pointers and should only be called with valid arrays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_box_list_from_array(
    elements: *mut *mut TributeBoxed, 
    count: usize
) -> *mut TributeBoxed {
    unsafe {
        let list_boxed = tribute_box_list_empty(count);
        
        if count == 0 {
            return list_boxed;
        }
        
        let list = (*list_boxed).value.list;
        let list_data = &mut *list;
        
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
pub unsafe extern "C" fn tribute_unbox_string(boxed: *mut TributeBoxed, length_out: *mut usize) -> *mut u8 {
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
pub unsafe extern "C" fn tribute_add_boxed(lhs: *mut TributeBoxed, rhs: *mut TributeBoxed) -> *mut TributeBoxed {
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
pub unsafe extern "C" fn tribute_sub_boxed(lhs: *mut TributeBoxed, rhs: *mut TributeBoxed) -> *mut TributeBoxed {
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
pub unsafe extern "C" fn tribute_mul_boxed(lhs: *mut TributeBoxed, rhs: *mut TributeBoxed) -> *mut TributeBoxed {
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
pub unsafe extern "C" fn tribute_div_boxed(lhs: *mut TributeBoxed, rhs: *mut TributeBoxed) -> *mut TributeBoxed {
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
pub unsafe extern "C" fn tribute_eq_boxed(lhs: *mut TributeBoxed, rhs: *mut TributeBoxed) -> *mut TributeBoxed {
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
            panic!("Type error: expected List, got {:?}", (*list_boxed).type_tag);
        }
        
        let list = (*list_boxed).value.list;
        if list.is_null() {
            return 0;
        }
        
        (*list).length
    }
}

/// Get element at index - O(1)
/// 
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_list_get(list_boxed: *mut TributeBoxed, index: usize) -> *mut TributeBoxed {
    unsafe {
        if list_boxed.is_null() {
            panic!("Attempted to access null list");
        }
        
        if (*list_boxed).type_tag != TributeType::List {
            panic!("Type error: expected List, got {:?}", (*list_boxed).type_tag);
        }
        
        let list = (*list_boxed).value.list;
        if list.is_null() {
            panic!("List data is null");
        }
        
        let list_data = &*list;
        if index >= list_data.length {
            panic!("Index {} out of bounds for list of length {}", index, list_data.length);
        }
        
        let element = *list_data.data.add(index);
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
    value: *mut TributeBoxed
) {
    unsafe {
        if list_boxed.is_null() {
            panic!("Attempted to modify null list");
        }
        
        if (*list_boxed).type_tag != TributeType::List {
            panic!("Type error: expected List, got {:?}", (*list_boxed).type_tag);
        }
        
        let list = (*list_boxed).value.list;
        if list.is_null() {
            panic!("List data is null");
        }
        
        let list_data = &mut *list;
        if index >= list_data.length {
            panic!("Index {} out of bounds for list of length {}", index, list_data.length);
        }
        
        // Release old value
        let old_element = *list_data.data.add(index);
        if !old_element.is_null() {
            (*old_element).release();
        }
        
        // Set new value and retain it
        if !value.is_null() {
            (*value).retain();
        }
        *list_data.data.add(index) = value;
    }
}

/// Append element to list - Amortized O(1)
/// 
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_list_push(list_boxed: *mut TributeBoxed, value: *mut TributeBoxed) {
    unsafe {
        if list_boxed.is_null() {
            panic!("Attempted to push to null list");
        }
        
        if (*list_boxed).type_tag != TributeType::List {
            panic!("Type error: expected List, got {:?}", (*list_boxed).type_tag);
        }
        
        let list = (*list_boxed).value.list;
        if list.is_null() {
            panic!("List data is null");
        }
        
        let list_data = &mut *list;
        
        // Check if we need to resize
        if list_data.length >= list_data.capacity {
            // Double the capacity
            let new_capacity = list_data.capacity * 2;
            let new_layout = Layout::from_size_align(
                new_capacity * mem::size_of::<*mut TributeBoxed>(),
                mem::align_of::<*mut TributeBoxed>()
            ).unwrap();
            let new_data = alloc(new_layout) as *mut *mut TributeBoxed;
            
            if new_data.is_null() {
                panic!("Failed to allocate memory for list resize");
            }
            
            // Copy existing elements
            for i in 0..list_data.length {
                *new_data.add(i) = *list_data.data.add(i);
            }
            
            // Initialize new slots to null
            for i in list_data.length..new_capacity {
                *new_data.add(i) = ptr::null_mut();
            }
            
            // Deallocate old data
            let old_layout = Layout::from_size_align_unchecked(
                list_data.capacity * mem::size_of::<*mut TributeBoxed>(),
                mem::align_of::<*mut TributeBoxed>()
            );
            dealloc(list_data.data as *mut u8, old_layout);
            
            // Update list data
            list_data.data = new_data;
            list_data.capacity = new_capacity;
        }
        
        // Add the new element
        if !value.is_null() {
            (*value).retain();
        }
        *list_data.data.add(list_data.length) = value;
        list_data.length += 1;
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
            panic!("Type error: expected List, got {:?}", (*list_boxed).type_tag);
        }
        
        let list = (*list_boxed).value.list;
        if list.is_null() {
            panic!("List data is null");
        }
        
        let list_data = &mut *list;
        if list_data.length == 0 {
            return ptr::null_mut(); // Empty list
        }
        
        list_data.length -= 1;
        let element = *list_data.data.add(list_data.length);
        *list_data.data.add(list_data.length) = ptr::null_mut();
        
        // Element already has correct ref count (transfer ownership to caller)
        element
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
                    let list = (*boxed).value.list;
                    if list.is_null() {
                        println!("[]");
                    } else {
                        let list_data = &*list;
                        print!("[");
                        for i in 0..list_data.length {
                            if i > 0 { print!(", "); }
                            let element = *list_data.data.add(i);
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
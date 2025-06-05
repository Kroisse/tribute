use std::ptr;
use crate::{array::TributeArray, value::{TributeBoxed, TributeValue}};

pub type TributeList = TributeArray<*mut TributeBoxed>;

impl TributeList {
    /// Initialize all pointers to null (for list arrays)
    /// 
    /// # Safety
    /// This function assumes the array has been properly allocated.
    pub unsafe fn init_null_pointers(&mut self) {
        for i in 0..self.capacity {
            unsafe {
                *self.data.add(i) = ptr::null_mut();
            }
        }
    }

    /// Push an element to the list with automatic resizing
    /// 
    /// # Safety
    /// This function manipulates raw pointers and may reallocate memory.
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
    /// 
    /// # Safety
    /// This function manipulates raw pointers and assumes the list is properly initialized.
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
    /// 
    /// # Safety
    /// This function dereferences pointers and calls release on them.
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

/// Create an empty list with initial capacity
#[unsafe(no_mangle)]
pub extern "C" fn tribute_box_list_empty(initial_capacity: usize) -> *mut TributeBoxed {
    let mut list = unsafe { TributeList::new_with_capacity(initial_capacity) };
    unsafe {
        list.init_null_pointers();
    }

    let boxed = TributeBoxed::new(TributeValue::List(list));
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

        match &mut (*list_boxed).value {
            TributeValue::List(list_data) => {
                // Copy elements and retain them
                for i in 0..count {
                    let element = *elements.add(i);
                    if !element.is_null() {
                        (*element).retain(); // Increment ref count
                        *list_data.data.add(i) = element;
                    }
                }

                list_data.length = count;
            }
            _ => panic!("Expected list value"),
        }

        list_boxed
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

        match &(*list_boxed).value {
            TributeValue::List(list) => list.len(),
            _ => panic!("Type error: expected List, got different type"),
        }
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

        match &(*list_boxed).value {
            TributeValue::List(list_data) => {
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
            _ => panic!("Type error: expected List, got different type"),
        }
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

        match &mut (*list_boxed).value {
            TributeValue::List(list_data) => {
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
            _ => panic!("Type error: expected List, got different type"),
        }
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

        match &mut (*list_boxed).value {
            TributeValue::List(list_data) => {
                // Retain the value before pushing
                if !value.is_null() {
                    (*value).retain();
                }

                list_data.push(value);
            }
            _ => panic!("Type error: expected List, got different type"),
        }
    }
}

/// Pop last element from list - O(1)
///
/// # Safety
/// This function dereferences raw pointers and should only be called with valid TributeBoxed pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tribute_list_pop(list_boxed: *mut TributeBoxed) -> *mut TributeBoxed {
    unsafe {
        match &mut (*list_boxed).value {
            TributeValue::List(list_data) => list_data.pop(),
            _ => panic!("Type error: expected List, got different type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tribute_box_number, tribute_unbox_number, tribute_get_type, tribute_release};

    #[test]
    fn test_list_operations() {
        unsafe {
            // Create an empty list
            let list = tribute_box_list_empty(10);
            assert!(!list.is_null());

            // Check type
            assert_eq!(tribute_get_type(list), TributeValue::TYPE_LIST);

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
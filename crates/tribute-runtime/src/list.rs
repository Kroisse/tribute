//! Handle-based list operations for the Tribute runtime

use crate::{
    array::TributeArray,
    handle::{TributeHandle, HandleTable},
    value::{TributeBoxed, TributeValue},
};
use std::sync::LazyLock;

// Temporary global handle table for list operations - TODO: Migrate to context-aware API
static LIST_HANDLE_TABLE: LazyLock<HandleTable> = LazyLock::new(HandleTable::new);

pub type TributeList = TributeArray<TributeHandle>;

impl TributeList {
    /// Initialize all handles to null (for list arrays)
    pub unsafe fn init_null_handles(&mut self) {
        for i in 0..self.capacity {
            unsafe {
                *self.data.add(i) = crate::handle::TRIBUTE_HANDLE_INVALID;
            }
        }
    }

    /// Push a handle to the list with automatic resizing
    pub unsafe fn push_handle(&mut self, handle: TributeHandle) {
        // Check if we need to resize
        if self.length >= self.capacity {
            let new_capacity = self.capacity * 2;
            unsafe {
                self.resize(new_capacity);
            }

            // Initialize new slots to null
            for i in self.length..self.capacity {
                unsafe {
                    *self.data.add(i) = crate::handle::TRIBUTE_HANDLE_INVALID;
                }
            }
        }

        // Add the new handle
        unsafe {
            *self.data.add(self.length) = handle;
        }
        self.length += 1;
    }

    /// Pop the last handle from the list
    pub unsafe fn pop_handle(&mut self) -> TributeHandle {
        if self.length == 0 {
            return crate::handle::TRIBUTE_HANDLE_INVALID;
        }

        self.length -= 1;
        let handle = unsafe { *self.data.add(self.length) };
        unsafe {
            *self.data.add(self.length) = crate::handle::TRIBUTE_HANDLE_INVALID;
        }
        handle
    }

    /// Release all handles in the list (calls handle release logic)
    pub fn release_all_handles(&self) {
        // In a full implementation, this would call the handle release logic
        // For now, we'll just mark it as a no-op since handles manage their own lifetime
    }
}

/// Create an empty list with initial capacity (returns handle)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_new_list_empty(initial_capacity: usize) -> TributeHandle {
    let mut list = unsafe { TributeList::new_with_capacity(initial_capacity) };
    unsafe {
        list.init_null_handles();
    }

    let boxed = TributeBoxed::new(TributeValue::List(list));
    LIST_HANDLE_TABLE.create_handle(boxed)
}

/// Get list length - O(1)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_list_length(list_handle: TributeHandle) -> usize {
    LIST_HANDLE_TABLE
        .with_value(list_handle, |boxed| {
            match &boxed.value {
                TributeValue::List(list) => list.len(),
                _ => 0, // Type error
            }
        })
        .unwrap_or(0)
}

/// Get handle at index - O(1)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_list_get(list_handle: TributeHandle, index: usize) -> TributeHandle {
    LIST_HANDLE_TABLE
        .with_value(list_handle, |boxed| {
            match &boxed.value {
                TributeValue::List(list_data) => {
                    if index >= list_data.len() {
                        return crate::handle::TRIBUTE_HANDLE_INVALID; // Index out of bounds
                    }

                    if list_data.data.is_null() {
                        return crate::handle::TRIBUTE_HANDLE_INVALID; // List data is null
                    }

                    unsafe {
                        list_data.get_unchecked(index)
                    }
                }
                _ => crate::handle::TRIBUTE_HANDLE_INVALID, // Type error
            }
        })
        .unwrap_or(crate::handle::TRIBUTE_HANDLE_INVALID)
}

/// Set handle at index - O(1)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_list_set(list_handle: TributeHandle, index: usize, value_handle: TributeHandle) {
    LIST_HANDLE_TABLE.with_value_mut(list_handle, |boxed| {
        if let TributeValue::List(list_data) = &mut boxed.value {
            if index >= list_data.len() {
                return; // Index out of bounds
            }

            if list_data.data.is_null() {
                return; // List data is null
            }

            unsafe {
                list_data.set_unchecked(index, value_handle);
            }
        }
    });
}

/// Append handle to list - Amortized O(1)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_list_push(list_handle: TributeHandle, value_handle: TributeHandle) {
    LIST_HANDLE_TABLE.with_value_mut(list_handle, |boxed| {
        if let TributeValue::List(list_data) = &mut boxed.value {
            unsafe {
                list_data.push_handle(value_handle);
            }
        }
    });
}

/// Pop last handle from list - O(1)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_list_pop(list_handle: TributeHandle) -> TributeHandle {
    LIST_HANDLE_TABLE
        .with_value_mut(list_handle, |boxed| {
            match &mut boxed.value {
                TributeValue::List(list_data) => unsafe { list_data.pop_handle() },
                _ => crate::handle::TRIBUTE_HANDLE_INVALID, // Type error
            }
        })
        .unwrap_or(crate::handle::TRIBUTE_HANDLE_INVALID)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Temporary wrapper functions for list tests - TODO: Update tests to use context-aware API
    unsafe fn tribute_new_number(value: i64) -> crate::handle::TributeHandle {
        use crate::value::{TributeBoxed, TributeValue};
        let boxed = TributeBoxed::new(TributeValue::Number(value));
        LIST_HANDLE_TABLE.create_handle(boxed)
    }
    
    unsafe fn tribute_release(handle: crate::handle::TributeHandle) {
        use crate::handle::{INTERNED_TRUE, INTERNED_FALSE, INTERNED_NIL, INTERNED_EMPTY_STRING};
        
        // Never release interned values
        if handle == INTERNED_TRUE || handle == INTERNED_FALSE || handle == INTERNED_NIL || handle == INTERNED_EMPTY_STRING {
            return;
        }

        let should_deallocate = LIST_HANDLE_TABLE
            .with_value(handle, |boxed| boxed.release() == 0)
            .unwrap_or(false);

        if should_deallocate {
            LIST_HANDLE_TABLE.release(handle);
        }
    }
    
    unsafe fn tribute_unbox_number(handle: crate::handle::TributeHandle) -> i64 {
        LIST_HANDLE_TABLE
            .with_value(handle, |boxed| {
                match &boxed.value {
                    TributeValue::Number(n) => *n,
                    _ => 0,
                }
            })
            .unwrap_or(0)
    }

    #[test]
    fn test_handle_list_operations() {
        unsafe {
            // Create an empty list
            let list = tribute_new_list_empty(10);
            assert_ne!(list, crate::handle::TRIBUTE_HANDLE_INVALID);

            // Check initial length
            assert_eq!(tribute_list_length(list), 0);

            // Push some numbers
            let num1 = tribute_new_number(10);
            let num2 = tribute_new_number(20);
            let num3 = tribute_new_number(30);

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

            // Pop an element
            let popped = tribute_list_pop(list);
            assert_eq!(tribute_unbox_number(popped), 30);
            assert_eq!(tribute_list_length(list), 2);

            // Clean up
            tribute_release(num1);
            tribute_release(num2);
            tribute_release(num3);
            tribute_release(list);
        }
    }
}

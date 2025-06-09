//! Handle-based list operations for the Tribute runtime

use crate::{
    array::TributeArray,
    handle::TributeHandle,
    value::{TributeBoxed, TributeValue},
};

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
pub extern "C" fn tribute_handle_new_list_empty(initial_capacity: usize) -> TributeHandle {
    let mut list = unsafe { TributeList::new_with_capacity(initial_capacity) };
    unsafe {
        list.init_null_handles();
    }

    let boxed = TributeBoxed::new(TributeValue::List(list));
    crate::handle::create_handle(boxed)
}

/// Get list length - O(1)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_list_length(list_handle: TributeHandle) -> usize {
    list_handle
        .with_value(|boxed| {
            match &boxed.value {
                TributeValue::List(list) => list.len(),
                _ => 0, // Type error
            }
        })
        .unwrap_or(0)
}

/// Get handle at index - O(1)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_list_get(list_handle: TributeHandle, index: usize) -> TributeHandle {
    list_handle
        .with_value(|boxed| {
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
pub extern "C" fn tribute_handle_list_set(list_handle: TributeHandle, index: usize, value_handle: TributeHandle) {
    list_handle.with_value_mut(|boxed| {
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
pub extern "C" fn tribute_handle_list_push(list_handle: TributeHandle, value_handle: TributeHandle) {
    list_handle.with_value_mut(|boxed| {
        if let TributeValue::List(list_data) = &mut boxed.value {
            unsafe {
                list_data.push_handle(value_handle);
            }
        }
    });
}

/// Pop last handle from list - O(1)
#[unsafe(no_mangle)]
pub extern "C" fn tribute_handle_list_pop(list_handle: TributeHandle) -> TributeHandle {
    list_handle
        .with_value_mut(|boxed| {
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
    use crate::handle::{
        tribute_handle_new_number, tribute_handle_release, tribute_handle_unbox_number,
    };

    #[test]
    fn test_handle_list_operations() {
        // Create an empty list
        let list = tribute_handle_new_list_empty(10);
        assert_ne!(list, crate::handle::TRIBUTE_HANDLE_INVALID);

        // Check initial length
        assert_eq!(tribute_handle_list_length(list), 0);

        // Push some numbers
        let num1 = tribute_handle_new_number(10);
        let num2 = tribute_handle_new_number(20);
        let num3 = tribute_handle_new_number(30);

        tribute_handle_list_push(list, num1);
        tribute_handle_list_push(list, num2);
        tribute_handle_list_push(list, num3);

        assert_eq!(tribute_handle_list_length(list), 3);

        // Get elements
        let elem0 = tribute_handle_list_get(list, 0);
        let elem1 = tribute_handle_list_get(list, 1);
        let elem2 = tribute_handle_list_get(list, 2);

        assert_eq!(tribute_handle_unbox_number(elem0), 10);
        assert_eq!(tribute_handle_unbox_number(elem1), 20);
        assert_eq!(tribute_handle_unbox_number(elem2), 30);

        // Pop an element
        let popped = tribute_handle_list_pop(list);
        assert_eq!(tribute_handle_unbox_number(popped), 30);
        assert_eq!(tribute_handle_list_length(list), 2);

        // Clean up
        tribute_handle_release(num1);
        tribute_handle_release(num2);
        tribute_handle_release(num3);
        tribute_handle_release(list);
    }
}

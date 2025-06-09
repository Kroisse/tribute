use std::{
    alloc::{alloc, dealloc, Layout},
    mem,
    ptr,
};

/// Generic array type for both strings and lists
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TributeArray<T> {
    pub data: *mut T,
    pub length: usize,
    pub capacity: usize,
}

impl<T> TributeArray<T> {
    /// Create a new empty array with given capacity
    /// 
    /// # Safety
    /// This function allocates raw memory and should only be used with proper cleanup.
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
    /// 
    /// # Safety
    /// This function deallocates raw memory and should only be called once per array.
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
    /// 
    /// # Safety
    /// This function assumes the index is within bounds and the data pointer is valid.
    pub unsafe fn get_unchecked(&self, index: usize) -> T
    where
        T: Copy,
    {
        unsafe { *self.data.add(index) }
    }

    /// Set element at index (unsafe)
    /// 
    /// # Safety
    /// This function assumes the index is within bounds and the data pointer is valid.
    pub unsafe fn set_unchecked(&mut self, index: usize, value: T) {
        unsafe {
            *self.data.add(index) = value;
        }
    }

    /// Resize the array to new capacity
    /// 
    /// # Safety
    /// This function reallocates memory and should only be called with valid arrays.
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
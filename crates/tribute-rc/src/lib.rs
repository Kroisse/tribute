//! Reference-counted box layout for the Tribute runtime.
//!
//! All heap-allocated objects in Tribute are prefixed with an RC header
//! containing a reference count and an RTTI index. `RcBox<T>` encodes
//! this layout as a `#[repr(C)]` struct so that both the runtime and
//! compiler passes can work with it type-safely.
//!
//! # Memory Layout
//!
//! ```text
//! ┌──────────────┬──────────────┬─────────────────┐
//! │ refcount: u32│ rtti_idx: u32│ payload: T       │
//! │  (offset 0)  │  (offset 4)  │  (offset 8)      │
//! └──────────────┴──────────────┴─────────────────┘
//! ```

#![no_std]

use core::mem::offset_of;
use core::sync::atomic::{AtomicU32, Ordering};

/// A reference-counted box with an RTTI index.
///
/// `refcount` tracks the number of live references. `rtti_idx` identifies
/// the type for the release dispatcher (which destructor to call when the
/// refcount drops to zero).
#[repr(C)]
pub struct RcBox<T> {
    pub refcount: AtomicU32,
    pub rtti_idx: u32,
    pub payload: T,
}

// Field offsets derived from the struct layout. `#[repr(C)]` guarantees
// these are stable and independent of `T` for the header fields.

/// Byte offset of the `refcount` field from the start of `RcBox`.
pub const REFCOUNT_OFFSET: usize = offset_of!(RcBox<()>, refcount);

/// Byte offset of the `rtti_idx` field from the start of `RcBox`.
pub const RTTI_IDX_OFFSET: usize = offset_of!(RcBox<()>, rtti_idx);

/// Byte offset of the `payload` field from the start of `RcBox`.
/// This equals the total size of the RC header.
pub const PAYLOAD_OFFSET: usize = offset_of!(RcBox<()>, payload);

/// Size of the RC header in bytes (equivalent to `PAYLOAD_OFFSET`).
///
/// This is the value previously known as `RC_HEADER_SIZE`.
pub const HEADER_SIZE: u64 = PAYLOAD_OFFSET as u64;

impl<T> RcBox<T> {
    /// Initialize an `RcBox` at a raw allocation pointer.
    ///
    /// Sets `refcount = 1` and the given `rtti_idx`. Returns a mutable
    /// reference to the initialized `RcBox`.
    ///
    /// # Safety
    ///
    /// `raw` must point to a valid allocation of at least
    /// `size_of::<RcBox<T>>()` bytes, properly aligned for `RcBox<T>`.
    pub unsafe fn init(raw: *mut u8, rtti_idx: u32) -> &'static mut RcBox<T> {
        let rc_box = unsafe { &mut *(raw as *mut RcBox<T>) };
        rc_box.refcount = AtomicU32::new(1);
        rc_box.rtti_idx = rtti_idx;
        rc_box
    }

    /// Compute an `RcBox` pointer from a payload pointer.
    ///
    /// # Safety
    ///
    /// `payload_ptr` must point to the `payload` field of a valid `RcBox<T>`.
    pub unsafe fn from_payload_ptr(payload_ptr: *const T) -> *const RcBox<T> {
        unsafe { (payload_ptr as *const u8).sub(PAYLOAD_OFFSET) as *const RcBox<T> }
    }

    /// Compute a mutable `RcBox` pointer from a payload pointer.
    ///
    /// # Safety
    ///
    /// `payload_ptr` must point to the `payload` field of a valid `RcBox<T>`.
    pub unsafe fn from_payload_ptr_mut(payload_ptr: *mut T) -> *mut RcBox<T> {
        unsafe { (payload_ptr as *mut u8).sub(PAYLOAD_OFFSET) as *mut RcBox<T> }
    }

    /// Atomically increment the reference count.
    pub fn retain(&self) {
        self.refcount.fetch_add(1, Ordering::Relaxed);
    }

    /// Return a pointer to the payload field.
    pub fn payload_ptr(&mut self) -> *mut T {
        &raw mut self.payload
    }
}

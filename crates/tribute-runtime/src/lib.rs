//! Tribute runtime library.
//!
//! Provides the native runtime functions required by Tribute's compiled output:
//! - Heap allocation (`__tribute_alloc`, `__tribute_dealloc`)
//! - TLS-based tag generation for ability dispatch
//! - Evidence-based ability dispatch (`__tribute_evidence_*`)

#![cfg_attr(panic = "abort", no_std)]
#![allow(private_interfaces)]

extern crate alloc;

use alloc::boxed::Box;

use smallvec::SmallVec;

// =============================================================================
// Global allocator and panic handler (no_std)
//
// Wraps libc malloc/free for Rust's alloc crate, and aborts on panic.
// Only compiled when panic="abort" (i.e., the `runtime` profile).
// In dev/test builds, std provides these via the test harness.
// =============================================================================

#[cfg(all(not(test), panic = "abort"))]
mod no_std_runtime {
    use core::alloc::{GlobalAlloc, Layout};
    use core::ffi::c_void;

    struct CAllocator;

    #[cfg(unix)]
    unsafe extern "C" {
        fn posix_memalign(
            memptr: *mut *mut c_void,
            alignment: usize,
            size: usize,
        ) -> core::ffi::c_int;
        fn free(ptr: *mut c_void);
    }

    #[cfg(windows)]
    unsafe extern "system" {
        fn _aligned_malloc(size: usize, alignment: usize) -> *mut c_void;
        fn _aligned_free(ptr: *mut c_void);
    }

    #[cfg(unix)]
    unsafe impl GlobalAlloc for CAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let align = layout.align().max(core::mem::size_of::<*mut c_void>());
            let ret = unsafe { posix_memalign(&mut ptr, align, layout.size()) };
            if ret == 0 {
                ptr as *mut u8
            } else {
                core::ptr::null_mut()
            }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            unsafe { free(ptr as *mut c_void) }
        }
    }

    #[cfg(windows)]
    unsafe impl GlobalAlloc for CAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            unsafe { _aligned_malloc(layout.size(), layout.align()) as *mut u8 }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            unsafe { _aligned_free(ptr as *mut c_void) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CAllocator = CAllocator;

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        unsafe extern "C" {
            fn abort() -> !;
        }
        unsafe { abort() }
    }

    // Pre-compiled alloc/core reference this symbol even with panic="abort".
    // Provide a dummy stub so the staticlib links cleanly.
    #[unsafe(no_mangle)]
    pub extern "C" fn rust_eh_personality() {}
}

mod asan;
mod tls;

use tls::{thread_state, tls_init};

// =============================================================================
// Initialization
// =============================================================================

/// Initialize the Tribute runtime (must be called once before any ability use).
///
/// Sets up TLS for tag generation.
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_init() {
    unsafe {
        tls_init();
    }
}

// =============================================================================
// Prompt tag generation
// =============================================================================

/// Generate a unique prompt tag for this thread.
///
/// Each call returns a distinct i32, ensuring that recursive/nested handlers
/// for the same ability get different tags.
///
/// Signature: `() -> i32`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_next_tag() -> i32 {
    unsafe { thread_state() }.next_tag()
}

// =============================================================================
// Debug I/O (temporary — for e2e test verification)
// =============================================================================

unsafe extern "C" {
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
}

/// Print a signed 32-bit integer to stdout, followed by a newline.
///
/// Matches Tribute's `Int` type which maps to `core.i32` in TrunkIR.
///
/// Signature: `(value: i32) -> ()`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_print_int(value: i32) {
    let mut buf = itoa::Buffer::new();
    let s = buf.format(value);
    unsafe {
        write(1, s.as_ptr(), s.len());
        write(1, b"\n".as_ptr(), 1);
    }
}

/// Print an unsigned 32-bit integer to stdout, followed by a newline.
///
/// Matches Tribute's `Nat` type which maps to `core.i32` (unsigned interpretation) in TrunkIR.
///
/// Signature: `(value: u32) -> ()`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_print_nat(value: u32) {
    let mut buf = itoa::Buffer::new();
    let s = buf.format(value);
    unsafe {
        write(1, s.as_ptr(), s.len());
        write(1, b"\n".as_ptr(), 1);
    }
}

/// Print a 64-bit float to stdout, followed by a newline.
///
/// Matches Tribute's `Float` type which maps to `core.f64` in TrunkIR.
///
/// Signature: `(value: f64) -> ()`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_print_float(value: f64) {
    let mut buf = zmij::Buffer::new();
    let s = buf.format(value);
    unsafe {
        write(1, s.as_ptr(), s.len());
        write(1, b"\n".as_ptr(), 1);
    }
}

// =============================================================================
// Allocator
// =============================================================================

/// # Safety
///
/// Caller must eventually free the returned pointer via `__tribute_dealloc`
/// with the same `size`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_alloc(size: u64) -> *mut u8 {
    if size == 0 {
        return core::ptr::null_mut();
    }
    let Ok(size) = usize::try_from(size) else {
        return core::ptr::null_mut();
    };
    if asan::is_enabled() {
        return unsafe { asan::alloc(size) };
    }
    let Ok(layout) = core::alloc::Layout::from_size_align(size, 8) else {
        return core::ptr::null_mut();
    };
    unsafe { alloc::alloc::alloc(layout) }
}

/// # Safety
///
/// `ptr` must have been allocated by `__tribute_alloc` with the same `size`,
/// or be null (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_dealloc(ptr: *mut u8, size: u64) {
    if ptr.is_null() || size == 0 {
        return;
    }
    let Ok(size) = usize::try_from(size) else {
        return;
    };
    if asan::is_enabled() {
        return unsafe { asan::dealloc(ptr, size) };
    }
    let Ok(layout) = core::alloc::Layout::from_size_align(size, 8) else {
        return;
    };
    unsafe { alloc::alloc::dealloc(ptr, layout) };
}

// =============================================================================
// Evidence-based ability dispatch
// =============================================================================

/// Marker for a single ability handler in the evidence.
///
/// `#[repr(C)]` so Cranelift can access individual fields by offset.
///
/// `tr_dispatch_fn` is a pointer to a tail-resumptive dispatch function
/// `(op_idx: i32, shift_value: ptr) -> ptr`, or null if the handler is
/// not fully tail-resumptive.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Marker {
    pub ability_id: i32,
    pub prompt_tag: i32,
    pub tr_dispatch_fn: *const u8,
}

/// Opaque evidence structure — a sorted array of `Marker`s keyed by `ability_id`.
///
/// IR-level code only sees `core.ptr`; this struct is never exposed across FFI
/// except through the three `__tribute_evidence_*` functions.
#[derive(Debug, Clone)]
struct Evidence {
    markers: SmallVec<[Marker; 4]>,
}

impl Evidence {
    fn new() -> Self {
        Self {
            markers: SmallVec::new(),
        }
    }

    fn lookup(&self, ability_id: i32) -> &Marker {
        let result = self
            .markers
            .binary_search_by_key(&ability_id, |m| m.ability_id);
        debug_assert!(
            result.is_ok(),
            "ICE: __tribute_evidence_lookup: ability_id {} not found (compiler bug)",
            ability_id
        );
        match result {
            Ok(idx) => &self.markers[idx],
            // SAFETY: The compiler guarantees that every ability_id passed here
            // has been previously inserted via __tribute_evidence_extend.
            // Reaching this branch means a compiler bug.
            Err(_) => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    fn extend(&self, marker: Marker) -> Self {
        let mut new = self.clone();
        match new
            .markers
            .binary_search_by_key(&marker.ability_id, |m| m.ability_id)
        {
            Ok(pos) => {
                // Same ability_id already exists (nested same-ability handler).
                // Replace with the new (inner) marker so lookup returns the
                // closest handler.
                new.markers[pos] = marker;
            }
            Err(pos) => {
                new.markers.insert(pos, marker);
            }
        }
        new
    }
}

/// Create an empty evidence.
///
/// Signature: `() -> ptr`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_evidence_empty() -> *mut Evidence {
    Box::into_raw(Box::new(Evidence::new()))
}

/// Look up a marker by ability ID in the `Evidence` and return its
/// `prompt_tag` (an `i32`).
///
/// Aborts if no marker with the given `ability_id` exists (compiler bug).
///
/// Signature: `(ev: ptr, ability_id: i32) -> i32`
///
/// # Safety
///
/// `ev` must be a valid pointer returned by `__tribute_evidence_empty` or
/// `__tribute_evidence_extend`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_lookup(ev: *const Evidence, ability_id: i32) -> i32 {
    let ev = unsafe { &*ev };
    ev.lookup(ability_id).prompt_tag
}

/// Extend evidence with a new marker (persistent — returns a new evidence).
///
/// Signature: `(ev: ptr, ability_id: i32, prompt_tag: i32, tr_dispatch_fn: ptr) -> ptr`
///
/// # Safety
///
/// `ev` must be a valid pointer returned by `__tribute_evidence_empty` or
/// a previous `__tribute_evidence_extend` call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_extend(
    ev: *const Evidence,
    ability_id: i32,
    prompt_tag: i32,
    tr_dispatch_fn: *const u8,
) -> *mut Evidence {
    let ev = unsafe { &*ev };
    let marker = Marker {
        ability_id,
        prompt_tag,
        tr_dispatch_fn,
    };
    Box::into_raw(Box::new(ev.extend(marker)))
}

/// Look up the tail-resumptive dispatch function pointer for an ability.
///
/// Returns the `tr_dispatch_fn` pointer from the marker, or null if
/// the handler is not tail-resumptive.
///
/// Signature: `(ev: ptr, ability_id: i32) -> ptr`
///
/// # Safety
///
/// `ev` must be a valid pointer returned by `__tribute_evidence_empty` or
/// `__tribute_evidence_extend`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_lookup_tr(
    ev: *const Evidence,
    ability_id: i32,
) -> *const u8 {
    let ev = unsafe { &*ev };
    ev.lookup(ability_id).tr_dispatch_fn
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_dealloc() {
        unsafe {
            let ptr = __tribute_alloc(64);
            assert!(!ptr.is_null());
            __tribute_dealloc(ptr, 64);
        }
    }

    #[test]
    fn test_alloc_zero() {
        unsafe {
            let ptr = __tribute_alloc(0);
            assert!(ptr.is_null());
        }
    }

    #[test]
    fn test_next_tag_sequential() {
        __tribute_init();
        let a = __tribute_next_tag();
        let b = __tribute_next_tag();
        let c = __tribute_next_tag();
        // Tags must be strictly sequential (unique per thread).
        assert_eq!(b, a + 1);
        assert_eq!(c, a + 2);
    }

    #[test]
    fn test_alloc_oversized_returns_null() {
        // Layout::from_size_align fails for sizes > isize::MAX - 7 (with align=8).
        // Should return null instead of panicking across FFI boundary.
        unsafe {
            let ptr = __tribute_alloc(u64::MAX);
            assert!(ptr.is_null());
        }
    }

    #[test]
    fn test_dealloc_invalid_is_noop() {
        // Should not panic on null pointer or invalid size
        unsafe {
            __tribute_dealloc(core::ptr::null_mut(), 64);
            __tribute_dealloc(core::ptr::null_mut(), 0);
        }
    }

    // =========================================================================
    // Evidence tests
    // =========================================================================

    #[test]
    fn test_evidence_empty() {
        let ev = __tribute_evidence_empty();
        assert!(!ev.is_null());
        let ev_ref = unsafe { &*ev };
        assert!(ev_ref.markers.is_empty());
        // Clean up
        let _ = unsafe { Box::from_raw(ev) };
    }

    #[test]
    fn test_evidence_extend_single() {
        unsafe {
            let ev = __tribute_evidence_empty();
            let ev2 = __tribute_evidence_extend(ev, 10, 1, core::ptr::null());

            let ev2_ref = &*ev2;
            assert_eq!(ev2_ref.markers.len(), 1);
            assert_eq!(ev2_ref.markers[0].ability_id, 10);
            assert_eq!(ev2_ref.markers[0].prompt_tag, 1);
            assert!(ev2_ref.markers[0].tr_dispatch_fn.is_null());

            // Original evidence is unchanged (persistent)
            let ev_ref = &*ev;
            assert!(ev_ref.markers.is_empty());

            let _ = Box::from_raw(ev);
            let _ = Box::from_raw(ev2);
        }
    }

    #[test]
    fn test_evidence_extend_sorted() {
        unsafe {
            let ev = __tribute_evidence_empty();
            // Insert in reverse order: 30, 10, 20
            let ev = __tribute_evidence_extend(ev, 30, 3, core::ptr::null());
            let ev = __tribute_evidence_extend(ev, 10, 1, core::ptr::null());
            let ev = __tribute_evidence_extend(ev, 20, 2, core::ptr::null());

            let ev_ref = &*ev;
            assert_eq!(ev_ref.markers.len(), 3);
            // Should be sorted by ability_id
            assert_eq!(ev_ref.markers[0].ability_id, 10);
            assert_eq!(ev_ref.markers[1].ability_id, 20);
            assert_eq!(ev_ref.markers[2].ability_id, 30);

            // Note: we leak intermediate evidences in this test; acceptable for testing
            let _ = Box::from_raw(ev);
        }
    }

    #[test]
    fn test_evidence_lookup_found() {
        unsafe {
            let ev = __tribute_evidence_empty();
            let ev = __tribute_evidence_extend(ev, 10, 1, core::ptr::null());
            let ev = __tribute_evidence_extend(ev, 20, 2, core::ptr::null());

            let prompt_tag = __tribute_evidence_lookup(ev, 20);
            assert_eq!(prompt_tag, 2);

            let _ = Box::from_raw(ev);
        }
    }
}

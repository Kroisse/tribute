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

/// Print raw bytes to stdout (no newline appended).
///
/// Used by the String rope implementation to output leaf byte sequences.
/// The caller is responsible for traversing the rope and calling this for
/// each `Leaf(bytes)` node.
///
/// Signature: `(ptr: *const u8, len: u64) -> ()`
///
/// # Safety
///
/// `ptr` must point to a valid byte buffer of at least `len` bytes,
/// or be null when `len` is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_print_bytes(ptr: *const u8, len: u64) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let Ok(len) = usize::try_from(len) else {
        return;
    };
    unsafe {
        write(1, ptr, len);
    }
}

/// Print a newline character to stdout.
///
/// Used by `print_line` after traversing and printing a String rope.
///
/// Signature: `() -> ()`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_print_newline() {
    unsafe {
        write(1, b"\n".as_ptr(), 1);
    }
}

// =============================================================================
// Bytes support
// =============================================================================

/// Bytes payload layout: [ptr: *const u8, len: u64].
///
/// Compiler passes emit code that stores this layout after the RC header.
/// Runtime functions receive a pointer to this payload area (not the raw allocation).
#[repr(C)]
pub struct TributeBytes {
    pub ptr: *const u8,
    pub len: u64,
}

/// Print the contents of a Bytes value to stdout (no trailing newline).
///
/// Signature: `(bytes: ptr) -> ()`
///
/// # Safety
///
/// `bytes` must be a valid pointer to a `TributeBytes` payload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_bytes_print(bytes: *const TributeBytes) {
    let b = unsafe { &*bytes };
    if b.len > 0 && !b.ptr.is_null() {
        unsafe {
            write(1, b.ptr, b.len as usize);
        }
    }
}

/// Return the byte length of a Bytes value.
///
/// Signature: `(bytes: ptr) -> u32`
///
/// # Safety
///
/// `bytes` must be a valid pointer to a `TributeBytes` payload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_bytes_len(bytes: *const TributeBytes) -> u32 {
    let b = unsafe { &*bytes };
    b.len as u32
}

/// Concatenate two Bytes values, returning a new RC-managed Bytes.
///
/// Allocates a `TributeRc<TributeBytes>` (RC header + TributeBytes payload),
/// copies both byte sequences into a fresh buffer, and returns a pointer
/// to the payload area.
///
/// Signature: `(a: ptr, b: ptr) -> ptr`
///
/// # Safety
///
/// Both `a` and `b` must be valid pointers to `TributeBytes` payloads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_bytes_concat(
    a: *const TributeBytes,
    b: *const TributeBytes,
) -> *mut TributeBytes {
    let a_ref = unsafe { &*a };
    let b_ref = unsafe { &*b };

    let total_len = a_ref.len + b_ref.len;

    // Allocate buffer for concatenated bytes
    let buf = if total_len > 0 {
        let buf = unsafe { __tribute_alloc(total_len) };
        if a_ref.len > 0 && !a_ref.ptr.is_null() {
            unsafe {
                core::ptr::copy_nonoverlapping(a_ref.ptr, buf, a_ref.len as usize);
            }
        }
        if b_ref.len > 0 && !b_ref.ptr.is_null() {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    b_ref.ptr,
                    buf.add(a_ref.len as usize),
                    b_ref.len as usize,
                );
            }
        }
        buf
    } else {
        core::ptr::null_mut()
    };

    // Allocate RcBox<TributeBytes>
    let alloc_size = tribute_rc::HEADER_SIZE + core::mem::size_of::<TributeBytes>() as u64;
    let raw = unsafe { __tribute_alloc(alloc_size) };
    let rc_box = unsafe { tribute_rc::RcBox::<TributeBytes>::init(raw, 0) };
    rc_box.payload.ptr = buf;
    rc_box.payload.len = total_len;

    &raw mut rc_box.payload
}

/// Slice a Bytes value, returning a new RC-managed Bytes pointing into the
/// original buffer (zero-copy).
///
/// The returned slice shares the original's data buffer. To prevent
/// use-after-free, this function bumps the original object's refcount so
/// it stays alive at least as long as the slice. The extra retain is
/// balanced when the compiler's RC insertion pass releases the original
/// at the caller's scope boundary.
///
/// Panics (aborts) if `start > end` or `end > bytes.len`.
///
/// Signature: `(bytes: ptr, start: u32, end: u32) -> ptr`
///
/// # Safety
///
/// `bytes` must be a valid pointer to a `TributeBytes` payload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_bytes_slice_or_panic(
    bytes: *const TributeBytes,
    start: u32,
    end: u32,
) -> *mut TributeBytes {
    let b = unsafe { &*bytes };
    let s = start as u64;
    let e = end as u64;

    // Bounds check
    if s > e || e > b.len {
        bounds_check_abort();
    }

    // Retain the original so its data buffer stays alive while the slice
    // points into it.
    unsafe {
        let rc_box = &*tribute_rc::RcBox::from_payload_ptr(bytes);
        rc_box.retain();
    }

    let new_len = e - s;
    let new_ptr = if new_len > 0 && !b.ptr.is_null() {
        unsafe { b.ptr.add(s as usize) }
    } else {
        core::ptr::null()
    };

    // Allocate RcBox<TributeBytes>
    let alloc_size = tribute_rc::HEADER_SIZE + core::mem::size_of::<TributeBytes>() as u64;
    let raw = unsafe { __tribute_alloc(alloc_size) };
    let rc_box = unsafe { tribute_rc::RcBox::<TributeBytes>::init(raw, 0) };
    rc_box.payload.ptr = new_ptr;
    rc_box.payload.len = new_len;

    &raw mut rc_box.payload
}

fn bounds_check_abort() -> ! {
    oom_abort();
}

// =============================================================================
// Allocator
// =============================================================================

pub(crate) fn oom_abort() -> ! {
    unsafe extern "C" {
        fn abort() -> !;
    }
    unsafe { abort() }
}

/// # Safety
///
/// Caller must eventually free the returned pointer via `__tribute_dealloc`
/// with the same `size`.
///
/// # Aborts
///
/// Aborts the process if any of the following occur:
/// - `size` exceeds `usize::MAX` (u64-to-usize conversion failure)
/// - `Layout::from_size_align` fails (size exceeds `isize::MAX - 7`)
/// - The underlying allocator returns null (OOM)
///
/// The only case that returns null is `size == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_alloc(size: u64) -> *mut u8 {
    if size == 0 {
        return core::ptr::null_mut();
    }
    let Ok(size) = usize::try_from(size) else {
        oom_abort();
    };
    if asan::is_enabled() {
        return unsafe { asan::alloc(size) };
    }
    let Ok(layout) = core::alloc::Layout::from_size_align(size, 8) else {
        oom_abort();
    };
    let ptr = unsafe { alloc::alloc::alloc(layout) };
    if ptr.is_null() {
        oom_abort();
    }
    ptr
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
///
/// `handler_dispatch` is a pointer to the full CPS handler dispatch closure
/// `(k: ptr, op_idx: i32, value: ptr) -> void`, or null if not using
/// full CPS. Used by the tail-call-based effect handling path.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Marker {
    pub ability_id: i32,
    pub prompt_tag: i32,
    pub tr_dispatch_fn: *const u8,
    pub handler_dispatch: *const u8,
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
/// Signature: `(ev: ptr, ability_id: i32, prompt_tag: i32, tr_dispatch_fn: ptr, handler_dispatch: ptr) -> ptr`
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
    handler_dispatch: *const u8,
) -> *mut Evidence {
    let ev = unsafe { &*ev };
    let marker = Marker {
        ability_id,
        prompt_tag,
        tr_dispatch_fn,
        handler_dispatch,
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

/// Look up the handler dispatch closure pointer for an ability.
///
/// Returns the `handler_dispatch` pointer from the marker, or null if
/// the handler does not use full CPS dispatch.
///
/// Signature: `(ev: ptr, ability_id: i32) -> ptr`
///
/// # Safety
///
/// `ev` must be a valid pointer returned by `__tribute_evidence_empty` or
/// `__tribute_evidence_extend`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_lookup_handler(
    ev: *const Evidence,
    ability_id: i32,
) -> *const u8 {
    let ev = unsafe { &*ev };
    ev.lookup(ability_id).handler_dispatch
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
            let ev2 = __tribute_evidence_extend(ev, 10, 1, core::ptr::null(), core::ptr::null());

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
            let ev = __tribute_evidence_extend(ev, 30, 3, core::ptr::null(), core::ptr::null());
            let ev = __tribute_evidence_extend(ev, 10, 1, core::ptr::null(), core::ptr::null());
            let ev = __tribute_evidence_extend(ev, 20, 2, core::ptr::null(), core::ptr::null());

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
            let ev = __tribute_evidence_extend(ev, 10, 1, core::ptr::null(), core::ptr::null());
            let ev = __tribute_evidence_extend(ev, 20, 2, core::ptr::null(), core::ptr::null());

            let prompt_tag = __tribute_evidence_lookup(ev, 20);
            assert_eq!(prompt_tag, 2);

            let _ = Box::from_raw(ev);
        }
    }
}

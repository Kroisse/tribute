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
use alloc::vec::Vec;

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
        libc::write(libc::STDOUT_FILENO, s.as_ptr().cast(), s.len());
        libc::write(libc::STDOUT_FILENO, b"\n".as_ptr().cast(), 1);
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
        libc::write(libc::STDOUT_FILENO, s.as_ptr().cast(), s.len());
        libc::write(libc::STDOUT_FILENO, b"\n".as_ptr().cast(), 1);
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
        libc::write(libc::STDOUT_FILENO, s.as_ptr().cast(), s.len());
        libc::write(libc::STDOUT_FILENO, b"\n".as_ptr().cast(), 1);
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

fn write_stdout_all(bytes: &[u8]) {
    let _ = write_all_with(bytes, |remaining| {
        let written = unsafe {
            libc::write(
                libc::STDOUT_FILENO,
                remaining.as_ptr().cast(),
                remaining.len(),
            )
        };
        if written >= 0 {
            Ok(written as usize)
        } else {
            Err(last_errno())
        }
    });
}

fn write_all_with(
    mut remaining: &[u8],
    mut write_once: impl FnMut(&[u8]) -> Result<usize, i32>,
) -> bool {
    while !remaining.is_empty() {
        match write_once(remaining) {
            Ok(0) => return false,
            Ok(written) if written <= remaining.len() => remaining = &remaining[written..],
            Ok(_) => return false,
            Err(code) if code == libc::EINTR => {}
            Err(_) => return false,
        }
    }
    true
}

// =============================================================================
// Native basic I/O
// =============================================================================

pub const IO_READ_LINE: u32 = 0;
pub const IO_READ_END_OF_FILE: u32 = 1;
pub const IO_READ_INVALID_ENCODING: u32 = 2;
pub const IO_READ_SYSTEM: u32 = 3;

/// Target-neutral line-read result returned across the native runtime ABI.
///
/// The compiler copies the fields into `std::io::ReadLineResult` and then
/// releases this descriptor with [`__tribute_io_read_line_result_dealloc`].
#[repr(C)]
pub struct NativeReadLineResult {
    pub tag: u32,
    pub code: i32,
    pub bytes: *mut TributeBytes,
    pub message: *mut TributeBytes,
}

enum ReadByte {
    Byte(u8),
    EndOfFile,
    Interrupted,
    Error(i32),
}

enum ReadChunk {
    Bytes(usize),
    EndOfFile,
    Interrupted,
    Error(i32),
}

const STDIN_BUFFER_SIZE: usize = 8 * 1024;

struct StdinBuffer {
    bytes: [u8; STDIN_BUFFER_SIZE],
    position: usize,
    length: usize,
}

impl StdinBuffer {
    const fn new() -> Self {
        Self {
            bytes: [0; STDIN_BUFFER_SIZE],
            position: 0,
            length: 0,
        }
    }

    fn read_byte(&mut self, mut refill: impl FnMut(&mut [u8]) -> ReadChunk) -> ReadByte {
        if self.position < self.length {
            let byte = self.bytes[self.position];
            self.position += 1;
            return ReadByte::Byte(byte);
        }

        match refill(&mut self.bytes) {
            ReadChunk::Bytes(length) => {
                debug_assert!(length > 0 && length <= self.bytes.len());
                self.position = 1;
                self.length = length;
                ReadByte::Byte(self.bytes[0])
            }
            ReadChunk::EndOfFile => ReadByte::EndOfFile,
            ReadChunk::Interrupted => ReadByte::Interrupted,
            ReadChunk::Error(code) => ReadByte::Error(code),
        }
    }
}

/// Write a flattened `Bytes` payload, optionally followed by one newline.
///
/// # Safety
///
/// `bytes` must point to a valid [`TributeBytes`] payload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_io_write(bytes: *const TributeBytes, newline: u32) {
    let b = unsafe { &*bytes };
    if b.len > 0 && !b.ptr.is_null() {
        write_stdout_all(unsafe { core::slice::from_raw_parts(b.ptr, b.len as usize) });
    }
    if newline == 1 {
        write_stdout_all(b"\n");
    }
}

/// Read one line from native stdin and return a C-compatible result descriptor.
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_io_read_line() -> *mut NativeReadLineResult {
    let mut stdin = unsafe { thread_state() }.stdin_buffer();
    read_line_with(|| {
        stdin.read_byte(|buffer| {
            let read =
                unsafe { libc::read(libc::STDIN_FILENO, buffer.as_mut_ptr().cast(), buffer.len()) };
            if read > 0 {
                return ReadChunk::Bytes(read as usize);
            }
            match read {
                0 => ReadChunk::EndOfFile,
                _ => {
                    let code = last_errno();
                    if code == libc::EINTR {
                        ReadChunk::Interrupted
                    } else {
                        ReadChunk::Error(code)
                    }
                }
            }
        })
    })
}

/// Release a descriptor returned by [`__tribute_io_read_line`].
///
/// The RC-managed `bytes` and `message` payloads, if any, have already been
/// transferred to the compiler-created result ADT and are not released here.
///
/// # Safety
///
/// `result` must be null or a pointer returned by [`__tribute_io_read_line`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_io_read_line_result_dealloc(result: *mut NativeReadLineResult) {
    unsafe {
        __tribute_dealloc(
            result.cast(),
            core::mem::size_of::<NativeReadLineResult>() as u64,
        )
    };
}

fn read_line_with(mut read_byte: impl FnMut() -> ReadByte) -> *mut NativeReadLineResult {
    let mut bytes = Vec::new();
    loop {
        match read_byte() {
            ReadByte::Byte(b'\n') => {
                if bytes.last() == Some(&b'\r') {
                    bytes.pop();
                }
                return line_result(bytes);
            }
            ReadByte::Byte(byte) => bytes.push(byte),
            ReadByte::EndOfFile if bytes.is_empty() => {
                return allocate_read_result(NativeReadLineResult {
                    tag: IO_READ_END_OF_FILE,
                    code: 0,
                    bytes: core::ptr::null_mut(),
                    message: core::ptr::null_mut(),
                });
            }
            ReadByte::EndOfFile => return line_result(bytes),
            ReadByte::Interrupted => {}
            ReadByte::Error(code) => {
                return allocate_read_result(NativeReadLineResult {
                    tag: IO_READ_SYSTEM,
                    code,
                    bytes: core::ptr::null_mut(),
                    message: allocate_bytes(b"stdin read failed"),
                });
            }
        }
    }
}

fn line_result(bytes: Vec<u8>) -> *mut NativeReadLineResult {
    if core::str::from_utf8(&bytes).is_err() {
        return allocate_read_result(NativeReadLineResult {
            tag: IO_READ_INVALID_ENCODING,
            code: 0,
            bytes: core::ptr::null_mut(),
            message: core::ptr::null_mut(),
        });
    }

    allocate_read_result(NativeReadLineResult {
        tag: IO_READ_LINE,
        code: 0,
        bytes: allocate_bytes(&bytes),
        message: core::ptr::null_mut(),
    })
}

fn allocate_read_result(result: NativeReadLineResult) -> *mut NativeReadLineResult {
    let size = core::mem::size_of::<NativeReadLineResult>() as u64;
    let raw = unsafe { __tribute_alloc(size) }.cast::<NativeReadLineResult>();
    unsafe { raw.write(result) };
    raw
}

fn allocate_bytes(bytes: &[u8]) -> *mut TributeBytes {
    let len = bytes.len() as u64;
    let data = if bytes.is_empty() {
        core::ptr::null_mut()
    } else {
        let data = unsafe { __tribute_alloc(len) };
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        data
    };

    let size = core::mem::size_of::<tribute_rc::RcBox<TributeBytes>>() as u64;
    let raw = unsafe { __tribute_alloc(size) };
    let rc_box = unsafe { tribute_rc::RcBox::<TributeBytes>::init(raw, 0) };
    unsafe {
        (*rc_box).payload.ptr = data;
        (*rc_box).payload.len = len;
        &raw mut (*rc_box).payload
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn last_errno() -> i32 {
    unsafe { *libc::__errno_location() }
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
fn last_errno() -> i32 {
    unsafe { *libc::__error() }
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

/// Compare equal-length ranges in two Bytes values.
///
/// Returns `1` when the ranges contain the same bytes and `0` otherwise.
/// This is the native target primitive used by rope String equality: callers
/// select contiguous spans from the current pair of leaves, so no flattening
/// or temporary byte buffer is required.
///
/// # Safety
///
/// `left` and `right` must be valid pointers to `TributeBytes` payloads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_bytes_range_equal(
    left: *const TributeBytes,
    left_start: u32,
    right: *const TributeBytes,
    right_start: u32,
    len: u32,
) -> u32 {
    let left = unsafe { &*left };
    let right = unsafe { &*right };
    let left_start = u64::from(left_start);
    let right_start = u64::from(right_start);
    let len = u64::from(len);

    let Some(left_end) = left_start.checked_add(len) else {
        return 0;
    };
    let Some(right_end) = right_start.checked_add(len) else {
        return 0;
    };
    if left_end > left.len || right_end > right.len {
        return 0;
    }
    if len == 0 {
        return 1;
    }

    let left_ptr = unsafe { left.ptr.add(left_start as usize) };
    let right_ptr = unsafe { right.ptr.add(right_start as usize) };
    u32::from(unsafe { libc::memcmp(left_ptr.cast(), right_ptr.cast(), len as usize) == 0 })
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

    let Some(total_len) = a_ref.len.checked_add(b_ref.len) else {
        oom_abort();
    };

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
    unsafe {
        (*rc_box).payload.ptr = buf;
        (*rc_box).payload.len = total_len;
        &raw mut (*rc_box).payload
    }
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
    unsafe {
        (*rc_box).payload.ptr = new_ptr;
        (*rc_box).payload.len = new_len;
        &raw mut (*rc_box).payload
    }
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
/// `shadowed` is the marker of the same ability this one shadows, or null.
/// It points into the evidence this marker's evidence was derived from. An
/// evidence is immutable and never freed once returned, so the marker it
/// points to stays valid.
///
/// `outer` is the evidence the handler was installed on, before the
/// installation's selection. It stays valid for the same reason.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Marker {
    pub ability_id: i32,
    pub prompt_tag: i32,
    pub tr_dispatch_fn: *const u8,
    pub shadowed: *const Marker,
    pub outer: *const Evidence,
}

/// Opaque evidence structure — an array of marker stacks sorted by
/// `ability_id`. Each slot holds the top `Marker` of its ability; the markers
/// it shadows follow its `shadowed` chain.
///
/// IR-level code only sees `core.ptr`; this struct is never exposed across FFI
/// except through the `__tribute_evidence_*` functions.
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

    fn position(&self, ability_id: i32) -> usize {
        let result = self
            .markers
            .binary_search_by_key(&ability_id, |m| m.ability_id);
        debug_assert!(
            result.is_ok(),
            "ICE: evidence has no marker for ability_id {} (compiler bug)",
            ability_id
        );
        match result {
            Ok(idx) => idx,
            // SAFETY: The compiler guarantees that every ability_id passed here
            // has been previously inserted via __tribute_evidence_extend.
            // Reaching this branch means a compiler bug.
            Err(_) => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    fn lookup(&self, ability_id: i32) -> &Marker {
        &self.markers[self.position(ability_id)]
    }

    /// Push `marker` onto its ability's stack. Its `shadowed` is set here.
    fn extend(&self, mut marker: Marker) -> Self {
        let mut new = self.clone();
        match self
            .markers
            .binary_search_by_key(&marker.ability_id, |m| m.ability_id)
        {
            Ok(pos) => {
                // The shadowed marker is the one stored in `self`, not its
                // copy in `new`: `new` moves when it is boxed.
                marker.shadowed = &self.markers[pos];
                new.markers[pos] = marker;
            }
            Err(pos) => {
                marker.shadowed = core::ptr::null();
                new.markers.insert(pos, marker);
            }
        }
        new
    }

    /// Pop the top marker of `ability_id`, exposing the one it shadows.
    fn mask(&self, ability_id: i32) -> Self {
        let pos = self.position(ability_id);
        let mut new = self.clone();
        let shadowed = self.markers[pos].shadowed;
        if shadowed.is_null() {
            new.markers.remove(pos);
        } else {
            // SAFETY: `shadowed` points into an evidence that is never freed.
            new.markers[pos] = unsafe { *shadowed };
        }
        new
    }

    /// The evidence of the row tail in `slot`, or `self` without that slot.
    fn tail(&self, slot: i32) -> *const Evidence {
        match self.markers.binary_search_by_key(&slot, |m| m.ability_id) {
            Ok(pos) => self.markers[pos].outer,
            Err(_) => self,
        }
    }

    /// Push a copy of the top marker of `ability_id` onto its stack.
    fn dup(&self, ability_id: i32) -> Self {
        let pos = self.position(ability_id);
        let mut new = self.clone();
        new.markers[pos].shadowed = &self.markers[pos];
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
/// Signature: `(ev: ptr, ability_id: i32, prompt_tag: i32, tr_dispatch_fn: ptr, outer: ptr) -> ptr`
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
    outer: *const Evidence,
) -> *mut Evidence {
    let ev = unsafe { &*ev };
    let marker = Marker {
        ability_id,
        prompt_tag,
        tr_dispatch_fn,
        shadowed: core::ptr::null(),
        outer,
    };
    Box::into_raw(Box::new(ev.extend(marker)))
}

/// Remove the top marker of an ability, exposing the marker it shadows
/// (persistent — returns a new evidence). The ability's slot is removed when
/// the top marker shadows nothing.
///
/// Signature: `(ev: ptr, ability_id: i32) -> ptr`
///
/// # Safety
///
/// `ev` must be a valid pointer returned by a `__tribute_evidence_*` function
/// and must hold a marker for `ability_id`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_mask(
    ev: *const Evidence,
    ability_id: i32,
) -> *mut Evidence {
    let ev = unsafe { &*ev };
    Box::into_raw(Box::new(ev.mask(ability_id)))
}

/// Push a copy of the top marker of an ability so that it shadows the
/// original (persistent — returns a new evidence).
///
/// Signature: `(ev: ptr, ability_id: i32) -> ptr`
///
/// # Safety
///
/// `ev` must be a valid pointer returned by a `__tribute_evidence_*` function
/// and must hold a marker for `ability_id`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_dup(
    ev: *const Evidence,
    ability_id: i32,
) -> *mut Evidence {
    let ev = unsafe { &*ev };
    Box::into_raw(Box::new(ev.dup(ability_id)))
}

/// Return the evidence the top handler of an ability was installed on.
///
/// Signature: `(ev: ptr, ability_id: i32) -> ptr`
///
/// # Safety
///
/// `ev` must be a valid pointer returned by a `__tribute_evidence_*` function
/// and must hold a marker for `ability_id`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_outer(
    ev: *const Evidence,
    ability_id: i32,
) -> *const Evidence {
    let ev = unsafe { &*ev };
    ev.lookup(ability_id).outer
}

/// Return the evidence of a row tail: the `outer` of the marker in `slot`, or
/// `ev` itself when it holds no such marker.
///
/// Signature: `(ev: ptr, slot: i32) -> ptr`
///
/// # Safety
///
/// `ev` must be a valid pointer returned by a `__tribute_evidence_*` function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_tail(
    ev: *const Evidence,
    slot: i32,
) -> *const Evidence {
    let ev = unsafe { &*ev };
    ev.tail(slot)
}

/// Set the evidence of a row tail: put a marker in `slot` whose `outer` is
/// `tail` (persistent — returns a new evidence).
///
/// Signature: `(ev: ptr, slot: i32, tail: ptr) -> ptr`
///
/// # Safety
///
/// Both pointers must be valid pointers returned by a `__tribute_evidence_*`
/// function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_with_tail(
    ev: *const Evidence,
    slot: i32,
    tail: *const Evidence,
) -> *mut Evidence {
    let ev = unsafe { &*ev };
    let marker = Marker {
        ability_id: slot,
        prompt_tag: 0,
        tr_dispatch_fn: core::ptr::null(),
        shadowed: core::ptr::null(),
        outer: tail,
    };
    Box::into_raw(Box::new(ev.extend(marker)))
}

/// Push the top marker that `source` holds for an ability onto `ev`
/// (persistent — returns a new evidence).
///
/// Signature: `(ev: ptr, source: ptr, ability_id: i32) -> ptr`
///
/// # Safety
///
/// Both pointers must be valid pointers returned by a `__tribute_evidence_*`
/// function, and `source` must hold a marker for `ability_id`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_evidence_push(
    ev: *const Evidence,
    source: *const Evidence,
    ability_id: i32,
) -> *mut Evidence {
    let ev = unsafe { &*ev };
    let source = unsafe { &*source };
    Box::into_raw(Box::new(ev.extend(*source.lookup(ability_id))))
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
    use alloc::collections::BTreeMap;
    use proptest::prelude::*;
    use proptest::sample::Index;

    fn scripted_read(events: impl IntoIterator<Item = ReadByte>) -> *mut NativeReadLineResult {
        let mut events = events.into_iter();
        read_line_with(|| events.next().expect("read script exhausted"))
    }

    unsafe fn bytes_contents(bytes: *const TributeBytes) -> Vec<u8> {
        if bytes.is_null() {
            return Vec::new();
        }
        let bytes = unsafe { &*bytes };
        if bytes.len == 0 {
            return Vec::new();
        }
        unsafe { core::slice::from_raw_parts(bytes.ptr, bytes.len as usize) }.to_vec()
    }

    unsafe fn dealloc_test_bytes(bytes: *mut TributeBytes) {
        if bytes.is_null() {
            return;
        }
        let payload = unsafe { &*bytes };
        unsafe { __tribute_dealloc(payload.ptr.cast_mut(), payload.len) };
        let raw = unsafe { tribute_rc::RcBox::from_payload_ptr_mut(bytes) }.cast();
        unsafe {
            __tribute_dealloc(
                raw,
                core::mem::size_of::<tribute_rc::RcBox<TributeBytes>>() as u64,
            )
        };
    }

    unsafe fn dealloc_test_read_result(result: *mut NativeReadLineResult) {
        let result_ref = unsafe { &*result };
        unsafe {
            dealloc_test_bytes(result_ref.bytes);
            dealloc_test_bytes(result_ref.message);
            __tribute_io_read_line_result_dealloc(result);
        }
    }

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

    #[test]
    fn test_native_io_abi_layout() {
        assert_eq!(core::mem::offset_of!(NativeReadLineResult, tag), 0);
        assert_eq!(core::mem::offset_of!(NativeReadLineResult, code), 4);
        assert_eq!(core::mem::offset_of!(NativeReadLineResult, bytes), 8);
        assert_eq!(core::mem::offset_of!(NativeReadLineResult, message), 16);
        assert_eq!(core::mem::size_of::<NativeReadLineResult>(), 24);

        let _: unsafe extern "C" fn(*const TributeBytes, u32) = __tribute_io_write;
        let _: extern "C" fn() -> *mut NativeReadLineResult = __tribute_io_read_line;
        let _: unsafe extern "C" fn(*mut NativeReadLineResult) =
            __tribute_io_read_line_result_dealloc;
        let _: unsafe extern "C" fn(
            *const TributeBytes,
            u32,
            *const TributeBytes,
            u32,
            u32,
        ) -> u32 = __tribute_bytes_range_equal;
    }

    fn bytes_view(data: &[u8]) -> TributeBytes {
        // Empty runtime payloads carry a null pointer.
        TributeBytes {
            ptr: if data.is_empty() {
                core::ptr::null()
            } else {
                data.as_ptr()
            },
            len: data.len() as u64,
        }
    }

    /// Range starts and lengths: mostly in bounds, sometimes near `u32::MAX`
    /// so that the end computation would overflow 32 bits.
    fn range_bound() -> impl Strategy<Value = u32> {
        prop_oneof![9 => 0u32..14, 1 => (u32::MAX - 14)..=u32::MAX]
    }

    proptest! {
        /// A range comparison is `1` exactly when both ranges lie within
        /// their payloads and hold the same bytes, and `0` otherwise.
        #[test]
        fn prop_bytes_range_equal_matches_slices(
            left in prop::collection::vec(0u8..2, 0..12),
            right in prop::collection::vec(0u8..2, 0..12),
            left_start in range_bound(),
            right_start in range_bound(),
            len in range_bound(),
        ) {
            let range = |data: &[u8], start: u32| {
                let start = start as usize;
                let end = start.checked_add(len as usize)?;
                data.get(start..end).map(<[u8]>::to_vec)
            };
            let expected = match (range(&left, left_start), range(&right, right_start)) {
                (Some(l), Some(r)) => u32::from(l == r),
                _ => 0,
            };
            let left_bytes = bytes_view(&left);
            let right_bytes = bytes_view(&right);
            let actual = unsafe {
                __tribute_bytes_range_equal(&left_bytes, left_start, &right_bytes, right_start, len)
            };
            prop_assert_eq!(actual, expected);
        }
    }

    /// One scripted outcome of a `write` call.
    #[derive(Clone, Debug)]
    enum WriteStep {
        /// Write between one byte and everything that remains.
        Partial(Index),
        Zero,
        Interrupted,
        Fail(i32),
        /// Report more bytes than were offered.
        Overlong(usize),
    }

    fn write_step() -> impl Strategy<Value = WriteStep> {
        prop_oneof![
            4 => any::<Index>().prop_map(WriteStep::Partial),
            1 => Just(WriteStep::Zero),
            2 => Just(WriteStep::Interrupted),
            1 => (1i32..200)
                .prop_filter("EINTR is retried", |code| *code != libc::EINTR)
                .prop_map(WriteStep::Fail),
            1 => (1usize..4).prop_map(WriteStep::Overlong),
        ]
    }

    proptest! {
        /// `write_all_with` offers the unwritten suffix on every attempt,
        /// retries `EINTR`, and fails on a zero, overlong, or failed write.
        /// A script that runs out writes everything that remains.
        #[test]
        fn prop_write_all_follows_write_script(
            data in prop::collection::vec(any::<u8>(), 0..24),
            script in prop::collection::vec(write_step(), 0..12),
        ) {
            let outcome = |step: Option<&WriteStep>, remaining: usize| match step {
                None => Ok(remaining),
                Some(WriteStep::Partial(index)) => Ok(index.index(remaining) + 1),
                Some(WriteStep::Zero) => Ok(0),
                Some(WriteStep::Interrupted) => Err(libc::EINTR),
                Some(WriteStep::Fail(code)) => Err(*code),
                Some(WriteStep::Overlong(extra)) => Ok(remaining + extra),
            };

            // Reference model.
            let mut offset = 0;
            let mut expected_attempts = Vec::new();
            let mut steps = script.iter();
            let expected = loop {
                if offset == data.len() {
                    break true;
                }
                expected_attempts.push(data[offset..].to_vec());
                let step = steps.next();
                match step {
                    Some(WriteStep::Interrupted) => {}
                    Some(WriteStep::Zero | WriteStep::Fail(_) | WriteStep::Overlong(_)) => {
                        break false;
                    }
                    None | Some(WriteStep::Partial(_)) => {
                        let Ok(written) = outcome(step, data.len() - offset) else {
                            unreachable!("successful steps write bytes");
                        };
                        offset += written;
                    }
                }
            };

            let mut attempts = Vec::new();
            let mut steps = script.iter();
            let completed = write_all_with(&data, |remaining| {
                attempts.push(remaining.to_vec());
                outcome(steps.next(), remaining.len())
            });

            prop_assert_eq!(completed, expected);
            prop_assert_eq!(attempts, expected_attempts);
        }
    }

    /// The line contents the read-line property feeds: valid text, text with
    /// interior carriage returns, or arbitrary bytes (usually invalid UTF-8).
    fn input_line() -> impl Strategy<Value = Vec<u8>> {
        prop_oneof![
            "[^\r\n]{0,6}".prop_map(String::into_bytes),
            "[a-c\r]{0,4}[a-c]".prop_map(String::into_bytes),
            prop::collection::vec(any::<u8>().prop_filter("line break", |b| *b != b'\n'), 0..6),
        ]
    }

    /// How a line ends in the input stream.
    #[derive(Clone, Copy, Debug)]
    enum LineEnd {
        Lf,
        CrLf,
    }

    #[derive(Debug, PartialEq)]
    enum ReadOutcome {
        Line(Vec<u8>),
        InvalidEncoding,
        EndOfFile,
    }

    /// Read one line through `buffer`, refilling it from `input` in chunks of
    /// the scripted sizes. A zero-sized chunk reports an interrupted read.
    fn chunked_read(
        buffer: &mut StdinBuffer,
        input: &[u8],
        offset: &mut usize,
        chunks: &[usize],
        next_chunk: &mut usize,
    ) -> Result<ReadOutcome, TestCaseError> {
        let result = read_line_with(|| {
            buffer.read_byte(|destination| {
                if *offset == input.len() {
                    return ReadChunk::EndOfFile;
                }
                let size = chunks[*next_chunk % chunks.len()];
                *next_chunk += 1;
                if size == 0 {
                    return ReadChunk::Interrupted;
                }
                let length = size.min(input.len() - *offset).min(destination.len());
                destination[..length].copy_from_slice(&input[*offset..*offset + length]);
                *offset += length;
                ReadChunk::Bytes(length)
            })
        });
        let result_ref = unsafe { &*result };
        let outcome = match result_ref.tag {
            IO_READ_LINE => ReadOutcome::Line(unsafe { bytes_contents(result_ref.bytes) }),
            IO_READ_INVALID_ENCODING => ReadOutcome::InvalidEncoding,
            IO_READ_END_OF_FILE => ReadOutcome::EndOfFile,
            tag => {
                unsafe { dealloc_test_read_result(result) };
                return Err(TestCaseError::fail(format!("unexpected read tag {tag}")));
            }
        };
        unsafe { dealloc_test_read_result(result) };
        Ok(outcome)
    }

    proptest! {
        /// Lines separated by LF or CRLF, with an optional unterminated last
        /// line, read back one by one regardless of how reads chunk the
        /// input; one trailing CR is stripped from terminated lines, invalid
        /// UTF-8 is reported per line, and the end of input reads as EOF.
        #[test]
        fn prop_read_line_round_trips_chunked_input(
            lines in prop::collection::vec(
                (input_line(), prop_oneof![Just(LineEnd::Lf), Just(LineEnd::CrLf)]),
                0..6,
            ),
            unterminated in prop::option::of(input_line()),
            chunks in prop::collection::vec(0usize..8, 1..6)
                .prop_filter("some chunk must make progress", |c| c.iter().any(|&s| s > 0)),
        ) {
            let classify = |line: Vec<u8>| match core::str::from_utf8(&line) {
                Ok(_) => ReadOutcome::Line(line),
                Err(_) => ReadOutcome::InvalidEncoding,
            };
            let mut input = Vec::new();
            let mut expected = Vec::new();
            for (line, end) in &lines {
                input.extend_from_slice(line);
                let mut content = line.clone();
                if let LineEnd::CrLf = end {
                    input.push(b'\r');
                    content.push(b'\r');
                }
                input.push(b'\n');
                if content.last() == Some(&b'\r') {
                    content.pop();
                }
                expected.push(classify(content));
            }
            if let Some(line) = unterminated.filter(|line| !line.is_empty()) {
                input.extend_from_slice(&line);
                expected.push(classify(line));
            }
            expected.push(ReadOutcome::EndOfFile);
            // A read past the end reports EOF again.
            expected.push(ReadOutcome::EndOfFile);

            let mut buffer = StdinBuffer::new();
            let mut offset = 0;
            let mut next_chunk = 0;
            for expected in expected {
                let actual =
                    chunked_read(&mut buffer, &input, &mut offset, &chunks, &mut next_chunk)?;
                prop_assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn test_read_line_retries_interrupt_and_reports_system_error() {
        let line = scripted_read([
            ReadByte::Interrupted,
            ReadByte::Byte(b'x'),
            ReadByte::EndOfFile,
        ]);
        assert_eq!(unsafe { &*line }.tag, IO_READ_LINE);
        assert_eq!(unsafe { bytes_contents((*line).bytes) }, b"x");
        unsafe { dealloc_test_read_result(line) };

        let error = scripted_read([ReadByte::Error(5)]);
        let error_ref = unsafe { &*error };
        assert_eq!(error_ref.tag, IO_READ_SYSTEM);
        assert_eq!(error_ref.code, 5);
        assert_eq!(
            unsafe { bytes_contents(error_ref.message) },
            b"stdin read failed"
        );
        unsafe { dealloc_test_read_result(error) };
    }

    // =========================================================================
    // Evidence tests
    // =========================================================================

    /// The marker slots of a live evidence.
    unsafe fn markers<'a>(ev: *const Evidence) -> &'a [Marker] {
        unsafe { &(*ev).markers }
    }

    #[test]
    fn test_evidence_runtime_function_signatures() {
        let _: extern "C" fn() -> *mut Evidence = __tribute_evidence_empty;
        let _: unsafe extern "C" fn(*const Evidence, i32) -> i32 = __tribute_evidence_lookup;
        let _: unsafe extern "C" fn(
            *const Evidence,
            i32,
            i32,
            *const u8,
            *const Evidence,
        ) -> *mut Evidence = __tribute_evidence_extend;
        let _: unsafe extern "C" fn(*const Evidence, i32) -> *const Evidence =
            __tribute_evidence_outer;
        let _: unsafe extern "C" fn(*const Evidence, i32) -> *mut Evidence =
            __tribute_evidence_mask;
        let _: unsafe extern "C" fn(*const Evidence, i32) -> *mut Evidence = __tribute_evidence_dup;
        let _: unsafe extern "C" fn(*const Evidence, i32) -> *const u8 =
            __tribute_evidence_lookup_tr;
    }

    #[test]
    fn test_marker_repr_c_field_order() {
        assert_eq!(core::mem::offset_of!(Marker, ability_id), 0);
        assert!(
            core::mem::offset_of!(Marker, ability_id) < core::mem::offset_of!(Marker, prompt_tag)
        );
        assert!(
            core::mem::offset_of!(Marker, prompt_tag)
                < core::mem::offset_of!(Marker, tr_dispatch_fn)
        );
        assert!(
            core::mem::offset_of!(Marker, tr_dispatch_fn) < core::mem::offset_of!(Marker, shadowed)
        );
    }

    #[test]
    fn test_evidence_empty() {
        let ev = __tribute_evidence_empty();
        assert!(!ev.is_null());
        let ev_ref = unsafe { &*ev };
        assert!(ev_ref.markers.is_empty());
        // Clean up
        let _ = unsafe { Box::from_raw(ev) };
    }

    /// Ability ids the evidence property draws from. The same ids serve as
    /// row-tail slots, as the runtime does not distinguish them.
    const EVIDENCE_IDS: core::ops::RangeInclusive<i32> = -2..=3;

    #[derive(Clone, Debug)]
    enum EvidenceAction {
        Extend {
            base: Index,
            ability: i32,
            prompt_tag: i32,
            tr: usize,
            outer: Index,
        },
        WithTail {
            base: Index,
            slot: i32,
            tail: Index,
        },
        Mask {
            base: Index,
            ability: Index,
        },
        Dup {
            base: Index,
            ability: Index,
        },
        Push {
            base: Index,
            source: Index,
            ability: Index,
        },
    }

    fn evidence_action() -> impl Strategy<Value = EvidenceAction> {
        prop_oneof![
            4 => (any::<Index>(), EVIDENCE_IDS, 0i32..100, 0usize..3, any::<Index>()).prop_map(
                |(base, ability, prompt_tag, tr, outer)| EvidenceAction::Extend {
                    base,
                    ability,
                    prompt_tag,
                    tr,
                    outer,
                }
            ),
            1 => (any::<Index>(), EVIDENCE_IDS, any::<Index>())
                .prop_map(|(base, slot, tail)| EvidenceAction::WithTail { base, slot, tail }),
            3 => (any::<Index>(), any::<Index>())
                .prop_map(|(base, ability)| EvidenceAction::Mask { base, ability }),
            1 => (any::<Index>(), any::<Index>())
                .prop_map(|(base, ability)| EvidenceAction::Dup { base, ability }),
            1 => (any::<Index>(), any::<Index>(), any::<Index>()).prop_map(
                |(base, source, ability)| EvidenceAction::Push {
                    base,
                    source,
                    ability,
                }
            ),
        ]
    }

    /// A handler of the reference model. `outer` indexes the evidence pool.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct ModelMarker {
        prompt_tag: i32,
        tr: usize,
        outer: usize,
    }

    /// Each ability's handler stack, bottom first. Stacks are never empty.
    type ModelEvidence = BTreeMap<i32, Vec<ModelMarker>>;

    /// A present ability of `model`, chosen by `index`.
    fn present_ability(model: &ModelEvidence, index: Index) -> Option<i32> {
        if model.is_empty() {
            return None;
        }
        model.keys().nth(index.index(model.len())).copied()
    }

    /// The real evidence at `pool[index]` must hold exactly the stacks of
    /// `models[index]`, sorted by ability id, and every query must agree.
    fn check_evidence(
        pool: &[*mut Evidence],
        models: &[ModelEvidence],
        index: usize,
    ) -> Result<(), TestCaseError> {
        let ev = pool[index].cast_const();
        let model = &models[index];
        let pool_index = |outer: *const Evidence| {
            pool.iter()
                .position(|&candidate| candidate.cast_const() == outer)
        };

        let slots = unsafe { markers(ev) };
        let ids: Vec<i32> = slots.iter().map(|marker| marker.ability_id).collect();
        let expected_ids: Vec<i32> = model.keys().copied().collect();
        prop_assert_eq!(ids, expected_ids, "evidence {} slots", index);

        for slot in slots {
            let mut stack = Vec::new();
            let mut current: *const Marker = slot;
            while !current.is_null() {
                let marker = unsafe { &*current };
                prop_assert_eq!(marker.ability_id, slot.ability_id);
                stack.push(ModelMarker {
                    prompt_tag: marker.prompt_tag,
                    tr: marker.tr_dispatch_fn.addr(),
                    outer: pool_index(marker.outer).unwrap_or(usize::MAX),
                });
                current = marker.shadowed;
            }
            stack.reverse();
            prop_assert_eq!(&stack, &model[&slot.ability_id], "evidence {} stack", index);
        }

        for id in EVIDENCE_IDS {
            let top = model.get(&id).and_then(|stack| stack.last());
            let expected_tail = top.map_or(index, |top| top.outer);
            prop_assert_eq!(
                unsafe { __tribute_evidence_tail(ev, id) },
                pool[expected_tail].cast_const()
            );
            if let Some(top) = top {
                prop_assert_eq!(unsafe { __tribute_evidence_lookup(ev, id) }, top.prompt_tag);
                prop_assert_eq!(
                    unsafe { __tribute_evidence_lookup_tr(ev, id) }.addr(),
                    top.tr
                );
                prop_assert_eq!(
                    unsafe { __tribute_evidence_outer(ev, id) },
                    pool[top.outer].cast_const()
                );
            }
        }
        Ok(())
    }

    proptest! {
        /// Evidence operations behave like persistent per-ability handler
        /// stacks: extend, `with_tail`, `push`, and `dup` push a handler,
        /// `mask` pops one (dropping the slot when it empties), queries read
        /// the top handler, a row tail is the top handler's `outer` or the
        /// evidence itself, and no operation changes an existing evidence.
        #[test]
        fn prop_evidence_matches_handler_stack_model(
            actions in prop::collection::vec(evidence_action(), 1..40),
        ) {
            let mut pool = vec![__tribute_evidence_empty()];
            let mut models = vec![ModelEvidence::new()];

            for action in actions {
                let pick = |index: Index| index.index(pool.len());
                let (ev, model) = match action {
                    EvidenceAction::Extend { base, ability, prompt_tag, tr, outer } => {
                        let (base, outer) = (pick(base), pick(outer));
                        let tr = tr * 0x10;
                        let ev = unsafe {
                            __tribute_evidence_extend(
                                pool[base],
                                ability,
                                prompt_tag,
                                core::ptr::without_provenance(tr),
                                pool[outer],
                            )
                        };
                        let mut model = models[base].clone();
                        model.entry(ability).or_default().push(ModelMarker { prompt_tag, tr, outer });
                        (ev, model)
                    }
                    EvidenceAction::WithTail { base, slot, tail } => {
                        let (base, tail) = (pick(base), pick(tail));
                        let ev = unsafe { __tribute_evidence_with_tail(pool[base], slot, pool[tail]) };
                        let mut model = models[base].clone();
                        model.entry(slot).or_default().push(ModelMarker {
                            prompt_tag: 0,
                            tr: 0,
                            outer: tail,
                        });
                        (ev, model)
                    }
                    EvidenceAction::Mask { base, ability } => {
                        let base = pick(base);
                        let Some(ability) = present_ability(&models[base], ability) else {
                            continue;
                        };
                        let ev = unsafe { __tribute_evidence_mask(pool[base], ability) };
                        let mut model = models[base].clone();
                        let stack = model.get_mut(&ability).expect("present ability");
                        stack.pop();
                        if stack.is_empty() {
                            model.remove(&ability);
                        }
                        (ev, model)
                    }
                    EvidenceAction::Dup { base, ability } => {
                        let base = pick(base);
                        let Some(ability) = present_ability(&models[base], ability) else {
                            continue;
                        };
                        let ev = unsafe { __tribute_evidence_dup(pool[base], ability) };
                        let mut model = models[base].clone();
                        let stack = model.get_mut(&ability).expect("present ability");
                        stack.push(*stack.last().expect("non-empty stack"));
                        (ev, model)
                    }
                    EvidenceAction::Push { base, source, ability } => {
                        let (base, source) = (pick(base), pick(source));
                        let Some(ability) = present_ability(&models[source], ability) else {
                            continue;
                        };
                        let ev =
                            unsafe { __tribute_evidence_push(pool[base], pool[source], ability) };
                        let top = *models[source][&ability].last().expect("non-empty stack");
                        let mut model = models[base].clone();
                        model.entry(ability).or_default().push(top);
                        (ev, model)
                    }
                };
                pool.push(ev);
                models.push(model);
                for index in 0..pool.len() {
                    check_evidence(&pool, &models, index)?;
                }
            }

            // Markers point into other evidences, so free them all at once.
            for ev in pool {
                let _ = unsafe { Box::from_raw(ev) };
            }
        }
    }
}

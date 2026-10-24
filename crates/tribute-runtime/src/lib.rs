//! Tribute runtime library.
//!
//! Provides the native runtime functions required by Tribute's compiled output:
//! - Heap allocation (`__tribute_alloc`, `__tribute_dealloc`)
//! - TLS-based tag generation for ability dispatch

#![cfg_attr(panic = "abort", no_std)]
#![allow(private_interfaces)]

extern crate alloc;

use alloc::vec::Vec;

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

/// Bytes payload layout: `[ptr] [len] [owner] [cap] [bytes...]`.
///
/// `ptr` and `len` are the range this value reads. `cap` bytes are stored
/// right after these fields, and `owner` is the `Bytes` that stores the range
/// when this value does not. See `new-plans/rc.md` (Bytes).
///
/// Compiler passes emit code that stores this layout after the RC header.
/// Runtime functions receive a pointer to this payload area (not the raw allocation).
#[repr(C)]
pub struct TributeBytes {
    pub ptr: *const u8,
    pub len: u64,
    pub owner: *const TributeBytes,
    pub cap: u64,
}

/// Allocate a `Bytes` that stores `cap` uninitialized bytes and reads all of
/// them. Returns the value and the address of its bytes.
fn allocate_storing_bytes(cap: u64) -> (*mut TributeBytes, *mut u8) {
    let fixed = core::mem::size_of::<tribute_rc::RcBox<TributeBytes>>() as u64;
    let Some(size) = fixed.checked_add(cap) else {
        oom_abort();
    };
    let raw = unsafe { __tribute_alloc(size) };
    let rc_box = unsafe { tribute_rc::RcBox::<TributeBytes>::init(raw, 0) };
    let data = if cap == 0 {
        core::ptr::null_mut()
    } else {
        unsafe { raw.add(fixed as usize) }
    };
    unsafe {
        (*rc_box).payload = TributeBytes {
            ptr: data,
            len: cap,
            owner: core::ptr::null(),
            cap,
        };
        (&raw mut (*rc_box).payload, data)
    }
}

/// Allocate a `Bytes` that reads `len` bytes at `ptr`, which `owner` stores.
/// The caller hands over one unit of a non-null `owner`.
fn allocate_sharing_bytes(
    ptr: *const u8,
    len: u64,
    owner: *const TributeBytes,
) -> *mut TributeBytes {
    let size = core::mem::size_of::<tribute_rc::RcBox<TributeBytes>>() as u64;
    let raw = unsafe { __tribute_alloc(size) };
    let rc_box = unsafe { tribute_rc::RcBox::<TributeBytes>::init(raw, 0) };
    unsafe {
        (*rc_box).payload = TributeBytes {
            ptr,
            len,
            owner,
            cap: 0,
        };
        &raw mut (*rc_box).payload
    }
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
    let (result, data) = allocate_storing_bytes(bytes.len() as u64);
    if !bytes.is_empty() {
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
    }
    result
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

/// Read one byte of a Bytes value, zero-extended.
///
/// Aborts if `index` is out of range.
///
/// Signature: `(bytes: ptr, index: u32) -> u32`
///
/// # Safety
///
/// `bytes` must be a valid pointer to a `TributeBytes` payload.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_bytes_get_or_panic(
    bytes: *const TributeBytes,
    index: u32,
) -> u32 {
    let b = unsafe { &*bytes };
    if u64::from(index) >= b.len {
        bounds_check_abort();
    }
    u32::from(unsafe { *b.ptr.add(index as usize) })
}

/// Concatenate two Bytes values, returning a new RC-managed Bytes that
/// stores the result.
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

    let (result, data) = allocate_storing_bytes(total_len);
    if a_ref.len > 0 && !a_ref.ptr.is_null() {
        unsafe {
            core::ptr::copy_nonoverlapping(a_ref.ptr, data, a_ref.len as usize);
        }
    }
    if b_ref.len > 0 && !b_ref.ptr.is_null() {
        unsafe {
            core::ptr::copy_nonoverlapping(
                b_ref.ptr,
                data.add(a_ref.len as usize),
                b_ref.len as usize,
            );
        }
    }
    result
}

/// Slice a Bytes value, returning a new RC-managed Bytes that reads part of
/// the original's bytes (zero-copy).
///
/// The slice retains the `Bytes` that stores the bytes: the original's
/// `owner` if it has one, the original itself if it stores bytes, and nothing
/// for static bytes. Releasing the slice releases that owner.
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

    let new_len = e - s;
    if new_len == 0 || b.ptr.is_null() {
        return allocate_sharing_bytes(core::ptr::null(), 0, core::ptr::null());
    }
    let owner = if !b.owner.is_null() {
        b.owner
    } else if b.cap != 0 {
        bytes
    } else {
        core::ptr::null()
    };
    if !owner.is_null() {
        unsafe { (*tribute_rc::RcBox::from_payload_ptr(owner)).retain() };
    }
    allocate_sharing_bytes(unsafe { b.ptr.add(s as usize) }, new_len, owner)
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
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::Ordering;
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
        assert!(payload.owner.is_null(), "test bytes store their own bytes");
        let raw = unsafe { tribute_rc::RcBox::from_payload_ptr_mut(bytes) }.cast();
        unsafe {
            __tribute_dealloc(
                raw,
                core::mem::size_of::<tribute_rc::RcBox<TributeBytes>>() as u64 + payload.cap,
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
            owner: core::ptr::null(),
            cap: 0,
        }
    }

    fn refcount(bytes: *const TributeBytes) -> u32 {
        let rc_box = unsafe { &*tribute_rc::RcBox::from_payload_ptr(bytes) };
        rc_box.refcount.load(Ordering::Relaxed)
    }

    #[test]
    fn concat_stores_its_bytes_in_the_result() {
        let (left, right) = (bytes_view(b"ab"), bytes_view(b"cde"));
        let joined = unsafe { __tribute_bytes_concat(&left, &right) };
        let payload = unsafe { &*joined };
        assert_eq!(unsafe { bytes_contents(joined) }, b"abcde");
        assert_eq!(payload.cap, 5);
        assert!(payload.owner.is_null());
        assert_eq!(
            payload.ptr,
            unsafe { joined.add(1) }.cast::<u8>().cast_const()
        );
        assert_eq!(
            unsafe { __tribute_bytes_get_or_panic(joined, 4) },
            u32::from(b'e')
        );
        unsafe { dealloc_test_bytes(joined) };
    }

    #[test]
    fn slice_retains_the_bytes_that_store_its_range() {
        let (left, right) = (bytes_view(b"ab"), bytes_view(b"cde"));
        let stored = unsafe { __tribute_bytes_concat(&left, &right) };
        let outer = unsafe { __tribute_bytes_slice_or_panic(stored, 1, 5) };
        let inner = unsafe { __tribute_bytes_slice_or_panic(outer, 1, 3) };
        assert_eq!(unsafe { bytes_contents(inner) }, b"cd");
        // Both slices hold the storing value, not one another.
        assert_eq!(unsafe { (*outer).owner }, stored.cast_const());
        assert_eq!(unsafe { (*inner).owner }, stored.cast_const());
        assert_eq!(refcount(stored), 3);
        assert_eq!(refcount(outer), 1);
        assert_eq!(unsafe { ((*inner).cap, (*outer).cap) }, (0, 0));

        // A slice of static bytes, and an empty slice, hold nothing.
        let literal = bytes_view(b"xyz");
        let of_literal = unsafe { __tribute_bytes_slice_or_panic(&literal, 1, 3) };
        let empty = unsafe { __tribute_bytes_slice_or_panic(stored, 2, 2) };
        assert_eq!(unsafe { bytes_contents(of_literal) }, b"yz");
        assert!(unsafe { (*of_literal).owner }.is_null());
        assert!(unsafe { (*empty).owner }.is_null());
        assert_eq!(refcount(stored), 3);

        let fixed = core::mem::size_of::<tribute_rc::RcBox<TributeBytes>>() as u64;
        for slice in [outer, inner, of_literal, empty] {
            let raw = unsafe { tribute_rc::RcBox::from_payload_ptr_mut(slice) }.cast();
            unsafe { __tribute_dealloc(raw, fixed) };
        }
        unsafe { dealloc_test_bytes(stored) };
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
}

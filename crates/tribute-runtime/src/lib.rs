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
// Byte array support
// =============================================================================

/// The reserved RTTI index of the byte array, the one descriptor number the
/// runtime shares with compiled code (`design/rc.md` § "RTTI Table").
pub const BYTE_ARRAY_RTTI_IDX: u32 = 0;

/// The payload of a byte array `adt.array<core.i8>`: its length, followed by
/// that many bytes (`design/rc.md` § "Array"). Runtime functions receive a
/// pointer to this payload, not to the raw allocation.
#[repr(C)]
pub struct ByteArray {
    pub len: u64,
}

impl ByteArray {
    /// The `len` bytes of the array from `start`, ending the execution when
    /// the range is not inside the array.
    ///
    /// # Safety
    ///
    /// `self` must be the payload of a live byte array.
    unsafe fn range(&self, start: impl Into<u64>, len: impl Into<u64>) -> &[u8] {
        let (start, len) = (start.into(), len.into());
        match start.checked_add(len) {
            Some(end) if end <= self.len => {}
            _ => bounds_check_abort(),
        }
        if len == 0 {
            return &[];
        }
        let elements = (self as *const Self).wrapping_add(1).cast::<u8>();
        unsafe { core::slice::from_raw_parts(elements.add(start as usize), len as usize) }
    }
}

/// Allocate a byte array of `len` uninitialized bytes with one unit. Returns
/// its payload and the address of its bytes.
fn allocate_byte_array(len: u64) -> (*mut ByteArray, *mut u8) {
    let fixed = core::mem::size_of::<tribute_rc::RcBox<ByteArray>>() as u64;
    let Some(size) = fixed.checked_add(len) else {
        oom_abort();
    };
    let raw = unsafe { __tribute_alloc(size) };
    let rc_box = unsafe { tribute_rc::RcBox::<ByteArray>::init(raw, BYTE_ARRAY_RTTI_IDX) };
    unsafe {
        (*rc_box).payload = ByteArray { len };
        (&raw mut (*rc_box).payload, raw.add(fixed as usize))
    }
}

/// A new byte array holding `bytes`.
fn byte_array_of(bytes: &[u8]) -> *mut ByteArray {
    let (array, data) = allocate_byte_array(bytes.len() as u64);
    if !bytes.is_empty() {
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
    }
    array
}

/// Copy `len` bytes at `data` into a new byte array, as a byte literal does.
///
/// # Safety
///
/// `data` must be readable for `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_byte_array_from(data: *const u8, len: u64) -> *mut ByteArray {
    if len == 0 {
        return byte_array_of(&[]);
    }
    byte_array_of(unsafe { core::slice::from_raw_parts(data, len as usize) })
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
    pub bytes: *mut ByteArray,
    pub message: *mut ByteArray,
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

/// Write `len` bytes of a byte array from `start`, optionally followed by one
/// newline. Ends the execution when the range is not inside the array.
///
/// # Safety
///
/// `data` must point to the payload of a live byte array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_io_write(
    data: *const ByteArray,
    start: usize,
    len: usize,
    newline: u32,
) {
    // `usize` is at most 64 bits wide on every supported target.
    let bytes = unsafe { (*data).range(start as u64, len as u64) };
    if !bytes.is_empty() {
        write_stdout_all(bytes);
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
/// The RC-managed `bytes` and `message` byte arrays, if any, have already been
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
                    message: byte_array_of(b"stdin read failed"),
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
        bytes: byte_array_of(&bytes),
        message: core::ptr::null_mut(),
    })
}

fn allocate_read_result(result: NativeReadLineResult) -> *mut NativeReadLineResult {
    let size = core::mem::size_of::<NativeReadLineResult>() as u64;
    let raw = unsafe { __tribute_alloc(size) }.cast::<NativeReadLineResult>();
    unsafe { raw.write(result) };
    raw
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

/// Compare `len` bytes of two byte arrays, from `left_start` and
/// `right_start`.
///
/// Returns `1` when the ranges contain the same bytes and `0` otherwise. This
/// is the native primitive behind `Bytes` and rope `String` equality: callers
/// select contiguous spans, so no flattening or temporary buffer is required.
/// Ends the execution when a range is not inside its array.
///
/// # Safety
///
/// `left` and `right` must point to the payloads of live byte arrays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_byte_array_range_equal(
    left: *const ByteArray,
    left_start: u32,
    right: *const ByteArray,
    right_start: u32,
    len: u32,
) -> u32 {
    let left = unsafe { (*left).range(left_start, len) };
    let right = unsafe { (*right).range(right_start, len) };
    u32::from(left == right)
}

/// Concatenate `left_len` bytes of `left` from `left_start` and `right_len`
/// bytes of `right` from `right_start` into a new byte array. Ends the
/// execution when a range is not inside its array.
///
/// # Safety
///
/// `left` and `right` must point to the payloads of live byte arrays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_byte_array_concat(
    left: *const ByteArray,
    left_start: u32,
    left_len: u32,
    right: *const ByteArray,
    right_start: u32,
    right_len: u32,
) -> *mut ByteArray {
    let left = unsafe { (*left).range(left_start, left_len) };
    let right = unsafe { (*right).range(right_start, right_len) };
    let total = left.len() as u64 + right.len() as u64;
    let (result, data) = allocate_byte_array(total);
    unsafe {
        core::ptr::copy_nonoverlapping(left.as_ptr(), data, left.len());
        core::ptr::copy_nonoverlapping(right.as_ptr(), data.add(left.len()), right.len());
    }
    result
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

    unsafe fn bytes_contents(bytes: *const ByteArray) -> Vec<u8> {
        if bytes.is_null() {
            return Vec::new();
        }
        let bytes = unsafe { &*bytes };
        unsafe { bytes.range(0u64, bytes.len) }.to_vec()
    }

    unsafe fn dealloc_test_bytes(bytes: *mut ByteArray) {
        if bytes.is_null() {
            return;
        }
        let len = unsafe { (*bytes).len };
        let raw = unsafe { tribute_rc::RcBox::from_payload_ptr_mut(bytes) }.cast();
        unsafe {
            __tribute_dealloc(
                raw,
                core::mem::size_of::<tribute_rc::RcBox<ByteArray>>() as u64 + len,
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

        let _: unsafe extern "C" fn(*const ByteArray, usize, usize, u32) = __tribute_io_write;
        let _: extern "C" fn() -> *mut NativeReadLineResult = __tribute_io_read_line;
        let _: unsafe extern "C" fn(*mut NativeReadLineResult) =
            __tribute_io_read_line_result_dealloc;
        let _: unsafe extern "C" fn(*const ByteArray, u32, *const ByteArray, u32, u32) -> u32 =
            __tribute_byte_array_range_equal;
        let _: unsafe extern "C" fn(
            *const ByteArray,
            u32,
            u32,
            *const ByteArray,
            u32,
            u32,
        ) -> *mut ByteArray = __tribute_byte_array_concat;
        let _: unsafe extern "C" fn(*const u8, u64) -> *mut ByteArray = __tribute_byte_array_from;
    }

    fn header(array: *const ByteArray) -> &'static tribute_rc::RcBox<ByteArray> {
        unsafe { &*tribute_rc::RcBox::from_payload_ptr(array) }
    }

    #[test]
    fn byte_arrays_use_the_array_layout_and_reserved_descriptor() {
        let array = unsafe { __tribute_byte_array_from(b"abc".as_ptr(), 3) };
        assert_eq!(unsafe { bytes_contents(array) }, b"abc");
        assert_eq!(unsafe { (*array).len }, 3);
        // The bytes follow the 8-byte length, as an `adt.array` element does.
        assert_eq!(
            unsafe { *array.cast::<u8>().add(8 + 2) },
            b'c',
            "the third byte"
        );
        assert_eq!(header(array).rtti_idx, BYTE_ARRAY_RTTI_IDX);
        assert_eq!(header(array).refcount.load(Ordering::Relaxed), 1);
        unsafe { dealloc_test_bytes(array) };

        let empty = unsafe { __tribute_byte_array_from(core::ptr::null(), 0) };
        assert_eq!(unsafe { (*empty).len }, 0);
        unsafe { dealloc_test_bytes(empty) };
    }

    #[test]
    fn concat_copies_both_ranges_into_a_new_array() {
        let left = byte_array_of(b"xabx");
        let right = byte_array_of(b"cde");
        let joined = unsafe { __tribute_byte_array_concat(left, 1, 2, right, 0, 3) };
        assert_eq!(unsafe { bytes_contents(joined) }, b"abcde");
        // The arguments are borrowed and keep their one unit.
        assert_eq!(header(left).refcount.load(Ordering::Relaxed), 1);
        for array in [left, right, joined] {
            unsafe { dealloc_test_bytes(array) };
        }
    }

    proptest! {
        /// A comparison of in-bounds ranges is `1` exactly when they hold the
        /// same bytes.
        #[test]
        fn prop_byte_array_range_equal_matches_slices(
            left in prop::collection::vec(0u8..2, 0..12),
            right in prop::collection::vec(0u8..2, 0..12),
            left_start in any::<Index>(),
            right_start in any::<Index>(),
            len in any::<Index>(),
        ) {
            let left_start = left_start.index(left.len() + 1);
            let right_start = right_start.index(right.len() + 1);
            let len = len.index((left.len() - left_start).min(right.len() - right_start) + 1);
            let expected =
                u32::from(left[left_start..left_start + len] == right[right_start..right_start + len]);
            let (left_array, right_array) = (byte_array_of(&left), byte_array_of(&right));
            let actual = unsafe {
                __tribute_byte_array_range_equal(
                    left_array,
                    left_start as u32,
                    right_array,
                    right_start as u32,
                    len as u32,
                )
            };
            unsafe {
                dealloc_test_bytes(left_array);
                dealloc_test_bytes(right_array);
            }
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

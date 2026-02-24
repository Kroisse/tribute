//! Allocator-level AddressSanitizer for Tribute-compiled programs.
//!
//! Provides red zone and quarantine based heap memory error detection.
//! Activated by `__asan_init()` which is called from the entrypoint when
//! `--sanitize=address` is passed to the compiler.
//!
//! ## Detection capabilities
//!
//! - **Heap buffer overflow/underflow**: Red zones (32 bytes) around each allocation
//!   are checked at deallocation time for corruption.
//! - **Use-after-free (probabilistic)**: Freed memory is filled with poison bytes
//!   and kept in a quarantine queue before actual deallocation.
//! - **Double-free**: Detected via allocation metadata tracking.

use alloc::collections::VecDeque;
use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Red zone size in bytes placed before and after each allocation.
const REDZONE_SIZE: usize = 32;

/// Magic byte written into freed memory regions.
const MAGIC_FREED: u8 = 0xFD;

/// Magic byte written into red zones at allocation time.
const MAGIC_REDZONE: u8 = 0xCC;

/// Maximum total bytes held in the quarantine before oldest entries are freed.
const QUARANTINE_MAX: usize = 1 << 20; // 1 MiB

/// Alignment used for all ASan allocations (matches tribute's 8-byte alignment).
const ALLOC_ALIGN: usize = 8;

/// Global flag: ASan is active.
static ASAN_ENABLED: AtomicBool = AtomicBool::new(false);

/// Check whether ASan is currently enabled.
#[inline]
pub fn is_enabled() -> bool {
    ASAN_ENABLED.load(Ordering::Relaxed)
}

// =============================================================================
// Quarantine
// =============================================================================

struct QuarantineEntry {
    /// Pointer to the base of the allocation (start of left red zone).
    base: *mut u8,
    /// Total allocation size including both red zones.
    total_size: usize,
}

/// Global quarantine state.
///
/// Safety: Tribute programs are single-threaded (fibers on one OS thread),
/// so mutable static access is safe as long as we don't re-enter from a
/// signal handler (which we avoid).
///
/// We use `UnsafeCell` to avoid `static mut` (denied in Rust 2024 edition).
struct QuarantineState {
    queue: UnsafeCell<Option<VecDeque<QuarantineEntry>>>,
}

unsafe impl Sync for QuarantineState {}

static QUARANTINE: QuarantineState = QuarantineState {
    queue: UnsafeCell::new(None),
};
static QUARANTINE_TOTAL: AtomicUsize = AtomicUsize::new(0);

unsafe fn quarantine() -> &'static mut VecDeque<QuarantineEntry> {
    unsafe { (*QUARANTINE.queue.get()).get_or_insert_with(VecDeque::new) }
}

/// Push an entry into the quarantine. If the quarantine exceeds its size
/// limit, the oldest entries are actually freed.
unsafe fn quarantine_push(base: *mut u8, total_size: usize) {
    let q = unsafe { quarantine() };
    q.push_back(QuarantineEntry { base, total_size });
    QUARANTINE_TOTAL.fetch_add(total_size, Ordering::Relaxed);

    // Evict oldest entries when over budget
    while QUARANTINE_TOTAL.load(Ordering::Relaxed) > QUARANTINE_MAX {
        if let Some(old) = q.pop_front() {
            if let Ok(layout) = Layout::from_size_align(old.total_size, ALLOC_ALIGN) {
                unsafe { alloc::alloc::dealloc(old.base, layout) };
            }
            QUARANTINE_TOTAL.fetch_sub(old.total_size, Ordering::Relaxed);
        } else {
            break;
        }
    }
}

// =============================================================================
// Error reporting
// =============================================================================

/// Write an error message to stderr using raw syscall (no_std compatible).
fn write_stderr(msg: &[u8]) {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn write(fd: i32, buf: *const u8, count: usize) -> isize;
        }
        unsafe {
            write(2, msg.as_ptr(), msg.len());
        }
    }
    #[cfg(windows)]
    {
        // Fallback: Windows stderr via GetStdHandle + WriteFile
        // For now, silently drop — Windows ASan support is secondary.
    }
}

/// Report a red zone violation and abort.
fn report_redzone_corruption(side: &str, offset: usize, expected: u8, actual: u8) -> ! {
    // Use a stack buffer for formatting (no_std compatible)
    let mut buf = [0u8; 256];
    let msg = format_redzone_error(&mut buf, side, offset, expected, actual);
    write_stderr(msg);

    unsafe extern "C" {
        fn abort() -> !;
    }
    unsafe { abort() }
}

/// Format a red zone error message into a stack buffer.
fn format_redzone_error<'a>(
    buf: &'a mut [u8; 256],
    side: &str,
    offset: usize,
    expected: u8,
    actual: u8,
) -> &'a [u8] {
    use core::fmt::Write;

    struct BufWriter<'b> {
        buf: &'b mut [u8],
        pos: usize,
    }

    impl<'b> Write for BufWriter<'b> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            let remaining = self.buf.len() - self.pos;
            let to_copy = bytes.len().min(remaining);
            self.buf[self.pos..self.pos + to_copy].copy_from_slice(&bytes[..to_copy]);
            self.pos += to_copy;
            Ok(())
        }
    }

    let mut w = BufWriter { buf, pos: 0 };
    let _ = write!(
        w,
        "==ERROR: TributeASan: heap-buffer-overflow\n  {} redzone corrupted at byte {} (expected 0x{:02X}, got 0x{:02X})\n",
        side, offset, expected, actual,
    );
    &w.buf[..w.pos]
}

// =============================================================================
// Red zone checking
// =============================================================================

/// Check that a red zone region contains only the expected magic byte.
/// Aborts with an error report on first corrupted byte.
unsafe fn check_redzone(ptr: *const u8, len: usize, side: &str) {
    for i in 0..len {
        let byte = unsafe { ptr.add(i).read() };
        if byte != MAGIC_REDZONE {
            report_redzone_corruption(side, i, MAGIC_REDZONE, byte);
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Initialize the ASan subsystem.
///
/// Called from the entrypoint before `__tribute_init()` when `--sanitize=address`
/// is used. Sets the global flag so that `__tribute_alloc`/`__tribute_dealloc`
/// route through the ASan allocator.
#[unsafe(no_mangle)]
pub extern "C" fn __asan_init() {
    ASAN_ENABLED.store(true, Ordering::SeqCst);
    // Pre-allocate the quarantine to avoid allocation during dealloc
    unsafe {
        *QUARANTINE.queue.get() = Some(VecDeque::with_capacity(64));
    }
}

/// ASan-instrumented allocation.
///
/// Layout: `[LEFT_REDZONE(32B)] [payload(size)] [RIGHT_REDZONE(32B)]`
///
/// Returns a pointer to the payload region. The caller sees the same pointer
/// semantics as a normal `__tribute_alloc` call.
///
/// # Safety
///
/// Same preconditions as `__tribute_alloc`.
pub unsafe fn alloc(size: usize) -> *mut u8 {
    let Some(total) = REDZONE_SIZE
        .checked_add(size)
        .and_then(|s| s.checked_add(REDZONE_SIZE))
    else {
        return core::ptr::null_mut();
    };
    let Ok(layout) = Layout::from_size_align(total, ALLOC_ALIGN) else {
        return core::ptr::null_mut();
    };
    let base = unsafe { alloc::alloc::alloc(layout) };
    if base.is_null() {
        return core::ptr::null_mut();
    }

    // Fill left red zone
    unsafe { core::ptr::write_bytes(base, MAGIC_REDZONE, REDZONE_SIZE) };

    let payload = unsafe { base.add(REDZONE_SIZE) };

    // Fill right red zone
    unsafe { core::ptr::write_bytes(payload.add(size), MAGIC_REDZONE, REDZONE_SIZE) };

    payload
}

/// ASan-instrumented deallocation.
///
/// Checks red zone integrity, poisons the entire allocation, and places it
/// in the quarantine instead of immediately freeing.
///
/// # Safety
///
/// `ptr` must have been returned by `asan::alloc` with the same `size`.
pub unsafe fn dealloc(ptr: *mut u8, size: usize) {
    let Some(total) = REDZONE_SIZE
        .checked_add(size)
        .and_then(|s| s.checked_add(REDZONE_SIZE))
    else {
        // Corrupted size — abort rather than silently ignoring
        write_stderr(b"==ERROR: TributeASan: dealloc called with overflowing size\n");
        unsafe extern "C" {
            fn abort() -> !;
        }
        unsafe { abort() }
    };

    let base = unsafe { ptr.sub(REDZONE_SIZE) };

    // Check red zone integrity
    unsafe { check_redzone(base, REDZONE_SIZE, "left") };
    unsafe { check_redzone(ptr.add(size), REDZONE_SIZE, "right") };

    // Poison the entire region (red zones + payload) with MAGIC_FREED
    unsafe { core::ptr::write_bytes(base, MAGIC_FREED, total) };

    // Move to quarantine instead of immediately freeing
    unsafe { quarantine_push(base, total) };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutex to serialize tests that mutate global ASan state.
    ///
    /// Tests modify `ASAN_ENABLED`, `QUARANTINE`, and `QUARANTINE_TOTAL` which
    /// are process-wide globals. Without serialization, parallel test execution
    /// causes data races.
    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Reset quarantine state for test isolation.
    unsafe fn reset_quarantine() {
        unsafe { *QUARANTINE.queue.get() = Some(VecDeque::new()) };
        QUARANTINE_TOTAL.store(0, Ordering::Relaxed);
    }

    #[test]
    fn test_asan_alloc_dealloc() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        unsafe {
            let ptr = alloc(64);
            assert!(!ptr.is_null());

            // Write some data to the payload
            core::ptr::write_bytes(ptr, 0x42, 64);

            // Dealloc should succeed (red zones intact)
            dealloc(ptr, 64);
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_asan_redzone_intact() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        unsafe {
            let ptr = alloc(32);
            assert!(!ptr.is_null());

            // Verify left red zone contains magic bytes
            let base = ptr.sub(REDZONE_SIZE);
            for i in 0..REDZONE_SIZE {
                assert_eq!(base.add(i).read(), MAGIC_REDZONE);
            }

            // Verify right red zone contains magic bytes
            for i in 0..REDZONE_SIZE {
                assert_eq!(ptr.add(32 + i).read(), MAGIC_REDZONE);
            }

            dealloc(ptr, 32);
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_asan_freed_memory_poisoned() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        unsafe {
            let ptr = alloc(16);
            assert!(!ptr.is_null());

            // Remember the base address (before left redzone)
            let base = ptr.sub(REDZONE_SIZE);
            let total = REDZONE_SIZE + 16 + REDZONE_SIZE;

            dealloc(ptr, 16);

            // The entire region should now be MAGIC_FREED
            // (only safe to read because the quarantine holds it)
            for i in 0..total {
                assert_eq!(
                    base.add(i).read(),
                    MAGIC_FREED,
                    "byte at offset {} was not poisoned",
                    i
                );
            }
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_asan_quarantine_eviction() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        // Allocate and free enough to exceed QUARANTINE_MAX (1 MiB)
        let alloc_size = 1024; // 1 KiB payload
        let count = (QUARANTINE_MAX / (alloc_size + 2 * REDZONE_SIZE)) + 2;

        unsafe {
            for _ in 0..count {
                let ptr = alloc(alloc_size);
                assert!(!ptr.is_null());
                dealloc(ptr, alloc_size);
            }

            // Quarantine should have evicted some entries
            assert!(
                QUARANTINE_TOTAL.load(Ordering::Relaxed)
                    <= QUARANTINE_MAX + alloc_size + 2 * REDZONE_SIZE
            );
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }
}

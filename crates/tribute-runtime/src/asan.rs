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
//! - **Use-after-free**: Freed memory is filled with poison bytes and kept in a
//!   quarantine queue before actual deallocation.
//! - **Double-free**: A second free of a quarantined block is reported as a
//!   double free, and a free of any address without a live block, including a
//!   block that already left the quarantine, as a free of unallocated memory.
//! - **Invalid access at the access site**: Compiled code calls
//!   [`__tribute_asan_load`] or [`__tribute_asan_store`] before each memory
//!   access; the region table classifies the address range.

use alloc::collections::{BTreeMap, VecDeque};
use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

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
// Shared state
// =============================================================================

struct QuarantineEntry {
    /// Address of the base of the allocation (start of left red zone).
    base: usize,
    /// Total allocation size including both red zones.
    total_size: usize,
}

/// One block this allocator handed out: both red zones and the payload.
#[derive(Clone, Copy)]
struct Region {
    /// Payload size the caller asked for.
    payload: usize,
    /// The block was freed and is held in the quarantine.
    freed: bool,
}

impl Region {
    fn total(self) -> usize {
        REDZONE_SIZE + self.payload + REDZONE_SIZE
    }
}

/// Everything the sanitizer records about the heap.
struct State {
    /// Live and quarantined blocks by base address.
    regions: BTreeMap<usize, Region>,
    /// Freed blocks, oldest first, kept until the budget is exceeded.
    quarantine: VecDeque<QuarantineEntry>,
    /// Total bytes held in `quarantine`.
    quarantine_total: usize,
}

/// The sanitizer state behind a spin lock.
///
/// Compiled code may allocate, free and access memory from several threads,
/// and a free evicts quarantine entries that the region table also names, so
/// one lock covers both. The runtime has no `std` in an aborting build, hence
/// the spin lock; every critical section is a few table operations.
struct StateLock {
    locked: AtomicBool,
    state: UnsafeCell<State>,
}

// Safety: `state` is only reached through `with_state`, which holds `locked`.
unsafe impl Sync for StateLock {}

static STATE: StateLock = StateLock {
    locked: AtomicBool::new(false),
    state: UnsafeCell::new(State {
        regions: BTreeMap::new(),
        quarantine: VecDeque::new(),
        quarantine_total: 0,
    }),
};

/// Run `f` with exclusive access to the sanitizer state.
///
/// `f` must not report: a report aborts while the lock is held, which is
/// harmless, but it must not call back into `with_state`.
fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    struct Unlock;
    impl Drop for Unlock {
        fn drop(&mut self) {
            STATE.locked.store(false, Ordering::Release);
        }
    }

    while STATE
        .locked
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
    let _unlock = Unlock;
    f(unsafe { &mut *STATE.state.get() })
}

impl State {
    /// Push a freed block into the quarantine. If the quarantine exceeds its
    /// size limit, the oldest entries are actually freed and forgotten.
    fn quarantine_push(&mut self, base: usize, total_size: usize) {
        self.quarantine
            .push_back(QuarantineEntry { base, total_size });
        self.quarantine_total += total_size;

        // Evict oldest entries when over budget
        while self.quarantine_total > QUARANTINE_MAX {
            let Some(old) = self.quarantine.pop_front() else {
                break;
            };
            self.regions.remove(&old.base);
            if let Ok(layout) = Layout::from_size_align(old.total_size, ALLOC_ALIGN) {
                unsafe { alloc::alloc::dealloc(old.base as *mut u8, layout) };
            }
            self.quarantine_total -= old.total_size;
        }
    }
}

/// Mark the live block at `base` freed, or name why it cannot be freed.
///
/// A block that already left the quarantine has no entry any more, so a second
/// free of it is reported as a free of memory that is not allocated.
fn mark_freed(base: usize) -> Result<(), &'static str> {
    with_state(|state| match state.regions.get_mut(&base) {
        Some(region) if region.freed => Err("attempting double-free"),
        Some(region) => {
            region.freed = true;
            Ok(())
        }
        None => Err("attempting free of memory that is not allocated"),
    })
}

/// What an access of `size` bytes at `addr` touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// Inside a live payload, or memory this allocator does not own.
    Valid,
    /// Overlaps a red zone of a live allocation.
    HeapBufferOverflow,
    /// Overlaps a block held in the quarantine.
    HeapUseAfterFree,
}

/// Classify an access against the region table.
pub(crate) fn classify(addr: usize, size: usize) -> Access {
    let Some(end) = addr.checked_add(size) else {
        return Access::Valid;
    };
    if size == 0 {
        return Access::Valid;
    }
    // The last block that starts before the access ends is the only one the
    // access can overlap, since blocks are disjoint.
    let found = with_state(|state| {
        state
            .regions
            .range(..end)
            .next_back()
            .map(|(&base, &region)| (base, region))
    });
    let Some((base, region)) = found else {
        return Access::Valid;
    };
    if base + region.total() <= addr {
        return Access::Valid;
    }
    if region.freed {
        return Access::HeapUseAfterFree;
    }
    let payload = base + REDZONE_SIZE;
    if payload <= addr && end <= payload + region.payload {
        Access::Valid
    } else {
        Access::HeapBufferOverflow
    }
}

// =============================================================================
// Error reporting
// =============================================================================

struct BufWriter<'b> {
    buf: &'b mut [u8],
    pos: usize,
}

impl core::fmt::Write for BufWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = self.buf.len() - self.pos;
        let to_copy = bytes.len().min(remaining);
        self.buf[self.pos..self.pos + to_copy].copy_from_slice(&bytes[..to_copy]);
        self.pos += to_copy;
        Ok(())
    }
}

/// Format a report into a stack buffer (no_std compatible), truncating it to
/// the buffer.
fn format_report<'a>(buf: &'a mut [u8; 256], args: core::fmt::Arguments<'_>) -> &'a [u8] {
    use core::fmt::Write;

    let mut w = BufWriter { buf, pos: 0 };
    let _ = w.write_fmt(args);
    &w.buf[..w.pos]
}

/// Write a formatted report to stderr and abort.
fn report(args: core::fmt::Arguments<'_>) -> ! {
    let mut buf = [0u8; 256];
    write_stderr(format_report(&mut buf, args));

    unsafe extern "C" {
        fn abort() -> !;
    }
    unsafe { abort() }
}

impl Access {
    /// The report name of an invalid access.
    fn error(self) -> Option<&'static str> {
        match self {
            Access::Valid => None,
            Access::HeapBufferOverflow => Some("heap-buffer-overflow"),
            Access::HeapUseAfterFree => Some("heap-use-after-free"),
        }
    }
}

/// Check one access made by compiled code and abort with a report if it
/// touches a red zone or freed memory.
fn check_access(addr: usize, size: usize, kind: &str) {
    if let Some(error) = classify(addr, size).error() {
        report(format_args!(
            "==ERROR: TributeASan: {error} on address {addr:#x}\n  {kind} of size {size}\n"
        ));
    }
}

/// Write an error message to stderr using raw syscall (no_std compatible).
fn write_stderr(msg: &[u8]) {
    #[cfg(unix)]
    {
        unsafe {
            libc::write(libc::STDERR_FILENO, msg.as_ptr().cast(), msg.len());
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
    report(format_args!(
        "==ERROR: TributeASan: heap-buffer-overflow\n  {side} redzone corrupted at byte {offset} (expected 0x{expected:02X}, got 0x{actual:02X})\n"
    ))
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
    with_state(|state| state.quarantine.reserve(64));
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
        crate::oom_abort();
    };
    let Ok(layout) = Layout::from_size_align(total, ALLOC_ALIGN) else {
        crate::oom_abort();
    };
    let base = unsafe { alloc::alloc::alloc(layout) };
    if base.is_null() {
        crate::oom_abort();
    }

    // Fill left red zone
    unsafe { core::ptr::write_bytes(base, MAGIC_REDZONE, REDZONE_SIZE) };

    let payload = unsafe { base.add(REDZONE_SIZE) };

    // Fill right red zone
    unsafe { core::ptr::write_bytes(payload.add(size), MAGIC_REDZONE, REDZONE_SIZE) };

    with_state(|state| {
        state.regions.insert(
            base as usize,
            Region {
                payload: size,
                freed: false,
            },
        )
    });

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

    // `ptr` may not point into a block of this allocator, so its base is
    // looked up as an integer before any pointer is derived from it.
    if let Err(error) = mark_freed((ptr as usize).wrapping_sub(REDZONE_SIZE)) {
        report(format_args!(
            "==ERROR: TributeASan: {error} on address {:#x}\n",
            ptr as usize
        ));
    }
    let base = unsafe { ptr.sub(REDZONE_SIZE) };

    // Check red zone integrity
    unsafe { check_redzone(base, REDZONE_SIZE, "left") };
    unsafe { check_redzone(ptr.add(size), REDZONE_SIZE, "right") };

    // Poison the entire region (red zones + payload) with MAGIC_FREED
    unsafe { core::ptr::write_bytes(base, MAGIC_FREED, total) };

    // Move to quarantine instead of immediately freeing
    with_state(|state| state.quarantine_push(base as usize, total));
}

/// Check a read of `size` bytes at `addr` before compiled code performs it.
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_asan_load(addr: *const u8, size: u64) {
    check_access(addr as usize, size as usize, "READ");
}

/// Check a write of `size` bytes at `addr` before compiled code performs it.
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_asan_store(addr: *const u8, size: u64) {
    check_access(addr as usize, size as usize, "WRITE");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutex to serialize tests that mutate global ASan state.
    ///
    /// Tests modify `ASAN_ENABLED` and reset the shared sanitizer state, which
    /// are process-wide globals. Without serialization, one test would observe
    /// another's blocks.
    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Reset quarantine state for test isolation.
    unsafe fn reset_quarantine() {
        with_state(|state| {
            state.regions.clear();
            state.quarantine.clear();
            state.quarantine_total = 0;
        });
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
                with_state(|state| state.quarantine_total)
                    <= QUARANTINE_MAX + alloc_size + 2 * REDZONE_SIZE
            );
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_asan_classifies_accesses_by_region() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        unsafe {
            let live = alloc(16) as usize;
            // The whole payload, and a zero-width access at its end, are valid.
            assert_eq!(classify(live, 16), Access::Valid);
            assert_eq!(classify(live + 8, 8), Access::Valid);
            assert_eq!(classify(live + 16, 0), Access::Valid);
            // Either red zone, or an access that straddles one, overflows.
            assert_eq!(classify(live - 1, 1), Access::HeapBufferOverflow);
            assert_eq!(classify(live + 16, 1), Access::HeapBufferOverflow);
            assert_eq!(classify(live + 12, 8), Access::HeapBufferOverflow);
            assert_eq!(
                classify(live - REDZONE_SIZE - 4, 8),
                Access::HeapBufferOverflow
            );
            // Memory this allocator never handed out is not its concern.
            let local = 0u64;
            assert_eq!(classify(&raw const local as usize, 8), Access::Valid);

            let freed = alloc(24);
            dealloc(freed, 24);
            let freed = freed as usize;
            assert_eq!(classify(freed, 8), Access::HeapUseAfterFree);
            assert_eq!(classify(freed + 23, 1), Access::HeapUseAfterFree);
            assert_eq!(classify(freed - 8, 8), Access::HeapUseAfterFree);

            dealloc(live as *mut u8, 16);
            assert_eq!(classify(live, 16), Access::HeapUseAfterFree);
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_asan_forgets_regions_evicted_from_quarantine() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        unsafe {
            let first = alloc(1024);
            dealloc(first, 1024);
            let base = first as usize - REDZONE_SIZE;
            assert!(with_state(|state| state.regions.contains_key(&base)));
            for _ in 0..(QUARANTINE_MAX / 1024 + 2) {
                dealloc(alloc(1024), 1024);
            }
            with_state(|state| assert_eq!(state.regions.len(), state.quarantine.len()));
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_asan_valid_accesses_return_from_the_checks() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        unsafe {
            let live = alloc(16);
            __tribute_asan_store(live, 16);
            __tribute_asan_load(live.add(8), 8);
            dealloc(live, 16);
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_asan_reports_name_the_violation_and_fit_the_buffer() {
        assert_eq!(Access::Valid.error(), None);
        assert_eq!(
            Access::HeapBufferOverflow.error(),
            Some("heap-buffer-overflow")
        );
        assert_eq!(
            Access::HeapUseAfterFree.error(),
            Some("heap-use-after-free")
        );

        let mut buf = [0u8; 256];
        let report = format_report(
            &mut buf,
            format_args!("{} on address {:#x}\n  READ of size {}\n", "name", 0x10, 8),
        );
        assert_eq!(report, b"name on address 0x10\n  READ of size 8\n");

        let mut buf = [0u8; 256];
        let long = [b'x'; 300];
        let report = format_report(
            &mut buf,
            format_args!("{}", core::str::from_utf8(&long).unwrap()),
        );
        assert_eq!(report.len(), 256);
    }

    #[test]
    fn test_asan_rejects_a_second_or_unknown_free() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        unsafe {
            let ptr = alloc(16);
            let base = ptr as usize - REDZONE_SIZE;
            dealloc(ptr, 16);
            assert_eq!(mark_freed(base), Err("attempting double-free"));

            // Once the block leaves the quarantine its entry is gone.
            for _ in 0..(QUARANTINE_MAX / 1024 + 2) {
                dealloc(alloc(1024), 1024);
            }
            assert_eq!(
                mark_freed(base),
                Err("attempting free of memory that is not allocated")
            );
            let local = 0u64;
            assert_eq!(
                mark_freed(&raw const local as usize),
                Err("attempting free of memory that is not allocated")
            );
        }

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_asan_state_is_consistent_under_concurrent_use() {
        let _lock = TEST_MUTEX.lock().unwrap();
        ASAN_ENABLED.store(true, Ordering::SeqCst);
        unsafe { reset_quarantine() };

        const THREADS: usize = 8;
        const ROUNDS: usize = 2000;
        std::thread::scope(|scope| {
            for thread in 0..THREADS {
                scope.spawn(move || {
                    for round in 0..ROUNDS {
                        let size = 8 + (thread + round) % 64;
                        unsafe {
                            let ptr = alloc(size);
                            assert_eq!(classify(ptr as usize, size), Access::Valid);
                            assert_eq!(
                                classify(ptr as usize + size, 1),
                                Access::HeapBufferOverflow
                            );
                            dealloc(ptr, size);
                        }
                    }
                });
            }
        });

        // Every block was freed, so the table holds exactly the quarantine.
        with_state(|state| {
            assert!(state.regions.values().all(|region| region.freed));
            assert_eq!(state.regions.len(), state.quarantine.len());
            let held: usize = state.quarantine.iter().map(|entry| entry.total_size).sum();
            assert_eq!(held, state.quarantine_total);
        });

        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }
}

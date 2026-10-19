//! Allocator-level AddressSanitizer for Tribute-compiled programs.
//!
//! Provides red zone, shadow memory and quarantine based heap memory error
//! detection. Activated by `__asan_init()` which is called from the entrypoint
//! when `--sanitize=address` is passed to the compiler.
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
//! - **Leaks**: [`__asan_exit`] reports allocations still live when `main`
//!   returns.
//! - **Invalid access at the access site**: Compiled code calls
//!   [`__tribute_asan_load`] or [`__tribute_asan_store`] before each memory
//!   access; the shadow bytes of the address range classify it.
//!
//! ## Shadow memory
//!
//! One shadow byte describes each 8-byte granule of the address space, at
//! `base + addr / 8`. `__asan_init` reserves the whole shadow as one lazily
//! committed anonymous mapping, so memory this allocator never touched reads
//! as 0, which is valid. An access check reads shadow bytes and takes no lock.

use alloc::collections::VecDeque;
use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicUsize, Ordering};

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

/// Bytes of memory one shadow byte describes.
const GRANULE: usize = 8;

/// Shadow of a granule whose every byte is addressable. A value from 1 to 7
/// makes only that many leading bytes addressable.
const SHADOW_VALID: u8 = 0;

/// Shadow of a red zone granule of a live block.
const SHADOW_REDZONE: u8 = 0xFA;

/// Shadow of the last left red zone granule of a live block, which holds the
/// block's state.
const SHADOW_ALLOCATED: u8 = 0xFC;

/// Shadow of the granules of a quarantined block other than its state granule.
const SHADOW_FREED: u8 = 0xFD;

/// Shadow of the state granule of a quarantined block.
const SHADOW_FREED_STATE: u8 = 0xFE;

/// Widths of the user address space the shadow may cover, widest first.
///
/// Linux on aarch64 configures a 48-, 42- or 39-bit user address space by
/// kernel build, and refuses a shadow larger than that space, so the
/// reservation falls back to narrower widths.
#[cfg(target_pointer_width = "32")]
const ADDRESS_BITS: &[u32] = &[32];
#[cfg(all(
    target_pointer_width = "64",
    target_os = "linux",
    target_arch = "aarch64"
))]
const ADDRESS_BITS: &[u32] = &[48, 47, 42, 39];
#[cfg(all(
    target_pointer_width = "64",
    not(all(target_os = "linux", target_arch = "aarch64"))
))]
const ADDRESS_BITS: &[u32] = &[47];

/// Global flag: ASan is active.
static ASAN_ENABLED: AtomicBool = AtomicBool::new(false);

/// Start of the shadow reservation, or null before `__asan_init`.
static SHADOW_BASE: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());

/// Length of the shadow reservation in bytes, stored before `SHADOW_BASE`.
static SHADOW_LEN: AtomicUsize = AtomicUsize::new(0);

/// Number of blocks allocated and not yet freed.
static LIVE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Payload bytes of the blocks counted by `LIVE_COUNT`.
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Check whether ASan is currently enabled.
#[inline]
pub fn is_enabled() -> bool {
    ASAN_ENABLED.load(Ordering::Relaxed)
}

// =============================================================================
// Shadow memory
// =============================================================================

/// The shadow bytes of the granules that `[addr, addr + len)` touches, or
/// `None` if the range is empty, the shadow is not reserved, or the range
/// lies outside the covered address space.
fn shadow(addr: usize, len: usize) -> Option<&'static [AtomicU8]> {
    let base = SHADOW_BASE.load(Ordering::Acquire);
    if base.is_null() || len == 0 {
        return None;
    }
    let first = addr / GRANULE;
    let last = addr.checked_add(len - 1)? / GRANULE;
    if last >= SHADOW_LEN.load(Ordering::Relaxed) {
        return None;
    }
    // Safety: the reservation spans `SHADOW_LEN` bytes and is never unmapped.
    Some(unsafe {
        core::slice::from_raw_parts(base.add(first).cast::<AtomicU8>(), last - first + 1)
    })
}

/// Set the shadow of the granule aligned range `[addr, addr + len)`.
fn fill_shadow(addr: usize, len: usize, value: u8) {
    for byte in shadow(addr, len).unwrap_or_default() {
        byte.store(value, Ordering::Relaxed);
    }
}

#[cfg(unix)]
fn reserve_shadow(len: usize) -> Result<*mut u8, i32> {
    let mapping = unsafe {
        libc::mmap(
            ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if mapping == libc::MAP_FAILED {
        return Err(crate::last_errno());
    }
    // A core dump walks every page of a mapping, which takes minutes for one
    // this large even when it is untouched, and a report aborts the process.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    unsafe {
        libc::madvise(mapping, len, libc::MADV_DONTDUMP)
    };
    Ok(mapping.cast())
}

#[cfg(not(unix))]
fn reserve_shadow(_len: usize) -> Result<*mut u8, i32> {
    report(format_args!(
        "==ERROR: TributeASan: shadow memory is not supported on this platform\n"
    ))
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

/// The quarantine of freed blocks.
struct State {
    /// Freed blocks, oldest first, kept until the budget is exceeded.
    quarantine: VecDeque<QuarantineEntry>,
    /// Total bytes held in `quarantine`.
    quarantine_total: usize,
}

/// The quarantine behind a spin lock.
///
/// Compiled code may free memory from several threads; only the free path
/// takes this lock. The runtime has no `std` in an aborting build, hence the
/// spin lock; every critical section is a few queue operations.
struct StateLock {
    locked: AtomicBool,
    state: UnsafeCell<State>,
}

// Safety: `state` is only reached through `with_state`, which holds `locked`.
unsafe impl Sync for StateLock {}

static STATE: StateLock = StateLock {
    locked: AtomicBool::new(false),
    state: UnsafeCell::new(State {
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
            self.quarantine_total -= old.total_size;
            release(old);
        }
    }
}

/// Return an evicted block to the underlying allocator.
///
/// The shadow is reset first: once freed, the memory may back a buffer the
/// runtime allocates itself, which compiled code then reads.
fn release(entry: QuarantineEntry) {
    fill_shadow(entry.base, entry.total_size, SHADOW_VALID);
    if let Ok(layout) = Layout::from_size_align(entry.total_size, ALLOC_ALIGN) {
        unsafe { alloc::alloc::dealloc(entry.base as *mut u8, layout) };
    }
}

/// Mark the live block whose payload starts at `payload` freed, or name why
/// it cannot be freed.
///
/// The state granule just before the payload must say the block is live; the
/// compare-and-swap lets only one of two concurrent frees pass. A block that
/// already left the quarantine has shadow 0 again, so a second free of it is
/// reported as a free of memory that is not allocated.
fn mark_freed(payload: usize) -> Result<(), &'static str> {
    const NOT_ALLOCATED: &str = "attempting free of memory that is not allocated";
    if !payload.is_multiple_of(GRANULE) {
        return Err(NOT_ALLOCATED);
    }
    let Some([state]) = payload
        .checked_sub(GRANULE)
        .and_then(|addr| shadow(addr, 1))
    else {
        return Err(NOT_ALLOCATED);
    };
    match state.compare_exchange(
        SHADOW_ALLOCATED,
        SHADOW_FREED_STATE,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => Ok(()),
        Err(SHADOW_FREED_STATE) => Err("attempting double-free"),
        Err(_) => Err(NOT_ALLOCATED),
    }
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

/// Classify an access by the shadow of the granules it touches.
pub(crate) fn classify(addr: usize, size: usize) -> Access {
    let Some(shadow) = shadow(addr, size) else {
        return Access::Valid;
    };
    let end = addr + size;
    for (granule, byte) in (addr / GRANULE..).zip(shadow) {
        let start = granule * GRANULE;
        match byte.load(Ordering::Relaxed) {
            SHADOW_VALID => {}
            valid @ 1..8 if end.min(start + GRANULE) - start <= usize::from(valid) => {}
            SHADOW_FREED | SHADOW_FREED_STATE => return Access::HeapUseAfterFree,
            _ => return Access::HeapBufferOverflow,
        }
    }
    Access::Valid
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

/// Report a refused shadow reservation and abort.
fn report_reservation_failure(len: usize, errno: i32) -> ! {
    report(format_args!(
        "==ERROR: TributeASan: cannot reserve {len} bytes of shadow memory (errno {errno})\n  a virtual memory limit or strict overcommit refused the mapping\n"
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
/// is used. Reserves the shadow, then sets the global flag so that
/// `__tribute_alloc`/`__tribute_dealloc` route through the ASan allocator.
/// A second call keeps the shadow the first one reserved.
#[unsafe(no_mangle)]
pub extern "C" fn __asan_init() {
    with_state(|state| {
        if SHADOW_BASE.load(Ordering::Acquire).is_null() {
            let (mapping, len) = reserve_widest_shadow()
                .unwrap_or_else(|(len, errno)| report_reservation_failure(len, errno));
            SHADOW_LEN.store(len, Ordering::Relaxed);
            SHADOW_BASE.store(mapping, Ordering::Release);
        }
        // Pre-allocate the quarantine to avoid allocation during dealloc
        state.quarantine.reserve(64);
    });
    ASAN_ENABLED.store(true, Ordering::SeqCst);
}

/// Reserve the shadow for the widest address space the system accepts.
///
/// Returns the mapping and its length, or the length and `errno` of the
/// narrowest attempt.
fn reserve_widest_shadow() -> Result<(*mut u8, usize), (usize, i32)> {
    let mut refused = (0, 0);
    for &bits in ADDRESS_BITS {
        let len = 1 << (bits - GRANULE.trailing_zeros());
        match reserve_shadow(len) {
            Ok(mapping) => return Ok((mapping, len)),
            Err(errno) => refused = (len, errno),
        }
    }
    Err(refused)
}

/// The size of the block that holds a payload of `size` bytes: the payload
/// rounded up to whole granules, between two red zones.
fn block_size(size: usize) -> Option<usize> {
    size.checked_next_multiple_of(GRANULE)?
        .checked_add(2 * REDZONE_SIZE)
}

/// ASan-instrumented allocation.
///
/// Layout: `[LEFT_REDZONE(32B)] [payload(size)] [RIGHT_REDZONE(32B + padding)]`
///
/// Returns a pointer to the payload region. The caller sees the same pointer
/// semantics as a normal `__tribute_alloc` call.
///
/// # Safety
///
/// Same preconditions as `__tribute_alloc`.
pub unsafe fn alloc(size: usize) -> *mut u8 {
    let Some(total) = block_size(size) else {
        crate::oom_abort();
    };
    let Ok(layout) = Layout::from_size_align(total, ALLOC_ALIGN) else {
        crate::oom_abort();
    };
    let base = unsafe { alloc::alloc::alloc(layout) };
    if base.is_null() {
        crate::oom_abort();
    }
    let base_addr = base as usize;
    if shadow(base_addr, total).is_none() {
        report(format_args!(
            "==ERROR: TributeASan: allocation at {base_addr:#x} lies outside the shadow\n"
        ));
    }

    let payload = unsafe { base.add(REDZONE_SIZE) };
    let right = total - REDZONE_SIZE - size;
    unsafe { ptr::write_bytes(base, MAGIC_REDZONE, REDZONE_SIZE) };
    unsafe { ptr::write_bytes(payload.add(size), MAGIC_REDZONE, right) };

    let state = base_addr + REDZONE_SIZE - GRANULE;
    let tail = base_addr + REDZONE_SIZE + size.next_multiple_of(GRANULE);
    fill_shadow(base_addr, REDZONE_SIZE - GRANULE, SHADOW_REDZONE);
    fill_shadow(state, GRANULE, SHADOW_ALLOCATED);
    fill_shadow(state + GRANULE, size, SHADOW_VALID);
    if !size.is_multiple_of(GRANULE) {
        fill_shadow(tail - GRANULE, GRANULE, (size % GRANULE) as u8);
    }
    fill_shadow(tail, REDZONE_SIZE, SHADOW_REDZONE);

    LIVE_COUNT.fetch_add(1, Ordering::Relaxed);
    LIVE_BYTES.fetch_add(size, Ordering::Relaxed);
    payload
}

/// Whether the live block at `payload` holds exactly `size` bytes.
///
/// The granule after the padded payload must start the right red zone, and
/// the last payload granule must hold `size % 8` valid bytes, or all 8; only
/// one size satisfies both.
fn payload_size_matches(payload: usize, size: usize) -> bool {
    let Some(padded) = size.checked_next_multiple_of(GRANULE) else {
        return false;
    };
    let Some(padded_end) = payload.checked_add(padded) else {
        return false;
    };
    let shadow_at = |addr| shadow(addr, 1).map(|byte| byte[0].load(Ordering::Relaxed));
    let last = (size % GRANULE) as u8;
    padded != 0
        && shadow_at(padded_end - GRANULE) == Some(last)
        && shadow_at(padded_end) == Some(SHADOW_REDZONE)
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
    let Some(total) = block_size(size) else {
        // Corrupted size — abort rather than silently ignoring
        report(format_args!(
            "==ERROR: TributeASan: dealloc called with overflowing size\n"
        ));
    };

    // `ptr` may not point into a block of this allocator, so only its shadow
    // is read before any header is.
    if let Err(error) = mark_freed(ptr as usize) {
        report(format_args!(
            "==ERROR: TributeASan: {error} on address {:#x}\n",
            ptr as usize
        ));
    }
    if !payload_size_matches(ptr as usize, size) {
        report(format_args!(
            "==ERROR: TributeASan: dealloc size {size} does not match the block at {:#x}\n",
            ptr as usize
        ));
    }
    let base = unsafe { ptr.sub(REDZONE_SIZE) };

    // Check red zone integrity
    unsafe { check_redzone(base, REDZONE_SIZE, "left") };
    unsafe { check_redzone(ptr.add(size), total - REDZONE_SIZE - size, "right") };

    // Poison the entire region (red zones + payload) with MAGIC_FREED
    unsafe { ptr::write_bytes(base, MAGIC_FREED, total) };
    let base_addr = base as usize;
    fill_shadow(base_addr, REDZONE_SIZE - GRANULE, SHADOW_FREED);
    fill_shadow(ptr as usize, total - REDZONE_SIZE, SHADOW_FREED);

    LIVE_COUNT.fetch_sub(1, Ordering::Relaxed);
    LIVE_BYTES.fetch_sub(size, Ordering::Relaxed);

    // Move to quarantine instead of immediately freeing
    with_state(|state| state.quarantine_push(base_addr, total));
}

/// Report the allocations still live when the program returns from `main`.
///
/// Called from the entrypoint before it returns. Every allocation compiled
/// code makes is reference counted, so one that is still live was leaked.
#[unsafe(no_mangle)]
pub extern "C" fn __asan_exit() {
    let (count, bytes) = live_allocations();
    if count != 0 {
        report(format_args!(
            "==ERROR: TributeASan: detected memory leaks\n  {bytes} byte(s) in {count} allocation(s)\n"
        ));
    }
}

/// The number of live blocks and the payload bytes they hold.
fn live_allocations() -> (usize, usize) {
    (
        LIVE_COUNT.load(Ordering::Relaxed),
        LIVE_BYTES.load(Ordering::Relaxed),
    )
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
    /// Tests modify `ASAN_ENABLED`, the shadow and the quarantine, which are
    /// process-wide globals. Without serialization, one test would observe
    /// another's blocks.
    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Enable the sanitizer and empty the quarantine for test isolation.
    fn enable() -> std::sync::MutexGuard<'static, ()> {
        let lock = TEST_MUTEX
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        __asan_init();
        with_state(|state| {
            while let Some(entry) = state.quarantine.pop_front() {
                state.quarantine_total -= entry.total_size;
                release(entry);
            }
        });
        lock
    }

    fn disable() {
        ASAN_ENABLED.store(false, Ordering::SeqCst);
    }

    /// Evict every block the quarantine holds now, allocating nothing after
    /// the first eviction, so no evicted memory is handed out again.
    unsafe fn flush_quarantine() {
        let size = QUARANTINE_MAX - 2 * REDZONE_SIZE;
        unsafe { dealloc(alloc(size), size) };
    }

    #[test]
    fn test_asan_alloc_dealloc() {
        let _lock = enable();

        unsafe {
            let ptr = alloc(64);
            assert!(!ptr.is_null());

            // Write some data to the payload
            core::ptr::write_bytes(ptr, 0x42, 64);

            // Dealloc should succeed (red zones intact)
            dealloc(ptr, 64);
        }

        disable();
    }

    #[test]
    fn test_asan_counts_the_allocations_still_live() {
        let _lock = enable();
        let before = live_allocations();

        unsafe {
            let first = alloc(64);
            let second = alloc(8);
            assert_eq!(live_allocations(), (before.0 + 2, before.1 + 72));
            // A freed block waits in the quarantine but is no longer live.
            dealloc(first, 64);
            assert_eq!(live_allocations(), (before.0 + 1, before.1 + 8));
            dealloc(second, 8);
        }
        assert_eq!(live_allocations(), before);
        // With nothing live the exit check returns instead of aborting.
        if before == (0, 0) {
            __asan_exit();
        }

        disable();
    }

    #[test]
    fn test_asan_redzone_intact() {
        let _lock = enable();

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

        disable();
    }

    #[test]
    fn test_asan_freed_memory_poisoned() {
        let _lock = enable();

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

        disable();
    }

    #[test]
    fn test_asan_quarantine_eviction() {
        let _lock = enable();

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

        disable();
    }

    #[test]
    fn test_asan_shadow_describes_each_granule_of_a_block() {
        let _lock = enable();

        unsafe {
            let ptr = alloc(13);
            let base = ptr as usize - REDZONE_SIZE;
            let shadow_bytes = || -> Vec<u8> {
                shadow(base, block_size(13).unwrap())
                    .unwrap()
                    .iter()
                    .map(|byte| byte.load(Ordering::Relaxed))
                    .collect()
            };
            let rz = SHADOW_REDZONE;
            assert_eq!(
                shadow_bytes(),
                [rz, rz, rz, SHADOW_ALLOCATED, 0, 5, rz, rz, rz, rz]
            );

            dealloc(ptr, 13);
            let mut freed = [SHADOW_FREED; 10];
            freed[3] = SHADOW_FREED_STATE;
            assert_eq!(shadow_bytes(), freed);

            flush_quarantine();
            assert_eq!(shadow_bytes(), [SHADOW_VALID; 10]);
        }

        disable();
    }

    #[test]
    fn test_asan_classifies_accesses_by_region() {
        let _lock = enable();

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
            // Neither is an address the shadow does not cover.
            assert_eq!(classify(usize::MAX - 3, 4), Access::Valid);
            assert_eq!(classify(usize::MAX, 2), Access::Valid);

            let freed = alloc(24);
            dealloc(freed, 24);
            let freed = freed as usize;
            assert_eq!(classify(freed, 8), Access::HeapUseAfterFree);
            assert_eq!(classify(freed + 23, 1), Access::HeapUseAfterFree);
            assert_eq!(classify(freed - 8, 8), Access::HeapUseAfterFree);

            dealloc(live as *mut u8, 16);
            assert_eq!(classify(live, 16), Access::HeapUseAfterFree);
        }

        disable();
    }

    #[test]
    fn test_asan_classifies_a_partial_granule_by_its_valid_bytes() {
        let _lock = enable();

        unsafe {
            let ptr = alloc(13);
            let addr = ptr as usize;
            assert_eq!(classify(addr, 13), Access::Valid);
            assert_eq!(classify(addr + 8, 5), Access::Valid);
            assert_eq!(classify(addr + 12, 1), Access::Valid);
            assert_eq!(classify(addr + 8, 6), Access::HeapBufferOverflow);
            assert_eq!(classify(addr + 13, 1), Access::HeapBufferOverflow);
            assert_eq!(classify(addr + 10, 4), Access::HeapBufferOverflow);
            assert_eq!(classify(addr + 4, 8), Access::Valid);
            dealloc(ptr, 13);
        }

        disable();
    }

    #[test]
    fn test_asan_memory_evicted_from_quarantine_is_valid_for_its_next_owner() {
        let _lock = enable();

        unsafe {
            let first = alloc(1024);
            dealloc(first, 1024);
            let old = first as usize - REDZONE_SIZE;
            let old_end = old + block_size(1024).unwrap();
            assert_eq!(classify(first as usize, 8), Access::HeapUseAfterFree);

            flush_quarantine();
            // The shadow forgets the block before the memory goes back to the
            // underlying allocator.
            assert_eq!(classify(old, old_end - old), Access::Valid);

            // Whoever receives the memory next, a runtime buffer or another
            // block of a different size, accesses it validly.
            let layout = Layout::from_size_align(old_end - old, ALLOC_ALIGN).unwrap();
            let mut buffers = Vec::new();
            let mut blocks = Vec::new();
            let mut reused = false;
            for _ in 0..256 {
                let buffer = alloc::alloc::alloc(layout);
                let block = alloc(200);
                for (addr, len) in [(buffer as usize, layout.size()), (block as usize, 200)] {
                    if addr < old_end && old < addr + len {
                        reused = true;
                        assert_eq!(classify(addr, len), Access::Valid);
                    }
                }
                buffers.push(buffer);
                blocks.push(block);
                if reused {
                    break;
                }
            }
            for block in blocks {
                assert_eq!(classify(block as usize, 200), Access::Valid);
                assert_eq!(
                    classify(block as usize + 200, 1),
                    Access::HeapBufferOverflow
                );
                dealloc(block, 200);
            }
            for buffer in buffers {
                assert_eq!(classify(buffer as usize, layout.size()), Access::Valid);
                alloc::alloc::dealloc(buffer, layout);
            }
        }

        disable();
    }

    #[test]
    fn test_asan_valid_accesses_return_from_the_checks() {
        let _lock = enable();

        unsafe {
            let live = alloc(16);
            __tribute_asan_store(live, 16);
            __tribute_asan_load(live.add(8), 8);
            dealloc(live, 16);
        }

        disable();
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
    fn test_asan_a_refused_reservation_yields_its_errno() {
        assert!(reserve_shadow(usize::MAX & !0xFFFF).is_err_and(|errno| errno != 0));
    }

    #[test]
    fn test_asan_matches_the_dealloc_size_to_the_block() {
        let _lock = enable();

        unsafe {
            for size in [8, 13, 16] {
                let ptr = alloc(size) as usize;
                assert!(payload_size_matches(ptr, size));
                for wrong in [size - 1, size + 1, size + 8, size - 8, 0, usize::MAX] {
                    assert!(!payload_size_matches(ptr, wrong), "{size} as {wrong}");
                }
                dealloc(ptr as *mut u8, size);
            }
        }

        disable();
    }

    #[test]
    fn test_asan_rejects_a_second_or_unknown_free() {
        let _lock = enable();

        unsafe {
            let ptr = alloc(16);
            dealloc(ptr, 16);
            assert_eq!(mark_freed(ptr as usize), Err("attempting double-free"));

            // Once the block leaves the quarantine its shadow is reset.
            flush_quarantine();
            assert_eq!(
                mark_freed(ptr as usize),
                Err("attempting free of memory that is not allocated")
            );
            let local = 0u64;
            assert_eq!(
                mark_freed(&raw const local as usize),
                Err("attempting free of memory that is not allocated")
            );

            // Only the start of a live payload is accepted.
            let live = alloc(32) as usize;
            for addr in [live + 8, live + 1, live - 8, live + 32, 0, 4] {
                assert_eq!(
                    mark_freed(addr),
                    Err("attempting free of memory that is not allocated"),
                    "{addr:#x}"
                );
            }
            dealloc(live as *mut u8, 32);
        }

        disable();
    }

    #[test]
    fn test_asan_state_is_consistent_under_concurrent_use() {
        let _lock = enable();
        let before = live_allocations();

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

        // Every block was freed, and the quarantine holds only freed blocks.
        assert_eq!(live_allocations(), before);
        with_state(|state| {
            for entry in &state.quarantine {
                assert!(
                    shadow(entry.base, entry.total_size)
                        .unwrap()
                        .iter()
                        .all(|byte| matches!(
                            byte.load(Ordering::Relaxed),
                            SHADOW_FREED | SHADOW_FREED_STATE
                        ))
                );
            }
            let held: usize = state.quarantine.iter().map(|entry| entry.total_size).sum();
            assert_eq!(held, state.quarantine_total);
        });

        disable();
    }
}

//! Platform-specific thread-local storage backend.
//!
//! Uses OS-level TLS APIs (`pthread_key_*` on POSIX, `FlsAlloc` on Windows)
//! to provide true thread-local storage without depending on `std::thread_local!`.
//!
//! All per-thread state is consolidated into a single [`ThreadState`] struct,
//! lazily allocated on first access via [`thread_state()`].

use alloc::boxed::Box;
use core::cell::{Cell, RefCell};
use core::ffi::c_void;
use core::ptr::NonNull;

use smallvec::SmallVec;

use crate::MpPrompt;

// =============================================================================
// Prompt tag registry
//
// Maps integer tags to their active prompt pointers. Uses SmallVec-based
// linear search instead of HashMap, since the number of simultaneously
// active prompt tags is typically 1–4.
// =============================================================================

type PromptStack = SmallVec<[NonNull<MpPrompt>; 2]>;

pub(crate) struct PromptRegistry {
    entries: SmallVec<[(i32, PromptStack); 4]>,
}

impl PromptRegistry {
    fn new() -> Self {
        Self {
            entries: SmallVec::new(),
        }
    }

    pub(crate) fn push(&mut self, tag: i32, prompt: NonNull<MpPrompt>) {
        if let Some((_, stack)) = self.entries.iter_mut().find(|(t, _)| *t == tag) {
            stack.push(prompt);
        } else {
            let mut stack = SmallVec::new();
            stack.push(prompt);
            self.entries.push((tag, stack));
        }
    }

    pub(crate) fn pop(&mut self, tag: i32, prompt: NonNull<MpPrompt>) {
        if let Some(pos) = self.entries.iter().position(|(t, _)| *t == tag) {
            let stack = &mut self.entries[pos].1;
            if let Some(idx) = stack.iter().rposition(|p| *p == prompt) {
                stack.remove(idx);
            }
            if stack.is_empty() {
                self.entries.remove(pos);
            }
        }
    }

    pub(crate) fn lookup(&self, tag: i32) -> Option<NonNull<MpPrompt>> {
        self.entries
            .iter()
            .find(|(t, _)| *t == tag)
            .and_then(|(_, stack)| stack.last().copied())
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// =============================================================================
// Thread-local state
//
// All per-thread state is consolidated into a single struct, allocated on
// first access via platform-specific TLS APIs.
// =============================================================================

pub(crate) struct ThreadState {
    pub(crate) yield_active: Cell<bool>,
    pub(crate) yield_resume: Cell<*mut u8>,
    pub(crate) yield_op_idx: Cell<i32>,
    pub(crate) yield_shift_value: Cell<*mut u8>,
    pub(crate) yield_rc_roots: Cell<(*mut u8, usize)>,
    pub(crate) prompt_registry: RefCell<PromptRegistry>,
}

impl ThreadState {
    fn new() -> Self {
        Self {
            yield_active: Cell::new(false),
            yield_resume: Cell::new(core::ptr::null_mut()),
            yield_op_idx: Cell::new(0),
            yield_shift_value: Cell::new(core::ptr::null_mut()),
            yield_rc_roots: Cell::new((core::ptr::null_mut(), 0)),
            prompt_registry: RefCell::new(PromptRegistry::new()),
        }
    }
}

// =============================================================================
// POSIX TLS backend (pthread_key_*)
// =============================================================================

#[cfg(unix)]
mod posix {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    // pthread_key_t type varies by platform:
    //   macOS (Darwin): unsigned long (c_ulong)
    //   Linux (glibc/musl): unsigned int (c_uint)
    #[cfg(target_vendor = "apple")]
    type PthreadKey = core::ffi::c_ulong;
    #[cfg(not(target_vendor = "apple"))]
    type PthreadKey = core::ffi::c_uint;

    unsafe extern "C" {
        fn pthread_key_create(
            key: *mut PthreadKey,
            dtor: Option<unsafe extern "C" fn(*mut c_void)>,
        ) -> core::ffi::c_int;
        fn pthread_getspecific(key: PthreadKey) -> *mut c_void;
        fn pthread_setspecific(key: PthreadKey, value: *const c_void) -> core::ffi::c_int;
    }

    /// Sentinel: no initialization attempted yet.
    const KEY_UNSET: usize = usize::MAX;
    /// Sentinel: initialization in progress by another thread.
    const KEY_INITIALIZING: usize = usize::MAX - 1;

    static TLS_KEY: AtomicUsize = AtomicUsize::new(KEY_UNSET);

    /// Destructor callback invoked by pthread when a thread exits.
    unsafe extern "C" fn destroy_thread_state(ptr: *mut c_void) {
        if !ptr.is_null() {
            let _ = unsafe { Box::from_raw(ptr as *mut ThreadState) };
        }
    }

    /// Create the TLS key. Idempotent: only the first call performs initialization;
    /// concurrent callers spin-wait until the key is ready.
    ///
    /// # Safety
    ///
    /// Calling `thread_state()` before `tls_init()` is undefined behavior.
    pub(crate) unsafe fn tls_init() {
        match TLS_KEY.compare_exchange(
            KEY_UNSET,
            KEY_INITIALIZING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // Winner: perform the actual initialization.
                let mut key: PthreadKey = 0;
                let ret = unsafe { pthread_key_create(&mut key, Some(destroy_thread_state)) };
                assert!(ret == 0, "pthread_key_create failed");
                TLS_KEY.store(key as usize, Ordering::Release);
            }
            Err(KEY_INITIALIZING) => {
                // Another thread is initializing; spin until the real key is published.
                while TLS_KEY.load(Ordering::Acquire) == KEY_INITIALIZING {
                    core::hint::spin_loop();
                }
            }
            Err(_) => {
                // Already initialized — nothing to do.
            }
        }
    }

    /// Get the current thread's `ThreadState`, lazily allocating on first access.
    ///
    /// # Safety
    ///
    /// `tls_init()` must have been called before this function.
    pub(crate) unsafe fn thread_state() -> &'static ThreadState {
        let key = TLS_KEY.load(Ordering::Acquire) as PthreadKey;
        let ptr = unsafe { pthread_getspecific(key) };
        if ptr.is_null() {
            let state = Box::new(ThreadState::new());
            let raw = Box::into_raw(state) as *mut c_void;
            let ret = unsafe { pthread_setspecific(key, raw) };
            assert!(ret == 0, "pthread_setspecific failed");
            unsafe { &*(raw as *const ThreadState) }
        } else {
            unsafe { &*(ptr as *const ThreadState) }
        }
    }
}

// =============================================================================
// Windows TLS backend (Fiber-Local Storage)
// =============================================================================

#[cfg(windows)]
mod windows {
    use super::*;
    use core::sync::atomic::{AtomicU32, Ordering};

    /// Sentinel returned by `FlsAlloc` on failure; also used as "unset".
    const FLS_OUT_OF_INDEXES: u32 = 0xFFFFFFFF;
    /// Sentinel: initialization in progress by another thread.
    const FLS_INITIALIZING: u32 = 0xFFFFFFFE;

    unsafe extern "system" {
        fn FlsAlloc(callback: Option<unsafe extern "system" fn(*mut c_void)>) -> u32;
        fn FlsGetValue(index: u32) -> *mut c_void;
        fn FlsSetValue(index: u32, value: *mut c_void) -> i32;
    }

    static FLS_INDEX: AtomicU32 = AtomicU32::new(FLS_OUT_OF_INDEXES);

    /// Destructor callback invoked by Windows FLS when a fiber/thread exits.
    unsafe extern "system" fn destroy_thread_state(ptr: *mut c_void) {
        if !ptr.is_null() {
            let _ = unsafe { Box::from_raw(ptr as *mut ThreadState) };
        }
    }

    /// Create the FLS index. Idempotent: only the first call performs initialization;
    /// concurrent callers spin-wait until the index is ready.
    ///
    /// # Safety
    ///
    /// Calling `thread_state()` before `tls_init()` is undefined behavior.
    pub(crate) unsafe fn tls_init() {
        match FLS_INDEX.compare_exchange(
            FLS_OUT_OF_INDEXES,
            FLS_INITIALIZING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // Winner: perform the actual initialization.
                let index = unsafe { FlsAlloc(Some(destroy_thread_state)) };
                assert!(index != FLS_OUT_OF_INDEXES, "FlsAlloc failed");
                FLS_INDEX.store(index, Ordering::Release);
            }
            Err(FLS_INITIALIZING) => {
                // Another thread is initializing; spin until the real index is published.
                while FLS_INDEX.load(Ordering::Acquire) == FLS_INITIALIZING {
                    core::hint::spin_loop();
                }
            }
            Err(_) => {
                // Already initialized — nothing to do.
            }
        }
    }

    /// Get the current thread's `ThreadState`, lazily allocating on first access.
    ///
    /// # Safety
    ///
    /// `tls_init()` must have been called before this function.
    pub(crate) unsafe fn thread_state() -> &'static ThreadState {
        let index = FLS_INDEX.load(Ordering::Acquire);
        let ptr = unsafe { FlsGetValue(index) };
        if ptr.is_null() {
            let state = Box::new(ThreadState::new());
            let raw = Box::into_raw(state) as *mut c_void;
            let ret = unsafe { FlsSetValue(index, raw) };
            assert!(ret != 0, "FlsSetValue failed");
            unsafe { &*(raw as *const ThreadState) }
        } else {
            unsafe { &*(ptr as *const ThreadState) }
        }
    }
}

// =============================================================================
// Public re-exports
// =============================================================================

#[cfg(unix)]
pub(crate) use posix::{thread_state, tls_init};

#[cfg(windows)]
pub(crate) use windows::{thread_state, tls_init};

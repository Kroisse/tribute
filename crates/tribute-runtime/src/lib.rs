//! Tribute runtime library.
//!
//! Provides the native runtime functions required by Tribute's compiled output:
//! - Heap allocation (`__tribute_alloc`, `__tribute_dealloc`)
//! - Delimited continuations via libmprompt (`__tribute_prompt`, `__tribute_yield`, etc.)
//! - TLS-based yield state for handler dispatch
//! - Evidence-based ability dispatch (`__tribute_evidence_*`)
//! - RC-safe continuation wrapping (`__tribute_cont_wrap_*`, `__tribute_resume`, etc.)

#![cfg_attr(panic = "abort", no_std)]
#![allow(private_interfaces)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr::NonNull;

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

// =============================================================================
// libmprompt FFI bindings
// =============================================================================

/// Opaque prompt type from libmprompt.
#[repr(C)]
pub struct MpPrompt {
    _opaque: [u8; 0],
}

/// Opaque resume type from libmprompt.
#[repr(C)]
pub struct MpResume {
    _opaque: [u8; 0],
}

/// Callback type for `mp_prompt`: `fn(prompt, arg) -> result`
type MpStartFun = unsafe extern "C" fn(*mut MpPrompt, *mut u8) -> *mut u8;

/// Callback type for `mp_yield`: `fn(resume, arg) -> result`
type MpYieldFun = unsafe extern "C" fn(*mut MpResume, *mut u8) -> *mut u8;

unsafe extern "C" {
    fn mp_init(config: *const c_void);
    fn mp_prompt(fun: MpStartFun, arg: *mut u8) -> *mut u8;
    fn mp_yield(p: *mut MpPrompt, fun: MpYieldFun, arg: *mut u8) -> *mut u8;
    fn mp_resume(r: *mut MpResume, result: *mut u8) -> *mut u8;
    fn mp_resume_drop(r: *mut MpResume);
}

mod tls;

use tls::{thread_state, tls_init};

// =============================================================================
// Initialization
// =============================================================================

/// Initialize the Tribute runtime (must be called once before any ability use).
///
/// Calls `mp_init(NULL)` to set up libmprompt's internal state (signal handlers,
/// thread-local storage, gstack pools, etc.).
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_init() {
    tls_init();
    unsafe { mp_init(core::ptr::null()) };
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
    let Ok(layout) = core::alloc::Layout::from_size_align(size, 8) else {
        return;
    };
    unsafe { alloc::alloc::dealloc(ptr, layout) };
}

// =============================================================================
// Prompt / Yield / Resume
// =============================================================================

/// Context struct passed through `mp_prompt` as the opaque `arg` pointer.
///
/// Contains the tag (for registry lookup) and the user's body function + env.
#[repr(C)]
struct PromptContext {
    tag: i32,
    body_fn: unsafe extern "C" fn(*mut u8) -> *mut u8,
    env: *mut u8,
}

/// The `mp_prompt` callback: registers the prompt pointer, calls the user's
/// body function, then unregisters.
///
/// `arg` is a `Box<PromptContext>` passed as a raw pointer. This function
/// takes ownership and drops it when done, ensuring the context remains
/// valid across yield/resume boundaries.
unsafe extern "C" fn prompt_start(prompt: *mut MpPrompt, arg: *mut u8) -> *mut u8 {
    let ctx = unsafe { Box::from_raw(arg as *mut PromptContext) };

    // Convert to NonNull (libmprompt guarantees non-null prompt pointers)
    let prompt_nn = NonNull::new(prompt).expect("ICE: libmprompt returned null prompt");

    // Register the prompt pointer for this tag (stack for nested prompts)
    thread_state()
        .prompt_registry
        .borrow_mut()
        .push(ctx.tag, prompt_nn);

    // Call the user's body function
    let result = unsafe { (ctx.body_fn)(ctx.env) };

    // Unregister (pop from stack; remove key when empty)
    thread_state().prompt_registry.borrow_mut().pop(ctx.tag);

    // ctx is dropped here (Box ownership)
    result
}

/// Establish a prompt with the given tag and run `body_fn(env)` under it.
///
/// Signature: `(tag: i32, body_fn: ptr, env: ptr) -> ptr`
///
/// # Safety
///
/// `body_fn` must be a valid function pointer. `env` must be valid for the
/// duration of `body_fn` execution (or null if unused).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_prompt(
    tag: i32,
    body_fn: unsafe extern "C" fn(*mut u8) -> *mut u8,
    env: *mut u8,
) -> *mut u8 {
    let ctx = Box::new(PromptContext { tag, body_fn, env });
    unsafe { mp_prompt(prompt_start, Box::into_raw(ctx) as *mut u8) }
}

/// The `mp_yield` callback: captures the resume pointer into TLS
/// and signals that a yield is active.
unsafe extern "C" fn yield_handler(resume: *mut MpResume, arg: *mut u8) -> *mut u8 {
    let ts = thread_state();
    ts.yield_resume.set(resume as *mut u8);
    ts.yield_active.set(true);
    // Return `arg` as the result of `mp_prompt` (which returns to the handler dispatch loop)
    arg
}

/// Yield to the prompt with the given tag.
///
/// Stores `op_idx` and `shift_val` in TLS so the handler dispatch loop
/// can read them. The yield handler callback captures the resume object.
///
/// Signature: `(tag: i32, op_idx: i32, shift_value: ptr) -> ptr`
///
/// # Safety
///
/// `tag` must refer to a currently registered prompt. `shift_value` is
/// passed through opaquely and must remain valid until the handler processes it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_yield(tag: i32, op_idx: i32, shift_value: *mut u8) -> *mut u8 {
    let ts = thread_state();

    // Store yield metadata in TLS before yielding
    ts.yield_op_idx.set(op_idx);
    ts.yield_shift_value.set(shift_value);

    // Look up the innermost prompt pointer for this tag
    let prompt = ts
        .prompt_registry
        .borrow()
        .lookup(tag)
        .expect("ICE: __tribute_yield called with unregistered tag");

    // Yield to the prompt; the yield_handler callback captures the resume
    // and returns shift_value as the result of mp_prompt
    unsafe { mp_yield(prompt.as_ptr(), yield_handler, shift_value) }
}

/// Resume a wrapped continuation with a value.
///
/// Extracts the inner `MpResume`, calls `mp_resume`, then frees the wrapper.
/// RC roots are NOT released here — the compiler inserts `tribute_rt.release`
/// after the yield returns (resume path).
///
/// Signature: `(wrapped: ptr, val: ptr) -> ptr`
///
/// # Safety
///
/// `wrapped` must be a valid `TributeContinuation` pointer from
/// `__tribute_cont_wrap_from_tls`. Each wrapped continuation can only
/// be resumed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_resume(wrapped: *mut u8, val: *mut u8) -> *mut u8 {
    let cont = unsafe { Box::from_raw(wrapped as *mut TributeContinuation) };
    // cont is dropped after this call — wrapper freed, rc_roots Vec freed
    // RC roots are NOT released; the compiler handles that on the resume path
    unsafe { mp_resume(cont.resume, val) }
}

/// Drop a wrapped continuation without resuming it.
///
/// For each RC root, performs **two** deep releases:
/// 1. Cancel the extra retain inserted by the compiler before yield
/// 2. Replace the body's normal release that will never execute
///    (since `mp_resume_drop` discards the captured stack)
///
/// Then calls `mp_resume_drop` to discard the captured stack, and frees
/// the wrapper.
///
/// Signature: `(wrapped: ptr) -> ()`
///
/// # Safety
///
/// `wrapped` must be a valid `TributeContinuation` pointer that has not
/// been resumed or dropped yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_resume_drop(wrapped: *mut u8) {
    let cont = unsafe { Box::from_raw(wrapped as *mut TributeContinuation) };

    // Two releases per root:
    // 1st: cancel the extra retain inserted before yield (refcount N+1 → N)
    // 2nd: replace the body's normal release that won't run (refcount N → N-1)
    // The 2nd release may reach refcount 0 → deep release via RTTI dispatch
    for root in cont.rc_roots.iter() {
        unsafe { root.release_deep() }; // cancel extra retain
        unsafe { root.release_deep() }; // replace body's missing release
    }

    // Drop the captured continuation (without executing its body)
    unsafe { mp_resume_drop(cont.resume) };
    // cont is dropped here — wrapper freed
}

// =============================================================================
// TLS yield state accessors
// =============================================================================

/// Check whether a yield is currently active.
///
/// Returns 1 (true) if `__tribute_yield` was called and the handler
/// dispatch loop should process the operation, 0 otherwise.
///
/// Signature: `() -> i1`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_yield_active() -> i8 {
    thread_state().yield_active.get() as i8
}

/// Get the operation index of the current yield.
///
/// Signature: `() -> i32`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_get_yield_op_idx() -> i32 {
    thread_state().yield_op_idx.get()
}

/// Get the captured continuation (resume object) from the current yield.
///
/// Signature: `() -> ptr`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_get_yield_continuation() -> *mut u8 {
    thread_state().yield_resume.get()
}

/// Get the shift value (argument) from the current yield.
///
/// Signature: `() -> ptr`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_get_yield_shift_value() -> *mut u8 {
    thread_state().yield_shift_value.get()
}

/// Reset all TLS yield state to default values.
///
/// Called after the handler dispatch loop processes a yield operation.
///
/// Signature: `() -> ()`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_reset_yield_state() {
    let ts = thread_state();
    ts.yield_active.set(false);
    ts.yield_resume.set(core::ptr::null_mut());
    ts.yield_op_idx.set(0);
    ts.yield_shift_value.set(core::ptr::null_mut());
}

// =============================================================================
// RC-safe continuation wrapping
// =============================================================================

// In production builds, __tribute_deep_release is provided by the compiler
// (generated in trunk-ir-cranelift-backend with RTTI dispatch). In tests,
// we use a stub that performs shallow dealloc.
#[cfg(not(test))]
unsafe extern "C" {
    fn __tribute_deep_release(ptr: *mut u8, alloc_size: u64);
}

/// Test stub for `__tribute_deep_release`: performs shallow dealloc only.
#[cfg(test)]
unsafe extern "C" fn __tribute_deep_release(ptr: *mut u8, alloc_size: u64) {
    if !ptr.is_null() {
        let raw = unsafe { ptr.sub(RC_HEADER_SIZE) };
        unsafe { __tribute_dealloc(raw, alloc_size) };
    }
}

/// Size of the RC header prepended to every heap-allocated object.
/// Layout: `[refcount: u32][-4: rtti_idx: u32][0: payload...]`
const RC_HEADER_SIZE: usize = 8;

/// An RC root captured at yield time: payload pointer + allocation size.
///
/// `ptr` points at the payload (offset 0); the refcount lives at `ptr - 8`.
/// `alloc_size` is the total allocation size (header + payload), used for
/// deep release when refcount reaches 0.
#[derive(Clone, Copy)]
struct RcRoot {
    ptr: NonNull<u8>,
    alloc_size: u64,
}

impl RcRoot {
    /// Decrement refcount; if it reaches 0, call `__tribute_deep_release`
    /// which handles RTTI-dispatched recursive field release and deallocation.
    unsafe fn release_deep(&self) {
        let rc_addr = unsafe { self.ptr.as_ptr().sub(RC_HEADER_SIZE) } as *mut u32;
        let rc = unsafe { rc_addr.read() };
        let new_rc = rc - 1;
        unsafe { rc_addr.write(new_rc) };
        if new_rc == 0 {
            unsafe { __tribute_deep_release(self.ptr.as_ptr(), self.alloc_size) };
        }
    }
}

/// Wrapper around a libmprompt resume object with RC root metadata.
///
/// Opaque to compiled code — only accessed through `__tribute_resume` / `__tribute_resume_drop` FFI.
struct TributeContinuation {
    resume: *mut MpResume,
    rc_roots: Box<[RcRoot]>,
}

/// Store RC root entries in TLS before yielding.
///
/// The compiler calls this just before `__tribute_yield` to pass
/// the set of live RC pointers that need protection across the yield.
///
/// Signature: `(roots: ptr, count: i32) -> ()`
///
/// # Safety
///
/// `roots` must point to a contiguous array of `count` entries, each 16
/// bytes laid out as `(ptr: *mut u8, alloc_size: u64)` in native byte
/// order. The buffer must have been allocated via `__tribute_alloc` with
/// size `count * 16`. Ownership of the buffer is transferred to the
/// runtime: `__tribute_cont_wrap_from_tls` will free it after copying.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_yield_set_rc_roots(roots: *mut u8, count: i32) {
    thread_state().yield_rc_roots.set((roots, count as usize));
}

/// Wrap a raw resume pointer into a `TributeContinuation`.
///
/// Reads RC roots from TLS (set by `__tribute_yield_set_rc_roots`) and
/// packages them together with the resume pointer. The TLS slot is cleared
/// after consumption.
///
/// The roots buffer uses a `(ptr, alloc_size)` pair layout (16 bytes per entry).
/// After copying, the original buffer is freed via `__tribute_dealloc` to
/// prevent leaking (the post-yield dealloc in the body never runs on the
/// drop path).
///
/// **Note:** Retains are NOT done here — the compiler inserts `tribute_rt.retain`
/// before the yield.
///
/// Signature: `(resume: ptr) -> ptr`
///
/// # Safety
///
/// `resume` must be a valid `MpResume` pointer from `__tribute_get_yield_continuation`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_cont_wrap_from_tls(resume: *mut u8) -> *mut u8 {
    let ts = thread_state();
    let (roots_ptr, count) = ts.yield_rc_roots.get();
    ts.yield_rc_roots.set((core::ptr::null_mut(), 0));

    let rc_roots = if roots_ptr.is_null() || count == 0 {
        Box::new([]) as Box<[RcRoot]>
    } else {
        // Parse (ptr, alloc_size) pairs: each entry is 16 bytes
        let mut roots = Vec::with_capacity(count);
        for i in 0..count {
            let entry_base = unsafe { roots_ptr.add(i * 16) };
            let ptr_val = unsafe { (entry_base as *const *mut u8).read() };
            let alloc_size = unsafe { (entry_base.add(8) as *const u64).read() };
            if let Some(nn) = NonNull::new(ptr_val) {
                roots.push(RcRoot {
                    ptr: nn,
                    alloc_size,
                });
            }
        }

        // Free the original roots buffer (ownership transferred here)
        let buf_size = (count * 16) as u64;
        unsafe { __tribute_dealloc(roots_ptr, buf_size) };

        roots.into_boxed_slice()
    };

    let cont = Box::new(TributeContinuation {
        resume: resume as *mut MpResume,
        rc_roots,
    });
    Box::into_raw(cont) as *mut u8
}

// =============================================================================
// Evidence-based ability dispatch
// =============================================================================

/// Marker for a single ability handler in the evidence.
///
/// `#[repr(C)]` so Cranelift can access individual fields by offset.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Marker {
    pub ability_id: i32,
    pub prompt_tag: i32,
    pub op_table_index: i32,
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
        let pos = new
            .markers
            .binary_search_by_key(&marker.ability_id, |m| m.ability_id)
            .unwrap_or_else(|pos| pos);
        new.markers.insert(pos, marker);
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
/// Signature: `(ev: ptr, ability_id: i32, prompt_tag: i32, op_table_index: i32) -> ptr`
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
    op_table_index: i32,
) -> *mut Evidence {
    let ev = unsafe { &*ev };
    let marker = Marker {
        ability_id,
        prompt_tag,
        op_table_index,
    };
    Box::into_raw(Box::new(ev.extend(marker)))
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
    fn test_yield_state_defaults() {
        assert_eq!(__tribute_yield_active(), 0);
        assert_eq!(__tribute_get_yield_op_idx(), 0);
        assert!(__tribute_get_yield_continuation().is_null());
        assert!(__tribute_get_yield_shift_value().is_null());
    }

    #[test]
    fn test_yield_state_reset() {
        thread_state().yield_active.set(true);
        thread_state().yield_op_idx.set(42);
        __tribute_reset_yield_state();
        assert_eq!(__tribute_yield_active(), 0);
        assert_eq!(__tribute_get_yield_op_idx(), 0);
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

    #[test]
    fn test_prompt_simple() {
        // Test a simple prompt that doesn't yield: body returns immediately.
        unsafe extern "C" fn body(_env: *mut u8) -> *mut u8 {
            42usize as *mut u8
        }

        unsafe {
            let result = __tribute_prompt(1, body, core::ptr::null_mut());
            assert_eq!(result as usize, 42);
        }
    }

    #[test]
    fn test_prompt_with_yield() {
        // Test that PromptContext survives across yield/resume (heap-allocated).
        // Body yields a value, handler resumes with a different value.
        unsafe extern "C" fn body(_env: *mut u8) -> *mut u8 {
            // Yield to the prompt with tag=10, op_idx=0, shift_value=100

            // Return the value we were resumed with
            unsafe { __tribute_yield(10, 0, 100usize as *mut u8) }
        }

        unsafe {
            let result = __tribute_prompt(10, body, core::ptr::null_mut());
            // After yield, mp_prompt returns the yield handler's return value (shift_value=100)
            assert_eq!(result as usize, 100);

            // Verify yield state was set
            assert_eq!(__tribute_yield_active(), 1);
            assert_eq!(__tribute_get_yield_op_idx(), 0);

            // Wrap and resume the continuation with value 999
            let k = __tribute_get_yield_continuation();
            assert!(!k.is_null());
            let wrapped_k = __tribute_cont_wrap_from_tls(k);
            let final_result = __tribute_resume(wrapped_k, 999usize as *mut u8);
            // body returns the resumed value (999)
            assert_eq!(final_result as usize, 999);

            __tribute_reset_yield_state();
        }
    }

    #[test]
    fn test_prompt_nested_same_tag() {
        // Test that nested prompts with the same tag work correctly (stack-based registry).
        unsafe extern "C" fn inner_body(_env: *mut u8) -> *mut u8 {
            77usize as *mut u8
        }

        unsafe extern "C" fn outer_body(_env: *mut u8) -> *mut u8 {
            // Nest another prompt with the same tag
            let inner_result = unsafe { __tribute_prompt(5, inner_body, core::ptr::null_mut()) };
            // Return inner result + 10
            (inner_result as usize + 10) as *mut u8
        }

        unsafe {
            let result = __tribute_prompt(5, outer_body, core::ptr::null_mut());
            assert_eq!(result as usize, 87); // 77 + 10
        }

        // Verify registry is clean after both prompts complete
        assert!(thread_state().prompt_registry.borrow().is_empty());
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
            let ev2 = __tribute_evidence_extend(ev, 10, 1, 0);

            let ev2_ref = &*ev2;
            assert_eq!(ev2_ref.markers.len(), 1);
            assert_eq!(ev2_ref.markers[0].ability_id, 10);
            assert_eq!(ev2_ref.markers[0].prompt_tag, 1);
            assert_eq!(ev2_ref.markers[0].op_table_index, 0);

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
            let ev = __tribute_evidence_extend(ev, 30, 3, 0);
            let ev = __tribute_evidence_extend(ev, 10, 1, 0);
            let ev = __tribute_evidence_extend(ev, 20, 2, 0);

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
            let ev = __tribute_evidence_extend(ev, 10, 1, 0);
            let ev = __tribute_evidence_extend(ev, 20, 2, 5);

            let prompt_tag = __tribute_evidence_lookup(ev, 20);
            assert_eq!(prompt_tag, 2);

            let _ = Box::from_raw(ev);
        }
    }

    // =========================================================================
    // TributeContinuation tests
    // =========================================================================

    #[test]
    fn test_yield_rc_roots_tls_defaults() {
        let (ptr, count) = thread_state().yield_rc_roots.get();
        assert!(ptr.is_null());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_yield_set_rc_roots_stores_and_clears() {
        // Set roots
        let mut roots: [*mut u8; 2] = [
            core::ptr::dangling_mut::<u8>(),
            core::ptr::without_provenance_mut(2),
        ];
        unsafe {
            __tribute_yield_set_rc_roots(roots.as_mut_ptr() as *mut u8, 2);
        }
        let (ptr, count) = thread_state().yield_rc_roots.get();
        assert!(!ptr.is_null());
        assert_eq!(count, 2);

        // Clear by setting to null
        unsafe {
            __tribute_yield_set_rc_roots(core::ptr::null_mut(), 0);
        }
        let (ptr, count) = thread_state().yield_rc_roots.get();
        assert!(ptr.is_null());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_cont_wrap_from_tls_null_roots() {
        // With no roots set, wrapping should succeed with empty rc_roots
        unsafe {
            __tribute_yield_set_rc_roots(core::ptr::null_mut(), 0);
        }

        // We can't create a real MpResume, so just test the wrapping logic
        // by using a sentinel value and checking the wrapper structure
        let fake_resume = 0xDEAD_BEEF_usize as *mut u8;
        let wrapped = unsafe { __tribute_cont_wrap_from_tls(fake_resume) };
        assert!(!wrapped.is_null());

        // Verify wrapper contents
        let cont = unsafe { Box::from_raw(wrapped as *mut TributeContinuation) };
        assert_eq!(cont.resume as usize, 0xDEAD_BEEF);
        assert!(cont.rc_roots.is_empty());
        // cont is dropped here

        // TLS should have been cleared
        let (ptr, count) = thread_state().yield_rc_roots.get();
        assert!(ptr.is_null());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_cont_wrap_from_tls_with_roots() {
        // Allocate a fake RC object (header + payload)
        let alloc_size = (RC_HEADER_SIZE + 8) as u64;
        let raw = unsafe { __tribute_alloc(alloc_size) };
        assert!(!raw.is_null());

        // Initialize refcount to 2 (one for "normal" use + one extra retain)
        unsafe {
            (raw as *mut u32).write(2); // refcount = 2
            (raw.add(4) as *mut u32).write(0); // rtti_idx = 0
        }
        let payload = unsafe { raw.add(RC_HEADER_SIZE) };

        // Set roots in TLS using (ptr, alloc_size) pair layout (16 bytes/entry)
        // Allocate a roots buffer via __tribute_alloc (will be freed by cont_wrap)
        let roots_buf = unsafe { __tribute_alloc(16) }; // 1 entry × 16 bytes
        assert!(!roots_buf.is_null());
        unsafe {
            (roots_buf as *mut *mut u8).write(payload); // ptr
            (roots_buf.add(8) as *mut u64).write(alloc_size); // alloc_size
        }
        unsafe {
            __tribute_yield_set_rc_roots(roots_buf, 1);
        }

        // Wrap — this also frees the roots_buf
        let fake_resume = 0xCAFE_usize as *mut u8;
        let wrapped = unsafe { __tribute_cont_wrap_from_tls(fake_resume) };
        assert!(!wrapped.is_null());

        // Verify wrapper
        let cont = unsafe { Box::from_raw(wrapped as *mut TributeContinuation) };
        assert_eq!(cont.rc_roots.len(), 1);
        assert_eq!(cont.rc_roots[0].ptr.as_ptr(), payload);
        assert_eq!(cont.rc_roots[0].alloc_size, alloc_size);

        // Clean up — dealloc the fake object
        unsafe { __tribute_dealloc(raw, alloc_size) };
    }

    #[test]
    fn test_rc_root_release_deep_decrements() {
        // Allocate a fake RC object
        let alloc_size = (RC_HEADER_SIZE + 8) as u64;
        let raw = unsafe { __tribute_alloc(alloc_size) };
        assert!(!raw.is_null());

        // Initialize refcount to 3
        unsafe {
            (raw as *mut u32).write(3); // refcount = 3
            (raw.add(4) as *mut u32).write(0); // rtti_idx = 0
        }
        let payload = unsafe { raw.add(RC_HEADER_SIZE) };
        let root = RcRoot {
            ptr: NonNull::new(payload).unwrap(),
            alloc_size,
        };

        // release_deep → refcount should be 2
        unsafe { root.release_deep() };
        let rc = unsafe { (raw as *mut u32).read() };
        assert_eq!(rc, 2);

        // release_deep → refcount should be 1
        unsafe { root.release_deep() };
        let rc = unsafe { (raw as *mut u32).read() };
        assert_eq!(rc, 1);

        // Clean up (don't release to 0 here since __tribute_deep_release
        // is a compiler-generated symbol not available in unit tests)
        unsafe { __tribute_dealloc(raw, alloc_size) };
    }
}

//! Tribute runtime library.
//!
//! Provides the native runtime functions required by Tribute's compiled output:
//! - Heap allocation (`__tribute_alloc`, `__tribute_dealloc`)
//! - Delimited continuations via libmprompt (`__tribute_prompt`, `__tribute_yield`, etc.)
//! - TLS-based yield state for handler dispatch

#![allow(clippy::missing_safety_doc)]

use std::cell::Cell;
use std::collections::HashMap;

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
    fn mp_prompt(fun: MpStartFun, arg: *mut u8) -> *mut u8;
    fn mp_yield(p: *mut MpPrompt, fun: MpYieldFun, arg: *mut u8) -> *mut u8;
    fn mp_resume(r: *mut MpResume, result: *mut u8) -> *mut u8;
    fn mp_resume_drop(r: *mut MpResume);
}

// =============================================================================
// TLS-based yield state
// =============================================================================

thread_local! {
    /// Whether a yield is active (handler dispatch should enter the loop).
    static YIELD_ACTIVE: Cell<bool> = const { Cell::new(false) };
    /// The resume object captured by the yield handler callback.
    static YIELD_RESUME: Cell<*mut u8> = const { Cell::new(std::ptr::null_mut()) };
    /// The operation index passed by the yielding code.
    static YIELD_OP_IDX: Cell<i32> = const { Cell::new(0) };
    /// The shift value (argument) passed by the yielding code.
    static YIELD_SHIFT_VALUE: Cell<*mut u8> = const { Cell::new(std::ptr::null_mut()) };
}

// =============================================================================
// Prompt tag registry
//
// Maps integer tags to their active prompt pointers. When `__tribute_prompt`
// establishes a prompt, the callback receives the prompt pointer from
// libmprompt, and we register it here so `__tribute_yield` can look it up.
// =============================================================================

thread_local! {
    static PROMPT_REGISTRY: std::cell::RefCell<HashMap<i32, Vec<*mut MpPrompt>>> =
        std::cell::RefCell::new(HashMap::new());
}

// =============================================================================
// Allocator
// =============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_alloc(size: i64) -> *mut u8 {
    if size <= 0 {
        return std::ptr::null_mut();
    }
    let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
    unsafe { std::alloc::alloc(layout) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_dealloc(ptr: *mut u8, size: i64) {
    if ptr.is_null() || size <= 0 {
        return;
    }
    let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
    unsafe { std::alloc::dealloc(ptr, layout) };
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
unsafe extern "C" fn prompt_start(prompt: *mut MpPrompt, arg: *mut u8) -> *mut u8 {
    let ctx = unsafe { &*(arg as *const PromptContext) };

    // Register the prompt pointer for this tag (stack for nested prompts)
    PROMPT_REGISTRY.with(|reg| {
        reg.borrow_mut().entry(ctx.tag).or_default().push(prompt);
    });

    // Call the user's body function
    let result = unsafe { (ctx.body_fn)(ctx.env) };

    // Unregister (pop from stack; remove key when empty)
    PROMPT_REGISTRY.with(|reg| {
        let mut reg = reg.borrow_mut();
        if let Some(stack) = reg.get_mut(&ctx.tag) {
            stack.pop();
            if stack.is_empty() {
                reg.remove(&ctx.tag);
            }
        }
    });

    result
}

/// Establish a prompt with the given tag and run `body_fn(env)` under it.
///
/// Signature: `(tag: i32, body_fn: ptr, env: ptr) -> ptr`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_prompt(
    tag: i32,
    body_fn: unsafe extern "C" fn(*mut u8) -> *mut u8,
    env: *mut u8,
) -> *mut u8 {
    let mut ctx = PromptContext { tag, body_fn, env };
    unsafe { mp_prompt(prompt_start, &mut ctx as *mut PromptContext as *mut u8) }
}

/// The `mp_yield` callback: captures the resume pointer into TLS
/// and signals that a yield is active.
unsafe extern "C" fn yield_handler(resume: *mut MpResume, arg: *mut u8) -> *mut u8 {
    YIELD_RESUME.set(resume as *mut u8);
    YIELD_ACTIVE.set(true);
    // Return `arg` as the result of `mp_prompt` (which returns to the handler dispatch loop)
    arg
}

/// Yield to the prompt with the given tag.
///
/// Stores `op_idx` and `shift_val` in TLS so the handler dispatch loop
/// can read them. The yield handler callback captures the resume object.
///
/// Signature: `(tag: i32, op_idx: i32, shift_value: ptr) -> ptr`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_yield(tag: i32, op_idx: i32, shift_value: *mut u8) -> *mut u8 {
    // Store yield metadata in TLS before yielding
    YIELD_OP_IDX.set(op_idx);
    YIELD_SHIFT_VALUE.set(shift_value);

    // Look up the innermost prompt pointer for this tag
    let prompt = PROMPT_REGISTRY.with(|reg| {
        reg.borrow()
            .get(&tag)
            .and_then(|stack| stack.last().copied())
            .expect("ICE: __tribute_yield called with unregistered tag")
    });

    // Yield to the prompt; the yield_handler callback captures the resume
    // and returns shift_value as the result of mp_prompt
    unsafe { mp_yield(prompt, yield_handler, shift_value) }
}

/// Resume a captured continuation with a value.
///
/// Signature: `(continuation: ptr, value: ptr) -> ptr`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_resume(k: *mut u8, val: *mut u8) -> *mut u8 {
    unsafe { mp_resume(k as *mut MpResume, val) }
}

/// Drop a captured continuation without resuming it.
///
/// Signature: `(continuation: ptr) -> ()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tribute_resume_drop(k: *mut u8) {
    unsafe { mp_resume_drop(k as *mut MpResume) }
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
    YIELD_ACTIVE.get() as i8
}

/// Get the operation index of the current yield.
///
/// Signature: `() -> i32`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_get_yield_op_idx() -> i32 {
    YIELD_OP_IDX.get()
}

/// Get the captured continuation (resume object) from the current yield.
///
/// Signature: `() -> ptr`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_get_yield_continuation() -> *mut u8 {
    YIELD_RESUME.get()
}

/// Get the shift value (argument) from the current yield.
///
/// Signature: `() -> ptr`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_get_yield_shift_value() -> *mut u8 {
    YIELD_SHIFT_VALUE.get()
}

/// Reset all TLS yield state to default values.
///
/// Called after the handler dispatch loop processes a yield operation.
///
/// Signature: `() -> ()`
#[unsafe(no_mangle)]
pub extern "C" fn __tribute_reset_yield_state() {
    YIELD_ACTIVE.set(false);
    YIELD_RESUME.set(std::ptr::null_mut());
    YIELD_OP_IDX.set(0);
    YIELD_SHIFT_VALUE.set(std::ptr::null_mut());
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
        YIELD_ACTIVE.set(true);
        YIELD_OP_IDX.set(42);
        __tribute_reset_yield_state();
        assert_eq!(__tribute_yield_active(), 0);
        assert_eq!(__tribute_get_yield_op_idx(), 0);
    }

    #[test]
    fn test_prompt_simple() {
        // Test a simple prompt that doesn't yield: body returns immediately.
        unsafe extern "C" fn body(_env: *mut u8) -> *mut u8 {
            42usize as *mut u8
        }

        unsafe {
            let result = __tribute_prompt(1, body, std::ptr::null_mut());
            assert_eq!(result as usize, 42);
        }
    }
}

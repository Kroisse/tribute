//! Memory dialect operations.
//!
//! Low-level memory operations for FFI and runtime support.

use crate::dialect;

dialect! {
    mod mem {
        /// `mem.data` operation: places bytes in data section, returns pointer.
        #[attr(bytes: any)]
        fn data() -> result;

        /// `mem.load` operation: reads from memory.
        #[attr(offset: u32)]
        fn load(ptr) -> result;

        /// `mem.store` operation: writes to memory.
        #[attr(offset: u32)]
        fn store(ptr, value);
    }
}

// === Pure operation registrations ===
// mem.data and mem.load are pure (non-mutating memory operations)
// mem.store is NOT pure (it modifies memory)

crate::register_pure_op!(mem.data);
crate::register_pure_op!(mem.load);

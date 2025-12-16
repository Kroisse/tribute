//! Memory dialect operations.
//!
//! Low-level memory operations for FFI and runtime support.

use crate::dialect;

dialect! {
    mem {
        /// `mem.data` operation: places bytes in data section, returns pointer.
        op data[bytes]() -> result {};

        /// `mem.load` operation: reads from memory.
        op load[offset](ptr) -> result {};

        /// `mem.store` operation: writes to memory.
        op store[offset](ptr, value) {};
    }
}

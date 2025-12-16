//! Memory dialect operations.
//!
//! Low-level memory operations for FFI and runtime support.

use crate::dialect;

dialect! {
    mem {
        /// `mem.data` operation: places bytes in data section, returns pointer.
        pub op data[bytes]() -> result {};

        /// `mem.load` operation: reads from memory.
        pub op load[offset](ptr) -> result {};

        /// `mem.store` operation: writes to memory.
        pub op store[offset](ptr, value) {};
    }
}

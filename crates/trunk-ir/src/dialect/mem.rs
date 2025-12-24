//! Memory dialect operations.
//!
//! Low-level memory operations for FFI and runtime support.

use crate::{dialect, op_interface};

dialect! {
    mod mem {
        /// `mem.data` operation: places bytes in data section, returns pointer.
        #[attr(bytes)]
        fn data() -> result;

        /// `mem.load` operation: reads from memory.
        #[attr(offset)]
        fn load(ptr) -> result;

        /// `mem.store` operation: writes to memory.
        #[attr(offset)]
        fn store(ptr, value);
    }
}

// === Pure trait implementations ===
// mem.data and mem.load are pure (non-mutating memory operations)
// mem.store is NOT pure (it modifies memory)

impl<'db> op_interface::Pure for Data<'db> {}
impl<'db> op_interface::Pure for Load<'db> {}

// Register pure operations for runtime lookup
inventory::submit! { op_interface::PureOps::register("mem", "data") }
inventory::submit! { op_interface::PureOps::register("mem", "load") }

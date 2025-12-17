//! List dialect operations.
//!
//! Operations for persistent list (RRB tree-backed) manipulation.
//! All operations that return a list or element carry `elem_type` for type information.

use crate::dialect;

dialect! {
    list {
        // === Creation ===

        /// `list.new` operation: creates a list from elements (empty if no elements).
        op new[elem_type](..elements) -> result {};

        // === Access ===

        /// `list.get` operation: gets element at index.
        op get[elem_type](list, index) -> result {};

        /// `list.len` operation: returns the length of the list.
        op len(list) -> result {};

        // === View (multi-result for pattern matching) ===

        /// `list.view_front` operation: returns (head, tail).
        /// Used for `[head, ..tail] = xs` pattern.
        op view_front[elem_type](list) -> head, tail {};

        /// `list.view_back` operation: returns (init, last).
        /// Used for `[..init, last] = xs` pattern.
        op view_back[elem_type](list) -> init, last {};

        // === Modification (persistent) ===

        /// `list.set` operation: returns a new list with element at index updated.
        op set[elem_type](list, index, value) -> result {};

        /// `list.push_front` operation: returns a new list with element prepended.
        op push_front[elem_type](list, value) -> result {};

        /// `list.push_back` operation: returns a new list with element appended.
        op push_back[elem_type](list, value) -> result {};

        // === Split/Concat ===

        /// `list.concat` operation: concatenates two lists.
        op concat[elem_type](left, right) -> result {};

        /// `list.slice` operation: returns a sublist [start, end).
        op slice[elem_type](list, start, end) -> result {};
    }
}

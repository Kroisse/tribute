//! List dialect operations.
//!
//! Operations for persistent list (RRB tree-backed) manipulation.
//! All operations that return a list or element carry `elem_type` for type information.

use crate::dialect;

dialect! {
    mod list {
        // === Creation ===

        /// `list.new` operation: creates a list from elements (empty if no elements).
        #[attr(elem_type: Type)]
        fn new(#[rest] elements) -> result;

        // === Access ===

        /// `list.get` operation: gets element at index.
        #[attr(elem_type: Type)]
        fn get(list, index) -> result;

        /// `list.len` operation: returns the length of the list.
        fn len(list) -> result;

        // === View (multi-result for pattern matching) ===

        /// `list.view_front` operation: returns (head, tail).
        /// Used for `[head, ..tail] = xs` pattern.
        #[attr(elem_type: Type)]
        fn view_front(list) -> (head, tail);

        /// `list.view_back` operation: returns (init, last).
        /// Used for `[..init, last] = xs` pattern.
        #[attr(elem_type: Type)]
        fn view_back(list) -> (init, last);

        // === Modification (persistent) ===

        /// `list.set` operation: returns a new list with element at index updated.
        #[attr(elem_type: Type)]
        fn set(list, index, value) -> result;

        /// `list.push_front` operation: returns a new list with element prepended.
        #[attr(elem_type: Type)]
        fn push_front(list, value) -> result;

        /// `list.push_back` operation: returns a new list with element appended.
        #[attr(elem_type: Type)]
        fn push_back(list, value) -> result;

        // === Split/Concat ===

        /// `list.concat` operation: concatenates two lists.
        #[attr(elem_type: Type)]
        fn concat(left, right) -> result;

        /// `list.slice` operation: returns a sublist [start, end).
        #[attr(elem_type: Type)]
        fn slice(list, start, end) -> result;
    }
}

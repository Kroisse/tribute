//! Data Registry for managing WASM data section entries.
//!
//! The DataRegistry centralizes management of all static data that goes into
//! the WASM data section: string literals, byte arrays, etc.
//!
//! This replaces the ad-hoc approach of adding custom attributes to IR operations
//! and provides a clean separation between IR and data section management.

use rustc_hash::FxHashMap as HashMap;

/// Registry for static data that will be emitted to WASM data section.
#[derive(Debug, Clone)]
pub struct DataRegistry {
    /// Entries stored in the registry
    entries: Vec<DataEntry>,
    /// Current offset in data section
    current_offset: u32,
    /// Map from content hash to entry index for deduplication
    content_map: HashMap<Vec<u8>, usize>,
}

/// A single entry in the data section.
#[derive(Debug, Clone)]
pub struct DataEntry {
    /// Offset in the data section
    pub offset: u32,
    /// Raw bytes of the data
    pub data: Vec<u8>,
    /// Optional label for debugging
    pub label: Option<String>,
}

impl DataRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            current_offset: 0,
            content_map: HashMap::default(),
        }
    }

    /// Add a string literal to the registry.
    /// Returns the offset in the data section.
    /// Deduplicates identical strings.
    pub fn add_string(&mut self, s: &str) -> (u32, u32) {
        self.add_bytes(s.as_bytes(), Some(format!("string: {:?}", s)))
    }

    /// Add raw bytes to the registry.
    /// Returns (offset, length).
    /// Deduplicates identical byte sequences.
    pub fn add_bytes(&mut self, data: &[u8], label: Option<String>) -> (u32, u32) {
        let bytes = data.to_vec();
        let len = bytes.len() as u32;

        // Check if we already have this data
        if let Some(&index) = self.content_map.get(&bytes) {
            let entry = &self.entries[index];
            return (entry.offset, len);
        }

        // Add new entry
        let offset = self.current_offset;
        let index = self.entries.len();

        self.entries.push(DataEntry {
            offset,
            data: bytes.clone(),
            label,
        });

        self.content_map.insert(bytes, index);
        self.current_offset += len;

        (offset, len)
    }

    /// Get all entries for emitting to data section.
    pub fn entries(&self) -> &[DataEntry] {
        &self.entries
    }

    /// Get total size of all data.
    pub fn total_size(&self) -> u32 {
        self.current_offset
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for DataRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest_state_machine::{ReferenceStateMachine, StateMachineTest, prop_state_machine};

    /// The reference model of a registry that byte payloads are added to.
    #[derive(Clone, Debug, Default)]
    struct RegistryModel {
        /// Each distinct payload with its offset, in first-addition order.
        entries: Vec<(Vec<u8>, u32)>,
        next_offset: u32,
        /// The `(offset, length)` the last addition returned.
        returned: Option<(u32, u32)>,
    }

    impl ReferenceStateMachine for RegistryModel {
        type State = Self;
        /// A payload to add. A small byte and length space makes repeated
        /// payloads common.
        type Transition = Vec<u8>;

        fn init_state() -> BoxedStrategy<Self::State> {
            Just(Self::default()).boxed()
        }

        fn transitions(_state: &Self::State) -> BoxedStrategy<Self::Transition> {
            prop::collection::vec(0u8..4, 0..5).boxed()
        }

        fn apply(mut state: Self::State, payload: &Self::Transition) -> Self::State {
            let offset = match state.entries.iter().find(|(data, _)| data == payload) {
                Some(&(_, offset)) => offset,
                None => {
                    let offset = state.next_offset;
                    state.entries.push((payload.clone(), offset));
                    state.next_offset += payload.len() as u32;
                    offset
                }
            };
            state.returned = Some((offset, payload.len() as u32));
            state
        }
    }

    /// Runs `DataRegistry::add_bytes` against [`RegistryModel`].
    struct RegistryMachine;

    impl StateMachineTest for RegistryMachine {
        type SystemUnderTest = DataRegistry;
        type Reference = RegistryModel;

        fn init_test(_model: &RegistryModel) -> Self::SystemUnderTest {
            DataRegistry::new()
        }

        fn apply(
            mut registry: Self::SystemUnderTest,
            model: &RegistryModel,
            payload: Vec<u8>,
        ) -> Self::SystemUnderTest {
            assert_eq!(Some(registry.add_bytes(&payload, None)), model.returned);
            registry
        }

        fn check_invariants(registry: &Self::SystemUnderTest, model: &RegistryModel) {
            assert_eq!(registry.total_size(), model.next_offset);
            let entries = registry.entries();
            assert_eq!(entries.len(), model.entries.len());
            assert_eq!(registry.is_empty(), model.entries.is_empty());
            let mut end = 0u32;
            for (entry, (data, offset)) in entries.iter().zip(&model.entries) {
                assert_eq!(&entry.data, data);
                assert_eq!(entry.offset, *offset);
                assert_eq!(entry.offset, end);
                end += entry.data.len() as u32;
            }
            assert_eq!(end, registry.total_size());
        }
    }

    prop_state_machine! {
        /// Each distinct payload gets the next unaligned offset; a repeated
        /// payload returns its first offset without growing the section.
        #[test]
        fn prop_offsets_contiguous_and_deduplicated(sequential 1..24 => RegistryMachine);
    }

    proptest! {
        /// Strings share the byte payload table and its deduplication.
        #[test]
        fn prop_strings_match_bytes(strings in prop::collection::vec("[ab]{0,3}", 0..16)) {
            let mut by_string = DataRegistry::new();
            let mut by_bytes = DataRegistry::new();
            for s in &strings {
                prop_assert_eq!(by_string.add_string(s), by_bytes.add_bytes(s.as_bytes(), None));
            }
            prop_assert_eq!(by_string.total_size(), by_bytes.total_size());
        }
    }
}

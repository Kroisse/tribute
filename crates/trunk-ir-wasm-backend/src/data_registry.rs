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

    proptest! {
        /// Each distinct payload gets the next unaligned offset; a repeated
        /// payload returns its first offset without growing the section.
        #[test]
        fn prop_offsets_contiguous_and_deduplicated(
            payloads in prop::collection::vec(prop::collection::vec(0u8..4, 0..5), 0..24),
        ) {
            let mut registry = DataRegistry::new();
            let mut model: Vec<(Vec<u8>, u32)> = Vec::new();
            let mut next_offset = 0u32;

            for payload in &payloads {
                let expected = match model.iter().find(|(data, _)| data == payload) {
                    Some(&(_, offset)) => offset,
                    None => {
                        let offset = next_offset;
                        model.push((payload.clone(), offset));
                        next_offset += payload.len() as u32;
                        offset
                    }
                };
                prop_assert_eq!(
                    registry.add_bytes(payload, None),
                    (expected, payload.len() as u32)
                );
                prop_assert_eq!(registry.total_size(), next_offset);
            }

            let entries = registry.entries();
            prop_assert_eq!(entries.len(), model.len());
            prop_assert_eq!(registry.is_empty(), model.is_empty());
            let mut end = 0u32;
            for (entry, (data, offset)) in entries.iter().zip(&model) {
                prop_assert_eq!(&entry.data, data);
                prop_assert_eq!(entry.offset, *offset);
                prop_assert_eq!(entry.offset, end);
                end += entry.data.len() as u32;
            }
            prop_assert_eq!(end, registry.total_size());
        }

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

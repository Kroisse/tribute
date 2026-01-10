//! Data Registry for managing WASM data section entries.
//!
//! The DataRegistry centralizes management of all static data that goes into
//! the WASM data section: string literals, byte arrays, etc.
//!
//! This replaces the ad-hoc approach of adding custom attributes to IR operations
//! and provides a clean separation between IR and data section management.

use std::collections::HashMap;

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
            content_map: HashMap::new(),
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

    #[test]
    fn test_add_string() {
        let mut registry = DataRegistry::new();

        let (offset1, len1) = registry.add_string("hello");
        assert_eq!(offset1, 0);
        assert_eq!(len1, 5);

        let (offset2, len2) = registry.add_string("world");
        assert_eq!(offset2, 5);
        assert_eq!(len2, 5);

        assert_eq!(registry.total_size(), 10);
    }

    #[test]
    fn test_deduplication() {
        let mut registry = DataRegistry::new();

        let (offset1, len1) = registry.add_string("hello");
        let (offset2, len2) = registry.add_string("hello");

        // Same string should return same offset
        assert_eq!(offset1, offset2);
        assert_eq!(len1, len2);

        // Should only have one entry
        assert_eq!(registry.entries().len(), 1);
        assert_eq!(registry.total_size(), 5);
    }

    #[test]
    fn test_add_bytes() {
        let mut registry = DataRegistry::new();

        let data1 = vec![1, 2, 3, 4];
        let data2 = vec![5, 6];

        let (offset1, len1) = registry.add_bytes(&data1, None);
        assert_eq!(offset1, 0);
        assert_eq!(len1, 4);

        let (offset2, len2) = registry.add_bytes(&data2, None);
        assert_eq!(offset2, 4);
        assert_eq!(len2, 2);

        assert_eq!(registry.total_size(), 6);
    }
}

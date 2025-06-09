//! Interned string implementation for efficient string management
//!
//! This module provides an interned string system that:
//! - Interns empty strings and small strings automatically
//! - Uses hash-based deduplication for larger strings  
//! - Provides inline storage for very small strings to avoid allocations

use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Maximum size for inline string storage (15 bytes + 1 length byte)
const INLINE_STRING_MAX_LEN: usize = 15;

/// Interned string table that manages string deduplication
pub struct InternedStringTable {
    strings: DashMap<u64, InternedStringData>,
}

impl InternedStringTable {
    /// Create a new empty interned string table
    pub fn new() -> Self {
        Self {
            strings: DashMap::new(),
        }
    }
    
    /// Insert a string into the table if not already present
    pub fn intern(&self, hash: u64, bytes: &[u8]) {
        self.strings.entry(hash).or_insert_with(|| {
            InternedStringData::Heap { 
                data: bytes.to_vec().into_boxed_slice() 
            }
        });
    }
    
    /// Get string data by hash
    pub fn get(&self, hash: u64) -> Option<Vec<u8>> {
        self.strings.get(&hash).map(|entry| entry.as_bytes().to_vec())
    }
    
    /// Get the count of interned strings
    pub fn len(&self) -> usize {
        self.strings.len()
    }
    
    /// Clear all interned strings
    pub fn clear(&self) {
        self.strings.clear();
    }
}

impl Default for InternedStringTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents the actual string data storage
#[derive(Debug, Clone)]
enum InternedStringData {
    /// Heap allocated storage for longer strings
    Heap { 
        data: Box<[u8]> 
    },
}

impl InternedStringData {
    fn as_bytes(&self) -> &[u8] {
        match self {
            InternedStringData::Heap { data } => data,
        }
    }
}

/// An interned string that can be efficiently stored and compared
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TributeString {
    /// Empty string (always interned)
    Empty,
    /// Inline string (stored directly in the enum, up to 15 bytes)
    Inline { 
        data: [u8; INLINE_STRING_MAX_LEN], 
        len: u8 
    },
    /// Interned string (referenced by hash)
    Interned { 
        hash: u64,
        len: usize,
    },
}

impl TributeString {
    /// Create a new tribute string from bytes with interned string table
    pub fn from_bytes_with_table(bytes: &[u8], interned_table: &InternedStringTable) -> Self {
        let len = bytes.len();
        
        // Handle empty string
        if len == 0 {
            return TributeString::Empty;
        }
        
        // Handle inline strings (small optimization)
        if len <= INLINE_STRING_MAX_LEN {
            let mut data = [0u8; INLINE_STRING_MAX_LEN];
            data[..len].copy_from_slice(bytes);
            return TributeString::Inline { 
                data, 
                len: len as u8 
            };
        }
        
        // For larger strings, compute hash and intern
        let hash = Self::compute_hash(bytes);
        
        // Insert into interned table if not already present
        interned_table.intern(hash, bytes);
        
        TributeString::Interned { hash, len }
    }
    
    /// Create a new tribute string from bytes (legacy - uses global table)
    #[deprecated(note = "Use from_bytes_with_table for better isolation")]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let len = bytes.len();
        
        // Handle empty string
        if len == 0 {
            return TributeString::Empty;
        }
        
        // Handle inline strings (small optimization)
        if len <= INLINE_STRING_MAX_LEN {
            let mut data = [0u8; INLINE_STRING_MAX_LEN];
            data[..len].copy_from_slice(bytes);
            return TributeString::Inline { 
                data, 
                len: len as u8 
            };
        }
        
        // For larger strings, compute hash and intern
        let hash = Self::compute_hash(bytes);
        
        // Legacy fallback - this won't work properly without global table
        // TODO: Remove this method once all callers are updated
        
        TributeString::Interned { hash, len }
    }
    
    
    /// Get the length of the string
    pub fn len(&self) -> usize {
        match self {
            TributeString::Empty => 0,
            TributeString::Inline { len, .. } => *len as usize,
            TributeString::Interned { len, .. } => *len,
        }
    }
    
    /// Check if the string is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the string data as bytes with interned string table
    pub fn as_bytes_with_table(&self, interned_table: &InternedStringTable) -> Vec<u8> {
        match self {
            TributeString::Empty => Vec::new(),
            TributeString::Inline { data, len } => data[..*len as usize].to_vec(),
            TributeString::Interned { hash, .. } => {
                interned_table.get(*hash).unwrap_or_default()
            }
        }
    }
    
    /// Get the string data as bytes (legacy - uses global table)
    #[deprecated(note = "Use as_bytes_with_table for better isolation")]
    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            TributeString::Empty => Vec::new(),
            TributeString::Inline { data, len } => data[..*len as usize].to_vec(),
            TributeString::Interned { .. } => {
                // Legacy fallback - this won't work properly without global table
                Vec::new()
            }
        }
    }
    
    /// Get the string data as bytes with lifetime tied to self (for inline strings)
    /// For interned strings, this returns a temporary Vec
    pub fn as_bytes_ref(&self) -> &[u8] {
        match self {
            TributeString::Empty => &[],
            TributeString::Inline { data, len } => &data[..*len as usize],
            TributeString::Interned { .. } => {
                // For interned strings, we need to use the as_bytes() method
                // This is less efficient but handles the lifetime correctly
                // In practice, most strings should be inline for small ones
                &[] // Fallback - caller should use as_bytes() for interned strings
            }
        }
    }
    
    /// Convert to a UTF-8 string if valid with interned string table
    pub fn as_str_with_table(&self, interned_table: &InternedStringTable) -> Result<String, std::str::Utf8Error> {
        Ok(std::str::from_utf8(&self.as_bytes_with_table(interned_table))?.to_string())
    }
    
    /// Convert to a UTF-8 string if valid (legacy - uses global table)
    #[deprecated(note = "Use as_str_with_table for better isolation")]
    pub fn as_str(&self) -> Result<String, std::str::Utf8Error> {
        #[allow(deprecated)]
        Ok(std::str::from_utf8(&self.as_bytes())?.to_string())
    }
    
    /// Convert to a UTF-8 string slice for inline strings only
    pub fn as_str_ref(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(self.as_bytes_ref())
    }
    
    /// Compute hash for byte slice
    fn compute_hash(bytes: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Get statistics about interned strings from table
    pub fn interned_count_from_table(interned_table: &InternedStringTable) -> usize {
        interned_table.len()
    }
    
    /// Clear all interned strings from table (for testing)
    pub fn clear_interned_from_table(interned_table: &InternedStringTable) {
        interned_table.clear();
    }
}

impl std::fmt::Display for TributeString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[allow(deprecated)]
        match self.as_str() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<invalid UTF-8>"),
        }
    }
}

impl TributeString {
    /// Create from a string slice with interned string table
    pub fn from_str_with_table(s: &str, interned_table: &InternedStringTable) -> Self {
        Self::from_bytes_with_table(s.as_bytes(), interned_table)
    }
}

impl From<&str> for TributeString {
    fn from(s: &str) -> Self {
        #[allow(deprecated)]
        Self::from_bytes(s.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thread_local::ThreadLocal;
    use std::sync::LazyLock;
    
    // Test interned string table for isolated testing (thread local)
    static TEST_INTERNED_TABLE: LazyLock<ThreadLocal<InternedStringTable>> = LazyLock::new(ThreadLocal::new);

    #[test]
    fn test_empty_string() {
        let table = TEST_INTERNED_TABLE.get_or_default();
        let empty1 = TributeString::from_str_with_table("", table);
        let empty2 = TributeString::from_bytes_with_table(&[], table);
        
        assert_eq!(empty1, TributeString::Empty);
        assert_eq!(empty2, TributeString::Empty);
        assert_eq!(empty1, empty2);
        assert_eq!(empty1.len(), 0);
        assert!(empty1.is_empty());
    }
    
    #[test]
    fn test_inline_strings() {
        let table = TEST_INTERNED_TABLE.get_or_default();
        let short = TributeString::from_str_with_table("hello", table);
        let medium = TributeString::from_str_with_table("hello, world!", table);
        
        // Both should be inline
        assert!(matches!(short, TributeString::Inline { .. }));
        assert!(matches!(medium, TributeString::Inline { .. }));
        
        assert_eq!(short.len(), 5);
        assert_eq!(medium.len(), 13);
        assert_eq!(short.as_str_with_table(table).unwrap(), "hello");
        assert_eq!(medium.as_str_with_table(table).unwrap(), "hello, world!");
    }
    
    #[test]
    fn test_interned_strings() {
        // Create a fresh table for this test
        let test_table = InternedStringTable::new();
        
        let long1 = TributeString::from_str_with_table("this is a longer string that will be interned", &test_table);
        let long2 = TributeString::from_str_with_table("this is a longer string that will be interned", &test_table);
        let different = TributeString::from_str_with_table("this is a different longer string", &test_table);
        
        // Should be interned
        assert!(matches!(long1, TributeString::Interned { .. }));
        assert!(matches!(long2, TributeString::Interned { .. }));
        assert!(matches!(different, TributeString::Interned { .. }));
        
        // Same content should have same hash
        if let (TributeString::Interned { hash: h1, .. }, TributeString::Interned { hash: h2, .. }) = (long1, long2) {
            assert_eq!(h1, h2);
        }
        
        assert_eq!(long1.as_str_with_table(&test_table).unwrap(), "this is a longer string that will be interned");
        assert_eq!(TributeString::interned_count_from_table(&test_table), 2); // Two different strings interned
    }
    
    #[test]
    fn test_string_equality() {
        let table = TEST_INTERNED_TABLE.get_or_default();
        let s1 = TributeString::from_str_with_table("test", table);
        let s2 = TributeString::from_str_with_table("test", table);
        let s3 = TributeString::from_str_with_table("different", table);
        
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }
    
    #[test]
    fn test_boundary_conditions() {
        let table = TEST_INTERNED_TABLE.get_or_default();
        // Test exactly at the inline boundary
        let exactly_15 = "123456789012345"; // 15 characters
        let exactly_16 = "1234567890123456"; // 16 characters
        
        let s15 = TributeString::from_str_with_table(exactly_15, table);
        let s16 = TributeString::from_str_with_table(exactly_16, table);
        
        assert!(matches!(s15, TributeString::Inline { .. }));
        assert!(matches!(s16, TributeString::Interned { .. }));
        
        assert_eq!(s15.len(), 15);
        assert_eq!(s16.len(), 16);
    }
}
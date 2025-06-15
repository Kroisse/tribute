//! Target platform information for code generation
//!
//! This module defines target-specific information that affects
//! code generation, type sizes, and calling conventions.

use crate::Db;
use target_lexicon::{Architecture, Triple};

/// Target platform information for code generation
#[salsa::input(debug)]
pub struct TargetInfo {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu")
    pub triple: Triple,
    /// Pointer size in bytes (4 for 32-bit, 8 for 64-bit)
    pub pointer_size: u8,
    /// Byte order of the target platform
    pub endianness: Endianness,
    /// Native integer size in bytes
    pub native_int_size: u8,
}

/// Byte order of the target platform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endianness {
    Little,
    Big,
}

impl TargetInfo {
    /// Create target info from a triple
    pub fn from_triple(db: &dyn Db, triple: Triple) -> Self {
        let pointer_size = match triple.pointer_width() {
            Ok(target_lexicon::PointerWidth::U16) => 2,
            Ok(target_lexicon::PointerWidth::U32) => 4,
            Ok(target_lexicon::PointerWidth::U64) => 8,
            _ => 8, // Default to 64-bit
        };

        let endianness = match triple.endianness() {
            Ok(target_lexicon::Endianness::Little) => Endianness::Little,
            Ok(target_lexicon::Endianness::Big) => Endianness::Big,
            _ => Endianness::Little, // Default to little endian
        };

        // Native int size is usually the same as pointer size on modern platforms
        let native_int_size = pointer_size;

        Self::new(db, triple, pointer_size, endianness, native_int_size)
    }

    /// Get the default target for the current host
    pub fn host(db: &dyn Db) -> Self {
        Self::from_triple(db, Triple::host())
    }

    /// Check if this is a 64-bit target
    pub fn is_64bit(&self, db: &dyn Db) -> bool {
        self.pointer_size(db) == 8
    }

    /// Check if this is a 32-bit target
    pub fn is_32bit(&self, db: &dyn Db) -> bool {
        self.pointer_size(db) == 4
    }

    /// Check if this target uses little endian byte order
    pub fn is_little_endian(&self, db: &dyn Db) -> bool {
        self.endianness(db) == Endianness::Little
    }

    /// Get the architecture of this target
    pub fn architecture(&self, db: &dyn Db) -> Architecture {
        self.triple(db).architecture
    }

    /// Get a string representation of the target
    pub fn as_str(&self, db: &dyn Db) -> String {
        self.triple(db).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TributeDatabaseImpl;
    use std::str::FromStr;

    #[test]
    fn test_host_target() {
        let db = TributeDatabaseImpl::default();
        let target = TargetInfo::host(&db);

        // Should have reasonable values
        let pointer_size = target.pointer_size(&db);
        assert!(pointer_size == 4 || pointer_size == 8);
        assert_eq!(target.native_int_size(&db), pointer_size);
    }

    #[test]
    fn test_x86_64_target() {
        let db = TributeDatabaseImpl::default();
        let triple = Triple::from_str("x86_64-unknown-linux-gnu").unwrap();
        let target = TargetInfo::from_triple(&db, triple);

        assert_eq!(target.pointer_size(&db), 8);
        assert_eq!(target.native_int_size(&db), 8);
        assert_eq!(target.endianness(&db), Endianness::Little);
        assert!(target.is_64bit(&db));
        assert!(!target.is_32bit(&db));
        assert!(target.is_little_endian(&db));
    }

    #[test]
    fn test_i386_target() {
        let db = TributeDatabaseImpl::default();
        let triple = Triple::from_str("i386-unknown-linux-gnu").unwrap();
        let target = TargetInfo::from_triple(&db, triple);

        assert_eq!(target.pointer_size(&db), 4);
        assert_eq!(target.native_int_size(&db), 4);
        assert_eq!(target.endianness(&db), Endianness::Little);
        assert!(!target.is_64bit(&db));
        assert!(target.is_32bit(&db));
        assert!(target.is_little_endian(&db));
    }
}

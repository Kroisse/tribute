//! Test database for tribute-trunk-ir tests.
//!
//! This provides a minimal Salsa database for testing IR operations
//! without depending on tribute-core.

/// Minimal test database for tribute-trunk-ir.
#[derive(Default, Clone)]
#[salsa::db]
pub struct TestDatabase {
    storage: salsa::Storage<Self>,
}

#[salsa::db]
impl salsa::Database for TestDatabase {}

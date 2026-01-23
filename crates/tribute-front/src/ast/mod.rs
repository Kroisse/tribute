//! Abstract Syntax Tree (AST) types for Tribute.
//!
//! This module defines the Salsa-tracked AST representation used throughout
//! the compiler frontend. The AST is designed with several key principles:
//!
//! ## Generic Phase Parameter
//!
//! AST types are parameterized by a "phase" type `V` that represents
//! what information is available about name references:
//!
//! - `UnresolvedName`: After parsing, before name resolution
//! - `ResolvedRef<'db>`: After name resolution
//! - `TypedRef<'db>`: After type checking
//!
//! This allows the same AST structure to be used throughout compilation
//! while the type system ensures correct phase handling.
//!
//! ## NodeId + SpanMap Pattern
//!
//! Following rust-analyzer's approach, AST nodes don't store spans directly.
//! Instead, each node has a `NodeId` that can be used to look up its span
//! in a separate `SpanMap`. This has several benefits:
//!
//! - AST is purely structural (easier to work with)
//! - Span changes don't invalidate Salsa caches
//! - Additional metadata can be added using the same pattern
//!
//! ## Type vs TypeScheme
//!
//! The type system distinguishes between:
//!
//! - `Type`: Monomorphic types (during inference, may contain type variables)
//! - `TypeScheme`: Polymorphic types with universally quantified parameters
//!
//! This separation is important for proper handling of let-polymorphism
//! and generalization.
//!
//! ## Example Usage
//!
//! ```ignore
//! // After parsing
//! let parsed: ParsedModule = lower_cst_to_ast(db, source, cst);
//!
//! // After name resolution
//! let resolved: ResolvedModule<'db> = resolve_module(db, parsed);
//!
//! // After type checking
//! let typed: TypedModule<'db> = typecheck_module(db, resolved);
//! ```

mod decl;
mod expr;
mod node_id;
mod pattern;
mod phases;
mod types;

// Re-export core types
pub use decl::*;
pub use expr::*;
pub use node_id::*;
pub use pattern::*;
pub use phases::*;
pub use types::*;

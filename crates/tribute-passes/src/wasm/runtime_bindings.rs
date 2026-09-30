//! The C runtime helpers the Wasm target binds after the representation/ABI
//! boundary.
//!
//! A bodyless `abi = "C"` declaration is bound by its symbol name, the C link
//! name it declares. Wasm has no C linker: only the helpers listed here have a
//! Wasm implementation, and the boundary exit verifier rejects a reference to
//! any other.

use tribute_ir::dialect::ability::evidence_abi;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::OpRef;
use trunk_ir::symbol_table::SymbolTable;

use super::intrinsic_to_wasm::{BYTES_CONCAT, BYTES_LEN, BYTES_RANGE_EQUAL};

/// Every C helper with a Wasm implementation: the evidence helpers
/// `evidence_to_wasm` binds and the bytes helpers `intrinsic_to_wasm` binds.
pub const PROVIDED: &[&str] = &[
    evidence_abi::LOOKUP,
    evidence_abi::LOOKUP_TR,
    evidence_abi::LOOKUP_HANDLER,
    evidence_abi::EXTEND,
    BYTES_LEN,
    BYTES_CONCAT,
    BYTES_RANGE_EQUAL,
];

/// Whether the Wasm target binds the C helper named `name`.
pub fn provides(name: Symbol) -> bool {
    PROVIDED.iter().any(|&provided| name == provided)
}

/// Whether `op` is a bodyless `abi = "C"` declaration.
pub fn is_c_declaration(ctx: &IrContext, op: OpRef) -> bool {
    let data = ctx.op(op);
    data.regions.is_empty() && data.attributes.get_str("abi") == Some("C")
}

/// The C link name of the declaration `callee` resolves to, if it is a
/// bodyless `abi = "C"` declaration.
pub(crate) fn c_helper(ctx: &IrContext, symbols: &SymbolTable, callee: Symbol) -> Option<Symbol> {
    let declaration = symbols.resolve(callee)?;
    is_c_declaration(ctx, declaration)
        .then(|| ctx.op(declaration).attributes.get_symbol("sym_name"))
        .flatten()
}

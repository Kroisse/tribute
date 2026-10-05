//! The C runtime helpers the Wasm target binds after the representation/ABI
//! boundary.
//!
//! A bodyless `abi = "C"` declaration is bound by its symbol name, the C link
//! name it declares. Wasm has no C linker: only the helpers listed here have a
//! Wasm implementation, and the boundary exit verifier rejects a reference to
//! any other.

use tribute_ir::dialect::ability::evidence_abi;
use trunk_ir::context::IrContext;
use trunk_ir::refs::OpRef;
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::{Symbol, SymbolPath};

use super::evidence_to_wasm::NEXT_TAG;
use super::intrinsic_to_wasm::{BYTES_CONCAT, BYTES_LEN, BYTES_RANGE_EQUAL, BYTES_SLICE_OR_PANIC};

/// Every C helper with a Wasm implementation: the evidence and prompt tag
/// helpers `evidence_to_wasm` binds and the bytes helpers `intrinsic_to_wasm`
/// binds.
pub const PROVIDED: &[&str] = &[
    evidence_abi::LOOKUP,
    evidence_abi::LOOKUP_TR,
    evidence_abi::EXTEND,
    evidence_abi::MASK,
    evidence_abi::DUP,
    evidence_abi::OUTER,
    evidence_abi::TAIL,
    evidence_abi::WITH_TAIL,
    evidence_abi::PUSH,
    NEXT_TAG,
    BYTES_LEN,
    BYTES_CONCAT,
    BYTES_RANGE_EQUAL,
    BYTES_SLICE_OR_PANIC,
];

/// Whether the Wasm target binds the C helper named `name`.
pub fn provides(name: &Symbol) -> bool {
    PROVIDED.iter().any(|&provided| name == provided)
}

/// Whether `op` is a bodyless `abi = "C"` declaration.
pub fn is_c_declaration(ctx: &IrContext, op: OpRef) -> bool {
    !ctx.op_has_regions(op) && ctx.op(op).attributes.get_str(ctx, "abi") == Some("C")
}

/// The C link name of the declaration `callee` resolves to, if it is a
/// bodyless `abi = "C"` declaration.
pub(crate) fn c_helper(
    ctx: &IrContext,
    symbols: &SymbolTable,
    callee: &SymbolPath,
) -> Option<Symbol> {
    let declaration = symbols.resolve(callee)?;
    is_c_declaration(ctx, declaration)
        .then(|| {
            ctx.op(declaration)
                .attributes
                .get_str(ctx, "sym_name")
                .map(Symbol::new)
        })
        .flatten()
}

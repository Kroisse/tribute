//! Compiler-owned runtime items, found by canonical path.
//!
//! Runtime code is Tribute source under `std::runtime` (`new-plans/modules.md`,
//! "Compiler 소유 runtime 모듈"). A runtime item's identity is its canonical
//! declaration path. Lowering finds a runtime function by the IR symbol built
//! from that path, which only the prelude definition has, and a runtime type
//! by its entry in the root module's prelude type dictionary.

use itertools::Itertools;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::symbol_table::SymbolTable;
use trunk_ir::types::Location;
use trunk_ir::{Symbol, SymbolPath};

/// The module that holds the compiler-owned runtime code.
pub const RUNTIME_MODULE: &[&str] = &["std", "runtime"];

/// One runtime item that lowering uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeItem {
    /// The canonical declaration path, one component per entry.
    pub path: &'static [&'static str],
    pub kind: RuntimeItemKind,
}

/// The declaration a runtime item must be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeItemKind {
    /// A function with a body.
    Function,
    /// A struct, published as a prelude type.
    Struct,
}

/// The evidence runtime module (`new-plans/cps-effects.md`).
pub mod evidence {
    pub const MARKER: &[&str] = &["std", "runtime", "evidence", "Marker"];
    pub const LOOKUP: &[&str] = &["std", "runtime", "evidence", "lookup"];
    pub const LOOKUP_TR: &[&str] = &["std", "runtime", "evidence", "lookup_tr"];
    pub const EXTEND: &[&str] = &["std", "runtime", "evidence", "extend"];
    pub const MASK: &[&str] = &["std", "runtime", "evidence", "mask"];
    pub const DUP: &[&str] = &["std", "runtime", "evidence", "dup"];
    pub const OUTER: &[&str] = &["std", "runtime", "evidence", "outer"];
    pub const TAIL: &[&str] = &["std", "runtime", "evidence", "tail"];
    pub const WITH_TAIL: &[&str] = &["std", "runtime", "evidence", "with_tail"];
    pub const PUSH: &[&str] = &["std", "runtime", "evidence", "push"];

    /// The evidence functions.
    pub const FUNCTIONS: [&[&str]; 9] = [
        LOOKUP, LOOKUP_TR, EXTEND, MASK, DUP, OUTER, TAIL, WITH_TAIL, PUSH,
    ];
}

const fn function(path: &'static [&'static str]) -> RuntimeItem {
    RuntimeItem {
        path,
        kind: RuntimeItemKind::Function,
    }
}

/// The runtime items lowering uses.
pub const RUNTIME_ITEMS: &[RuntimeItem] = &[
    RuntimeItem {
        path: evidence::MARKER,
        kind: RuntimeItemKind::Struct,
    },
    function(evidence::LOOKUP),
    function(evidence::LOOKUP_TR),
    function(evidence::EXTEND),
    function(evidence::MASK),
    function(evidence::DUP),
    function(evidence::OUTER),
    function(evidence::TAIL),
    function(evidence::WITH_TAIL),
    function(evidence::PUSH),
];

/// The compiler intrinsic identities of the runtime array `Slots(a)`
/// (`new-plans/types.md`, "Slots").
pub const SLOTS_INTRINSICS: [&str; 6] = [
    "std::runtime::Slots::empty",
    "std::runtime::Slots::len",
    "std::runtime::Slots::get",
    "std::runtime::Slots::insert",
    "std::runtime::Slots::remove",
    "std::runtime::Slots::replace",
];

/// The IR symbol of the root-module definition at `path`.
///
/// This is the one place that turns a runtime item's path into a symbol.
pub fn runtime_symbol(path: &[&str]) -> Symbol {
    Symbol::new(&path.iter().format("::").to_string())
}

/// The root-qualified references to the function items of `items`, for
/// dead-function elimination to keep as roots.
pub fn function_entry_points(items: &[RuntimeItem]) -> Vec<SymbolPath> {
    items
        .iter()
        .filter(|item| item.kind == RuntimeItemKind::Function)
        .map(|item| SymbolPath::new([runtime_symbol(item.path)]))
        .collect()
}

/// Global DCE configuration that roots every function of `runtime`.
///
/// A runtime function is a root until the target lowering stages that call
/// it have run (`new-plans/modules.md`, "생존 범위"). Every runtime function
/// is called by target evidence lowering or earlier; the cleanup DCE after
/// it removes the ones nothing calls.
pub fn kept_alive(runtime: &[RuntimeItem]) -> trunk_ir::transforms::GlobalDceConfig {
    trunk_ir::transforms::GlobalDceConfig {
        extra_entry_points: function_entry_points(runtime),
        ..Default::default()
    }
}

/// The runtime type at `path`, published in the prelude type dictionary.
pub fn runtime_type(ctx: &IrContext, module: Module, path: &[&str]) -> Option<TypeRef> {
    crate::metadata::prelude_type(ctx, module.op(), runtime_symbol(path).as_str())
}

/// A runtime function the module does not define as expected.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[display("runtime function `{path}` {kind}")]
pub struct RuntimeFunctionError {
    pub path: String,
    pub kind: RuntimeFunctionErrorKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display)]
pub enum RuntimeFunctionErrorKind {
    #[display("has no unique definition")]
    Missing,
    #[display("is not a func.func with a body")]
    NotAFunction,
}

/// The `func.func` definition of one runtime function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeFunction {
    function: func::Func,
}

impl RuntimeFunction {
    /// The root-module definition of the runtime function at `path`.
    pub fn lookup(
        ctx: &IrContext,
        module: Module,
        path: &[&str],
    ) -> Result<Self, RuntimeFunctionError> {
        let error = |kind| RuntimeFunctionError {
            path: path.iter().format("::").to_string(),
            kind,
        };
        let reference = SymbolPath::new([runtime_symbol(path)]);
        let op = SymbolTable::collect(ctx, module)
            .resolve_in(ctx, module.op(), &reference)
            .ok_or_else(|| error(RuntimeFunctionErrorKind::Missing))?;
        match func::Func::from_op(ctx, op) {
            Ok(function) if ctx.op_has_regions(op) => Ok(Self { function }),
            _ => Err(error(RuntimeFunctionErrorKind::NotAFunction)),
        }
    }

    pub fn function(self) -> func::Func {
        self.function
    }

    pub fn signature(self, ctx: &IrContext) -> func::FuncSig {
        func::FuncSig::from_type_ref(ctx, self.function.r#type(ctx))
            .expect("a runtime function has a func.func_sig type")
    }

    /// The reference a call in the root module makes to this definition.
    pub fn callee(self, ctx: &IrContext) -> SymbolPath {
        SymbolPath::new([Symbol::new(self.function.sym_name(ctx))])
    }

    /// Build a direct call to this function with `args`, typed by its
    /// signature. The caller inserts it.
    pub fn call(
        self,
        ctx: &mut IrContext,
        location: Location,
        args: impl IntoIterator<Item = ValueRef>,
    ) -> func::Call {
        let callee = self.callee(ctx);
        let results = self.signature(ctx).results(ctx).to_vec();
        func::Call::operands(args)
            .callee(callee)
            .results(results)
            .build(ctx, location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"core.module @main {tribute.types = {"rt::Cell" = core.i32}} {
  func.func @"rt::probe"(%value: core.i32) -> core.i32 {
    func.return %value
  }
  func.func @"rt::declared"(%value: core.i32) -> core.i32
  func.func @main() -> core.i32 {
    %zero = arith.const {value = 0} : core.i32
    func.return %zero
  }
}"#;

    #[test]
    fn finds_runtime_items_by_path() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, FIXTURE);
        let probe = RuntimeFunction::lookup(&ctx, module, &["rt", "probe"]).expect("probe");
        assert_eq!(probe.function().sym_name(&ctx), "rt::probe");
        assert_eq!(
            RuntimeFunction::lookup(&ctx, module, &["rt", "absent"]).map_err(|error| error.kind),
            Err(RuntimeFunctionErrorKind::Missing)
        );
        assert_eq!(
            RuntimeFunction::lookup(&ctx, module, &["rt", "declared"]).map_err(|error| error.kind),
            Err(RuntimeFunctionErrorKind::NotAFunction)
        );
        let i32_ty = trunk_ir::types::TypeDataBuilder::new("core.i32").intern(&mut ctx);
        assert_eq!(runtime_type(&ctx, module, &["rt", "Cell"]), Some(i32_ty));
        assert_eq!(runtime_type(&ctx, module, &["rt", "probe"]), None);
    }

    #[test]
    fn builds_calls_from_the_definition_signature() {
        let mut ctx = IrContext::new();
        let module = trunk_ir::parser::parse_test_module(&mut ctx, FIXTURE);
        let probe = RuntimeFunction::lookup(&ctx, module, &["rt", "probe"]).expect("probe");
        let location = ctx.op(probe.function().op_ref()).location;
        let body = probe.function().body(&ctx);
        let entry = ctx.region(body).blocks[0];
        let argument = ctx.block_arg(entry, 0);
        let call = probe.call(&mut ctx, location, [argument]);
        assert_eq!(call.callee(&ctx), &SymbolPath::new(["rt::probe"]));
        assert_eq!(
            ctx.value_ty(call.result(&ctx)),
            probe.signature(&ctx).results(&ctx)[0]
        );
    }

    #[test]
    fn roots_only_function_items() {
        let items = [
            RuntimeItem {
                path: &["std", "runtime", "evidence", "extend"],
                kind: RuntimeItemKind::Function,
            },
            RuntimeItem {
                path: &["std", "runtime", "evidence", "Marker"],
                kind: RuntimeItemKind::Struct,
            },
        ];
        assert_eq!(
            function_entry_points(&items),
            [SymbolPath::new(["std::runtime::evidence::extend"])]
        );
    }
}

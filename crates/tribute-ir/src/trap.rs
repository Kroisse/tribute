//! The causes of a trap and the intrinsics that raise them
//! (`design/modules.md`, "Trap intrinsic").

use std::fmt;
use std::str::FromStr;

use trunk_ir::context::IrContext;
use trunk_ir::types::Attribute;

/// Why an execution ends in a trap.
///
/// It displays as, and parses from, its [keyword](Self::keyword).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrapCause {
    /// A position or range outside its sequence.
    OutOfRange,
    /// Evidence without the marker of an ability the program performs,
    /// which is a compiler bug.
    MissingAbility,
}

impl TrapCause {
    /// Every cause.
    pub const ALL: [Self; 2] = [Self::OutOfRange, Self::MissingAbility];

    /// The keyword that spells this cause: the stored attribute value of
    /// `tribute_control.trap`, and the `code` of the trap it lowers to.
    pub fn keyword(self) -> &'static str {
        match self {
            Self::OutOfRange => "out_of_range",
            Self::MissingAbility => "missing_ability",
        }
    }

    /// The compiler intrinsic identity of the function that raises this
    /// cause.
    pub fn intrinsic(self) -> &'static str {
        match self {
            Self::OutOfRange => "std::runtime::trap_out_of_range",
            Self::MissingAbility => "std::runtime::trap_missing_ability",
        }
    }

    /// The cause the compiler intrinsic `identity` raises; `None` for any
    /// other intrinsic.
    pub fn of_intrinsic(identity: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|cause| cause.intrinsic() == identity)
    }

    fn from_attribute(ctx: &IrContext, value: &Attribute) -> Option<Self> {
        match value {
            Attribute::String(keyword) => ctx.str(*keyword).parse().ok(),
            _ => None,
        }
    }
}

/// `Attr<TrapCause>`: a keyword string, which the schema checks.
impl trunk_ir::attr_kind::AttrKind for TrapCause {
    const KIND: trunk_ir::op_schema::AttributeKind = trunk_ir::op_schema::AttributeKind::String;
    type Out<'ctx> = TrapCause;
    type In = TrapCause;
    const DOMAIN: Option<&'static str> = Some("one of \"out_of_range\" and \"missing_ability\"");

    fn accepts(ctx: &IrContext, attr: &Attribute) -> bool {
        Self::from_attribute(ctx, attr).is_some()
    }

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> TrapCause {
        Self::from_attribute(ctx, attr).expect("unverified trap cause keyword")
    }

    fn write(ctx: &mut IrContext, value: TrapCause) -> Attribute {
        ctx.string_attr(value.keyword())
    }

    /// A cause is written as its keyword, `out_of_range`.
    fn print_text(p: &mut trunk_ir::printer::AsmPrinter<'_>, attr: &Attribute) -> fmt::Result {
        fmt::Write::write_str(p, Self::read(p.ctx(), attr).keyword())
    }

    fn parse_text(
        p: &mut trunk_ir::parser::AsmParser<'_>,
    ) -> trunk_ir::parser::ParseResult<Attribute> {
        trunk_ir::asm_format::parse_keyword(p, |ctx, word| {
            let cause: Self = word.parse().ok()?;
            Some(ctx.string_attr(cause.keyword()))
        })
    }
}

impl fmt::Display for TrapCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.keyword())
    }
}

/// A string that is not a trap cause keyword.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[display("unknown trap cause")]
pub struct UnknownTrapCause;

impl FromStr for TrapCause {
    type Err = UnknownTrapCause;

    fn from_str(keyword: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|cause| cause.keyword() == keyword)
            .ok_or(UnknownTrapCause)
    }
}

//! Arena-based ability dialect.

#[trunk_ir::arena_dialect]
mod ability {
    #[attr(ability_ref: Type)]
    fn evidence_lookup(evidence: ()) -> result {}

    #[attr(ability_ref: Type, prompt_tag: any)]
    fn evidence_extend(evidence: ()) -> result {}

    #[attr(max_ops_per_handler: u32)]
    fn handler_table() {
        #[region(entries)]
        {}
    }

    #[attr(tag: u32, op_count: u32)]
    fn handler_entry() {
        #[region(funcs)]
        {}
    }
}

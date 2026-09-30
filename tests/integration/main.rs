//! Integration tests, linked into one binary.

mod common;

mod active_pipeline_goldens;
mod cli_diagnostics;
mod constructor_instances;
mod cps_closure_lowering;
mod diagnostic_snapshots;
mod e2e_ability_core;
mod e2e_ability_effect_row;
mod e2e_ability_handler;
mod e2e_ability_nested;
mod e2e_add;
mod e2e_effect_instances;
mod e2e_float;
mod e2e_int_text;
mod e2e_local_callables;
mod e2e_native;
mod effect_var_collision;
mod evidence_lambda;
mod frontend_instances;
mod frontend_type_roots;
mod open_callback_evidence_root;
mod optimization_conformance;
mod salsa_integration;
mod wasm_compilation;

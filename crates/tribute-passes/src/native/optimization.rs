//! Native optimization policies and the observable native pipeline stages
//! at which they are selected.

use super::ownership_plan::NativeOwnershipPlanOptions;

/// Independently selectable native-backend optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeOptimizationOptions {
    pub paired_rc_elimination: PairedRcEliminationPolicy,
    pub borrowed_parameters: BorrowedParameterPolicy,
    pub temporary_borrows: TemporaryBorrowPolicy,
    pub last_use_moves: LastUseMovePolicy,
    pub field_moves: FieldMovePolicy,
}

impl NativeOptimizationOptions {
    pub const fn production() -> Self {
        Self {
            paired_rc_elimination: PairedRcEliminationPolicy::Enabled,
            borrowed_parameters: BorrowedParameterPolicy::ElideProvenBorrowed,
            temporary_borrows: TemporaryBorrowPolicy::ElideProvenFieldBorrows,
            last_use_moves: LastUseMovePolicy::Move,
            field_moves: FieldMovePolicy::Move,
        }
    }

    pub const fn baseline() -> Self {
        Self {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows: TemporaryBorrowPolicy::Preserve,
            last_use_moves: LastUseMovePolicy::Acquire,
            field_moves: FieldMovePolicy::Acquire,
        }
    }
}

/// Policy for local retain/release pair elimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PairedRcEliminationPolicy {
    Disabled,
    Enabled,
}

/// Policy for eliding ownership of proven borrowed native parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorrowedParameterPolicy {
    Preserve,
    ElideProvenBorrowed,
}

/// Policy for eliding ownership of proven field-derived native temporaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporaryBorrowPolicy {
    Preserve,
    ElideProvenFieldBorrows,
}

/// Policy for moving a native value's ownership unit into a store or an
/// ordinary consumed call that is its last use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LastUseMovePolicy {
    Acquire,
    Move,
}

/// Policy for moving a field's unit out of an owner that dies in the block
/// where the field is read, instead of acquiring one for the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldMovePolicy {
    Acquire,
    Move,
}

/// Stable native-pipeline boundaries available to optimization tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativePipelineStage {
    /// Immediately after reference-counting operations are inserted.
    AfterRcInsertion,
    /// After borrowed-parameter-aware insertion, before paired elimination.
    AfterBorrowedParameterOptimization,
    /// After temporary field-borrow-aware insertion, before last-use moves and
    /// paired elimination.
    AfterTemporaryBorrowOptimization,
    /// After optional local RC optimization, before cast resolution and lowering.
    AfterRcOptimization,
}

/// Select ownership-plan policy for each observable native pipeline boundary.
///
/// The staged dump boundaries deliberately suppress later elisions so their
/// output remains comparable with the corresponding optimization policy.
pub fn ownership_plan_options(
    stop_after: Option<NativePipelineStage>,
    optimizations: NativeOptimizationOptions,
) -> NativeOwnershipPlanOptions {
    let configured = NativeOwnershipPlanOptions {
        elide_proven_borrowed_parameters: matches!(
            optimizations.borrowed_parameters,
            BorrowedParameterPolicy::ElideProvenBorrowed
        ),
        elide_proven_field_borrows: matches!(
            optimizations.temporary_borrows,
            TemporaryBorrowPolicy::ElideProvenFieldBorrows
        ),
        move_units_at_last_use: matches!(optimizations.last_use_moves, LastUseMovePolicy::Move),
        move_fields_from_dying_owners: matches!(optimizations.field_moves, FieldMovePolicy::Move),
    };

    match stop_after {
        Some(NativePipelineStage::AfterRcInsertion) => NativeOwnershipPlanOptions {
            elide_proven_borrowed_parameters: false,
            elide_proven_field_borrows: false,
            move_units_at_last_use: false,
            move_fields_from_dying_owners: false,
        },
        Some(NativePipelineStage::AfterBorrowedParameterOptimization) => {
            NativeOwnershipPlanOptions {
                elide_proven_borrowed_parameters: configured.elide_proven_borrowed_parameters,
                elide_proven_field_borrows: false,
                move_units_at_last_use: false,
                move_fields_from_dying_owners: false,
            }
        }
        Some(NativePipelineStage::AfterTemporaryBorrowOptimization) => NativeOwnershipPlanOptions {
            move_units_at_last_use: false,
            move_fields_from_dying_owners: false,
            ..configured
        },
        Some(NativePipelineStage::AfterRcOptimization) | None => configured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_options_disable_every_optional_native_optimization() {
        assert_eq!(
            NativeOptimizationOptions::baseline(),
            NativeOptimizationOptions {
                paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
                borrowed_parameters: BorrowedParameterPolicy::Preserve,
                temporary_borrows: TemporaryBorrowPolicy::Preserve,
                last_use_moves: LastUseMovePolicy::Acquire,
                field_moves: FieldMovePolicy::Acquire,
            }
        );
        assert_eq!(
            ownership_plan_options(None, NativeOptimizationOptions::baseline()),
            NativeOwnershipPlanOptions {
                elide_proven_borrowed_parameters: false,
                elide_proven_field_borrows: false,
                move_units_at_last_use: false,
                move_fields_from_dying_owners: false,
            }
        );
    }

    #[test]
    fn ownership_plan_options_follow_stage_policy_table() {
        let parameter_elision_only = NativeOptimizationOptions {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::ElideProvenBorrowed,
            temporary_borrows: TemporaryBorrowPolicy::Preserve,
            last_use_moves: LastUseMovePolicy::Acquire,
            field_moves: FieldMovePolicy::Acquire,
        };
        let field_elision_only = NativeOptimizationOptions {
            paired_rc_elimination: PairedRcEliminationPolicy::Disabled,
            borrowed_parameters: BorrowedParameterPolicy::Preserve,
            temporary_borrows: TemporaryBorrowPolicy::ElideProvenFieldBorrows,
            last_use_moves: LastUseMovePolicy::Acquire,
            field_moves: FieldMovePolicy::Acquire,
        };
        let all_optimizations = NativeOptimizationOptions::production();

        let cases = [
            (
                Some(NativePipelineStage::AfterRcInsertion),
                all_optimizations,
                (false, false, false),
            ),
            (
                Some(NativePipelineStage::AfterBorrowedParameterOptimization),
                all_optimizations,
                (true, false, false),
            ),
            (
                Some(NativePipelineStage::AfterBorrowedParameterOptimization),
                field_elision_only,
                (false, false, false),
            ),
            (
                Some(NativePipelineStage::AfterTemporaryBorrowOptimization),
                parameter_elision_only,
                (true, false, false),
            ),
            (
                Some(NativePipelineStage::AfterTemporaryBorrowOptimization),
                all_optimizations,
                (true, true, false),
            ),
            (
                Some(NativePipelineStage::AfterRcOptimization),
                field_elision_only,
                (false, true, false),
            ),
            (None, all_optimizations, (true, true, true)),
        ];

        // The last element selects both unit moves, which only the final
        // stages make.
        for (stage, optimizations, (borrowed_parameters, field_borrows, moves)) in cases {
            assert_eq!(
                ownership_plan_options(stage, optimizations),
                NativeOwnershipPlanOptions {
                    elide_proven_borrowed_parameters: borrowed_parameters,
                    elide_proven_field_borrows: field_borrows,
                    move_units_at_last_use: moves,
                    move_fields_from_dying_owners: moves,
                },
                "unexpected ownership plan options at {stage:?}",
            );
        }
    }
}

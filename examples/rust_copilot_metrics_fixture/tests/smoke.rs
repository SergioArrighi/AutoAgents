use rust_copilot_metrics_fixture::engine::run_default;
use rust_copilot_metrics_fixture::types::{ValueState, pick_left};

#[test]
fn run_default_returns_zero_for_mixed_values() {
    // Mixed-sign inputs fail the default positive rule and collapse to Zero.
    assert!(matches!(run_default(5, -5), ValueState::Zero));
}

#[test]
fn pick_left_generic_returns_first_value() {
    assert_eq!(pick_left(7_i32, 3_i32), 7_i32);
}

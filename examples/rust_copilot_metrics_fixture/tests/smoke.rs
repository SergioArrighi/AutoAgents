use rust_copilot_metrics_fixture::engine::run_default;
use rust_copilot_metrics_fixture::syntax_playground::{classify_window, fold_signals};
use rust_copilot_metrics_fixture::types::{pick_left, ValueState};

#[test]
fn run_default_returns_zero_for_mixed_values() {
    // Mixed-sign inputs fail the default positive rule and collapse to Zero.
    assert!(matches!(run_default(5, -5), ValueState::Zero));
}

#[test]
fn pick_left_generic_returns_first_value() {
    assert_eq!(pick_left(7_i32, 3_i32), 7_i32);
}

#[test]
fn fold_signals_accumulates_signed_values() {
    assert_eq!(fold_signals(&[2, 0, -3, 4]), 3);
}

#[test]
fn classify_window_detects_cross_zero_window() {
    let Some((left, right)) = classify_window(&[-2, 0, 7]) else {
        panic!("expected a classified window");
    };
    assert!(matches!(left, ValueState::Negative));
    assert!(matches!(right, ValueState::Positive));
}

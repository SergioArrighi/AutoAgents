use rust_copilot_metrics_fixture::engine::run_default;
use rust_copilot_metrics_fixture::types::ValueState;

#[test]
fn run_default_returns_zero_for_mixed_values() {
    assert!(matches!(run_default(5, -5), ValueState::Zero));
}

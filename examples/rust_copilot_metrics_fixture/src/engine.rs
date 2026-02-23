use crate::types::{classify, PositiveRule, ValueState};
use crate::{evaluate_pair, Rule};

/// Thin engine wrapper used to exercise trait + associated-function extraction.
pub struct Engine;

impl Engine {
    /// Runs the provided rule against both inputs.
    pub fn run(rule: &dyn Rule, left: i32, right: i32) -> bool {
        evaluate_pair(rule, left, right)
    }
}

/// Fixture default behavior:
/// - If both values are positive, classify their sum.
/// - Otherwise return [`ValueState::Zero`].
pub fn run_default(left: i32, right: i32) -> ValueState {
    let rule = PositiveRule;
    let allowed = Engine::run(&rule, left, right);
    if allowed {
        classify(fixture_helper::add(left, right))
    } else {
        ValueState::Zero
    }
}

use crate::types::{classify, PositiveRule, ValueState};
use crate::{evaluate_pair, Rule};

pub struct Engine;

impl Engine {
    pub fn run(rule: &dyn Rule, left: i32, right: i32) -> bool {
        evaluate_pair(rule, left, right)
    }
}

pub fn run_default(left: i32, right: i32) -> ValueState {
    let rule = PositiveRule;
    let allowed = Engine::run(&rule, left, right);
    if allowed {
        classify(left + right)
    } else {
        ValueState::Zero
    }
}

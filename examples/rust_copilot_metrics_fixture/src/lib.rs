pub mod engine;
pub mod types;

pub trait Rule {
    fn allows(&self, value: i32) -> bool;
}

pub fn evaluate(rule: &dyn Rule, value: i32) -> bool {
    rule.allows(value)
}

pub fn evaluate_pair(rule: &dyn Rule, left: i32, right: i32) -> bool {
    evaluate(rule, left) && evaluate(rule, right)
}

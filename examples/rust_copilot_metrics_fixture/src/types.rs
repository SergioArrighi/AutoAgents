pub struct PositiveRule;

impl crate::Rule for PositiveRule {
    fn allows(&self, value: i32) -> bool {
       value > 0
    }
}

pub enum ValueState {
    Positive,
    Zero,
    Negative,
}

pub fn classify(value: i32) -> ValueState {
       if value > 0 {
        ValueState::Positive
    } else if value < 0 {
        ValueState::Negative
    } else {
        ValueState::Zero
    }
}

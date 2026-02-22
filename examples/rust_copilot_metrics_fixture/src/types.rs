/// Rule that allows strictly positive integers.
pub struct PositiveRule;

impl crate::Rule for PositiveRule {
    fn allows(&self, value: i32) -> bool {
        value > 0
    }
}

/// Coarse value classification used by fixture logic.
pub enum ValueState {
    /// Value is greater than zero.
    Positive,
    /// Value equals zero.
    Zero,
    /// Value is less than zero.
    Negative,
}

/// Converts a raw integer into a [`ValueState`].
pub fn classify(value: i32) -> ValueState {
    if value > 0 {
        ValueState::Positive
    } else if value < 0 {
        ValueState::Negative
    } else {
        ValueState::Zero
    }
}

/// Minimal generic helper for extraction coverage of generics + where clauses.
pub fn pick_left<T>(left: T, _right: T) -> T
where
    T: Copy,
{
    left
}

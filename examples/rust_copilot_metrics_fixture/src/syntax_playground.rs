use crate::types::{classify, ValueState};

/// Folds positive inputs, subtracts negatives, and ignores zeros.
pub fn fold_signals(inputs: &[i32]) -> i32 {
    inputs.iter().fold(0_i32, |acc, value| match value {
        v if *v > 0 => acc + v,
        v if *v < 0 => acc - v.abs(),
        _ => acc,
    })
}

/// Returns `(first, last)` only when at least two values are present and ordered.
pub fn classify_window(inputs: &[i32]) -> Option<(ValueState, ValueState)> {
    let [first, .., last] = inputs else {
        return None;
    };

    let (left, right) = (classify(*first), classify(*last));
    match (&left, &right) {
        (ValueState::Negative, ValueState::Positive) => Some((left, right)),
        (ValueState::Positive, ValueState::Negative) => Some((left, right)),
        _ => None,
    }
}

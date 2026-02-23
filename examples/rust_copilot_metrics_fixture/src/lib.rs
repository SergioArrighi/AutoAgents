//! Minimal deterministic Rust fixture used by `rust_copilot_daemon` evaluation.
//!
//! The crate intentionally exposes a small set of symbols (trait, types, and
//! functions) with cross-module references so extraction metrics stay stable.

/// Engine entry points used by tests and fixture scanners.
pub mod engine;
/// Syntax-heavy helpers used to stress syntax artifact extraction.
pub mod syntax_playground;
/// Shared types and classification helpers used by the engine.
pub mod types;

/// Predicate abstraction used by the fixture engine.
pub trait Rule {
    /// Returns `true` when the provided value is allowed by the rule.
    fn allows(&self, value: i32) -> bool;
}

/// Evaluates a single value against a [`Rule`].
pub fn evaluate(rule: &dyn Rule, value: i32) -> bool {
    rule.allows(value)
}

/// Evaluates two values and requires both to be allowed by the same [`Rule`].
pub fn evaluate_pair(rule: &dyn Rule, left: i32, right: i32) -> bool {
    evaluate(rule, left) && evaluate(rule, right)
}

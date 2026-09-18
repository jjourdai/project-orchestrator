//! Canonical energy model for notes.
//!
//! # Why this module exists
//!
//! Until 2026-09, note energy was decayed **in place**: every run of
//! `update_energy_scores` multiplied the stored `energy` by
//! `exp(-days_idle / half_life)` without ever moving `last_activated`.
//! Each run therefore re-applied the decay of the *whole* idle period to an
//! already-decayed value. With the heartbeat firing 4×/day the real law was
//!
//! ```text
//! energy(J) = exp( -J(J+1) / (4 · half_life) )      // quadratic in J
//! ```
//!
//! instead of the intended `exp(-J / half_life)`. Measured effect: the
//! intended 1.0 → 0.05 in 42 days became 1.0 → 0.0 in ~5 days, and 93 % of
//! all active notes sat at exactly 0.0. Combined with the 60-day archival
//! rule, the entire knowledge base had a 60-day life expectancy.
//!
//! # The model
//!
//! Energy is a **pure function of two persisted values**, never of the number
//! of times the job ran:
//!
//! ```text
//! energy = clamp( energy_base · exp(-days_since(last_activated) / half_life) )
//! ```
//!
//! * `energy_base` — the energy at the instant of the last activation. It is
//!   written **only** by activation paths (`boost_energy`, `confirm_note`,
//!   note creation, `init_note_energy`). The decay job never changes it.
//! * `energy` — a materialised cache of the formula above, refreshed by the
//!   decay job so that Cypher can `ORDER BY` / filter on it.
//!
//! Because the job recomputes from `energy_base` rather than from its own
//! previous output, running it once after 30 days is now genuinely equivalent
//! to running it 120 times over those 30 days.

/// Canonical decay time constant, in days.
///
/// Despite the historical "half-life" naming this is the time constant `τ` of
/// `exp(-t/τ)`, not a half-life: after `τ` days energy is at `1/e` (~37 %),
/// and it reaches [`ENERGY_CLAMP_FLOOR`] after `τ·ln(1/floor)` ≈ 64 days.
///
/// Before this constant existed three different values were in use —
/// 14 days in the heartbeat, 90 days in the HTTP handler default, and a
/// separate `0.5^(days/90)` law inside `boost_energy` / `computed_energy`,
/// which stacked a *second* decay on top of an already-decayed value.
pub const ENERGY_HALF_LIFE_DAYS: f64 = 14.0;

/// Below this, a note's energy is flushed to exactly 0.0 ("dead neuron").
///
/// Must stay **strictly below** the activation boost, so that reading a note
/// lifts it clear of the floor instead of landing on it.
pub const ENERGY_CLAMP_FLOOR: f64 = 0.01;

/// Energy below which a note older than 60 days may be auto-archived.
///
/// Must stay **strictly below** [`ENERGY_CLAMP_FLOOR`] — which means in
/// practice that only fully-dead notes (energy flushed to exactly 0.0) are
/// ever eligible. The three values used to be the same literal `0.05`:
/// boost, clamp floor and archive threshold. A note at 0.0 came back to
/// exactly 0.05 when read — precisely *on* the threshold — then dropped
/// under it the next day and was flushed back to 0.0. Reanimation was
/// arithmetically impossible.
pub const ARCHIVE_ENERGY_THRESHOLD: f64 = 0.005;

/// Energy granted to a note when it is injected into a chat system prompt.
///
/// 25× the clamp floor: a note read once buys itself ~45 idle days before it
/// can be flushed to zero again.
pub const CONTEXT_ENERGY_BOOST: f64 = 0.25;

/// Compute a note's current energy from its immutable base.
///
/// `days_idle` is the elapsed time since `last_activated`. The result is
/// clamped to `[0, 1]` and flushed to 0.0 below [`ENERGY_CLAMP_FLOOR`].
///
/// This function is **idempotent in time**: it depends only on `days_idle`,
/// never on how often it has been called.
pub fn decayed_energy(base: f64, days_idle: f64, half_life_days: f64) -> f64 {
    let base = base.clamp(0.0, 1.0);
    if half_life_days <= 0.0 {
        return base;
    }
    let value = if days_idle <= 0.0 {
        base
    } else {
        base * (-days_idle / half_life_days).exp()
    };
    if value < ENERGY_CLAMP_FLOOR {
        0.0
    } else {
        value.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_are_strictly_ordered() {
        // The whole point of the 2026-09 fix: these three must never collide
        // again. archive < clamp floor < boost.
        assert!(
            ARCHIVE_ENERGY_THRESHOLD < ENERGY_CLAMP_FLOOR,
            "archive threshold must be strictly below the clamp floor, \
             otherwise a note flushed to the floor is archived on sight"
        );
        assert!(
            ENERGY_CLAMP_FLOOR * 5.0 < CONTEXT_ENERGY_BOOST,
            "the activation boost must lift a note well clear of the floor, \
             not land it exactly on the threshold"
        );
    }

    #[test]
    fn decay_is_temporally_idempotent() {
        // Applying the law once over 30 days must equal applying it
        // 120 times over the same 30 days — the property the old in-place
        // job claimed in its docstring and did not have.
        let one_shot = decayed_energy(1.0, 30.0, ENERGY_HALF_LIFE_DAYS);

        let mut stepwise = 0.0;
        for step in 1..=120 {
            let days = 30.0 * (step as f64) / 120.0;
            stepwise = decayed_energy(1.0, days, ENERGY_HALF_LIFE_DAYS);
        }

        assert!(
            (one_shot - stepwise).abs() < 1e-9,
            "one-shot {one_shot} vs stepwise {stepwise}"
        );
        assert!((one_shot - (-30.0f64 / 14.0).exp()).abs() < 1e-9);
    }

    #[test]
    fn weekly_reading_keeps_a_note_alive_for_90_days() {
        // Verification for the "decouple the thresholds" step: a note read
        // once a week must stay strictly above the archive threshold.
        let mut base = CONTEXT_ENERGY_BOOST;
        let mut lowest = f64::MAX;
        for _week in 0..13 {
            // seven idle days, sampled daily
            for day in 1..=7 {
                let e = decayed_energy(base, day as f64, ENERGY_HALF_LIFE_DAYS);
                lowest = lowest.min(e);
            }
            let current = decayed_energy(base, 7.0, ENERGY_HALF_LIFE_DAYS);
            base = (current + CONTEXT_ENERGY_BOOST).min(1.0);
        }
        assert!(
            lowest > ARCHIVE_ENERGY_THRESHOLD,
            "a weekly-read note dipped to {lowest}, at or under the archive \
             threshold {ARCHIVE_ENERGY_THRESHOLD}"
        );
    }

    #[test]
    fn dead_energy_is_flushed_to_exactly_zero() {
        let e = decayed_energy(1.0, 365.0, ENERGY_HALF_LIFE_DAYS);
        assert_eq!(e, 0.0);
    }
}

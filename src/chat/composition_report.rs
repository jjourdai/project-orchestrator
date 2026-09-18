//! What actually went into the system prompt — and what was cut out of it.
//!
//! # Why this exists
//!
//! Two defects lived for months in prompt composition: an energy decay that
//! emptied the knowledge base, and a truncation that beheaded whole sections
//! of context. Neither was noticed by reading code or by watching the tests.
//! Both were found only by asking a live session to quote a marker back and
//! seeing it fail to.
//!
//! The reason they hid so well is that the composer's output is a single
//! opaque string, and the part it drops is discarded on the spot. Nothing —
//! no log, no metric, no UI — could answer "why did this session not see that
//! guideline?". This module makes the composition observable: sizes per
//! source, which sections lost items, and the exact text that was removed.
//!
//! The report is *persisted* rather than merely emitted live, because the
//! question is almost always asked after the fact, about a session that has
//! already ended.

use serde::{Deserialize, Serialize};

/// Which of the three dynamic sources a [`SourceReport`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicSource {
    /// Freshly searched knowledge for this message (skills, propagated notes).
    Enrichment,
    /// Resume data carried over from the previous session.
    Continuity,
    /// Standing project context: plans, guidelines, gotchas, globals.
    ProjectContext,
}

impl DynamicSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Enrichment => "enrichment",
            Self::Continuity => "continuity",
            Self::ProjectContext => "project_context",
        }
    }
}

/// A markdown section that lost items to the budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DroppedSection {
    /// Header line, e.g. `## Global Guidelines`.
    pub section: String,
    /// Items rendered into the prompt.
    pub items_kept: usize,
    /// Items the budget removed.
    pub items_dropped: usize,
    /// Characters removed from this section.
    pub chars_dropped: usize,
}

/// Per-source accounting for one of the three dynamic inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceReport {
    pub source: DynamicSource,
    /// Size of the source before truncation.
    pub original_chars: usize,
    /// Size of what actually reached the prompt.
    pub kept_chars: usize,
    /// Character budget this source was given.
    pub budget_chars: usize,
    /// Sections that lost items, in document order.
    pub dropped_sections: Vec<DroppedSection>,
}

impl SourceReport {
    pub fn dropped_chars(&self) -> usize {
        self.original_chars.saturating_sub(self.kept_chars)
    }
}

/// A full account of one prompt composition.
///
/// The three headline numbers map one-to-one onto the chips in the session
/// banner: static context injected, dynamic context injected, dynamic context
/// truncated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionReport {
    /// Base prompt + FSM fragments + tool reference: the parts that do not
    /// depend on the knowledge graph and are never truncated.
    pub static_chars: usize,
    /// Dynamic context that reached the prompt.
    pub dynamic_chars: usize,
    /// Dynamic context that did not — the headline number for the alert chip.
    pub truncated_chars: usize,
    /// Budget the dynamic context had to fit into.
    pub dynamic_budget_chars: usize,
    /// Size of the final assembled prompt, separators included.
    pub total_chars: usize,
    /// Breakdown of the static half, for the detail panel.
    pub base_prompt_chars: usize,
    pub fsm_section_chars: usize,
    pub tool_reference_chars: usize,
    /// Per-source accounting.
    pub sources: Vec<SourceReport>,
    /// The text that was removed, verbatim.
    ///
    /// Kept in full rather than summarised: a count tells you *that* something
    /// was lost, only the content tells you *whether it mattered*.
    pub truncated_content: String,
}

/// The part of a [`CompositionReport`] that travels with the `system_init`
/// event: enough to render the three chips, without the full cut text.
///
/// The removed content stays in the persisted report and is fetched on demand
/// — a prompt's truncated half can be tens of kilobytes, and it does not
/// belong in every session-open payload.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionSummary {
    pub static_chars: usize,
    pub dynamic_chars: usize,
    pub truncated_chars: usize,
    pub dynamic_budget_chars: usize,
    pub total_chars: usize,
    /// Sections that lost items, worst first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_sections: Vec<DroppedSection>,
}

impl CompositionReport {
    /// The chip-sized view of this report.
    pub fn summary(&self) -> CompositionSummary {
        CompositionSummary {
            static_chars: self.static_chars,
            dynamic_chars: self.dynamic_chars,
            truncated_chars: self.truncated_chars,
            dynamic_budget_chars: self.dynamic_budget_chars,
            total_chars: self.total_chars,
            dropped_sections: self
                .all_dropped_sections()
                .into_iter()
                .cloned()
                .collect(),
        }
    }

    /// Sections that lost items, across every source, worst first.
    pub fn all_dropped_sections(&self) -> Vec<&DroppedSection> {
        let mut all: Vec<&DroppedSection> = self
            .sources
            .iter()
            .flat_map(|s| s.dropped_sections.iter())
            .collect();
        all.sort_by(|a, b| b.chars_dropped.cmp(&a.chars_dropped));
        all
    }

    /// Whether anything was withheld from the prompt.
    pub fn is_truncated(&self) -> bool {
        self.truncated_chars > 0
    }
}

/// What a single budgeted render removed.
///
/// Produced by the truncation itself, so the numbers cannot drift from what
/// was actually written.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TruncationDetail {
    pub dropped_sections: Vec<DroppedSection>,
    /// Every dropped item, rendered as it would have appeared.
    pub dropped_content: String,
}

impl TruncationDetail {
    pub fn is_empty(&self) -> bool {
        self.dropped_sections.is_empty() && self.dropped_content.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropped_sections_are_ranked_by_volume() {
        let report = CompositionReport {
            sources: vec![
                SourceReport {
                    source: DynamicSource::ProjectContext,
                    original_chars: 1000,
                    kept_chars: 400,
                    budget_chars: 400,
                    dropped_sections: vec![DroppedSection {
                        section: "## Guidelines".into(),
                        items_kept: 1,
                        items_dropped: 2,
                        chars_dropped: 200,
                    }],
                },
                SourceReport {
                    source: DynamicSource::Enrichment,
                    original_chars: 900,
                    kept_chars: 500,
                    budget_chars: 500,
                    dropped_sections: vec![DroppedSection {
                        section: "## Global Guidelines".into(),
                        items_kept: 0,
                        items_dropped: 4,
                        chars_dropped: 400,
                    }],
                },
            ],
            ..Default::default()
        };

        let ranked = report.all_dropped_sections();
        assert_eq!(ranked[0].section, "## Global Guidelines");
        assert_eq!(ranked[1].section, "## Guidelines");
    }

    #[test]
    fn a_report_with_no_losses_is_not_truncated() {
        let report = CompositionReport {
            static_chars: 1000,
            dynamic_chars: 500,
            ..Default::default()
        };
        assert!(!report.is_truncated());
    }
}

#[cfg(test)]
mod transport_tests {
    use super::*;
    use crate::chat::types::ChatEvent;

    /// A session created before this feature carries no report. The field must
    /// then be absent from the wire payload entirely — not `null`, not zeros —
    /// so the frontend renders its banner exactly as it did before and shows
    /// no chips. Zeros would read as "nothing was truncated", which is a claim
    /// we cannot make about a session we never measured.
    #[test]
    fn system_init_without_a_report_is_wire_compatible() {
        let event = ChatEvent::SystemInit {
            cli_session_id: "sid-legacy".to_string(),
            model: Some("claude-sonnet-4".to_string()),
            tools: vec!["Bash".to_string()],
            mcp_servers: vec![],
            permission_mode: Some("default".to_string()),
            composition: None,
        };

        let json = serde_json::to_value(&event).unwrap();
        assert!(
            json.get("composition").is_none(),
            "absent report must not serialize at all, got: {json}"
        );
        // The pre-existing metadata is untouched.
        assert_eq!(json["cli_session_id"], "sid-legacy");
        assert_eq!(json["model"], "claude-sonnet-4");
        assert_eq!(json["permission_mode"], "default");
    }

    /// And a payload written before the field existed must still deserialize.
    #[test]
    fn legacy_system_init_payload_still_parses() {
        let json = r#"{"type":"system_init","cli_session_id":"sid-1","model":"claude-sonnet-4"}"#;
        let event: ChatEvent = serde_json::from_str(json).unwrap();
        match event {
            ChatEvent::SystemInit { composition, .. } => assert!(composition.is_none()),
            other => panic!("expected SystemInit, got {other:?}"),
        }
    }

    /// With a report, the three headline numbers reach the wire.
    #[test]
    fn system_init_carries_the_three_chip_numbers() {
        let report = CompositionReport {
            static_chars: 42_000,
            dynamic_chars: 9_800,
            truncated_chars: 5_100,
            dynamic_budget_chars: 10_000,
            total_chars: 51_800,
            sources: vec![SourceReport {
                source: DynamicSource::ProjectContext,
                original_chars: 14_900,
                kept_chars: 9_800,
                budget_chars: 10_000,
                dropped_sections: vec![DroppedSection {
                    section: "## Global Guidelines".into(),
                    items_kept: 0,
                    items_dropped: 6,
                    chars_dropped: 5_100,
                }],
            }],
            truncated_content: "## Global Guidelines\n- [Critical] warm up\n".into(),
            ..Default::default()
        };

        let event = ChatEvent::SystemInit {
            cli_session_id: "sid-2".to_string(),
            model: None,
            tools: vec![],
            mcp_servers: vec![],
            permission_mode: None,
            composition: Some(report.summary()),
        };

        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["composition"]["static_chars"], 42_000);
        assert_eq!(json["composition"]["dynamic_chars"], 9_800);
        assert_eq!(json["composition"]["truncated_chars"], 5_100);
        assert_eq!(
            json["composition"]["dropped_sections"][0]["section"],
            "## Global Guidelines"
        );

        // The cut text itself stays out of the session-open payload — it is
        // fetched on demand from the persisted report.
        assert!(
            !json.to_string().contains("warm up"),
            "the truncated content rode along in the system_init payload"
        );
    }
}

//! FsmPromptComposer — modular, FSM-aware prompt assembly.
//!
//! Replaces the monolithic `build_system_prompt()` in manager.rs with a composable
//! system that adapts the prompt based on:
//! - Scaffolding level (0-4) — progressive complexity reduction
//! - Active protocol runs — inject prompt_fragment, available_tools, forbidden_actions
//! - Project context — plans, constraints, guidelines, topology
//! - Session continuity — previous session resume data
//! - User message — intent detection for tool group selection
//! - Routing hints — future: DualTrackRouter provides section weights
//!
//! Architecture:
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   FsmPromptComposer                     │
//! │                                                         │
//! │  Inputs:                                                │
//! │  1. scaffolding_level: u8                               │
//! │  2. protocol_runs: Vec<ProtocolRunStatus>               │
//! │  3. project_context_markdown: String                    │
//! │  4. continuity_markdown: String                         │
//! │  5. user_message: &str                                  │
//! │  6. routing_hints: Option<Vec<SectionHint>>             │
//! │                                                         │
//! │  Pipeline:                                              │
//! │  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐ │
//! │  │ Select   │→ │ Inject    │→ │ Select tool groups   │ │
//! │  │ sections │  │ FSM frags │  │ (intent+FSM+level)   │ │
//! │  └──────────┘  └───────────┘  └──────────────────────┘ │
//! │       ↓              ↓               ↓                  │
//! │  ┌──────────────────────────────────────────────┐      │
//! │  │ Assemble: base + FSM + dynamic + tool_ref    │      │
//! │  └──────────────────────────────────────────────┘      │
//! │       ↓                                                 │
//! │  ┌──────────┐                                           │
//! │  │ Truncate  │ → String                                 │
//! │  └──────────┘                                           │
//! └─────────────────────────────────────────────────────────┘
//! ```

use super::prompt::TOOL_REFERENCE;
use super::prompt_sections::{
    assemble_sections, assemble_sections_weighted, extract_tool_reference,
};
use super::composition_report::{
    CompositionReport, DroppedSection, DynamicSource, SourceReport, TruncationDetail,
};
use super::routing::{HeuristicRouter, RoutingContext, RoutingDecisionRecord, RoutingProvider};
use super::stages::status_injection::ProtocolRunStatus;

// Re-export SectionHint from routing for backward compatibility
pub use super::routing::SectionHint;

// ============================================================================
// FsmPromptComposer — the central prompt assembly engine
// ============================================================================

/// Inputs for prompt composition, gathered before calling `compose()`.
///
/// All async data fetching happens outside the composer — it receives
/// pre-rendered markdown strings for dynamic sections.
#[derive(Debug, Clone, Default)]
pub struct ComposerInput<'a> {
    /// Scaffolding level (0=full guidance, 4=expert).
    pub scaffolding_level: u8,
    /// Active protocol runs with their FSM state data.
    pub protocol_runs: &'a [ProtocolRunStatus],
    /// Dynamic project context rendered as markdown (from `context_to_markdown`).
    pub project_context_markdown: &'a str,
    /// Session continuity rendered as markdown (from `load_session_context`).
    pub continuity_markdown: &'a str,
    /// Enrichment pipeline output rendered as markdown (from `to_system_prompt_markdown`).
    /// Integrated into the system prompt instead of being prepended to the user message.
    pub enrichment_markdown: &'a str,
    /// The user's current message (for intent detection).
    pub user_message: &'a str,
    /// Optional routing hints (ignored by HeuristicRouter, used by DualTrackRouter).
    pub routing_hints: Option<&'a [SectionHint]>,
    /// Whether the project has sibling projects (multi-project workspace).
    pub is_multi_project: bool,
    /// Whether there are active plans.
    pub has_active_plan: bool,
    /// Number of tasks across active plans.
    pub task_count: usize,
    /// Model name (e.g., "claude-sonnet-4-20250514"). Used for adaptive context budgeting.
    pub model: &'a str,
    /// Pre-computed embedding of the user message (for DualTrackRouter).
    /// `None` if embeddings are unavailable → routing falls back to heuristics.
    pub message_embedding: Option<&'a Vec<f32>>,
    /// Whether external MCP servers are connected (triggers External tool group).
    pub external_tools_available: bool,
}

// Default is derived — all numeric fields default to 0, bools to false,
// &str to "", Option to None, &[] to empty slice.

/// Fallback character budget when the model is unknown (~2500 tokens at 4 chars/token).
const DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET: usize = 10_000;

/// Compute the dynamic context character budget based on the model's context window
/// and the current system prompt length.
///
/// Strategy: allocate up to 15% of the model's remaining context window (after base prompt)
/// for dynamic context, clamped between 5K and 40K chars.
///
/// Known context windows (in tokens):
/// - claude-opus-4 / claude-sonnet-4: 200K
/// - claude-haiku-3-5: 200K
/// - claude-3-opus: 200K
/// - Smaller/unknown models: fallback to 10K chars
fn compute_dynamic_budget(model: &str, base_prompt_chars: usize) -> usize {
    // Estimate model context window in tokens
    let context_window_tokens: usize =
        if model.contains("opus") || model.contains("sonnet") || model.contains("haiku") {
            200_000
        } else {
            // Unknown model → use conservative fallback
            return DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET;
        };

    // Convert to chars (~4 chars per token)
    let context_window_chars = context_window_tokens * 4;

    // Remaining space after base prompt
    let remaining = context_window_chars.saturating_sub(base_prompt_chars);

    // Allocate 15% of remaining space for dynamic context
    let budget = remaining * 15 / 100;

    // Clamp between 5K and 40K chars
    budget.clamp(5_000, 40_000)
}

/// The FsmPromptComposer assembles the full system prompt from modular sections.
///
/// It is stateless — each call to `compose()` produces a fresh prompt string.
/// All state (project context, protocol runs, etc.) is passed via `ComposerInput`.
pub struct FsmPromptComposer;

impl FsmPromptComposer {
    /// Compose the full system prompt using the default [`HeuristicRouter`].
    ///
    /// This is the standard entry point. For custom routing (e.g., DualTrackRouter),
    /// use [`compose_with_router()`] instead.
    pub fn compose(input: &ComposerInput<'_>) -> String {
        Self::compose_with_router(input, &HeuristicRouter).0
    }

    /// Compose the full system prompt and return the routing decision record.
    ///
    /// Use this when you need to emit the routing decision to the
    /// `TrajectoryCollector` for the neural feedback loop.
    pub fn compose_with_record(
        input: &ComposerInput<'_>,
        router: &dyn RoutingProvider,
    ) -> (String, RoutingDecisionRecord) {
        Self::compose_with_router(input, router)
    }

    /// Compose the full system prompt using a custom [`RoutingProvider`].
    ///
    /// Assembly pipeline:
    /// 1. Build [`RoutingContext`] from the input
    /// 2. Call `router.route()` to get section + tool group decisions
    /// 3. Assemble base sections into markdown
    /// 4. Inject FSM prompt fragments (from active protocol runs)
    /// 5. Render selected tool reference groups
    /// 6. Append dynamic context (project + continuity + enrichment), truncated
    /// 7. Join everything into the final prompt
    ///
    /// ## Integrating a custom router (e.g., DualTrackRouter)
    ///
    /// ```rust,ignore
    /// use crate::chat::routing::{RoutingProvider, RoutingContext, RoutingDecision};
    ///
    /// let dual_track = DualTrackRouter::new(model_weights);
    /// let prompt = FsmPromptComposer::compose_with_router(&input, &dual_track);
    /// ```
    pub fn compose_with_router(
        input: &ComposerInput<'_>,
        router: &dyn RoutingProvider,
    ) -> (String, RoutingDecisionRecord) {
        let (prompt, record, _report) = Self::compose_reported(input, router);
        (prompt, record)
    }

    /// Compose the prompt and account for what went into it.
    ///
    /// Same bytes as [`compose_with_router`] — the report is observation, not
    /// policy. It exists because the composer's output is one opaque string
    /// and the part it drops used to be discarded on the spot, which is how a
    /// truncation that beheaded whole sections of context survived for
    /// months. See [`CompositionReport`].
    pub fn compose_reported(
        input: &ComposerInput<'_>,
        router: &dyn RoutingProvider,
    ) -> (String, RoutingDecisionRecord, CompositionReport) {
        // ── Step 1: Build routing context ─────────────────────────────
        let fsm_tools: Vec<String> = input
            .protocol_runs
            .iter()
            .filter_map(|r| r.available_tools.as_ref())
            .flatten()
            .cloned()
            .collect();

        let routing_ctx = RoutingContext {
            scaffolding_level: input.scaffolding_level,
            has_active_plan: input.has_active_plan,
            has_active_protocol: !input.protocol_runs.is_empty(),
            task_count: input.task_count,
            is_multi_project: input.is_multi_project,
            external_tools_available: input.external_tools_available,
            fsm_available_tools: fsm_tools,
            user_message: input.user_message.to_string(),
            detected_intent: None, // Future: from enrichment hints
            message_embedding: input.message_embedding.cloned(),
        };

        // ── Step 2: Route — get section + tool group decisions ────────
        let decision = router.route(&routing_ctx);

        // ── Step 3: Assemble base sections ────────────────────────────
        // Merge section_hints from the router decision with optional input hints
        let merged_hints: Vec<(super::prompt_sections::PromptSectionId, f32)> = {
            let mut hints: Vec<_> = decision
                .section_hints
                .iter()
                .map(|h| (h.section_id, h.weight))
                .collect();
            // Input routing_hints override/extend router hints
            if let Some(input_hints) = input.routing_hints {
                for h in input_hints {
                    // Remove existing hint for same section, then add the override
                    hints.retain(|(id, _)| *id != h.section_id);
                    hints.push((h.section_id, h.weight));
                }
            }
            hints
        };

        let base_prompt = if !merged_hints.is_empty() {
            // Weighted assembly: truncate low-weight base sections when over budget.
            // Use 60% of the model's context window (in chars) as the base section budget.
            let model_budget = compute_dynamic_budget(input.model, 0) * 6;
            assemble_sections_weighted(&decision.sections, &merged_hints, model_budget)
        } else {
            assemble_sections(&decision.sections)
        };

        // ── Step 4: Inject FSM prompt fragments ───────────────────────
        let fsm_section = Self::build_fsm_section(input.protocol_runs);

        // ── Step 5: Render selected tool reference groups ─────────────
        let tool_ref = extract_tool_reference(TOOL_REFERENCE, &decision.tool_groups);

        // ── Step 6: Build dynamic context (truncated) ─────────────────
        // Compute adaptive budget based on model context window and current prompt size
        let base_prompt_len = base_prompt.len() + fsm_section.len() + tool_ref.len();
        let char_budget = compute_dynamic_budget(input.model, base_prompt_len);
        let (dynamic, source_reports, truncated_content) = Self::build_dynamic_section_reported(
            input.continuity_markdown,
            input.project_context_markdown,
            input.enrichment_markdown,
            char_budget,
        );

        // ── Step 7: Build trajectory record ────────────────────────────
        let routing_record = decision.to_trajectory_record(&routing_ctx);

        // ── Step 8: Assemble final prompt ─────────────────────────────
        let mut parts: Vec<&str> = Vec::with_capacity(4);
        parts.push(&base_prompt);

        let fsm_len = fsm_section.len();
        let fsm_owned;
        if !fsm_section.is_empty() {
            fsm_owned = fsm_section;
            parts.push(&fsm_owned);
        }

        let dynamic_len = dynamic.len();
        let dynamic_owned;
        if !dynamic.is_empty() {
            dynamic_owned = dynamic;
            parts.push(&dynamic_owned);
        }

        parts.push(&tool_ref);

        let prompt = parts.join("\n\n---\n\n");

        let report = CompositionReport {
            static_chars: base_prompt.len() + fsm_len + tool_ref.len(),
            dynamic_chars: dynamic_len,
            truncated_chars: source_reports.iter().map(|r| r.dropped_chars()).sum(),
            dynamic_budget_chars: char_budget,
            total_chars: prompt.len(),
            base_prompt_chars: base_prompt.len(),
            fsm_section_chars: fsm_len,
            tool_reference_chars: tool_ref.len(),
            sources: source_reports,
            truncated_content,
        };

        (prompt, routing_record, report)
    }

    /// Build the FSM context section from active protocol runs.
    ///
    /// Injects prompt_fragment, available_tools whitelist, and forbidden_actions
    /// from each active run's current state.
    fn build_fsm_section(runs: &[ProtocolRunStatus]) -> String {
        if runs.is_empty() {
            return String::new();
        }

        let mut lines = vec!["## Active Protocol Context".to_string()];

        for run in runs {
            lines.push(format!(
                "\n### Protocol: {} (state: `{}`)",
                run.protocol_name, run.current_state
            ));

            if !run.status_message.is_empty() {
                lines.push(format!("Status: {}", run.status_message));
            }

            // Inject the prompt fragment (contextual instructions for this state)
            if let Some(ref fragment) = run.prompt_fragment {
                lines.push(String::new());
                lines.push(fragment.clone());
            }

            // Render available tools whitelist
            if let Some(ref tools) = run.available_tools {
                if !tools.is_empty() {
                    lines.push(String::new());
                    lines.push(format!(
                        "**Allowed tools in state `{}`**: {}",
                        run.current_state,
                        tools.join(", ")
                    ));
                }
            }

            // Render forbidden actions as warnings
            if let Some(ref forbidden) = run.forbidden_actions {
                if !forbidden.is_empty() {
                    lines.push(String::new());
                    lines.push(format!(
                        "⚠️ **Forbidden in state `{}`**:",
                        run.current_state
                    ));
                    for action in forbidden {
                        lines.push(format!("- {}", action));
                    }
                }
            }
        }

        lines.join("\n")
    }

    /// Build the dynamic context section (continuity + project context + enrichment)
    /// with semantic truncation.
    ///
    /// Instead of blindly cutting at a character boundary, this function:
    /// 1. Parses the markdown into sections (## headers) with their list items (- lines)
    /// 2. Scores each item by importance markers ([Critical] > [High] > [Medium] > [Low])
    /// 3. When over budget, keeps only the top-N items per section (never drops entire sections)
    ///
    /// `char_budget` is the maximum character count for the dynamic section, computed
    /// by [`compute_dynamic_budget()`] based on the model's context window.
    #[cfg(test)]
    fn build_dynamic_section(
        continuity: &str,
        project_context: &str,
        enrichment: &str,
        char_budget: usize,
    ) -> String {
        Self::build_dynamic_section_reported(continuity, project_context, enrichment, char_budget).0
    }

    /// As [`build_dynamic_section`], but also returns per-source accounting
    /// and the exact text the budget removed.
    fn build_dynamic_section_reported(
        continuity: &str,
        project_context: &str,
        enrichment: &str,
        char_budget: usize,
    ) -> (String, Vec<SourceReport>, String) {
        let has_continuity = !continuity.is_empty();
        let has_enrichment = !enrichment.is_empty();
        let has_project = !project_context.is_empty();

        if !has_continuity && !has_enrichment && !has_project {
            return (String::new(), Vec::new(), String::new());
        }

        // Quick path: everything fits without truncation
        let total_len = continuity.len()
            + enrichment.len()
            + project_context.len()
            + if has_continuity { 2 } else { 0 }
            + if has_enrichment { 2 } else { 0 };

        if total_len <= char_budget {
            let mut parts = Vec::new();
            if has_enrichment {
                parts.push(enrichment);
            }
            if has_continuity {
                parts.push(continuity);
            }
            if has_project {
                parts.push(project_context);
            }
            let untouched = |source: DynamicSource, text: &str| SourceReport {
                source,
                original_chars: text.len(),
                kept_chars: text.len(),
                budget_chars: char_budget,
                dropped_sections: Vec::new(),
            };
            let mut reports = Vec::new();
            if has_enrichment {
                reports.push(untouched(DynamicSource::Enrichment, enrichment));
            }
            if has_continuity {
                reports.push(untouched(DynamicSource::Continuity, continuity));
            }
            if has_project {
                reports.push(untouched(DynamicSource::ProjectContext, project_context));
            }
            return (parts.join("\n\n"), reports, String::new());
        }

        // Over budget → sub-budget allocation: enrichment 40%, continuity 30%, project 30%
        // Enrichment gets the largest share because it's the most contextually relevant
        // (freshly searched knowledge, active skills, propagated notes).
        let active_count = [has_enrichment, has_continuity, has_project]
            .iter()
            .filter(|&&b| b)
            .count();

        let (enrichment_budget, continuity_budget, project_budget) = if active_count == 1 {
            // Single source gets the whole budget
            (char_budget, char_budget, char_budget)
        } else {
            (
                if has_enrichment {
                    (char_budget * 40) / 100
                } else {
                    0
                },
                if has_continuity {
                    (char_budget * 30) / 100
                } else {
                    0
                },
                if has_project {
                    (char_budget * 30) / 100
                } else {
                    0
                },
            )
        };

        // Truncate each source within its own budget
        let (trunc_enrichment, det_enrichment) = if has_enrichment {
            truncate_with_boost_reported(enrichment, enrichment_budget, ENRICHMENT_SCORE_BONUS)
        } else {
            (String::new(), TruncationDetail::default())
        };
        let (trunc_continuity, det_continuity) = if has_continuity {
            truncate_markdown_semantically_reported(continuity, continuity_budget)
        } else {
            (String::new(), TruncationDetail::default())
        };
        let (trunc_project, det_project) = if has_project {
            truncate_markdown_semantically_reported(project_context, project_budget)
        } else {
            (String::new(), TruncationDetail::default())
        };

        // Redistribute surplus: if a source used less than its budget,
        // give the remainder to the others (enrichment gets priority).
        // Only compute surplus for ACTIVE sources to avoid phantom surplus.
        let enrichment_surplus = if has_enrichment {
            enrichment_budget.saturating_sub(trunc_enrichment.len())
        } else {
            0
        };
        let continuity_surplus = if has_continuity {
            continuity_budget.saturating_sub(trunc_continuity.len())
        } else {
            0
        };
        let project_surplus = if has_project {
            project_budget.saturating_sub(trunc_project.len())
        } else {
            0
        };
        let total_surplus = enrichment_surplus + continuity_surplus + project_surplus;

        // Re-truncate with expanded budgets if there's meaningful surplus
        let (final_enrichment, final_continuity, final_project) = if total_surplus > 500 {
            let extra_for_enrichment = continuity_surplus + project_surplus;
            let extra_for_continuity = enrichment_surplus + project_surplus;
            let extra_for_project = enrichment_surplus + continuity_surplus;

            let fe = if has_enrichment && extra_for_enrichment > 0 {
                truncate_with_boost_reported(
                    enrichment,
                    enrichment_budget + extra_for_enrichment,
                    ENRICHMENT_SCORE_BONUS,
                )
            } else {
                (trunc_enrichment, det_enrichment)
            };
            let fc = if has_continuity && extra_for_continuity > 0 {
                truncate_markdown_semantically_reported(
                    continuity,
                    continuity_budget + extra_for_continuity,
                )
            } else {
                (trunc_continuity, det_continuity)
            };
            let fp = if has_project && extra_for_project > 0 {
                truncate_markdown_semantically_reported(
                    project_context,
                    project_budget + extra_for_project,
                )
            } else {
                (trunc_project, det_project)
            };
            (fe, fc, fp)
        } else {
            (
                (trunc_enrichment, det_enrichment),
                (trunc_continuity, det_continuity),
                (trunc_project, det_project),
            )
        };
        let ((final_enrichment, det_enrichment), (final_continuity, det_continuity), (final_project, det_project)) =
            (final_enrichment, final_continuity, final_project);

        // Assemble: enrichment first (most relevant), then continuity, then project
        let mut reports = Vec::new();
        let mut dropped_content = String::new();
        let record = |source: DynamicSource,
                          original: &str,
                          kept: &str,
                          budget: usize,
                          detail: TruncationDetail,
                          reports: &mut Vec<SourceReport>,
                          dropped_content: &mut String| {
            if original.is_empty() {
                return;
            }
            if !detail.dropped_content.is_empty() {
                dropped_content.push_str(&format!("# {}\n\n", source.as_str()));
                dropped_content.push_str(&detail.dropped_content);
                dropped_content.push('\n');
            }
            reports.push(SourceReport {
                source,
                original_chars: original.len(),
                kept_chars: kept.len(),
                budget_chars: budget,
                dropped_sections: detail.dropped_sections,
            });
        };
        record(
            DynamicSource::Enrichment,
            enrichment,
            &final_enrichment,
            enrichment_budget,
            det_enrichment,
            &mut reports,
            &mut dropped_content,
        );
        record(
            DynamicSource::Continuity,
            continuity,
            &final_continuity,
            continuity_budget,
            det_continuity,
            &mut reports,
            &mut dropped_content,
        );
        record(
            DynamicSource::ProjectContext,
            project_context,
            &final_project,
            project_budget,
            det_project,
            &mut reports,
            &mut dropped_content,
        );

        let mut parts = Vec::new();
        if !final_enrichment.is_empty() {
            parts.push(final_enrichment);
        }
        if !final_continuity.is_empty() {
            parts.push(final_continuity);
        }
        if !final_project.is_empty() {
            parts.push(final_project);
        }
        (parts.join("\n\n"), reports, dropped_content)
    }

    /// Estimate token count (~4 chars per token).
    #[allow(dead_code)]
    fn estimate_tokens(text: &str) -> usize {
        text.len().div_ceil(4)
    }

    /// Count total tool groups selected (for metrics/logging).
    #[allow(dead_code)]
    pub fn count_tool_groups(input: &ComposerInput<'_>) -> usize {
        Self::count_tool_groups_with_router(input, &HeuristicRouter)
    }

    /// Count tool groups using a custom router.
    #[allow(dead_code)]
    pub fn count_tool_groups_with_router(
        input: &ComposerInput<'_>,
        router: &dyn RoutingProvider,
    ) -> usize {
        let fsm_tools: Vec<String> = input
            .protocol_runs
            .iter()
            .filter_map(|r| r.available_tools.as_ref())
            .flatten()
            .cloned()
            .collect();

        let routing_ctx = RoutingContext {
            scaffolding_level: input.scaffolding_level,
            has_active_plan: input.has_active_plan,
            has_active_protocol: !input.protocol_runs.is_empty(),
            task_count: input.task_count,
            is_multi_project: input.is_multi_project,
            external_tools_available: input.external_tools_available,
            fsm_available_tools: fsm_tools,
            user_message: input.user_message.to_string(),
            detected_intent: None,
            message_embedding: input.message_embedding.cloned(),
        };
        router.route(&routing_ctx).tool_groups.len()
    }
}

/// Find the largest char boundary <= the given byte index.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// A parsed markdown section with header and scored items.
struct MdSection {
    /// The header line (e.g., "## Guidelines\n")
    header: String,
    /// Non-item lines (description text between header and items)
    preamble: Vec<String>,
    /// List items with importance scores (higher = more important)
    items: Vec<(u8, String)>,
}

/// Score bonus applied to enrichment items during truncation.
/// Enrichment content is freshly searched and contextually relevant,
/// so it should survive truncation over static project_context/continuity.
const ENRICHMENT_SCORE_BONUS: u8 = 30;

/// Score a markdown list item by importance markers.
///
/// Recognizes patterns like `[Critical]`, `[High]`, `[Medium]`, `[Low]`
/// and also priority numbers (priority 90 > priority 10).
fn score_item(line: &str) -> u8 {
    let lower = line.to_lowercase();
    if lower.contains("[critical]") || lower.contains("critical") {
        return 100;
    }
    if lower.contains("[high]") {
        return 80;
    }
    if lower.contains("[medium]") {
        return 50;
    }
    if lower.contains("[low]") {
        return 20;
    }
    // Check for priority numbers: "priority 90" → score 90
    if let Some(pos) = lower.find("priority ") {
        let rest = &lower[pos + 9..];
        if let Some(num_str) = rest.split(|c: char| !c.is_ascii_digit()).next() {
            if let Ok(n) = num_str.parse::<u8>() {
                return n;
            }
        }
    }
    // Default: medium importance
    40
}

/// Parse markdown into sections (## headers) with their list items.
fn parse_markdown_sections(text: &str) -> Vec<MdSection> {
    let mut sections: Vec<MdSection> = Vec::new();
    let mut current: Option<MdSection> = None;
    let mut in_items = false;

    for line in text.lines() {
        if line.starts_with("## ") {
            // Save previous section
            if let Some(sec) = current.take() {
                sections.push(sec);
            }
            current = Some(MdSection {
                header: line.to_string(),
                preamble: Vec::new(),
                items: Vec::new(),
            });
            in_items = false;
        } else if let Some(ref mut sec) = current {
            if line.starts_with("- ") || line.starts_with("* ") {
                in_items = true;
                let score = score_item(line);
                sec.items.push((score, line.to_string()));
            } else if in_items {
                // Anything after the first item, up to the next header or the
                // next item, belongs to that item.
                //
                // This used to require the line to be indented or blank;
                // anything else fell through to `preamble`. Note contents are
                // raw markdown pasted in as a single `- [Critical] <body>`
                // entry, so every unindented body line landed in the preamble
                // — and preambles are rendered unconditionally, outside the
                // item budget. A handful of long notes could therefore push
                // the document past the budget no matter how many items were
                // dropped, which is what forced the final byte cut on large
                // projects.
                if let Some(last) = sec.items.last_mut() {
                    last.1.push('\n');
                    last.1.push_str(line);
                }
            } else {
                sec.preamble.push(line.to_string());
            }
        } else {
            // Lines before any section header — treat as preamble of a virtual section
            if current.is_none() {
                current = Some(MdSection {
                    header: String::new(),
                    preamble: vec![line.to_string()],
                    items: Vec::new(),
                });
            }
        }
    }

    if let Some(sec) = current {
        sections.push(sec);
    }

    sections
}

/// Render sections back to markdown, dropping the lowest-scoring items until
/// the whole document fits `char_budget`.
///
/// # Why this is not a character truncation
///
/// This function used to do two passes. The first was score-aware: sort each
/// section's items by score, then binary-search a uniform `best_n` items per
/// section, guaranteeing at least one item per section. The second undid it:
///
/// ```ignore
/// if output.len() > char_budget {
///     output.truncate(floor_char_boundary(&output, char_budget));
///     output.push_str("\n\n[... context truncated]");
/// }
/// ```
///
/// The guarantee was expressed in SCORE; the cut was applied by POSITION. On
/// any project whose context did not fit even at one item per section, the
/// document was simply beheaded at a byte offset — and since
/// `context_to_markdown` emits `Guidelines, Gotchas, Global Guidelines,
/// Global Gotchas, Milestones, Releases, GDS` in that order, the global
/// sections were always the first to go. Measured: on `website` (1 active
/// note) a `[Critical]` global guideline was quoted back verbatim by the
/// session; on `core` (309 active notes) the same note was invisible and the
/// document ended just after `## Gotchas`.
///
/// The removal is now global and score-ordered: the lowest-scoring item still
/// present is dropped, whichever section it belongs to, until the budget is
/// met. A `[Critical]` item at the bottom of the document outlives a `[Low]`
/// item at the top. Section headers survive even when all their items are
/// dropped, so the reader can still see that a section existed and how much
/// of it was withheld.
/// Also returns what it removed.
///
/// The detail is built by the truncation itself rather than reconstructed
/// afterwards, so the numbers reported cannot drift from the bytes written —
/// the same discipline `section_size` follows against `render_section`.
fn render_sections_budgeted_reported(
    sections: &mut [MdSection],
    char_budget: usize,
) -> (String, TruncationDetail) {
    // Within a section, the lowest-scoring items are the tail.
    for sec in sections.iter_mut() {
        sec.items.sort_by(|a, b| b.0.cmp(&a.0));
    }

    let mut kept: Vec<usize> = sections.iter().map(|s| s.items.len()).collect();
    let mut sizes: Vec<usize> = sections
        .iter()
        .enumerate()
        .map(|(i, sec)| section_size(sec, kept[i]))
        .collect();
    let mut total: usize = sizes.iter().sum();

    // Drop items one at a time, always the globally lowest-scoring item still
    // kept. On a tie, take it from the section that still has the most items,
    // so the last survivor of a small section outlives the tail of a big one
    // and breadth degrades gracefully.
    while total > char_budget {
        let victim = sections
            .iter()
            .enumerate()
            .filter(|(i, _)| kept[*i] > 0)
            .min_by_key(|(i, sec)| (sec.items[kept[*i] - 1].0, std::cmp::Reverse(kept[*i])))
            .map(|(i, _)| i);

        let Some(i) = victim else {
            // Every item is gone and the headers alone still overflow.
            break;
        };

        kept[i] -= 1;
        let new_size = section_size(&sections[i], kept[i]);
        total = total + new_size - sizes[i];
        sizes[i] = new_size;
    }

    let mut output = sections
        .iter()
        .enumerate()
        .map(|(i, sec)| render_section(sec, kept[i]))
        .collect::<Vec<_>>()
        .join("\n");

    // Degenerate case: headers and preambles alone exceed the budget. Shed
    // preambles (descriptive prose, no scored content) before resorting to a
    // byte cut, and say so explicitly rather than ending mid-sentence.
    if output.len() > char_budget {
        output = sections
            .iter()
            .enumerate()
            .map(|(i, sec)| {
                let stripped = MdSection {
                    header: sec.header.clone(),
                    preamble: Vec::new(),
                    items: sec.items.clone(),
                };
                render_section(&stripped, kept[i])
            })
            .collect::<Vec<_>>()
            .join("\n");
    }

    if output.len() > char_budget {
        const NOTICE: &str = "\n\n[... context truncated: section headers alone exceed the budget]";
        let room = char_budget.saturating_sub(NOTICE.len());
        let boundary = floor_char_boundary(&output, room);
        output.truncate(boundary);
        output.push_str(NOTICE);
    }

    // Report what was removed, instead of dropping it on the floor. The
    // question this answers — "why did this session not see that guideline?"
    // — is almost always asked after the session has ended, so the content
    // has to survive the call, not just a log line.
    let mut detail = TruncationDetail::default();
    for (i, sec) in sections.iter().enumerate() {
        let dropped = sec.items.len().saturating_sub(kept[i]);
        if dropped == 0 {
            continue;
        }
        let chars_dropped: usize = sec
            .items
            .iter()
            .skip(kept[i])
            .map(|(_, l)| l.len() + 1)
            .sum();
        detail.dropped_sections.push(DroppedSection {
            section: sec.header.clone(),
            items_kept: kept[i],
            items_dropped: dropped,
            chars_dropped,
        });
        if !sec.header.is_empty() {
            detail.dropped_content.push_str(&sec.header);
            detail.dropped_content.push('\n');
        }
        for (_, line) in sec.items.iter().skip(kept[i]) {
            detail.dropped_content.push_str(line);
            detail.dropped_content.push('\n');
        }
        detail.dropped_content.push('\n');
    }

    (output, detail)
}

/// Rendered size of a section when only its first `kept` items are shown.
///
/// Must stay in step with [`render_section`], which is what actually writes
/// the bytes: a size estimate that drifts from the renderer is how a budget
/// silently stops being a budget.
fn section_size(sec: &MdSection, kept: usize) -> usize {
    let mut size = 0usize;
    if !sec.header.is_empty() {
        size += sec.header.len() + 1;
    }
    size += sec.preamble.iter().map(|l| l.len() + 1).sum::<usize>();
    size += sec
        .items
        .iter()
        .take(kept)
        .map(|(_, l)| l.len() + 1)
        .sum::<usize>();
    let omitted = sec.items.len().saturating_sub(kept);
    if omitted > 0 {
        size += format!("  [{} more items omitted]", omitted).len() + 1;
    }
    size
}

/// Render a single section with at most `max_items` items (sorted by score descending).
fn render_section(sec: &MdSection, max_items: usize) -> String {
    let mut lines = Vec::new();
    if !sec.header.is_empty() {
        lines.push(sec.header.clone());
    }
    for p in &sec.preamble {
        lines.push(p.clone());
    }
    let total_items = sec.items.len();
    let kept = total_items.min(max_items);
    for (_, item_line) in sec.items.iter().take(kept) {
        lines.push(item_line.clone());
    }
    let omitted = total_items.saturating_sub(kept);
    if omitted > 0 {
        lines.push(format!("  [{} more items omitted]", omitted));
    }
    lines.join("\n")
}

/// Semantically truncate markdown with an additive score boost on all items.
///
/// Same as `truncate_markdown_semantically` but applies `score_boost` to every item's
/// importance score before truncation. Used for enrichment content which should
/// survive truncation over lower-value static content.
#[cfg(test)]
fn truncate_with_boost(text: &str, char_budget: usize, score_boost: u8) -> String {
    truncate_with_boost_reported(text, char_budget, score_boost).0
}

fn truncate_with_boost_reported(
    text: &str,
    char_budget: usize,
    score_boost: u8,
) -> (String, TruncationDetail) {
    if text.len() <= char_budget {
        return (text.to_string(), TruncationDetail::default());
    }
    let mut sections = parse_markdown_sections(text);
    // Apply score boost to all items
    for sec in &mut sections {
        for item in &mut sec.items {
            item.0 = item.0.saturating_add(score_boost);
        }
    }
    render_sections_budgeted_reported(&mut sections, char_budget)
}

/// Semantically truncate markdown by parsing sections and keeping top-N items by importance.
///
/// Unlike blind character truncation, this:
/// - Never drops entire sections (headers are always kept)
/// - Prioritizes items by importance markers ([Critical] > [High] > [Medium] > [Low])
/// - Uniformly reduces items per section to fit the budget
#[cfg(test)]
fn truncate_markdown_semantically(text: &str, char_budget: usize) -> String {
    truncate_markdown_semantically_reported(text, char_budget).0
}

fn truncate_markdown_semantically_reported(
    text: &str,
    char_budget: usize,
) -> (String, TruncationDetail) {
    if text.len() <= char_budget {
        return (text.to_string(), TruncationDetail::default());
    }
    let mut sections = parse_markdown_sections(text);
    render_sections_budgeted_reported(&mut sections, char_budget)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {

    // ── T24: composition report ──────────────────────────────────────────

    fn over_budget_input<'a>(project_md: &'a str) -> ComposerInput<'a> {
        ComposerInput {
            model: "claude-sonnet-4",
            user_message: "what are the project guidelines?",
            project_context_markdown: project_md,
            ..Default::default()
        }
    }

    /// The report is observation, not policy: the prompt must come out byte
    /// for byte the same whether or not anyone asks for the accounting.
    #[test]
    fn report_does_not_change_the_prompt() {
        let long = "x".repeat(400);
        let mut md = String::new();
        for i in 0..80 {
            md.push_str(&format!("## Section {i}\n- [Low] {long}\n\n"));
        }
        md.push_str("## Global Guidelines\n- [Critical] GLOBAL-TEST-ALPHA-7749\n");
        let input = over_budget_input(&md);

        let plain = FsmPromptComposer::compose(&input);
        let (reported, _record, _report) =
            FsmPromptComposer::compose_reported(&input, &HeuristicRouter);

        assert_eq!(plain, reported, "the report changed the prompt bytes");
    }

    /// Over budget: the cut must be named, counted, and recoverable.
    #[test]
    fn report_names_what_was_cut_and_keeps_it() {
        let long = "x".repeat(400);
        let mut md = String::new();
        for i in 0..200 {
            md.push_str(&format!("## Section {i}\n- [Low] {long} marker-{i}\n\n"));
        }
        let input = over_budget_input(&md);

        let (_prompt, _record, report) =
            FsmPromptComposer::compose_reported(&input, &HeuristicRouter);

        assert!(report.is_truncated(), "an 80 KB context reported no loss");
        assert!(report.truncated_chars > 0);
        assert!(
            report.static_chars > 0,
            "the static half was not accounted for"
        );
        assert!(
            report.dynamic_chars <= report.dynamic_budget_chars,
            "dynamic context {} exceeds its own budget {}",
            report.dynamic_chars,
            report.dynamic_budget_chars
        );

        let dropped = report.all_dropped_sections();
        assert!(!dropped.is_empty(), "no section was named as losing items");
        assert!(
            dropped.iter().any(|d| d.section.starts_with("## Section")),
            "dropped sections carry no header: {dropped:?}"
        );

        // The content itself must survive, not just a count of it.
        assert!(
            !report.truncated_content.is_empty(),
            "the cut text was discarded — the report answers 'how much', not 'what'"
        );
        assert!(report.truncated_content.contains("marker-"));
    }

    /// Within budget: nothing is reported as lost, and no phantom sections.
    #[test]
    fn report_is_quiet_when_everything_fits() {
        let md = "## Guidelines\n- [Critical] keep it small\n";
        let input = over_budget_input(md);

        let (_prompt, _record, report) =
            FsmPromptComposer::compose_reported(&input, &HeuristicRouter);

        assert_eq!(report.truncated_chars, 0);
        assert!(!report.is_truncated());
        assert!(report.truncated_content.is_empty());
        assert!(report.all_dropped_sections().is_empty());
        assert_eq!(report.dynamic_chars, md.len());
    }

    /// T22 — a `[Critical]` item at the BOTTOM of an over-budget document
    /// must survive, and low-scoring items above it must be the ones to go.
    ///
    /// The budget here is tight enough that even ONE item per section does
    /// not fit — which is precisely when the old code fell through to its
    /// byte-offset cut. `context_to_markdown` emits `Guidelines, Gotchas,
    /// Global Guidelines, Global Gotchas, Milestones, Releases, GDS` in that
    /// order, so the global sections sat at the bottom and the cut always
    /// took them first, however critical they were. Measured on the real
    /// system: `website` (1 active note) quoted the critical global guideline
    /// back verbatim; `core` (309 active notes) never saw it, its prompt
    /// ending just after `## Gotchas`.
    #[test]
    fn critical_item_at_the_bottom_survives_truncation() {
        let long = "x".repeat(300);
        let mut md = String::new();
        for i in 0..8 {
            md.push_str(&format!("## Project Section {i}\n- [Low] {long}\n\n"));
        }
        md.push_str("## Global Guidelines\n- [Critical] GLOBAL-TEST-ALPHA-7749 always warm up\n");

        // Even one item per section is ~2.6 KB, so no uniform per-section cap
        // can fit this: the old code reached its final byte truncation here.
        let budget = 1_000;
        let result = truncate_markdown_semantically(&md, budget);

        assert!(
            result.len() <= budget,
            "budget blown: {} chars",
            result.len()
        );
        assert!(
            result.contains("GLOBAL-TEST-ALPHA-7749"),
            "the critical global guideline was dropped — position beat score again; rendered:\n{result}"
        );
        assert!(
            result.contains("## Global Guidelines"),
            "the section header itself disappeared; rendered:\n{result}"
        );
        assert!(
            !result.contains("[... context truncated"),
            "fell back to a byte cut instead of dropping items by score"
        );
    }

    /// Every section must stay visible as a header even when the budget is so
    /// tight that none of its items fit — a reader has to be able to tell
    /// "this section was emptied" from "this section did not exist".
    #[test]
    fn section_headers_survive_when_all_items_are_dropped() {
        let filler: String = (0..30)
            .map(|i| format!("- [Medium] item {i} with enough text to matter for the budget\n"))
            .collect();
        let md = format!("## Guidelines\n{filler}\n## Global Guidelines\n{filler}");

        let result = truncate_markdown_semantically(&md, 260);

        assert!(result.len() <= 260, "budget blown: {} chars", result.len());
        assert!(result.contains("## Guidelines"));
        assert!(result.contains("## Global Guidelines"));
        assert!(result.contains("more items omitted"));
    }

    /// A note body is pasted in as a single `- [Critical] <body>` entry whose
    /// continuation lines are not indented. Those lines must be attached to
    /// the item — and therefore droppable — not parked in the preamble, which
    /// is rendered outside the budget.
    #[test]
    fn unindented_note_body_counts_against_the_budget() {
        let body: String = (0..60)
            .map(|i| format!("line {i} of a long note body pasted straight into the prompt\n"))
            .collect();
        let md = format!("## Guidelines\n- [Low] a note\n{body}\n## Global Guidelines\n- [Critical] KEEP-ME\n");

        let result = truncate_markdown_semantically(&md, 600);

        assert!(result.len() <= 600, "budget blown: {} chars", result.len());
        assert!(
            result.contains("KEEP-ME"),
            "an unbounded preamble crowded out the critical item; rendered:\n{result}"
        );
    }
    use super::*;

    #[test]
    fn test_compose_no_project_no_fsm() {
        let input = ComposerInput {
            scaffolding_level: 0,
            has_active_plan: true,
            task_count: 10,
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        // Should contain the identity section
        assert!(
            prompt.contains("# Development Agent"),
            "Should have identity"
        );
        assert!(
            prompt.contains("MCP Mega-Tools Reference"),
            "Should have tool reference"
        );
        assert!(
            !prompt.contains("Active Protocol Context"),
            "No FSM section"
        );
    }

    #[test]
    fn test_compose_with_fsm_fragment() {
        let runs = vec![ProtocolRunStatus {
            protocol_name: "code-review".to_string(),
            current_state: "analyzing".to_string(),
            progress: 25,
            status_message: "Analyzing changes".to_string(),
            prompt_fragment: Some("Focus on test coverage and error handling.".to_string()),
            available_tools: Some(vec!["code".to_string(), "note".to_string()]),
            forbidden_actions: Some(vec!["Do NOT commit without review approval".to_string()]),
        }];
        let input = ComposerInput {
            protocol_runs: &runs,
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        assert!(
            prompt.contains("Active Protocol Context"),
            "Should have FSM section"
        );
        assert!(prompt.contains("code-review"), "Should name the protocol");
        assert!(prompt.contains("analyzing"), "Should name the state");
        assert!(
            prompt.contains("Focus on test coverage"),
            "Should inject prompt fragment"
        );
        assert!(
            prompt.contains("Allowed tools"),
            "Should list available tools"
        );
        assert!(
            prompt.contains("Forbidden in state"),
            "Should list forbidden actions"
        );
    }

    #[test]
    fn test_compose_with_dynamic_context() {
        let input = ComposerInput {
            project_context_markdown: "## Project: my-app\n- 3 active plans\n- Rust codebase",
            continuity_markdown: "## Previous Session\n- Last worked on auth module",
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        assert!(
            prompt.contains("Previous Session"),
            "Should have continuity"
        );
        assert!(
            prompt.contains("Project: my-app"),
            "Should have project context"
        );
    }

    #[test]
    fn test_compose_l4_smaller_than_l0() {
        let input_l0 = ComposerInput {
            scaffolding_level: 0,
            has_active_plan: true,
            task_count: 10,
            ..Default::default()
        };
        let input_l4 = ComposerInput {
            scaffolding_level: 4,
            has_active_plan: true,
            task_count: 10,
            ..Default::default()
        };
        let prompt_l0 = FsmPromptComposer::compose(&input_l0);
        let prompt_l4 = FsmPromptComposer::compose(&input_l4);

        assert!(
            prompt_l4.len() < prompt_l0.len(),
            "L4 ({} chars) should be smaller than L0 ({} chars)",
            prompt_l4.len(),
            prompt_l0.len()
        );
    }

    #[test]
    fn test_compose_dynamic_truncation() {
        // Build a large structured markdown context
        let mut big_context = String::from("## Items\n");
        for i in 0..500 {
            big_context.push_str(&format!(
                "- [Low] Item {} with padding text to make it much longer\n",
                i
            ));
        }
        let input = ComposerInput {
            project_context_markdown: &big_context,
            ..Default::default()
        };
        let prompt = FsmPromptComposer::compose(&input);

        // The dynamic section should be semantically truncated
        assert!(
            prompt.contains("more items omitted") || prompt.contains("context truncated"),
            "Should indicate truncation"
        );
        // Total prompt should be reasonable (base + tool_ref + truncated dynamic)
        assert!(
            prompt.len() < 100_000,
            "Total prompt should be bounded, got {}",
            prompt.len()
        );
    }

    #[test]
    fn test_fsm_section_empty_when_no_runs() {
        let section = FsmPromptComposer::build_fsm_section(&[]);
        assert!(section.is_empty(), "No runs → empty FSM section");
    }

    #[test]
    fn test_fsm_section_multiple_runs() {
        let runs = vec![
            ProtocolRunStatus {
                protocol_name: "deploy".to_string(),
                current_state: "staging".to_string(),
                progress: 50,
                status_message: "Deploying to staging".to_string(),
                prompt_fragment: Some("Check staging logs.".to_string()),
                available_tools: None,
                forbidden_actions: None,
            },
            ProtocolRunStatus {
                protocol_name: "review".to_string(),
                current_state: "pending".to_string(),
                progress: 0,
                status_message: "".to_string(),
                prompt_fragment: None,
                available_tools: None,
                forbidden_actions: None,
            },
        ];
        let section = FsmPromptComposer::build_fsm_section(&runs);
        assert!(section.contains("deploy"), "Should contain first protocol");
        assert!(section.contains("review"), "Should contain second protocol");
        assert!(
            section.contains("Check staging logs"),
            "Should contain fragment"
        );
    }

    #[test]
    fn test_tool_groups_fsm_filtering() {
        let runs = vec![ProtocolRunStatus {
            protocol_name: "test".to_string(),
            current_state: "run".to_string(),
            progress: 0,
            status_message: "".to_string(),
            prompt_fragment: None,
            available_tools: Some(vec!["code".to_string(), "note".to_string()]),
            forbidden_actions: None,
        }];
        let input = ComposerInput {
            protocol_runs: &runs,
            ..Default::default()
        };
        let groups = FsmPromptComposer::count_tool_groups(&input);
        // Core + Knowledge always, + CodeExploration (has "code")
        assert!(
            (2..=4).contains(&groups),
            "FSM whitelist should limit groups, got {}",
            groups
        );
    }

    // ── Semantic truncation tests ────────────────────────────────────

    #[test]
    fn test_score_item_importance_ordering() {
        assert!(score_item("- [Critical] Never do X") > score_item("- [High] Avoid Y"));
        assert!(score_item("- [High] Avoid Y") > score_item("- [Medium] Consider Z"));
        assert!(score_item("- [Medium] Consider Z") > score_item("- [Low] Maybe W"));
    }

    #[test]
    fn test_score_item_priority_number() {
        let score = score_item("- **Plan A** (InProgress, priority 90)");
        assert!(score == 90, "Should extract priority 90, got {}", score);
    }

    #[test]
    fn test_parse_markdown_sections() {
        let md =
            "## Guidelines\n- [Critical] Rule 1\n- [Low] Rule 2\n\n## Gotchas\n- Watch out for X\n";
        let sections = parse_markdown_sections(md);
        assert_eq!(sections.len(), 2, "Should parse 2 sections");
        assert_eq!(sections[0].items.len(), 2, "Guidelines should have 2 items");
        assert_eq!(sections[1].items.len(), 1, "Gotchas should have 1 item");
    }

    #[test]
    fn test_semantic_truncation_keeps_critical_drops_low() {
        // Build a dynamic context with 8 guidelines of varying importance
        let mut md = String::from("## Guidelines\n");
        md.push_str("- [Critical] Never expose API keys\n");
        md.push_str("- [Critical] Always validate input\n");
        md.push_str("- [Critical] Use parameterized queries\n");
        md.push_str("- [High] Log all errors\n");
        md.push_str("- [High] Use structured logging\n");
        md.push_str("- [Medium] Prefer composition over inheritance\n");
        md.push_str("- [Low] Use snake_case for variables\n");
        md.push_str("- [Low] Max line length 100\n");

        // Set a very tight budget that can only fit ~3 items
        let budget = md.len() / 2;
        let result = truncate_markdown_semantically(&md, budget);

        // Should keep the Critical items, drop Low items
        assert!(
            result.contains("Never expose API keys"),
            "Should keep Critical item"
        );
        assert!(
            result.contains("Always validate input"),
            "Should keep Critical item"
        );
        assert!(
            result.contains("## Guidelines"),
            "Should keep section header"
        );
        // Should indicate omitted items
        assert!(
            result.contains("more items omitted"),
            "Should show omission indicator"
        );
    }

    #[test]
    fn test_semantic_truncation_preserves_all_section_headers() {
        let md = "## Guidelines\n- [Low] Rule 1\n- [Low] Rule 2\n- [Low] Rule 3\n- [Low] Rule 4\n\n\
                   ## Gotchas\n- [Low] Gotcha 1\n- [Low] Gotcha 2\n- [Low] Gotcha 3\n- [Low] Gotcha 4\n\n\
                   ## Plans\n- [Low] Plan 1\n- [Low] Plan 2\n- [Low] Plan 3\n- [Low] Plan 4\n";

        // Budget that fits headers + 1 item each but not all items
        let budget = md.len() * 2 / 3;
        let result = truncate_markdown_semantically(md, budget);

        // All section headers must be preserved
        assert!(
            result.contains("## Guidelines"),
            "Must keep Guidelines header"
        );
        assert!(result.contains("## Gotchas"), "Must keep Gotchas header");
        assert!(result.contains("## Plans"), "Must keep Plans header");
        // Should have omitted some items
        assert!(
            result.contains("more items omitted"),
            "Should indicate omitted items"
        );
    }

    #[test]
    fn test_semantic_truncation_under_budget_unchanged() {
        let md = "## Small\n- Item 1\n- Item 2\n";
        let result = truncate_markdown_semantically(md, 10_000);
        // Should return as-is (no omission markers)
        assert!(
            !result.contains("omitted"),
            "Under budget should not truncate"
        );
        assert!(result.contains("Item 1"));
        assert!(result.contains("Item 2"));
    }

    #[test]
    fn test_dynamic_section_uses_semantic_truncation() {
        // Build a big project context that exceeds the default budget
        let budget = DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET;
        let mut big_context = String::new();
        big_context.push_str("## Guidelines\n");
        for i in 0..100 {
            let importance = if i < 10 {
                "Critical"
            } else if i < 30 {
                "High"
            } else {
                "Low"
            };
            big_context.push_str(&format!(
                "- [{}] Guideline number {} with some padding text to make it longer and exceed the budget easily\n",
                importance, i
            ));
        }
        big_context.push_str("\n## Gotchas\n");
        for i in 0..50 {
            big_context.push_str(&format!("- [Medium] Gotcha number {} with extra text\n", i));
        }

        assert!(
            big_context.len() > budget,
            "Test context should exceed budget"
        );

        let result = FsmPromptComposer::build_dynamic_section("", &big_context, "", budget);

        // Should be within budget (with some margin for omission markers)
        assert!(
            result.len() <= budget + 200,
            "Result ({} chars) should be near budget ({})",
            result.len(),
            budget
        );

        // Should keep section headers
        assert!(result.contains("## Guidelines"));
        assert!(result.contains("## Gotchas"));

        // Should indicate truncation
        assert!(result.contains("more items omitted"));
    }

    // ── Sub-budgeting and enrichment priority tests ──────────────────

    #[test]
    fn test_enrichment_gets_priority_in_truncation() {
        // Enrichment should survive truncation better than project_context
        // thanks to the +30 score boost and 40% budget allocation.
        let budget = 2000;

        // Build enrichment with critical items
        let mut enrichment = String::from("## Active Skills\n");
        for i in 0..20 {
            enrichment.push_str(&format!("- Skill {} with relevant context\n", i));
        }

        // Build project context with low items
        let mut project = String::from("## Project Info\n");
        for i in 0..20 {
            project.push_str(&format!("- [Low] Generic info item {}\n", i));
        }

        let result = FsmPromptComposer::build_dynamic_section("", &project, &enrichment, budget);

        // Enrichment should appear first (it's assembled first)
        let enrichment_pos = result.find("Active Skills");
        let project_pos = result.find("Project Info");
        assert!(
            enrichment_pos.is_some(),
            "Enrichment should be present in output"
        );
        assert!(
            project_pos.is_some(),
            "Project context should be present in output"
        );
        assert!(
            enrichment_pos.unwrap() < project_pos.unwrap(),
            "Enrichment should come before project context"
        );
    }

    #[test]
    fn test_sub_budget_reserves_enrichment_space() {
        // When both enrichment and project_context are large, enrichment
        // should get 40% of the budget (more than the 30% for project).
        let budget = 3000;

        let mut enrichment = String::from("## Enrichment\n");
        for i in 0..50 {
            enrichment.push_str(&format!(
                "- Enrichment item {} with important context data\n",
                i
            ));
        }

        let mut project = String::from("## Project\n");
        for i in 0..50 {
            project.push_str(&format!("- Project item {} with some padding text\n", i));
        }

        let result = FsmPromptComposer::build_dynamic_section("", &project, &enrichment, budget);

        // Count how much enrichment content survived vs project content
        let enrichment_lines = result
            .lines()
            .filter(|l| l.contains("Enrichment item"))
            .count();
        let project_lines = result
            .lines()
            .filter(|l| l.contains("Project item"))
            .count();

        assert!(
            enrichment_lines >= project_lines,
            "Enrichment ({} items) should get at least as many items as project ({} items) due to 40% vs 30% budget",
            enrichment_lines,
            project_lines
        );
    }

    #[test]
    fn test_enrichment_score_boost_preserves_items() {
        // The ENRICHMENT_SCORE_BONUS (+30) should make enrichment items
        // score higher than default (40) → they become 70, surviving over
        // low-priority items at 20.
        let boosted = truncate_with_boost(
            "## Skills\n- Skill A\n- Skill B\n- Skill C\n",
            200,
            ENRICHMENT_SCORE_BONUS,
        );
        // All items should survive since the text is under budget
        assert!(boosted.contains("Skill A"));
        assert!(boosted.contains("Skill B"));
        assert!(boosted.contains("Skill C"));
    }

    #[test]
    fn test_surplus_redistribution() {
        // If enrichment is small, its surplus should be given to project_context.
        let budget = 3000;
        let enrichment = "## Skills\n- One active skill\n";
        let mut project = String::from("## Project\n");
        for i in 0..60 {
            project.push_str(&format!("- Project item {} with padding text\n", i));
        }

        let result = FsmPromptComposer::build_dynamic_section("", &project, enrichment, budget);

        // Project should get more than its 30% base allocation thanks to
        // surplus from enrichment's unused 40%.
        assert!(
            result.len() > budget * 30 / 100,
            "Result ({} chars) should exceed the 30% base project allocation ({})",
            result.len(),
            budget * 30 / 100
        );
    }

    // ── compute_dynamic_budget tests ─────────────────────────────────

    #[test]
    fn test_budget_known_model_sonnet() {
        let budget = compute_dynamic_budget("claude-sonnet-4-20250514", 50_000);
        // 200K tokens × 4 chars = 800K chars; remaining = 750K; 15% = 112.5K → clamped to 40K
        assert_eq!(budget, 40_000);
    }

    #[test]
    fn test_budget_known_model_haiku() {
        let budget = compute_dynamic_budget("claude-haiku-3-5-20241022", 50_000);
        assert_eq!(budget, 40_000);
    }

    #[test]
    fn test_budget_known_model_opus() {
        let budget = compute_dynamic_budget("claude-opus-4-20250514", 50_000);
        assert_eq!(budget, 40_000);
    }

    #[test]
    fn test_budget_unknown_model_uses_default() {
        let budget = compute_dynamic_budget("gpt-4o", 50_000);
        assert_eq!(budget, DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET);
    }

    #[test]
    fn test_budget_empty_model_uses_default() {
        let budget = compute_dynamic_budget("", 50_000);
        assert_eq!(budget, DEFAULT_DYNAMIC_CONTEXT_CHAR_BUDGET);
    }

    #[test]
    fn test_budget_large_base_prompt_reduces_budget() {
        // Base prompt nearly fills the context window
        // 200K tokens × 4 = 800K chars; remaining = 800K - 780K = 20K; 15% = 3K → clamped to 5K
        let budget = compute_dynamic_budget("claude-sonnet-4-20250514", 780_000);
        assert_eq!(budget, 5_000);
    }

    #[test]
    fn test_budget_small_base_prompt_gets_max() {
        // Tiny base prompt → 15% of ~800K = ~120K → clamped to 40K
        let budget = compute_dynamic_budget("claude-sonnet-4-20250514", 1_000);
        assert_eq!(budget, 40_000);
    }

    #[test]
    fn test_budget_adapts_via_compose() {
        // Verify that compose() with a model name produces a different budget than without
        let input_with_model = ComposerInput {
            model: "claude-sonnet-4-20250514",
            project_context_markdown: "## Context\n- item",
            ..Default::default()
        };
        let input_no_model = ComposerInput {
            model: "",
            project_context_markdown: "## Context\n- item",
            ..Default::default()
        };
        // Both should produce valid prompts (just with different internal budgets)
        let prompt_with = FsmPromptComposer::compose(&input_with_model);
        let prompt_without = FsmPromptComposer::compose(&input_no_model);
        assert!(!prompt_with.is_empty());
        assert!(!prompt_without.is_empty());
    }
}

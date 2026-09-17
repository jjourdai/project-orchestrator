//! Asserts that every literal Cypher query in the source tree actually parses.
//!
//! Requires Neo4j to be running (see `integration_tests.rs` for the env vars).
//! Run with: cargo test --test cypher_parse_tests
//!
//! # Why this exists
//!
//! `get_avg_multi_signal_score` (src/neo4j/code.rs) shipped with a query that
//! Neo4j rejects outright — error 42I18, an implicitly grouped expression:
//!
//!     WITH f, max_pr, max_bt, out_deg + count(DISTINCT imp_in) AS total_degree
//!
//! It never once executed. Its caller, `get_code_health`, wraps the call in
//! `.unwrap_or(0.0)` — best-effort error handling intended for "GDS properties
//! not yet computed" — so the health report's `avg_impact_score` silently
//! returned 0.0 for every project, indefinitely, instead of surfacing a
//! failure. It was found only because someone profiled every query in that
//! handler by hand and noticed this one did no work at all.
//!
//! Roughly ten of that one handler's ~16 calls swallow errors the same way, so
//! a broken query is invisible by construction. `EXPLAIN` parses and plans a
//! query without running it and without needing parameter values, which makes
//! this whole class of defect cheap to catch here rather than in production.
//!
//! This test would have failed on the bug above the day it was written.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Cypher leading keywords used to decide whether a string literal is a query.
/// A literal must begin (ignoring whitespace/comments) with one of these.
const CYPHER_HEADS: &[&str] = &[
    "MATCH", "OPTIONAL MATCH", "MERGE", "CREATE", "WITH", "UNWIND", "RETURN", "CALL", "SHOW",
    "DETACH DELETE", "PROFILE", "EXPLAIN",
];

/// Queries known to be unparseable, deliberately NOT fixed, with the reason.
///
/// Keyed on a distinctive fragment of the query text rather than a line number,
/// so an entry identifies exactly one query and survives the file being edited
/// around it. A file-wide exemption would blind every other query in that file.
///
/// This list is asserted to be non-stale: if an exempt query starts parsing,
/// the test fails and tells you to remove the entry, so it cannot quietly rot
/// into a permanent exemption. Add to it only when a fix needs a product
/// decision — never to silence a defect you could fix.
const KNOWN_BROKEN: &[(&str, &str)] = &[(
    "duration.inMilliseconds",
    "backfill_often_follows (src/neo4j/mcp_federation.rs): queries label `ChatEventRecord`, which is a Rust \
     struct name, not a Neo4j label — the written label is `ChatEvent` (801k \
     nodes; zero ChatEventRecord). Also calls the nonexistent function \
     `duration.inMilliseconds`, and `created_at` is an ISO-8601 String rather \
     than a DateTime, so duration.between() could not apply to it either. \
     Even fully repaired it would match zero nodes today, and its \
     `MATCH (e1),(e2)` is a cartesian product over that 801k-node label — the \
     same unbounded shape that made get_health time out. Needs a redesign \
     (scope by session, drive off the seq ordering) plus a decision about \
     whether the feature is still wanted; fixing only the syntax would arm it.",
)];

/// A Cypher literal recovered from the source tree.
#[derive(Debug)]
struct FoundQuery {
    file: PathBuf,
    /// 1-indexed line where the literal starts.
    line: usize,
    text: String,
    /// True when the literal had `format!`-style placeholders substituted.
    was_templated: bool,
}

fn main_src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Recursively collect .rs files under `dir`.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Extract the contents of every `r#"..."#` raw string literal, with the
/// 1-indexed line number on which each begins.
fn raw_string_literals(src: &str) -> Vec<(usize, String)> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    // Precompute line starts so we can map byte offset -> line cheaply.
    let mut line_of = vec![1usize; bytes.len() + 1];
    {
        let mut line = 1usize;
        for (idx, b) in bytes.iter().enumerate() {
            line_of[idx] = line;
            if *b == b'\n' {
                line += 1;
            }
        }
        line_of[bytes.len()] = line;
    }

    while i + 2 < bytes.len() {
        if bytes[i] == b'r' {
            // Count the run of '#' after 'r'.
            let mut hashes = 0usize;
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if hashes > 0 && j < bytes.len() && bytes[j] == b'"' {
                let content_start = j + 1;
                // Closing delimiter is '"' followed by the same number of '#'.
                let mut k = content_start;
                let close = loop {
                    if k >= bytes.len() {
                        break None;
                    }
                    if bytes[k] == b'"' {
                        let mut h = 0usize;
                        let mut m = k + 1;
                        while m < bytes.len() && bytes[m] == b'#' && h < hashes {
                            h += 1;
                            m += 1;
                        }
                        if h == hashes {
                            break Some((k, m));
                        }
                    }
                    k += 1;
                };
                if let Some((content_end, after)) = close {
                    let text = &src[content_start..content_end];
                    out.push((line_of[content_start], text.to_string()));
                    i = after;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// Strip leading whitespace and `//` comment lines, then test whether the
/// literal begins with a Cypher clause keyword.
fn looks_like_cypher(text: &str) -> bool {
    let head = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("//"))
        .unwrap_or("");
    let upper = head.to_ascii_uppercase();
    CYPHER_HEADS.iter().any(|kw| upper.starts_with(kw))
}

/// Resolve `format!` placeholders so the query can be planned.
///
/// Raw strings passed to `format!` escape literal braces by doubling them, so
/// `{{id: $id}}` is a Cypher map and `{}` / `{depth}` are interpolation slots.
/// Returns `None` when a placeholder cannot be substituted safely — those are
/// reported as unchecked rather than guessed at, so this test never produces a
/// false failure from a bad guess.
fn resolve_placeholders(text: &str) -> Option<(String, bool)> {
    // Doubled braces are literal; hide them while we inspect the rest.
    const L: char = '\u{1}';
    const R: char = '\u{2}';
    let hidden = text.replace("{{", &L.to_string()).replace("}}", &R.to_string());

    let mut out = String::with_capacity(hidden.len());
    let mut rest = hidden.as_str();
    let mut templated = false;

    while let Some(open) = rest.find('{') {
        let Some(close_rel) = rest[open..].find('}') else {
            return None; // unbalanced; don't guess
        };
        let close = open + close_rel;
        let inner = &rest[open + 1..close];
        out.push_str(&rest[..open]);

        // A Cypher map literal in a NON-format string also reaches here (it has
        // no doubled braces). Those contain ':' or '$'; leave them intact.
        if inner.contains(':') || inner.contains('$') {
            out.push('{');
            out.push_str(inner);
            out.push('}');
            rest = &rest[close + 1..];
            continue;
        }

        // Otherwise it is a format slot: `{}` or `{ident}`. Substitute only in
        // the shapes we can be certain about.
        let before = out.as_str();
        let sub = if before.ends_with("..") {
            // Variable-length depth bound, e.g. `[:CALLS*1..{}]`.
            "3"
        } else if before.ends_with("[:") || before.ends_with('|') {
            // Relationship type slot, e.g. `[:{}]`.
            "CALLS"
        } else {
            return None; // unknown position — report as unchecked
        };
        out.push_str(sub);
        templated = true;
        rest = &rest[close + 1..];
    }
    out.push_str(rest);

    let restored = out.replace(L, "{").replace(R, "}");
    Some((restored, templated))
}

fn collect_queries() -> (Vec<FoundQuery>, Vec<(PathBuf, usize)>) {
    let mut files = Vec::new();
    rust_files(&main_src_dir(), &mut files);
    files.sort();

    let mut found = Vec::new();
    let mut unchecked = Vec::new();

    for file in files {
        // mock.rs holds fixture strings for the in-memory test double, not
        // queries that are ever sent to Neo4j.
        if file.file_name().is_some_and(|n| n == "mock.rs") {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&file) else {
            continue;
        };
        for (line, text) in raw_string_literals(&src) {
            if !looks_like_cypher(&text) {
                continue;
            }
            match resolve_placeholders(&text) {
                Some((resolved, was_templated)) => found.push(FoundQuery {
                    file: file.clone(),
                    line,
                    text: resolved,
                    was_templated,
                }),
                None => unchecked.push((file.clone(), line)),
            }
        }
    }
    (found, unchecked)
}

#[tokio::test]
async fn every_literal_cypher_query_parses() {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password =
        std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "orchestrator123".into());

    let graph = neo4rs::Graph::new(&uri, &user, &password)
        .await
        .unwrap_or_else(|e| {
            panic!(
                "cypher_parse_tests needs a reachable Neo4j at {uri}: {e}\n\
                 Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD."
            )
        });

    let (queries, unchecked) = collect_queries();

    assert!(
        queries.len() > 100,
        "only found {} Cypher literals — the extractor is probably broken, \
         which would make this test silently vacuous",
        queries.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let mut templated = 0usize;

    let mut known_broken_that_now_parse: Vec<String> = Vec::new();

    for q in &queries {
        let rel_path = q
            .file
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(&q.file)
            .display()
            .to_string();
        let exempt = KNOWN_BROKEN
            .iter()
            .any(|(fragment, _)| q.text.contains(fragment));

        // EXPLAIN parses and plans without executing, and does not require
        // parameter values to be supplied.
        let stmt = format!("EXPLAIN {}", q.text);
        let outcome = graph.execute(neo4rs::query(&stmt)).await;
        if exempt {
            if outcome.is_ok() {
                known_broken_that_now_parse.push(format!("{rel_path}:{}", q.line));
            }
            continue;
        }
        if let Err(e) = outcome {
            let msg = e.to_string();
            // A query naming a procedure this deployment lacks (APOC, GDS) is a
            // deployment fact, not a syntax defect.
            if msg.contains("no procedure") || msg.contains("There is no procedure") {
                continue;
            }
            let rel = rel_path.clone();
            failures.push(format!(
                "\n── {}:{}{}\n   {}\n   query:\n{}",
                rel,
                q.line,
                if q.was_templated {
                    " (placeholders substituted)"
                } else {
                    ""
                },
                msg.lines().next().unwrap_or(&msg),
                q.text
                    .lines()
                    .map(|l| format!("     {l}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        checked += 1;
        if q.was_templated {
            templated += 1;
        }
    }

    // Report coverage so a growing blind spot is visible rather than silent.
    let unchecked_files: BTreeSet<String> = unchecked
        .iter()
        .map(|(f, _)| {
            f.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                .unwrap_or(f)
                .display()
                .to_string()
        })
        .collect();
    println!(
        "cypher_parse_tests: planned {checked} queries ({templated} after placeholder \
         substitution); {} dynamic queries not checked across {} files",
        unchecked.len(),
        unchecked_files.len()
    );

    assert!(
        known_broken_that_now_parse.is_empty(),
        "these files are in KNOWN_BROKEN but now parse — remove them from the \
         list so it does not become a permanent exemption: {known_broken_that_now_parse:?}"
    );

    assert!(
        failures.is_empty(),
        "{} literal Cypher quer{} failed to parse. These can never succeed at \
         runtime; if the caller swallows errors, the failure is invisible in \
         production.{}",
        failures.len(),
        if failures.len() == 1 { "y" } else { "ies" },
        failures.join("")
    );
}

/// Guards the extractor itself. If these stop holding, the test above can go
/// quietly vacuous, which is worse than failing.
#[test]
fn extractor_handles_raw_strings_and_placeholders() {
    let src = r##"
        let a = query(r#"MATCH (n:File) RETURN n"#);
        let b = format!(r#"MATCH (f:Function {{id: $id}})-[:CALLS*1..{}]->(c) RETURN c"#, depth);
        let c = "not a raw string";
        let d = query(r#"// leading comment
            MERGE (p:Project {id: $pid}) RETURN p"#);
    "##;

    let lits = raw_string_literals(src);
    assert_eq!(lits.len(), 3, "expected 3 raw literals, got {}", lits.len());

    let cypher: Vec<_> = lits
        .iter()
        .filter(|(_, t)| looks_like_cypher(t))
        .collect();
    assert_eq!(cypher.len(), 3, "all three raw literals are Cypher");

    // Doubled braces become a real Cypher map; the depth slot becomes a number.
    let (resolved, templated) = resolve_placeholders(&lits[1].1).expect("should resolve");
    assert!(templated, "literal had a placeholder");
    assert!(
        resolved.contains("{id: $id}"),
        "doubled braces should unescape to a Cypher map, got: {resolved}"
    );
    assert!(
        resolved.contains("*1..3"),
        "depth slot should be substituted, got: {resolved}"
    );
    assert!(
        !resolved.contains('{') || !resolved.contains("{}"),
        "no unresolved slots should remain, got: {resolved}"
    );

    // A map literal in a non-format string must survive untouched.
    let (plain, was_templated) = resolve_placeholders(&lits[2].1).expect("should resolve");
    assert!(!was_templated, "no placeholders in a plain raw string");
    assert!(plain.contains("{id: $pid}"), "got: {plain}");
}

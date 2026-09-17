//! Svelte component extractor.
//!
//! A `.svelte` file is not one language: it is HTML-ish markup, one or two
//! `<script>` blocks, and a `<style>` block. Only the script blocks carry the
//! symbols the code graph cares about — functions, imports, types.
//!
//! WHY THERE IS NO SVELTE GRAMMAR HERE. In every tree-sitter Svelte grammar the
//! `<script>` body is a single opaque `raw_text` LEAF with no children; the
//! grammars ship an injection query precisely to hand that text to TypeScript.
//! Walking a Svelte AST would therefore yield zero functions and zero imports.
//! The grammar would buy nothing, while adding a dependency whose Svelte 5
//! coverage we cannot verify. So we locate the script spans by byte scan and
//! reuse the TypeScript grammar that is already here.
//!
//! WHY MASKING AND NOT SLICING. Every extractor computes `line_start` as
//! `node.start_position().row + 1`, relative to whatever string it was handed.
//! Parsing a sliced-out `<script>` body would offset every symbol by the position
//! of the opening tag, and nothing would fail loudly — Neo4j would simply carry
//! wrong line anchors forever. There is no line-offset parameter anywhere in the
//! extractor API to correct it afterwards. So instead of slicing we build a
//! SAME-LENGTH copy of the file with every byte outside the script bodies
//! replaced by a space (newlines preserved). Byte offsets, rows and columns then
//! match the original file exactly, and `typescript::extract` needs no changes.

use super::typescript;
use crate::parser::ParsedFile;
use anyhow::Result;

/// Byte ranges of every `<script>` body in the source.
///
/// Handles both `<script>` and `<script context="module">` / `<script module>`:
/// a component may carry two, and taking only the first would silently drop half
/// its symbols.
///
/// ON `</script>` INSIDE A STRING: the span ends at the first `</script`, even if
/// it sits inside a JS string literal. That is not a shortcut — it is what HTML
/// raw-text parsing does, and what the Svelte compiler does. `const s = '</script>'`
/// is rejected by `svelte.compile` with "Unterminated string constant", so such a
/// component cannot exist in a buildable project. Matching the compiler is the
/// correct behaviour; diverging from it would be the bug.
fn script_spans(source: &str) -> Vec<(usize, usize)> {
    let bytes = source.as_bytes();
    let mut spans = Vec::new();
    let mut cursor = 0usize;

    while cursor < source.len() {
        let Some(rel) = source[cursor..].find("<script") else {
            break;
        };
        let tag_start = cursor + rel;

        // An HTML comment before this point may enclose it. A commented-out
        // <script> is not code, and indexing it invents functions and imports
        // that do not exist in the built component.
        if let Some(comment_start) = source[..tag_start].rfind("<!--") {
            let closed = source[comment_start..tag_start].contains("-->");
            if !closed {
                // Inside a comment — resume after it ends, or stop if unterminated.
                match source[tag_start..].find("-->") {
                    Some(rel_end) => {
                        cursor = tag_start + rel_end + 3;
                        continue;
                    }
                    None => break,
                }
            }
        }

        // "<script" must be followed by something that ends the tag NAME,
        // otherwise this is `<scriptish` or similar. 7 == "<script".len()
        let after_name = tag_start + 7;
        match bytes.get(after_name) {
            Some(c) if c.is_ascii_whitespace() || *c == b'>' || *c == b'/' => {}
            _ => {
                cursor = after_name;
                continue;
            }
        }

        // Walk the opening tag honouring quotes, so an attribute value containing
        // '>' does not close it early. Svelte 5 really does this:
        //   <script lang="ts" generics="T extends Item<K>">
        let mut i = after_name;
        let mut quote: Option<u8> = None;
        let mut tag_end: Option<usize> = None;
        while i < bytes.len() {
            let b = bytes[i];
            match quote {
                Some(q) if b == q => quote = None,
                Some(_) => {}
                None if b == b'"' || b == b'\'' => quote = Some(b),
                None if b == b'>' => {
                    tag_end = Some(i);
                    break;
                }
                None => {}
            }
            i += 1;
        }
        let Some(tag_end) = tag_end else {
            break; // unterminated opening tag
        };

        let attrs = &source[tag_start..tag_end];
        let self_closing = attrs.trim_end().ends_with('/');
        let has_src = attrs.contains("src=");
        let body_start = tag_end + 1;

        if self_closing {
            // `<script src="..." />` has no body and no closing tag. Resume just
            // after it — NOT after some later `</script`, which would swallow the
            // next real block whole.
            cursor = body_start;
            continue;
        }

        let Some(rel_end) = source[body_start..].find("</script") else {
            break; // unterminated block
        };
        let body_end = body_start + rel_end;

        if !has_src && body_end > body_start {
            spans.push((body_start, body_end));
        }
        cursor = body_end + 8; // past "</script"
    }

    spans
}

/// Build a same-length copy of `source` where only the script bodies survive.
///
/// Every other byte becomes a space, except newlines, which are kept so line
/// numbers are preserved. Multi-byte characters outside the spans are replaced
/// byte-for-byte with spaces, which keeps the length identical and is safe
/// because the replaced bytes never form part of the TypeScript being parsed.
fn mask_to_scripts(source: &str, spans: &[(usize, usize)]) -> String {
    let mut masked: Vec<u8> = source
        .as_bytes()
        .iter()
        .map(|b| if *b == b'\n' { b'\n' } else { b' ' })
        .collect();

    for (start, end) in spans {
        masked[*start..*end].copy_from_slice(&source.as_bytes()[*start..*end]);
    }

    // Every replaced byte is ASCII space or newline and every surviving range is a
    // char-boundary-aligned slice of the original, so this is valid UTF-8.
    String::from_utf8(masked).expect("masking preserves UTF-8 validity")
}

/// Mask a Svelte component down to just its `<script>` bodies.
///
/// Returns a string of EXACTLY the same length as `source`, with every byte
/// outside a script body replaced by a space (newlines preserved). The result is
/// valid TypeScript-ish input whose byte offsets, rows and columns all still refer
/// to the original `.svelte` file, so any extractor run over it records positions
/// that point at the real component.
///
/// This runs BEFORE parsing, in `CodeParser::parse_file`, so a component is parsed
/// once. Parsing the raw markup first and re-parsing the mask afterwards cost a
/// wasted full-file parse per component — around 99ms on a 219KB component, about
/// half the corpus parse time.
pub fn mask_scripts(source: &str) -> String {
    let spans = script_spans(source);
    mask_to_scripts(source, &spans)
}

/// Extract symbols from a masked Svelte component.
///
/// `source` is already the masked buffer (see [`mask_scripts`]), so this is a
/// straight delegation to the TypeScript extractor — everything that is not a
/// script body has been blanked out and parses to nothing.
pub fn extract(
    root: &tree_sitter::Node,
    source: &str,
    file_path: &str,
    parsed: &mut ParsedFile,
) -> Result<()> {
    typescript::extract(root, source, file_path, parsed)
}

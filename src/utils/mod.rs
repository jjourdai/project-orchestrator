//! Shared utility helpers.

pub mod file_path_extractor;
pub mod paths;

/// Stable equivalent of the nightly-only `str::floor_char_boundary`.
/// Returns the largest byte index `<= index` that is a valid UTF-8 char boundary.
///
/// Useful for safely truncating a `&str` at a maximum byte length without
/// panicking on multi-byte characters.
pub(crate) fn floor_char_boundary(s: &str, index: usize) -> usize {
    let index = index.min(s.len());
    (0..=index)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

/// Truncate `s` to at most `max_bytes`, never splitting a UTF-8 character.
///
/// Returns the whole string when it already fits. Unlike `&s[..max_bytes]`,
/// this never panics: `str::len()` counts BYTES while slicing requires a CHAR
/// boundary, so a multi-byte character straddling `max_bytes` would abort the
/// process. Always prefer this over a raw byte slice on text that can contain
/// non-ASCII (human prose, compiler output, file paths, error messages).
pub(crate) fn truncate_chars(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        s
    } else {
        &s[..floor_char_boundary(s, max_bytes)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A multi-byte char straddling the cut point is the exact shape that
    /// panicked `&s[..N]` in production (RFC previews, rustc stderr).
    #[test]
    fn test_truncate_chars_multibyte_on_boundary() {
        // 199 ASCII bytes then 'é' (2 bytes): byte 200 falls INSIDE the 'é'.
        let s = format!("{}é trailing", "a".repeat(199));
        assert!(!s.is_char_boundary(200), "fixture must straddle byte 200");
        let out = truncate_chars(&s, 200);
        assert_eq!(out.len(), 199);
        assert!(s.starts_with(out));
        // Negative control: the pattern this replaces would abort the process.
        assert!(std::panic::catch_unwind(|| &s[..200]).is_err());
    }

    #[test]
    fn test_truncate_chars_shorter_than_limit_is_identity() {
        assert_eq!(truncate_chars("héllo", 4096), "héllo");
    }

    #[test]
    fn test_truncate_chars_exact_fit_and_zero() {
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("é", 1), "");
    }

    /// rustc emits box-drawing characters in stderr; truncating its output at a
    /// fixed byte count is what makes the PlanRunner verifier panic on a long
    /// compile failure — i.e. exactly when it is needed.
    #[test]
    fn test_truncate_chars_on_rustc_like_stderr() {
        let line = "  │ error[E0308]: mismatched types ─→ src/lib.rs:1:1\n";
        let stderr = line.repeat(200);
        assert!(stderr.len() > 2000);
        let out = truncate_chars(&stderr, 2000);
        assert!(out.len() <= 2000);
        assert!(std::str::from_utf8(out.as_bytes()).is_ok());
    }

    #[test]
    fn test_floor_char_boundary_ascii() {
        let s = "hello world";
        assert_eq!(floor_char_boundary(s, 5), 5);
    }

    #[test]
    fn test_floor_char_boundary_multibyte() {
        // "é" is 2 bytes (0xC3 0xA9); boundary at byte 1 is invalid
        let s = "héllo";
        assert_eq!(floor_char_boundary(s, 2), 1); // step back to valid boundary
        assert_eq!(floor_char_boundary(s, 3), 3);
    }

    #[test]
    fn test_floor_char_boundary_beyond_len() {
        let s = "hi";
        assert_eq!(floor_char_boundary(s, 100), 2);
    }
}

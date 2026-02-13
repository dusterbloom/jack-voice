// Supertonic TTS - Unicode Text Processor
// Converts text to token IDs using unicode_indexer.json mapping

use regex::Regex;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

use crate::TtsError;

/// Unicode indexer loaded from JSON file
/// Format: Array where index = Unicode codepoint, value = vocabulary index (-1 = unmapped)
#[derive(Clone)]
pub struct UnicodeIndexer {
    /// Lookup table: index = codepoint, value = vocab index (-1 means unmapped)
    lookup: Vec<i64>,
    /// Unknown token ID (default: 0)
    unk_id: i64,
}

impl UnicodeIndexer {
    /// Load unicode indexer from a JSON file
    /// Format: Array of integers where index = Unicode codepoint, value = vocab ID (-1 = unmapped)
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TtsError> {
        let content =
            std::fs::read_to_string(path.as_ref()).map_err(|e| TtsError::IoError(e.to_string()))?;

        let lookup: Vec<i64> = serde_json::from_str(&content).map_err(|e| {
            TtsError::IoError(format!("Failed to parse unicode_indexer.json: {}", e))
        })?;

        let valid_count = lookup.iter().filter(|&&x| x >= 0).count();
        log::info!(
            "Loaded unicode indexer with {} entries ({} valid mappings)",
            lookup.len(),
            valid_count
        );

        Ok(Self {
            lookup,
            unk_id: 0, // Unknown token ID (space character typically)
        })
    }

    /// Convert a character to its vocabulary index
    pub fn char_to_index(&self, ch: char) -> i64 {
        let codepoint = ch as usize;
        if codepoint < self.lookup.len() {
            let idx = self.lookup[codepoint];
            if idx >= 0 {
                return idx;
            }
        }
        self.unk_id
    }

    /// Convert text to token IDs with full preprocessing matching official supertonic-2.
    /// The `lang` parameter specifies the language tag (e.g., "en", "ko", "es", "pt", "fr").
    /// Returns (text_ids, text_mask, seq_len).
    pub fn text_to_ids(&self, text: &str, lang: &str) -> (Vec<i64>, Vec<f32>, usize) {
        // Full preprocessing: normalization + emoji removal + punctuation + language tags
        let preprocessed = preprocess_text(text, lang);

        // Convert to token IDs
        let ids: Vec<i64> = preprocessed
            .chars()
            .map(|ch| self.char_to_index(ch))
            .collect();

        let seq_len = ids.len();

        // Create mask (all 1.0s) with shape [1, 1, seq_len]
        let mask: Vec<f32> = vec![1.0; seq_len];

        (ids, mask, seq_len)
    }
}

/// Normalize text for TTS input
fn normalize_text(text: &str) -> String {
    // Use NFKD normalization (decomposes characters)
    let text = text.nfkd().collect::<String>();

    // Replace various dashes with regular hyphen
    let text = text
        .replace('\u{2013}', "-") // en dash
        .replace('\u{2014}', "-") // em dash
        .replace('\u{2015}', "-"); // horizontal bar

    // Replace fancy quotes
    let text = text
        .replace('\u{2018}', "'") // left single quote
        .replace('\u{2019}', "'") // right single quote
        .replace('\u{201C}', "\"") // left double quote
        .replace('\u{201D}', "\""); // right double quote

    // Replace ellipsis
    let text = text.replace('\u{2026}', "...");

    // Replace other special characters
    let text = text
        .replace('\u{00A0}', " ") // non-breaking space
        .replace('\t', " "); // tab to space

    // Collapse multiple spaces
    let space_re = Regex::new(r"\s+").unwrap();
    let text = space_re.replace_all(&text, " ").to_string();

    // Trim
    text.trim().to_string()
}

/// Full text preprocessing matching the official supertonic-2 implementation.
/// Normalizes Unicode, removes emojis, fixes punctuation, ensures terminal punctuation,
/// and wraps with language tags (e.g., `<en>Hello world.</en>`).
fn preprocess_text(text: &str, lang: &str) -> String {
    // Step 1: NFKD normalize
    let mut text: String = text.nfkd().collect();

    // Step 2: Remove emojis (wide Unicode ranges)
    if let Ok(emoji_re) = Regex::new(
        r"[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}\x{1F700}-\x{1F77F}\x{1F780}-\x{1F7FF}\x{1F800}-\x{1F8FF}\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}\x{1F1E6}-\x{1F1FF}]+",
    ) {
        text = emoji_re.replace_all(&text, "").to_string();
    }

    // Step 3: Replace dashes and symbols
    text = text
        .replace('\u{2013}', "-") // en dash
        .replace('\u{2011}', "-") // non-breaking hyphen
        .replace('\u{2014}', "-") // em dash
        .replace('\u{2015}', "-") // horizontal bar
        .replace('_', " ")
        .replace('\u{201C}', "\"") // left double quote
        .replace('\u{201D}', "\"") // right double quote
        .replace('\u{2018}', "'") // left single quote
        .replace('\u{2019}', "'") // right single quote
        .replace('\u{00B4}', "'") // acute accent
        .replace('`', "'")
        .replace('[', " ")
        .replace(']', " ")
        .replace('|', " ")
        .replace('/', " ")
        .replace('#', " ")
        .replace('\u{2192}', " ") // right arrow
        .replace('\u{2190}', " "); // left arrow

    // Step 4: Remove special symbols
    for sym in &['\u{2665}', '\u{2606}', '\u{2661}', '\u{00A9}', '\\'] {
        text = text.replace(*sym, "");
    }

    // Step 5: Replace known expressions
    text = text.replace('@', " at ");
    text = text.replace("e.g.,", "for example, ");
    text = text.replace("i.e.,", "that is, ");

    // Step 6: Fix spacing around punctuation
    if let Ok(re) = Regex::new(r" ([,\.!?;:'])") {
        text = re.replace_all(&text, "$1").to_string();
    }

    // Step 7: Remove duplicate quotes
    while text.contains("\"\"") {
        text = text.replace("\"\"", "\"");
    }
    while text.contains("''") {
        text = text.replace("''", "'");
    }

    // Step 8: Replace non-breaking space, collapse multiple spaces
    text = text.replace('\u{00A0}', " ").replace('\t', " ");
    text = text.replace('\u{2026}', "..."); // ellipsis
    let space_re = Regex::new(r"\s+").unwrap();
    text = space_re.replace_all(&text, " ").to_string();
    text = text.trim().to_string();

    // Step 9: Ensure text ends with punctuation (official adds period if missing)
    if !text.is_empty() {
        let last_ch = text.chars().last().unwrap();
        if !matches!(
            last_ch,
            '.' | '!' | '?' | ';' | ':' | ',' | '\'' | '"' | ')' | ']' | '}'
        ) {
            text.push('.');
        }
    }

    // Step 10: Wrap with language tags (MUST be last - the v2 model expects these)
    format!("<{}>{}</{}>", lang, text, lang)
}

/// Split text into sentences for chunked processing
pub fn split_sentences(text: &str) -> Vec<String> {
    let normalized = normalize_text(text);

    // Split on sentence boundaries
    let sentence_re = Regex::new(r"[.!?]+\s*").unwrap();
    let sentences: Vec<String> = sentence_re
        .split(&normalized)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    if sentences.is_empty() {
        vec![normalized]
    } else {
        sentences
    }
}

/// Chunk text to prevent memory issues with long inputs
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let sentences = split_sentences(text);
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();

    for sentence in sentences {
        if current_chunk.len() + sentence.len() > max_chars && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = String::new();
        }

        if !current_chunk.is_empty() {
            current_chunk.push(' ');
        }
        current_chunk.push_str(&sentence);

        // Add back the period
        if !sentence.ends_with('.') && !sentence.ends_with('!') && !sentence.ends_with('?') {
            current_chunk.push('.');
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() {
        chunks.push(normalize_text(text));
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text() {
        let text = "Hello\u{2014}world";
        let normalized = normalize_text(text);
        assert_eq!(normalized, "Hello-world");
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello. How are you? I'm fine!");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_chunk_text() {
        let text = "This is a test. Another sentence. And one more.";
        let chunks = chunk_text(text, 30);
        assert!(!chunks.is_empty());
    }
}

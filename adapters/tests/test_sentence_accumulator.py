"""Tests for SentenceAccumulator and _strip_inline_markdown."""

from __future__ import annotations

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ADAPTERS_DIR = REPO_ROOT / "adapters"
SDK_DIR = REPO_ROOT / "sdk" / "python" / "jack-voice-sdk"

# SDK must be on path first so cli_voice can import jack_voice
if str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))
if str(ADAPTERS_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTERS_DIR))

from cli_voice import SentenceAccumulator, _find_sentence_boundary, _strip_inline_markdown


def collect_sentences(text_chunks: list[str]) -> list[str]:
    """Push text chunks through SentenceAccumulator and return collected sentences."""
    results: list[str] = []
    acc = SentenceAccumulator(on_sentence=results.append)
    for chunk in text_chunks:
        acc.push(chunk)
    acc.flush()
    return results


class TestSentenceSplitting:
    def test_single_sentence(self):
        result = collect_sentences(["Hello world."])
        assert result == ["Hello world."]

    def test_two_sentences_one_push(self):
        result = collect_sentences(["Hello world. How are you?"])
        assert result == ["Hello world.", "How are you?"]

    def test_split_across_push_boundary(self):
        result = collect_sentences(["Hello wor", "ld. Good", "bye."])
        assert result == ["Hello world.", "Goodbye."]

    def test_exclamation_and_question(self):
        result = collect_sentences(["Wow! Really? Yes."])
        assert result == ["Wow!", "Really?", "Yes."]

    def test_no_boundary_flushes_remainder(self):
        result = collect_sentences(["No period here"])
        assert result == ["No period here"]

    def test_empty_input(self):
        result = collect_sentences([""])
        assert result == []

    def test_whitespace_only(self):
        result = collect_sentences(["   \n  "])
        assert result == []


class TestCodeBlockSkipping:
    def test_code_block_skipped(self):
        result = collect_sentences([
            "Before code. ```python\nprint('hello')\n``` After code."
        ])
        assert "print" not in " ".join(result)
        assert "Before code." in result
        assert "After code." in result

    def test_multiline_code_block(self):
        result = collect_sentences([
            "Start. ```\nline1\nline2\nline3\n``` End."
        ])
        assert "line1" not in " ".join(result)
        assert "Start." in result

    def test_only_code_block(self):
        result = collect_sentences(["```\ncode only\n```"])
        assert result == []


class TestInlineMarkdownStripping:
    def test_bold(self):
        assert _strip_inline_markdown("**bold text**") == "bold text"

    def test_italic(self):
        assert _strip_inline_markdown("_italic text_") == "italic text"

    def test_backtick(self):
        assert _strip_inline_markdown("`code`") == "code"

    def test_heading(self):
        assert _strip_inline_markdown("## Heading") == "Heading"

    def test_mixed(self):
        assert _strip_inline_markdown("**bold** and _italic_ and `code`") == "bold and italic and code"

    def test_no_markdown(self):
        assert _strip_inline_markdown("plain text") == "plain text"


class TestFindSentenceBoundary:
    def test_period_space(self):
        assert _find_sentence_boundary("Hello. World") == 5

    def test_question_space(self):
        assert _find_sentence_boundary("Why? Because") == 3

    def test_no_boundary(self):
        assert _find_sentence_boundary("Hello world") is None

    def test_period_at_end_no_space(self):
        assert _find_sentence_boundary("Hello world.") is None

    def test_abbreviation_like(self):
        # e.g. "Dr. Smith" — will split, which is acceptable for TTS
        assert _find_sentence_boundary("Dr. Smith") == 2

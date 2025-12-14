"""
Module 3: Matching Engine
Two-tier matching: regex -> fuzzy
Prioritizes fast local matching.
"""

import re
from dataclasses import dataclass
from enum import Enum
from rapidfuzz import fuzz, process
from mmucc_loader import MMUCCDictionary, MatchResult, get_dictionary


class MatchMethod(Enum):
    """How the match was found."""
    REGEX = "regex"
    FUZZY = "fuzzy"
    NONE = "none"


@dataclass
class EngineResult:
    """Result from the matching engine."""
    success: bool
    code: str | None
    label: str | None
    confidence: float  # 0-100
    method: MatchMethod
    matched_on: str | None  # The synonym or text that matched
    original_text: str
    needs_review: bool


class MatchingEngine:
    """
    Two-tier matching engine for crash data standardization.

    Tier 1: Regex matching (fast, exact)
    Tier 2: Fuzzy matching (handles typos/variations)
    """

    def __init__(self, dictionary: MMUCCDictionary = None, fuzzy_threshold: int = 85):
        """
        Initialize the matching engine.

        Args:
            dictionary: MMUCC dictionary instance (uses default if None)
            fuzzy_threshold: Minimum score (0-100) for fuzzy match acceptance
        """
        self.dictionary = dictionary or get_dictionary()
        self.fuzzy_threshold = fuzzy_threshold

    def _preprocess(self, text: str) -> str:
        """
        Clean input text before matching.
        Removes noise that interferes with fuzzy matching.
        """
        if not text:
            return ""

        cleaned = text.lower().strip()

        # Remove trailing punctuation
        cleaned = re.sub(r'[?!.,;:]+$', '', cleaned)

        # Remove common hedging suffixes
        cleaned = re.sub(r'-?(ish|like|esque|type|kind|sort of|maybe)$', '', cleaned, flags=re.IGNORECASE)

        # Normalize common separators to spaces
        cleaned = re.sub(r'[-_/]+', ' ', cleaned)

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def match(self, text: str, category: str) -> EngineResult:
        """
        Match input text to a standardized category.
        Tries regex -> fuzzy in order.

        Args:
            text: Raw input text to match
            category: MMUCC category to match against

        Returns:
            EngineResult with match details
        """
        if not text or not text.strip():
            return EngineResult(
                success=False,
                code=None,
                label=None,
                confidence=0,
                method=MatchMethod.NONE,
                matched_on=None,
                original_text=text or "",
                needs_review=True
            )

        original = text.strip()
        cleaned = self._preprocess(text)

        # Tier 1: Regex matching (try both original and cleaned)
        result = self._regex_match(original, category)
        if result:
            return result

        if cleaned != original.lower():
            result = self._regex_match(cleaned, category)
            if result:
                result.original_text = original  # Preserve original
                return result

        # Tier 2: Fuzzy matching (use cleaned text)
        result = self._fuzzy_match(cleaned, category, original)
        if result:
            return result

        # No match found
        return EngineResult(
            success=False,
            code="99",
            label="Unknown",
            confidence=0,
            method=MatchMethod.NONE,
            matched_on=None,
            original_text=original,
            needs_review=True
        )

    def _regex_match(self, text: str, category: str) -> EngineResult | None:
        """Tier 1: Regex/exact matching."""
        result = self.dictionary.regex_lookup(category, text)

        if result:
            return EngineResult(
                success=True,
                code=result.code,
                label=result.label,
                confidence=100,
                method=MatchMethod.REGEX,
                matched_on=result.matched_synonym,
                original_text=text,
                needs_review=False
            )
        return None

    def _fuzzy_match(self, text: str, category: str, original_text: str = None) -> EngineResult | None:
        """
        Tier 2: Fuzzy matching with rapidfuzz.
        Uses multiple scorers for better coverage.
        """
        # Get all synonyms for this category
        all_synonyms = self.dictionary.get_all_synonyms_flat(category)
        synonym_to_code = self.dictionary.get_synonym_to_code_map(category)

        if not all_synonyms:
            return None

        best_match = None
        best_score = 0

        # Try token_set_ratio (good for phrases, word order doesn't matter)
        result = process.extractOne(
            text,
            all_synonyms,
            scorer=fuzz.token_set_ratio
        )
        if result and result[1] > best_score:
            best_match = result
            best_score = result[1]

        # Try ratio (good for single-word typos like "daylite" -> "daylight")
        result = process.extractOne(
            text,
            all_synonyms,
            scorer=fuzz.ratio
        )
        if result and result[1] > best_score:
            best_match = result
            best_score = result[1]

        # Try partial_ratio (good when input contains the synonym)
        result = process.extractOne(
            text,
            all_synonyms,
            scorer=fuzz.partial_ratio
        )
        if result and result[1] > best_score:
            best_match = result
            best_score = result[1]

        if best_match and best_score >= self.fuzzy_threshold:
            matched_synonym, score, _ = best_match
            code = synonym_to_code.get(matched_synonym.lower())
            label = self.dictionary.lookup_by_code(category, code)

            return EngineResult(
                success=True,
                code=code,
                label=label,
                confidence=score,
                method=MatchMethod.FUZZY,
                matched_on=matched_synonym,
                original_text=original_text or text,
                needs_review=(score < 95)  # Flag for review if not highly confident
            )

        return None

    def match_batch(self, texts: list[str], category: str) -> list[EngineResult]:
        """
        Match multiple texts to a category.

        Args:
            texts: List of raw input texts
            category: MMUCC category to match against

        Returns:
            List of EngineResults in same order as input
        """
        return [self.match(text, category) for text in texts]

    def get_stats(self, results: list[EngineResult]) -> dict:
        """
        Get statistics on matching results.

        Args:
            results: List of EngineResults from match_batch

        Returns:
            Dict with counts by method and review status
        """
        stats = {
            "total": len(results),
            "matched": sum(1 for r in results if r.success),
            "unmatched": sum(1 for r in results if not r.success),
            "needs_review": sum(1 for r in results if r.needs_review),
            "by_method": {
                "regex": sum(1 for r in results if r.method == MatchMethod.REGEX),
                "fuzzy": sum(1 for r in results if r.method == MatchMethod.FUZZY),
                "none": sum(1 for r in results if r.method == MatchMethod.NONE),
            },
            "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0
        }
        return stats


# === CLI for testing ===
if __name__ == "__main__":
    print("Matching Engine Test")
    print("=" * 50)

    engine = MatchingEngine(fuzzy_threshold=80)

    # Test cases - mix of clean and messy inputs
    test_cases = [
        ("manner_of_collision", "rear end"),           # Clean - should regex match
        ("manner_of_collision", "REAR-ENDED"),         # Caps variation
        ("manner_of_collision", "rearended vehicle"),  # Typo
        ("manner_of_collision", "t bone collision"),   # Common term
        ("manner_of_collision", "hit from behind"),    # Synonym
        ("manner_of_collision", "xyzabc123"),          # Garbage - should fail
        ("injury_severity", "fatal"),
        ("injury_severity", "minor injuries"),
        ("injury_severity", "PDO"),                    # Acronym
        ("weather_condition", "it was raining hard"),
        ("weather_condition", "clear and sunny"),
        ("light_condition", "daylite-ish"),            # Typo + suffix
        ("light_condition", "bright sun?"),            # Punctuation
        ("light_condition", "overcast wet"),           # Mixed category
        ("contributing_factor_driver", "driver was texting"),
        ("contributing_factor_driver", "DUI"),
        ("contributing_factor_driver", "fell asleep at wheel"),
    ]

    results = []

    for category, text in test_cases:
        result = engine.match(text, category)
        results.append(result)

        status = "+" if result.success else "x"
        review = " [REVIEW]" if result.needs_review else ""

        print(f"{status} '{text}'")
        print(f"   -> {result.code}: {result.label}")
        print(f"   -> Method: {result.method.value}, Confidence: {result.confidence}%{review}")
        print()

    # Print stats
    print("=" * 50)
    stats = engine.get_stats(results)
    print(f"Stats: {stats['matched']}/{stats['total']} matched")
    print(f"By method: regex={stats['by_method']['regex']}, fuzzy={stats['by_method']['fuzzy']}, none={stats['by_method']['none']}")
    print(f"Needs review: {stats['needs_review']}")
    print(f"Avg confidence: {stats['avg_confidence']:.1f}%")
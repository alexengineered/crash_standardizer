"""
Module 2: MMUCC Dictionary Loader
Loads standardized MMUCC dictionaries for crash data categorization.
Provides lookup by code and reverse lookup by synonym.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of a dictionary lookup."""
    code: str
    label: str
    matched_synonym: str
    category: str


class MMUCCDictionary:
    """
    MMUCC Dictionary manager.
    Loads dictionaries from JSON and provides lookup methods.
    """
    
    def __init__(self, json_path: str = None):
        """Load dictionaries from JSON file."""
        if json_path is None:
            # Default to same directory as this module
            json_path = Path(__file__).parent / "mmucc_dictionaries.json"
        
        with open(json_path, 'r') as f:
            self._data = json.load(f)
        
        # Remove metadata from category list
        self._metadata = self._data.pop('_metadata', {})
        
        # Build reverse lookup indexes for fast matching
        self._build_indexes()
    
    def _build_indexes(self):
        """Build synonym-to-code indexes for each category."""
        self._indexes = {}
        
        for category, data in self._data.items():
            index = {}
            for code, synonyms in data.get('synonyms', {}).items():
                for syn in synonyms:
                    # Normalize synonym for matching
                    normalized = syn.lower().strip()
                    index[normalized] = (code, data['codes'].get(code, 'Unknown'))
            self._indexes[category] = index
    
    @property
    def categories(self) -> list[str]:
        """List of available dictionary categories."""
        return list(self._data.keys())
    
    def get_category_label(self, category: str) -> str:
        """Get the human-readable label for a category."""
        if category not in self._data:
            raise KeyError(f"Unknown category: {category}")
        return self._data[category].get('label', category)
    
    def get_codes(self, category: str) -> dict[str, str]:
        """Get all code -> label mappings for a category."""
        if category not in self._data:
            raise KeyError(f"Unknown category: {category}")
        return self._data[category].get('codes', {})
    
    def get_synonyms(self, category: str) -> dict[str, list[str]]:
        """Get all code -> synonym list mappings for a category."""
        if category not in self._data:
            raise KeyError(f"Unknown category: {category}")
        return self._data[category].get('synonyms', {})
    
    def lookup_by_code(self, category: str, code: str) -> str:
        """Look up the label for a given code."""
        codes = self.get_codes(category)
        return codes.get(str(code), None)
    
    def lookup_by_text(self, category: str, text: str) -> MatchResult | None:
        """
        Look up code and label by matching input text against synonyms.
        Uses exact matching on normalized text.
        Returns None if no match found.
        """
        if category not in self._indexes:
            raise KeyError(f"Unknown category: {category}")
        
        if not text:
            return None
        
        normalized = text.lower().strip()
        index = self._indexes[category]
        
        # Try exact match first
        if normalized in index:
            code, label = index[normalized]
            return MatchResult(
                code=code,
                label=label,
                matched_synonym=normalized,
                category=category
            )
        
        # Try substring matching for longer text
        for synonym, (code, label) in index.items():
            if synonym in normalized or normalized in synonym:
                return MatchResult(
                    code=code,
                    label=label,
                    matched_synonym=synonym,
                    category=category
                )
        
        return None
    
    def regex_lookup(self, category: str, text: str) -> MatchResult | None:
        """
        Look up using regex patterns built from synonyms.
        More flexible than exact matching.
        """
        if category not in self._data:
            raise KeyError(f"Unknown category: {category}")
        
        if not text:
            return None
        
        text_lower = text.lower().strip()
        
        for code, synonyms in self.get_synonyms(category).items():
            for syn in synonyms:
                # Build pattern with word boundaries where possible
                pattern = r'\b' + re.escape(syn) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return MatchResult(
                        code=code,
                        label=self.lookup_by_code(category, code),
                        matched_synonym=syn,
                        category=category
                    )
        
        return None
    
    def get_all_synonyms_flat(self, category: str) -> list[str]:
        """Get flat list of all synonyms for fuzzy matching."""
        synonyms = []
        for code, syn_list in self.get_synonyms(category).items():
            synonyms.extend(syn_list)
        return synonyms
    
    def get_synonym_to_code_map(self, category: str) -> dict[str, str]:
        """Get flat synonym -> code mapping for a category."""
        result = {}
        for code, syn_list in self.get_synonyms(category).items():
            for syn in syn_list:
                result[syn.lower()] = code
        return result


# Convenience function for quick access
_default_dictionary = None

def get_dictionary() -> MMUCCDictionary:
    """Get the default MMUCC dictionary instance."""
    global _default_dictionary
    if _default_dictionary is None:
        _default_dictionary = MMUCCDictionary()
    return _default_dictionary


# === CLI for testing ===
if __name__ == "__main__":
    import sys
    
    d = MMUCCDictionary()
    
    print("MMUCC Dictionary Loader")
    print("=" * 50)
    print(f"\nAvailable categories ({len(d.categories)}):")
    for cat in d.categories:
        codes = d.get_codes(cat)
        print(f"  - {cat}: {d.get_category_label(cat)} ({len(codes)} codes)")
    
    # Test lookups
    print("\n" + "=" * 50)
    print("Test Lookups:")
    
    tests = [
        ("manner_of_collision", "rear end"),
        ("manner_of_collision", "t-bone"),
        ("injury_severity", "fatal"),
        ("injury_severity", "no injury"),
        ("weather_condition", "raining"),
        ("contributing_factor_driver", "texting"),
        ("contributing_factor_driver", "drunk"),
    ]
    
    for category, text in tests:
        result = d.regex_lookup(category, text)
        if result:
            print(f"  '{text}' -> Code {result.code}: {result.label}")
        else:
            print(f"  '{text}' -> NO MATCH")
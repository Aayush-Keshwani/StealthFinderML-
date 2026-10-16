"""
Merchant normalization and alias mapping module for Stealth-Spend.
Harmonizes diverse merchant description formats to canonical business names.
"""

import re
from typing import Dict, Optional, Tuple, Set


COMMON_MERCHANT_PATTERNS = {
    r"\buber\b|\buber\s*\*?trip\b": "Uber",
    r"\buber\s*eats\b": "Uber Eats",
    r"\bcostco\b": "Costco",
    r"\bwhole\s*foods\b": "Whole Foods",
    r"\btrader\s*joe\b": "Trader Joe's",
    r"\bamazon\b|\bamzn\b": "Amazon",
    r"\btarget\b": "Target",
    r"\bwalmart\b": "Walmart",
    r"\bstarbucks\b": "Starbucks",
    r"\bchipotle\b": "Chipotle",
    r"\bmcdonald\b": "McDonald's",
    r"\bnetflix\b": "Netflix",
    r"\bspotify\b": "Spotify",
    r"\bapple\b|\bitunes\b": "Apple",
    r"\bchevron\b": "Chevron",
    r"\bshell\b": "Shell",
    r"\bcvs\b": "CVS Pharmacy",
    r"\bwalgreens\b": "Walgreens",
    r"\bbest\s*buy\b": "Best Buy",
    r"\bat&t\b|\batt\b": "AT&T",
    r"\bcomcast\b|\bxfinity\b": "Comcast",
    r"\bconed\b|\bcon\s*edison\b": "ConEd",
    r"\brealty\b|\brent\b": "Realty Partners",
    r"\bsteam\b": "Steam Games",
    r"\bhm\b|\bh&m\b": "H&M",
    r"\betsy\b": "Etsy",
    r"\bspothero\b": "SpotHero",
    r"\bsafeway\b": "Safeway",
    r"\baldi\b": "Aldi",
    r"\bmarriott\b": "Marriott",
}


class MerchantNormalizer:
    """
    Normalizes messy bank statement merchant strings into canonical names
    using regex patterns, keyword rules, and an alias lookup table.
    """

    def __init__(self, custom_aliases: Optional[Dict[str, str]] = None):
        self.aliases = custom_aliases or {}
        self.known_merchants: Set[str] = set()

    def normalize(self, raw_description: str) -> str:
        """
        Extracts and normalizes the merchant name from a raw transaction description.
        """
        if not isinstance(raw_description, str) or not raw_description.strip():
            return "Unknown Merchant"

        text = raw_description.lower().strip()

        # Check explicit custom aliases first
        if text in self.aliases:
            return self.aliases[text]

        # Check known regex patterns
        for pattern, canonical_name in COMMON_MERCHANT_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                self.known_merchants.add(canonical_name)
                return canonical_name

        # Fallback: clean extraneous punctuation/numbers and capitalize words
        cleaned = re.sub(r"^(purchase\s*at|pos\s*debit|card\s*charge|check\s*card)\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"[#*0-9\-_]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        tokens = cleaned.split()
        if tokens:
            name = " ".join(tokens[:3]).title()
            self.known_merchants.add(name)
            return name
        return "Unknown Merchant"

    def is_new_merchant(self, merchant_name: str, historical_merchants: Set[str]) -> bool:
        """
        Checks if a merchant has never been seen in the historical training window.
        """
        return merchant_name not in historical_merchants

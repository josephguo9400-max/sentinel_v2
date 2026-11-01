"""Cheap, deterministic language identification for the Stage 1 language gate.

Method: langdetect 1.0.9 (character n-gram naive Bayes, the same library and seed that labelled the splits'
`lang` field, see training/language_mix.py) on a private DetectorFactory with `seed = lang_seed` (default 0).
langdetect seeds a fresh `random.Random(seed)` for every detection, so the same text always gets the same answer,
independent of call order and of other users of the library. Each text is cut to its first `max_chars` characters
(default 2,000).

Decision rule, per text: "unknown" when the text has fewer than `min_letters` letters, when langdetect finds no
features, or when the top language's probability is below `min_prob`; otherwise the top language code (ISO 639-1,
"zh-cn"/"zh-tw" reduced to "zh").

Failure mode: short texts (a greeting, a one-line command, a code snippet, a URL) and mixed-language texts come
back as "unknown" or as the wrong language. The router treats "unknown" exactly like an out-of-scope language: it
escalates to Stage 2. So a misidentification can only make Stage 1 *more* conservative, except one way round: a
non-English text identified as English with probability >= min_prob is admitted to Stage 1 finalization. That is
the residual risk; langdetect is not robust to adversarial text that pads an injection with English filler.
Cost: about 2-4 ms per 2,000-character chunk on this host (one-off measurement, not a benchmark).
"""
from __future__ import annotations

import threading

UNKNOWN = "unknown"


class LanguageDetector:
    def __init__(self, *, seed: int = 0, min_letters: int = 20, min_prob: float = 0.80, max_chars: int = 2000):
        from langdetect.detector_factory import PROFILES_DIRECTORY, DetectorFactory

        self.seed = seed
        self.min_letters = min_letters
        self.min_prob = min_prob
        self.max_chars = max_chars
        factory = DetectorFactory()
        factory.load_profile(PROFILES_DIRECTORY)
        factory.seed = seed  # instance attribute: never touches the library-wide DetectorFactory.seed
        self._factory = factory
        self._lock = threading.Lock()  # langdetect's factory is not documented as thread-safe

    def describe(self) -> dict:
        import langdetect

        return {"method": f"langdetect {getattr(langdetect, '__version__', '1.0.9')}", "seed": self.seed,
                "min_letters": self.min_letters, "min_prob": self.min_prob, "max_chars": self.max_chars}

    def detect(self, text: str) -> tuple[str, float]:
        """(language code or "unknown", top probability; 0.0 for unknown-by-rule)."""
        from langdetect.lang_detect_exception import LangDetectException

        sample = text[: self.max_chars]
        if sum(ch.isalpha() for ch in sample) < self.min_letters:
            return UNKNOWN, 0.0
        try:
            with self._lock:
                det = self._factory.create()
                det.append(sample)
                probs = det.get_probabilities()
        except LangDetectException:
            return UNKNOWN, 0.0
        if not probs:
            return UNKNOWN, 0.0
        top = probs[0]
        lang = top.lang.split("-")[0]
        if top.prob < self.min_prob:
            return UNKNOWN, float(top.prob)
        return lang, float(top.prob)


class FixedLanguage:
    """Test double: returns one language for every text."""

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def describe(self) -> dict:
        return {"method": "fixed", "lang": self.lang}

    def detect(self, text: str) -> tuple[str, float]:
        return self.lang, 1.0

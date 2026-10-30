"""Strict recipient parsing and canonicalization for the authorization firewall.

Design rule: a recipient is reduced to one canonical ASCII address, and that canonical address is
what the sandbox would deliver to. Normalization is only applied where it cannot change the
destination mailbox:

* surrounding ASCII spaces are stripped (tabs and line breaks are rejected as control characters,
  other Unicode spacing such as NBSP is rejected as non-ASCII);
* display names ("Dana <dana@example.com>") are dropped, only the angle address counts;
* ASCII case is folded (the sandbox treats local parts as case-insensitive, like most providers);
* lists may be given as a Python list or as a comma/semicolon separated string.

Everything else that could make the destination ambiguous is rejected instead of normalized:
non-ASCII characters in the address (Unicode look-alikes such as Cyrillic "a", fullwidth forms,
the Kelvin sign, zero-width characters), control characters (header injection), quoted local
parts, comments, nested or unbalanced brackets, multiple "@", trailing dots and empty entries.
For rejected look-alikes we still compute a confusable "skeleton" so the block reason can say which
authorized address was being imitated; the skeleton is diagnostic only and never authorizes.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Conservative ASCII address grammar: dot-atom local part, LDH domain labels, at least one dot.
_LOCAL = r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_ADDRESS = re.compile(rf"^{_LOCAL}@{_LABEL}(?:\.{_LABEL})+$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f\x85  ]")
_MAX_ENTRIES = 50
_MAX_LEN = 320

# A small confusables table (Cyrillic, Greek, Latin extensions) mapped to ASCII. Diagnostic only.
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ɡ": "g", "ı": "i", "ȷ": "j",
    "А": "a", "В": "b", "Е": "e", "К": "k", "М": "m", "Н": "h", "О": "o", "Р": "p", "С": "c",
    "Т": "t", "Х": "x", "У": "y", "α": "a", "ο": "o", "ν": "v", "τ": "t", "ρ": "p", "ε": "e",
    "ι": "i", "κ": "k", "Α": "a", "Β": "b", "Ε": "e", "Ζ": "z", "Η": "h", "Ι": "i", "Κ": "k",
    "Μ": "m", "Ν": "n", "Ο": "o", "Ρ": "p", "Τ": "t", "Υ": "y", "Χ": "x", "ℓ": "l",
}


class RecipientError(ValueError):
    """A recipient value that cannot be reduced to exactly one unambiguous ASCII address."""

    def __init__(self, code: str, raw: str, message: str):
        super().__init__(f"{code}: {message} ({raw!r})")
        self.code = code
        self.raw = raw


@dataclass(frozen=True)
class Recipient:
    raw: str
    address: str  # canonical: lowercase ASCII, no display name


def skeleton(text: str) -> str:
    """Confusable skeleton: NFKC, drop format/invisible chars, map look-alikes, casefold."""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        if unicodedata.category(ch) in ("Cf", "Mn", "Zs", "Cc"):
            continue
        out.append(_CONFUSABLES.get(ch, ch))
    return "".join(out).casefold()


def _split_entries(value: str) -> list[str]:
    """Split on ',' or ';' outside double quotes and angle brackets; reject unbalanced syntax."""
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    depth = 0
    escape = False
    for ch in value:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\" and in_quote:
            buf.append(ch)
            escape = True
            continue
        if ch == '"':
            in_quote = not in_quote
        elif not in_quote and ch == "<":
            depth += 1
            if depth > 1:
                raise RecipientError("nested_brackets", value, "nested angle brackets")
        elif not in_quote and ch == ">":
            depth -= 1
            if depth < 0:
                raise RecipientError("unbalanced_brackets", value, "unbalanced angle brackets")
        elif not in_quote and depth == 0 and ch in ",;":
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if in_quote or depth != 0:
        raise RecipientError("unbalanced_syntax", value, "unterminated quote or angle bracket")
    parts.append("".join(buf))
    return parts


def _address_part(entry: str) -> str:
    """Return the routable address of one entry: the angle address if present, else the entry."""
    if "<" not in entry and ">" not in entry:
        return entry
    m = re.fullmatch(r'(?s)(?P<name>[^<>]*)<(?P<addr>[^<>]*)>\s*', entry)
    if not m:
        raise RecipientError("bad_angle_address", entry, "text after or around the angle address")
    name = m.group("name").strip()
    if name.startswith('"') != name.endswith('"') and name:
        raise RecipientError("bad_display_name", entry, "unbalanced quotes in display name")
    return m.group("addr")


def parse_one(entry: str) -> Recipient:
    raw = entry
    if _CONTROL.search(entry):
        raise RecipientError("control_character", raw, "control or line-break character")
    entry = entry.strip(" ")  # ASCII space only; other Unicode spacing is rejected below
    if not entry:
        raise RecipientError("empty_recipient", raw, "empty recipient entry")
    if len(entry) > _MAX_LEN:
        raise RecipientError("too_long", raw, "recipient entry too long")
    addr = _address_part(entry).strip(" ")
    if "(" in addr or ")" in addr:
        raise RecipientError("comment_in_address", raw, "comments are not allowed in addresses")
    if not addr.isascii():
        raise RecipientError("non_ascii_address", raw, "address contains non-ASCII or invisible characters")
    if addr.lower().startswith("mailto:"):
        raise RecipientError("mailto_prefix", raw, "mailto: URIs are not addresses")
    canonical = addr.lower()
    if not _ADDRESS.fullmatch(canonical):
        raise RecipientError("invalid_address", raw, "not a plain dot-atom address")
    return Recipient(raw=raw, address=canonical)


def parse_recipients(value: object, field: str = "to") -> list[Recipient]:
    """Parse a to/cc/bcc value: None, a string (comma/semicolon separated) or a list of strings.

    Duplicates (after canonicalization) collapse to one entry, preserving first occurrence.
    Raises RecipientError on anything ambiguous.
    """
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip(" "):
            return []
        entries = _split_entries(value)
    elif isinstance(value, (list, tuple)):
        entries = []
        for item in value:
            if not isinstance(item, str):
                raise RecipientError("bad_type", repr(item), f"{field} entries must be strings")
            entries.extend(_split_entries(item))
    else:
        raise RecipientError("bad_type", repr(value), f"{field} must be a string or list of strings")
    if len(entries) > _MAX_ENTRIES:
        raise RecipientError("too_many_recipients", field, "too many recipients")
    seen: dict[str, Recipient] = {}
    for e in entries:
        r = parse_one(e)
        seen.setdefault(r.address, r)
    return list(seen.values())


def canonical_or_none(value: str) -> str | None:
    try:
        return parse_one(value).address
    except RecipientError:
        return None

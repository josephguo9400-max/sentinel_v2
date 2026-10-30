"""Structure-aware segmentation and window packing.

1. `segment(text)` splits an input into ordered segments of one kind each:
   - "body"       paragraphs of the visible text (blank-line separated)
   - "quoted"     quoted replies: lines starting with ">" (markers stripped for scoring), an attribution line
                  "On ... wrote:" (one or two lines) and everything after an Outlook "-----Original Message-----"
   - "signature"  everything after a "-- " (or "--") delimiter line, until a quoted block starts
   - "hidden"     text a mail client would not show: HTML comments, elements hidden by style (display:none,
                  visibility:hidden, font-size:0, opacity:0, zero width/height or max-height), the `hidden`
                  and aria-hidden="true" attributes, <head>/<script>/<style>/<template>/<noscript> contents,
                  and text-bearing attributes (alt, title, aria-label, placeholder, value, content)
   HTML is detected by the presence of a tag; tags are stripped, block tags become paragraph breaks, and
   hidden text is never mixed into visible text: each hidden element is its own segment. Zero-width characters
   (U+200B-U+200D, U+2060, U+FEFF) are removed from segment text before scoring.

2. `build_chunks(segments, tokenize, window, stride)` turns segments into model windows of at most `window`
   content tokens (see `plan_windows`):
   - consecutive segments of the same kind form one token stream; segments of different kinds never share a
     window (hidden text is never diluted with visible text, a signature never with the body);
   - inside a stream, a window ends on a segment boundary whenever one falls in its second half, and the next
     window starts on a segment boundary when one gives an overlap between `window - stride` (126 by default) and
     `window // 2` tokens;
   - where no boundary is in reach (a segment longer than a window), the cut falls inside the segment and the next
     window overlaps by exactly `window - stride` tokens: plain sliding windows.
   Consecutive windows always overlap by at least `window - stride` tokens, so any span of at most
   `window - stride + 1` (127) tokens of one stream is fully inside at least one window, whether or not it
   crosses a paragraph boundary or a fixed 510-token position.

Not handled (known failure modes): text hidden by colour (white on white), off-screen positioning or CSS classes;
markup other than the listed attributes (e.g. URLs in href) is not scored and does not count as content; quoted
replies in HTML (<blockquote>) are treated as body text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable, Literal

Kind = Literal["body", "quoted", "signature", "hidden"]
TokenizeFn = Callable[[str], tuple[list[int], list[tuple[int, int]] | None]]  # ids, char offsets per id

ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_HTML_TAG = re.compile(r"<(?:!--|/?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>)")
_ATTRIBUTION_ONE = re.compile(r"^\s*On\s.{1,300}\swrote:\s*$")
_ATTRIBUTION_START = re.compile(r"^\s*On\s.{1,300}$")
_WROTE_END = re.compile(r"^.{0,300}\bwrote:\s*$")
_ORIGINAL_MESSAGE = re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE)
_QUOTE_PREFIX = re.compile(r"^\s*(?:>\s?)+")


@dataclass(frozen=True)
class Segment:
    kind: Kind
    text: str


# ----------------------------------------------------------------------------------------------- plain text


def _plain_segments(text: str) -> list[Segment]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kinds: list[Kind | None] = []  # None = blank line (paragraph break)
    in_sig = False
    in_original = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kinds.append(None)
            continue
        if in_original or _ORIGINAL_MESSAGE.match(line):
            in_original = True
            kinds.append("quoted")
            continue
        if _QUOTE_PREFIX.match(line) and stripped.startswith(">"):
            kinds.append("quoted")
            in_sig = False
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if _ATTRIBUTION_ONE.match(line):
            kinds.append("quoted")
            in_sig = False
            continue
        if _ATTRIBUTION_START.match(line) and _WROTE_END.match(nxt) and not nxt.lstrip().startswith(">"):
            kinds.append("quoted")  # "On Mon, ... John <j@x>" / "wrote:"
            continue
        if kinds and kinds[-1] == "quoted" and _WROTE_END.match(line) and i > 0 and _ATTRIBUTION_START.match(
                lines[i - 1]):
            kinds.append("quoted")
            continue
        if line.rstrip("\n") in ("-- ", "--"):
            in_sig = True
            kinds.append(None)  # the delimiter itself carries no content
            continue
        kinds.append("signature" if in_sig else "body")

    out: list[Segment] = []
    cur_kind: Kind | None = None
    cur: list[str] = []

    def flush() -> None:
        if cur and cur_kind is not None:
            body = "\n".join(cur).strip()
            if body:
                out.append(Segment(cur_kind, body))
        cur.clear()

    for line, kind in zip(lines, kinds):
        if kind is None:
            flush()
            cur_kind = None
            continue
        if kind != cur_kind:
            flush()
            cur_kind = kind
        cur.append(_QUOTE_PREFIX.sub("", line, count=1) if kind == "quoted" and line.lstrip().startswith(">")
                   else line)
    flush()
    return out


# ------------------------------------------------------------------------------------------------------ HTML

_HIDDEN_STYLE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden|"
    r"font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?\s*(?:!important)?\s*(?:;|$)|"
    r"opacity\s*:\s*0(?:\.0+)?\s*(?:!important)?\s*(?:;|$)|"
    r"(?:max-)?(?:width|height)\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?\s*(?:!important)?\s*(?:;|$))",
    re.IGNORECASE)
_HIDDEN_TAGS = {"head", "script", "style", "template", "noscript", "title"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
               "section", "article", "header", "footer", "hr", "pre", "td", "th", "dd", "dt"}
_TEXT_ATTRS = {"alt", "title", "aria-label", "placeholder", "value", "content"}


def _is_hidden(tag: str, attrs: dict[str, str | None]) -> bool:
    if tag in _HIDDEN_TAGS or "hidden" in attrs:
        return True
    if (attrs.get("aria-hidden") or "").strip().lower() == "true":
        return True
    return bool(_HIDDEN_STYLE.search(attrs.get("style") or ""))


class _MailHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible: list[str] = []
        self.hidden: list[str] = []
        self._stack: list[tuple[str, bool]] = []  # (tag, opened a hidden region)
        self._hidden_depth = 0
        self._hidden_buf: list[str] = []

    def _attr_text(self, attrs: dict[str, str | None]) -> None:
        for k in sorted(attrs):
            v = attrs[k]
            if v and (k in _TEXT_ATTRS or k.startswith("data-")) and v.strip():
                self.hidden.append(v.strip())

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): v for k, v in attrs}
        self._attr_text(a)
        if tag in _BLOCK_TAGS and not self._hidden_depth:
            self.visible.append("\n\n")
        if tag in _VOID_TAGS:
            return
        hides = _is_hidden(tag, a)
        if hides:
            if self._hidden_depth == 0:
                self._hidden_buf = []
            self._hidden_depth += 1
        self._stack.append((tag, hides))

    def handle_startendtag(self, tag, attrs):
        a = {k.lower(): v for k, v in attrs}
        self._attr_text(a)
        if tag in _BLOCK_TAGS and not self._hidden_depth:
            self.visible.append("\n\n")

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        if not any(t == tag for t, _ in self._stack):
            return  # stray end tag
        while self._stack:
            t, hides = self._stack.pop()
            if hides:
                self._hidden_depth -= 1
                if self._hidden_depth == 0:
                    self._flush_hidden()
            if t == tag:
                break
        if tag in _BLOCK_TAGS and not self._hidden_depth:
            self.visible.append("\n\n")

    def _flush_hidden(self) -> None:
        txt = " ".join("".join(self._hidden_buf).split())
        if txt:
            self.hidden.append(txt)
        self._hidden_buf = []

    def handle_data(self, data):
        (self._hidden_buf if self._hidden_depth else self.visible).append(data)

    def handle_comment(self, data):
        if data.strip():
            self.hidden.append(" ".join(data.split()))

    def close(self):
        super().close()
        if self._hidden_depth:  # unclosed hidden element at end of input
            self._hidden_depth = 0
            self._flush_hidden()


def looks_like_html(text: str) -> bool:
    return bool(_HTML_TAG.search(text))


def segment(text: str) -> list[Segment]:
    """Ordered segments with zero-width characters removed. Visible text first, then hidden segments."""
    if looks_like_html(text):
        p = _MailHTML()
        p.feed(text)
        p.close()
        visible = re.sub(r"[ \t]*\n[ \t]*", "\n", "".join(p.visible))
        visible = re.sub(r"\n{3,}", "\n\n", visible)
        segs = _plain_segments(visible) + [Segment("hidden", h) for h in p.hidden]
    else:
        segs = _plain_segments(text)
    out = []
    for s in segs:
        t = ZERO_WIDTH.sub("", s.text).strip()
        if t:
            out.append(Segment(s.kind, t))
    return out


# --------------------------------------------------------------------------------------------------- windows


@dataclass
class Chunk:
    ids: list[int]  # content ids, without [CLS]/[SEP]
    text: str | None  # the text these ids came from (None if the tokenizer reports no offsets)
    kind: Kind
    spans: list[tuple[int, int, int]] = field(default_factory=list)  # (segment index, token start, token end)
    parts: list[str] = field(default_factory=list)  # text of each span (one per segment piece), when known


@dataclass
class ChunkSet:
    chunks: list[Chunk]
    seg_tokens: list[int]  # content tokens per segment
    segments: list[Segment]

    @property
    def n_tokens(self) -> int:
        return sum(self.seg_tokens)

    def covered(self, chosen) -> bool:
        """True iff every token of every segment lies in at least one chunk whose index is in `chosen`."""
        per_seg: dict[int, list[tuple[int, int]]] = {}
        for i in set(chosen):
            for seg, a, b in self.chunks[i].spans:
                per_seg.setdefault(seg, []).append((a, b))
        for seg, n in enumerate(self.seg_tokens):
            if n == 0:
                continue
            reach = 0
            for a, b in sorted(per_seg.get(seg, [])):
                if a > reach:
                    return False
                reach = max(reach, b)
            if reach < n:
                return False
        return True


def plan_windows(n: int, boundaries: list[int], window: int, min_overlap: int) -> list[tuple[int, int]]:
    """Windows [s, e) over a stream of n tokens whose segment boundaries are `boundaries` (strictly inside (0, n)).

    - A window ends at the last boundary in [s + window // 2, s + window] if there is one, else at s + window.
    - The next window starts at the last boundary in [e - window // 2, e - min_overlap] if there is one, else at
      e - min_overlap; so consecutive windows always overlap by at least min_overlap tokens.
    - The last window ends at n.
    Guarantee (tested): every span of at most min_overlap + 1 tokens lies inside one window.
    """
    if n <= window:
        return [(0, n)]
    if not 0 <= min_overlap < window // 2 + 1:
        raise ValueError("need 0 <= min_overlap <= window // 2")
    bset = sorted(b for b in set(boundaries) if 0 < b < n)
    out: list[tuple[int, int]] = []
    s = 0
    while True:
        if s + window >= n:
            out.append((s, n))
            return out
        lo, hi = s + window // 2, s + window
        cands = [b for b in bset if lo <= b <= hi]
        e = cands[-1] if cands else hi
        out.append((s, e))
        lo, hi = max(s + 1, e - window // 2), e - min_overlap
        cands = [b for b in bset if lo <= b <= hi]
        s = cands[-1] if cands else max(s + 1, hi)


def build_chunks(segments: list[Segment], tokenize: TokenizeFn, *, window: int = 510, stride: int = 384,
                 structure_aware: bool = True) -> ChunkSet:
    """Windows of at most `window` content tokens with a minimum overlap of `window - stride` tokens.

    With structure_aware, each maximal run of consecutive segments of the same kind is one token stream whose
    segment starts are preferred window boundaries; runs of different kinds never share a window. Without it,
    all segments form one stream with no preferred boundaries (plain sliding windows, stride `stride`).
    """
    if window < 1 or not 0 < stride <= window:
        raise ValueError("need window >= 1 and 0 < stride <= window")
    min_overlap = min(window - stride, window // 2)
    toks: list[tuple[list[int], list[tuple[int, int]] | None]] = []
    seg_tokens: list[int] = []
    for seg in segments:
        ids, offs = tokenize(seg.text)
        toks.append((list(ids), offs))
        seg_tokens.append(len(ids))

    runs: list[list[int]] = []  # segment indices per run
    for i, seg in enumerate(segments):
        if structure_aware and runs and segments[runs[-1][-1]].kind == seg.kind:
            runs[-1].append(i)
        elif not structure_aware and runs:
            runs[-1].append(i)
        else:
            runs.append([i])

    chunks: list[Chunk] = []
    for run in runs:
        stream: list[tuple[int, int]] = []  # (segment, token index) per position
        boundaries: list[int] = []
        for si in run:
            if seg_tokens[si] and stream and structure_aware:
                boundaries.append(len(stream))
            stream.extend((si, t) for t in range(seg_tokens[si]))
        if not stream:
            continue
        for s, e in plan_windows(len(stream), boundaries, window, min_overlap):
            spans: list[tuple[int, int, int]] = []
            for si, t in stream[s:e]:
                if spans and spans[-1][0] == si:
                    spans[-1] = (si, spans[-1][1], t + 1)
                else:
                    spans.append((si, t, t + 1))
            ids = [tid for si, a, b in spans for tid in toks[si][0][a:b]]
            texts = []
            for si, a, b in spans:
                offs = toks[si][1]
                texts.append(None if offs is None else segments[si].text[offs[a][0]: offs[b - 1][1]])
            text = None if any(t is None for t in texts) else "\n\n".join(texts)
            chunks.append(Chunk(ids=ids, text=text, kind=segments[run[0]].kind, spans=spans,
                                parts=[] if text is None else texts))
    if not chunks:  # empty or whitespace-only input: one empty chunk, so there is still something to score
        chunks.append(Chunk(ids=[], text="", kind="body", spans=[]))
    return ChunkSet(chunks=chunks, seg_tokens=seg_tokens, segments=segments)

"""Model-independent input construction for BERT-style detectors.

The window logic is copied from benchmarks/detector_baseline/encoding.py (serving must not import from
benchmarks/) and extended with a per-request chunk cap. Every model input is [CLS] + content + [SEP].
Chunking works on token ids directly (no decode/re-encode round trip).
"""
from dataclasses import dataclass


def truncate_ids(content_ids: list[int], cls_id: int, sep_id: int, max_len: int) -> tuple[list[int], bool]:
    """Return ([CLS] + first (max_len - 2) content ids + [SEP], truncated?)."""
    room = max_len - 2
    return [cls_id, *content_ids[:room], sep_id], len(content_ids) > room


def window_starts(n_content: int, window: int, stride: int) -> list[int]:
    """Start offsets of overlapping windows covering every content id, including the tail."""
    if not 0 < stride <= window:
        raise ValueError("stride must be in (0, window]")
    if n_content <= window:
        return [0]
    starts = list(range(0, n_content - window, stride))
    starts.append(n_content - window)  # last window ends exactly at the tail
    return starts


def chunk_ids(content_ids: list[int], cls_id: int, sep_id: int, max_len: int, stride: int) -> list[list[int]]:
    """Overlapping windows of (max_len - 2) content ids, each wrapped in [CLS] ... [SEP]."""
    window = max_len - 2
    return [[cls_id, *content_ids[s : s + window], sep_id] for s in window_starts(len(content_ids), window, stride)]


@dataclass(frozen=True)
class ChunkPlan:
    chunks: list[list[int]]  # model inputs, each [CLS] ... [SEP]
    n_windows: int  # windows needed to cover the whole input
    truncated: bool  # True iff some content id is in none of the selected chunks


def select_starts(starts: list[int], max_chunks: int) -> list[int]:
    """Keep at most max_chunks windows: the first (max_chunks - 1) plus the last one.

    Keeping the tail window means an injection appended to the end of a long input is still scored
    when the cap applies; the middle of the input is what gets dropped. With max_chunks == 1 this is
    plain head truncation.
    """
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")
    if len(starts) <= max_chunks:
        return starts
    if max_chunks == 1:
        return starts[:1]
    return [*starts[: max_chunks - 1], starts[-1]]


def covers(starts: list[int], window: int, n_content: int) -> bool:
    """True iff the windows [s, s + window) cover every index in [0, n_content)."""
    reach = 0
    for s in sorted(starts):
        if s > reach:
            return False
        reach = max(reach, s + window)
    return reach >= n_content


def plan_chunks(content_ids: list[int], cls_id: int, sep_id: int, max_len: int, stride: int,
                max_chunks: int) -> ChunkPlan:
    window = max_len - 2
    if window < 1:
        raise ValueError("max_len must be >= 3")
    all_starts = window_starts(len(content_ids), window, stride)
    starts = select_starts(all_starts, max_chunks)
    chunks = [[cls_id, *content_ids[s : s + window], sep_id] for s in starts]
    return ChunkPlan(chunks=chunks, n_windows=len(all_starts),
                     truncated=not covers(starts, window, len(content_ids)))

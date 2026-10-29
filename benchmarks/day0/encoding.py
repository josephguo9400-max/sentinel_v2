"""Model-independent input construction for BERT-style detectors.

Both the truncated and the chunked path produce [CLS] + content + [SEP], so they feed the model the
same way. Chunking works on token ids directly (no decode/re-encode round trip).
"""


def truncate_ids(content_ids: list[int], cls_id: int, sep_id: int, max_len: int) -> tuple[list[int], bool]:
    """Return ([CLS] + first (max_len - 2) content ids + [SEP], truncated?)."""
    room = max_len - 2
    return [cls_id, *content_ids[:room], sep_id], len(content_ids) > room


def chunk_ids(content_ids: list[int], cls_id: int, sep_id: int, max_len: int, stride: int) -> list[list[int]]:
    """Overlapping windows of (max_len - 2) content ids, each wrapped in [CLS] ... [SEP].

    Every content id appears in at least one window, including the tail.
    """
    window = max_len - 2
    if not 0 < stride <= window:
        raise ValueError("stride must be in (0, max_len - 2]")
    if len(content_ids) <= window:
        return [[cls_id, *content_ids, sep_id]]
    starts = list(range(0, len(content_ids) - window, stride))
    starts.append(len(content_ids) - window)  # last window ends exactly at the tail
    return [[cls_id, *content_ids[s : s + window], sep_id] for s in starts]

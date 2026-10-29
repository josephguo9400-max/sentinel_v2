import pytest

from benchmarks.day0.encoding import chunk_ids, truncate_ids

CLS, SEP = 1, 2


def test_truncate_keeps_sep_and_flags_truncation():
    ids, truncated = truncate_ids(list(range(10, 1010)), CLS, SEP, 512)
    assert ids[0] == CLS and ids[-1] == SEP and len(ids) == 512 and truncated


def test_short_input_not_truncated():
    ids, truncated = truncate_ids([10, 11], CLS, SEP, 512)
    assert ids == [CLS, 10, 11, SEP] and not truncated


@pytest.mark.parametrize("n", [1, 509, 510, 511, 700, 1021, 1022, 4999])
@pytest.mark.parametrize("stride", [128, 384, 510])
def test_chunks_cover_every_token_including_tail(n, stride):
    content = list(range(100, 100 + n))
    chunks = chunk_ids(content, CLS, SEP, 512, stride)
    seen = set()
    for c in chunks:
        assert c[0] == CLS and c[-1] == SEP and len(c) <= 512
        seen.update(c[1:-1])
    assert seen == set(content)
    assert chunks[-1][-2] == content[-1]


def test_invalid_stride_rejected():
    with pytest.raises(ValueError):
        chunk_ids(list(range(1000)), CLS, SEP, 512, 0)
    with pytest.raises(ValueError):
        chunk_ids(list(range(1000)), CLS, SEP, 512, 511)

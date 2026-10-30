import pytest

from serving.encoding import chunk_ids, covers, plan_chunks, select_starts, truncate_ids, window_starts

CLS, SEP = 1, 2


def test_truncate_keeps_sep_and_flags_truncation():
    ids, truncated = truncate_ids(list(range(10, 1010)), CLS, SEP, 512)
    assert ids[0] == CLS and ids[-1] == SEP and len(ids) == 512 and truncated


def test_short_input_not_truncated():
    ids, truncated = truncate_ids([10, 11], CLS, SEP, 512)
    assert ids == [CLS, 10, 11, SEP] and not truncated


@pytest.mark.parametrize("n", [0, 1, 509, 510, 511, 700, 1021, 1022, 4999])
@pytest.mark.parametrize("stride", [128, 384, 510])
def test_chunks_cover_every_token_including_tail(n, stride):
    content = list(range(100, 100 + n))
    chunks = chunk_ids(content, CLS, SEP, 512, stride)
    seen = set()
    for c in chunks:
        assert c[0] == CLS and c[-1] == SEP and len(c) <= 512
        seen.update(c[1:-1])
    assert seen == set(content)
    if n:
        assert chunks[-1][-2] == content[-1]


def test_chunk_geometry_510_384():
    assert window_starts(510, 510, 384) == [0]
    assert window_starts(511, 510, 384) == [0, 1]
    assert window_starts(1000, 510, 384) == [0, 384, 490]


def test_invalid_stride_rejected():
    with pytest.raises(ValueError):
        chunk_ids(list(range(1000)), CLS, SEP, 512, 0)
    with pytest.raises(ValueError):
        chunk_ids(list(range(1000)), CLS, SEP, 512, 511)


def test_plan_uncapped_is_not_truncated():
    plan = plan_chunks(list(range(100, 1100)), CLS, SEP, 512, 384, max_chunks=10)
    assert plan.n_windows == 3 and len(plan.chunks) == 3 and not plan.truncated


def test_plan_cap_drops_middle_keeps_tail_and_flags_truncation():
    content = list(range(100, 5100))
    plan = plan_chunks(content, CLS, SEP, 512, 384, max_chunks=3)
    assert plan.n_windows > 3 and len(plan.chunks) == 3 and plan.truncated
    assert plan.chunks[0][1] == content[0] and plan.chunks[-1][-2] == content[-1]


def test_plan_cap_one_is_head_truncation():
    content = list(range(100, 1100))
    plan = plan_chunks(content, CLS, SEP, 512, 384, max_chunks=1)
    assert plan.chunks == [truncate_ids(content, CLS, SEP, 512)[0]] and plan.truncated


def test_cap_that_still_covers_is_not_truncated():
    # 2 windows needed, cap 2: nothing dropped.
    plan = plan_chunks(list(range(100, 800)), CLS, SEP, 512, 384, max_chunks=2)
    assert len(plan.chunks) == 2 and not plan.truncated


@pytest.mark.parametrize("n", range(1, 3000, 97))
@pytest.mark.parametrize("cap", [1, 2, 3, 5])
def test_truncated_flag_matches_actual_coverage(n, cap):
    content = list(range(100, 100 + n))
    plan = plan_chunks(content, CLS, SEP, 512, 384, cap)
    seen = {t for c in plan.chunks for t in c[1:-1]}
    assert plan.truncated == (seen != set(content))
    assert len(plan.chunks) <= cap


def test_select_and_covers_helpers():
    assert select_starts([0, 1, 2, 3], 2) == [0, 3]
    assert covers([0, 384], 510, 894) and not covers([0, 490], 384, 874)
    with pytest.raises(ValueError):
        select_starts([0], 0)

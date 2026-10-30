"""Traffic replay set (replay-v1): ~2,000 items in none of train / val / frozen_test, with delayed labels and a
language-drift schedule. Systems measurements (shadow agreement, latency, errors, drift visibility), not a test set.

    python -m training.replay --fetch      # new raw sources -> data/raw/ (gitignored), sha256-verified
    flock /home/claude/measure.lock -c 'python -m training.replay'   # -> data/replay/replay-v1.jsonl + MANIFEST.json

Sources (none used by the splits; licenses from each card at the pinned revision, see data/replay/NOTICE.md):
  mosscap          Lakera/mosscap_prompt_injection (MIT), test split: game prompts, NOT labeled (card: "many of
                   them are not actual prompt injections") -> label null
  gandalf_summ     Lakera/gandalf_summarization (MIT): indirect injections hidden in text to summarize -> label 1
  itw_jailbreak    TrustAIRLab/in-the-wild-jailbreak-prompts (MIT), jailbreak_2023_12_25 -> label 1
                   (a jailbreak is not an injection; same caveat as jackhhao in DATA.md)
  itw_regular      same repo, regular_2023_12_25 (ordinary prompts) -> label 0
  aya              CohereLabs/aya_dataset (Apache-2.0), test split: human-written prompts in 7 languages -> label 0
  replay_email     training/replay_emails.py (new templates, generator replay-emails-v1): English benign emails
                   and hand-written de/fr/es/it/pt benign and injection emails
  sandbox_dev      fixtures/scenarios/dev/*.json email bodies (Agent C's dev scenarios, read only); label 1 for the
                   scenario's attacker_email_id. fixtures/scenarios/heldout is never read.
No item is drawn from train, val or frozen_test.

Leakage: every candidate is deduplicated, exactly (dedup_key) and near (char-5gram MinHash, verified Jaccard >= 0.8,
same definition as build_splits), against every record of train, val and frozen_test. frozen_test.jsonl is read
only to compute those keys; no text, label or metric of it is used or written. Synthetic emails must also share
no benign-hard sentence (> 30 chars, exact or skeleton) with val or frozen benign-hard records. Inside the replay
set exact duplicates are removed; near duplicates are removed except between synthetic template emails, which
repeat by design (as templated mail does in real traffic) and are counted in the manifest.
check_leakage re-checks all of this from the committed files.

Each item: id, text, source, lang (langdetect, same settings as the splits), label (0 / 1 / null),
label_available_after_s (seconds after the item's arrival when its label may be used; null = never revealed),
arrival_offset_s (nominal schedule at NOMINAL_RPS, Poisson arrivals), seq, plus provenance fields.
Selection: replay_email, sandbox_dev and gandalf_summ are included in full; the rest is a seeded random draw.
Drift: the target non-English share rises linearly from 5% at the first item to 40% at the last; the order is
built greedily so the running non-English count tracks that target. Delayed labels: a labeled item gets its label
revealed with probability LABEL_REVEAL_P, after a delay uniform in LABEL_DELAY_S (nominal seconds; the runner scales
offsets and delays by NOMINAL_RPS / rps). Every random choice is seeded (sha256 of SEED and the item).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from training.common import (RAW_DIR, ROOT, SPLIT_NAMES, SPLITS_DIR, clean_text, dedup_key, key_hash,
                             near_duplicate_pairs, read_jsonl, sha256_file, stable_unit, write_json, write_jsonl)
from training.sources import RawFile, Source

REPLAY_VERSION = "replay-v1"
REPLAY_DIR = ROOT / "data" / "replay"
OUT = REPLAY_DIR / f"{REPLAY_VERSION}.jsonl"
SEED = "replay-v1:20261030"
N_ITEMS = 2000
NON_EN_START, NON_EN_END = 0.05, 0.40
NOMINAL_RPS = 10.0
LABEL_REVEAL_P = 0.30
LABEL_DELAY_S = (30.0, 120.0)
MAX_CHARS = 20_000  # longer texts are dropped (not cut) so stored text is always a real record
# Candidate pool sizes before dedup (sampled by seeded hash); the drift schedule then picks N_ITEMS.
POOL = {"mosscap": 700, "gandalf_summ": 140, "itw_jailbreak": 260, "itw_regular": 420, "aya": 1750}
N_EMAIL_EN = 320
N_EMAIL_ML_PER_LANG_KIND = 14


def _load_mosscap(src: Source) -> list[dict]:
    import pandas as pd

    (p,) = [p for p in src.verified() if p.suffix == ".parquet"]
    df = pd.read_parquet(p)
    return [{"text": t, "label": None, "source_label": f"level={lv}(unlabeled)"} for t, lv in zip(df["prompt"], df["level"])]


def _load_gandalf_summ(src: Source) -> list[dict]:
    import pandas as pd

    rows = []
    for p in src.verified():
        if p.suffix == ".parquet":
            rows += [{"text": t, "label": 1, "source_label": "gandalf_summarization(all positive)"}
                     for t in pd.read_parquet(p)["text"]]
    return rows


def _load_itw(config: str, label: int):
    def load(src: Source) -> list[dict]:
        import pandas as pd

        (p,) = [p for p in src.verified() if p.suffix == ".parquet" and config in str(p)]
        df = pd.read_parquet(p)
        return [{"text": t, "label": label, "source_label": f"{config}:jailbreak={bool(j)}:platform={pl}"}
                for t, j, pl in zip(df["prompt"], df["jailbreak"], df["platform"])]
    return load


def _load_aya(src: Source) -> list[dict]:
    import pandas as pd

    (p,) = [p for p in src.verified() if p.suffix == ".parquet"]
    df = pd.read_parquet(p)
    return [{"text": t, "label": 0, "source_label": f"aya:language_code={c}"} for t, c in zip(df["inputs"], df["language_code"])]


_ITW_REPO, _ITW_REV = "TrustAIRLab/in-the-wild-jailbreak-prompts", "a10aab8eff1c73165a442d4464dce192bd28b9c5"
REPLAY_SOURCES: dict[str, Source] = {s.name: s for s in [
    Source(name="mosscap", repo_id="Lakera/mosscap_prompt_injection",
           revision="b7e495ff63373ff7f7dabc1e9390cf62b5838570", license="mit", redistributable=True,
           files=(RawFile("data/test-00000-of-00001-06acab2d6a2bcf9f.parquet",
                          "9311f64ca579c947e43ff64ad4996585c51e4dae8ae747342926cf0091160d21"),),
           stage2_overlap="unknown", label_definition="unlabeled game prompts (null)", loader=_load_mosscap),
    Source(name="gandalf_summ", repo_id="Lakera/gandalf_summarization",
           revision="8e213cc8dd65f099c61dc32dbcfdb2ff6b1c92d8", license="mit", redistributable=True,
           files=(RawFile("data/train-00000-of-00001-44378fdb62cbf9b7.parquet",
                          "8d160db880b4eb8efd8e6dc5425f10466ba594fa32da8aa6da4b2ceb22899214"),
                  RawFile("data/validation-00000-of-00001-c7b174a69fca3472.parquet",
                          "d1a29b60a10228db16e0cf75dae9cd9be9904f905eefdaa5f9a48a0f557ea52b"),
                  RawFile("data/test-00000-of-00001-1696577a5023b698.parquet",
                          "cf63d8ee1a9a4c0ce45f53f2e6702d526cf35544bb4b4c86f746503923dc43b1")),
           stage2_overlap="unknown", label_definition="successful indirect injections (all 1)",
           loader=_load_gandalf_summ),
    Source(name="itw_jailbreak", repo_id=_ITW_REPO, revision=_ITW_REV, license="mit", redistributable=True,
           files=(RawFile("jailbreak_2023_12_25/train-00000-of-00001.parquet",
                          "55bbc5e2be771ee16ed67aed4edb5ecef5f44ac69cea317a67e5405c5c6057a4"),),
           stage2_overlap="unknown", label_definition="jailbreak prompts (1; jailbreak != injection)",
           loader=_load_itw("jailbreak_2023_12_25", 1)),
    Source(name="itw_regular", repo_id=_ITW_REPO, revision=_ITW_REV, license="mit", redistributable=True,
           files=(RawFile("regular_2023_12_25/train-00000-of-00001.parquet",
                          "23e90010ccbc85574bf46678528cddce5b02abc86d9a047209218af65cb37c68"),),
           stage2_overlap="unknown", label_definition="regular prompts (0)", loader=_load_itw("regular_2023_12_25", 0)),
    Source(name="aya", repo_id="CohereLabs/aya_dataset", revision="f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
           license="apache-2.0", redistributable=True,
           files=(RawFile("data/test-00000-of-00001.parquet",
                          "52bbb79461bff9887c9987a5737d9a06f5ec6c78c08d770828c37fc182ce2cca"),),
           stage2_overlap="unknown", label_definition="human-written instruction prompts, benign (0)", loader=_load_aya),
]}


def fetch() -> None:
    from huggingface_hub import hf_hub_download

    for src in REPLAY_SOURCES.values():
        for f in src.files:
            hf_hub_download(src.repo_id, f.path, repo_type="dataset", revision=src.revision,
                            local_dir=RAW_DIR / src.repo_id.replace("/", "__"))
        src.verified()
        print(f"fetched {src.repo_id}@{src.revision[:10]} (sha256 verified)")


def _lang(text: str) -> str:
    from training.build_splits import _lang as split_lang

    return split_lang(text)


def sandbox_dev_emails(root: Path = ROOT) -> list[dict]:
    out = []
    for p in sorted((root / "fixtures" / "scenarios" / "dev").glob("*.json")):
        sc = json.loads(p.read_text())
        for e in sc["emails"]:
            out.append({"text": e["body"], "label": int(e["id"] == sc.get("attacker_email_id")),
                        "source_label": f"{sc['scenario_id']}:{e['id']}", "synthetic": True})
    return out


def candidates() -> list[dict]:
    """All candidate records in a fixed order, each with source / label / text (cleaned)."""
    from training import replay_emails

    out = []
    for name, src in REPLAY_SOURCES.items():
        rows = []
        for i, r in enumerate(src.loader(src)):
            t = clean_text(r["text"])
            if t is None or len(t) > MAX_CHARS:
                continue
            rows.append({**r, "text": t, "source": name, "source_index": i})
        rows.sort(key=lambda r: stable_unit(f"{name}:{r['source_index']}", SEED))
        out += rows[: POOL[name]]
    for i, r in enumerate(replay_emails.generate(SEED, N_EMAIL_EN, N_EMAIL_ML_PER_LANG_KIND)):
        out.append({"text": r["text"], "label": r["label"], "source": "replay_email", "source_index": i,
                    "source_label": f"{r['family']}:{r['lang_hint']}", "family": r["family"], "synthetic": True,
                    "generator": r["generator"]})
    for i, r in enumerate(sandbox_dev_emails()):
        out.append({**r, "source": "sandbox_dev", "source_index": i})
    return out


def split_records(splits_dir: Path = SPLITS_DIR) -> list[dict]:
    """(split, id, text) for every split record. Only dedup keys and benign-hard sentences are derived."""
    recs = []
    for s in SPLIT_NAMES:
        for r in read_jsonl(splits_dir / f"{s}.jsonl"):
            recs.append({"split": s, "id": r["id"], "text": r["text"], "source": r.get("source")})
    return recs


def dedup(cands: list[dict], split_recs: list[dict]) -> tuple[list[dict], dict]:
    """Drop candidates that duplicate a split record (exact or near) or another kept candidate."""
    from training.benign_hard import sentence_skeletons, sentences

    stats = Counter()
    split_keys = {dedup_key(r["text"]) for r in split_recs}
    bh = [r["text"] for r in split_recs if r["source"] == "benign_hard"]
    bh_sent = {x for t in bh for x in sentences(t)}
    bh_skel = {x for t in bh for x in sentence_skeletons(t)}
    keep, seen = [], set()
    for c in cands:
        k = dedup_key(c["text"])
        if not k:
            stats["empty_key"] += 1
            continue
        if k in split_keys:
            stats[f"exact_split:{c['source']}"] += 1
            continue
        if k in seen:
            stats[f"exact_within:{c['source']}"] += 1
            continue
        if c.get("synthetic") and (set(sentences(c["text"])) & bh_sent or sentence_skeletons(c["text"]) & bh_skel):
            stats[f"benign_hard_sentence:{c['source']}"] += 1
            continue
        seen.add(k)
        keep.append(c)
    # Near duplicates: one MinHash pass over split records + kept candidates.
    n_split = len(split_recs)
    keys = [dedup_key(r["text"]) for r in split_recs] + [dedup_key(c["text"]) for c in keep]
    drop = set()
    within_synthetic = 0
    for i, j, _ in near_duplicate_pairs(keys):
        if i < n_split and j < n_split:
            continue
        if i < n_split <= j:
            drop.add(j - n_split)
            stats[f"near_split:{keep[j - n_split]['source']}"] += 1
            continue
        a, b = keep[i - n_split], keep[j - n_split]
        if a.get("synthetic") and b.get("synthetic"):
            within_synthetic += 1
            continue
        drop.add(j - n_split)  # keep the earlier candidate
        stats[f"near_within:{b['source']}"] += 1
    kept = [c for idx, c in enumerate(keep) if idx not in drop]
    stats["near_within_synthetic_pairs_kept"] = within_synthetic
    return kept, dict(sorted(stats.items()))


def target_non_en(i: int, n: int) -> float:
    return NON_EN_START + (NON_EN_END - NON_EN_START) * (i / (n - 1) if n > 1 else 0.0)


MUST_SOURCES = ("replay_email", "sandbox_dev", "gandalf_summ")  # scarce sources: always included in full


def _pick(pool: list[dict], k: int, what: str) -> list[dict]:
    must = [x for x in pool if x["source"] in MUST_SOURCES]
    rest = sorted([x for x in pool if x["source"] not in MUST_SOURCES],
                  key=lambda x: stable_unit("pick:" + x["id"], SEED))
    if len(must) > k or len(must) + len(rest) < k:
        raise SystemExit(f"cannot pick {k} {what} items from {len(must)} required + {len(rest)} others")
    chosen = must + rest[: k - len(must)]
    return sorted(chosen, key=lambda x: stable_unit("order:" + x["id"], SEED))


def schedule(items: list[dict], n: int = N_ITEMS) -> list[dict]:
    """Pick n items and order them so the running non-English share tracks the linear 5% -> 40% target.

    Items of MUST_SOURCES are always included; the remainder is a seeded random draw. Within each language
    group the order is a seeded shuffle, so sources are mixed evenly along the run.
    """
    need_ne = round(sum(target_non_en(i, n) for i in range(n)))
    ne = _pick([x for x in items if x["lang"] != "en"], need_ne, "non-English")
    en = _pick([x for x in items if x["lang"] == "en"], n - need_ne, "English")
    out, n_ne, cum = [], 0, 0.0
    ei = ni = 0
    for i in range(n):
        cum += target_non_en(i, n)
        if (n_ne + 0.5 <= cum and ni < len(ne)) or ei >= len(en):
            out.append(ne[ni]); ni += 1; n_ne += 1
        else:
            out.append(en[ei]); ei += 1
    return out


def assign_timing(seq: list[dict]) -> None:
    t = 0.0
    for i, x in enumerate(seq):
        if i:
            u = stable_unit(f"gap:{x['id']}", SEED)
            t += -math.log(1.0 - u) / NOMINAL_RPS
        x["seq"] = i
        x["arrival_offset_s"] = round(t, 4)
        if x["label"] is not None and stable_unit(f"reveal:{x['id']}", SEED) < LABEL_REVEAL_P:
            lo, hi = LABEL_DELAY_S
            x["label_available_after_s"] = round(lo + (hi - lo) * stable_unit(f"delay:{x['id']}", SEED), 3)
        else:
            x["label_available_after_s"] = None


def build(out: Path = OUT, splits_dir: Path = SPLITS_DIR, n: int = N_ITEMS) -> dict:
    cands = candidates()
    kept, stats = dedup(cands, split_records(splits_dir))
    items = []
    for c in kept:
        x = {"id": "rp-" + key_hash(c["text"])[:20], "text": c["text"], "source": c["source"],
             "source_label": c.get("source_label"), "label": c["label"], "lang": _lang(c["text"]),
             "synthetic": bool(c.get("synthetic", False))}
        if c.get("family"):
            x["family"] = c["family"]
        if c.get("generator"):
            x["generator"] = c["generator"]
        items.append(x)
    seq = schedule(items, n)
    assign_timing(seq)
    write_jsonl(out, seq)
    windows = []
    for w in range(10):
        part = seq[w * n // 10:(w + 1) * n // 10]
        windows.append(round(sum(x["lang"] != "en" for x in part) / len(part), 4))
    comp = Counter((x["source"], x["label"]) for x in seq)
    manifest = {
        "replay_version": REPLAY_VERSION, "seed": SEED, "n_items": len(seq), "file": out.name,
        "sha256": sha256_file(out),
        "purpose": "systems measurements (shadow agreement, latency, errors, drift visibility); NOT an accuracy "
                   "test set: labels are partial, mosscap is unlabeled, and sources differ from the gate's",
        "drawn_from_train_val_or_frozen": False,
        "sources": {k: {"repo_id": v.repo_id, "revision": v.revision, "license": v.license,
                        "files": {f.path: f.sha256 for f in v.files}, "label_definition": v.label_definition,
                        "stage2_overlap": v.stage2_overlap} for k, v in REPLAY_SOURCES.items()}
                   | {"replay_email": {"generator": "training/replay_emails.py replay-emails-v1", "license": "this repo"},
                      "sandbox_dev": {"path": "fixtures/scenarios/dev/*.json", "license": "this repo"}},
        "candidate_pool": dict(Counter(c["source"] for c in cands)),
        "dedup": {"against": "train, val, frozen_test (exact dedup_key + MinHash near >= 0.8) and benign-hard "
                             "sentences of val/frozen", "removed": stats},
        "composition": {f"{s}:label={lab}": v for (s, lab), v in sorted(comp.items(), key=str)},
        "by_source": dict(sorted(Counter(x["source"] for x in seq).items())),
        "by_lang": dict(Counter(x["lang"] for x in seq).most_common()),
        "labels": {"positive": sum(x["label"] == 1 for x in seq), "negative": sum(x["label"] == 0 for x in seq),
                   "null": sum(x["label"] is None for x in seq),
                   "revealed": sum(x["label_available_after_s"] is not None for x in seq)},
        "drift": {"schedule": f"target non-English share linear {NON_EN_START} -> {NON_EN_END} over seq",
                  "non_en_share_by_tenth": windows},
        "timing": {"nominal_rps": NOMINAL_RPS, "arrivals": "Poisson (exponential gaps) at nominal_rps",
                   "duration_s_nominal": seq[-1]["arrival_offset_s"],
                   "label_reveal_probability": LABEL_REVEAL_P, "label_delay_s_nominal": list(LABEL_DELAY_S),
                   "scaling": "a runner at rate R multiplies arrival offsets and label delays by nominal_rps / R"},
        "max_chars": MAX_CHARS,
    }
    write_json(out.parent / "MANIFEST.json", manifest)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args(argv)
    if a.fetch:
        fetch()
        return 0
    m = build()
    print(json.dumps({k: m[k] for k in ("n_items", "by_source", "labels", "drift")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

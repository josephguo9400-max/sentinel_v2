"""Empirical overlap probe: our public sources vs the datasets the Stage 2 model card names.

The Stage 2 card (protectai/deberta-v3-base-prompt-injection-v2 @ 90c9989b) names 7 datasets but says it
used 22; 15 are unnamed. This script can only check the named, publicly downloadable ones. A "none
found" result therefore does NOT prove that Stage 2 never saw a source.

Fetch the references (about 200 MB, gitignored) and run:
    python -m training.stage2_overlap --fetch
    python -m training.stage2_overlap
Output: data/stage2_overlap.json (counts only, no raw text).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from datasketch import MinHashLSH

from training.common import (NEAR_DUP_JACCARD, NUM_PERM, RAW_DIR, ROOT, clean_text, dedup_key, jaccard,
                             key_hash, minhashes, sha256_file, shingles, write_json)
from training.sources import SOURCES, load_all

REF_DIR = RAW_DIR / "stage2_named"
OUT = ROOT / "data" / "stage2_overlap.json"
# sha256 of dedup_key(text) for every record of ours that matched; build_splits excludes these from frozen test.
KEYS_OUT = ROOT / "data" / "stage2_overlap_keys.txt"

# (repo_id, revision, file, sha256) of the named Stage 2 datasets that are publicly reachable.
REFERENCES = [
    ("alespalla/chatbot_instruction_prompts", "6ea5ce09647e5451860801a46db6a46b28e259bf",
     "data/train-00000-of-00001-d3450385c0ae3f98.parquet",
     "d2b26dc88cef5844ec60fd54e5e9ffdc7c5cc76d0708d8b0ba1dbad451532cc0"),
    ("alespalla/chatbot_instruction_prompts", "6ea5ce09647e5451860801a46db6a46b28e259bf",
     "data/test-00000-of-00001-8d5da67d5c6856ed.parquet",
     "0b6e57d49470b63b3f195fbc15db7ebffb88a8bc5f4eafc290179a918a6f02fe"),
    ("VMware/open-instruct", "9e53e85f37988e3bfbebbee4dc9ad4942cfd37fa",
     "data/train-00000-of-00001-c6f4e090ee7100b6.parquet",
     "9ac59527bdbeee747ce4de6abdff89a7e4678aa339cb4063c27812df22098f4d"),
    ("HuggingFaceH4/grok-conversation-harmless", "01185ad52860775c65cbd6d0c9254ac8e9089fe0",
     "data/train_sft-00000-of-00001-71f7b2a54cfc90b9.parquet",
     "816145a51c8d5baeec39d312b2325924391b71547f4e36e00d2e7036b7706b92"),
    ("HuggingFaceH4/grok-conversation-harmless", "01185ad52860775c65cbd6d0c9254ac8e9089fe0",
     "data/test_sft-00000-of-00001-8e8f87780d2fd798.parquet",
     "3766c8d98d76f27fceb962007ae833201faf328002db1add7963a4c88320d6dc"),
    ("OpenSafetyLab/Salad-Data", "d21a325e276a99bd69b1fbb8aa51a9f249486b72", "base_set.json",
     "0987630d953961ac9d9e3e272db72dea20079aa29df769378fa96e9f653f2ca1"),
    ("OpenSafetyLab/Salad-Data", "d21a325e276a99bd69b1fbb8aa51a9f249486b72", "attack_enhanced_set.json",
     "389da12f029a7f0c4a562f753e1dd5e53aed46d873ef1f568919d9f4b72e7062"),
    ("OpenSafetyLab/Salad-Data", "d21a325e276a99bd69b1fbb8aa51a9f249486b72", "defense_enhanced_set.json",
     "df1434a7e2c2057d7fe0a7b34f86f0eb3ec368aad998f9800fa3f46025c95ee0"),
    ("natolambert/xstest-v2-copy", "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
     "data/prompts-00000-of-00001.parquet",
     "322d4e89df9fb419c296d5b360067f3265845d40a561a37d9be77a078d219522"),
]
# Named in the card but not checkable here.
UNREACHABLE = {"Harelix/Prompt-Injection-Mixed-Techniques-2024": "HF API returns 401 (removed or private)"}
# jackhhao/jailbreak-classification is also named; it is one of our sources, so its overlap is confirmed by name.


def _path(repo, f):
    return REF_DIR / repo.replace("/", "__") / f


def fetch():
    from huggingface_hub import hf_hub_download

    for repo, rev, f, digest in REFERENCES:
        hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=REF_DIR / repo.replace("/", "__"))
        assert sha256_file(_path(repo, f)) == digest, f"sha256 mismatch {repo}/{f}"
        print("fetched", repo, f)


def ref_texts(repo, f) -> list[str]:
    import pandas as pd

    p = _path(repo, f)
    if p.suffix == ".json":
        rows = json.loads(p.read_text())
        cols = [c for c in ("question", "augq", "daugq") if c in rows[0]]
        return [r[c] for r in rows for c in cols]
    df = pd.read_parquet(p)
    if repo.startswith("alespalla"):
        return list(df["prompt"])
    if repo.startswith("VMware"):
        return list(df["instruction"])
    if repo.startswith("HuggingFaceH4"):
        return list(df["prompt"])
    if repo.startswith("natolambert"):
        return list(df["prompt"])
    raise ValueError(repo)


def main():
    ours = load_all()
    keys = [dedup_key(r["text"]) for r in ours]
    hashes = {}
    for i, r in enumerate(ours):
        hashes.setdefault(key_hash(r["text"]), []).append(i)
    sets = [shingles(k) for k in keys]
    lsh = MinHashLSH(threshold=NEAR_DUP_JACCARD, num_perm=NUM_PERM)
    for i, m in enumerate(minhashes(sets)):
        lsh.insert(str(i), m)

    exact_hit, near_hit = defaultdict(set), defaultdict(set)
    per_ref = {}
    for repo, _rev, f, digest in REFERENCES:
        assert sha256_file(_path(repo, f)) == digest, f"sha256 mismatch {repo}/{f}"
        texts = [t for t in (clean_text(x) for x in ref_texts(repo, f)) if t]
        rkeys = [dedup_key(t) for t in texts]
        rsets = [shingles(k) for k in rkeys]
        n_exact = n_near = 0
        for t, rs, m in zip(texts, rsets, minhashes(rsets)):
            for i in hashes.get(key_hash(t), []):
                exact_hit[repo].add(i)
                n_exact += 1
            for c in lsh.query(m):
                i = int(c)
                if jaccard(sets[i], rs) >= NEAR_DUP_JACCARD:
                    near_hit[repo].add(i)
                    n_near += 1
        per_ref[f"{repo}/{f}"] = {"n_reference_texts": len(texts)}
        print(repo, f, len(texts), "texts")

    by_source = {}
    for name in SOURCES:
        idx = {i for i, r in enumerate(ours) if r["source"] == name}
        entry = {"n_records": len(idx), "matches": {}}
        for repo in sorted({r[0] for r in REFERENCES}):
            e, n = exact_hit[repo] & idx, near_hit[repo] & idx
            if e or n:
                entry["matches"][repo] = {
                    "exact": len(e), "near_or_exact": len(n | e),
                    "by_label": {str(lab): sum(1 for i in (n | e) if ours[i]["label"] == lab) for lab in (0, 1)},
                }
        by_source[name] = entry
    matched = sorted({key_hash(ours[i]["text"]) for d in (exact_hit, near_hit) for s in d.values() for i in s})
    KEYS_OUT.write_text("".join(h + "\n" for h in matched))
    write_json(OUT, {
        "description": "Records of our sources that exactly or near-duplicate (char-5gram Jaccard >= "
                       f"{NEAR_DUP_JACCARD}) a text in a dataset named in the Stage 2 model card. Counts are "
                       "pre-dedup records.",
        "stage2_model": "protectai/deberta-v3-base-prompt-injection-v2@90c9989b1a342275dd0d1a95aad283c04e075671",
        "references_checked": per_ref,
        "named_but_unreachable": UNREACHABLE,
        "named_and_used_by_us": ["jackhhao/jailbreak-classification"],
        "unnamed_datasets_in_card": 15,
        "by_source": by_source,
    })
    print(json.dumps(by_source, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    fetch() if a.fetch else main()

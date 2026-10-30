"""Pinned public data sources: revision, file sha256, license, label mapping, Stage 2 overlap status.

Fetch raw files (into the gitignored data/raw/):  python -m training.sources --fetch
Every loader verifies the sha256 of each raw file before reading it.

See training/DATA.md for the audit behind every field here.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from training.common import RAW_DIR, clean_text, sha256_file


@dataclass(frozen=True)
class RawFile:
    path: str  # path inside the HF repo
    sha256: str


@dataclass(frozen=True)
class Source:
    name: str  # short name used in records
    repo_id: str
    revision: str
    license: str
    redistributable: bool
    files: tuple[RawFile, ...]
    # Source-level tier (record-level matches are tracked separately as stage2_named_match):
    # "confirmed":  named in the Stage 2 (v2) model card.
    # "suspected":  named in the v1 card (gandalf), or treated as suspect as a precaution (deepset, the most
    #               widely used public set; see DATA.md section 2).
    # "none_known": not named in either card. This does NOT prove Stage 2 never saw it.
    stage2_overlap: str
    label_definition: str
    loader: Callable[["Source"], list[dict]] = field(repr=False, compare=False, default=None)

    def local(self, f: RawFile) -> Path:
        return RAW_DIR / self.repo_id.replace("/", "__") / f.path

    def verified(self) -> list[Path]:
        out = []
        for f in self.files:
            p = self.local(f)
            if not p.exists():
                raise FileNotFoundError(f"{p} missing; run: python -m training.sources --fetch")
            digest = sha256_file(p)
            if digest != f.sha256:
                raise ValueError(f"sha256 mismatch for {p}: {digest} != {f.sha256}")
            out.append(p)
        return out


def _rec(text, label: int, source_label: str, group: str | None = None) -> dict | None:
    t = clean_text(text)
    if t is None:
        return None
    return {"text": t, "label": int(label), "source_label": source_label, "group": group}


def _load_deepset(src: Source) -> list[dict]:
    import pandas as pd

    rows = []
    for p in src.verified():
        if p.suffix != ".parquet":
            continue
        split = "train" if "/train-" in str(p) else "test"
        df = pd.read_parquet(p)
        for t, y in zip(df["text"], df["label"]):
            rows.append(_rec(t, int(y), f"label={int(y)}", group=f"deepset:{split}"))
    return [r for r in rows if r]


def _load_gandalf(src: Source) -> list[dict]:
    import pandas as pd

    rows = []
    for p in src.verified():
        if p.suffix != ".parquet":
            continue
        df = pd.read_parquet(p)
        rows += [_rec(t, 1, "gandalf_ignore_instructions(all positive)") for t in df["text"]]
    return [r for r in rows if r]


def _load_spml(src: Source) -> list[dict]:
    import pandas as pd

    from training.common import sha256_text

    (p,) = [p for p in src.verified() if p.suffix == ".csv"]
    df = pd.read_csv(p)
    rows = []
    for sp, up, y, deg in zip(df["System Prompt"], df["User Prompt"], df["Prompt injection"], df["Degree"]):
        # Group by system prompt: all user prompts written against one system prompt stay in one split.
        g = "spml:" + sha256_text(str(sp))[:16]
        rows.append(_rec(up, int(y), f"prompt_injection={int(y)},degree={int(deg)}", group=g))
    return [r for r in rows if r]


def _load_jackhhao(src: Source) -> list[dict]:
    import pandas as pd

    (p,) = [p for p in src.verified() if p.suffix == ".csv"]
    df = pd.read_csv(p)
    mapping = {"jailbreak": 1, "benign": 0}
    return [r for r in (_rec(t, mapping[y], f"type={y}") for t, y in zip(df["prompt"], df["type"])) if r]


def _load_yanis(src: Source) -> list[dict]:
    import pandas as pd

    (p,) = [p for p in src.verified() if p.suffix == ".csv"]
    df = pd.read_csv(p)
    return [r for r in (_rec(t, 1, "prompt_injections(all positive)") for t in df["prompt_injections"]) if r]


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            name="deepset",
            repo_id="deepset/prompt-injections",
            revision="4f61ecb038e9c3fb77e21034b22511b523772cdd",
            license="apache-2.0",
            redistributable=True,
            files=(
                RawFile("data/train-00000-of-00001-9564e8b05b4757ab.parquet",
                        "2e10bc7ab30f542c97e4e83e2a5683000b5057d25ec10908784c631d44124c04"),
                RawFile("data/test-00000-of-00001-701d16158af87368.parquet",
                        "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e"),
            ),
            stage2_overlap="suspected",
            label_definition="label 1 = prompt injection / goal hijacking (incl. role-play hijacks); 0 = normal query",
            loader=_load_deepset,
        ),
        Source(
            name="spml",
            repo_id="reshabhs/SPML_Chatbot_Prompt_Injection",
            revision="02ce8084e979bc7d4c24ee35d22ecb7f2db96ff5",
            license="mit",
            redistributable=True,
            files=(RawFile("spml_prompt_injection.csv",
                           "4ca6590cd8ec10832fcebee5e243acb32be5c3e79b7a260f8638eed738c784f8"),),
            stage2_overlap="none_known",
            label_definition="'Prompt injection'=1 if the user prompt tries to make the bot violate its system prompt",
            loader=_load_spml,
        ),
        Source(
            name="yanis",
            repo_id="yanismiraoui/prompt_injections",
            revision="bd55359f2f332afc35f277ac3dd08f7111b024c9",
            license="apache-2.0",
            redistributable=True,
            files=(RawFile("prompt_injections.csv",
                           "f4843f1841fa19b980f804796a68fc72f06841775eaba2723c768c7d772aabad"),),
            stage2_overlap="none_known",
            label_definition="every row is a prompt injection (prompt leaking, jailbreak, mode switching); no negatives",
            loader=_load_yanis,
        ),
        Source(
            name="gandalf",
            repo_id="Lakera/gandalf_ignore_instructions",
            revision="04737b65e90a6794ec227012e4a255a7def6344b",
            license="mit",
            redistributable=True,
            files=(
                RawFile("data/train-00000-of-00001-ded53be747ff55cd.parquet",
                        "5b6acf3e5a5998d21f8e1222bb45bbdec25a14408747b1cd63bebef4a75fa439"),
                RawFile("data/validation-00000-of-00001-94481a2a09ff2fff.parquet",
                        "f51ab3e3407a368845b0f57932cc745c09280429416d7507bd178a16326a79f6"),
                RawFile("data/test-00000-of-00001-bc92128b9288a6d1.parquet",
                        "56b646d133335ebc535266bd55dbe1b5bee7caa4b95bf49d040684b9b5dd9972"),
            ),
            stage2_overlap="suspected",
            label_definition="Gandalf game prompts with embedding similarity >= 0.825 to 'Ignore all previous "
                             "instructions'; all treated as injection (card warns of some noise)",
            loader=_load_gandalf,
        ),
        Source(
            name="jackhhao",
            repo_id="jackhhao/jailbreak-classification",
            revision="2f2ceeb39658696fd3f462403562b6eea5306287",
            license="apache-2.0",
            redistributable=True,
            files=(RawFile("default/jailbreak_dataset_full.csv",
                           "79a7b90b0abe00e3586cc5048353c3236543cca228a1ee261fe3b57a7cb7e29f"),),
            stage2_overlap="confirmed",
            label_definition="type=jailbreak -> 1, type=benign -> 0 (jailbreak is not identical to injection)",
            loader=_load_jackhhao,
        ),
    ]
}


def fetch(names: list[str] | None = None) -> None:
    from huggingface_hub import hf_hub_download

    for name, src in SOURCES.items():
        if names and name not in names:
            continue
        for f in src.files:
            hf_hub_download(src.repo_id, f.path, repo_type="dataset", revision=src.revision,
                            local_dir=RAW_DIR / src.repo_id.replace("/", "__"))
        src.verified()
        print(f"fetched {src.repo_id}@{src.revision[:10]} ({len(src.files)} files, sha256 verified)")


def load_all() -> list[dict]:
    """All public records with source metadata attached, in a fixed order."""
    out = []
    for name, src in SOURCES.items():
        for i, r in enumerate(src.loader(src)):
            r.update(source=name, source_index=i, stage2_overlap=src.stage2_overlap)
            out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.only)
    else:
        json.dump({k: {"repo_id": v.repo_id, "revision": v.revision, "license": v.license} for k, v in SOURCES.items()},
                  sys.stdout, indent=2)

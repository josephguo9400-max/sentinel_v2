"""Stage 1 anti-dilution: what scoring segment pieces and sub-windows costs (benign escalation) and buys (diluted
injections no longer cleared). Uses the Stage 1 detector the registry points at and the val split only.

    flock /home/claude/measure.lock -c 'python -m serving.bench.stage1_dilution --out serving/bench_results/stage1_dilution.json'

Measured, per setting of `stage1_subwindow_tokens` (0 = off, 32, 64, 128; stride = half):
- benign val texts (label 0, all sources): fraction Stage 1 would NOT clear (score >= gray_zone low), i.e. that
  escalate to Stage 2 on score alone. The language gate is not applied here (it is independent of the setting).
- diluted injections: each val injection (label 1), first 300 characters with whitespace collapsed, appended after a fixed 30-sentence benign
  paragraph (one paragraph, no blank line, so structure-aware packing cannot separate it): fraction Stage 1 clears
  (score < gray_zone low), i.e. that would never reach Stage 2. Synthetic construction; not a red-team result.
- the same for the injection placed in its own paragraph after the benign paragraph (segment-piece scoring).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from serving.bench.common import ROOT, host_info, now, read_val, require_lock, uptime, write_result
from serving.config import GatewayConfig
from serving.detector import tfidf_subwindows
from serving.registry import load_detector, read_registry
from serving.structure import segment

FILLER = ("The quarterly review covered the budget, the hiring plan and next year's roadmap, and the team agreed "
          "to send the updated slides before Thursday. ") * 30


def stage1_score(det, text: str, sub: int) -> float:
    """Mirror of ClassifierService._run_stage1 for one chunk text (pieces = paragraphs of a packed chunk)."""
    pieces = [s.text for s in segment(text)]
    texts = [text] + (pieces if len(pieces) > 1 else [])
    windows = [w for t in texts for w in det.windows(t)[0]]
    if sub:
        windows += [w for t in texts for w in tfidf_subwindows(t, sub, max(1, sub // 2))]
    return max(det.infer(windows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "serving" / "bench_results" / "stage1_dilution.json")
    ap.add_argument("--settings", default="0,32,64,128")
    ap.add_argument("--n-injections", type=int, default=500)
    ap.add_argument("--no-lock-check", action="store_true")
    args = ap.parse_args(argv)
    require_lock(args.no_lock_check)
    cfg = GatewayConfig()
    rel = read_registry(Path(cfg.registry_path))["stage1"]
    spec = load_detector(Path(cfg.artifacts_root), "stage1", rel)
    lo = spec.gray_zone[0]
    rows = read_val()
    benign = [r["text"] for r in rows if r["label"] == 0]
    # Whitespace collapsed so an injection's own line breaks cannot split the paragraph.
    inj = [" ".join(r["text"][:300].split()) for r in rows if r["label"] == 1][: args.n_injections]
    same_para = [FILLER + " " + t for t in inj]
    own_para = [FILLER + "\n\n" + t for t in inj]
    # The same-paragraph construction must be a single segment, or segment-piece scoring would separate it.
    assert all(len(segment(t)) == 1 for t in same_para)
    before = uptime()
    out = {"generated_by": "python -m serving.bench.stage1_dilution", "at": now(), "host": host_info(),
           "uptime_before": before, "stage1": {"pointer": rel, "detector_version": spec.detector_version,
                                                "gray_zone": list(spec.gray_zone)},
           "data": {"split": "val", "n_benign": len(benign), "n_injections": len(inj),
                    "construction": "first 300 chars of each val injection after a 30-sentence benign paragraph"},
           "settings": {}}
    for sub in [int(x) for x in args.settings.split(",")]:
        b = [stage1_score(spec.detector, t, sub) for t in benign]
        s = [stage1_score(spec.detector, t, sub) for t in same_para]
        o = [stage1_score(spec.detector, t, sub) for t in own_para]
        out["settings"][str(sub)] = {
            "subwindow_tokens": sub, "subwindow_stride": max(1, sub // 2) if sub else None,
            "benign_not_cleared": {"k": sum(x >= lo for x in b), "n": len(b)},
            "diluted_same_paragraph_cleared": {"k": sum(x < lo for x in s), "n": len(s)},
            "diluted_own_paragraph_cleared": {"k": sum(x < lo for x in o), "n": len(o)},
        }
    out["uptime_after"] = uptime()
    write_result(args.out, out)
    for k, v in out["settings"].items():
        print(k, {kk: f"{vv['k']}/{vv['n']}" for kk, vv in v.items() if isinstance(vv, dict)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Copy aggregate-only experiment JSON into reports/handoff/ (safe to commit).

Strips absolute local paths. Does not copy models, row-level files, or raw HMDA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "handoff"

COPIES = {
    "baseline_latest_metrics.json": ROOT / "outputs" / "baselines" / "latest_metrics.json",
    "privacy_latest_privacy.json": ROOT / "outputs" / "privacy" / "latest_privacy.json",
    "data_profile.json": ROOT / "outputs" / "data_profile.json",
}


def _sanitize(obj: object, root: Path) -> object:
    if isinstance(obj, dict):
        return {k: _sanitize(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v, root) for v in obj]
    if isinstance(obj, str):
        s = obj
        for prefix in (str(root.resolve()) + "/", str(root) + "/"):
            if s.startswith(prefix):
                s = s[len(prefix) :]
        norm = s.replace("\\", "/")
        if "/data/raw/" in norm:
            s = "data/raw/" + norm.split("/data/raw/")[-1]
        return s
    return obj


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in COPIES.values() if not p.is_file()]
    if missing:
        print("Missing local aggregate outputs:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("Run make profile / make baseline / make privacy first.", file=sys.stderr)
        return 1
    for name, src in COPIES.items():
        payload = json.loads(src.read_text(encoding="utf-8"))
        if name != "data_profile.json":
            synthetic = bool(
                (payload.get("manifest") or {}).get("synthetic", False)
            )
            if synthetic:
                print(
                    f"Refusing to sync {src}: manifest.synthetic=true. "
                    "Run real HMDA baseline/privacy (--no-synthetic) first.",
                    file=sys.stderr,
                )
                return 1
        clean = _sanitize(payload, ROOT)
        if isinstance(clean, dict) and name == "data_profile.json":
            clean["data_root"] = "data/raw"
        dest = OUT / name
        dest.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

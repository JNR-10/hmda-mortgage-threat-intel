"""Aggregate result tables and run manifests (no row-level outputs)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def new_run_manifest(
    *,
    run_id: str,
    command: str,
    config_paths: list[str],
    seed: int,
    device: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "config_paths": config_paths,
        "seed": seed,
        "device": device,
        "aggregate_only": True,
        "row_level_outputs": False,
    }
    if extra:
        manifest.update(extra)
    return manifest

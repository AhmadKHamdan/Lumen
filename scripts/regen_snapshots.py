"""Regenerate snapshot files used by the test suite.

Run after a legitimate edit to LANDMARK_ALIASES (or any other snapshotted
data). The snapshot test will then expect the new content.

Usage:
    python scripts/regen_snapshots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Put `backend/` on sys.path so `navigation` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from navigation.landmark_map import snapshot_dict  # noqa: E402


def main() -> None:
    snapshot_dir = (
        Path(__file__).resolve().parent.parent / "backend" / "tests" / "snapshots"
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    out_path = snapshot_dir / "landmark_map.json"
    data = snapshot_dict()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(f"wrote {out_path}  ({len(data)} aliases)")


if __name__ == "__main__":
    main()

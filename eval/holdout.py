"""Business-level holdout split.

The private test uses unseen business types, so the holdout must be split by
business, never by question. Every fifth business (sorted) is held out.
"""

from __future__ import annotations

from pathlib import Path

HOLDOUT_BUSINESSES = {"biz_105", "biz_110", "biz_115", "biz_120", "biz_125", "biz_130"}

import os

if os.environ.get("LEDGERAI_HOLDOUT"):
    HOLDOUT_BUSINESSES = {
        name.strip()
        for name in os.environ["LEDGERAI_HOLDOUT"].split(",")
        if name.strip()
    }


def split_businesses(data_dir: Path) -> tuple[list[str], list[str]]:
    names = sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    holdout = [n for n in names if n in HOLDOUT_BUSINESSES]
    train = [n for n in names if n not in HOLDOUT_BUSINESSES]
    return train, holdout

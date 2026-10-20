#!/usr/bin/env python3
"""Stage 1 (laptop): build the pilot task manifests from the pinned sources and write an
audit next to each. The manifests are committed; the pod only reads them.

    python scripts/01_prepare_tasks.py                 # both pilots
    python scripts/01_prepare_tasks.py --only srh_pilot
"""

import argparse
import collections
import json

import _common  # noqa: F401

from eforensics.config import load_experiment, manifest_path
from eforensics.io import write_json, write_jsonl
from eforensics.tasks import impossible, srh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=["srh_pilot", "ib_pilot"], default=None)
    args = p.parse_args()
    exp = load_experiment()
    builders = {"srh_pilot": srh.build_pilot_manifest, "ib_pilot": impossible.build_pilot_manifest}
    for name, build in builders.items():
        if args.only and name != args.only:
            continue
        rows, audit = build(exp)
        path = manifest_path(exp, name)
        write_jsonl(path, rows)
        audit["design_version"] = exp["design_version"]
        audit["n_rows"] = len(rows)
        audit["by_family"] = dict(collections.Counter(r["exploit_family"] for r in rows))
        write_json(path.with_suffix(".audit.json"), audit)
        print(f"[prepare] {name}: {len(rows)} rows -> {path}")
        print(json.dumps({k: v for k, v in audit.items() if k != "problems"}, indent=1)[:2500])


if __name__ == "__main__":
    main()

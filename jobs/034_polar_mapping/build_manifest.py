"""Two encoder investigations; reuse fixed ODMA-polar and dense controls from 032/033."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def manifest():
    previous = [json.loads(line) for line in (ROOT / "jobs/033_matched_d2/manifest.jsonl").read_text().splitlines()]
    references = [r for r in previous if r["family"] == "odma_polar"]
    return [{**r, "name": r["name"].replace("odma_polar_d2", f"{family}_joint_d2"), "family": family,
             "mode": "joint", "reference_033": r["name"]}
            for family in ("polar_energy", "polar_mixer") for r in references]


if __name__ == "__main__":
    rows = manifest()
    Path(__file__).with_name("manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} rows: 6 energy profiles + 6 shared mixers; fixed controls reused")

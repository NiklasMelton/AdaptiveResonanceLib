#!/usr/bin/env python3
import json, re, sys
from pathlib import Path
import pandas as pd

# Where your jobs saved outputs
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("all_results.csv")

rows = []
fname_re = re.compile(
    r'(?i)(?P<method>BinaryFuzzyARTMAP|FuzzyARTMAP|HypersphereARTMAP|GaussianARTMAP)[-_]'
    r'(?P<backend>python|torch|c\+\+|cpp)[-_]?'
    r'(?:(?P<device>cpu|gpu))?.*'
)

for p in RESULTS_DIR.glob("**/*.json"):
    try:
        with p.open() as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Skip {p}: {e}", file=sys.stderr)
        continue

    # Try to enrich with filename-derived metadata if not in JSON
    m = fname_re.search(p.stem)
    meta = {}
    if m:
        meta.update({k: v for k, v in m.groupdict().items() if v})
    # Slurm job id if present in name like ...-<jobid>.json
    jid_match = re.search(r'(\d{6,})', p.stem)
    if jid_match:
        meta["job_id"] = jid_match.group(1)

    # Promote common fields if your JSON already has them
    for k in ("method", "backend", "device", "seed"):
        if k in data and data[k] is not None:
            meta.setdefault(k, data[k])

    # Flatten top-level only; nest under 'metrics_' for dicts if needed
    flat = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}_{sk}"] = sv
        else:
            flat[k] = v

    row = {**meta, **flat, "_source": str(p)}
    rows.append(row)

if not rows:
    print(f"[ERROR] No JSON files found under {RESULTS_DIR}", file=sys.stderr)
    sys.exit(2)

df = pd.DataFrame(rows)

# Consistent column order (metadata first)
meta_cols = [c for c in ["method", "backend", "device", "seed", "job_id"] if c in df.columns]
other_cols = [c for c in df.columns if c not in meta_cols + ["_source"]]
df = df[meta_cols + other_cols + ["_source"]]

df.to_csv(OUT_CSV, index=False)
print(f"[OK] Wrote {OUT_CSV} with {len(df)} rows from {RESULTS_DIR}")

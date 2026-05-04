"""
Aggregate per-mode metric CSVs into 4 organized comparison tables.
Run from modeling/ directory:
    python build_result_tables.py            # uses unversioned results
    python build_result_tables.py --tag v4   # uses v4 results
Outputs:
    results/table1_backpack.csv
    results/table2_patchuniformity.csv
    results/table3_distance.csv
    results/table4_reciprocity.csv
"""

import argparse
import os
import pandas as pd

REWARD_MODES = ["selfish", "capacity", "proximity", "reciprocity", "capacity_proximity"]
DISPLAY_NAMES = {
    "selfish": "Selfish",
    "capacity": "Capacity",
    "proximity": "Proximity",
    "reciprocity": "Reciprocity",
    "capacity_proximity": "Cap+Prox",
}


def _path(metric_file, mode, tag):
    suffix = f"_{tag}" if tag else ""
    return f"results/metrics_{mode}{suffix}_metric{metric_file}.csv"


def _load(metric_file, mode, tag):
    p = _path(metric_file, mode, tag)
    return pd.read_csv(p) if os.path.exists(p) else None


def build_wide_table(metric_file, bin_col, tag):
    """
    Produce a wide table: rows = bins, columns = Human + each agent mode.
    """
    rows = None
    human_col = None
    out = {}

    for mode in REWARD_MODES:
        df = _load(metric_file, mode, tag)
        if df is None:
            print(f"  [skip] {mode}: file missing")
            continue
        if rows is None:
            rows = df[bin_col].tolist()
            human_col = df["human_helping_rate"].tolist()
        out[DISPLAY_NAMES[mode]] = df["agent_helping_rate"].tolist()

    if rows is None:
        return None

    table = pd.DataFrame({bin_col: rows, "Human": human_col})
    for mode_name, vals in out.items():
        table[mode_name] = vals

    # Round for report-friendly display
    for c in table.columns:
        if c != bin_col:
            table[c] = pd.to_numeric(table[c], errors="coerce").round(3)
    return table


def build_reciprocity_table(tag):
    """
    Metric 5 is special: rows are (turn, partner_helped_last).
    Returns a table with one row per (turn, partner_helped_last)
    and one column per mode + Human.
    """
    rows = None
    human_col = None
    out = {}

    for mode in REWARD_MODES:
        df = _load("5_reciprocity", mode, tag)
        if df is None:
            continue
        df = df.sort_values(["turn", "partner_helped_last"]).reset_index(drop=True)
        if rows is None:
            rows = list(zip(df["turn"], df["partner_helped_last"]))
            human_col = df["human_helping_rate"].tolist()
        out[DISPLAY_NAMES[mode]] = df["agent_helping_rate"].tolist()

    if rows is None:
        return None

    table = pd.DataFrame(rows, columns=["turn", "partner_helped_last"])
    table["Human"] = human_col
    for mode_name, vals in out.items():
        table[mode_name] = vals

    for c in table.columns:
        if c not in ("turn", "partner_helped_last"):
            table[c] = pd.to_numeric(table[c], errors="coerce").round(3)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="Run tag (e.g. v4)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    print(f"Building tables (tag='{args.tag}')...\n")

    specs = [
        ("1_backpack",       "backpack_size",     "table1_backpack",       "Helping rate by backpack size"),
        ("2_patchuniformity","patchUniformity",   "table2_patchuniformity","Helping rate by patch uniformity"),
        ("3_distance",       "distance_bin",      "table3_distance",       "Helping rate by distance to partner veg"),
        ("4_energy",         "energy_bin",        "table4_energy",         "Helping rate by remaining energy"),
    ]
    for metric_file, bin_col, out_name, label in specs:
        print(f"--- {label} ---")
        t = build_wide_table(metric_file, bin_col, args.tag)
        if t is None:
            print("  (no data)\n")
            continue
        path = f"results/{out_name}{suffix}.csv"
        t.to_csv(path, index=False)
        print(t.to_string(index=False))
        print(f"  saved -> {path}\n")

    print("--- Helping rate by turn (reciprocity) ---")
    t = build_reciprocity_table(args.tag)
    if t is not None:
        path = f"results/table5_reciprocity{suffix}.csv"
        t.to_csv(path, index=False)
        print(t.to_string(index=False))
        print(f"  saved -> {path}\n")
    else:
        print("  (no data)\n")


if __name__ == "__main__":
    main()
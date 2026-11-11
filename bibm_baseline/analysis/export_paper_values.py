#!/usr/bin/env python3
"""Export every table value in the paper from the aggregate JSON, and build the table gate.

Writes
  outputs/aggregate/paper_values.csv   one row per table cell: table,row,column,value
  harness/table_manifest.yaml          JSON-compatible manifest for scripts/check_table_manifest.py
and prints the LaTeX table rows, formatted exactly as they must appear in the manuscript.

Each manifest entry checks (a) the CSV value against the manuscript value within rounding
tolerance and (b) that the full LaTeX row containing the cell is present in the manuscript,
so a hand edit that drifts from the data fails the gate.

Sources: e5_controls.json (Table I), e2_trainonly_pca_fills.json (Table II),
e4_signal_perturbations.json + e3_eegnet_signal.json (Table III).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "aggregate"
TEX = "manuscript/main_bibm2026.tex"


def j(name):
    return json.load(open(A / name))


def f3(v):
    return f"{v:.3f}"


def cell3(v):
    return f"${f3(v)}$"


def ci3(lo, hi):
    return f"[${f3(lo)}$, ${f3(hi)}$]"


def diff3(m, lo, hi):
    return f"${m:+.3f}$ [${lo:+.3f},{hi:+.3f}$]"


def have(*names):
    missing = [n for n in names if not (A / n).exists()]
    if missing:
        print(f"[export] skipping a table: missing {', '.join(missing)}")
    return not missing


def tables():
    out = {}
    if have("e5_controls.json"):
        out.update(table_clean())
    if have("e2_trainonly_pca_fills.json"):
        out.update(table_dropconf())
    if have("e4_signal_perturbations.json", "e3_eegnet_signal.json"):
        out.update(table_rawsignal())
    return out


def table_clean():
    out = {}
    # Table I: clean metrics, fixed encoders
    cm = j("e5_controls.json")["clean_metrics"]
    rows = []
    for cfg, name, dims in (("A0$_e$", "ERP-window", 102), ("A0", "Band-power", 170),
                            ("A1", "Reservoir", 2176)):
        r = cm[name]
        label = f"{name} ({dims})" if name != "Reservoir" else "Reservoir \\BSC{} (2176)"
        tex = (f"{cfg} & {label} & {cell3(r['BA'])} {ci3(*r['BA_ci95'])} & "
               f"{cell3(r['macro_F1'])} & {cell3(r['macro_OvR_AUC'])} \\\\")
        rows.append((name, tex, {"BA": r["BA"], "BA_lo": r["BA_ci95"][0], "BA_hi": r["BA_ci95"][1],
                                 "F1": r["macro_F1"], "AUC": r["macro_OvR_AUC"]}))
    out["tab:clean"] = rows
    return out


def table_dropconf():
    out = {}
    # Table II: feature-coordinate experiment at 30%
    e2 = j("e2_trainonly_pca_fills.json")
    rows = []
    for fill, lab in (("zero", "Zero"), ("mean", "Mean"), ("knn", "$k$NN"), ("spatial", "Spatial")):
        d = e2["dropout_30"][fill]; p = e2["paired_ERPwindow_minus_Reservoir_30"][fill]
        tex = (f"{lab} & {cell3(d['Band-power']['drop_BA'])} & {cell3(d['ERP-window']['drop_BA'])} & "
               f"{cell3(d['Reservoir']['drop_BA'])} & {diff3(p['mean_diff'], *p['ci95'])} \\\\")
        rows.append((fill, tex, {"Band-power": d["Band-power"]["drop_BA"], "ERP-window": d["ERP-window"]["drop_BA"],
                                 "Reservoir": d["Reservoir"]["drop_BA"], "ERP_minus_Res": p["mean_diff"]}))
    out["tab:dropconf"] = rows
    return out


def table_rawsignal():
    out = {}
    # Table III: signal level, all encoders
    e4 = j("e4_signal_perturbations.json")["BA"]; e3 = j("e3_eegnet_signal.json")["BA"]
    rows = []
    for c4, c3, lab in (("clean", "remove_0.0", "Clean"), ("remove_0.1", "remove_0.1", "$10\\%$ removed"),
                        ("remove_0.3", "remove_0.3", "$30\\%$ removed"), ("remove_0.5", "remove_0.5", "$50\\%$ removed"),
                        ("amp_5dB", "amp_5dB", "Noise $5$~dB"), ("jitter_50ms", "jitter_50ms", "Jitter $\\pm50$~ms")):
        v = {"Band-power": e4[c4]["Band-power"]["BA"], "ERP-window": e4[c4]["ERP-window"]["BA"],
             "Reservoir": e4[c4]["Reservoir"]["BA"], "EEGNet": e3[c3]["EEGNet"]["BA"],
             "EEGNet+aug": e3[c3]["EEGNet+aug"]["BA"]}
        tex = f"{lab} & " + " & ".join(cell3(v[k]) for k in v) + " \\\\"
        rows.append((c4, tex, v))
    out["tab:rawsignal"] = rows
    return out


def main():
    t = tables()
    csv_path = A / "paper_values.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["table", "row", "column", "value"])
        for tab, rows in t.items():
            for row, _tex, vals in rows:
                for col, v in vals.items():
                    w.writerow([tab, row, col, f"{v:.6f}"])
    manifest = {"tables": []}
    for tab, rows in t.items():
        matches = []
        for row, tex, vals in rows:
            for col, v in vals.items():
                shown = f"{v:+.3f}" if col == "ERP_minus_Res" else f3(v)
                matches.append({"label": f"{row}.{col}",
                                "csv_row": {"table": tab, "row": row, "column": col},
                                "csv_column": "value", "manuscript_value": shown,
                                "required_text": tex, "tolerance": 0.0006})
        manifest["tables"].append({"name": tab, "latex_file": TEX,
                                   "csv_file": str(csv_path.relative_to(ROOT)), "matches": matches})
    (ROOT / "harness" / "table_manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[export] wrote {csv_path.relative_to(ROOT)} and harness/table_manifest.yaml")
    for tab, rows in t.items():
        print(f"\n% {tab}")
        for _row, tex, _vals in rows:
            print(tex)


if __name__ == "__main__":
    main()

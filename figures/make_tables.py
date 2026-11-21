"""Generate LaTeX tables from outputs/*.json. No number in a table is typed by hand."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "manuscript" / "tables"
OUTD.mkdir(parents=True, exist_ok=True)


def fmt(d):
    return f"{d['ba']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"


def shape_table():
    b = json.loads((ROOT / "outputs/X01/x01_results.json").read_text())["balanced_accuracy"]
    p4 = ROOT / "outputs/X04/x04_results.json"
    if p4.exists():
        b.update(json.loads(p4.read_text())["balanced_accuracy"])
    rows = [("pre_mean", "Pre-stimulus mean (34)"), ("pre_bins6", "Pre-stimulus, 6 bins (204)"),
            ("reservoir_pre", "Reservoir, truncated at onset"), ("reservoir_bibm", "Reservoir, BIBM window"),
            ("erp_windows", "ERP windows (102)"), ("eegnet_pre", "EEGNet, pre-stimulus input"),
            ("eegnet_full", "EEGNet, full epoch")]
    norms = ["epoch_z", "post_z", "pre_sd", "none"]
    head = " & ".join([r"\texttt{epoch\_z}", r"\texttt{post\_z}", r"\texttt{pre\_sd}", r"\texttt{none}"])
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{SHAPE: balanced accuracy (participant-bootstrap 95\% CI) by encoder and scaling. Three classes, chance 0.333; 211 participants, 633 observations. Only \texttt{epoch\_z} and \texttt{post\_z} use post-stimulus samples to scale earlier ones. A dash marks a combination not run.}",
             r"\label{tab:shape}", r"\begin{tabular}{lcccc}", r"\toprule",
             f"Encoder & {head} \\\\", r"\midrule"]
    for key, name in rows:
        cells = [fmt(b[f"{key}|{n}"]) if f"{key}|{n}" in b else "--" for n in norms]
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUTD / "tab_shape.tex").write_text("\n".join(lines) + "\n")


def erpcore_table():
    p = ROOT / "outputs/X02/x02_results.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())["results"]
    norms = ["none", "pre_sd", "epoch_z", "post_z"]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{ERP CORE: mean pre-stimulus decoding accuracy ($-200$ to $0$~ms, 15.6~ms bins), two classes, chance 0.5, 40 participants per paradigm. Across participants: trials scaled, then averaged; $p$ from 200 label permutations. Within participants: single trials; mean [95\% CI] and one-sample $t$-test $p$ against 0.5.}",
             r"\label{tab:erpcore}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Paradigm & Regime & \texttt{none} & \texttt{pre\_sd} & \texttt{epoch\_z} & \texttt{post\_z} \\", r"\midrule"]
    for par, d in r.items():
        a = " & ".join(f"{d['across'][n]['pre_mean']:.3f} ($p={d['across'][n]['p']:.3f}$)" for n in norms)
        w = " & ".join(f"{d['within'][n]['pre_mean']:.3f} ($p={d['within'][n]['p']:.2g}$)" for n in norms)
        lines.append(f"\\multirow{{2}}{{*}}{{{par}}} & across & {a} \\\\")
        lines.append(f" & within & {w} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUTD / "tab_erpcore.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    shape_table(); erpcore_table()
    print("tables written to", OUTD)

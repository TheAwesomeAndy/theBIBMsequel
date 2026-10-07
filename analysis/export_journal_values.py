#!/usr/bin/env python3
"""Export every number of the journal manuscript from the aggregate JSON.

Writes
  manuscript/journal/generated/values.tex   \\V{key} lookups for every inline number
  manuscript/journal/generated/tab_*.tex    table bodies (\\input by the manuscript)
  outputs/aggregate/journal/journal_values.csv   key,value for every exported number
The manuscript never types a result by hand: inline numbers use \\V{key} (an unknown key
is a LaTeX error) and tables are \\input from generated/. Rerun after any experiment.

Sources: conference outputs (e1-e5, deap_replication_v2) for SHAPE and DEAP; journal
outputs (j1-j6) for the external TCRZEM cohort ("x" = conference-matched epoch,
"xl" = pre-specified long epoch).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "outputs" / "aggregate"
JAG = AGG / "journal"
GEN = ROOT / "manuscript" / "journal" / "generated"
VALUES: dict[str, str] = {}
ENC = {"Band-power": "band", "ERP-window": "erp", "Reservoir": "res", "EEGNet": "eegnet",
       "EEGNet+aug": "aug"}


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def f3(v):
    return f"{v:.3f}"


def s3(v):
    return f"{v:+.3f}"


def put(key, val):
    VALUES[key] = val
    return val


def put_ba(key, ba, ci):
    put(key, f3(ba)); put(key + "-lo", f3(ci[0])); put(key + "-hi", f3(ci[1]))
    put(key + "-ci", f"[{f3(ci[0])}, {f3(ci[1])}]")


def put_diff(key, d):
    put(key, s3(d["mean_diff"])); put(key + "-lo", s3(d["ci95"][0])); put(key + "-hi", s3(d["ci95"][1]))
    put(key + "-ci", f"[{s3(d['ci95'][0])}, {s3(d['ci95'][1])}]")
    put(key + "-abs", f"{abs(d['mean_diff']):.3f}")


def cell(ba, ci=None):
    return f"${f3(ba)}$" if ci is None else f"${f3(ba)}$ [${f3(ci[0])}$, ${f3(ci[1])}$]"


def dcell(d):
    return f"${s3(d['mean_diff'])}$ [${s3(d['ci95'][0])},{s3(d['ci95'][1])}$]"


# ----------------------------------------------------------------------------- sources
def shape_values():
    e5, e2, e4, e3, e1 = (load(AGG / n) for n in ("e5_controls.json", "e2_trainonly_pca_fills.json",
                                                  "e4_signal_perturbations.json", "e3_eegnet_signal.json",
                                                  "e1_rho_sweep.json"))
    for n, k in ENC.items():
        if n in e5["clean_metrics"]:
            r = e5["clean_metrics"][n]
            put_ba(f"s-clean-{k}", r["BA"], r["BA_ci95"])
            put(f"s-f1-{k}", f3(r["macro_F1"])); put(f"s-auc-{k}", f3(r["macro_OvR_AUC"]))
            put(f"s-cent-{k}", f"{e5['centeredness_mean_abs_mu_over_sigma'][n]['mean']:.2f}")
            put(f"s-nullmax-{k}", f3(e5["permutation_null"][n]["max"]))
    for m in ("EEGNet", "EEGNet+aug"):
        r = e3["BA"]["remove_0.0"][m]
        put_ba(f"s-clean-{ENC[m]}", r["BA"], r["ci95"])
    for fill in ("zero", "mean", "knn", "spatial"):
        for n, k in list(ENC.items())[:3]:
            r = e2["dropout_30"][fill][n]
            put_ba(f"s-fc30-{fill}-{k}", r["drop_BA"], r["drop_ci95"])
        put_diff(f"s-fc30-{fill}-erpres", e2["paired_ERPwindow_minus_Reservoir_30"][fill])
    for n, k in list(ENC.items())[:3]:
        put_diff(f"s-meanzero30-{k}", e2["mean_minus_zero_30"][n])
    put(f"s-rhostar", "0.82")
    put_ba("s-rho09-clean", e1["BA"]["0.9"]["clean"]["BA"], e1["BA"]["0.9"]["clean"]["ci95"])
    put_diff("s-rho0-minus-09", e1["paired_vs_rho0.9"]["0.0"]["clean"])
    put_diff("s-rho15-minus-09", e1["paired_vs_rho0.9"]["1.5"]["clean"])
    return e5, e2, e4, e3, e1


def ext_values(tag, suffix):
    j1, j2, j3, j5, j6 = (load(JAG / f"{n}{suffix}.json") for n in
                          ("j1_tcrzem_core", "j2_tcrzem_signal", "j3_tcrzem_eegnet", "j5_tcrzem_origin",
                           "j6_tcrzem_prestimulus"))
    if j1:
        for n, k in list(ENC.items())[:3]:
            r = j1["clean_metrics"][n]
            put_ba(f"{tag}-clean-{k}", r["BA"], r["ci95"])
            put(f"{tag}-f1-{k}", f3(r["macro_F1"])); put(f"{tag}-auc-{k}", f3(r["macro_OvR_AUC"]))
            put(f"{tag}-cent-{k}", f"{j1['centeredness_mean_abs_mu_over_sigma'][n]['mean']:.2f}")
            put(f"{tag}-nullmax-{k}", f3(j1["permutation_null"][n]["max"]))
            put(f"{tag}-nullmean-{k}", f3(j1["permutation_null"][n]["mean"]))
        put(f"{tag}-cent-rawcounts",
            f"{j1['centeredness_mean_abs_mu_over_sigma']['Reservoir raw BSC6 counts (before PCA)']['mean']:.2f}")
        for kk, r in j1["pca_components"].items():
            put_ba(f"{tag}-pcak{kk}", r["BA"], r["ci95"])
        put_ba(f"{tag}-pooledpca", j1["pooled_pca_sensitivity"]["BA"], j1["pooled_pca_sensitivity"]["ci95"])
        put_diff(f"{tag}-pooled-minus-train", j1["pooled_pca_sensitivity"]["pooled_minus_trainonly"])
        for c in ("clean", "signal30"):
            put_diff(f"{tag}-bsc6minus1-{c}", j1["temporal_binning"][c]["BSC6_minus_BSC1"])
        put_diff(f"{tag}-clean-erpres", j1["clean_paired"]["ERP-window_minus_Reservoir"])
        for lv in ("10", "20", "30", "40", "50"):
            for fill in ("zero", "mean", "knn", "spatial"):
                e = j1["dropout"][lv][fill]
                for n, k in list(ENC.items())[:3]:
                    put_ba(f"{tag}-fc{lv}-{fill}-{k}", e[n]["BA"], e[n]["ci95"])
                    put(f"{tag}-fc{lv}-{fill}-{k}-rawret", f"{e[n]['raw_retention']:.2f}")
                    put(f"{tag}-fc{lv}-{fill}-{k}-acret", f"{e[n]['above_chance_retention']:.2f}")
                put_diff(f"{tag}-fc{lv}-{fill}-erpres", e["ERP-window_minus_Reservoir"])
        for n, k in list(ENC.items())[:3]:
            for fill in ("mean", "knn", "spatial"):
                for lv, d in j1["fill_minus_zero"][n][fill].items():
                    put_diff(f"{tag}-fmz-{fill}{lv}-{k}", d)
    if j2:
        put(f"{tag}-trials-median", f"{j2['trials_per_observation']['median']:.0f}")
        for lv, r in j2["spline_mean_correlation_missing_channels"].items():
            put(f"{tag}-splinecorr-{int(float(lv) * 100)}", f"{r:.2f}")
        for c, row in j2["BA"].items():
            for n, k in list(ENC.items())[:3]:
                put_ba(f"{tag}-sig-{c}-{k}", row[n]["BA"], row[n]["ci95"])
            put_diff(f"{tag}-sig-{c}-erpres", j2["paired_ERPwindow_minus_Reservoir"][c])
            if c != "clean":
                for n, k in list(ENC.items())[:3]:
                    put_diff(f"{tag}-loss-{c}-{k}", j2["paired_clean_minus_condition"][c][n])
        for lv, row in j2["paired_spline_minus_zero"].items():
            for n, k in list(ENC.items())[:3]:
                put_diff(f"{tag}-splminuszero{int(float(lv) * 100)}-{k}", row[n])
    if j3:
        for m in ("EEGNet", "EEGNet+aug"):
            r = j3["clean_metrics"][m]
            put(f"{tag}-f1-{ENC[m]}", f3(r["macro_F1"])); put(f"{tag}-auc-{ENC[m]}", f3(r["macro_OvR_AUC"]))
        for c, row in j3["BA"].items():
            for m in ("EEGNet", "EEGNet+aug"):
                put_ba(f"{tag}-sig-{c}-{ENC[m]}", row[m]["BA"], row[m]["ci95"])
            put_diff(f"{tag}-augminus-{c}", j3["paired_aug_minus_unaug"][c])
            for m in ("EEGNet", "EEGNet+aug"):
                put_diff(f"{tag}-{ENC[m]}minuserp-{c}", j3["paired_minus_ERPwindow"][c][m])
            if c != "clean":
                for m in ("EEGNet", "EEGNet+aug"):
                    put_diff(f"{tag}-loss-{c}-{ENC[m]}", j3["paired_clean_minus_condition"][c][m])
        put_ba(f"{tag}-clean-eegnet", j3["BA"]["clean"]["EEGNet"]["BA"], j3["BA"]["clean"]["EEGNet"]["ci95"])
        put_ba(f"{tag}-clean-aug", j3["BA"]["clean"]["EEGNet+aug"]["BA"], j3["BA"]["clean"]["EEGNet+aug"]["ci95"])
    if j5:
        for n, k in list(ENC.items())[:3]:
            put(f"{tag}-inv-{k}", f"{j5['clean_invariance_max_abs_proba_diff'][n]:.0e}".replace("e-0", "e-"))
            for lv, e in j5["dropout"][n].items():
                for fk, r in e.items():
                    if fk.startswith("kappa_") or fk == "native_zero":
                        nm = fk.replace("kappa_", "k").replace("+", "p").replace("-", "m").replace(".", "")
                        put_ba(f"{tag}-o{lv}-{nm}-{k}", r["BA"], r["ci95"])
                put_diff(f"{tag}-o{lv}-nativeminusmean-{k}", e["native_minus_mean"])
                put_diff(f"{tag}-o{lv}-km2minuskp2-{k}", e["kappa-2_minus_kappa+2"])
            p = j5["predicted_logit_shift_30"][n]
            put(f"{tag}-dlogit-{k}", f"{p['mean_class_centered_norm']:.2f}")
            put(f"{tag}-flip-{k}", f"{100 * p['fraction_predictions_flipped_zero_vs_mean']:.1f}")
            put(f"{tag}-zpos-{k}", f"{p['mean_abs_mu_over_sigma_dropped']:.2f}")
    if j6:
        put(f"{tag}-prestim-corr", f"{j6['corr_prestim_mean_Z_vs_poststim_mean_uV']:.2f}")
        for v, row in j6["BA"].items():
            key = v.replace("Reservoir:", "res-").replace("ERP-window:", "erp-").replace("_", "")
            for nm in ("Z", "G"):
                put_ba(f"{tag}-pre-{key}-{nm}", row[nm]["BA"], row[nm]["ci95"])
            put_diff(f"{tag}-pre-{key}-ZminusG", j6["paired_Z_minus_G"][v])
    return j1, j2, j3, j5, j6


def edge_values():
    j4 = load(JAG / "j4_tcrzem_edge_long.json") or load(JAG / "j4_tcrzem_edge.json")
    if not j4 or "summary" not in j4:
        return None
    S = j4["summary"]
    import numpy as np
    rs = np.array(S["rho_star"])
    put("e-rhostar-mean", f"{rs.mean():.2f}"); put("e-rhostar-sd", f"{rs.std(ddof=1):.2f}")
    put("e-rhostar-min", f"{rs.min():.2f}"); put("e-rhostar-max", f"{rs.max():.2f}")
    put("e-ndraws", str(len(rs))); put("e-naccdraws", str(len(S["acc_draws"])))
    for i, r in enumerate(j4["rho_acc"]):
        put(f"e-ba-{r}-mean", f3(S["BA_clean_mean"][i])); put(f"e-ba-{r}-sd", f3(S["BA_clean_sd"][i]))
        put(f"e-ba30-{r}-mean", f3(S["BA_signal30_mean"][i]))
    for c in ("clean", "signal30"):
        put(f"e-n-rho0-sig-{c}", str(S[f"n_draws_rho0_below_rho0.9_ci_excludes0_{c}"]))
        put(f"e-n-rho15-sig-{c}", str(S[f"n_draws_rho1.5_below_rho0.9_ci_excludes0_{c}"]))
        put(f"e-argmax-{c}", ", ".join(str(x) for x in S[f"argmax_rho_{c}"]))
    em = j4.get("edge_maps_draw42", {})
    for th, e in em.get("theta", {}).items():
        put(f"e-rhostar-theta{th}", f"{e['rho_star']:.2f}")
    for be, e in em.get("beta", {}).items():
        put(f"e-rhostar-beta{be}", f"{e['rho_star']:.2f}")
    put("e-rhostar-42", f"{j4['draws']['42']['rho_star']:.2f}")
    return j4


# ----------------------------------------------------------------------------- tables
def table_clean(e5, e3, x, xl, x3, xl3):
    rows = []
    def fixed(lbl, src, tag):
        for n in ("ERP-window", "Band-power", "Reservoir"):
            r = src["clean_metrics"][n]
            ci = r.get("ci95", r.get("BA_ci95"))
            null = src["permutation_null"][n]["max"]
            rows.append(f" & {n} & {cell(r['BA'], ci)} & ${f3(r['macro_F1'])}$ & ${f3(r['macro_OvR_AUC'])}$ & ${f3(null)}$ \\\\")
    def eeg(src, cond_key, has_metrics):
        for m, lab in (("EEGNet", "EEGNet"), ("EEGNet+aug", "EEGNet + aug.")):
            r = src["BA"][cond_key][m]
            if has_metrics:
                cm = src["clean_metrics"][m]
                rows.append(f" & {lab} & {cell(r['BA'], r['ci95'])} & ${f3(cm['macro_F1'])}$ & ${f3(cm['macro_OvR_AUC'])}$ & -- \\\\")
            else:
                rows.append(f" & {lab} & {cell(r['BA'], r['ci95'])} & -- & -- & -- \\\\")
    blocks = [("SHAPE ($N{=}211$)", lambda: (fixed(None, e5, "s"), eeg(e3, "remove_0.0", False)))]
    if x:
        blocks.append(("External, $-200$..$800$~ms", lambda: (fixed(None, x, "x"), x3 and eeg(x3, "clean", True))))
    if xl:
        blocks.append(("External, long epoch", lambda: (fixed(None, xl, "xl"), xl3 and eeg(xl3, "clean", True))))
    out = []
    for i, (lab, fn) in enumerate(blocks):
        start = len(rows)
        fn()
        n = len(rows) - start
        rows[start] = f"\\multirow{{{n}}}{{*}}{{\\shortstack[l]{{{lab}}}}}" + rows[start]
        if i < len(blocks) - 1:
            rows.append("\\midrule")
    (GEN / "tab_clean.tex").write_text("\n".join(rows) + "\n")


def table_fills(e2, x, xl):
    rows = []
    blocks = [("SHAPE", "s", e2)]
    if xl:
        blocks.append(("External, long", "xl", xl))
    if x:
        blocks.append(("External, $-200$..$800$", "x", x))
    for bi, (lab, tag, src) in enumerate(blocks):
        for fi, fill in enumerate(("zero", "mean", "knn", "spatial")):
            if tag == "s":
                e = src["dropout_30"][fill]
                vals = [e[n]["drop_BA"] for n in ("Band-power", "ERP-window", "Reservoir")]
                d = src["paired_ERPwindow_minus_Reservoir_30"][fill]
            else:
                e = src["dropout"]["30"][fill]
                vals = [e[n]["BA"] for n in ("Band-power", "ERP-window", "Reservoir")]
                d = e["ERP-window_minus_Reservoir"]
            name = {"zero": "Zero", "mean": "Mean", "knn": "$k$NN", "spatial": "Spatial"}[fill]
            pre = f"\\multirow{{4}}{{*}}{{{lab}}}" if fi == 0 else ""
            rows.append(f"{pre} & {name} & " + " & ".join(f"${f3(v)}$" for v in vals) + f" & {dcell(d)} \\\\")
        if bi < len(blocks) - 1:
            rows.append("\\midrule")
    (GEN / "tab_fills.tex").write_text("\n".join(rows) + "\n")


def table_signal_ext(tag, j2, j3):
    if not j2:
        return
    lab = {"clean": "Clean", "remove_0.1": "$10\\%$ removed", "remove_0.3": "$30\\%$ removed",
           "remove_0.5": "$50\\%$ removed", "spline_0.1": "$10\\%$ spline", "spline_0.3": "$30\\%$ spline",
           "spline_0.5": "$50\\%$ spline", "amp_5dB": "Noise $5$~dB", "jitter_50ms": "Jitter $\\pm50$~ms",
           "trials_1": "$1$ trial", "trials_4": "$4$ trials", "trials_16": "$16$ trials"}
    rows = []
    for c, l in lab.items():
        r = j2["BA"][c]
        cells = [f"${f3(r[n]['BA'])}$" for n in ("Band-power", "ERP-window", "Reservoir")]
        for m in ("EEGNet", "EEGNet+aug"):
            cells.append(f"${f3(j3['BA'][c][m]['BA'])}$" if j3 and c in j3["BA"] else "--")
        rows.append(f"{l} & " + " & ".join(cells) + " \\\\")
        if c in ("clean", "remove_0.5", "spline_0.5", "jitter_50ms"):
            rows.append("\\midrule")
    (GEN / f"tab_signal_{tag}.tex").write_text("\n".join(rows) + "\n")


def table_signal_shape(e4, e3):
    lab = {"clean": ("clean", "remove_0.0", "Clean"), "remove_0.1": ("remove_0.1", "remove_0.1", "$10\\%$ removed"),
           "remove_0.3": ("remove_0.3", "remove_0.3", "$30\\%$ removed"),
           "remove_0.5": ("remove_0.5", "remove_0.5", "$50\\%$ removed"),
           "amp_5dB": ("amp_5dB", "amp_5dB", "Noise $5$~dB"), "jitter_50ms": ("jitter_50ms", "jitter_50ms", "Jitter $\\pm50$~ms")}
    rows = []
    for _, (c4, c3, l) in lab.items():
        r = e4["BA"][c4]
        cells = [f"${f3(r[n]['BA'])}$" for n in ("Band-power", "ERP-window", "Reservoir")]
        cells += [f"${f3(e3['BA'][c3][m]['BA'])}$" for m in ("EEGNet", "EEGNet+aug")]
        rows.append(f"{l} & " + " & ".join(cells) + " \\\\")
    (GEN / "tab_signal_shape.tex").write_text("\n".join(rows) + "\n")


def table_edge(j4):
    if not j4:
        return
    rows = []
    ra = j4["rho_acc"]
    for d in j4["summary"]["acc_draws"]:
        r = j4["draws"][str(d)]
        best = max(ra, key=lambda x: r["BA"][str(x)]["clean"]["BA"])
        p0 = r["paired_vs_rho0.9"]["0.0"]["clean"]; p15 = r["paired_vs_rho0.9"]["1.5"]["clean"]
        rows.append(f"{d} & ${r['rho_star']:.2f}$ & " + " & ".join(f"${f3(r['BA'][str(x)]['clean']['BA'])}$" for x in ra)
                    + f" & ${best}$ & {dcell(p0)} \\\\")
    S = j4["summary"]
    rows.append("\\midrule")
    rows.append(f"Mean & ${sum(S['rho_star']) / len(S['rho_star']):.2f}$ & " + " & ".join(f"${f3(v)}$" for v in S["BA_clean_mean"]) + " & & \\\\")
    (GEN / "tab_edge.tex").write_text("\n".join(rows) + "\n")


def table_origin(tag, j5, j1):
    if not j5:
        return
    rows = []
    for n in ("Band-power", "ERP-window", "Reservoir"):
        e = j5["dropout"][n]["30"]
        p = j5["predicted_logit_shift_30"][n]
        rows.append(f"{n} & ${p['mean_abs_mu_over_sigma_dropped']:.2f}$ & ${f3(j5['clean_BA'][n]['BA'])}$ & "
                    f"${f3(e['native_zero']['BA'])}$ & ${f3(e['kappa_-2.0']['BA'])}$ & ${f3(e['kappa_+0.0']['BA'])}$ & "
                    f"${f3(e['kappa_+2.0']['BA'])}$ & ${p['mean_class_centered_norm']:.2f}$ & "
                    f"${100 * p['fraction_predictions_flipped_zero_vs_mean']:.1f}$ \\\\")
    (GEN / f"tab_origin_{tag}.tex").write_text("\n".join(rows) + "\n")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    e5, e2, e4, e3, e1 = shape_values()
    x = ext_values("x", "")
    xl = ext_values("xl", "_long")
    j4 = edge_values()
    table_clean(e5, e3, x[0], xl[0], x[2], xl[2])
    table_fills(e2, x[0], xl[0])
    table_signal_shape(e4, e3)
    table_signal_ext("x", x[1], x[2])
    table_signal_ext("xl", xl[1], xl[2])
    table_edge(j4)
    table_origin("x", x[3], x[0])
    table_origin("xl", xl[3], xl[0])
    lines = ["% Generated by analysis/export_journal_values.py; do not edit by hand.",
             "\\makeatletter",
             "\\newcommand{\\V}[1]{\\ifcsname jv@#1\\endcsname\\csname jv@#1\\endcsname"
             "\\else\\PackageError{values}{Unknown value key #1}{Run analysis/export_journal_values.py}\\fi}"]
    for k, v in sorted(VALUES.items()):
        lines.append(f"\\expandafter\\def\\csname jv@{k}\\endcsname{{{v}}}")
    lines.append("\\makeatother")
    (GEN / "values.tex").write_text("\n".join(lines) + "\n")
    with open(JAG / "journal_values.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["key", "value"])
        for k, v in sorted(VALUES.items()):
            w.writerow([k, v])
    print(f"[export] {len(VALUES)} values; tables: {sorted(p.name for p in GEN.glob('tab_*.tex'))}")


if __name__ == "__main__":
    main()

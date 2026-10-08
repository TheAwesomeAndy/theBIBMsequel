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


def ceil3(v):
    import math
    return f"{math.ceil(v * 1000 - 1e-9) / 1000:.3f}"


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
    for c, row in e4["BA"].items():
        cc = c.replace("_", "")
        for n, k in list(ENC.items())[:3]:
            put_ba(f"s-sig-{cc}-{k}", row[n]["BA"], row[n]["ci95"])
        if c != "clean":
            for n, k in list(ENC.items())[:3]:
                put_diff(f"s-loss-{cc}-{k}", e4["paired_clean_minus_condition"][c][n])
        put_diff(f"s-sig-{cc}-erpres", e4["paired_ERPwindow_minus_Reservoir"][c])
    for c, row in e3["BA"].items():
        cc = "clean" if c == "remove_0.0" else c.replace("_", "")
        for m in ("EEGNet", "EEGNet+aug"):
            put_ba(f"s-sig-{cc}-{ENC[m]}", row[m]["BA"], row[m]["ci95"])
        put_diff(f"s-augminus-{cc}", e3["paired_aug_minus_unaug"][c])
        if c != "remove_0.0":
            for m in ("EEGNet", "EEGNet+aug"):
                put_diff(f"s-loss-{cc}-{ENC[m]}", e3["paired_clean_minus_condition"][c][m])
    deap = load(AGG / "deap_replication_v2.json")
    if deap:
        bp = deap["BandPower"]
        put_ba("d-clean", bp["clean"]["subject_mean_BA"], bp["clean"]["subject_ci95"])
        for lv in ("0.10", "0.30", "0.50"):
            for fill, r in bp[lv].items():
                put_ba(f"d-{int(float(lv) * 100)}-{fill}", r["subject_mean_BA"], r["subject_ci95"])
        gap = max(max(bp[lv]["mean"]["subject_mean_BA"], bp[lv]["knn"]["subject_mean_BA"]) - bp[lv]["zero"]["subject_mean_BA"]
                  for lv in ("0.10", "0.30", "0.50"))
        put("d-maxgap", f"{gap:.2f}")
        put("d-margin", f"{bp['clean']['subject_mean_BA'] - 0.5:.2f}")
    put(f"s-rhostar", "0.82")
    put_ba("s-rho09-clean", e1["BA"]["0.9"]["clean"]["BA"], e1["BA"]["0.9"]["clean"]["ci95"])
    put_diff("s-rho0-minus-09", e1["paired_vs_rho0.9"]["0.0"]["clean"])
    put_diff("s-rho15-minus-09", e1["paired_vs_rho0.9"]["1.5"]["clean"])
    return e5, e2, e4, e3, e1


def ext_values(tag, suffix, cohort="tcrzem"):
    """Values from the journal runs of one cohort and epoch. Tags: x / xl (external,
    matched / long epoch); sj / so / sp (SHAPE rerun: conference windows / windows from
    true onset / whole post-onset window)."""
    j1, j2, j3, j5, j6 = (load(JAG / f"{n.replace('tcrzem', cohort)}{suffix}.json") for n in
                          ("j1_tcrzem_core", "j2_tcrzem_signal", "j3_tcrzem_eegnet", "j5_tcrzem_origin",
                           "j6_tcrzem_prestimulus"))
    if j3 is None and cohort == "tcrzem":  # long-epoch EEGNet runs on the seed-42 partition only
        j3 = load(JAG / f"j3_tcrzem_eegnet_s42{suffix}.json")
    if j3 is not None:
        put(f"{tag}-eegnet-seeds", ", ".join(str(x) for x in j3.get("cv_seeds", [42, 43, 44, 45, 46])))
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
            cc = c.replace("_", "")
            for n, k in list(ENC.items())[:3]:
                put_ba(f"{tag}-sig-{cc}-{k}", row[n]["BA"], row[n]["ci95"])
                put(f"{tag}-rawret-{cc}-{k}", f"{row[n]['BA'] / j2['BA']['clean'][n]['BA']:.2f}")
            put_diff(f"{tag}-sig-{cc}-erpres", j2["paired_ERPwindow_minus_Reservoir"][c])
            if c != "clean":
                for n, k in list(ENC.items())[:3]:
                    put_diff(f"{tag}-loss-{cc}-{k}", j2["paired_clean_minus_condition"][c][n])
        for lv, row in j2["paired_spline_minus_zero"].items():
            for n, k in list(ENC.items())[:3]:
                put_diff(f"{tag}-splminuszero{int(float(lv) * 100)}-{k}", row[n])
    if j3:
        for m in ("EEGNet", "EEGNet+aug"):
            r = j3["clean_metrics"][m]
            put(f"{tag}-f1-{ENC[m]}", f3(r["macro_F1"])); put(f"{tag}-auc-{ENC[m]}", f3(r["macro_OvR_AUC"]))
        for c, row in j3["BA"].items():
            cc = c.replace("_", "")
            for m in ("EEGNet", "EEGNet+aug"):
                put_ba(f"{tag}-sig-{cc}-{ENC[m]}", row[m]["BA"], row[m]["ci95"])
            put_diff(f"{tag}-augminus-{cc}", j3["paired_aug_minus_unaug"][c])
            for m in ("EEGNet", "EEGNet+aug"):
                put_diff(f"{tag}-{ENC[m]}minuserp-{cc}", j3["paired_minus_ERPwindow"][c][m])
            if c != "clean":
                for m in ("EEGNet", "EEGNet+aug"):
                    put_diff(f"{tag}-loss-{cc}-{ENC[m]}", j3["paired_clean_minus_condition"][c][m])
        put(f"{tag}-augminus-max", s3(max(v["mean_diff"] for v in j3["paired_aug_minus_unaug"].values())))
        put(f"{tag}-augminus-sigconds", ", ".join(c for c, v in j3["paired_aug_minus_unaug"].items() if v["ci95"][0] > 0))
        put_ba(f"{tag}-clean-eegnet", j3["BA"]["clean"]["EEGNet"]["BA"], j3["BA"]["clean"]["EEGNet"]["ci95"])
        put_ba(f"{tag}-clean-aug", j3["BA"]["clean"]["EEGNet+aug"]["BA"], j3["BA"]["clean"]["EEGNet+aug"]["ci95"])
    if j5:
        for n, k in list(ENC.items())[:3]:
            m, e = f"{j5['clean_invariance_max_abs_proba_diff'][n]:.1e}".split("e")
            put(f"{tag}-inv-{k}", f"{m}\\times10^{{{int(e)}}}")
            if "clean_invariance_fraction_labels_changed_per_fold_max" in j5:
                put(f"{tag}-invlabels-{k}", f"{100 * j5['clean_invariance_fraction_labels_changed_per_fold_max'][n]:.1f}")
            for lv, e in j5["dropout"][n].items():
                for fk, d in e.get("kappa0_minus", {}).items():
                    nm = fk.replace("kappa_", "k").replace("+", "p").replace("-", "m").replace(".", "")
                    put_diff(f"{tag}-o{lv}-k0minus{nm}-{k}", d)
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
    j7 = load(JAG / f"j7_{cohort}_prestim_null{suffix}.json")
    if j7:
        for nm, r in j7["results"].items():
            put(f"{tag}-pnull-{nm}-p", f"{r['p_value']:.3f}" if r["p_value"] >= 0.001 else "<0.001")
            put(f"{tag}-pnull-{nm}-p95", f3(r["null_p95"])); put(f"{tag}-pnull-{nm}-mean", f3(r["null_mean"]))
            put(f"{tag}-pnull-{nm}-ba", f3(r["BA"]))
        put(f"{tag}-pnull-n", str(j7["n_perm"]))
    if j6:
        put(f"{tag}-prestim-corr", f"{j6['corr_prestim_mean_Z_vs_poststim_mean_uV']:.2f}")
        for v, row in j6["BA"].items():
            key = v.replace("Reservoir:", "res-").replace("ERP-window:", "erp-").replace("_", "")
            for nm in ("Z", "G"):
                put_ba(f"{tag}-pre-{key}-{nm}", row[nm]["BA"], row[nm]["ci95"])
            put_diff(f"{tag}-pre-{key}-ZminusG", j6["paired_Z_minus_G"][v])
        for nm, d in j6["paired_post_minus_conference_reservoir"].items():
            put_diff(f"{tag}-pre-postminusconf-{nm}", d)
        for w, r in j6["firing_rate_Z"].items():
            put(f"{tag}-pre-rate-{w.replace('_', '')}", f"{r:.3f}")
    return j1, j2, j3, j5, j6


def derived(e5, x1, xl1, more=()):
    """Summary values computed across sources (margins, agreement bounds, CI widths)."""
    third = 1.0 / 3.0
    for n, k in list(ENC.items())[:3]:
        ms = e5["clean_metrics"][n]["BA"] - third
        put(f"s-margin-{k}", f3(ms))
        for tag, src in (("x", x1), ("xl", xl1)) + tuple(more):
            if src:
                mx = src["clean_metrics"][n]["BA"] - third
                put(f"{tag}-margin-{k}", f3(mx))
                put(f"{tag}-marginratio-{k}", f"{mx / ms:.2f}")
    for tag, src in (("x", x1), ("xl", xl1)) + tuple(more):
        if not src:
            continue
        r = src["fill_minus_zero"]["Reservoir"]["mean"]
        put(f"{tag}-resfill-maxabs", f"{max(abs(v['mean_diff']) for v in r.values()):.3f}")
        w = [v["ci95"][1] - v["ci95"][0] for v in r.values()]
        put(f"{tag}-resfill-wmin", f"{min(w):.3f}"); put(f"{tag}-resfill-wmax", f"{max(w):.3f}")
        w2 = [v["ci95"][1] - v["ci95"][0] for n in ("Band-power", "ERP-window")
              for v in src["fill_minus_zero"][n]["mean"].values()]
        put(f"{tag}-nonfill-wmin", f"{min(w2):.3f}"); put(f"{tag}-nonfill-wmax", f"{max(w2):.3f}")
    if x1 and xl1:
        allr = [abs(v["mean_diff"]) for src in (x1, xl1) for v in src["fill_minus_zero"]["Reservoir"]["mean"].values()]
        put("x-resfill-maxabs-both", ceil3(max(allr)))


def recipe_values():
    rows = []
    names = {"default": "Default ($80$ ep., $p{=}0.25$)", "epochs40": "$40$ epochs",
             "epochs160": "$160$ epochs", "dropout50": "Dropout $0.5$"}
    for r, lab in names.items():
        d = load(JAG / ("j3_tcrzem_eegnet_s42.json" if r == "default" else f"j3_tcrzem_eegnet_{r}_s42.json"))
        if not d:
            continue
        cells = []
        for m in ("EEGNet", "EEGNet+aug"):
            for c in ("clean", "remove_0.3", "remove_0.5", "amp_5dB"):
                put_ba(f"rc-{r}-{c}-{ENC[m]}", d["BA"][c][m]["BA"], d["BA"][c][m]["ci95"])
                cells.append(f"${f3(d['BA'][c][m]['BA'])}$")
        for c in ("clean", "remove_0.3", "remove_0.5"):
            put_diff(f"rc-{r}-augminus-{c}", d["paired_aug_minus_unaug"][c])
        a = d["paired_aug_minus_unaug"]["remove_0.5"]
        rows.append(f"{lab} & " + " & ".join(cells) + f" & {dcell(a)} \\\\")
    if rows:
        (GEN / "tab_recipes.rows").write_text("\n".join(rows) + "\n")
    got = [r for r in names if load(JAG / ("j3_tcrzem_eegnet_s42.json" if r == "default" else f"j3_tcrzem_eegnet_{r}_s42.json"))]
    if got:
        aug50 = [float(VALUES[f"rc-{r}-augminus-remove_0.5"]) for r in got]
        put("rc-aug50-min", s3(min(aug50))); put("rc-aug50-max", s3(max(aug50)))
        sig = [r for r in got if float(VALUES[f"rc-{r}-augminus-remove_0.5-lo"]) > 0]
        put("rc-aug50-nsig", str(len(sig))); put("rc-n", str(len(got)))
        cl = [float(VALUES[f"rc-{r}-clean-eegnet"]) for r in got]
        put("rc-clean-min", f3(min(cl))); put("rc-clean-max", f3(max(cl)))
        augc = [float(VALUES[f"rc-{r}-augminus-clean"]) for r in got]
        put("rc-augclean-min", s3(min(augc))); put("rc-augclean-max", s3(max(augc)))
        put("rc-augclean-nsig", str(sum(float(VALUES[f"rc-{r}-augminus-clean-lo"]) > 0 for r in got)))
        lu = [float(VALUES[f"rc-{r}-clean-eegnet"]) - float(VALUES[f"rc-{r}-remove_0.5-eegnet"]) for r in got]
        la = [float(VALUES[f"rc-{r}-clean-aug"]) - float(VALUES[f"rc-{r}-remove_0.5-aug"]) for r in got]
        put("rc-loss50-unaug-min", f3(min(lu))); put("rc-loss50-unaug-max", f3(max(lu)))
        put("rc-loss50-aug-min", f3(min(la))); put("rc-loss50-aug-max", f3(max(la)))
        red = [a - b for a, b in zip(lu, la)]
        put("rc-lossred-min", f3(min(red))); put("rc-lossred-max", ceil3(max(red)))


def erp_summary():
    e = load(JAG / "tcrzem_erp_summary.json")
    if e:
        put("t-sal-uv", f"{e['salience_500_1300ms_uV']:.2f}"); put("t-sal-dz", f"{e['salience_within_subject_dz']:.2f}")
        put("t-val-uv", f"{e['valence_1500_2500ms_uV']:.2f}"); put("t-val-dz", f"{e['valence_within_subject_dz']:.2f}")
        put("t-between-sd", f"{e['neutral_amplitude_between_subject_sd_uV']:.2f}"); put("t-channel", e["channel"])


def extra_values(e1, xl2):
    if xl2:
        losses = [xl2["paired_clean_minus_condition"]["spline_0.5"][n]["mean_diff"] for n in list(ENC)[:3]]
        put("xl-spline50-maxloss", ceil3(max(losses)))
    for r in e1["rho_grid"]:
        put(f"s-e1-ba30-{r}", f3(e1["BA"][str(r)]["signal"]["BA"]))
        put(f"s-e1-loss30-{r}", f3(e1["BA"][str(r)]["clean"]["BA"] - e1["BA"][str(r)]["signal"]["BA"]))
    j4 = load(JAG / "j4_tcrzem_edge_long.json")
    if j4:
        d = max(abs(e1["damage"][k] - j4["draws"]["42"]["damage"][k]) for k in e1["damage"])
        put("e-damage-maxdiff-42", ceil3(d))


def edge_values(pre="e", j4name="j4_tcrzem_edge_long.json", j8name="j8_tcrzem_silence_long.json"):
    """Operating-point values. Prefixes: e (external, long epoch), se / seo / sep (SHAPE,
    accuracy on the conference / onset / post windows; order parameter always on the
    conference drives and window)."""
    j4 = load(JAG / j4name)
    if not j4 or "summary" not in j4:
        return None
    S = j4["summary"]
    import numpy as np
    rs = np.array(S["rho_star"])
    put(f"{pre}-rhostar-mean", f"{rs.mean():.2f}"); put(f"{pre}-rhostar-sd", f"{rs.std(ddof=1):.2f}")
    put(f"{pre}-rhostar-min", f"{rs.min():.2f}"); put(f"{pre}-rhostar-max", f"{rs.max():.2f}")
    put(f"{pre}-ndraws", str(len(rs))); put(f"{pre}-naccdraws", str(len(S["acc_draws"])))
    for i, r in enumerate(j4["rho_acc"]):
        put(f"{pre}-ba-{r}-mean", f3(S["BA_clean_mean"][i])); put(f"{pre}-ba-{r}-sd", f3(S["BA_clean_sd"][i]))
        put(f"{pre}-ba-{r}-min", f3(S["BA_clean_min"][i])); put(f"{pre}-ba-{r}-max", f3(S["BA_clean_max"][i]))
        put(f"{pre}-ba30-{r}-mean", f3(S["BA_signal30_mean"][i]))
        put(f"{pre}-loss30-{r}", f3(S["BA_clean_mean"][i] - S["BA_signal30_mean"][i]))
        put(f"{pre}-rate-{r}", f"{S['rate_mean'][i]:.2f}")
    losses = [S["BA_clean_mean"][i] - S["BA_signal30_mean"][i] for i in range(len(j4["rho_acc"]))]
    put(f"{pre}-loss30-max", ceil3(max(losses)))
    lo_i = [i for i, r in enumerate(j4["rho_acc"]) if r <= 0.6]; hi_i = [i for i, r in enumerate(j4["rho_acc"]) if r >= 0.9]
    put(f"{pre}-loss30-low-max", ceil3(max(losses[i] for i in lo_i)))
    put(f"{pre}-loss30-high-min", f3(min(losses[i] for i in hi_i)))
    put(f"{pre}-loss30-high-max", f3(max(losses[i] for i in hi_i)))
    for c in ("clean", "signal30"):
        put(f"{pre}-n-rho0-sig-{c}", str(S[f"n_draws_rho0_below_rho0.9_ci_excludes0_{c}"]))
        put(f"{pre}-n-rho15-sig-{c}", str(S[f"n_draws_rho1.5_below_rho0.9_ci_excludes0_{c}"]))
        put(f"{pre}-argmax-{c}", ", ".join(str(x) for x in S[f"argmax_rho_{c}"]))
        if f"n_draws_rho0_point_below_rho0.9_{c}" in S:
            put(f"{pre}-n-rho0-below-{c}", str(S[f"n_draws_rho0_point_below_rho0.9_{c}"]))
    for r, row in S.get("drawmean_paired_vs_rho0.9", {}).items():
        for c, d in row.items():
            put_diff(f"{pre}-dm-{r}minus09-{c}", d)
            put(f"{pre}-dm-{r}minus09-{c}-point", s3(d["point"]))
    em = j4.get("edge_maps_draw42", {})
    for th, e in em.get("theta", {}).items():
        put(f"{pre}-rhostar-theta{th}", f"{e['rho_star']:.2f}")
    for be, e in em.get("beta", {}).items():
        put(f"{pre}-rhostar-beta{be}", f"{e['rho_star']:.2f}")
    put(f"{pre}-rhostar-42", f"{j4['draws']['42']['rho_star']:.2f}")
    j8 = load(JAG / j8name)
    if j8:
        for r in j8["rho"]:
            put(f"{pre}-z0-{r}", f"{j8['silent_block_mean_abs_z'][str(r)]:.2f}")
            put(f"{pre}-rawcent-{r}", f"{j8['raw_counts_mean_abs_mu_over_sigma'][str(r)]:.2f}")
        low = [j8["silent_block_mean_abs_z"][str(r)] for r in j8["rho"] if r <= 0.6]
        put(f"{pre}-z0-low-min", f"{min(low):.2f}"); put(f"{pre}-z0-low-max", f"{max(low):.2f}")
        high = [j8["silent_block_mean_abs_z"][str(r)] for r in j8["rho"] if r >= 0.9]
        put(f"{pre}-z0-high-min", f"{min(high):.2f}"); put(f"{pre}-z0-high-max", f"{max(high):.2f}")
        put(f"{pre}-z0-all-max", f"{max(j8['silent_block_mean_abs_z'].values()):.2f}")
    return j4


def repro_values(e5, e2, e1, sj1, sj4):
    """Largest absolute difference between the SHAPE rerun and every conference value it
    repeats (clean BA and CI bounds; 30% fills; the draw-42 rho sweep and damage curve)."""
    diffs = []
    if sj1:
        for n in ("Band-power", "ERP-window", "Reservoir"):
            a, b = sj1["clean_metrics"][n], e5["clean_metrics"][n]
            diffs += [abs(a["BA"] - b["BA"])] + [abs(x - y) for x, y in zip(a["ci95"], b["BA_ci95"])]
            for fill in ("zero", "mean", "knn", "spatial"):
                diffs.append(abs(sj1["dropout"]["30"][fill][n]["BA"] - e2["dropout_30"][fill][n]["drop_BA"]))
    if sj4:
        r = sj4["draws"]["42"]
        for rho in e1["rho_grid"]:
            for c, ce in (("clean", "clean"), ("signal30", "signal")):
                diffs.append(abs(r["BA"][str(rho)][c]["BA"] - e1["BA"][str(rho)][ce]["BA"]))
        diffs += [abs(e1["damage"][k] - r["damage"][k]) for k in e1["damage"]]
    if diffs:
        put("sj-repro-maxdiff", "0" if max(diffs) == 0 else f"{max(diffs):.1e}")
        put("sj-repro-n", str(len(diffs)))


def zscore_values():
    """J9: does the reservoir read the per-epoch normalization offset (sz SHAPE, xz external)."""
    for pre, cohort in (("sz", "shape"), ("xz", "tcrzem")):
        j9 = load(JAG / f"j9_{cohort}_zscore.json")
        if not j9:
            continue
        for inp in ("Z", "G"):
            for w in ("pre", "conference"):
                e = j9[f"{inp}_{w}"]
                put(f"{pre}-{inp}-{w}-r2", f"{e['oof_R2_m_from_BSC6']:.3f}")
                put(f"{pre}-{inp}-{w}-r2count", f"{e['oof_R2_m_from_count']:.3f}")
                for kk in ("r_count_m", "r_count_absm"):
                    nm = "r" if kk == "r_count_m" else "rabs"
                    put(f"{pre}-{inp}-{w}-{nm}-med", f"{e[kk]['median']:+.2f}")
                    put(f"{pre}-{inp}-{w}-{nm}-min", f"{e[kk]['min']:+.2f}")
                    put(f"{pre}-{inp}-{w}-{nm}-max", f"{e[kk]['max']:+.2f}")
                put(f"{pre}-{inp}-{w}-rate", f"{e['rate']:.3f}")


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
    (GEN / "tab_clean.rows").write_text("\n".join(rows) + "\n")


def table_fills(e2, x, xl):
    rows = []
    blocks = [("SHAPE", "s", e2)]
    if xl:
        blocks.append(("Ext.\\ long", "xl", xl))
    if x:
        blocks.append(("Ext.\\ matched", "x", x))
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
    (GEN / "tab_fills.rows").write_text("\n".join(rows) + "\n")


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
    (GEN / f"tab_signal_{tag}.rows").write_text("\n".join(rows) + "\n")


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
    (GEN / "tab_signal_shape.rows").write_text("\n".join(rows) + "\n")


def table_edge(j4, name="tab_edge"):
    if not j4 or "summary" not in j4:
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
    rows.append(f"Mean & ${sum(S['rho_star']) / len(S['rho_star']):.2f}$ & " + " & ".join(f"${f3(v)}$" for v in S["BA_clean_mean"])
                + (f" & & {dcell(S['drawmean_paired_vs_rho0.9']['0.0']['clean'])} \\\\" if "drawmean_paired_vs_rho0.9" in S else " & & \\\\"))
    rows.append(f"$30\\%$ silent & & " + " & ".join(f"${f3(v)}$" for v in S["BA_signal30_mean"])
                + (f" & & {dcell(S['drawmean_paired_vs_rho0.9']['0.0']['signal30'])} \\\\" if "drawmean_paired_vs_rho0.9" in S else " & & \\\\"))
    (GEN / f"{name}.rows").write_text("\n".join(rows) + "\n")


def table_windows():
    """SHAPE: the conference windows against windows measured from true onset."""
    rows = []
    for ep, lab in (("", "Conference ($-160$..$+73$)"), ("_onset", "Onset ($+39$..$+273$)"),
                    ("_post", "Post ($0$..$+793$)")):
        j1 = load(JAG / f"j1_shape_core{ep}.json"); j4 = load(JAG / f"j4_shape_edge{ep}.json")
        if not j1:
            continue
        cm = j1["clean_metrics"]
        z = j1["dropout"]["30"]["zero"]["ERP-window_minus_Reservoir"]
        m = j1["dropout"]["30"]["mean"]["ERP-window_minus_Reservoir"]
        r0 = (dcell(j4["summary"]["drawmean_paired_vs_rho0.9"]["0.0"]["clean"])
              if j4 and "drawmean_paired_vs_rho0.9" in j4.get("summary", {}) else "--")
        rows.append(f"{lab} & ${f3(cm['ERP-window']['BA'])}$ & ${f3(cm['Reservoir']['BA'])}$ & {dcell(z)} & {dcell(m)} & {r0} \\\\")
    if rows:
        (GEN / "tab_windows.rows").write_text("\n".join(rows) + "\n")


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
    (GEN / f"tab_origin_{tag}.rows").write_text("\n".join(rows) + "\n")


HEADERS = {
    "tab_clean": ("llcccc", "Cohort & Encoder & BA [95\\% CI] & F1 & AUC & Null max \\\\"),
    "tab_fills": ("llcccc", "Cohort & Fill & Band & ERP & Res. & ERP$-$Res [95\\% CI] \\\\"),
    "tab_signal_shape": ("lccccc", "Condition & Band & ERP & Res. & EEGNet & +aug. \\\\"),
    "tab_signal_x": ("lccccc", "Condition & Band & ERP & Res. & EEGNet & +aug. \\\\"),
    "tab_signal_xl": ("lccccc", "Condition & Band & ERP & Res. & EEGNet & +aug. \\\\"),
    "tab_edge": ("lccccccccc", "Draw & $\\rho^\\ast$ & $0$ & $0.3$ & $0.6$ & $0.9$ & $1.2$ & $1.5$ & Best & $\\rho{=}0$ $-$ $0.9$ [95\\% CI] \\\\"),
    "tab_edge_s": ("lccccccccc", "Draw & $\\rho^\\ast$ & $0$ & $0.3$ & $0.6$ & $0.9$ & $1.2$ & $1.5$ & Best & $\\rho{=}0$ $-$ $0.9$ [95\\% CI] \\\\"),
    "tab_windows": ("lccccc", "Window (ms) & ERP & Res. & Zero fill & Mean fill & $\\rho{=}0$ $-$ $0.9$ \\\\ & \\multicolumn{2}{c}{clean BA} & \\multicolumn{2}{c}{ERP$-$Res., $30\\%$ dropout} & draw mean \\\\"),
    "tab_origin_sj": ("lcccccccc", "Encoder & $\\frac{|\\mu|}{\\sigma}$ & Clean & Zero & $\\kappa{=}{-2}$ & $\\kappa{=}0$ & $\\kappa{=}{+2}$ & $\\lVert\\Delta\\ell\\rVert$ & Flip \\% \\\\"),
    "tab_origin_x": ("lcccccccc", "Encoder & $\\frac{|\\mu|}{\\sigma}$ & Clean & Zero & $\\kappa{=}{-2}$ & $\\kappa{=}0$ & $\\kappa{=}{+2}$ & $\\lVert\\Delta\\ell\\rVert$ & Flip \\% \\\\"),
    "tab_origin_xl": ("lcccccccc", "Encoder & $\\frac{|\\mu|}{\\sigma}$ & Clean & Zero & $\\kappa{=}{-2}$ & $\\kappa{=}0$ & $\\kappa{=}{+2}$ & $\\lVert\\Delta\\ell\\rVert$ & Flip \\% \\\\"),
    "tab_recipes": ("lccccccccc", "Recipe & \\multicolumn{4}{c}{EEGNet} & \\multicolumn{4}{c}{EEGNet + aug.} & Aug.$-$unaug., $50\\%$ \\\\ & Clean & $30\\%$ & $50\\%$ & Noise & Clean & $30\\%$ & $50\\%$ & Noise & [95\\% CI] \\\\"),
}


def wrap_tables():
    """Turn every generated row file into a complete tabular (\\input outside the tabular)."""
    for name, (spec, head) in HEADERS.items():
        f = GEN / f"{name}.rows"
        if f.exists():
            body = f.read_text()
            (GEN / f"{name}.tex").write_text(
                f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n{body}\\bottomrule\n\\end{{tabular}}\n")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    e5, e2, e4, e3, e1 = shape_values()
    x = ext_values("x", "")
    xl = ext_values("xl", "_long")
    sj = ext_values("sj", "", "shape")
    so = ext_values("so", "_onset", "shape")
    sp = ext_values("sp", "_post", "shape")
    derived(e5, x[0], xl[0], more=(("sj", sj[0]), ("so", so[0]), ("sp", sp[0])))
    recipe_values()
    erp_summary()
    extra_values(e1, xl[1])
    j4 = edge_values()
    sj4 = edge_values("se", "j4_shape_edge.json", "j8_shape_silence.json")
    edge_values("seo", "j4_shape_edge_onset.json", "j8_shape_silence_onset.json")
    edge_values("sep", "j4_shape_edge_post.json", "j8_shape_silence_post.json")
    repro_values(e5, e2, e1, sj[0], sj4)
    zscore_values()
    table_clean(e5, e3, x[0], xl[0], x[2], xl[2])
    table_fills(e2, x[0], xl[0])
    table_signal_shape(e4, e3)
    table_signal_ext("x", x[1], x[2])
    table_signal_ext("xl", xl[1], xl[2])
    table_edge(j4)
    table_edge(sj4, "tab_edge_s")
    table_windows()
    table_origin("sj", sj[3], sj[0])
    table_origin("x", x[3], x[0])
    table_origin("xl", xl[3], xl[0])
    wrap_tables()
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

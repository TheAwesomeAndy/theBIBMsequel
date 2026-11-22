"""Paper B tables from outputs/X11 and outputs/X12. No number is typed by hand."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
OUT = R / "paperB" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
x11 = json.loads((R / "outputs/X11/x11_shape.json").read_text()); b = x11["balanced_accuracy"]
x12 = json.loads((R / "outputs/X12/x12_results.json").read_text())["results"]
law = json.loads((R / "outputs/X12/x12_law_summary.json").read_text())
e11 = json.loads((R / "outputs/X11/x11_erpcore.json").read_text())["results"]
f = lambda d: f"{d['ba']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
g = lambda d: f"${d['diff']:+.3f}$ [${d['ci'][0]:+.3f}$, ${d['ci'][1]:+.3f}$]"
rows = [("uncentered", "None", "--"), ("ea", "Euclidean alignment", "second"), ("tangent", "Tangent space, cohort reference", "--"),
        ("tangent_recenter", "Tangent space, participant recentering", "second"), ("panel_center", "Panel centering", "first"),
        ("subject_z", "Panel centering and scaling", "first"), ("ea_center", "Euclidean alignment, then centering", "both"),
        ("panel_center_assign", "Panel centering, joint assignment", "first")]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{SHAPE, three affective conditions (chance 0.333), 211 participants, unscaled epochs, participant-grouped 10-fold cross-validation. Balanced accuracy with participant-bootstrap 95\% intervals. Every transform uses only the participant's own unlabeled panel.}",
     r"\label{tab:shape}", r"\begin{tabular}{llc}", r"\toprule", r"Transform & Order & Balanced accuracy \\", r"\midrule"]
L += [f"{n} & {o} & {f(b[k])} \\\\" for k, n, o in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "tab_shape.tex").write_text("\n".join(L) + "\n")
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{ERP CORE. Left: participant-condition averages, complete two-condition panels (chance 0.5). Right: cross-participant single-trial decoding with the participant mean taken over all trials, the gain over the uncentered readout with participant-bootstrap 95\% interval, the gain of Euclidean alignment, the break-even reference size $m^\ast$ predicted by \eqref{eq:breakeven}, and the smallest tested $m\in\{1,2,4,8,16,32\}$ whose accuracy exceeds the uncentered readout.}",
     r"\label{tab:erpcore}", r"\begin{tabular}{lcc|cccccc}", r"\toprule",
     r" & \multicolumn{2}{c|}{Averages} & \multicolumn{6}{c}{Single trials} \\",
     r"Paradigm & Uncentered & Panel centered & Uncentered & Centered & Centering gain & Alignment gain & $m^\ast$ & First $m$ above \\", r"\midrule"]
for p in x12:
    s = law["per_paradigm"][p]; ms = s["m_star_predicted"]
    L.append(f"{p} & {e11[p]['uncentered']['ba']:.3f} & {e11[p]['panel_center']['ba']:.3f} & {x12[p]['uncentered']['ba']:.3f} & {x12[p]['center_all']['ba']:.3f} & {g(x12[p]['gain_center_all'])} & {g(x12[p]['gain_ea'])} & "
             + (f"{ms:.1f}" if ms < 1e6 else r"$\infty$") + f" & {s['smallest_tested_m_above_uncentered'] or 'none'} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "tab_erpcore.tex").write_text("\n".join(L) + "\n")
print("Paper B tables written")

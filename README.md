# theBIBMsequel

Journal extension of the IEEE BIBM 2026 paper *On the Edge of Stability: Spiking
Reservoir State-Space Encoding of Affective EEG* (Doctoral Forum S60204).

Working journal title: **Encoding Nothing at the Edge of Stability: Missing
Electrodes, Coordinate Origins, and Empty Baselines in Spiking Reservoir
Representations of Affective EEG** (`manuscript/journal/main_journal.tex`).

This repository started as a full-history merge of
[`TheAwesomeAndy/BIBM`](https://github.com/TheAwesomeAndy/BIBM) at commit
`ad79109` (the accepted camera-ready). The conference repository is left
untouched; all journal work happens here.

## What the journal version adds

| Addition | Script | Output (`outputs/aggregate/journal/`) |
|---|---|---|
| External cohort loader (public IAPS ERP data, doi:10.34894/TCRZEM, 228 subjects), conference-matched epoch and a pre-specified long epoch | `analysis/tcrzem_data.py` | (data stay in `/home/user/data_local/tcrzem/`) |
| Bit-exact, dataset-agnostic port of the conference code | `analysis/jcore.py` | |
| Clean metrics, permutation null, PCA controls, BSC1 vs BSC6, centeredness, four fills at 10-50% | `analysis/j1_core_replication.py` | `j1_tcrzem_core{,_long}.json` |
| Signal-level removal, spherical-spline repair, noise, jitter, trial-count re-averaging | `analysis/j2_signal_level.py`, `analysis/spline.py` | `j2_tcrzem_signal{,_long}.json` |
| EEGNet with and without augmentation; recipe variants | `analysis/j3_eegnet.py` | `j3_tcrzem_eegnet*.json` |
| Edge of stability across ten reservoir draws, thresholds, leaks | `analysis/j4_edge_of_stability.py` | `j4_tcrzem_edge_long.json` |
| Direct test of Proposition 1 (origin sweep) | `analysis/j5_origin_shift.py` | `j5_tcrzem_origin{,_long}.json` |
| Pre-stimulus windows and normalization (Proposition 2) | `analysis/j6_prestimulus.py`, `analysis/j7_prestim_null.py` | `j6_*`, `j7_*` |
| Distance of a silent electrode from typical reservoir activity, per spectral radius | `analysis/j8_silence_distance.py` | `j8_tcrzem_silence_long.json` |
| Every number, table body, and inline value of the journal manuscript | `analysis/export_journal_values.py` | `manuscript/journal/generated/`, `journal_values.csv` |

Long-epoch runs set `TCRZEM_EPOCH=long`. SHAPE (restricted) and DEAP were not
reanalyzed; their numbers come from the conference aggregate outputs. See
`CLAUDE.md` (journal section) for the working rules and `analysis/README.md`
for the run order.

Build and check the journal manuscript:

```sh
python scripts/check_journal.py      # export values, build, run the journal gates
python analysis/verify_jcore.py      # journal code vs conference code on synthetic data
```

The conference README follows unchanged for provenance.

---

# BIBM 2026 Conference Paper 1

This repository contains the IEEE BIBM 2026 conference-paper carve-out from the operational-distinctness work.

## Working title

**On the Edge of Stability: Spiking Reservoir State-Space Encoding of Affective EEG**

## Submission target

IEEE International Conference on Bioinformatics and Biomedicine (BIBM) 2026. The intended route is the BIBM 2026 Doctoral Forum submission portal, with the manuscript written and formatted as a standard IEEE conference proceedings paper.

## Scope

This paper isolates the reservoir temporal-encoding component. It focuses on affective ERP observations, fixed LIF reservoir encoding, six-bin spike-count encoding, PCA-compressed reservoir embeddings, subject-level validation, and perturbation characterization.

## Reserved for later journal work

The repository should not contain the full integrated ARSPI-Net manuscript, graph-topological tPLV analysis, structure-function coupling, closed-loop EFE results, or broad diagnostic claims.

## Repository layout

| Path | What it is | Owner |
|---|---|---|
| `manuscript/` | IEEE two-column paper (`main_bibm2026.tex`), cover letter, CV appendix, build guide (`BUILD.md`) | paper |
| `manuscript/figures/imported/` | figure PDFs used by the paper (produced by the `analysis/` scripts) | paper |
| `analysis/` | every script that produces a number, table, or figure in the paper; see `analysis/README.md` for the full paper-element → script → output map | paper |
| `outputs/aggregate/` | aggregate, de-identified results (JSON/CSV) written by `analysis/` | paper |
| `docs/CODE_AUDIT.md` | camera-ready code audit: findings, fixes, and reproduction checks | paper |
| `scripts/`, `papercheck/`, `harness/`, `tools/`, `makefiles/`, `playbooks/`, `.claude/` | research/audit harness (quality gates, memory journal, agent skills) | harness |

## Harness provenance and sync status

The harness directories are installed from the audit repository
[`TheAwesomeAndy/myResearchAssitantHarness`](https://github.com/TheAwesomeAndy/myResearchAssitantHarness)
with `scripts/install_harness.py`. As of 2026-10-01 they are byte-identical to its
`main` at commit `b549c07`, with these deliberate, project-local differences:

- `harness/papercheck.toml` — project configuration (main file, single-blind venue);
- `harness/memory/memory.jsonl` — this project's memory journal;
- `scripts/verify_latex.py` — bug fix not yet upstream: the build ran inside the
  manuscript directory but passed the caller-relative path, so any manuscript in a
  subdirectory (e.g. `manuscript/main.tex`) failed to build (the harness's own
  `good_latex_build` fixture fails without the fix; 70/70 fixtures pass with it).

To re-sync, run the installer from a fresh clone of the harness repository and
re-apply the three differences above.

## Building the paper

```sh
cd manuscript
latexmk -pdf -interaction=nonstopmode main_bibm2026.tex
```

The submission file is the paper followed by the one-page CV
(`main_bibm2026_with_cv.pdf`); see `manuscript/BUILD.md`. Committed PDFs are
build artifacts and can lag the source; always rebuild before submitting.

## Reproducing the results

All analysis code, its inputs, run order, and the map from each paper element to
its script and output are in `analysis/README.md`. The table-value gate
(`python scripts/check_table_manifest.py harness/table_manifest.yaml`) checks
that the numbers printed in the paper match the aggregate outputs.

## Data access and privacy

Raw EEG and subject-level derivatives are never committed. The SHAPE data are
access-controlled (Laboratory for Clinical Affective Neuroscience, Stony Brook
University); DEAP is available from its authors under the DEAP licence. Only
aggregate, de-identified results are stored in `outputs/aggregate/`.

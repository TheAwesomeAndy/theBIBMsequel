# Figure inventory for BIBM 2026 Conference Paper 1

## Source repository reviewed

Original repository: `TheAwesomeAndy/operational-distinctness-paper`

The original TAFFC manuscript defines a broad ARSPI-Net overview figure at:

- `figures/taffc/fig_overview_evidence_streams.pdf`

The generator is:

- `experiments/tcds_ready9/generate_taffc_overview_figure.py`

That generator builds a data-backed figure with three panels relevant to this BIBM carve-out:

1. trial-averaged ERPs by affective condition;
2. LIF reservoir spike raster for an exemplar ERP observation;
3. diagnostic BSC6 reservoir projection using PCA-2.

The full TAFFC overview also includes the broader E/D/T/C evidence-stream map and closed-loop evaluation language. Those portions are journal material and should not be reused verbatim in Conference Paper 1.

## BIBM figure policy

Conference Paper 1 should use a narrowed reservoir-only figure set:

| BIBM Figure | Source status | Purpose |
|---|---|---|
| Fig. 1: Reservoir temporal-encoding pipeline | created as `fig_pipeline_bibm2026.tex` | Shows ERP -> fixed LIF reservoir -> BSC6 -> embedding -> subject-level evaluation |
| Fig. 2: BSC6 temporal-bin schematic | created as `fig_bsc6_bins_bibm2026.tex` | Defines temporal traceability of spike-count bins |
| Fig. 3: Representation-level perturbation response | created as `fig_perturbation_response_bibm2026.tex` | Shows A0 vs A1 balanced accuracy under clean, 5 dB noise, and 30% dropout |
| Optional replacement Fig. 1 | regenerate from original `generate_taffc_overview_figure.py` | Use only the ERP/raster/BSC6 panels; remove E/D/T/C and closed-loop blocks |

## Required next figure hardening

The current repository now contains compileable vector/TikZ figure sources, but the strongest final BIBM version should regenerate a narrowed data-backed overview from the original script. That regenerated figure should retain only the reservoir-relevant panels and should not copy the full TAFFC figure because the full TAFFC figure consumes journal-only material.

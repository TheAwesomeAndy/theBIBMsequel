# Building the BIBM 2026 submission PDF

The submission is the **paper followed by a one-page first-author CV**, as a
single PDF (`main_bibm2026_with_cv.pdf`). The paper and CV are different
LaTeX document classes, so they are compiled separately and merged.

## Build order (run from `manuscript/` with a TeX Live install)

```sh
pdflatex main_bibm2026.tex          # run twice so cross-references resolve
pdflatex main_bibm2026.tex
pdflatex cv_appendix_template.tex   # the one-page first-author CV
pdflatex main_bibm2026_with_cv.tex  # merges both -> main_bibm2026_with_cv.pdf
```

The bibliography uses an inline `thebibliography` (no BibTeX/`.bib`), so two
`pdflatex` passes on the paper are enough.

## What to submit

Upload **`main_bibm2026_with_cv.pdf`** to the CyberChair portal.

## Page limit

BIBM 2026 Doctoral Forum: **6 pages excluding references and appendices**,
plus the appended one-page CV. Confirm the body (through the Conclusion,
before `\begin{thebibliography}`) is within 6 pages after compiling.

## Note on committed PDFs

The `*.pdf` files checked into this directory are rendered artifacts and can
fall out of date whenever a `.tex` source changes. **Always rebuild from
source before submitting** so the PDF reflects the current text.

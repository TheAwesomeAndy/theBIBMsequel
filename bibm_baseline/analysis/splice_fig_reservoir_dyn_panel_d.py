#!/usr/bin/env python3
"""Replace panel (d) of manuscript/figures/imported/fig_reservoir_dyn.pdf in place.

Panels (a) and (b) of Fig. 1 need the restricted SHAPE pickle, so the full figure can only
be regenerated locally (python3 analysis/make_fig_reservoir_dynamics.py). Panel (d) reads
only outputs/aggregate/e1_rho_sweep.json. This script renders panel (d) with the patched
source and splices it into the committed PDF, so the committed figure can be updated from
the aggregate data alone. Preferred procedure remains the full regeneration; this one is
the fallback, and the two produce the same panel (d) because both call panel_d().

Steps
  1. render panel (d) alone:   make_fig_reservoir_dynamics.py --panel-d <tmp.pdf>
  2. locate the panel-(d) axes frame in both PDFs (the white axes-background rectangle
     with the largest x), and the right edge of panel (c)'s frame in the committed PDF;
  3. redact (remove, not cover) everything right of the midpoint between the (c) frame
     and the (d) y-label region in the committed PDF, and draw the new panel there, scaled and translated so
     that its axes frame coincides with the old one.

Usage: python3 analysis/splice_fig_reservoir_dyn_panel_d.py [--check-only]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "manuscript/figures/imported/fig_reservoir_dyn.pdf"


def axes_frames(page):
    """White filled rectangles larger than 40 x 40 pt: the axes backgrounds, by x."""
    fr = [d["rect"] for d in page.get_drawings()
          if d.get("fill") == (1.0, 1.0, 1.0) and d["rect"].width > 40 and d["rect"].height > 40
          and d["rect"].get_area() < 0.9 * page.rect.get_area()]  # drop the page background
    return sorted(fr, key=lambda r: r.x0)


def main():
    check_only = "--check-only" in sys.argv
    with tempfile.TemporaryDirectory() as td:
        new_pdf = Path(td) / "panel_d.pdf"
        subprocess.run([sys.executable, str(ROOT / "analysis/make_fig_reservoir_dynamics.py"),
                        "--panel-d", str(new_pdf)], check=True, capture_output=True)
        old = pymupdf.open(FIG); op = old[0]
        new = pymupdf.open(new_pdf); np_ = new[0]
        old_frames = axes_frames(op); new_frames = axes_frames(np_)
        assert len(old_frames) == 4, f"expected 4 axes frames in the committed figure, found {len(old_frames)}"
        assert len(new_frames) == 1, f"expected 1 axes frame in the panel render, found {len(new_frames)}"
        fc, fd_old = old_frames[2], old_frames[3]; fd_new = new_frames[0]
        print(f"old (d) frame {fd_old}; new (d) frame {fd_new}; (c) frame right edge {fc.x1:.1f}")
        if check_only:
            return
        # 3. white-out and overlay. The cut sits between panel (c)'s frame and panel (d)'s
        #    y-label; both figures place the y-label about 18 pt left of the frame.
        cut_x = (fc.x1 + (fd_old.x0 - 22)) / 2
        clip_old = pymupdf.Rect(cut_x, op.rect.y0 - 5, op.rect.x1 + 5, op.rect.y1 + 5)
        # redaction removes the old text and vector content, not just covers it
        op.add_redact_annot(clip_old, fill=(1, 1, 1))
        op.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
        # source clip in the new page: same offset from its frame as the old clip has from the old frame
        scale = fd_old.width / fd_new.width
        assert abs(scale - fd_old.height / fd_new.height) < 1e-3, "aspect mismatch between renders"
        src = pymupdf.Rect(fd_new.x0 - (fd_old.x0 - cut_x) / scale, fd_new.y0 - (fd_old.y0 - op.rect.y0) / scale,
                           fd_new.x0 + (op.rect.x1 - fd_old.x0) / scale, fd_new.y0 + (op.rect.y1 - fd_old.y0) / scale)
        src = src & np_.rect
        dst = pymupdf.Rect(fd_old.x0 - (fd_new.x0 - src.x0) * scale, fd_old.y0 - (fd_new.y0 - src.y0) * scale,
                           fd_old.x0 + (src.x1 - fd_new.x0) * scale, fd_old.y0 + (src.y1 - fd_new.y0) * scale)
        op.show_pdf_page(dst, new, 0, clip=src, overlay=True)
        out = Path(td) / "spliced.pdf"
        old.subset_fonts()
        old.save(out, garbage=4, deflate=True, clean=True)
        old.close(); new.close()
        FIG.write_bytes(out.read_bytes())
        chk = pymupdf.open(FIG)[0]
        for word in ("ordered", "irregular", "transition"):
            assert not chk.search_for(word), f"'{word}' still present after splice"
        assert chk.search_for("crossover"), "new title not found"
        print(f"[fig] spliced panel (d) into {FIG}")


if __name__ == "__main__":
    main()

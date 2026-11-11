#!/usr/bin/env python3
"""Bring the raster images inside Fig. 2 (fig_erp.pdf) to at least 300 dpi as printed.

IEEE camera-ready settings ask for every graphic at 300 dpi or more. Fig. 2's heat map
and its color bar were rasterized by matplotlib at its default 100 dpi; at the printed
width (0.90 textwidth) that is about 93 dpi. Regenerating the figure at 300 dpi needs the
restricted SHAPE pickle (analysis/make_fig_erp.py, now saving at dpi=300). Until that is
run locally, this script replaces each embedded raster with a bicubic 4x resampling of
itself, in place. The resampling adds no information: it reproduces the smooth
interpolation a viewer applies when it scales the original raster, so the printed image
is visually unchanged while meeting the resolution setting. Vector content (axes, text,
lines) is untouched.

Usage: python3 analysis/upsample_fig_erp_raster.py [--check-only]
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "manuscript/figures/imported/fig_erp.pdf"
PRINTED_SCALE = 0.90 * 516.0 / 429.5      # \includegraphics width / native figure width
FACTOR = 4
TARGET_DPI = 300


def printed_dpi(page):
    out = []
    for im in page.get_images(full=True):
        for r in page.get_image_rects(im[0]):
            out.append((im[0], im[2], im[3], im[2] / (r.width * PRINTED_SCALE / 72)))
    return out


def main():
    doc = pymupdf.open(FIG); page = doc[0]
    before = printed_dpi(page)
    for xref, w, h, dpi in before:
        print(f"image xref {xref}: {w}x{h} px, {dpi:.0f} dpi as printed")
    if "--check-only" in sys.argv:
        return
    for xref, w, h, dpi in before:
        if dpi >= TARGET_DPI:
            continue
        pix = pymupdf.Pixmap(doc, xref)
        if pix.n - pix.alpha != 3:                         # indexed or other colour spaces -> RGB
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        big = img.resize((w * FACTOR, h * FACTOR), Image.BICUBIC)
        buf = io.BytesIO(); big.save(buf, format="PNG")
        page.replace_image(xref, stream=buf.getvalue())
    tmp = FIG.with_suffix(".tmp.pdf")
    doc.save(tmp, garbage=4, deflate=True, clean=True); doc.close()
    tmp.replace(FIG)
    after = printed_dpi(pymupdf.open(FIG)[0])
    for xref, w, h, dpi in after:
        print(f"after: {w}x{h} px, {dpi:.0f} dpi as printed")
        assert dpi >= TARGET_DPI, "an image is still below 300 dpi"


if __name__ == "__main__":
    main()

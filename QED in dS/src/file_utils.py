"""Small file utilities used by the reproduction scripts."""
from __future__ import annotations

from pathlib import Path


def clean_pdf_metadata(path: str | Path) -> None:
    path = Path(path)
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        return
    reader = PdfReader(str(path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({
        "/Title": "",
        "/Author": "",
        "/Subject": "",
        "/Keywords": "",
        "/Creator": "",
        "/Producer": "",
    })
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        writer.write(f)
    tmp.replace(path)


def clean_png_metadata(path: str | Path) -> None:
    path = Path(path)
    try:
        from PIL import Image
    except Exception:
        return
    with Image.open(path) as img:
        data = list(img.getdata())
        clean = Image.new(img.mode, img.size)
        clean.putdata(data)
        clean.save(path)


def clean_figure_metadata(paths) -> None:
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.suffix.lower() == ".pdf":
            clean_pdf_metadata(p)
        elif p.suffix.lower() == ".png":
            clean_png_metadata(p)

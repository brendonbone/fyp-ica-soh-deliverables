"""
Shared helpers for the report items: reading results, formatting tables,
figure style and saving. Nothing here fits a model.

Every report item module exposes

    LABEL   = "Table V"             the report's own label
    TITLE   = "..."                 the caption title (one sentence)
    SOURCES = ["eng16_...csv", ...] result files it reads (names under results/)

    def make(stem: Path) -> list[Path]
        write the item next to `stem` (e.g. report/05_table_V_feature_soh_association)
        and return the files written: <stem>.csv + <stem>.md for a table,
        <stem>.pdf + <stem>.png + <stem>.svg for a figure.

Use `write_table` and `save_figure` below so every item looks the same.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # headless: the report script never opens a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402

FEATURES: list[str] = ["height", "position", "area", "area35", "area25",
                       "fwhm", "slope_left", "slope_right", "prominence"]
COMPONENT_LABELS = {"level": "level (raw)", "rm20": "rolling mean (rm20)", "drift": "drift",
                    "slope20": "slope, 20 cycles (slope20)", "slope10": "slope, 10 cycles (slope10)",
                    "d20": "change, 20 cycles (d20)", "d10": "change, 10 cycles (d10)"}
PARTITIONS = ["lobo_b1", "lobo_b2", "lobo_b3", "cv3_f0", "cv3_f1", "cv3_f2"]
PARTITION_NAMES = {"lobo_b1": "Batch 1 out", "lobo_b2": "Batch 2 out",
                   "lobo_b3": "Batch 3 out (study split)", "cv3_f0": "Mixed third 1",
                   "cv3_f1": "Mixed third 2", "cv3_f2": "Mixed third 3"}


# ------------------------ reading results ------------------------

def result_path(name: str) -> Path:
    p = RESULTS_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run the stage that writes it (see run_results.py)")
    return p


def load_json(name: str) -> dict:
    return json.loads(result_path(name).read_text(encoding="utf-8"))


def load_csv(name: str, **kw) -> pd.DataFrame:
    """Full-precision read: pandas' default parser can be one ulp off the
    written value, which would change a CSV written back from it."""
    kw.setdefault("float_precision", "round_trip")
    return pd.read_csv(result_path(name), **kw)


def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(result_path(name))


# ------------------------ number formatting ------------------------

MINUS = "\u2212"   # typographic minus in every formatted (markdown) number


def _isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA
    except TypeError:
        return False


def minus(s: str) -> str:
    """'-0.122' -> '−0.122'. Formatted output only; the CSVs stay numeric."""
    return s.replace("-", MINUS)


def fmt_rmse(x, nd: int = 5) -> str:
    """SoH RMSE / MAE as a fraction of nominal capacity, 5 decimals."""
    return "" if _isnan(x) else minus(f"{x:.{nd}f}")


def fmt_r(x, nd: int = 3) -> str:
    """Correlation, R2, Kendall tau."""
    return "" if _isnan(x) else minus(f"{x:.{nd}f}")


def fmt_signed(x, nd: int = 3) -> str:
    return "" if _isnan(x) else minus(f"{x:+.{nd}f}")


def fmt_pct(x, nd: int = 1, signed: bool = False) -> str:
    """x is a fraction (0.134 -> '13.4%')."""
    if _isnan(x):
        return ""
    return minus(f"{100 * x:+.{nd}f}%" if signed else f"{100 * x:.{nd}f}%")


def fmt_int(x) -> str:
    return "" if _isnan(x) else minus(f"{int(round(x)):,}")


def fmt_float(nd: int) -> Callable[[Any], str]:
    return lambda x: "" if _isnan(x) else minus(f"{x:.{nd}f}")


def fmt_pm(mean, sd, nd: int = 5) -> str:
    if _isnan(mean):
        return ""
    return minus(f"{mean:.{nd}f}" if _isnan(sd) else f"{mean:.{nd}f} \u00b1 {sd:.{nd}f}")


def join_features(blocks) -> str:
    """['fwhm', 'height'] -> 'height, fwhm' in the fixed pool order."""
    blocks = list(blocks)
    order = {f: i for i, f in enumerate(FEATURES)}
    if all(b in order for b in blocks):
        blocks = sorted(blocks, key=order.get)
    return ", ".join(blocks)


# ------------------------ markdown tables ------------------------

def md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def to_markdown(df: pd.DataFrame, formats: dict[str, Callable] | None = None,
                bold_rows: list[int] | None = None) -> str:
    """GitHub-flavoured markdown table. `formats` maps column -> formatter;
    other columns use str(). Rows in `bold_rows` (positional) are bolded."""
    formats = formats or {}
    bold = set(bold_rows or [])
    cols = list(df.columns)
    lines = ["| " + " | ".join(md_escape(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for c in cols:
            v = row[c]
            s = formats[c](v) if c in formats else ("" if _isnan(v) else str(v))
            s = md_escape(s)
            if i in bold and s:
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_table(stem: Path, df: pd.DataFrame, *, label: str, title: str,
                formats: dict[str, Callable] | None = None, csv_df: pd.DataFrame | None = None,
                bold_rows: list[int] | None = None, notes: list[str] | None = None,
                sources: list[str] | None = None) -> list[Path]:
    """<stem>.csv (full precision: `csv_df` if given, else `df`) and <stem>.md
    (the formatted table under its label and title, then any short notes and
    the source files)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    (csv_df if csv_df is not None else df).to_csv(csv_path, index=False, lineterminator="\n")
    md = [f"**{label}. {title}**", "", to_markdown(df, formats, bold_rows)]
    for n in notes or []:
        md += ["", n]
    if sources:
        md += ["", "Sources: " + ", ".join(f"`results/{s}`" for s in sources)]
    md_path = stem.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return [csv_path, md_path]


# ------------------------ figure style (IEEE two-column) ------------------------

COL_WIDTH: float = 3.5    # single IEEE column, inches
FULL_WIDTH: float = 7.16  # full IEEE page width, inches

_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
}


def apply_style() -> None:
    """Call at the start of every figure item."""
    plt.rcParams.update(_RC)


def save_figure(stem: Path, fig, *, formats=("pdf", "png", "svg"), dpi: int = 600) -> list[Path]:
    """<stem>.pdf (print), <stem>.png (600 dpi) and <stem>.svg (text kept as
    text) for Word; deterministic metadata so reruns are byte-identical."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = stem.with_suffix("." + ext)
        kw = {"bbox_inches": "tight", "dpi": dpi}
        if ext == "svg":
            with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "report"}):
                fig.savefig(p, metadata={"Date": None}, **kw)
        elif ext == "pdf":
            fig.savefig(p, metadata={"CreationDate": None, "ModDate": None}, **kw)
        else:
            fig.savefig(p, **kw)
        paths.append(p)
    plt.close(fig)
    return paths


# Figs. 8 and 9 (subset searches against k) are a matched pair: the same line
# has the same symbol, colour and legend label. Okabe-Ito colours.
C_BE, C_PEARSON, C_PI = "#D55E00", "#0072B2", "#009E73"
PATH_STYLE = {
    "subsets": dict(color="0.72", s=3, lw=0),
    "best": dict(color="k", ls="-", lw=1.0, marker="o", ms=3, label="best subset of each size"),
    "be": dict(color=C_BE, ls=(0, (4, 2)), lw=0.9, marker="s", ms=5, mfc="none", mew=0.8,
               label="backward elimination (BE)"),
    "pearson": dict(color=C_PEARSON, ls=":", lw=0.9, marker="^", ms=3.4, label="Pearson ranking (forward)"),
    "pi": dict(color=C_PI, ls="-.", lw=0.9, marker="D", ms=2.6, label="PI ranking, tuned RF (forward)"),
    "ref": dict(color="0.35", ls="-", lw=0.7),
}
Y_CAP = 1.10            # y capped 10 % above the highest drawn line
OFF_ARROW = dict(arrowstyle="-|>", color="0.45", lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0)
OFF_TEXT = " off scale (up to {:.3f})"
TEXT_PT = 6.5           # in-plot labels and legend of Figs. 8 and 9


def off_scale_labels(k: np.ndarray, y: np.ndarray, ytop: float) -> list[tuple[int, str]]:
    """(k, label) for each k with subsets above ytop, in k order: the count,
    and after the last one ' off scale (up to <largest of all>)', with
    'at k = ..' when the largest is not in the last k."""
    off = y > ytop
    ks = sorted({int(v) for v in k[off]})
    if not ks:
        return []
    i_max = int(np.flatnonzero(off)[np.argmax(y[off])])
    tail = OFF_TEXT.format(y[i_max])
    if int(k[i_max]) != ks[-1]:
        tail = tail[:-1] + f" at k = {int(k[i_max])})"
    return [(kv, f"{int((off & (k == kv)).sum())}" + (tail if kv == ks[-1] else "")) for kv in ks]

# src/components/plot_generator.py
from __future__ import annotations

import io, os, re, base64, difflib
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
try:
    from PIL import Image
except Exception:
    Image = None

# ---------- parsing helpers ----------
_FIRST_N_RE = re.compile(r"\b(first|top)\s+(\d+)\s+rows?\b", re.I)
_LAST_N_RE  = re.compile(r"\b(last|bottom)\s+(\d+)\s+rows?\b", re.I)

# NEW: top/bottom N by category (e.g., "top 5", "bottom 10")
_TOP_ANY_RE    = re.compile(r"\b(top|highest)\s+(\d+)\b", re.I)
_BOTTOM_ANY_RE = re.compile(r"\b(bottom|lowest)\s+(\d+)\b", re.I)

_PIE_RE     = re.compile(r"\b(pie|donut)\b", re.I)
_BAR_RE     = re.compile(r"\b(bar|column)\b", re.I)
_LINE_RE    = re.compile(r"\b(line|trend)\b", re.I)
_SCATTER_RE = re.compile(r"\b(scatter)\b", re.I)
_HIST_RE    = re.compile(r"\b(hist|histogram)\b", re.I)
_BOX_RE     = re.compile(r"\b(box|box[-\s]?plot)\b", re.I)

# explicit label phrases
_BY_RE       = re.compile(r"\bby\s+([a-zA-Z0-9 \-_]+)", re.I)                 # "... by region East"
_WISE_RE     = re.compile(r"\b([a-zA-Z0-9 \-_]+?)\s*(?:-|\s)?wise\b", re.I)  # "state wise"
_FOR_BY_RE   = re.compile(r"\bfor\s+([a-zA-Z0-9 \-_]+?)\s+by\s+([a-zA-Z0-9 \-_]+)\b", re.I)  # (legacy)
_PER_RE      = re.compile(r"\bper\s+([a-zA-Z0-9 \-_]+)\b", re.I)             # "sales per state"

# NEW: explicit filter forms – "where Region = East", "for Category: Technology"
_FILTER_EXPR_RE = re.compile(
    r"\b(?:where|for)\s+([a-zA-Z0-9 _\-]+?)\s*(?:=|:|is)\s*([a-zA-Z0-9 _\-]+)\b", re.I
)

# time parsing
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DATE_PREFS = ["order date","order_date","orderdate","date","ship date","ship_date","shipdate"]
_YEAR_LABEL_WORDS = ["year", "yr", "fy", "fiscal year"]

# ----- metric aliases -----
_VALUE_ALIASES = {
    "sales":   ["sales", "sale", "amount", "revenue", "total sales", "net sales"],
    "profit":  ["profit", "margin", "gain"],
    "quantity":["quantity", "qty", "units", "count"],
    "price":   ["unit price", "price", "avg price", "rate", "mrp", "unit cost", "selling price", "sp"],
}

# broadened defaults so we still have a sensible fallback order
_LABEL_PREFS = [
    "state","city","country","region","market",
    "payment mode","payment method","payment type","pay mode",
    "ship mode","shipping mode","channel","order priority",
    "segment","category","sub-category","sub category",
    "product name","product","customer name","order id","year"
]

_CURRENCY_HINT = ["sales","revenue","profit","amount","price","cost","mrp","rate"]

# export sizes
_DEF_W, _DEF_H, _SCALE = 1200, 650, 2
_DARK_COLORS = px.colors.qualitative.Dark24  # dark palette

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("_", " ").replace("-", " ")

# --- basic helpers (ABOVE any usage) ---
def _parse_subset_slice(question: str, nrows: int) -> Tuple[int, str]:
    """Supports 'first 10 rows' / 'last 20 rows'. Returns (take, which)."""
    q = _norm(question)
    m = _FIRST_N_RE.search(q)
    if m:
        return max(1, min(int(m.group(2)), nrows)), "first"
    m = _LAST_N_RE.search(q)
    if m:
        return max(1, min(int(m.group(2)), nrows)), "last"
    return nrows, "all"

# NEW: parse generic "top/bottom N"
def _parse_top_bottom(question: str) -> Tuple[Optional[int], Optional[str]]:
    q = _norm(question)
    m = _TOP_ANY_RE.search(q)
    if m:
        return int(m.group(2)), "top"
    m = _BOTTOM_ANY_RE.search(q)
    if m:
        return int(m.group(2)), "bottom"
    return None, None

def _is_count_based(question: str) -> bool:
    q = _norm(question)
    return any(w in q for w in ["count", "how many", "no. of", "number of", "volume"])

def _detect_kind(question: str) -> str:
    if _PIE_RE.search(question):     return "pie"
    if _BAR_RE.search(question):     return "bar"
    if _LINE_RE.search(question):    return "line"
    if _SCATTER_RE.search(question): return "scatter"
    if _HIST_RE.search(question):    return "hist"
    if _BOX_RE.search(question):     return "box"
    q = _norm(question)
    if "wise" in q or "share" in q or "distribution" in q: return "pie"
    return "pie"

def _match_col(cols: List[str], aliases: List[str]) -> Optional[str]:
    ncols = [_norm(c) for c in cols]
    for alias in aliases:
        al = _norm(alias)
        for i, nc in enumerate(ncols):
            if al == nc:
                return cols[i]
            if re.search(rf"(?<![a-z0-9]){re.escape(al)}(?![a-z0-9])", nc):
                return cols[i]
    return None

def _best_label_from_mentioned_cols(df: pd.DataFrame, q: str) -> Optional[str]:
    cols = list(df.columns); qn = _norm(q)
    best = None; best_score = -1
    for c in cols:
        cn = _norm(c)
        if re.search(rf"(?<![a-z0-9]){re.escape(cn)}(?![a-z0-9])", qn) or all(tok in qn for tok in cn.split()):
            score = len(cn) + (5 if not pd.api.types.is_numeric_dtype(df[c]) else 0)
            if score > best_score: best, best_score = c, score
    return best

# ---------- PRICE: ensure a real/derived price column ----------
def _ensure_price_column_inplace(df: pd.DataFrame) -> Optional[str]:
    """
    1) If a price-like column exists -> return it.
    2) Else if Sales & Quantity exist -> create 'Unit Price (calc)' = Sales / Quantity (0/NaN safe).
    3) Else return None.
    """
    cols = list(df.columns)
    pcol = _match_col(cols, _VALUE_ALIASES["price"])
    if pcol is not None and pd.api.types.is_numeric_dtype(df[pcol]): return pcol

    sales = _match_col(cols, _VALUE_ALIASES["sales"])
    qty   = _match_col(cols, _VALUE_ALIASES["quantity"])
    if sales and qty and pd.api.types.is_numeric_dtype(df[sales]) and pd.api.types.is_numeric_dtype(df[qty]):
        q = df[qty].replace(0, np.nan)
        df["Unit Price (calc)"] = df[sales] / q
        return "Unit Price (calc)"
    return None

def _pick_value_columns(df: pd.DataFrame, question: str, want_two: bool = False) -> List[str]:
    """
    If the question mentions 'price', force a price metric (real or derived),
    so "price" never silently falls back to Sales.
    """
    cols = list(df.columns); q = _norm(question); chosen: List[str] = []

    def add_if(col: Optional[str]):
        if col and pd.api.types.is_numeric_dtype(df[col]) and col not in chosen:
            chosen.append(col)

    if any(k in q for k in ["price","rate","mrp"]):
        add_if(_ensure_price_column_inplace(df))

    for key, aliases in _VALUE_ALIASES.items():
        if key in q:
            add_if(_match_col(cols, aliases))

    need = 2 if want_two else 1
    for key in ("sales","profit","quantity","price"):
        if len(chosen) >= need: break
        add_if(_match_col(cols, _VALUE_ALIASES[key]))

    if len(chosen) < need:
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]) and c not in chosen:
                chosen.append(c)
                if len(chosen) >= need: break
    return chosen[:need]

# ----- time helpers -----
def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for pref in _DATE_PREFS:
        c = _match_col(cols, [pref])
        if c is not None: return c
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]): return c
    for c in cols:
        if df[c].dtype == object:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                pass
    return None

def _ensure_year_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    cols = list(df.columns)
    ycol = _match_col(cols, ["year","fiscal year","fy","yr"])
    if ycol is not None: return df, ycol, None
    dcol = _find_date_col(df)
    if not dcol: return df, None, None
    df2 = df.copy()
    dt = pd.to_datetime(df2[dcol], errors="coerce")
    df2["__Year__"] = dt.dt.year
    return df2, "__Year__", dcol

# ----- label chooser -----
def _pick_label_column_resolved(df: pd.DataFrame, question: str) -> Tuple[pd.DataFrame, str]:
    cols = list(df.columns); q = _norm(question)

    if any(w in q for w in _YEAR_LABEL_WORDS):
        df2, ycol, _src = _ensure_year_column(df)
        if ycol is not None: return df2, ycol

    m = _FOR_BY_RE.search(q)
    if m:
        want = m.group(1).strip()
        col = _match_col(cols, [want])
        if col is not None and not pd.api.types.is_numeric_dtype(df[col]): return df, col
        if any(want == w for w in _YEAR_LABEL_WORDS):
            df2, ycol, _ = _ensure_year_column(df)
            if ycol is not None: return df2, ycol

    m = _BY_RE.search(q)
    if m:
        want = m.group(1).strip()          # could be "region east"
        lbl, _ = _split_label_value(want, cols)
        if lbl is not None: return df, lbl

    m = _WISE_RE.search(q)
    if m:
        want = m.group(1).strip()
        col = _match_col(cols, [want])
        if col is not None: return df, col
        if any(want == w for w in _YEAR_LABEL_WORDS):
            df2, ycol, _ = _ensure_year_column(df)
            if ycol is not None: return df2, ycol

    m = _PER_RE.search(q)
    if m:
        want = m.group(1).strip()
        col = _match_col(cols, [want])
        if col is not None: return df, col

    for pref in _LABEL_PREFS:
        if _norm(pref) in q:
            col = _match_col(cols, [pref])
            if col is not None: return df, col

    col = _best_label_from_mentioned_cols(df, q)
    if col is not None: return df, col

    for pref in _LABEL_PREFS:
        col = _match_col(cols, [pref])
        if col is not None: return df, col

    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]): return df, c
    return df, cols[0]

# ----- label-value filter parsing -----
def _split_label_value(want: str, cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Input like 'region east' -> ('Region','East') by matching the longest prefix to a column."""
    toks = [t for t in want.strip().split() if t]
    if not toks: return None, None
    for k in range(len(toks), 0, -1):
        cand = " ".join(toks[:k])
        col = _match_col(cols, [cand])
        if col is not None:
            value = " ".join(toks[k:]).strip() or None
            return col, value
    for k in range(1, len(toks)):
        cand = " ".join(toks[k:])
        col = _match_col(cols, [cand])
        if col is not None:
            value = " ".join(toks[:k]).strip() or None
            return col, value
    return None, None

def _resolve_category_value(values: pd.Series, target_raw: str) -> Optional[str]:
    """Map a noisy value string to the closest category (case/space tolerant)."""
    if target_raw is None: return None
    target = _norm(str(target_raw))
    uniques = pd.Series(values.astype(str).unique())
    norms = uniques.map(_norm)
    if target in set(norms): return uniques[norms == target].iloc[0]
    mask = norms.str.startswith(target) | norms.str.contains(re.escape(target))
    if mask.any(): return uniques[mask].iloc[0]
    match = difflib.get_close_matches(target, norms.tolist(), n=1, cutoff=0.72)
    if match:
        return uniques[norms == match[0]].iloc[0]
    return None

def _apply_label_value_filter(df: pd.DataFrame, question: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """
    If question says 'by <label> <value>' e.g., 'by region East',
    or 'where/for <label> = <value>', filter df to that label=value. Returns (df_filtered, note, label_used).
    """
    q = _norm(question)
    # by <label/value> form
    m = _BY_RE.search(q)
    if m:
        want = m.group(1).strip()
        lbl, raw_val = _split_label_value(want, list(df.columns))
        if lbl is not None and raw_val:
            resolved = _resolve_category_value(df[lbl], raw_val)
            if resolved is not None:
                mask = _norm_series(df[lbl]) == _norm(resolved)
                if mask.any():
                    return df.loc[mask].copy(), f"{lbl}: {resolved}", lbl
                return df, f"{lbl}: {resolved} (no rows matched)", lbl

    # where/for label = value
    m2 = _FILTER_EXPR_RE.search(q)
    if m2:
        want_label = m2.group(1).strip()
        want_value = m2.group(2).strip()
        col = _match_col(list(df.columns), [want_label])
        if col is not None and want_value:
            resolved = _resolve_category_value(df[col], want_value)
            if resolved is not None:
                mask = _norm_series(df[col]) == _norm(resolved)
                if mask.any():
                    return df.loc[mask].copy(), f"{col}: {resolved}", col
                return df, f"{col}: {resolved} (no rows matched)", col

    return df, "", None

def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace("_"," ", regex=False).str.replace("-"," ", regex=False).str.strip()

# ----- time filter -----
def _parse_year_from_question(q: str) -> Optional[int]:
    m = _YEAR_RE.search(q)
    return int(m.group(0)) if m else None

def _apply_time_filter(df: pd.DataFrame, question: str) -> Tuple[pd.DataFrame, str]:
    year = _parse_year_from_question(_norm(question))
    if year is None: return df, ""
    date_col = _find_date_col(df)
    if not date_col: return df, f"Year: {year} (date column not found)"
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df = df.copy(); df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            return df, f"Year: {year} (invalid dates)"
    mask = df[date_col].dt.year == year
    if not mask.any(): return df, f"Year: {year} (no rows matched)"
    return df.loc[mask].copy(), f"Year: {year}"

# ----- stats & currency -----
def _stat_from_question(q: str) -> Optional[str]:
    q = _norm(q)
    if any(w in q for w in ["median", "madhyik"]): return "median"
    if any(w in q for w in ["average", "avg", "mean", "ausat"]): return "mean"
    if any(w in q for w in ["maximum", "max", "highest", "adhiktam", "sabse zyada"]): return "max"
    if any(w in q for w in ["minimum", "min", "lowest", "nyuntam", "sabse kam"]): return "min"
    if any(w in q for w in ["sum", "total", "overall", "yog", "kul", "jod"]): return "sum"
    if any(w in q for w in ["count", "how many", "number of"]): return "count"
    return None

def _default_agg_for_metric(metric_name: str) -> str:
    n = _norm(metric_name)
    if "price" in n or "discount" in n or "rate" in n: return "mean"
    return "sum"

def _currency_symbol_for(df: pd.DataFrame) -> str:
    joined = " ".join(map(str, df.columns))
    for sym in ("₹","€","£","$"):
        if sym in joined: return sym
    cols = [_norm(c) for c in df.columns]
    return "$" if any(any(h in c for h in _CURRENCY_HINT) for c in cols) else ""

def _dataset_context_line(df: pd.DataFrame, cur: str, which: str, take: int, extra_note: str = "") -> str:
    want = [
        ("Region",   ["region","state","city","country","market"]),
        ("Product",  ["product name","product"]),
        ("Category", ["category","sub-category","sub category","segment"]),
        ("Price",    ["price"]),
        ("Sales",    ["sales","amount","revenue"]),
        ("Profit",   ["profit","margin"]),
        ("Quantity", ["quantity","qty","units"]),
        ("Discount", ["discount"]),
    ]
    found: List[str] = []
    cols = list(df.columns)
    for label, aliases in want:
        col = _match_col(cols, aliases)
        if col:
            if label in ("Sales","Profit","Price"): found.append(f"{label} ({cur or '$'})")
            elif label == "Discount":               found.append("Discount (%)")
            else:                                   found.append(label)
    subset = "" if which == "all" else f"Subset: {which} {take} rows"
    head = "Fields: " + ", ".join(dict.fromkeys(found)) if found else ""
    tail = extra_note
    parts = [x for x in [head, subset, tail] if x]
    return " • ".join(parts)

def _summary_by_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    cols = list(df.columns); agg: Dict[str, str] = {}
    sales  = _match_col(cols, _VALUE_ALIASES["sales"])
    profit = _match_col(cols, _VALUE_ALIASES["profit"])
    qty    = _match_col(cols, _VALUE_ALIASES["quantity"])
    price  = _match_col(cols, _VALUE_ALIASES["price"])

    # Derive price if missing but Sales & Quantity exist
    if (price is None) and sales and qty \
       and pd.api.types.is_numeric_dtype(df[sales]) and pd.api.types.is_numeric_dtype(df[qty]):
        q = df[qty].replace(0, np.nan)
        df = df.copy()
        df["Unit Price (calc)"] = df[sales] / q
        price = "Unit Price (calc)"

    if sales:  agg[sales]  = "sum"
    if profit: agg[profit] = "sum"
    if qty:    agg[qty]    = "sum"
    if price:  agg[price]  = "mean"

    g = df.groupby(label_col, dropna=False).agg(agg)
    g["Count"] = df.groupby(label_col, dropna=False).size()

    rename_map = {}
    if sales:  rename_map[sales]  = "Sales"
    if profit: rename_map[profit] = "Profit"
    if qty:    rename_map[qty]    = "Quantity"
    if price:  rename_map[price]  = "Avg Price"
    g = g.rename(columns=rename_map)
    return g.reset_index()

def _fmt_currency_list(values: pd.Series, cur: str) -> List[str]:
    return [("—" if pd.isna(v) else f"{cur}{v:,.2f}") for v in values]

def _fmt_int_list(values: pd.Series) -> List[str]:
    return [("—" if pd.isna(v) else f"{int(round(v)):,}") for v in values]

def _export_png_base64(fig: go.Figure) -> str:
    try:
        img_bytes = pio.to_image(fig, format="png", width=_DEF_W, height=_DEF_H, scale=_SCALE, engine="kaleido")
    except Exception as e:
        raise RuntimeError("Plotly static export failed. Install Kaleido: pip install -U kaleido") from e
    return base64.b64encode(img_bytes).decode("utf-8")

def _add_subtitle(fig: go.Figure, subtitle: str):
    if not subtitle: return
    fig.add_annotation(text=subtitle, x=0.5, xref="paper", y=1.08, yref="paper",
                       showarrow=False, align="center", font=dict(size=12, color="#889"))

# ---------- main ----------
class PlotGenerator:
    """
    You can pass a single DataFrame OR a list of DataFrames (combined mode).
    In combined mode we vertically append rows (outer columns union).
    """
    def __init__(self, df_or_dfs: Union[pd.DataFrame, List[pd.DataFrame]]):
        if isinstance(df_or_dfs, list):
            # combine by row; align different schemas by outer join
            aligned = []
            all_cols = set()
            for d in df_or_dfs:
                all_cols.update(d.columns.tolist())
            all_cols = list(all_cols)
            for d in df_or_dfs:
                aligned.append(d.reindex(columns=all_cols))
            self.df = pd.concat(aligned, ignore_index=True)
        else:
            self.df = df_or_dfs.copy()

        _ensure_price_column_inplace(self.df)  # prepare price column upfront if possible

    def generate_plot_and_info(self, question: str) -> Tuple[str, Dict]:
        q = question or ""
        df_all = self.df
        take, which = _parse_subset_slice(q, len(df_all))
        df = df_all.head(take) if which == "first" else df_all.tail(take) if which == "last" else df_all

        # time filter
        df, time_note = _apply_time_filter(df, q)

        kind = _detect_kind(q); cur = _currency_symbol_for(df)
        if kind == "pie":     return self._build_pie_or_donut(df, q, which, take, cur, time_note)
        if kind == "bar":     return self._build_bar(df, q, which, take, cur, time_note)
        if kind == "line":    return self._build_line(df, q, which, take, cur, time_note)
        if kind == "scatter": return self._build_scatter(df, q, which, take, cur, time_note)
        if kind == "hist":    return self._build_hist(df, q, which, take, cur, time_note)
        if kind == "box":     return self._build_box(df, q, which, take, cur, time_note)
        return self._build_pie_or_donut(df, q, which, take, cur, time_note)

    def generate_plot_to_file(self, question: str, out_dir: Optional[str] = None, plot_id: Optional[str] = None, **_kwargs) -> Tuple[str, str, Dict]:
        image_b64, info = self.generate_plot_and_info(question)
        out_dir = out_dir or "."; os.makedirs(out_dir, exist_ok=True)
        pid = plot_id or (info.get("title") or info.get("kind") or "plot")
        pid = re.sub(r"[^a-zA-Z0-9]+", "_", pid).strip("_").lower()
        png_path   = os.path.join(out_dir, f"{pid}.png")
        thumb_path = os.path.join(out_dir, f"{pid}_thumb.png")
        html_path  = os.path.join(out_dir, f"{pid}.html")
        raw = base64.b64decode(image_b64.encode("utf-8"))
        with open(png_path, "wb") as f: f.write(raw)
        if Image is not None:
            try:
                img = Image.open(io.BytesIO(raw)); img.thumbnail((520, 520)); img.save(thumb_path, format="PNG")
            except Exception:
                with open(thumb_path, "wb") as f2: f2.write(raw)
        else:
            with open(thumb_path, "wb") as f2: f2.write(raw)
        if "fig_html" in info:
            with open(html_path, "w", encoding="utf-8") as fh: fh.write(info["fig_html"])
            info["html_path"] = html_path
        return png_path, thumb_path, info

    # ---------- Quick numeric answer (for cards) ----------
    def quick_stat(self, question: str) -> Tuple[str, float, Dict]:
        """Return a single numeric answer honoring time + 'by <label> <value>' filters."""
        q = question or ""
        df = self.df.copy()

        # subset (first/last N rows)
        take, which = _parse_subset_slice(q, len(df))
        df = df.head(take) if which == "first" else df.tail(take) if which == "last" else df

        # filters
        df, time_note = _apply_time_filter(df, q)
        df, filter_note, _ = _apply_label_value_filter(df, q)

        stat = _stat_from_question(q) or "sum"

        # choose metric; derive Unit Price if user asked for price
        metrics = _pick_value_columns(df, q, want_two=False)
        metric = metrics[0] if metrics else None

        # pure count
        if stat == "count" or (metric is None and any(w in _norm(q) for w in ["count", "how many", "number of"])):  # noqa: E501
            value = int(df.shape[0])
            title = "Count" + (f" — {filter_note}" if filter_note else "") + (f" — {time_note}" if time_note else "")
            return title.strip(" —"), float(value), {
                "metric": "count", "stat": "count",
                "notes": "row count", "filters": {"time": time_note, "label": filter_note},
                "value_display": f"{value:,}"
            }

        # fallback to any numeric col
        if metric is None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not num_cols:
                raise ValueError("No numeric metric available for this question.")
            metric = num_cols[0]

        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            raise ValueError("No numeric values available after filtering.")

        if stat == "mean":
            value = float(series.mean())
        elif stat == "median":
            value = float(series.median())
        elif stat == "max":
            value = float(series.max())
        elif stat == "min":
            value = float(series.min())
        elif stat == "sum":
            value = float(series.sum())
        else:
            value = float(series.sum())

        cur = _currency_symbol_for(df)
        is_money = any(h in _norm(metric) for h in _CURRENCY_HINT)
        val_disp = f"{cur}{value:,.2f}" if is_money else f"{value:,.2f}"
        title = f"{stat.title()} of {metric}" + (f" — {filter_note}" if filter_note else "") + (f" — {time_note}" if time_note else "")
        return title.strip(" —"), value, {
            "metric": metric, "stat": stat, "value_display": val_disp,
            "filters": {"time": time_note, "label": filter_note}
        }

    # ---------- PIE / DONUT ----------
    def _build_pie_or_donut(self, df: pd.DataFrame, question: str, which: str, take: int, cur: str, time_note: str):
        # label + optional filter from "by <label> <value>" or where/for
        df, label_col = _pick_label_column_resolved(df, question)
        df, filter_note, filter_label = _apply_label_value_filter(df, question)
        if filter_label:
            label_col = filter_label
        subtitle = _dataset_context_line(df, cur, which, take, " • ".join([x for x in [time_note, filter_note] if x]))
        count_mode = _is_count_based(question)

        # top/bottom N support
        n_take, tb_flag = _parse_top_bottom(question)

        table_df = _summary_by_label(df, label_col)
        sort_key = "Sales" if "Sales" in table_df.columns else "Count"
        table_df = table_df.sort_values(by=sort_key, ascending=False)

        if n_take:
            if tb_flag == "top":
                table_df = table_df.head(n_take)
            elif tb_flag == "bottom":
                table_df = table_df.tail(n_take)

        labels = table_df[label_col].astype(str).tolist()
        colors = (_DARK_COLORS * ((len(labels)//len(_DARK_COLORS))+1))[:len(labels)]
        color_map = {lab: col for lab, col in zip(labels, colors)}

        def _table_trace(tdf: pd.DataFrame) -> go.Table:
            return go.Table(
                columnwidth=[40,28,28,28,28,20],
                header=dict(values=[label_col, "Sales", "Profit", "Quantity", "Avg Price", "Count"],
                            fill_color="#1f2937", font=dict(color="white", size=12), align="left"),
                cells=dict(values=[
                        tdf[label_col].astype(str).tolist(),
                        _fmt_currency_list(tdf.get("Sales", pd.Series([np.nan]*len(tdf))), cur or "$"),
                        _fmt_currency_list(tdf.get("Profit", pd.Series([np.nan]*len(tdf))), cur or "$"),
                        _fmt_int_list(tdf.get("Quantity", pd.Series([np.nan]*len(tdf)))),
                        _fmt_currency_list(tdf.get("Avg Price", pd.Series([np.nan]*len(tdf))), cur or "$"),
                        _fmt_int_list(tdf["Count"]),
                    ],
                    height=24, align="left")
            )

        # Count pie
        if count_mode or not _pick_value_columns(df, question):
            values = table_df["Count"].values
            custom = np.column_stack([
                table_df.get("Sales", pd.Series([np.nan]*len(table_df))),
                table_df.get("Profit", pd.Series([np.nan]*len(table_df))),
                table_df["Count"].values,
                table_df.get("Avg Price", pd.Series([np.nan]*len(table_df))),
                table_df.get("Quantity", pd.Series([np.nan]*len(table_df))),
            ])
            hover = (
                "<b>%{label}</b><br>"
                "Share: %{percent:.1%}<br>"
                "Count: %{customdata[2]:,}"
                "<br>Sales: " + (cur or "$") + "%{customdata[0]:,.2f}"
                "<br>Profit: " + (cur or "$") + "%{customdata[1]:,.2f}"
                "<br>Avg Price: " + (cur or "$") + "%{customdata[3]:,.2f}"
                "<br>Quantity: %{customdata[4]:,}<extra></extra>"
            )

            fig = make_subplots(rows=1, cols=2,
                                specs=[[{"type":"domain"}, {"type":"table"}]],
                                column_widths=[0.66, 0.34], horizontal_spacing=0.03)
            fig.add_trace(go.Pie(labels=labels, values=values, hole=0.4,
                                 textinfo="percent",
                                 marker=dict(colors=[color_map[l] for l in labels]),
                                 hovertemplate=hover, customdata=custom,
                                 sort=False, direction="clockwise", name=f"Count by {label_col}"), 1, 1)
            fig.add_trace(_table_trace(table_df), 1, 2)
            title = f"Count by {label_col}" + (f" — {which} {take} rows" if which!="all" else "")
            fig.update_layout(title=title, width=_DEF_W, height=_DEF_H, legend_title=label_col,
                              margin=dict(t=90,l=60,r=40,b=60))
            _add_subtitle(fig, subtitle)
            img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
            info = {"kind":"pie","title":title,"x":label_col,"y":"count","group":None,
                    "subset":{"which":which,"n":take},"agg":"count","currency_symbol":cur,"notes":"count-based",
                    "fig_html": html}
            return img, info

        # SINGLE-METRIC pie
        metric_candidates = _pick_value_columns(df, question, want_two=True)
        metric = next((m for m in metric_candidates if any(k in _norm(m) for k in ["sale","revenue"])), None) or metric_candidates[0]

        sums = df.groupby(label_col, dropna=False)[metric].sum().sort_values(ascending=False)
        order = sums.index.astype(str).tolist()
        if n_take:
            order = (order[:n_take] if tb_flag == "top" else order[-n_take:])
            sums = sums.loc[[s for s in sums.index if str(s) in set(order)]]

        tdf = table_df.set_index(label_col).reindex(order).reset_index()

        custom = np.column_stack([
            tdf.get("Sales", pd.Series([np.nan]*len(tdf))),
            tdf.get("Profit", pd.Series([np.nan]*len(tdf))),
            tdf["Count"].values,
            tdf.get("Avg Price", pd.Series([np.nan]*len(tdf))),
            tdf.get("Quantity", pd.Series([np.nan]*len(tdf))),
        ])
        hover = (
            "<b>%{label}</b><br>"
            "Share: %{percent:.1%}<br>"
            "Count: %{customdata[2]:,}"
            "<br>Sales: " + (cur or "$") + "%{customdata[0]:,.2f}"
            "<br>Profit: " + (cur or "$") + "%{customdata[1]:,.2f}"
            "<br>Avg Price: " + (cur or "$") + "%{customdata[3]:,.2f}"
            "<br>Quantity: %{customdata[4]:,}<extra></extra>"
        )

        fig = make_subplots(rows=1, cols=2,
                            specs=[[{"type":"domain"}, {"type":"table"}]],
                            column_widths=[0.66, 0.34], horizontal_spacing=0.03)
        fig.add_trace(go.Pie(labels=order, values=sums.values, hole=0.40, textinfo="percent",
                             marker=dict(colors=[color_map[l] for l in order]),
                             hovertemplate=hover, customdata=custom, name=metric, sort=False), 1, 1)
        fig.add_trace(_table_trace(tdf), 1, 2)
        title = f"{metric} by {label_col}" + (f" — {which} {take} rows" if which!="all" else "")
        fig.update_layout(title=title, width=_DEF_W, height=_DEF_H, legend_title=label_col, margin=dict(t=90,l=60,r=40,b=60))
        _add_subtitle(fig, subtitle)
        img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
        info = {"kind":"pie","title":title,"x":label_col,"y":metric,"group":None,
                "subset":{"which":which,"n":take},"agg":"sum","currency_symbol":cur or "$","notes":"",
                "fig_html": html}
        return img, info

    # ---------- BAR ----------
    def _build_bar(self, df, q, which, take, cur, time_note):
        df, label_col = _pick_label_column_resolved(df, q)
        df, filter_note, filter_label = _apply_label_value_filter(df, q)
        if filter_label:
            label_col = filter_label
        subtitle = _dataset_context_line(df, cur, which, take, " • ".join([x for x in [time_note, filter_note] if x]))

        n_take, tb_flag = _parse_top_bottom(q)  # top/bottom N
        metrics = _pick_value_columns(df, q, want_two=True)
        forced_stat = _stat_from_question(q)
        table_df_raw = _summary_by_label(df, label_col)
        table_df = table_df_raw.sort_values(by=("Sales" if "Sales" in table_df_raw.columns else "Count"), ascending=False)

        has_price = any("price" in _norm(m) for m in metrics)
        has_sales = any(any(a in _norm(m) for a in ["sales","revenue"]) for m in metrics)

        if metrics and len(metrics) == 2 and has_price and has_sales:
            price_col = [m for m in metrics if "price" in _norm(m)][0]
            sales_col = [m for m in metrics if any(a in _norm(m) for a in ["sales","revenue"])][0]

            g_sum = df.groupby(label_col, dropna=False)[sales_col].sum()
            g_avg = df.groupby(label_col, dropna=False)[price_col].mean()
            g = pd.DataFrame({label_col: g_sum.index, sales_col: g_sum.values, price_col: g_avg.reindex(g_sum.index).values})
            g = g.sort_values(by=sales_col, ascending=False).reset_index(drop=True)

            if n_take:
                g = g.head(n_take) if tb_flag == "top" else g.tail(n_take)

            fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {"type":"table"}]],
                                column_widths=[0.66,0.34], horizontal_spacing=0.03)

            fig.add_trace(go.Bar(
                x=g[label_col], y=g[sales_col], name=sales_col, marker_color=_DARK_COLORS[2],
                hovertemplate=f"<b>%{{x}}</b><br>{sales_col}: {(cur or '$')}%{{y:,.2f}}<extra></extra>"),
                row=1, col=1, secondary_y=False)

            fig.add_trace(go.Bar(
                x=g[label_col], y=g[price_col], name=f"Avg {price_col}", marker_color=_DARK_COLORS[0],
                hovertemplate=f"<b>%{{x}}</b><br>Avg {price_col}: {(cur or '$')}%{{y:,.2f}}<extra></extra>"),
                row=1, col=1, secondary_y=True)

            fig.add_trace(go.Table(
                columnwidth=[40,28,28,28,28,20],
                header=dict(values=[label_col, "Sales", "Profit", "Quantity", "Avg Price", "Count"],
                            fill_color="#1f2937", font=dict(color="white", size=12), align="left"),
                cells=dict(values=[
                    table_df[label_col].astype(str).tolist(),
                    _fmt_currency_list(table_df.get("Sales", pd.Series([np.nan]*len(table_df))), cur or "$"),
                    _fmt_currency_list(table_df.get("Profit", pd.Series([np.nan]*len(table_df))), cur or "$"),
                    _fmt_int_list(table_df.get("Quantity", pd.Series([np.nan]*len(table_df)))),
                    _fmt_currency_list(table_df.get("Avg Price", pd.Series([np.nan]*len(table_df))), cur or "$"),
                    _fmt_int_list(table_df["Count"]),
                ], height=24, align="left")
            ), 1, 2)

            title = f"Revenue & Avg Price by {label_col}" + (f" — {which} {take} rows" if which!='all' else "")
            fig.update_layout(barmode="group", title=title, width=_DEF_W, height=_DEF_H, xaxis_title=label_col, legend_title_text="")
            fig.update_yaxes(title_text="Revenue", secondary_y=False, tickprefix=(cur or "$"))
            fig.update_yaxes(title_text="Avg Price", secondary_y=True, tickprefix=(cur or "$"))
            _add_subtitle(fig, subtitle)
            img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
            info = {"kind":"bar","title":title,"x":label_col,"y":f"{sales_col},{price_col}","group":None,
                    "subset":{"which":which,"n":take},"agg":"sum/mean","currency_symbol":cur or "$","notes":"dual-axis",
                    "fig_html": html}
            return img, info

        # general case
        fig = make_subplots(rows=1, cols=2, specs=[[{"type":"xy"},{"type":"table"}]],
                            column_widths=[0.66,0.34], horizontal_spacing=0.03)
        if not metrics:
            grouped = table_df
            if n_take:
                grouped = grouped.head(n_take) if tb_flag == "top" else grouped.tail(n_take)
            fig.add_trace(go.Bar(x=grouped[label_col], y=grouped["Count"], name="Count", marker_color=_DARK_COLORS[0]), 1, 1)
            title = f"Count by {label_col}" + (f" — {which} {take} rows" if which!="all" else "")
        else:
            agg_map: Dict[str, str] = {m: (forced_stat or _default_agg_for_metric(m)) for m in metrics}
            grouped = df.groupby(label_col, dropna=False).agg(agg_map).reset_index().sort_values(by=metrics[0], ascending=False)
            if n_take:
                grouped = grouped.head(n_take) if tb_flag == "top" else grouped.tail(n_take)
            for i, m in enumerate(metrics):
                disp = m if agg_map[m]=="sum" else f"{agg_map[m].title()} {m}"
                fig.add_trace(go.Bar(
                    x=grouped[label_col], y=grouped[m], name=disp,
                    marker_color=_DARK_COLORS[i % len(_DARK_COLORS)],
                    hovertemplate=f"<b>%{{x}}</b><br>{disp}: %{{y:,.2f}}<extra></extra>"
                ), 1, 1)
            title = " & ".join([(m if agg_map[m]=='sum' else f"{agg_map[m].title()} {m}") for m in metrics]) \
                    + f" by {label_col}" + (f" — {which} {take} rows" if which!='all' else "")

        fig.add_trace(go.Table(
            columnwidth=[40,28,28,28,28,20],
            header=dict(values=[label_col, "Sales", "Profit", "Quantity", "Avg Price", "Count"],
                        fill_color="#1f2937", font=dict(color="white", size=12), align="left"),
            cells=dict(values=[
                table_df[label_col].astype(str).tolist(),
                _fmt_currency_list(table_df.get("Sales", pd.Series([np.nan]*len(table_df))), cur or "$"),
                _fmt_currency_list(table_df.get("Profit", pd.Series([np.nan]*len(table_df))), cur or "$"),
                _fmt_int_list(table_df.get("Quantity", pd.Series([np.nan]*len(table_df)))),
                _fmt_currency_list(table_df.get("Avg Price", pd.Series([np.nan]*len(table_df))), cur or "$"),
                _fmt_int_list(table_df["Count"]),
            ], height=24, align="left")
        ), 1, 2)

        if metrics and any(_norm(m) in _CURRENCY_HINT for m in metrics):
            fig.update_layout(yaxis_tickprefix=(cur or "$"))
        fig.update_layout(title=title, width=_DEF_W, height=_DEF_H, xaxis_title=label_col, yaxis_title="Value", xaxis_tickangle=-45)
        _add_subtitle(fig, subtitle)
        img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
        info = {"kind":"bar","title":title,"x":label_col,"y":",".join(metrics) if metrics else "count","group":None,
                "subset":{"which":which,"n":take},"agg":",".join([(forced_stat or _default_agg_for_metric(m)) for m in metrics]) if metrics else "count",
                "currency_symbol":cur or "$","notes":"","fig_html": html}
        return img, info

    # ---------- LINE ----------
    def _build_line(self, df, q, which, take, cur, time_note):
        # apply label-value filter even if x-axis is time
        df, filter_note, filter_label = _apply_label_value_filter(df, q)

        df, label_col = _pick_label_column_resolved(df, q)
        if filter_label:
            label_col = filter_label
        metrics = _pick_value_columns(df, q, want_two=False)
        subtitle = _dataset_context_line(df, cur, which, take, " • ".join([x for x in [time_note, filter_note] if x]))

        metric = metrics[0] if metrics else None
        x = df[label_col]
        try:
            x_dt = pd.to_datetime(x, errors="raise")
            if metric:
                group = df.groupby(x_dt.dt.to_period("D"))[metric].sum()
                series = group.reset_index(); time_col = series.columns[0]; series[time_col] = series[time_col].dt.to_timestamp()
                y_col = metric; title = f"{metric} over {label_col}" + (f" — {which} {take} rows" if which!='all' else "")
            else:
                group = df.groupby(x_dt.dt.to_period("D")).size()
                series = group.reset_index(name="Count"); time_col = series.columns[0]; series[time_col] = series[time_col].dt.to_timestamp()
                y_col = "Count"; title = f"Count over {label_col}" + (f" — {which} {take} rows" if which!='all' else "")
            fig = px.line(series, x=time_col, y=y_col, title=title, width=_DEF_W, height=_DEF_H)
        except Exception:
            if metric:
                grouped = df.groupby(label_col, dropna=False)[metric].sum().reset_index()
                fig = px.line(grouped, x=label_col, y=metric, title=f"{metric} over {label_col}" + (f" — {which} {take} rows" if which!='all' else ""), width=_DEF_W, height=_DEF_H)
            else:
                grouped = df.groupby(label_col, dropna=False).size().reset_index(name="Count")
                fig = px.line(grouped, x=label_col, y="Count", title=f"Count over {label_col}" + (f" — {which} {take} rows" if which!='all' else ""), width=_DEF_W, height=_DEF_H)

        if metric and _norm(metric) in _CURRENCY_HINT: fig.update_layout(yaxis_tickprefix=(cur or "$"))
        _add_subtitle(fig, subtitle)
        img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
        info = {"kind":"line","title":fig.layout.title.text,"x":label_col,"y":metric or "count","group":None,
                "subset":{"which":which,"n":take},"agg":"sum" if metric else "count","currency_symbol":cur or "$","notes":"","fig_html": html}
        return img, info

    # ---------- SCATTER ----------
    def _build_scatter(self, df, q, which, take, cur, time_note):
        df, filter_note, _filter_label = _apply_label_value_filter(df, q)
        metrics = _pick_value_columns(df, q, want_two=True)
        if len(metrics) < 2:
            m1 = _pick_value_columns(df, "sales", want_two=False) or metrics
            m2 = _pick_value_columns(df, "profit", want_two=False) or metrics
            metrics = list(dict.fromkeys((metrics + m1 + m2)))[:2]
        if len(metrics) < 2:
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]; metrics = (metrics + nums)[:2]
        if len(metrics) < 2:
            df2, label_col = _pick_label_column_resolved(df, q)
            table_df = _summary_by_label(df2, label_col)
            fig = px.bar(table_df, x=label_col, y="Count",
                         title=f"Count by {label_col}" + (f" — {which} {take} rows" if which!='all' else ""),
                         width=_DEF_W, height=_DEF_H, color_discrete_sequence=_DARK_COLORS)
            _add_subtitle(fig, _dataset_context_line(df2, cur, which, take, time_note if not filter_note else f"{time_note} • {filter_note}".strip(" •")))
            img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
            info = {"kind":"bar","title":fig.layout.title.text,"x":label_col,"y":"count","group":None,
                    "subset":{"which":which,"n":take},"agg":"count","currency_symbol":cur or "$","notes":"scatter->bar fallback",
                    "fig_html": html}
            return img, info

        x, y = metrics[0], metrics[1]
        subtitle = _dataset_context_line(df, cur, which, take, " • ".join([x for x in [time_note, filter_note] if x]))
        fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}" + (f" — {which} {take} rows" if which!='all' else ""),
                         width=_DEF_W, height=_DEF_H, opacity=0.9, color_discrete_sequence=_DARK_COLORS)
        if _norm(x) in _CURRENCY_HINT: fig.update_layout(xaxis_tickprefix=(cur or "$"))
        if _norm(y) in _CURRENCY_HINT: fig.update_layout(yaxis_tickprefix=(cur or "$"))
        _add_subtitle(fig, subtitle)
        img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
        info = {"kind":"scatter","title":fig.layout.title.text,"x":x,"y":y,"group":None,
                "subset":{"which":which,"n":take},"agg":"none","currency_symbol":cur or "$","notes":"","fig_html": html}
        return img, info

    # ---------- HIST ----------
    def _build_hist(self, df, q, which, take, cur, time_note):
        df, filter_note, _filter_label = _apply_label_value_filter(df, q)
        metrics = _pick_value_columns(df, q, want_two=False)
        subtitle = _dataset_context_line(df, cur, which, take, " • ".join([x for x in [time_note, filter_note] if x]))
        if metrics:
            metric = metrics[0]
            fig = px.histogram(df, x=metric, nbins=20, title=f"Distribution of {metric}" + (f" — {which} {take} rows" if which!='all' else ""),
                               width=_DEF_W, height=_DEF_H, color_discrete_sequence=_DARK_COLORS)
            if _norm(metric) in _CURRENCY_HINT: fig.update_layout(xaxis_tickprefix=(cur or "$"))
        else:
            df2, label_col = _pick_label_column_resolved(df, q)
            fig = px.histogram(df2, x=label_col, title=f"Frequency of {label_col}" + (f" — {which} {take} rows" if which!='all' else ""),
                               width=_DEF_W, height=_DEF_H, color_discrete_sequence=_DARK_COLORS)
        _add_subtitle(fig, subtitle)
        img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
        info = {"kind":"hist","title":fig.layout.title.text,"x":metrics[0] if metrics else "label","y":"frequency","group":None,
                "subset":{"which":which,"n":take},"agg":"none","currency_symbol":cur or "$","notes":"bins=20" if metrics else "categorical frequency",
                "fig_html": html}
        return img, info

    # ---------- BOX ----------
    def _build_box(self, df, q, which, take, cur, time_note):
        df, label_col = _pick_label_column_resolved(df, q)
        df, filter_note, filter_label = _apply_label_value_filter(df, q)
        if filter_label:
            label_col = filter_label
        subtitle = _dataset_context_line(df, cur, which, take, " • ".join([x for x in [time_note, filter_note] if x]))

        metrics = _pick_value_columns(df, q, want_two=False)
        if not metrics:
            table_df = _summary_by_label(df, label_col)
            fig = px.bar(table_df, x=label_col, y="Count",
                         title=f"Count by {label_col}" + (f" — {which} {take} rows" if which!='all' else ""),
                         width=_DEF_W, height=_DEF_H, color_discrete_sequence=_DARK_COLORS)
            _add_subtitle(fig, subtitle)
            img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
            info = {"kind":"bar","title":fig.layout.title.text,"x":label_col,"y":"count","group":None,
                    "subset":{"which":which,"n":take},"agg":"count","currency_symbol":cur or "$","notes":"box->bar fallback",
                    "fig_html": html}
            return img, info

        metric = metrics[0]
        groupable = not pd.api.types.is_numeric_dtype(df[label_col])
        if groupable:
            fig = px.box(df, x=label_col, y=metric, points="outliers",
                         title=f"{metric} by {label_col} (box plot)" + (f" — {which} {take} rows" if which!='all' else ""),
                         width=_DEF_W, height=_DEF_H, color_discrete_sequence=_DARK_COLORS)
        else:
            fig = px.box(df, y=metric, points="outliers",
                         title=f"{metric} distribution (box plot)" + (f" — {which} {take} rows" if which!='all' else ""),
                         width=_DEF_W, height=_DEF_H, color_discrete_sequence=_DARK_COLORS)
        if _norm(metric) in _CURRENCY_HINT: fig.update_layout(yaxis_tickprefix=(cur or "$"))
        _add_subtitle(fig, subtitle)
        img = _export_png_base64(fig); html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
        info = {"kind":"box","title":fig.layout.title.text,"x":label_col if groupable else None,"y":metric,"group":label_col if groupable else None,
                "subset":{"which":which,"n":take},"agg":"none","currency_symbol":cur or "$","notes":"","fig_html": html}
        return img, info

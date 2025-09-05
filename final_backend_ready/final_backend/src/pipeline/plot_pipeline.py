# src/pipeline/plot_pipeline.py
from __future__ import annotations

import uuid, json, re, difflib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ------------------------ Storage ------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT.parent / "static"
VIS_DIR = STATIC_ROOT / "visualizations"
META_INDEX = VIS_DIR / "index.jsonl"
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------ Parsing helpers ------------------------
_PLOT_PATS = [
    ("bar",     re.compile(r"\b(bar|bar\s*chart)\b", re.I)),
    ("line",    re.compile(r"\b(line|trend)\b", re.I)),
    ("pie",     re.compile(r"\b(pie)\b", re.I)),
    ("hist",    re.compile(r"\b(hist|histogram)\b", re.I)),
    ("scatter", re.compile(r"\b(scatter|bubble)\b", re.I)),
    ("box",     re.compile(r"\b(box|box[-\s]*plot)\b", re.I)),
]

# measures & dimensions
_VALUE_ALIASES = {
    "sales":    ["sales", "sale", "amount", "revenue", "sales amount", "net sales"],
    "profit":   ["profit", "margin", "gross profit", "net profit", "profit amount"],
    "quantity": ["quantity", "qty", "units", "order quantity", "units sold"],
    "price":    ["price", "unit price", "avg price", "mrp", "rate"],
    "discount": ["discount", "disc"],
}

_LABEL_ALIASES = {
    "region":       ["region", "market", "territory", "area", "geo region", "state region"],
    "state":        ["state", "province"],
    "city":         ["city"],
    "country":      ["country"],
    "category":     ["category", "product category", "cat"],
    "sub-category": ["sub-category", "sub category", "subcategory", "sub cat", "sub-cat", "product subcategory"],
    "segment":      ["segment", "customer segment"],
    "product":      ["product", "product name", "item", "sku"],
    "ship mode":    ["ship mode", "shipping mode"],
    "payment mode": ["payment mode", "payment method", "payment"],
    "customer":     ["customer", "customer name", "client"],
}

# “of/for X by Y”, “for X and Y”, generic “X by Y”
_OF_BY_RE        = re.compile(r"\b(?:of|for)\s+([a-z0-9 _\-]+?)\s+(?:by|per)\s+([a-z0-9 _\-]+)\b", re.I)
_FOR_AND_RE      = re.compile(r"\bfor\s+([a-z0-9 _\-]+?)\s+(?:and|vs|by)\s+([a-z0-9 _\-]+)\b", re.I)
_X_BY_Y_RE       = re.compile(r"\b([a-z0-9 _\-]+?)\s+by\s+([a-z0-9 _\-]+)\b", re.I)
_WISE_RE         = re.compile(r"\b([a-z0-9 _\-]+?)\s*(?:-|\s)?wise\b", re.I)

# scatter pairs
_SCATTER_VS_RE   = re.compile(r"([a-z0-9 _\-]+)\s*(?:vs|versus|against)\s*([a-z0-9 _\-]+)", re.I)
_SCATTER_AND_RE  = re.compile(r"([a-z0-9 _\-]+?)\s*(?:and|,)\s*([a-z0-9 _\-]+)\b", re.I)

# time
_YEAR_RE         = re.compile(r"\b(19|20)\d{2}\b")
_TIME_RE         = re.compile(r"\b(year|quarter|month|week|day)s?\b", re.I)
_TIME_TO_PERIOD  = {"year":"Y", "quarter":"Q", "month":"M", "week":"W", "day":"D"}

# aggregations
_AGG_WORDS = {
    "mean":   re.compile(r"\b(average|avg|mean)\b", re.I),
    "median": re.compile(r"\bmedian\b", re.I),
    "max":    re.compile(r"\b(max|maximum|highest|largest|top)\b", re.I),
    "min":    re.compile(r"\b(min|minimum|lowest|smallest|bottom)\b", re.I),
    "sum":    re.compile(r"\b(sum|total|overall)\b", re.I),
    "count":  re.compile(r"\b(count|how many|number of|no\.?\s*of|orders?|transactions?|rows?|records?)\b", re.I),
}
_UNIQUE_RE = re.compile(r"\b(unique|distinct)\b", re.I)

# sheet hints
_SHEET_PATS = [
    re.compile(r"\bfrom\s+sheet\s+([A-Za-z0-9 _\-]+)\b", re.I),
    re.compile(r"\bsheet\s*[:=]\s*([A-Za-z0-9 _\-]+)\b", re.I),
    re.compile(r"\btab\s*[:=]?\s*([A-Za-z0-9 _\-]+)\b", re.I),
    re.compile(r"\bsheet\s+([A-Za-z0-9 _\-]+)\b", re.I),
]

_STOP = set("""
the a an and or of for in on to me give show make plot chart graph wise by vs versus against with using please
bar line pie scatter hist histogram box trend over time first last top bottom overall bubble
""".split())

def _normalize_term(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    if s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s.strip()

def _ngrams_from_question(q: str, max_n: int = 3) -> List[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", (q or "").lower()) if t not in _STOP]
    phrases: List[str] = []
    n = len(toks)
    for k in range(min(max_n, 3), 0, -1):
        for i in range(n - k + 1):
            seg = " ".join(toks[i:i+k]).strip()
            if len(seg) >= 3 and seg not in _STOP:
                phrases.append(seg)
    seen = set(); out = []
    for p in phrases:
        if p not in seen:
            out.append(p); seen.add(p)
    return out[:50]

# dtype cache
_cols_dtype_cache: Dict[str, Any] = {}

def _best_col_match(cols: List[str], term: str, *, numeric: Optional[bool]=None, min_ratio: float=0.76) -> Optional[str]:
    if not term:
        return None
    target = _normalize_term(term)
    cand_cols = []
    for c in cols:
        dt = _cols_dtype_cache.get(c, object)
        if numeric is True and (not pd.api.types.is_numeric_dtype(dt)):
            continue
        if numeric is False and pd.api.types.is_numeric_dtype(dt):
            continue
        cand_cols.append(c)

    nmap = {c: _normalize_term(c) for c in cand_cols}
    # exact
    for c, n in nmap.items():
        if n == target:
            return c
    # token containment
    for c, n in nmap.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])", n):
            return c
    # fuzzy
    best, score = None, 0.0
    for c, n in nmap.items():
        r = difflib.SequenceMatcher(None, n, target).ratio()
        if r > score:
            best, score = c, r
    return best if score >= min_ratio else None

def _detect_plot_kind(q: str) -> str:
    for kind, pat in _PLOT_PATS:
        if pat.search(q):
            return kind
    if re.search(r"\b(over\s+time|trend|per\s+month|monthly|yearly|time\s+series)\b", q, re.I):
        return "line"
    return "bar"

def _parse_sheet_from_question(q: str) -> Optional[str]:
    for pat in _SHEET_PATS:
        m = pat.search(q)
        if m:
            name = (m.group(1) or "").strip()
            if name and name.lower() not in {"the", "first", "sheet"}:
                return name
    return None

def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    prefs = ["order date","ship date","date","invoice date","posting date","order_date","ship_date","orderdate","shipdate"]
    # fast exacts
    for p in ["date", "order date"]:
        m = _best_col_match(list(df.columns), p, numeric=False)
        if m: return m
    # aliases
    for p in prefs:
        m = _best_col_match(list(df.columns), p, numeric=False)
        if m: return m
    # convertible objects
    for c in df.columns:
        if df[c].dtype == object:
            vals = pd.to_datetime(df[c], errors="coerce")
            if vals.notna().any():
                return c
    # already datetime
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def _apply_year_filter(df: pd.DataFrame, q: str) -> Tuple[pd.DataFrame, Optional[int]]:
    m = _YEAR_RE.search(q or "")
    if not m:
        return df, None
    year = int(m.group(0))
    dcol = _find_date_col(df)
    if not dcol:
        return df, year
    dfx = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(dfx[dcol]):
        dfx[dcol] = pd.to_datetime(dfx[dcol], errors="coerce")
    mask = dfx[dcol].dt.year == year
    if mask.any():
        return dfx.loc[mask].copy(), year
    return dfx, year

def _detect_agg(q: str) -> Tuple[str, bool]:
    uq = bool(_UNIQUE_RE.search(q))
    for name, pat in _AGG_WORDS.items():
        if pat.search(q):
            return name, uq
    return "sum", uq

def _word_present(q: str, words: List[str]) -> bool:
    txt = (q or "").lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", txt) for w in words)

def _extract_requested_fields(q: str) -> Tuple[Optional[str], Optional[str]]:
    txt = (q or "").lower()
    m = _FOR_AND_RE.search(txt)
    if m: return (m.group(1).strip(), m.group(2).strip())
    m = _OF_BY_RE.search(txt)
    if m: return (m.group(1).strip(), m.group(2).strip())
    m = _X_BY_Y_RE.search(txt)
    if m: return (m.group(1).strip(), m.group(2).strip())
    label_hint = None
    m = _WISE_RE.search(txt)
    if m: label_hint = m.group(1).strip()
    metric_hint = None
    if _word_present(txt, _VALUE_ALIASES["profit"]):
        metric_hint = "profit"
    elif _word_present(txt, _VALUE_ALIASES["sales"]):
        metric_hint = "sales"
    else:
        for canon, words in _VALUE_ALIASES.items():
            if _word_present(txt, words):
                metric_hint = canon; break
    return metric_hint, label_hint

# ---------- strict metric resolver (profit ≠ margin/ratio) ----------
def _resolve_metric(df: pd.DataFrame, hint: str) -> Optional[str]:
    if not hint: return None
    h = _normalize_term(hint)
    cols = list(df.columns)
    nmap = {c: _normalize_term(c) for c in cols}

    def exact(name: str) -> Optional[str]:
        for c, n in nmap.items():
            if n == name: return c
        return None

    if h == "profit":
        c = exact("profit")
        if c: return c
        for c, n in nmap.items():
            if "profit" in n and not any(bad in n for bad in ["margin", "ratio", "percent", "pct", "%"]):
                if pd.api.types.is_numeric_dtype(df[c]): return c

    if h in ("sales", "sale"):
        c = exact("sales") or exact("sale")
        if c: return c

    canon = None
    for k, vs in _VALUE_ALIASES.items():
        if _normalize_term(k) == h or any(_normalize_term(v) == h for v in vs):
            canon = k; break
    return _best_col_match(cols, canon or hint, numeric=True)
# -------------------------------------------------------------------

def _resolve_label(df: pd.DataFrame, hint: str) -> Optional[str]:
    if not hint: return None
    h = _normalize_term(hint)
    canon = None
    for k, vs in _LABEL_ALIASES.items():
        if _normalize_term(k) == h or any(_normalize_term(v) == h for v in vs):
            canon = k; break
    if canon:
        m = _best_col_match(list(df.columns), canon, numeric=False)
        if m: return m
    return _best_col_match(list(df.columns), hint, numeric=False)

def _resolve_any_from_terms(df: pd.DataFrame, q: str, *, numeric: Optional[bool]) -> Optional[str]:
    for term in _ngrams_from_question(q):
        c = _best_col_match(list(df.columns), term, numeric=numeric)
        if c:
            return c
    return None

def _detect_time_grain(q: str) -> Optional[str]:
    m = _TIME_RE.search(q or "")
    return (m.group(1).lower() if m else None)


# ------------------------ Reader ------------------------
def _read_any_table(path: str, question: Optional[str] = None) -> pd.DataFrame:
    suf = Path(path).suffix.lower()
    if suf == ".csv":
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="latin1")
    xl = pd.ExcelFile(path)
    sheet = _parse_sheet_from_question(question or "")
    if sheet:
        low = [s.lower() for s in xl.sheet_names]
        if sheet.lower() in low:
            sname = xl.sheet_names[low.index(sheet.lower())]
        else:
            m = difflib.get_close_matches(sheet.lower(), low, n=1, cutoff=0.7)
            sname = xl.sheet_names[low.index(m[0])] if m else xl.sheet_names[0]
    else:
        sname = xl.sheet_names[0]
    return pd.read_excel(xl, sheet_name=sname)


# ------------------------ Plot engine ------------------------
@dataclass
class PlotMeta:
    id: str
    kind: str
    title: str
    question: str
    chat_id: Optional[str]
    x: Optional[str]
    y: Optional[str]
    created_at: str
    dataset_path: Optional[str]
    table: Optional[List[Dict[str, Any]]] = None
    table_preview: Optional[List[Dict[str, Any]]] = None

def _save_fig(plot_id: str) -> Tuple[str, str]:
    img = VIS_DIR / f"{plot_id}.png"
    thumb = VIS_DIR / f"{plot_id}_thumb.png"
    plt.tight_layout()
    plt.savefig(img, dpi=140, bbox_inches="tight")
    try:
        from PIL import Image
        im = Image.open(img); im.thumbnail((480, 320)); im.save(thumb)
    except Exception:
        plt.savefig(thumb, dpi=80, bbox_inches="tight")
    plt.close()
    return str(img), str(thumb)

def _write_meta(meta: PlotMeta) -> Dict[str, Any]:
    data = vars(meta).copy()

    # ✅ write CSV for values table and expose URL (snake_case + camelCase)
    try:
        if meta.table:
            csv_path = VIS_DIR / f"{meta.id}.csv"
            pd.DataFrame(meta.table).to_csv(csv_path, index=False, encoding="utf-8-sig")
            data["table_csv_url"] = f"/api/visualizations/{meta.id}/table"
            data["tableCsvUrl"] = data["table_csv_url"]
    except Exception:
        pass

    META_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(META_INDEX, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    data["image_url"] = f"/api/visualizations/{meta.id}/image"
    data["thumb_url"] = f"/api/visualizations/{meta.id}/thumb"
    return data

def _table_preview(tbl: List[Dict[str, Any]], n=10) -> List[Dict[str, Any]]:
    return tbl[:n] if isinstance(tbl, list) else None

def _safe_title(title: Optional[str], kind: str, metric: Optional[str], label: Optional[str]) -> str:
    if title:
        return title
    if label and metric:
        return f"{kind.title()} of {metric} by {label}"
    if metric:
        return f"{kind.title()} of {metric}"
    return kind.title()


# ------------------------ Public pipeline ------------------------
class PlotGenerationPipeline:
    def __init__(self):
        VIS_DIR.mkdir(parents=True, exist_ok=True)
        META_INDEX.parent.mkdir(parents=True, exist_ok=True)

    def list_meta(self, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not META_INDEX.exists():
            return []
        items: List[Dict[str, Any]] = []
        with open(META_INDEX, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if (not chat_id) or (obj.get("chat_id") == chat_id):
                        items.append(obj)
                except Exception:
                    continue
        return items

    def get_image_path(self, plot_id: str) -> Optional[str]:
        p = VIS_DIR / f"{plot_id}.png"
        return str(p) if p.exists() else None

    def get_thumb_path(self, plot_id: str) -> Optional[str]:
        p = VIS_DIR / f"{plot_id}_thumb.png"
        return str(p) if p.exists() else None

    # -------- high level --------
    def generate_and_store(self, file_path: str, question: str, title: Optional[str]=None, chat_id: Optional[str]=None) -> Dict[str, Any]:
        df = _read_any_table(file_path, question=question)
        return self._plot_and_store(df, question, title=title, chat_id=chat_id, dataset_path=file_path)

    def generate_and_store_combine(self, file_paths: List[str], question: str, title: Optional[str]=None, chat_id: Optional[str]=None) -> Dict[str, Any]:
        frames = []
        for p in file_paths:
            try:
                frames.append(_read_any_table(p, question=question))
            except Exception:
                pass
        if not frames:
            raise ValueError("No readable tables.")
        df = pd.concat(frames, ignore_index=True, sort=False)
        return self._plot_and_store(df, question, title=title, chat_id=chat_id, dataset_path=";".join(file_paths))

    # -------- core --------
    def _plot_and_store(self, df: pd.DataFrame, question: str, title: Optional[str], chat_id: Optional[str], dataset_path: Optional[str]) -> Dict[str, Any]:
        q = (question or "").strip()
        kind = _detect_plot_kind(q)

        df = df.copy()
        df = df.dropna(how="all").reset_index(drop=True)

        # year filter + dtype cache
        df, _ = _apply_year_filter(df, q)
        global _cols_dtype_cache
        _cols_dtype_cache = {c: df[c].dtype for c in df.columns}

        # fields
        metric_hint, label_hint = _extract_requested_fields(q)
        date_col   = _find_date_col(df)
        time_grain = _detect_time_grain(q)

        # metric (strict)
        metric = _resolve_metric(df, metric_hint) if metric_hint else None
        if not metric:
            metric = _resolve_any_from_terms(df, q, numeric=True)
        if not metric and kind == "scatter":
            pass
        elif not metric:
            raise ValueError("Could not identify a numeric measure from your request.")

        # label (only force time when explicitly requested)
        label = _resolve_label(df, label_hint) if label_hint else None
        if not label:
            if time_grain and date_col:
                label = "__period__"
            else:
                label = _resolve_any_from_terms(df, q, numeric=False)
        if label == metric:
            others = [c for c in df.columns if c != metric]
            label = _resolve_any_from_terms(df[others], q, numeric=False) or label

        # scatter: extract X & Y robustly (supports numeric-numeric and numeric-categorical)
        xcol = ycol = None
        scatter_mode = None  # "num-num" or "num-cat"
        num_col = cat_col = None

        if kind == "scatter":
            m = _SCATTER_VS_RE.search(q) or (_SCATTER_AND_RE.search(q) if "scatter" in q.lower() else None)
            if m:
                t1, t2 = m.group(1).strip(), m.group(2).strip()

                # try numeric-numeric first
                t1_num = _best_col_match(list(df.columns), t1, numeric=True)
                t2_num = _best_col_match(list(df.columns), t2, numeric=True)
                if t1_num and t2_num:
                    xcol, ycol = t1_num, t2_num
                    scatter_mode = "num-num"
                else:
                    # numeric + categorical (either order)
                    t1_cat = _best_col_match(list(df.columns), t1, numeric=False)
                    t2_cat = _best_col_match(list(df.columns), t2, numeric=False)

                    if t1_num and t2_cat and not pd.api.types.is_numeric_dtype(df[t2_cat]):
                        num_col, cat_col = t1_num, t2_cat
                        scatter_mode = "num-cat"
                    elif t2_num and t1_cat and not pd.api.types.is_numeric_dtype(df[t1_cat]):
                        num_col, cat_col = t2_num, t1_cat
                        scatter_mode = "num-cat"

            # fallback via metric/label hints
            if not scatter_mode:
                if metric and label and not pd.api.types.is_numeric_dtype(df[label]):
                    num_col, cat_col = metric, label
                    scatter_mode = "num-cat"
                elif metric:
                    other_nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != metric]
                    if other_nums:
                        xcol, ycol = metric, other_nums[0]
                        scatter_mode = "num-num"

            # last resort: any two numerics
            if not scatter_mode:
                nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(nums) >= 2:
                    xcol, ycol = nums[:2]
                    scatter_mode = "num-num"

            # keep meta x/y correct for num-cat
            if scatter_mode == "num-cat":
                xcol, ycol = cat_col, num_col

        agg, unique_flag = _detect_agg(q)

        # build period if needed
        if label == "__period__" and date_col:
            tmp = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(tmp[date_col]):
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            period = _TIME_TO_PERIOD.get(time_grain or "month", "M")
            tmp["__period__"] = tmp[date_col].dt.to_period(period).dt.to_timestamp()
            df = tmp

        # -------- KPI (single value) --------
        if kind != "scatter" and (label is None or label == "") and metric:
            if any(pat.search(q) for pat in _AGG_WORDS.values()):
                plt.figure(figsize=(6.5, 3.6))
                val_series = pd.to_numeric(df[metric], errors="coerce")
                if agg == "count":
                    value = int(val_series.notna().sum())
                    yname = "count"
                else:
                    fun = {"sum":np.nansum, "mean":np.nanmean, "median":np.nanmedian, "max":np.nanmax, "min":np.nanmin}.get(agg, np.nansum)
                    value = float(fun(val_series.values))
                    yname = metric
                plt.bar([f"{agg}"], [value])
                plt.ylabel(yname)
                ttl = _safe_title(title, "bar", f"{agg} {metric}", None)
                plt.title(ttl)
                table = [{yname: value, "aggregation": agg}]
                pid = uuid.uuid4().hex; _save_fig(pid)
                meta = PlotMeta(
                    id=pid, kind="kpi", title=ttl, question=question,
                    chat_id=chat_id, x=None, y=yname,
                    created_at=datetime.utcnow().isoformat(),
                    dataset_path=dataset_path, table=table, table_preview=_table_preview(table)
                )
                return _write_meta(meta)

        # -------- plotting --------
        if kind != "scatter" and metric is None:
            raise ValueError("Could not find a numeric column to plot.")

        ttl = _safe_title(
            title, kind,
            (ycol if kind == "scatter" else metric),
            (xcol if kind == "scatter" else (label if label != "__period__" else f"{time_grain or 'month'}"))
        )

        table: Optional[List[Dict[str, Any]]] = None
        plt.figure(figsize=(8.8, 5.4))

        def _aggregate(gframe: pd.DataFrame, grp_key: str, value_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
            if agg == "count":
                if value_col:
                    if unique_flag:
                        g = gframe.groupby(grp_key, dropna=False)[value_col].nunique().reset_index(name="count")
                    else:
                        g = gframe.groupby(grp_key, dropna=False)[value_col].count().reset_index(name="count")
                else:
                    g = gframe.groupby(grp_key, dropna=False).size().rename("count").reset_index()
                return g, "count"
            if value_col is None:
                raise ValueError("No numeric metric specified for aggregation.")
            vals = pd.to_numeric(gframe[value_col], errors="coerce")
            gframe = gframe.copy(); gframe["_val_"] = vals
            fun = {"sum":"sum","mean":"mean","median":"median","max":"max","min":"min"}.get(agg,"sum")
            g = gframe.groupby(grp_key, dropna=False)["_val_"].agg(fun).reset_index().rename(columns={"_val_": value_col})
            return g, value_col

        if kind == "line":
            # explicit categorical/label wins
            if label and label != "__period__":

                g, yname = _aggregate(df, label, metric if agg!="count" else metric)
                g = g.sort_values(by=yname, ascending=False)
                plt.plot(range(len(g[label])), g[yname])
                plt.xticks(range(len(g[label])), g[label].astype(str), rotation=45, ha="right")
                plt.ylabel(yname); plt.xlabel(label); plt.title(ttl)
                table = g.to_dict(orient="records")
            elif label == "__period__" and "__period__" in df.columns:
                g, yname = _aggregate(df, "__period__", metric if agg!="count" else metric)
                g = g.sort_values("__period__")
                plt.plot(g["__period__"], g[yname]); plt.xticks(rotation=45, ha="right")
                plt.xlabel(time_grain.title()); plt.ylabel(yname); plt.title(ttl)
                table = g.rename(columns={"__period__": time_grain}).to_dict(orient="records")
            elif date_col is not None:
                tmp = df.copy()
                if not pd.api.types.is_datetime64_any_dtype(tmp[date_col]):
                    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col])
                tmp["__period__"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
                g, yname = _aggregate(tmp, "__period__", metric if agg!="count" else metric)
                g = g.sort_values("__period__")
                plt.plot(g["__period__"], g[yname]); plt.xticks(rotation=45, ha="right")
                plt.xlabel("Month"); plt.ylabel(yname); plt.title(ttl)
                table = g.rename(columns={"__period__": "month"}).to_dict(orient="records")
            else:
                s = pd.to_numeric(df[metric], errors="coerce").dropna().reset_index(drop=True)
                plt.plot(s.index, s.values)
                plt.xlabel("Index"); plt.ylabel(metric); plt.title(ttl)
                table = [{"index": int(i), metric: float(v)} for i, v in enumerate(s.values)]

        elif kind == "bar":
            if label:
                g, yname = _aggregate(df, label, metric if agg!="count" else metric)
                g = g.sort_values(by=yname, ascending=False).head(25)
                plt.bar(g[label].astype(str), g[yname].values)
                plt.xticks(rotation=45, ha="right")
                plt.ylabel(yname); plt.xlabel(label); plt.title(ttl)
                table = g.to_dict(orient="records")
            elif date_col:
                tmp = df.copy()
                if not pd.api.types.is_datetime64_any_dtype(tmp[date_col]):
                    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col])
                tmp["__period__"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
                g, yname = _aggregate(tmp, "__period__", metric if agg!="count" else metric)
                g = g.sort_values("__period__")
                plt.bar(g["__period__"], g[yname]); plt.xticks(rotation=45, ha="right")
                plt.ylabel(yname); plt.xlabel("Month"); plt.title(ttl)
                table = g.rename(columns={"__period__": "month"}).to_dict(orient="records")
            else:
                s = pd.to_numeric(df[metric], errors="coerce").dropna().reset_index(drop=True)
                plt.bar(range(len(s)), s.values)
                plt.xlabel("Index"); plt.ylabel(metric); plt.title(ttl)
                table = [{"index": int(i), metric: float(v)} for i, v in enumerate(s.values)]

        elif kind == "pie":
            if not label:
                raise ValueError("Pie chart needs a category (e.g., 'by Region').")
            g, yname = _aggregate(df, label, metric if agg!="count" else metric)
            g = g.sort_values(by=yname, ascending=False)
            top = g.head(10).copy()
            if len(g) > 10:
                top.loc[len(top)] = {label: "Others", yname: g.iloc[10:][yname].sum()}
            plt.pie(top[yname].values, labels=top[label].astype(str).values, autopct="%1.1f%%")
            plt.title(ttl)
            table = top.to_dict(orient="records")
            pid = uuid.uuid4().hex; _save_fig(pid)
            meta = PlotMeta(
                id=pid, kind="pie", title=ttl, question=question,
                chat_id=chat_id, x=label, y=yname,
                created_at=datetime.utcnow().isoformat(),
                dataset_path=dataset_path, table=table, table_preview=_table_preview(table)
            )
            return _write_meta(meta)

        elif kind == "hist":
            s = pd.to_numeric(df[metric], errors="coerce").dropna()
            if s.empty:
                raise ValueError(f"No numeric data in {metric} for histogram.")
            bins = min(50, max(10, int(np.sqrt(len(s)))))
            counts, edges, _ = plt.hist(s, bins=bins)
            plt.xlabel(metric); plt.ylabel("count"); plt.title(ttl)
            table = [{"bin_start": float(edges[i]), "bin_end": float(edges[i+1]), "count": int(counts[i])} for i in range(len(counts))]

        elif kind == "box":
            if label and pd.to_numeric(df[metric], errors="coerce").notna().any():
                groups, tbl = [], []
                for val, grp in df.groupby(label, dropna=False):
                    arr = pd.to_numeric(grp[metric], errors="coerce").dropna().values
                    if len(arr):
                        groups.append(arr)
                        tbl.append({label: str(val), "count": int(len(arr)), "median": float(np.median(arr)), "q1": float(np.percentile(arr,25)), "q3": float(np.percentile(arr,75))})
                if not groups:
                    raise ValueError("No numeric data for box plot.")
                plt.boxplot(groups, labels=[t[label] for t in tbl], showfliers=False)
                plt.xticks(rotation=45, ha="right"); plt.ylabel(metric); plt.title(ttl)
                table = tbl
            else:
                s = pd.to_numeric(df[metric], errors="coerce").dropna().values
                if len(s) == 0:
                    raise ValueError("No numeric data for box plot.")
                plt.boxplot(s, vert=True, showfliers=False)
                plt.ylabel(metric); plt.title(ttl)
                table = [{"count": int(len(s)), "median": float(np.median(s)), "q1": float(np.percentile(s,25)), "q3": float(np.percentile(s,75))}]

        elif kind == "scatter":
            if scatter_mode == "num-num":
                dfx = df[[xcol, ycol]].copy()
                dfx[xcol] = pd.to_numeric(dfx[xcol], errors="coerce")
                dfx[ycol] = pd.to_numeric(dfx[ycol], errors="coerce")
                dfx = dfx.dropna()
                draw = dfx.sample(5000, random_state=42) if len(dfx) > 5000 else dfx

                plt.scatter(draw[xcol], draw[ycol], s=12, alpha=0.7)
                plt.xlabel(xcol); plt.ylabel(ycol); plt.title(_safe_title(title, "scatter", ycol, xcol))

                # values table: first 1000 points
                table = draw.head(1000).round(6).to_dict(orient="records")

                # put a quick correlation summary at row 0 (nice UX)
                try:
                    corr = float(draw[xcol].corr(draw[ycol])) if len(draw) >= 2 else float("nan")
                    table.insert(0, {"__summary__": "correlation", "value": corr})
                except Exception:
                    pass

            elif scatter_mode == "num-cat":
                # jittered scatter for numeric by category (e.g., Sales by Payment Mode)
                cat_col, num_col = xcol, ycol
                dfx = df[[cat_col, num_col]].copy()
                dfx[num_col] = pd.to_numeric(dfx[num_col], errors="coerce")
                dfx = dfx.dropna()

                dfx["__code__"] = dfx[cat_col].astype("category").cat.codes
                cats = list(dfx[cat_col].astype("category").cat.categories)
                jitter = (np.random.rand(len(dfx)) - 0.5) * 0.6

                plt.scatter(dfx["__code__"] + jitter, dfx[num_col], s=12, alpha=0.7)
                plt.xticks(range(len(cats)), [str(c) for c in cats], rotation=45, ha="right")
                plt.xlabel(cat_col); plt.ylabel(num_col)
                plt.title(_safe_title(title, "scatter", num_col, cat_col))

                # values table = aggregated numeric by category
                g, yname = _aggregate(df, cat_col, num_col if agg != "count" else num_col)
                g = g.sort_values(by=yname, ascending=False)
                table = g.to_dict(orient="records")

            else:
                raise ValueError("Could not determine fields for scatter plot (need numeric vs numeric OR numeric vs categorical).")

        else:
            # fallback bar
            if label:
                g, yname = _aggregate(df, label, metric if agg!="count" else metric)
                g = g.sort_values(by=yname, ascending=False).head(25)
                plt.bar(g[label].astype(str), g[yname].values)
                plt.xticks(rotation=45, ha="right")
                plt.ylabel(yname); plt.xlabel(label); plt.title(ttl)
                table = g.to_dict(orient="records")
            else:
                s = pd.to_numeric(df[metric], errors="coerce").dropna().reset_index(drop=True)
                plt.bar(range(len(s)), s.values)
                plt.xlabel("Index"); plt.ylabel(metric); plt.title(ttl)
                table = [{"index": int(i), metric: float(v)} for i, v in enumerate(s.values)]

        pid = uuid.uuid4().hex
        _save_fig(pid)
        meta = PlotMeta(
            id=pid, kind=kind, title=ttl, question=question,
            chat_id=chat_id,
            x=(xcol or (label if label != "__period__" else (time_grain or "period")) or ("date" if kind in ("bar","line") and _find_date_col(df) else None)),
            y=(ycol if kind == "scatter" else (metric if kind!="pie" else None)),
            created_at=datetime.utcnow().isoformat(),
            dataset_path=dataset_path,
            table=table,
            table_preview=_table_preview(table),
        )
        return _write_meta(meta)

# src/core/select_candidates.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from src.core.catalog_cache import get_catalog

# ---------------------------
# Tokenization & stopwords
# ---------------------------

STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "by", "with", "and", "or",
    "from", "at", "as", "is", "are", "be", "was", "were", "this", "that", "these",
    "those", "all", "any", "how", "many", "much", "count", "list", "top", "average",
    "avg", "sum", "min", "max", "where", "between", "group", "order", "sort",
    "show", "give", "get", "please", "records", "rows", "data", "report", "year",
    "month", "day", "today", "yesterday", "now"
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t]


def filter_tokens(tokens: Iterable[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]


# ---------------------------
# Scoring utilities
# ---------------------------

@dataclass
class ScorePart:
    name: str
    value: float


@dataclass
class TableScore:
    table: str
    score: float
    parts: List[ScorePart]


def _score_table(table: str, info: Dict, q_tokens: List[str]) -> TableScore:
    """
    Score a single table based on:
      - table name match
      - column name matches
      - sample value matches (distinct examples)
      - rowCount prior (log1p)
    """
    parts: List[ScorePart] = []
    base = table.split(".", 1)[-1].lower()

    # 1) Table-name hits
    tn_hits = sum(1 for t in q_tokens if t == base or (t in base and len(t) >= 3))
    if tn_hits:
        parts.append(ScorePart("table_name", 3.0 * tn_hits))

    # 2) Column-name hits
    cols: Dict[str, str] = (info.get("columns") or {})
    col_hits = 0
    for c in cols.keys():
        cl = c.lower()
        for t in q_tokens:
            if t == cl or (t in cl and len(t) >= 3):
                col_hits += 1
    if col_hits:
        parts.append(ScorePart("column_names", 1.2 * col_hits))

    # 3) Sample value hits (light booster)
    samples: Dict[str, List[str]] = info.get("samples") or {}
    sample_hits = 0
    if samples:
        for vals in samples.values():
            for v in vals:
                vl = str(v).lower()
                for t in q_tokens:
                    # exact token or substring in sample (avoid too-small tokens)
                    if len(t) >= 3 and (t == vl or t in vl):
                        sample_hits += 1
                        break
    if sample_hits:
        parts.append(ScorePart("sample_values", 0.8 * min(sample_hits, 6)))  # cap effect

    # 4) Row-count prior (bigger tables get a tiny prior)
    rc = int(info.get("rowCount") or 0)
    if rc > 0:
        parts.append(ScorePart("rowcount_prior", 0.3 * math.log1p(rc)))

    total = sum(p.value for p in parts)
    return TableScore(table=table, score=total, parts=parts)


def _fk_adjacency(cat: Dict) -> Dict[str, List[str]]:
    """
    Build undirected adjacency from catalog fks:
      dbo.A -> dbo.B  means neighbors[A].add(B) and neighbors[B].add(A)
    """
    adj: Dict[str, List[str]] = {}
    for fk in (cat.get("fks") or []):
        src = fk.get("from")
        dst = fk.get("to")
        if not (src and dst):
            continue
        adj.setdefault(src, [])
        adj.setdefault(dst, [])
        if dst not in adj[src]:
            adj[src].append(dst)
        if src not in adj[dst]:
            adj[dst].append(src)
    return adj


def _expand_neighbors(seeds: List[str], adj: Dict[str, List[str]], per_table: int = 2) -> List[str]:
    out = list(seeds)
    seen = set(out)
    for t in seeds:
        for nb in adj.get(t, [])[:max(0, per_table)]:
            if nb not in seen:
                out.append(nb)
                seen.add(nb)
    return out


# ---------------------------
# Public API
# ---------------------------

def pick_candidates(question: str, top_k: int = 6, expand_fks: bool = True, neighbors_per_table: int = 2) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Rank tables from the catalog for the given natural-language question.

    Returns:
        (tables, debug):
          tables -> ordered list like ["dbo.Transactions","dbo.Branches",...]
          debug  -> per-table score breakdown {table: {"score": float, "parts":[[name,value],...]}}
    """
    cat = get_catalog()
    tables: Dict[str, Dict] = cat.get("tables") or {}
    if not tables:
        return [], {}

    q_tokens = filter_tokens(tokenize(question))

    # Score each table
    scored: List[TableScore] = []
    for tname, tinfo in tables.items():
        scored.append(_score_table(tname, tinfo, q_tokens))

    # Sort by score desc, then name for stability
    scored.sort(key=lambda s: (-s.score, s.table))

    # Top-K seeds (non-zero scores first; if none, just take by row count)
    seeds = [s.table for s in scored if s.score > 0][:max(1, top_k)]
    if not seeds:
        seeds = [s.table for s in scored[:max(1, top_k)]]

    # Optional FK expansion to include obvious join partners
    if expand_fks:
        adj = _fk_adjacency(cat)
        seeds = _expand_neighbors(seeds, adj, per_table=max(0, neighbors_per_table))

    # Deduplicate but keep order
    seen = set()
    final: List[str] = []
    for t in seeds:
        if t not in seen:
            final.append(t)
            seen.add(t)

    # Build debug map
    debug: Dict[str, Dict] = {
        s.table: {
            "score": round(s.score, 3),
            "parts": [[p.name, round(p.value, 3)] for p in s.parts],
            "rowCount": int((tables.get(s.table) or {}).get("rowCount") or 0),
        }
        for s in scored
    }

    return final[:max(1, top_k + (neighbors_per_table if expand_fks else 0))], debug


def make_schema_snippet(tables: List[str]) -> str:
    """
    Produce a compact schema snippet for the chosen tables:
      TABLE dbo.Foo:
        - Col1 (int)
        - Col2 (nvarchar(100))
    """
    cat = get_catalog()
    catalog_tables: Dict[str, Dict] = cat.get("tables") or {}
    lines: List[str] = []
    for t in tables:
        info = catalog_tables.get(t)
        if not info:
            continue
        lines.append(f"TABLE {t}:")
        cols = info.get("columns") or {}
        for c, typ in sorted(cols.items()):
            if typ:
                lines.append(f"  - {c} ({typ})")
            else:
                lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines).rstrip()

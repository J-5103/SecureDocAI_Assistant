# src/pipeline/plot_pipeline.py
from __future__ import annotations

import json
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import csv

import pandas as pd
from src.components.plot_generator import PlotGenerator

# Try to use Pillow for thumbnails; fall back gracefully if unavailable
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None  # type: ignore
    _HAS_PIL = False

# ----- stable locations -----
BASE_DIR = Path(__file__).resolve().parents[2]
VIS_DIR = BASE_DIR / "static" / "visualizations"
META_PATH = VIS_DIR / "metadata.json"


class PlotGenerationPipeline:
    """Generate plots from Excel/CSV (single file; optional safe combine) and persist PNG + thumbnail + metadata."""

    def __init__(self) -> None:
        VIS_DIR.mkdir(parents=True, exist_ok=True)
        if not META_PATH.exists():
            META_PATH.write_text("[]", encoding="utf-8")

    # ---------- IO helpers ----------
    def _resolve_path(self, file_path: str) -> Path:
        p = Path(file_path)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return p

    def _sniff_delimiter(self, path: Path) -> Optional[str]:
        try:
            sample = path.open("r", encoding="utf-8", errors="ignore").read(8192)
            return csv.Sniffer().sniff(sample).delimiter
        except Exception:
            return None

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Light normalisation: strip column names, drop fully empty cols/rows."""
        # ensure col names are strings
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        # drop columns/rows that are entirely empty
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")
        return df

    def _read_csv_safe(self, path: Path) -> pd.DataFrame:
        sep = self._sniff_delimiter(path)
        try:
            return pd.read_csv(
                path,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                encoding="utf-8-sig",
            )
        except UnicodeDecodeError:
            # encoding fallback for odd CSVs
            return pd.read_csv(
                path,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
                encoding="latin1",
            )

    def _read_table_safely_single(self, path: Path) -> pd.DataFrame:
        """Read a single CSV/XLS/XLSX to a non-empty DataFrame; raise ValueError on empty/unusable."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suf = path.suffix.lower()
        if suf == ".csv":
            df = self._read_csv_safe(path)
        elif suf in (".xlsx", ".xls"):
            xl = pd.ExcelFile(path)
            if not xl.sheet_names:
                raise ValueError("Excel file has no sheets.")
            df = pd.read_excel(xl, sheet_name=0)
        else:
            raise ValueError("Only CSV, XLS, or XLSX supported.")

        df = self._clean_df(df)

        if df is None or df.empty or df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError("The file was read but contains no usable rows/columns.")
        return df

    def _load_df(self, file_path: str) -> pd.DataFrame:
        """Load one dataset robustly."""
        p = self._resolve_path(file_path)
        return self._read_table_safely_single(p)

    def _load_and_concat(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Safely load several files and concatenate rows (align by column names).
        Guards against empty frames so we never call pd.concat([]).
        """
        if not file_paths or len(file_paths) < 2:
            raise ValueError("At least two files are required to combine.")
        frames: List[pd.DataFrame] = []
        for fp in file_paths:
            try:
                df = self._read_table_safely_single(self._resolve_path(fp)).copy()
                df["_source_file"] = self._resolve_path(fp).name
                frames.append(df)
            except Exception:
                # Skip unreadable/empty files; we'll check at the end
                continue

        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            raise ValueError("No usable tables were found across the selected files.")
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True, sort=False)

    def _read_meta(self) -> List[Dict]:
        try:
            text = META_PATH.read_text(encoding="utf-8")
            return json.loads(text) if text.strip() else []
        except Exception:
            META_PATH.write_text("[]", encoding="utf-8")
            return []

    def _write_meta(self, data: List[Dict]) -> None:
        tmp = META_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(META_PATH)

    def _append_meta(self, meta: Dict) -> None:
        arr = self._read_meta()
        arr.insert(0, meta)  # newest first
        self._write_meta(arr)

    def _save_b64_png(self, image_b64: str, out_path: Path) -> None:
        raw = base64.b64decode(image_b64.encode("utf-8"))
        out_path.write_bytes(raw)

    def _make_thumb(self, image_path: Path, thumb_path: Path, max_px: int = 360) -> None:
        if _HAS_PIL:
            with Image.open(image_path) as im:
                im_copy = im.copy()
                im_copy.thumbnail((max_px, max_px))
                im_copy.save(thumb_path, format="PNG")
        else:
            # Minimal fallback: just copy full image as "thumb"
            thumb_path.write_bytes(image_path.read_bytes())

    # ---------- public ----------
    def list_meta(self, chat_id: Optional[str] = None) -> List[Dict]:
        arr = self._read_meta()
        if chat_id:
            arr = [m for m in arr if (m.get("chat_id") == chat_id)]
        return arr

    def generate_and_store(
        self,
        file_path: str,
        question: str,
        title: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Dict:
        """
        Generate a plot image + thumbnail using a SINGLE file's data,
        save under VIS_DIR and append a metadata entry.
        """
        df = self._load_df(file_path)

        # Render in-memory (base64) and write our own files named by plot_id
        plot_id = uuid.uuid4().hex
        gen = PlotGenerator(df)
        try:
            image_b64, info = gen.generate_plot_and_info(question)
        except Exception as e:
            msg = str(e)
            # Normalize common pandas concat/empty patterns to a clear message
            if "No objects to concatenate" in msg or "empty" in msg.lower():
                raise ValueError("No usable data was found to create the plot. Check your columns/filters.")
            raise

        img_path = VIS_DIR / f"{plot_id}.png"
        thumb_path = VIS_DIR / f"{plot_id}_thumb.png"
        self._save_b64_png(image_b64, img_path)
        self._make_thumb(img_path, thumb_path)

        kind = (info or {}).get("kind")
        x = (info or {}).get("x")
        y = (info or {}).get("y")

        meta = {
            "id": plot_id,
            "title": title or self._auto_title(kind, x, y),
            "kind": kind,
            "x": x,
            "y": y,
            "image_url": f"/api/visualizations/{plot_id}/image",
            "thumb_url": f"/api/visualizations/{plot_id}/thumb",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_file": self._resolve_path(file_path).name,
            "source_files": [self._resolve_path(file_path).name],
            "combined": False,
            "chat_id": chat_id,
            "question": question,   # <— store it so FE can show it
        }

        self._append_meta(meta)
        return meta

    def generate_and_store_combine(
        self,
        file_paths: List[str],
        question: str,
        title: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Dict:
        """
        (Optional) Generate a plot from multiple files combined.
        This is SAFE: skips unreadable/empty files and errors cleanly if nothing usable remains.
        """
        df = self._load_and_concat(file_paths)

        plot_id = uuid.uuid4().hex
        gen = PlotGenerator(df)
        try:
            image_b64, info = gen.generate_plot_and_info(question)
        except Exception as e:
            msg = str(e)
            if "No objects to concatenate" in msg or "empty" in msg.lower():
                raise ValueError("No usable data was found across the selected files.")
            raise

        img_path = VIS_DIR / f"{plot_id}.png"
        thumb_path = VIS_DIR / f"{plot_id}_thumb.png"
        self._save_b64_png(image_b64, img_path)
        self._make_thumb(img_path, thumb_path)

        kind = (info or {}).get("kind")
        x = (info or {}).get("x")
        y = (info or {}).get("y")
        src_names = [self._resolve_path(p).name for p in file_paths]

        meta = {
            "id": plot_id,
            "title": title or self._auto_title(kind, x, y),
            "kind": kind,
            "x": x,
            "y": y,
            "image_url": f"/api/visualizations/{plot_id}/image",
            "thumb_url": f"/api/visualizations/{plot_id}/thumb",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_file": None,
            "source_files": src_names,
            "combined": True,
            "chat_id": chat_id,
            "question": question,   # <— store it here as well
        }

        self._append_meta(meta)
        return meta

    # ---------- utils ----------
    def _auto_title(self, kind: Optional[str], x: Optional[str], y: Optional[str]) -> str:
        k = (kind or "plot").title()
        if (kind or "").lower() in ("hist", "histogram", "box", "pie"):
            return f"{k} • {x or 'Value'}"
        if x and y:
            return f"{k} • {y} vs {x}"
        if x:
            return f"{k} • {x}"
        return k

    def get_image_path(self, plot_id: str) -> Optional[str]:
        p = VIS_DIR / f"{plot_id}.png"
        return str(p) if p.exists() else None

    def get_thumb_path(self, plot_id: str) -> Optional[str]:
        p = VIS_DIR / f"{plot_id}_thumb.png"
        return str(p) if p.exists() else None

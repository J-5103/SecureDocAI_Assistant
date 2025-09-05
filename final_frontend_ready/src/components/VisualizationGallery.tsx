// src/components/VisualizationGallery.tsx
import React, { useEffect, useMemo, useState } from "react";
import { vizList, vizImageUrl } from "@/api/api";

type VizItem = {
  id: string;
  title: string;
  kind: string;
  x?: string | null;
  y?: string | null;
  image_url?: string;
  thumb_url?: string;
  created_at: string;
  question?: string;
  source_file?: string | null;
  source_files?: string[];
  combined?: boolean;
  chat_id?: string | null;
};

interface Props {
  refreshKey?: string;
  chatId?: string;     // optional filter by chat
  query?: string;      // optional text filter (matches title/question)
}

export default function VisualizationGallery({ refreshKey, chatId, query }: Props) {
  const [items, setItems] = useState<VizItem[]>([]);
  const [selected, setSelected] = useState<VizItem | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = async () => {
    setBusy(true);
    setErr(null);
    try {
      const data = await vizList(); // returns items[] from /api/visualizations/list
      const arr: VizItem[] = Array.isArray(data) ? data : data?.items || data?.plots || [];
      setItems(arr);
    } catch (e: any) {
      console.error(e);
      setErr(e?.message || "Failed to load visualizations");
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  const filtered = useMemo(() => {
    let out = items.slice();
    if (chatId) {
      out = out.filter((it) => (it.chat_id || "") === chatId);
    }
    if (query && query.trim()) {
      const q = query.trim().toLowerCase();
      out = out.filter(
        (it) =>
          (it.title || "").toLowerCase().includes(q) ||
          (it.question || "").toLowerCase().includes(q) ||
          (it.kind || "").toLowerCase().includes(q)
      );
    }
    // sort newest first by created_at
    out.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
    return out;
  }, [items, chatId, query]);

  const fallbackThumb = (it: VizItem): string | undefined => {
    if (it.thumb_url) return vizImageUrl(it.thumb_url);
    if (it.image_url) return vizImageUrl(it.image_url);
    return undefined;
  };

  const fullUrl = (it: VizItem): string | undefined =>
    it.image_url ? vizImageUrl(it.image_url) : undefined;

  const handleDownload = (it: VizItem) => {
    const url = fullUrl(it);
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(it.title || it.kind || "plot").replace(/\s+/g, "_")}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const handleShare = async (it: VizItem) => {
    const url = fullUrl(it);
    if (!url) return;
    try {
      if (navigator.share) {
        await navigator.share({ title: it.title || "Visualization", url });
      } else {
        await navigator.clipboard.writeText(url);
        alert("Link copied to clipboard");
      }
    } catch {
      /* no-op */
    }
  };

  return (
    <div className="grid gap-6">
      {/* Top bar */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Saved Visualizations</h2>
          <p className="text-sm text-muted-foreground">
            {chatId ? `For chat: ${chatId}` : "All plots"}
            {query ? ` • Filter: “${query}”` : ""}
          </p>
        </div>
        <button
          onClick={refresh}
          className="inline-flex items-center h-9 px-3 rounded-md border border-input text-sm hover:bg-accent/10 disabled:opacity-50"
          disabled={busy}
        >
          {busy ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {/* Empty / error / loading states */}
      {err && (
        <div className="p-3 rounded border border-red-200 bg-red-50 text-red-700 text-sm">
          {err}
        </div>
      )}

      {busy && !items.length ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ) : null}

      {!busy && filtered.length === 0 ? (
        <div className="h-full flex items-center justify-center bg-background">
          <div className="text-center">
            <div className="w-20 h-20 rounded-full bg-accent/15 flex items-center justify-center mx-auto mb-4">
              <span className="text-3xl">📊</span>
            </div>
            <div className="font-medium mb-1">No visualizations yet</div>
            <div className="text-sm text-muted-foreground">
              Generate a plot from the Excel/CSV tab — saved charts will show up here.
            </div>
          </div>
        </div>
      ) : null}

      {/* Cards */}
      <div className="grid md:grid-cols-3 sm:grid-cols-2 grid-cols-1 gap-4">
        {filtered.map((it) => {
          const thumb = fallbackThumb(it);
          return (
            <div
              key={it.id}
              className="border rounded-xl overflow-hidden cursor-pointer hover:shadow transition"
              onClick={() => setSelected(it)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setSelected(it);
                }
              }}
              role="button"
              tabIndex={0}
            >
              <div className="relative">
                {thumb ? (
                  <img
                    className="w-full h-48 object-cover"
                    src={thumb}
                    alt={it.title}
                    loading="lazy"
                  />
                ) : (
                  <div className="w-full h-48 flex items-center justify-center bg-accent/10">
                    <span className="text-2xl">📈</span>
                  </div>
                )}
                {it.combined ? (
                  <span className="absolute top-2 left-2 text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-700">
                    Combined
                  </span>
                ) : null}
                {it.kind ? (
                  <span className="absolute top-2 right-2 text-[10px] px-2 py-0.5 rounded bg-black/60 text-white uppercase tracking-wide">
                    {it.kind}
                  </span>
                ) : null}
              </div>
              <div className="p-3">
                <div className="font-medium truncate" title={it.title}>
                  {it.title}
                </div>
                <div className="text-xs text-muted-foreground">
                  {new Date(it.created_at).toLocaleString()}
                </div>
                {it.question ? (
                  <div className="text-xs text-muted-foreground mt-1 line-clamp-2" title={it.question}>
                    Q: {it.question}
                  </div>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>

      {/* Detail viewer */}
      {selected && (
        <div className="border rounded-xl p-4">
          <div className="flex items-start justify-between gap-3 mb-2">
            <div>
              <div className="font-semibold">{selected.title}</div>
              <div className="text-sm text-muted-foreground">
                {selected.kind} • {new Date(selected.created_at).toLocaleString()}
              </div>
            </div>
            <div className="flex gap-2">
              <button
                className="inline-flex items-center h-9 px-3 rounded-md border border-input text-sm hover:bg-accent/10 disabled:opacity-50"
                onClick={() => handleShare(selected)}
                disabled={!fullUrl(selected)}
              >
                Share
              </button>
              <button
                className="inline-flex items-center h-9 px-3 rounded-md border border-input text-sm hover:bg-accent/10 disabled:opacity-50"
                onClick={() => handleDownload(selected)}
                disabled={!fullUrl(selected)}
              >
                Download
              </button>
              <button
                className="inline-flex items-center h-9 px-3 rounded-md border border-input text-sm hover:bg-accent/10"
                onClick={() => setSelected(null)}
              >
                Close
              </button>
            </div>
          </div>

          <img
            className="w-full rounded-md border"
            src={fullUrl(selected)}
            alt={selected.title}
          />

          <div className="text-sm mt-3 space-y-1">
            {selected.question ? (
              <div>
                <span className="font-medium">Question:</span> {selected.question}
              </div>
            ) : null}
            <div>
              <span className="font-medium">Axes:</span> x: {selected.x || "—"}
              {selected.y ? ` • y: ${selected.y}` : ""}
            </div>
            <div className="text-muted-foreground">
              {selected.combined
                ? `Sources: ${(selected.source_files || []).join(" | ") || "—"}`
                : `Source: ${selected.source_file || "—"}`}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// src/components/VisualizationLightbox.tsx
import React, { useEffect, useMemo } from "react";
import { vizImageUrl } from "@/api/api";

type VizItem = {
  id: string;
  title?: string;
  kind?: string;
  x?: string | null;
  y?: string | null;
  image_url?: string;
  thumb_url?: string;
  created_at?: string;
  question?: string;
  chat_name?: string;
  chat_id?: string | null;
  combined?: boolean;
  source_file?: string | null;
  source_files?: string[];
};

interface Props {
  open: boolean;
  item?: VizItem | null;
  onClose: () => void;
}

export default function VisualizationLightbox({ open, item, onClose }: Props) {
  // ESC to close + lock scroll while open
  useEffect(() => {
    const onEsc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    if (open) {
      document.addEventListener("keydown", onEsc);
      const prev = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => {
        document.removeEventListener("keydown", onEsc);
        document.body.style.overflow = prev;
      };
    }
  }, [open, onClose]);

  const fullUrl = useMemo(() => {
    if (!item) return undefined;
    if (item.image_url) return vizImageUrl(item.image_url);
    if (item.thumb_url) return vizImageUrl(item.thumb_url);
    return undefined;
  }, [item]);

  const handleShare = async () => {
    if (!fullUrl) return;
    try {
      if (navigator.share) {
        await navigator.share({ title: item?.title || "Visualization", url: fullUrl });
      } else {
        await navigator.clipboard.writeText(fullUrl);
        alert("Link copied to clipboard");
      }
    } catch {}
  };

  const handleDownload = () => {
    if (!fullUrl) return;
    const a = document.createElement("a");
    a.href = fullUrl;
    a.download = `${(item?.title || item?.kind || "plot").replace(/\s+/g, "_")}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  if (!open || !item) return null;

  return (
    <div
      className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      <div
        className="max-w-6xl w-[92%] max-h-[88vh] bg-white rounded-xl overflow-hidden shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <div className="min-w-0">
            <div className="font-medium truncate">
              {(item.title || "Plot") + (item.chat_name ? ` • ${item.chat_name}` : "")}
            </div>
            <div className="text-xs text-gray-500">
              {(item.kind || "").toUpperCase()}
              {item.created_at ? ` • ${new Date(item.created_at).toLocaleString()}` : ""}
              {item.combined ? " • Combined" : ""}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              className="px-3 py-1 rounded-md border text-sm hover:bg-gray-50 disabled:opacity-50"
              onClick={handleShare}
              disabled={!fullUrl}
              title="Share / Copy link"
            >
              Share
            </button>
            <button
              className="px-3 py-1 rounded-md border text-sm hover:bg-gray-50 disabled:opacity-50"
              onClick={handleDownload}
              disabled={!fullUrl}
              title="Download PNG"
            >
              Download
            </button>
            <button className="px-3 py-1 rounded-md border text-sm" onClick={onClose}>
              Close
            </button>
          </div>
        </div>

        {/* Image */}
        <div className="p-3 flex items-center justify-center bg-neutral-50">
          {fullUrl ? (
            <img
              src={fullUrl}
              alt={item.title || "Visualization"}
              className="max-h-[72vh] w-auto object-contain"
            />
          ) : (
            <div className="h-[72vh] w-full flex items-center justify-center text-gray-400">
              No image available
            </div>
          )}
        </div>

        {/* Meta footer */}
        <div className="px-4 py-3 border-t text-sm">
          {item.question ? (
            <div className="mb-1">
              <span className="font-medium">Question:</span> {item.question}
            </div>
          ) : null}
          <div className="mb-1">
            <span className="font-medium">Axes:</span> x: {item.x || "—"}
            {item.y ? ` • y: ${item.y}` : ""}
          </div>
          {item.combined ? (
            <div className="text-gray-600">
              <span className="font-medium">Sources:</span>{" "}
              {(item.source_files || []).join(" | ") || "—"}
            </div>
          ) : (
            <div className="text-gray-600">
              <span className="font-medium">Source:</span> {item.source_file || "—"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

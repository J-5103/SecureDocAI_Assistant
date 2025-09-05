// src/components/Vizchatcard.tsx
// @ts-nocheck
import React from "react";
import { Link } from "react-router-dom";
import { vizImageUrl } from "@/api/api";

export default function Vizchatcard({
  item,
  onClick,
  to,
  className = "",
}) {
  // Distinguish cards:
  // - Plot meta objects have kind/image/thumb and created_at
  // - Chat rows (from /api/excel/chats) have chat_id/chat_name/latest_file
  const isPlot =
    !!item?.kind ||
    !!item?.image_url ||
    !!item?.thumb_url ||
    !!item?.created_at;

  const isChat = !isPlot && !!item?.chat_id;

  // Normalize fields used in UI
  const title =
    (isPlot ? item?.title || `${item?.kind || "Plot"}` : item?.chat_name) ||
    "Untitled";

  const subtitle = isPlot
    ? item?.question ||
      (item?.source_files?.length
        ? item?.source_files.slice(0, 2).join(" + ") +
          (item?.source_files.length > 2
            ? ` + ${item?.source_files.length - 2} more`
            : "")
        : item?.source_file || "")
    : item?.latest_file || "No data file";

  const createdAt =
    (isPlot ? item?.created_at : item?.created_at) || null; // chat list may not include it

  const combined =
    !!item?.combined || (Array.isArray(item?.source_files) && item.source_files.length > 1);

  const thumbUrl =
    isPlot && item?.thumb_url ? vizImageUrl(item.thumb_url) : undefined;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (!onClick) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onClick(item);
    }
  };

  const CardInner = (
    <div
      className={
        "rounded-xl border border-gray-200 hover:border-gray-300 cursor-pointer p-5 bg-white transition " +
        (className || "")
      }
      style={{ minHeight: 96 }}
      onClick={onClick ? () => onClick(item) : undefined}
      onKeyDown={onClick ? handleKeyDown : undefined}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : -1}
    >
      {/* Header row */}
      <div className="flex items-start gap-3">
        <span className="text-2xl" aria-hidden="true">
          {isPlot ? "📊" : "💬"}
        </span>
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-lg truncate flex items-center gap-2">
            <span className="truncate">{title}</span>
            {isPlot && combined && (
              <span className="px-2 py-0.5 rounded text-[10px] bg-amber-100 text-amber-700">
                Combined
              </span>
            )}
          </div>

          {/* subtitle: question (for plots) or latest file (for chats) */}
          {subtitle ? (
            <div
              className={`text-gray-500 text-sm mt-0.5 ${
                isPlot ? "line-clamp-2" : "truncate"
              }`}
              title={subtitle}
            >
              {isPlot ? `Q: ${subtitle}` : subtitle}
            </div>
          ) : null}

          {/* created at (when available) */}
          {createdAt ? (
            <div className="text-xs text-gray-400 mt-0.5">
              {new Date(createdAt).toLocaleString()}
            </div>
          ) : null}
        </div>
      </div>

      {/* Thumbnail only for plot items */}
      {isPlot && thumbUrl ? (
        <div className="mt-3 relative">
          <img
            src={thumbUrl}
            alt={title}
            className="w-full h-32 object-cover rounded-lg border"
            loading="lazy"
          />
          <div className="absolute top-2 left-2 flex items-center gap-2">
            {item?.kind ? (
              <span className="px-2 py-0.5 text-[10px] rounded bg-black/60 text-white uppercase tracking-wide">
                {item.kind}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );

  return to ? <Link to={to}>{CardInner}</Link> : CardInner;
}

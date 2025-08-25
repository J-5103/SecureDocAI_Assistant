// @ts-nocheck
import React from "react";
import { Link } from "react-router-dom";

export default function Vizchatcard({ item, onClick, to, className = "" }) {
  const isChat = !!item?.chat_id;

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
      <div className="flex items-center gap-3">
        <span className="text-2xl">💬</span>
        <div className="min-w-0">
          <div className="font-semibold text-lg truncate">
            {item?.chat_name || "Unknown chat"}
          </div>
          <div className="text-gray-500 text-sm mt-0.5 truncate">
            {isChat ? item?.latest_file || "No data file" : item?.title || "Untitled plot"}
          </div>
        </div>
      </div>

      {/* Thumbnail only for plot items */}
      {!isChat && item?.thumb_url ? (
        <div className="mt-3">
          <img
            src={item.thumb_url}
            alt={item?.title || "Plot thumbnail"}
            className="w-full h-32 object-cover rounded-lg"
            loading="lazy"
          />
        </div>
      ) : null}
    </div>
  );

  return to ? <Link to={to}>{CardInner}</Link> : CardInner;
}

import React, { useEffect } from "react";

export default function VisualizationLightbox({ open, item, onClose }) {
  useEffect(() => {
    const onEsc = (e) => e.key === "Escape" && onClose();
    if (open) document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [open, onClose]);

  if (!open || !item) return null;

  return (
    <div
      className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center"
      onClick={onClose}
    >
      <div className="max-w-6xl w-[92%] max-h-[88vh] bg-white rounded-xl overflow-hidden"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <div className="font-medium">
            {item.title || "Plot"} • {item.chat_name}
          </div>
          <button className="px-3 py-1 rounded-md border" onClick={onClose}>Close</button>
        </div>
        <div className="p-3 flex items-center justify-center bg-neutral-50">
          <img
            src={item.image_url}
            alt={item.title}
            className="max-h-[78vh] w-auto object-contain"
          />
        </div>
      </div>
    </div>
  );
}

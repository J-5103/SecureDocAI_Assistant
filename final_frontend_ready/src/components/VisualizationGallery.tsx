// src/components/VisualizationGallery.tsx
import React, { useEffect, useState } from "react";
import { vizList, vizImageUrl } from "@/api/api";

export default function VisualizationGallery({ refreshKey }: { refreshKey?: string }) {
  const [items, setItems] = useState<any[]>([]);
  const [selected, setSelected] = useState<any | null>(null);

  const refresh = async () => {
    try {
      const data = await vizList();
      setItems(Array.isArray(data) ? data : data?.plots || []);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => { refresh(); }, [refreshKey]);

  return (
    <div className="grid gap-6">
      <div className="grid md:grid-cols-3 sm:grid-cols-2 grid-cols-1 gap-4">
        {items.map((it) => (
          <div
            key={it.id}
            className="border rounded-xl overflow-hidden cursor-pointer hover:shadow"
            onClick={() => setSelected(it)}
          >
            <img className="w-full h-48 object-cover" src={vizImageUrl(it.thumb_url)} alt={it.title} />
            <div className="p-3">
              <div className="font-medium truncate">{it.title}</div>
              <div className="text-xs text-muted-foreground">
                {it.kind} • {new Date(it.created_at).toLocaleString()}
              </div>
            </div>
          </div>
        ))}
      </div>

      {selected && (
        <div className="border rounded-xl p-4">
          <div className="mb-2 font-semibold">{selected.title}</div>
          <img className="w-full" src={vizImageUrl(selected.image_url)} alt={selected.title} />
          <div className="text-sm mt-2">
            Source: {selected.source_file} • x: {selected.x}
            {selected.y ? ` • y: ${selected.y}` : ""} • type: {selected.kind}
          </div>
        </div>
      )}
    </div>
  );
}

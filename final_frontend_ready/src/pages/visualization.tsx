import React, { useEffect, useMemo, useState } from "react";
import { fetchVisualizations } from "@/api/visualizations_API";
import VisualizationCard from "@/components/Vizchatcard";
import VisualizationLightbox from "@/components/VisualizationLightbox";

export default function Visualizations() {
  const [items, setItems] = useState([]);
  const [chatFilter, setChatFilter] = useState("");
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async (chat = "") => {
    setLoading(true);
    try {
      const data = await fetchVisualizations(chat);
      setItems(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(chatFilter); }, [chatFilter]);

  const chats = useMemo(() => {
    const s = new Set(items.map(i => (i.chat_name || "").trim()).filter(Boolean));
    return Array.from(s).sort();
  }, [items]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(i =>
      (i.title || "").toLowerCase().includes(q) ||
      (i.chat_name || "").toLowerCase().includes(q)
    );
  }, [items, query]);

  return (
    <div className="px-6 py-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Visualizations</h1>
        <div className="flex gap-2">
          <input
            className="border rounded-md px-3 py-2 w-64"
            placeholder="Search by title or chat…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select
            className="border rounded-md px-3 py-2"
            value={chatFilter}
            onChange={(e) => setChatFilter(e.target.value)}
          >
            <option value="">All chats</option>
            {chats.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {/* “Chat Sessions” style blocks */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5 mt-6">
        {loading && <div className="text-gray-500">Loading…</div>}
        {!loading && filtered.length === 0 && (
          <div className="text-gray-500">No visualizations found.</div>
        )}
        {!loading && filtered.map(item => (
          <VisualizationCard key={item.id || item.filename} item={item} onClick={setActive} />
        ))}
      </div>

      <VisualizationLightbox open={!!active} item={active} onClose={() => setActive(null)} />
    </div>
  );
}

// src/pages/VisualizationPage.tsx
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Header } from "@/components/Header";
import { Button } from "@/components/ui/button";
import { RefreshCw } from "lucide-react";
import { vizList } from "@/api/api";

export default function VisualizationPage() {
  const [items, setItems] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [chatIds, setChatIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [sp, setSp] = useSearchParams();
  const chatId = sp.get("chatId") || "";

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const { items, total, chatIds } = await vizList({
        chatId: chatId || undefined,
        order: "desc",
        limit: 200,
      });
      setItems(items);
      setTotal(total || items.length);
      setChatIds(chatIds || []);
    } catch (e: any) {
      setErr(e?.message || "Failed to load visualizations.");
      setItems([]);
      setTotal(0);
      setChatIds([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  const handleChatChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const v = e.target.value || "";
    const next = new URLSearchParams(sp);
    if (v) next.set("chatId", v);
    else next.delete("chatId");
    setSp(next, { replace: true });
  };

  return (
    <div className="min-h-screen bg-gradient-secondary">
      <Header />
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-between mb-4 gap-2">
          <h1 className="text-3xl font-bold">Visualizations</h1>
          <div className="flex items-center gap-2">
            <select
              value={chatId}
              onChange={handleChatChange}
              className="px-2 py-1 border rounded text-sm"
              aria-label="Filter by chat"
            >
              <option value="">All chats</option>
              {chatIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
            <Button variant="outline" size="sm" onClick={load} disabled={loading}>
              <RefreshCw className={`w-4 h-4 mr-1 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </div>

        {err && (
          <div className="mb-4 text-sm text-red-600 border border-red-200 bg-red-50 rounded p-3">
            {err}
          </div>
        )}

        <div className="space-y-4">
          {loading ? (
            <div>Loading…</div>
          ) : items.length === 0 ? (
            <div>No visualizations found.</div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {items.map((item: any) => (
                <div key={item.id} className="border rounded-lg bg-card p-3">
                  <div className="font-medium mb-2 truncate" title={item.title || item.kind}>
                    {item.title || item.kind || "Plot"}
                  </div>
                  <a href={item.image_url} target="_blank" rel="noreferrer" title="Open full image">
                    <img
                      src={item.thumb_url || item.image_url}
                      alt={item.title || "plot"}
                      className="w-full h-auto rounded border object-contain bg-white"
                      loading="lazy"
                    />
                  </a>
                  <div className="mt-2 text-xs text-muted-foreground">
                    {item.created_at && new Date(item.created_at).toLocaleString()}
                    {item.chat_id ? (
                      <>
                        {" "}
                        · <span className="font-mono">{item.chat_id}</span>
                      </>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

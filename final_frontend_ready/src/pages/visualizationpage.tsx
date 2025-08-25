import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Header } from "@/components/Header";
import { Button } from "@/components/ui/button";
import { RefreshCw } from "lucide-react";
import { vizList } from "@/api/api";

export default function VisualizationPage() {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [sp, setSp] = useSearchParams();
  const chatId = sp.get("chatId") || undefined;
  const navigate = useNavigate();

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const data = await vizList(chatId);
      setItems(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setErr(e?.message || "Failed to load visualizations.");
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [chatId]);

  return (
    <div className="min-h-screen bg-gradient-secondary">
      <Header />
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-between mb-2">
          <h1 className="text-3xl font-bold">Visualizations</h1>
          <Button variant="outline" size="sm" onClick={load} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-1 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>

        <div className="space-y-4">
          {loading ? (
            <div>Loading…</div>
          ) : items.length === 0 ? (
            <div>No visualizations found.</div>
          ) : (
            <div>
              {items.map((item: any) => (
                <div key={item.id}>
                  <div>{item.title}</div>
                  {/* Display Plot/Image */}
                  <img src={item.image_url} alt={item.title} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

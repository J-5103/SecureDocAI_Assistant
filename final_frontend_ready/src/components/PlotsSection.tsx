import { BarChart, LineChart, PieChart, TrendingUp, Download, Share2, Trash2, ScatterChart } from "lucide-react";
import { vizImageUrl } from "@/api/api";

// Old shape from your Index page
export interface LegacyPlot {
  id: string;
  type: string;
  title: string;
  createdAt: string;
}

// New shape returned by /api/visualizations/*
interface VizMeta {
  id: string;
  title: string;
  kind: string;          // "bar" | "line" | "scatter" | "hist" | "box" | ...
  x?: string | null;
  y?: string | null;
  image_url: string;     // /api/visualizations/{id}/image
  thumb_url: string;     // /api/visualizations/{id}/thumb
  created_at: string;    // ISO
  source_file?: string;
}

// Unified display shape
type DisplayPlot = {
  id: string;
  kind: string;
  title: string;
  createdAt: string;
  imageUrl?: string;
  thumbUrl?: string;
};

interface PlotsSectionProps {
  plots: Array<LegacyPlot | VizMeta>;
}

const normalizePlot = (p: LegacyPlot | VizMeta): DisplayPlot => {
  if ("kind" in p) {
    return {
      id: p.id,
      kind: p.kind || "unknown",
      title: p.title || `${p.kind} plot`,
      createdAt: p.created_at,
      imageUrl: p.image_url,
      thumbUrl: p.thumb_url,
    };
  }
  return {
    id: p.id,
    kind: p.type || "unknown",
    title: p.title,
    createdAt: p.createdAt,
  };
};

const getPlotIcon = (kind: string) => {
  switch (kind) {
    case "bar":
      return <BarChart className="w-5 h-5 text-accent" />;
    case "line":
      return <LineChart className="w-5 h-5 text-success" />;
    case "pie":
      return <PieChart className="w-5 h-5 text-primary" />;
    case "scatter":
      return <ScatterChart className="w-5 h-5 text-purple-600" />;
    case "hist":
    case "histogram":
      // No Lucide Histogram — fallback to BarChart
      return <BarChart className="w-5 h-5 text-amber-600" />;
    case "box":
      return <TrendingUp className="w-5 h-5 text-pink-600" />;
    default:
      return <TrendingUp className="w-5 h-5 text-muted-foreground" />;
  }
};

export const PlotsSection = ({ plots }: PlotsSectionProps) => {
  const items: DisplayPlot[] = (plots || []).map(normalizePlot);

  const handleShare = async (fullUrl: string) => {
    try {
      if (navigator.share) {
        await navigator.share({ title: "Visualization", url: fullUrl });
      } else {
        await navigator.clipboard.writeText(fullUrl);
        alert("Link copied to clipboard");
      }
    } catch {}
  };

  if (!items.length) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="w-24 h-24 bg-gradient-accent rounded-full flex items-center justify-center mx-auto mb-6 shadow-glow animate-glow-pulse">
            <TrendingUp className="w-12 h-12 text-accent-foreground" />
          </div>
          <h3 className="text-xl font-semibold text-foreground mb-2">No Plots Generated Yet</h3>
          <p className="text-muted-foreground mb-6 max-w-md mx-auto">
            Ask the AI to create charts, graphs, or visualizations from your data. Saved plots will appear here.
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {["Create a bar chart", "Show me a line chart", "Generate a scatter plot", "Histogram of sales"].map((s) => (
              <span key={s} className="px-3 py-1 text-sm bg-secondary rounded-full text-muted-foreground">
                {s}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // shadcn "outline" style approximation
  const btnOutline = "inline-flex items-center justify-center h-8 px-3 rounded-md border border-input bg-background text-sm hover:bg-accent/10 disabled:opacity-50";

  return (
    <div className="h-full bg-background">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-foreground">Generated Plots</h2>
            <p className="text-muted-foreground">All your AI-generated visualizations</p>
          </div>
          <div className="flex space-x-2">
            <button className={btnOutline} disabled>
              <Download className="w-4 h-4 mr-2" />
              Export All
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {items.map((plot) => {
            const hasImage = !!plot.thumbUrl || !!plot.imageUrl;
            const thumb = plot.thumbUrl ? vizImageUrl(plot.thumbUrl) : plot.imageUrl ? vizImageUrl(plot.imageUrl) : undefined;
            const full = plot.imageUrl ? vizImageUrl(plot.imageUrl) : undefined;

            return (
              <div
                key={plot.id}
                className="group bg-card rounded-lg border border-border shadow-card hover:shadow-glow transition-all duration-200 animate-slide-up"
              >
                {/* Plot Preview */}
                <div className="h-48 rounded-t-lg overflow-hidden bg-gradient-secondary p-0 flex items-center justify-center">
                  {hasImage ? (
                    <img
                      src={thumb}
                      alt={plot.title}
                      className="w-full h-48 object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <div className="text-center">
                      {getPlotIcon(plot.kind)}
                      <p className="text-sm text-muted-foreground mt-2">
                        {plot.kind?.toUpperCase() || "PLOT"} Preview
                      </p>
                      <div className="mt-4 w-32 h-20 bg-accent/10 rounded border-2 border-dashed border-accent/30 flex items-center justify-center">
                        <TrendingUp className="w-8 h-8 text-accent/50" />
                      </div>
                    </div>
                  )}
                </div>

                {/* Plot Info */}
                <div className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-foreground truncate mb-1">{plot.title}</h3>
                      <p className="text-sm text-muted-foreground">
                        {new Date(plot.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex justify-between items-center opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                    <div className="flex space-x-2">
                      <a
                        href={full || "#"}
                        download
                        target="_blank"
                        rel="noreferrer"
                        className={`${btnOutline} ${!full ? "pointer-events-none" : ""}`}
                        title={full ? "Download PNG" : "No image"}
                      >
                        <Download className="w-3 h-3" />
                      </a>
                      <button
                        className={btnOutline}
                        disabled={!full}
                        onClick={() => full && handleShare(full)}
                        title="Share link"
                      >
                        <Share2 className="w-3 h-3" />
                      </button>
                    </div>
                    <button className={`${btnOutline} text-red-600 border-red-300`} disabled title="Delete coming soon">
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

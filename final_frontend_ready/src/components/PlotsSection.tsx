// src/components/plotsection.tsx
import {
  BarChart,
  LineChart,
  PieChart,
  TrendingUp,
  Download,
  Share2,
  Trash2,
  ScatterChart,
} from "lucide-react";
import { vizImageUrl } from "@/api/api";

// Old shape from your Index page
export interface LegacyPlot {
  id: string;
  type: string;
  title: string;
  createdAt: string;
}

// New shape returned by /api/visualizations/list and /generate*
interface VizMeta {
  id: string;
  title: string;
  kind: string;           // "bar" | "line" | "scatter" | "hist" | "box" | ...
  x?: string | null;
  y?: string | null;
  image_url: string;      // /api/visualizations/{id}/image
  thumb_url: string;      // /api/visualizations/{id}/thumb
  created_at: string;     // ISO
  source_file?: string | null;
  source_files?: string[]; // present for combined
  combined?: boolean;
  chat_id?: string | null;
  question?: string | null;
}

// Unified shape for rendering
type DisplayPlot = {
  id: string;
  kind: string;
  title: string;
  createdAt: string;
  imageUrl?: string;
  thumbUrl?: string;
  // extras for richer UI
  question?: string;
  x?: string | null;
  y?: string | null;
  sourceLabel?: string; // "file.xlsx" or "file1 + file2" etc.
  combined?: boolean;
};

interface PlotsSectionProps {
  plots: Array<LegacyPlot | VizMeta>;
  showQuestion?: boolean; // default true
  showSource?: boolean;   // default true
}

const normalizePlot = (p: LegacyPlot | VizMeta): DisplayPlot => {
  if ("kind" in p) {
    const srcSingle = (p.source_file || "")?.trim();
    const srcMany = (p.source_files || []) as string[];
    const sourceLabel =
      (srcMany && srcMany.length > 0
        ? srcMany.slice(0, 2).join(" + ") + (srcMany.length > 2 ? ` + ${srcMany.length - 2} more` : "")
        : srcSingle) || undefined;

    return {
      id: p.id,
      kind: p.kind || "unknown",
      title: p.title || `${p.kind} plot`,
      createdAt: p.created_at,
      imageUrl: p.image_url,
      thumbUrl: p.thumb_url,
      question: (p as VizMeta).question || undefined,
      x: p.x,
      y: p.y,
      sourceLabel,
      combined: !!p.combined,
    };
  }
  // legacy
  return {
    id: p.id,
    kind: p.type || "unknown",
    title: p.title,
    createdAt: p.createdAt,
  };
};

const getPlotIcon = (kind: string) => {
  switch ((kind || "").toLowerCase()) {
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
      return <BarChart className="w-5 h-5 text-amber-600" />;
    case "box":
      return <TrendingUp className="w-5 h-5 text-pink-600" />;
    default:
      return <TrendingUp className="w-5 h-5 text-muted-foreground" />;
  }
};

export const PlotsSection = ({
  plots,
  showQuestion = true,
  showSource = true,
}: PlotsSectionProps) => {
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
            Ask the AI to create charts from your Excel/CSV. Saved plots will appear here.
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {[
              "Bar chart of Sales by Region",
              "Line of revenue over date",
              "Scatter of Profit vs Sales",
              "Histogram of amount",
            ].map((s) => (
              <span key={s} className="px-3 py-1 text-sm bg-secondary rounded-full text-muted-foreground">
                {s}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const btnOutline =
    "inline-flex items-center justify-center h-8 px-3 rounded-md border border-input bg-background text-sm hover:bg-accent/10 disabled:opacity-50";

  return (
    <div className="h-full bg-background">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-foreground">Generated Plots</h2>
            <p className="text-muted-foreground">All your AI-generated visualizations</p>
          </div>
          <div className="flex space-x-2">
            <button className={btnOutline} disabled title="Coming soon">
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
                {/* Preview */}
                <div className="relative h-48 rounded-t-lg overflow-hidden bg-gradient-secondary p-0 flex items-center justify-center">
                  {hasImage ? (
                    <>
                      <img
                        src={thumb}
                        alt={plot.title}
                        className="w-full h-48 object-cover"
                        loading="lazy"
                      />
                      {/* corner badges */}
                      <div className="absolute top-2 left-2 flex items-center gap-2">
                        <span className="px-2 py-0.5 text-[10px] rounded bg-black/60 text-white uppercase tracking-wide">
                          {plot.kind || "plot"}
                        </span>
                        {plot.combined && (
                          <span className="px-2 py-0.5 text-[10px] rounded bg-amber-600 text-white">Combined</span>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="text-center">
                      {getPlotIcon(plot.kind)}
                      <p className="text-sm text-muted-foreground mt-2">{(plot.kind || "PLOT").toUpperCase()} Preview</p>
                      <div className="mt-4 w-32 h-20 bg-accent/10 rounded border-2 border-dashed border-accent/30 flex items-center justify-center">
                        <TrendingUp className="w-8 h-8 text-accent/50" />
                      </div>
                    </div>
                  )}
                </div>

                {/* Info */}
                <div className="p-4 space-y-2">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-foreground truncate">{plot.title}</h3>
                      <p className="text-xs text-muted-foreground">
                        {new Date(plot.createdAt).toLocaleString()}
                      </p>
                    </div>
                  </div>

                  {/* Question context */}
                  {showQuestion && plot.question && (
                    <div className="text-xs text-muted-foreground line-clamp-2">
                      <span className="font-medium text-foreground/80">Q:</span> {plot.question}
                    </div>
                  )}

                  {/* Axis + source */}
                  <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                    {plot.x && <span className="px-2 py-0.5 rounded bg-secondary/60">X: {plot.x}</span>}
                    {plot.y && <span className="px-2 py-0.5 rounded bg-secondary/60">Y: {plot.y}</span>}
                    {showSource && plot.sourceLabel && (
                      <span className="px-2 py-0.5 rounded bg-secondary/60" title={plot.sourceLabel}>
                        {plot.sourceLabel}
                      </span>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex justify-between items-center pt-1">
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

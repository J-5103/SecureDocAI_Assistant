// src/pages/index.tsx
import { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Header } from "@/components/Header";
import {
  MessageCircle,
  Users,
  BarChart3,
  FileText,
  IdCard,
  Database,
  Upload,
  FilePieChart,
  GitBranch,
} from "lucide-react";
import { vizList } from "@/api/api";

export interface Document {
  id: string;
  name?: string;
  type?: "pdf" | "excel" | "word" | string;
  size?: string;
  status: "uploaded" | "processing" | "ready";
  uploadDate: string;
  chatId?: string;
}

export interface Plot {
  id: string;
  title: string;
  type: string;
  data: any;
  createdAt: string;
}

const Index = () => {
  const navigate = useNavigate();

  // 🔝 Always start at TOP on home load/refresh
  useEffect(() => {
    if ("scrollRestoration" in window.history) {
      (window.history as any).scrollRestoration = "manual";
    }
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, []);

  // Live count for Visualizations card (upper part unchanged)
  const [vizCount, setVizCount] = useState<number | null>(null);
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const { total } = await vizList({});
        if (alive) setVizCount(typeof total === "number" ? total : 0);
      } catch {
        if (alive) setVizCount(0);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Informational feature cards
  const featureCards = useMemo(
    () => [
      {
        icon: <FileText className="w-6 h-6" />,
        title: "Files: Data Analysis with Chat",
        desc:
          "PDF / scanned images / Excel / CSV. Ask anything; get accurate counts, lists & summaries (OCR + tables + SQL).",
        tags: ["OCR", "Tables", "SQL-accurate"],
      },
      {
        icon: <IdCard className="w-6 h-6" />,
        title: "Business & Visiting Card Extraction",
        desc:
          "Extract name, phone, email & company. Export to Excel/CSV and view quick insights.",
        tags: ["OCR", "vCard", "CSV Export"],
      },
      {
        icon: <BarChart3 className="w-6 h-6" />,
        title: "Plot Generator",
        desc: "Create bar/line/pie charts from any table using plain English prompts.",
        tags: ["Auto-EDA", "Charts"],
      },
      {
        icon: <Database className="w-6 h-6" />,
        title: "Chat with Your Database",
        desc: "Read-only NL→SQL with review for safe execution on your DB.",
        tags: ["NL→SQL", "Read-only"],
      },
      {
        icon: <FilePieChart className="w-6 h-6" />,
        title: "Visualizations Library",
        desc: `Browse & manage saved visuals. Currently ${vizCount ?? "…"} item${
          (vizCount ?? 0) === 1 ? "" : "s"
        }.`,
        tags: ["Saved", "Shareable"],
      },
      {
        icon: <Upload className="w-6 h-6" />,
        title: "Excel Upload & Tagging",
        desc:
          "Bulk import contacts with tags; smart de-dup and tag updates on re-upload.",
        tags: ["De-dup", "Updater"],
      },
      {
        icon: <GitBranch className="w-6 h-6" />,
        title: "Branch Master",
        desc: "Manage branch directory; feeds filters across the app.",
        tags: ["Directory"],
      },
      {
        icon: <Users className="w-6 h-6" />,
        title: "Contact List",
        desc: "Unified contacts with tags & search. Excel import supported.",
        tags: ["Tags", "Search"],
      },
    ],
    [vizCount]
  );

  return (
    <div className="min-h-screen bg-gradient-secondary">
      <Header />

      <main className="container mx-auto p-6">
        {/* Welcome */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Welcome to SecureDocAI
          </h1>
          {/* keep this line as requested */}
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Explore powerful tools to analyze files, extract business cards, generate plots, and chat with your database—all in one place.
          </p>
        </div>

        {/* === UPPER PART (unchanged layout) === */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          <button
            onClick={() => navigate("/chats")}
            className="group p-8 bg-gradient-primary text-primary-foreground rounded-xl hover:shadow-glow transition-all duration-300 text-left transform hover:scale-105"
          >
            <div className="flex items-center space-x-4 mb:mb-0 md:mb-4">
              <div className="p-3 bg-primary-foreground/20 rounded-xl">
                <MessageCircle className="w-8 h-8" />
              </div>
              <h3 className="text-2xl font-bold">Chat Sessions</h3>
            </div>
            <p className="text-lg text-primary-foreground/90 leading-relaxed">
              Create dedicated chats for specific documents. Upload files, analyze content,
              and continue conversations across sessions with full history.
            </p>
          </button>

          {/* Quick Chat (light blue card) — CLICKABLE to /quick-chat */}
          <button
            onClick={() => navigate("/quick-chat")}
            className="group p-8 rounded-xl transition-all duration-300 text-left transform hover:scale-105
                       border-2 bg-sky-50 text-sky-900 border-sky-200 hover:border-sky-300 hover:shadow-card"
          >
            <div className="flex items-center space-x-4 md:mb-4">
              <div className="p-3 rounded-xl bg-sky-100">
                <Users className="w-8 h-8 text-sky-600" />
              </div>
              <h3 className="text-2xl font-bold">Quick Chat</h3>
            </div>
            <p className="text-lg leading-relaxed text-sky-900/90">
              Run quick queries to extract information from your database — no file
              upload needed. Uses natural-language to SQL (read-only).
            </p>
          </button>

          <button
            onClick={() => navigate("/visualizations")}
            className="group p-8 rounded-xl transition-all duration-300 text-left transform hover:scale-105 border-2 bg-card text-card-foreground border-border hover:border-accent/50 hover:shadow-card"
          >
            <div className="flex items-center space-x-4 md:mb-4">
              <div className="p-3 bg-accent/20 rounded-xl">
                <BarChart3 className="w-8 h-8 text-accent" />
              </div>
              <h3 className="text-2xl font-bold">Visualizations</h3>
            </div>
            <p className="text-lg opacity-90 leading-relaxed">
              View and manage generated plots from your chats. Currently showing{" "}
              {vizCount ?? "…"} visualization{(vizCount ?? 0) !== 1 ? "s" : ""}.
            </p>
          </button>
        </div>

        {/* === INFO CARDS SECTION (non-clickable) === */}
        <div className="bg-card rounded-xl border border-border shadow-card overflow-hidden">
          <div className="p-6 border-b border-border">
            <h2 className="text-xl font-semibold">What you can do here</h2>
            <p className="text-sm text-muted-foreground">
              A quick overview of key features available in SecureDocAI.
            </p>
          </div>

          <div className="p-6">
            <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {featureCards.map((f) => (
                <div
                  key={f.title}
                  className="rounded-2xl border border-border bg-background p-4 hover:shadow-card transition-shadow"
                >
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-xl bg-accent/15 text-accent">{f.icon}</div>
                    <h3 className="text-base font-semibold">{f.title}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground min-h-[56px]">{f.desc}</p>
                  {f.tags?.length ? (
                    <div className="flex flex-wrap gap-2 mt-3">
                      {f.tags.map((t) => (
                        <span
                          key={t}
                          className="text-xs px-2 py-1 rounded-full border border-border bg-card/50"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        </div>
        {/* (seed query bar removed as requested) */}
      </main>
    </div>
  );
};

export default Index;

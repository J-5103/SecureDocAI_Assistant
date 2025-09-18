// src/pages/QuickChat.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Plus, Search, Clock, ArrowLeft, Trash2 } from "lucide-react";
import {
  listQuickChats,
  getQuickChatMessages,
  createQuickChat,
  sendQuickChatMessage,
  deleteQuickChat, // must exist in src/api/quickchat.(ts|js)
} from "@/api/quickchat";

type ChatSummary = { id: string; title: string; createdAt: string };
type Message = { id: string; text: string; sender: "user" | "ai"; at: string };

type LocState = { seedPrompt?: string };

const timeAgo = (iso: string) => {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const diff = (Date.now() - t) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(iso).toLocaleString();
};

const DEFAULT_GREETING =
  "Hello! I’m ready to help. Ask anything—counts, lists, or quick insights.";

const QuickChat: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const seedPrompt = ((location.state as LocState | null)?.seedPrompt || "").trim();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [filter, setFilter] = useState("");
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [composer, setComposer] = useState("");

  const threadRef = useRef<HTMLDivElement>(null);

  // ===== initial load =====
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const items = (await listQuickChats()) as ChatSummary[] | undefined;
        if (!alive) return;
        const list = items ?? [];
        setChats(list);

        // preselect newest chat if present
        if (list[0]) {
          const firstId = list[0].id;
          setActiveChatId(firstId);
          const msgs = (await getQuickChatMessages(firstId)) as Message[] | undefined;
          if (!alive) return;
          setMessages(msgs ?? []);
        }
      } catch (e) {
        if (alive) setError("Failed to load quick chats.");
        console.error(e);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // ===== handle seed prompt (optional) =====
  useEffect(() => {
    if (!seedPrompt) return;
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const newId = (await createQuickChat(seedPrompt)) as string;
        const createdAt = new Date().toISOString();
        if (!alive) return;
        setChats((prev) => [{ id: newId, title: seedPrompt.slice(0, 64), createdAt }, ...prev]);
        setActiveChatId(newId);
        setMessages([{ id: `u_${Date.now()}`, text: seedPrompt, sender: "user", at: createdAt }]);
      } catch (e) {
        if (alive) setError("Failed to start a new quick chat.");
        console.error(e);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [seedPrompt]);

  // ===== autoscroll thread =====
  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // ===== derived =====
  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return chats;
    return chats.filter((c) => (c.title || "").toLowerCase().includes(q));
  }, [chats, filter]);

  const activeChat = useMemo(
    () => chats.find((c) => c.id === activeChatId) ?? null,
    [chats, activeChatId]
  );
  const activeTitle = activeChat?.title ?? "Quick Chat";

  // ===== actions =====
  const selectChat = async (id: string) => {
    if (!id || id === activeChatId) return;
    try {
      setLoading(true);
      setActiveChatId(id);
      const msgs = (await getQuickChatMessages(id)) as Message[] | undefined;
      setMessages(msgs ?? []);
    } catch (e) {
      setError("Failed to load messages for this chat.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // >>> UPDATED: seed the new chat with a default assistant greeting
  const startNewChat = async () => {
    try {
      setLoading(true);
      const newId = (await createQuickChat()) as string;
      const now = new Date().toISOString();

      // add to list & activate
      setChats((prev) => [{ id: newId, title: "New quick chat", createdAt: now }, ...prev]);
      setActiveChatId(newId);

      // seed right panel with greeting
      setMessages([{ id: `a_${Date.now()}`, text: DEFAULT_GREETING, sender: "ai", at: now }]);

      setComposer("");
    } catch (e) {
      setError("Failed to create a new quick chat.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const removeChat = async (id: string | null) => {
    if (!id) return;
    if (!window.confirm("Delete this chat permanently?")) return;
    try {
      setLoading(true);
      await deleteQuickChat(id);

      // choose a next selection (neighbor: next, else prev)
      setChats((prev) => {
        const idx = prev.findIndex((c) => c.id === id);
        const neighbor = idx >= 0 ? prev[idx + 1] || prev[idx - 1] : null;
        const remaining = prev.filter((c) => c.id !== id);

        if (activeChatId === id) setActiveChatId(neighbor ? neighbor.id : null);
        return remaining;
      });

      if (activeChatId === id) {
        const remaining = chats.filter((c) => c.id !== id);
        if (remaining.length) {
          const pick = remaining[0];
          const msgs = (await getQuickChatMessages(pick.id)) as Message[] | undefined;
          setMessages(msgs ?? []);
        } else {
          setMessages([]);
        }
      }
    } catch (e) {
      setError("Failed to delete chat.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (text: string) => {
    const t = (text || "").trim();
    if (!t) return;
    setError("");
    setLoading(true);

    try {
      let chatId = activeChatId;
      if (!chatId) {
        chatId = (await createQuickChat()) as string;
        const createdAt = new Date().toISOString();
        setActiveChatId(chatId);
        setChats((prev) => [{ id: chatId, title: "New quick chat", createdAt }, ...prev]);

        // give a greeting if we had to make a chat implicitly
        setMessages([{ id: `a_${Date.now()}`, text: DEFAULT_GREETING, sender: "ai", at: createdAt }]);
      }

      const at = new Date().toISOString();
      // optimistic user message
      setMessages((prev) => [...prev, { id: `u_${Date.now()}`, text: t, sender: "user", at }]);
      setComposer("");

      const reply = (await sendQuickChatMessage(chatId, t)) as string;
      setMessages((prev) => [
        ...prev,
        { id: `a_${Date.now()}`, text: reply, sender: "ai", at: new Date().toISOString() },
      ]);
    } catch (e) {
      setError("Failed to send message.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // ===== render =====
  return (
    <div className="h-screen">
      <div className="container mx-auto p-4 h-full">
        <div className="grid grid-cols-12 gap-4 h-full">
          {/* LEFT: New + Search + List (scrollable) */}
          <aside className="col-span-12 md:col-span-4 lg:col-span-3 bg-card border border-border rounded-2xl grid grid-rows-[auto_auto_1fr] overflow-hidden">
            {/* header */}
            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
              <div>
                <h2 className="text-[15px] font-semibold">Quick Chats</h2>
                <p className="text-xs text-muted-foreground">
                  {chats.length} total • {loading ? "syncing…" : "up to date"}
                </p>
              </div>
              <button
                onClick={startNewChat}
                className="inline-flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg bg-gradient-primary text-primary-foreground disabled:opacity-60"
                disabled={loading}
                title="Start a new quick chat"
              >
                <Plus className="w-4 h-4" />
                New
              </button>
            </div>

            {/* search */}
            <div className="px-3 py-3 border-b border-border">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="Search chats…"
                  className="w-full pl-9 pr-3 py-2 rounded-lg bg-muted text-foreground border border-border focus:outline-none focus:ring-2 focus:ring-accent"
                />
              </div>
            </div>

            {/* list */}
            <div className="overflow-auto divide-y divide-border">
              {filtered.length === 0 ? (
                <div className="p-4 text-sm text-muted-foreground">
                  {chats.length ? "No chats match your search." : "No chats yet — start one!"}
                </div>
              ) : (
                filtered.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => selectChat(c.id)}
                    className={`w-full text-left px-4 py-3 transition-colors ${
                      activeChatId === c.id ? "bg-muted" : "hover:bg-muted/60"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-sm font-medium truncate">
                        {c.title || "Untitled chat"}
                      </div>
                      <span className="shrink-0 text-[10px] text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {timeAgo(c.createdAt)}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      {new Date(c.createdAt).toLocaleString()}
                    </div>
                  </button>
                ))
              )}
            </div>
          </aside>

          {/* RIGHT: Header + Messages(scroll) + Composer(fixed) */}
          <section className="col-span-12 md:col-span-8 lg:col-span-9 bg-card border border-border rounded-2xl grid grid-rows-[auto_1fr_auto] overflow-hidden">
            {/* top bar */}
            <div className="px-5 py-4 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => navigate(-1)}
                  className="p-2 rounded-lg bg-muted hover:bg-muted/70"
                  title="Back"
                >
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <h3 className="text-lg font-semibold">{activeTitle}</h3>
              </div>
              <div className="flex items-center gap-2">
                {activeChatId && (
                  <button
                    onClick={() => removeChat(activeChatId)}
                    className="p-2 rounded-lg bg-muted hover:bg-muted/70 text-red-600 disabled:opacity-60"
                    title="Delete chat"
                    disabled={loading}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
                {loading && (
                  <span className="text-xs text-muted-foreground animate-pulse">thinking…</span>
                )}
              </div>
            </div>

            {/* messages */}
            <div ref={threadRef} className="overflow-auto px-5 py-4 space-y-3">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`px-4 py-3 rounded-xl shadow-sm max-w-[85%] ${
                    m.sender === "user"
                      ? "ml-auto bg-gradient-primary text-primary-foreground"
                      : "bg-muted text-foreground"
                  }`}
                >
                  <div className="text-sm whitespace-pre-wrap leading-relaxed">{m.text}</div>
                </div>
              ))}
              {!messages.length && (
                <div className="text-sm text-muted-foreground">
                  No messages yet. Ask your first question below.
                </div>
              )}
              {error && <div className="text-xs text-red-600">{error}</div>}
            </div>

            {/* composer */}
            <div className="px-4 py-3 border-t border-border">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!composer.trim() || loading) return;
                  sendMessage(composer);
                }}
                className="flex items-center gap-3"
              >
                <input
                  value={composer}
                  onChange={(e) => setComposer(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (composer.trim() && !loading) sendMessage(composer);
                    }
                  }}
                  placeholder="Ask anything… e.g., Count banks with STD code 079"
                  className="flex-1 px-4 py-3 rounded-lg bg-muted text-foreground border border-border focus:outline-none focus:ring-2 focus:ring-accent"
                  aria-label="Message input"
                />
                <button
                  type="submit"
                  className="px-5 py-3 rounded-lg bg-gradient-primary text-primary-foreground shadow-glow disabled:opacity-60"
                  disabled={loading || !composer.trim()}
                >
                  Send
                </button>
              </form>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default QuickChat;

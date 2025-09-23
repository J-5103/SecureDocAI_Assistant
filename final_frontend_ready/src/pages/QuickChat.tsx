// src/pages/QuickChat.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Plus, Search, Clock, ArrowLeft, Edit2, Check, X , Trash2 } from "lucide-react";
import MessageList, { type ChatMessage } from "@/components/messagelist";
import {
  listQuickChats as _listQuickChats,
  getQuickChatMessages as _getQuickChatMessages,
  createQuickChat as _createQuickChat,
  // NOTE: we won't use _sendQuickChatMessage to avoid losing clientMsgId
  deleteQuickChat as _deleteQuickChat,
  renameQuickChat as _renameQuickChat,
} from "@/api/quickchat";

type ChatSummary = { id: string; title: string; createdAt: string; lastMessageAt?: string };
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

const DEFAULT_GREETING = "Hello! I’m ready to help you for your queries.";

// ---------- API wrappers (server is source of truth) ----------
const extractId = (res: any): string => {
  if (!res) throw new Error("Empty response");
  if (typeof res === "string") return res;
  if (res?.id) return String(res.id);
  if (res?.chatId) return String(res.chatId);
  throw new Error("Unable to read chat id");
};

async function createQuickChatSafe(title?: string) {
  try {
    const raw = await _createQuickChat?.(title);
    return extractId(raw);
  } catch {
    const body: any = title ? { title } : {};
    const res = await fetch("/api/quick-chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Create failed (${res.status})`);
    const data = await res.json();
    return extractId(data);
  }
}

async function listQuickChatsSafe(): Promise<ChatSummary[]> {
  try {
    const data = (await _listQuickChats?.()) as any;
    return (Array.isArray(data) ? data : data?.items) ?? [];
  } catch {
    const r = await fetch("/api/quick-chats");
    if (!r.ok) throw new Error(`List failed (${r.status})`);
    const data = await r.json();
    return (data?.items ?? data) as ChatSummary[];
  }
}

async function getQuickChatMessagesSafe(id: string): Promise<ChatMessage[]> {
  try {
    const data = (await _getQuickChatMessages?.(id)) as any;
    return (Array.isArray(data) ? data : data?.items) ?? [];
  } catch {
    const r = await fetch(`/api/quick-chats/${id}/messages`);
    if (!r.ok) throw new Error(`Messages failed (${r.status})`);
    const data = await r.json();
    return (data?.items ?? data) as ChatMessage[];
  }
}

// IMPORTANT: always use fetch so clientMsgId reaches backend
async function sendQuickChatMessageSafe(id: string, text: string, clientMsgId: string) {
  const r = await fetch(`/api/quick-chats/${id}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, clientMsgId }),
  });
  if (!r.ok) throw new Error(`Send failed (${r.status})`);
  const data = await r.json();
  return typeof data === "string" ? data : data?.reply ?? "";
}

async function deleteQuickChatSafe(id: string) {
  try {
    return await _deleteQuickChat?.(id);
  } catch {
    const r = await fetch(`/api/quick-chats/${id}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`Delete failed (${r.status})`);
  }
}

async function renameQuickChatSafe(id: string, title: string) {
  try {
    return await _renameQuickChat?.(id, title);
  } catch {
    const r = await fetch(`/api/quick-chats/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (!r.ok) throw new Error(`Rename failed (${r.status})`);
  }
}

const newClientId = () =>
  (crypto?.randomUUID?.() ?? `cm_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);

// Red trash emoji-style icon (matches the screenshot)
const RedTrashEmoji: React.FC<{ className?: string }> = ({ className = "" }) => (
  <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
    {/* lid */}
    <path d="M9 6h6l.8 2H8.2L9 6Z" fill="#fff" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
    {/* top bar */}
    <path d="M4.5 8h15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    {/* bin */}
    <rect x="6" y="8.5" width="12" height="11" rx="2" fill="#fff" stroke="currentColor" strokeWidth="1.5"/>
    {/* two vertical bars */}
    <path d="M10 11v6.5M14 11v6.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);


// -----------------------------------------------------------------------------

const QuickChat: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const seedPrompt = ((location.state as LocState | null)?.seedPrompt || "").trim();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [filter, setFilter] = useState("");
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [composer, setComposer] = useState("");

  // rename UI states
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");

  const threadRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<number | null>(null);

  // ===== initial load =====
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const serverList = await listQuickChatsSafe();
        if (!alive) return;
        setChats(serverList);

        if (serverList[0]) {
          const id = serverList[0].id;
          setActiveChatId(id);
          const serverMsgs = await getQuickChatMessagesSafe(id);
          if (!alive) return;
          setMessages(serverMsgs);
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
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  // ===== handle seed prompt (optional) =====
  useEffect(() => {
    if (!seedPrompt) return;
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const newId = await createQuickChatSafe(seedPrompt);
        if (!alive) return;
        const createdAt = new Date().toISOString();
        setChats((prev) => [{ id: newId, title: seedPrompt.slice(0, 64), createdAt }, ...prev]);
        setActiveChatId(newId);
        // optimistic user line
        setMessages([{ id: `u_seed_${Date.now()}`, text: seedPrompt, sender: "user", at: createdAt }]);
        // sync from server once created
        const synced = await getQuickChatMessagesSafe(newId);
        if (!alive) return;
        setMessages(synced);
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
  const activeTitle = editingTitle ? titleDraft : activeChat?.title ?? "Quick Chat";

  // ===== actions =====
  const selectChat = async (id: string) => {
    if (!id || id === activeChatId) return;
    try {
      setLoading(true);
      setActiveChatId(id);
      const serverMsgs = await getQuickChatMessagesSafe(id);
      setMessages(serverMsgs);
      setEditingTitle(false);
    } catch (e) {
      setError("Failed to load messages for this chat.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const startNewChat = async () => {
    try {
      setLoading(true);
      const newId = await createQuickChatSafe();
      const now = new Date().toISOString();
      setChats((prev) => [{ id: newId, title: "New quick chat", createdAt: now }, ...prev]);
      setActiveChatId(newId);
      const serverMsgs = await getQuickChatMessagesSafe(newId);
      setMessages(serverMsgs);
      setComposer("");
      setEditingTitle(false);
    } catch (e: any) {
      setError(e?.message || "Failed to create a new quick chat.");
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
      await deleteQuickChatSafe(id);
      setChats((prev) => prev.filter((c) => c.id !== id));
      if (activeChatId === id) {
        const next = chats.filter((c) => c.id !== id)[0]?.id ?? null;
        setActiveChatId(next);
        setMessages(next ? await getQuickChatMessagesSafe(next) : []);
      }
    } catch (e) {
      setError("Failed to delete chat.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // Merge helper: keep optimistic user visible until server echoes the same clientMsgId
  const mergeOptimisticUser = (
    serverMsgs: ChatMessage[],
    optimisticUser: ChatMessage,
    clientMsgId: string
  ): ChatMessage[] => {
    // if server already has this user turn (by clientMsgId), just return server list
    const hasTurn = serverMsgs.some(
      // @ts-ignore: backend returns clientMsgId on user messages
      (m: any) => m.sender === "user" && m.clientMsgId === clientMsgId
    );
    if (hasTurn) return serverMsgs;

    // otherwise, append optimistic user at the right place (end)
    const merged = [...serverMsgs, optimisticUser];
    return merged;
  };

  const pollUntilAnswered = (chatId: string, clientMsgId: string, optimisticUser: ChatMessage) => {
    if (pollRef.current) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(async () => {
      try {
        const serverMsgs = await getQuickChatMessagesSafe(chatId);

        // keep optimistic user until server echoes it
        const nextMsgs = mergeOptimisticUser(serverMsgs, optimisticUser, clientMsgId);
        setMessages(nextMsgs);

        // if server already has this user, check AI after it
        const uIdx = nextMsgs.findIndex(
          (m: any) => m.sender === "user" && (m as any).clientMsgId === clientMsgId
        );
        if (uIdx >= 0) {
          const ai = nextMsgs.slice(uIdx + 1).find((m) => m.sender === "ai") as any;
          if (ai && !ai.thinking) {
            // done, final answer present
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = null;
            setLoading(false);
          }
        }
      } catch (e) {
        console.error(e);
      }
    }, 700);
  };

  const sendMessage = async (text: string) => {
    const t = (text || "").trim();
    if (!t) return;
    setError("");
    setLoading(true);

    try {
      let chatId = activeChatId;
      if (!chatId) {
        chatId = await createQuickChatSafe();
        const now = new Date().toISOString();
        setChats((prev) => [{ id: chatId!, title: "New quick chat", createdAt: now }, ...prev]);
        setActiveChatId(chatId);
        const serverMsgs = await getQuickChatMessagesSafe(chatId);
        setMessages(serverMsgs);
      }

      // optimistic user message (stays visible while we poll)
      const clientMsgId = newClientId();
      const optimisticUser: ChatMessage = {
        id: `u_${clientMsgId}`,
        text: t,
        sender: "user",
        at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, optimisticUser]);

      // start polling before POST returns, so we can catch the backend "🧠 Thinking…" asap
      pollUntilAnswered(chatId!, clientMsgId, optimisticUser);

      // POST with clientMsgId so backend can dedupe and link the turn
      await sendQuickChatMessageSafe(chatId!, t, clientMsgId);

      // one last sync (if polling already finished this is a no-op)
      const finalMsgs = await getQuickChatMessagesSafe(chatId!);
      setMessages(finalMsgs);
      setLoading(false);
    } catch (e) {
      setError("Failed to send message.");
      console.error(e);
      setLoading(false);
    } finally {
      setComposer("");
    }
  };

  // ===== rename handlers =====
  const beginRename = () => {
    if (!activeChatId) return;
    const current = chats.find((c) => c.id === activeChatId)?.title ?? "";
    setTitleDraft(current);
    setEditingTitle(true);
  };

  const cancelRename = () => {
    setEditingTitle(false);
    setTitleDraft("");
  };

  const saveRename = async () => {
    const t = (titleDraft || "").trim();
    if (!activeChatId || !t) return cancelRename();
    try {
      setLoading(true);
      setChats((prev) => prev.map((c) => (c.id === activeChatId ? { ...c, title: t } : c)));
      await renameQuickChatSafe(activeChatId, t);
    } catch (e) {
      setError("Failed to rename chat.");
      console.error(e);
    } finally {
      setLoading(false);
      setEditingTitle(false);
    }
  };

  return (
    <div className="h-screen">
      <div className="container mx-auto p-4 h-full">
        <div className="grid grid-cols-12 gap-4 h-full">
          {/* LEFT */}
          <aside className="col-span-12 md:col-span-4 lg:col-span-3 bg-card border border-border rounded-2xl grid grid-rows-[auto_auto_1fr] overflow-hidden">
            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
              <div>
                <h2 className="text-[15px] font-semibold">Quick Chats</h2>
                <p className="text-xs text-muted-foreground">{chats.length} total</p>
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

            <div className="overflow-auto divide-y divide-border">
              {(() => {
                const list = filtered;
                if (!list.length) {
                  return (
                    <div className="p-4 text-sm text-muted-foreground">
                      {chats.length ? "No chats match your search." : "No chats yet — start one!"}
                    </div>
                  );
                }
                return list.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => selectChat(c.id)}
                    className={`w-full text-left px-4 py-3 transition-colors ${
                      activeChatId === c.id ? "bg-muted" : "hover:bg-muted/60"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-sm font-medium truncate">{c.title || "Untitled chat"}</div>
                      <span className="shrink-0 text-[10px] text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {timeAgo(c.lastMessageAt || c.createdAt)}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      {(c.lastMessageAt || c.createdAt) &&
                        new Date(c.lastMessageAt || c.createdAt).toLocaleString()}
                    </div>
                  </button>
                ));
              })()}
            </div>
          </aside>

          {/* RIGHT */}
          <section className="col-span-12 md:col-span-8 lg:col-span-9 bg-card border border-border rounded-2xl grid grid-rows-[auto_1fr_auto] overflow-hidden">
            <div className="px-5 py-4 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => navigate(-1)}
                  className="p-2 rounded-lg bg-muted hover:bg-muted/70"
                  title="Back"
                >
                  <ArrowLeft className="w-4 h-4" />
                </button>

                {!editingTitle ? (
                  <div className="flex items-center gap-2">
                    <h3
                      className="text-lg font-semibold select-text"
                      onDoubleClick={beginRename}
                      title="Double-click to rename"
                    >
                      {activeTitle}
                    </h3>
                    {activeChatId && (
                      <button
                        onClick={beginRename}
                        className="p-2 rounded-lg bg-muted hover:bg-muted/70"
                        title="Rename chat"
                        disabled={loading}
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <input
                      autoFocus
                      value={titleDraft}
                      onChange={(e) => setTitleDraft(e.target.value)}
                      onBlur={saveRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") saveRename();
                        if (e.key === "Escape") cancelRename();
                      }}
                      className="px-3 py-1.5 rounded-lg bg-muted border border-border focus:outline-none focus:ring-2 focus:ring-accent"
                    />
                    <button onClick={saveRename} className="p-2 rounded-lg bg-muted hover:bg-muted/70" title="Save">
                      <Check className="w-4 h-4" />
                    </button>
                    <button onClick={cancelRename} className="p-2 rounded-lg bg-muted hover:bg-muted/70" title="Cancel">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
              <div className="flex items-center">
                {activeChatId && (
                  <button
                    onClick={() => removeChat(activeChatId)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded-lg"
                    title="Delete chat"
                    aria-label="Delete chat"
                    disabled={loading}
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                )}
              </div>
            </div>

            <div ref={threadRef} className="overflow-auto px-5 py-4">
              <MessageList messages={messages} />
              {!messages.length && (
                <div className="text-sm text-muted-foreground mt-2">
                  No messages yet. Ask your first question below.
                </div>
              )}
              {error && <div className="text-xs text-red-600 mt-2">{error}</div>}
            </div>

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
                  placeholder="Ask anything… about your database..."
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

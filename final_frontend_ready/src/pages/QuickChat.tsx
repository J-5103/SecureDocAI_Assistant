// src/pages/QuickChat.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Plus, Search, Clock, ArrowLeft, Trash2, Edit2, Check, X } from "lucide-react";
import {
  listQuickChats as _listQuickChats,
  getQuickChatMessages as _getQuickChatMessages,
  createQuickChat as _createQuickChat,
  sendQuickChatMessage as _sendQuickChatMessage,
  deleteQuickChat as _deleteQuickChat,
  renameQuickChat as _renameQuickChat,
} from "@/api/quickchat";

// ---------- types ----------
type ChatSummary = { id: string; title: string; createdAt: string };
type Message = { id: string; text: string; sender: "user" | "ai"; at: string };
type LocState = { seedPrompt?: string };

// ---------- helpers ----------
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

// normalize create-id: supports string | {id} | {chatId}
const extractId = (res: any): string => {
  if (!res) throw new Error("Empty response");
  if (typeof res === "string") return res;
  if (res?.id) return String(res.id);
  if (res?.chatId) return String(res.chatId);
  throw new Error("Unable to read new chat id from response");
};

// ---- API wrappers ----------------------------------------------------------
async function createQuickChatSafe(title?: string) {
  try {
    const raw = await _createQuickChat?.(title);
    return extractId(raw);
  } catch (err: any) {
    const status = err?.response?.status;
    if (status === 404 || err?.code === "ERR_BAD_REQUEST") {
      const body: any = title ? { title } : {};
      const res = await fetch("/api/quick-chats", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`Create failed (${res.status}): ${txt || res.statusText}`);
      }
      const data = await res.json();
      return extractId(data);
    }
    throw err;
  }
}

async function listQuickChatsSafe() {
  try {
    const data = (await _listQuickChats?.()) as any;
    return (Array.isArray(data) ? data : data?.items) as ChatSummary[] | undefined;
  } catch (err: any) {
    if (err?.response?.status === 404) {
      const r = await fetch("/api/quick-chats", { method: "GET" });
      if (!r.ok) throw new Error(`List failed (${r.status})`);
      const data = await r.json();
      return (data?.items ?? data) as ChatSummary[];
    }
    throw err;
  }
}

async function getQuickChatMessagesSafe(id: string) {
  try {
    const data = (await _getQuickChatMessages?.(id)) as any;
    return (Array.isArray(data) ? data : data?.items) as Message[] | undefined;
  } catch (err: any) {
    if (err?.response?.status === 404) {
      const r = await fetch(`/api/quick-chats/${id}/messages`);
      if (!r.ok) throw new Error(`Messages failed (${r.status})`);
      const data = await r.json();
      return (data?.items ?? data) as Message[];
    }
    throw err;
  }
}

async function sendQuickChatMessageSafe(id: string, text: string) {
  // NOTE: your api/quickchat.js ignores `text` and only sends chatId
  try {
    const data = (await _sendQuickChatMessage?.(id, text)) as any;
    return typeof data === "string" ? data : data?.reply ?? "";
  } catch (err: any) {
    if (err?.response?.status === 404) {
      const r = await fetch(`/api/quick-chats/${id}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }), // backend tolerates empty, but ok to send
      });
      if (!r.ok) {
        const t = await r.text().catch(() => "");
        throw new Error(`Send failed (${r.status}): ${t || r.statusText}`);
      }
      const data = await r.json();
      return typeof data === "string" ? data : data?.reply ?? "";
    }
    throw err;
  }
}

async function deleteQuickChatSafe(id: string) {
  try {
    return await _deleteQuickChat?.(id);
  } catch (err: any) {
    if (err?.response?.status === 404) {
      const r = await fetch(`/api/quick-chats/${id}`, { method: "DELETE" });
      if (!r.ok) throw new Error(`Delete failed (${r.status})`);
      return;
    }
    throw err;
  }
}

async function renameQuickChatSafe(id: string, title: string) {
  try {
    return await _renameQuickChat?.(id, title);
  } catch (err: any) {
    if (err?.response?.status === 404) {
      const r = await fetch(`/api/quick-chats/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!r.ok) {
        const t = await r.text().catch(() => "");
        throw new Error(`Rename failed (${r.status}): ${t || r.statusText}`);
      }
      return true;
    }
    throw err;
  }
}

// ---------- client-side persistence ----------
const LS_PREFIX = "qc_msgs_v1:";
const lsKey = (id: string) => `${LS_PREFIX}${id}`;

const loadLocal = (id: string): Message[] => {
  try {
    const raw = localStorage.getItem(lsKey(id));
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
};

const saveLocal = (id: string, msgs: Message[]) => {
  try {
    localStorage.setItem(lsKey(id), JSON.stringify(msgs.slice(-500)));
  } catch {}
};

const clearLocal = (id: string) => {
  try {
    localStorage.removeItem(lsKey(id));
  } catch {}
};

const mergeMessages = (server: Message[], local: Message[]) => {
  const map = new Map<string, Message>();
  [...server, ...local].forEach((m) => map.set(m.id, m));
  return [...map.values()].sort(
    (a, b) => new Date(a.at).getTime() - new Date(b.at).getTime()
  );
};

// strongly-typed message creator
const makeMsg = (
  sender: Message["sender"],
  text: string,
  at: string = new Date().toISOString()
): Message => ({
  id: `${sender === "user" ? "u" : "a"}_${Date.now()}`,
  text,
  sender,
  at,
});

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
  const [messages, setMessages] = useState<Message[]>([]);
  const [composer, setComposer] = useState("");

  // rename UI states
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");

  const threadRef = useRef<HTMLDivElement>(null);

  // ===== initial load =====
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const items = (await listQuickChatsSafe()) as ChatSummary[] | undefined;
        if (!alive) return;
        const list = items ?? [];
        setChats(list);

        if (list[0]) {
          const firstId = list[0].id;
          setActiveChatId(firstId);

          // merge server + local and persist AFTER we have the correct thread
          const serverMsgs = (await getQuickChatMessagesSafe(firstId)) ?? [];
          const localMsgs = loadLocal(firstId);
          const merged = mergeMessages(serverMsgs, localMsgs);
          if (!alive) return;
          setMessages(merged);
          saveLocal(firstId, merged); // ✅ important
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
        const newId = await createQuickChatSafe(seedPrompt);
        const createdAt = new Date().toISOString();
        if (!alive) return;

        setChats((prev) => [{ id: newId, title: seedPrompt.slice(0, 64), createdAt }, ...prev]);
        setActiveChatId(newId);

        // show & persist the user's seeded prompt locally
        const seeded = makeMsg("user", seedPrompt, createdAt);
        setMessages([seeded]);
        saveLocal(newId, [seeded]);
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

      const serverMsgs = (await getQuickChatMessagesSafe(id)) ?? [];
      const localMsgs = loadLocal(id);
      const merged = mergeMessages(serverMsgs, localMsgs);
      setMessages(merged);
      saveLocal(id, merged); // ✅ important

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

      const initial: Message[] = [makeMsg("ai", DEFAULT_GREETING, now)];
      setMessages(initial);
      saveLocal(newId, initial);

      setComposer("");
      setEditingTitle(false);
    } catch (e: any) {
      const msg =
        (e?.message as string) ||
        (e?.response?.statusText as string) ||
        "Failed to create a new quick chat.";
      setError(msg);
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
      clearLocal(id);

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
          const serverMsgs = (await getQuickChatMessagesSafe(pick.id)) ?? [];
          const localMsgs = loadLocal(pick.id);
          const merged = mergeMessages(serverMsgs, localMsgs);
          setMessages(merged);
          saveLocal(pick.id, merged);
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
        chatId = await createQuickChatSafe();
        const createdAt = new Date().toISOString();
        setActiveChatId(chatId);
        setChats((prev) => [{ id: chatId, title: "New quick chat", createdAt }, ...prev]);

        const greet = makeMsg("ai", DEFAULT_GREETING, createdAt);
        setMessages([greet]);
        saveLocal(chatId, [greet]);
      }

      // add user message locally and persist
      const userMsg = makeMsg("user", t);
      setMessages((prev) => {
        const next = [...prev, userMsg];
        if (chatId) saveLocal(chatId, next);
        return next;
      });
      setComposer("");

      // call backend (chatId only) and add AI reply locally
      const reply = await sendQuickChatMessageSafe(chatId!, t);
      const aiMsg = makeMsg("ai", reply);
      setMessages((prev) => {
        const next = [...prev, aiMsg];
        if (chatId) saveLocal(chatId, next);
        return next;
      });
    } catch (e) {
      setError("Failed to send message.");
      console.error(e);
    } finally {
      setLoading(false);
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
      // optimistic update list & header
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

  // ===== render =====
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
                      <div className="text-sm font-medium truncate">{c.title || "Untitled chat"}</div>
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

                {/* Title / Rename */}
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
                {loading && <span className="text-xs text-muted-foreground animate-pulse">thinking…</span>}
              </div>
            </div>

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
                <div className="text-sm text-muted-foreground">No messages yet. Ask your first question below.</div>
              )}
              {error && <div className="text-xs text-red-600">{error}</div>}
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

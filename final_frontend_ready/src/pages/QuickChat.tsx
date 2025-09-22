// src/pages/QuickChat.tsx
import React, { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Plus, Search, Clock, ArrowLeft, Trash2, Edit2, Check, X } from "lucide-react";

import MessageList, { type ChatMessage } from "@/components/messagelist";

import {
  listQuickChats as _listQuickChats,
  getQuickChatMessages as _getQuickChatMessages,
  createQuickChat as _createQuickChat,
  sendQuickChatMessage as _sendQuickChatMessage,
  deleteQuickChat as _deleteQuickChat,
  renameQuickChat as _renameQuickChat,
} from "@/api/quickchat";

// ---------- types ----------
type ChatSummary = {
  id: string;
  title: string;
  createdAt: string;
  lastMessageAt?: string;
  isDeleted?: boolean;
};
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
const DEFAULT_GREETING =
  "Hello! I’m ready to help you for your queries.\n\nI can run data queries if you start with `sql:` or ask a question about your tables.";

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
    return (Array.isArray(data) ? data : data?.items) as ChatMessage[] | undefined;
  } catch (err: any) {
    if (err?.response?.status === 404) {
      const r = await fetch(`/api/quick-chats/${id}/messages`);
      if (!r.ok) throw new Error(`Messages failed (${r.status})`);
      const data = await r.json();
      return (data?.items ?? data) as ChatMessage[];
    }
    throw err;
  }
}
async function sendQuickChatMessageSafe(id: string, text: string) {
  try {
    const data = (await _sendQuickChatMessage?.(id, text)) as any;
    return typeof data === "string" ? data : data?.reply ?? "";
  } catch (err: any) {
    if (err?.response?.status === 404) {
      const r = await fetch(`/api/quick-chats/${id}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
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

// ---------- client-side index persistence (list only; messages come from server) ----------
const LS_PREFIX_IDX = "qc_index_v1";
type ChatIndex = Record<string, ChatSummary>;
const loadIndex = (): ChatIndex => {
  try {
    const raw = localStorage.getItem(LS_PREFIX_IDX);
    return raw ? (JSON.parse(raw) as ChatIndex) : {};
  } catch {
    return {};
  }
};
const saveIndex = (idx: ChatIndex) => {
  try {
    localStorage.setItem(LS_PREFIX_IDX, JSON.stringify(idx));
  } catch {}
};
const upsertIndex = (chat: ChatSummary) => {
  const idx = loadIndex();
  const prev = idx[chat.id] || {};
  idx[chat.id] = { ...prev, ...chat, isDeleted: false };
  saveIndex(idx);
};
const markDeletedIndex = (id: string) => {
  const idx = loadIndex();
  if (idx[id]) {
    idx[id].isDeleted = true;
    saveIndex(idx);
  }
};
const removeIndexHard = (id: string) => {
  const idx = loadIndex();
  delete idx[id];
  saveIndex(idx);
};
const mergeChats = (server: ChatSummary[] = []): ChatSummary[] => {
  const localIdx = loadIndex();
  const map = new Map<string, ChatSummary>();
  Object.values(localIdx).forEach((c) => {
    if (!c.isDeleted) map.set(c.id, c);
  });
  server.forEach((c) => {
    const prev = map.get(c.id);
    map.set(c.id, { ...prev, ...c, isDeleted: false });
  });
  const arr = [...map.values()].filter((c) => !c.isDeleted);
  arr.sort((a, b) => {
    const atA = new Date(a.lastMessageAt || a.createdAt).getTime();
    const atB = new Date(b.lastMessageAt || b.createdAt).getTime();
    return atB - atA;
  });
  return arr;
};

// strongly-typed message creator (TEMP ids – not persisted)
const makeTempId = (prefix: string) => `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
const tempUser = (text: string): ChatMessage => ({
  id: makeTempId("temp_u"),
  text,
  sender: "user",
  at: new Date().toISOString(),
});
const tempThinking = (): ChatMessage => ({
  id: makeTempId("temp_th"),
  text: "Thinking…",
  sender: "ai",
  at: new Date().toISOString(),
  thinking: true,
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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [composer, setComposer] = useState("");

  // rename UI states
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");

  // ===== initial load =====
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const serverList = (await listQuickChatsSafe()) ?? [];
        const mergedList = mergeChats(serverList);
        if (!alive) return;

        mergedList.forEach(upsertIndex);
        setChats(mergedList);

        if (mergedList[0]) {
          const firstId = mergedList[0].id;
          setActiveChatId(firstId);
          const serverMsgs = (await getQuickChatMessagesSafe(firstId)) ?? [];
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
    };
  }, []);

  // ===== optional seed prompt: create chat then (optionally) you can auto-send it =====
  useEffect(() => {
    if (!seedPrompt) return;
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        const newId = await createQuickChatSafe(seedPrompt);
        const createdAt = new Date().toISOString();
        const newChat: ChatSummary = { id: newId, title: seedPrompt.slice(0, 64), createdAt };
        upsertIndex(newChat);

        if (!alive) return;
        setChats((prev) => mergeChats([newChat, ...prev]));
        setActiveChatId(newId);

        // Load server messages (should contain the greeting)
        const serverMsgs = (await getQuickChatMessagesSafe(newId)) ?? [];
        if (!alive) return;
        setMessages(serverMsgs);
        // If you want to immediately ask the seed prompt, uncomment:
        // await sendMessage(seedPrompt);
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
      const newChat: ChatSummary = { id: newId, title: "New quick chat", createdAt: now };

      upsertIndex(newChat);
      setChats((prev) => mergeChats([newChat, ...prev]));
      setActiveChatId(newId);

      // Load server messages to display the server-side greeting (avoids duplicates later)
      const serverMsgs = (await getQuickChatMessagesSafe(newId)) ?? [
        { id: "greet_local", text: DEFAULT_GREETING, sender: "ai", at: now },
      ];
      setMessages(serverMsgs);

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
      markDeletedIndex(id);
      await deleteQuickChatSafe(id);

      const remaining = mergeChats(chats.filter((c) => c.id !== id));
      setChats(remaining);

      if (activeChatId === id) {
        const next = remaining[0]?.id ?? null;
        setActiveChatId(next);
        setMessages(next ? (await getQuickChatMessagesSafe(next)) ?? [] : []);
      }
      removeIndexHard(id);
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
        // Create chat on the server and load its greeting to stay in sync
        chatId = await createQuickChatSafe();
        const createdAt = new Date().toISOString();
        const newChat: ChatSummary = { id: chatId, title: "New quick chat", createdAt };
        upsertIndex(newChat);
        setActiveChatId(chatId);
        setChats((prev) => mergeChats([newChat, ...prev]));
        const srv0 = (await getQuickChatMessagesSafe(chatId)) ?? [];
        setMessages(srv0);
      }

      // Show temp user + temp thinking (NOT saved to localStorage)
      const u = tempUser(t);
      const th = tempThinking();
      setMessages((prev) => [...prev, u, th]);
      setComposer("");

      // bump list timestamp so the chat rises
      upsertIndex({
        id: chatId!,
        title: chats.find((c) => c.id === chatId)?.title || "New quick chat",
        createdAt: chats.find((c) => c.id === chatId)?.createdAt || new Date().toISOString(),
        lastMessageAt: new Date().toISOString(),
      });
      setChats((prev) => mergeChats(prev));

      // Call backend
      await sendQuickChatMessageSafe(chatId!, t);

      // After backend replies, reload the canonical messages so IDs match server state
      const serverMsgs = (await getQuickChatMessagesSafe(chatId!)) ?? [];
      setMessages(serverMsgs);

      // bump again after reply
      upsertIndex({
        id: chatId!,
        title: chats.find((c) => c.id === chatId)?.title || "New quick chat",
        createdAt: chats.find((c) => c.id === chatId)?.createdAt || new Date().toISOString(),
        lastMessageAt: new Date().toISOString(),
      });
      setChats((prev) => mergeChats(prev));
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
      setChats((prev) => prev.map((c) => (c.id === activeChatId ? { ...c, title: t } : c)));
      upsertIndex({
        id: activeChatId,
        title: t,
        createdAt: chats.find((c) => c.id === activeChatId)?.createdAt || new Date().toISOString(),
        lastMessageAt: chats.find((c) => c.id === activeChatId)?.lastMessageAt,
      });
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
                        {timeAgo(c.lastMessageAt || c.createdAt)}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      {(c.lastMessageAt || c.createdAt) &&
                        new Date(c.lastMessageAt || c.createdAt).toLocaleString()}
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
                <button onClick={() => navigate(-1)} className="p-2 rounded-lg bg-muted hover:bg-muted/70" title="Back">
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

            {/* Thread */}
            <MessageList
              messages={messages}
              emptyHint="No messages yet. Ask your first question below."
              errorText={error}
            />

            {/* Composer */}
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

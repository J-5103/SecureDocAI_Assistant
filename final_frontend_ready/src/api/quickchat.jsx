// src/api/quickchat.js
import axios from "axios";

/**
 * Toggle mock mode while wiring the backend:
 *  - true  => returns local fake data (no server needed)
 *  - false => uses your real API endpoints below
 */
const LOCAL_MOCK = false;

// Optional default greeting text (used only if you seed a greeting)
export const DEFAULT_GREETING =
  "Hello! I’m ready to help you for your queries.";

// -------------------- AXIOS CLIENT --------------------
const normalize = (u) => (u ? String(u).replace(/\/+$/, "") : "");
const isDev = !!import.meta.env?.DEV;

/**
 * In dev we keep baseURL empty so requests go to `/api/...`
 * and Vite's proxy forwards to your FastAPI on :8000.
 * In prod builds you can set VITE_API_BASE or VITE_BACKEND.
 */
const API_BASE = isDev
  ? ""
  : normalize(import.meta.env?.VITE_API_BASE) ||
    normalize(import.meta.env?.VITE_BACKEND) ||
    "";

export const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json", Accept: "application/json" },
  // timeout: 20000, // optional
});

// ----- low level helpers: retry with fetch() on CORS/network -----
async function getJSON(path) {
  try {
    const { data } = await api.get(path);
    return data;
  } catch (err) {
    if (!err?.response) {
      const r = await fetch(path, { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error(`GET ${path} failed: ${r.status}`);
      return await r.json();
    }
    console.error("[quickchat.getJSON] error", path, err?.response?.status, err?.response?.data);
    throw err;
  }
}

async function postJSON(path, body) {
  try {
    const { data } = await api.post(path, body);
    return data;
  } catch (err) {
    if (!err?.response) {
      const r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body ?? {}),
      });
      if (!r.ok) {
        const t = await r.text().catch(() => "");
        throw new Error(`POST ${path} failed: ${r.status} ${t}`);
      }
      try {
        return await r.json();
      } catch {
        return {};
      }
    }
    console.error("[quickchat.postJSON] error", path, err?.response?.status, err?.response?.data);
    throw err;
  }
}

async function patchJSON(path, body) {
  try {
    const { data } = await api.patch(path, body);
    return data;
  } catch (err) {
    if (!err?.response) {
      const r = await fetch(path, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body ?? {}),
      });
      if (!r.ok) {
        const t = await r.text().catch(() => "");
        throw new Error(`PATCH ${path} failed: ${r.status} ${t}`);
      }
      return await r.json();
    }
    console.error("[quickchat.patchJSON] error", path, err?.response?.status, err?.response?.data);
    throw err;
  }
}

async function deleteJSON(path) {
  try {
    const { data } = await api.delete(path);
    return data;
  } catch (err) {
    if (!err?.response) {
      const r = await fetch(path, { method: "DELETE", headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error(`DELETE ${path} failed: ${r.status}`);
      try {
        return await r.json();
      } catch {
        return { ok: true };
      }
    }
    console.error("[quickchat.deleteJSON] error", path, err?.response?.status, err?.response?.data);
    throw err;
  }
}

// Retry helper (also supports legacy fallback)
async function tryOrFallback(mainFn, fallbackFn) {
  try {
    return await mainFn();
  } catch (err) {
    const status = err?.response?.status;
    // 404/405 route shape mismatch, 422 strict validator → try legacy
    if ((status === 404 || status === 405 || status === 422) && typeof fallbackFn === "function") {
      return await fallbackFn();
    }
    throw err;
  }
}

// -------------------- MOCKS (optional) --------------------
const mockChats = [
  { id: "c2", title: "Banks with STD code 079", createdAt: "2025-09-17T09:00:00Z" },
  { id: "c1", title: "Top branches by volume", createdAt: "2025-09-16T10:00:00Z" },
];

const mockMsgs = {
  c2: [
    { id: "m1", text: "Show banks with STD code 079", sender: "user", at: "2025-09-17T09:01:00Z" },
    { id: "m2", text: "There are 8 banks with STD code 079.", sender: "ai", at: "2025-09-17T09:01:05Z" },
  ],
  c1: [
    { id: "m3", text: "Top branches by volume?", sender: "user", at: "2025-09-16T10:02:00Z" },
    { id: "m4", text: "Here are the top 5…", sender: "ai", at: "2025-09-16T10:02:04Z" },
  ],
};
// --------------------------------------------------------

// -------------------- API FUNCTIONS --------------------

/** List all quick chats (newest first) */
export const listQuickChats = async () => {
  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 150));
    return [...mockChats].sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }

  return tryOrFallback(
    async () => {
      const data = await getJSON("/api/quick-chats");
      return data?.items ?? data ?? [];
    },
    async () => {
      const data = await getJSON("/api/quickchat/list");
      return data?.items ?? data ?? [];
    }
  );
};

/** Get chat meta (id, title, createdAt) */
export const getQuickChatMeta = async (chatId) => {
  if (!chatId) return null;

  if (LOCAL_MOCK) {
    const found = mockChats.find((c) => c.id === chatId);
    return found ? { ...found } : null;
  }

  const id = encodeURIComponent(chatId);
  return tryOrFallback(
    async () => await getJSON(`/api/quick-chats/${id}`),
    async () => await getJSON(`/api/quickchat/${id}/meta`)
  );
};

/** Get messages for a chat */
export const getQuickChatMessages = async (chatId) => {
  if (!chatId) return [];
  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 120));
    return mockMsgs[chatId] ?? [];
  }

  const id = encodeURIComponent(chatId);
  return tryOrFallback(
    async () => {
      const data = await getJSON(`/api/quick-chats/${id}/messages`);
      return data?.items ?? data ?? [];
    },
    async () => {
      const data = await getJSON(`/api/quickchat/${id}`);
      return data?.items ?? data ?? [];
    }
  );
};

/**
 * Create a new quick chat.
 * Params:
 *  - seedPrompt?: string
 *  - seedGreeting?: boolean
 *  - greetingText?: string
 * Return: string (new chat id)
 */
export const createQuickChat = async (
  seedPrompt,
  seedGreeting = false,
  greetingText = DEFAULT_GREETING
) => {
  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 120));
    const id = `c_${Date.now()}`;
    const createdAt = new Date().toISOString();

    mockChats.unshift({
      id,
      title: (seedPrompt || "New quick chat").slice(0, 64),
      createdAt,
    });

    mockMsgs[id] = mockMsgs[id] || [];
    if (seedGreeting && greetingText) {
      mockMsgs[id].push({
        id: `a_${Date.now()}`,
        text: greetingText,
        sender: "ai",
        at: createdAt,
      });
    }
    if (seedPrompt) {
      mockMsgs[id].push({
        id: `u_${Date.now() + 1}`,
        text: seedPrompt,
        sender: "user",
        at: createdAt,
      });
      mockMsgs[id].push({
        id: `a_${Date.now() + 2}`,
        text: "Mock reply to your seed prompt.",
        sender: "ai",
        at: createdAt,
      });
    }
    return id;
  }

  const body = { seedPrompt, seedGreeting, greetingText };

  const data = await tryOrFallback(
    async () => await postJSON("/api/quick-chats", body),
    async () => await postJSON("/api/quickchat", body)
  );

  if (!data) return undefined;
  if (typeof data === "string") return data;
  if (typeof data === "object" && data.id) return data.id;
  return data?.id ?? undefined;
};

/** Rename a quick chat (title) */
export const renameQuickChat = async (chatId, title) => {
  const t = (title || "").trim();
  if (!chatId || !t) return false;

  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 80));
    const i = mockChats.findIndex((c) => c.id === chatId);
    if (i >= 0) mockChats[i] = { ...mockChats[i], title: t };
    return true;
  }

  const id = encodeURIComponent(chatId);
  const data = await tryOrFallback(
    async () => await patchJSON(`/api/quick-chats/${id}`, { title: t }),
    async () => await patchJSON(`/api/quickchat/${id}`, { title: t })
  );
  return data?.ok === true;
};

/**
 * Send a user message so the backend can run Text-to-SQL.
 * Preferred route:  POST /api/quick-chats/:chatId/messages  { text }
 * Legacy fallback:  POST /api/quickchat/:chatId             { text }
 * Returns the reply string.
 */
export const sendQuickChatMessage = async (chatId, text) => {
  if (!chatId) return "";
  const t = String(text ?? "").trim();
  if (!t) return "";

  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 250));
    const at = new Date().toISOString();
    (mockMsgs[chatId] = mockMsgs[chatId] || []).push({
      id: `a_${Date.now()}`,
      text: "Mock reply",
      sender: "ai",
      at,
    });
    return "Mock reply";
  }

  const id = encodeURIComponent(chatId);

  const data = await tryOrFallback(
    // NEW: hit /messages with the user's text
    async () => await postJSON(`/api/quick-chats/${id}/messages`, { text: t }),
    // FALLBACK: older singular route some builds used
    async () => await postJSON(`/api/quickchat/${id}`, { text: t })
  );

  return typeof data === "string"
    ? data
    : data?.reply ?? data?.message ?? data?.text ?? "";
};

/** Delete a quick chat */
export const deleteQuickChat = async (chatId) => {
  if (!chatId) return false;
  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 120));
    const idx = mockChats.findIndex((c) => c.id === chatId);
    if (idx >= 0) mockChats.splice(idx, 1);
    delete mockMsgs[chatId];
    return true;
  }

  const id = encodeURIComponent(chatId);
  const data = await tryOrFallback(
    async () => await deleteJSON(`/api/quick-chats/${id}`),
    async () => await deleteJSON(`/api/quickchat/${id}`)
  );

  return data?.ok ?? true;
};

// src/api/quickchat.js
import axios from "axios";

/**
 * Toggle mock mode easily while wiring the backend:
 * - true  => returns local fake data (no server needed)
 * - false => uses your real API endpoints below
 */
const LOCAL_MOCK = false;

// Optional default greeting text (used only if you seed a greeting)
export const DEFAULT_GREETING =
  "Hello! I’m ready to help. Ask anything—counts, lists, or quick insights.";

// ---------- MOCKS (optional) ----------
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
// -------------------------------------

/**
 * List all quick chats (newest first)
 * Return: Array<{ id, title, createdAt }>
 */
export const listQuickChats = async () => {
  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 150));
    return [...mockChats].sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }
  const { data } = await axios.get("/api/quickchat/list");
  return data?.items ?? [];
};

/**
 * Get messages for a chat
 * Return: Array<{ id, text, sender: 'user'|'ai', at }>
 */
export const getQuickChatMessages = async (chatId) => {
  if (!chatId) return [];
  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 120));
    return mockMsgs[chatId] ?? [];
  }
  const { data } = await axios.get(`/api/quickchat/${encodeURIComponent(chatId)}`);
  return data?.items ?? [];
};

/**
 * Create a new quick chat.
 * Params:
 *  - seedPrompt?: string – optional first user question
 *  - seedGreeting?: boolean – if true, seed an initial assistant greeting (mock only unless backend supports it)
 *  - greetingText?: string – optional custom greeting (defaults to DEFAULT_GREETING)
 *
 * Return: string (new chat id)
 */
export const createQuickChat = async (seedPrompt, seedGreeting = false, greetingText = DEFAULT_GREETING) => {
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

  // Real backend: pass the optional flags if your server supports them
  const { data } = await axios.post("/api/quickchat", {
    seedPrompt,
    seedGreeting,
    greetingText,
  });
  return data?.id;
};

/**
 * Send a message to a chat and get the AI reply
 * Return: string (AI reply text)
 */
export const sendQuickChatMessage = async (chatId, text) => {
  if (!chatId || !text?.trim()) return "";
  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 250));
    const at = new Date().toISOString();
    (mockMsgs[chatId] = mockMsgs[chatId] || []).push({
      id: `u_${Date.now()}`,
      text,
      sender: "user",
      at,
    });
    const reply = `Demo answer for: ${text}`;
    mockMsgs[chatId].push({
      id: `a_${Date.now() + 1}`,
      text: reply,
      sender: "ai",
      at,
    });
    return reply;
  }
  const { data } = await axios.post(`/api/quickchat/${encodeURIComponent(chatId)}/message`, { text });
  return data?.reply ?? "";
};

/**
 * Delete a quick chat
 * Return: boolean (ok)
 */
export const deleteQuickChat = async (chatId) => {
  if (!chatId) return false;
  if (LOCAL_MOCK) {
    await new Promise((r) => setTimeout(r, 120));
    const idx = mockChats.findIndex((c) => c.id === chatId);
    if (idx >= 0) mockChats.splice(idx, 1);
    delete mockMsgs[chatId];
    return true;
  }
  const { data } = await axios.delete(`/api/quickchat/${encodeURIComponent(chatId)}`);
  return data?.ok ?? true;
};

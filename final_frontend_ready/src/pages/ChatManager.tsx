// src/pages/ChatManager.tsx
import React, { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Plus, MessageCircle, ArrowLeft, Trash2 } from "lucide-react";
import { Header } from "@/components/Header";
import { useToast } from "@/hooks/use-toast";

import { DocumentSidebar } from "@/components/DocumentSidebar";
import { ChatInterface } from "@/components/ChatInterface";
import type { Document as Doc } from "@/components/DocumentSidebar";

import {
  askDocs,          // ✅ docs/vectorstore endpoint (always /api/ask)
  uploadDocument,
  chatUploadImages,
  extractBusinessCard, // ✅ SAME extractor as Viz chats
} from "../api/api";

type Msg = {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  imageUrl?: string;         // legacy (first image)
  imageUrls?: string[];      // ✅ multi-image thumbnails
  imageAlt?: string;
  status?: "Thinking" | "ok" | "error";
};

export interface ChatSession {
  id: string;
  name: string;
  createdAt: string;
  lastMessage: string;
  messageCount: number;
  messages: Msg[];
  documentName?: string | null;
}

const CHAT_STORAGE = "chatSessions";
const docsKeyFor = (id: string) => `documents:${id}`;
const imgMapKeyFor = (id: string) => `imageDocUrlMap:${id}`;
const imgAnsCacheKeyFor = (id: string) => `imageAnswerCache:${id}`;

const genId = () =>
  Date.now().toString() + Math.random().toString(36).substring(2, 9);

// ---- helpers
const IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".gif"];
const isImageName = (name = "") =>
  IMG_EXTS.some((ext) => name.toLowerCase().endsWith(ext));

function getDocType(filename: string): Doc["type"] {
  const f = filename.toLowerCase();
  if (f.endsWith(".pdf")) return "pdf";
  return "other";
}

/** Merge & de-dupe by documentId||name */
function mergeDocs(chatId: string, incoming: Doc[]): Doc[] {
  const key = docsKeyFor(chatId);
  const existing: Doc[] = JSON.parse(localStorage.getItem(key) || "[]");
  const byKey = new Map<string, Doc>();
  [...existing, ...incoming].forEach((d) => {
    const k = (d.documentId || d.name).toLowerCase();
    byKey.set(k, d);
  });
  const merged = Array.from(byKey.values());
  localStorage.setItem(key, JSON.stringify(merged));
  return merged;
}

/* ---------- sanitize saved sessions (strip blob:/data: URLs) ---------- */
const stripHeavyUrl = (url?: string) =>
  url && (url.startsWith("blob:") || url.startsWith("data:")) ? "" : (url || "");

function sanitizeSessions(sessions: ChatSession[]): ChatSession[] {
  return sessions.map((cs) => ({
    ...cs,
    messages: (cs.messages || []).map((m) => {
      const mm: Msg = { ...m };
      if (typeof mm.imageUrl === "string") mm.imageUrl = stripHeavyUrl(mm.imageUrl);
      if (Array.isArray(mm.imageUrls))
        mm.imageUrls = mm.imageUrls.map((u) => stripHeavyUrl(u)).filter(Boolean);
      return mm;
    }),
  }));
}

/* ---------- safe saver ---------- */
const safeSaveSessions = (chs: ChatSession[]) => {
  try {
    const cleaned = sanitizeSessions(chs);
    localStorage.setItem(CHAT_STORAGE, JSON.stringify(cleaned));
  } catch {}
};

// Turn a saved URL into a File (for extractor)
async function urlToFile(
  url: string,
  fallbackName = "image_from_doc.png"
): Promise<File> {
  const res = await fetch(url, { credentials: "omit" });
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.status}`);
  const blob = await res.blob();
  let name = fallbackName;
  try {
    const u = new URL(url);
    const last = (u.pathname.split("/").pop() || "").trim();
    if (last) name = last;
  } catch {}
  const type = blob.type || "image/png";
  return new File([blob], name, { type });
}

/* ---------- strict business-card prompt (kept for parity) ---------- */
const buildCardPrompt = (userText: string) => `
You are an OCR + contact extractor for business cards.
Return only what the card prints; no guessing. If a field is missing, omit it.
User request: ${userText || "Extract contact details from this card"}
`.trim();

/* ---------- normalize extractor response -> items[] ---------- */
type CardItem = {
  first_name?: string;
  last_name?: string;
  // name?: string; // (optional if your backend returns a single "name")
  organization?: string;
  job_title?: string;
  phones?: string[];
  emails?: string[];
  websites?: string[];
  address?: {
    street?: string;
    city?: string;
    state?: string;
    postal_code?: string;
    country?: string;
  };
  source_url?: string;
};

function toArray<T>(x: T | T[] | undefined | null): T[] {
  if (!x) return [];
  return Array.isArray(x) ? x : [x];
}

function normalizeToItems(r: any): CardItem[] {
  // common shapes handled: { items: [...] }, { card: {...} }, { data: { card/items/... } }
  const data = r?.data ?? r;
  const items = toArray<CardItem>(data?.items) as CardItem[];
  if (items.length) return items;
  const card = data?.card;
  if (card && typeof card === "object") return [card as CardItem];
  // sometimes backend may return {cards:[...]} or {result:[...]}
  const cards = toArray<CardItem>(data?.cards) || toArray<CardItem>(data?.result);
  if (cards.length) return cards as CardItem[];
  return [];
}

/* ========= NEW: smart merge when it's the same person ========= */
const norm = (s?: string) => (s || "").toLowerCase().replace(/\s+/g, " ").trim();

const fullNameOf = (it: CardItem) => {
  const fn = (it.first_name || "").trim();
  const ln = (it.last_name || "").trim();
  const raw = (fn || ln) ? `${fn} ${ln}`.trim() : "";
  return { raw, key: norm(raw) };
};

const pickLongest = (a?: string, b?: string) => {
  const aa = (a || "").trim();
  const bb = (b || "").trim();
  if (!aa) return bb;
  if (!bb) return aa;
  return bb.length > aa.length ? bb : aa;
};

const mergeArrayDedup = (...arrs: (string[] | undefined)[]) => {
  const out: string[] = [];
  const seen = new Set<string>();
  arrs.flat().filter(Boolean).forEach((x) => {
    String(x)
      .split(",")
      .map((v) => v.trim())
      .forEach((v) => {
        const key = norm(v);
        if (v && !seen.has(key)) {
          seen.add(key);
          out.push(v);
        }
      });
  });
  return out;
};

const mergeAddress = (a?: CardItem["address"], b?: CardItem["address"]) => {
  const aa = a || {};
  const bb = b || {};
  return {
    street: pickLongest(aa.street, bb.street),
    city: pickLongest(aa.city, bb.city),
    state: pickLongest(aa.state, bb.state),
    postal_code: pickLongest(aa.postal_code, bb.postal_code),
    country: pickLongest(aa.country, bb.country),
  };
};

/** Group by normalized full name. If zero/one unique name -> merge into ONE contact. */
function smartMergeContacts(items: CardItem[]): CardItem[] {
  const groups = new Map<string, CardItem[]>();
  const uniqueNameKeys = new Set<string>();

  items.forEach((it) => {
    const { key } = fullNameOf(it);
    const k = key || "__noname__";
    uniqueNameKeys.add(key);
    groups.set(k, [...(groups.get(k) || []), it]);
  });

  const uniqueNamed = Array.from(uniqueNameKeys).filter(Boolean);
  const mergeAll = uniqueNamed.length <= 1;

  const mergeGroup = (arr: CardItem[]): CardItem =>
    arr.reduce<CardItem>((acc: any, cur) => {
      const accNm = fullNameOf(acc).raw.split(/\s+/);
      const curNm = fullNameOf(cur).raw.split(/\s+/);
      const [afn, aln] = [accNm[0] || "", accNm.slice(1).join(" ") || ""];
      const [fn, ln] = [curNm[0] || "", curNm.slice(1).join(" ") || ""];

      return {
        first_name: pickLongest(afn, fn),
        last_name: pickLongest(aln, ln),
        organization: pickLongest(acc.organization, cur.organization),
        job_title: pickLongest(acc.job_title, cur.job_title),
        phones: mergeArrayDedup(acc.phones, cur.phones),
        emails: mergeArrayDedup(acc.emails, cur.emails),
        websites: mergeArrayDedup(acc.websites, cur.websites),
        address: mergeAddress(acc.address, cur.address),
        source_url: acc.source_url || cur.source_url,
      };
    }, {} as CardItem);

  if (mergeAll) return [mergeGroup(items)];

  const out: CardItem[] = [];
  for (const [, arr] of groups.entries()) out.push(mergeGroup(arr));
  return out.sort((a, b) => norm(fullNameOf(a).raw).localeCompare(norm(fullNameOf(b).raw)));
}

/* ---------- VizChat-style pretty formatter (no raw JSON)
 * Shows "Contact 1/2..." ONLY when >1 contacts after merge ----------
 */
function formatContactsViz(items: CardItem[] = [], meta: { totalImages?: number } = {}): string {
  const { totalImages = 0 } = meta;
  if (!items.length) return `No details found from ${totalImages} image${totalImages > 1 ? "s" : ""}.`;

  const single = items.length === 1;

  const blockFor = (it: CardItem, idx: number) => {
    const L: string[] = [];
    if (!single) {
      L.push(`Contact ${idx + 1}`);
      L.push("───────────────");
    }

    const fullName = `${it.first_name || ""} ${it.last_name || ""}`.trim();
    if (fullName) L.push(`Name       : ${fullName}`);
    if (it.organization) L.push(`Company    : ${it.organization}`);
    if (it.job_title)    L.push(`Title      : ${it.job_title}`);

    if (Array.isArray(it.phones) && it.phones.length)   L.push(`Phones     : ${it.phones.join(", ")}`);
    if (Array.isArray(it.emails) && it.emails.length)   L.push(`Emails     : ${it.emails.join(", ")}`);
    if (Array.isArray(it.websites) && it.websites.length) L.push(`Websites   : ${it.websites.join(", ")}`);

    const a = it.address || {};
    const addr = [a.street, a.city, a.state, a.postal_code, a.country].filter(Boolean).join(", ");
    if (addr) L.push(`Address    : ${addr}`);

    if (it.source_url && !single) L.push(`Source     : ${it.source_url}`);

    return L.join("\n");
  };

  const header = `Extracted contacts from ${totalImages} image${totalImages > 1 ? "s" : ""}:`;
  if (single) return [header, "", blockFor(items[0], 0)].join("\n");
  return [header, "", ...items.map(blockFor)].join("\n");
}

/* ---------- pick display text (fallback) ---------- */
const pickExtractorText = (r: any) => {
  const whats = r?.whatsapp || r?.data?.whatsapp;
  if (typeof whats === "string" && whats.trim()) return whats.trim();
  const card = r?.card || r?.data?.card;
  if (card && typeof card === "object") return JSON.stringify(card, null, 2);
  const txt = r?.text || r?.answer || r?.data?.text || "";
  return String(txt || "").trim();
};

export const ChatManager = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [documents, setDocuments] = useState<Doc[]>([]);
  const [selectedChat, setSelectedChat] = useState<ChatSession | null>(null);

  // sidebar state (for highlight only)
  const [selectedDocId, setSelectedDocId] = useState<string | undefined>(undefined);
  const [selectedCombineDocs, setSelectedCombineDocs] = useState<string[]>([]);

  const [showNewChatDialog, setShowNewChatDialog] = useState(false);
  const [newChatName, setNewChatName] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  const [isAsking, setIsAsking] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [docsHydrated, setDocsHydrated] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  // persistent map for image-type docs to their server URLs
  const [imageDocUrlMap, setImageDocUrlMap] = useState<Record<string, string>>({});

  // stable ref for selectedChat to avoid stale closures in async
  const selectedChatRef = useRef<ChatSession | null>(null);
  useEffect(() => {
    selectedChatRef.current = selectedChat;
  }, [selectedChat]);

  /* ---- image-answer cache (same image => same answer forever) ---- */
  const readImgAnsCache = (): Record<string, string> => {
    if (!chatId) return {};
    try {
      return JSON.parse(localStorage.getItem(imgAnsCacheKeyFor(chatId)) || "{}");
    } catch {
      return {};
    }
  };
  const writeImgAnsCache = (m: Record<string, string>) => {
    if (!chatId) return;
    localStorage.setItem(imgAnsCacheKeyFor(chatId), JSON.stringify(m));
  };
  const getCachedAnswer = (url?: string) => {
    if (!url || !chatId) return undefined;
    const m = readImgAnsCache();
    return m[url];
  };
  const setCachedAnswer = (url?: string, text?: string) => {
    if (!url || !chatId || !text) return;
    const m = readImgAnsCache();
    if (!m[url]) {
      m[url] = text;
      writeImgAnsCache(m);
    }
  };

  /* ---- hydrate sessions ---- */
  useEffect(() => {
    try {
      const saved = localStorage.getItem(CHAT_STORAGE);
      const parsed: ChatSession[] = saved ? JSON.parse(saved) : [];
      const fixed = sanitizeSessions(parsed);
      setChatSessions(fixed);
    } catch {
      setChatSessions([]);
    } finally {
      setHydrated(true);
    }
  }, []);

  useEffect(() => {
    safeSaveSessions(chatSessions);
  }, [chatSessions]);

  /* ---- select chat & docs ---- */
  useEffect(() => {
    if (!chatId) {
      setSelectedChat(null);
      setDocuments([]);
      setSelectedDocId(undefined);
      setSelectedCombineDocs([]);
      setDocsHydrated(false);
      setImageDocUrlMap({});
      return;
    }

    const chat =
      selectedChat?.id === chatId
        ? selectedChat
        : chatSessions.find((c) => c.id === chatId) || null;

    setSelectedChat(chat || null);

    try {
      const saved = localStorage.getItem(docsKeyFor(chatId));
      const parsed: Doc[] = saved ? JSON.parse(saved) : [];
      setDocuments(Array.isArray(parsed) ? parsed : []);
      if (parsed.length > 0) setSelectedDocId(parsed[0].documentId);
    } catch {
      setDocuments([]);
      setSelectedDocId(undefined);
    }
    setDocsHydrated(true);

    // hydrate imageDocUrlMap
    try {
      const m = localStorage.getItem(imgMapKeyFor(chatId));
      setImageDocUrlMap(m ? JSON.parse(m) : {});
    } catch {
      setImageDocUrlMap({});
    }

    if (hydrated && !chat) {
      setSelectedChat(null);
      navigate("/chats", { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, chatSessions, hydrated]);

  useEffect(() => {
    if (!chatId || !docsHydrated) return;
    localStorage.setItem(docsKeyFor(chatId), JSON.stringify(documents));
  }, [chatId, documents, docsHydrated]);

  const saveImageDocUrl = (docId: string, url: string) => {
    if (!chatId || !docId || !url) return;
    setImageDocUrlMap((prev) => {
      const next = { ...prev, [docId]: url };
      localStorage.setItem(imgMapKeyFor(chatId), JSON.stringify(next));
      return next;
    });
  };

  /* ---- uploads ---- */
  const uploadOneFile = async (
    file: File,
    targetChatId: string,
    indexHint = 0
  ): Promise<Doc> => {
    const fallbackId = `${Date.now()}-${indexHint}-${file.name}`;

    // Images → save in chat uploads to get a persistent URL
    if (isImageName(file.name)) {
      const up = await chatUploadImages({ chatId: targetChatId, files: [file] });
      const url =
        (up?.attachments?.[0]?.url as string) ||
        (up?.image_urls?.[0] as string) ||
        "";
      const docId = fallbackId; // our local id
      if (url) saveImageDocUrl(docId, url);

      return {
        id: docId,
        name: file.name,
        type: getDocType(file.name),
        size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
        uploadDate: new Date().toISOString(),
        status: "ready",
        chatId: targetChatId,
        documentId: docId,
      } as any;
    }

    // Non-image → vectorstore pipeline
    const formData = new FormData();
    formData.append("file", file);
    formData.append("chat_id", targetChatId);
    const res = await uploadDocument(formData);

    return {
      id: (res as any)?.document_id || fallbackId,
      name: file.name,
      type: getDocType(file.name),
      size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
      uploadDate: new Date().toISOString(),
      status: (res as any)?.status || "ready",
      chatId: targetChatId,
      documentId: (res as any)?.document_id || file.name,
    } as any;
  };

  const handleDocumentUpload = async (files: FileList): Promise<void> => {
    if (!chatId) {
      const { toast: t } = useToast();
      t({ title: "Error", description: "Chat ID is missing." });
      return;
    }
    setIsUploading(true);
    const uploaded: Doc[] = [];
    for (const [i, file] of Array.from(files).entries()) {
      try {
        const doc = await uploadOneFile(file, chatId, i);
        uploaded.push(doc);
      } catch (error: any) {
        const { toast: t } = useToast();
        t({
          title: "Upload Failed",
          description: error?.message || `Error uploading ${file.name}`,
          variant: "destructive",
        });
      }
    }
    if (uploaded.length) {
      const merged = mergeDocs(chatId, uploaded);
      setDocuments(merged);
      setSelectedDocId(merged[0]?.documentId);
      toast({
        title: "Upload Successful",
        description: `${uploaded.length} file(s) uploaded.`,
      });
    }
    setIsUploading(false);
  };

  /* ---- create new chat ---- */
  const createNewChat = async () => {
    if (!uploadedFiles.length || !newChatName.trim()) {
      toast({
        title: "Error",
        description: "Chat name and at least one document are required.",
      });
      return;
    }

    const newChatId = genId();
    const newDocs: Doc[] = [];
    setIsUploading(true);

    for (const [i, file] of uploadedFiles.entries()) {
      try {
        const doc = await uploadOneFile(file, newChatId, i);
        newDocs.push(doc);
      } catch (err: any) {
        toast({
          title: "Upload Error",
          description: err?.message || `Failed to upload ${file.name}`,
          variant: "destructive",
        });
      }
    }

    if (!newDocs.length) {
      toast({
        title: "Upload Error",
        description: "No files were uploaded successfully.",
        variant: "destructive",
      });
      setIsUploading(false);
      return;
    }

    const greeting = `Hello! I'm ready to help you analyze ${uploadedFiles
      .map((f) => `"${f.name}"`)
      .join(", ")}. What would you like to know?`;

    const newChat: ChatSession = {
      id: newChatId,
      name: newChatName,
      createdAt: new Date().toISOString(),
      lastMessage: "",
      messageCount: 0,
      messages: [
        {
          id: "1",
          text: greeting,
          sender: "ai",
          timestamp: new Date().toISOString(),
        },
      ],
    };

    setChatSessions((prev) => [...prev, newChat]);
    const merged = mergeDocs(newChatId, newDocs);
    setSelectedChat(newChat);
    setDocuments(merged);
    setSelectedDocId(merged[0]?.documentId);
    setDocsHydrated(true);

    setShowNewChatDialog(false);
    setUploadedFiles([]);
    setNewChatName("");
    setIsUploading(false);

    navigate(`/chat/${newChatId}`);
    toast({
      title: "Chat Created",
      description: `New chat with ${merged.length} document(s) created.`,
    });
  };

  const deleteChat = (id: string) => {
    setChatSessions((prev) => prev.filter((c) => c.id !== id));
    localStorage.removeItem(docsKeyFor(id));
    localStorage.removeItem(imgMapKeyFor(id));
    localStorage.removeItem(imgAnsCacheKeyFor(id));
    if (selectedChat?.id === id) {
      setSelectedChat(null);
      setDocuments([]);
      setSelectedDocId(undefined);
      setSelectedCombineDocs([]);
      navigate("/chats");
    }
  };

  /* ---- send message ---- */
  const handleSendMessage = async (
    text: string,
    documentId?: string,
    combineDocs?: string[],
    images?: File[]
  ) => {
    if (!selectedChat || isAsking) return;
    setIsAsking(true);

    const hasImages = Array.isArray(images) && images.length > 0;

    // ----- Push user message with ALL local thumbnails (multi-image) -----
    const blobUrls: string[] =
      hasImages && typeof URL !== "undefined"
        ? images!.map((f) => URL.createObjectURL(f))
        : [];

    const userMsgId = genId();
    const userMessage: Msg = {
      id: userMsgId,
      text,
      sender: "user",
      timestamp: new Date().toISOString(),
      ...(blobUrls.length
        ? { imageUrls: blobUrls, imageUrl: blobUrls[0], imageAlt: "attachment(s)" }
        : {}),
    };

    const thinkingId = genId();
    const Thinking: Msg = {
      id: thinkingId,
      text: "Thinking…",
      sender: "ai",
      timestamp: new Date().toISOString(),
      status: "Thinking",
    };

    let updatedChat: ChatSession = {
      ...selectedChat,
      messages: [...selectedChat.messages, userMessage, Thinking],
      lastMessage: text,
      messageCount: selectedChat.messageCount + 1,
    };
    setChatSessions((prev) =>
      prev.map((c) => (c.id === updatedChat.id ? updatedChat : c))
    );
    setSelectedChat(updatedChat);

    // timeout fallback so "Thinking…" never gets stuck
    const timeoutId = window.setTimeout(() => {
      setChatSessions((prev) =>
        prev.map((c) =>
          c.id === updatedChat.id
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === thinkingId
                    ? { ...m, text: "Processing complete. (fallback)", status: "ok" }
                    : m
                ),
              }
            : c
        )
      );
    }, 15000);

    const replaceThinking = (next: Partial<Msg>) => {
      window.clearTimeout(timeoutId);
      const final = updatedChat.messages.map((m) =>
        m.id === thinkingId ? { ...m, ...next, status: next.status ?? "ok" } : m
      );
      const finalChat: ChatSession = {
        ...updatedChat,
        messages: final,
        lastMessage: (next.text || updatedChat.lastMessage || "").slice(0, 100),
        messageCount: updatedChat.messageCount + 1,
      };
      setChatSessions((prev) =>
        prev.map((c) => (c.id === finalChat.id ? finalChat : c))
      );
      setSelectedChat(finalChat);
      updatedChat = finalChat;
    };

    const patchUserImages = (urls: string[]) => {
      const msgs = updatedChat.messages.map((m) =>
        m.id === userMsgId ? { ...m, imageUrls: urls, imageUrl: urls[0] } : m
      );
      const finalChat = { ...updatedChat, messages: msgs };
      setChatSessions((prev) =>
        prev.map((c) => (c.id === finalChat.id ? finalChat : c))
      );
      setSelectedChat(finalChat);
      updatedChat = finalChat;
    };

    try {
      /** A) Vision / Image flow — now supports MULTIPLE images */
      if (hasImages) {
        // 1) Upload all images (server should return attachments[] or image_urls[])
        let serverUrls: string[] = [];
        try {
          const up = await chatUploadImages({
            chatId: selectedChat.id,
            text,
            files: images!, // <-- MULTIPLE files
          });
          const att = Array.isArray(up?.attachments) ? up.attachments : [];
          const urlsA = att.map((a: any) => a?.url).filter(Boolean);
          const urlsB = Array.isArray(up?.image_urls) ? up.image_urls : [];
          serverUrls = (urlsA.length ? urlsA : urlsB).filter(Boolean);
          if (serverUrls.length) patchUserImages(serverUrls);
        } catch {
          // best-effort; keep blob thumbnails if upload preview fails
        } finally {
          // clean local blob memory
          if (blobUrls.length)
            setTimeout(() => blobUrls.forEach((u) => URL.revokeObjectURL(u)), 15000);
        }

        // 2) Try cache per image (if we have server URLs)
        const allItems: CardItem[] = [];
        const perImageBlocksFromCache: string[] = [];
        const needsExtraction: number[] = [];

        if (serverUrls.length === images!.length) {
          serverUrls.forEach((u, idx) => {
            const cached = getCachedAnswer(u);
            if (cached) perImageBlocksFromCache.push(cached);
            else needsExtraction.push(idx);
          });
        } else {
          // If we don't have corresponding URLs, just extract all
          needsExtraction.push(...images!.map((_, i) => i));
        }

        // 3) Extract for images that are not cached
        for (const idx of needsExtraction) {
          const file = images![idx];
          const r = await extractBusinessCard({
            file,
            returnVcard: true,
            prompt: buildCardPrompt(text),
          });
          const items = normalizeToItems(r);
          if (items.length) {
            allItems.push(...items);
            // store per-image formatted text in cache (if server url is known)
            const u = serverUrls[idx];
            if (u) {
              const mergedOne = smartMergeContacts(items);
              const perBlock = formatContactsViz(mergedOne, { totalImages: 1 });
              setCachedAnswer(u, perBlock);
              perImageBlocksFromCache.push(perBlock);
            }
          } else {
            // fallback: whatever text extractor returned
            const fallbackTxt = pickExtractorText(r);
            if (fallbackTxt) perImageBlocksFromCache.push(fallbackTxt);
          }
        }

        // 4) Build final VizChat-style message (merge to avoid duplicate Contact 1/2)
        let finalText = "";
        if (allItems.length) {
          const merged = smartMergeContacts(allItems);
          finalText = formatContactsViz(merged, {
            totalImages: serverUrls.length || images!.length,
          });
        } else if (perImageBlocksFromCache.length) {
          finalText =
            `Extracted contacts from ${serverUrls.length || images!.length} image` +
            `${(serverUrls.length || images!.length) > 1 ? "s" : ""}:\n\n` +
            perImageBlocksFromCache.join("\n\n");
        } else {
          finalText = "No recognizable details were found.";
        }

        replaceThinking({ text: finalText, sender: "ai" });
        return;
      }

      /** B) Docs QA route — send vectorstore doc IDs (no change) */
      const selectedIds = Array.isArray(combineDocs) ? combineDocs.filter(Boolean) : [];

      if (selectedIds.length > 0) {
        const compareHint =
          "\n\n---\nYou are an AI analyst comparing multiple documents. " +
          "Answer the user's question, then produce a markdown summary with sections:\n" +
          "1) Executive Summary (3–6 bullet points)\n" +
          "2) Comparison Table with columns: Criteria | Doc Name | Evidence (quote or paraphrase) | Page/Section\n" +
          "3) Key Differences / Conflicts\n" +
          "4) Gaps / Missing Info\n" +
          "Cite document names exactly as uploaded. Keep it concise.";
        const questionWithHint = text + compareHint;

        const qa = await askDocs({
          chatId: selectedChat.id,
          question: questionWithHint,
          docIds: selectedIds,
        });
        replaceThinking({
          text: qa?.answer || "❌ No answer returned.",
          sender: "ai",
          ...(qa?.plotImageUrl ? { imageUrl: qa.plotImageUrl, imageAlt: "Generated image" } : {}),
        });
        return;
      }

      // SINGLE-FILE (by ID)
      const activeDoc =
        documents.find((d) => d.documentId === (documentId || selectedDocId)) || null;

      if (activeDoc) {
        // If the active doc is an image we uploaded earlier, run the same extractor
        if (isImageName(activeDoc.name)) {
          const url = imageDocUrlMap[activeDoc.documentId];
          if (url) {
            const cached = getCachedAnswer(url);
            if (cached) {
              replaceThinking({ text: cached, sender: "ai" });
              return;
            }
            try {
              const file = await urlToFile(url, activeDoc.name);
              const r = await extractBusinessCard({
                file,
                returnVcard: true,
                prompt: buildCardPrompt(text),
              });
              const items = normalizeToItems(r);
              const merged = smartMergeContacts(items);
              const out = merged.length
                ? formatContactsViz(merged, { totalImages: 1 })
                : pickExtractorText(r) || "No recognizable details were found.";
              replaceThinking({ text: out, sender: "ai" });
              if (out) setCachedAnswer(url, out);
              return;
            } catch (e: any) {
              console.warn("Image fetch→File failed, falling back to doc-QA:", e?.message);
            }
          }
        }

        const qa = await askDocs({
          chatId: selectedChat.id,
          question: text,
          documentId: activeDoc.documentId,
        });
        replaceThinking({
          text: qa?.answer || "❌ No answer returned.",
          sender: "ai",
          ...(qa?.plotImageUrl ? { imageUrl: qa.plotImageUrl, imageAlt: "Generated image" } : {}),
        });
        return;
      }

      // No file selected → general QA
      const qa = await askDocs({
        chatId: selectedChat.id,
        question: text,
      });
      replaceThinking({
        text: qa?.answer || "❌ No answer returned.",
        sender: "ai",
        ...(qa?.plotImageUrl ? { imageUrl: qa.plotImageUrl, imageAlt: "Generated image" } : {}),
      });
    } catch (err: any) {
      replaceThinking({
        text: `❌ Backend Error: ${err?.message || "Request failed."}`,
        sender: "ai",
        status: "error",
      });
    } finally {
      setIsAsking(false);
    }
  };

  /* ---- sidebar selection wiring ---- */
  const handleSelectDocument = (docId: string) => {
    setSelectedDocId(docId);
    if (docId !== "combine") {
      setDocuments((prev) => {
        const idx = prev.findIndex((d) => d.documentId === docId);
        if (idx <= 0) return prev;
        const copy = [...prev];
        const [picked] = copy.splice(idx, 1);
        copy.unshift(picked);
        return copy;
      });
    }
  };

  /* ---- render ---- */
  return (
    <div className="min-h-screen bg-white text-black">
      <Header />

      {selectedChat ? (
        <div className="flex h-[calc(100vh-4rem)]">
          {/* LEFT: uploaded docs */}
          <DocumentSidebar
            chatId={chatId!}
            documentList={documents}
            onDocumentUpload={handleDocumentUpload}
            onSelectDocument={handleSelectDocument}
            selectedDocId={selectedDocId}
            selectedCombineDocs={selectedCombineDocs}
            setSelectedCombineDocs={setSelectedCombineDocs}
          />

          {/* RIGHT: chat interface */}
          <div className="flex-1 flex flex-col">
            <div className="bg-gray-100 border-b p-4 flex items-center justify-between">
              <button
                onClick={() => navigate("/chats")}
                className="p-2 hover:bg-gray-200 rounded-full"
              >
                <ArrowLeft className="w-5 h-5 text-gray-600" />
              </button>
              <h2 className="font-semibold">{selectedChat.name}</h2>
              <button
                onClick={() => deleteChat(selectedChat.id)}
                className="p-2 text-red-500 hover:bg-red-50 rounded-lg"
              >
                <Trash2 />
              </button>
            </div>

            <ChatInterface
              messages={selectedChat?.messages || []}
              onSendMessage={handleSendMessage}
              documents={documents.map((d) => ({
                documentId: d.documentId,
                name: d.name,
              }))}
              isLoading={isAsking}
            />
          </div>
        </div>
      ) : (
        <main className="container mx-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate("/")}
                className="p-2 hover:bg-gray-200 rounded-lg flex items-center"
              >
                <ArrowLeft className="mr-2" />
              </button>
              <h1 className="text-2xl font-bold">Chat Sessions</h1>
            </div>
            <button
              onClick={() => setShowNewChatDialog(true)}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg"
            >
              <Plus className="w-5 h-5 mr-2" />
              <span>New Chat</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {chatSessions.length > 0 ? (
              chatSessions.map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => navigate(`/chat/${chat.id}`)}
                  className="bg-gray-100 p-4 rounded-lg hover:shadow-md cursor-pointer"
                >
                  <div className="flex items-center space-x-2 mb-2">
                    <MessageCircle />
                    <span>{chat.name}</span>
                  </div>
                  <p className="text-sm text-gray-600">
                    {chat.documentName || "No document name"}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-center mt-12 text-gray-600">
                No chat sessions found.
              </p>
            )}
          </div>
        </main>
      )}

      {showNewChatDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg w-full max-w-md">
            <h2 className="text-xl mb-4">New Chat</h2>
            <label className="block mb-2">Upload Document(s):</label>
            <input
              type="file"
              multiple
              // Docs/Images only for this chat
              accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.webp,.gif"
              onChange={(e) =>
                setUploadedFiles(e.target.files ? Array.from(e.target.files) : [])
              }
              className="mb-4"
              disabled={isUploading}
            />
            {isUploading && (
              <p className="text-center text-blue-600">
                Uploading and processing files...
              </p>
            )}
            <label className="block mb-2">Chat Name:</label>
            <input
              type="text"
              value={newChatName}
              onChange={(e) => setNewChatName(e.target.value)}
              className="w-full border rounded p-2 mb-4"
              disabled={isUploading}
            />
            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setShowNewChatDialog(false)}
                className="px-4 py-2 bg-gray-200 rounded"
                disabled={isUploading}
              >
                Cancel
              </button>
              <button
                onClick={createNewChat}
                className="px-4 py-2 bg-blue-600 text-white rounded"
                disabled={isUploading}
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

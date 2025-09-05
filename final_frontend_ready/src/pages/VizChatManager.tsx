// src/pages/VizChatManager.tsx
import React, { useState, useEffect, useRef, useCallback, useLayoutEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Plus, ArrowLeft, Trash2 } from "lucide-react";
import { Header } from "@/components/Header";
import { VizDocumentSidebar } from "@/components/VizDocumentSidebar";
import { VizChatInterface } from "@/components/VizChatInterface";
import { useToast } from "@/hooks/use-toast";
import {
  askQuestion,
  askViz,
  uploadDocument,
  excelUpload,
  excelPlot,
  excelPlotCombine,
  chatUploadImages,
  extractBusinessCard,
  vizGenerate,
  vizGenerateCombined,
} from "@/api/api";
import type { VizDocument } from "@/components/VizDocumentSidebar";

/* ========================= Types ========================= */
type VizMsg = {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  imageUrl?: string;
  imageUrls?: string[];
  imageAlt?: string;
  tableCsvUrl?: string;
  kind?: string; // "kpi", "bar", ...
};

type VizChat = {
  id: string;
  name: string;
  createdAt: string;
  lastMessage: string;
  messageCount: number;
  messages: VizMsg[];
  documentName?: string | null;
};

/* ========================= Local storage ========================= */
const VIZ_CHAT_STORAGE = "vizChats";
const docsKeyFor = (id: string) => `documents:${id}`;
const vizMapKeyFor = (id: string) => `vizFilePathMap:${id}`; // id -> file_path
const imgAnsCacheKeyFor = (id: string) => `vizImgAnswerCache:${id}`;

const safeMsgs = (m: unknown): VizMsg[] => (Array.isArray(m) ? (m as VizMsg[]) : []);
const cleanFileName = (s?: string) =>
  (String(s || "").split(/[?#]/)[0].split(/[/\\]/).pop() || "").trim().replace(/["']/g, "");

/** keep tableCsvUrl & kind while sanitizing */
const sanitizeChats = (chats: VizChat[]): VizChat[] =>
  chats.map((c) => ({
    ...c,
    messages: safeMsgs(c.messages).map((m) => {
      const imgs = Array.isArray(m.imageUrls) ? m.imageUrls.filter((u) => !/^data:|^blob:/i.test(u)) : undefined;
      const single = typeof m.imageUrl === "string" && !/^data:|^blob:/i.test(m.imageUrl) ? m.imageUrl : undefined;
      const csvUrl = (m as any).tableCsvUrl || (m as any).table_csv_url || undefined;
      const kind = (m as any).kind || undefined;
      return { ...m, imageUrls: imgs, imageUrl: single, tableCsvUrl: csvUrl, kind };
    }),
  }));

const safeSaveChats = (chs: VizChat[]) => {
  try {
    const cleaned = sanitizeChats(chs);
    localStorage.setItem(VIZ_CHAT_STORAGE, JSON.stringify(cleaned));
  } catch {}
};

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substring(2, 9);
}
function getDocType(filename: string): VizDocument["type"] {
  const f = filename.toLowerCase();
  if (f.endsWith(".xlsx") || f.endsWith(".xls") || f.endsWith(".csv")) return "excel";
  return "pdf";
}
function mergeDocs(chatId: string, incoming: VizDocument[]): VizDocument[] {
  const key = docsKeyFor(chatId);
  const existing: VizDocument[] = JSON.parse(localStorage.getItem(key) || "[]");
  const byKey = new Map<string, VizDocument>();
  [...existing, ...incoming].forEach((d) => {
    const k = (d.documentId || d.name).toLowerCase();
    byKey.set(k, d);
  });
  const merged = Array.from(byKey.values());
  localStorage.setItem(key, JSON.stringify(merged));
  return merged;
}

/* ========================= DB persistence (NEW) ========================= */
type AppendExtras = {
  image_url?: string;
  image_urls?: string[];
  table_csv_url?: string;
  kind?: string;
};

async function ensureVizChat(params: { chatId: string; name?: string; createdAt?: string }) {
  try {
    await fetch("/api/db/chat/ensure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        chat_id: params.chatId,
        source: "viz",
        name: params.name ?? null,
        created_at: params.createdAt ?? new Date().toISOString(),
      }),
    });
  } catch {
    /* best-effort; don't block UI */
  }
}

async function appendVizEvent(params: {
  chatId: string;
  role: "user" | "ai";
  text: string;
  extras?: AppendExtras;
}) {
  try {
    await fetch("/api/db/chat/append", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        chat_id: params.chatId,
        source: "viz",
        role: params.role,
        text: params.text || "",
        ...(params.extras || {}),
      }),
    });
  } catch {
    /* best-effort */
  }
}

/* ========================= UI scroll ========================= */
const isScrollableEl = (el: HTMLElement) => {
  const st = getComputedStyle(el);
  return (st.overflowY === "auto" || st.overflowY === "scroll") && el.scrollHeight > el.clientHeight;
};
const findScrollableWithin = (root: HTMLElement | null): HTMLElement | null => {
  if (!root) return null;
  if (isScrollableEl(root)) return root;
  for (const n of Array.from(root.querySelectorAll<HTMLElement>("*"))) {
    if (isScrollableEl(n)) return n;
  }
  return null;
};
const scrollToBottomEl = (el: HTMLElement, smooth = true) =>
  el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
const isNearBottomEl = (el: HTMLElement, px = 120) =>
  el.scrollHeight - (el.scrollTop + el.clientHeight) <= px;

/* ========================= Misc ========================= */
const stringifyErr = (e: any): string => {
  if (!e) return "Request failed.";
  if (typeof e === "string") return e;
  if (typeof e?.message === "string" && e.message) return e.message;
  const d = e?.response?.data ?? e?.data ?? e?.detail ?? e?.error ?? e?.message;
  if (typeof d === "string" && d) return d;
  try {
    return JSON.stringify(d || e, null, 2);
  } catch {
    return String(d ?? e) || "Unknown error";
  }
};
const isGeminiKeyError = (msg: string) =>
  /gemini api key missing|GEMINI_API_KEY|GOOGLE_API_KEY|GENAI_API_KEY/i.test(String(msg || ""));

const fileToDataUrl = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result || ""));
    r.onerror = reject;
    r.readAsDataURL(file);
  });

/* ========================= Legacy plot prompts (fallback only) ========================= */
const BASE_READONLY = `
DATA IS READ-ONLY.
- Never call DataFrame.insert, never assign back to df to create/overwrite columns.
- Reuse existing columns; if needed, create locals (tmp variables) only.
- Do not write to disk.
- Respect user's requested chart type, x/y fields, filters, and aggregation.
- If multiple sheets exist, pick the one whose columns best match the request (e.g., has Sales/Profit/Date for sales time-series).
`;
const COLUMN_MAPPING = `
Map common columns case-insensitively (examples/synonyms):
- date/time: order_date, date, orderdate, txn_date, ship_date, year, month, quarter
- category: category, cat, product_category
- subcategory: sub-category, subcategory, sub_cat, product_subcategory
- product: product_name, item, sku, product
- region: region, state, province, city, market
- segment: segment, customer_segment
- customer: customer_name, customer, client
- sales: sales, sale_amount, revenue, net_sales, sales_amount, amount
- profit: profit, net_profit, margin, gross_profit, profit_amount
- quantity: quantity, qty, units_sold, order_quantity
- discount: discount, disc
- ship_mode: ship_mode, shipping_mode
- salary/comp: salary, salary_usd, annual_salary, ctc, compensation, pay, income, wage, "salary($)", salary_in_usd
- experience: experience, years_experience, yoe, yrs_exp, exp
`;
const CHART_TASK = (userText: string) =>
  `Create EXACTLY the chart the user asked for:\n"${userText}"\n- Choose appropriate axes/encodings; label axes and add a clear title.\n- Apply grouping/aggregation if the question implies it (sum of sales by category, etc.).\n- If dates exist, sort chronologically and format nicely.\n- Avoid adding any extra commentary in the image; just render the chart.`;

const buildExcelPlotPrompt = (userText: string) =>
  `${BASE_READONLY}\n${COLUMN_MAPPING}\n${CHART_TASK(userText)}\nIf the user mentions 'profit', you must use a profit-like column (not sales).`;
const ULTRA_STRICT_PROMPT = (userText: string) =>
  `${BASE_READONLY}\nABSOLUTE BAN on df inserts or in-place mutations.\n${CHART_TASK(userText)}\nIf the user mentions 'profit', you must use a profit-like column (not sales).`;
const DIAGNOSTIC_PROMPT = (userText: string) =>
  `${BASE_READONLY}\nFirst list columns + dtypes + 3 sample rows you actually used (brief), then render: ${CHART_TASK(userText)}`;

const extractImageFrom = (res: any): string =>
  res?.plotImageUrl || res?.image_url || (res?.image_base64 ? `data:image/png;base64,${res.image_base64}` : "");

/* ========================= Vision: business-card helpers (copied from ChatManager) ========================= */
const buildCardPrompt = (userText: string) =>
  `You are an OCR + contact extractor for business cards.
Return only what the card prints; no guessing. If a field is missing, omit it.
User request: ${userText || "Extract contact details from this card"}`.trim();

type CardItem = {
  first_name?: string;
  last_name?: string;
  organization?: string;
  job_title?: string;
  phones?: string[];
  emails?: string[];
  websites?: string[];
  address?: { street?: string; city?: string; state?: string; postal_code?: string; country?: string };
  source_url?: string;
};
const toArray = <T,>(x: T | T[] | undefined | null): T[] => (!x ? [] : Array.isArray(x) ? x : [x]);

function normalizeToItems(r: any): CardItem[] {
  const data = r?.data ?? r;
  const items = toArray<CardItem>(data?.items) as CardItem[];
  if (items.length) return items;
  const card = data?.card;
  if (card && typeof card === "object") return [card as CardItem];
  const cards = toArray<CardItem>(data?.cards) || toArray<CardItem>(data?.result);
  if (cards.length) return cards as CardItem[];
  return [];
}
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
  arrs
    .flat()
    .filter(Boolean)
    .forEach((x) => {
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
    if (it.job_title) L.push(`Title      : ${it.job_title}`);
    if (Array.isArray(it.phones) && it.phones.length) L.push(`Phones     : ${it.phones.join(", ")}`);
    if (Array.isArray(it.emails) && it.emails.length) L.push(`Emails     : ${it.emails.join(", ")}`);
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
const pickExtractorText = (r: any) => {
  const whats = r?.whatsapp || r?.data?.whatsapp;
  if (typeof whats === "string" && whats.trim()) return whats.trim();
  const card = r?.card || r?.data?.card;
  if (card && typeof card === "object") return JSON.stringify(card, null, 2);
  const txt = r?.text || r?.answer || r?.data?.text || "";
  return String(txt || "").trim();
};

/* Toggle: avoid uploading attachments to server in Viz chat (previews only) */
const SHOULD_UPLOAD_ATTACHMENTS = false;

/* ========================= Intent helpers ========================= */
const KPI_AGG_RE = /\b(sum|total|overall|avg|average|mean|median|max|maximum|min|minimum|count|how many|number of)\b/i;
const DIMENSION_HINT_RE =
  /\b(by|per|vs|versus|against|wise|breakdown|split|group(ed)? by|region|state|city|country|segment|category|subcategory|product|customer|payment|ship|month|year|week|day|date|time|over time|trend)\b/i;
const isKpiIntent = (txt: string) => KPI_AGG_RE.test(txt || "") && !DIMENSION_HINT_RE.test(txt || "");

/* ========================= Component ========================= */
export const VizChatManager = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [vizChatSessions, setVizChatSessions] = useState<VizChat[]>([]);
  const [documents, setDocuments] = useState<VizDocument[]>([]);
  const [selectedChat, setSelectedChat] = useState<VizChat | null>(null);

  // documentId -> server file_path mapping
  const [vizFilePathMap, setVizFilePathMap] = useState<Record<string, string>>({});

  const [showNewChatDialog] = useState(false);
  const [newChatName, setNewChatName] = useState("");

  const [uploadedFiles] = useState<File[]>([]);
  const [isAsking, setIsAsking] = useState(false);

  const pendingRef = useRef(false);
  const [hydrated, setHydrated] = useState(false);
  const [docsHydrated, setDocsHydrated] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [inputResetKey, setInputResetKey] = useState(0);

  const nudgeBottom = useCallback((smooth = true) => {
    const sc = findScrollableWithin(scrollAreaRef.current);
    if (!sc) return;
    const go = (s: boolean) => scrollToBottomEl(sc, s);
    go(smooth);
    setTimeout(() => go(false), 0);
    setTimeout(() => go(false), 150);
    setTimeout(() => go(false), 350);
  }, []);

  /* small cache so same image isn't re-parsed */
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

  useEffect(() => {
    const sc = findScrollableWithin(scrollAreaRef.current);
    if (!sc) return;
    const mo =
      typeof MutationObserver !== "undefined"
        ? new MutationObserver(() => {
            if (isNearBottomEl(sc)) scrollToBottomEl(sc, true);
          })
        : null;
    const ro =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => {
            if (isNearBottomEl(sc)) scrollToBottomEl(sc, true);
          })
        : null;
    mo?.observe?.(sc, { childList: true, subtree: true });
    ro?.observe?.(sc);
    return () => {
      mo?.disconnect?.();
      ro?.disconnect?.();
    };
  }, [selectedChat?.id]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(VIZ_CHAT_STORAGE);
      const parsed: VizChat[] = saved ? JSON.parse(saved) : [];
      const normalized = parsed.map((c) => ({
        ...c,
        messages: safeMsgs((c as any).messages),
        messageCount:
          typeof (c as any).messageCount === "number"
            ? (c as any).messageCount
            : safeMsgs((c as any).messages).length,
      }));
      setVizChatSessions(normalized);
    } catch {
      setVizChatSessions([]);
    } finally {
      setHydrated(true);
    }
  }, []);
  useEffect(() => {
    safeSaveChats(vizChatSessions);
  }, [vizChatSessions]);

  useEffect(() => {
    if (!chatId) {
      setSelectedChat(null);
      setDocuments([]);
      setDocsHydrated(false);
      setVizFilePathMap({});
      return;
    }
    const chat =
      selectedChat?.id === chatId ? selectedChat : vizChatSessions.find((c) => c.id === chatId) || null;
    setSelectedChat(
      chat
        ? {
            ...chat,
            messages: safeMsgs(chat.messages),
            messageCount:
              typeof chat.messageCount === "number" ? chat.messageCount : safeMsgs(chat.messages).length,
          }
        : null
    );

    try {
      const savedDocs = localStorage.getItem(docsKeyFor(chatId));
      const parsed: VizDocument[] = savedDocs ? JSON.parse(savedDocs) : [];
      setDocuments(Array.isArray(parsed) ? parsed : []);
    } catch {
      setDocuments([]);
    }
    setDocsHydrated(true);

    try {
      const m = localStorage.getItem(vizMapKeyFor(chatId));
      setVizFilePathMap(m ? JSON.parse(m) : {});
    } catch {
      setVizFilePathMap({});
    }

    if (hydrated && !chat) navigate("/visualizations", { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, vizChatSessions, hydrated]);

  useLayoutEffect(() => {
    if (selectedChat) requestAnimationFrame(() => requestAnimationFrame(() => nudgeBottom(false)));
  }, [selectedChat?.id, nudgeBottom]);

  useEffect(() => {
    if (!selectedChat) return;
    const sc = findScrollableWithin(scrollAreaRef.current);
    if (sc && isNearBottomEl(sc)) nudgeBottom(true);
  }, [selectedChat?.messages.length, nudgeBottom]);

  const saveVizFilePath = (docId: string, path: string) => {
    if (!chatId || !docId || !path) return;
    setVizFilePathMap((prev) => {
      const next = { ...prev, [docId]: path };
      localStorage.setItem(vizMapKeyFor(chatId), JSON.stringify(next));
      return next;
    });
  };

  const handleCreateChat = useCallback(() => {
    let name = (newChatName || "").trim();
    if (!name) {
      const reply = window.prompt("Name your chat (optional):", "");
      name = (reply || "").trim();
    }
    const id = generateId();
    const chat: VizChat = {
      id,
      name: name || `Chat ${new Date().toLocaleString()}`,
      createdAt: new Date().toISOString(),
      lastMessage: "",
      messageCount: 0,
      messages: [],
      documentName: null,
    };
    try {
      localStorage.setItem(docsKeyFor(id), "[]");
      localStorage.setItem(vizMapKeyFor(id), "{}");
    } catch {}
    setVizChatSessions((prev) => {
      const next = [chat, ...prev];
      safeSaveChats(next);
      return next;
    });
    setSelectedChat(chat);
    setDocuments([]);

    // ensure row in documents_chat (source='viz')
    ensureVizChat({ chatId: id, name: chat.name, createdAt: chat.createdAt });

    navigate(`/visualizations/chat/${id}`, { replace: true });
  }, [navigate, newChatName]);

  const uploadFile = async (file: File, targetChatId: string): Promise<VizDocument> => {
    const type = getDocType(file.name);
    if (type === "excel") {
      const up = await excelUpload(file, targetChatId); // { file_path, chat_id, message }
      const docId = generateId();
      if (up?.file_path) saveVizFilePath(docId, up.file_path);
      return {
        id: docId,
        name: file.name,
        type,
        size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
        uploadDate: new Date().toISOString(),
        status: "ready",
        chatId: targetChatId,
        documentId: docId, // path via vizFilePathMap
      } as VizDocument;
    } else {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("chat_id", targetChatId);
      await uploadDocument(formData);
      return {
        id: generateId(),
        name: file.name,
        type,
        size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
        uploadDate: new Date().toISOString(),
        status: "ready",
        chatId: targetChatId,
        documentId: file.name,
      } as VizDocument;
    }
  };

  const handleDocumentUpload = async (files: FileList) => {
    if (!chatId) {
      toast({ title: "Error", description: "Chat ID is missing." });
      return;
    }
    const uploaded: VizDocument[] = [];
    for (const file of Array.from(files)) {
      try {
        uploaded.push(await uploadFile(file, chatId));
      } catch (error: any) {
        toast({
          title: "Upload Failed",
          description: stringifyErr(error) || `Error uploading ${file.name}`,
        });
      }
    }
    if (uploaded.length) {
      const merged = mergeDocs(chatId, uploaded);
      setDocuments(merged);
      toast({ title: "Upload Successful", description: `${uploaded.length} file(s) uploaded.` });
      nudgeBottom(true);
    }
  };

  const isExcelDoc = (d?: VizDocument) => d && (d.type === "excel" || d.type === "csv");
  const isExcelName = (name?: string) => {
    const f = (name || "").toLowerCase();
    return f.endsWith(".xlsx") || f.endsWith(".xls") || f.endsWith(".csv");
  };
  const looksLikeImage = (f?: File) =>
    !!f &&
    ((f.type && (f.type.startsWith("image/") || f.type.toLowerCase() === "application/pdf")) ||
      /\.(png|jpe?g|webp|gif|bmp|tiff|tif|pdf)$/i.test(f?.name || ""));

  const deleteChat = useCallback(
    (id: string) => {
      if (!window.confirm("Delete this chat and its documents?")) return;
      try {
        localStorage.removeItem(docsKeyFor(id));
        localStorage.removeItem(vizMapKeyFor(id));
        localStorage.removeItem(imgAnsCacheKeyFor(id));
      } catch {}
      setVizChatSessions((prev) => {
        const next = prev.filter((c) => c.id !== id);
        safeSaveChats(next);
        return next;
      });
      if (selectedChat?.id === id) {
        setSelectedChat(null);
        setDocuments([]);
        setDocsHydrated(false);
        navigate("/visualizations", { replace: true });
      }
      toast({ title: "Chat deleted", description: "The chat and its documents were removed." });
    },
    [navigate, toast, selectedChat?.id]
  );

  /* ========================= Send ========================= */
  const handleSendMessage = async (
    text: string,
    documentId?: string | null,
    combineDocs?: string[],
    images?: File[]
  ) => {
    if (!selectedChat) return;
    if (pendingRef.current) return;
    pendingRef.current = true;
    setIsAsking(true);

    const imageFiles = (images || []).filter((f) => looksLikeImage(f));
    const previewUrls = await Promise.all(imageFiles.map(fileToDataUrl));

    const userMsgId = generateId();
    const userMsg: VizMsg = {
      id: userMsgId,
      text: text.trim(),
      sender: "user",
      timestamp: new Date().toISOString(),
      imageUrls: previewUrls.length ? previewUrls : undefined,
      imageAlt: previewUrls.length ? "attachments" : undefined,
    };

    let chatSnapshot: VizChat = {
      ...selectedChat,
      messages: [...safeMsgs(selectedChat.messages), userMsg],
      lastMessage: userMsg.text,
      messageCount: (selectedChat.messageCount || safeMsgs(selectedChat.messages).length) + 1,
    };

    setVizChatSessions((prev) => prev.map((c) => (c.id === chatSnapshot.id ? chatSnapshot : c)));
    setSelectedChat(chatSnapshot);

    // persist user message (text only)
    appendVizEvent({ chatId: chatSnapshot.id, role: "user", text: userMsg.text });

    const thinkingId = generateId();
    const thinkingMsg: VizMsg = {
      id: thinkingId,
      text: "Thinking…",
      sender: "ai",
      timestamp: new Date().toISOString(),
    };
    chatSnapshot = {
      ...chatSnapshot,
      messages: [...safeMsgs(chatSnapshot.messages), thinkingMsg],
      lastMessage: "Thinking…",
      messageCount: chatSnapshot.messageCount + 1,
    };
    setVizChatSessions((prev) => prev.map((c) => (c.id === chatSnapshot.id ? chatSnapshot : c)));
    setSelectedChat(chatSnapshot);
    requestAnimationFrame(() => requestAnimationFrame(() => nudgeBottom(true)));

    const replaceThinking = (patch: Partial<VizMsg>) => {
      setVizChatSessions((prev) =>
        prev.map((c) => {
          if (c.id !== chatSnapshot.id) return c;
          const msgs = safeMsgs(c.messages).map((m) => (m.id === thinkingId ? { ...m, ...patch } : m));
          const last = msgs[msgs.length - 1];
          return { ...c, messages: msgs, lastMessage: (last?.text || "").slice(0, 100) };
        })
      );
      setSelectedChat((cur) => {
        if (!cur || cur.id !== chatSnapshot.id) return cur;
        const msgs = safeMsgs(cur.messages).map((m) => (m.id === thinkingId ? { ...m, ...patch } : m));
        const last = msgs[msgs.length - 1];
        const next = { ...cur, messages: msgs, lastMessage: (last?.text || "").slice(0, 100) };
        chatSnapshot = next;
        return next;
      });

      // persist AI final answer (with extras)
      const aiText = String(patch.text || "").trim();
      if (aiText) {
        const extras: AppendExtras = {};
        if (patch.imageUrl) extras.image_url = patch.imageUrl;
        if (Array.isArray(patch.imageUrls)) extras.image_urls = patch.imageUrls;
        if ((patch as any).tableCsvUrl) extras.table_csv_url = (patch as any).tableCsvUrl;
        if ((patch as any).kind) extras.kind = (patch as any).kind;
        appendVizEvent({ chatId: chatSnapshot.id, role: "ai", text: aiText, extras });
      }
      requestAnimationFrame(() => requestAnimationFrame(() => nudgeBottom(true)));
    };

    const replaceUserImageUrls = (urls?: string[]) => {
      if (!urls || !urls.length) return;
      setVizChatSessions((prev) =>
        prev.map((c) => {
          if (c.id !== chatSnapshot.id) return c;
          const msgs = safeMsgs(c.messages).map((m) =>
            m.id === userMsgId ? { ...m, imageUrls: urls, imageAlt: "attachments" } : m
          );
          return { ...c, messages: msgs };
        })
      );
      setSelectedChat((cur) => {
        if (!cur || cur.id !== chatSnapshot.id) return cur;
        const msgs = safeMsgs(cur.messages).map((m) =>
          m.id === userMsgId ? { ...m, imageUrls: urls, imageAlt: "attachments" } : m
        );
        const next = { ...cur, messages: msgs };
        chatSnapshot = next;
        return next;
      });
    };

    try {
      /* ===== Vision: business card ===== */
      if (imageFiles.length) {
        // (optional) upload to server to persist; disabled for demo to avoid extra GETs
        let serverUrls: string[] = [];
        if (SHOULD_UPLOAD_ATTACHMENTS) {
          try {
            const up = await chatUploadImages({ chatId: selectedChat.id, text, files: imageFiles.slice(0, 4) });
            const urls = (up?.attachments || []).map((a: any) => a?.url).filter(Boolean);
            if (urls?.length) {
              serverUrls = urls;
              replaceUserImageUrls(urls);
            }
          } catch {
            /* ignore; previews already shown */
          }
        }

        // use cache where available
        const perImageBlocks: string[] = [];
        const itemsAll: CardItem[] = [];

        for (let i = 0; i < imageFiles.length; i++) {
          const f = imageFiles[i];
          const url = serverUrls[i];
          const cached = getCachedAnswer(url);
          if (cached) {
            perImageBlocks.push(cached);
            continue;
          }
          const r = await extractBusinessCard({
            file: f,
            returnVcard: false,
            prompt: buildCardPrompt(text),
          });
          const items = normalizeToItems(r);
          if (items.length) {
            itemsAll.push(...items);
            const mergedOne = smartMergeContacts(items);
            const block = formatContactsViz(mergedOne, { totalImages: 1 });
            perImageBlocks.push(block);
            if (url) setCachedAnswer(url, block);
          } else {
            const fallbackTxt = pickExtractorText(r) || "No recognizable details were found.";
            perImageBlocks.push(fallbackTxt);
            if (url) setCachedAnswer(url, fallbackTxt);
          }
        }

        let finalText = "";
        if (itemsAll.length) {
          const merged = smartMergeContacts(itemsAll);
          finalText = formatContactsViz(merged, { totalImages: imageFiles.length });
        } else {
          finalText =
            `Extracted contacts from ${imageFiles.length} image${imageFiles.length > 1 ? "s" : ""}:\n\n` +
            perImageBlocks.join("\n\n");
        }

        replaceThinking({ text: finalText, sender: "ai" });
        return;
      }

      /* ===== Multi-file (combine) ===== */
      if (Array.isArray(combineDocs) && combineDocs.length > 1) {
        const filePaths = combineDocs.map((id) => vizFilePathMap[id] || "").filter(Boolean);
        const fileNames = combineDocs
          .map((id) => documents.find((d) => d.documentId === id)?.name)
          .filter(Boolean)
          .map(cleanFileName);

        const payload: any = { question: text, title: undefined, chatId: selectedChat.id };
        if (filePaths.length) payload.filePaths = filePaths;
        if (!filePaths.length && fileNames.length) payload.fileNames = fileNames;

        if (payload.filePaths || payload.fileNames) {
          const res = await vizGenerateCombined(payload);
          const kind = (res as any)?.kind;
          replaceThinking({
            text:
              kind === "kpi"
                ? (res as any)?.title || "Calculated value shown below."
                : (res as any)?.message || `Generated from ${combineDocs.length} files.`,
            imageUrl: kind === "kpi" ? undefined : (res as any)?.image_url,
            imageAlt: (res as any)?.title || "Generated plot",
            tableCsvUrl: (res as any)?.table_csv_url || (res as any)?.tableCsvUrl,
            kind,
          });
          return;
        }
      }

      /* ===== Single Excel doc ===== */
      const doc = documents.find((d) => d.documentId === (documentId || undefined));
      const filePath = doc ? vizFilePathMap[doc.documentId] : undefined;
      const treatAsExcel = !!filePath || isExcelDoc(doc) || isExcelName(doc?.name);

      if (treatAsExcel) {
        const payload: any = { question: text, title: undefined, chatId: selectedChat.id };
        if (filePath) payload.filePath = filePath;
        else if (doc?.name) payload.fileName = cleanFileName(doc.name);

        try {
          const res = await vizGenerate(payload);
          const kind = (res as any)?.kind;
          replaceThinking({
            text:
              kind === "kpi"
                ? (res as any)?.title || "Calculated value shown below."
                : (res as any)?.message || "Here’s the chart and the values table.",
            imageUrl: kind === "kpi" ? undefined : (res as any)?.image_url,
            imageAlt: (res as any)?.title || "Generated plot",
            tableCsvUrl: (res as any)?.table_csv_url || (res as any)?.tableCsvUrl,
            kind,
          });
          return;
        } catch (e) {
          console.warn("vizGenerate failed, using legacy path:", e);
        }

        // Legacy fallbacks
        const fileName = cleanFileName(doc?.name);
        const tryDirectPlot = async (prompt: string) => {
          const res = await excelPlot(fileName, prompt, undefined, selectedChat.id);
          return {
            answerText: (res as any)?.meta?.title ? `### ${(res as any).meta.title}` : "Visualization created.",
            plot: extractImageFrom(res || {}),
          };
        };
        const tryAskViz = async (prompt: string) => {
          const res = await askViz({ question: prompt, chatId: selectedChat.id, fileName });
          return {
            answerText: (res as any)?.answer || (res as any)?.text || "Visualization created.",
            plot: extractImageFrom(res || {}),
          };
        };

        let answerText = "";
        let plotUrl = "";
        try {
          const r1 = await tryDirectPlot(buildExcelPlotPrompt(text));
          answerText = r1.answerText;
          plotUrl = r1.plot;
          if (!plotUrl) {
            const r2 = await tryDirectPlot(ULTRA_STRICT_PROMPT(text));
            answerText = r2.answerText;
            plotUrl = r2.plot;
          }
          if (!plotUrl) {
            const r3 = await tryDirectPlot(DIAGNOSTIC_PROMPT(text));
            answerText = r3.answerText;
            plotUrl = r3.plot;
          }
          if (!plotUrl) {
            const r4 = await tryAskViz(buildExcelPlotPrompt(text));
            answerText = r4.answerText;
            plotUrl = r4.plot;
          }
          if (!plotUrl) {
            const r5 = await tryAskViz(DIAGNOSTIC_PROMPT(text));
            answerText = r5.answerText;
            plotUrl = r5.plot;
          }
        } catch (err: any) {
          const msg = stringifyErr(err);
          replaceThinking({
            text:
              `❌ Plotting failed.\n` +
              `Tip: check that the Excel file opens normally and headers are not merged.\n\nDetails: ${msg}`,
          });
          toast({ title: "Plot error", description: msg });
          setIsAsking(false);
          pendingRef.current = false;
          setInputResetKey((k) => k + 1);
          return;
        }

        if (!plotUrl) {
          replaceThinking({
            text:
              "❌ I couldn't render a chart from the file.\n" +
              "- Make sure the sheet has the columns implied by your request (e.g., Sales/Profit/Date).\n" +
              "- If there are multiple sheets, try specifying the sheet name in your request.",
          });
        } else {
          replaceThinking({ text: answerText, imageUrl: plotUrl, imageAlt: "Generated plot" });
        }
      } else {
        // non-excel Q&A
        const res = await askQuestion({
          chatId: selectedChat.id,
          documentId: documentId || undefined,
          question: text,
          combineDocs: combineDocs || [],
        });
        const answerText = (res as any)?.answer || "❌ No answer returned.";
        const plotImageUrl = extractImageFrom(res || {});
        replaceThinking({
          text: answerText,
          ...(plotImageUrl ? { imageUrl: plotImageUrl, imageAlt: "Generated plot" } : {}),
        });
      }
    } catch (err: any) {
      const msg = stringifyErr(err);
      const friendly = isGeminiKeyError(msg)
        ? "Gemini API key missing on backend. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in server .env and restart."
        : msg;
      replaceThinking({ text: `❌ Backend Error: ${friendly}` });
      toast({ title: "Error", description: friendly });
    } finally {
      setIsAsking(false);
      pendingRef.current = false;
      setInputResetKey((k) => k + 1);
    }
  };

  const onSendWrapper = async (
    text: string,
    documentId?: string | null,
    combineDocs?: string[],
    images?: File[]
  ): Promise<void> => {
    await handleSendMessage(text, documentId, combineDocs, images);
  };

  return (
    <div className="min-h-screen bg-white text-black">
      <Header />
      {selectedChat ? (
        <div className="flex h-[calc(100vh-4rem)]">
          <VizDocumentSidebar
            chatId={chatId!}
            documentList={documents}
            onDocumentUpload={handleDocumentUpload}
          />
          <div className="flex-1 flex flex-col">
            <div className="bg-gray-100 border-b p-4 flex items-center justify-between">
              <button
                onClick={() => navigate("/visualizations")}
                className="p-2 hover:bg-gray-200 rounded-lg"
                aria-label="Back"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <h2 className="font-semibold truncate">{selectedChat.name}</h2>
              <button
                onClick={() => deleteChat(selectedChat.id)}
                className="p-2 text-red-500 hover:bg-red-50 rounded-lg"
                aria-label="Delete chat"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
            <div ref={scrollAreaRef} data-viz-scroll className="flex-1 min-h-0 overflow-y-auto">
              <VizChatInterface
                key={inputResetKey}
                messages={selectedChat?.messages || []}
                onSendMessage={onSendWrapper}
                documents={documents}
                isLoading={isAsking}
              />
            </div>
          </div>
        </div>
      ) : (
        <main className="container mx-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold">Visualization Chat</h1>
            <button
              onClick={handleCreateChat}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              <Plus className="w-5 h-5 mr-2" />
              <span>New Chat</span>
            </button>
          </div>
          <p className="text-gray-600">No chat selected.</p>
        </main>
      )}
    </div>
  );
};

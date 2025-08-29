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
  askImage,
  uploadDocument,
  excelUpload,
  excelPlotCombine,
  chatUploadImages,
} from "@/api/api";
import type { VizDocument } from "@/components/VizDocumentSidebar";

type VizMsg = {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  imageUrl?: string;
  imageAlt?: string;
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

const VIZ_CHAT_STORAGE = "vizChats";
const docsKeyFor = (id: string) => `documents:${id}`;
const safeMsgs = (m: unknown): VizMsg[] => (Array.isArray(m) ? (m as VizMsg[]) : []);
const cleanFileName = (s?: string) =>
  (String(s || "").split(/[?#]/)[0].split(/[/\\]/).pop() || "").trim().replace(/["']/g, "");

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

/* ---------- scroll helpers ---------- */
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

/* ---------- error helper ---------- */
const stringifyErr = (e: any): string => {
  if (!e) return "Request failed.";
  if (typeof e === "string") return e;
  if (typeof e?.message === "string" && e.message) return e.message;
  const d = e?.response?.data ?? e?.data ?? e?.detail ?? e?.error ?? e?.message;
  if (typeof d === "string" && d) return d;
  try { return JSON.stringify(d || e, null, 2); } catch { return String(d ?? e) || "Unknown error"; }
};

/* ---------- preview helper ---------- */
const fileToDataUrl = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result || ""));
    r.onerror = reject;
    r.readAsDataURL(file);
  });

/* ---------- business-card prompt (no JSON) ---------- */
const buildCardPrompt = (userText: string) => `
You are an OCR + contact parser for business cards and vCards.

OUTPUT RULES:
- Reply ONLY in clean, human-readable Markdown (no code blocks, no JSON, no tables).
- Use short topic lines like:
  **Name:** …
  **Title:** …
  **Company:** …
  **Phones:** add tel: links.
  **Emails:** add mailto: links.
  **Website:** clickable URL.
  **Address:** one neat line if visible.
  **Social:** list profiles with clickable links.
  **Extras:** slogans, “QR present/decoded URL”, notes.
- Include EVERY link on the card (front/back/QR). If you can't read a QR, say “QR present”.
- Fix obvious OCR errors (O↔0, l↔1, etc). Don’t invent data.

User request: ${userText || "Extract contact details from this card"}
`;

/* ---------- READ-ONLY plotting prompts & retry variants ---------- */
const BASE_READONLY = `
DATA IS READ-ONLY.
- Never call DataFrame.insert, never assign back to df to create/overwrite columns.
- Reuse existing columns. If you must transform, make a local Series or use df.assign(**with a unique, temporary name**), but prefer locals.
- Do not write to disk. Just build the plot and return it.
`;

const COLUMN_MAPPING = `
Try to map case-insensitively:
- sales: sales, sale_amount, revenue, net_sales, sales_amount
- profit: profit, net_profit, margin, gross_profit, profit_amount
- salary: salary, salary_usd, annual_salary, ctc, compensation, pay, income, wage, "salary($)", salary_in_usd
- experience: experience, years_experience, years_of_experience, yoe, yrs_exp, exp, experience_years
Clean numbers (strip currency/commas; convert "10 yrs" -> 10) before plotting.
`;

const buildExcelPlotPrompt = (userText: string) =>
  `${BASE_READONLY}
${COLUMN_MAPPING}
Create exactly the chart user asked (e.g., "${userText}"). If ambiguous, choose the best matching columns and proceed.
`;

const ULTRA_STRICT_PROMPT = (userText: string) => `
${BASE_READONLY}
ABSOLUTE BAN: do not use DataFrame.insert(), df["..."]=..., or overwrite any column names.
Use only existing columns; if you need a helper, use a local variable (NOT df.assign).
If suitable columns are not obvious, list columns and your picks, then plot.
User request: ${userText}
`;

const DIAGNOSTIC_PROMPT = (userText: string) => `
${BASE_READONLY}
List ALL columns with inferred dtypes and 3 sample values. Then pick the best columns and plot: ${userText}
`;

/* ---------- util to pull an image from any backend shape ---------- */
const extractImageFrom = (res: any): string => {
  return (
    res?.plotImageUrl ||
    res?.image_url ||
    (res?.image_base64 ? `data:image/png;base64,${res.image_base64}` : "")
  );
};

export const VizChatManager = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [vizChatSessions, setVizChatSessions] = useState<VizChat[]>([]);
  const [documents, setDocuments] = useState<VizDocument[]>([]);
  const [selectedChat, setSelectedChat] = useState<VizChat | null>(null);

  const [showNewChatDialog, setShowNewChatDialog] = useState(false);
  const [newChatName, setNewChatName] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isAsking, setIsAsking] = useState(false);

  const [hydrated, setHydrated] = useState(false);
  const [docsHydrated, setDocsHydrated] = useState(false);

  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const nudgeBottom = useCallback((smooth = true) => {
    const sc = findScrollableWithin(scrollAreaRef.current);
    if (!sc) return;
    const go = (s: boolean) => scrollToBottomEl(sc, s);
    go(smooth); setTimeout(() => go(false), 0); setTimeout(() => go(false), 150); setTimeout(() => go(false), 350);
  }, []);

  useEffect(() => {
    const sc = findScrollableWithin(scrollAreaRef.current);
    if (!sc) return;
    const mo = typeof MutationObserver !== "undefined"
      ? new MutationObserver(() => { if (isNearBottomEl(sc)) scrollToBottomEl(sc, true); })
      : null;
    const ro = typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(() => { if (isNearBottomEl(sc)) scrollToBottomEl(sc, true); })
      : null;
    mo?.observe?.(sc, { childList: true, subtree: true }); ro?.observe?.(sc);
    return () => { mo?.disconnect?.(); ro?.disconnect?.(); };
  }, [selectedChat?.id]);

  /* hydrate chats */
  useEffect(() => {
    try {
      const saved = localStorage.getItem(VIZ_CHAT_STORAGE);
      const parsed: VizChat[] = saved ? JSON.parse(saved) : [];
      const normalized = parsed.map((c) => ({
        ...c,
        messages: safeMsgs((c as any).messages),
        messageCount: typeof (c as any).messageCount === "number"
          ? (c as any).messageCount
          : safeMsgs((c as any).messages).length,
      }));
      setVizChatSessions(normalized);
    } catch {
      setVizChatSessions([]);
    } finally { setHydrated(true); }
  }, []);
  useEffect(() => { localStorage.setItem(VIZ_CHAT_STORAGE, JSON.stringify(vizChatSessions)); }, [vizChatSessions]);

  /* select chat & docs */
  useEffect(() => {
    if (!chatId) { setSelectedChat(null); setDocuments([]); setDocsHydrated(false); return; }
    const chat = selectedChat?.id === chatId ? selectedChat : (vizChatSessions.find((c) => c.id === chatId) || null);
    setSelectedChat(chat ? {
      ...chat,
      messages: safeMsgs(chat.messages),
      messageCount: typeof chat.messageCount === "number" ? chat.messageCount : safeMsgs(chat.messages).length,
    } : null);

    try {
      const savedDocs = localStorage.getItem(docsKeyFor(chatId));
      const parsed: VizDocument[] = savedDocs ? JSON.parse(savedDocs) : [];
      setDocuments(Array.isArray(parsed) ? parsed : []);
    } catch { setDocuments([]); }
    setDocsHydrated(true);

    if (hydrated && !chat) navigate("/visualizations", { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, vizChatSessions, hydrated]);

  useEffect(() => { if (chatId && docsHydrated) localStorage.setItem(docsKeyFor(chatId), JSON.stringify(documents)); }, [chatId, documents, docsHydrated]);

  /* one-time document id fix */
  useEffect(() => {
    if (!chatId || !docsHydrated || !documents.length) return;
    const needsFix = (d: any) =>
      typeof d?.documentId === "string" &&
      !/\.(xlsx|xls|csv|pdf)$/i.test(d.documentId) &&
      /\.(xlsx|xls|csv|pdf)$/i.test(d.name || "");
    const fixed = documents.map((d) => (needsFix(d) ? { ...d, documentId: d.name } : d));
    if (JSON.stringify(fixed) !== JSON.stringify(documents)) {
      setDocuments(fixed); localStorage.setItem(docsKeyFor(chatId), JSON.stringify(fixed));
    }
  }, [chatId, docsHydrated, documents]);

  useLayoutEffect(() => {
    if (selectedChat) requestAnimationFrame(() => requestAnimationFrame(() => nudgeBottom(false)));
  }, [selectedChat?.id, nudgeBottom]);

  useEffect(() => {
    if (!selectedChat) return;
    const sc = findScrollableWithin(scrollAreaRef.current);
    if (sc && isNearBottomEl(sc)) nudgeBottom(true);
  }, [selectedChat?.messages.length, nudgeBottom]);

  /* uploads */
  const uploadFile = async (file: File, targetChatId: string): Promise<VizDocument> => {
    const type = getDocType(file.name);
    if (type === "excel") await excelUpload(file, targetChatId);
    else {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("chat_id", targetChatId);
      await uploadDocument(formData);
    }
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
  };

  const handleDocumentUpload = async (files: FileList) => {
    if (!chatId) { toast({ title: "Error", description: "Chat ID is missing." }); return; }
    const uploaded: VizDocument[] = [];
    for (const file of Array.from(files)) {
      try { uploaded.push(await uploadFile(file, chatId)); }
      catch (error: any) {
        toast({ title: "Upload Failed", description: stringifyErr(error) || `Error uploading ${file.name}` });
      }
    }
    if (uploaded.length) {
      const merged = mergeDocs(chatId, uploaded);
      setDocuments(merged);
      toast({ title: "Upload Successful", description: `${uploaded.length} file(s) uploaded.` });
      nudgeBottom(true);
    }
  };

  /* helpers */
  const docById = (docId?: string | null) => documents.find((d) => d.documentId === docId);
  const isExcelDoc = (d?: VizDocument) => d && (d.type === "excel" || d.type === "csv");
  const looksLikeImage = (f?: File) =>
    !!f && ((f.type && f.type.startsWith("image/")) || /\.(png|jpe?g|webp|gif|bmp|tiff)$/i.test(f.name || ""));
  const persistFirstImage = async (text: string, files?: File[]) => {
    if (!chatId || !files || files.length === 0) return;
    const img = files.find((f) => looksLikeImage(f));
    if (!img) return;
    try { await chatUploadImages({ chatId, text, files: [img] }); } catch {}
  };

  /* delete chat */
  const deleteChat = useCallback((id: string) => {
    if (!window.confirm("Delete this chat and its documents?")) return;
    try { localStorage.removeItem(docsKeyFor(id)); } catch {}
    setVizChatSessions((prev) => {
      const next = prev.filter((c) => c.id !== id);
      localStorage.setItem(VIZ_CHAT_STORAGE, JSON.stringify(next));
      return next;
    });
    if (selectedChat?.id === id) {
      setSelectedChat(null); setDocuments([]); setDocsHydrated(false);
      navigate("/visualizations", { replace: true });
    }
    toast({ title: "Chat deleted", description: "The chat and its documents were removed." });
  }, [navigate, toast, selectedChat?.id]);

  /* send */
  const handleSendMessage = async (
    text: string,
    documentId?: string | null,
    combineDocs?: string[],
    images?: File[]
  ) => {
    if (!selectedChat || isAsking) return;
    setIsAsking(true);

    // NOTE: We no longer push a user bubble here (the UI already does optimistic echo).
    const thinkingId = generateId();
    const thinkingMsg: VizMsg = { id: thinkingId, text: "thinking…", sender: "ai", timestamp: new Date().toISOString() };
    let chatSnapshot: VizChat = {
      ...selectedChat,
      messages: [...safeMsgs(selectedChat.messages), thinkingMsg],
      lastMessage: "thinking…",
      messageCount: (selectedChat.messageCount || safeMsgs(selectedChat.messages).length) + 1,
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
      requestAnimationFrame(() => requestAnimationFrame(() => nudgeBottom(true)));
    };

    try {
      if (images && images.length > 0) {
        void persistFirstImage(text, images);
        const res = await askImage({ images, prompt: buildCardPrompt(text), question: text, text });
        const nl =
          (res as any)?.data?.whatsapp ??
          (res as any)?.answer ??
          (res as any)?.data?.text ??
          (res as any)?.text ??
          (typeof (res as any)?.status === "string" ? `Image processed (${(res as any).status}).` : "Processed the image(s).");
        replaceThinking({ text: String(nl).trim(), imageUrl: undefined, imageAlt: undefined });
      } else {
        // ===== Excel / CSV routes =====
        if (Array.isArray(combineDocs) && combineDocs.length > 1) {
          const allExcel = combineDocs.every((id) => isExcelDoc(documents.find((d) => d.documentId === id)));
          if (allExcel) {
            const filePaths = combineDocs.map((id) => cleanFileName(documents.find((d) => d.documentId === id)?.name || id));
            const res = await excelPlotCombine(filePaths, buildExcelPlotPrompt(text), undefined, selectedChat.id);
            const img = extractImageFrom(res || {});
            const title = (res as any)?.meta?.title || "Visualization";
            replaceThinking({ text: `### ${title}\nPlot generated from ${combineDocs.length} files.`, ...(img ? { imageUrl: img, imageAlt: title } : {}) });
            setIsAsking(false);
            return;
          }
        }

        const doc = docById(documentId || undefined);
        if (isExcelDoc(doc)) {
          const fileName = cleanFileName(doc?.name);

          const tryAskViz = async (prompt: string) => {
            const res = await askViz({ question: prompt, chatId: selectedChat.id, fileName });
            return { answerText: (res as any)?.answer || (res as any)?.text || "Visualization created.", plot: extractImageFrom(res || {}) };
          };
          const tryCombine = async (prompt: string) => {
            const res = await excelPlotCombine([fileName], prompt, undefined, selectedChat.id);
            return { answerText: (res as any)?.meta?.title ? `### ${(res as any).meta.title}` : "Visualization created.", plot: extractImageFrom(res || {}) };
          };

          let answerText = "";
          let plotUrl = "";

          try {
            // 1) Base read-only prompt
            ({ answerText, plotUrl: plotUrl as any } = await (async () => {
              const r = await tryAskViz(buildExcelPlotPrompt(text));
              return { answerText: r.answerText, plotUrl: r.plot };
            })());

            if (!plotUrl) {
              const r = await tryCombine(buildExcelPlotPrompt(text));
              answerText = r.answerText; plotUrl = r.plot;
            }
            if (!plotUrl) {
              const r = await tryAskViz(DIAGNOSTIC_PROMPT(text));
              answerText = r.answerText; plotUrl = r.plot;
            }
          } catch (errFirst: any) {
            const msg = stringifyErr(errFirst);

            // 2) If the failure mentions "insert" or "already exists", do a **strict** retry
            if (/insert|already exists/i.test(msg)) {
              try {
                const strict1 = await tryAskViz(ULTRA_STRICT_PROMPT(text));
                answerText = strict1.answerText; plotUrl = strict1.plot;

                if (!plotUrl) {
                  const strict2 = await tryCombine(ULTRA_STRICT_PROMPT(text));
                  answerText = strict2.answerText; plotUrl = strict2.plot;
                }
                if (!plotUrl) {
                  const diag = await tryAskViz(DIAGNOSTIC_PROMPT(text));
                  answerText = diag.answerText; plotUrl = diag.plot;
                }
              } catch (errStrict: any) {
                const msg2 = stringifyErr(errStrict);
                replaceThinking({ text: `❌ Backend Error: ${msg2}` });
                toast({ title: "Error", description: msg2 });
                setIsAsking(false);
                return;
              }
            } else {
              // generic failure → diagnostic
              try {
                const diag = await tryAskViz(DIAGNOSTIC_PROMPT(text));
                answerText = diag.answerText; plotUrl = diag.plot;
              } catch (errDiag: any) {
                const msg2 = stringifyErr(errDiag);
                replaceThinking({ text: `❌ Backend Error: ${msg2}` });
                toast({ title: "Error", description: msg2 });
                setIsAsking(false);
                return;
              }
            }
          }

          replaceThinking({ text: answerText, ...(plotUrl ? { imageUrl: plotUrl, imageAlt: "Generated plot" } : {}) });
        } else {
          // PDFs / general Q&A
          const res = await askQuestion({ chatId: selectedChat.id, documentId: documentId || undefined, question: text, combineDocs: combineDocs || [] });
          const answerText = (res as any)?.answer || "❌ No answer returned.";
          const plotImageUrl = extractImageFrom(res || {});
          replaceThinking({ text: answerText, ...(plotImageUrl ? { imageUrl: plotImageUrl, imageAlt: "Generated plot" } : {}) });
        }
      }
    } catch (err: any) {
      const msg = stringifyErr(err);
      replaceThinking({ text: `❌ Backend Error: ${msg}` });
      toast({ title: "Error", description: msg });
    } finally {
      setIsAsking(false);
    }
  };

  const onSendWrapper = async (text: string, documentId?: string | null, combineDocs?: string[], images?: File[]) => {
    await handleSendMessage(text, documentId, combineDocs, images);
  };

  return (
    <div className="min-h-screen bg-white text-black">
      <Header />
      {selectedChat ? (
        <div className="flex h-[calc(100vh-4rem)]">
          <VizDocumentSidebar chatId={chatId!} documentList={documents} onDocumentUpload={handleDocumentUpload} />
          <div className="flex-1 flex flex-col">
            <div className="bg-gray-100 border-b p-4 flex items-center justify-between">
              <button onClick={() => navigate("/visualizations")} className="p-2 hover:bg-gray-200 rounded-lg" aria-label="Back">
                <ArrowLeft className="w-5 h-5" />
              </button>
              <h2 className="font-semibold truncate">{selectedChat.name}</h2>
              <button onClick={() => deleteChat(selectedChat.id)} className="p-2 text-red-500 hover:bg-red-50 rounded-lg" aria-label="Delete chat">
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
            <div ref={scrollAreaRef} data-viz-scroll className="flex-1 min-h-0 overflow-y-auto">
              <VizChatInterface
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
            <button onClick={() => setShowNewChatDialog(true)} className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              <Plus className="w-5 h-5 mr-2" /><span>New Chat</span>
            </button>
          </div>
          <p className="text-gray-600">No chat selected.</p>
        </main>
      )}

      {showNewChatDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg w/full max-w-md">
            <h2 className="text-xl mb-4">New Chat</h2>
            <label className="block mb-2">Upload Document(s):</label>
            <input type="file" multiple accept=".pdf,.xls,.xlsx,.csv" onChange={(e) => setUploadedFiles(e.target.files ? Array.from(e.target.files) : [])} className="mb-4" />
            <label className="block mb-2">Chat Name:</label>
            <input type="text" value={newChatName} onChange={(e) => setNewChatName(e.target.value)} className="w-full border rounded p-2 mb-4" />
            <div className="flex justify-end space-x-2">
              <button onClick={() => setShowNewChatDialog(false)} className="px-4 py-2 bg-gray-200 rounded">Cancel</button>
              <button
                onClick={async () => {
                  if (!uploadedFiles.length || !newChatName.trim()) {
                    toast({ title: "Error", description: "Chat name and at least one document are required." }); return;
                  }
                  const newChatId = generateId();
                  const newDocs: VizDocument[] = [];
                  for (const file of uploadedFiles) {
                    try { newDocs.push(await uploadFile(file, newChatId)); }
                    catch (err: any) { toast({ title: "Upload Error", description: stringifyErr(err) || `Failed to upload ${file.name}` }); }
                  }
                  const greeting = `Hello! I'm ready to help you analyze ${uploadedFiles.map((f) => `"${f.name}"`).join(", ")}. What would you like to know?`;
                  const newChat: VizChat = {
                    id: newChatId,
                    name: newChatName,
                    createdAt: new Date().toISOString(),
                    lastMessage: greeting,
                    messageCount: 1,
                    messages: [{ id: "1", text: greeting, sender: "ai", timestamp: new Date().toISOString() }],
                  };
                  setVizChatSessions((prev) => {
                    const updated = [...prev, newChat];
                    localStorage.setItem(VIZ_CHAT_STORAGE, JSON.stringify(updated));
                    return updated;
                  });
                  const merged = mergeDocs(newChatId, newDocs);
                  setSelectedChat(newChat); setDocuments(merged); setDocsHydrated(true);
                  setShowNewChatDialog(false); setUploadedFiles([]); setNewChatName("");
                  navigate(`/visualizations/chat/${newChatId}`);
                  toast({ title: "Chat Created", description: `New chat with ${merged.length} document(s) created.` });
                  requestAnimationFrame(() => requestAnimationFrame(() => nudgeBottom(false)));
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
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

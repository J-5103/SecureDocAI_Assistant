// src/pages/VizChatManager.tsx
import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Plus, ArrowLeft, Trash2 } from "lucide-react";
import { Header } from "@/components/Header";
import { VizDocumentSidebar } from "@/components/VizDocumentSidebar";
import { VizChatInterface } from "@/components/VizChatInterface";
import { useToast } from "@/hooks/use-toast";
import {
  askQuestion,
  askViz,
  askImage,            // ← NEW: vision endpoint
  uploadDocument,
  excelUpload,
  excelPlot,
  excelPlotCombine,
} from "../api/api";
import type { VizDocument } from "@/components/VizDocumentSidebar";

type VizMsg = {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  imageUrl?: string; // will show attached image in the bubble
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

const API_BASE =
  (import.meta as any)?.env?.VITE_API_URL ||
  (globalThis as any)?.process?.env?.REACT_APP_API_URL ||
  "http://192.168.0.109:8000";

function generateId() {
  return Date.now().toString() + Math.random().toString(36).substring(2, 9);
}

// Map CSV → excel bucket for UI logic
function getDocType(filename: string): VizDocument["type"] {
  const f = filename.toLowerCase();
  if (f.endsWith(".xlsx") || f.endsWith(".xls") || f.endsWith(".csv")) return "excel";
  return "pdf";
}

// merge docs (de-dupe by documentId||name)
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

  // hydrate chats
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
    localStorage.setItem(VIZ_CHAT_STORAGE, JSON.stringify(vizChatSessions));
  }, [vizChatSessions]);

  // select chat & docs
  useEffect(() => {
    if (!chatId) {
      setSelectedChat(null);
      setDocuments([]);
      setDocsHydrated(false);
      return;
    }

    const chat =
      selectedChat?.id === chatId
        ? selectedChat
        : vizChatSessions.find((c) => c.id === chatId) || null;

    setSelectedChat(
      chat
        ? {
            ...chat,
            messages: safeMsgs(chat.messages),
            messageCount:
              typeof chat.messageCount === "number"
                ? chat.messageCount
                : safeMsgs(chat.messages).length,
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

    if (hydrated && !chat) navigate("/visualizations", { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, vizChatSessions, hydrated]);

  useEffect(() => {
    if (!chatId || !docsHydrated) return;
    localStorage.setItem(docsKeyFor(chatId), JSON.stringify(documents));
  }, [chatId, documents, docsHydrated]);

  // numeric documentId → filename (one-time migration)
  useEffect(() => {
    if (!chatId || !docsHydrated || !documents.length) return;

    const needsFix = (d: any) =>
      typeof d?.documentId === "string" &&
      !/\.(xlsx|xls|csv)$/i.test(d.documentId) &&
      /\.(xlsx|xls|csv)$/i.test(d.name || "");

    const fixed = documents.map((d) => (needsFix(d) ? { ...d, documentId: d.name } : d));
    if (JSON.stringify(fixed) !== JSON.stringify(documents)) {
      setDocuments(fixed);
      localStorage.setItem(docsKeyFor(chatId), JSON.stringify(fixed));
    }
  }, [chatId, docsHydrated, documents]);

  // upload one file (excel/pdf/etc)
  const uploadFile = async (file: File, targetChatId: string): Promise<VizDocument> => {
    const type = getDocType(file.name);

    if (type === "excel") {
      await excelUpload(file, targetChatId);
    } else {
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
    };
  };

  const handleDocumentUpload = async (files: FileList) => {
    if (!chatId) {
      toast({ title: "Error", description: "Chat ID is missing." });
      return;
    }
    const uploaded: VizDocument[] = [];
    for (const file of Array.from(files)) {
      try {
        const doc = await uploadFile(file, chatId);
        uploaded.push(doc);
      } catch (error: any) {
        toast({
          title: "Upload Failed",
          description: error?.message || `Error uploading ${file.name}`,
        });
      }
    }
    if (uploaded.length) {
      const merged = mergeDocs(chatId, uploaded);
      setDocuments(merged);
      toast({ title: "Upload Successful", description: `${uploaded.length} file(s) uploaded.` });
    }
  };

  // create new chat
  const createNewChat = async () => {
    if (!uploadedFiles.length || !newChatName.trim()) {
      toast({ title: "Error", description: "Chat name and at least one document are required." });
      return;
    }

    const newChatId = generateId();
    const newDocs: VizDocument[] = [];

    for (const file of uploadedFiles) {
      try {
        const doc = await uploadFile(file, newChatId);
        newDocs.push(doc);
      } catch (err: any) {
        toast({ title: "Upload Error", description: err?.message || `Failed to upload ${file.name}` });
      }
    }

    const greeting = `Hello! I'm ready to help you analyze ${uploadedFiles
      .map((f) => `"${f.name}"`)
      .join(", ")}. What would you like to know?`;

    const newChat: VizChat = {
      id: newChatId,
      name: newChatName,
      createdAt: new Date().toISOString(),
      lastMessage: greeting,
      messageCount: 1,
      messages: [
        {
          id: "1",
          text: greeting,
          sender: "ai",
          timestamp: new Date().toISOString(),
        },
      ],
    };

    setVizChatSessions((prev) => {
      const updated = [...prev, newChat];
      localStorage.setItem(VIZ_CHAT_STORAGE, JSON.stringify(updated));
      return updated;
    });

    const merged = mergeDocs(newChatId, newDocs);
    setSelectedChat(newChat);
    setDocuments(merged);
    setDocsHydrated(true);

    setShowNewChatDialog(false);
    setUploadedFiles([]);
    setNewChatName("");

    navigate(`/visualizations/chat/${newChatId}`);
    toast({ title: "Chat Created", description: `New chat with ${merged.length} document(s) created.` });
  };

  // delete chat
  const deleteChat = (id: string) => {
    setVizChatSessions((prev) => prev.filter((c) => c.id !== id));
    localStorage.removeItem(docsKeyFor(id));
    if (selectedChat?.id === id) {
      setSelectedChat(null);
      setDocuments([]);
      navigate("/visualizations");
    }
  };

  const docById = (docId?: string | null) => documents.find((d) => d.documentId === docId);
  const isExcelDoc = (d?: VizDocument) => d && (d.type === "excel" || d.type === "csv");

  // Upload any images to /api/chat (so the backend sees them with the question)
  const uploadImagesIfAny = async (text: string, images?: File[]) => {
    if (!chatId || !images || images.length === 0) return;
    const fd = new FormData();
    fd.append("chat_id", chatId);
    fd.append("text", text);
    images.forEach((f) => fd.append("files", f, f.name));
    try {
      await fetch(`${API_BASE}/api/chat`, { method: "POST", body: fd });
    } catch (e) {
      // don't block the rest of the flow if this fails
      console.error("Image upload failed", e);
    }
  };

  // Main send handler (original logic preserved; now supports image → /api/ask-image)
  const handleSendMessage = async (
    text: string,
    documentId?: string | null,
    combineDocs?: string[],
    images?: File[]
  ) => {
    if (!selectedChat || isAsking) return;
    setIsAsking(true);

    // show the first attached image in the user bubble (UI echo)
    const firstImgUrl = images && images.length > 0 ? URL.createObjectURL(images[0]) : undefined;

    const userMessage: VizMsg = {
      id: generateId(),
      text,
      sender: "user",
      timestamp: new Date().toISOString(),
      ...(firstImgUrl ? { imageUrl: firstImgUrl, imageAlt: "attachment" } : {}),
    };

    const baseMsgs = safeMsgs(selectedChat.messages);

    let updatedChat: VizChat = {
      ...selectedChat,
      messages: [...baseMsgs, userMessage],
      lastMessage: text,
      messageCount: (selectedChat.messageCount || baseMsgs.length) + 1,
    };

    setVizChatSessions((prev) => prev.map((c) => (c.id === updatedChat.id ? updatedChat : c)));
    setSelectedChat(updatedChat);

    try {
      // Save images for preview history (non-blocking)
      await uploadImagesIfAny(text, images);

      let aiMessage: VizMsg | null = null;

      // === NEW: if user attached images, run the Vision pipeline first ===
      if (images && images.length > 0) {
        try {
          const res = await askImage({
            frontFile: images[0],
            backFile: images[1] || null,
          });

          // Prefer whatsapp/plain text if available; else stringify json
          let visionText = "";
          const data = (res as any)?.data || {};
          if (typeof data.whatsapp === "string" && data.whatsapp.trim()) {
            visionText = data.whatsapp.trim();
          } else if (data.json) {
            try {
              visionText = "```json\n" + JSON.stringify(data.json, null, 2) + "\n```";
            } catch {
              visionText = String(data.json);
            }
          } else if (typeof (res as any)?.status === "string") {
            visionText = `Image processed (${(res as any).status}).`;
          } else {
            visionText = "Processed the image(s).";
          }

          aiMessage = {
            id: generateId(),
            text: visionText,
            sender: "ai",
            timestamp: new Date().toISOString(),
          };
        } catch (visionErr: any) {
          // Don't break the flow; fall back to old logic below
          console.error("askImage failed:", visionErr?.message || visionErr);
        }
      }

      // === Old logic preserved (runs when no images OR if vision failed) ===
      if (!aiMessage) {
        // combine multiple excel files
        if (Array.isArray(combineDocs) && combineDocs.length > 1) {
          const allExcel = combineDocs.every((id) => {
            const d = documents.find((dd) => dd.documentId === id);
            return isExcelDoc(d);
          });

          if (allExcel) {
            const filePaths = combineDocs.map((id) => {
              const d = documents.find((dd) => dd.documentId === id);
              return cleanFileName(d?.name || id);
            });
            const res = await excelPlotCombine(filePaths, text, undefined, selectedChat.id);
            const img =
              (res as any)?.image_url ||
              ((res as any)?.image_base64 ? `data:image/png;base64,${(res as any).image_base64}` : "");
            const title = (res as any)?.meta?.title || "Visualization";
            aiMessage = {
              id: generateId(),
              text: `### ${title}\nPlot generated from ${combineDocs.length} files.`,
              sender: "ai",
              timestamp: new Date().toISOString(),
              ...(img ? { imageUrl: img, imageAlt: title } : {}),
            };
          }
        }
      }

      // single doc: excel → askViz, others → askQuestion
      if (!aiMessage) {
        const doc = docById(documentId || undefined);

        if (isExcelDoc(doc)) {
          const res = await askViz({
            question: text,
            chatId: selectedChat.id,
            fileName: cleanFileName(doc?.name),
          });
          const answerText = (res as any)?.answer || "Visualization created.";
          const plotImageUrl =
            (res as any)?.image_url ||
            ((res as any)?.image_base64 ? `data:image/png;base64,${(res as any).image_base64}` : "");
          aiMessage = {
            id: generateId(),
            text: answerText,
            sender: "ai",
            timestamp: new Date().toISOString(),
            ...(plotImageUrl ? { imageUrl: plotImageUrl, imageAlt: "Generated plot" } : {}),
          };
        } else {
          const res = await askQuestion({
            chatId: selectedChat.id,
            documentId: documentId || undefined,
            question: text,
            combineDocs: combineDocs || [],
          });
          const answerText = (res as any)?.answer || "❌ No answer returned.";
          const maybeImg =
            (res as any)?.image_url ||
            ((res as any)?.image_base64 ? `data:image/png;base64,${(res as any).image_base64}` : "");
          aiMessage = {
            id: generateId(),
            text: answerText,
            sender: "ai",
            timestamp: new Date().toISOString(),
            ...(maybeImg ? { imageUrl: maybeImg, imageAlt: "Generated plot" } : {}),
          };
        }
      }

      // fallback: askViz with newest excel
      if (!aiMessage) {
        const res = await askViz({ question: text, chatId: selectedChat.id });
        const answerText = (res as any)?.answer || "Visualization created.";
        const maybeImg =
          (res as any)?.image_url ||
          ((res as any)?.image_base64 ? `data:image/png;base64,${(res as any).image_base64}` : "");
        aiMessage = {
          id: generateId(),
          text: answerText,
          sender: "ai",
          timestamp: new Date().toISOString(),
          ...(maybeImg ? { imageUrl: maybeImg, imageAlt: "Generated plot" } : {}),
        };
      }

      const finalChat: VizChat = {
        ...updatedChat,
        messages: [...updatedChat.messages, aiMessage!],
        lastMessage: aiMessage!.text.slice(0, 100),
        messageCount: updatedChat.messageCount + 1,
      };

      setVizChatSessions((prev) => prev.map((c) => (c.id === finalChat.id ? finalChat : c)));
      setSelectedChat(finalChat);
    } catch (err: any) {
      const errorMsg: VizMsg = {
        id: generateId(),
        text: `❌ Backend Error: ${err?.message || "Request failed."}`,
        sender: "ai",
        timestamp: new Date().toISOString(),
      };
      const erroredChat: VizChat = {
        ...updatedChat,
        messages: [...updatedChat.messages, errorMsg],
        lastMessage: errorMsg.text.slice(0, 100),
        messageCount: updatedChat.messageCount + 1,
      };
      setVizChatSessions((prev) => prev.map((c) => (c.id === erroredChat.id ? erroredChat : c)));
      setSelectedChat(erroredChat);
      toast({ title: "Error", description: err?.message || "Failed to get response from backend." });
    } finally {
      // cleanup object URL after a short delay to avoid memory leak
      if (typeof firstImgUrl === "string") {
        setTimeout(() => URL.revokeObjectURL(firstImgUrl), 10000);
      }
      setIsAsking(false);
    }
  };

  // wrapper (typed as any so it works whether VizChatInterface passes 3 or 4 args)
  const onSendWrapper: any = async (
    text: string,
    documentId?: string | null,
    combineDocs?: string[] | undefined,
    images?: File[] | undefined
  ) => {
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
              <button onClick={() => navigate("/visualizations")} className="p-2 hover:bg-gray-200 rounded-lg">
                <ArrowLeft className="w-5 h-5" />
              </button>
              <h2 className="font-semibold">{selectedChat.name}</h2>
              <button
                onClick={() => deleteChat(selectedChat.id)}
                className="p-2 text-red-500 hover:bg-red-50 rounded-lg"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>

            <VizChatInterface
              messages={selectedChat?.messages || []}
              onSendMessage={onSendWrapper}
              documents={documents}
              isLoading={isAsking}
            />
          </div>
        </div>
      ) : (
        <main className="container mx-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold">Visualization Chat</h1>
            <button
              onClick={() => setShowNewChatDialog(true)}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              <Plus className="w-5 h-5 mr-2" />
              <span>New Chat</span>
            </button>
          </div>
          <p className="text-gray-600">No chat selected.</p>
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
              accept=".pdf,.xls,.xlsx,.csv"
              onChange={(e) => setUploadedFiles(e.target.files ? Array.from(e.target.files) : [])}
              className="mb-4"
            />

            <label className="block mb-2">Chat Name:</label>
            <input
              type="text"
              value={newChatName}
              onChange={(e) => setNewChatName(e.target.value)}
              className="w-full border rounded p-2 mb-4"
            />

            <div className="flex justify-end space-x-2">
              <button onClick={() => setShowNewChatDialog(false)} className="px-4 py-2 bg-gray-200 rounded">
                Cancel
              </button>
              <button onClick={createNewChat} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

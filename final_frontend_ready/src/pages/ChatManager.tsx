// src/pages/ChatManager.tsx
import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Plus, MessageCircle, ArrowLeft, Trash2 } from "lucide-react";
import { Header } from "@/components/Header";
import { useToast } from "@/hooks/use-toast";

// ✅ DOC chat components (not visualization)
import { DocumentSidebar } from "@/components/DocumentSidebar";
import { ChatInterface } from "@/components/ChatInterface";
import type { Document as Doc } from "@/components/DocumentSidebar";

import {
  askQuestion,
  askImage,
  uploadDocument,
  chatUploadImages,
} from "../api/api";

type Msg = {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  imageUrl?: string;
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

const genId = () =>
  Date.now().toString() + Math.random().toString(36).substring(2, 9);

// ---- helpers
const IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".gif"];
const isImageName = (name = "") =>
  IMG_EXTS.some((ext) => name.toLowerCase().endsWith(ext));

// Map filename: image → treat as "other" for sidebar; pdf stays pdf
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

// MIGRATION: drop dead blob: URLs from saved messages
function sanitizeSessions(sessions: ChatSession[]): ChatSession[] {
  return sessions.map((cs) => ({
    ...cs,
    messages: (cs.messages || []).map((m) =>
      typeof m?.imageUrl === "string" && m.imageUrl.startsWith("blob:")
        ? { ...m, imageUrl: "" }
        : m
    ),
  }));
}

// Turn a saved URL into a File for askImage()
async function urlToFile(url: string, fallbackName = "image_from_doc.png"): Promise<File> {
  const res = await fetch(url, { credentials: "omit" });
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.status}`);
  const blob = await res.blob();
  let name = fallbackName;
  try {
    const u = new URL(url);
    const last = (u.pathname.split("/").pop() || "").trim();
    if (last) name = last;
  } catch {
    /* ignore */
  }
  const type = blob.type || "image/png";
  return new File([blob], name, { type });
}

export const ChatManager = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [documents, setDocuments] = useState<Doc[]>([]);
  const [selectedChat, setSelectedChat] = useState<ChatSession | null>(null);

  // sync selection with sidebar highlight
  const [selectedDocId, setSelectedDocId] = useState<string | undefined>(undefined);
  const [selectedCombineDocs, setSelectedCombineDocs] = useState<string[]>([]);

  const [showNewChatDialog, setShowNewChatDialog] = useState(false);
  const [newChatName, setNewChatName] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  const [isAsking, setIsAsking] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [docsHydrated, setDocsHydrated] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  // NEW: persistent map for image-type docs to their server URLs
  const [imageDocUrlMap, setImageDocUrlMap] = useState<Record<string, string>>({});

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
    localStorage.setItem(CHAT_STORAGE, JSON.stringify(chatSessions));
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

    setSelectedChat(chat);

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
      const url = up?.attachments?.[0]?.url || "";
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
      id: res.document_id || fallbackId,
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
      toast({ title: "Error", description: "Chat ID is missing." });
      return;
    }
    setIsUploading(true);
    const uploaded: Doc[] = [];
    for (const [i, file] of Array.from(files).entries()) {
      try {
        const doc = await uploadOneFile(file, chatId, i);
        uploaded.push(doc);
      } catch (error: any) {
        toast({
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
    if (selectedChat?.id === id) {
      setSelectedChat(null);
      setDocuments([]);
      setSelectedDocId(undefined);
      setSelectedCombineDocs([]);
      navigate("/chats");
    }
  };

  /* ---- send message (routing rules implemented) ---- */
  const handleSendMessage = async (
    text: string,
    documentId?: string,
    _combineDocs?: string[],
    images?: File[]
  ) => {
    if (!selectedChat || isAsking) return;
    setIsAsking(true);

    const hasImages = Array.isArray(images) && images.length > 0;

    // 1) push user message (temporary blob preview if any image)
    const tmpBlob =
      hasImages && typeof URL !== "undefined"
        ? URL.createObjectURL(images![0])
        : undefined;

    const userMsgId = genId();
    const userMessage: Msg = {
      id: userMsgId,
      text,
      sender: "user",
      timestamp: new Date().toISOString(),
      ...(tmpBlob ? { imageUrl: tmpBlob, imageAlt: "attachment" } : {}),
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

    const replaceThinking = (next: Partial<Msg>) => {
      const finalMessages = updatedChat.messages.map((m) =>
        m.id === thinkingId ? { ...m, ...next, status: next.status ?? "ok" } : m
      );
      const finalChat: ChatSession = {
        ...updatedChat,
        messages: finalMessages,
        lastMessage: (next.text || updatedChat.lastMessage || "").slice(0, 100),
        messageCount: updatedChat.messageCount + 1,
      };
      setChatSessions((prev) =>
        prev.map((c) => (c.id === finalChat.id ? finalChat : c))
      );
      setSelectedChat(finalChat);
      updatedChat = finalChat;
    };

    const replaceUserImageUrl = (url: string) => {
      const msgs = updatedChat.messages.map((m) =>
        m.id === userMsgId ? { ...m, imageUrl: url } : m
      );
      const finalChat = { ...updatedChat, messages: msgs };
      setChatSessions((prev) =>
        prev.map((c) => (c.id === finalChat.id ? finalChat : c))
      );
      setSelectedChat(finalChat);
      updatedChat = finalChat;
    };

    try {
      /** RULE 2: If question has image(s) → always image extraction (LLaVA). */
      if (hasImages) {
        // persist to server for permanent URL
        try {
          const up = await chatUploadImages({
            chatId: selectedChat.id,
            text,
            files: [images![0]],
          });
          const serverUrl = up?.attachments?.[0]?.url;
          if (serverUrl) replaceUserImageUrl(serverUrl);
        } catch {
          // non-blocking
        } finally {
          if (tmpBlob) setTimeout(() => URL.revokeObjectURL(tmpBlob), 15000);
        }

        // Vision extraction (multi-file capable)
        const res: any = await askImage({ images, prompt: text });
        const data = res?.data ?? res;
        let out =
          (typeof data?.whatsapp === "string" && data.whatsapp.trim()) ||
          (data?.text ? String(data.text) : "");
        if (!out && data?.json) {
          try {
            out = "```json\n" + JSON.stringify(data.json, null, 2) + "\n```";
          } catch {
            out = String(data.json);
          }
        }
        replaceThinking({
          text: out || "No text detected in image.",
          sender: "ai",
        });
        return;
      }

      /** RULE 1: No images → answer strictly from the selected file (if any). */
      const activeDoc = documents.find((d) => d.documentId === (documentId || selectedDocId));

      if (activeDoc) {
        // If selected file is an IMAGE → route through LLaVA by fetching it and sending as File
        if (isImageName(activeDoc.name)) {
          const url = imageDocUrlMap[activeDoc.documentId];
          if (url) {
            try {
              const file = await urlToFile(url, activeDoc.name);
              const res: any = await askImage({ images: [file], prompt: text });
              const data = res?.data ?? res;
              const out =
                (typeof data?.whatsapp === "string" && data.whatsapp.trim()) ||
                data?.answer ||
                data?.text ||
                "No text detected in image.";
              replaceThinking({ text: out, sender: "ai" });
              return;
            } catch (e: any) {
              // fall back to normal doc-QA if conversion fails
              console.warn("Image fetch→File failed, falling back to doc-QA:", e?.message);
            }
          }
          // if no URL saved, fall through to normal doc-QA
        }

        // PDF/Doc/Excel → LLama doc-QA confined to that document
        const qa = await askQuestion({
          chatId: selectedChat.id,
          documentId: activeDoc.documentId,
          combineDocs: [],
          question: text,
        });
        replaceThinking({
          text: qa?.answer || "❌ No answer returned.",
          sender: "ai",
          ...(qa?.plotImageUrl
            ? { imageUrl: qa.plotImageUrl, imageAlt: "Generated image" }
            : {}),
        });
        return;
      }

      // No file selected → general doc-QA
      const qa = await askQuestion({
        chatId: selectedChat.id,
        documentId: undefined,
        combineDocs: [],
        question: text,
      });
      replaceThinking({
        text: qa?.answer || "❌ No answer returned.",
        sender: "ai",
        ...(qa?.plotImageUrl
          ? { imageUrl: qa.plotImageUrl, imageAlt: "Generated image" }
          : {}),
      });
    } catch (err: any) {
      replaceThinking({
        text: `❌ Backend Error: ${err?.message || "Request failed."}`,
        sender: "ai",
        status: "error",
      });
      toast({
        title: "Error",
        description: err?.message || "Failed to get response from backend.",
        variant: "destructive",
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

          {/* RIGHT: chat interface (viz-style composer) */}
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
              accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.png,.jpg,.jpeg,.webp,.gif"
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

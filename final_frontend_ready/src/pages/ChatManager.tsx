// src/pages/ChatManager.tsx
import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Plus, MessageCircle, ArrowLeft, Trash2 } from "lucide-react";
import { Header } from "@/components/Header";
import { DocumentSidebar } from "@/components/DocumentSidebar";
import { ChatInterface } from "@/components/ChatInterface";
import { useToast } from "@/hooks/use-toast";
import { askQuestion, uploadDocument, excelUpload } from "../api/api";

export interface Document {
  id: string;
  name: string;
  type: "pdf" | "word" | "excel" | "image";
  size: string;
  uploadDate: string;
  status: "uploaded" | "processing" | "ready";
  chatId: string;
  documentId: string;
}

export interface ChatSession {
  id: string;
  name: string;
  createdAt: string;
  lastMessage: string;
  messageCount: number;
  messages: Array<{
    id: string;
    text: string;
    sender: "user" | "ai";
    timestamp: string;
    /** NEW (optional): show attached image in the bubble if present */
    imageUrl?: string;
    imageAlt?: string;
  }>;
  documentName?: string | null;
}

const CHAT_STORAGE = "chatSessions";
const docsKeyFor = (id: string) => `documents:${id}`;

function generateId(): string {
  return Date.now().toString() + Math.random().toString(36).substring(2, 9);
}

function getDocType(filename: string): Document["type"] {
  const f = filename.toLowerCase();
  if (f.endsWith(".pdf")) return "pdf";
  if (f.endsWith(".doc") || f.endsWith(".docx")) return "word";
  if (f.endsWith(".xls") || f.endsWith(".xlsx") || f.endsWith(".csv")) return "excel";
  if (f.endsWith(".png") || f.endsWith(".jpg") || f.endsWith(".jpeg")) return "image";
  return "pdf";
}

/** Merge documents into a chat’s store and de-dupe by documentId||name */
function mergeDocs(chatId: string, incoming: Document[]): Document[] {
  const key = docsKeyFor(chatId);
  const existing: Document[] = JSON.parse(localStorage.getItem(key) || "[]");
  const byKey = new Map<string, Document>();

  [...existing, ...incoming].forEach((d) => {
    const k = (d.documentId || d.name).toLowerCase();
    byKey.set(k, d);
  });

  const merged = Array.from(byKey.values());
  localStorage.setItem(key, JSON.stringify(merged));
  return merged;
}

export const ChatManager = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedChat, setSelectedChat] = useState<ChatSession | null>(null);
  const [showNewChatDialog, setShowNewChatDialog] = useState(false);
  const [newChatName, setNewChatName] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState<string | undefined>(undefined);
  const [selectedCombineDocs, setSelectedCombineDocs] = useState<string[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const [docsHydrated, setDocsHydrated] = useState(false);
  const [isUploading, setIsUploading] = useState(false); // upload progress

  useEffect(() => {
    try {
      const saved = localStorage.getItem(CHAT_STORAGE);
      const parsed: ChatSession[] = saved ? JSON.parse(saved) : [];
      setChatSessions(parsed);
    } catch {
      setChatSessions([]);
    } finally {
      setHydrated(true);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(CHAT_STORAGE, JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    if (!chatId) {
      setSelectedChat(null);
      setDocuments([]);
      setSelectedDocId(undefined);
      setDocsHydrated(false);
      return;
    }

    const chat =
      selectedChat?.id === chatId
        ? selectedChat
        : chatSessions.find((c) => c.id === chatId) || null;

    setSelectedChat(chat);

    try {
      const savedDocs = localStorage.getItem(docsKeyFor(chatId));
      const parsed: Document[] = savedDocs ? JSON.parse(savedDocs) : [];
      setDocuments(Array.isArray(parsed) ? parsed : []);
    } catch {
      setDocuments([]);
    }
    setSelectedDocId(undefined);
    setDocsHydrated(true);

    if (hydrated && !chat) {
      setSelectedChat(null);
      navigate("/chats", { replace: true });
    }
  }, [chatId, chatSessions, hydrated]);

  useEffect(() => {
    if (!chatId || !docsHydrated) return;
    localStorage.setItem(docsKeyFor(chatId), JSON.stringify(documents));
  }, [chatId, documents, docsHydrated]);

  const uploadOneFile = async (file: File, targetChatId: string, indexHint = 0): Promise<Document> => {
    const kind = getDocType(file.name);
    const fallbackId = `${Date.now()}-${indexHint}-${file.name}`;

    if (kind === "excel") {
      const res = await excelUpload(file, targetChatId);
      const id = (res?.documentId || res?.id || fallbackId) as string;
      return {
        id,
        name: (res?.name || file.name) as string,
        type: "excel",
        size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
        uploadDate: new Date().toISOString(),
        status: "ready",
        chatId: targetChatId,
        documentId: id,
      };
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("chat_id", targetChatId);
    try {
      const res = await uploadDocument(formData);
      return {
        id: res.document_id || fallbackId,
        name: file.name,
        type: kind,
        size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
        uploadDate: new Date().toISOString(),
        status: res.status || "ready",
        chatId: targetChatId,
        documentId: res.document_id,
      };
    } catch (error: any) {
      throw new Error(error?.message || `Error uploading ${file.name}`);
    }
  };

  const handleDocumentUpload = async (files: FileList): Promise<void> => {
    if (!chatId) {
      toast({ title: "Error", description: "Chat ID is missing." });
      return;
    }
    setIsUploading(true);
    const uploadedDocs: Document[] = [];
    for (const [i, file] of Array.from(files).entries()) {
      try {
        const doc = await uploadOneFile(file, chatId, i);
        uploadedDocs.push(doc);
      } catch (error: any) {
        toast({ title: "Upload Failed", description: error?.message || `Error uploading ${file.name}` });
      }
    }
    if (uploadedDocs.length) {
      const merged = mergeDocs(chatId, uploadedDocs);
      setDocuments(merged);
      toast({ title: "Upload Successful", description: `${uploadedDocs.length} file(s) uploaded.` });
    }
    setIsUploading(false);
  };

  const createNewChat = async () => {
    if (!uploadedFiles.length || !newChatName.trim()) {
      toast({ title: "Error", description: "Chat name and at least one document are required." });
      return;
    }

    const newChatId = generateId();
    const newDocs: Document[] = [];
    setIsUploading(true);

    for (const [i, file] of uploadedFiles.entries()) {
      try {
        const doc = await uploadOneFile(file, newChatId, i);
        newDocs.push(doc);
      } catch (err: any) {
        toast({ title: "Upload Error", description: err?.message || `Failed to upload ${file.name}` });
      }
    }

    if (!newDocs.length) {
      toast({ title: "Upload Error", description: "No files were uploaded successfully." });
      setIsUploading(false);
      return;
    }

    const newChat: ChatSession = {
      id: newChatId,
      name: newChatName,
      createdAt: new Date().toISOString(),
      lastMessage: "",
      messageCount: 0,
      messages: [
        {
          id: "1",
          text: `Hello! I'm ready to help you analyze ${uploadedFiles.map((f) => `"${f.name}"`).join(", ")}. What would you like to know?`,
          sender: "ai",
          timestamp: new Date().toISOString(),
        },
      ],
    };

    setChatSessions((prev) => [...prev, newChat]);
    const merged = mergeDocs(newChatId, newDocs);
    setSelectedChat(newChat);
    setDocuments(merged);
    setDocsHydrated(true);

    setShowNewChatDialog(false);
    setUploadedFiles([]);
    setNewChatName("");
    setIsUploading(false);

    navigate(`/chat/${newChatId}`);
    toast({ title: "Chat Created", description: `New chat with ${merged.length} document(s) created.` });
  };

  const deleteChat = (id: string) => {
    setChatSessions((prev) => prev.filter((c) => c.id !== id));
    localStorage.removeItem(docsKeyFor(id));
    if (selectedChat?.id === id) {
      setSelectedChat(null);
      setDocuments([]);
      navigate("/chats");
    }
  };

  /** NEW: upload any selected images alongside the question (non-blocking) */
  const uploadImagesIfAny = async (text: string, images?: File[]) => {
    if (!selectedChat || !images || images.length === 0) return;
    try {
      const fd = new FormData();
      fd.append("chat_id", selectedChat.id);
      fd.append("text", text);
      images.forEach((f) => fd.append("files", f, f.name));
      await fetch("/api/chat", { method: "POST", body: fd });
    } catch (e) {
      // Do not block the rest of the flow
      console.error("Image upload failed", e);
    }
  };

  /**
   * UPDATED: accepts optional `images` (4th arg).
   * If your ChatInterface doesn't pass images, this still behaves exactly like before.
   */
  const handleSendMessage = async (
    text: string,
    documentId?: string,
    _combineDocsMaybe?: string[],
    images?: File[]
  ) => {
    if (!selectedChat) return;
    if (isAsking) return;
    setIsAsking(true);

    // If an image is attached, show the first one in the user's bubble for immediate feedback.
    const firstImgUrl = images && images.length > 0 ? URL.createObjectURL(images[0]) : undefined;

    const userMessage = {
      id: generateId(),
      text,
      sender: "user" as const,
      timestamp: new Date().toISOString(),
      ...(firstImgUrl ? { imageUrl: firstImgUrl, imageAlt: "attachment" } : {}),
    };

    let updatedChat: ChatSession = {
      ...selectedChat,
      messages: [...selectedChat.messages, userMessage],
      lastMessage: text,
      messageCount: selectedChat.messageCount + 1,
    };

    setChatSessions((prev) => prev.map((c) => (c.id === updatedChat.id ? updatedChat : c)));
    setSelectedChat(updatedChat);

    try {
      // Upload images in parallel (non-blocking for Q&A)
      uploadImagesIfAny(text, images).catch(() => { /* ignore */ });

      const payload = {
        chatId: selectedChat.id,
        documentId: documentId || selectedDocId || undefined,
        combineDocs: selectedCombineDocs.length > 0 ? selectedCombineDocs : undefined,
        question: text,
      };
      const res = await askQuestion(payload);
      const answerText =
        res && typeof res === "object" && "answer" in res
          ? String((res as any).answer)
          : "❌ No answer returned.";

      const aiMessage = {
        id: generateId(),
        text: answerText,
        sender: "ai" as const,
        timestamp: new Date().toISOString(),
      };

      const finalChat: ChatSession = {
        ...updatedChat,
        messages: [...updatedChat.messages, aiMessage],
        lastMessage: aiMessage.text.slice(0, 100),
        messageCount: updatedChat.messageCount + 1,
      };

      setChatSessions((prev) => prev.map((c) => (c.id === finalChat.id ? finalChat : c)));
      setSelectedChat(finalChat);
    } catch (err: any) {
      toast({ title: "Error", description: err?.message || "Failed to get response from backend." });
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="min-h-screen bg-white text-black">
      <Header />

      {selectedChat ? (
        <div className="flex h-[calc(100vh-4rem)]">
          <DocumentSidebar
            chatId={chatId!}
            documentList={documents}
            onDocumentUpload={handleDocumentUpload}
            onSelectDocument={(docId) => setSelectedDocId(docId)}
            selectedDocId={selectedDocId}
            selectedCombineDocs={selectedCombineDocs}
            setSelectedCombineDocs={setSelectedCombineDocs}
          />

          <div className="flex-1 flex flex-col">
            <div className="bg-gray-100 border-b p-4 flex items-center justify-between">
              <button onClick={() => navigate("/chats")} className="p-2 hover:bg-gray-200 rounded-lg">
                <ArrowLeft />
              </button>
              <h2 className="font-semibold">{selectedChat.name}</h2>
              <button onClick={() => deleteChat(selectedChat.id)} className="p-2 text-red-500">
                <Trash2 />
              </button>
            </div>

            <ChatInterface
              messages={selectedChat?.messages || []}
              onSendMessage={handleSendMessage} // will accept (text, docId, combine?, images?) if provided
              documents={documents.map((d) => ({ documentId: d.documentId, name: d.name }))}
            />
          </div>
        </div>
      ) : (
        <main className="container mx-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <button onClick={() => navigate("/")} className="p-2 hover:bg-gray-200 rounded-lg flex items-center">
                <ArrowLeft className="mr-2" />
                Back
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
                  <p className="text-sm text-gray-600">{chat.documentName || "No document name"}</p>
                </div>
              ))
            ) : (
              <p className="text-center mt-12 text-gray-600">No chat sessions found.</p>
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
              accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.png,.jpg,.jpeg"
              onChange={(e) => setUploadedFiles(e.target.files ? Array.from(e.target.files) : [])}
              className="mb-4"
              disabled={isUploading}
            />
            {isUploading && <p className="text-center text-blue-600">Uploading and processing files...</p>}
            <label className="block mb-2">Chat Name:</label>
            <input
              type="text"
              value={newChatName}
              onChange={(e) => setNewChatName(e.target.value)}
              className="w-full border rounded p-2 mb-4"
              disabled={isUploading}
            />
            <div className="flex justify-end space-x-2">
              <button onClick={() => setShowNewChatDialog(false)} className="px-4 py-2 bg-gray-200 rounded" disabled={isUploading}>
                Cancel
              </button>
              <button onClick={createNewChat} className="px-4 py-2 bg-blue-600 text-white rounded" disabled={isUploading}>
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

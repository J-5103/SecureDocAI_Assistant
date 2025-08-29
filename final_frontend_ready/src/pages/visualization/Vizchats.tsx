// src/pages/visualization/Vizchats.tsx
import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Plus, MessageCircle, Trash2, ArrowLeft } from "lucide-react";
import { Header } from "@/components/Header";
import NewVizChatModal from "@/components/models/NewVizChatModal";
import { useToast } from "@/hooks/use-toast";
import { excelUpload } from "@/api/api";

type VizChat = {
  id: string;
  name: string;
  documentName?: string | null;
  createdAt?: number;
};

type AppDocument = {
  id: string;
  name: string;
  type: "excel";
  size: string;
  uploadDate: string;
  status: "uploaded" | "processing" | "ready";
  chatId: string;
  documentId: string; // must match filename so VizChatManager finds it
};

const STORAGE_KEY = "vizChats";

const Vizchats = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [visualizationChats, setVisualizationChats] = useState<VizChat[]>([]);
  const [showNewModal, setShowNewModal] = useState(false);

  // ✅ Always keep this listing page at TOP on refresh/mount
  useEffect(() => {
    // force top for SPA mount
    if ("scrollRestoration" in window.history) window.history.scrollRestoration = "manual";
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, []);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      const parsed: VizChat[] = saved ? JSON.parse(saved) : [];
      parsed.sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0));
      setVisualizationChats(parsed);
    } catch {
      setVisualizationChats([]);
    }
  }, []);

  const persistChats = (items: VizChat[]) =>
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));

  const openNewChatModal = () => setShowNewModal(true);
  const closeNewChatModal = () => setShowNewModal(false);

  const handleCreateChat = async (chatName: string, files: FileList) => {
    const id = window.crypto?.randomUUID?.() ?? `${Date.now()}`;

    // 1) Create chat locally
    const newChat: VizChat = { id, name: chatName, documentName: null, createdAt: Date.now() };
    const updated = [newChat, ...visualizationChats];
    setVisualizationChats(updated);
    persistChats(updated);

    // 2) Upload files
    const docsKey = `documents:${id}`;
    const uploadedDocs: AppDocument[] = [];
    let firstDocName: string | null = null;

    for (const file of Array.from(files)) {
      try {
        const res = await excelUpload(file, id);
        const docName = file.name;
        if (!firstDocName) firstDocName = docName;

        uploadedDocs.push({
          id: `${Date.now()}-${docName}`,
          name: docName,
          type: "excel",
          size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
          uploadDate: new Date().toISOString(),
          status: "ready",
          chatId: id,
          documentId: docName,
        });
      } catch (e: any) {
        toast({
          title: "Upload failed",
          description: e?.message || `Could not upload ${file.name}`,
          variant: "destructive",
        });
      }
    }

    if (uploadedDocs.length) {
      localStorage.setItem(docsKey, JSON.stringify(uploadedDocs));
      const refreshed = updated.map((c) => (c.id === id ? { ...c, documentName: firstDocName } : c));
      setVisualizationChats(refreshed);
      persistChats(refreshed);
      toast({ title: "Chat created", description: `${uploadedDocs.length} file(s) uploaded to this chat.` });
    } else {
      toast({ title: "Chat created", description: "No files uploaded." });
    }

    closeNewChatModal();

    // ✅ Navigate to chat detail and hint it to stick-bottom (via query)
    navigate(`/visualizations/chat/${id}?sb=1`);
  };

  const deleteChat = (id: string) => {
    const next = visualizationChats.filter((c) => c.id !== id);
    setVisualizationChats(next);
    persistChats(next);
    localStorage.removeItem(`documents:${id}`);
  };

  return (
    // ❌ Removed data-viz-scroll from listing page (should be on detail page wrapper)
    <div className="min-h-screen bg-white">
      <Header />
      <div className="container mx-auto p-6">
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/")}
              className="inline-flex items-center rounded-xl border px-3 py-2 hover:bg-gray-50"
              aria-label="Back to Home"
              title="Back to Home"
            >
              <ArrowLeft className="h-5 w-5 md:h-6 md:w-6" />
            </button>
            <h1 className="text-2xl font-bold">Visualization Chats</h1>
          </div>

          <button
            onClick={openNewChatModal}
            className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-white shadow hover:bg-blue-700"
          >
            <Plus className="h-5 w-5" />
            <span className="text-sm md:text-base">New Chat</span>
          </button>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {visualizationChats.length > 0 ? (
            visualizationChats.map((chat) => (
              <div key={chat.id} className="relative">
                {/* Whole card navigates */}
                <Link
                  // ✅ Pass stick-bottom hint to detail
                  to={`/visualizations/chat/${chat.id}?sb=1`}
                  className="block cursor-pointer rounded-lg bg-gray-100 p-4 transition hover:shadow-md no-underline text-inherit"
                >
                  <div className="mb-2 flex items-center space-x-2">
                    <MessageCircle className="h-5 w-5 md:h-6 md:w-6" />
                    <span className="line-clamp-1 font-medium">{chat.name}</span>
                  </div>
                  <p className="text-sm text-gray-600">
                    {chat.documentName || "No document name"}
                  </p>
                </Link>

                {/* Delete button (won’t trigger navigation) */}
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    deleteChat(chat.id);
                  }}
                  className="absolute right-3 top-3 rounded-lg p-2 text-red-600 hover:bg-red-50"
                  aria-label="Delete chat"
                  title="Delete chat"
                >
                  <Trash2 className="h-5 w-5 md:h-6 md:w-6" />
                </button>
              </div>
            ))
          ) : (
            <p className="mt-12 text-center text-gray-600">
              No visualization chats found.
            </p>
          )}
        </div>
      </div>

      {/* Modal */}
      <NewVizChatModal open={showNewModal} onClose={closeNewChatModal} onCreate={handleCreateChat} />
    </div>
  );
};

export default Vizchats;

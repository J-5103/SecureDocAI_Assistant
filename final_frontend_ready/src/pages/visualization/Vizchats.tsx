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

    // 1) Create chat locally (documentName will be set after first upload)
    const newChat: VizChat = { id, name: chatName, documentName: null, createdAt: Date.now() };
    const updated = [newChat, ...visualizationChats];
    setVisualizationChats(updated);
    persistChats(updated);

    // 2) Upload files for this chat (Excel/CSV only)
    const docsKey = `documents:${id}`;
    const uploadedDocs: AppDocument[] = [];
    let firstDocName: string | null = null;

    for (const file of Array.from(files)) {
      try {
        const res = await excelUpload(file, id); // { file_path, chat_id, message }
        // Build document entry from the file (backend doesn't return name/id)
        const docName = file.name;
        if (!firstDocName) firstDocName = docName;

        uploadedDocs.push({
          id: `${Date.now()}-${docName}`, // local unique id for UI
          name: docName,
          type: "excel",
          size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
          uploadDate: new Date().toISOString(),
          status: "ready",
          chatId: id,
          documentId: docName, // IMPORTANT: must be the filename; VizChatManager uses this
        });
      } catch (e: any) {
        toast({
          title: "Upload failed",
          description: e?.message || `Could not upload ${file.name}`,
          variant: "destructive",
        });
      }
    }

    // 3) Persist uploaded docs for this chat
    if (uploadedDocs.length) {
      localStorage.setItem(docsKey, JSON.stringify(uploadedDocs));
      // Update the card subtitle with first file name
      const nextChats = (prev: VizChat[]) =>
        prev.map((c) => (c.id === id ? { ...c, documentName: firstDocName } : c));
      const refreshed = nextChats(updated);
      setVisualizationChats(refreshed);
      persistChats(refreshed);

      toast({
        title: "Chat created",
        description: `${uploadedDocs.length} file(s) uploaded to this chat.`,
      });
    } else {
      toast({ title: "Chat created", description: "No files uploaded." });
    }

    // 4) Navigate to the chat detail page
    closeNewChatModal();
    navigate(`/visualizations/chat/${id}`);
  };

  const deleteChat = (id: string) => {
    const next = visualizationChats.filter((c) => c.id !== id);
    setVisualizationChats(next);
    persistChats(next);
    localStorage.removeItem(`documents:${id}`);
  };

  return (
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
                  to={`/visualizations/chat/${chat.id}`}
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

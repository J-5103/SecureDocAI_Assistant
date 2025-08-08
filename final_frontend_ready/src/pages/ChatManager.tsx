import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Plus, MessageCircle, ArrowLeft, Trash2 } from "lucide-react";
import { Header } from "@/components/Header";
import { DocumentSidebar } from "@/components/DocumentSidebar";
import { ChatInterface } from "@/components/ChatInterface";
import { useToast } from "@/hooks/use-toast";
import { askQuestion, uploadPdf } from "../api/api";

export interface Document {
  id: string;
  name: string;
  type: "pdf" | "word" | "excel";
  size: string;
  uploadDate: string;
  status: "uploaded" | "processing" | "ready";
  chatId: string;
  documentId: string;
}

function generateId() {
  return Date.now().toString() + Math.random().toString(36).substring(2, 9);
}

export const ChatManager = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [chatSessions, setChatSessions] = useState<any[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedChat, setSelectedChat] = useState<any | null>(null);

  const [showNewChatDialog, setShowNewChatDialog] = useState(false);
  const [newChatName, setNewChatName] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  // Sidebar’s multi-select (IDs). ChatInterface passes names; this is a fallback.
  const [selectedCombineDocs, setSelectedCombineDocs] = useState<string[]>([]);
  const [isAsking, setIsAsking] = useState(false);

  // --- hydrate from localStorage ---
  useEffect(() => {
    const savedChats = localStorage.getItem("chatSessions");
    const savedDocs = localStorage.getItem("documents");
    try {
      if (savedChats) setChatSessions(JSON.parse(savedChats));
      if (savedDocs) setDocuments(JSON.parse(savedDocs));
    } catch (e) {
      console.error("❌ Failed to parse localStorage", e);
      localStorage.clear();
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("chatSessions", JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    localStorage.setItem("documents", JSON.stringify(documents));
  }, [documents]);

  useEffect(() => {
    if (chatId) {
      const chat = chatSessions.find((c) => c.id === chatId);
      setSelectedChat(chat || null);
      // clear combine selections when changing chats
      setSelectedCombineDocs([]);
    } else {
      setSelectedChat(null);
    }
  }, [chatId, chatSessions]);

  // --- upload documents to backend and record locally ---
  const handleDocumentUpload = async (files: FileList) => {
    if (!chatId) {
      toast({ title: "Error", description: "Chat ID is missing." });
      return;
    }
    const uploadedDocs: Document[] = [];
    for (const file of Array.from(files)) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("chat_id", chatId);
        await uploadPdf(formData);

        uploadedDocs.push({
          id: generateId(),
          name: file.name,
          type: "pdf",
          size: (file.size / 1024 / 1024).toFixed(2) + " MB",
          uploadDate: new Date().toISOString(),
          status: "uploaded",
          chatId,
          documentId: file.name,
        });
      } catch (error: any) {
        toast({
          title: "Upload Failed",
          description: error.message || `Error uploading ${file.name}`,
        });
      }
    }

    if (uploadedDocs.length > 0) {
      setDocuments((prev) => {
        const updated = [...prev, ...uploadedDocs];
        localStorage.setItem("documents", JSON.stringify(updated));
        return updated;
      });
      toast({
        title: "Upload Successful",
        description: `${uploadedDocs.length} file(s) uploaded.`,
      });
    }
  };

  // --- new chat creation flow ---
  const createNewChat = async () => {
    if (!uploadedFiles.length || !newChatName.trim()) {
      toast({ title: "Error", description: "Chat name and at least one document are required." });
      return;
    }

    const newChatId = generateId();
    const newDocs: Document[] = [];

    for (const file of uploadedFiles) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("chat_id", newChatId);
        await uploadPdf(formData);

        newDocs.push({
          id: generateId(),
          name: file.name,
          type: "pdf",
          size: (file.size / 1024 / 1024).toFixed(2) + " MB",
          uploadDate: new Date().toISOString(),
          status: "uploaded",
          chatId: newChatId,
          documentId: file.name,
        });
      } catch (err: any) {
        toast({ title: "Upload Error", description: err.message || `Failed to upload ${file.name}` });
      }
    }

    if (!newDocs.length) {
      toast({ title: "Upload Error", description: "No files were uploaded successfully." });
      return;
    }

    const newChat = {
      id: newChatId,
      name: newChatName,
      createdAt: new Date().toISOString(),
      lastMessage: "",
      messageCount: 0,
      messages: [
        {
          id: "1",
          text: `Hello! I'm ready to help you analyze ${uploadedFiles
            .map((f) => `"${f.name}"`)
            .join(", ")}. What would you like to know?`,
          sender: "ai",
          timestamp: new Date().toISOString(),
        },
      ],
    };

    setChatSessions((prev) => {
      const updated = [...prev, newChat];
      localStorage.setItem("chatSessions", JSON.stringify(updated));
      return updated;
    });
    setDocuments((prev) => {
      const updatedDocs = [...prev, ...newDocs];
      localStorage.setItem("documents", JSON.stringify(updatedDocs));
      return updatedDocs;
    });

    setShowNewChatDialog(false);
    setUploadedFiles([]);
    setNewChatName("");
    navigate(`/chat/${newChatId}`);
    toast({ title: "Chat Created", description: `New chat with ${newDocs.length} document(s) created.` });
  };

  // --- delete chat + its docs locally ---
  const deleteChat = (id: string) => {
    setChatSessions((prev) => prev.filter((c) => c.id !== id));
    setDocuments((prev) => prev.filter((d) => d.chatId !== id));
    if (selectedChat?.id === id) {
      setSelectedChat(null);
      navigate("/chats");
    }
  };

  // --- send a question ---
  const handleSendMessage = async (
    text: string,
    documentId?: string,
    combineDocs?: string[]
  ) => {
    if (!selectedChat) return;
    if (isAsking) return; // avoid double send
    setIsAsking(true);

    // If ChatInterface didn't pass combine docs, fall back to sidebar's selection (IDs)
    const selectedForCombine = Array.isArray(combineDocs) ? combineDocs : selectedCombineDocs;

    if (!documentId && (!selectedForCombine || selectedForCombine.length === 0)) {
      toast({ title: "Error", description: "Select a document or enable combine mode." });
      setIsAsking(false);
      return;
    }

    const userMessage = {
      id: generateId(),
      text,
      sender: "user" as const,
      timestamp: new Date().toISOString(),
    };

    let updatedChat = {
      ...selectedChat,
      messages: [...selectedChat.messages, userMessage],
      lastMessage: text,
      messageCount: selectedChat.messageCount + 1,
    };

    setChatSessions((prev) =>
      prev.map((chat) => (chat.id === selectedChat.id ? updatedChat : chat))
    );
    setSelectedChat(updatedChat);

    try {
      const payload = {
        chatId: selectedChat.id,
        documentId: documentId || (selectedForCombine?.length ? "combine" : undefined),
        question: text,
        combineDocs: selectedForCombine || [],
      };

      const res = await askQuestion(payload);

      const answerText =
        (res && typeof res === "object" && "answer" in res) ? String(res.answer) :
        (typeof res === "string") ? res :
        "❌ No answer returned.";

      const aiMessage = {
        id: generateId(),
        text: answerText,
        sender: "ai" as const,
        timestamp: new Date().toISOString(),
      };

      const finalChat = {
        ...updatedChat,
        messages: [...updatedChat.messages, aiMessage],
        lastMessage: aiMessage.text.slice(0, 100),
        messageCount: updatedChat.messageCount + 1,
      };

      setChatSessions((prev) =>
        prev.map((chat) => (chat.id === selectedChat.id ? finalChat : chat))
      );
      setSelectedChat(finalChat);
    } catch (err: any) {
      toast({
        title: "Error",
        description: err?.message || "Failed to get response from backend.",
      });
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
            documentList={documents.filter((doc) => doc.chatId === chatId)}
            onDocumentUpload={handleDocumentUpload}
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
              onSendMessage={handleSendMessage}
              documents={documents.filter((doc) => doc.chatId === chatId)}
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
                  <p className="text-sm text-gray-600">
                    {chat.documentName || "No document name"}
                  </p>
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
              accept=".pdf"
              multiple
              onChange={(e) =>
                setUploadedFiles(e.target.files ? Array.from(e.target.files) : [])
              }
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
              <button
                onClick={() => setShowNewChatDialog(false)}
                className="px-4 py-2 bg-gray-200 rounded"
              >
                Cancel
              </button>
              <button
                onClick={createNewChat}
                className="px-4 py-2 bg-blue-600 text-white rounded"
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

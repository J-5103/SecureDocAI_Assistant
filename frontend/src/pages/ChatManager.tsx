import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Plus, MessageCircle, FileText, ArrowLeft, Trash2 } from "lucide-react";
import { Header } from "@/components/Header";
import { DocumentSidebar } from "@/components/DocumentSidebar";
import { ChatInterface } from "@/components/ChatInterface";
import { useToast } from "@/hooks/use-toast";
import { askQuestion } from "../api/api";
import { uploadFile } from "@/api/document_API";

export const ChatManager = () => {
  const { chatId } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [chatSessions, setChatSessions] = useState<any[]>([]);
  const [documents, setDocuments] = useState<any[]>([]);
  const [selectedChat, setSelectedChat] = useState<any | null>(null);
  const [showNewChatDialog, setShowNewChatDialog] = useState(false);
  const [chatPlots, setChatPlots] = useState<any[]>([]);
  const [newChatName, setNewChatName] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Load data from localStorage
  useEffect(() => {
    const savedChats = localStorage.getItem("chatSessions");
    const savedDocs = localStorage.getItem("documents");
    const savedPlots = localStorage.getItem("chatPlots");

    if (savedChats) setChatSessions(JSON.parse(savedChats));
    if (savedDocs) setDocuments(JSON.parse(savedDocs));
    if (savedPlots) setChatPlots(JSON.parse(savedPlots));
  }, []);

  // Save data to localStorage
  useEffect(() => {
    localStorage.setItem("chatSessions", JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    localStorage.setItem("documents", JSON.stringify(documents));
  }, [documents]);

  useEffect(() => {
    localStorage.setItem("chatPlots", JSON.stringify(chatPlots));
  }, [chatPlots]);

  // Handle selected chat by param
  useEffect(() => {
    if (chatId) {
      const chat = chatSessions.find((c) => c.id === chatId);
      if (chat) setSelectedChat(chat);
    }
  }, [chatId, chatSessions]);

  // Handle document upload
  const handleDocumentUpload = async (files: FileList) => {
    if (!files || files.length === 0) {
      toast({ title: "Upload Error", description: "No files selected." });
      return;
    }

    const uploadedDocs: any[] = [];

    for (const file of Array.from(files)) {
      if (!(file instanceof File)) continue;

      try {
        // Upload file to backend
        const response = await uploadFile(file);

        // Create new document object (safe, no name/size used)
        const newDoc = {
          id: crypto.randomUUID(),
          documentPath: response.document_path,
          status: "uploaded",
          uploadDate: new Date().toISOString(),
        };

        uploadedDocs.push(newDoc);

      } catch (error: any) {
        console.error(`❌ Failed to upload ${file}:`, error.message);
        toast({
          title: "Upload Failed",
          description: `Error uploading file: ${error.message}`,
        });
      }
    }

    if (uploadedDocs.length > 0) {
      setDocuments((prev) => [...prev, ...uploadedDocs]);

      toast({
        title: "Upload Successful",
        description: `${uploadedDocs.length} file(s) uploaded.`,
      });
    }
  };

  // Create new chat
  const createNewChat = () => {
    if (!(uploadedFile instanceof File)) {
      toast({ title: "Error", description: "Please upload a valid document." });
      return;
    }

    if (!newChatName.trim()) {
      toast({ title: "Error", description: "Chat name is required." });
      return;
    }

    const newDoc = {
      id: Date.now().toString() + Math.random().toString(36).slice(2, 11),
      name: uploadedFile.name,
      type: uploadedFile.name.endsWith(".pdf")
        ? "pdf"
        : uploadedFile.name.endsWith(".xlsx") || uploadedFile.name.endsWith(".xls")
        ? "excel"
        : "word",
      size: (uploadedFile.size / 1024 / 1024).toFixed(2) + " MB",
      uploadDate: new Date().toISOString(),
      status: "uploaded",
    };

    setDocuments((prev) => [...prev, newDoc]);

    const newChat = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      name: newChatName,
      documentId: newDoc.id,
      documentName: newDoc.name,
      createdAt: new Date().toISOString(),
      lastMessage: "",
      messageCount: 0,
      messages: [
        {
          id: "1",
          text: `Hello! I'm ready to help you analyze "${newDoc.name}". What would you like to know?`,
          sender: "ai",
          timestamp: new Date().toISOString(),
        },
      ],
    };

    setChatSessions((prev) => [...prev, newChat]);

    setShowNewChatDialog(false);
    setUploadedFile(null);
    setNewChatName("");
    navigate(`/chat/${newChat.id}`);

    toast({
      title: "Chat Created",
      description: `New chat "${newChatName}" created with document "${newDoc.name}".`,
    });
  };

  // Delete chat
  const deleteChat = (chatId: string) => {
    setChatSessions((prev) => prev.filter((c) => c.id !== chatId));
    if (selectedChat?.id === chatId) {
      setSelectedChat(null);
      navigate("/chats");
    }
  };

  // Send message
  const handleSendMessage = async (text: string) => {
    if (!selectedChat) return;

    const userMessage = {
      id: Date.now().toString(),
      text,
      sender: "user",
      timestamp: new Date().toISOString(),
    };

    const updatedChat = {
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
      const response = await askQuestion(selectedChat.id, selectedChat.documentName, text);

      const aiMessage = {
        id: (Date.now() + 1).toString(),
        text: response.answer,
        sender: "ai",
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
    } catch (err) {
      console.error(err);
      toast({ title: "Error", description: "Failed to get response from backend." });
    }
  };

  // Chat Session View
  if (selectedChat) {
    return (
      <div className="min-h-screen bg-gradient-secondary">
        <Header />
        <div className="flex h-[calc(100vh-4rem)]">
          <DocumentSidebar documents={documents} onDocumentUpload={handleDocumentUpload} />
          <div className="flex-1 flex flex-col">
            <div className="bg-card border-b p-4 flex items-center justify-between">
              <button onClick={() => navigate("/chats")} className="p-2 hover:bg-muted rounded-lg">
                <ArrowLeft />
              </button>
              <h2>{selectedChat.name}</h2>
              <button
                onClick={() => deleteChat(selectedChat.id)}
                className="p-2 text-red-500"
              >
                <Trash2 />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 bg-card">
              {selectedChat.messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`mb-2 ${msg.sender === "user" ? "text-right" : "text-left"}`}
                >
                  <div className="inline-block bg-muted px-4 py-2 rounded">{msg.text}</div>
                </div>
              ))}
            </div>
            <ChatInterface
              messages={selectedChat.messages}
              onSendMessage={handleSendMessage}
            />
          </div>
        </div>
      </div>
    );
  }

  // Chat List View with Back Navigation
  return (
    <div className="min-h-screen bg-gradient-secondary">
      <Header />
      <main className="container mx-auto p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigate("/")}
              className="p-2 hover:bg-muted rounded-lg flex items-center"
            >
              <ArrowLeft className="mr-2" />
              Back
            </button>
            <h1 className="text-2xl font-bold">Chat Sessions</h1>
          </div>

          <button
            onClick={() => setShowNewChatDialog(true)}
            className="flex items-center px-4 py-2 bg-gradient-primary text-white rounded-lg"
          >
            <Plus className="w-5 h-5" />
            <span>New Chat</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {chatSessions.map((chat) => (
            <div
              key={chat.id}
              onClick={() => navigate(`/chat/${chat.id}`)}
              className="bg-card p-4 rounded-lg hover:shadow-md cursor-pointer"
            >
              <div className="flex items-center space-x-2 mb-2">
                <MessageCircle />
                <span>{chat.name}</span>
              </div>
              <p className="text-sm text-muted-foreground">{chat.documentName}</p>
            </div>
          ))}
        </div>

        {chatSessions.length === 0 && (
          <p className="text-center mt-12 text-muted-foreground">
            No chat sessions found.
          </p>
        )}
      </main>

      {showNewChatDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card p-6 rounded-lg w-full max-w-md">
            <h2 className="text-xl mb-4">New Chat</h2>

            <label className="block mb-2">Upload Document:</label>
            <input
              type="file"
              onChange={(e) =>
                setUploadedFile(e.target.files && e.target.files[0] instanceof File ? e.target.files[0] : null)
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
                className="px-4 py-2 bg-muted rounded"
              >
                Cancel
              </button>
              <button
                onClick={createNewChat}
                className="px-4 py-2 bg-gradient-primary text-white rounded"
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

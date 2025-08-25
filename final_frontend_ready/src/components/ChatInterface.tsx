// src/components/ChatInterface.tsx
import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";

interface Message {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  /** NEW: optional image shown in bubble (user echo or AI image) */
  imageUrl?: string;
  imageAlt?: string;
}

interface Document {
  documentId: string;
  name: string;
}

interface ChatInterfaceProps {
  messages?: Message[];
  /** SUPerset: adding optional `images` keeps old logic compatible */
  onSendMessage: (
    question: string,
    documentId: string | null,
    combineDocs?: string[],
    images?: File[]
  ) => void | Promise<void>;
  documents?: Document[];
}

export const ChatInterface = ({
  messages = [],
  onSendMessage,
  documents = [],
}: ChatInterfaceProps) => {
  const [inputText, setInputText] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [isCombineMode, setIsCombineMode] = useState(false);
  const [combineDialogOpen, setCombineDialogOpen] = useState(false);
  const [selectedCombineDocs, setSelectedCombineDocs] = useState<string[]>([]);

  // NEW: local image selection + previews
  const [images, setImages] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const imgInputRef = useRef<HTMLInputElement>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto scroll when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Keep selection valid with docs changes
  useEffect(() => {
    if (!documents || documents.length === 0) {
      if (selectedDocId !== "") setSelectedDocId("");
      if (isCombineMode) setIsCombineMode(false);
      if (selectedCombineDocs.length > 0) setSelectedCombineDocs([]);
      return;
    }
    const hasSelected =
      selectedDocId && documents.some((d) => d.documentId === selectedDocId);

    if (!isCombineMode && !hasSelected) {
      setSelectedDocId(documents[0].documentId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents, isCombineMode, selectedDocId]);

  // --- image helpers ---
  const addImages = (files: FileList | null) => {
    if (!files) return;
    const nextFiles: File[] = [];
    const nextUrls: string[] = [];
    for (const f of Array.from(files)) {
      if (!f.type.startsWith("image/")) continue;
      nextFiles.push(f);
      nextUrls.push(URL.createObjectURL(f));
    }
    if (nextFiles.length === 0) return;
    setImages((p) => [...p, ...nextFiles]);
    setPreviews((p) => [...p, ...nextUrls]);
    if (imgInputRef.current) imgInputRef.current.value = "";
  };

  const removePreview = (idx: number) => {
    setPreviews((prev) => {
      const url = prev[idx];
      if (url) URL.revokeObjectURL(url);
      const copy = [...prev];
      copy.splice(idx, 1);
      return copy;
    });
    setImages((prev) => {
      const copy = [...prev];
      copy.splice(idx, 1);
      return copy;
    });
  };

  const clearImages = () => {
    previews.forEach((u) => URL.revokeObjectURL(u));
    setPreviews([]);
    setImages([]);
    if (imgInputRef.current) imgInputRef.current.value = "";
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed) return;

    if (isCombineMode && selectedCombineDocs.length > 0) {
      await onSendMessage(trimmed, null, selectedCombineDocs, images);
    } else if (!isCombineMode && selectedDocId) {
      await onSendMessage(trimmed, selectedDocId, undefined, images);
    } else {
      return;
    }

    setInputText("");
    clearImages(); // reset previews after send
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSubmit(e as unknown as React.FormEvent);
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    const textarea = e.target;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
  };

  const handleDropdownChange = (value: string) => {
    if (value === "combine") {
      setCombineDialogOpen(true);
    } else {
      if (isCombineMode) setIsCombineMode(false);
      if (selectedCombineDocs.length > 0) setSelectedCombineDocs([]);
      if (value !== selectedDocId) setSelectedDocId(value);
    }
  };

  const toggleCombineDoc = (docId: string) => {
    setSelectedCombineDocs((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const handleCombineConfirm = () => {
    if (selectedCombineDocs.length < 2) {
      alert("Please select at least two files to combine.");
      return;
    }
    if (!isCombineMode) setIsCombineMode(true);
    if (selectedDocId !== "combine") setSelectedDocId("combine");
    setCombineDialogOpen(false);
  };

  const handleCombineCancel = () => {
    setCombineDialogOpen(false);
    if (!isCombineMode && !selectedDocId && documents[0]) {
      setSelectedDocId(documents[0].documentId);
    }
  };

  const clearContext = () => {
    setIsCombineMode(false);
    setSelectedCombineDocs([]);
    setSelectedDocId(documents[0]?.documentId || "");
    clearImages(); // clear images along with context
  };

  const canSend =
    inputText.trim() !== "" &&
    (isCombineMode ? selectedCombineDocs.length > 0 : !!selectedDocId);

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-start space-x-3 animate-slide-up ${
              message.sender === "user" ? "flex-row-reverse space-x-reverse" : ""
            }`}
          >
            <div
              className={`flex items-center justify-center w-8 h-8 rounded-full ${
                message.sender === "ai" ? "bg-blue-600 shadow-glow" : "bg-gray-200"
              }`}
            >
              {message.sender === "ai" ? (
                <Bot className="w-5 h-5 text-white" />
              ) : (
                <User className="w-5 h-5 text-gray-800" />
              )}
            </div>

            <div
              className={`max-w-[70%] p-4 rounded-lg shadow-soft break-words ${
                message.sender === "ai"
                  ? "bg-blue-50 border border-blue-200"
                  : "bg-gray-100 text-gray-800"
              }`}
            >
              {/* Show image inside bubble if provided */}
              {message.imageUrl && (
                <img
                  src={message.imageUrl}
                  alt={message.imageAlt || "attachment"}
                  className="mb-3 rounded border max-w-full h-auto"
                />
              )}

              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {message.text}
              </p>
              <p
                className={`text-xs mt-2 ${
                  message.sender === "ai" ? "text-blue-600" : "text-gray-600"
                }`}
              >
                {new Date(message.timestamp).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Controls */}
      <div className="p-4 border-t border-gray-200 bg-white">
        {/* Selection row */}
        <div className="flex items-center gap-2 mb-2">
          <select
            className="px-2 py-1 border rounded text-sm bg-white text-gray-800"
            value={isCombineMode ? "combine" : selectedDocId || ""}
            onChange={(e) => handleDropdownChange(e.target.value)}
          >
            <option value="" disabled>
              Select a file
            </option>
            <option value="combine">🔗 Combine files</option>
            {documents.map((doc) => (
              <option key={doc.documentId} value={doc.documentId}>
                {doc.name}
              </option>
            ))}
          </select>

          {isCombineMode && (
            <span className="text-xs text-blue-700 bg-blue-50 border border-blue-200 px-2 py-1 rounded">
              {selectedCombineDocs.length} selected
            </span>
          )}

          {/* IMAGE button (back where the controls are) */}
          <input
            ref={imgInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => addImages(e.target.files)}
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => imgInputRef.current?.click()}
            title="Attach image(s)"
            aria-label="Attach image(s)"
          >
            📷
          </Button>

          {/* Small previews (with per-thumb remove) */}
          {previews.length > 0 && (
            <div className="flex items-center gap-2 overflow-x-auto">
              {previews.map((u, i) => (
                <div key={u} className="relative w-[44px] h-[34px] rounded border overflow-hidden">
                  <img src={u} alt={`preview-${i}`} className="w-full h-full object-cover" />
                  <button
                    type="button"
                    onClick={() => removePreview(i)}
                    className="absolute -top-2 -right-2 bg-white rounded-full shadow p-0.5"
                    title="Remove"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Clear (placed next to image controls) */}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={clearContext}
            className="ml-auto"
            title="Clear selection & images"
          >
            <X className="w-4 h-4 mr-1" />
            Clear
          </Button>
        </div>

        {/* Input row */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={inputText}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your file(s) or image…"
              className="min-h-[2.5rem] max-h-[120px] resize-none pr-12 border-gray-300 focus:border-blue-500 focus:ring-blue-200"
              rows={1}
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <Sparkles className="w-4 h-4 text-gray-400" />
            </div>
          </div>
          <Button
            type="submit"
            disabled={!canSend}
            className="bg-blue-600 hover:bg-blue-700 text-white transition-all duration-200 disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>

      {/* Combine dialog */}
      <Dialog
        open={combineDialogOpen}
        onOpenChange={(open) => (open ? setCombineDialogOpen(true) : handleCombineCancel())}
      >
        <DialogContent
          className="max-w-md bg-white"
          aria-labelledby="combine-dialog-title"
          aria-describedby="combine-dialog-description"
        >
          <DialogHeader>
            <DialogTitle id="combine-dialog-title" className="text-blue-700">
              Select files to Combine
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-2 max-h-[250px] overflow-y-auto" id="combine-dialog-description">
            {documents.length === 0 && (
              <p className="text-xs text-gray-500">No files available in this chat.</p>
            )}
            {documents.map((doc) => (
              <label key={doc.documentId} className="flex items-center space-x-2">
                <Checkbox
                  checked={selectedCombineDocs.includes(doc.documentId)}
                  onCheckedChange={() => toggleCombineDoc(doc.documentId)}
                  className="border-gray-300"
                />
                <span className="text-sm text-gray-800">{doc.name}</span>
              </label>
            ))}
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={handleCombineCancel}
              className="border-gray-300 text-gray-700"
            >
              Cancel
            </Button>
            <Button
              onClick={handleCombineConfirm}
              disabled={selectedCombineDocs.length < 2}
              className="bg-blue-600 text-white hover:bg-blue-700"
            >
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

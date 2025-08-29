// src/components/ChatInterface.tsx
import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { Send, Bot, User, Sparkles, X, Link as LinkIcon } from "lucide-react";
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
  imageUrl?: string;
  imageAlt?: string;
}

interface Document {
  documentId: string; // this is the doc_id/folder key under vectorstores/<chat_id>/
  name: string;
}

interface ChatInterfaceProps {
  messages?: Message[];
  /** onSendMessage(question, documentId?, docIds?, images?) */
  onSendMessage: (
    question: string,
    documentId?: string | null,
    docIds?: string[],
    images?: File[]
  ) => void | Promise<void>;
  documents?: Document[];
  isLoading?: boolean;
}

export const ChatInterface = ({
  messages = [],
  onSendMessage,
  documents = [],
  isLoading = false,
}: ChatInterfaceProps) => {
  const [inputText, setInputText] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [isCombineMode, setIsCombineMode] = useState(false);
  const [combineDialogOpen, setCombineDialogOpen] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]); // renamed for clarity

  // images (attach + previews)
  const [images, setImages] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const imgInputRef = useRef<HTMLInputElement>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composingRef = useRef(false);

  // ---- memo helpers ----
  const nameById = useMemo(() => {
    const m = new Map<string, string>();
    for (const d of documents) m.set(d.documentId, d.name);
    return m;
  }, [documents]);

  const chipNames = useMemo(() => {
    if (isCombineMode) return selectedDocIds.map((id) => nameById.get(id) || id);
    if (selectedDocId) return [nameById.get(selectedDocId) || selectedDocId];
    return [];
  }, [isCombineMode, selectedDocIds, selectedDocId, nameById]);

  // ---- effects ----
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // keep selection valid — only set when it would actually change
  useEffect(() => {
    if (!documents || documents.length === 0) {
      if (selectedDocId !== "" || isCombineMode || selectedDocIds.length) {
        setSelectedDocId("");
        setIsCombineMode(false);
        setSelectedDocIds([]);
      }
      return;
    }

    const hasSelected =
      !!selectedDocId && documents.some((d) => d.documentId === selectedDocId);

    if (!isCombineMode && !hasSelected) {
      const first = documents[0].documentId;
      if (selectedDocId !== first) {
        setSelectedDocId(first);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents]); // intentionally only on documents change

  // cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      previews.forEach((u) => URL.revokeObjectURL(u));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- image helpers ----
  const addImages = useCallback((files: FileList | null) => {
    if (!files) return;
    const nextFiles: File[] = [];
    const nextUrls: string[] = [];
    for (const f of Array.from(files)) {
      if (!f.type.startsWith("image/")) continue;
      nextFiles.push(f);
      nextUrls.push(URL.createObjectURL(f));
    }
    if (!nextFiles.length) return;
    setImages((p) => [...p, ...nextFiles]);
    setPreviews((p) => [...p, ...nextUrls]);
    if (imgInputRef.current) imgInputRef.current.value = "";
  }, []);

  const removePreview = useCallback((idx: number) => {
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
  }, []);

  const clearImages = useCallback(() => {
    previews.forEach((u) => URL.revokeObjectURL(u));
    setPreviews([]);
    setImages([]);
    if (imgInputRef.current) imgInputRef.current.value = "";
  }, [previews]);

  // ---- handlers ----
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = inputText.trim();
      if (!trimmed || isLoading) return;

      if (isCombineMode && selectedDocIds.length > 0) {
        // MULTI: pass as docIds (array)
        await onSendMessage(trimmed, null, selectedDocIds, images);
      } else if (!isCombineMode && selectedDocId) {
        // SINGLE
        await onSendMessage(trimmed, selectedDocId, undefined, images);
      } else {
        return;
      }

      setInputText("");
      clearImages();
      if (textareaRef.current) textareaRef.current.style.height = "auto";
    },
    [inputText, isLoading, isCombineMode, selectedDocIds, selectedDocId, onSendMessage, images, clearImages]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (composingRef.current) return;
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void handleSubmit(e as unknown as React.FormEvent);
      }
    },
    [handleSubmit]
  );

  const handleTextareaChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  }, []);

  const handleDropdownChange = useCallback((value: string) => {
    if (value === "combine") {
      setCombineDialogOpen(true);
    } else {
      setIsCombineMode(false);
      setSelectedDocIds([]);
      setSelectedDocId(value);
    }
  }, []);

  const toggleCombineDoc = useCallback(
    (docId: string, checked?: boolean | string) => {
      const willSelect =
        checked === true || (checked === undefined && !selectedDocIds.includes(docId));
      setSelectedDocIds((prev) =>
        willSelect ? [...prev, docId] : prev.filter((id) => id !== docId)
      );
    },
    [selectedDocIds]
  );

  const handleCombineConfirm = useCallback(() => {
    if (selectedDocIds.length < 2) {
      alert("Please select at least two files to combine.");
      return;
    }
    setIsCombineMode(true);
    setSelectedDocId("combine");
    setCombineDialogOpen(false);
  }, [selectedDocIds]);

  const handleCombineCancel = useCallback(() => {
    setCombineDialogOpen(false);
    if (!isCombineMode && !selectedDocId && documents[0]) {
      setSelectedDocId(documents[0].documentId);
    }
  }, [documents, isCombineMode, selectedDocId]);

  const clearContext = useCallback(() => {
    setIsCombineMode(false);
    setSelectedDocIds([]);
    setSelectedDocId(documents[0]?.documentId || "");
    clearImages();
  }, [documents, clearImages]);

  const canSend =
    inputText.trim() !== "" &&
    (isCombineMode ? selectedDocIds.length > 0 : !!selectedDocId) &&
    !isLoading;

  // ---- UI ----
  return (
    <div className="flex flex-col h-full bg-white">
      {/* messages */}
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
              aria-hidden
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
              {!!message.imageUrl && (
                <img
                  src={message.imageUrl}
                  alt={message.imageAlt || "attachment"}
                  className="mb-3 rounded border max-w-full h-auto"
                />
              )}
              <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.text}</p>
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

      {/* controls */}
      <div className="p-4 border-t border-gray-200 bg-white">
        {/* row 1: select */}
        <div className="flex items-center gap-2 mb-2">
          <select
            className="px-2 py-1 border rounded text-sm bg-white text-gray-800"
            value={isCombineMode ? "combine" : selectedDocId || ""}
            onChange={(e) => handleDropdownChange(e.target.value)}
            disabled={isLoading || documents.length === 0}
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
        </div>

        {/* row 2: Using chips + camera + Clear */}
        <div className="mb-2 flex items-center gap-2 flex-wrap">
          {chipNames.length > 0 && (
            <>
              <span className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-muted text-foreground/80 border">
                <LinkIcon className="w-3 h-3" />
                Using:
              </span>
              <div className="flex items-center gap-2 flex-wrap max-h-20 overflow-y-auto">
                {chipNames.map((n, i) => (
                  <span
                    key={`${n}-${i}`}
                    className="inline-flex items-center text-[11px] px-2 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200"
                  >
                    {n}
                  </span>
                ))}
              </div>
            </>
          )}

          {/* attach image */}
          <input
            ref={imgInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => addImages(e.target.files)}
            disabled={isLoading}
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => imgInputRef.current?.click()}
            disabled={isLoading}
            title="Attach image(s)"
            aria-label="Attach image(s)"
          >
            📷
          </Button>

          {/* previews */}
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

          {/* Clear on the right */}
          <button
            type="button"
            onClick={clearContext}
            className="ml-auto inline-flex items-center gap-1 text-sm text-gray-600 hover:text-gray-800"
            title="Clear selection & images"
            disabled={isLoading}
          >
            <X className="w-4 h-4" />
            Clear
          </button>
        </div>

        {/* input row */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={inputText}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              onCompositionStart={() => (composingRef.current = true)}
              onCompositionEnd={() => (composingRef.current = false)}
              placeholder="Ask anything about your file(s) or image…"
              className="min-h-[2.5rem] max-h-[120px] resize-none pr-12 border-gray-300 focus:border-blue-500 focus:ring-blue-200"
              rows={1}
              disabled={isLoading}
              aria-label="Ask a question"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <Sparkles className="w-4 h-4 text-gray-400" aria-hidden />
            </div>
          </div>

          <Button
            type="submit"
            disabled={!canSend}
            className="bg-blue-600 hover:bg-blue-700 text-white transition-all duration-200 disabled:opacity-50"
            aria-label="Send message"
          >
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>

      {/* combine dialog */}
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
              Select files to combine
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-2 max-h-[260px] overflow-y-auto" id="combine-dialog-description">
            {documents.length === 0 && (
              <p className="text-xs text-gray-500">No files available in this chat.</p>
            )}
            {documents.map((doc) => {
              const checked = selectedDocIds.includes(doc.documentId);
              return (
                <label key={doc.documentId} className="flex items-center space-x-2">
                  <Checkbox
                    checked={checked}
                    onCheckedChange={(val) => toggleCombineDoc(doc.documentId, val as boolean)}
                    className="border-gray-300"
                  />
                  <span className="text-sm text-gray-800">{doc.name}</span>
                </label>
              );
            })}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={handleCombineCancel} className="border-gray-300 text-gray-700">
              Cancel
            </Button>
            <Button
              onClick={handleCombineConfirm}
              disabled={selectedDocIds.length < 2}
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

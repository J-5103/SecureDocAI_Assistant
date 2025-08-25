// src/components/VizChatInterface.tsx
import React, { useState, useRef, useEffect, useMemo } from "react";
import { Send, Bot, User, Sparkles, Link as LinkIcon, X } from "lucide-react";
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
import ReactMarkdown from "react-markdown";

interface Message {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  imageUrl?: string;
  imageAlt?: string;
}

interface Document {
  documentId: string;
  name: string;
}

interface ChatInterfaceProps {
  messages?: Message[];
  onSendMessage: (
    question: string,
    documentId?: string | null,
    combineDocs?: string[],
    images?: File[]           // ← pass selected images up
  ) => Promise<void> | void;
  documents?: Document[];
  isLoading?: boolean;
}

export const VizChatInterface = ({
  messages = [],
  onSendMessage,
  documents = [],
  isLoading = false,
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
  const composingRef = useRef(false);

  const fileDocuments = documents;

  const nameById = useMemo(() => {
    const m = new Map<string, string>();
    for (const d of fileDocuments) m.set(d.documentId, d.name);
    return m;
  }, [fileDocuments]);

  const chipNames = useMemo(() => {
    if (isCombineMode) return selectedCombineDocs.map((id) => nameById.get(id) || id);
    if (selectedDocId) return [nameById.get(selectedDocId) || selectedDocId];
    return [];
  }, [isCombineMode, selectedCombineDocs, selectedDocId, nameById]);

  const derivedMessages: Message[] = useMemo(() => {
    if (messages.length > 0) return messages;
    if (fileDocuments.length > 0) {
      const names = fileDocuments.map((d) => `"${d.name}"`).join(", ");
      return [
        {
          id: "welcome",
          text: `Hello! I'm ready to help you analyze ${names}. What would you like to know?`,
          sender: "ai",
          timestamp: new Date().toISOString(),
        },
      ];
    }
    return [];
  }, [messages, fileDocuments]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [derivedMessages, isLoading]);

  useEffect(() => {
    if (!selectedDocId && fileDocuments.length > 0) {
      setSelectedDocId(fileDocuments[0].documentId);
    }
  }, [fileDocuments, selectedDocId]);

  // ---- image helpers ----
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

  // ----- Handlers -----
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || isLoading) return;

    if (isCombineMode && selectedCombineDocs.length > 0) {
      await onSendMessage(trimmed, null, selectedCombineDocs, images);
    } else if (!isCombineMode && selectedDocId) {
      await onSendMessage(trimmed, selectedDocId, undefined, images);
    } else {
      // no doc context chosen
      return;
    }

    setInputText("");
    clearImages();
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (composingRef.current) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  };

  const handleDropdownChange = (value: string) => {
    if (value === "combine") setCombineDialogOpen(true);
    else {
      setIsCombineMode(false);
      setSelectedCombineDocs([]);
      setSelectedDocId(value);
    }
  };

  const toggleCombineDoc = (docId: string, checked?: boolean | string) => {
    const willSelect = checked === true || (checked === undefined && !selectedCombineDocs.includes(docId));
    setSelectedCombineDocs((prev) => (willSelect ? [...prev, docId] : prev.filter((id) => id !== docId)));
  };

  const handleCombineConfirm = () => {
    if (selectedCombineDocs.length < 2) {
      alert("Please select at least two files to combine.");
      return;
    }
    setIsCombineMode(true);
    setSelectedDocId("combine");
    setCombineDialogOpen(false);
  };

  const handleCombineCancel = () => {
    setCombineDialogOpen(false);
    if (!isCombineMode && !selectedDocId && fileDocuments[0]) {
      setSelectedDocId(fileDocuments[0].documentId);
    }
  };

  const clearContext = () => {
    setIsCombineMode(false);
    setSelectedCombineDocs([]);
    setSelectedDocId(fileDocuments[0]?.documentId || "");
    clearImages(); // also clear images when clearing context
  };

  const canSend =
    inputText.trim() !== "" &&
    (isCombineMode ? selectedCombineDocs.length > 0 : !!selectedDocId) &&
    !isLoading;

  // ----- UI -----
  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {derivedMessages.map((message) => {
          const isAI = message.sender === "ai";
          const isGreeting = message.id === "welcome";

          return (
            <div
              key={message.id}
              className={`flex items-start space-x-3 animate-slide-up ${
                message.sender === "user" ? "flex-row-reverse space-x-reverse" : ""
              }`}
            >
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full ${
                  isAI ? "bg-gradient-accent shadow-glow" : "bg-gradient-primary"
                }`}
                aria-hidden
              >
                {isAI ? (
                  <Bot className="w-5 h-5 text-accent-foreground" />
                ) : (
                  <User className="w-5 h-5 text-primary-foreground" />
                )}
              </div>

              <div
                className={`max-w-[70%] p-4 rounded-lg shadow-soft break-words ${
                  isAI
                    ? isGreeting
                      ? "bg-blue-50 border border-blue-200 text-blue-900"
                      : "bg-card border border-border prose prose-sm dark:prose-invert"
                    : "bg-gradient-primary text-primary-foreground"
                }`}
              >
                {!!message.imageUrl && (
                  <img
                    src={message.imageUrl}
                    alt={message.imageAlt || "attachment"}
                    className="mb-3 rounded-lg border border-border max-w-full h-auto"
                  />
                )}
                {isAI ? (
                  <div className="whitespace-pre-wrap">
                    <ReactMarkdown>{message.text}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{message.text}</p>
                )}

                <p
                  className={`text-xs mt-2 ${isAI ? "text-muted-foreground" : "text-primary-foreground/70"}`}
                >
                  {new Date(message.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          );
        })}

        {isLoading && (
          <div className="flex items-start space-x-3">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-accent">
              <Bot className="w-5 h-5 text-accent-foreground animate-pulse" />
            </div>
            <div className="max-w-[70%] p-4 rounded-lg border border-border bg-card text-muted-foreground">
              Thinking…
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-border bg-card">
        {/* Selection row */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <label htmlFor="viz-file-select" className="sr-only">
              Select file context
            </label>
            <select
              id="viz-file-select"
              className="px-2 py-1 border rounded text-sm"
              value={isCombineMode ? "combine" : selectedDocId || ""}
              onChange={(e) => handleDropdownChange(e.target.value)}
              disabled={isLoading || fileDocuments.length === 0}
            >
              <option value="" disabled>
                Select a file
              </option>
              <option value="combine">🔗 Combine files</option>
              {fileDocuments.map((doc) => (
                <option key={doc.documentId} value={doc.documentId}>
                  {doc.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Context chips + IMAGE BUTTON + previews + Clear (image button before Clear) */}
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

          {/* Attach image button (BEFORE Clear) */}
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

          {/* small previews */}
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

          <button
            type="button"
            onClick={clearContext}
            className="ml-auto inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-transparent hover:bg-gray-100 border"
            title="Clear selection"
          >
            <X className="w-3 h-3" />
            Clear
          </button>
        </div>

        {/* Input row */}
        <form onSubmit={handleSubmit} className="flex space-x-3">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={inputText}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              onCompositionStart={() => (composingRef.current = true)}
              onCompositionEnd={() => (composingRef.current = false)}
              placeholder="Ask anything about your file(s) or image…"
              className="min-h-[2.5rem] max-h-[120px] resize-none pr-12 border-accent/30 focus:border-accent focus:ring-accent/20"
              rows={1}
              disabled={isLoading}
              aria-label="Ask a question"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <Sparkles className="w-4 h-4 text-muted-foreground" aria-hidden />
            </div>
          </div>
          <Button
            type="submit"
            disabled={!canSend}
            className="bg-gradient-accent hover:shadow-glow transition-all duration-200 disabled:opacity-50"
            aria-label="Send message"
          >
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>

      {/* Combine selection dialog */}
      <Dialog
        open={combineDialogOpen}
        onOpenChange={(open) => (open ? setCombineDialogOpen(true) : handleCombineCancel())}
      >
        <DialogContent className="max-w-md" aria-labelledby="combine-dialog-title" aria-describedby="combine-dialog-description">
          <DialogHeader>
            <DialogTitle id="combine-dialog-title">Select files to combine</DialogTitle>
          </DialogHeader>

          <div className="space-y-2 max-h-[260px] overflow-y-auto" id="combine-dialog-description">
            {fileDocuments.map((doc) => {
              const checked = selectedCombineDocs.includes(doc.documentId);
              return (
                <label key={doc.documentId} className="flex items-center space-x-2">
                  <Checkbox
                    checked={checked}
                    onCheckedChange={(val) => toggleCombineDoc(doc.documentId, val)}
                    disabled={isLoading}
                    aria-label={`Select ${doc.name}`}
                  />
                  <span className="text-sm">{doc.name}</span>
                </label>
              );
            })}
            {fileDocuments.length === 0 && (
              <p className="text-xs text-muted-foreground">No files available in this chat.</p>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={handleCombineCancel} disabled={isLoading}>
              Cancel
            </Button>
            <Button onClick={handleCombineConfirm} disabled={isLoading || selectedCombineDocs.length < 2}>
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div ref={messagesEndRef} />
    </div>
  );
};

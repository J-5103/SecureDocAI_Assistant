import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useLayoutEffect,
  useCallback,
} from "react";
import { Send, Bot, User, Sparkles, Link as LinkIcon, X, FileText } from "lucide-react";
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
  /** Some API paths return plotImageUrl instead of imageUrl */
  plotImageUrl?: string;
  imageAlt?: string;
}

interface VizDocument {
  documentId: string;
  name: string;
}

interface VizChatInterfaceProps {
  messages?: Message[];
  onSendMessage: (
    question: string,
    documentId?: string | null,
    combineDocs?: string[],
    images?: File[]
  ) => Promise<void> | void;
  documents?: VizDocument[];
  isLoading?: boolean;
}

// stable key helper to avoid duplicates when merging lists
const mkKey = (m: Message) =>
  m.id || `${m.sender}|${m.timestamp}|${(m.text || "").slice(0, 80)}`;

/* --------------------- helpers --------------------- */
const fileToDataUrl = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result || ""));
    r.onerror = reject;
    r.readAsDataURL(file);
  });

export const VizChatInterface = ({
  messages = [],
  onSendMessage,
  documents = [],
  isLoading = false,
}: VizChatInterfaceProps) => {
  const [inputText, setInputText] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [isCombineMode, setIsCombineMode] = useState(false);
  const [combineDialogOpen, setCombineDialogOpen] = useState(false);
  const [selectedCombineDocs, setSelectedCombineDocs] = useState<string[]>([]);

  // local attachments (images + PDFs)
  const [images, setImages] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]); // data URLs now
  const [pdfNames, setPdfNames] = useState<string[]>([]);
  const imgInputRef = useRef<HTMLInputElement>(null);

  // scrollable list container + sentinel
  const listRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composingRef = useRef(false);

  const fileDocuments = documents;

  /* --------------------- ALWAYS-SHOW WELCOME + MERGE --------------------- */
  const welcomeText = useMemo(() => {
    if (!fileDocuments.length) return "Hello! I'm ready to help. What would you like to know?";
    const names = fileDocuments.map((d) => `"${d.name}"`).join(", ");
    return `Hello! I'm ready to help you analyze ${names}. What would you like to know?`;
  }, [fileDocuments]);

  const welcomeMsg: Message = useMemo(
    () => ({
      id: "welcome",
      text: welcomeText,
      sender: "ai",
      timestamp: new Date().toISOString(),
    }),
    [welcomeText]
  );

  // local buffer keeps welcome + optimistic user messages
  const [localMsgs, setLocalMsgs] = useState<Message[]>([welcomeMsg]);

  // persist local messages per selected file so the user's question doesn't vanish
  const storageKey = useMemo(() => `vizLocal:${selectedDocId || "default"}`, [selectedDocId]);

  // load from storage when selected file (context) changes
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(storageKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length) {
          setLocalMsgs(parsed);
          return;
        }
      }
    } catch {}
    setLocalMsgs([welcomeMsg]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey]);

  // save local buffer whenever it changes
  useEffect(() => {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(localMsgs));
    } catch {}
  }, [storageKey, localMsgs]);

  // merge: keep welcome at top, then local, then server messages — de-duplicated
  const mergedMessages = useMemo(() => {
    const seen = new Set<string>();
    const base = [welcomeMsg, ...localMsgs.filter((m) => mkKey(m) !== "welcome")];
    const out: Message[] = [];
    for (const m of [...base, ...messages]) {
      const k = mkKey(m);
      if (!seen.has(k)) {
        seen.add(k);
        out.push(m);
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, welcomeMsg, localMsgs]);

  // refresh the welcome when docs change (keeps it visible/updated)
  useEffect(() => {
    setLocalMsgs((prev) => {
      const others = prev.filter((m) => mkKey(m) !== "welcome");
      return [welcomeMsg, ...others];
    });
  }, [welcomeMsg]);

  /* --------------------- names / chips --------------------- */
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

  /* --------------------- Sticky bottom helpers --------------------- */
  const isNearBottom = useCallback((px = 120) => {
    const el = listRef.current;
    if (!el) return false;
    return el.scrollHeight - (el.scrollTop + el.clientHeight) <= px;
  }, []);

  const scrollToBottom = useCallback((smooth = true) => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
    bottomRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "end" });
  }, []);

  // initial jump to bottom on mount/refresh
  useLayoutEffect(() => {
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        scrollToBottom(false);
        setTimeout(() => scrollToBottom(false), 0);
        setTimeout(() => scrollToBottom(false), 150);
        setTimeout(() => scrollToBottom(false), 350);
      })
    );
  }, [scrollToBottom]);

  // keep pinned when new messages append (user/ai)
  useEffect(() => {
    if (isNearBottom()) scrollToBottom(true);
  }, [mergedMessages.length, isLoading, isNearBottom, scrollToBottom]);

  // stay pinned on DOM growth (images/markdown)
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;

    const mo =
      typeof MutationObserver !== "undefined"
        ? new MutationObserver(() => {
            if (isNearBottom()) scrollToBottom(true);
          })
        : null;

    const ro =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => {
            if (isNearBottom()) scrollToBottom(true);
          })
        : null;

    mo?.observe?.(el, { childList: true, subtree: true });
    ro?.observe?.(el);

    return () => {
      mo?.disconnect?.();
      ro?.disconnect?.();
    };
  }, [isNearBottom, scrollToBottom]);

  // ensure a valid selected doc WHEN documents change (avoids loops)
  useEffect(() => {
    if (!fileDocuments.length) {
      setSelectedDocId("");
      setIsCombineMode(false);
      setSelectedCombineDocs([]);
      return;
    }
    const stillExists = selectedDocId && fileDocuments.some((d) => d.documentId === selectedDocId);
    if (!isCombineMode && (!selectedDocId || !stillExists)) {
      setSelectedDocId(fileDocuments[0].documentId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileDocuments]); // intentionally only when documents change

  /* --------------------- attachment helpers --------------------- */
  const looksLikeImage = useCallback(
    (f: File) =>
      (f.type && f.type.startsWith("image/")) ||
      /\.(png|jpe?g|webp|gif|bmp|tiff)$/i.test(f.name || ""),
    []
  );

  const looksLikePdf = useCallback(
    (f: File) =>
      (f.type && f.type.toLowerCase() === "application/pdf") || /\.pdf$/i.test(f.name || ""),
    []
  );

  const addImages = useCallback(
    async (files: FileList | null) => {
      if (!files) return;

      const toSend: File[] = [];
      const newPdfNames: string[] = [];
      const imgFiles: File[] = [];

      for (const f of Array.from(files)) {
        if (looksLikeImage(f)) {
          toSend.push(f);
          imgFiles.push(f);
        } else if (looksLikePdf(f)) {
          toSend.push(f); // still send to backend (first page → PNG)
          newPdfNames.push(f.name);
        }
      }
      if (toSend.length === 0) return;

      // build data URLs for previews (never revoked, so they don't break)
      const dataUrls = await Promise.all(imgFiles.map(fileToDataUrl));

      setImages((p) => [...p, ...toSend]);
      setPreviews((p) => [...p, ...dataUrls]);
      setPdfNames((p) => [...p, ...newPdfNames]);

      if (imgInputRef.current) imgInputRef.current.value = "";
      if (isNearBottom()) requestAnimationFrame(() => scrollToBottom(true));
    },
    [isNearBottom, looksLikeImage, looksLikePdf, scrollToBottom]
  );

  const removePreview = useCallback(
    (idx: number) => {
      setPreviews((prev) => {
        const copy = [...prev];
        copy.splice(idx, 1);
        return copy;
      });
      // remove the corresponding *image* file (not PDFs)
      setImages((prev) => {
        const imgIndexes = prev
          .map((f, i) => ({ i, f }))
          .filter(({ f }) => looksLikeImage(f))
          .map(({ i }) => i);
        const targetFileIndex = imgIndexes[idx];
        if (targetFileIndex == null) return prev;
        const copy = [...prev];
        copy.splice(targetFileIndex, 1);
        return copy;
      });
    },
    [looksLikeImage]
  );

  const removePdf = useCallback(
    (idx: number) => {
      const nameToRemove = pdfNames[idx];
      setPdfNames((prev) => {
        const copy = [...prev];
        copy.splice(idx, 1);
        return copy;
      });
      setImages((prev) => {
        const pos = prev.findIndex((f) => looksLikePdf(f) && f.name === nameToRemove);
        if (pos < 0) return prev;
        const copy = [...prev];
        copy.splice(pos, 1);
        return copy;
      });
    },
    [pdfNames, looksLikePdf]
  );

  const clearImages = useCallback(() => {
    setPreviews([]);
    setPdfNames([]);
    setImages([]);
    if (imgInputRef.current) imgInputRef.current.value = "";
  }, []);

  /* --------------------- Handlers --------------------- */
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = inputText.trim();
      // allow sending if there's text OR attachments
      if ((!trimmed && images.length === 0) || isLoading) return;

      // optimistic echo: show the user's question immediately
      if (trimmed) {
        const optimistic: Message = {
          id: `local-${Date.now()}`,
          text: trimmed,
          sender: "user",
          timestamp: new Date().toISOString(),
        };
        setLocalMsgs((prev) => [...prev, optimistic]);
      }

      if (isNearBottom()) {
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            scrollToBottom(true);
            setTimeout(() => scrollToBottom(true), 0);
          })
        );
      }

      try {
        if (isCombineMode && selectedCombineDocs.length > 0) {
          await onSendMessage(trimmed, null, selectedCombineDocs, images);
        } else if (!isCombineMode && selectedDocId) {
          await onSendMessage(trimmed, selectedDocId, undefined, images);
        } else if (images.length > 0) {
          // allow image/PDF-only ask without forcing a file selection
          await onSendMessage(trimmed, null, undefined, images);
        } else {
          return;
        }
      } finally {
        // ALWAYS clear the composer after submit
        setInputText("");
        clearImages();
        if (textareaRef.current) textareaRef.current.style.height = "auto";
        requestAnimationFrame(() => setTimeout(() => scrollToBottom(true), 0));
      }
    },
    [
      inputText,
      images,
      isLoading,
      isCombineMode,
      selectedCombineDocs,
      selectedDocId,
      onSendMessage,
      isNearBottom,
      scrollToBottom,
      clearImages,
    ]
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

  const handleDropdownChange = useCallback(
    (value: string) => {
      if (value === "combine") setCombineDialogOpen(true);
      else {
        setIsCombineMode(false);
        setSelectedCombineDocs([]);
        setSelectedDocId(value);
      }
      if (isNearBottom()) requestAnimationFrame(() => scrollToBottom(true));
    },
    [isNearBottom, scrollToBottom]
  );

  // shadcn Checkbox emits boolean | "indeterminate"
  const toggleCombineDoc = useCallback(
    (docId: string, checked?: boolean | "indeterminate") => {
      const willSelect =
        checked === true || (checked === undefined && !selectedCombineDocs.includes(docId));
      setSelectedCombineDocs((prev) =>
        willSelect ? [...prev, docId] : prev.filter((id) => id !== docId)
      );
    },
    [selectedCombineDocs]
  );

  const handleCombineConfirm = useCallback(() => {
    if (selectedCombineDocs.length < 2) {
      alert("Please select at least two files to combine.");
      return;
    }
    setIsCombineMode(true);
    setSelectedDocId("combine");
    setCombineDialogOpen(false);
    if (isNearBottom()) requestAnimationFrame(() => scrollToBottom(true));
  }, [selectedCombineDocs, isNearBottom, scrollToBottom]);

  const handleCombineCancel = useCallback(() => {
    setCombineDialogOpen(false);
    if (!isCombineMode && !selectedDocId && fileDocuments[0]) {
      setSelectedDocId(fileDocuments[0].documentId);
    }
  }, [fileDocuments, isCombineMode, selectedDocId]);

  const clearContext = useCallback(() => {
    setIsCombineMode(false);
    setSelectedCombineDocs([]);
    setSelectedDocId(fileDocuments[0]?.documentId || "");
    // reset local bubbles to just the welcome and clear persisted cache for this context
    try {
      sessionStorage.removeItem(storageKey);
    } catch {}
    setLocalMsgs([welcomeMsg]);
    clearImages();
    if (isNearBottom()) requestAnimationFrame(() => scrollToBottom(true));
  }, [fileDocuments, clearImages, isNearBottom, scrollToBottom, storageKey, welcomeMsg]);

  // allow send if text OR attachments present; if no attachments, require context
  const canSend =
    (inputText.trim() !== "" || images.length > 0) &&
    (images.length > 0 || (isCombineMode ? selectedCombineDocs.length > 0 : !!selectedDocId)) &&
    !isLoading;

  /* --------------------- UI --------------------- */
  return (
    <div className="flex flex-col h-full bg-background">
      {/* messages list */}
      <div ref={listRef} className="chat-body flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
        {mergedMessages.map((message) => {
          const isAI = message.sender === "ai";
          const isGreeting = message.id === "welcome";
          // unified width for visual alignment of Q&A
          const bubbleMaxWidth = "max-w-[75%]";
          const displayImageUrl = message.imageUrl || message.plotImageUrl;

          return (
            <div
              key={mkKey(message)}
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
                className={`${bubbleMaxWidth} p-4 rounded-lg shadow-soft break-words ${
                  isAI
                    ? isGreeting
                      ? "bg-blue-50 border border-blue-200 text-blue-900"
                      : "bg-card border border-border prose prose-sm dark:prose-invert"
                    : "bg-gradient-primary text-primary-foreground"
                }`}
              >
                {!!displayImageUrl && (
                  <a
                    href={displayImageUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-block mb-3"
                    title="Open full size"
                  >
                    <img
                      src={displayImageUrl}
                      alt={message.imageAlt || "attachment"}
                      loading="lazy"
                      className={
                        isAI
                          ? "rounded-lg border border-border w-full max-w-[1100px] h-auto object-contain bg-white"
                          : "rounded-lg border border-border max-w-[220px] sm:max-w-[320px] max-h-48 sm:max-h-64 w-auto h-auto object-contain bg-white"
                      }
                      onLoad={() => {
                        if (isNearBottom()) requestAnimationFrame(() => scrollToBottom(true));
                      }}
                    />
                  </a>
                )}

                {isAI ? (
                  <div className="whitespace-pre-wrap">
                    <ReactMarkdown
                      components={{
                        img: (props: any) => (
                          <img
                            {...props}
                            loading="lazy"
                            className={
                              `w-full max-w-[1100px] h-auto object-contain rounded-lg border border-border my-2 bg-white ` +
                              (props?.className || "")
                            }
                            onLoad={() => {
                              if (isNearBottom()) requestAnimationFrame(() => scrollToBottom(true));
                            }}
                          />
                        ),
                      }}
                    >
                      {message.text}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{message.text}</p>
                )}

                <p
                  className={`text-xs mt-2 ${
                    isAI ? "text-muted-foreground" : "text-primary-foreground/70"
                  }`}
                >
                  {(() => {
                    try {
                      return new Date(message.timestamp).toLocaleTimeString();
                    } catch {
                      return "";
                    }
                  })()}
                </p>
              </div>
            </div>
          );
        })}

        {isLoading && (
          <div className="flex items-start space-x-3" aria-live="polite" aria-busy="true">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-accent">
              <Bot className="w-5 h-5 text-accent-foreground animate-pulse" />
            </div>
            <div className="max-w-[75%] p-4 rounded-lg border border-border bg-card">
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2"></span>
                </span>
                Thinking…
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* composer */}
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

        {/* Context chips + ATTACH + previews + Clear */}
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

          {/* Attach button (images + PDF) */}
          <input
            ref={imgInputRef}
            type="file"
            accept="image/*,.pdf"
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
            title="Attach image(s) or PDF"
            aria-label="Attach image(s) or PDF"
          >
            📎
          </Button>

          {/* image previews (data URLs) */}
          {previews.length > 0 && (
            <div className="flex items-center gap-2 overflow-x-auto">
              {previews.map((u, i) => (
                <div key={u} className="relative w-[44px] h-[34px] rounded border overflow-hidden">
                  <img
                    src={u}
                    alt={`preview-${i}`}
                    className="w-full h-full object-cover"
                    onLoad={() => {
                      if (isNearBottom()) requestAnimationFrame(() => scrollToBottom(true));
                    }}
                  />
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

          {/* pdf chips */}
          {pdfNames.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              {pdfNames.map((name, i) => (
                <span
                  key={`${name}-${i}`}
                  className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200"
                  title={name}
                >
                  <FileText className="w-3 h-3" />
                  <span className="max-w-[120px] truncate">{name}</span>
                  <button
                    type="button"
                    onClick={() => removePdf(i)}
                    className="ml-1 rounded-full hover:bg-white/60"
                    aria-label={`Remove ${name}`}
                    title="Remove"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
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
              placeholder="Ask anything about your file(s) or image/PDF…"
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
        <DialogContent
          className="max-w-md"
          aria-labelledby="combine-dialog-title"
          aria-describedby="combine-dialog-description"
        >
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
    </div>
  );
};

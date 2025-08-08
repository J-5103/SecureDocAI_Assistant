import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Sparkles, Link as LinkIcon } from "lucide-react";
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
}

interface Document {
  documentId: string;
  name: string;
}

interface ChatInterfaceProps {
  messages?: Message[];
  onSendMessage: (question: string, documentId: string | null, combineDocs?: string[]) => void;
  documents?: Document[];
  isLoading?: boolean; // optional typing indicator
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
  const [selectedCombineDocs, setSelectedCombineDocs] = useState<string[]>([]);
  const [selectedCombineDocNames, setSelectedCombineDocNames] = useState<string[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!selectedDocId && documents.length > 0) {
      setSelectedDocId(documents[0].documentId);
    }
  }, [documents]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || isLoading) return;

    if (isCombineMode && selectedCombineDocNames.length > 0) {
      onSendMessage(trimmed, null, selectedCombineDocNames);
    } else if (!isCombineMode && selectedDocId) {
      onSendMessage(trimmed, selectedDocId);
    }

    setInputText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
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
      // open selection dialog
      setCombineDialogOpen(true);
    } else {
      // switch to single-doc mode
      setIsCombineMode(false);
      setSelectedCombineDocs([]);
      setSelectedCombineDocNames([]);
      setSelectedDocId(value);
    }
  };

  const toggleCombineDoc = (docId: string, docName: string, checked?: boolean | string) => {
    // shadcn Checkbox passes true/false/"indeterminate"
    const willSelect = checked === true || (checked === undefined && !selectedCombineDocs.includes(docId));

    setSelectedCombineDocs((prev) =>
      willSelect ? [...prev, docId] : prev.filter((id) => id !== docId)
    );
    setSelectedCombineDocNames((prev) =>
      willSelect ? [...prev, docName] : prev.filter((name) => name !== docName)
    );
  };

  const handleCombineConfirm = () => {
    if (selectedCombineDocs.length < 2) {
      alert("Please select at least two PDFs to combine.");
      return;
    }
    setIsCombineMode(true);
    setSelectedDocId("combine");
    setCombineDialogOpen(false);
  };

  const handleCombineCancel = () => {
    // If user cancels without confirming, keep previous mode & selection untouched.
    setCombineDialogOpen(false);
    if (!isCombineMode) {
      // ensure dropdown doesn't stay on "combine"
      if (!selectedDocId && documents[0]) {
        setSelectedDocId(documents[0].documentId);
      }
    }
  };

  const canSend =
    inputText.trim() !== "" &&
    (isCombineMode ? selectedCombineDocNames.length > 0 : !!selectedDocId) &&
    !isLoading;

  return (
    <div className="flex flex-col h-full bg-background">
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
                message.sender === "ai" ? "bg-gradient-accent shadow-glow" : "bg-gradient-primary"
              }`}
            >
              {message.sender === "ai" ? (
                <Bot className="w-5 h-5 text-accent-foreground" />
              ) : (
                <User className="w-5 h-5 text-primary-foreground" />
              )}
            </div>

            <div
              className={`max-w-[70%] p-4 rounded-lg shadow-soft break-words ${
                message.sender === "ai"
                  ? "bg-card border border-border prose prose-sm dark:prose-invert"
                  : "bg-gradient-primary text-primary-foreground"
              }`}
            >
              {message.sender === "ai" ? (
  <div className="whitespace-pre-wrap">
    <ReactMarkdown>{message.text}</ReactMarkdown>
  </div>
) : (
  <p className="whitespace-pre-wrap">{message.text}</p>
)}

              <p
                className={`text-xs mt-2 ${
                  message.sender === "ai" ? "text-muted-foreground" : "text-primary-foreground/70"
                }`}
              >
                {new Date(message.timestamp).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

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
            <select
              className="px-2 py-1 border rounded text-sm"
              value={isCombineMode ? "combine" : (selectedDocId || "")}
              onChange={(e) => handleDropdownChange(e.target.value)}
              disabled={isLoading}
            >
              <option value="" disabled>
                Select a document
              </option>
              <option value="combine">🔗 Combine PDFs</option>
              {documents.map((doc) => (
                <option key={doc.documentId} value={doc.documentId}>
                  {doc.name}
                </option>
              ))}
            </select>

            {isCombineMode && (
              <span className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded bg-blue-50 text-blue-700 border border-blue-200">
                <LinkIcon className="w-3 h-3" />
                {selectedCombineDocNames.length} selected
              </span>
            )}
          </div>
        </div>

        {/* Input row */}
        <form onSubmit={handleSubmit} className="flex space-x-3">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={inputText}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your document(s)…"
              className="min-h-[2.5rem] max-h-[120px] resize-none pr-12 border-accent/30 focus:border-accent focus:ring-accent/20"
              rows={1}
              disabled={isLoading}
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <Sparkles className="w-4 h-4 text-muted-foreground" />
            </div>
          </div>
          <Button
            type="submit"
            disabled={!canSend}
            className="bg-gradient-accent hover:shadow-glow transition-all duration-200 disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>

      {/* Combine selection dialog */}
      <Dialog open={combineDialogOpen} onOpenChange={(open) => (open ? setCombineDialogOpen(true) : handleCombineCancel())}>
        <DialogContent className="max-w-md" aria-labelledby="combine-dialog-title" aria-describedby="combine-dialog-description">
          <DialogHeader>
            <DialogTitle id="combine-dialog-title">Select PDFs to Combine</DialogTitle>
          </DialogHeader>

          <div className="space-y-2 max-h-[260px] overflow-y-auto" id="combine-dialog-description">
            {documents.map((doc) => {
              const checked = selectedCombineDocs.includes(doc.documentId);
              return (
                <label key={doc.documentId} className="flex items-center space-x-2">
                  <Checkbox
                    checked={checked}
                    onCheckedChange={(val) => toggleCombineDoc(doc.documentId, doc.name, val)}
                  />
                  <span className="text-sm">{doc.name}</span>
                </label>
              );
            })}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={handleCombineCancel}>
              Cancel
            </Button>
            <Button onClick={handleCombineConfirm} disabled={selectedCombineDocs.length < 2}>
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

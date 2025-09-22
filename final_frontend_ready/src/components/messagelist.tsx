// src/components/MessageList.tsx
import React, { useEffect, useMemo, useRef } from "react";

export type ChatMessage = {
  id: string;
  text: string;
  sender: "user" | "ai";
  at: string;              // ISO datetime
  thinking?: boolean;      // show spinner instead of text
};

type Props = {
  messages: ChatMessage[];
  className?: string;
  emptyHint?: string;
  errorText?: string;
};

/**
 * Basic fenced-code parser (```lang\n...\n``` → <pre>).
 * This avoids pulling in markdown libs while keeping SQL blocks readable.
 */
function useCodeBlocks(text: string) {
  return useMemo(() => {
    const parts: Array<{ type: "code" | "text"; lang?: string; value: string }> = [];
    const re = /```(\w+)?\s*([\s\S]*?)```/g;
    let lastIdx = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      if (m.index > lastIdx) {
        parts.push({ type: "text", value: text.slice(lastIdx, m.index) });
      }
      parts.push({ type: "code", lang: (m[1] || "").toLowerCase(), value: m[2] || "" });
      lastIdx = re.lastIndex;
    }
    if (lastIdx < text.length) {
      parts.push({ type: "text", value: text.slice(lastIdx) });
    }
    return parts;
  }, [text]);
}

const Bubble: React.FC<{ msg: ChatMessage }> = ({ msg }) => {
  const isUser = msg.sender === "user";
  const blocks = useCodeBlocks(msg.text || "");

  return (
    <div
      className={`px-4 py-3 rounded-xl shadow-sm max-w-[85%] ${
        isUser
          ? "ml-auto bg-gradient-primary text-primary-foreground"
          : "bg-muted text-foreground"
      }`}
    >
      {msg.thinking ? (
        <div className="text-sm inline-flex items-center gap-2">
          <span
            className="inline-block w-3 h-3 rounded-full border-2 border-muted-foreground/40 border-t-blue-500 animate-spin"
            aria-hidden
          />
          Thinking…
        </div>
      ) : (
        <div className="text-sm whitespace-pre-wrap leading-relaxed">
          {blocks.map((b, i) =>
            b.type === "code" ? (
              <pre
                key={`code-${i}`}
                className="my-2 rounded-lg border border-border bg-background/60 p-3 overflow-x-auto"
                aria-label={b.lang ? `${b.lang} code` : "code"}
              >
                <code>{b.value}</code>
              </pre>
            ) : (
              <span key={`txt-${i}`}>{b.value}</span>
            )
          )}
        </div>
      )}
    </div>
  );
};

const MessageList: React.FC<Props> = ({ messages, className, emptyHint, errorText }) => {
  const threadRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    const el = threadRef.current;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  return (
    <div ref={threadRef} className={`overflow-auto px-5 py-4 space-y-3 ${className || ""}`}>
      {messages.length === 0 && !errorText && (
        <div className="text-sm text-muted-foreground">
          {emptyHint || "No messages yet. Ask your first question below."}
        </div>
      )}

      {messages.map((m) => (
        <Bubble key={m.id} msg={m} />
      ))}

      {!!errorText && <div className="text-xs text-red-600">{errorText}</div>}
    </div>
  );
};

export default MessageList;

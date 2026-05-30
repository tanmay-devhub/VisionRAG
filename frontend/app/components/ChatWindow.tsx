"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import { queryDocuments, Source } from "@/lib/api";
import UploadPanel from "./UploadPanel";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8081";

interface Message {
  role: "user" | "ai";
  text: string;
  sources?: Source[];
}

const SUGGESTIONS = [
  "What entities appear across multiple documents?",
  "Summarize the key findings in the uploaded materials",
  "Which figures are most relevant to the main topic?",
  "What relationships exist between the extracted entities?",
];

// ── Icons ──────────────────────────────────────────────────────────────────────

function SparklesIcon() {
  return (
    <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/>
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="m22 2-7 20-4-9-9-4z"/><path d="M22 2 11 13"/>
    </svg>
  );
}

function ChevronIcon({ rotated }: { rotated?: boolean }) {
  return (
    <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"
      className="chev" style={{ transform: rotated ? "rotate(-90deg)" : undefined, transition: "transform .2s" }}>
      <path d="m6 9 6 6 6-6"/>
    </svg>
  );
}

function LinkIcon() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <rect width={18} height={18} x={3} y={3} rx={2} ry={2}/>
      <circle cx={9} cy={9} r={2}/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
    </svg>
  );
}

function VideoIcon() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <path d="m22 8-6 4 6 4V8z"/><rect width={14} height={12} x={2} y={6} rx={2} ry={2}/>
    </svg>
  );
}

function TableIcon() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <path d="M3 3h18v18H3zM3 9h18M3 15h18M9 3v18M15 3v18"/>
    </svg>
  );
}

function FileTextIcon() {
  return (
    <svg width={11} height={11} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>
    </svg>
  );
}

// ── Source card ────────────────────────────────────────────────────────────────

function scoreClass(s: number) {
  if (s >= 0.85) return "score-hi";
  if (s >= 0.7)  return "score-mid";
  return "score-lo";
}

function ChunkBadge({ type }: { type: string }) {
  const map: Record<string, { cls: string; label: string; icon: JSX.Element }> = {
    figure: { cls: "chunk-figure", label: "Figure",  icon: <ImageIcon />    },
    video:  { cls: "chunk-video",  label: "Video",   icon: <VideoIcon />    },
    table:  { cls: "chunk-table",  label: "Table",   icon: <TableIcon />    },
    text:   { cls: "chunk-text",   label: "Text",    icon: <FileTextIcon /> },
  };
  const m = map[type] ?? map.text;
  return (
    <span className={"chunk-badge " + m.cls}>
      {m.icon} {m.label}
    </span>
  );
}

function SourceCard({ src }: { src: Source }) {
  const hasImage = !!src.image_url;
  const ts = src.timestamp_ms != null
    ? `${Math.floor(src.timestamp_ms / 1000 / 60)}:${String(Math.floor((src.timestamp_ms / 1000) % 60)).padStart(2, "0")}`
    : null;
  const title = src.caption || src.text.slice(0, 80);
  const filename = src.source.split("/").pop() ?? src.source;

  return (
    <div className="source-card">
      <div className="sc-thumb">
        {hasImage
          ? <img src={src.image_url!} alt={src.caption ?? "source"} />
          : (
            <div style={{
              position: "absolute", inset: 0,
              background: `var(--surface-2)`,
              display: "grid", placeItems: "center",
            }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 9, fontWeight: 600, letterSpacing: ".1em", color: "var(--text-3)" }}>
                {(src.chunk_type ?? src.type ?? "TEXT").toUpperCase()}
              </span>
            </div>
          )
        }
        {ts && <span className="sc-ts">{ts}</span>}
      </div>
      <div className="sc-body">
        <div className="sc-row1">
          <ChunkBadge type={src.chunk_type ?? src.type ?? "text"} />
          <span className={"score-badge " + scoreClass(src.score)}>
            {Math.round(src.score * 100)}%
          </span>
        </div>
        <div className="sc-title">{title}</div>
        <div className="sc-src">
          <LinkIcon />
          {filename.length > 28 ? filename.slice(0, 26) + "…" : filename}
          {src.page_number != null && <span style={{ marginLeft: 4, color: "var(--text-3)" }}>p.{src.page_number + 1}</span>}
        </div>
      </div>
    </div>
  );
}

// ── AI message ─────────────────────────────────────────────────────────────────

function AiMessage({ m }: { m: Message }) {
  const [open, setOpen] = useState(true);

  function renderText(text: string) {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((p, i) =>
      p.startsWith("**") && p.endsWith("**")
        ? <b key={i}>{p.slice(2, -2)}</b>
        : p
    );
  }

  return (
    <div className="msg-row ai">
      <div className="ai-block">
        <div className="ai-row">
          <div className="ai-avatar"><SparklesIcon /></div>
          <div className="bubble ai">{renderText(m.text)}</div>
        </div>
        {m.sources && m.sources.length > 0 && (
          <div className="sources">
            <div
              className={"sources-head" + (!open ? " collapsed" : "")}
              onClick={() => setOpen(o => !o)}
            >
              <ChevronIcon rotated={!open} />
              Grounded sources
              <span className="sources-count">{m.sources.length}</span>
            </div>
            {open && (
              <div className="source-grid">
                {m.sources.map((s, i) => <SourceCard key={i} src={s} />)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput]       = useState("");
  const [thinking, setThinking] = useState(false);
  const streamRef = useRef<HTMLDivElement>(null);
  const taRef     = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (streamRef.current) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight;
    }
  }, [messages, thinking]);

  const send = useCallback(async (text?: string) => {
    const q = (text ?? input).trim();
    if (!q || thinking) return;
    setMessages(prev => [...prev, { role: "user", text: q }]);
    setInput("");
    if (taRef.current) { taRef.current.style.height = "auto"; }
    setThinking(true);
    try {
      const res = await queryDocuments(q, 6);
      setMessages(prev => [...prev, { role: "ai", text: res.answer, sources: res.sources }]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed";
      setMessages(prev => [...prev, { role: "ai", text: `Error: ${msg}` }]);
    } finally {
      setThinking(false);
    }
  }, [input, thinking]);

  return (
    <div className="chat-layout page-anim">
      {/* ── Left: upload panel ── */}
      <aside className="upload-panel">
        <div className="upload-head">
          <h2>Sources</h2>
          <p>Upload images, PDFs and videos to ground answers.</p>
        </div>
        <div className="upload-body">
          <UploadPanel />
        </div>
      </aside>

      {/* ── Right: chat ── */}
      <section className="chat-side">
        <div className="chat-stream" ref={streamRef}>
          <div className="chat-inner">
            {messages.length === 0 && !thinking && (
              <div className="empty" style={{ height: "calc(100vh - 260px)" }}>
                <div className="empty-art">
                  <SparklesIcon />
                </div>
                <h3>Ask about your sources</h3>
                <p>Upload a document or image, then ask a question to get grounded answers with citations.</p>
              </div>
            )}

            {messages.map((m, i) =>
              m.role === "user"
                ? <div className="msg-row user" key={i}><div className="bubble user">{m.text}</div></div>
                : <AiMessage key={i} m={m} />
            )}

            {thinking && (
              <div className="msg-row ai">
                <div className="ai-block">
                  <div className="ai-row">
                    <div className="ai-avatar"><SparklesIcon /></div>
                    <div className="bubble ai">
                      <div className="typing"><span /><span /><span /></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="composer-wrap">
          {messages.length === 0 && (
            <div className="suggest-row">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} className="suggest-chip" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          )}
          <div className="composer">
            <textarea
              ref={taRef}
              rows={1}
              placeholder="Ask about your sources…"
              value={input}
              onChange={e => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = Math.min(120, e.target.scrollHeight) + "px";
              }}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
              }}
            />
            <button className="send-btn" disabled={!input.trim() || thinking} onClick={() => send()}>
              <SendIcon />
            </button>
          </div>
          <div className="composer-hint">Answers are grounded in your uploaded sources · Enter to send</div>
        </div>
      </section>
    </div>
  );
}

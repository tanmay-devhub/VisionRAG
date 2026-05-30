"use client";
import { useEffect, useState } from "react";
import ChatPage from "./components/ChatWindow";
import GraphView from "./components/GraphView";
import FilesPage from "./components/FilesPage";
import StatusPage from "./components/StatusPage";
import { FileEntry } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8081";

type Tab = "chat" | "graph" | "files" | "status";

function LayersIcon({ size = 17 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>
    </svg>
  );
}

function TabIcon({ tab, size = 16 }: { tab: Tab; size?: number }) {
  const props = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (tab === "chat") return <svg {...props}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>;
  if (tab === "graph") return <svg {...props}><circle cx="5" cy="6" r="2.4"/><circle cx="19" cy="7" r="2.4"/><circle cx="12" cy="17" r="2.6"/><path d="M7.1 7 9.8 15M16.7 8.4 13.2 15"/></svg>;
  if (tab === "files") return <svg {...props}><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M3 11h18"/></svg>;
  return <svg {...props}><path d="M3 12h3.5l2-6 4 13 3-9 1.8 2H21"/></svg>;
}

function SunIcon() {
  return <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>;
}
function MoonIcon() {
  return <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><path d="M20 13.6A8 8 0 1 1 10.4 4 6.5 6.5 0 0 0 20 13.6Z"/></svg>;
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("chat");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [visionLabel, setVisionLabel] = useState("Vision");
  const [visionBackend, setVisionBackend] = useState("ollama");
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [graphInitFile, setGraphInitFile] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => {
        if (d?.vision_label) setVisionLabel(d.vision_label);
        if (d?.vision_backend) setVisionBackend(d.vision_backend);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // files are fetched by FilesPage on mount

  const openGraph = (filename: string) => {
    setGraphInitFile(filename);
    setTab("graph");
  };

  const TABS: { k: Tab; label: string }[] = [
    { k: "chat", label: "Chat" },
    { k: "graph", label: "Graph" },
    { k: "files", label: "Files" },
    { k: "status", label: "Status" },
  ];

  const badgeColor =
    visionBackend === "gemini" ? { bg: "var(--blue-soft)", color: "#1d4ed8", dot: "#2563eb" } :
    visionBackend === "openai" ? { bg: "var(--green-soft)", color: "#15803d", dot: "#16a34a" } :
    { bg: "var(--purple-soft)", color: "var(--purple)", dot: "var(--purple)" };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* ── Header ── */}
      <header style={{
        height: "var(--header-h)", flexShrink: 0,
        display: "flex", alignItems: "center", gap: 20, padding: "0 22px",
        background: theme === "dark" ? "rgba(15,23,39,0.82)" : "rgba(255,255,255,0.86)",
        backdropFilter: "saturate(1.4) blur(14px)",
        borderBottom: "1px solid var(--border)",
        zIndex: 40, position: "relative",
      }}>
        {/* Brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 11, userSelect: "none" }}>
          <div style={{
            width: 30, height: 30, borderRadius: 9,
            background: "linear-gradient(145deg, var(--accent) 0%, var(--accent-ink) 100%)",
            display: "grid", placeItems: "center",
            boxShadow: "inset 0 1px 0 rgba(255,255,255,.25), var(--shadow-xs)",
          }}>
            <LayersIcon size={17} />
          </div>
          <span style={{ fontSize: 16.5, fontWeight: 650, letterSpacing: "-0.025em", color: "var(--text)" }}>
            <span style={{ color: "var(--accent)" }}>Vision</span>RAG
          </span>
        </div>

        {/* Backend badge */}
        <span style={{
          display: "inline-flex", alignItems: "center", gap: 7,
          height: 28, padding: "0 11px 0 9px", borderRadius: "var(--r-pill)",
          background: badgeColor.bg,
          border: `1px solid rgba(${visionBackend === "gemini" ? "37,99,235" : visionBackend === "openai" ? "22,163,74" : "124,58,237"}, 0.18)`,
          color: badgeColor.color,
          fontSize: 12, fontWeight: 600, letterSpacing: "-0.01em", whiteSpace: "nowrap",
        }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: badgeColor.dot, boxShadow: `0 0 0 3px ${badgeColor.dot}28` }} />
          {visionLabel}
        </span>

        {/* Tabs */}
        <nav style={{ display: "flex", alignItems: "center", gap: 2, marginLeft: 8 }}>
          {TABS.map(t => (
            <button key={t.k} onClick={() => { setTab(t.k); if (t.k !== "graph") setGraphInitFile(null); }}
              style={{
                display: "inline-flex", alignItems: "center", gap: 8,
                height: 36, padding: "0 14px", borderRadius: "var(--r-sm)",
                color: tab === t.k ? "var(--accent-ink)" : "var(--text-2)",
                background: tab === t.k ? "var(--accent-soft)" : "transparent",
                fontSize: 13.5, fontWeight: 550, letterSpacing: "-0.01em",
                transition: "background .15s, color .15s",
              }}>
              <TabIcon tab={t.k} size={16} />
              {t.label}
              {t.k === "files" && files.length > 0 && (
                <span style={{
                  fontSize: 11, fontWeight: 600, padding: "1px 6px", borderRadius: "var(--r-pill)",
                  background: tab === "files" ? "rgba(var(--accent-rgb),.16)" : "var(--surface-3)",
                  color: tab === "files" ? "var(--accent-ink)" : "var(--text-2)",
                }}>{files.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div style={{ flex: 1 }} />

        {/* Theme toggle */}
        <button onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
          style={{
            width: 34, height: 34, borderRadius: "var(--r-sm)",
            display: "grid", placeItems: "center",
            color: "var(--text-2)", border: "1px solid var(--border)",
            background: "var(--surface)", transition: "background .15s",
          }}>
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>

        {/* Avatar */}
        <div style={{
          width: 32, height: 32, borderRadius: "50%",
          background: "linear-gradient(140deg, #6d7686, #444b58)",
          color: "#fff", display: "grid", placeItems: "center",
          fontSize: 12, fontWeight: 600, flexShrink: 0,
        }}>AI</div>
      </header>

      {/* ── Page ── */}
      <main style={{ flex: 1, minHeight: 0, position: "relative", overflow: "hidden" }}>
        {tab === "chat" && <ChatPage />}
        {tab === "graph" && <GraphView initFile={graphInitFile} />}
        {tab === "files" && <FilesPage files={files} setFiles={setFiles} onViewGraph={openGraph} />}
        {tab === "status" && <StatusPage />}
      </main>
    </div>
  );
}

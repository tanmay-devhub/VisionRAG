"use client";
import { useEffect, useState } from "react";
import UploadPanel from "./components/UploadPanel";
import ChatWindow from "./components/ChatWindow";
import { getFiles, deleteGraph, FileEntry } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8081";

type Tab = "chat" | "files";

export default function Home() {
  const [visionLabel, setVisionLabel] = useState<string>("Vision");
  const [visionBackend, setVisionBackend] = useState<string>("ollama");
  const [tab, setTab]                 = useState<Tab>("chat");
  const [files, setFiles]             = useState<FileEntry[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => {
        if (data?.vision_label)   setVisionLabel(data.vision_label);
        if (data?.vision_backend) setVisionBackend(data.vision_backend);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (tab === "files") {
      setLoadingFiles(true);
      getFiles()
        .then(d => setFiles(d.files))
        .catch(() => {})
        .finally(() => setLoadingFiles(false));
    }
  }, [tab]);

  const handleDelete = async (filename: string) => {
    if (!confirm(`Delete all data for "${filename}"?`)) return;
    await deleteGraph();
    setFiles(prev => prev.filter(f => f.filename !== filename));
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xl font-bold text-gray-900 tracking-tight">VisionRAG</span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            visionBackend === "gemini" ? "bg-blue-100 text-blue-700" :
            visionBackend === "openai" ? "bg-green-100 text-green-700" :
            "bg-teal-100 text-teal-700"
          }`}>
            {visionLabel}
          </span>
        </div>

        <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
          {(["chat", "files"] as Tab[]).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors capitalize ${
                tab === t
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <div />
      </header>

      <div className="flex flex-1 overflow-hidden">
        {tab !== "files" && (
          <aside className="w-[30%] border-r border-gray-200 bg-white p-5 overflow-y-auto shrink-0">
            <UploadPanel />
          </aside>
        )}

        <main className="flex-1 overflow-hidden">
          {tab === "chat"  && <ChatWindow />}
          {tab === "files" && (
            <div className="p-6 overflow-y-auto h-full">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">Ingested Documents</h2>
              {loadingFiles ? (
                <p className="text-sm text-gray-400">Loading…</p>
              ) : files.length === 0 ? (
                <p className="text-sm text-gray-400">No documents ingested yet.</p>
              ) : (
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-gray-200 text-left text-xs text-gray-500 uppercase tracking-wide">
                      <th className="py-2 pr-4">Filename</th>
                      <th className="py-2 pr-4 text-right">Text chunks</th>
                      <th className="py-2 pr-4 text-right">Figures</th>
                      <th className="py-2 pr-4 text-right">Tables</th>
                      <th className="py-2 pr-4">Date</th>
                      <th className="py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {files.map(f => (
                      <tr key={f.filename} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-2 pr-4 font-medium text-gray-800 max-w-[200px] truncate" title={f.filename}>
                          {f.filename}
                        </td>
                        <td className="py-2 pr-4 text-right text-gray-600">
                          {(f.chunks - (f.figures || 0) - (f.tables || 0)) || f.chunks}
                        </td>
                        <td className="py-2 pr-4 text-right">
                          <span className="text-teal-600 font-medium">{f.figures || 0}</span>
                        </td>
                        <td className="py-2 pr-4 text-right">
                          <span className="text-amber-600 font-medium">{f.tables || 0}</span>
                        </td>
                        <td className="py-2 pr-4 text-gray-400 text-xs">
                          {f.ingested_at ? new Date(f.ingested_at).toLocaleDateString() : "—"}
                        </td>
                        <td className="py-2">
                          <button
                            onClick={() => handleDelete(f.filename)}
                            className="text-red-400 hover:text-red-600 text-xs"
                          >
                            Delete all
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

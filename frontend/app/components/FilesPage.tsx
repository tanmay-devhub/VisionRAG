"use client";
import { useState, useCallback, useEffect, Dispatch, SetStateAction } from "react";
import { FileEntry, getFiles, deleteFile } from "@/lib/api";

interface Props {
  files: FileEntry[];
  setFiles: Dispatch<SetStateAction<FileEntry[]>>;
  onViewGraph: (filename: string) => void;
}

function RefreshIcon({ spinning }: { spinning?: boolean }) {
  return (
    <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"
      className={spinning ? "spin" : ""}>
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
      <path d="M21 3v5h-5"/>
      <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
      <path d="M8 16H3v5"/>
    </svg>
  );
}

function GraphIcon() {
  return (
    <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <circle cx={5}  cy={6}  r={2.4}/>
      <circle cx={19} cy={7}  r={2.4}/>
      <circle cx={12} cy={17} r={2.6}/>
      <path d="M7.1 7 9.8 15M16.7 8.4 13.2 15"/>
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>
    </svg>
  );
}

function FilesIcon() {
  return (
    <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/>
      <path d="M14 2v4a2 2 0 0 0 2 2h4M9 12h6M9 16h6"/>
    </svg>
  );
}

function FileTextIcon() {
  return (
    <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <rect width={18} height={18} x={3} y={3} rx={2}/>
      <circle cx={9} cy={9} r={2}/>
      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
    </svg>
  );
}

function TableIcon() {
  return (
    <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <path d="M3 3h18v18H3zM3 9h18M3 15h18M9 3v18M15 3v18"/>
    </svg>
  );
}

function kindOf(filename: string): "vid" | "img" | "doc" {
  const ext = filename.toLowerCase().slice(filename.lastIndexOf("."));
  if ([".mp4", ".mov", ".avi", ".mkv", ".webm"].includes(ext)) return "vid";
  if ([".png", ".jpg", ".jpeg", ".webp"].includes(ext)) return "img";
  return "doc";
}

function formatDate(iso: string | null) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  } catch { return iso; }
}

function CountPill({ n, kind }: { n: number; kind: "text" | "fig" | "tab" }) {
  if (!n) return (
    <span className="count-pill count-zero">
      <span className="cdot" />0
    </span>
  );
  const cls = kind === "text" ? "count-text" : kind === "fig" ? "count-fig" : "count-tab";
  return (
    <span className={"count-pill " + cls}>
      <span className="cdot" />{n}
    </span>
  );
}

export default function FilesPage({ files, setFiles, onViewGraph }: Props) {
  const [refreshing, setRefreshing]   = useState(false);
  const [removing, setRemoving]       = useState<Set<string>>(new Set());
  const [fetchError, setFetchError]   = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setFetchError(null);
    try {
      const data = await getFiles();
      setFiles(data.files);
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : "Failed to load files");
    } finally { setRefreshing(false); }
  }, [setFiles]);

  // fetch on mount
  useEffect(() => { refresh(); }, []);

  const del = useCallback(async (filename: string) => {
    setRemoving(prev => new Set(prev).add(filename));
    try {
      await deleteFile(filename);
      setFiles(prev => prev.filter(f => f.filename !== filename));
    } catch {
      setRemoving(prev => { const n = new Set(prev); n.delete(filename); return n; });
    }
  }, [setFiles]);

  const totals = files.reduce(
    (a, f) => ({ text: a.text + f.chunks, fig: a.fig + f.figures, tab: a.tab + f.tables }),
    { text: 0, fig: 0, tab: 0 }
  );

  return (
    <div className="page-scroll page-anim">
      <div className="page-pad">
        <div className="page-head">
          <div>
            <h1 className="page-title">Files</h1>
            <p className="page-sub">{files.length} source{files.length !== 1 ? "s" : ""} ingested into the knowledge base.</p>
          </div>
          <button className="btn btn-ghost" onClick={refresh}>
            <RefreshIcon spinning={refreshing} /> Refresh
          </button>
        </div>

        {/* Stats row */}
        <div className="files-stats">
          <div className="card stat-card">
            <div className="stat-label"><FilesIcon /> Total files</div>
            <div className="stat-value">{files.length}</div>
          </div>
          <div className="card stat-card">
            <div className="stat-label"><FileTextIcon /> Text chunks</div>
            <div className="stat-value">{totals.text}</div>
          </div>
          <div className="card stat-card">
            <div className="stat-label"><ImageIcon /> Figures</div>
            <div className="stat-value">{totals.fig}</div>
          </div>
          <div className="card stat-card">
            <div className="stat-label"><TableIcon /> Tables</div>
            <div className="stat-value">{totals.tab}</div>
          </div>
        </div>

        {fetchError ? (
          <div className="card" style={{ padding: 0 }}>
            <div className="empty" style={{ height: 200 }}>
              <h3 style={{ color: "var(--red)" }}>Could not load files</h3>
              <p>{fetchError}</p>
              <button className="btn btn-ghost" style={{ marginTop: 12 }} onClick={refresh}>Retry</button>
            </div>
          </div>
        ) : refreshing && files.length === 0 ? (
          <div className="card" style={{ padding: 0 }}>
            <div className="empty" style={{ height: 200 }}>
              <div className="spin" style={{ width: 28, height: 28, borderRadius: "50%", border: "2px solid var(--border)", borderTopColor: "var(--accent)" }} />
              <p style={{ marginTop: 16 }}>Loading files…</p>
            </div>
          </div>
        ) : files.length === 0 ? (
          <div className="card" style={{ padding: 0 }}>
            <div className="empty" style={{ height: 340 }}>
              <div className="empty-art">
                <FilesIcon />
              </div>
              <h3>No files yet</h3>
              <p>Upload a document, image, or video from the Chat tab to get started.</p>
            </div>
          </div>
        ) : (
          <div className="card table-card">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Filename</th>
                  <th className="num">Text</th>
                  <th className="num">Figures</th>
                  <th className="num">Tables</th>
                  <th>Ingested</th>
                  <th style={{ textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {files.map(f => {
                  const kind = kindOf(f.filename);
                  return (
                    <tr key={f.filename} className={removing.has(f.filename) ? "row-removing" : ""}>
                      <td>
                        <div className="file-cell">
                          <span className={"file-ico " + kind}>
                            <FileTextIcon />
                          </span>
                          <div>
                            <div className="file-name">{f.filename}</div>
                            <div className="file-sub">{kind}</div>
                          </div>
                        </div>
                      </td>
                      <td className="num"><CountPill n={f.chunks}  kind="text" /></td>
                      <td className="num"><CountPill n={f.figures} kind="fig"  /></td>
                      <td className="num"><CountPill n={f.tables}  kind="tab"  /></td>
                      <td style={{ color: "var(--text-2)", whiteSpace: "nowrap" }}>{formatDate(f.ingested_at)}</td>
                      <td>
                        <div className="row-actions">
                          <button className="icon-btn" onClick={() => onViewGraph(f.filename)}>
                            <GraphIcon /> View graph
                          </button>
                          <button className="icon-btn danger" onClick={() => del(f.filename)} title="Delete file">
                            <TrashIcon />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

"use client";
import { useState, useCallback, useEffect, useRef, DragEvent, ChangeEvent } from "react";
import { ingestFile, getJobStatus, deleteFile, JobStatus } from "@/lib/api";

type LocalStatus = "queued" | "submitted" | "pending" | "processing" | "done" | "error" | "cancelled";
type VideoBackend = "gemini" | "ollama";

interface FileEntry {
  id:     string;
  file:   File;
  jobId:  string | null;
  status: LocalStatus;
  job:    JobStatus | null;
  error:  string | null;
}

let _ctr = 0;
const uid = () => String(++_ctr);

const _VIDEO_EXTS     = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm"]);
const _SUPPORTED_EXTS = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".avi", ".mkv", ".webm"];

function getExt(name: string) { return name.toLowerCase().slice(name.lastIndexOf(".")); }
function isSupportedFile(name: string) { return _SUPPORTED_EXTS.includes(getExt(name)); }
function isVideoFile(name: string) { return _VIDEO_EXTS.has(getExt(name)); }
function isImageFile(name: string) { return [".png", ".jpg", ".jpeg", ".webp"].includes(getExt(name)); }

function fmtSize(bytes: number) {
  if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + " MB";
  return Math.max(1, Math.round(bytes / 1024)) + " KB";
}

function kindOf(file: File): "vid" | "img" | "doc" {
  if (isVideoFile(file.name)) return "vid";
  if (isImageFile(file.name)) return "img";
  return "doc";
}

function getSubStep(job: JobStatus | null, video: boolean, backend: VideoBackend): string {
  if (!job) return "Queued";
  const { chunks_done, total_chunks, figure_count, table_count } = job;
  if (video) {
    if (backend === "gemini") {
      if (total_chunks === 0) return "Uploading to Gemini…";
      if (chunks_done < total_chunks) return "Gemini analyzing video…";
      return "Storing results…";
    }
    if (total_chunks === 0) return "Extracting frames…";
    if (chunks_done < total_chunks) return "Describing frames via cloud…";
    return "Indexing frames…";
  }
  if (total_chunks > 0 && chunks_done < total_chunks) {
    const textDone = chunks_done - figure_count - table_count;
    if (textDone < total_chunks - figure_count - table_count) return "Extracting text…";
    if (figure_count > 0) return "Analyzing figures…";
    return "Indexing chunks…";
  }
  if (figure_count > 0) return "Analyzing figures…";
  return "Indexing…";
}

// ── Icons ──────────────────────────────────────────────────────────────────────

function CloudIcon() {
  return (
    <svg width={26} height={26} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round">
      <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/>
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
      <path d="m9 11 3 3L22 4"/>
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} className="spin">
      <circle cx={12} cy={12} r={10} strokeOpacity={0.25}/>
      <path d="M12 2a10 10 0 0 1 10 10" stroke="var(--accent)"/>
    </svg>
  );
}

function VideoIcon() {
  return (
    <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round">
      <path d="m22 8-6 4 6 4V8z"/><rect width={14} height={12} x={2} y={6} rx={2}/>
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round">
      <rect width={18} height={18} x={3} y={3} rx={2}/>
      <circle cx={9} cy={9} r={2}/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
    </svg>
  );
}

function FileTextIcon() {
  return (
    <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/>
    </svg>
  );
}

function XIcon() {
  return (
    <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round">
      <path d="M18 6 6 18M6 6l12 12"/>
    </svg>
  );
}

// ── File card ──────────────────────────────────────────────────────────────────

function FileCard({ entry, onCancel, videoBackend }: { entry: FileEntry; onCancel: (id: string) => void; videoBackend: VideoBackend }) {
  const { file, status, job, error } = entry;
  const video  = isVideoFile(file.name);
  const kind   = kindOf(file);
  const pct    = job && job.total_chunks > 0 ? Math.round((job.chunks_done / job.total_chunks) * 100) : null;
  const canCancel = ["pending", "processing", "queued", "submitted"].includes(status);

  return (
    <div className="upload-item">
      <div className="ui-top">
        <div className={"ui-thumb " + kind}>
          {kind === "vid" ? <VideoIcon /> : kind === "img" ? <ImageIcon /> : <FileTextIcon />}
        </div>
        <div className="ui-meta">
          <div className="ui-name" title={file.name}>{file.name}</div>
          <div className="ui-info">
            <span className="size-badge">{fmtSize(file.size)}</span>
            {status === "done"
              ? <span className="ui-step" style={{ color: "var(--green)" }}>
                  Ready · {job?.chunks_done ?? 0} chunks
                </span>
              : status === "error"
              ? <span className="ui-step" style={{ color: "var(--red)" }}>Error</span>
              : status === "cancelled"
              ? <span className="ui-step" style={{ color: "var(--orange)" }}>Cancelled</span>
              : <span className="ui-step">{getSubStep(job, video, videoBackend)}</span>
            }
          </div>
        </div>
        {(status === "pending" || status === "processing" || status === "queued" || status === "submitted") && (
          <SpinnerIcon />
        )}
        {status === "done" && (
          <span className="ui-done"><CheckIcon /></span>
        )}
        {canCancel && (
          <button className="ui-cancel" onClick={() => onCancel(entry.id)}>Cancel</button>
        )}
        {status === "error" && (
          <button className="ui-cancel" onClick={() => onCancel(entry.id)}><XIcon /></button>
        )}
      </div>

      {(status === "pending" || status === "processing") && (
        <>
          <div className="progress">
            <div className="progress-fill" style={{ width: `${pct ?? 5}%` }} />
          </div>
          {pct != null && (
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 4 }}>
              <span className="ui-pct">{pct}%</span>
            </div>
          )}
        </>
      )}

      {status === "error" && error && (
        <p style={{ margin: "8px 0 0", fontSize: 11.5, color: "var(--red)", wordBreak: "break-word" }}>{error}</p>
      )}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function UploadPanel() {
  const [dragging, setDragging]         = useState(false);
  const [queue, setQueue]               = useState<FileEntry[]>([]);
  const [videoBackend, setVideoBackend] = useState<VideoBackend>("gemini");

  const queueRef        = useRef(queue);
  const videoBackendRef = useRef(videoBackend);
  queueRef.current      = queue;
  videoBackendRef.current = videoBackend;
  const fileRef = useRef<HTMLInputElement>(null);

  const update = useCallback((id: string, patch: Partial<FileEntry>) => {
    setQueue(q => q.map(e => e.id === id ? { ...e, ...patch } : e));
  }, []);

  const submitFiles = useCallback(async (entries: FileEntry[]) => {
    await Promise.all(entries.map(async entry => {
      update(entry.id, { status: "submitted" });
      try {
        const backend = isVideoFile(entry.file.name) ? videoBackendRef.current : "gemini";
        const res = await ingestFile(entry.file, backend);
        update(entry.id, { jobId: res.job_id, status: "pending" });
      } catch (err) {
        update(entry.id, { status: "error", error: err instanceof Error ? err.message : "Submission failed" });
      }
    }));
  }, [update]);

  const processFiles = useCallback((files: File[]) => {
    const supported = files.filter(f => isSupportedFile(f.name));
    if (!supported.length) return;
    const entries: FileEntry[] = supported.map(f => ({ id: uid(), file: f, jobId: null, status: "queued", job: null, error: null }));
    setQueue(q => { const next = [...q, ...entries]; setTimeout(() => submitFiles(entries), 0); return next; });
  }, [submitFiles]);

  const cancelUpload = useCallback(async (id: string) => {
    const entry = queueRef.current.find(e => e.id === id);
    update(id, { status: "cancelled" });
    if (entry) {
      try { await deleteFile(entry.file.name); } catch {}
    }
    setQueue(q => q.filter(e => e.id !== id));
  }, [update]);

  useEffect(() => {
    const timer = setInterval(async () => {
      const active = queueRef.current.filter(e => e.jobId && (e.status === "pending" || e.status === "processing"));
      if (!active.length) return;
      await Promise.all(active.map(async entry => {
        if (!entry.jobId) return;
        try {
          const job = await getJobStatus(entry.jobId);
          update(entry.id, { status: job.status as LocalStatus, job });
        } catch {}
      }));
    }, 2000);
    return () => clearInterval(timer);
  }, [update]);

  const activeCount = queue.filter(e => e.status === "pending" || e.status === "processing").length;

  return (
    <>
      {/* Video backend selector */}
      <div className="backend-select">
        <label>Video analysis backend</label>
        <div className="radio-group">
          {([
            { k: "gemini" as const, name: "Gemini",       sub: "Cloud · fast"     },
            { k: "ollama" as const, name: "Ollama Cloud",  sub: "Private · open"   },
          ]).map(o => (
            <div key={o.k}
              className={"radio-card" + (videoBackend === o.k ? " sel" : "")}
              onClick={() => setVideoBackend(o.k)}>
              <span className="radio-dot" />
              <div>
                <div className="rc-name">{o.name}</div>
                <div className="rc-sub">{o.sub}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Dropzone */}
      <div
        className={"dropzone" + (dragging ? " drag" : "")}
        onClick={() => fileRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setDragging(false); processFiles(Array.from(e.dataTransfer.files)); }}
      >
        <div className="dz-icon"><CloudIcon /></div>
        <div className="dz-title">Drop files or <b>browse</b></div>
        <div className="dz-sub">Images &amp; documents up to 50 MB · video up to 500 MB</div>
        <div className="dz-types">
          {["PNG", "JPG", "PDF", "MP4", "MOV", "WEBM"].map(x => <span key={x} className="dz-type">{x}</span>)}
        </div>
        <input
          ref={fileRef} type="file" multiple
          accept=".pdf,.png,.jpg,.jpeg,.webp,.mp4,.mov,.avi,.mkv,.webm"
          style={{ display: "none" }}
          onChange={(e: ChangeEvent<HTMLInputElement>) => { if (e.target.files) processFiles(Array.from(e.target.files)); e.target.value = ""; }}
        />
      </div>

      {/* Queue */}
      {queue.length > 0 && (
        <div className="upload-list-wrap">
          <div className="ul-head">
            <span>Queue</span>
            <span>{activeCount > 0 ? `${activeCount} processing` : `${queue.length} files`}</span>
          </div>
          {queue.map(entry => (
            <FileCard key={entry.id} entry={entry} onCancel={cancelUpload} videoBackend={videoBackend} />
          ))}
        </div>
      )}
    </>
  );
}

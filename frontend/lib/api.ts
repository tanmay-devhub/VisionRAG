const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8081";

// ── types ─────────────────────────────────────────────────────────────────────

export interface IngestResponse {
  job_id:   string;
  filename: string;
  status:   string;
}

export interface JobStatus {
  job_id:                string;
  filename:              string;
  status:                "pending" | "processing" | "done" | "error";
  chunks_done:           number;
  total_chunks:          number;
  figure_count:          number;
  table_count:           number;
  entities_created:      number;
  relationships_created: number;
  error:                 string | null;
}

export interface Source {
  text:         string;
  source:       string;
  chunk_index:  number;
  score:        number;
  type:         string;
  chunk_type:   string;
  media_type:   string | null;
  image_url:    string | null;
  figure_type:  string | null;
  caption:      string | null;
  page_number:  number | null;
  timestamp_ms: number | null;
}

export interface QueryResponse {
  answer:  string;
  sources: Source[];
}

export interface GraphNode {
  id:         string;
  label:      string;
  filename:   string;
  nodeType:   "chunk" | "entity";
  chunk_type?: string;
  type?:      string;
}

export interface GraphLink {
  source: string;
  target: string;
  type:   string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface FileEntry {
  filename:    string;
  chunks:      number;
  figures:     number;
  tables:      number;
  ingested_at: string | null;
}

export interface FileList {
  files: FileEntry[];
}

// ── functions ─────────────────────────────────────────────────────────────────

export async function ingestFile(file: File, videoBackend: string = "gemini"): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const url = `${API_BASE}/ingest?video_backend=${encodeURIComponent(videoBackend)}`;
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Ingest failed with status ${res.status}`);
  }
  return res.json() as Promise<IngestResponse>;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/ingest/status/${jobId}`);
  if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
  return res.json() as Promise<JobStatus>;
}

export async function listJobs(): Promise<JobStatus[]> {
  const res = await fetch(`${API_BASE}/ingest/jobs`);
  if (!res.ok) return [];
  const data = await res.json() as { jobs: JobStatus[] };
  return data.jobs;
}

export async function queryDocuments(question: string, topK: number): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ question, top_k: topK }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Query failed with status ${res.status}`);
  }
  return res.json() as Promise<QueryResponse>;
}

export async function getGraph(filename?: string): Promise<GraphData> {
  const url = filename
    ? `${API_BASE}/graph?filename=${encodeURIComponent(filename)}`
    : `${API_BASE}/graph`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Graph fetch failed: ${res.status}`);
  return res.json() as Promise<GraphData>;
}

export async function getFiles(): Promise<FileList> {
  const res = await fetch(`${API_BASE}/graph/files`);
  if (!res.ok) return { files: [] };
  return res.json() as Promise<FileList>;
}

export async function deleteGraph(): Promise<{ deleted: number }> {
  const res = await fetch(`${API_BASE}/graph`, { method: "DELETE" });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Clear failed with status ${res.status}`);
  }
  return res.json() as Promise<{ deleted: number }>;
}

export async function deleteFile(filename: string): Promise<{ deleted_chunks: number; deleted_entities: number }> {
  const res = await fetch(`${API_BASE}/ingest/${encodeURIComponent(filename)}`, { method: "DELETE" });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Delete failed with status ${res.status}`);
  }
  return res.json();
}

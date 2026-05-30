"use client";
import { useState, useEffect, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8081";

interface GeminiUsage {
  date:           string;
  requests_today: number;
  daily_limit:    number;
  remaining:      number;
  last_request:   string | null;
  error?:         string;
}

interface HealthData {
  status:              string;
  neo4j:               string;
  ollama:              string;
  gemini:              string;           // "configured" | "missing"
  vision_backend:      string;
  vision_label:        string;
  video_cloud_model:   string;
  video_max_duration:  number;
  gemini_video:        string;
  gemini_video_model:  string;
  auth_enabled:        boolean;
  rate_limit_rpm:      number;
  gemini_usage?:       GeminiUsage;
}

// ── Icons ──────────────────────────────────────────────────────────────────────

function DatabaseIcon()   { return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><ellipse cx={12} cy={5} rx={9} ry={3}/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>; }
function ServerIcon()     { return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><rect width={20} height={8} x={2} y={2} rx={2}/><rect width={20} height={8} x={2} y={14} rx={2}/><line x1={6} x2={6.01} y1={6} y2={6}/><line x1={6} x2={6.01} y1={18} y2={18}/></svg>; }
function SparklesIcon()   { return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/></svg>; }
function CpuIcon()        { return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><rect width={16} height={16} x={4} y={4} rx={2}/><rect width={8}  height={8}  x={8} y={8} rx={1}/><path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2"/></svg>; }
function CloudCogIcon()   { return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><path d="M12 17.5a5.5 5.5 0 1 1 0-11 5.5 5.5 0 0 1 0 11z" /><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" opacity=".3"/><circle cx={12} cy={12} r={1.5}/></svg>; }
function ShieldIcon()     { return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>; }
function GaugeIcon()      { return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10"/><path d="M12 6v6l4 2"/></svg>; }
function KeyIcon()        { return <svg width={19} height={19} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"><circle cx={7.5} cy={15.5} r={5.5}/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3"/></svg>; }
function CheckCircleIcon(){ return <svg width={20} height={20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>; }
function AlertCircleIcon(){ return <svg width={20} height={20} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><circle cx={12} cy={12} r={10}/><path d="M12 8v4M12 16h.01"/></svg>; }
function RefreshIcon({ spinning }: { spinning?: boolean }) {
  return <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" className={spinning ? "spin" : ""}><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>;
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function Pill({ state, children }: { state: "ok" | "warn" | "err" | "idle"; children: React.ReactNode }) {
  return (
    <span className={"status-pill " + state}>
      <span className="live" />
      {children}
    </span>
  );
}

interface CardProps {
  icon: JSX.Element;
  tone: string;
  name: string;
  kind: string;
  pill: JSX.Element;
  rows?: [string, string][];
  children?: React.ReactNode;
}

function StatusCard({ icon, tone, name, kind, pill, rows, children }: CardProps) {
  return (
    <div className="card status-card">
      <div className="sc-head">
        <span className={"sc-icon " + tone}>{icon}</span>
        <div>
          <div className="sc-name">{name}</div>
          <div className="sc-kind">{kind}</div>
        </div>
        {pill}
      </div>
      <div className="sc-detail">
        {rows?.map((r, i) => (
          <div className="detail-row" key={i}>
            <span className="dk">{r[0]}</span>
            <span className="dv">{r[1]}</span>
          </div>
        ))}
        {children}
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function StatusPage() {
  const [health, setHealth]     = useState<HealthData | null>(null);
  const [loading, setLoading]   = useState(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const check = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json() as HealthData;
        setHealth(data);
        setLastChecked(new Date());
      }
    } catch {}
    finally { setLoading(false); }
  }, []);

  useEffect(() => { check(); }, [check]);

  function secondsAgo() {
    if (!lastChecked) return "—";
    const s = Math.round((Date.now() - lastChecked.getTime()) / 1000);
    return `checked ${s}s ago`;
  }

  const neo4jOk    = health?.neo4j === "ok";
  const ollamaOk   = health?.ollama === "ok";
  const geminiAvail = health?.gemini === "configured";
  const allOk      = neo4jOk && ollamaOk;
  const rlTotal    = health?.rate_limit_rpm ?? 60;
  const rlPct      = 0;

  return (
    <div className="page-scroll page-anim">
      <div className="page-pad">
        <div className="page-head">
          <div>
            <h1 className="page-title">System status</h1>
            <p className="page-sub">Live health of connected services and models.</p>
          </div>
          <button className="btn btn-ghost" onClick={check}>
            <RefreshIcon spinning={loading} /> Re-check
          </button>
        </div>

        {/* Health banner */}
        {health && (
          <div className="status-banner" style={!allOk ? { background: "var(--orange-soft)", borderColor: "rgba(234,88,12,.2)" } : undefined}>
            <span className="sb-ico" style={!allOk ? { background: "var(--orange)" } : undefined}>
              {allOk ? <CheckCircleIcon /> : <AlertCircleIcon />}
            </span>
            <div>
              <div className="sb-title" style={!allOk ? { color: "var(--orange)" } : undefined}>
                {allOk ? "All systems operational" : "Some services degraded"}
              </div>
              <div className="sb-sub" style={!allOk ? { color: "var(--orange)" } : undefined}>
                {allOk
                  ? "Services healthy · pipeline ready to ingest"
                  : "Check service configuration below"}
              </div>
            </div>
            <span className="sb-time" style={{ fontFamily: "var(--mono)" }}>{secondsAgo()}</span>
          </div>
        )}

        {loading && !health && (
          <div style={{ textAlign: "center", padding: 60, color: "var(--text-3)", fontSize: 14 }}>
            <div className="spin" style={{ display: "inline-block", width: 28, height: 28, borderRadius: "50%", border: "2px solid var(--border)", borderTopColor: "var(--accent)" }} />
            <p style={{ marginTop: 16 }}>Checking services…</p>
          </div>
        )}

        {health && (
          <div className="status-grid">
            <StatusCard
              icon={<DatabaseIcon />} tone="teal" name="Neo4j" kind="Graph database"
              pill={<Pill state={neo4jOk ? "ok" : "err"}>{neo4jOk ? "Connected" : "Unreachable"}</Pill>}
              rows={[["Endpoint", "bolt://localhost:7687"], ["Status", health.neo4j]]}
            />

            <StatusCard
              icon={<ServerIcon />} tone="orange" name="Ollama" kind="Local inference"
              pill={<Pill state={ollamaOk ? "ok" : "warn"}>{ollamaOk ? "Online" : "Offline"}</Pill>}
              rows={[["Host", "127.0.0.1:11434"], ["Status", health.ollama]]}
            />

            <StatusCard
              icon={<SparklesIcon />} tone="purple" name="Gemini" kind="Cloud vision API"
              pill={<Pill state={geminiAvail ? "ok" : "idle"}>{geminiAvail ? "Configured" : "Not configured"}</Pill>}
              rows={[
                ["Video model", health.gemini_video_model || "gemini-2.5-flash"],
                ["Status",      health.gemini_video === "configured" ? "Ready" : "Key missing"],
                ...(health.gemini_usage && !health.gemini_usage.error ? [
                  ["Requests today", `${health.gemini_usage.requests_today} / ${health.gemini_usage.daily_limit}`],
                  ["Remaining",      `${health.gemini_usage.remaining}`],
                ] as [string, string][] : []),
              ]}
            />

            <StatusCard
              icon={<CpuIcon />} tone="teal" name="Vision model" kind="Image understanding"
              pill={<Pill state="ok">Active</Pill>}
              rows={[
                ["Backend", health.vision_backend],
                ["Label",   health.vision_label],
              ]}
            />

            <StatusCard
              icon={<CloudCogIcon />} tone="blue" name="Video cloud model" kind="Frame analysis"
              pill={<Pill state="ok">Active</Pill>}
              rows={[
                ["Model",        health.video_cloud_model || "—"],
                ["Max duration", `${health.video_max_duration ?? 180}s`],
              ]}
            />

            <StatusCard
              icon={<ShieldIcon />} tone="green" name="Authentication" kind="API access"
              pill={<Pill state={health.auth_enabled ? "ok" : "idle"}>{health.auth_enabled ? "Enabled" : "Disabled (dev)"}</Pill>}
              rows={[["Method", health.auth_enabled ? "API key" : "None (dev mode)"]]}
            />

            <StatusCard
              icon={<GaugeIcon />} tone="orange" name="Rate limit" kind="Requests per minute"
              pill={<Pill state="ok">OK</Pill>}
              rows={[["Limit", `${rlTotal} req/min`]]}
            />

            <StatusCard
              icon={<KeyIcon />} tone="slate" name="Embeddings" kind="Vector index"
              pill={<Pill state="ok">Synced</Pill>}
              rows={[["Model", "all-MiniLM-L6-v2"]]}
            />
          </div>
        )}
      </div>
    </div>
  );
}

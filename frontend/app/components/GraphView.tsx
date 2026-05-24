"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8081";

interface GraphNode extends d3.SimulationNodeDatum {
  id: string; label: string; filename: string; nodeType: "chunk" | "entity"; chunk_type?: string; type?: string;
}
interface RawLink  { source: string; target: string; type: string; }
interface SimLink  extends d3.SimulationLinkDatum<GraphNode> { type: string; }
interface GraphData { nodes: GraphNode[]; links: RawLink[]; }

function nodeOf(v: string | GraphNode): GraphNode | null { return typeof v === "object" ? v : null; }

const NODE_COLORS = {
  text:   "#4cd7f6",
  figure: "#34d399",
  table:  "#ffb95f",
  entity: "#c4abff",
};

function nodeColor(d: GraphNode): string {
  if (d.nodeType === "entity") return NODE_COLORS.entity;
  return NODE_COLORS[d.chunk_type as keyof typeof NODE_COLORS] ?? NODE_COLORS.text;
}

const LINK_COLORS = {
  NEXT_CHUNK:     "#4cd7f6",
  DEPICTS:        "#34d399",
  CO_OCCURS_WITH: "#c4abff",
};

function linkColor(t: string): string {
  return LINK_COLORS[t as keyof typeof LINK_COLORS] ?? "rgba(255,255,255,.25)";
}
function linkWidth(t: string): number { return t === "NEXT_CHUNK" ? 1.8 : t === "DEPICTS" ? 1.2 : 0.9; }
function linkDash(t: string): string | null { return t === "CO_OCCURS_WITH" ? "3,3" : null; }

export default function GraphView({ initFile }: { initFile?: string | null }) {
  const svgRef          = useRef<SVGSVGElement>(null);
  const nodeSelRef      = useRef<d3.Selection<SVGGElement, GraphNode, SVGGElement, unknown> | null>(null);
  const [status,        setStatus]       = useState<"loading"|"empty"|"ready"|"error">("loading");
  const [counts,        setCounts]       = useState({ nodes: 0, links: 0 });
  const [files,         setFiles]        = useState<{ filename: string; chunks: number }[]>([]);
  const [selectedFile,  setSelectedFile] = useState<string | null>(initFile ?? null);
  const [labels,        setLabels]       = useState(true);
  const [search,        setSearch]       = useState("");

  useEffect(() => { setSelectedFile(initFile ?? null); }, [initFile]);

  useEffect(() => {
    fetch(`${API_BASE}/graph/files`)
      .then(r => r.json())
      .then((d: { files: { filename: string; chunks: number }[] }) => setFiles(d.files))
      .catch(() => {});
  }, []);

  // Dim nodes that don't match search
  useEffect(() => {
    if (!nodeSelRef.current) return;
    const s = search.toLowerCase();
    nodeSelRef.current.style("opacity", (d: GraphNode) =>
      s && !d.label?.toLowerCase().includes(s) && !d.id?.toLowerCase().includes(s) ? 0.12 : 1
    );
  }, [search]);

  const renderGraph = useCallback((data: GraphData) => {
    if (!svgRef.current) return;
    const el    = svgRef.current;
    const width  = el.clientWidth  || 900;
    const height = el.clientHeight || 600;
    d3.select(el).selectAll("*").remove();
    nodeSelRef.current = null;
    const svg = d3.select(el);
    const g   = svg.append("g");

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.05, 6])
        .on("zoom", ({ transform }) => g.attr("transform", transform.toString()))
    );

    const nodes: GraphNode[] = data.nodes.map(n => ({ ...n }));
    const nodeIdSet = new Set(nodes.map(n => n.id));
    const links: SimLink[] = data.links
      .filter(l => nodeIdSet.has(l.source as string) && nodeIdSet.has(l.target as string))
      .map(l => ({ ...l } as unknown as SimLink));

    const sim = d3.forceSimulation<GraphNode>(nodes)
      .force("link", d3.forceLink<GraphNode, SimLink>(links).id(d => d.id).distance(d => {
        const t = (d as SimLink).type;
        return t === "NEXT_CHUNK" ? 160 : t === "DEPICTS" ? 80 : 100;
      }))
      .force("charge", d3.forceManyBody<GraphNode>().strength(d => (d as GraphNode).nodeType === "chunk" ? -400 : -180))
      .force("center",    d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide<GraphNode>(d => (d as GraphNode).nodeType === "chunk" ? 30 : 20));

    const defs = svg.append("defs");
    Object.entries(LINK_COLORS).forEach(([key, col]) => {
      defs.append("marker").attr("id", `arr-${key}`).attr("viewBox", "0 -5 10 10")
        .attr("refX", 24).attr("refY", 0).attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
        .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", col);
    });

    const linkEls = g.append("g").selectAll<SVGLineElement, SimLink>("line")
      .data(links).enter().append("line")
      .attr("stroke",           d => linkColor(d.type))
      .attr("stroke-width",     d => linkWidth(d.type))
      .attr("stroke-dasharray", d => linkDash(d.type))
      .attr("stroke-opacity",   d => d.type === "CO_OCCURS_WITH" ? 0.45 : 0.85)
      .attr("marker-end",       d => `url(#arr-${d.type})`);

    const nodeEls = g.append("g").selectAll<SVGGElement, GraphNode>("g")
      .data(nodes).enter().append("g").style("cursor", "grab")
      .call(d3.drag<SVGGElement, GraphNode>()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

    nodeSelRef.current = nodeEls;

    nodeEls.append("circle")
      .attr("r", d => d.nodeType === "chunk" ? 14 : 9)
      .attr("fill",         d => `${nodeColor(d)}18`)
      .attr("stroke",       d => nodeColor(d))
      .attr("stroke-width", d => d.nodeType === "chunk" ? 2 : 1.5);

    nodeEls.filter(d => d.nodeType === "entity").append("circle")
      .attr("r", 14).attr("fill", "none")
      .attr("stroke", d => nodeColor(d)).attr("stroke-width", 0.5).attr("stroke-opacity", 0.3);

    nodeEls.append("title").text(d =>
      `${d.label}\n${d.nodeType}${d.chunk_type ? ` (${d.chunk_type})` : ""}\n${d.filename}`
    );

    if (labels) {
      nodeEls.append("text")
        .text(d => { const l = d.label || d.id; return l.length > 14 ? l.slice(0, 13) + "…" : l; })
        .attr("text-anchor", "middle")
        .attr("dy",          d => d.nodeType === "chunk" ? 28 : 22)
        .attr("fill", "#bcc9cd")
        .attr("font-size",   "9px")
        .attr("font-family", "Inter, sans-serif")
        .attr("pointer-events", "none");
    }

    sim.on("tick", () => {
      linkEls
        .attr("x1", d => nodeOf(d.source)?.x ?? 0).attr("y1", d => nodeOf(d.source)?.y ?? 0)
        .attr("x2", d => nodeOf(d.target)?.x ?? 0).attr("y2", d => nodeOf(d.target)?.y ?? 0);
      nodeEls.attr("transform", d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });
  }, [labels]);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    const url = selectedFile
      ? `${API_BASE}/graph?filename=${encodeURIComponent(selectedFile)}`
      : `${API_BASE}/graph`;
    fetch(url)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then((data: GraphData) => {
        if (cancelled) return;
        setCounts({ nodes: data.nodes.length, links: data.links.length });
        if (!data.nodes.length) { setStatus("empty"); return; }
        renderGraph(data);
        setStatus("ready");
      })
      .catch(() => { if (!cancelled) setStatus("error"); });
    return () => { cancelled = true; };
  }, [selectedFile, renderGraph]);

  const short = (f: string) => f.length > 18 ? f.slice(0, 16) + "…" : f;

  return (
    <div className="relative h-full w-full overflow-hidden bg-surface-container-lowest">
      {/* Grid background */}
      <div className="absolute inset-0 z-0 pointer-events-none" style={{
        backgroundImage: "linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px)",
        backgroundSize: "40px 40px",
        maskImage: "radial-gradient(ellipse at center, black 30%, transparent 80%)",
      }} />
      {/* Glow blobs */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/3 w-80 h-80 rounded-full blur-[80px]"  style={{ background: "rgba(76,215,246,0.05)" }} />
        <div className="absolute bottom-1/4 right-1/3 w-60 h-60 rounded-full blur-[100px]" style={{ background: "rgba(87,27,193,0.08)" }} />
      </div>

      {/* SVG canvas */}
      <svg ref={svgRef} className="absolute inset-0 w-full h-full z-10" />

      {/* Stats overlay — top left */}
      {status === "ready" && (
        <div className="absolute top-4 left-4 z-20 glass-panel-high rounded-xl px-4 py-3 flex gap-5">
          {[
            { label: "Nodes", val: counts.nodes.toLocaleString(), color: "#4cd7f6" },
            { label: "Edges", val: counts.links.toLocaleString(), color: "#d0bcff" },
            { label: "Files", val: String(files.length),          color: "#ffb95f" },
          ].map(s => (
            <div key={s.label} className="flex flex-col gap-0.5">
              <span className="text-[9px] font-mono text-on-surface-variant uppercase tracking-wider">{s.label}</span>
              <span className="text-lg font-bold" style={{ color: s.color }}>{s.val}</span>
            </div>
          ))}
        </div>
      )}

      {/* Floating toolbar pill — top center */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1.5 px-2 py-1.5 glass-panel-high rounded-full shadow-lg">
        {/* Search */}
        <div className="flex items-center gap-2 h-7 px-3 rounded-full bg-black/30 border border-white/[0.08]">
          <svg className="w-3 h-3 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="#bcc9cd" strokeWidth={2.5}>
            <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
          </svg>
          <input
            placeholder="Filter nodes…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="bg-transparent border-none outline-none text-xs text-on-surface placeholder:text-on-surface-variant w-28"
          />
        </div>
        <div className="w-px h-5 bg-white/10 mx-0.5" />
        {/* Doc tabs */}
        <button
          onClick={() => setSelectedFile(null)}
          className={`h-7 px-3 rounded-full text-xs font-medium transition-colors ${
            !selectedFile
              ? "border text-primary" : "text-on-surface-variant hover:text-on-surface hover:bg-white/5"
          }`}
          style={!selectedFile ? { background: "rgba(76,215,246,0.1)", borderColor: "rgba(76,215,246,0.25)" } : undefined}
        >All</button>
        {files.slice(0, 4).map(f => (
          <button key={f.filename} onClick={() => setSelectedFile(f.filename)} title={f.filename}
            className={`h-7 px-3 rounded-full text-xs font-medium transition-colors ${
              selectedFile === f.filename
                ? "border text-primary" : "text-on-surface-variant hover:text-on-surface hover:bg-white/5"
            }`}
            style={selectedFile === f.filename ? { background: "rgba(76,215,246,0.1)", borderColor: "rgba(76,215,246,0.25)" } : undefined}
          >
            {short(f.filename)}
          </button>
        ))}
        <div className="w-px h-5 bg-white/10 mx-0.5" />
        <button
          onClick={() => setLabels(v => !v)}
          className={`h-7 px-3 rounded-full text-xs font-medium transition-colors ${
            labels
              ? "border text-primary" : "text-on-surface-variant hover:text-on-surface hover:bg-white/5"
          }`}
          style={labels ? { background: "rgba(76,215,246,0.1)", borderColor: "rgba(76,215,246,0.25)" } : undefined}
        >Labels</button>
      </div>

      {/* Legend — bottom left */}
      <div className="absolute bottom-4 left-4 z-20 glass-panel-high rounded-xl p-3 flex flex-col gap-2 min-w-[155px]">
        <p className="text-[9px] font-mono text-on-surface-variant uppercase tracking-wider font-semibold mb-1">Node types</p>
        {([
          { color: NODE_COLORS.text,   label: "Text chunk" },
          { color: NODE_COLORS.figure, label: "Figure"     },
          { color: NODE_COLORS.table,  label: "Table"      },
          { color: NODE_COLORS.entity, label: "Entity"     },
        ] as const).map(n => (
          <div key={n.label} className="flex items-center gap-2 text-xs text-on-surface-variant">
            <div className="w-2.5 h-2.5 rounded-full border flex-shrink-0"
              style={{ borderColor: n.color, background: `${n.color}22`, boxShadow: `0 0 6px ${n.color}44` }} />
            <span>{n.label}</span>
          </div>
        ))}
        <div className="border-t border-white/[0.08] pt-2 mt-1 flex flex-col gap-1.5">
          {([
            { color: LINK_COLORS.NEXT_CHUNK,     label: "NEXT_CHUNK",     dash: false },
            { color: LINK_COLORS.DEPICTS,        label: "DEPICTS",        dash: false },
            { color: LINK_COLORS.CO_OCCURS_WITH, label: "CO_OCCURS_WITH", dash: true  },
          ] as const).map(e => (
            <div key={e.label} className="flex items-center gap-2 text-[10px] text-on-surface-variant">
              <svg width={18} height={6}>
                <line x1={0} y1={3} x2={18} y2={3} stroke={e.color} strokeWidth={1.5}
                  strokeDasharray={e.dash ? "3,2" : undefined} />
              </svg>
              <span>{e.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Loading */}
      {status === "loading" && (
        <div className="absolute inset-0 z-20 flex items-center justify-center">
          <div className="flex flex-col items-center gap-4 text-on-surface-variant">
            <div className="w-10 h-10 border-2 border-white/10 rounded-full spin" style={{ borderTopColor: "#4cd7f6" }} />
            <span className="text-sm font-mono">Loading graph…</span>
          </div>
        </div>
      )}

      {/* Empty */}
      {status === "empty" && (
        <div className="absolute inset-0 z-20 flex items-center justify-center">
          <div className="text-center text-on-surface-variant glass-panel rounded-2xl p-10">
            <svg className="w-12 h-12 mx-auto mb-3 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.2}>
              <circle cx="5"  cy="12" r="2.5" />
              <circle cx="19" cy="5"  r="2.5" />
              <circle cx="19" cy="19" r="2.5" />
              <line x1="7.2" y1="10.9" x2="17" y2="6.1" />
              <line x1="7.2" y1="13.1" x2="17" y2="17.9" />
            </svg>
            <p className="text-base font-medium text-on-surface-variant mb-1">No graph data</p>
            <p className="text-sm">Upload a document to build the knowledge graph.</p>
          </div>
        </div>
      )}

      {/* Error */}
      {status === "error" && (
        <div className="absolute inset-0 z-20 flex items-center justify-center">
          <p className="text-sm font-mono glass-panel rounded-xl px-6 py-4" style={{ color: "#ffb4ab", borderColor: "rgba(255,180,171,0.2)" }}>
            Failed to load graph. Is the backend running?
          </p>
        </div>
      )}
    </div>
  );
}

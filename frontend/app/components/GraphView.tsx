"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8081";

// ── Types ──────────────────────────────────────────────────────────────────────

interface RawNode {
  id: string; label: string; filename: string;
  nodeType: "chunk" | "entity"; chunk_type?: string;
}
interface RawLink { source: string; target: string; type: string; }
interface GraphData { nodes: RawNode[]; links: RawLink[]; }
interface FileInfo { filename: string; chunks: number; }

interface SimNode extends d3.SimulationNodeDatum {
  id: string; label: string; type: string; file: string; r: number;
}
interface SimLink extends d3.SimulationLinkDatum<SimNode> { kind: string; }

// ── Constants ──────────────────────────────────────────────────────────────────

const NODE_TYPES: Record<string, { color: string; label: string }> = {
  text:   { color: "#4cd7f6", label: "Text chunk"  },
  figure: { color: "#34d399", label: "Figure"      },
  table:  { color: "#ffb95f", label: "Table"       },
  entity: { color: "#c4abff", label: "Entity"      },
  video:  { color: "#60a5fa", label: "Video frame" },
};
const FILE_PALETTE = ["#4cd7f6","#34d399","#ffb95f","#c4abff","#60a5fa","#f87171","#fb923c","#a78bfa"];

function nodeType(n: RawNode): string {
  return n.nodeType === "entity" ? "entity" : (n.chunk_type ?? "text");
}
function nodeColor(type: string): string {
  return (NODE_TYPES[type] ?? NODE_TYPES.text).color;
}
function linkDist(kind: string): number {
  return kind === "NEXT_CHUNK" ? 48 : kind === "DEPICTS" ? 72 : 100;
}

// ── Icons ──────────────────────────────────────────────────────────────────────
function SearchIcon()  { return <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><circle cx={11} cy={11} r={8}/><path d="m21 21-4.35-4.35"/></svg>; }
function PlusIcon()    { return <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><path d="M12 5v14M5 12h14"/></svg>; }
function MinusIcon()   { return <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><path d="M5 12h14"/></svg>; }
function TargetIcon()  { return <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><circle cx={12} cy={12} r={10}/><circle cx={12} cy={12} r={6}/><circle cx={12} cy={12} r={2}/></svg>; }
function GraphIcon()   { return <svg width={42} height={42} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.2} strokeLinecap="round"><circle cx={5} cy={6} r={2.4}/><circle cx={19} cy={7} r={2.4}/><circle cx={12} cy={17} r={2.6}/><path d="M7.1 7 9.8 15M16.7 8.4 13.2 15"/></svg>; }

// ── Component ──────────────────────────────────────────────────────────────────

export default function GraphView({ initFile }: { initFile?: string | null }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef       = useRef<SVGSVGElement>(null);
  const zoomRef      = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const [status,     setStatus]     = useState<"loading"|"empty"|"ready"|"error">("loading");
  const [files,      setFiles]      = useState<FileInfo[]>([]);
  const [typeCounts, setTypeCounts] = useState<Record<string, number>>({});
  const [counts,     setCounts]     = useState({ nodes: 0, edges: 0, files: 0 });
  const [offFiles,   setOffFiles]   = useState<Set<string>>(new Set());
  const [offTypes,   setOffTypes]   = useState<Set<string>>(new Set());
  const [search,     setSearch]     = useState("");
  const [zoomPct,    setZoomPct]    = useState(100);

  // Refs to current SVG selections for live filtering without re-simulating
  const nodeGRef = useRef<d3.Selection<SVGGElement, SimNode, SVGGElement, unknown> | null>(null);
  const linkGRef = useRef<d3.Selection<SVGLineElement, SimLink, SVGGElement, unknown> | null>(null);

  // Fetch files list
  useEffect(() => {
    fetch(`${API_BASE}/graph/files`)
      .then(r => r.json())
      .then((d: { files: FileInfo[] }) => setFiles(d.files))
      .catch(() => {});
  }, []);

  // Apply visibility filter whenever toggles or search change
  useEffect(() => {
    if (!nodeGRef.current || !linkGRef.current) return;
    const q = search.trim().toLowerCase();

    nodeGRef.current.style("display", (d: SimNode) => {
      if (offFiles.has(d.file) || offTypes.has(d.type)) return "none";
      return null;
    }).style("opacity", (d: SimNode) => {
      if (!q) return null;
      return (d.label?.toLowerCase().includes(q) || d.id?.toLowerCase().includes(q)) ? "1" : "0.1";
    });

    linkGRef.current.style("display", (d: SimLink) => {
      const s = d.source as SimNode, t = d.target as SimNode;
      if (!s || !t) return null;
      if (offFiles.has(s.file) || offFiles.has(t.file)) return "none";
      if (offTypes.has(s.type) || offTypes.has(t.type)) return "none";
      return null;
    });
  }, [offFiles, offTypes, search]);

  // Build and render the graph
  const buildGraph = useCallback((data: GraphData, fileList: FileInfo[]) => {
    if (!svgRef.current || !containerRef.current) return;

    const rect   = containerRef.current.getBoundingClientRect();
    const width  = rect.width  || 900;
    const height = rect.height || 600;

    // Type counts
    const tc: Record<string, number> = {};
    data.nodes.forEach(n => { const t = nodeType(n); tc[t] = (tc[t] ?? 0) + 1; });
    setTypeCounts(tc);

    // File color map
    const fileColorMap: Record<string, string> = {};
    fileList.forEach((f, i) => { fileColorMap[f.filename] = FILE_PALETTE[i % FILE_PALETTE.length]; });

    // Build sim data
    const nodes: SimNode[] = data.nodes.map(n => ({
      id: n.id,
      label: n.label ?? n.id,
      type: nodeType(n),
      file: n.filename,
      r: n.nodeType === "entity" ? 10 : 7,
    }));
    const nodeIds = new Set(nodes.map(n => n.id));
    const links: SimLink[] = data.links
      .filter(l => nodeIds.has(l.source) && nodeIds.has(l.target))
      .map(l => ({ source: l.source, target: l.target, kind: l.type }));

    setCounts({ nodes: nodes.length, edges: links.length, files: new Set(nodes.map(n => n.file)).size });

    // Clear old content
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    svg.attr("width", width).attr("height", height);

    // Background grid pattern
    const defs = svg.append("defs");
    defs.append("pattern")
      .attr("id", "grid").attr("width", 38).attr("height", 38)
      .attr("patternUnits", "userSpaceOnUse")
      .append("path").attr("d", "M 38 0 L 0 0 0 38")
      .attr("fill", "none").attr("stroke", "rgba(125,155,210,0.07)").attr("stroke-width", 1);

    svg.append("rect").attr("width", width).attr("height", height)
      .attr("fill", "url(#grid)");

    // Radial gradient overlay
    const rg = defs.append("radialGradient").attr("id", "bgGrad")
      .attr("cx", "50%").attr("cy", "42%").attr("r", "70%");
    rg.append("stop").attr("offset", "0%").attr("stop-color", "rgba(20,40,80,0.4)");
    rg.append("stop").attr("offset", "100%").attr("stop-color", "rgba(8,13,26,0)");
    svg.append("rect").attr("width", width).attr("height", height).attr("fill", "url(#bgGrad)");

    // Zoom container
    const g = svg.append("g");

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.05, 6])
      .on("zoom", ev => {
        g.attr("transform", ev.transform.toString());
        setZoomPct(Math.round(ev.transform.k * 100));
      });
    svg.call(zoom);
    zoomRef.current = zoom;

    // Arrowhead markers
    ["NEXT_CHUNK","DEPICTS","CO_OCCURS_WITH"].forEach(kind => {
      const col = kind === "CO_OCCURS_WITH" ? "rgba(196,171,255,0.5)"
                : kind === "DEPICTS"        ? "rgba(52,211,153,0.5)"
                :                             "rgba(76,215,246,0.5)";
      defs.append("marker")
        .attr("id", `arr-${kind}`).attr("viewBox", "0 -4 8 8")
        .attr("refX", 20).attr("refY", 0)
        .attr("markerWidth", 4).attr("markerHeight", 4)
        .attr("orient", "auto")
        .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", col);
    });

    // Edge colors / styles
    function edgeStroke(kind: string) {
      return kind === "CO_OCCURS_WITH" ? "rgba(196,171,255,0.28)"
           : kind === "DEPICTS"        ? "rgba(52,211,153,0.35)"
           :                             "rgba(76,215,246,0.22)";
    }

    // Draw links
    const linkG = g.append("g").selectAll<SVGLineElement, SimLink>("line")
      .data(links).join("line")
      .attr("stroke",           d => edgeStroke(d.kind))
      .attr("stroke-width",     d => d.kind === "NEXT_CHUNK" ? 1.2 : 1)
      .attr("stroke-dasharray", d => d.kind === "CO_OCCURS_WITH" ? "3,3" : null)
      .attr("marker-end",       d => `url(#arr-${d.kind})`);

    linkGRef.current = linkG as unknown as typeof linkGRef.current;

    // Draw nodes
    const nodeG = g.append("g").selectAll<SVGGElement, SimNode>("g")
      .data(nodes).join("g")
      .style("cursor", "grab")
      .call(
        d3.drag<SVGGElement, SimNode>()
          .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.15).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
          .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    nodeG.append("circle")
      .attr("r",            d => d.r)
      .attr("fill",         d => `${nodeColor(d.type)}22`)
      .attr("stroke",       d => nodeColor(d.type))
      .attr("stroke-width", d => d.type === "entity" ? 1.8 : 1.4);

    // Glow circles for entity nodes
    nodeG.filter(d => d.type === "entity").append("circle")
      .attr("r",              d => d.r + 5)
      .attr("fill",           "none")
      .attr("stroke",         d => nodeColor(d.type))
      .attr("stroke-width",   0.5)
      .attr("stroke-opacity", 0.25);

    // Labels (entities always, others on hover via CSS — we just append all)
    nodeG.append("text")
      .text(d => { const l = d.label || d.id; return l.length > 16 ? l.slice(0, 14) + "…" : l; })
      .attr("text-anchor",    "middle")
      .attr("dy",             d => d.r + 12)
      .attr("fill",           d => d.type === "entity" ? "#e7ecf5" : "rgba(207,220,230,0.7)")
      .attr("font-size",      d => d.type === "entity" ? "9.5px" : "8px")
      .attr("font-family",    "Geist, sans-serif")
      .attr("pointer-events", "none")
      .style("display",       d => d.type === "entity" ? null : "none"); // only show entity labels by default

    nodeG.append("title").text(d => `${d.label}\n${d.type}\n${d.file}`);

    nodeGRef.current = nodeG as unknown as typeof nodeGRef.current;

    // D3 force simulation with alpha decay (settles naturally)
    const sim = d3.forceSimulation<SimNode>(nodes)
      .force("link",
        d3.forceLink<SimNode, SimLink>(links)
          .id(d => d.id)
          .distance(d => linkDist(d.kind))
          .strength(d => d.kind === "NEXT_CHUNK" ? 0.9 : 0.5)
      )
      .force("charge", d3.forceManyBody<SimNode>().strength(d => d.type === "entity" ? -180 : -120))
      .force("center",    d3.forceCenter(width / 2, height / 2).strength(0.05))
      .force("collision", d3.forceCollide<SimNode>(d => d.r + 4))
      .alphaDecay(0.028)   // settles in ~300 ticks (~5 s)
      .on("tick", () => {
        linkG
          .attr("x1", d => (d.source as SimNode).x ?? 0)
          .attr("y1", d => (d.source as SimNode).y ?? 0)
          .attr("x2", d => (d.target as SimNode).x ?? 0)
          .attr("y2", d => (d.target as SimNode).y ?? 0);
        nodeG.attr("transform", d => `translate(${d.x ?? 0},${d.y ?? 0})`);
      });

    setStatus("ready");
    return sim;
  }, []);

  // Fetch graph and build
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    Promise.all([
      fetch(`${API_BASE}/graph/files`).then(r => r.json()).catch(() => ({ files: [] })),
      fetch(`${API_BASE}/graph`).then(r => { if (!r.ok) throw new Error(); return r.json(); }),
    ])
      .then(([filesRes, data]: [{ files: FileInfo[] }, GraphData]) => {
        if (cancelled) return;
        const fl: FileInfo[] = filesRes.files ?? [];
        setFiles(fl);
        if (!data.nodes.length) { setStatus("empty"); return; }
        buildGraph(data, fl);
      })
      .catch(() => { if (!cancelled) setStatus("error"); });
    return () => { cancelled = true; };
  }, [buildGraph]);

  // Handle initFile focus: dim everything else
  useEffect(() => {
    if (!initFile || files.length === 0) return;
    setOffFiles(new Set(files.map(f => f.filename).filter(fn => fn !== initFile)));
  }, [initFile, files]);

  const zoomBy = useCallback((f: number) => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current).transition().duration(200)
      .call(zoomRef.current.scaleBy, f);
  }, []);

  const resetZoom = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current).transition().duration(300)
      .call(zoomRef.current.transform, d3.zoomIdentity);
    setZoomPct(100);
  }, []);

  const toggleFile = useCallback((fn: string) => {
    setOffFiles(prev => { const n = new Set(prev); n.has(fn) ? n.delete(fn) : n.add(fn); return n; });
  }, []);

  const toggleType = useCallback((type: string) => {
    setOffTypes(prev => { const n = new Set(prev); n.has(type) ? n.delete(type) : n.add(type); return n; });
  }, []);

  return (
    <div className="graph-page" ref={containerRef}>
      {/* SVG canvas */}
      <svg ref={svgRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />

      {/* ── Top toolbar ── */}
      {status === "ready" && (
        <div className="graph-toolbar glass">
          <div className="g-search">
            <SearchIcon />
            <input
              placeholder="Search nodes…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          {files.length > 0 && (
            <>
              <div className="g-divider" />
              <div className="file-pills">
                {files.map((f, i) => (
                  <button
                    key={f.filename}
                    className={"file-pill " + (offFiles.has(f.filename) ? "off" : "on")}
                    onClick={() => toggleFile(f.filename)}
                    title={f.filename}
                  >
                    <span className="fp-dot" style={{ background: FILE_PALETTE[i % FILE_PALETTE.length] }} />
                    {f.filename.replace(/\.(pdf|mp4|mov|avi|mkv|webm|png|jpe?g|webp)$/i, "")}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Live stats ── */}
      {status === "ready" && (
        <div className="graph-stats glass">
          <div className="gs-title"><span className="pulse" /> Live graph</div>
          <div className="gs-row"><span className="gs-k">Nodes</span><span className="gs-v">{counts.nodes}</span></div>
          <div className="gs-row"><span className="gs-k">Edges</span><span className="gs-v">{counts.edges}</span></div>
          <div className="gs-row"><span className="gs-k">Files</span><span className="gs-v">{counts.files}</span></div>
        </div>
      )}

      {/* ── Legend ── */}
      {status === "ready" && (
        <div className="graph-legend glass">
          <div className="lg-title">Node types</div>
          {Object.entries(NODE_TYPES)
            .filter(([k]) => (typeCounts[k] ?? 0) > 0)
            .map(([k, nt]) => (
              <div
                key={k}
                className={"lg-item" + (offTypes.has(k) ? " dim" : "")}
                onClick={() => toggleType(k)}
              >
                <span className="lg-swatch" style={{ background: nt.color, color: nt.color }} />
                {nt.label}
                <span className="lg-count">{typeCounts[k] ?? 0}</span>
              </div>
            ))}
        </div>
      )}

      {/* ── Zoom controls ── */}
      {status === "ready" && (
        <div className="graph-zoom glass">
          <button className="zoom-btn" onClick={() => zoomBy(1.25)}><PlusIcon /></button>
          <div className="zoom-level">{zoomPct}%</div>
          <button className="zoom-btn" onClick={() => zoomBy(0.8)}><MinusIcon /></button>
          <div className="zoom-divider" />
          <button className="zoom-btn" onClick={resetZoom} title="Reset view"><TargetIcon /></button>
        </div>
      )}

      {/* ── Loading ── */}
      {status === "loading" && (
        <div style={{ position: "absolute", inset: 0, zIndex: 20, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16, color: "var(--glass-text-2)" }}>
            <div className="spin" style={{ width: 40, height: 40, borderRadius: "50%", border: "2px solid rgba(76,215,246,0.2)", borderTopColor: "#4cd7f6" }} />
            <span style={{ fontSize: 13, fontFamily: "var(--mono)" }}>Loading graph…</span>
          </div>
        </div>
      )}

      {/* ── Empty ── */}
      {status === "empty" && (
        <div className="graph-empty">
          <div className="graph-empty-inner">
            <div className="ge-art"><GraphIcon /></div>
            <h3>No graph data</h3>
            <p>Upload a document to build the knowledge graph.</p>
          </div>
        </div>
      )}

      {/* ── Error ── */}
      {status === "error" && (
        <div style={{ position: "absolute", inset: 0, zIndex: 20, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <p className="glass" style={{ borderRadius: 12, padding: "16px 24px", fontSize: 13, fontFamily: "var(--mono)", color: "#ffb4ab" }}>
            Failed to load graph. Is the backend running?
          </p>
        </div>
      )}
    </div>
  );
}

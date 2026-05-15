"use client";
import { Source } from "@/lib/api";

interface Props {
  sources: Source[];
}

function ChunkTypeBadge({ type }: { type: string }) {
  if (type === "figure") {
    return (
      <span className="bg-teal-100 text-teal-700 px-1.5 py-0.5 rounded text-[10px] font-semibold">
        Figure
      </span>
    );
  }
  if (type === "table") {
    return (
      <span className="bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded text-[10px] font-semibold">
        Table
      </span>
    );
  }
  return (
    <span className="bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded text-[10px] font-semibold">
      Text
    </span>
  );
}

function FigureCard({ src }: { src: Source }) {
  return (
    <div className="border border-teal-200 rounded-lg p-2 bg-white text-xs">
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <ChunkTypeBadge type="figure" />
        {src.figure_type && (
          <span className="bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded text-[10px]">
            {src.figure_type}
          </span>
        )}
        <span className="font-medium text-gray-700 truncate max-w-[120px]" title={src.source}>
          {src.source.length > 30 ? src.source.slice(0, 30) + "…" : src.source}
        </span>
        {src.page_number != null && (
          <span className="text-gray-400">p.{src.page_number + 1}</span>
        )}
        <span className="ml-auto text-gray-500">{Math.round(src.score * 100)}%</span>
      </div>

      {src.image_url && (
        <div className="mb-2 rounded overflow-hidden border border-gray-100">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src.image_url}
            alt={src.caption || "Figure"}
            className="w-full object-contain max-h-48"
          />
        </div>
      )}

      {src.caption && (
        <p className="text-gray-400 text-[10px] mb-1 italic">{src.caption}</p>
      )}

      <details className="mt-1">
        <summary className="cursor-pointer text-gray-400 hover:text-gray-600 select-none">
          Description
        </summary>
        <p className="mt-1 text-gray-600 leading-snug">
          {src.text.length > 300 ? src.text.slice(0, 300) + "…" : src.text}
        </p>
      </details>
    </div>
  );
}

function TableCard({ src }: { src: Source }) {
  const rows = parseMarkdownTable(src.text);

  return (
    <div className="border border-amber-200 rounded-lg p-2 bg-white text-xs">
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <ChunkTypeBadge type="table" />
        <span className="font-medium text-gray-700 truncate max-w-[120px]" title={src.source}>
          {src.source.length > 30 ? src.source.slice(0, 30) + "…" : src.source}
        </span>
        {src.page_number != null && (
          <span className="text-gray-400">p.{src.page_number + 1}</span>
        )}
        <span className="ml-auto text-gray-500">{Math.round(src.score * 100)}%</span>
      </div>

      {rows.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[10px]">
            <thead>
              <tr>
                {rows[0].map((cell, i) => (
                  <th
                    key={i}
                    className="border border-gray-200 bg-gray-50 px-1.5 py-1 text-left font-semibold text-gray-700"
                  >
                    {cell}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(1).map((row, ri) => (
                <tr key={ri} className="even:bg-gray-50">
                  {row.map((cell, ci) => (
                    <td
                      key={ci}
                      className="border border-gray-200 px-1.5 py-1 text-gray-600"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-gray-600 leading-snug">
          {src.text.length > 300 ? src.text.slice(0, 300) + "…" : src.text}
        </p>
      )}
    </div>
  );
}

function TextCard({ src }: { src: Source }) {
  return (
    <div className="border border-gray-200 rounded-lg p-2 bg-white text-xs">
      <div className="flex items-center gap-2 mb-1 flex-wrap">
        <ChunkTypeBadge type="text" />
        <span className="font-medium text-gray-700 truncate max-w-[120px]" title={src.source}>
          {src.source.length > 30 ? src.source.slice(0, 30) + "…" : src.source}
        </span>
        {src.page_number != null && (
          <span className="text-gray-400">p.{src.page_number + 1}</span>
        )}
        <span className="ml-auto text-gray-500">{Math.round(src.score * 100)}%</span>
      </div>
      <p className="text-gray-600 leading-snug">
        {src.text.length > 200 ? src.text.slice(0, 200) + "…" : src.text}
      </p>
    </div>
  );
}

export default function SourcePanel({ sources }: Props) {
  return (
    <div className="grid grid-cols-2 gap-2 mt-2">
      {sources.map((src, i) => {
        if (src.chunk_type === "figure") return <FigureCard key={i} src={src} />;
        if (src.chunk_type === "table")  return <TableCard  key={i} src={src} />;
        return <TextCard key={i} src={src} />;
      })}
    </div>
  );
}

// ── helpers ───────────────────────────────────────────────────────────────────

function parseMarkdownTable(text: string): string[][] {
  const lines = text.trim().split("\n").filter(l => l.trim().startsWith("|"));
  if (lines.length < 2) return [];

  const parseRow = (line: string): string[] =>
    line
      .split("|")
      .slice(1, -1)
      .map(cell => cell.trim());

  const header    = parseRow(lines[0]);
  const separator = lines[1];
  if (!separator.includes("---")) return [];

  const dataRows = lines.slice(2).map(parseRow);
  return [header, ...dataRows];
}

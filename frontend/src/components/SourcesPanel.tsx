import { useState } from 'react';
import type { ChatSource } from '../types';

interface SourcesPanelProps {
  sources: ChatSource[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3 border-t border-slate-100 pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-xs font-medium text-teal-700 hover:underline"
      >
        {open ? '▲' : '▼'} Nguồn ({sources.length} kết quả)
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {sources.map((src, i) => {
            const score = typeof src.score === 'number' ? src.score : undefined;
            const similarity = score !== undefined ? Math.max(0, Math.min(1, 1 - score)) : undefined;
            return (
              <li key={String(src.id ?? i)} className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
                <span className="font-semibold text-slate-700">[{i + 1}] </span>
                {String(src.question ?? 'N/A')}
                {similarity !== undefined && (
                  <span className="ml-1 text-slate-400">· {similarity.toFixed(2)} similarity</span>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

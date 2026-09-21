import { useState } from 'react';
import type { FormEvent } from 'react';

interface InputBarProps {
  onSend: (text: string, topK: number) => void;
  disabled: boolean;
}

export function InputBar({ onSend, disabled }: InputBarProps) {
  const [text, setText] = useState('');
  const [topK, setTopK] = useState(5);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, topK);
    setText('');
  };

  return (
    <form onSubmit={submit} className="border-t border-white/60 bg-white/70 px-4 py-3 backdrop-blur-sm sm:px-8">
      {showAdvanced && (
        <div className="mb-2 flex items-center gap-2 text-xs text-slate-500">
          <label htmlFor="top-k">top_k</label>
          <input
            id="top-k"
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-16 rounded border border-slate-300 px-2 py-1"
          />
        </div>
      )}
      <div className="flex items-end gap-2">
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="shrink-0 rounded-full border border-slate-300 px-2 py-1 text-xs text-slate-500 transition-transform duration-150 hover:scale-105 hover:bg-slate-50"
          aria-label="Tùy chọn nâng cao"
        >
          ⚙️
        </button>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit(e as unknown as FormEvent);
            }
          }}
          rows={1}
          placeholder="Hỏi về sức khỏe, bệnh tật, thuốc men..."
          className="flex-1 resize-none rounded-2xl border border-slate-300 bg-white px-4 py-2 text-sm shadow-sm transition-shadow focus:border-brand-500 focus:shadow-md focus:outline-none focus:ring-2 focus:ring-brand-200"
        />
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="shrink-0 rounded-full bg-gradient-to-br from-brand-500 to-brand-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-all duration-150 hover:shadow-md hover:brightness-105 active:scale-95 disabled:opacity-40 disabled:hover:shadow-sm disabled:active:scale-100"
        >
          Gửi
        </button>
      </div>
    </form>
  );
}

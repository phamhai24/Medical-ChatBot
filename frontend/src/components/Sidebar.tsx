import type { SessionEntry } from '../hooks/useSessions';

interface SidebarProps {
  sessions: SessionEntry[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onNewChat: () => void;
  onDelete: (sessionId: string) => void;
}

export function Sidebar({ sessions, activeSessionId, onSelect, onNewChat, onDelete }: SidebarProps) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-slate-50">
      <div className="p-3">
        <button
          type="button"
          onClick={onNewChat}
          className="w-full rounded-xl bg-teal-600 px-3 py-2 text-sm font-medium text-white hover:bg-teal-700"
        >
          + Cuộc trò chuyện mới
        </button>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto px-2">
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={`group flex items-center justify-between rounded-lg px-3 py-2 text-sm ${
              s.session_id === activeSessionId ? 'bg-teal-100 text-teal-900' : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            <button type="button" onClick={() => onSelect(s.session_id)} className="flex-1 truncate text-left" title={s.title}>
              {s.title}
            </button>
            <button
              type="button"
              onClick={() => onDelete(s.session_id)}
              className="ml-2 hidden text-slate-400 hover:text-red-500 group-hover:block"
              aria-label="Xóa cuộc trò chuyện"
            >
              ✕
            </button>
          </div>
        ))}
      </nav>
      <div className="border-t border-slate-200 p-3 text-center text-[11px] text-slate-400">
        ⚠️ Chatbot demo giáo dục. Không dùng cho chẩn đoán y khoa.
      </div>
    </aside>
  );
}

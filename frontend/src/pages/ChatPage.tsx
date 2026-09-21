import { useCallback, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { deleteSession as deleteSessionOnServer } from '../api/client';
import { Sidebar } from '../components/Sidebar';
import { ChatWindow } from '../components/ChatWindow';
import { InputBar } from '../components/InputBar';
import { StatusBadge } from '../components/StatusBadge';
import { useChat } from '../hooks/useChat';
import { useSessions } from '../hooks/useSessions';

export function ChatPage() {
  const { sessions, addSession, removeSession } = useSessions();
  const { messages, isSending, send, hydrate, reset } = useChat();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const titledRef = useRef(false);

  const selectSession = useCallback(
    async (sessionId: string) => {
      setActiveSessionId(sessionId);
      titledRef.current = true;
      await hydrate(sessionId);
    },
    [hydrate],
  );

  const handleNewChat = useCallback(() => {
    setActiveSessionId(null);
    titledRef.current = false;
    reset();
  }, [reset]);

  const handleSend = useCallback(
    async (text: string, topK: number) => {
      const isFirstMessage = messages.length === 0 && !titledRef.current;
      await send(text, activeSessionId, topK, (sessionId) => {
        setActiveSessionId((prev) => prev ?? sessionId);
        if (isFirstMessage) {
          addSession(sessionId, text.slice(0, 40));
          titledRef.current = true;
        }
      });
    },
    [activeSessionId, addSession, messages.length, send],
  );

  return (
    <div
      className="flex h-screen bg-gradient-to-br from-brand-50 via-sky-50 to-brand-100 bg-[length:200%_200%] animate-gradient-shift"
    >
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={selectSession}
        onNewChat={handleNewChat}
        onDelete={(id) => {
          removeSession(id);
          if (id === activeSessionId) handleNewChat();
          void deleteSessionOnServer(id).catch(() => {
            // best-effort: the Redis-backed history still expires via its own 30-day TTL
          });
        }}
      />
      <div className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-white/60 bg-white/70 px-4 py-3 backdrop-blur-sm sm:px-8">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 text-lg shadow-sm">
              🏥
            </span>
            <div>
              <h1 className="text-base font-semibold text-slate-800">Medical RAG Chatbot</h1>
              <p className="text-xs text-slate-400">Trợ lý y tế sử dụng Retrieval-Augmented Generation</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <StatusBadge />
            <Link
              to="/admin"
              className="text-xs font-medium text-brand-700 transition-colors hover:text-brand-500 hover:underline"
            >
              Quản trị
            </Link>
          </div>
        </header>
        <ChatWindow messages={messages} />
        <InputBar onSend={handleSend} disabled={isSending} />
      </div>
    </div>
  );
}

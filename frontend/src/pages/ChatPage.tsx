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
    <div className="flex h-screen bg-slate-100">
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
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 sm:px-8">
          <div>
            <h1 className="text-base font-semibold text-slate-800">🏥 Medical RAG Chatbot</h1>
            <p className="text-xs text-slate-400">Trợ lý y tế sử dụng Retrieval-Augmented Generation</p>
          </div>
          <div className="flex items-center gap-4">
            <StatusBadge />
            <Link to="/admin" className="text-xs font-medium text-teal-700 hover:underline">
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

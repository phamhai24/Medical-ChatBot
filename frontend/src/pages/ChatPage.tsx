import { useCallback, useEffect, useRef, useState } from 'react';
import { Icon } from '../components/Icon';
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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessionError, setSessionError] = useState('');
  const [isLoadingSession, setIsLoadingSession] = useState(false);

  useEffect(() => {
    if (!sidebarOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSidebarOpen(false);
        document.querySelector<HTMLButtonElement>('[aria-controls="chat-sidebar"]')?.focus();
      }
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [sidebarOpen]);

  const selectSession = useCallback(
    async (sessionId: string) => {
      setIsLoadingSession(true);
      try {
        await hydrate(sessionId);
        setActiveSessionId(sessionId);
        titledRef.current = true;
        setSessionError('');
      } catch {
        setSessionError('Không tải được cuộc trò chuyện. Vui lòng thử lại.');
      } finally {
        setIsLoadingSession(false);
      }
    },
    [hydrate],
  );

  const handleNewChat = useCallback(() => {
    setSessionError('');
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
      className="app-shell"
    >
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        disabled={isSending || isLoadingSession}
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
      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-title"><button className="mobile-menu tool-button" aria-label="Mở lịch sử trò chuyện" aria-controls="chat-sidebar" aria-expanded={sidebarOpen} onClick={() => setSidebarOpen(v => !v)}><Icon name="menu" /></button><span className="assistant-mark"><Icon name="spark" size={18} /></span><span>Trợ lý sức khỏe<span className="ai-label">AI</span></span></div>
          <div className="topbar-right"><StatusBadge /><span className="topbar-separator" /><span className="private-label">Không gian của bạn</span><span className="profile-mark">M</span></div>
        </header>
        {sessionError && <div role="alert" className="session-error">{sessionError}<button onClick={() => setSessionError('')} aria-label="Đóng thông báo"><Icon name="close" size={16} /></button></div>}
        <ChatWindow messages={messages} onSuggest={text => { if (!isSending && !isLoadingSession) void handleSend(text, 5); }} disabled={isSending || isLoadingSession} />
        <InputBar onSend={handleSend} disabled={isSending || isLoadingSession} />
      </main>
    </div>
  );
}

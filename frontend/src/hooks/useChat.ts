import { useCallback, useState } from 'react';
import { askChat, getHistory, streamChat } from '../api/client';
import type { ChatMessage, ChatSource } from '../types';

export interface UiMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
  sourcesCount?: number;
  latencyMs?: number;
}

export function useChat() {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [isSending, setIsSending] = useState(false);

  const hydrate = useCallback(async (sessionId: string) => {
    const history = await getHistory(sessionId);
    setMessages(
      history.messages.map((m: ChatMessage) => ({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.content,
        sourcesCount: typeof m.metadata?.sources === 'number' ? (m.metadata.sources as number) : undefined,
        latencyMs: typeof m.metadata?.latency_ms === 'number' ? (m.metadata.latency_ms as number) : undefined,
      })),
    );
  }, []);

  const reset = useCallback(() => setMessages([]), []);

  const send = useCallback(
    async (text: string, sessionId: string | null, topK: number, onSessionId: (id: string) => void) => {
      setMessages((prev) => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '' }]);
      setIsSending(true);

      let streamSessionId: string | null = null;

      try {
        let receivedAnyChunk = false;
        await streamChat({ message: text, top_k: topK, session_id: sessionId ?? undefined }, (chunk) => {
          receivedAnyChunk = true;
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + chunk };
            return next;
          });
        }, (id) => {
          streamSessionId = id;
        });
        if (!receivedAnyChunk) {
          throw new Error('Stream resolved with no chunks');
        }
        if (streamSessionId) onSessionId(streamSessionId);
      } catch {
        try {
          const res = await askChat({
            message: text,
            top_k: topK,
            session_id: sessionId ?? undefined,
            include_sources: true,
          });
          if (res.session_id) onSessionId(res.session_id);
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = {
              role: 'assistant',
              content: res.answer,
              sources: res.sources,
              latencyMs: res.latency_ms,
            };
            return next;
          });
        } catch (fallbackError) {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = {
              role: 'assistant',
              content: `⚠️ Lỗi: ${(fallbackError as Error).message}`,
            };
            return next;
          });
        }
      } finally {
        setIsSending(false);
      }
    },
    [],
  );

  return { messages, isSending, send, hydrate, reset };
}

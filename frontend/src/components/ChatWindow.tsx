import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import type { UiMessage } from '../hooks/useChat';

export function ChatWindow({ messages }: { messages: UiMessage[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center text-center text-slate-400">
        <div>
          <p className="text-lg font-medium text-slate-500">🏥 Trợ lý Y tế</p>
          <p className="mt-1 text-sm">Hỏi về sức khỏe, bệnh tật, thuốc men...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-8">
      {messages.map((m, i) => (
        <MessageBubble key={i} message={m} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

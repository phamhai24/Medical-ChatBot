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
        <div className="animate-bubble-in">
          <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-400 to-brand-600 text-2xl shadow-md">
            🏥
          </span>
          <p className="mt-4 text-lg font-medium text-slate-600">Trợ lý Y tế</p>
          <p className="mt-1 text-sm text-slate-400">Hỏi về sức khỏe, bệnh tật, thuốc men...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="thin-scrollbar flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-8">
      {messages.map((m, i) => (
        <MessageBubble key={i} message={m} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

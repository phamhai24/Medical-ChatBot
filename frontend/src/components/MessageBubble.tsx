import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { SourcesPanel } from './SourcesPanel';
import type { UiMessage } from '../hooks/useChat';

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-2 w-2 rounded-full bg-brand-400 animate-dot-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

export function MessageBubble({ message }: { message: UiMessage }) {
  const isUser = message.role === 'user';
  const isWaiting = !isUser && message.content === '';

  return (
    <div className={`flex animate-bubble-in ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm transition-shadow hover:shadow-md ${
          isUser
            ? 'bg-gradient-to-br from-brand-500 to-brand-600 text-white'
            : 'border border-slate-200/70 bg-white text-slate-800'
        }`}
      >
        {isWaiting ? (
          <TypingDots />
        ) : (
          <div className="markdown-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        )}
        {!isUser && message.sources && message.sources.length > 0 && <SourcesPanel sources={message.sources} />}
        {!isUser && !message.sources && !!message.sourcesCount && (
          <p className="mt-2 text-xs text-slate-400">📚 {message.sourcesCount} nguồn tham khảo</p>
        )}
        {!isUser && typeof message.latencyMs === 'number' && (
          <div className="mt-2 text-xs text-slate-400">⏱️ {Math.round(message.latencyMs)}ms</div>
        )}
      </div>
    </div>
  );
}

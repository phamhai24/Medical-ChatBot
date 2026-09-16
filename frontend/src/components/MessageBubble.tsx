import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { SourcesPanel } from './SourcesPanel';
import type { UiMessage } from '../hooks/useChat';

export function MessageBubble({ message }: { message: UiMessage }) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser ? 'bg-teal-600 text-white' : 'border border-slate-200 bg-white text-slate-800'
        }`}
      >
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content || '…'}</ReactMarkdown>
        </div>
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

import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import { Icon } from './Icon';
import type { UiMessage } from '../hooks/useChat';
const suggestions = [
 { icon: 'heart' as const, label: 'Hiểu cơ thể', question: 'Làm thế nào để chăm sóc sức khỏe tim mạch?', color: 'coral' },
 { icon: 'moon' as const, label: 'Sống khỏe mỗi ngày', question: 'Tôi có thể cải thiện chất lượng giấc ngủ như thế nào?', color: 'lavender' },
 { icon: 'book' as const, label: 'Khám phá kiến thức', question: 'Cần lưu ý những gì khi sử dụng thuốc kháng sinh?', color: 'mint' },
];
export function ChatWindow({ messages, onSuggest, disabled }: { messages: UiMessage[]; onSuggest: (text: string) => void; disabled: boolean }) {
 const bottomRef = useRef<HTMLDivElement>(null);
 useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'auto' }); }, [messages]);
 if (messages.length === 0) return <div className="welcome thin-scrollbar"><section className="welcome-content">
 <div className="orb-scene" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orb"><div className="orb-sheen" /><Icon name="pulse" size={78} /></div><span className="orb-satellite satellite-one">+</span><span className="orb-satellite satellite-two"><Icon name="spark" size={17} /></span><span className="orb-caption">Một chút tò mò. Nhiều điều hữu ích.</span></div>
 <h1>Lắng nghe cơ thể.<br />Mở lối hiểu biết.</h1><p className="welcome-description">Một không gian để hỏi, khám phá và hiểu hơn về sức khỏe.<br className="desktop-break" /> Bắt đầu từ điều bạn đang quan tâm.</p>
 <div className="suggestions">{suggestions.map(item => <button key={item.label} className={`suggestion ${item.color}`} onClick={() => onSuggest(item.question)} disabled={disabled}><span className="suggestion-top"><span className="suggestion-icon"><Icon name={item.icon} /></span><Icon name="arrow" size={18} /></span><span className="suggestion-label">{item.label}</span><span className="suggestion-question">{item.question}</span></button>)}</div>
 </section></div>;
 return <div className="conversation thin-scrollbar" role="log" aria-label="Nội dung trò chuyện" aria-busy={disabled}><div className="conversation-inner">{messages.map((m, i) => <MessageBubble key={i} message={m} />)}<div ref={bottomRef} /></div></div>;
}

import { useState } from 'react';
import type { FormEvent } from 'react';
import { Icon } from './Icon';
export function InputBar({ onSend, disabled }: { onSend: (text: string, topK: number) => void; disabled: boolean }) {
 const [text, setText] = useState('');
 const [topK, setTopK] = useState(5);
 const [showAdvanced, setShowAdvanced] = useState(false);
 const submit = (e: FormEvent) => { e.preventDefault(); const trimmed = text.trim(); if (!trimmed || disabled) return; onSend(trimmed, topK); setText(''); };
 return <div className="composer-area"><form onSubmit={submit} className="composer">
 {showAdvanced && <div className="advanced-options"><label htmlFor="top-k">Số nguồn tham khảo tối đa</label><input id="top-k" type="number" min={1} max={20} value={topK} onChange={e => setTopK(Math.max(1, Math.min(20, Math.trunc(Number(e.target.value)) || 1)))} /><span>1–20 nguồn</span></div>}
 <textarea aria-label="Câu hỏi của bạn" value={text} onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) submit(e); }} rows={2} placeholder="Hôm nay, bạn muốn tìm hiểu điều gì?" disabled={disabled} />
 <div className="composer-toolbar"><div className="composer-tools"><button type="button" className={`tool-button ${showAdvanced ? 'active' : ''}`} onClick={() => setShowAdvanced(v => !v)} aria-label="Tùy chọn nguồn tham khảo" aria-expanded={showAdvanced}><Icon name="settings" size={18} /></button><span className="composer-divider" /><span className="source-label"><Icon name="book" size={14} />Có nguồn tham khảo</span></div><div className="send-tools"><span className="enter-hint">Enter để gửi</span><button type="submit" className="send-button" disabled={disabled || !text.trim()} aria-label="Gửi câu hỏi"><Icon name={disabled ? 'pulse' : 'send'} size={20} /></button></div></div>
 </form><p className="disclaimer">Thông tin mang tính tham khảo, không thay thế tư vấn từ chuyên gia y tế.</p></div>;
}

import { Link } from 'react-router-dom';
import type { SessionEntry } from '../hooks/useSessions';
import { Icon } from './Icon';
interface SidebarProps {
 sessions: SessionEntry[]; activeSessionId: string | null; onSelect: (id: string) => void; onNewChat: () => void; onDelete: (id: string) => void; open: boolean; onClose: () => void; disabled: boolean;
}
export function Sidebar({ sessions, activeSessionId, onSelect, onNewChat, onDelete, open, onClose, disabled }: SidebarProps) {
 return <>
 {open && <button className="sidebar-backdrop" onClick={onClose} aria-label="Đóng lịch sử" />}
 <aside id="chat-sidebar" className={`sidebar ${open ? 'is-open' : ''}`}>
 <Link to="/" className="brand"><span className="brand-symbol"><Icon name="pulse" size={26} /></span><span>medora<span className="brand-dot">.</span><small>Không gian tri thức sức khỏe</small></span></Link>
 <button className="new-chat" onClick={() => { onNewChat(); onClose(); }} disabled={disabled}><Icon name="plus" />Cuộc trò chuyện mới<kbd>+</kbd></button>
 <div className="history-heading"><span>Cuộc trò chuyện</span><span>{sessions.length.toString().padStart(2, '0')}</span></div>
 <nav className="session-list thin-scrollbar" aria-label="Lịch sử trò chuyện">
 {sessions.length === 0 && <div className="empty-history"><Icon name="chat" size={25} /><p>Mỗi câu hỏi là một khởi đầu.</p><span>Các cuộc trò chuyện của bạn sẽ xuất hiện ở đây.</span></div>}
 {sessions.map(s => <div key={s.session_id} className={`session-row ${s.session_id === activeSessionId ? 'selected' : ''}`}><Icon name="chat" size={16} /><button disabled={disabled} onClick={() => { onSelect(s.session_id); onClose(); }} className="session-title" title={s.title}>{s.title}</button><button disabled={disabled} onClick={() => onDelete(s.session_id)} className="delete-session" aria-label={`Xóa cuộc trò chuyện ${s.title}`}><Icon name="close" size={14} /></button></div>)}
 </nav>
 <div className="sidebar-note"><span className="note-icon"><Icon name="book" /></span><h3>Hiểu hơn. Chăm sóc tốt hơn.</h3><p>Khám phá kiến thức sức khỏe với câu trả lời có nguồn tham khảo.</p><div className="note-line" /></div>
 <Link to="/admin" className="admin-link"><Icon name="settings" size={18} />Quản trị hệ thống<Icon name="arrow" size={16} /></Link>
 <div className="sidebar-foot"><span className="tiny-cross">+</span> Medical RAG <span>Phiên bản thử nghiệm</span></div>
 </aside></>;
}

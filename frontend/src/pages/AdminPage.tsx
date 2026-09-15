import { Link } from 'react-router-dom';
import { AdminPanel } from '../components/AdminPanel';

export function AdminPage() {
  return (
    <div className="min-h-screen bg-slate-100">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
        <h1 className="text-base font-semibold text-slate-800">🗄️ Quản trị hệ thống</h1>
        <Link to="/" className="text-xs font-medium text-teal-700 hover:underline">
          ← Về Chat
        </Link>
      </header>
      <AdminPanel />
    </div>
  );
}

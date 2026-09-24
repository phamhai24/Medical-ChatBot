import { Link } from 'react-router-dom';
import { AdminPanel } from '../components/AdminPanel';

export function AdminPage() {
  return (
    <div className="admin-page">
      <header className="topbar">
        <h1 className="text-base font-semibold">Medora / Quản trị hệ thống</h1>
        <Link to="/" className="admin-back">
          ← Về Chat
        </Link>
      </header>
      <AdminPanel />
    </div>
  );
}

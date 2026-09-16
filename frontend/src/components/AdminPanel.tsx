import { useState } from 'react';
import { getHealth, getStats, ingest, reindex } from '../api/client';
import { useAdminKey } from '../hooks/useAdminKey';
import type { HealthResponse, StatsResponse } from '../types';

export function AdminPanel() {
  const { adminKey, setAdminKey } = useAdminKey();
  const [rebuild, setRebuild] = useState(false);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const appendLog = (line: string) => setLog((prev) => [line, ...prev].slice(0, 20));

  const runIngest = async () => {
    setBusy(true);
    try {
      const res = await ingest({ rebuild, batch_size: 100 }, adminKey);
      appendLog(`✅ Ingest xong: ${res.total_records} records, ${res.total_chunks} chunks, ${res.duration_seconds}s`);
    } catch (e) {
      appendLog(`❌ Ingest lỗi: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const runReindex = async () => {
    if (!window.confirm('Xóa toàn bộ vector index hiện tại?')) return;
    setBusy(true);
    try {
      const res = await reindex(adminKey);
      appendLog(`✅ ${res.message}`);
    } catch (e) {
      appendLog(`❌ Reindex lỗi: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const refreshStatus = async () => {
    try {
      const [s, h] = await Promise.all([getStats(), getHealth()]);
      setStats(s);
      setHealth(h);
    } catch (e) {
      appendLog(`❌ Không lấy được stats/health: ${(e as Error).message}`);
    }
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <section>
        <label className="block text-sm font-medium text-slate-700">Admin API Key</label>
        <input
          type="password"
          value={adminKey}
          onChange={(e) => setAdminKey(e.target.value)}
          placeholder="Nhập ADMIN_API_KEY (để trống nếu backend chưa cấu hình)"
          className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
        />
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <h2 className="text-sm font-semibold text-slate-800">📥 Ingest dữ liệu</h2>
        <label className="mt-2 flex items-center gap-2 text-sm text-slate-600">
          <input type="checkbox" checked={rebuild} onChange={(e) => setRebuild(e.target.checked)} />
          Rebuild (xóa index cũ trước khi ingest)
        </label>
        <button
          type="button"
          disabled={busy}
          onClick={runIngest}
          className="mt-3 rounded-lg bg-teal-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          Chạy Ingest
        </button>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <h2 className="text-sm font-semibold text-slate-800">🗑️ Reindex</h2>
        <p className="mt-1 text-xs text-slate-500">Xóa toàn bộ vector index hiện tại. Cần chạy Ingest lại sau đó.</p>
        <button
          type="button"
          disabled={busy}
          onClick={runReindex}
          className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          Xóa Index
        </button>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-800">📊 Trạng thái hệ thống</h2>
          <button type="button" onClick={refreshStatus} className="text-xs font-medium text-teal-700 hover:underline">
            Cập nhật
          </button>
        </div>
        {health && (
          <p className="mt-2 text-xs text-slate-600">
            Status: <span className="font-medium">{health.status}</span>
          </p>
        )}
        {stats && (
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-600">
            <div>Documents: {stats.document_count}</div>
            <div>Vector store: {stats.vector_store_type}</div>
            <div>Embedding: {stats.embedding_model.split('/').pop()}</div>
            <div>Generator: {stats.generation_model.split('/').pop()}</div>
          </div>
        )}
      </section>

      {log.length > 0 && (
        <section className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-600">
          {log.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </section>
      )}
    </div>
  );
}

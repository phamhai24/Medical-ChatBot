import { useEffect, useState } from 'react';
import { getHealth } from '../api/client';

type Status = 'checking' | 'healthy' | 'degraded' | 'unhealthy';

const DOT_COLOR: Record<Status, string> = {
  checking: 'bg-slate-300',
  healthy: 'bg-emerald-500',
  degraded: 'bg-amber-500',
  unhealthy: 'bg-red-500',
};

const LABEL: Record<Status, string> = {
  checking: 'Đang kiểm tra...',
  healthy: 'API hoạt động',
  degraded: 'API suy giảm',
  unhealthy: 'API lỗi',
};

export function StatusBadge() {
  const [status, setStatus] = useState<Status>('checking');

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const health = await getHealth();
        if (!cancelled) setStatus(health.status);
      } catch {
        if (!cancelled) setStatus('unhealthy');
      }
    }

    check();
    const interval = setInterval(check, 30000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <span className="status-badge" role="status">
      <span className={`h-2 w-2 rounded-full ${DOT_COLOR[status]} ${status === 'healthy' ? 'animate-pulse-ring' : ''}`} />
      {LABEL[status]}
    </span>
  );
}

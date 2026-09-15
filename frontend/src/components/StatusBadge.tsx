import { useEffect, useState } from 'react';
import { getHealth } from '../api/client';

type Status = 'checking' | 'healthy' | 'degraded' | 'unhealthy';

const LABEL: Record<Status, string> = {
  checking: '⏳ Đang kiểm tra...',
  healthy: '🟢 API hoạt động',
  degraded: '🟡 API suy giảm',
  unhealthy: '🔴 API lỗi',
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

  return <span className="text-xs font-medium text-slate-500">{LABEL[status]}</span>;
}

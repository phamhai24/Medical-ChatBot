import type {
  ChatHistoryResponse,
  ChatRequest,
  ChatResponse,
  ChatStreamRequest,
  HealthResponse,
  IngestRequest,
  IngestResponse,
  StatsResponse,
} from '../types';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body.error ?? detail;
    } catch {
      // non-JSON error body - keep statusText
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function askChat(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch('/api/v1/chat/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return handle<ChatResponse>(res);
}

export async function streamChat(
  req: ChatStreamRequest,
  onChunk: (text: string) => void,
  onSessionId: (sessionId: string) => void,
): Promise<void> {
  const res = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok || !res.body) {
    throw new Error(`Stream request failed: ${res.status}`);
  }

  const sessionId = res.headers.get('X-Session-ID');
  if (sessionId) onSessionId(sessionId);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    onChunk(decoder.decode(value, { stream: true }));
  }
}

export async function getHistory(sessionId: string): Promise<ChatHistoryResponse> {
  const res = await fetch(`/api/v1/chat/history/${sessionId}`);
  return handle<ChatHistoryResponse>(res);
}

export async function newSession(): Promise<{ session_id: string }> {
  const res = await fetch('/api/v1/chat/history/new', { method: 'POST' });
  return handle<{ session_id: string }>(res);
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`/api/v1/chat/history/${sessionId}`, { method: 'DELETE' });
  await handle<{ success: boolean }>(res);
}

export async function ingest(req: IngestRequest, adminKey: string): Promise<IngestResponse> {
  const res = await fetch('/api/v1/admin/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Admin-Key': adminKey },
    body: JSON.stringify(req),
  });
  return handle<IngestResponse>(res);
}

export async function reindex(adminKey: string): Promise<{ success: boolean; message: string }> {
  const res = await fetch('/api/v1/admin/reindex', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Admin-Key': adminKey },
    body: JSON.stringify({ confirm: true }),
  });
  return handle<{ success: boolean; message: string }>(res);
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch('/health');
  return handle<HealthResponse>(res);
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch('/stats');
  return handle<StatsResponse>(res);
}

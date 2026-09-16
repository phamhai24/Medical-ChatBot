import { useCallback, useEffect, useState } from 'react';

export interface SessionEntry {
  session_id: string;
  title: string;
  createdAt: number;
}

const STORAGE_KEY = 'medical-rag.sessions';

function readSessions(): SessionEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as SessionEntry[]) : [];
  } catch {
    return [];
  }
}

function writeSessions(sessions: SessionEntry[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    // localStorage unavailable (private mode, quota) - list just won't persist
  }
}

export function useSessions() {
  const [sessions, setSessions] = useState<SessionEntry[]>(() => readSessions());

  useEffect(() => {
    writeSessions(sessions);
  }, [sessions]);

  const addSession = useCallback((session_id: string, title = 'Cuộc trò chuyện mới') => {
    setSessions((prev) => [{ session_id, title, createdAt: Date.now() }, ...prev]);
  }, []);

  const renameSession = useCallback((session_id: string, title: string) => {
    setSessions((prev) => prev.map((s) => (s.session_id === session_id ? { ...s, title } : s)));
  }, []);

  const removeSession = useCallback((session_id: string) => {
    setSessions((prev) => prev.filter((s) => s.session_id !== session_id));
  }, []);

  return { sessions, addSession, renameSession, removeSession };
}

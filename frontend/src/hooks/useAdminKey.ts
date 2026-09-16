import { useCallback, useState } from 'react';

const STORAGE_KEY = 'medical-rag.admin-key';

export function useAdminKey() {
  const [adminKey, setAdminKeyState] = useState<string>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) ?? '';
    } catch {
      return '';
    }
  });

  const setAdminKey = useCallback((key: string) => {
    setAdminKeyState(key);
    try {
      localStorage.setItem(STORAGE_KEY, key);
    } catch {
      // localStorage unavailable - key stays in memory for this session only
    }
  }, []);

  return { adminKey, setAdminKey };
}

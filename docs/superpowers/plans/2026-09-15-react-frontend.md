# React Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit UI with a React + Vite + TypeScript SPA served over the existing FastAPI backend, closing two backend gaps (unauthenticated admin endpoints, wide-open CORS) along the way.

**Architecture:** The FastAPI backend (`src/api/`) is already UI-agnostic — chat, session, admin, and health endpoints stay unchanged in shape. A new `frontend/` Vite project talks to it over relative URLs (`/api/v1/...`, `/health`, `/stats`); in dev Vite's proxy forwards those to `http://localhost:8000`, in prod an nginx container does the same, so the browser never crosses origins and CORS is defense-in-depth only. Streamlit (`src/web/`) is deleted once the new UI covers chat and admin (ingest/reindex/stats).

**Tech Stack:** FastAPI (unchanged) · React 18 + TypeScript + Vite 5 + Tailwind CSS 3 + react-router-dom 6 + react-markdown 9 · nginx:alpine for the production static build.

**Spec:** `docs/superpowers/specs/2026-09-15-react-frontend-design.md`

## Global Constraints

- No end-user login system — only a single shared admin API key for `/admin` screens (from spec Non-goals).
- No dark mode (from spec Non-goals).
- No automated frontend test suite — verification is manual browser testing with both servers running (from spec Non-goals / Testing).
- Frontend calls the backend via **relative URLs only** (`/api/v1/...`, `/health`, `/stats`) — never a hardcoded `http://localhost:8000` — so dev proxy and prod nginx both work unmodified (from spec Frontend architecture).
- `ADMIN_API_KEY` unset ⇒ `/api/v1/admin/*` stays open (logs one warning); set ⇒ `X-Admin-Key` header must match exactly, else `401` (from spec Backend changes).
- `allow_credentials=False` on CORS — the frontend never sends cookies (from spec Backend changes).

---

### Task 1: Backend — admin API key authentication

**Files:**
- Modify: `src/core/config.py` (add `admin_api_key` setting)
- Modify: `src/api/deps.py` (add `verify_admin_key` dependency)
- Modify: `src/api/routes/admin.py` (require the dependency on the router)
- Modify: `.env.example` (document `ADMIN_API_KEY`)
- Test: `tests/unit/test_admin_auth.py`

**Interfaces:**
- Consumes: `src.api.deps.get_settings_dep() -> Settings` (already exists).
- Produces: `src.api.deps.verify_admin_key(x_admin_key: Optional[str], settings: Settings) -> None` — a FastAPI dependency, raises `HTTPException(401)` on mismatch, returns `None` (allow) otherwise. Later tasks (docker-compose, README) reference the env var name `ADMIN_API_KEY`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_admin_auth.py`:

```python
"""Unit tests for the admin API key dependency."""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.deps import get_settings_dep, verify_admin_key
from src.core.config import Settings


def _make_app(settings: Settings) -> FastAPI:
    app = FastAPI()
    app.dependency_overrides[get_settings_dep] = lambda: settings

    @app.get("/protected", dependencies=[Depends(verify_admin_key)])
    def protected():
        return {"ok": True}

    return app


class TestVerifyAdminKey:
    def test_allows_request_when_key_unset(self):
        client = TestClient(_make_app(Settings(admin_api_key=None)))
        response = client.get("/protected")
        assert response.status_code == 200

    def test_rejects_missing_header_when_key_set(self):
        client = TestClient(_make_app(Settings(admin_api_key="secret")))
        response = client.get("/protected")
        assert response.status_code == 401

    def test_rejects_wrong_key(self):
        client = TestClient(_make_app(Settings(admin_api_key="secret")))
        response = client.get("/protected", headers={"X-Admin-Key": "wrong"})
        assert response.status_code == 401

    def test_accepts_correct_key(self):
        client = TestClient(_make_app(Settings(admin_api_key="secret")))
        response = client.get("/protected", headers={"X-Admin-Key": "secret"})
        assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_admin_auth.py -v`
Expected: FAIL/ERROR — `verify_admin_key` does not exist yet, and `Settings` has no `admin_api_key` field (it's ignored as extra by default config, so the real failure is `ImportError: cannot import name 'verify_admin_key'`).

- [ ] **Step 3: Add the `admin_api_key` setting**

In `src/core/config.py`, add next to the other `# ─── API ──...` fields (after `api_warmup_query`):

```python
    admin_api_key: Optional[str] = None
```

- [ ] **Step 4: Add the `verify_admin_key` dependency**

In `src/api/deps.py`, add imports and the dependency at the end of the file:

```python
from typing import Optional

from fastapi import Depends, Header, HTTPException
```

(merge with the existing `from typing import Optional` import at the top rather than duplicating it)

```python
# ─── Admin Auth ───────────────────────────────────────────────────────────────

_admin_key_warning_logged = False


def verify_admin_key(
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
    settings=Depends(get_settings_dep),
) -> None:
    """Require X-Admin-Key to match ADMIN_API_KEY when it's configured."""
    global _admin_key_warning_logged

    if not settings.admin_api_key:
        if not _admin_key_warning_logged:
            logger.warning("ADMIN_API_KEY not set - /api/v1/admin/* is unauthenticated")
            _admin_key_warning_logged = True
        return

    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
```

- [ ] **Step 5: Apply the dependency to the admin router**

In `src/api/routes/admin.py`, change the imports and router declaration:

```python
from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_pipeline, verify_admin_key
from src.api.schemas import IngestRequest, IngestResponse, ReindexRequest
from src.core.logging import logger

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin"],
    dependencies=[Depends(verify_admin_key)],
)
```

(only the `router = APIRouter(...)` line and the import line change; the two endpoint functions below are untouched)

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/unit/test_admin_auth.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Document the env var**

In `.env.example`, add under `# ─── API Server ──...` (after `API_WARMUP_QUERY`):

```env
ADMIN_API_KEY=
```

- [ ] **Step 8: Run the full unit test suite to check for regressions**

Run: `pytest tests/unit -v`
Expected: PASS (no new failures beyond any pre-existing ones unrelated to this change)

- [ ] **Step 9: Commit**

```bash
git add src/core/config.py src/api/deps.py src/api/routes/admin.py .env.example tests/unit/test_admin_auth.py
git commit -m "feat(api): require X-Admin-Key on /api/v1/admin/* when ADMIN_API_KEY is set"
```

---

### Task 2: Backend — tighten CORS defaults

**Files:**
- Modify: `src/core/config.py` (`api_cors_origins` default)
- Modify: `src/api/main.py` (`allow_credentials`)
- Modify: `.env.example` (`API_CORS_ORIGINS`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Settings.api_cors_origins` now defaults to `["http://localhost:5173"]` (Vite's default dev port) instead of `["*"]"`; later tasks (docker-compose) override it per-environment via the existing `API_CORS_ORIGINS` env var.

- [ ] **Step 1: Change the Settings default**

In `src/core/config.py`, change:

```python
    api_cors_origins: list[str] = ["*"]
```

to:

```python
    api_cors_origins: list[str] = ["http://localhost:5173"]
```

- [ ] **Step 2: Turn off `allow_credentials`**

In `src/api/main.py`, in the `CORSMiddleware` block, change:

```python
        allow_credentials=True,
```

to:

```python
        allow_credentials=False,
```

- [ ] **Step 3: Update `.env.example`**

In `.env.example`, change:

```env
API_CORS_ORIGINS=["*"]
```

to:

```env
API_CORS_ORIGINS=["http://localhost:5173"]
```

- [ ] **Step 4: Verify the new default loads correctly**

Run: `python -c "from src.core.config import Settings; s = Settings(_env_file=None); print(s.api_cors_origins)"`
Expected: `['http://localhost:5173']`

- [ ] **Step 5: Run the full unit test suite to check for regressions**

Run: `pytest tests/unit -v`
Expected: PASS (existing `test_config.py` doesn't assert on `api_cors_origins`, so this should be unaffected)

- [ ] **Step 6: Commit**

```bash
git add src/core/config.py src/api/main.py .env.example
git commit -m "fix(api): default CORS to the frontend dev origin and drop allow_credentials"
```

---

### Task 3: Frontend — project scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/.gitignore`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/index.css`

**Interfaces:**
- Produces: a buildable Vite React+TS project rooted at `frontend/`, with the dev server proxying `/api`, `/health`, `/metrics` to `http://localhost:8000` (relied on by every later frontend task). `App` is the root component later tasks extend with routes.

- [ ] **Step 1: Create `package.json`**

```json
{
  "name": "medical-rag-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.39",
    "tailwindcss": "^3.4.6",
    "typescript": "^5.5.3",
    "vite": "^5.3.4"
  }
}
```

- [ ] **Step 2: Create `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "Bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [ ] **Step 3: Create `tsconfig.node.json`**

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 4: Create `vite.config.ts`**

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
    },
  },
});
```

- [ ] **Step 5: Create `tailwind.config.js`**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {},
  },
  plugins: [],
};
```

- [ ] **Step 6: Create `postcss.config.js`**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

- [ ] **Step 7: Create `index.html`**

```html
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Medical RAG Chatbot</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 8: Create `.gitignore`**

```
node_modules
dist
.env
*.local
```

- [ ] **Step 9: Create `src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

.markdown-content p {
  margin: 0.25rem 0;
}
.markdown-content ul,
.markdown-content ol {
  margin: 0.25rem 0 0.25rem 1.25rem;
  list-style: disc;
}
.markdown-content strong {
  font-weight: 600;
}
```

- [ ] **Step 10: Create a placeholder `src/App.tsx`**

```tsx
export function App() {
  return (
    <div className="flex h-screen items-center justify-center bg-slate-100 text-slate-500">
      Medical RAG Chatbot — scaffold OK
    </div>
  );
}
```

- [ ] **Step 11: Create `src/main.tsx`**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 12: Install dependencies and verify the build**

Run: `cd frontend && npm install && npm run build`
Expected: build completes and produces `frontend/dist/`. Then run `cd frontend && npm run dev` and open `http://localhost:5173` — expect to see "Medical RAG Chatbot — scaffold OK" centered on the page. Stop the dev server after confirming.

- [ ] **Step 13: Commit**

```bash
git add frontend/package.json frontend/tsconfig.json frontend/tsconfig.node.json frontend/vite.config.ts frontend/tailwind.config.js frontend/postcss.config.js frontend/index.html frontend/.gitignore frontend/src/main.tsx frontend/src/App.tsx frontend/src/index.css
git commit -m "feat(frontend): scaffold Vite + React + TypeScript + Tailwind project"
```

---

### Task 4: Frontend — shared types and API client

**Files:**
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/api/client.ts`

**Interfaces:**
- Consumes: nothing (pure types + `fetch`).
- Produces: types `ChatRequest`, `ChatSource`, `ChatResponse`, `ChatStreamRequest`, `ChatMessage`, `ChatHistoryResponse`, `IngestRequest`, `IngestResponse`, `HealthResponse`, `StatsResponse` (mirroring `src/api/schemas.py`); functions `askChat`, `streamChat`, `getHistory`, `newSession`, `deleteSession`, `ingest`, `reindex`, `getHealth`, `getStats` from `frontend/src/api/client.ts` — every later frontend task calls the backend exclusively through these.

- [ ] **Step 1: Create `src/types/index.ts`**

```typescript
export interface ChatRequest {
  message: string;
  top_k?: number;
  temperature?: number;
  include_sources?: boolean;
  session_id?: string;
}

export interface ChatSource {
  id?: string;
  question?: string;
  score?: number;
  chunk_index?: number;
  [key: string]: unknown;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  latency_ms: number;
  model: string;
  top_k: number;
  session_id?: string;
  generated_at: string;
}

export interface ChatStreamRequest {
  message: string;
  top_k?: number;
  session_id?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  metadata?: Record<string, unknown>;
  created_at?: string;
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: ChatMessage[];
  total: number;
}

export interface IngestRequest {
  data_path?: string;
  rebuild?: boolean;
  batch_size?: number;
}

export interface IngestResponse {
  success: boolean;
  total_records: number;
  total_chunks: number;
  avg_chunk_length: number;
  embedding_model: string;
  embedding_dimension: number;
  vector_store_type: string;
  documents_indexed: number;
  duration_seconds: number;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  components: Record<string, string>;
  stats: Record<string, unknown>;
  timestamp: string;
}

export interface StatsResponse {
  initialized: boolean;
  embedding_model: string;
  vector_store_type: string;
  collection: string;
  document_count: number;
  generation_model: string;
  retrieval_top_k: number;
  generator_mode: string;
}
```

- [ ] **Step 2: Create `src/api/client.ts`**

```typescript
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
```

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types/index.ts frontend/src/api/client.ts
git commit -m "feat(frontend): add typed API client for the FastAPI backend"
```

---

### Task 5: Frontend — session list and chat hooks

**Files:**
- Create: `frontend/src/hooks/useSessions.ts`
- Create: `frontend/src/hooks/useChat.ts`

**Interfaces:**
- Consumes: `askChat`, `streamChat`, `getHistory` from `frontend/src/api/client.ts` (Task 4); types `ChatMessage`, `ChatSource` from `frontend/src/types` (Task 4).
- Produces: `SessionEntry { session_id: string; title: string; createdAt: number }` and `useSessions() -> { sessions: SessionEntry[]; addSession(session_id, title?): void; renameSession(session_id, title): void; removeSession(session_id): void }`; `UiMessage { role: 'user'|'assistant'; content: string; sources?: ChatSource[]; sourcesCount?: number; latencyMs?: number }` and `useChat() -> { messages: UiMessage[]; isSending: boolean; send(text, sessionId, topK, onSessionId): Promise<void>; hydrate(sessionId): Promise<void>; reset(): void }` — both consumed by `ChatPage` in Task 6.

- [ ] **Step 1: Create `src/hooks/useSessions.ts`**

```typescript
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
```

- [ ] **Step 2: Create `src/hooks/useChat.ts`**

The backend only persists a message *count* in Redis history metadata
(`{"sources": len(response.sources)}` — see `src/api/routes/chat.py`), never
the full source list, so messages restored via `hydrate` can only carry
`sourcesCount`, not `sources`:

```typescript
import { useCallback, useState } from 'react';
import { askChat, getHistory, streamChat } from '../api/client';
import type { ChatMessage, ChatSource } from '../types';

export interface UiMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
  sourcesCount?: number;
  latencyMs?: number;
}

export function useChat() {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [isSending, setIsSending] = useState(false);

  const hydrate = useCallback(async (sessionId: string) => {
    const history = await getHistory(sessionId);
    setMessages(
      history.messages.map((m: ChatMessage) => ({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.content,
        sourcesCount: typeof m.metadata?.sources === 'number' ? (m.metadata.sources as number) : undefined,
        latencyMs: typeof m.metadata?.latency_ms === 'number' ? (m.metadata.latency_ms as number) : undefined,
      })),
    );
  }, []);

  const reset = useCallback(() => setMessages([]), []);

  const send = useCallback(
    async (text: string, sessionId: string | null, topK: number, onSessionId: (id: string) => void) => {
      setMessages((prev) => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '' }]);
      setIsSending(true);

      try {
        await streamChat({ message: text, top_k: topK, session_id: sessionId ?? undefined }, (chunk) => {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + chunk };
            return next;
          });
        }, onSessionId);
      } catch {
        try {
          const res = await askChat({
            message: text,
            top_k: topK,
            session_id: sessionId ?? undefined,
            include_sources: true,
          });
          if (res.session_id) onSessionId(res.session_id);
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = {
              role: 'assistant',
              content: res.answer,
              sources: res.sources,
              latencyMs: res.latency_ms,
            };
            return next;
          });
        } catch (fallbackError) {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = {
              role: 'assistant',
              content: `⚠️ Lỗi: ${(fallbackError as Error).message}`,
            };
            return next;
          });
        }
      } finally {
        setIsSending(false);
      }
    },
    [],
  );

  return { messages, isSending, send, hydrate, reset };
}
```

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/hooks/useSessions.ts frontend/src/hooks/useChat.ts
git commit -m "feat(frontend): add session-list and chat hooks"
```

---

### Task 6: Frontend — chat UI and routing (first working milestone)

**Files:**
- Create: `frontend/src/components/MessageBubble.tsx`
- Create: `frontend/src/components/SourcesPanel.tsx`
- Create: `frontend/src/components/ChatWindow.tsx`
- Create: `frontend/src/components/InputBar.tsx`
- Create: `frontend/src/components/Sidebar.tsx`
- Create: `frontend/src/components/StatusBadge.tsx`
- Create: `frontend/src/pages/ChatPage.tsx`
- Modify: `frontend/src/App.tsx` (add router, render `ChatPage`)

**Interfaces:**
- Consumes: `useChat`, `useSessions`, `UiMessage`, `SessionEntry` (Task 5); `getHealth`, `deleteSession` (Task 4).
- Produces: route `/` rendering the chat UI — `AdminPage` (Task 7) will be added as route `/admin` alongside it in `App.tsx`.

- [ ] **Step 1: Create `src/components/SourcesPanel.tsx`**

```tsx
import { useState } from 'react';
import type { ChatSource } from '../types';

interface SourcesPanelProps {
  sources: ChatSource[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3 border-t border-slate-100 pt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="text-xs font-medium text-teal-700 hover:underline"
      >
        {open ? '▲' : '▼'} Nguồn ({sources.length} kết quả)
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {sources.map((src, i) => {
            const score = typeof src.score === 'number' ? src.score : undefined;
            const similarity = score !== undefined ? Math.max(0, Math.min(1, 1 - score)) : undefined;
            return (
              <li key={String(src.id ?? i)} className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
                <span className="font-semibold text-slate-700">[{i + 1}] </span>
                {String(src.question ?? 'N/A')}
                {similarity !== undefined && (
                  <span className="ml-1 text-slate-400">· {similarity.toFixed(2)} similarity</span>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create `src/components/MessageBubble.tsx`**

```tsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { SourcesPanel } from './SourcesPanel';
import type { UiMessage } from '../hooks/useChat';

export function MessageBubble({ message }: { message: UiMessage }) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser ? 'bg-teal-600 text-white' : 'border border-slate-200 bg-white text-slate-800'
        }`}
      >
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content || '…'}</ReactMarkdown>
        </div>
        {!isUser && message.sources && message.sources.length > 0 && <SourcesPanel sources={message.sources} />}
        {!isUser && !message.sources && !!message.sourcesCount && (
          <p className="mt-2 text-xs text-slate-400">📚 {message.sourcesCount} nguồn tham khảo</p>
        )}
        {!isUser && typeof message.latencyMs === 'number' && (
          <div className="mt-2 text-xs text-slate-400">⏱️ {Math.round(message.latencyMs)}ms</div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create `src/components/ChatWindow.tsx`**

```tsx
import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import type { UiMessage } from '../hooks/useChat';

export function ChatWindow({ messages }: { messages: UiMessage[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center text-center text-slate-400">
        <div>
          <p className="text-lg font-medium text-slate-500">🏥 Trợ lý Y tế</p>
          <p className="mt-1 text-sm">Hỏi về sức khỏe, bệnh tật, thuốc men...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6 sm:px-8">
      {messages.map((m, i) => (
        <MessageBubble key={i} message={m} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 4: Create `src/components/InputBar.tsx`**

```tsx
import { useState } from 'react';
import type { FormEvent } from 'react';

interface InputBarProps {
  onSend: (text: string, topK: number) => void;
  disabled: boolean;
}

export function InputBar({ onSend, disabled }: InputBarProps) {
  const [text, setText] = useState('');
  const [topK, setTopK] = useState(5);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, topK);
    setText('');
  };

  return (
    <form onSubmit={submit} className="border-t border-slate-200 bg-white px-4 py-3 sm:px-8">
      {showAdvanced && (
        <div className="mb-2 flex items-center gap-2 text-xs text-slate-500">
          <label htmlFor="top-k">top_k</label>
          <input
            id="top-k"
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-16 rounded border border-slate-300 px-2 py-1"
          />
        </div>
      )}
      <div className="flex items-end gap-2">
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="shrink-0 rounded-full border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-slate-50"
          aria-label="Tùy chọn nâng cao"
        >
          ⚙️
        </button>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit(e as unknown as FormEvent);
            }
          }}
          rows={1}
          placeholder="Hỏi về sức khỏe, bệnh tật, thuốc men..."
          className="flex-1 resize-none rounded-2xl border border-slate-300 px-4 py-2 text-sm focus:border-teal-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="shrink-0 rounded-full bg-teal-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          Gửi
        </button>
      </div>
    </form>
  );
}
```

- [ ] **Step 5: Create `src/components/Sidebar.tsx`**

```tsx
import type { SessionEntry } from '../hooks/useSessions';

interface SidebarProps {
  sessions: SessionEntry[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onNewChat: () => void;
  onDelete: (sessionId: string) => void;
}

export function Sidebar({ sessions, activeSessionId, onSelect, onNewChat, onDelete }: SidebarProps) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-slate-50">
      <div className="p-3">
        <button
          type="button"
          onClick={onNewChat}
          className="w-full rounded-xl bg-teal-600 px-3 py-2 text-sm font-medium text-white hover:bg-teal-700"
        >
          + Cuộc trò chuyện mới
        </button>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto px-2">
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={`group flex items-center justify-between rounded-lg px-3 py-2 text-sm ${
              s.session_id === activeSessionId ? 'bg-teal-100 text-teal-900' : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            <button type="button" onClick={() => onSelect(s.session_id)} className="flex-1 truncate text-left" title={s.title}>
              {s.title}
            </button>
            <button
              type="button"
              onClick={() => onDelete(s.session_id)}
              className="ml-2 hidden text-slate-400 hover:text-red-500 group-hover:block"
              aria-label="Xóa cuộc trò chuyện"
            >
              ✕
            </button>
          </div>
        ))}
      </nav>
      <div className="border-t border-slate-200 p-3 text-center text-[11px] text-slate-400">
        ⚠️ Chatbot demo giáo dục. Không dùng cho chẩn đoán y khoa.
      </div>
    </aside>
  );
}
```

- [ ] **Step 6: Create `src/components/StatusBadge.tsx`**

```tsx
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
```

- [ ] **Step 7: Create `src/pages/ChatPage.tsx`**

```tsx
import { useCallback, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { deleteSession as deleteSessionOnServer } from '../api/client';
import { Sidebar } from '../components/Sidebar';
import { ChatWindow } from '../components/ChatWindow';
import { InputBar } from '../components/InputBar';
import { StatusBadge } from '../components/StatusBadge';
import { useChat } from '../hooks/useChat';
import { useSessions } from '../hooks/useSessions';

export function ChatPage() {
  const { sessions, addSession, removeSession } = useSessions();
  const { messages, isSending, send, hydrate, reset } = useChat();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const titledRef = useRef(false);

  const selectSession = useCallback(
    async (sessionId: string) => {
      setActiveSessionId(sessionId);
      titledRef.current = true;
      await hydrate(sessionId);
    },
    [hydrate],
  );

  const handleNewChat = useCallback(() => {
    setActiveSessionId(null);
    titledRef.current = false;
    reset();
  }, [reset]);

  const handleSend = useCallback(
    async (text: string, topK: number) => {
      const isFirstMessage = messages.length === 0 && !titledRef.current;
      await send(text, activeSessionId, topK, (sessionId) => {
        setActiveSessionId((prev) => prev ?? sessionId);
        if (isFirstMessage) {
          addSession(sessionId, text.slice(0, 40));
          titledRef.current = true;
        }
      });
    },
    [activeSessionId, addSession, messages.length, send],
  );

  return (
    <div className="flex h-screen bg-slate-100">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={selectSession}
        onNewChat={handleNewChat}
        onDelete={(id) => {
          removeSession(id);
          if (id === activeSessionId) handleNewChat();
          void deleteSessionOnServer(id).catch(() => {
            // best-effort: the Redis-backed history still expires via its own 30-day TTL
          });
        }}
      />
      <div className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 sm:px-8">
          <div>
            <h1 className="text-base font-semibold text-slate-800">🏥 Medical RAG Chatbot</h1>
            <p className="text-xs text-slate-400">Trợ lý y tế sử dụng Retrieval-Augmented Generation</p>
          </div>
          <div className="flex items-center gap-4">
            <StatusBadge />
            <Link to="/admin" className="text-xs font-medium text-teal-700 hover:underline">
              Quản trị
            </Link>
          </div>
        </header>
        <ChatWindow messages={messages} />
        <InputBar onSend={handleSend} disabled={isSending} />
      </div>
    </div>
  );
}
```

- [ ] **Step 8: Wire routing into `src/App.tsx`**

Replace the whole file:

```tsx
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ChatPage } from './pages/ChatPage';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
      </Routes>
    </BrowserRouter>
  );
}
```

(Task 7 adds the `/admin` route back into this file.)

- [ ] **Step 9: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: no errors.

- [ ] **Step 10: Manual verification**

Run in two terminals:
1. `python -m src.api.main` (repo root; requires the vector store already ingested — if `data/vectorstore` is empty, run `python -m src.rag.ingest --config config/rag_config.yaml --rebuild` first)
2. `cd frontend && npm run dev`

Open `http://localhost:5173` and confirm:
- The status badge shows "🟢 API hoạt động".
- Sending a message streams an answer into a bubble with a typewriter effect, sources are expandable, latency shows.
- "+ Cuộc trò chuyện mới" clears the window and starts a new session; the old conversation still appears in the sidebar and can be reopened (reloading history).
- Reloading the page and clicking a sidebar entry restores its messages (with a "📚 N nguồn tham khảo" caption instead of expandable cards, since hydrated history only carries the count).

- [ ] **Step 11: Commit**

```bash
git add frontend/src/components frontend/src/pages/ChatPage.tsx frontend/src/App.tsx
git commit -m "feat(frontend): chat UI with streaming, sidebar sessions, and history restore"
```

---

### Task 7: Frontend — admin UI

**Files:**
- Create: `frontend/src/hooks/useAdminKey.ts`
- Create: `frontend/src/components/AdminPanel.tsx`
- Create: `frontend/src/pages/AdminPage.tsx`
- Modify: `frontend/src/App.tsx` (add `/admin` route)

**Interfaces:**
- Consumes: `ingest`, `reindex`, `getStats`, `getHealth` (Task 4); types `HealthResponse`, `StatsResponse` (Task 4).
- Produces: route `/admin`, completing the routes referenced by the `Sidebar`/`ChatPage` "Quản trị" link from Task 6.

- [ ] **Step 1: Create `src/hooks/useAdminKey.ts`**

```typescript
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
```

- [ ] **Step 2: Create `src/components/AdminPanel.tsx`**

```tsx
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
```

- [ ] **Step 3: Create `src/pages/AdminPage.tsx`**

```tsx
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
```

- [ ] **Step 4: Add the `/admin` route in `src/App.tsx`**

```tsx
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ChatPage } from './pages/ChatPage';
import { AdminPage } from './pages/AdminPage';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/admin" element={<AdminPage />} />
      </Routes>
    </BrowserRouter>
  );
}
```

- [ ] **Step 5: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: no errors.

- [ ] **Step 6: Manual verification**

With both servers still running (Task 6, Step 10):
- Navigate to `http://localhost:5173/admin` (or click "Quản trị").
- Click "Cập nhật" under "Trạng thái hệ thống" — confirm stats/health populate.
- Leave the Admin API Key field empty and click "Chạy Ingest" — since `ADMIN_API_KEY` is unset by default, this should succeed (backend logs a warning, doesn't reject).
- Set `ADMIN_API_KEY=test123` in `.env`, restart `python -m src.api.main`, reload `/admin`, try "Chạy Ingest" with an empty key — expect the `❌ Ingest lỗi: ...` log line (401); then paste `test123` into the field and retry — expect success.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/hooks/useAdminKey.ts frontend/src/components/AdminPanel.tsx frontend/src/pages/AdminPage.tsx frontend/src/App.tsx
git commit -m "feat(frontend): admin page for ingest, reindex, and system status"
```

---

### Task 8: Frontend — Docker image

**Files:**
- Create: `frontend/Dockerfile`
- Create: `frontend/nginx.conf`

**Interfaces:**
- Consumes: `frontend/dist/` build output (from `npm run build`, Task 3-7's code).
- Produces: a `frontend:latest`-style image serving the SPA on port 80 with `/api`, `/health`, `/metrics` proxied to `http://api:8000` — consumed by the `web` service in Task 9's `docker-compose.yml`.

- [ ] **Step 1: Create `frontend/nginx.conf`**

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    location /api/ {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }

    location /health {
        proxy_pass http://api:8000/health;
    }

    location /metrics {
        proxy_pass http://api:8000/metrics;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 2: Create `frontend/Dockerfile`**

```dockerfile
# =============================================================================
# Stage 1: Build the static assets
# =============================================================================
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# =============================================================================
# Stage 2: Serve with nginx
# =============================================================================
FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost/ || exit 1
```

- [ ] **Step 3: Build the image and verify it serves the SPA**

Run: `cd frontend && docker build -t medical-rag-frontend .`
Expected: build succeeds.

Run: `docker run --rm -p 8080:80 medical-rag-frontend` then `curl -s http://localhost:8080/ | grep -o '<title>[^<]*</title>'`
Expected: `<title>Medical RAG Chatbot</title>`. Stop the container after confirming (Ctrl+C).

- [ ] **Step 4: Commit**

```bash
git add frontend/Dockerfile frontend/nginx.conf
git commit -m "feat(frontend): add production Docker image (nginx + static build)"
```

---

### Task 9: docker-compose — replace the `streamlit` service with `web`

**Files:**
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `frontend/Dockerfile` (Task 8); `admin_api_key` / `ADMIN_API_KEY` (Task 1).
- Produces: `docker compose up` starts `api`, `redis`, `web` (no more `streamlit`).

- [ ] **Step 1: Replace the `streamlit` service block**

In `docker-compose.yml`, delete the entire `# ─── Streamlit UI ──...` service block (from `streamlit:` through its `restart: unless-stopped` line) and replace it with:

```yaml
  # ─── React Frontend ──────────────────────────────────────────────────────
  web:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: medical-rag-web
    ports:
      - "${WEB_PORT:-3000}:80"
    depends_on:
      - api
    restart: unless-stopped
```

- [ ] **Step 2: Pass `ADMIN_API_KEY` through to the `api` service and tighten CORS**

In the `api` service's `environment:` block, change:

```yaml
      - API_CORS_ORIGINS=*
```

to:

```yaml
      - API_CORS_ORIGINS=["http://localhost:${WEB_PORT:-3000}"]
      - ADMIN_API_KEY=${ADMIN_API_KEY:-}
```

- [ ] **Step 3: Verify the compose file parses**

Run: `docker compose config --quiet`
Expected: no output, exit code 0 (confirms valid YAML/interpolation; does not require the images to actually build).

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "chore(compose): replace the streamlit service with the React frontend"
```

---

### Task 10: Remove Streamlit and update docs

**Files:**
- Delete: `src/web/app.py`
- Delete: `src/web/streamlit_rag_app.py`
- Delete: `src/web/__init__.py`
- Delete: `src/web/.gitignore`
- Delete: `src/web/__pycache__/` (if present)
- Modify: `requirements.txt` (remove the Streamlit section)
- Modify: `README.md` (architecture diagram, Quick Start, Project Structure, Tech Stack, API table if needed)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing consumed by later tasks — this is the final cleanup task before Task 11's verification pass.

- [ ] **Step 1: Delete the Streamlit source directory**

```bash
git rm -r src/web
```

- [ ] **Step 2: Remove Streamlit from `requirements.txt`**

Delete these lines from `requirements.txt`:

```
# ─── Streamlit UI ───────────────────────────────────────────────────────────
streamlit>=1.29.0
streamlit-chat>=0.1.1
```

- [ ] **Step 3: Update `README.md` — Architecture diagram**

Replace the `┌── Streamlit UI ──┐ ... (http://localhost:8501)` box and its arrow with:

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite + TS)                 │
│               (http://localhost:3000, dev: :5173)             │
└──────────────────────┬────────────────────────────────────────┘
                       │ HTTP / Streaming (proxied by nginx or Vite)
```

(the FastAPI Backend box and everything below it stays unchanged)

- [ ] **Step 4: Update `README.md` — Quick Start**

In the Docker section, change:

```
# API: http://localhost:8000/docs
# UI:   http://localhost:8501
```

to:

```
# API: http://localhost:8000/docs
# UI:   http://localhost:3000
```

In the Local Development section, change step 6 from:

```
# 6. (Separate terminal) Start UI
streamlit run src/web/app.py
```

to:

```
# 6. (Separate terminal) Start UI
cd frontend && npm install && npm run dev
# UI: http://localhost:5173
```

- [ ] **Step 5: Update `README.md` — Project Structure**

Remove the `│   ├── web/             # Streamlit UI` block (and its `│   │   └── app.py` line) from the `src/` tree, and add a top-level `frontend/` entry alongside `src/` and `tests/`:

```
├── frontend/             # React + Vite + TypeScript SPA
│   ├── src/
│   │   ├── api/          # Typed fetch client
│   │   ├── components/   # Chat/Admin UI components
│   │   ├── hooks/        # useChat, useSessions, useAdminKey
│   │   ├── pages/        # ChatPage, AdminPage
│   │   └── types/        # Mirrors src/api/schemas.py
│   ├── Dockerfile
│   └── nginx.conf
```

- [ ] **Step 6: Update `README.md` — Tech Stack table**

Change the `| **UI** | Streamlit | 1.29+ |` row to:

```
| **UI** | React + Vite + TypeScript + Tailwind | 18 / 5 / 5 / 3 |
```

- [ ] **Step 7: Verify no remaining references**

Run: `grep -ril streamlit --include=*.py --include=*.md --include=*.txt --include=*.yml . 2>/dev/null`
Expected: no matches (aside from unrelated historical files this task doesn't touch, e.g. `reports/` or `SYSTEM_FLOWS_REPORT.md`, which are out of scope).

- [ ] **Step 8: Run the backend unit test suite**

Run: `pytest tests/unit -v`
Expected: PASS — deleting `src/web/` must not break any backend import.

- [ ] **Step 9: Commit**

```bash
git add -A -- src/web requirements.txt README.md
git commit -m "chore: remove Streamlit UI in favor of the React frontend"
```

---

### Task 11: End-to-end verification

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Verify local (non-Docker) dev flow end-to-end**

Run in three terminals from the repo root:
1. `python -m src.api.main`
2. `cd frontend && npm run dev`
3. (only if `data/vectorstore` is empty) `python -m src.rag.ingest --config config/rag_config.yaml --rebuild`

In a browser, walk through:
- `http://localhost:5173/` — send at least 2 questions in one conversation, confirm streamed answers + sources + latency render.
- Start a new chat, send a message, confirm it appears as a new sidebar entry; switch back to the first conversation and confirm its messages are still there.
- Reload the page entirely; confirm the sidebar list survives (localStorage) and clicking an entry restores its messages via `/history/{id}`.
- `http://localhost:5173/admin` — refresh stats/health, run Ingest (rebuild off), confirm the log line shows success.

- [ ] **Step 2: Verify the Docker Compose flow**

Run: `docker compose up -d --build`
Expected: `api`, `redis`, `web` all report healthy/running (`docker compose ps`).

Open `http://localhost:3000/` and repeat the chat golden-path check from Step 1 (send a message, confirm streamed answer). Open `http://localhost:3000/admin` and confirm stats load (proxied through nginx, same-origin, no CORS errors in the browser console).

Run: `docker compose down`

- [ ] **Step 3: Final full-repo test pass**

Run: `pytest tests/ -v`
Expected: PASS (no regressions from the whole plan).

- [ ] **Step 4: Report status**

No commit for this task — if Steps 1-3 all pass, the plan is complete; if anything fails, fix it under the task that owns the broken file and re-run this task's steps before considering the plan done.

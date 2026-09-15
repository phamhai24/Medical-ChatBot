# React Frontend — Design Spec

Date: 2026-09-15
Status: Approved

## Context

The project currently ships a Streamlit UI (`src/web/app.py`) as a thin HTTP
client over an already-decoupled FastAPI backend (`src/api/`). The backend
exposes chat (ask/stream/history), admin (ingest/reindex), and system
(health/stats/metrics) endpoints. Because the backend is already
UI-agnostic, it can be replaced with a custom frontend without backend
redesign — only two pre-existing gaps need closing along the way:

- `/api/v1/admin/*` has no authentication — anyone who can reach the API
  can trigger ingestion or wipe the vector index.
- CORS is configured as `allow_origins=["*"]` with `allow_credentials=True`,
  a combination browsers reject once credentials are actually used, and
  wide open regardless.

## Goals

- Replace Streamlit entirely with a React + Vite + TypeScript SPA styled
  with Tailwind CSS, in a ChatGPT/Claude-like layout (sidebar + chat).
- Close the two backend gaps above as part of this work.
- Ship a docker-compose setup where the new frontend replaces the
  `streamlit` service.

## Non-goals

- No end-user authentication/login system (single shared admin key only,
  for the admin/ingest screen).
- No dark mode.
- No automated frontend test suite (manual browser verification is the
  bar for this change).
- No backend feature changes beyond the admin-auth and CORS fixes.

## Backend changes

**Admin auth** (`src/core/config.py`, `src/api/deps.py`, `src/api/routes/admin.py`):
- Add `admin_api_key: Optional[str] = None` to `Settings`, sourced from
  the `ADMIN_API_KEY` env var.
- Add a `verify_admin_key` FastAPI dependency in `src/api/deps.py`:
  - If `settings.admin_api_key` is unset, log a warning once and allow
    the request through (keeps zero-config local dev working).
  - If set, require the request header `X-Admin-Key` to match exactly,
    else raise `HTTPException(401)`.
- Apply `Depends(verify_admin_key)` to both routes in
  `src/api/routes/admin.py` (`/ingest`, `/reindex`).

**CORS** (`src/api/main.py`, `src/core/config.py`):
- Keep `api_cors_origins` as a configurable list (default now
  `["http://localhost:5173"]` for Vite dev instead of `["*"]"`).
- Set `allow_credentials=False` (the frontend never sends cookies; the
  session id and admin key travel as explicit headers/body fields).
- In docker-compose/production, the frontend is served through nginx
  which proxies `/api`, `/health`, `/metrics` to the `api` service, so
  the browser only ever talks to one origin and CORS is not on the hot
  path — this setting is defense-in-depth for direct API access (e.g.
  hitting `/docs` from a different host).

## Frontend architecture

New top-level `frontend/` directory (Vite + React + TypeScript + Tailwind).

```
frontend/
  src/
    api/client.ts        # typed fetch wrapper, relative URLs (/api/v1/...)
    types/                # mirrors src/api/schemas.py Pydantic models
    hooks/
      useChat.ts          # send/stream message, owns in-flight message state
      useSessions.ts      # localStorage-backed list of {session_id, title, createdAt}
      useAdminKey.ts       # localStorage-backed admin key, attaches X-Admin-Key
    components/
      Sidebar.tsx          # conversation list, New Chat, delete
      ChatWindow.tsx        # message list + autoscroll
      MessageBubble.tsx     # user/assistant bubble, markdown rendering
      SourcesPanel.tsx      # expandable citation cards
      InputBar.tsx           # textarea, send, advanced (top_k/temperature) toggle
      StatusBadge.tsx         # API health indicator (polls /health)
      AdminPanel.tsx          # /admin route: key prompt, ingest/reindex, stats/health cards
    pages/
      ChatPage.tsx
      AdminPage.tsx
    App.tsx                    # React Router: "/" -> ChatPage, "/admin" -> AdminPage
    main.tsx
  index.html
  package.json
  vite.config.ts               # dev proxy: /api, /health, /metrics -> http://localhost:8000
  tailwind.config.js
  Dockerfile                    # multi-stage: node build -> nginx serve
  nginx.conf                    # proxies /api, /health, /metrics to api:8000
```

**API client contract** (`src/api/client.ts`), one function per backend
endpoint, matching `src/api/schemas.py` types:
- `askChat(req: ChatRequest): Promise<ChatResponse>` → `POST /api/v1/chat/ask`
- `streamChat(req: ChatStreamRequest, onChunk, onSessionId): Promise<void>`
  → `POST /api/v1/chat/stream`, reads the `ReadableStream` body, forwards
  each decoded chunk to `onChunk`, reads `X-Session-ID` response header
  into `onSessionId`.
- `getHistory(sessionId): Promise<ChatHistoryResponse>` → `GET /api/v1/chat/history/{id}`
- `newSession(): Promise<{session_id: string}>` → `POST /api/v1/chat/history/new`
- `deleteSession(sessionId): Promise<void>` → `DELETE /api/v1/chat/history/{id}`
- `ingest(req: IngestRequest, adminKey): Promise<IngestResponse>` → `POST /api/v1/admin/ingest` (+ `X-Admin-Key` header)
- `reindex(adminKey): Promise<{success, message}>` → `POST /api/v1/admin/reindex` (+ `X-Admin-Key` header)
- `getHealth(): Promise<HealthResponse>` → `GET /health`
- `getStats(): Promise<StatsResponse>` → `GET /stats`

**Multi-conversation sidebar**: the backend only persists messages for a
given `session_id` (Redis-backed, no "list all sessions" endpoint), so the
list of conversations is client-side only: `useSessions` keeps
`{session_id, title, createdAt}[]` in `localStorage`. "New chat" calls
`newSession()`; selecting a past entry calls `getHistory(id)` to hydrate
`ChatWindow`. Title is set from the first user message once sent.

**Streaming UX**: `InputBar` submit calls `streamChat`; `onChunk` appends
text into the last assistant `MessageBubble` (typewriter effect). If the
stream call throws (network/parse error), fall back to `askChat` for that
same message so the user still gets an answer.

**Styling**: Tailwind, medical-calm palette (soft blue/teal accents on an
off-white/neutral base), ChatGPT-style layout — fixed-width sidebar on the
left, centered chat column, sticky input bar at the bottom. Source
citations render as expandable cards below the assistant bubble that
produced them, matching the score/question fields already returned by
`ChatResponse.sources`.

## Streamlit removal

- Delete `src/web/` (including `streamlit_rag_app.py`, `app.py`, its
  `__pycache__`).
- Remove `streamlit` (and any Streamlit-only deps) from `requirements.txt`.
- Update `README.md`: architecture diagram, Quick Start (`streamlit run …`
  → `cd frontend && npm run dev` / docker-compose), Project Structure,
  Tech Stack table.

## Deploy

- `frontend/Dockerfile`: stage 1 `node:20-alpine` → `npm ci && npm run
  build`; stage 2 `nginx:alpine` copying the static `dist/` output plus
  `nginx.conf`.
- `frontend/nginx.conf`: serves the SPA (`try_files ... /index.html`) and
  reverse-proxies `/api/`, `/health`, `/metrics` to `http://api:8000`.
- `docker-compose.yml`: remove the `streamlit` service; add a `web`
  service building `frontend/Dockerfile`, `depends_on: [api]`, port
  mapping replacing the old `STREAMLIT_PORT` (e.g. `${WEB_PORT:-3000}:80`).
- Local dev (no Docker): `vite.config.ts` dev-server proxy forwards
  `/api`, `/health`, `/metrics` to `http://localhost:8000`, so
  `npm run dev` needs no CORS config and no `.env` base URL.

## Testing

- Backend: add `tests/unit/test_admin_auth.py` covering `verify_admin_key`
  — request without `X-Admin-Key` when `ADMIN_API_KEY` is set → 401;
  correct key → passes through; `ADMIN_API_KEY` unset → passes through
  (with a logged warning).
- Frontend: no automated suite. Verification is running
  `python -m src.api.main` + `npm run dev` together and exercising the
  golden path (send a message, see streamed answer + sources, open a
  second chat, switch between them, reload and confirm history restores,
  hit `/admin`, run ingest with and without a valid key) in a real
  browser before calling the work done.

## Risks / open questions

- `generate_streaming`'s `callback` in `src/rag/generator.py` is assumed
  to be synchronous per existing `/chat/stream` route; no changes to that
  contract are in scope, the frontend only consumes the existing
  plain-text stream.
- If `ADMIN_API_KEY` is left unset in a public deployment, `/admin/*`
  stays open — this mirrors today's behavior by default and only tightens
  when the operator opts in by setting the env var, which is documented
  in `README.md`.

/** An earlier message, sent so the backend can resolve follow-up questions. */
export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string;
  history?: HistoryMessage[];
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
  /** The [n] label this source was shown as in the LLM's context — display
   * citations under this number so it matches any inline "[n]" markers the
   * model wrote in the answer, instead of renumbering from 1. */
  position?: number;
  /** Excerpt of the actual retrieved chunk text. Source records in this corpus
   * can be very long articles (one record can be 100+ chunks), so `question`
   * alone (the whole article's title) doesn't reliably describe what this
   * specific chunk says — show the excerpt too. */
  snippet?: string;
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
  history?: HistoryMessage[];
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

# Báo cáo đánh giá hệ thống Medical RAG Chatbot

- Thời gian chạy: 2026-06-11T15:25:15+07:00 -> 2026-06-11T15:26:00+07:00
- Trạng thái tổng quát: **CẦN CẢI THIỆN CHẤT LƯỢNG TRUY HỒI**
- Bộ đánh giá: 12 câu hỏi lấy từ chính `data/processed/data.json`
- File JSON chi tiết: `evaluation_report_20260611_152515.json`

## 1. Kết luận nhanh

- Vectorstore/Chroma đọc được dữ liệu: warm-up báo `document_count=708714` và `probe_results=1`.
- Retrieval trên benchmark bám corpus đạt hit rate `66.67%`, MRR trung bình `0.6667`.
- Generation thành công `8/12` câu; lỗi generation `4` câu.
- Provider Groq trả `429 Too Many Requests` khi gọi liên tiếp; đây là lỗi giới hạn tốc độ API, không phải lỗi Chroma/HNSW.

## 2. Chỉ số tổng hợp

| Nhóm | Chỉ số | Giá trị |
|---|---:|---:|
| Execution | Tổng thời gian | 45.14 giây |
| Execution | Benchmark questions | 12 |
| Execution | Generation success | 8 |
| Execution | Generation errors | 4 |
| Retrieval | Hit rate | 66.67% |
| Retrieval | Precision@K trung bình | 0.5333 |
| Retrieval | Recall proxy@K trung bình | 0.6667 |
| Retrieval | MRR trung bình | 0.6667 |
| Generation | Term coverage trung bình | 0.8333 |
| Generation | Context grounding trung bình | 0.7500 |
| Generation | Tỷ lệ trả lời 'không tìm thấy' | 0.00% |
| Generation | Tỷ lệ có khuyến cáo y tế | 0.00% |
| Latency | End-to-end trung bình | 0.8334 giây |
| Latency | End-to-end P50 | 0.7305 giây |
| Latency | End-to-end P95 | 1.4429 giây |
| Latency | Retrieval trung bình | 0.0301 giây |
| Latency | Generation trung bình | 0.5613 giây |

## 3. Cấu hình được kiểm tra

| Thành phần | Giá trị |
|---|---|
| `data_records` | `16506` |
| `processed_data_path` | `C:\Users\pc\Documents\code\Medical_RAG_Chatbot\data\processed\data.json` |
| `embedding_model` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `embedding_device` | `cuda` |
| `vectorstore_type` | `chroma` |
| `vectorstore_path` | `C:\Users\pc\Documents\code\Medical_RAG_Chatbot\data\vectorstore` |
| `collection_name` | `medical_qa` |
| `vector_search_type` | `similarity` |
| `retrieval_top_k` | `5` |
| `retrieval_score_threshold` | `0.3` |
| `generator_mode` | `api` |
| `generator_provider` | `groq` |
| `generator_model` | `llama-3.3-70b-versatile` |
| `api_host` | `0.0.0.0` |
| `api_port` | `8000` |
| `python` | `3.10.20` |
| `platform` | `Windows-10-10.0.26200-SP0` |

## 4. Warm-up/vectorstore

| Trường | Giá trị |
|---|---|
| `document_count` | `708714` |
| `probe_results` | `1` |
| `generator_mode` | `api` |
| `timings` | `{'embedder_load_seconds': 10.798, 'probe_embedding_seconds': 0.264, 'vector_count_seconds': 6.396, 'probe_search_seconds': 0.004, 'generator_warmup_seconds': 0.109, 'total_seconds': 17.571}` |

Nhận xét: persisted Chroma/HNSW đã load được, count được collection và query probe trả kết quả. Vì vậy lỗi hiện tại, nếu chatbot trả lời không đúng, nghiêng về chất lượng retrieval/context/generation hơn là lỗi index không load được.

## 5. Kết quả từng câu

| ID | Câu hỏi | Status | Hit | P@K | MRR | Retrieval | Generation | Top metadata question |
|---|---|---|---:|---:|---:|---:|---:|---|
| q01 | Phương pháp điều trị áp xe não do amip | success | True | 0.6 | 1.0 | 0.03s | 0.9985s | Cách điều trị áp xe não do amip |
| q02 | Acid uric là gì? | success | False | 0.0 | 0.0 | 0.028s | 0.5464s | Chỉ định xét nghiệm acid uric máu |
| q03 | Tăng acid uric máu và bệnh Gout | success | True | 0.2 | 1.0 | 0.0297s | 1.5949s | Tăng acid uric máu và bệnh Gout |
| q04 | Triệu chứng lâm sàng và chẩn đoán bệnh | success | True | 1.0 | 1.0 | 0.0326s | 0.7382s | Một số biểu hiện của bệnh |
| q05 | Dùng thuốc trong điều trị áp xe não | success | False | 0.0 | 0.0 | 0.0295s | 0.3905s | Điều trị hôn mê như thế nào? |
| q06 | Dự phòng nhiễm amip | success | False | 0.0 | 0.0 | 0.0318s | 0.4777s | Tổng quan về bệnh bạch hầu |
| q07 | Tầm quan trọng của chế độ ăn cho người bị áp xe não do amip | success | True | 1.0 | 1.0 | 0.0316s | 0.8584s | Tầm quan trọng của chế độ ăn cho người bị áp xe não do amip |
| q08 | Tham khảo chế độ dinh dưỡng cho người bị áp xe não do amip | success | True | 1.0 | 1.0 | 0.031s | 0.6046s | Các chất dinh dưỡng cần thiết cho người bệnh bại não |
| q09 | Vai trò của tập luyện với người bệnh áp xe não do amip | generation_error | True | 1.0 | 1.0 | 0.0291s | 0.1233s | Vai trò của tập luyện đối với người dị dạng mạch não |
| q10 | Các bài tập cho bệnh nhân áp xe não do amip | generation_error | True | 0.6 | 1.0 | 0.0283s | 0.1487s | Các bài tập cho bệnh nhân áp xe não do amip |
| q11 | Những lưu ý dành cho người áp xe não do amip khi tập luyện | generation_error | True | 1.0 | 1.0 | 0.0304s | 0.1283s | Những lưu ý dành cho người áp xe não do amip khi tập luyện |
| q12 | Đông y có chữa được áp xe não do amip không? | generation_error | False | 0.0 | 0.0 | 0.0297s | 0.1256s | Áp xe não do amip có chữa khỏi được không? |

## 6. Phát hiện chính

- Retriever chưa tìm lại ổn ngay cả với câu hỏi bám corpus. Cần ưu tiên kiểm tra embedding model, dữ liệu chunk, search params và khả năng trùng/loãng dữ liệu.
- API generation bị rate limit `429` khi benchmark gọi nhiều câu liên tiếp. Khi demo/chat thật, nên thêm retry/backoff hoặc giảm tốc độ gọi.
- Latency retrieval rất thấp so với generation; phần chậm chính nằm ở generator API và cold/warm state, không phải Chroma search.
- `RAGPipeline.query()` hiện retrieve lặp lại nhiều lần cho cùng một câu hỏi: một lần lấy docs, một lần build context, và một lần build sources.
- Benchmark có sẵn trong repo nên được chuẩn hóa UTF-8 để tránh mojibake khi đo metric hoặc đọc báo cáo.

## 7. Khuyến nghị

1. Thêm retry exponential backoff cho Groq/OpenAI API khi gặp `429`, đồng thời log rõ provider/model/request latency.
2. Sửa `RAGPipeline.query()` để dùng lại `retrieved_docs` khi build context và sources, không retrieve lặp lại.
3. Tạo benchmark UTF-8 chính thức gồm câu hỏi expected source/answer, rồi chạy định kỳ sau mỗi lần ingest hoặc đổi embedding model.
4. Nếu các câu hỏi phổ thông như tiểu đường/tăng huyết áp vẫn trả về Đông y không liên quan, cần bổ sung corpus phù hợp hoặc thêm reranker BM25/hybrid/rerank cross-encoder.
5. Giữ `vector_search_type=similarity`; chỉ bật threshold khi đã calibrate khoảng cách thật, vì threshold quá thấp từng làm retrieval rỗng.

## 8. Giới hạn của lần đánh giá

- Đây là smoke evaluation tự động, chưa phải đánh giá y khoa bởi chuyên gia.
- Chỉ số relevance dựa trên exact/keyword match với câu hỏi nguồn, nên chưa đo đầy đủ tính đúng-sai lâm sàng.
- Generation phụ thuộc quota/rate limit của provider tại thời điểm chạy.

## 9. Ghi chú an toàn

Chatbot y tế chỉ nên hỗ trợ tham khảo thông tin. Với triệu chứng nặng, cấp cứu, phụ nữ có thai, trẻ nhỏ, người cao tuổi hoặc bệnh nền phức tạp, hệ thống nên khuyến nghị liên hệ bác sĩ/cơ sở y tế.

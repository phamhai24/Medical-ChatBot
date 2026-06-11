# Báo cáo đánh giá lại hệ thống Medical RAG Chatbot

- Thời gian chạy: 2026-06-11T16:11:48+07:00 -> 2026-06-11T16:12:32+07:00
- Trạng thái tổng quát: **ĐẠT smoke test RAG sau tối ưu**
- Bộ đánh giá: 12 câu hỏi bám theo corpus
- File JSON chi tiết: `evaluation_report_20260611_161148.json`

## 1. Kết luận nhanh

- Vectorstore/Chroma load được: `document_count=708714`, `probe_results=1`.
- Kiểm tra tối ưu pipeline: `retrieve_calls=1`, trạng thái `ok`.
- Retrieval hit rate: `83.33%`, MRR trung bình `0.7667`.
- Generation thành công `12/12` câu; lỗi generation `0` câu.
- Có ghi nhận `429 Too Many Requests` tạm thời trong log, nhưng retry/backoff đã phục hồi thành công; lỗi generation cuối cùng là `0`.

## 2. So sánh với lần trước

| Chỉ số | Trước | Hiện tại |
|---|---:|---:|
| Retrieval hit rate | 66.67% | 83.33% |
| MRR | 0.6667 | 0.7667 |
| Generation success | 8 | 12 |
| Generation errors | 4 | 0 |
| Generation avg seconds | 0.5613 | 1.2554 |
| Retrieve calls trong `pipeline.query()` | chưa đo | 1 |

## 3. Chỉ số tổng hợp

| Nhóm | Chỉ số | Giá trị |
|---|---:|---:|
| Execution | Tổng thời gian | 44.36 giây |
| Retrieval | Hit rate | 83.33% |
| Retrieval | Precision@K trung bình | 0.6333 |
| Retrieval | Recall proxy@K trung bình | 0.8333 |
| Retrieval | MRR trung bình | 0.7667 |
| Retrieval | Avg/P50/P95 | 0.0186 / 0.0176 / 0.023 giây |
| Generation | Success/errors | 12 / 0 |
| Generation | Avg/P50/P95 | 1.2554 / 0.7555 / 3.5584 giây |
| Generation | Term coverage trung bình | 0.6528 |
| Generation | Context grounding trung bình | 0.7639 |
| Generation | Tỷ lệ trả lời 'không tìm thấy' | 8.33% |
| Generation | Tỷ lệ có khuyến cáo y tế | 83.33% |

## 4. Cấu hình được kiểm tra

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
| `api_max_retries` | `2` |
| `api_retry_backoff` | `1.0` |
| `api_retry_max_backoff` | `8.0` |
| `api_host` | `0.0.0.0` |
| `api_port` | `8000` |
| `python` | `3.10.20` |
| `platform` | `Windows-10-10.0.26200-SP0` |

## 5. Kết quả retrieval theo từng câu

| ID | Query | Hit | P@K | MRR | Retrieval | Top metadata question |
|---|---|---:|---:|---:|---:|---|
| q01 | Phương pháp điều trị áp xe não do amip | True | 0.4 | 1.0 | 0.0183s | Cách điều trị áp xe não do amip |
| q02 | Dùng thuốc trong điều trị áp xe não | False | 0.0 | 0.0 | 0.0174s | Các phương pháp điều trị áp xe não |
| q03 | Nguyên nhân gây viêm tai xương chũm | True | 0.6 | 1.0 | 0.0178s | Viêm tai xương chũm là gì? |
| q04 | Acid uric là gì? | False | 0.0 | 0.0 | 0.0162s | Chỉ định xét nghiệm acid uric máu |
| q05 | Tăng acid uric máu và bệnh Gout | True | 0.4 | 1.0 | 0.0215s | Tăng acid uric máu và bệnh Gout |
| q06 | Triệu chứng lâm sàng và chẩn đoán bệnh | True | 1.0 | 1.0 | 0.0164s | Một số biểu hiện của bệnh |
| q07 | Virus cúm A lây qua đường nào? | True | 1.0 | 1.0 | 0.0166s | Phân loại virus cúm |
| q08 | Đông y có chữa được Alkapton niệu không? | True | 1.0 | 1.0 | 0.0165s | Các phương pháp điều trị Alkapton niệu |
| q09 | Điều trị áp xe gan do amip | True | 1.0 | 1.0 | 0.0219s | Biện pháp điều trị áp xe gan do amip |
| q10 | Tầm quan trọng của chế độ ăn cho người bị áp xe não do amip | True | 1.0 | 1.0 | 0.0244s | Tầm quan trọng của chế độ ăn cho người bị áp xe não do amip |
| q11 | Vai trò của tập luyện với người bệnh áp xe não do amip | True | 1.0 | 1.0 | 0.017s | Vai trò của tập luyện đối với người bị amip ăn não |
| q12 | Dự phòng nhiễm amip | True | 0.2 | 0.2 | 0.0186s | Cách điều trị bạch hầu thanh quản |

## 6. Kết quả generation theo từng câu

| ID | Status | Generation | Error/preview |
|---|---|---:|---|
| q01 | success | 0.9385s | Để điều trị áp xe não do amip, các bác sĩ sẽ áp dụng cả nội khoa và ngoại khoa. Phác đồ điều trị cần sử dụn... |
| q02 | success | 1.1716s | Trong điều trị áp xe não, thuốc được sử dụng để loại bỏ nguồn lây nhiễm và giảm áp lực nội sọ. Một loại thu... |
| q03 | success | 0.9078s | Nguyên nhân gây viêm tai xương chũm bao gồm: - Viêm tai giữa không được điều trị tốt. - Biến chứng của viêm... |
| q04 | success | 0.5893s | Acid uric là một chất được tạo ra khi cơ thể phân hủy các bazơ purin ngoại sinh có trong thực phẩm hoặc pur... |
| q05 | success | 0.8835s | Tăng acid uric máu và bệnh Gout có mối quan hệ chặt chẽ nhưng không phải bất kỳ ai tăng acid uric máu đều b... |
| q06 | success | 0.7177s | Dựa trên thông tin được cung cấp trong ngữ cảnh, triệu chứng lâm sàng của bệnh bao gồm: - Đau bụng dữ dội -... |
| q07 | success | 0.7716s | Virus cúm A lây qua đường hô hấp, chủ yếu là "giọt bắn" và "không khí" khi người bệnh ho, hắt hơi, những gi... |
| q08 | success | 0.6417s | Tôi không tìm thấy thông tin về việc Đông y có thể chữa được Alkapton niệu trong dữ liệu được cung cấp. Dữ ... |
| q09 | success | 0.6407s | Phương pháp chính điều trị áp xe gan do amip là điều trị nội khoa bằng thuốc chống ký sinh trùng như metron... |
| q10 | success | 0.5877s | Chế độ ăn đóng vai trò quan trọng trong việc hỗ trợ điều trị và phục hồi cho người bị áp xe não do amip. Ng... |
| q11 | success | 0.7395s | Vai trò của tập luyện với người bệnh áp xe não do amip bao gồm: - Giảm các triệu chứng stress, lo lắng; giả... |
| q12 | success | 6.4755s | Để dự phòng nhiễm amip, bạn nên thực hiện các biện pháp sau: - Rửa tay thường xuyên bằng xà phòng hoặc dung... |

## 7. Nhận xét

- Tối ưu retrieve lặp đã có hiệu lực: `pipeline.query()` chỉ gọi retrieval một lần trong smoke check.
- Vectorstore không còn là lỗi chặn chính: warm-up count và probe đều đạt.
- Retrieval bám corpus vẫn chưa đạt mức lý tưởng; bước tiếp theo nên là hybrid BM25 + vector hoặc reranker.

## 8. Khuyến nghị tiếp theo

1. Thêm hybrid retrieval BM25 + vector để cải thiện khả năng bắt đúng tên bệnh, thuốc, triệu chứng.
2. Thêm reranker cho top 20 candidates trước khi đưa top 5 vào context.
3. Thêm circuit breaker/cache thông báo thân thiện khi provider trả `429` liên tục.
4. Tạo benchmark chính thức UTF-8 có expected source/answer để đo ổn định sau mỗi lần ingest.

## 9. Ghi chú an toàn

Báo cáo này là smoke evaluation tự động, không thay thế đánh giá y khoa bởi chuyên gia.

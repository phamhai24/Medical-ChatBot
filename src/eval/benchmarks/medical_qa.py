"""Medical Q&A benchmark dataset for evaluation."""

import logging
import json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BENCHMARK_DATASET = [
    {
        "id": "med_001",
        "question": "Triệu chứng của bệnh tiểu đường type 2 là gì?",
        "expected_topics": ["tiểu đường", "type 2", "đường huyết", "triệu chứng"],
        "category": "nội tiết",
        "difficulty": "easy",
    },
    {
        "id": "med_002",
        "question": "Cách phòng ngừa bệnh cao huyết áp?",
        "expected_topics": ["cao huyết áp", "phòng ngừa", "huyết áp"],
        "category": "tim mạch",
        "difficulty": "easy",
    },
    {
        "id": "med_003",
        "question": "Khi nào cần tiêm vaccine COVID-19?",
        "expected_topics": ["vaccine", "covid-19", "tiêm chủng"],
        "category": "bệnh truyền nhiễm",
        "difficulty": "easy",
    },
    {
        "id": "med_004",
        "question": "Cách điều trị bệnh viêm amidan?",
        "expected_topics": ["viêm amidan", "điều trị", "amidan"],
        "category": "tai mũi họng",
        "difficulty": "medium",
    },
    {
        "id": "med_005",
        "question": "Bệnh gút nên ăn gì và kiêng gì?",
        "expected_topics": ["gút", "chế độ ăn", "kiêng", "uric"],
        "category": "cơ xương khớp",
        "difficulty": "medium",
    },
    {
        "id": "med_006",
        "question": "Triệu chứng bệnh viêm gan B là gì?",
        "expected_topics": ["viêm gan B", "triệu chứng", "viêm gan"],
        "category": "bệnh truyền nhiễm",
        "difficulty": "medium",
    },
    {
        "id": "med_007",
        "question": "Cách chăm sóc người bệnh sau phẫu thuật ruột thừa?",
        "expected_topics": ["ruột thừa", "phẫu thuật", "chăm sóc", "hồi phục"],
        "category": "ngoại khoa",
        "difficulty": "medium",
    },
    {
        "id": "med_008",
        "question": "Thuốc paracetamol có tác dụng phụ gì?",
        "expected_topics": ["paracetamol", "tác dụng phụ", "thuốc"],
        "category": "dược lý",
        "difficulty": "easy",
    },
    {
        "id": "med_009",
        "question": "Bệnh viêm khớp dạng thấp là gì?",
        "expected_topics": ["viêm khớp dạng thấp", "viêm khớp", "tự miễn"],
        "category": "cơ xương khớp",
        "difficulty": "medium",
    },
    {
        "id": "med_010",
        "question": "Cách xử lý khi bị bỏng nước sôi?",
        "expected_topics": ["bỏng", "xử lý", "sơ cứu"],
        "category": "cấp cứu",
        "difficulty": "easy",
    },
    {
        "id": "med_011",
        "question": "Triệu chứng và cách điều trị bệnh viêm phổi?",
        "expected_topics": ["viêm phổi", "triệu chứng", "điều trị", "phổi"],
        "category": " hô hấp",
        "difficulty": "medium",
    },
    {
        "id": "med_012",
        "question": "Bệnh trầm cảm có những dấu hiệu gì?",
        "expected_topics": ["trầm cảm", "tâm thần", "triệu chứng", "tâm lý"],
        "category": "sức khỏe tâm thần",
        "difficulty": "easy",
    },
    {
        "id": "med_013",
        "question": "Cách phòng bệnh sốt xuất huyết?",
        "expected_topics": ["sốt xuất huyết", "phòng bệnh", "muỗi"],
        "category": "bệnh truyền nhiễm",
        "difficulty": "easy",
    },
    {
        "id": "med_014",
        "question": "Bệnh hen suyễn được điều trị như thế nào?",
        "expected_topics": ["hen suyễn", "điều trị", "hen", "phổi"],
        "category": "hô hấp",
        "difficulty": "medium",
    },
    {
        "id": "med_015",
        "question": "Chỉ số BMI là gì và cách tính?",
        "expected_topics": ["bmi", "cân nặng", "chiều cao", "béo phì"],
        "category": "dinh dưỡng",
        "difficulty": "easy",
    },
    {
        "id": "med_016",
        "question": "Bệnh viêm dạ dày HP dương tính là gì?",
        "expected_topics": ["viêm dạ dày", "hp", "vi khuẩn", "dạ dày"],
        "category": "tiêu hóa",
        "difficulty": "hard",
    },
    {
        "id": "med_017",
        "question": "Cách chữa táo bón cho trẻ em?",
        "expected_topics": ["táo bón", "trẻ em", "điều trị", "tiêu hóa"],
        "category": "nhi",
        "difficulty": "easy",
    },
    {
        "id": "med_018",
        "question": "Triệu chứng bệnh nhồi máu cơ tim?",
        "expected_topics": ["nhồi máu cơ tim", "tim", "triệu chứng", "cấp cứu"],
        "category": "tim mạch",
        "difficulty": "hard",
    },
    {
        "id": "med_019",
        "question": "Vitamin D có vai trò gì cho cơ thể?",
        "expected_topics": ["vitamin d", "canxi", "xương", "dinh dưỡng"],
        "category": "dinh dưỡng",
        "difficulty": "easy",
    },
    {
        "id": "med_020",
        "question": "Bệnh Zona thần kinh là gì?",
        "expected_topics": ["zona", "thần kinh", "virus", "herpes"],
        "category": "bệnh truyền nhiễm",
        "difficulty": "medium",
    },
]


def get_benchmark_dataset() -> list[dict[str, Any]]:
    """Return the benchmark dataset."""
    file_dataset = _load_json_benchmark()
    if file_dataset:
        return file_dataset
    return BENCHMARK_DATASET


def _load_json_benchmark() -> list[dict[str, Any]]:
    """Load the UTF-8 corpus-grounded benchmark when available."""
    path = Path(__file__).parents[3] / "data" / "eval" / "medical_benchmark.json"
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        logger.warning("Could not load benchmark file %s: %s", path, exc)
        return []

    dataset = []
    for item in payload.get("items", []):
        question = item.get("question")
        if not question:
            continue
        dataset.append({
            "id": item.get("id"),
            "question": question,
            "expected_topics": item.get("expected_terms", []),
            "expected_source_question": item.get("expected_source_question"),
            "category": item.get("category", "corpus_smoke"),
            "difficulty": item.get("difficulty", "medium"),
            "source_index": item.get("source_index"),
        })
    return dataset


def get_benchmark_by_category(category: str) -> list[dict[str, Any]]:
    """Get benchmark questions filtered by category."""
    return [q for q in BENCHMARK_DATASET if q.get("category") == category]


def get_benchmark_by_difficulty(difficulty: str) -> list[dict[str, Any]]:
    """Get benchmark questions filtered by difficulty."""
    return [q for q in BENCHMARK_DATASET if q.get("difficulty") == difficulty]

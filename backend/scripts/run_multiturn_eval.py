"""Multi-turn topic-tracking evaluation against a running API.

Replays each conversation in data/eval/multiturn_topic_v1.json through
/api/v1/chat/stream the way the React UI does (sending recent history, and the
active topic if the server returns one), then asks a judge model whether each
answer is about the topic that turn expects. The number that matters is the
share of medical turns answered about the right topic; "wrong_topic" turns are
the bug users see (asking about brain abscess, getting an answer about Crohn).

Usage (API must be running; from backend/):
    python scripts/run_multiturn_eval.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_llm_graded_eval import Judge  # noqa: E402
from src.core.config import get_settings  # noqa: E402

HISTORY_MESSAGES = 6

TOPIC_PROMPT = """Bạn kiểm tra xem câu trả lời của một chatbot y tế có đúng chủ đề người dùng đang hỏi không.

Chủ đề người dùng đang hỏi: {topic}
Tin nhắn của người dùng: {message}

Câu trả lời của chatbot:
\"\"\"{answer}\"\"\"

Nhãn:
- "on_topic": câu trả lời nói về đúng chủ đề trên (kể cả khi nói không tìm thấy thông tin về chủ đề đó, hoặc khuyên đi khám)
- "wrong_topic": câu trả lời nói về một bệnh/thuốc/chủ đề KHÁC
- "unclear": không xác định được chủ đề

Chỉ trả về JSON: {{"label": "on_topic"|"wrong_topic"|"unclear", "answer_topic": "<chủ đề câu trả lời thực sự nói tới>"}}"""

NON_MEDICAL_PROMPT = """Người dùng đang trò chuyện với chatbot y tế và vừa nhắn một câu không phải câu hỏi y tế.

Tin nhắn: {message}
Câu trả lời của chatbot:
\"\"\"{answer}\"\"\"

"wrong_topic" nếu câu trả lời tự đưa ra thông tin về một bệnh/thuốc mà người dùng không hỏi; ngược lại "on_topic".
Chỉ trả về JSON: {{"label": "on_topic"|"wrong_topic", "answer_topic": "<nếu có>"}}"""


def ask_stream(client: httpx.Client, message: str, history: list, topic):
    body = {"message": message, "history": history[-HISTORY_MESSAGES:]}
    if topic:
        body["topic"] = topic
    text, done = "", {}
    with client.stream("POST", "/api/v1/chat/stream", json=body) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "chunk":
                text += event.get("text", "")
            elif event.get("type") == "done":
                done = event
            elif event.get("type") == "error":
                raise RuntimeError(event.get("message"))
    return text, done


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="data/eval/multiturn_topic_v1.json")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--label", default="run", help="Tag for the output file, e.g. baseline")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is required for the judge model.")
    judge = Judge(args.judge_model, settings.openai_api_key)
    dialogues = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["dialogues"]
    client = httpx.Client(base_url=args.base_url, timeout=180)

    rows = []
    for dlg in dialogues:
        history, topic = [], None
        for i, turn in enumerate(dlg["turns"]):
            answer, done = ask_stream(client, turn["message"], history, topic)
            topic = done.get("topic") or topic  # only present once the server tracks topics
            history += [{"role": "user", "content": turn["message"]},
                        {"role": "assistant", "content": answer}]
            if turn["topic"]:
                verdict = judge.ask(TOPIC_PROMPT.format(
                    topic=turn["topic"], message=turn["message"], answer=answer))
            else:
                verdict = judge.ask(NON_MEDICAL_PROMPT.format(message=turn["message"], answer=answer))
            rows.append({
                "dialogue": dlg["id"], "turn": i + 1, "message": turn["message"],
                "expected_topic": turn["topic"], "server_topic": done.get("topic"),
                "label": verdict.get("label"), "answer_topic": verdict.get("answer_topic"),
                "answer": answer,
            })
            mark = {"on_topic": "ok", "wrong_topic": "WRONG"}.get(verdict.get("label"), "??")
            print(f"{dlg['id']} t{i + 1} {mark:5} {turn['message'][:45]:45} -> "
                  f"{str(verdict.get('answer_topic'))[:40]}", flush=True)

    medical = [r for r in rows if r["expected_topic"]]
    follow_ups = [r for r in medical if r["turn"] > 1]
    summary = {
        "label": args.label,
        "turns": len(rows),
        "on_topic_all_medical": f"{sum(r['label'] == 'on_topic' for r in medical)}/{len(medical)}",
        "on_topic_follow_ups": f"{sum(r['label'] == 'on_topic' for r in follow_ups)}/{len(follow_ups)}",
        "wrong_topic": [f"{r['dialogue']} t{r['turn']}: {r['message']} -> {r['answer_topic']}"
                        for r in rows if r["label"] == "wrong_topic"],
        "unclear": [f"{r['dialogue']} t{r['turn']}" for r in rows if r["label"] not in ("on_topic", "wrong_topic")],
    }
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"multiturn_eval_{args.label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Raw results: {out}")


if __name__ == "__main__":
    main()

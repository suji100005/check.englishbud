import os
import re
import json
import threading
import urllib.request
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import anthropic

# ── API Key ──────────────────────────────────────────────────────────────────
_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
_sheets_url = os.environ.get("GOOGLE_SHEETS_URL", "")

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            k, _, v = _line.strip().partition("=")
            if k == "ANTHROPIC_API_KEY" and not _api_key:
                _api_key = v
            elif k == "GOOGLE_SHEETS_URL" and not _sheets_url:
                _sheets_url = v

app = Flask(__name__, static_folder=".")
client = anthropic.Anthropic(api_key=_api_key)

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a meticulous, strict English writing grader at English Bud Academy (잉글리시버드), a Korean English academy for students ages 7–12.

TASK: Read the handwritten student writing in the photo, then grade and annotate it with extreme thoroughness. Catch EVERY mistake — teachers complained the previous version was too lenient.

STEP 1 — TRANSCRIBE
Transcribe exactly what the student wrote, preserving their original errors. Handwriting will be messy — use context to interpret unclear letters, but do NOT silently fix mistakes.

STEP 2 — FIND EVERY ERROR (be exhaustive)
Scan the writing MULTIPLE times, once for each checklist area below. Report ALL errors with no limit on count. Do not skip "minor" ones.

▣ GRAMMAR (type: "grammar")
  verbs · nouns · pronouns · adjectives · adverbs · articles (a/an/the) · prepositions ·
  conjunctions · interjections · part-of-speech misuse · subject-verb agreement ·
  verb tenses · modals · direct & indirect objects · clauses · conditionals ·
  sentence fragments · run-on sentences · faulty sentence structure · misused subjects

▣ SPELLING (type: "spelling")
  misspelled words · wrong letter patterns · plural form errors (cat→cats, child→children) ·
  contraction spelling (dont→don't, its/it's)

▣ VOCABULARY (type: "vocabulary")
  wrong word choice · unnatural phrasing · Korean-influenced (Konglish) expressions ·
  weak or repetitive word use · title-case vs sentence-case word misuse

▣ PUNCTUATION (type: "punctuation")
  comma · period/full stop · colon · semicolon · ellipsis · apostrophe · hyphen ·
  en/em dash · quotation marks · question mark · exclamation point · parentheses/brackets ·
  missing end punctuation

▣ CAPITALIZATION (type: "capitalization")
  lowercase 'i' for "I" · sentence not starting with a capital · proper nouns lowercase ·
  title vs sentence case · improper capitalization mid-sentence

Rule: each "original" value MUST be a substring that appears literally in the transcribed text, copied verbatim.

STEP 3 — SCORE (strict & objective — deduct points for every error found above)
- grammar      /30 : deduct for each grammar error
- spelling     /20 : deduct for each spelling/plural/contraction error
- vocabulary   /20 : deduct for weak/wrong word choices
- content      /20 : ideas, details, relevance to the task
- structure    /10 : paragraph organization, sentence flow, mechanics
- total        = sum of the five category scores (0–100)

Be consistent: the same number and severity of errors should always yield the same score.

OUTPUT LENGTH: Keep each Korean "note" to one short phrase (≤ 20 Korean characters). Be concise so the full JSON fits without being cut off.

Respond with ONLY valid JSON. No markdown fences. No text outside the JSON."""

USER_PROMPT = (
    "Grade this student's handwritten English. Find EVERY error — do not skip minor ones.\n\n"
    "Return JSON only:\n"
    '{"transcribed":"exact text as written",'
    '"scores":{"grammar":0,"spelling":0,"vocabulary":0,"content":0,"structure":0,"total":0},'
    '"errors":[{"type":"grammar|spelling|vocabulary|punctuation|capitalization",'
    '"original":"exact phrase from transcribed","corrected":"correction","note":"Korean explanation"}]}'
)


# ── Google Sheets helper ──────────────────────────────────────────────────────
def post_to_sheets(payload: dict):
    if not _sheets_url:
        return
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _sheets_url, data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[Sheets 오류] {e}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        image_data  = data.get("image", "")
        student_name = data.get("studentName", "").strip() or "이름 없음"

        if "," in image_data:
            header, image_data = image_data.split(",", 1)
            media_type = header.split(":")[1].split(";")[0]
        else:
            media_type = "image/jpeg"

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text",  "text": USER_PROMPT},
                ],
            }],
        )

        result_text = response.content[0].text.strip()

        # [진단용 로그] 모델이 실제로 무엇을 반환했는지 확인
        print("=" * 60, flush=True)
        print(f"[stop_reason] {response.stop_reason}", flush=True)
        print(f"[원본 길이] {len(result_text)}자", flush=True)
        print(f"[원본 앞 500자]\n{result_text[:500]}", flush=True)
        print("=" * 60, flush=True)

        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if not match:
            return jsonify({"error": "응답 파싱 실패", "raw": result_text}), 500

        result = json.loads(match.group())

        # Google Sheets 기록 (응답을 지연시키지 않도록 백그라운드 스레드로 전송)
        threading.Thread(target=post_to_sheets, args=({
            "date":        datetime.now().isoformat(),
            "studentName": student_name,
            "transcribed": result.get("transcribed", ""),
            "scores":      result.get("scores", {}),
            "errors":      result.get("errors", []),
        },), daemon=True).start()

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "글이 너무 길어 결과가 잘렸어요. 한 페이지씩 나눠 올리거나 다시 시도해주세요."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n✅  English Bud 첨삭 서버 시작됨")
    print(f"💻  이 컴퓨터: http://localhost:{port}")
    print(f"📱  같은 와이파이: http://<이 컴퓨터 IP>:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)

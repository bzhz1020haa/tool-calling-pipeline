import os
import json
import time
import argparse
from pathlib import Path

try:
    import google.generativeai as genai
except Exception as e:
    genai = None  # Lazy error if not installed

def load_api_key(explicit_key: str | None) -> str:
    """Resolve API key: CLI arg > GOOGLE_API_KEY > fallback file tool_calling/1.apikey."""
    if explicit_key:
        return explicit_key
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key
    fallback = Path(__file__).with_name("1.apikey")
    if fallback.exists():
        return fallback.read_text(encoding="utf-8").strip()
    raise SystemExit("Missing Google API key. Set --api-key, or GOOGLE_API_KEY, or place it in tool_calling/1.apikey")


def read_jsonl_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines
                continue


def extract_question(conversations: list) -> str:
    """Return the first human message as the question."""
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == "human":
            return str(turn.get("value", ""))
    return ""


def generate_with_gemini(model: str, api_key: str, prompt: str, max_retries: int = 5) -> str:
    if genai is None:
        raise SystemExit("google-generativeai is not installed. pip install google-generativeai")

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)

    delay_seconds = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.generate_content(prompt)
            # Handle response text per SDK behavior
            if hasattr(resp, "text") and resp.text:
                return resp.text
            # Fallback to candidates
            candidates = getattr(resp, "candidates", None)
            if candidates:
                # Join all text parts if available
                parts = []
                for c in candidates:
                    content = getattr(c, "content", None)
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            txt = getattr(p, "text", None)
                            if txt:
                                parts.append(txt)
                if parts:
                    return "\n".join(parts)
            return ""
        except Exception:
            if attempt == max_retries - 1:
                return ""
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)


def main():
    parser = argparse.ArgumentParser(description="Fill responses with Gemini and output question-response pairs JSONL")
    parser.add_argument("--input", default=str(Path("data/2_converted/Tool-Calling-Dataset-UIGEN-X.jsonl")))
    parser.add_argument("--output", default=str(Path("data/4_filled/filled_pairs_gemini.jsonl")))
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--api-key", default=None, help="Google API key (overrides env/file)")
    parser.add_argument("--max", type=int, default=0, help="Process at most N lines (0 = all)")
    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for obj in read_jsonl_lines(in_path):
            total += 1
            if args.max and written >= args.max:
                break
            conversations = obj.get("conversations", []) if isinstance(obj, dict) else []
            question = extract_question(conversations)
            if not question:
                continue

            prompt = (
                "You are a helpful assistant. Answer the user's question clearly and concisely.\n\n"
                f"User question: {question}\n\n"
            )
            response = generate_with_gemini(args.model, api_key, prompt)

            pair = {
                "question": question,
                "response": response or "",
            }
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1

    print(f"[INFO] Processed {total} items, wrote {written} pairs to {out_path}")


if __name__ == "__main__":
    main()



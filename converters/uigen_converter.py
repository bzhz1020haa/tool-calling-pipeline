import json
import re
import argparse
from pathlib import Path
from collections import deque


THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>([\s\S]*?)</tool_call>", re.IGNORECASE)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>([\s\S]*?)</tool_response>", re.IGNORECASE)


def extract_calls_responses_and_text(assistant_text: str):
    """从文本中提取按顺序出现的 tool_call 与 tool_response 片段，并返回去标签后的剩余文本。
    返回 (calls: list[str], responses: list[str], remainder: str)
    """
    if not isinstance(assistant_text, str):
        return [], [], ""

    text_wo_think = THINK_RE.sub("", assistant_text)

    calls = [m.group(1).strip() for m in TOOL_CALL_RE.finditer(text_wo_think)]
    responses = [m.group(1).strip() for m in TOOL_RESPONSE_RE.finditer(text_wo_think)]
    remainder = TOOL_RESPONSE_RE.sub("", TOOL_CALL_RE.sub("", text_wo_think)).strip()
    return calls, responses, remainder


def parse_tool_name_from_call(call_str: str) -> str:
    """从 tool_call JSON 字符串中尝试解析 name 字段，用于生成占位 observation。失败则返回空串。"""
    try:
        # 尝试精确解析 JSON（常见为 {"name":..., "arguments":{...}}）
        obj = json.loads(call_str)
        if isinstance(obj, dict):
            name = obj.get("name")
            if isinstance(name, str):
                return name
    except Exception:
        pass

    # fallback：用正则弱匹配 "name": "xxx"
    m = re.search(r"\"name\"\s*:\s*\"([^\"]+)\"", call_str)
    return m.group(1) if m else ""


def build_observation_from_tool_response(resp_str: str) -> str:
    # 允许传入带 <tool_response> 包裹的文本；剥壳后再解析
    # 复用你已有的正则
    m = TOOL_RESPONSE_RE.search(resp_str)
    if m:
        resp_str = m.group(1).strip()

    try:
        obj = json.loads(resp_str)
        if isinstance(obj, dict):
            name = obj.get("name")
            content = obj.get("content")
            return json.dumps([{
                "type": "tool_result",
                "name": name,
                "content": content
            }], ensure_ascii=False)
        # 若有的工具直接返回数组、字符串，也包成 raw 以免丢
        return json.dumps([{"type": "raw", "text": obj}], ensure_ascii=False)
    except Exception:
        return json.dumps([{"type": "raw", "text": resp_str}], ensure_ascii=False)


def parse_conversations(conversations):
    samples = []
    current_round = []
    pending_calls = []   # 缓存 tool_call
    pending_resps = []   # 缓存 tool_response

    def append_placeholder_gpt(round_):
        round_.append({"from": "gpt", "value": "Placeholder GPT"})
        return round_

    for msg in conversations:
        role = msg.get("from")
        value = msg.get("value", "")

        if role == "gpt":
            # 只提取 tool_call（不再配对 response）
            extracted_calls, _, _ = extract_calls_responses_and_text(value)
            for call_str in extracted_calls:
                pending_calls.append(call_str)

            # ⚠️ 这里直接跳过，不生成 observation，等 tool 消息来了再配对
            continue

        elif role == "tool":
            # 提取 tool_response
            resp_strs = [m.group(1).strip() for m in TOOL_RESPONSE_RE.finditer(value)]
            if not resp_strs:
                # 如果没有标签，直接原样作为 raw
                pending_resps.append(value)
            else:
                pending_resps.extend(resp_strs)

            # 将 call 和 response 按顺序配对
            while pending_calls:
                call_str = pending_calls.pop(0)
                current_round.append({"from": "function_call", "value": call_str})

                if pending_resps:
                    resp_str = pending_resps.pop(0)
                    obs_val = build_observation_from_tool_response(resp_str)
                else:
                    tool_name = parse_tool_name_from_call(call_str)
                    obs_val = json.dumps([{
                        "type": "tool_result",
                        "name": tool_name,
                        "content": "[Stub] Placeholder response"
                    }], ensure_ascii=False)

                current_round.append({"from": "observation", "value": obs_val})

            continue

        elif role == "system":
            continue  # 跳过 system

        elif role == "human":
            # 如果有遗留的未配对 call → 填充 stub
            while pending_calls:
                call_str = pending_calls.pop(0)
                current_round.append({"from": "function_call", "value": call_str})
                if pending_resps:
                    resp_str = pending_resps.pop(0)
                    obs_val = build_observation_from_tool_response(resp_str)
                else:
                    tool_name = parse_tool_name_from_call(call_str)
                    obs_val = json.dumps([{
                        "type": "tool_result",
                        "name": tool_name,
                        "content": "[Stub] Placeholder response"
                    }], ensure_ascii=False)
                current_round.append({"from": "observation", "value": obs_val})

            if current_round:
                # 结束一轮对话
                current_round = append_placeholder_gpt(current_round)
                samples.append({"conversations": current_round})
                current_round = []

            current_round.append({"from": "human", "value": value})

        elif role == "function_call":
            # 少数场景 tool_call 单独出现
            pending_calls.append(value)

    # 🔚 最后一轮收尾
    while pending_calls:
        call_str = pending_calls.pop(0)
        current_round.append({"from": "function_call", "value": call_str})
        if pending_resps:
            resp_str = pending_resps.pop(0)
            obs_val = build_observation_from_tool_response(resp_str)
        else:
            tool_name = parse_tool_name_from_call(call_str)
            obs_val = json.dumps([{
                "type": "tool_result",
                "name": tool_name,
                "content": "[Stub] Placeholder response"
            }], ensure_ascii=False)
        current_round.append({"from": "observation", "value": obs_val})

    if current_round:
        current_round = append_placeholder_gpt(current_round)
        samples.append({"conversations": current_round})

    return samples



def transform_jsonl_file(in_path: Path, overwrite_in_place: bool = True, backup: bool = True) -> Path:
    """流式读取巨大 JSONL 文件，逐行规范化 conversations，并写回。
    - 写入临时文件 .tmp
    - 成功后可选重命名为原文件，原文件重命名为 .bak
    返回输出文件路径。
    """
    out_dir = Path("data/2_converted")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / in_path.with_suffix(".jsonl").name
    print(f"[INFO] 写入路径: {tmp_path}")
    total = 0
    fixed = 0
    with in_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 非法行，原样写回
                fout.write(line + "\n")
                continue

            if isinstance(obj, dict) and obj.get("conversations"):
                new_obj = parse_conversations(obj["conversations"])
                fixed += 1
            else:
                new_obj = obj

            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    print(f"[INFO] 已处理 {total} 行，其中规范化 {fixed} 行")
import re

def append_placeholder_gpt(conversations: list) -> list:
    """
    在 conversations 尾部检查是否已有有效 gpt 回复：
    - 若无 → 添加占位 gpt；
    """
    conversations.append({
            "from": "gpt",
            "value": "Placeholder GPT"
        })
    return conversations

def main():
    parser = argparse.ArgumentParser(description="Normalize Tool-Calling conversations JSONL to eval format (streaming)")
    parser.add_argument("--input", default="data/1_raw/Tool-Calling-Dataset-UIGEN-X.jsonl")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backup when overwriting")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite input; keep .tmp output")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"[FATAL] 输入文件不存在：{in_path}")

    out_path = transform_jsonl_file(
        in_path,
        overwrite_in_place=not args.no_overwrite,
        backup=not args.no_backup,
    )

if __name__ == "__main__":
    main()

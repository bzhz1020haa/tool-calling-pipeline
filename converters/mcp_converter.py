# mcp_convert_pipeline.py
# pip install jsonschema tqdm orjson google-generativeai (可选) pyarrow (可选)
# 用法示例：
#   python mcp_convert_pipeline.py \
#       --input ./raw_dataset \
#       --output ./converted_dataset \
#       --refine --score --model gemini-1.5-flash

import os
import re
import json
import orjson
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from jsonschema import validate, Draft7Validator
from tqdm import tqdm

# =============== 1) 目标 MCP Schema（精简版） ===============
MCP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "enum": ["system", "human", "assistant", "tool", "function_call", "observation"]},
                    "value": {"type": "string"}
                },
                "required": ["from", "value"]
            }
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "inputSchema": {"type": "object"}
                },
                "required": ["name"]
            }
        }
    },
    "required": ["conversations"]
}

# =============== 2) 工具：健壮加载 ===============
def json_loads_relaxed(s: Union[str, bytes]) -> Any:
    """尽量解析字符串化 JSON；自动修补常见引号/逗号问题。"""
    if not isinstance(s, (str, bytes)):
        return s
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="ignore")
    s_strip = s.strip()
    # 先尝试直接 orjson
    try:
        return orjson.loads(s_strip)
    except Exception:
        pass
    # 尝试把单引号替换为双引号、去掉尾逗号
    s_fix = re.sub(r"(['\"])\s*,\s*}", r"\1}", s_strip)
    s_fix = re.sub(r"(['\"])\s*,\s*]", r"\1]", s_fix)
    s_fix = s_fix.replace("'", '"')
    try:
        return orjson.loads(s_fix)
    except Exception:
        # 实在不行就返回原串
        return s

def load_any(path: Path) -> List[Any]:
    """支持 .json / .jsonl；可扩到 Parquet。"""
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json_loads_relaxed(line))
        return items
    elif path.suffix.lower() == ".json":
        with path.open("rb") as f:
            data = json_loads_relaxed(f.read())
        # 统一为列表
        if isinstance(data, list):
            return data
        return [data]
    elif path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as e:
            raise RuntimeError("需要安装 pyarrow 才能读取 Parquet: pip install pyarrow") from e
        table = pq.read_table(path)
        return table.to_pylist()  # list[dict]
    else:
        return []

# =============== 3) 本地规则归一化（核心） ===============
COMMON_ARG_KEYS = ["arguments", "args", "params", "parameters"]
TOOL_CALL_KEYS = ["tool_call", "tool_calls", "function_call", "function_calls"]

def normalize_role(x: str) -> str:
    r = x.lower()
    mapping = {
        "user": "human",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
        "function_call": "function_call",
        "observation": "observation"
    }
    return mapping.get(r, r)

def ensure_string(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def coerce_tool_catalog(tools_field: Any) -> List[Dict[str, Any]]:
    """把各种 tools 字段（字符串化/简写/错键）统一成 MCP tools[]。"""
    if tools_field is None:
        return []
    t = json_loads_relaxed(tools_field)
    if isinstance(t, dict) and "tools" in t and isinstance(t["tools"], list):
        t = t["tools"]
    if isinstance(t, list):
        # 规范化键名
        fixed = []
        for tool in t:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name") or tool.get("tool_name") or tool.get("id")
            desc = tool.get("description", "")
            schema = tool.get("inputSchema") or tool.get("parameters") or tool.get("input_schema") or {}
            fixed.append({
                "name": name,
                "description": desc,
                "inputSchema": schema if isinstance(schema, dict) else {}
            })
        return fixed
    return []

def extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """
    从 assistant 的自然语言里“捞”出可能的函数调用（简易版）。
    支持模式：get_weather(city="Paris")
    """
    calls = []
    pattern = r'([A-Za-z_][A-Za-z0-9_:.-]*)\s*\((.*?)\)'
    for m in re.finditer(pattern, text):
        name = m.group(1)
        args_src = m.group(2).strip()
        args_obj: Dict[str, Any] = {}
        if args_src:
            # 粗糙把 k=v, k="v" 解析为 dict
            parts = re.split(r'\s*,\s*', args_src)
            for p in parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    k = k.strip().strip('"\'')
                    v = v.strip()
                    v = v.strip().strip('"\'')
                    # 尝试转成数字/布尔
                    if v.lower() in ["true", "false"]:
                        v = v.lower() == "true"
                    else:
                        try:
                            if '.' in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except Exception:
                            pass
                    args_obj[k] = v
        calls.append({"name": name, "arguments": args_obj})
    return calls

def normalize_record_to_mcp(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    将“各种样式”的单条样本转为 MCP 目标结构：
      { "conversations": [ {from, value}, ... ], "tools": [ {name, description, inputSchema}, ... ] }
    """
    mcp: Dict[str, Any] = {"conversations": [], "tools": []}

    # 1) 工具目录
    tools_field = rec.get("tools") or rec.get("tool_catalog") or rec.get("available_tools")
    mcp["tools"] = coerce_tool_catalog(tools_field)

    # 2) conversations 归一：支持多来源字段名
    convs = rec.get("conversations") or rec.get("messages") or rec.get("dialogue") or []
    if not convs and "query" in rec:
        # 单轮问答样式：用 query 作为 human
        convs = [{"from": "human", "value": ensure_string(rec["query"])}]
        # answers 若是函数调用数组，也塞进 assistant
        if "answers" in rec:
            tool_calls = json_loads_relaxed(rec["answers"])
            convs.append({
                "from": "assistant",
                "value": f"<tool_call>\n{ensure_string(tool_calls)}\n</tool_call>"
            })

    # 3) 遍历每个对话 turn，规范化 role/内容 & 提取 tool_call
    for msg in convs:
        role = normalize_role(msg.get("from") or msg.get("role", ""))
        value = msg.get("value") or msg.get("content") or ""

        # 捕捉结构化 tool_calls/object
        tool_calls_struct = None
        for k in TOOL_CALL_KEYS:
            if k in msg:
                tool_calls_struct = msg[k]
                break

        if tool_calls_struct:
            tc_list = json_loads_relaxed(tool_calls_struct)
            # 统一成 <tool_call> 包起来（MCP 常见包裹风格）
            value_str = ""
            if isinstance(tc_list, list):
                for tc in tc_list:
                    # 兼容 args/params → arguments
                    args_obj = {}
                    for ak in COMMON_ARG_KEYS:
                        if ak in tc:
                            args_obj = tc[ak]
                            break
                    if not args_obj and isinstance(tc.get("arguments"), str):
                        args_obj = json_loads_relaxed(tc["arguments"])
                    name = tc.get("name")
                    value_str += "<tool_call>\n" + ensure_string({"name": name, "arguments": args_obj}) + "\n</tool_call>\n"
            else:
                value_str = "<tool_call>\n" + ensure_string(tc_list) + "\n</tool_call>\n"

            mcp["conversations"].append({"from": role or "assistant", "value": value_str.strip()})
        else:
            # 如果 assistant 文本里像 foo(a=1) 这种，尽量抽取
            if role == "assistant" and isinstance(value, str) and "(" in value and ")" in value:
                extracted = extract_tool_calls_from_text(value)
                if extracted:
                    wrapped = "\n".join(f"<tool_call>\n{ensure_string(tc)}\n</tool_call>" for tc in extracted)
                    mcp["conversations"].append({"from": "assistant", "value": wrapped})
                else:
                    mcp["conversations"].append({"from": role, "value": ensure_string(value)})
            else:
                mcp["conversations"].append({"from": role or "human", "value": ensure_string(value)})

    return mcp

# =============== 4) 本地校验 & 质量评分（无模型版） ===============
def validate_mcp(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs = []
    v = Draft7Validator(MCP_SCHEMA)
    for e in sorted(v.iter_errors(obj), key=lambda e: e.path):
        errs.append(f"{list(e.path)}: {e.message}")
    return (len(errs) == 0, errs)

def local_quality_score(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    简易质量指标（0~1）：
      - schema_valid: 是否通过 schema
      - has_multi_turn: 是否多轮
      - has_tool_catalog: 是否包含 tools
      - has_tool_calls: assistant 是否出现 <tool_call>
      - role_sequence_ok: system→human→assistant 基本模式是否出现
    """
    ok, errs = validate_mcp(obj)
    conv = obj.get("conversations", [])
    text = "\n".join(m.get("value", "") for m in conv if isinstance(m, dict))
    has_tool_calls = "<tool_call>" in text
    has_tool_catalog = bool(obj.get("tools"))
    has_multi_turn = len(conv) >= 3
    roles = [m.get("from") for m in conv]
    role_seq_ok = any(r == "system" for r in roles) and any(r == "human" for r in roles) and any(r == "assistant" for r in roles)

    score = 0.0
    score += 0.4 if ok else 0.0
    score += 0.2 if has_tool_calls else 0.0
    score += 0.15 if has_tool_catalog else 0.0
    score += 0.15 if has_multi_turn else 0.0
    score += 0.10 if role_seq_ok else 0.0

    return {
        "schema_valid": ok,
        "errors": errs,
        "has_tool_calls": has_tool_calls,
        "has_tool_catalog": has_tool_catalog,
        "has_multi_turn": has_multi_turn,
        "role_sequence_ok": role_seq_ok,
        "local_score_0_1": round(score, 3)
    }

# =============== 5) 可选：Gemini 精修 & 评分 ===============
def refine_with_gemini(record: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    """
    需要：pip install google-generativeai 且配置环境变量 GOOGLE_API_KEY
    """
    try:
        import google.generativeai as genai
    except Exception:
        print("[REFINE] google-generativeai 未安装，跳过。")
        return None

    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[REFINE] 未检测到 GOOGLE_API_KEY，跳过。")
        return None

    genai.configure(api_key=api_key)
    prompt = f"""
You are a data normalizer. Convert the following record into **valid MCP JSON** with fields:
- "conversations": list of objects {{ "from": "system|human|assistant|tool|function_call|observation", "value": string }}
- "tools": optional list of tool specs {{ "name", "description", "inputSchema" }}

Rules:
- If tool calls exist (structured or in text), rewrite assistant messages to include <tool_call>{{"name":..., "arguments":{{...}}}}</tool_call>.
- Fix common key drifts: role/from, answers/tools as stringified JSON, args/params→arguments, function_call/tool_calls→assistant value with <tool_call>.
- Do NOT invent content. If information is missing, leave it out.
- Return only JSON. No markdown.

Record to normalize:
{json.dumps(record, ensure_ascii=False)}
"""
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    text = resp.text.strip()
    try:
        out = json.loads(text)
        return out
    except Exception:
        # 有些模型会加 ```json 包裹
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def score_with_gemini(mcp_obj: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    try:
        import google.generativeai as genai
    except Exception:
        print("[SCORE] google-generativeai 未安装，跳过。")
        return None

    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[SCORE] 未检测到 GOOGLE_API_KEY，跳过。")
        return None

    genai.configure(api_key=api_key)
    prompt = f"""
You are a data quality rater for MCP tool-calling datasets.
Score the JSON on these axes (0-1): schema_validity, tool_call_correctness, argument_schema_alignment, dialogue_coherence, faithfulness(no hallucination), overall.
Return a JSON with numeric fields and a short 'notes'.

JSON to rate:
{json.dumps(mcp_obj, ensure_ascii=False)}
"""
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    text = resp.text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

# =============== 6) 主处理流程 ===============
@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    refine: bool = False
    score: bool = False
    model_name: str = "gemini-1.5-flash"
    max_workers: int = 4

def process_file(path: Path, cfg: PipelineConfig) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    raw_items = load_any(path)
    results = []
    for rec in raw_items:
        # 本地归一
        mcp = normalize_record_to_mcp(rec)

        # 可选：Gemini 精修
        if cfg.refine:
            refined = refine_with_gemini(mcp, cfg.model_name)
            if refined and isinstance(refined, dict):
                mcp = refined

        # 本地评分
        local_score = local_quality_score(mcp)
        # 可选：Gemini 评分
        g_score = score_with_gemini(mcp, cfg.model_name) if cfg.score else None

        results.append((mcp, {"local": local_score, "gemini": g_score}))
    return results

def write_outputs(base_out: Path, src_file: Path, records: List[Tuple[Dict[str, Any], Dict[str, Any]]]):
    rel = src_file.with_suffix(".json").name
    out_data = [r[0] for r in records]
    out_meta = [r[1] for r in records]
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / rel).write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (base_out / (src_file.stem + ".scores.json")).write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="文件或目录：支持 .json / .jsonl / .parquet")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--refine", action="store_true", help="启用 Gemini 精修")
    p.add_argument("--score", action="store_true", help="启用 Gemini 评分")
    p.add_argument("--model", default="gemini-1.5-flash", help="Gemini 模型名")
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    cfg = PipelineConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        refine=args.refine,
        score=args.score,
        model_name=args.model,
        max_workers=args.max_workers,
    )

    files: List[Path] = []
    if cfg.input_path.is_dir():
        for ext in (".json", ".jsonl", ".parquet"):
            files.extend(cfg.input_path.rglob(f"*{ext}"))
    else:
        files = [cfg.input_path]

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {ex.submit(process_file, f, cfg): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            f = futures[fut]
            try:
                recs = fut.result()
                write_outputs(cfg.output_path, f, recs)
            except Exception as e:
                print(f"[ERROR] {f}: {e}")

if __name__ == "__main__":
    main()

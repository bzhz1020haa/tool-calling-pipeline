#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional, List, Any, Tuple, Union
from fsspec import config
import requests
from datasets import load_dataset

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# ---------- IO helpers ----------

def load_hf_dataset(name: str, config_name: str= None, split: str = "train", sample_items: int = 200, streaming: bool = False) -> str:
    """
    从 Hugging Face 加载数据集并转为 digest 字符串
    支持 streaming 模式
    """
    print(f"[INFO] 正在从 Hugging Face 加载数据集：{name} (config={config_name}, split={split}, streaming={streaming})")
    if config_name:
        dataset = load_dataset(name, config_name, split=split, streaming=streaming)
    else:
        dataset = load_dataset(name, split=split, streaming=streaming)
    if streaming:
        # IterableDataset → 逐条遍历，只取前 sample_items 条
        from itertools import islice
        sampled = list(islice(dataset, sample_items))
    else:
        sampled=dataset.select(range(min(len(dataset), sample_items)))
    return json.dumps(list(sampled), ensure_ascii=False, separators=(",", ":"))

def read_key_hardcoded() -> str:
    return "AIzaSyDikJjktaSUq3sJCAHUIu7JmMEgP1DeHSI"  # key

def load_text(path: str) -> Optional[str]:
    try:
        txt = Path(path).read_text(encoding="utf-8")
        print(f"[INFO] 读取 {path} 成功（{len(txt)} 字符）")
        return txt
    except Exception as e:
        print(f"[FATAL] 读取 {path} 失败：{e}")
        return None

# ---------- sampling / digest ----------

def try_parse_json(s: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(s)
    except Exception:
        return False, None

def sample_json_content(obj: Any, max_items: int = 200) -> Any:
    if isinstance(obj, list):
        if len(obj) <= max_items:
            return obj
        idx = sorted(random.sample(range(len(obj)), max_items))
        return [obj[i] for i in idx]
    return obj

def parse_maybe_jsonl(raw: str) -> Union[List[Any], None]:
    lines = raw.splitlines()
    if len(lines) > 3 and sum(1 for ln in lines[:1000] if ln.strip().startswith(("{","["))) > 3:
        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            ok, item = try_parse_json(ln)
            if ok:
                out.append(item)
            else:
                return None
        return out
    return None

def build_digest(raw: str, max_bytes: int = 400_000, sample_items: int = 200) -> str:
    ok, obj = try_parse_json(raw)
    if ok:
        sampled = sample_json_content(obj, max_items=sample_items)
        out = json.dumps(sampled, ensure_ascii=False, separators=(",", ":"))
        return out[:max_bytes]

    arr = parse_maybe_jsonl(raw)
    if arr is not None:
        sampled = sample_json_content(arr, max_items=sample_items)
        out = json.dumps(sampled, ensure_ascii=False, separators=(",", ":"))
        return out[:max_bytes]

    length = len(raw)
    cut = int(length * 0.3)
    head = raw[:cut]
    tail = raw[-cut:]
    out = head + "\n...\n[TRUNCATED: middle 40% removed]\n...\n" + tail

    if len(out) > max_bytes:
        half = max_bytes // 2
        out = raw[:half] + "\n...\n[TRUNCATED]\n...\n" + raw[-half:]

    return out

# ---------- model call ----------
def call_model_api(
    prompt: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash-exp",
    temperature: float = 0.3,
    top_p: float = 0.95,
    top_k: int = 40,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> str:
    url = f"{API_BASE}/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "topP": float(top_p),
            "topK": int(top_k),
            "maxOutputTokens": int(max_tokens),
        },
    }
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    dt = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    if not text:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    print(f"[INFO] 模型返回成功（{dt:.2f}s，{len(text)} chars）")
    return text

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", help="本地输入文件 (.json/.jsonl)，和 --hf-dataset 二选一")
    ap.add_argument("--hf-config", default=None, help="Hugging Face 数据集的 config 名称（有多个子配置时必须指定）")
    ap.add_argument("--hf-dataset", help="Hugging Face 数据集名称 (如 username/my_dataset)")
    ap.add_argument("--hf-split", default="train", help="Hugging Face 数据集 split (默认 train)")
    ap.add_argument("--streaming", action="store_true", help="是否启用 Hugging Face streaming 模式（避免全量下载大数据集）")
    ap.add_argument("--guideline", default=None)
    ap.add_argument("--outdir", default="data/3_evaluated")
    ap.add_argument("--model", default="gemini-2.0-flash-exp")
    ap.add_argument("--max-bytes", type=int, default=400_000)
    ap.add_argument("--sample-items", type=int, default=200)
    args = ap.parse_args()

    api_key = read_key_hardcoded().strip()
    if not api_key:
        print("[FATAL] API Key 未配置")
        return

    # 根据输入来源选择数据
    raw_sample = None
    if args.hf_dataset:
        digest = load_hf_dataset(
            args.hf_dataset, 
            config_name=args.hf_config,
            split=args.hf_split, 
            sample_items=args.sample_items,
            streaming=args.streaming)
        sample_name = args.hf_dataset.replace("/", "_") + "_" + args.hf_split
    elif args.sample:
        raw_sample = load_text(args.sample)
        if raw_sample is None:
            return
        digest = build_digest(raw_sample, max_bytes=args.max_bytes, sample_items=args.sample_items)
        sample_name = Path(args.sample).stem
    else:
        print("[FATAL] 必须提供 --sample 或 --hf-dataset 参数之一")
        return

    guidelines_text = load_text(args.guideline) if args.guideline else ""

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 仅在本地文件模式下保存 preview_raw
    if raw_sample is not None:
        raw_preview_path = outdir / f"{sample_name}_preview_raw.txt"
        raw_preview_path.write_text(raw_sample[:200000], encoding="utf-8")

    # 保存 digest
    digest_path = outdir / f"{sample_name}_digest.txt"
    digest_path.write_text(digest, encoding="utf-8")

    # 构造提示
    print("[INFO] 构造提示")
    prompt = (
        "你是一个大模型评估器。任务是评估工具调用数据集样本的质量。\n"
        f"{guidelines_text}\n\n"
        "输出要求：\n"
        "- 所有评分字段必须是 0–100 的整数\n"
        "- 必须包含 ChainTypeRatio.parallel / ChainTypeRatio.sequential / ChainTypeRatio.hybrid 三个占比（整数，0–100）\n"
        "- Schema 需包含 InputOutputSchemaDesc 自然语言说明\n"
        "- 只输出 JSON 对象，不能有额外文字\n\n"
        f"【以下为经过抽样与裁剪的样本摘要，不包含全部内容】\n{digest}"
    )

    print(f"[INFO] 调用模型：{args.model}")
    try:
        result_text = call_model_api(prompt=prompt, api_key=api_key, model_name=args.model)
    except Exception as e:
        print(f"[FATAL] 调用失败：{e}")
        return

    # 保存结果
    out_path = outdir / f"{sample_name}_eval_result.json"
    txt = result_text.strip()
    if txt.startswith("```"):
        stripped = txt.strip("`")
        fb, lb = stripped.find("{"), stripped.rfind("}")
        if fb != -1 and lb != -1 and lb > fb:
            txt = stripped[fb:lb+1]
    out_path.write_text(txt, encoding="utf-8")
    print(f"[INFO] 评估结果已写入 {out_path}")

if __name__ == "__main__":
    main()

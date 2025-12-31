#!/usr/bin/env python3
"""
收集专家分布数据

流程：
1. start_expert_distribution_record
2. 发送 generate 请求
3. stop_expert_distribution_record
4. dump_expert_distribution_record
5. 移动 pt 文件到 recorder 目录

用法:
    python collect_expert_distribution.py --data /path/to/ShareGPT.json --num 1000 --output ./recorder
"""

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import requests

SERVER_URL = "http://localhost:30000"


def start_record():
    """开始记录"""
    resp = requests.post(f"{SERVER_URL}/start_expert_distribution_record")
    print(f"start_record: {resp.status_code}")


def stop_record():
    """停止记录"""
    resp = requests.post(f"{SERVER_URL}/stop_expert_distribution_record")
    print(f"stop_record: {resp.status_code}")


def dump_record():
    """导出记录"""
    resp = requests.post(f"{SERVER_URL}/dump_expert_distribution_record")
    print(f"dump_record: {resp.status_code}")


def generate(text: str, max_new_tokens: int = 256):
    """发送生成请求"""
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "top_p": 1
        },
        "stream": False
    }
    resp = requests.post(
        f"{SERVER_URL}/generate",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    return resp


def move_pt_file(output_dir: Path, index: int):
    """移动 /tmp 下的 pt 文件到输出目录"""
    pt_files = glob.glob("/tmp/expert_distribution_recorder_*.pt")
    if pt_files:
        # 按修改时间排序，取最新的
        pt_files.sort(key=os.path.getmtime, reverse=True)
        src = pt_files[0]
        # 重命名为带序号的文件名
        dst = output_dir / f"expert_distribution_{index:04d}.pt"
        shutil.move(src, dst)
        print(f"移动: {src} -> {dst}")
        return True
    else:
        print("警告: 未找到 pt 文件")
        return False


def load_sharegpt_data(filepath: str, num: int):
    """加载 ShareGPT 数据，提取用户问题"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = []
    for item in data:
        conversations = item.get("conversations", [])
        # 提取第一个 human 消息作为 prompt
        for conv in conversations:
            if conv.get("from") == "human":
                text = conv.get("value", "").strip()
                if text:
                    prompts.append(text)
                    break
        if len(prompts) >= num:
            break

    return prompts[:num]


def main():
    parser = argparse.ArgumentParser(description="收集专家分布数据")
    parser.add_argument("--data", type=str, required=True, help="ShareGPT JSON 文件路径")
    parser.add_argument("--num", type=int, default=1000, help="处理的数据条数 (默认: 1000)")
    parser.add_argument("--output", type=str, default="./recorder", help="输出目录 (默认: ./recorder)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="最大生成 token 数 (默认: 256)")
    parser.add_argument("--start-idx", type=int, default=0, help="起始索引 (默认: 0)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载数据: {args.data}")
    prompts = load_sharegpt_data(args.data, args.num + args.start_idx)
    prompts = prompts[args.start_idx:]
    print(f"共 {len(prompts)} 条数据待处理")

    for i, prompt in enumerate(prompts):
        idx = i + args.start_idx
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{args.num}] 处理中...")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")

        try:
            # 1. 开始记录
            start_record()

            # 2. 发送请求
            t0 = time.time()
            resp = generate(prompt, max_new_tokens=args.max_new_tokens)
            elapsed = time.time() - t0
            print(f"generate: {resp.status_code}, 耗时: {elapsed:.2f}s")

            # 3. 停止记录
            stop_record()

            # 4. 导出记录
            dump_record()

            # 5. 移动文件
            time.sleep(0.1)  # 等待文件写入完成
            move_pt_file(output_dir, idx)

        except Exception as e:
            print(f"错误: {e}")
            continue

    print(f"\n完成! 输出目录: {output_dir}")


if __name__ == "__main__":
    main()

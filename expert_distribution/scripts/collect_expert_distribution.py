#!/usr/bin/env python3
"""
收集专家分布数据

流程：
1. start_expert_distribution_record
2. 发送 generate 请求
3. stop_expert_distribution_record
4. dump_expert_distribution_record
5. 移动 pt 文件到 recorder 目录
6. 聚合所有 pt 文件生成 activation_stats.pt

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
import torch

SERVER_URL = "http://localhost:30005"


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


def aggregate_expert_distributions(
    pt_files: list,
    output_path: Path,
    num_layers: int,
    num_experts: int,
) -> None:
    """聚合多个专家分布文件为单个统计文件

    Args:
        pt_files: pt 文件路径列表
        output_path: 输出文件路径
        num_layers: MoE 层数
        num_experts: 每层专家数
    """
    print(f"\n聚合 {len(pt_files)} 个专家分布文件...")

    # 初始化累加器
    total_activation_counts = torch.zeros(num_layers, num_experts, dtype=torch.int64)

    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=True)

            if "logical_count" in data:
                logical_count = data["logical_count"]
                # logical_count shape: [forward_pass, num_layers, num_experts]

                # 跨 forward pass 求和得到每个专家的激活次数
                # 结果: [num_layers, num_experts]
                activation_counts = (logical_count > 0).sum(dim=0).long()
                total_activation_counts += activation_counts
            else:
                print(f"警告: {pt_file} 不包含 'logical_count'")

        except Exception as e:
            print(f"加载 {pt_file} 出错: {e}")

    # 保存聚合统计
    # 包含两种格式:
    # 1. activation_counts: 用于分析 (2D)
    # 2. logical_count: 用于 --init-expert-location 兼容性 (3D)
    logical_count = total_activation_counts.unsqueeze(0)  # 添加 batch 维度

    torch.save(
        {
            "activation_counts": total_activation_counts,  # 2D 用于分析
            "logical_count": logical_count,  # 3D 用于 --init-expert-location
            "num_layers": num_layers,
            "num_experts": num_experts,
            "num_samples": len(pt_files),
        },
        output_path,
    )

    print(f"已保存聚合统计到 {output_path}")
    print(f"  activation_counts shape: {total_activation_counts.shape} (2D)")
    print(f"  logical_count shape: {logical_count.shape} (3D)")
    print(f"  总激活次数: {total_activation_counts.sum().item()}")
    print(f"  每专家平均激活: {total_activation_counts.float().mean().item():.2f}")
    print(f"\n可使用 --init-expert-location {output_path} 加载此文件")


def main():
    parser = argparse.ArgumentParser(description="收集专家分布数据")
    parser.add_argument("--data", type=str, required=True, help="ShareGPT JSON 文件路径")
    parser.add_argument("--num", type=int, default=100, help="处理的数据条数 (默认: 1000)")
    parser.add_argument("--output", type=str, default="./recorder", help="输出目录 (默认: ./recorder)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="最大生成 token 数 (默认: 256)")
    parser.add_argument("--start-idx", type=int, default=0, help="起始索引 (默认: 0)")
    parser.add_argument("--num-layers", type=int, default=92, help="MoE 层数 (默认: 48)")
    parser.add_argument("--num-experts", type=int, default=160, help="每层专家数 (默认: 512)")
    parser.add_argument("--skip-aggregate", action="store_true", help="跳过聚合步骤")

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
            # start_record()

            # 2. 发送请求
            t0 = time.time()
            resp = generate(prompt, max_new_tokens=args.max_new_tokens)
            elapsed = time.time() - t0
            print(f"generate: {resp.status_code}, 耗时: {elapsed:.2f}s")

            # 3. 停止记录
            # stop_record()

            # 4. 导出记录
            # dump_record()

            # 5. 移动文件
            time.sleep(0.1)  # 等待文件写入完成
            # move_pt_file(output_dir, idx)

        except Exception as e:
            print(f"错误: {e}")
            continue

    print(f"\n收集完成! 输出目录: {output_dir}")

    # 聚合所有 pt 文件
    if not args.skip_aggregate:
        pt_files = sorted(output_dir.glob("expert_distribution_*.pt"))
        if pt_files:
            aggregate_path = output_dir / "activation_stats.pt"
            aggregate_expert_distributions(
                pt_files,
                aggregate_path,
                args.num_layers,
                args.num_experts,
            )
        else:
            print("\n警告: 未找到专家分布文件，跳过聚合")
    else:
        print("\n跳过聚合步骤 (--skip-aggregate)")

    print(f"\n全部完成!")


if __name__ == "__main__":
    main()

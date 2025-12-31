#!/usr/bin/env python3
"""
查看 ExpertDistributionRecorder 生成的 .pt 文件内容

支持三种模式生成的文件：
- stat / stat_approx: 聚合统计模式
- per_pass: 每次 forward pass 单独记录
- per_token: 每个 token 级别的详细记录

用法:
    python view_expert_distribution.py <pt_file_path> [--top-k N] [--layer L]
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def detect_mode(data: Dict[str, Any]) -> str:
    """根据文件内容检测记录模式"""
    if "logical_count" in data:
        return "stat"
    elif "records" in data:
        # per_pass 和 per_token 都有 records，需要进一步区分
        if len(data["records"]) > 0:
            record = data["records"][0]
            if "topk_ids_of_layer" in record:
                return "per_token"
        return "per_pass"
    else:
        return "unknown"


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """打印分隔线"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def print_tensor_info(name: str, tensor: torch.Tensor, indent: int = 0):
    """打印张量基本信息"""
    prefix = "  " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  shape: {list(tensor.shape)}")
    print(f"{prefix}  dtype: {tensor.dtype}")
    print(f"{prefix}  device: {tensor.device}")
    if tensor.numel() > 0:
        print(f"{prefix}  min: {tensor.min().item()}, max: {tensor.max().item()}, sum: {tensor.sum().item()}")


def view_stat_mode(data: Dict[str, Any], args):
    """
    查看 stat/stat_approx 模式的文件

    数据结构:
    - rank: int, 生成此文件的 GPU rank
    - logical_count: Tensor, shape=(buffer_size, num_layers, num_logical_experts)
        - buffer_size: 缓冲区大小（记录的 forward pass 数量）
        - num_layers: MoE 层数
        - num_logical_experts: 逻辑专家数量
        - 值: 每个位置表示该专家在对应 forward pass 中被选中的次数
    - average_utilization_rate_over_window: float 或 None
        - 滑动窗口内的平均利用率（仅当 enable_expert_distribution_metrics 开启时有值）
    """
    print_separator("STAT 模式文件内容")

    # 基本信息
    print(f"\n[基本信息]")
    print(f"  rank: {data['rank']} (生成此文件的 GPU rank)")
    print(f"  average_utilization_rate_over_window: {data.get('average_utilization_rate_over_window')}")
    print(f"    (滑动窗口内平均利用率，None 表示未启用 metrics)")

    # logical_count 详细信息
    logical_count = data["logical_count"]
    print(f"\n[logical_count 张量]")
    print(f"  shape: {list(logical_count.shape)}")
    print(f"    - dim0 ({logical_count.shape[0]}): buffer_size，记录的 forward pass 数量")
    print(f"    - dim1 ({logical_count.shape[1]}): num_layers，MoE 层数")
    print(f"    - dim2 ({logical_count.shape[2]}): num_logical_experts，逻辑专家数")
    print(f"  dtype: {logical_count.dtype}")
    print(f"  总激活次数: {logical_count.sum().item()}")

    # 找出有专家激活的 forward pass 范围
    pass_activation = logical_count.sum(dim=(1, 2))  # (buffer_size,)
    active_passes = torch.where(pass_activation > 0)[0]
    if len(active_passes) > 0:
        first_active_pass = active_passes[0].item()
        last_active_pass = active_passes[-1].item()
        print(f"  有专家激活的 forward pass 范围: [{first_active_pass}, {last_active_pass}]")
        print(f"  有效 forward pass 数量: {len(active_passes)}")
    else:
        print(f"  有专家激活的 forward pass 范围: (无)")

    # 聚合所有 forward pass 的统计
    total_count = logical_count.sum(dim=0)  # (num_layers, num_experts)
    num_layers = total_count.shape[0]
    num_experts = total_count.shape[1]
    top_k = args.top_k

    # 确定要显示的层
    if args.layer is not None:
        if 0 <= args.layer < num_layers:
            layers_to_show = [args.layer]
        else:
            print(f"\n错误: layer {args.layer} 超出范围 [0, {num_layers - 1}]")
            return
    else:
        layers_to_show = list(range(num_layers))

    # 按层统计
    print(f"\n[各层专家激活统计]")
    for layer_idx in layers_to_show:
        layer_count = total_count[layer_idx].float()  # (num_experts,)
        layer_mean = layer_count.mean().item()
        layer_std = layer_count.std().item()
        zero_count = (layer_count == 0).sum().item()

        print(f"\n{'=' * 70}")
        print(f"Layer {layer_idx}  |  平均激活: {layer_mean:.2f}  |  标准差: {layer_std:.2f}  |  未激活专家数: {int(zero_count)}")
        print(f"{'=' * 70}")

        # Top-K 热门专家
        top_values, top_indices = torch.topk(layer_count, min(top_k, num_experts))
        print(f"  Top-{top_k} 热门专家:")
        for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
            print(f"    {i + 1:2d}. Expert {idx:3d}: {int(val):6d} 次")

        # Bottom-K 冷门专家（排除激活0次的）
        nonzero_mask = layer_count > 0
        nonzero_count = layer_count[nonzero_mask]
        nonzero_indices = torch.arange(num_experts)[nonzero_mask]
        if len(nonzero_count) > 0:
            k = min(top_k, len(nonzero_count))
            bottom_values, bottom_pos = torch.topk(nonzero_count, k, largest=False)
            bottom_indices = nonzero_indices[bottom_pos]
            print(f"  Bottom-{k} 冷门专家 (不含未激活):")
            for i, (idx, val) in enumerate(zip(bottom_indices.tolist(), bottom_values.tolist())):
                print(f"    {i + 1:2d}. Expert {idx:3d}: {int(val):6d} 次")
        else:
            print(f"  Bottom-{top_k} 冷门专家: (所有专家激活次数均为0)")


def view_per_pass_mode(data: Dict[str, Any], args):
    """
    查看 per_pass 模式的文件

    数据结构:
    - records: List[Dict], 每个 forward pass 的记录
        - forward_pass_id: int, forward pass 的唯一标识
        - rank: int, GPU rank
        - gatherer_key: str, 收集器类型（通常是 "primary"）
        - global_physical_count: Tensor, shape=(num_layers, num_physical_experts)
            - 该 forward pass 中每个物理专家被选中的次数
    - last_physical_to_logical_map: Tensor, shape=(num_layers, num_physical_experts)
        - 物理专家到逻辑专家的映射（用于 EPLB 场景）
    """
    print_separator("PER_PASS 模式文件内容")

    records = data["records"]
    physical_to_logical_map = data.get("last_physical_to_logical_map")

    print(f"\n[基本信息]")
    print(f"  记录数量: {len(records)} 个 forward pass")

    if physical_to_logical_map is not None:
        print(f"\n[physical_to_logical_map]")
        print(f"  shape: {list(physical_to_logical_map.shape)}")
        print(f"    - 用于将物理专家 ID 映射到逻辑专家 ID")
        print(f"    - EPLB 场景下，一个逻辑专家可能有多个物理副本")

    if len(records) == 0:
        print("\n  (无记录)")
        return

    # 第一条记录的结构
    first_record = records[0]
    print(f"\n[单条记录结构] (以第一条为例)")
    print(f"  forward_pass_id: {first_record.get('forward_pass_id')} (forward pass 唯一标识)")
    print(f"  rank: {first_record.get('rank')} (GPU rank)")
    print(f"  gatherer_key: {first_record.get('gatherer_key')} (收集器类型)")

    if "global_physical_count" in first_record:
        gpc = first_record["global_physical_count"]
        print(f"  global_physical_count:")
        print(f"    shape: {list(gpc.shape)}")
        print(f"      - dim0 ({gpc.shape[0]}): num_layers")
        print(f"      - dim1 ({gpc.shape[1]}): num_physical_experts")
        print(f"    dtype: {gpc.dtype}")

    # 聚合统计
    print(f"\n[聚合统计]")
    all_counts = []
    for record in records:
        if "global_physical_count" in record:
            all_counts.append(record["global_physical_count"])

    if all_counts:
        # 尝试 stack，如果形状不一致则跳过
        try:
            stacked = torch.stack(all_counts)  # (num_passes, num_layers, num_experts)
            total_count = stacked.sum(dim=0)  # (num_layers, num_experts)
            total_count_all_layers = total_count.sum(dim=0)  # (num_experts,)

            print(f"  总 forward pass 数: {len(all_counts)}")
            print(f"  总激活次数: {total_count.sum().item()}")

            top_k = args.top_k
            top_values, top_indices = torch.topk(total_count_all_layers, min(top_k, len(total_count_all_layers)))
            print(f"\n  >> Top-{top_k} 热门物理专家:")
            for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
                print(f"     {i+1}. Expert {idx}: {val} 次")
        except Exception as e:
            print(f"  聚合失败: {e}")

    # Forward pass ID 范围
    pass_ids = [r.get("forward_pass_id") for r in records if r.get("forward_pass_id") is not None]
    if pass_ids:
        print(f"\n[Forward Pass ID 范围]")
        print(f"  最小: {min(pass_ids)}, 最大: {max(pass_ids)}")


def view_per_token_mode(data: Dict[str, Any], args):
    """
    查看 per_token 模式的文件

    数据结构:
    - records: List[Dict], 每个 forward pass 的详细记录
        - forward_pass_id: int
        - rank: int
        - gatherer_key: str
        - input_ids: List[int], 输入 token IDs
        - positions: List[int], token 位置
        - extend_seq_lens: List[int], 扩展序列长度
        - forward_mode: int, forward 模式（prefill/decode 等）
        - topk_ids_of_layer: Tensor, shape=(num_layers, num_tokens, top_k)
            - 每层每个 token 选择的 top-k 专家 ID
        - misc_objects: List[Dict], 其他杂项数据（如 DeepEP 的调度信息）
        - global_physical_count: Tensor, shape=(num_layers, num_physical_experts)
    - last_physical_to_logical_map: Tensor
    """
    print_separator("PER_TOKEN 模式文件内容")

    records = data["records"]
    physical_to_logical_map = data.get("last_physical_to_logical_map")

    print(f"\n[基本信息]")
    print(f"  记录数量: {len(records)} 个 forward pass")

    if physical_to_logical_map is not None:
        print(f"\n[physical_to_logical_map]")
        print(f"  shape: {list(physical_to_logical_map.shape)}")

    if len(records) == 0:
        print("\n  (无记录)")
        return

    # 第一条记录的结构
    first_record = records[0]
    print(f"\n[单条记录结构] (以第一条为例)")
    print(f"  forward_pass_id: {first_record.get('forward_pass_id')}")
    print(f"  rank: {first_record.get('rank')}")
    print(f"  gatherer_key: {first_record.get('gatherer_key')}")
    print(f"  forward_mode: {first_record.get('forward_mode')} (forward 模式)")

    if "input_ids" in first_record:
        input_ids = first_record["input_ids"]
        print(f"  input_ids: List[int], len={len(input_ids)} (输入 token IDs)")

    if "positions" in first_record:
        positions = first_record["positions"]
        print(f"  positions: List[int], len={len(positions)} (token 位置)")

    if "extend_seq_lens" in first_record:
        print(f"  extend_seq_lens: {first_record['extend_seq_lens']} (扩展序列长度)")

    if "topk_ids_of_layer" in first_record:
        topk = first_record["topk_ids_of_layer"]
        print(f"  topk_ids_of_layer:")
        print(f"    shape: {list(topk.shape)}")
        print(f"      - dim0 ({topk.shape[0]}): num_layers")
        print(f"      - dim1 ({topk.shape[1]}): num_tokens")
        print(f"      - dim2 ({topk.shape[2]}): top_k (每个 token 选择的专家数)")
        print(f"    dtype: {topk.dtype}")
        print(f"    说明: 值为 -1 表示无效/padding")

    if "global_physical_count" in first_record:
        gpc = first_record["global_physical_count"]
        print(f"  global_physical_count: shape={list(gpc.shape)}")

    if "misc_objects" in first_record and first_record["misc_objects"]:
        misc = first_record["misc_objects"]
        print(f"  misc_objects: List[Dict], len={len(misc)}")
        if len(misc) > 0:
            print(f"    第一项 keys: {list(misc[0].keys())}")

    # 聚合统计
    print(f"\n[聚合统计]")
    total_tokens = 0
    all_topk = []

    for record in records:
        if "input_ids" in record:
            total_tokens += len(record["input_ids"])
        if "topk_ids_of_layer" in record:
            all_topk.append(record["topk_ids_of_layer"])

    print(f"  总 token 数: {total_tokens}")

    if all_topk:
        # 统计专家选择频率
        first_topk = all_topk[0]
        num_layers = first_topk.shape[0]

        # 从 topk_ids 统计
        all_expert_ids = []
        for topk in all_topk:
            valid_mask = topk != -1
            all_expert_ids.extend(topk[valid_mask].flatten().tolist())

        if all_expert_ids:
            from collections import Counter
            counter = Counter(all_expert_ids)

            print(f"  有效专家选择总数: {len(all_expert_ids)}")
            print(f"  不同专家数: {len(counter)}")

            top_k = args.top_k
            print(f"\n  >> Top-{top_k} 热门专家:")
            for i, (expert_id, count) in enumerate(counter.most_common(top_k)):
                percentage = count / len(all_expert_ids) * 100
                print(f"     {i+1}. Expert {expert_id}: {count} 次 ({percentage:.2f}%)")


def view_unknown_mode(data: Dict[str, Any], args):
    """查看未知格式的文件"""
    print_separator("未知模式 - 原始内容")

    print(f"\n[所有 Keys]")
    for key in data.keys():
        print(f"  - {key}")

    print(f"\n[各字段详情]")
    for key, value in data.items():
        print(f"\n  {key}:")
        if isinstance(value, torch.Tensor):
            print(f"    类型: Tensor")
            print(f"    shape: {list(value.shape)}")
            print(f"    dtype: {value.dtype}")
        elif isinstance(value, list):
            print(f"    类型: List, len={len(value)}")
            if len(value) > 0:
                print(f"    第一项类型: {type(value[0]).__name__}")
        elif isinstance(value, dict):
            print(f"    类型: Dict, keys={list(value.keys())}")
        else:
            print(f"    类型: {type(value).__name__}")
            print(f"    值: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="查看 ExpertDistributionRecorder 生成的 .pt 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python view_expert_distribution.py record.pt
  python view_expert_distribution.py record.pt --top-k 20
  python view_expert_distribution.py record.pt --layer 5
        """
    )
    parser.add_argument("pt_file", type=str, help=".pt 文件路径")
    parser.add_argument("--top-k", type=int, default=10, help="显示 Top-K 热门/冷门专家 (默认: 10)")
    parser.add_argument("--layer", type=int, default=None, help="指定查看某一层的详细统计")
    parser.add_argument("--raw", action="store_true", help="显示原始数据结构")
    parser.add_argument("--output", type=str, default=None, help="输出目录，保存为 txt 文件")

    args = parser.parse_args()

    pt_path = Path(args.pt_file)
    if not pt_path.exists():
        print(f"错误: 文件不存在 - {pt_path}")
        return 1

    # 设置输出重定向
    output_file = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"{pt_path.stem}.txt"
        output_file = open(output_file_path, "w", encoding="utf-8")
        sys.stdout = output_file
        print(f"# 输出文件: {output_file_path}")

    try:
        print(f"加载文件: {pt_path}")
        data = torch.load(pt_path, weights_only=False, map_location="cpu")

        if args.raw:
            print("\n[原始数据]")
            print(data)
            return 0

        mode = detect_mode(data)
        print(f"检测到模式: {mode}")

        if mode == "stat":
            view_stat_mode(data, args)
        elif mode == "per_pass":
            view_per_pass_mode(data, args)
        elif mode == "per_token":
            view_per_token_mode(data, args)
        else:
            view_unknown_mode(data, args)

        print_separator()
    finally:
        if output_file:
            sys.stdout = sys.__stdout__
            output_file.close()
            print(f"已保存到: {output_file_path}")

    return 0


if __name__ == "__main__":
    exit(main())

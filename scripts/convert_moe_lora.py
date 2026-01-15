#!/usr/bin/env python3
"""
将 PEFT 格式的 MoE LoRA adapter 转换为 kt-kernel 友好格式。

Usage:
    python scripts/convert_moe_lora.py \
        --input /path/to/adapter_model.safetensors \
        --config /path/to/adapter_config.json \
        --output /path/to/moe_lora.pt

支持两种 key 命名模式：
    - base_model.model.model.layers.{L}.mlp.original_moe.experts.{E}.{proj}.lora_{A,B}.weight
    - base_model.model.model.layers.{L}.mlp.experts.{E}.{proj}.lora_{A,B}.weight
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# MoE expert LoRA key pattern
# Matches: layers.{L}.mlp.original_moe.experts.{E}.{proj}.lora_{type}.weight
# Or:      layers.{L}.mlp.experts.{E}.{proj}.lora_{type}.weight
MOE_PATTERN = re.compile(
    r".*layers\.(\d+)\.mlp\.(original_moe\.)?experts\.(\d+)\."
    r"(gate|up|down)_proj\.lora_(A|B)\.weight"
)


def parse_moe_key(key: str) -> Optional[Tuple[int, int, str, str]]:
    """
    解析 MoE expert LoRA key。

    Args:
        key: safetensors 中的 key 名称

    Returns:
        (layer_idx, expert_id, proj_type, lora_type) or None
        - layer_idx: 层索引
        - expert_id: 专家 ID
        - proj_type: "gate", "up", "down"
        - lora_type: "a", "b"
    """
    match = MOE_PATTERN.match(key)
    if match:
        layer_idx = int(match.group(1))
        expert_id = int(match.group(3))
        proj_type = match.group(4)  # gate, up, down
        lora_type = match.group(5).lower()  # a, b
        return (layer_idx, expert_id, proj_type, lora_type)
    return None


def load_adapter_config(config_path: str) -> dict:
    """加载 adapter_config.json"""
    with open(config_path, "r") as f:
        return json.load(f)


def convert_peft_to_kt_format(
    input_path: str,
    output_path: str,
    lora_alpha: float = 32.0,
    verbose: bool = False,
) -> dict:
    """
    转换 PEFT 格式的 MoE LoRA adapter 为 kt-kernel 格式。

    Args:
        input_path: PEFT adapter_model.safetensors 路径
        output_path: 输出 .pt 文件路径
        lora_alpha: LoRA alpha 值
        verbose: 是否输出详细信息

    Returns:
        转换统计信息字典
    """
    logger.info(f"Loading adapter from: {input_path}")

    # 1. 读取 safetensors 文件，获取所有 keys
    with safe_open(input_path, framework="pt") as f:
        all_keys = list(f.keys())

    logger.info(f"Total keys in adapter: {len(all_keys)}")

    # 2. 扫描 MoE expert LoRA keys
    # 结构: moe_weights[layer_idx][expert_id][proj_type][lora_type] = tensor
    moe_weights: Dict[int, Dict[int, Dict[str, Dict[str, torch.Tensor]]]] = {}

    moe_key_count = 0
    lora_rank = None

    with safe_open(input_path, framework="pt") as f:
        for key in all_keys:
            parsed = parse_moe_key(key)
            if parsed is None:
                continue

            layer_idx, expert_id, proj_type, lora_type = parsed
            tensor = f.get_tensor(key)

            # 初始化嵌套字典
            if layer_idx not in moe_weights:
                moe_weights[layer_idx] = {}
            if expert_id not in moe_weights[layer_idx]:
                moe_weights[layer_idx][expert_id] = {}
            if proj_type not in moe_weights[layer_idx][expert_id]:
                moe_weights[layer_idx][expert_id][proj_type] = {}

            moe_weights[layer_idx][expert_id][proj_type][lora_type] = tensor
            moe_key_count += 1

            # 推断 lora_rank (从 lora_A 的第一维)
            if lora_type == "a" and lora_rank is None:
                lora_rank = tensor.shape[0]

            if verbose:
                logger.debug(f"  {key}: {tensor.shape}")

    logger.info(f"Found {moe_key_count} MoE expert LoRA keys")

    if moe_key_count == 0:
        logger.warning("No MoE expert LoRA keys found. Nothing to convert.")
        return {"status": "skipped", "reason": "no_moe_keys"}

    # 3. 堆叠权重为 kt-kernel 格式
    result = {
        "metadata": {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
        }
    }

    layer_indices = sorted(moe_weights.keys())
    num_layers = len(layer_indices)
    result["metadata"]["num_layers"] = num_layers

    logger.info(f"Processing {num_layers} layers...")

    for layer_idx in layer_indices:
        layer_data = moe_weights[layer_idx]
        expert_ids = sorted(layer_data.keys())
        num_experts = len(expert_ids)

        # 在第一层记录元数据
        if layer_idx == layer_indices[0]:
            result["metadata"]["num_experts"] = num_experts

            # 推断 hidden_size 和 intermediate_size
            sample_expert = layer_data[expert_ids[0]]
            if "gate" in sample_expert and "a" in sample_expert["gate"]:
                result["metadata"]["hidden_size"] = sample_expert["gate"]["a"].shape[1]
            if "gate" in sample_expert and "b" in sample_expert["gate"]:
                result["metadata"]["intermediate_size"] = sample_expert["gate"][
                    "b"
                ].shape[0]

            logger.info(
                f"  Detected: num_experts={num_experts}, "
                f"lora_rank={lora_rank}, "
                f"hidden_size={result['metadata'].get('hidden_size')}, "
                f"intermediate_size={result['metadata'].get('intermediate_size')}"
            )

        layer_weights = {}

        for proj_type in ["gate", "up", "down"]:
            for lora_type in ["a", "b"]:
                weight_name = f"{proj_type}_lora_{lora_type}"
                tensors = []

                for expert_id in expert_ids:
                    if (
                        expert_id in layer_data
                        and proj_type in layer_data[expert_id]
                        and lora_type in layer_data[expert_id][proj_type]
                    ):
                        tensors.append(layer_data[expert_id][proj_type][lora_type])
                    else:
                        raise ValueError(
                            f"Missing weight: layer={layer_idx}, expert={expert_id}, "
                            f"proj={proj_type}, lora={lora_type}"
                        )

                # 堆叠为 [num_experts, ...]
                stacked = torch.stack(tensors, dim=0).contiguous()
                layer_weights[weight_name] = stacked

                if verbose and layer_idx == layer_indices[0]:
                    logger.info(f"    {weight_name}: {stacked.shape}")

        result[f"layer_{layer_idx}"] = layer_weights

    # 4. 保存结果
    logger.info(f"Saving to: {output_path}")
    torch.save(result, output_path)

    # 5. 输出统计
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    stats = {
        "status": "success",
        "num_layers": num_layers,
        "num_experts": result["metadata"]["num_experts"],
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "hidden_size": result["metadata"].get("hidden_size"),
        "intermediate_size": result["metadata"].get("intermediate_size"),
        "total_keys": moe_key_count,
        "output_file_size_mb": round(file_size_mb, 2),
    }
    logger.info(f"Conversion complete!")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert PEFT MoE LoRA adapter to kt-kernel format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 基本用法
    python scripts/convert_moe_lora.py \\
        --input /path/to/adapter_model.safetensors \\
        --output /path/to/moe_lora.pt

    # 从 config 读取 lora_alpha
    python scripts/convert_moe_lora.py \\
        --input /path/to/adapter_model.safetensors \\
        --config /path/to/adapter_config.json \\
        --output /path/to/moe_lora.pt

    # 详细输出
    python scripts/convert_moe_lora.py \\
        --input /path/to/adapter_model.safetensors \\
        --output /path/to/moe_lora.pt \\
        --verbose
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to adapter_model.safetensors",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to adapter_config.json (optional, for reading lora_alpha)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for .pt file",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=32.0,
        help="LoRA alpha value (default: 32.0, overridden by config if provided)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # 从 config 读取 lora_alpha
    lora_alpha = args.lora_alpha
    if args.config:
        if not Path(args.config).exists():
            logger.warning(f"Config file not found: {args.config}, using default lora_alpha")
        else:
            config = load_adapter_config(args.config)
            lora_alpha = config.get("lora_alpha", lora_alpha)
            logger.info(f"Using lora_alpha from config: {lora_alpha}")

    # 创建输出目录
    output_dir = Path(args.output).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.info(f"Created output directory: {output_dir}")

    # 执行转换
    result = convert_peft_to_kt_format(
        input_path=args.input,
        output_path=args.output,
        lora_alpha=lora_alpha,
        verbose=args.verbose,
    )

    if result["status"] == "skipped":
        logger.warning("Conversion skipped: no MoE expert LoRA keys found")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

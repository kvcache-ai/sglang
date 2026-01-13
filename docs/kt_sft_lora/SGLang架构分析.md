# SGLang 架构分析

本文档记录了 SGLang 中 LoRA 和 MoE 相关模块的架构分析，以及之前的调试尝试。

## 1. 问题背景与调试历程

### 1.1 初始问题

用户使用以下命令启动 SGLang 服务：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
    --lora-paths klora=/mnt/data/lpl/inject_old_test_adapter/Kllama_deepseekV2_WEST/checkpoint-1321_converted \
    --tp 4 --port 31288
```

### 1.2 调试历程

#### 问题1: API 使用错误
- **现象**: 用户使用 `completions` API 且未指定 LoRA adapter
- **解决**: 改用 `chat.completions` API，model 参数使用 `DeepSeek:klora` 格式

#### 问题2: Shape Mismatch 错误
```
AssertionError: LoRA buffer shape torch.Size([144, 8]) does not match weight shape torch.Size([576, 8])
```
- **原因**: `kv_a_proj_with_mqa` 使用 `ReplicatedLinear`（权重在所有 TP rank 上复制），但 LoRA 内存池错误地对其进行了 TP 分片
- **分析**:
  - `kv_a_proj_with_mqa` 输出维度 = `kv_lora_rank(512) + qk_rope_head_dim(64) = 576`
  - 错误地除以 `tp_size=4` 得到 `144`
- **解决**:
  - 在 `utils.py` 添加 `REPLICATED_LINEAR_LORA_NAMES = ["kv_a_proj_with_mqa"]`
  - 在 `mem_pool.py` 的 `get_lora_B_shape()` 中排除 ReplicatedLinear 的 TP 分片

#### 问题3: 输出乱码
- **现象**: 模型输出与输入无关的随机内容
- **根本原因**: SGLang 不支持 MoE routed experts 的 LoRA
- **分析**: 用户 adapter 包含 9984 个 routed experts 权重，占总量 96%，这些权重无法被应用

### 1.3 已修复的代码

**文件: `/home/lpl/sglang/python/sglang/srt/lora/utils.py`**
```python
# 第 170-173 行
EMBEDDING_NAMES = ["embed_tokens", "lm_head"]
ROW_PARALLELISM_LINEAR_LORA_NAMES = ["o_proj", "down_proj"]
# ReplicatedLinear modules: weights are replicated across all TP ranks, not sharded
REPLICATED_LINEAR_LORA_NAMES = ["kv_a_proj_with_mqa"]
```

**文件: `/home/lpl/sglang/python/sglang/srt/lora/mem_pool.py`**
```python
# 第 12-21 行
from sglang.srt.lora.utils import (
    EMBEDDING_NAMES,
    REPLICATED_LINEAR_LORA_NAMES,
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    ...
)

# 第 178-184 行
def get_lora_B_shape(...):
    ...
    if (
        self.tp_size > 1
        and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES
        and module_name not in REPLICATED_LINEAR_LORA_NAMES
    ):
        output_dim = divide(output_dim, self.tp_size)
```

## 2. SGLang LoRA 架构

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                        LoRA Manager                              │
│  - 管理多个 LoRA adapters                                        │
│  - 协调权重加载/卸载                                              │
│  - 处理批处理 LoRA 信息                                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  LoRAAdapter  │   │LoRAMemoryPool │   │ LoRABackend   │
│  - 加载权重    │   │  - GPU 缓冲区  │   │  - Triton 核  │
│  - CPU 存储    │   │  - Eviction    │   │  - SGEMM 实现 │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 2.2 关键文件

| 文件 | 功能 |
|-----|------|
| `lora/lora_manager.py` | LoRA 适配器管理，批处理准备 |
| `lora/mem_pool.py` | GPU 内存池管理，权重缓冲区 |
| `lora/lora.py` | LoRAAdapter 类，权重加载和归一化 |
| `lora/layers.py` | 各种 Layer 的 LoRA wrapper |
| `lora/utils.py` | 工具函数，LoRABatchInfo 数据结构 |
| `lora/backend/triton_backend.py` | Triton 内核调度 |
| `lora/triton_ops/*.py` | Triton SGEMM 内核实现 |

### 2.3 LoRA 权重流程

```
1. 加载阶段 (启动时)
   adapter.safetensors → LoRAAdapter (CPU) → 权重归一化

2. 准备阶段 (每个批次)
   ForwardBatch.lora_ids → LoRAMemoryPool.prepare_lora_batch()
   → 按需从 CPU 复制到 GPU 缓冲区
   → 创建 LoRABatchInfo

3. 推理阶段 (Forward)
   BaseLayerWithLoRA.forward(x)
   → base_layer.forward(x)
   → lora_backend.run_lora_a_sgemm(x, A_buffer)
   → lora_backend.run_lora_b_sgemm(intermediate, B_buffer)
   → output += lora_delta
```

### 2.4 支持的 Layer 类型

| Wrapper 类 | 基础 Layer | 应用模块 |
|-----------|-----------|---------|
| `ColumnParallelLinearWithLoRA` | ColumnParallelLinear | 单独的 q/k/v_proj |
| `QKVParallelLinearWithLoRA` | QKVParallelLinear | 合并的 qkv_proj |
| `MergedColumnParallelLinearWithLoRA` | MergedColumnParallelLinear | gate_up_proj |
| `RowParallelLinearWithLoRA` | RowParallelLinear | o_proj, down_proj |
| `ReplicatedLinearWithLoRA` | ReplicatedLinear | kv_a_proj_with_mqa |
| `VocabParallelEmbeddingWithLoRA` | VocabParallelEmbedding | embed_tokens, lm_head |

### 2.5 LoRABatchInfo 结构

```python
@dataclass
class LoRABatchInfo:
    use_cuda_graph: bool          # 是否使用 CUDA Graph
    bs: int                       # 批大小
    num_segments: int             # 段数量
    seg_indptr: torch.Tensor      # (num_segments+1,) 段起始位置
    weight_indices: torch.Tensor  # (num_segments,) 每段使用的 adapter ID
    lora_ranks: torch.Tensor      # (max_loras,) 每个 adapter 的 rank
    scalings: torch.Tensor        # (max_loras,) 每个 adapter 的 alpha/r
    max_len: Optional[int]        # 最大段长度
    seg_lens: Optional[torch.Tensor]     # (num_segments,) 每段长度
    permutation: Optional[torch.Tensor]  # (num_tokens,) token 重排序
```

## 3. SGLang MoE 架构

### 3.1 FusedMoE 组件

```
┌─────────────────────────────────────────────────────────────────┐
│                         FusedMoE Layer                          │
│  - 融合的 expert 计算                                            │
│  - 支持 TP/EP 分布式                                             │
│  - 支持多种量化方法                                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────────┐
    ▼                       ▼                           ▼
┌──────────────┐   ┌────────────────┐   ┌────────────────────┐
│   MoeGate    │   │  MoeRunner     │   │ StandardDispatcher │
│  - Router    │   │  - 计算核心     │   │  - Token 调度      │
│  - TopK 选择  │   │  - Triton 实现  │   │  - Expert 映射     │
└──────────────┘   └────────────────┘   └────────────────────┘
```

### 3.2 关键文件

| 文件 | 功能 |
|-----|------|
| `layers/moe/fused_moe_triton/layer.py` | FusedMoE 主类 |
| `layers/moe/moe_runner/runner.py` | MoE 计算执行器 |
| `layers/moe/token_dispatcher/standard.py` | Token 调度器 |
| `layers/moe/topk.py` | TopK 选择逻辑 |
| `models/deepseek_v2.py` | DeepSeek-V2 模型实现 |

### 3.3 FusedMoE 权重存储

```python
# Expert 权重 shape
w13_weight: (num_experts, intermediate_size * 2, hidden_size)  # gate + up fused
w2_weight: (num_experts, hidden_size, intermediate_size)        # down proj

# 权重按 expert_id 索引
param.data[expert_id] = loaded_weight  # weight_loader 中
```

### 3.4 FusedMoE Forward 流程

```
1. Router 计算
   router_logits = gate(hidden_states)  # (bs, num_experts)
   topk_weights, topk_ids = topk(router_logits)  # (bs, top_k)

2. Expert 调度
   dispatch_output = dispatcher.dispatch(hidden_states, topk_output)

3. Expert 计算
   for expert_id in selected_experts:
       intermediate = act(gate_up @ x)  # w13
       output = down @ intermediate      # w2

4. 结果合并
   final = dispatcher.combine(expert_outputs, topk_weights)
```

### 3.5 DeepSeek-V2 MoE 配置

```python
# DeepSeek-V2-Lite 配置
n_routed_experts = 64              # 路由专家数量
n_shared_experts = 2               # 共享专家数量
num_experts_per_tok = 6            # 每 token 激活专家数
first_k_dense_replace = 1          # layer 0 使用 dense MLP
moe_intermediate_size = 1408       # 专家中间层大小
intermediate_size = 10944          # Dense MLP 中间层大小
```

### 3.6 Expert 权重路径格式

```
# Routed Experts (layer 1-26)
model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight
model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight
model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight

# Shared Experts
model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight
model.layers.{layer_id}.mlp.shared_experts.up_proj.weight
model.layers.{layer_id}.mlp.shared_experts.down_proj.weight

# Dense MLP (layer 0)
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
```

## 4. LoRA 权重格式分析

### 4.1 PEFT 格式 (转换前)

```
base_model.model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.lora_A.default.weight
base_model.model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.lora_B.default.weight
```

### 4.2 SGLang 格式 (转换后)

```
model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.lora_A.weight
model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.lora_B.weight
```

### 4.3 用户 Adapter 权重统计

```python
# 分类统计
routed_experts: 9984 个权重  # mlp.experts.X
shared_experts: 156 个权重   # shared_experts
attention: 216 个权重        # q_proj, o_proj, kv_*
regular_mlp: 6 个权重        # layer 0 的 mlp
```

## 5. 当前 LoRA 不支持 MoE 的原因

### 5.1 架构差异

| 方面 | 标准 Linear LoRA | MoE Expert LoRA |
|-----|-----------------|-----------------|
| 权重组织 | 单个 weight | per-expert weights |
| Forward | 固定路径 | 动态路由 |
| 调度 | 批量 SGEMM | expert-aware dispatch |
| TP 分片 | 简单切分 | expert 维度 + 内部切分 |

### 5.2 缺失的组件

1. **权重加载**: 无法识别 `mlp.experts.X.*` 格式
2. **内存池**: 无 per-expert 缓冲区
3. **Layer wrapper**: 无 `FusedMoEWithLoRA` 类
4. **Triton kernels**: 无 expert-aware LoRA kernel
5. **Batch info**: 无 expert routing 信息

### 5.3 集成点分析

```
FusedMoE.forward()
    │
    ├─► gate() → topk_ids, topk_weights
    │
    ├─► dispatcher.dispatch() → hidden_states (reordered by expert)
    │
    ├─► runner.run_moe_core() → expert 计算
    │        │
    │        └─► ★ 需要在这里注入 LoRA ★
    │             for each expert:
    │               y = w2 @ act(w13 @ x)
    │               y += lora_B[expert_id] @ (lora_A[expert_id] @ x)
    │
    └─► dispatcher.combine() → final output
```

## 6. Triton Kernel 分析

### 6.1 现有 LoRA Kernels

| Kernel | 功能 | 文件 |
|--------|-----|------|
| `sgemm_lora_a_kernel` | x @ lora_A^T | triton_ops/sgemm_lora_a.py |
| `sgemm_lora_b_kernel` | intermediate @ lora_B^T + fused add | triton_ops/sgemm_lora_b.py |
| `qkv_lora_b_kernel` | 3-way fused QKV projection | triton_ops/qkv_lora_b.py |
| `gate_up_lora_b_kernel` | 2-way fused gate_up projection | triton_ops/gate_up_lora_b.py |

### 6.2 MoE Kernels

| Kernel | 功能 | 文件 |
|--------|-----|------|
| `invoke_fused_moe_kernel` | Expert GEMM | layers/moe/fused_moe_triton/fused_moe.py |
| `moe_align_block_size_kernel` | Token-to-expert 对齐 | layers/moe/topk.py |

### 6.3 需要新增的 Kernel

```python
# MoE LoRA A kernel
# 输入: hidden_states (num_tokens, hidden_size)
#       expert_ids (num_tokens,)  # 每个 token 选择的 expert
#       lora_a_weights (num_experts, rank, hidden_size)
# 输出: intermediate (num_tokens, rank)
moe_lora_a_kernel(hidden_states, expert_ids, lora_a_weights) → intermediate

# MoE LoRA B kernel (fused add)
# 输入: intermediate (num_tokens, rank)
#       expert_ids (num_tokens,)
#       lora_b_weights (num_experts, output_dim, rank)
#       output (num_tokens, output_dim)  # 原始 expert 输出
# 输出: output += lora_b @ intermediate
moe_lora_b_kernel(intermediate, expert_ids, lora_b_weights, output)
```

## 7. 总结

### 7.1 核心发现

1. SGLang 的 LoRA 架构设计良好，支持多种 Linear layer 类型
2. MoE 的 FusedMoE 实现是独立的，未与 LoRA 系统集成
3. 实现 MoE LoRA 需要：
   - 扩展权重加载识别 expert 格式
   - 新增 per-expert 内存缓冲区
   - 实现 expert-aware Triton kernels
   - 创建 FusedMoEWithLoRA wrapper

### 7.2 实现复杂度评估

| 组件 | 复杂度 | 工作量 |
|-----|-------|-------|
| 权重加载 | 中 | 1-2天 |
| 内存池扩展 | 中 | 1-2天 |
| Triton kernels | 高 | 3-5天 |
| FusedMoE 集成 | 高 | 2-3天 |
| 测试验证 | 中 | 2-3天 |
| **总计** | - | **10-15天** |

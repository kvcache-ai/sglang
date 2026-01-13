# SGLang MoE LoRA 调试记录

本文档记录 SGLang 集成 DeepSeek-V2 MoE LoRA 推理过程中遇到的问题及修复状态。

---

## 问题背景

- **模型**: DeepSeek-V2-Lite-Chat
- **LoRA Adapter**: `/mnt/data/lpl/kernel_new_test_adapter/NoKT_deepseekV2_WEST_ALL/checkpoint-133`
- **测试文件**: `/home/lpl/sglang/demo.py`
- **启动命令**:
  ```bash
  CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
      --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
      --lora-paths klora=/mnt/data/lpl/kernel_new_test_adapter/NoKT_deepseekV2_WEST_ALL/checkpoint-133 \
      --tp 1 --port 30000
  ```

---

## Bug 1: dtype 不匹配错误

### 状态: ✅ 已解决

### 症状
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```
错误位置: `moe_lora.py:235`

### 原因分析
在 `moe_lora_forward_batched` 函数中：
- `x.float()` 将 hidden_states 转换为 float32
- `ga[expert_id].T` (LoRA 权重) 仍然是 BFloat16
- `torch.mm` 要求两个矩阵 dtype 相同

### 修复方案
在 `python/sglang/srt/lora/moe_lora.py` 中，将所有 LoRA 权重在计算时也转换为 float32：

```python
# 修改前
intermediate_a = torch.mm(x.float(), ga[expert_id].T)

# 修改后
intermediate_a = torch.mm(x.float(), ga[expert_id].float().T)
```

### 修改文件
| 文件 | 修改位置 |
|------|---------|
| `python/sglang/srt/lora/moe_lora.py` | Line 235, 238, 247, 250 添加 `.float()` |

---

## Bug 2: MoE LoRA 权重未加载

### 状态: ✅ 已解决

### 症状
- 服务启动日志中没有 "loaded X MoE expert weights" 信息
- LoRA 请求输出乱码

### 原因分析
在 `python/sglang/srt/lora/lora.py` 中：

```python
# Line 46-48: 正则表达式
MOE_EXPERT_PATTERN = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_([AB])\.weight"
)

# Line 65: 使用 re.match（要求从字符串开头匹配）
match = MOE_EXPERT_PATTERN.match(weight_name)
```

**问题**: PEFT 格式的权重名带有 `base_model.model.` 前缀：
- 实际权重名: `base_model.model.model.layers.1.mlp.experts.0.gate_proj.lora_A.weight`
- 期望格式: `model.layers.1.mlp.experts.0.gate_proj.lora_A.weight`

**对比**: 普通 LoRA 权重使用 `get_layer_id()` 函数（使用 `re.search`），可以在任意位置匹配。但 `MOE_EXPERT_PATTERN` 使用 `re.match`，导致匹配失败。

### 修复方案
```python
# 修改前
match = MOE_EXPERT_PATTERN.match(weight_name)

# 修改后
match = MOE_EXPERT_PATTERN.search(weight_name)
```

### 修改文件
| 文件 | 修改位置 |
|------|---------|
| `python/sglang/srt/lora/lora.py` | Line 65: `.match()` → `.search()` |

### 验证
修复后，启动日志显示:
```
LoRA adapter loaded 9984 MoE expert weights across 26 layers, covering 1664 unique experts
```

---

## Bug 3: MoE LoRA 未使用 Base 权重

### 状态: ✅ 已解决

### 症状
权重加载成功后，LoRA 请求仍然输出乱码或不正确的结果。

### 原因分析
原始实现把 LoRA 当作独立网络计算，而不是对 base 权重的修正：

```python
# 错误的计算逻辑（原实现）
# 完全独立的 LoRA 路径，没有 base 权重的贡献
gate_up_out = gate_up_B @ gate_up_A @ x
activated = silu(gate_up_out[:half]) * gate_up_out[half:]
down_out = down_B @ down_A @ activated
```

**正确的 LoRA 计算应该是**:
```python
# 对于每个专家，正确计算 (W + BA) @ x
gate_up_out = W_gate_up @ x + (gate_up_B @ gate_up_A @ x) * scaling
activated = silu(gate) * up
down_out = W_down @ activated + (down_B @ down_A @ activated) * scaling
```

### 修复方案
重写 MoE LoRA 计算逻辑，使用 base MoE 权重 (`w13_weight`, `w2_weight`) 进行正确计算：

1. **deepseek_v2.py**: 修改 `_apply_moe_lora` 传入 base 权重
2. **lora_manager.py**: 修改 `apply_moe_lora` 接收并传递 base 权重
3. **moe_lora.py**: 重写 `moe_lora_forward_batched` 和 `moe_lora_forward`，使用 base + LoRA 权重计算

### 修改文件
| 文件 | 修改内容 |
|------|---------|
| `python/sglang/srt/models/deepseek_v2.py` | `_apply_moe_lora` 获取并传入 base 权重 |
| `python/sglang/srt/lora/lora_manager.py` | `apply_moe_lora` 添加 base 权重参数 |
| `python/sglang/srt/lora/moe_lora.py` | 完全重写计算逻辑 |

---

## Bug 4: gate_proj 和 up_proj 的 LoRA A 矩阵错误合并

### 状态: ✅ 已解决

### 症状
Bug 3 修复后，输出仍然是乱码。

### 原因分析
在 `lora.py` 的 `MoELoRALayer.finalize()` 函数中，代码错误地假设 `gate_proj` 和 `up_proj` 的 `lora_A` 矩阵相同：

```python
# 错误的假设！！！
# For LoRA A: gate_a and up_a should be the same (both project from hidden to rank)
# We use gate_a as the representative
if expert_id in gate_a:
    self.gate_up_lora_a[expert_id] = gate_a[expert_id]  # 错误：丢失了 up_a
elif expert_id in up_a:
    self.gate_up_lora_a[expert_id] = up_a[expert_id]
```

**问题**：在 PEFT LoRA 中，每个线性层（`gate_proj`、`up_proj`）都有**独立的** `lora_A` 和 `lora_B` 矩阵。它们是分别训练的，不能互相替代。

**错误的计算**：
```
gate_up_out = W_gate_up @ x + cat([B_gate, B_up]) @ A_gate @ x
                                                    ^^^^^^ 错误：用 A_gate 替代了 A_up
```

**正确的计算**：
```
gate_out = W_gate @ x + B_gate @ A_gate @ x
up_out = W_up @ x + B_up @ A_up @ x
gate_up_out = cat([gate_out, up_out])
```

### 修复方案
分开存储和计算 gate 和 up 的 LoRA 权重：

1. **lora.py**: 重构 `MoELoRALayer`，分开存储 `gate_lora_a`, `gate_lora_b`, `up_lora_a`, `up_lora_b`
2. **moe_mem_pool.py**: 分开缓冲区，从 4 个变为 6 个：`gate_A`, `gate_B`, `up_A`, `up_B`, `down_A`, `down_B`
3. **moe_lora.py**: 分开计算 gate 和 up 的 LoRA 贡献，然后合并
4. **lora_manager.py**: 传递 6 个缓冲区

### 修改文件
| 文件 | 修改内容 |
|------|---------|
| `python/sglang/srt/lora/lora.py` | `MoELoRALayer` 分开存储 gate 和 up |
| `python/sglang/srt/lora/moe_mem_pool.py` | 6 个独立缓冲区 |
| `python/sglang/srt/lora/moe_lora.py` | 分开计算 gate 和 up LoRA |
| `python/sglang/srt/lora/lora_manager.py` | 传递 6 个缓冲区 |

### 核心修改代码
```python
# moe_lora.py 中的正确计算
# 1. gate with LoRA (使用独立的 A_gate)
base_gate_out = torch.mm(x, W_gate.T)
lora_gate_out = torch.mm(torch.mm(x, ga[expert_id].T), gb[expert_id].T)
gate_out = base_gate_out + lora_gate_out * scaling

# 2. up with LoRA (使用独立的 A_up)
base_up_out = torch.mm(x, W_up.T)
lora_up_out = torch.mm(torch.mm(x, ua[expert_id].T), ub[expert_id].T)
up_out = base_up_out + lora_up_out * scaling

# 3. Activation
activated = F.silu(gate_out) * up_out

# 4. down with LoRA
base_down_out = torch.mm(activated, W_down.T)
lora_down_out = torch.mm(torch.mm(activated, da[expert_id].T), db[expert_id].T)
expert_out = base_down_out + lora_down_out * scaling
```

---

## LoRA Adapter 信息

### 权重结构
```
Attention LoRA 权重数量: 216
MoE Expert LoRA 权重数量: 10140
Dense MLP LoRA 权重数量: 6 (layer 0)

MoE Expert 权重分布:
  gate_proj: 3380 个权重
  up_proj: 3380 个权重
  down_proj: 3380 个权重
```

### 权重形状 (rank=8)
```
gate_proj.lora_A: (8, 2048)  = (rank, hidden_size)
gate_proj.lora_B: (1408, 8)  = (intermediate_size, rank)
up_proj.lora_A:   (8, 2048)  = (rank, hidden_size)
up_proj.lora_B:   (1408, 8)  = (intermediate_size, rank)
down_proj.lora_A: (8, 1408)  = (rank, intermediate_size)
down_proj.lora_B: (2048, 8)  = (hidden_size, rank)
```

---

## 待办事项

- [x] Bug 1: dtype 不匹配 - ✅ 已解决
- [x] Bug 2: 权重加载正则匹配 - ✅ 已解决
- [x] Bug 3: 未使用 base 权重 - ✅ 已解决
- [x] Bug 4: gate/up LoRA A 矩阵合并错误 - ✅ 已解决
- [x] Bug 5: 基础权重布局不匹配 triton kernels - ✅ 已解决
- [ ] 与 LLaMA-Factory 推理结果对比验证
- [ ] 性能优化（当前使用非融合计算，较慢）

---

## Bug 5: 基础权重布局不匹配 triton kernels

### 状态: ✅ 已解决

### 症状
Bug 4 修复后，LoRA 请求仍然输出乱码。

### 原因分析
SGLang 的 FusedMoE 权重布局取决于后端：

| Backend | w13_weight (gate_up) | w2_weight (down) |
|---------|---------------------|------------------|
| 非 triton | (E, inter*2, hidden) | (E, hidden, inter) |
| triton_kernel | (E, hidden, inter*2) | (E, inter, hidden) |

代码假设非 triton 布局：
```python
W_gate = W_gate_up[:inter_size, :]   # 错误：triton 布局是 (hidden, inter*2)
```

但 SGLang 默认使用 triton kernels！

### 修复方案
在 `moe_lora.py` 中检测权重布局并正确处理：

**gate_up 权重处理：**
```python
if W_gate_up.shape[0] == hidden_size:
    # triton kernels 布局: (hidden, inter*2)
    W_gate = W_gate_up[:, :inter_size]   # 切片第二维
    base_gate_out = torch.mm(x, W_gate)  # 不转置
else:
    # 标准布局: (inter*2, hidden)
    W_gate = W_gate_up[:inter_size, :]   # 切片第一维
    base_gate_out = torch.mm(x, W_gate.T)  # 转置
```

**down 权重处理：**
```python
if W_down.shape[0] == inter_size:
    # triton kernels 布局: (inter, hidden)
    base_down_out = torch.mm(activated, W_down)  # 不转置
else:
    # 标准布局: (hidden, inter)
    base_down_out = torch.mm(activated, W_down.T)  # 转置
```

### 修改文件
| 文件 | 修改位置 |
|------|---------|
| `python/sglang/srt/lora/moe_lora.py` | `moe_lora_forward_batched` 和 `moe_lora_forward` 函数 |

### 注意事项
初始修复时 W_down 的转置逻辑写反了，导致 RuntimeError: mat1 and mat2 shapes cannot be multiplied。
正确逻辑：triton 布局下 W_down 需要保持原样（因为已经是 inter x hidden），标准布局下需要转置。

---

## 相关文件

| 文件路径 | 描述 |
|---------|------|
| `python/sglang/srt/lora/moe_lora.py` | MoE LoRA 核心计算逻辑 |
| `python/sglang/srt/lora/moe_mem_pool.py` | MoE LoRA 内存管理 |
| `python/sglang/srt/lora/lora.py` | LoRA 数据结构和权重加载 |
| `python/sglang/srt/lora/lora_manager.py` | LoRA 管理器 |
| `python/sglang/srt/models/deepseek_v2.py` | DeepSeek V2 模型定义 |
| `/home/lpl/sglang/demo.py` | 测试脚本 |

# CUDA Graph 与 kt-kernel 兼容性分析

## 1. CUDA Graph 基本机制

### 1.1 Capture 阶段

CUDA Graph 通过"录制"CUDA 操作来工作：

```python
# cuda_graph_runner.py 中的 capture 逻辑
def capture(self):
    with torch.cuda.graph(self.graph, stream=self.stream):
        # 这里的所有 CUDA 操作都会被录制
        output = model.forward(input)
```

**关键点：**
- 在 capture 阶段，**所有 Python 代码都会执行**
- 但只有 **CUDA 操作（kernel launch、memory copy 等）** 会被录制到 graph 中
- CPU 操作（Python 计算、CPU tensor 操作）会执行但**不会被录制**

### 1.2 Replay 阶段

```python
# cuda_graph_runner.py 中的 replay 逻辑
def replay(self):
    self.graph.replay()  # 只执行录制的 CUDA 操作
```

**关键点：**
- Replay 时只执行录制的 CUDA 操作序列
- **不执行任何 Python 代码**
- 输入/输出必须使用 capture 时的同一块内存（通过 `copy_()` 更新）

### 1.3 CUDA Graph 的限制

| 操作类型 | 能否被 Capture | 说明 |
|---------|--------------|------|
| CUDA kernel | ✅ | GPU 计算核心 |
| cudaMemcpyAsync (D2D) | ✅ | GPU 内显存拷贝 |
| cudaMemcpyAsync (H2D/D2H) | ❌ | 涉及 CPU 内存 |
| CPU 计算 | ❌ | 完全不在 CUDA 域 |
| Python 控制流 | ❌ | 只在 capture 时执行一次 |
| 动态 shape | ❌ | Graph 要求固定 shape |

---

## 2. sglang CUDA Graph 实现

### 2.1 普通 CUDA Graph

位置：`python/sglang/srt/model_executor/cuda_graph_runner.py`

```python
class CudaGraphRunner:
    def capture(self):
        # 为不同 batch size 预先捕获 graph
        for bs in self.batch_size_list:
            with model_capture_mode():
                self._capture_one_stream(bs)

    def replay(self, batch_size):
        # 直接 replay 对应的 graph
        self.graphs[batch_size].replay()
```

检测 capture 模式的方法：
```python
from sglang.srt.model_executor.forward_batch_info import get_is_capture_mode

if get_is_capture_mode():
    # 当前在 CUDA graph capture 中
    pass
```

### 2.2 Piecewise CUDA Graph

位置：`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

分段捕获，MoE 层有特殊处理：
```python
# 在 MoE forward 中
if is_in_piecewise_cuda_graph():
    return moe_forward_piecewise_cuda_graph_impl(...)
```

### 2.3 自动禁用 CUDA Graph 的场景

在 `server_args.py` 中已有多处自动禁用逻辑：

```python
def _handle_a2a_moe(self):
    """a2a MoE 需要通信，禁用 CUDA graph"""
    if self.enable_moe_a2a:
        self.disable_cuda_graph = True

def _handle_dllm_inference(self):
    """DLLM 推理需要动态控制流，禁用 CUDA graph"""
    if self.dllm_inference:
        self.disable_cuda_graph = True
```

---

## 3. kt-kernel CPU Offloading 机制

### 3.1 推理模式 (Inference Mode)

```python
# kt_ep_wrapper.py
class KTEPOffloadingMethod:
    def apply(self, layer, dispatch_output):
        # Step 1: 异步提交 CPU 计算
        if self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        # Step 2-3: GPU 计算（可以被 CUDA graph 捕获）
        gpu_result = self.gpu_method.apply(layer, masked_dispatch_output)

        # Step 4: 合并结果
        output = gpu_result.hidden_states  # GPU tensor
        if self.tp_rank == 0:
            cpu_output = self.sync(x, dispatch_output)  # 等待 CPU 完成
            output = output + cpu_output  # 合并

        return StandardCombineInput(hidden_states=output)
```

`sync()` 方法调用 `wrapper.sync_forward()`：
```python
# experts_base.py
def sync_forward(self, hidden_states, cuda_stream):
    # 等待 CPU 计算完成
    self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)  # CPU 操作！

    # 将结果从 CPU 拷贝到 GPU buffer
    output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)

    return output_gpu[current_slot]  # 返回 GPU tensor
```

### 3.2 SFT 模式

```python
# kt_ep_wrapper.py 中 sync() 方法
def sync(self, x, dispatch_output=None):
    if self.kt_config.moe_lora_enabled:
        # SFT 模式：直接返回 CPU tensor
        return self.wrapper.forward_sft(x, topk_ids, topk_weights, ...)
    else:
        # 推理模式：返回 GPU tensor
        return self.wrapper.sync_forward(x, cuda_stream)
```

`forward_sft()` 返回 CPU tensor：
```python
# amx_sft.py
def forward_sft(self, hidden_states, expert_ids, weights, save_for_backward=True):
    # ... CPU 上的计算 ...
    return buffer.output_cpu.clone()  # 返回 CPU tensor！
```

---

## 4. CUDA Graph 兼容性分析（修正版）

### 4.1 重要澄清

**之前的分析有误！** 经过实测验证：

- **标准推理模式**（不含 MoE LoRA）：✅ **可以兼容 CUDA graph**
- **SFT 模式**（使用 `--kt-moe-lora-path`）：❌ **无法兼容 CUDA graph**

### 4.2 标准推理模式为何兼容

标准推理模式使用 `sync_forward()` 方法：

```python
# experts_base.py
def sync_forward(self, hidden_states, cuda_stream):
    # 等待 CPU 计算完成
    self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)

    # 将结果从 CPU 拷贝到 GPU buffer
    output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)

    return output_gpu[current_slot]  # 返回 GPU tensor ✓
```

关键点：`sync_forward()` 返回 **GPU tensor**，在内部完成了 CPU→GPU 拷贝。

### 4.3 SFT 模式为何不兼容

SFT 模式使用 `forward_sft()` 方法：

```python
# amx_sft.py
def forward_sft(self, hidden_states, expert_ids, weights, save_for_backward=True):
    # ... CPU 上的计算 ...
    return buffer.output_cpu.clone()  # 返回 CPU tensor ❌
```

`forward_sft()` 返回 **CPU tensor**，导致：

```python
# kt_ep_wrapper.py apply() 方法
output = gpu_combine_input.hidden_states  # cuda:0
cpu_output = self.sync(x, dispatch_output)  # cpu (SFT 模式)
output = output + cpu_output  # 设备不匹配!
```

### 4.4 解决方案

**已实现以下修复**：

1. **修复设备不匹配** (`kt_ep_wrapper.py`):
   ```python
   cpu_output = self.sync(x, dispatch_output)
   if cpu_output.device != output.device:
       cpu_output = cpu_output.to(output.device, non_blocking=True)
   output = output + cpu_output
   ```

2. **自动禁用 CUDA Graph** (`server_args.py`):
   - 当 `--kt-moe-lora-path` 设置时，自动禁用 CUDA graph
   - 因为 SFT 模式的 `forward_sft()` 在 replay 时不会执行

---

## 5. 模式对比总结

| 模式 | sync 方法 | 返回设备 | CUDA Graph 兼容 |
|------|----------|---------|----------------|
| **标准推理** | `sync_forward()` | GPU | ✅ 兼容 |
| **SFT (MoE LoRA)** | `forward_sft()` | CPU | ❌ 需禁用 |

---

## 6. CUDA Graph 对不同组件的收益分析

### 6.1 收益来源

CUDA Graph 的主要收益来自**减少 kernel launch overhead**：
- 每次 kernel launch 有约 5-10 微秒的 CPU 开销
- 对于大量小 kernel 的场景收益明显
- 对于少量大 kernel 的场景收益有限

### 6.2 各组件分析

| 组件 | Kernel 数量 | 单 Kernel 大小 | Launch Overhead 占比 | CUDA Graph 收益 |
|------|------------|---------------|---------------------|-----------------|
| **Attention** | 多（Q/K/V proj, softmax, output proj...）| 小到中 | **高** | **高** |
| **MLP (Dense)** | 少 | 大 | 低 | 低 |
| **MoE GPU** | 少 | 大 | 低 | 低 |
| **MoE CPU (kt-kernel)** | N/A | N/A | N/A | **不适用** |

### 6.3 结论

- CUDA Graph 主要加速 **Attention 部分**
- MoE 部分收益有限，即使禁用 CUDA graph 影响也不大
- **标准 kt-kernel 推理**可以兼容 CUDA graph
- **SFT 模式 (MoE LoRA)** 需要禁用 CUDA graph

---

## 7. 解决方案（已实现）

### 7.1 修复设备不匹配

在 `kt_ep_wrapper.py` 的 `apply()` 方法中：

```python
cpu_output = self.sync(x, dispatch_output)
# SFT 模式返回 CPU tensor，需要移动到 GPU
if cpu_output.device != output.device:
    cpu_output = cpu_output.to(output.device, non_blocking=True)
output = output + cpu_output
```

### 7.2 自动禁用 CUDA Graph（仅 SFT 模式）

在 `server_args.py` 中添加：

```python
def _handle_kt_moe_lora(self):
    """Disable CUDA graph when kt-kernel MoE LoRA (SFT mode) is enabled."""
    if self.kt_moe_lora_path is not None:
        if not self.disable_cuda_graph:
            logger.warning(
                "CUDA graph is disabled because kt-kernel MoE LoRA is enabled. "
                "SFT mode requires synchronous CPU computation."
            )
            self.disable_cuda_graph = True
```

**注意**：只有 SFT 模式（使用 `--kt-moe-lora-path`）需要禁用 CUDA graph，标准 kt-kernel 推理不需要。

---

## 8. 已修改文件清单

| 文件 | 修改内容 |
|-----|---------|
| `python/sglang/srt/layers/moe/kt_ep_wrapper.py` | 修复设备不匹配，添加 `.to(output.device)` |
| `python/sglang/srt/server_args.py` | 添加 `_handle_kt_moe_lora()` 方法 |

---

## 9. 总结

1. **标准 kt-kernel 推理**：✅ 兼容 CUDA graph（`sync_forward()` 返回 GPU tensor）
2. **SFT 模式 (MoE LoRA)**：❌ 需禁用 CUDA graph（`forward_sft()` 返回 CPU tensor）
3. **解决方案**：
   - 修复设备不匹配：将 CPU tensor 移动到 GPU
   - 自动禁用：当 `--kt-moe-lora-path` 设置时禁用 CUDA graph
5. **影响评估**：CUDA Graph 主要加速 Attention，禁用对整体性能影响有限

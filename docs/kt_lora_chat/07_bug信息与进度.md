# MoE Expert LoRA Bug 信息与进度

## Bug 列表

### Bug #1: `kt_num_gpu_experts` 为 None 导致启动失败

**状态**: 未解决

**发现日期**: 2026-01-15

**错误信息**:
```
TypeError: empty() received an invalid combination of arguments - got (NoneType, int, int, dtype=torch.dtype), but expected one of:
 * (tuple of ints size, *, tuple of names names, torch.memory_format memory_format = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
 * (tuple of ints size, *, torch.memory_format memory_format = None, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
```

**触发命令**:
```bash
python -m sglang.launch_server \
      --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
      --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
      --kt-moe-lora-path /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model_converted.safetensors \
      --kt-moe-lora-rank 8 \
      --kt-moe-lora-alpha 16.0 \
      --kt-moe-sft-method AMXBF16_SFT \
      --lora-paths /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model.safetensors
```

**错误调用栈**:
```
File "kt_ep_wrapper.py", line 1154, in create_weights
    self.gpu_method.create_weights(
File "unquant.py", line 181, in create_weights
    torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
TypeError: empty() received an invalid combination of arguments
```

**根本原因分析**:

1. **问题位置**: `python/sglang/srt/layers/moe/kt_ep_wrapper.py:1154`

2. **问题流程**:
   ```
   server_args.py:500
       kt_num_gpu_experts: Optional[int] = None  (默认值为 None)
           ↓
   用户未指定 --kt-num-gpu-experts 参数
           ↓
   create_kt_config_from_server_args()
       num_gpu_experts=server_args.kt_num_gpu_experts  (传入 None)
           ↓
   KTEPWrapperMethod.__init__()
       self.num_gpu_experts = kt_config.num_gpu_experts  (保存 None)
           ↓
   KTEPWrapperMethod.create_weights()
       self.gpu_method.create_weights(num_experts=self.num_gpu_experts)  (传入 None)
           ↓
   UnquantizedFusedMoEMethod.create_weights()
       torch.empty(num_experts, w13_weight_n, w13_weight_k, ...)  (num_experts=None 导致错误)
   ```

3. **核心问题**: 当启用 MoE LoRA (SFT 模式) 时，代码没有处理 `kt_num_gpu_experts` 为 `None` 的情况。在 SFT 模式下，用户期望所有专家在 CPU 上运行（因为 SFT 模式处理 LoRA 计算），但代码要求必须显式指定 GPU 专家数量。

**临时解决方案**:

在启动命令中添加 `--kt-num-gpu-experts 0`:
```bash
python -m sglang.launch_server \
      --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
      --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
      --kt-moe-lora-path /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model_converted.safetensors \
      --kt-moe-lora-rank 8 \
      --kt-moe-lora-alpha 16.0 \
      --kt-moe-sft-method AMXBF16_SFT \
      --kt-num-gpu-experts 0 \
      --lora-paths /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model.safetensors
```

**建议的代码修复方案**:

在 `kt_ep_wrapper.py` 的 `KTEPWrapperMethod.__init__()` 中添加自动默认值逻辑：

```python
def __init__(
    self,
    gpu_method: FusedMoEMethodBase,
    kt_config: KTConfig,
):
    # ... existing code ...

    # 当启用 MoE LoRA 时，如果未指定 GPU 专家数，默认为 0（全部在 CPU）
    if kt_config.moe_lora_enabled and kt_config.num_gpu_experts is None:
        self.num_gpu_experts = 0
    else:
        self.num_gpu_experts = kt_config.num_gpu_experts
```

或者在 `create_kt_config_from_server_args()` 中添加验证：

```python
def create_kt_config_from_server_args(server_args, layer_idx):
    # ...
    moe_lora_enabled = server_args.kt_moe_lora_path is not None

    # 自动设置 num_gpu_experts 默认值
    num_gpu_experts = server_args.kt_num_gpu_experts
    if num_gpu_experts is None:
        num_gpu_experts = 0  # 默认全部在 CPU

    return KTConfig(
        # ...
        num_gpu_experts=num_gpu_experts,
        # ...
    )
```

**相关文件**:
- `python/sglang/srt/layers/moe/kt_ep_wrapper.py` (第 1100, 1154 行)
- `python/sglang/srt/server_args.py` (第 500 行)
- `python/sglang/srt/layers/quantization/unquant.py` (第 181 行)

---

### Bug #2: `kt_cpuinfer` 为 None 导致 KTMoEWrapper 初始化失败

**状态**: 未解决

**发现日期**: 2026-01-15

**错误信息**:
```
TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'
```

**触发命令**:
```bash
python -m sglang.launch_server \
      --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
      --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
      --kt-moe-lora-path /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model_converted.safetensors \
      --kt-moe-lora-rank 8 \
      --kt-moe-lora-alpha 16.0 \
      --kt-moe-sft-method AMXBF16_SFT \
      --kt-num-gpu-experts 0 \
      --lora-paths /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model.safetensors
```

**错误调用栈**:
```
File "kt_ep_wrapper.py", line 1168, in create_weights
    self.wrapper = KTMoEWrapper(
File "kt_kernel/experts.py", line 200, in __new__
    return _create_sft_wrapper(
File "kt_kernel/experts.py", line 353, in _create_sft_wrapper
    return AMXSFTMoEWrapper(
File "kt_kernel/utils/amx_sft.py", line 117, in __init__
    super().__init__(
File "kt_kernel/experts_sft.py", line 229, in __init__
    self.cpu_infer = self._get_cpu_infer(cpuinfer_threads, threadpool_count)
File "kt_kernel/experts_base.py", line 122, in _get_cpu_infer
    subpool_thread_count = [
File "kt_kernel/experts_base.py", line 123, in <listcomp>
    cpuinfer_threads // threadpool_count + ...
TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'
```

**根本原因分析**:

1. **问题位置**: `kt_ep_wrapper.py:1168` → `kt_kernel/experts_base.py:123`

2. **问题流程**:
   ```
   server_args.py:498
       kt_cpuinfer: Optional[int] = None  (默认值为 None)
           ↓
   用户未指定 --kt-cpuinfer 参数
           ↓
   create_kt_config_from_server_args()
       cpuinfer_threads=server_args.kt_cpuinfer  (传入 None)
           ↓
   KTMoEWrapper() 创建时
       cpuinfer_threads=self.kt_config.cpuinfer_threads  (传入 None)
           ↓
   kt_kernel/experts_base.py:123
       cpuinfer_threads // threadpool_count  (None // 2 导致 TypeError)
   ```

3. **核心问题**: `kt_cpuinfer` 参数没有默认值，必须由用户显式指定。

**临时解决方案**:

在启动命令中添加 `--kt-cpuinfer` 参数（建议设置为 CPU 核心数，如 60）:
```bash
python -m sglang.launch_server \
      --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
      --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
      --kt-moe-lora-path /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model_converted.safetensors \
      --kt-moe-lora-rank 8 \
      --kt-moe-lora-alpha 16.0 \
      --kt-moe-sft-method AMXBF16_SFT \
      --kt-num-gpu-experts 0 \
      --kt-cpuinfer 60 \
      --lora-paths /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model.safetensors
```

**建议的代码修复方案**:

在 `create_kt_config_from_server_args()` 中为 `kt_cpuinfer` 设置合理的默认值：

```python
import os

def create_kt_config_from_server_args(server_args, layer_idx):
    # ...

    # 自动设置 cpuinfer_threads 默认值
    cpuinfer_threads = server_args.kt_cpuinfer
    if cpuinfer_threads is None:
        # 默认使用 CPU 核心数的一半或 60，取较小值
        cpuinfer_threads = min(os.cpu_count() // 2, 60)

    return KTConfig(
        # ...
        cpuinfer_threads=cpuinfer_threads,
        # ...
    )
```

**相关文件**:
- `python/sglang/srt/layers/moe/kt_ep_wrapper.py` (第 1168, 1175 行)
- `python/sglang/srt/server_args.py` (第 498 行)
- `kt_kernel/experts_base.py` (第 122-123 行)

---

### Bug #3: SFT 模式下 `load_weights()` 调用顺序错误

**状态**: 已解决 ✅

**发现日期**: 2026-01-15

**解决日期**: 2026-01-15

**错误信息**:
```
RuntimeError: Base weights not set. Call load_weights_from_tensors() first, or ensure gate_proj, up_proj, down_proj are set before calling load_weights().
```

**触发命令**:
```bash
python -m sglang.launch_server \
      --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
      --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
      --kt-moe-lora-path /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model_converted.safetensors \
      --kt-moe-lora-rank 8 \
      --kt-moe-lora-alpha 16.0 \
      --kt-moe-sft-method AMXBF16_SFT \
      --kt-num-gpu-experts 0 \
      --kt-cpuinfer 60 \
      --lora-paths /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model.safetensors
```

**错误调用栈**:
```
File "kt_ep_wrapper.py", line 1226, in process_weights_after_loading
    self.wrapper.load_weights(physical_to_logical_map_cpu)
File "kt_kernel/utils/amx_sft.py", line 172, in load_weights
    raise RuntimeError(
RuntimeError: Base weights not set. Call load_weights_from_tensors() first...
```

**根本原因分析**:

1. **问题位置**: `kt_ep_wrapper.py:1226` → `kt_kernel/utils/amx_sft.py:172`

2. **问题描述**:
   - **Inference 模式**: `load_weights(physical_to_logical_map)` 直接从文件加载权重
   - **SFT 模式**: 需要先调用 `load_weights_from_tensors(gate_proj, up_proj, down_proj, ...)` 设置基础权重，然后再调用 `load_weights()`

3. **核心问题**: SFT 模式 (`AMXSFTMoEWrapper`) 的权重加载 API 与 Inference 模式 (`AMXMoEWrapper`) 不同

**解决方案**:

**关键发现**: `MOESFTConfig` 继承自 `GeneralMOEConfig`，因此 SFT 模式也可以使用与 Inference 模式相同的文件加载机制。继承关系详见 `kt-kernel/operators/common.hpp`:

```cpp
struct MOESFTConfig : public GeneralMOEConfig {
    int lora_rank = 16;
    float lora_alpha = 32.0f;
    // LoRA weight pointers...
};

struct GeneralMOEConfig {
    std::vector<std::vector<void*>> gate_projs;
    std::vector<std::vector<void*>> gate_scales;
    std::string path;
    bool save = false;
    bool load = false;
    // ...
};
```

**加载策略（按 SFT 方法类型）**:

| SFT Method | 权重来源 | 加载器 | 是否需要 Scale |
|------------|----------|--------|----------------|
| AMXBF16_SFT | HuggingFace 模型路径 (`--model-path`) | `BF16SafeTensorLoader` | 否 |
| AMXINT8_SFT | KT 量化权重路径 (`--kt-weight-path`) | `SafeTensorLoader` | 是 |
| AMXINT4_SFT | KT 量化权重路径 (`--kt-weight-path`) | `SafeTensorLoader` | 是 |

**实现修改**:

1. **新增 `BF16SafeTensorLoader`** (`kt-kernel/python/utils/loader.py`):
   - 专门用于加载 HuggingFace 格式的 BF16 权重
   - 自动检测 DeepSeek/Mixtral 格式
   - 返回 `gate_scale: None, up_scale: None, down_scale: None`

2. **修改 `AMXSFTMoEWrapper.load_weights()`** (`kt-kernel/python/utils/amx_sft.py`):
   - 支持从文件加载，类似于 Inference 模式
   - 根据 `method` 选择正确的 Loader

3. **修改 `kt_ep_wrapper.py`**:
   - SFT 模式下设置正确的权重路径：
     - AMXBF16_SFT: 使用 `model_path` (HuggingFace 模型路径)
     - AMXINT8/INT4_SFT: 使用 `kt_weight_path`

**代码修复** (`kt_ep_wrapper.py`):

```python
def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    # ...
    if self.tp_rank == 0 and self.wrapper is not None:
        # SFT 模式和 Inference 模式都使用相同的 load_weights() 接口
        # AMXSFTMoEWrapper.load_weights() 已修改为支持文件加载
        self.wrapper.load_weights(physical_to_logical_map_cpu)

        # 加载 MoE LoRA 权重
        if self.kt_config.moe_lora_enabled:
            self._load_moe_lora_weights()
```

**代码修复** (`kt-kernel/python/utils/amx_sft.py`):

```python
def load_weights(self, physical_to_logical_map_cpu: torch.Tensor) -> None:
    """Load base weights for this layer."""
    if self._weights_loaded:
        return

    # 如果基础权重未设置，从文件加载
    if self.gate_proj is None or self.up_proj is None or self.down_proj is None:
        self._load_base_weights_from_file()

    # ... 继续原有逻辑
```

**相关文件**:
- `kt-kernel/python/utils/loader.py` (新增 `BF16SafeTensorLoader`)
- `kt-kernel/python/utils/amx_sft.py` (修改 `load_weights()`)
- `python/sglang/srt/layers/moe/kt_ep_wrapper.py` (设置正确的权重路径)

---

### Bug #4: `safe_open` 对象没有 `close()` 方法

**状态**: 已解决 ✅

**发现日期**: 2026-01-16

**解决日期**: 2026-01-16

**错误信息**:
```
File "kt_kernel/utils/loader.py", line 168, in close_all_handles
    handle.close()
    ^^^^^^^^^^^^
AttributeError: 'builtins.safe_open' object has no attribute 'close'
```

**触发命令**:
```bash
# 任何使用 SFT 模式加载权重的命令
python -m sglang.launch_server \
    --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
    --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
    --kt-moe-lora-path /path/to/moe_lora.pt \
    --kt-moe-sft-method AMXBF16_SFT \
    --kt-num-gpu-experts 0 \
    --kt-cpuinfer 60
```

**根本原因分析**:

1. **问题位置**: `kt-kernel/python/utils/loader.py:168`

2. **问题描述**:
   - `safetensors.safe_open` 返回的对象没有 `close()` 方法
   - 该对象被设计为使用 Python 的 context manager（`with` 语句）来管理资源
   - 代码尝试调用 `handle.close()` 会导致 `AttributeError`

3. **触发流程**:
   ```
   AMXSFTMoEWrapper._load_base_weights_from_file()
       ↓
   BF16SafeTensorLoader.load_experts()
       ↓
   loader.close_all_handles()
       ↓
   handle.close()  ← safe_open 没有此方法，报错
   ```

**解决方案**:

修改 `close_all_handles()` 方法，不调用 `close()`，只清除引用让垃圾回收处理：

```python
# kt-kernel/python/utils/loader.py
def close_all_handles(self):
    """Close all file handles and clear the handle map.

    Note: safetensors.safe_open doesn't have a close() method,
    so we just clear the references and let garbage collection handle cleanup.
    """
    # safetensors.safe_open doesn't have close(), just clear references
    self.file_handle_map.clear()
```

**相关文件**:
- `kt-kernel/python/utils/loader.py` (第 166-173 行)

---

## 进度跟踪

| 日期 | 事项 | 状态 |
|------|------|------|
| 2026-01-15 | 发现 Bug #1: `kt_num_gpu_experts` 为 None 问题 | 临时方案可用，待代码修复 |
| 2026-01-15 | 发现 Bug #2: `kt_cpuinfer` 为 None 问题 | 临时方案可用，待代码修复 |
| 2026-01-15 | 发现 Bug #3: SFT 模式权重加载 API 不兼容 | ✅ 已解决 |
| 2026-01-15 | 新增 `BF16SafeTensorLoader` 到 kt-kernel | ✅ 完成 |
| 2026-01-15 | 设计 SFT 权重加载策略 (BF16/INT8/INT4) | ✅ 完成 |
| 2026-01-16 | 发现并修复 Bug #4: `safe_open.close()` 不存在 | ✅ 已解决 |

---

## 完整临时解决方案

需要添加以下参数才能正常启动 MoE LoRA 推理：
- `--kt-num-gpu-experts 0` (所有专家在 CPU 上运行)
- `--kt-cpuinfer 60` (CPU 推理线程数，根据实际 CPU 核心数调整)

**完整启动命令**:
```bash
python -m sglang.launch_server \
      --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
      --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
      --kt-moe-lora-path /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model_converted.safetensors \
      --kt-moe-lora-rank 8 \
      --kt-moe-lora-alpha 16.0 \
      --kt-moe-sft-method AMXBF16_SFT \
      --kt-num-gpu-experts 0 \
      --kt-cpuinfer 60 \
      --lora-paths /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model.safetensors
```

---

### Bug #5: "header too large" Error - 转换脚本输出格式错误

**状态**: 已解决 ✅

**发现日期**: 2026-01-16

**解决日期**: 2026-01-16

**错误信息**:
```
RuntimeError: Failed to load LoRA adapter /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/: Error while deserializing header: header too large
```

**触发命令**:
```bash
# 用户使用了 .safetensors 扩展名作为转换脚本输出
python scripts/convert_moe_lora.py \
    --input .../adapter_model.safetensors \
    --output .../adapter_model_converted.safetensors  # 错误！
```

**根本原因分析**:

1. **问题位置**: `convert_moe_lora.py` 输出文件 + sglang LoRA 加载器

2. **问题描述**:
   - `convert_moe_lora.py` 使用 `torch.save()` 保存为 **PyTorch `.pt` 格式**
   - 但用户指定了 `.safetensors` 扩展名作为输出
   - sglang 的 LoRA 加载器使用 `glob(*.safetensors)` 找到这个文件
   - 尝试用 safetensors 格式解析 PyTorch 文件，导致 "header too large" 错误

3. **问题流程**:
   ```
   convert_moe_lora.py
       torch.save(result, "output.safetensors")  # 实际是 PyTorch 格式
           ↓
   sglang LoRA 加载
       glob("*.safetensors")  # 找到这个文件
           ↓
   safetensors_weights_iterator()
       safe_open("output.safetensors")  # 尝试用 safetensors 格式解析
           ↓
   "Error: header too large"  # PyTorch 格式无法被 safetensors 解析
   ```

**解决方案**:

1. **删除错误文件**:
```bash
rm /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model_converted.safetensors
```

2. **使用正确的扩展名和路径重新转换**:
```bash
python scripts/convert_moe_lora.py \
    --input /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model.safetensors \
    --config /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_config.json \
    --output /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/moe_lora.pt
```

**注意事项**:
- 输出文件必须使用 `.pt` 扩展名
- 建议将输出文件放在 checkpoint 目录**外部**，避免被 sglang 的 `glob(*.safetensors)` 误识别

**相关文件**:
- `scripts/convert_moe_lora.py` (第 207 行: `torch.save()`)
- `python/sglang/srt/model_loader/loader.py` (第 434 行: `glob.glob()`)
- `python/sglang/srt/lora/lora.py` (第 90 行: `_get_weights_iterator`)

---

### Bug #6: SFT 模式 (MoE LoRA) CUDA Graph 设备不匹配

**状态**: ✅ 已解决

**发现日期**: 2026-01-16

**解决日期**: 2026-01-16

**错误信息**:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**触发命令**:
```bash
# 只有使用 --kt-moe-lora-path 参数时才会触发
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
    --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
    --kt-moe-lora-path /path/to/moe_lora.pt \  # 启用 SFT 模式
    --kt-moe-sft-method AMXBF16_SFT \
    --kt-num-gpu-experts 0 \
    --kt-cpuinfer 60 \
    --lora-paths /path/to/checkpoint-133
```

**重要澄清**:

**之前的分析有误！** kt-kernel 原先的推理模式（不含 MoE LoRA）是**可以兼容 CUDA graph** 的。

问题**只发生**在启用 `--kt-moe-lora-path` 参数时（SFT 模式），因为：
- 推理模式的 `sync_forward()` 返回 **GPU tensor**（在内部完成 CPU→GPU 拷贝）
- SFT 模式的 `forward_sft()` 返回 **CPU tensor**

**根本原因分析**:

1. **问题位置**: `kt_ep_wrapper.py:1476`
   ```python
   output = output + cpu_output
   # output: cuda:0 (GPU experts 计算结果)
   # cpu_output: cpu (SFT 模式下 forward_sft 返回 CPU tensor)
   ```

2. **核心差异**:
   ```python
   # kt_ep_wrapper.py 中 sync() 方法
   def sync(self, x, dispatch_output=None):
       if self.kt_config.moe_lora_enabled:  # --kt-moe-lora-path 设置时
           return self.wrapper.forward_sft(...)  # 返回 CPU tensor ❌
       else:
           return self.wrapper.sync_forward(...)  # 返回 GPU tensor ✓
   ```

3. **forward_sft 返回 CPU tensor**:
   ```python
   # amx_sft.py
   def forward_sft(...):
       return buffer.output_cpu.clone()  # CPU tensor!
   ```

**详细分析**: 见 `/home/lpl/sglang-debug/sglang/docs/kt_lora_chat/08_cuda_graph分析.md`

**解决方案**:

**已实现以下两个修复**：

1. **修复设备不匹配** (`kt_ep_wrapper.py`):
   ```python
   # apply() 方法中
   cpu_output = self.sync(x, dispatch_output)
   if cpu_output.device != output.device:
       cpu_output = cpu_output.to(output.device, non_blocking=True)
   output = output + cpu_output
   ```

2. **自动禁用 CUDA Graph** (`server_args.py`):
   ```python
   def _handle_kt_moe_lora(self):
       """Disable CUDA graph when kt-kernel MoE LoRA is enabled."""
       if self.kt_moe_lora_path is not None:
           if not self.disable_cuda_graph:
               logger.warning(
                   "CUDA graph is disabled because kt-kernel MoE LoRA is enabled. "
                   "SFT mode requires synchronous CPU computation."
               )
               self.disable_cuda_graph = True
   ```

**相关文件**:
- `python/sglang/srt/layers/moe/kt_ep_wrapper.py` (第 1476 行)
- `python/sglang/srt/server_args.py` (新增 `_handle_kt_moe_lora()` 方法)

---

## 进度跟踪

| 日期 | 事项 | 状态 |
|------|------|------|
| 2026-01-15 | 发现 Bug #1: `kt_num_gpu_experts` 为 None 问题 | 临时方案可用，待代码修复 |
| 2026-01-15 | 发现 Bug #2: `kt_cpuinfer` 为 None 问题 | 临时方案可用，待代码修复 |
| 2026-01-15 | 发现 Bug #3: SFT 模式权重加载 API 不兼容 | ✅ 已解决 |
| 2026-01-15 | 新增 `BF16SafeTensorLoader` 到 kt-kernel | ✅ 完成 |
| 2026-01-15 | 设计 SFT 权重加载策略 (BF16/INT8/INT4) | ✅ 完成 |
| 2026-01-16 | 发现并修复 Bug #4: `safe_open.close()` 不存在 | ✅ 已解决 |
| 2026-01-16 | 发现 Bug #5: 转换脚本输出格式错误 | ✅ 已解决 |
| 2026-01-16 | 发现并修复 Bug #6: SFT 模式 CUDA Graph 设备不匹配 | ✅ 已解决 |

---

## 完整启动命令

需要添加以下参数才能正常启动 MoE LoRA 推理：
- `--kt-num-gpu-experts 0` (所有专家在 CPU 上运行)
- `--kt-cpuinfer 60` (CPU 推理线程数，根据实际 CPU 核心数调整)
- 不需要手动添加 `--disable-cuda-graph`，系统会自动禁用

**完整启动命令**:
```bash
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
    --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
    --kt-moe-lora-path /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/moe_lora.pt \
    --kt-moe-lora-rank 8 \
    --kt-moe-lora-alpha 16.0 \
    --kt-moe-sft-method AMXBF16_SFT \
    --kt-num-gpu-experts 0 \
    --kt-cpuinfer 60 \
    --lora-paths /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133
```

---

### Bug #7: 新版本 MoE LoRA 输出乱码

**状态**: ✅ 已解决

**发现日期**: 2026-01-16

**解决日期**: 2026-01-16

**问题描述**:

- **旧版本**（只有 Attention + Shared Experts LoRA）: 输出正常
- **新版本**（加上 Routed Experts LoRA）: 输出乱码（如 "balenabalenabalena..."）

**根本原因 1**: `topk_ids` 被原地修改

`mask_cpu_expert_ids` 函数在 `kt_ep_wrapper.py:1062` **原地修改**了 `topk_ids`：

```python
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    topk_ids[topk_ids >= num_gpu_experts] = -1  # ← 原地修改！
    return topk_ids
```

**修复**: 添加 `.clone()` 避免原地修改 ✅ 已修复

**根本原因 2**: `hidden_states` 被 GPU 计算修改

在 `apply()` 方法中，`x = dispatch_output.hidden_states` 只是获取引用，而 `masked_dispatch_output.hidden_states` 和 `x` 是同一个引用。GPU 方法 `gpu_method.apply()` 可能原地修改了这个 tensor（用作输出缓冲区），导致 `sync()` 接收到的 `x` 已经是零。

**修复**: 在 SFT 模式下，GPU 计算前保存 `hidden_states` 的副本 ✅ 已修复

```python
if self.kt_config.moe_lora_enabled:
    x_for_cpu = x.clone()
else:
    x_for_cpu = x
```

**验证结果**:
- `x.abs().mean(): 0.255859` ✓ (非零输入)
- `topk_ids.min(): 0, topk_ids.max(): 60` ✓ (正确的专家 ID)
- 输出不再乱码 ✓

**相关文件**:
- `python/sglang/srt/layers/moe/kt_ep_wrapper.py:1062` (topk_ids 修复)
- `python/sglang/srt/layers/moe/kt_ep_wrapper.py:1464-1467` (hidden_states 修复)

---

### Bug #8: MoE LoRA 输出与基座模型完全一样

**状态**: 🔍 调试中（进入 C++ 层调试阶段）

**发现日期**: 2026-01-16

**问题描述**:

使用 `demo.py` 测试时，`DeepSeek-V2-Lite-Chat:lora0` 和 `DeepSeek-V2-Lite-Chat` 的输出**逐字相同**。这表明 MoE LoRA 虽然正确加载，但没有实际生效。

---

#### 调试阶段 1: Python 层验证 ✅ 完成

**验证项目**:

| 检查项 | 状态 | 说明 |
|--------|------|------|
| LoRA 权重形状 | ✅ | `[64, 8, 2048]` 等，与预期完全匹配 |
| LoRA 权重值 | ✅ | 非零，std≈0.004-0.005，正常分布 |
| `init_lora_weights()` 调用 | ✅ | 日志确认被调用 |
| `_weights_loaded` | ✅ | True |
| `moe is not None` | ✅ | True |
| `update_lora_weights()` 调用 | ✅ | 日志确认被调用 |

**日志证据**:
```
[DEBUG init_lora_weights] layer=1, _weights_loaded=True, moe=True
[MoE LoRA] Layer 1: update_lora_weights() called
```

**结论**: Python 层完全正确，问题在 C++ 后端。

---

#### 调试阶段 2: C++ 层分析 🔍 当前阶段

**重要说明**: 当前使用的是**非 TP 模式**，代码路径如下：

```
Python: AMXSFTMoEWrapper (amx_sft.py)
  → self.moe = AMXBF16_SFT_MOE(config)  # 绑定到 C++ AMX_SFT_MOE_TP

Python: update_lora_weights()
  → self.moe.update_lora_weights_task(gate_lora_a.data_ptr(), ...)
  → C++: AMX_SFT_MOE_TP::update_lora_weights()
    → 设置 gate_lora_a_, gate_lora_b_, etc. 指针
    → 设置 lora_weights_prepared_ = false

Python: forward_sft()
  → C++: AMX_SFT_MOE_TP::forward_sft()
    → 检查 if (gate_lora_a_ != nullptr && gate_lora_b_ != nullptr)
    → 调用 compute_lora_gate_up_amx() 或 compute_lora_gate_up()
      → prepare_lora_weights() (转换为 BufferB 格式)
      → 执行 LoRA 计算
```

**关键 C++ 代码** (`kt-kernel/operators/amx/sft_moe.hpp`):

```cpp
// Line 523-535: forward_sft() 中的 LoRA 分支
// Step 5.5: Gate + Up LoRA
if (gate_lora_a_ != nullptr && gate_lora_b_ != nullptr) {
    if constexpr (supports_standard_mat_mul_v<T>) {
        compute_lora_gate_up_amx(qlen, activated_expert);  // AMX-optimized path
    } else {
        compute_lora_gate_up(qlen, activated_expert);  // For-loop fallback
    }
}

// Line 688-702: update_lora_weights() 设置指针
void update_lora_weights(void* gate_lora_a, void* gate_lora_b, ...) {
    gate_lora_a_ = (ggml_bf16_t*)gate_lora_a;
    gate_lora_b_ = (ggml_bf16_t*)gate_lora_b;
    // ...
    lora_weights_prepared_ = false;  // 需要重新转换为 BufferB
}
```

**可能问题点**:
1. C++ `update_lora_weights()` 没有正确接收到 Python 传来的指针
2. `forward_sft()` 中 `gate_lora_a_` 指针为 nullptr（未被设置）
3. `prepare_lora_weights()` 转换失败
4. LoRA 计算逻辑有 bug

---

#### 下一步调试方案

**Step 1**: 在 C++ 层添加调试打印

已在以下位置添加调试代码：

1. `sft_moe.hpp:690-692` - `update_lora_weights()`:
```cpp
printf("[C++ AMX_SFT_MOE_TP::update_lora_weights] tp_part=%d, gate_lora_a=%p, gate_lora_b=%p\n",
       tp_part_idx, gate_lora_a, gate_lora_b);
```

2. `sft_moe.hpp:525-527` - `forward_sft()`:
```cpp
if (tp_part_idx == 0 && qlen > 0) {
    printf("[C++ forward_sft] tp_part=%d, qlen=%d, gate_lora_a_=%p, gate_lora_b_=%p\n",
           tp_part_idx, qlen, (void*)gate_lora_a_, (void*)gate_lora_b_);
}
```

**Step 2**: 重新编译 kt-kernel

```bash
cd /home/lpl/ktransformers-sglang/kt-kernel
pip install -e . --no-build-isolation
```

**Step 3**: 重启服务，观察日志

期望看到:
- `[C++ AMX_SFT_MOE_TP::update_lora_weights]` - 确认 C++ 层收到 LoRA 指针
- `[C++ forward_sft]` - 确认 forward 时 LoRA 指针状态

**预期结果分析**:
- 如果 `gate_lora_a` 为 `0x0`（nullptr）：问题在 Python→C++ 调用链
- 如果 `gate_lora_a` 非空但 LoRA 仍无效：问题在 C++ 计算逻辑

---

**相关文件**:
- `kt-kernel/operators/amx/sft_moe.hpp` (C++ LoRA 实现，非 TP 模式)
- `kt-kernel/python/utils/amx_sft.py` (Python wrapper)
- `python/sglang/srt/layers/moe/kt_ep_wrapper.py` (sglang 集成)

**注意**: 修改 C++ `.hpp` 文件后必须重新编译 kt-kernel 才能生效

---

## 进度跟踪

| 日期 | 事项 | 状态 |
|------|------|------|
| 2026-01-15 | 发现 Bug #1: `kt_num_gpu_experts` 为 None 问题 | 临时方案可用，待代码修复 |
| 2026-01-15 | 发现 Bug #2: `kt_cpuinfer` 为 None 问题 | 临时方案可用，待代码修复 |
| 2026-01-15 | 发现 Bug #3: SFT 模式权重加载 API 不兼容 | ✅ 已解决 |
| 2026-01-15 | 新增 `BF16SafeTensorLoader` 到 kt-kernel | ✅ 完成 |
| 2026-01-15 | 设计 SFT 权重加载策略 (BF16/INT8/INT4) | ✅ 完成 |
| 2026-01-16 | 发现并修复 Bug #4: `safe_open.close()` 不存在 | ✅ 已解决 |
| 2026-01-16 | 发现 Bug #5: 转换脚本输出格式错误 | ✅ 已解决 |
| 2026-01-16 | 发现并修复 Bug #6: SFT 模式 CUDA Graph 设备不匹配 | ✅ 已解决 |
| 2026-01-16 | Bug #7: 新版本输出乱码 (两处原地修改问题) | ✅ 已解决 |
| 2026-01-16 | Bug #8: MoE LoRA 输出与基座模型一样 | 🔍 调试中 |
| 2026-01-16 | 删除 kt_ep_wrapper.py 调试打印（优化性能） | ✅ 完成 |
| 2026-01-16 | 添加 amx_sft.py 精简调试输出 | ✅ 完成 |

---

## 完整启动命令

需要添加以下参数才能正常启动 MoE LoRA 推理：
- `--kt-num-gpu-experts 0` (所有专家在 CPU 上运行)
- `--kt-cpuinfer 60` (CPU 推理线程数，根据实际 CPU 核心数调整)
- 不需要手动添加 `--disable-cuda-graph`，系统会自动禁用

**完整启动命令**:
```bash
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
    --model-path /mnt/data3/models/DeepSeek-V2-Lite-Chat \
    --kt-weight-path /mnt/data3/models/DeepSeek-V2-Lite-Chat-CPU-weight-INT8 \
    --kt-moe-lora-path /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/moe_lora.pt \
    --kt-moe-lora-rank 8 \
    --kt-moe-lora-alpha 16.0 \
    --kt-moe-sft-method AMXBF16_SFT \
    --kt-num-gpu-experts 0 \
    --kt-cpuinfer 60 \
    --lora-paths /mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133
```

---

## 备注

- 转换脚本 `convert_moe_lora.py` 输出必须使用 `.pt` 扩展名
- 当前测试环境：DeepSeek-V2-Lite-Chat 模型
- LoRA 配置：rank=8, alpha=16.0
- **重要澄清**：kt-kernel 标准推理模式可以兼容 CUDA graph，只有 SFT 模式 (MoE LoRA) 需要禁用
- 当使用 `--kt-moe-lora-path` 时，CUDA graph 会自动禁用

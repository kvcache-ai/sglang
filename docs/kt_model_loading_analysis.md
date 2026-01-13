# SGLang 模型加载流水线分析：`--kt-num-gpu-experts` 如何控制 GPU/CPU 加载

## 概述

本文档详细分析 SGLang 中模型加载流水线，重点是 `--kt-num-gpu-experts` 参数如何控制 MoE 模型的 Expert 在 GPU 或 CPU 上的加载与执行。

---

## 1. KT 参数定义

### 1.1 ServerArgs 字段定义

**文件：** `python/sglang/srt/server_args.py:491-498`

```python
# Ktransformers/AMX expert parallelism
kt_weight_path: Optional[str] = None
kt_method: Optional[str] = None
kt_cpuinfer: Optional[int] = None
kt_threadpool_count: Optional[int] = None
kt_num_gpu_experts: Optional[int] = None
kt_max_deferred_experts_per_token: Optional[int] = None
kt_gpu_prefill_token_threshold: Optional[int] = None
```

### 1.2 命令行参数定义

**文件：** `python/sglang/srt/server_args.py:3756-3760`

```python
parser.add_argument(
    "--kt-num-gpu-experts",
    type=int,
    help="[ktransformers parameter] The number of GPU experts.",
)
```

---

## 2. 标准模型加载流水线（无 KT）

### 2.1 整体流程

```
Engine.__init__
    └── _launch_subprocesses()
        └── Scheduler.__init__()
            └── TpModelWorker.__init__()
                └── ModelRunner.__init__()
                    └── load_model()
                        ├── get_model_loader()
                        └── loader.load_model()
                            ├── _initialize_model()
                            ├── _get_all_weights()
                            └── load_weights_and_postprocess()
```

### 2.2 ModelRunner.load_model() 入口

**文件：** `python/sglang/srt/model_executor/model_runner.py:853`

```python
def load_model(self):
    before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
    logger.info(
        f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
    )
    # ... 配置准备 ...
```

**文件：** `python/sglang/srt/model_executor/model_runner.py:933-940`

```python
self.loader = get_model_loader(
    load_config=self.load_config,
    model_config=self.model_config,
)
self.model = self.loader.load_model(
    model_config=self.model_config,
    device_config=DeviceConfig(self.device, self.gpu_id),
)
```

### 2.3 DefaultModelLoader.load_model()

**文件：** `python/sglang/srt/model_loader/loader.py:608-634`

```python
def load_model(
    self,
    *,
    model_config: ModelConfig,
    device_config: DeviceConfig,
) -> nn.Module:

    target_device = torch.device(device_config.device)
    with set_default_torch_dtype(model_config.dtype):
        with target_device:
            # Step 1: 初始化模型架构（无权重）
            model = _initialize_model(
                model_config,
                self.load_config,
            )

        # Step 2: 加载权重并后处理
        self.load_weights_and_postprocess(
            model, self._get_all_weights(model_config, model), target_device
        )

    return model.eval()
```

### 2.4 load_weights_and_postprocess()

**文件：** `python/sglang/srt/model_loader/loader.py:636-651`

```python
@staticmethod
def load_weights_and_postprocess(model, weights, target_device):
    # Step 3: 调用模型的 load_weights 方法
    model.load_weights(weights)

    # Step 4: 量化后处理
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
```

---

## 3. KT 集成：FusedMoE 层初始化

### 3.1 检测 KT 配置并包装

**文件：** `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:239-254`

```python
self.quant_method: Optional[FusedMoEMethodBase] = None
server_args = get_global_server_args()
kt_config = create_kt_config_from_server_args(server_args, layer_id)
if kt_config is not None:
    # KT 模式：用 KTEPWrapperMethod 包装 GPU 方法
    if quant_config is not None:
        gpu_method = quant_config.get_quant_method(self, prefix)
    else:
        gpu_method = UnquantizedFusedMoEMethod(self.use_triton_kernels)
    self.quant_method = KTEPWrapperMethod(gpu_method, kt_config)
else:
    # 标准模式
    if quant_config is not None:
        self.quant_method = quant_config.get_quant_method(self, prefix)
    if self.quant_method is None:
        self.quant_method = UnquantizedFusedMoEMethod(
            self.use_triton_kernels, self.use_flashinfer_trtllm_moe
        )
```

### 3.2 创建 KTConfig

**文件：** `python/sglang/srt/layers/moe/kt_ep_wrapper.py:658-688`

```python
def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured."""
    if server_args.kt_weight_path is None:
        return None

    # Get num_layers from model config
    hf_config = server_args.get_hf_config()
    num_layers = getattr(hf_config, "num_hidden_layers", None)

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=server_args.kt_num_gpu_experts,  # ← 核心参数
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        gpu_prefill_token_threshold=server_args.kt_gpu_prefill_token_threshold,
    )
```

---

## 4. GPU/CPU 权重加载控制（核心）

### 4.1 权重加载跳过逻辑

**文件：** `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:600-623`

```python
def weight_loader(
    self,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
) -> None:
    # 1. 全局 Expert ID → 本地 Expert ID 映射 (EP 并行)
    if not getattr(param, "_sglang_require_global_experts", False):
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id < 0 or expert_id >= self.num_local_experts:
            return  # 不属于当前 rank，跳过

    # 2. KT 模式：跳过 CPU experts 的 GPU 权重加载
    if isinstance(
        self.quant_method,
        KTEPWrapperMethod,
    ):
        if self.quant_method.num_gpu_experts != -1:
            if expert_id >= self.quant_method.num_gpu_experts:
                return  # ← 核心控制点！跳过 CPU experts

    # 3. 实际加载权重到 GPU
    self._weight_loader_impl(
        param=param,
        loaded_weight=loaded_weight,
        weight_name=weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
    )
```

**关键逻辑：**
- `expert_id < num_gpu_experts` → 加载到 GPU
- `expert_id >= num_gpu_experts` → 跳过（由 KT 从 `kt_weight_path` 加载到 CPU）

### 4.2 KTEPWrapperMethod 初始化

**文件：** `python/sglang/srt/layers/moe/kt_ep_wrapper.py:729-754`

```python
def __init__(
    self,
    gpu_method: FusedMoEMethodBase,
    kt_config: KTConfig,
):
    """Initialize the KT EP wrapper."""
    if not KTRANSFORMERS_AVAILABLE:
        raise ImportError(
            "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
        )

    self.gpu_method = gpu_method
    self.kt_config = kt_config
    self.num_gpu_experts = kt_config.num_gpu_experts  # ← 存储 GPU expert 数量
    self.override_num_local_experts = True
    self.gpu_method.num_gpu_experts = self.num_gpu_experts
    self.tp_rank = get_tensor_model_parallel_rank()

    self.gpu_prefill_token_threshold = kt_config.gpu_prefill_token_threshold or 0
    self._full_init_args = None
    self.wrapper: Optional[KTMoEWrapper] = None
```

### 4.3 创建 GPU 和 CPU Expert 权重

**文件：** `python/sglang/srt/layers/moe/kt_ep_wrapper.py:799-826`

```python
# 1. Create weights for GPU experts using the wrapped method
# GPU experts: 0 to num_gpu_experts-1
self.gpu_method.create_weights(
    layer=layer,
    num_experts=self.num_gpu_experts,  # ← 只创建 GPU experts 的权重
    hidden_size=hidden_size,
    intermediate_size_per_partition=intermediate_size_per_partition,
    params_dtype=params_dtype,
    **extra_weight_attrs,
)

# 2. Initialize KT wrapper for CPU experts
# CPU experts: num_gpu_experts to num_experts-1
if self.tp_rank == 0:
    self.wrapper = KTMoEWrapper(
        layer_idx=self.kt_config.layer_idx,
        num_experts=num_experts,                    # 总 expert 数
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size_full,
        num_gpu_experts=self.num_gpu_experts,       # GPU expert 数
        cpuinfer_threads=self.kt_config.cpuinfer_threads,
        threadpool_count=self.kt_config.threadpool_count,
        weight_path=self.kt_config.weight_path,     # CPU 权重路径
        chunked_prefill_size=self.kt_config.chunked_prefill_size,
        method=self.kt_config.method,
        max_deferred_experts_per_token=layer_max_deferred,
    )
```

### 4.4 CPU 权重加载

**文件：** `python/sglang/srt/layers/moe/kt_ep_wrapper.py:828-852`

```python
def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Process weights after loading from checkpoint."""
    # 1. Process GPU weights
    if hasattr(self.gpu_method, "process_weights_after_loading"):
        self.gpu_method.process_weights_after_loading(layer)

    # 2. Load CPU weights using KT wrapper
    if self.tp_rank == 0 and self.wrapper is not None:
        torch.cuda.synchronize()

        # Get expert location metadata for CPU expert mapping
        from sglang.srt.eplb.expert_location_dispatch import (
            get_global_expert_location_metadata,
        )

        physical_to_logical_map_cpu = (
            get_global_expert_location_metadata()
            .physical_to_logical_map_cpu[self.kt_config.layer_idx]
            .contiguous()
        )
        # 从 kt_weight_path 加载 CPU experts 权重
        self.wrapper.load_weights(physical_to_logical_map_cpu)
```

---

## 5. 前向传播时的 GPU/CPU 分流

### 5.1 Expert ID 掩码函数

**文件：** `python/sglang/srt/layers/moe/kt_ep_wrapper.py:691-707`

```python
@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    topk_ids[topk_ids >= num_gpu_experts] = -1
    return topk_ids
```

### 5.2 混合 CPU+GPU 前向传播

**文件：** `python/sglang/srt/layers/moe/kt_ep_wrapper.py:918-988`

```python
def apply(
    self,
    layer: torch.nn.Module,
    dispatch_output: "StandardDispatchOutput",
) -> "CombineInput":
    """Execute hybrid CPU+GPU MoE forward pass with parallelism.

    This is the main computation method that coordinates:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize CPU results and merge with GPU results
    """
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    num_tokens = int(x.shape[0]) if x.dim() > 0 else 0

    # Check for full GPU fallback (大 prefill 时回退到全 GPU)
    if (
        self.gpu_prefill_token_threshold > 0
        and num_tokens >= self.gpu_prefill_token_threshold
    ):
        ctx = self._build_full_context(layer)
        result = ctx.gpu_method.apply(ctx.gpu_layer, dispatch_output)
        return result

    # Step 1: Submit CPU expert computation (non-blocking)
    if self.tp_rank == 0:
        self.submit(layer, dispatch_output)

    # Step 2: Prepare GPU computation by masking CPU expert IDs
    # CPU expert IDs (>= num_gpu_experts) are set to -1 so GPU kernel skips them
    topk_ids = topk_output.topk_ids
    masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

    # Create modified dispatch output for GPU computation
    masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
    masked_dispatch_output = dispatch_output._replace(
        topk_output=masked_topk_output
    )

    # Step 3: Execute GPU expert computation (any quantization method)
    # This runs in parallel with CPU computation
    gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)

    # Step 4: Synchronize CPU results and merge with GPU results
    output = gpu_combine_input.hidden_states
    if self.tp_rank == 0:
        cpu_output = self.sync(x)
        output = output + cpu_output  # 合并 GPU 和 CPU 结果

    return StandardCombineInput(hidden_states=output)
```

---

## 6. 数据流图

### 6.1 Expert 分布

```
Expert IDs:  0   1   2   3   4   5   6   7   ...  N-1
             |<-- GPU Experts -->|<-- CPU Experts -->|
             |   num_gpu_experts |
```

### 6.2 权重加载流程对比

```
┌──────────────────────────────────────────────────────────────────┐
│                     标准加载 (无 KT)                               │
├──────────────────────────────────────────────────────────────────┤
│  safetensors/pt 文件                                              │
│         ↓                                                        │
│  DefaultModelLoader._get_all_weights()                            │
│         ↓                                                        │
│  model.load_weights(weights)                                      │
│         ↓                                                        │
│  FusedMoE.weight_loader(expert_id)                               │
│         ↓                                                        │
│  GPU tensor[expert_id] ← 所有 expert 都加载到 GPU                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        KT 加载                                    │
├──────────────────────────────────────────────────────────────────┤
│  【Phase 1: GPU Experts 加载】                                    │
│  safetensors/pt 文件                                              │
│         ↓                                                        │
│  model.load_weights(weights)                                      │
│         ↓                                                        │
│  FusedMoE.weight_loader(expert_id)                               │
│         ↓                                                        │
│  if expert_id < num_gpu_experts:                                 │
│      GPU tensor[expert_id] ← 加载                                │
│  else:                                                            │
│      return  ← 跳过                                               │
│                                                                   │
│  【Phase 2: CPU Experts 加载】                                    │
│  kt_weight_path                                                   │
│         ↓                                                        │
│  KTEPWrapperMethod.process_weights_after_loading()               │
│         ↓                                                        │
│  KTMoEWrapper.load_weights(physical_to_logical_map)              │
│         ↓                                                        │
│  CPU (AMX/AVX 格式) ← experts [num_gpu_experts, total-1]          │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 前向传播流程

```
┌──────────────────────────────────────────────────────────────────┐
│                        KT 前向传播                                │
├──────────────────────────────────────────────────────────────────┤
│  tokens                                                          │
│    ↓                                                             │
│  Router → top-k expert IDs                                       │
│    ↓                                                             │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Step 1: submit()                                        │     │
│  │   提交 CPU expert 计算 (非阻塞)                          │     │
│  │   wrapper.submit_forward(x, topk_ids, topk_weights)     │     │
│  └─────────────────────────────────────────────────────────┘     │
│    ↓                                                             │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Step 2: mask_cpu_expert_ids()                           │     │
│  │   topk_ids[topk_ids >= num_gpu_experts] = -1            │     │
│  │   GPU kernel 会跳过 ID=-1 的 expert                      │     │
│  └─────────────────────────────────────────────────────────┘     │
│    ↓                                                             │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Step 3: GPU expert 计算                                 │     │
│  │   gpu_method.apply(layer, masked_dispatch_output)       │     │
│  │   与 CPU 计算并行执行                                    │     │
│  └─────────────────────────────────────────────────────────┘     │
│    ↓                                                             │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ Step 4: sync() + merge                                  │     │
│  │   cpu_output = wrapper.sync_forward()                   │     │
│  │   output = gpu_output + cpu_output                      │     │
│  └─────────────────────────────────────────────────────────┘     │
│    ↓                                                             │
│  final output                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| `kt_num_gpu_experts` 字段定义 | `server_args.py` | 496 |
| `--kt-num-gpu-experts` 参数定义 | `server_args.py` | 3756-3760 |
| `create_kt_config_from_server_args()` | `kt_ep_wrapper.py` | 658-688 |
| FusedMoE 层检测 KT 配置 | `fused_moe_triton/layer.py` | 240-247 |
| **GPU 权重跳过逻辑** | `fused_moe_triton/layer.py` | 609-615 |
| `KTEPWrapperMethod.__init__()` | `kt_ep_wrapper.py` | 729-754 |
| GPU/CPU expert 权重创建 | `kt_ep_wrapper.py` | 799-826 |
| CPU 权重加载 | `kt_ep_wrapper.py` | 828-852 |
| `mask_cpu_expert_ids()` | `kt_ep_wrapper.py` | 691-707 |
| **混合前向传播 `apply()`** | `kt_ep_wrapper.py` | 918-988 |
| `ModelRunner.load_model()` | `model_runner.py` | 853 |
| `get_model_loader()` 调用 | `model_runner.py` | 933-940 |
| `DefaultModelLoader.load_model()` | `loader.py` | 608-634 |
| `load_weights_and_postprocess()` | `loader.py` | 636-651 |

---

## 8. 总结

`--kt-num-gpu-experts` 参数是 KTransformers 混合计算的核心控制点：

1. **定义分界线**：决定哪些 experts 在 GPU（ID 0 到 N-1），哪些在 CPU（ID N 到 total-1）

2. **控制权重加载**：
   - `fused_moe_triton/layer.py:609-615` 中，`expert_id >= num_gpu_experts` 时跳过 GPU 权重加载
   - CPU experts 权重从 `kt_weight_path` 加载

3. **控制前向计算**：
   - `mask_cpu_expert_ids()` 将 CPU expert IDs 设为 -1，使 GPU kernel 跳过
   - CPU 和 GPU 计算并行执行，最后合并结果

4. **影响内存使用**：减少 GPU 显存占用，利用 CPU 内存扩展模型规模

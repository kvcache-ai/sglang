# KVCache Sparsity Engine (KSE)

KSE is a **sidecar sparsity module** for sglang that provides pluggable KV Cache sparsity for models without native sparse attention.

Its core principle is **Metadata Rewriting**: instead of moving or copying any KV data, KSE rewrites the attention backend's index structures (`page_table`, `kv_indices`, `seq_lens`) just before each decode attention call, so the attention kernel only sees the selected KV entries. This design provides:

- **Zero-copy**: no KV Cache data is moved
- **Backend-agnostic**: a `MetadataAdapter` abstraction handles each attention backend
- **CUDA Graph compatible**: only tensor *contents* are modified, not tensor shapes

---

## Directory Structure

```
python/sglang/srt/layers/kse/
│
├── __init__.py              # Public API exports; auto-registers built-in policies and adapters
├── types.py                 # Core types: Granularity, Frequency, SelectionResult
├── config.py                # KSEConfig dataclass
├── base_policy.py           # Abstract base classes: SparsityPolicy, EvictionPolicy
├── base_adapter.py          # Abstract base class: MetadataAdapter
├── registry.py              # Registry + create_kse_controller factory
├── controller.py            # KSEController: central coordinator with lifecycle hooks
│
├── policies/
│   ├── __init__.py          # Imports all built-in policies (triggers registration)
│   ├── quest.py             # Quest: Query-Aware, Page granularity, Per-Layer
│   └── streaming_llm.py     # StreamingLLM: Fixed pattern, Token granularity, with eviction
│
└── adapters/
    ├── __init__.py          # Imports all built-in adapters (triggers registration)
    ├── triton_adapter.py    # Triton backend (kv_indptr / kv_indices)
    ├── flashattention_adapter.py  # FlashAttention backend (page_table / cache_seqlens)
    └── flashinfer_adapter.py      # FlashInfer backend (in-place buffer overwrite)
```

---

## Quick Start: Enabling KSE via CLI

KSE is activated by passing `--enable-kse` when launching the sglang server. All other `--kse-*` flags are optional and have sensible defaults.

The KSE backend is derived automatically from `--attention-backend`, so you never need to configure it separately.

### Quest (Query-Aware Page Sparsity)

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend triton \
    --enable-kse \
    --kse-policy quest \
    --kse-page-size 64 \
    --kse-min-seq-len 2048 \
    --kse-token-budget-ratio 0.3 \
    --kse-num-recent-pages 4
```

### StreamingLLM (Sink + Sliding Window)

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend triton \
    --enable-kse \
    --kse-policy streaming_llm \
    --kse-min-seq-len 2048 \
    --kse-num-sink-tokens 4 \
    --kse-window-size 1024
```

### Quest with FlashAttention backend

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend flashattention \
    --enable-kse \
    --kse-policy quest \
    --kse-page-size 256 \
    --kse-token-budget-ratio 0.3
```

> **Note**: When using FlashAttention, `--kse-page-size` must be ≥ the backend's internal page size (256) and an integer multiple of it.

---

## All KSE CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-kse` | `False` | Enable the KVCache Sparsity Engine |
| `--kse-policy` | `quest` | Sparsity policy: `quest` or `streaming_llm` |
| `--kse-page-size` | `64` | Sparse page size in tokens (page-granularity policies only) |
| `--kse-min-seq-len` | `2048` | Minimum sequence length before KSE activates |
| `--kse-start-layer` | `0` | First layer index (inclusive) to apply KSE |
| `--kse-end-layer` | `-1` | Last layer index (exclusive); `-1` means all layers |
| `--kse-token-budget-ratio` | `0.3` | *(Quest)* Fraction of pages to retain per layer |
| `--kse-num-recent-pages` | `4` | *(Quest)* Recent pages always included regardless of score |
| `--kse-num-sink-tokens` | `4` | *(StreamingLLM)* Number of initial sink tokens to keep |
| `--kse-window-size` | `1024` | *(StreamingLLM)* Sliding window size in tokens |

---

## Core Components

### `KSEConfig`

```python
@dataclass
class KSEConfig:
    policy_name: str       # e.g. "quest", "streaming_llm"
    backend_name: str      # auto-derived from attention_backend; e.g. "triton", "flashattention", "flashinfer"
    start_layer: int = 0
    end_layer: int = -1    # exclusive; -1 → all layers
    min_seq_len: int = 2048
    page_size: int = 64    # sparse page size in tokens
    policy_kwargs: dict    # policy-specific parameters
```

### `SparsityPolicy` (Abstract Base)

```
SparsityPolicy (ABC)
   granularity() → Granularity          # TOKEN or PAGE
   frequency()   → Frequency            # PER_REQUEST / PER_STEP / PER_LAYER
   select(query, layer_id, ...) → SelectionResult
   on_prefill_complete(layer_id, ...)   # optional: build representations after prefill
   on_attention_complete(layer_id, ...) # optional: incremental update after decode
```

### `MetadataAdapter` (Abstract Base)

```
MetadataAdapter (ABC)
   save_dense_metadata(forward_metadata)   # snapshot original metadata
   apply(result, forward_metadata, ...)    # rewrite metadata with sparse selection
   restore_dense_metadata(forward_metadata)
```

### `KSEController`

The central coordinator. It is wired into 4 call-sites in the sglang forward path:

| sglang call-site | KSE hook |
|---|---|
| After extend (prefill) forward in `ModelRunner._forward_raw` | `controller.after_prefill(batch)` |
| Start of `ModelRunner.forward_decode` | `controller.before_forward(batch)` |
| Inside `AttentionBackend.forward` before decode kernel | `controller.before_attention(q, layer_id, batch, metadata)` |
| Inside `AttentionBackend.forward` after decode kernel | `controller.after_attention(layer_id, batch)` |

**Selection frequency caching**:
- `PER_REQUEST`: `select()` is called once after prefill; result is reused for the entire request
- `PER_STEP`: `select()` is called once per decode step; result is shared across all layers in that step
- `PER_LAYER`: `select()` is called independently for every layer

**Granularity compatibility rules** (validated at init):

| Policy granularity | Backend | Result |
|---|---|---|
| TOKEN | FlashInfer / Triton | ✅ allowed |
| PAGE | FlashInfer / Triton | ✅ allowed (pages expanded to tokens) |
| TOKEN | FlashAttention | ❌ error |
| PAGE | FlashAttention | ✅ allowed; `page_size` must be ≥ backend page size and an integer multiple |

---

## Built-in Policies

### Quest (`--kse-policy quest`)

**Paper**: Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*, ICML 2024

**How it works**: Maintains per-page key bounding boxes (min/max) for each physical KV page. At decode time, uses the current query to compute an upper-bound attention score for each page via the bounding box, then selects the top-scoring pages plus a fixed number of recent pages.

| Property | Value |
|---|---|
| Granularity | PAGE |
| Frequency | PER_LAYER |
| Eviction | No (KV data is retained; only visibility changes) |

**Policy-specific parameters** (via `--kse-token-budget-ratio`, `--kse-num-recent-pages`):

| Parameter | Default | Description |
|---|---|---|
| `token_budget_ratio` | `0.3` | Fraction of pages to select per layer |
| `num_recent_pages` | `4` | Recent pages always included |

---

### StreamingLLM (`--kse-policy streaming_llm`)

**Paper**: Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024

**How it works**: Retains a fixed number of "sink tokens" (initial tokens that absorb attention mass) and a sliding window of the most recent tokens. As new tokens arrive during decode, the window slides forward. Tokens outside the sink+window range are **masked out** via `select()` every decode step — the attention kernel never sees them.

| Property | Value |
|---|---|
| Granularity | TOKEN |
| Frequency | PER_STEP |
| Eviction | No (masking only; KV slots freed when request finishes) |

**Design**: `select()` produces a mask each step covering only sink + recent window tokens. No physical eviction is performed — the masked-out KV slots remain allocated but invisible to attention. This avoids complex interactions with the scheduler's `kv_committed_len` bookkeeping. The compute savings (sparse attention) are the primary benefit.

**Policy-specific parameters** (via `--kse-num-sink-tokens`, `--kse-window-size`):

| Parameter | Default | Description |
|---|---|---|
| `num_sink_tokens` | `4` | Number of initial sink tokens to keep |
| `window_size` | `1024` | Sliding window size (most recent tokens to keep) |

---

## Built-in Adapters

| `--kse-backend` | Class | Rewrites | Supported granularity |
|---|---|---|---|
| `triton` | `TritonAdapter` | `kv_indptr` / `kv_indices` (CSR, replaced) | TOKEN and PAGE |
| `flashattention` | `FlashAttentionAdapter` | `page_table` / `cache_seqlens_int32` (in-place) | PAGE only |
| `flashinfer` | `FlashInferAdapter` | `_kse_kv_indptr` / `_kse_kv_indices` (in-place overwrite) | TOKEN and PAGE |

---

## Data Flow

```
Prefill phase
    ModelRunner._forward_raw  (is_extend)
        └─► controller.after_prefill(batch)
                ├─ policy.on_prefill_complete(layer_id, ...)  [per layer]
                └─ policy.select(...)                         [PER_REQUEST policies]

Decode phase  (per step)
    ModelRunner.forward_decode
        └─► controller.before_forward(batch)
                └─ clear PER_STEP cache

    Per-layer attention  (AttentionBackend.forward)
        ├─► controller.before_attention(q, layer_id, ...)
        │       ├─ adapter.save_dense_metadata(...)           [first layer only]
        │       ├─ policy.select(...)                         [PER_STEP / PER_LAYER]
        │       └─ adapter.apply(result, metadata, ...)       [rewrite metadata]
        │
        ├─  attention_kernel(q, sparse_metadata)
        │
        └─► controller.after_attention(layer_id, ...)
                └─ policy.on_attention_complete(...)          [e.g. update bbox]
```

---

## Programmatic Usage

If you need to create a `KSEController` directly (e.g., for testing or custom integration):

```python
import torch
from sglang.srt.layers.kse import KSEConfig, create_kse_controller

config = KSEConfig(
    policy_name="quest",
    backend_name="triton",
    page_size=64,
    min_seq_len=2048,
    policy_kwargs={
        "token_budget_ratio": 0.3,
        "num_recent_pages": 4,
    },
)

controller = create_kse_controller(
    config=config,
    req_to_token_pool=model_runner.req_to_token_pool,
    token_to_kv_pool=model_runner.token_to_kv_pool,
    device=torch.device("cuda"),
)
```

### Registering a Custom Policy

```python
from sglang.srt.layers.kse import register_policy
from sglang.srt.layers.kse.base_policy import SparsityPolicy
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

@register_policy("my_policy")
class MyPolicy(SparsityPolicy):
    def __init__(self, config, device): ...
    def granularity(self) -> Granularity: return Granularity.TOKEN
    def frequency(self) -> Frequency: return Frequency.PER_STEP
    def select(self, query, layer_id, req_pool_indices,
               seq_lens, forward_batch, **kwargs) -> SelectionResult: ...
```

Once registered, use it with `--kse-policy my_policy`.

---

## Tests

Unit tests are in `test/srt/kse/`. All tests run on CPU without a GPU or sglang server:

```bash
# Run all KSE tests from the repo root
python test/srt/kse/run_all_kse_tests.py

# Run individual test files
python test/srt/kse/test_quest_policy.py
python test/srt/kse/test_streaming_llm_policy.py
python test/srt/kse/test_triton_adapter.py
python test/srt/kse/test_flashattention_adapter.py
```

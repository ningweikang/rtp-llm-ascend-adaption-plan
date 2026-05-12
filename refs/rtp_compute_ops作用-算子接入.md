# rtp_compute_ops 作用说明

> **Workspace:** c:\WorkingSpace\Code\rtp-llm

---

## User
rtp_compute_ops作用是什么

## Assistant
## `librtp_compute_ops` 的作用

`librtp_compute_ops` 是 rtp-llm 项目中**核心的高性能计算算子库**，通过 C++/CUDA 实现并经 pybind11 绑定到 Python。它是整个推理引擎的底层计算加速层，主要提供以下几大类功能：

---

### 1. 注意力机制（Attention）算子
- **FlashInferDecodeOp / FlashInferPrefillOp** — 基于 FlashInfer 的 decode/prefill 注意力实现
- **FlashInferMlaAttnParams / SparseMlaParams** — 支持 MLA（Multi-Head Latent Attention，DeepSeek V3 风格）的注意力参数
- **XQAAttnOp** — XQA 注意力实现
- **TRTAttnOp / TRTPagedAttnOp** — 基于 TensorRT 的注意力实现
- 各 Op 都有 `forward()`、`prepare()`、`support()` 方法，支持 CUDA Graph 捕获

### 2. 归一化（Normalization）算子
- `rmsnorm` — RMS 归一化
- `layernorm` — Layer 归一化
- `fused_add_rmsnorm` — 融合 Residual Add + RMSNorm
- `fused_add_layernorm` — 融合 Residual Add + LayerNorm
- `fused_qk_rmsnorm` — 融合 QK RMSNorm

### 3. 量化（Quantization）算子
- `per_tensor_quant_fp8` — 逐张量 FP8 量化
- `per_token_quant_fp8` — 逐 token FP8 量化
- `per_token_group_quant_fp8/v2` — 逐 token 组 FP8 量化
- `per_token_group_quant_int8` — 逐 token 组 INT8 量化
- `trt_fp8_quantize_128` — 基于 128 元素块的 FP8 量化
- `scaled_fp4_quant / scaled_fp4_experts_quant` — FP4 量化（SM100+）
- `cutlass_scaled_fp4_mm` — FP4 矩阵乘法

### 4. MoE（Mixture of Experts）算子
- `FusedMoEOp` — 融合 MoE 前向
- `SelectTopkOp / GroupTopKOp` — TopK 路由选择
- `moe_pre_reorder / moe_post_reorder` — MoE Expert Parallelism 的 permute/unpermute
- `moe_topk_softmax` — MoE TopK Softmax
- `fast_topk_v2 / fast_topk_transform_fused` — 快速 TopK 变换

### 5. KV Cache 管理算子
- `KVCache / LayerKVCache` — KV Cache 数据结构
- `concat_and_cache_mla` — MLA KV Cache 拼接写入
- `indexer_k_quant_and_cache` — K 量化缓存索引
- `reuse_kv_cache_indexed_batched` — KV Cache 复用
- `cp_gather_and_upconvert_fp8_kv_cache` — FP8 KV Cache 反量化收集
- `write_cache_store` — Cache Store 写入

### 6. 通信与分布式算子
- `TrtllmArFusionHandle` — AllReduce 融合句柄（含 IPC 共享内存管理）
- `allreduce / allreduce_rms` — 融合 AllReduce + RMSNorm
- `userbuffers_send / userbuffers_recv / userbuffers_ring_all_gather` — 基于用户态缓冲区的通信原语
- `init_communicator / dispose_communicator / register_buffer_to_communicator` — 通信器管理

### 7. 其他算子
- `embedding / embedding_bert` — Embedding 查找
- `silu_and_mul` — SiLU 激活 + 逐元素乘
- `silu_and_mul_scaled_fp4_experts_quant` — 融合 SiLU + FP4 量化
- `mla_k_merge` — MLA K 融并
- `cuda_graph_copy_small2large / cuda_graph_copy_large2small` — CUDA Graph 张量拷贝
- `allocate_shared_buffer / open_ipc_handle` — 跨进程共享内存

### 8. 运行时上下文与数据结构
- `init_exec_ctx / get_exec_ctx` — 初始化/获取执行上下文
- `PyAttentionInputs / PyModelInputs / PyModelOutputs` — 推理输入输出数据结构
- `DeviceType` — 设备类型枚举（CUDA、ROCm、CPU、ARM、Yitian、PPU）

---

### 架构关系

```
Python 层 (rtp_llm/ops/compute_ops.py)
    ↓ from librtp_compute_ops import *
C++ pybind11 模块 (rtp_llm/cpp/pybind/ComputeInit.cc)
    ↓ PYBIND11_MODULE(librtp_compute_ops, m)
CUDA/C++ 内核实现 (rtp_llm/models_py/bindings/cuda/)
```

[compute_ops.py](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/ops/compute_ops.py) 作为 Python 入口，将 `librtp_compute_ops` 的所有符号重新导出，并按平台条件加载额外的算子（如 CUDA 上的 FusedRopeKVCacheOp）。

**总结**：`librtp_compute_ops` 是 rtp-llm 推理引擎的**高性能算子核心**，将 LLM 推理中计算密集的操作（注意力、归一化、量化、MoE、KV Cache、分布式通信等）全部用 C++/CUDA 实现，通过 pybind11 暴露给 Python 层调用，以实现极致的推理性能。

## User
适配NPU的时候，是否可以使用torch_npu提供的python API

## Assistant
现在我对 rtp-llm 的多硬件后端架构有了全面的了解，可以回答你的问题了。

## 适配 NPU 时是否可以使用 `torch_npu` 提供的 Python API？

**可以，而且这是推荐的做法**，但需要理解 rtp-llm 现有的多硬件适配架构，并按其模式来集成。下面详细分析：

---

### rtp-llm 现有的多硬件适配模式

rtp-llm 已经有一套成熟的**按设备类型分支加载不同实现**的架构，核心模式如下：

```
modules/base/
├── common/      ← 架构无关的通用实现
├── cuda/        ← CUDA 实现（调用 rtp_llm_ops / 自定义 CUDA kernel）
└── rocm/        ← ROCm 实现（调用 aiter 库）
```

在 [modules/base/\_\_init\_\_.py](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/models_py/modules/base/__init__.py) 中通过 `DeviceType` 做分发：

```python
device_type = get_exec_ctx().get_device_type()

if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.activation import FusedSiluAndMul
    from rtp_llm.models_py.modules.base.rocm.norm import RMSNorm, ...
else:
    from rtp_llm.models_py.modules.base.cuda.activation import FusedSiluAndMul
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm, ...
```

**ROCm 后端就是用第三方 Python 库 `aiter` 的一个先例**，与 NPU 使用 `torch_npu` 完全类似：

| 对比项 | ROCm 后端 | NPU 后端（待实现） |
|--------|-----------|-------------------|
| Python 算子库 | `aiter` | `torch_npu` |
| 激活函数 | `aiter.silu_and_mul()` | `torch_npu.npu.silu_and_mul()` 等 |
| 归一化 | `aiter.rmsnorm2d_fwd()` | `torch_npu` 对应 API |
| TopK | `aiter.topk_softmax()` | `torch_npu` 对应 API |
| 未实现算子 | `NotImplementedOp` 占位 | 同理 |

---

### 适配 NPU 需要做的事情

#### 1. 扩展 `DeviceType` 枚举

在 [cpp/core/DeviceData.h](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/cpp/core/DeviceData.h#L12-L18) 中添加 NPU 类型：

```cpp
enum class DeviceType {
    Cpu    = 0,
    Cuda   = 1,
    Yitian = 2,
    ArmCpu = 3,
    ROCm   = 4,
    Ppu    = 5,
    NPU    = 6,  // 新增
};
```

#### 2. 添加 NPU 设备实现

在 [device/device_impl.py](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/device/device_impl.py) 中添加 `NpuImpl` 类，并在 [device/\_\_init\_\_.py](file:///c:/WorkingSpace/Code/rtp-llm/rtp_llm/device/__init__.py) 中注册。

#### 3. 创建 `modules/base/npu/` 目录

参照 ROCm 的做法，创建 NPU 专属实现：

```
modules/base/npu/
├── __init__.py
├── activation.py    ← 用 torch_npu API 实现 FusedSiluAndMul
├── norm.py          ← 用 torch_npu API 实现 RMSNorm 等
├── select_topk.py   ← 用 torch_npu API 实现 SelectTopk
└── not_implemented_ops.py  ← NPU 暂不支持的算子占位
```

例如 `activation.py` 可能长这样：

```python
import torch
import torch_npu  # noqa: F401
from rtp_llm.models_py.modules.base.common.activation import SiluAndMulBase

class FusedSiluAndMul(SiluAndMulBase):
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        d = gate_up.shape[-1] // 2
        gate, up = gate_up[..., :d], gate_up[..., d:]
        # torch_npu 兼容 PyTorch 原生算子，silu 和 mul 可直接在 NPU 上运行
        return torch.nn.functional.silu(gate) * up
```

#### 4. 在 `modules/base/__init__.py` 中添加 NPU 分支

```python
if device_type == DeviceType.NPU:
    from rtp_llm.models_py.modules.base.npu.activation import FusedSiluAndMul
    from rtp_llm.models_py.modules.base.npu.norm import RMSNorm, ...
elif device_type == DeviceType.ROCm:
    ...
else:
    ...
```

---

### `torch_npu` 能提供什么

`torch_npu` 是华为昇腾的 PyTorch 扩展包，它提供：

| 能力 | 说明 |
|------|------|
| **设备管理** | `torch.npu.current_device()`, `torch.npu.device_count()` 等，接口与 `torch.cuda` 对齐 |
| **张量计算** | 将张量放到 NPU 上后，大部分 PyTorch 原生算子自动在 NPU 上执行（`torch_npu` 注册了 NPU dispatch key） |
| **自定义算子** | `torch_npu.npu.*` 命名空间提供昇腾特有的融合算子，如 `npu_bert_apply_adam`, `npu_rotation_gradient` 等 |
| **CANN 集成** | 底层调用 CANN (Compute Architecture for Neural Networks) 算子库 |

**关键优势**：由于 `torch_npu` 实现了 PyTorch 的 dispatch 机制，很多标准 PyTorch 算子（如 `torch.silu`, `torch.matmul`, `F.layer_norm` 等）在 `torch_npu` 初始化后可以直接在 NPU tensor 上运行，**不需要逐个算子适配**。

---

### 需要注意的难点

1. **C++ 层的 `librtp_compute_ops`**：这个库是用 CUDA/ROCm C++ kernel 编译的，NPU 无法直接使用。需要：
   - C++ 层新增 `USING_NPU` 编译分支
   - 或者在 Python 层完全绕过 `librtp_compute_ops`，用 `torch_npu` 的 Python API 替代

2. **注意力算子**：FlashInfer、XQA、TRT-Attn 都是 CUDA 专属的，NPU 上需要使用昇腾的 FlashAttention 实现（如 `torch_npu.npu_fusion_attention`）

3. **KV Cache 管理**：`KVCache`、`PyAttentionInputs` 等数据结构定义在 C++ pybind 中，NPU 适配需要重新实现或提供兼容层

4. **分布式通信**：HCCL 替代 NCCL/RCCL，`torch.distributed` 在 `torch_npu` 下会自动使用 HCCL 后端

5. **CUDA Graph**：NPU 有类似的图捕获机制（`torch_npu.npu.Graph`），但 API 不完全相同

---

### 总结

| 问题 | 回答 |
|------|------|
| 能否用 `torch_npu` Python API？ | **可以**，ROCm 用 `aiter` 就是先例 |
| 是否推荐？ | **推荐**，尤其是对于 Python 层的算子适配 |
| 能覆盖所有算子吗？ | **不能完全覆盖**，注意力、KV Cache、通信等核心模块需要 C++ 层配合或使用 `torch_npu` 的专用 API |
| 最小适配路径 | 先在 Python 层用 `torch_npu` + PyTorch 原生算子跑通基础模型，再逐步替换为昇腾融合算子优化性能 |


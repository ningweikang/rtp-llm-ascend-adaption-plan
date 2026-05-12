# 以 Qwen3-4B 为验证目标的适配计划调整建议

## 一、Qwen3-4B 架构特征分析

### 1.1 模型规格

| 参数 | Qwen3-4B |
|------|----------|
| 架构 | `Qwen3ForCausalLM` → `qwen_3` |
| 继承链 | `QWenV2` → `QWen` → `BaseModel` |
| 模型描述 | `rtp_llm/models_py/model_desc/qwen3.py` → `Qwen3Model` |
| 层数 | 36 |
| hidden_size | 2560 |
| intermediate_size (FFN) | ~? (需从 config.json 确认，通常 2.7x hidden) |
| num_attention_heads | 32 (GQA) |
| num_key_value_heads | 8 |
| head_dim | 128 |
| vocab_size | 151936 |
| **qk_norm** | **✅ True**（关键特征） |
| activation | SiLU (gated FFN) |
| norm | RMSNorm (eps=1e-5) |
| RoPE | dim=128, style=1 |
| MoE | **否** |
| MLA | **否** |
| MTP | **否** |
| bias | False (QwenV3 覆盖了 QWenV2 的 True) |
| tie_word_embeddings | 可能为 False |

### 1.2 推理路径追踪

```
QwenV3._create_config()
  → config.qk_norm = True          # ★ Qwen3 独有

QwenV3._create_python_model()
  → Qwen3Model(
       layers = Qwen3DecoderLayer(
         self_attn = CausalAttention(    # 标准 MHA (GQA)
           qkv_proj = LinearFactory.create()
           o_proj   = LinearFactory.create()
           qk_fuse_norm = FusedQKRMSNorm(...)  # ★ qk_norm=True 时启用
         )
         mlp = DenseMLP(SiLU, gate+up+down)
         input_layernorm  = RMSNorm
         post_attention_layernorm = RMSNorm
       )
       norm = RMSNorm
       embed_tokens = Embedding
     )
```

### 1.3 涉及的关键代码路径

| 模块 | 文件 | 关键依赖 |
|------|------|---------|
| 模型入口 | `models/qwen_v3.py` | `QWenV2` → `QWen` → `BaseModel` |
| 模型描述 | `models_py/model_desc/qwen3.py` | `Qwen3Model`, `Qwen3DecoderLayer` |
| Attention | `models_py/modules/hybrid/causal_attention.py` | `CausalAttention` → `LinearFactory` + `FusedQKRMSNorm` |
| QK Norm (CUDA) | `models_py/modules/base/cuda/norm.py` | `FusedQKRMSNorm` → **`flashinfer.norm.rmsnorm`** |
| QK Norm (ROCm) | `models_py/modules/base/rocm/norm.py` | `FusedQKRMSNorm` → **`rtp_llm_ops.fused_qk_rmsnorm`** |
| Attention FMHA | `attn_factory.py` → `PREFILL_MHA_IMPS/DECODE_MHA_IMPS` | FlashInfer / XQA / TRT |
| MLP | `models_py/modules/hybrid/dense_mlp.py` | `DenseMLP` → `LinearFactory` |
| RMSNorm | `modules/base/cuda/norm.py` | `rtp_llm_ops.rmsnorm` (C++ 自定义算子) |
| Linear | `modules/factory/linear/factory.py` | `LinearFactory` → 量化和非量化路径 |
| 权重加载 | `models/qwen_v2.py` → `QWenV2Weight` | HF 格式权重映射 |

---

## 二、对适配计划的影响分析

### 2.1 好消息：Qwen3-4B 避开了最复杂的模块

| 复杂模块 | 是否涉及 | 说明 |
|----------|---------|------|
| MoE (FusedMoE) | ❌ 不涉及 | Qwen3-4B 是 dense 模型，**Phase 5 的 MoE dispatch/reorder 可跳过** |
| MLA Attention | ❌ 不涉及 | 无 FlashMLA 依赖，Phase 3 不需要 MLA 适配 |
| MTP | ❌ 不涉及 | 无 multi-token prediction |
| FP8/FP4 | ❌ 不涉及（首阶段） | BF16/FP16 推理即可验证 |
| FP4 kernel | ❌ 不涉及 | |
| Triton kernels | ❌ 不涉及（首阶段） | Qwen3 的标准路径不使用 Triton |
| XQA (SM90) | ❌ 可跳过 | 有 fallback 机制 |
| Speculative Decoding | ❌ 不涉及 | |
| Expert Parallelism (DeepEP) | ❌ 不涉及 | |
| Vision/Multimodal | ❌ 不涉及 | |

### 2.2 需要特别关注的：QK Norm

Qwen3 的 `qk_norm = True` 触发了一个**独特的代码路径**：

```python
# causal_attention.py:66
if W.q_ln_gamma in weights and W.k_ln_gamma in weights:
    self.qk_fuse_norm = FusedQKRMSNorm(...)
```

CUDA 路径的 `FusedQKRMSNorm` 调用了 **`flashinfer.norm.rmsnorm`**（`cuda/norm.py:109-110`）：
```python
flashinfer.norm.rmsnorm(q, self.q_weight, eps=self.eps, out=q, enable_pdl=self.enable_pdl)
flashinfer.norm.rmsnorm(k, self.k_weight, eps=self.eps, out=k, enable_pdl=self.enable_pdl)
```

而 ROCm 路径调用的是 **`rtp_llm_ops.fused_qk_rmsnorm`**（`rocm/norm.py:179-189`）。

**这意味着**：Ascend 路径需要提供一个 `FusedQKRMSNorm` 实现，替代 `flashinfer.norm.rmsnorm`。

### 2.3 需要适配的完整算子清单（Qwen3-4B 路径）

| 算子 | 来源 | 说明 | 优先级 |
|------|------|------|--------|
| RMSNorm | `rtp_llm_ops.rmsnorm` | 每层 2 个 + final norm = 73 次/forward | **P0** |
| FusedQKRMSNorm | `flashinfer.norm.rmsnorm` / `rtp_llm_ops.fused_qk_rmsnorm` | Qwen3 独有，每层 attention | **P0** |
| SiLU + Gate (SwiGLU FFN) | PyTorch 标准操作 | `torch.nn.functional.silu` | **P0** |
| Embedding | PyTorch 标准操作 | `torch.nn.functional.embedding` | **P0** |
| Linear (MatMul) | `LinearFactory` | QKV proj, O proj, Gate/Up/Down proj | **P0** |
| Attention (Prefill) | FlashInfer / 自定义 | Paged KV Cache Prefill | **P0** |
| Attention (Decode) | FlashInfer / 自定义 | Paged KV Cache Decode | **P0** |
| RoPE | FlashInfer 内置 | 旋转位置编码 | **P0** |
| TopK Sampling | `rtp_llm_ops` | 采样 | **P1** |
| Softmax | PyTorch / CANN | 采样中 | **P1** |

---

## 三、适配计划调整建议

### 3.1 可跳过的 Phase/子任务

| 原计划任务 | 调整 | 原因 |
|-----------|------|------|
| Phase 5: MoE dispatch/reorder | **跳过** | Qwen3-4B 无 MoE |
| Phase 5: FP8 quant kernel | **跳过** | 首阶段 FP16/BF16 即可 |
| Phase 5: FP4 kernel | **跳过** | 不需要 |
| Phase 3.2: MLA Attention | **跳过** | Qwen3 无 MLA |
| Phase 3.4: XQA | **跳过** | 有 fallback，非必须 |
| Phase 6: Expert Parallelism | **跳过** | 无 MoE |
| Phase 8: FP8 全栈 | **延后** | 首阶段不涉及 |
| Phase 5: Triton kernels | **跳过** | Qwen3 标准路径不用 Triton |

### 3.2 需要新增/调整的任务

#### 新增：Ascend QK Norm 适配

这是原计划**未提及**的关键任务，Qwen3 必需：

**位置**：插入到 Phase 3（Attention 适配）中，作为 Phase 3 的前置子任务

```
rtp_llm/models_py/modules/base/ascend/
├── __init__.py
├── norm.py              # AscendRMSNorm + AscendFusedQKRMSNorm
└── BUILD
```

**实现方案**：
- `AscendRMSNorm`：使用 `aclnnRmsNorm` 或 PyTorch 原生 `torch.nn.functional.rms_norm`
- `AscendFusedQKRMSNorm`：调用 `aclnnRmsNorm` 对 Q 和 K 分头做 norm（替代 `flashinfer.norm.rmsnorm`）

**CausalAttention 中的设备分派**需要修改：

```python
# causal_attention.py 当前代码：
device_type = get_exec_ctx().get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm

# 需要改为：
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
elif device_type == DeviceType.Ascend:
    from rtp_llm.models_py.modules.base.ascend.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm
```

同样，`RMSNorm` 也需要 Ascend 版本（`causal_attention.py` 不直接用，但 `Qwen3Model` 的 `self.norm` 和 `Qwen3DecoderLayer` 的 layernorm 用到了 `RMSNorm`——实际上看 `qwen3.py` 的 import，它从 `modules` 导入的 `RMSNorm` 需要确认走的是哪条路径）。

#### 调整：Phase 1 需要提前确认 norm 模块的设备分派机制

当前 `causal_attention.py:14-18` 使用了**模块级 import** 而非 Factory 模式：
```python
device_type = get_exec_ctx().get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm
```

这意味着**所有类似的模块级设备分派**都需要增加 Ascend 分支。这比 Factory 注册更侵入式。

**需要全局搜索的模式**：
```python
device_type == DeviceType.ROCm
# 或者
if_rocm / USING_ROCM
```
这些位置都需要同步添加 Ascend 分支。

### 3.3 调整后的 Phase 执行顺序

以 Qwen3-4B 为最小可验证目标，建议调整为：

```
Phase 0: 编译基础设施                    （不变）
Phase 1: 内存管理 + 设备运行时            （不变）
Phase 2: GEMM (FP16/BF16 MatMul only)   （不变，跳过 FP8 GEMM）
Phase 3: Attention + QK Norm             （增加 QK Norm 子任务）
Phase 4: KV Cache                        （不变）
Phase 5': 精简算子迁移                    （只迁移 Qwen3-4B 需要的算子）
  ├── RMSNorm        (aclnnRmsNorm)
  ├── FusedQKRMSNorm (aclnnRmsNorm × 2)
  ├── Sampling       (aclnnTopk + PyTorch)
  ├── Embedding      (PyTorch native)
  └── 跳过: MoE, FP8, FP4, Triton
Phase 6': 单卡验证                        （跳过分布式，先验证单卡）
Phase 7: CUDA Graph → Ascend Graph        （Qwen3 支持 graph，需要适配）
Phase 8: FP8/量化                         （延后）
Phase 9: Python 全量适配                  （不变）
Phase 10: 性能优化                        （不变）
```

### 3.4 关键里程碑调整

| 里程碑 | 原计划 | 调整后 | 交付物 |
|--------|-------|--------|--------|
| **M1** | Phase 0 | Phase 0 | Bazel 编译通过 |
| **M2** | Phase 1-2 | Phase 0-2 | 单卡 FP16 **单层 forward** 通过 |
| **M3** | Phase 3-4 | **Phase 3-4 + 5'** | **Qwen3-4B 单卡 FP16 推理跑通** ⭐ |
| M4 | Phase 5-6 | Phase 6'-7 | Graph 模式 + 性能初步优化 |
| M5 | Phase 7-8 | Phase 8-9 | FP8 + 全量功能 |
| M6 | Phase 9-10 | Phase 10 | 性能优化 |

**核心变化**：将 Qwen3-4B 单卡推理跑通作为 M3 的明确验收标准，从原来的"第 7-11 周"提前到**第 6-8 周**。

### 3.5 模块级设备分派 — 需修改文件清单

以下位置的模块级 if/else 设备分派需要增加 Ascend 分支：

| 文件 | 当前逻辑 | 需新增 |
|------|---------|--------|
| `modules/hybrid/causal_attention.py:14-18` | `if ROCm: ... else: cuda.norm.FusedQKRMSNorm` | `elif Ascend: ascend.norm.FusedQKRMSNorm` |
| `qwen_v3.py` (QWenV3) | 通过继承走 QWenV2 路径 | 需确认 norm 模块分派 |
| `model_desc/qwen3.py` | import `RMSNorm` from `modules` | 需确认 RMSNorm 的设备分派路径 |

还需要搜索所有类似的模式（全局搜索 `DeviceType.ROCm`）。

---

## 四、Qwen3-4B 验证的具体测试用例

### 4.1 分层验证策略

```
Step 1: 单算子验证
  ├── RMSNorm on NPU vs CPU 参考
  ├── FusedQKRMSNorm on NPU vs CPU 参考
  ├── FP16/BF16 GEMM on NPU vs cuBLAS 参考
  └── SiLU, Embedding on NPU

Step 2: 单层 Transformer forward
  └── 构造 Qwen3DecoderLayer，输入随机 hidden_states
      → 检查输出无 NaN/Inf
      → 与 PyTorch CPU 参考实现对比 (cosine sim > 0.99)

Step 3: 完整模型 forward (prefill)
  └── 加载 Qwen3-4B 权重到 NPU
      → 输入 "你好" (token_ids)
      → 检查 logits 形状和值分布

Step 4: 完整模型 generate (decode)
  └── 多轮 decode 生成 50-100 tokens
      → 检查输出文本连贯性
      → 检查无重复/乱码

Step 5: 端到端 serving
  └── 启动 rtp-llm server
      → HTTP 请求推理
      → 对比与 CUDA 部署的输出
```

### 4.2 精度验收标准

| 指标 | 验收标准 |
|------|---------|
| 单算子精度 | max_diff < 1e-2 (FP16), < 2e-2 (BF16) |
| 单层 forward | cosine_similarity > 0.99 |
| Full model logits | top-5 token IDs 与 CUDA 参考一致率 > 90% |
| 生成文本 | 人工评估连贯性 + 与 CUDA 输出 Rouge-L > 0.8 |

---

## 五、总结

### 需要的调整

| 调整项 | 类型 | 优先级 |
|--------|------|--------|
| 新增 `ascend/norm.py` (`AscendRMSNorm` + `AscendFusedQKRMSNorm`) | **新增** | P0 |
| `causal_attention.py` 设备分派增加 Ascend | **修改** | P0 |
| 全局搜索 `DeviceType.ROCm` 增加Ascend分支 (~15-20处) | **修改** | P0 |
| 跳过 MoE/MLA/FP8/FP4/Triton 相关子任务 | **删减** | P1 |
| M3 里程碑改为 "Qwen3-4B 单卡推理跑通" | **修改** | P1 |
| 总周期从 25 周 → **~15 周**（精简后） | **修改** | P1 |

### 风险

| 风险 | 说明 |
|------|------|
| `FusedQKRMSNorm` 依赖 `flashinfer.norm.rmsnorm` | CANN 无 FlashInfer，需用 `aclnnRmsNorm` 替代，精度需验证 |
| 模块级 if/else 设备分派遗漏 | 比工厂模式更容易遗漏，需全局搜索确保覆盖 |
| GQA (32 heads / 8 kv_heads) | CANN FlashAttention 需支持 GQA 模式（非标准 MHA） |

### 最终建议

以 Qwen3-4B 为验证目标后，适配工作量可减少约 **40%**（跳过 MoE/MLA/FP8/FP4/Triton/DeepEP），但需要**额外关注 QK Norm** 这一原计划未覆盖的关键路径。建议将 Qwen3-4B 单卡 FP16/BF16 推理跑通作为 Phase 0-5' 的统一验收标准。

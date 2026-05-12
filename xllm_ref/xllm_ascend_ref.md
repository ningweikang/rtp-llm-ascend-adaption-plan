# xllm 昇腾适配参考点

> 基于 xllm 代码仓（`C:\WorkingSpace\Code\xllm`）的深度分析，提炼 rtp-llm 昇腾适配可借鉴的设计模式、实现方案和工具类。
> rtp-llm 已选定 torch_npu Python API 路径（类似 vllm-ascend），xllm 的 ATB C++ 路径仅作参考，不直接照搬。

---

## 一、架构层面

### 1.1 Executor Factory + ACL Graph 模式

**文件**: `xllm/core/runtime/executor_impl_factory.h`, `acl_graph_executor_impl.h`

xllm 使用注册宏模式管理不同后端的执行器：

```cpp
// 注册宏
#define REGISTER_EXECUTOR(backend, class_type) \
  static bool _reg_##class_type = \
    ExecutorImplFactory::get_instance().register_creator( \
        backend, [](auto... args) { return std::make_unique<class_type>(args...); });

// NPU 执行器注册
REGISTER_EXECUTOR("npu", AclGraphExecutorImpl);
```

运行时根据 backend 字符串选择执行器：
```cpp
std::string backend = FLAGS_enable_graph ? Device::type_str() : options.backend();
impl_ = ExecutorImplFactory::get_instance().create_executor_impl(model, args, device, options, backend);
```

**`AclGraphExecutorImpl`** 核心结构（1131 行）：

```cpp
class AclGraph {
  c10_npu::NPUGraph graph_;                        // torch_npu 的 NPUGraph
  GraphPersistentParam& persistent_param_;          // 共享持久化 tensor
  std::optional<c10_npu::NPUStream> capture_stream_; // 非默认 stream 用于捕获

  bool capture(CausalLM* model, ..., uint32_t bucket_num_tokens);
  ModelOutput replay(...);
};

class AclGraphExecutorImpl : public ExecutorImpl {
  absl::flat_hash_map<uint32_t, std::unique_ptr<AclGraph>> graphs_;  // bucket -> graph
  std::unique_ptr<GraphPersistentParam> persistent_param_;
  uint32_t get_bucket_num_tokens(uint32_t num_tokens) const;
};
```

**Graph 捕获流程**：
```cpp
bool AclGraph::capture(...) {
  // 1. 用 padded 数据更新 persistent buffers
  auto graph_params = persistent_param_.update(..., /*return_capture_params=*/true);

  // 2. 切换到非默认 stream（NPUGraph 要求）
  auto& capture_lock = DeviceCaptureLock::get_instance().get_lock(device_idx);
  std::lock_guard<std::mutex> lock_guard(capture_lock);
  c10_npu::setCurrentNPUStream(capture_stream_.value());

  // 3. 开始捕获（thread-local 模式）
  graph_.capture_begin({0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);

  // 4. 执行前向（所有 ATB 操作在捕获模式下执行）
  auto forward_result = model->forward(persistent_tokens, persistent_positions, kv_cache, graph_params);

  // 5. 结束捕获
  graph_.capture_end();

  // 6. 恢复默认 stream
  c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream(device_idx));
}
```

**Graph 重放流程**：
```cpp
ModelOutput AclGraph::replay(...) {
  persistent_param_.update(..., false);  // 更新实际数据
  graph_.replay();                       // 重放
  return ModelOutput(get_hidden_states(actual_num_tokens));
}
```

**Bucket 策略**：
```cpp
uint32_t get_bucket_num_tokens(uint32_t num_tokens) const {
  if (FLAGS_enable_graph_mode_decode_no_padding) return num_tokens;
  if (num_tokens <= 1) return 1;
  if (num_tokens <= 2) return 2;
  if (num_tokens <= 4) return 4;
  if (num_tokens <= 8) return 8;
  return ((num_tokens + 15) / 16) * 16;
}
```

**rtp-llm 借鉴**：Phase 7（CUDA Graph → Ascend Graph）可直接参考此模式。核心要素：
- `c10_npu::NPUGraph` 替代 `torch.cuda.CUDAGraph`
- `GraphPersistentParam` 预分配所有输入输出 tensor
- Bucket 分桶策略减少 graph 数量
- Device Capture Lock 保证捕获安全

---

### 1.2 三层模板继承体系

**文件**: `xllm/models/llm/npu/llm_model_base.h`

```
LlmDecoderLayerImplBase<DecoderType>   # 封装 decoder + block_copy
  ↑
LlmModelImplBase<DecoderLayerType>     # embed_tokens + layers + norm + forward loop
  ↑
LlmForCausalLMImplBase<LlmModelType>  # model + lm_head + weight loading + rolling
```

每个具体模型只需填充模板参数：
```cpp
// xllm/models/llm/npu/qwen2.h
namespace xllm::npu::model {
class QWen2DecoderLayerImpl
    : public LlmDecoderLayerImplBase<layer::NpuQwen2DecoderLayer> { ... };
class QWen2ModelImpl
    : public LlmModelImplBase<QWen2DecoderLayer> { ... };
class QWen2ForCausalLMImpl
    : public LlmForCausalLMImplBase<QWen2Model> { ... };

REGISTER_CAUSAL_MODEL(qwen2, QWen2ForCausalLM);
REGISTER_MODEL_ARGS(qwen2, [&] { LOAD_ARG_OR(...); });
}
```

**rtp-llm 借鉴**：rtp-llm 的 Python Factory 模式已类似（`attn_factory.py` 等），无需照搬 C++ 模板。但在 Python 端可以参考 NPU model 的注册/分派机制。

---

### 1.3 Prefill/Decode 双节点分离

**文件**: `xllm/core/layers/npu/npu_qwen2_decoder_layer_impl.h`

每个 NPU decoder layer 维护两个 ATB Node：

```cpp
class NpuQwen2DecoderLayerImpl : public BaseLayer {
  atb_speed::Model::Node prefill_node_;    // Prefill ATB 操作节点
  atb_speed::Model::Node decode_node_;     // Decode ATB 操作节点
  atb_speed::qwen::DecoderLayerParam prefill_param_;
  atb_speed::qwen::DecoderLayerParam decode_param_;
};
```

初始化时创建两个不同参数的 ATB 操作：
```cpp
int64_t init_node(atb_speed::Model::Node& node, atb_speed::qwen::DecoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::DecoderLayer(param, &operation);  // ATB Speed 创建操作
  node.operation.reset(operation);
  // ... 绑定权重 tensor
}
```

前向时选择执行：
```cpp
void forward(...) {
  if (is_prefill) {
    build_node_variant_pack(prefill_node_, ...);
    execute_node(prefill_node_);
  } else {
    build_node_variant_pack(decode_node_, ...);
    execute_node(decode_node_);
  }
}
```

**rtp-llm 借鉴**：rtp-llm 已有 Prefill/Decode 分离的 Python 实现（`PyFlashinferPrefillPagedAttnOp` vs `PyFlashinferDecodeAttnOp`），NPU 版本维持相同的分离模式即可。

---

## 二、设备抽象

### 2.1 编译时 `#if defined(USE_NPU)` 宏分派

**文件**: `xllm/core/platform/device.h`, `device.cpp`

xllm 的核心设备抽象类，使用编译时宏而非运行时多态：

```cpp
// 设备类型
torch::DeviceType Device::type_torch() {
#if defined(USE_NPU) || defined(USE_MLU)
  return torch::kPrivateUse1;
#elif defined(USE_CUDA) || defined(USE_ILU)
  return torch::kCUDA;
#elif defined(USE_MUSA)
  return torch::kMUSA;
#endif
}

// 设备名称字符串
std::string Device::type_str() {
#if defined(USE_NPU)   return "npu";
#elif defined(USE_CUDA) return "cuda";
#elif defined(USE_MLU)  return "mlu";
#endif
}

// 设备数量
int Device::device_count() {
#if defined(USE_NPU)   return c10_npu::device_count();
#elif defined(USE_CUDA) return c10::cuda::device_count();
#endif
}

// 设置设备
void Device::set_device() const {
#if defined(USE_NPU)   c10_npu::set_device(index());
#elif defined(USE_CUDA) c10::cuda::set_device(index());
#endif
}

// 内存查询
Device::DeviceMem Device::get_device_mem() const {
#if defined(USE_NPU)
  aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
#elif defined(USE_CUDA)
  cudaMemGetInfo(&free_memory, &total_memory);
#endif
}

// Stream 同步
int Device::synchronize_default_stream() {
#if defined(USE_NPU)
  return aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(index()).stream());
#elif defined(USE_CUDA)
  c10::cuda::getCurrentCUDAStream().synchronize();
#endif
}

// 缓存清理
void Device::empty_cache(int32_t device_index) {
#if defined(USE_NPU)
  c10_npu::NPUCachingAllocator::emptyCache();
  c10_npu::NPUCachingAllocator::FreeDeviceCachedMemory(device_index);
#endif
}
```

**rtp-llm 借鉴**：rtp-llm 已有 `#if USING_CUDA` / `#if USING_ROCM` 模式，只需增加 `#if USING_ASCEND` 分支。建议创建统一的设备抽象类（`rtp_llm/cpp/core/Device.h`），替代散落各处的条件编译。

---

### 2.2 NPU 使用 `torch::kPrivateUse1`

**关键发现**：torch_npu 通过 PyTorch 的 PrivateUse1 机制注册 NPU 设备。

```cpp
// torch_npu 内部注册
#include <torch/library.h>
static auto register_npu = []() {
  torch::register_privateuse1_backend("npu");
  return true;
}();
```

**rtp-llm 借鉴**：在参数化设备类型时，NPU 应使用 `torch::kPrivateUse1`（或通过 `torch_npu` 提供的 `torch_npu.npu.set_device()` 等 Python API），而非假设存在 `torch::kNPU`。在 Python 端可以通过 `torch.device("npu:0")` 访问。

---

## 三、KV Cache

### 3.1 NPU 内存格式选择策略

**文件**: `xllm/core/framework/kv_cache/kv_cache_utils.cpp:154-158`

```cpp
aclFormat get_npu_kv_cache_format(const std::string& model_type) {
  return model_type == "deepseek_v3" && FLAGS_enable_prefix_cache
             ? ACL_FORMAT_FRACTAL_NZ    // DeepSeek-V3 + prefix cache 用 NZ 格式
             : ACL_FORMAT_ND;           // 其他所有场景用 ND 格式
}
```

| 场景 | 格式 | 原因 |
|------|------|------|
| 通用模型（Qwen2/3, LLaMA, GLM4） | `ACL_FORMAT_ND` | N-D 维度标准布局，兼容性好 |
| DeepSeek-V3 + Prefix Cache | `ACL_FORMAT_FRACTAL_NZ` | NZ 格式对 MLA 低秩 KV 更友好，对齐到 16 字节 |

**rtp-llm 借鉴**：
- 标准模型全部使用 `ACL_FORMAT_ND`
- DeepSeek-V3 MLA + prefix cache 场景考虑 `ACL_FORMAT_FRACTAL_NZ`（Phase 4 后期）

---

### 3.2 NPU Tensor 创建：`torch::empty` + `npu_format_cast`

**文件**: `xllm/core/framework/kv_cache/kv_cache_utils.cpp:34-49`

```cpp
// NPU: 使用 torch::empty + npu_format_cast（而非 torch::zeros）
tensors.key_cache = at_npu::native::npu_format_cast(
    torch::empty(kv_cache_shape.key_cache_shape(),
                 torch::dtype(create_options.dtype())
                     .device(create_options.device())),
    npu_format_type);  // ACL_FORMAT_ND 或 ACL_FORMAT_FRACTAL_NZ
```

**关键差异**：
- **GPU/CUDA**: 使用 `torch::zeros` 初始化
- **NPU**: 使用 `torch::empty` + `npu_format_cast`，因为 NPU 的 5HD/NZ 格式需要显式转换

**rtp-llm 借鉴**：`BlockPool::initializeCacheBuffer()` 中分配 KV cache buffer 时：
```cpp
#if USING_ASCEND
  auto tensor = torch::empty({total_size_bytes},
      torch::dtype(torch::kUInt8).device(torch::kPrivateUse1));
  // KV cache 使用 ND 格式，无需 npu_format_cast
  // 如果后续需要 NZ 格式，加上:
  // tensor = at_npu::native::npu_format_cast(tensor, ACL_FORMAT_ND);
#endif
```

---

### 3.3 Block Copy 的 ATB 算子实现

**文件**: `xllm/core/layers/npu/npu_block_copy_impl.cpp`

xllm 使用 ATB 的 `BlockCopy` operation 做 KV cache block 级拷贝：

```cpp
// NpuBlockCopyImpl 接收 5 个输入：
// 1. key_cache    - 完整 paged K cache
// 2. value_cache  - 完整 paged V cache
// 3. src_block_ids - 源 block 索引
// 4. dst_block_ids - 目标 block 索引
// 5. cum_sum       - 批量拷贝累积和

atb::Status atbStatus = atb::CreateOperation(param, &operation);
```

对比 PyTorch 原生操作（xllm 的 fallback）：
```cpp
// KVCacheImpl::swap_blocks() 的 fallback 实现
selected = torch::index_select(key_cache_, 0, src);  // 读取源 block
key_cache_.index_copy_(0, dst, selected);             // 写入目标 block
```

**rtp-llm 借鉴**：短期使用 PyTorch 原生 `index_select` + `index_copy_`（设备无关），性能优化阶段考虑 ATB BlockCopy 或 `torch_npu` 等效算子。

---

### 3.4 ACL Graph 兼容的 Paged Attention

**文件**: `xllm/core/kernels/npu/attention.cpp:64-88`

标准 `npu_paged_attention` 内部有 `.to(kCPU)` 操作会打断 ACL Graph 捕获。xllm 提供了专用变体：

```cpp
// 标准版本（ACL Graph 不兼容）
void batch_decode(const torch::Tensor& query, ...) {
  atb::npu_paged_attention(q, k_cache, v_cache, num_kv_heads,
                           num_heads, scale, block_table, seq_lens, o);
}

// ACL Graph 兼容版本
void batch_decode_acl_graph(const torch::Tensor& query, ...,
                            const torch::Tensor& tiling_data, ...) {
  atb::npu_custom_paged_attention(q, k_cache, v_cache, num_kv_heads,
                                  num_heads, scale, block_table,
                                  seq_lens, tiling_data, o);
}
```

**核心差异**：custom 版本接受预计算的 `tiling_data` tensor，消除了运行时 CPU 读取需求。

**rtp-llm 借鉴**：Phase 7 Ascend Graph 适配时，需要验证 `torch_npu._npu_paged_attention` 是否兼容 ACL Graph。如果不兼容，需要找到或实现类似 `npu_custom_paged_attention` 的 graph-safe 变体。vllm-ascend 通过 workspace 预分配和 `torch.npu.graph_task_update_begin/end` 部分解决了这个问题。

---

### 3.5 DeepSeek-V3 MLA NZ 格式 Shape 对齐

**文件**: `xllm/core/framework/kv_cache/kv_cache_shape.cpp:207-266`

DeepSeek-V3 启用 prefix cache 时，shape 需要对齐到 16 字节（NZ 格式要求）：

```cpp
// 标准 MLA shape
key_cache:   [n_blocks, block_size, 1, kv_lora_rank]
value_cache: [n_blocks, block_size, 1, qk_rope_head_dim]

// NZ MLA shape (prefix cache enabled)
key_cache:   [n_blocks, ceil_div(kv_lora_rank, 16), block_size, 16]
value_cache: [n_blocks, ceil_div(qk_rope_head_dim, 16), block_size, 16]
```

将最后一维拆分为 `(ceil_div(dim, 16), 16)` 满足 Ascend NPU NZ 格式对齐要求。

**rtp-llm 借鉴**：MLA 适配阶段（Phase 4.7），如需支持 DeepSeek-V3 + prefix cache，需实现 NZ shape 变换。

---

## 四、权重加载

### 4.1 Manual Loader + 异步 H2D 模式

**文件**: `xllm/core/layers/npu/loader/base_manual_loader.cpp`

针对 NPU 内存受限场景的权重加载模式：

```cpp
// Step 1: 将权重合并到 pinned host memory
void BaseManualLoader::copy_weights_to_pinned_host() {
  aclrtMallocHost(&host_pinned_storage_, storage_size_);  // Pinned host 分配
  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    if (is_nz_format_tensor(i)) {
      copy_host_nd_to_nz(host_tensor, dst, slice.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    } else {
      std::memcpy(dst, host_tensor.data_ptr(), slice.bytes);
    }
  }
}

// Step 2: 异步 H2D 拷贝
void BaseManualLoader::copy_weights_to_device_async(aclrtStream stream) {
  aclrtMemcpyAsync(device_storage_, storage_size_, host_pinned_storage_,
                   storage_size_, ACL_MEMCPY_HOST_TO_DEVICE, stream);
}

// Step 3: 从裸设备指针创建 torch tensor（不分配新内存）
torch::Tensor BaseManualLoader::convert_to_torch_tensor(...) {
  auto fptr = c10::GetStorageImplCreate(device_type);
  torch::Storage storage = fptr(c10::StorageImpl::use_byte_size_t(),
                                 c10::SymInt(tensor_nbytes),
                                 std::move(c10_data_ptr), allocator, true);
  tensor.set_(storage, 0, dims);
  // 强制 NZ 格式
  if (acl_format == ACL_FORMAT_FRACTAL_NZ) {
    tensor_storage->npu_desc_.npu_format_ = ACL_FORMAT_FRACTAL_NZ;
  }
}
```

**rtp-llm 借鉴**：rtp-llm 当前的权重加载（Python 端 `load_state_dict`）一般够用。但在大模型场景（如 671B DeepSeek-V3），可以参考此模式实现 NPU 端的内存高效加载。

---

### 4.2 Rolling Weight Buffer（大模型 Offloading）

**文件**: `xllm/core/layers/npu/loader/rolling_weight_buffer.h`, `rolling_load_manager.h`

对于超过设备内存的大模型：

```
┌──────────────────────────────────────────────┐
│            Host Memory (Pinned)               │
│  Layer 0 weights | Layer 1 weights | ...      │
└────────┬──────────────┬───────────────────────┘
         │ async H2D    │ async H2D
         ▼              ▼
┌──────────────────────────────────────────────┐
│          NPU HBM (Rolling Buffer)             │
│  ┌─── Slot 0 ───┐  ┌─── Slot 1 ───┐          │
│  │ Layer N weights │  │ Layer N+1 weights │    │
│  └───────────────┘  └───────────────┘          │
│  (rolling_load_num_cached_layers = 2)          │
└──────────────────────────────────────────────┘
```

- `RollingWeightBuffer`：在 HBM 中维护 N 个 decoder layer 权重槽位
- `RollingLoadManager`：前向推理时按需异步加载当前层权重
- 支持 `aclrtMalloc` 和 `XTensorAllocator` 两种内存后端

**控制参数**：
```cpp
DECLARE_bool(enable_rolling_load);             // 启用 rolling load
DECLARE_int32(rolling_load_num_cached_layers);  // HBM 中的层数（默认 2）
DECLARE_int32(rolling_load_num_rolling_slots);  // rolling vs fixed 槽位数
```

**rtp-llm 借鉴**：rtp-llm 当前不支持 offloading。如果后续需要支持超大模型在单卡运行，可参考此模式。

---

## 五、分布式通信

### 5.1 HCCL ProcessGroup

**文件**: `xllm/core/framework/parallel_state/npu_process_group.cpp`

```cpp
// 使用 torch_npu 的 ProcessGroupHCCL（封装 HCCL）
auto store = create_tcp_store(host, port, rank);
auto hccl_pg_options = c10d_npu::ProcessGroupHCCL::Options();
pg_ = std::make_unique<c10d_npu::ProcessGroupHCCL>(store, rank, rank_size, hccl_pg_options);
```

**rtp-llm 借鉴**：Phase 6 中将 `backend="nccl"` 改为 `backend="hccl"`：
```python
# rtp-llm Python 端
torch.distributed.init_process_group(backend="hccl", ...)
# torch_npu 已集成 HCCL 后端，直接可用
```

---

### 5.2 通信后端选择：`lccl` vs `hccl`

**文件**: `xllm/core/framework/parallel_state/mapping_npu.h`

xllm 区分两种通信后端：
- **`lccl`**：节点内低延迟集合通信，用于 TP（Tensor Parallel）
- **`hccl`**：跨节点通信，用于 DP（Data Parallel）

```cpp
// MappingNPU 管理多维度并行拓扑
struct MappingNPU {
  ParallelInfo attn_tp_;        // TP: lccl
  ParallelInfo word_embed_tp_;  // TP: lccl
  ParallelInfo mlp_tp_;         // TP: lccl
  ParallelInfo moe_ep_;         // EP: hccl
  ParallelInfo word_embed_dp_;  // DP: hccl
  ParallelInfo attn_dp_;        // DP: hccl
  ParallelInfo attn_cp_;        // CP: lccl/hccl
};

// ATB 参数设置
param.backend = "lccl";
param.tensorParallelInfo = {rank, world_size, "lccl"};
```

**rtp-llm 借鉴**：rtp-llm 的 TP 通信通过 `collective_torch.py` 使用 `torch.distributed`，切换到 `hccl` 后端即可。如果需要极致的 TP 性能，可以考虑 `lccl`。EP/CP 等高级并行模式的通信后端选择可后续评估。

---

## 六、工具类

### 6.1 Device Capture Lock（ACL Graph 安全）

**文件**: `xllm/core/platform/npu/device_capture_lock.h`

ACL Graph 捕获期间，并发的 H2D 操作会破坏捕获。此 singleton 提供设备级互斥：

```cpp
class DeviceCaptureLock {
 public:
  static DeviceCaptureLock& get_instance() {
    static DeviceCaptureLock instance;
    return instance;
  }
  std::mutex& get_lock(c10::DeviceIndex device_index) {
    std::lock_guard<std::mutex> lock(locks_mutex_);
    if (locks_.find(device_index) == locks_.end()) {
      locks_[device_index] = std::make_unique<std::mutex>();
    }
    return *locks_[device_index];
  }
 private:
  std::mutex locks_mutex_;
  std::unordered_map<c10::DeviceIndex, std::unique_ptr<std::mutex>> locks_;
};
```

**使用方式**：
```cpp
auto& capture_lock = DeviceCaptureLock::get_instance().get_lock(device_idx);
std::lock_guard<std::mutex> lock_guard(capture_lock);
// ... 执行 graph capture
```

**rtp-llm 借鉴**：Phase 7 Ascend Graph 适配时**必须**引入此机制。没有它，并发的权重加载或 KV cache 传输 H2D 操作会导致 graph capture 失败。

---

### 6.2 NPU Layer Synchronizer（PD 分离逐层同步）

**文件**: `xllm/core/platform/npu/npu_layer_synchronizer.h`

Prefill-Decode 分离场景中，逐层同步 KV cache 传输：

```cpp
class NPULayerSynchronizerImpl {
 public:
  aclrtEvent* get_event(const int64_t layer_index) {
    return &events_[layer_index];
  }
  std::atomic<bool>* get_event_flag(const int64_t layer_index) {
    return &event_record_flags_[layer_index];
  }
  bool synchronize_layer(const int64_t layer_index) {
    if (!event_record_flags_[layer_index].load()) return true;
    aclrtSynchronizeEvent(events_[layer_index]);
    return true;
  }
 private:
  std::vector<aclrtEvent> events_;
  std::vector<std::atomic<bool>> event_record_flags_;
};
```

**工作流程**：
1. Layer N 的计算完成后，record event
2. KV cache 传输线程等待该 event 后再 push Layer N 的 KV cache
3. 避免传输未完成的 KV 数据

**rtp-llm 借鉴**：如果 rtp-llm 后续支持 PD 分离部署，此逐层同步模式是必需的。

---

### 6.3 NPU Profiling（MSPTI + MSTX）

**文件**: `xllm/core/common/mspti_helper.h`

类似 NVTX 的 NPU 性能标记工具：

```cpp
// RAII range 标记
class MstxRange {
 public:
  MstxRange(const char* message) { mstxRangeStartA(message, &id_); }
  ~MstxRange() { mstxRangeEnd(id_); }
 private:
  uint64_t id_;
};

// 函数级自动标记宏
#define LLM_MSTX_RANGE() \
  MstxRange CONCATENATE(llm_mstx_range_, __LINE__) { __PRETTY_FUNCTION__ }
```

**MSPTI Subscriber**（可选编译 `USE_MSPTI=ON`）：
```cpp
class MsptiMetrics {
  static void register_subscriber();
  static std::string handle_hccl_event(msptiActivity* pRecord);    // HCCL 通信
  static std::string handle_kernel_event(msptiActivity* pRecord);  // 算子执行
  static std::string handle_memory_event(msptiActivity* pRecord);  // 内存操作
};
```

**rtp-llm 借鉴**：性能调优阶段使用。在关键路径添加 `LLM_MSTX_RANGE()` 宏，配合 `msprof` 工具分析性能瓶颈。

---

### 6.4 NPU Kernel 工具函数

**文件**: `xllm/core/kernels/npu/utils.h`

```cpp
namespace xllm::kernel::npu {

// 从 torch::Tensor 创建 ACL tensor 描述符
aclTensor* create_acltensor(const torch::Tensor& tensor, const std::vector<int64_t>& view_dims);

// 类型映射：torch::ScalarType -> aclDataType
namespace type_info {
  aclDataType get_acl_type(torch::ScalarType type);
}

// Tensor 检查
void check_tensor(const torch::Tensor& tensor, const std::string& name);
}
```

**rtp-llm 借鉴**：如果需要在 C++ 端调用 `aclnn` API，这些工具函数可以直接复用。

---

### 6.5 ATB Workspace 管理

**文件**: `xllm/core/layers/npu/buffer/atb_workspace.cpp`, `atb_buffer.cpp`

ATB 操作需要 workspace buffer，xllm 实现了自动增长的共享 workspace pool：

```cpp
// AtbWorkspace: per-device workspace pool
void* AtbWorkspace::get_workspace_buffer(uint64_t bufferSize) {
  return buffer_map_[device_id]->get_buffer(bufferSize);  // 自动扩容
}

// AtbBuffer: torch tensor 支持的自动扩容 buffer
torch::Tensor AtbBuffer::create_attensor(uint64_t buffer_size) const {
  return at_npu::native::empty_with_format(
      at::IntArrayRef({KB_1, buffer_size / KB_1 + 1}),
      options_, ACL_FORMAT_ND);
}
```

**rtp-llm 借鉴**：如果使用 ATB 算子，workspace 管理是必需的。torch_npu 路径下 workspace 由算子内部管理，不需要显式分配。

---

## 七、构建系统

### 7.1 CMake 编译选项（xllm）

**文件**: `xllm/CMakeLists.txt`

```cmake
option(USE_NPU "Enable NPU support" OFF)
option(USE_CUDA "Enable CUDA support" OFF)
# ...

if(USE_NPU)
  add_definitions(-DUSE_NPU -DBUILD_LIBTORCH -DTORCH_SEFCUSTOMHANDLER=ON -DTORCH_HIGHER_THAN_PTA6)

  # 环境变量
  # ASCEND_HOME_PATH          - CANN 工具包路径
  # PYTORCH_NPU_INSTALL_PATH  - torch_npu 安装路径
  # ATB_HOME_PATH             - ATB 库路径
  # NPU_TOOLKIT_HOME          - Ascend 工具包路径

  # ATB layers 子模块
  add_subdirectory(third_party/xllm_atb_layers)

  # 自定义 AscendC 算子预编译
  execute_process(COMMAND bash ${CMAKE_SOURCE_DIR}/third_party/xllm_ops/build.sh)

  # 包含路径
  include_directories($ENV{PYTORCH_NPU_INSTALL_PATH}/include)
  include_directories($ENV{NPU_HOME_PATH}/include)
  include_directories($ENV{ATB_HOME_PATH}/include)

  # 链接路径
  link_directories($ENV{NPU_HOME_PATH}/lib64)
  link_directories($ENV{ATB_HOME_PATH}/lib)

  # 链接库
  find_library(NPU_RUNTIME_SO NAMES "runtime" PATHS $ENV{NPU_HOME_PATH}/lib64)
  target_link_libraries(xllm cust_opapi ${NPU_RUNTIME_SO} nnopbase)
endif()
```

### 7.2 rtp-llm Bazel 对应关系

rtp-llm 使用 Bazel 构建，对应配置：

```python
# .bazelrc
build:npu --define=torch_npu_path=$PYTORCH_NPU_INSTALL_PATH
build:npu --define=ascend_home=$ASCEND_HOME_PATH
build:npu --define=atb_home=$ATB_HOME_PATH
build:npu --copt=-DUSING_ASCEND

# def.bzl 或 BUILD
torch_npu_deps = [
    "@local_torch_npu//:torch_npu",
]
ascend_deps = [
    "@local_ascend//:runtime",
    "@local_ascend//:cust_opapi",
]
```

**rtp-llm 借鉴**：需要在 Bazel 配置中增加 NPU 编译选项，引入 torch_npu 和 CANN 依赖。具体配置方式取决于 rtp-llm 现有的 Bazel workspace 结构。

---

## 八、需要谨慎的部分

### 8.1 ATB vs torch_npu 路径取舍

xllm **重度依赖 ATB C++ 算子**——每个模型都有独立的 ATB decoder layer 实现（`NpuQwen2DecoderLayer`, `NpuLlamaDecoderLayer` 等），通过 ATB 的 Node + VariantPack 模式将整个 decoder layer 编译为单一 ATB operation graph。

rtp-llm 已选择 torch_npu Python API 路径，因此：

| xllm 模式 | rtp-llm 应对 |
|-----------|-------------|
| ATB monolithic decoder op | **不需要**。rtp-llm 用 torch_npu 算子 + Python attention 实现 |
| `atb_speed::qwen::DecoderLayer(param, &op)` | **不需要**。rtp-llm 用 `torch_npu._npu_paged_attention` 等 |
| `Node` + `VariantPack` binding | **不需要**。rtp-llm 的 tensor 通过 Python 传递 |
| `atb::Context` + workspace | **不需要**。torch_npu 算子内部管理 |
| `BaseLoader` + `BaseManualLoader` | **参考但不照搬**。rtp-llm 用 Python 端加载 |
| ACL Graph `AclGraphExecutorImpl` | **参考**。persistent param + bucket 模式有价值 |
| `DeviceCaptureLock` | **必须引入** |
| `NPULayerSynchronizer` | **参考**。PD 分离场景需要 |

### 8.2 xllm ATB 模式的优缺点

**优点**（rtp-llm 可借鉴但不必须）：
- 单一 ATB op 融合整个 decoder layer，减少 kernel launch 开销
- ATB 内部自动优化算子调度和内存复用
- Prefill/Decode 双节点参数化灵活

**缺点**（rtp-llm 选择 torch_npu 的原因）：
- 每个新模型需要独立的 C++ decoder layer 实现（xllm 有 ~22 个 NPU layer 文件）
- 依赖 ATB SDK 的算子支持和更新节奏
- 调试困难（ATB 内部执行对开发者不透明）
- 开发周期长（C++ 编译 + ATB 调试）

### 8.3 rtp-llm 仍可参考的 xllm 模式

即使不采用 ATB 路径，以下模式仍有价值：

1. **ACL Graph 的 persistent param + bucket 策略** — 通用模式
2. **Device Capture Lock** — NPU Graph 安全必需
3. **NPU 内存格式处理（ND/NZ）** — KV cache 必需
4. **Device 抽象类** — 替代散落的条件编译
5. **`torch::kPrivateUse1` 设备类型** — torch_npu 使用方式
6. **`torch::empty` + `npu_format_cast`** — NPU tensor 创建模式
7. **MSPTI/MSTX profiling** — 性能调优工具
8. **`aclrtEvent` 逐层同步** — PD 分离场景
9. **Rolling weight buffer** — 大模型 offloading
10. **`lccl` vs `hccl` 后端选择** — 分布式通信优化

---

## 九、NPU Kernel API 参考

### 9.1 xllm NPU Kernel 命名空间

**文件**: `xllm/core/kernels/npu/npu_ops_api.h`

```cpp
namespace xllm::kernel::npu {
  // Attention
  void reshape_paged_cache(torch::Tensor& key, torch::Tensor& value,
                           torch::Tensor& k_cache, torch::Tensor& v_cache,
                           const torch::Tensor& slot_mapping);
  void batch_prefill(const torch::Tensor& query, ...);
  void batch_decode(const torch::Tensor& query, ...);
  void batch_decode_acl_graph(const torch::Tensor& query, ...,
                              const torch::Tensor& tiling_data, ...);

  // Matmul
  torch::Tensor matmul(const torch::Tensor& a, const torch::Tensor& b, ...);

  // MoE
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  apply_moe_gating_topk_softmax(const torch::Tensor& x, int64_t top_k, ...);
  std::vector<torch::Tensor> apply_npu_grouped_matmul(...);

  // Activation
  torch::Tensor silu_and_mul(const torch::Tensor& x);
  torch::Tensor gelu_and_mul(const torch::Tensor& x);
}
```

### 9.2 对应的 torch_npu API（rtp-llm 使用）

| xllm ATB API | torch_npu 等效 API | 用途 |
|--------------|-------------------|------|
| `atb::npu_reshape_and_cache` | `torch_npu._npu_reshape_and_cache` | KV cache 写入 |
| `atb::npu_paged_attention` | `torch_npu._npu_paged_attention` | Decode paged attention |
| `atb::npu_flash_attention` | `torch_npu.npu_fused_infer_attention_score` | Prefill attention |
| `atb::npu_custom_paged_attention` | `torch_npu._npu_paged_attention` + workspace | ACL Graph decode |
| ATB `BlockCopy` | `torch.index_copy_` / `torch.index_select` | Block swap |

---

## 十、xllm 支持的 NPU 模型清单

xllm 已适配昇腾的模型（22 个 LLM + 多个 VLM）：

| 模型 | ATB 实现 | Torch 实现 | NPU Layer 文件 |
|------|---------|-----------|---------------|
| Qwen2 | ✅ | - | `npu_qwen2_decoder_layer_impl` |
| Qwen3 | ✅ | ✅ | `npu_qwen3_decoder_layer_impl` |
| Qwen3 MoE | ✅ | - | `npu_qwen3_moe_decoder_layer_impl` |
| LLaMA / Llama3 | ✅ | - | `npu_llama_decoder_layer_impl` |
| GLM4 | ✅ | - | `npu_glm4_decoder_layer_impl` |
| GLM4 MoE | ✅ | - | `npu_glm4_moe_decoder_layer_impl` |
| DeepSeek-V2 | ✅ | - | `npu_deepseek_v2_decoder_layer_impl` |
| DeepSeek-V3.2 | ✅ | - | `npu_deepseek_v32_decoder_layer_impl` |
| Qwen3.5 (Linear Attn) | - | ✅ | `npu_torch/` 路径 |
| Eagle3 (Speculative) | ✅ | - | `npu_eagle3_decoder_layer_impl` |
| Qwen2-VL | ✅ | - | Vision encoder |
| Qwen3-VL | ✅ | - | Vision encoder |

**rtp-llm 借鉴**：xllm 的模型覆盖范围证明了昇腾 NPU 可以支持主流 LLM 模型。rtp-llm 优先适配 Qwen2/Qwen3 和 DeepSeek-V2/V3 的路径是合理的。

---

## 附录：关键全局 Flags

| Flag | 默认值 | 说明 |
|------|--------|------|
| `--npu_kernel_backend` | `AUTO` | NPU 内核后端：`AUTO`/`ATB`/`TORCH` |
| `--enable_graph` | `false` | 启用 ACL Graph（decode 加速） |
| `--enable_graph_mode_decode_no_padding` | `false` | Graph 模式不做 bucket padding |
| `--enable_chunked_prefill` | `false` | 分块 prefill |
| `--enable_prefix_cache` | `false` | Prefix cache（影响 NZ 格式选择） |
| `--enable_block_copy_kernel` | `false` | 使用 ATB BlockCopy 算子 |
| `--enable_manual_loader` | `false` | Pinned host + 异步 H2D 权重加载 |
| `--enable_rolling_load` | `false` | Rolling weight offloading |
| `--rolling_load_num_cached_layers` | `2` | HBM 中缓存的 decoder layer 数 |
| `--enable_xtensor` | `false` | XTensor 虚拟内存管理 |
| `--enable_customize_mla_kernel` | `false` | 自定义 MLA kernel（ACL Graph 兼容） |
| `--communication_backend` | `hccl` | 通信后端：`lccl`/`hccl` |
| `--max_tokens_for_graph_mode` | `2048` | Graph 模式最大 token 数 |

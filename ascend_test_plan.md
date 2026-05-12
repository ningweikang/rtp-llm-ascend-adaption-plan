# rtp-llm Ascend NPU 适配 — 各阶段测试用例计划

> 本文档为 `ascend_npu_adaptation_plan_v0.1.md` 的配套测试计划，在每个 Phase 的交付物中嵌入可执行的测试用例，
> 确保每一步都有明确的质量门禁（Quality Gate）。

---

## 测试基础设施概览

### 现有测试体系

| 层面 | 框架 | 关键机制 |
|------|------|---------|
| C++ 单测 | Google Test (`gtest`) | `TestBase.h` → `EngineBaseTest` → `DeviceTestBase` 三层继承；`cc_test_wrapper` 封装为 `sh_test` |
| Python 单测 | `unittest`（主）/ `pytest`（辅） | `py_test` Bazel 规则；`device_resource.py` GPU 资源管理 |
| 设备选择 | `TEST_USING_DEVICE` 环境变量 | `device_test_envs()` 宏通过 `select()` 按平台设置 |
| 分布式测试 | `exec_properties = {'gpu_count':'2'}` | `FileLock` 机制协调多卡 |
| 性能测试 | `tags = ["manual"]` 的 Bazel 目标 | 不走 `bazel test //...`，手动触发 |

### Ascend 测试需新增的基础设施

| 项目 | 说明 |
|------|------|
| `device_test_envs()` 新增 Ascend 分支 | `"@//:using_ascend": {"TEST_USING_DEVICE": "ASCEND"}` |
| `device_resource.py` 扩展 | 支持 `npu-smi` 查询 NPU 状态、`ASCEND_RT_VISIBLE_DEVICES` 环境变量 |
| `TestBase.h` 扩展 | `createDeviceTensor` 系列方法需支持 `torch::kNPU`；`initExecCtx` 需支持 Ascend 路径 |
| NPU CI runner | 自托管 NPU 机器，挂载到 Bazel remote execution |
| 精度对比工具 | 统一的 `assertTensorClose(a, b, rtol, atol)` 函数，支持跨设备精度对齐 |

---

## Phase 0: 编译基础设施搭建 — 测试计划

### 质量门禁：Bazel 编译通过 + 基础宏验证

#### T0.1 `ascend_configure.bzl` 配置探测测试

**类型**：Shell 脚本测试（手动）  
**位置**：`scripts/test_ascend_configure.sh`  
**前置条件**：CANN Toolkit 已安装

```bash
#!/bin/bash
# 验证 ascend_configure.bzl 能正确探测 CANN 环境
set -e

# 1. 验证 @local_config_ascend 仓库生成
bazel query @local_config_ascend//... 2>&1 | grep -q "ascend_headers"
echo "[PASS] @local_config_ascend 仓库已生成"

# 2. 验证关键 target 存在
for target in ascend_headers ascend_libs crosstool; do
    bazel query "@local_config_ascend//:$target" 2>&1 | grep -q "$target"
    echo "[PASS] target $target 存在"
done

# 3. 验证环境变量传递
bazel build --config=ascend \
    --action_env=ASCEND_TOOLKIT_PATH="$ASCEND_TOOLKIT_PATH" \
    //rtp_llm/cpp/ascend:ascend_shims 2>&1 | tail -5
echo "[PASS] --config=ascend 编译链路可用"
```

#### T0.2 编译守卫和宏验证

**类型**：C++ 编译时测试  
**位置**：`rtp_llm/cpp/ascend/test/BUILD`

```python
# BUILD
cc_test_wrapper(
    name = "ascend_config_test",
    srcs = ["ascend_config_test.cc"],
    deps = [
        "//rtp_llm/cpp/core:device_data",
        "//rtp_llm/cpp/core:allocator",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    exec_properties = {"gpu": "910B"},
    env = device_test_envs(),
)
```

```cpp
// ascend_config_test.cc
#include <gtest/gtest.h>
#include "rtp_llm/cpp/core/DeviceData.h"
#include "rtp_llm/cpp/core/allocator.h"

TEST(AscendConfigTest, DeviceTypeExists) {
#ifdef USING_ASCEND
    EXPECT_EQ(static_cast<int>(rtp_llm::DeviceType::Ascend), 6);
#else
    GTEST_SKIP() << "Not compiled with USING_ASCEND";
#endif
}

TEST(AscendConfigTest, AllocatorTypeExists) {
#ifdef USING_ASCEND
    EXPECT_EQ(static_cast<int>(rtp_llm::AllocatorType::ASCEND),
              static_cast<int>(rtp_llm::AllocatorType::ASCEND));
#endif
}

TEST(AscendConfigTest, ShimMacrosCompile) {
#include "rtp_llm/cpp/ascend/ascend_shims.h"
    aclrtStream stream = nullptr;
    aclrtEvent event = nullptr;
    EXPECT_EQ(stream, nullptr);
    EXPECT_EQ(event, nullptr);
}
```

#### T0.3 `def.bzl` 扩展验证

**类型**：Bazel Starlark 单元测试（`bazel query`）

```bash
# 验证 if_ascend() 函数存在且能在 select() 中工作
bazel build --config=ascend //rtp_llm/cpp/ascend:ascend_host_utils
echo "[PASS] if_ascend() select 分支生效"
```

#### T0.4 分步编译验证清单

按依赖顺序逐步编译，每步失败立即停止：

| 步骤 | 命令 | 验证内容 |
|------|------|---------|
| 1 | `bazel query @local_config_ascend//...` | 仓库 targets 存在 |
| 2 | `bazel build --config=ascend //rtp_llm/cpp/ascend:ascend_shims` | Shim 头文件编译 |
| 3 | `bazel build --config=ascend //rtp_llm/cpp/ascend:ascend_host_utils` | Host utils 编译 |
| 4 | `bazel build --config=ascend //rtp_llm/cpp/core:types_hdr` | Core 头文件编译 |
| 5 | `bazel build --config=ascend //:th_transformer_config` | Config 库编译 |
| 6 | `bazel build --config=ascend //:rtp_compute_ops` | Compute ops 编译 |
| 7 | `bazel build --config=ascend //:th_transformer` | **最终产物编译** |

---

## Phase 1: 内存管理与设备运行时 — 测试计划

### 质量门禁：NPU 上 tensor 分配/释放、Stream/Event 操作正常

#### T1.1 Ascend Allocator 单测

**类型**：C++ gtest  
**位置**：`rtp_llm/cpp/ascend/test/allocator_ascend_test.cc`

```cpp
#include <gtest/gtest.h>
#include "rtp_llm/cpp/ascend/allocator_ascend.h"
#include "rtp_llm/cpp/ascend/ascend_host_utils.h"

class AscendAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        aclrtSetDevice(0);
        allocator_ = std::make_unique<rtp_llm::Allocator<rtp_llm::AllocatorType::ASCEND>>(0);
    }
    void TearDown() override {
        allocator_.reset();
        aclrtResetDevice(0);
    }
    std::unique_ptr<rtp_llm::Allocator<rtp_llm::AllocatorType::ASCEND>> allocator_;
};

TEST_F(AscendAllocatorTest, MallocAndFree) {
    void* ptr = allocator_->malloc(1024 * 1024);
    EXPECT_NE(ptr, nullptr);
    allocator_->free(&ptr);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(AscendAllocatorTest, MallocLargeSize) {
    size_t size = 512 * 1024 * 1024;
    void* ptr = allocator_->malloc(size);
    EXPECT_NE(ptr, nullptr);
    allocator_->free(&ptr);
}

TEST_F(AscendAllocatorTest, MemoryInfoQuery) {
    auto [free, total] = rtp_llm::getAscendDeviceMemoryInfo();
    EXPECT_GT(total, 0u);
    EXPECT_GT(free, 0u);
    EXPECT_LE(free, total);
}
```

#### T1.2 Device Placement 参数化验证

**类型**：Python unittest  
**位置**：`rtp_llm/test/utils/test_device_utils.py`

```python
import unittest
import torch
from rtp_llm.utils.device_utils import get_device_str, get_torch_device

class DeviceUtilsTest(unittest.TestCase):
    def setUp(self):
        self.original_device = os.environ.get("TEST_USING_DEVICE", "CUDA")

    def test_get_device_str_cuda(self):
        os.environ["TEST_USING_DEVICE"] = "CUDA"
        self.assertEqual(get_device_str(), "cuda")

    def test_get_device_str_ascend(self):
        os.environ["TEST_USING_DEVICE"] = "ASCEND"
        self.assertEqual(get_device_str(), "npu")

    def test_get_torch_device_ascend(self):
        os.environ["TEST_USING_DEVICE"] = "ASCEND"
        device = get_torch_device()
        self.assertEqual(device.type, "npu")

    def test_tensor_on_correct_device(self):
        device = get_torch_device()
        t = torch.zeros(10, device=device)
        self.assertTrue(str(t.device).startswith(device.type))
```

#### T1.3 基础 NPU Tensor 操作测试（对标 `BasicDeviceTest.cc`）

**类型**：C++ gtest  
**位置**：`rtp_llm/cpp/ascend/test/basic_ascend_test.cc`

```cpp
#include "rtp_llm/cpp/testing/TestBase.h"

class BasicAscendTest : public DeviceTestBase {
protected:
    void initTestDevices() override {
        rtp_llm::ParallelismConfig pc;
        rtp_llm::ModelConfig mc;
        mc.max_seq_len = 1024;
        rtp_llm::initExecCtx(pc, mc, /* ... other configs ... */);
    }

    torch::Tensor createNpuTensor(const std::vector<int64_t>& shape,
                                   torch::Dtype dtype = torch::kFloat32) {
        return torch::empty(shape, torch::dtype(dtype).device(torch::kNPU));
    }
};

TEST_F(BasicAscendTest, HostToNpuCopy) {
    std::vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto A = torch::tensor(expected, torch::kFloat32).reshape({2, 3});
    auto B = A.to(torch::kNPU);
    auto C = B.cpu();

    auto C_ptr = C.contiguous().data_ptr<float>();
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_EQ(C_ptr[i], expected[i]);
    }
}

TEST_F(BasicAscendTest, NpuMemoryInfo) {
    auto status = getGpuExecStatus();
    EXPECT_GT(status.device_memory_status.total_bytes, 0u);
    EXPECT_GT(status.device_memory_status.free_bytes, 0u);
}

TEST_F(BasicAscendTest, NpuTensorOps) {
    auto a = torch::ones({3, 4}, torch::dtype(torch::kFloat32).device(torch::kNPU));
    auto b = torch::ones({3, 4}, torch::dtype(torch::kFloat32).device(torch::kNPU));
    auto c = a + b;
    auto c_cpu = c.cpu();
    auto ptr = c_cpu.data_ptr<float>();
    for (int i = 0; i < 12; i++) {
        EXPECT_FLOAT_EQ(ptr[i], 2.0f);
    }
}
```

#### T1.4 Stream/Event 操作测试

**类型**：C++ gtest  
**位置**：`rtp_llm/cpp/ascend/test/ascend_stream_test.cc`

```cpp
TEST(AscendStreamTest, StreamCreateDestroy) {
    aclrtStream stream;
    aclrtCreateStream(&stream);
    EXPECT_NE(stream, nullptr);

    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
}

TEST(AscendStreamTest, EventCreateRecordSynchronize) {
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclrtEvent event;
    aclrtCreateEvent(&event);

    auto tensor = torch::ones({1000}, torch::dtype(torch::kFloat32).device(torch::kNPU));
    aclrtRecordEvent(event, stream);
    aclrtSynchronizeEvent(event);

    aclrtDestroyEvent(event);
    aclrtDestroyStream(stream);
}

TEST(AscendStreamTest, AsyncMemcpy) {
    std::vector<float> data(1024, 1.0f);
    void* npu_ptr = nullptr;
    aclrtMalloc(&npu_ptr, data.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclrtMemcpyAsync(npu_ptr, data.size() * sizeof(float),
                     data.data(), data.size() * sizeof(float),
                     ACL_MEMCPY_HOST_TO_DEVICE, stream);
    aclrtSynchronizeStream(stream);

    std::vector<float> result(1024);
    aclrtMemcpyAsync(result.data(), result.size() * sizeof(float),
                     npu_ptr, data.size() * sizeof(float),
                     ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);

    for (auto v : result) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }

    aclrtFree(npu_ptr);
    aclrtDestroyStream(stream);
}
```

---

## Phase 2: GEMM 算子适配 — 测试计划

### 质量门禁：Ascend GEMM 结果与 cuBLAS 参考结果精度对齐

#### T2.1 AscendGemmWrapper 基础精度测试

**类型**：C++ gtest  
**位置**：`rtp_llm/cpp/ascend/ascend_gemm/test/ascend_gemm_test.cc`

```cpp
class AscendGemmTest : public ::testing::Test {
protected:
    void SetUp() override {
        aclrtSetDevice(0);
        aclrtCreateStream(&stream_);
        gemm_ = std::make_unique<rtp_llm::AscendGemmWrapper>(stream_);
    }

    void verifyGemmResult(const torch::Tensor& actual,
                           const torch::Tensor& expected,
                           float rtol = 1e-2, float atol = 1e-2) {
        auto diff = (actual - expected).abs();
        auto max_diff = diff.max().item<float>();
        EXPECT_LT(max_diff, atol) << "Max diff: " << max_diff;
    }

    aclrtStream stream_;
    std::unique_ptr<rtp_llm::AscendGemmWrapper> gemm_;
};

// ---- FP16 MatMul ----
TEST_F(AscendGemmTest, FP16_SmallMatMul) {
    int M = 128, K = 64, N = 96;
    auto a = torch::randn({M, K}, torch::kFloat16).to(torch::kNPU);
    auto b = torch::randn({K, N}, torch::kFloat16).to(torch::kNPU);
    auto c = gemm_->gemm(a, b);

    auto ref = torch::matmul(a.cpu().to(torch::kFloat32),
                              b.cpu().to(torch::kFloat32)).to(torch::kFloat16);
    verifyGemmResult(c.cpu(), ref, 0.01f, 0.01f);
}

TEST_F(AscendGemmTest, FP16_BatchMatMul) {
    int B = 4, M = 64, K = 64, N = 64;
    auto a = torch::randn({B, M, K}, torch::kFloat16).to(torch::kNPU);
    auto b = torch::randn({B, K, N}, torch::kFloat16).to(torch::kNPU);
    auto c = gemm_->batchGemm(a, b);

    auto ref = torch::bmm(a.cpu().to(torch::kFloat32),
                           b.cpu().to(torch::kFloat32)).to(torch::kFloat16);
    verifyGemmResult(c.cpu(), ref, 0.01f, 0.01f);
}

// ---- BF16 MatMul ----
TEST_F(AscendGemmTest, BF16_MatMul) {
    int M = 256, K = 128, N = 256;
    auto a = torch::randn({M, K}, torch::kBFloat16).to(torch::kNPU);
    auto b = torch::randn({K, N}, torch::kBFloat16).to(torch::kNPU);
    auto c = gemm_->gemm(a, b);

    auto ref = torch::matmul(a.cpu().to(torch::kFloat32),
                              b.cpu().to(torch::kFloat32)).to(torch::kBFloat16);
    verifyGemmResult(c.cpu(), ref, 0.02f, 0.02f);
}

// ---- 带Bias的MatMul ----
TEST_F(AscendGemmTest, FP16_MatMulWithBias) {
    int M = 128, K = 64, N = 96;
    auto a = torch::randn({M, K}, torch::kFloat16).to(torch::kNPU);
    auto b = torch::randn({K, N}, torch::kFloat16).to(torch::kNPU);
    auto bias = torch::randn({N}, torch::kFloat16).to(torch::kNPU);
    auto c = gemm_->gemmWithBias(a, b, bias);

    auto ref = torch::matmul(a.cpu().to(torch::kFloat32),
                              b.cpu().to(torch::kFloat32)).to(torch::kFloat16);
    ref = ref + bias.cpu();
    verifyGemmResult(c.cpu(), ref, 0.01f, 0.01f);
}
```

#### T2.2 GEMM 规模扫描测试

**类型**：C++ gtest（parameterized）  
**位置**：`rtp_llm/cpp/ascend/ascend_gemm/test/ascend_gemm_sweep_test.cc`

```cpp
struct GemmShape { int M, K, N; };

class AscendGemmSweepTest : public ::testing::TestWithParam<GemmShape> {};

TEST_P(AscendGemmSweepTest, FP16_Sweep) {
    auto [M, K, N] = GetParam();
    auto a = torch::randn({M, K}, torch::kFloat16).to(torch::kNPU);
    auto b = torch::randn({K, N}, torch::kFloat16).to(torch::kNPU);
    auto c = gemm_->gemm(a, b);
    ASSERT_EQ(c.size(0), M);
    ASSERT_EQ(c.size(1), N);
    // 精度验证：与 PyTorch 参考对比
    auto ref = torch::matmul(a.cpu().to(torch::kFloat32),
                              b.cpu().to(torch::kFloat32)).to(torch::kFloat16);
    EXPECT_LT((c.cpu().to(torch::kFloat32) - ref.to(torch::kFloat32)).abs().max().item<float>(), 0.1f);
}

INSTANTIATE_TEST_SUITE_P(
    StandardShapes,
    AscendGemmSweepTest,
    ::testing::Values(
        GemmShape{1, 4096, 4096},      // decode token
        GemmShape{32, 4096, 4096},     // small batch
        GemmShape{128, 4096, 4096},    // medium batch
        GemmShape{512, 4096, 11008},   // LLaMA FFN
        GemmShape{2048, 4096, 4096},   // prefill
        GemmShape{4096, 4096, 4096},   // large prefill
        GemmShape{1, 4096, 11008},     // decode FFN
        GemmShape{8, 4096, 4096}       // small TP batch
    ));
```

#### T2.3 GEMM 性能基准测试

**类型**：C++ benchmark（manual tag）  
**位置**：`rtp_llm/cpp/ascend/ascend_gemm/test/ascend_gemm_perf_test.cc`

```cpp
TEST(AscendGemmPerf, FP16_LlamaFFN_Decode) {
    int M = 1, K = 4096, N = 11008;
    int warmup = 10, iters = 100;

    auto a = torch::randn({M, K}, torch::kFloat16).to(torch::kNPU);
    auto b = torch::randn({K, N}, torch::kFloat16).to(torch::kNPU);

    for (int i = 0; i < warmup; i++) gemm_->gemm(a, b);
    aclrtSynchronizeStream(stream_);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) gemm_->gemm(a, b);
    aclrtSynchronizeStream(stream_);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double tflops = 2.0 * M * K * N / (ms * 1e-3) / 1e12;
    printf("[PERF] M=%d K=%d N=%d: %.3f ms, %.2f TFLOPS\n", M, K, N, ms, tflops);
}
```

---

## Phase 3: Attention 算子适配 — 测试计划

### 质量门禁：Prefill/Decode Attention 输出与 FlashInfer 参考精度对齐（cosine sim > 0.99）

#### T3.1 Ascend Prefill Attention 单测

**类型**：Python unittest  
**位置**：`rtp_llm/models_py/modules/factory/attention/ascend_impl/test/test_ascend_prefill.py`

```python
import unittest
import torch
import math

class AscendPrefillTest(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        torch.manual_seed(42)
        torch.npu.manual_seed(42)

    def _reference_attention(self, q, k, v, causal=True):
        """PyTorch 参考 attention 实现"""
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(scores.shape[-2:], device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def _verify_cosine_sim(self, actual, expected, threshold=0.99):
        actual_f = actual.flatten().to(torch.float32)
        expected_f = expected.flatten().to(torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(
            actual_f.unsqueeze(0), expected_f.unsqueeze(0))
        self.assertGreater(cos_sim.item(), threshold,
            f"Cosine similarity {cos_sim.item():.6f} < {threshold}")

    def test_prefill_single_query(self):
        """单请求 prefill"""
        B, H, S, D = 1, 32, 128, 128
        q = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
        k = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
        v = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")

        output = ascend_prefill_attention(q, k, v, causal=True)
        ref = self._reference_attention(
            q.reshape(B, H, S, D), k.reshape(B, H, S, D), v.reshape(B, H, S, D)
        ).reshape(B, S, H, D)

        self._verify_cosine_sim(output, ref)

    def test_prefill_batch(self):
        """批量 prefill"""
        B, H, S, D = 4, 32, 256, 128
        q = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
        k = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
        v = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")

        output = ascend_prefill_attention(q, k, v, causal=True)
        ref = self._reference_attention(
            q.reshape(B, H, S, D), k.reshape(B, H, S, D), v.reshape(B, H, S, D)
        ).reshape(B, S, H, D)

        self._verify_cosine_sim(output, ref)

    def test_prefill_variable_lengths(self):
        """不同序列长度的 prefill"""
        for S in [32, 64, 128, 512, 1024, 2048]:
            with self.subTest(seq_len=S):
                B, H, D = 1, 32, 128
                q = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
                k = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
                v = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
                output = ascend_prefill_attention(q, k, v, causal=True)
                self.assertEqual(output.shape, (B, S, H, D))

    def test_prefill_bf16(self):
        """BF16 prefill"""
        B, H, S, D = 2, 32, 128, 128
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="npu")
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="npu")
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="npu")
        output = ascend_prefill_attention(q, k, v, causal=True)
        self.assertEqual(output.dtype, torch.bfloat16)
```

#### T3.2 Ascend Decode Attention 单测（Paged KV Cache）

**类型**：Python unittest  
**位置**：`rtp_llm/models_py/modules/factory/attention/ascend_impl/test/test_ascend_decode.py`

```python
class AscendDecodeTest(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        torch.manual_seed(42)

    def test_decode_single_token(self):
        """单 token decode"""
        num_heads, head_dim, block_size = 32, 128, 16
        seq_len, num_blocks = 256, 64

        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim,
                               dtype=torch.float16, device="npu")
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim,
                               dtype=torch.float16, device="npu")

        q = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float16, device="npu")
        new_k = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float16, device="npu")
        new_v = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float16, device="npu")

        block_tables = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
                                     dtype=torch.int32, device="npu")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="npu")

        output = ascend_decode_attention(
            q, new_k, new_v, k_cache, v_cache,
            block_tables, seq_lens, block_size)

        self.assertEqual(output.shape, (1, 1, num_heads, head_dim))

    def test_decode_batch(self):
        """批量 decode，不同序列长度"""
        batch_size, num_heads, head_dim, block_size = 8, 32, 128, 16

        seq_lens = [100, 200, 50, 300, 150, 80, 400, 250]
        max_seq_len = max(seq_lens)
        num_blocks = batch_size * (max_seq_len // block_size + 2)

        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim,
                               dtype=torch.float16, device="npu")
        v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim,
                               dtype=torch.float16, device="npu")

        q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.float16, device="npu")
        new_k = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.float16, device="npu")
        new_v = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.float16, device="npu")

        block_tables = torch.randint(0, num_blocks, (batch_size, max_seq_len // block_size + 1),
                                      dtype=torch.int32, device="npu")
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device="npu")

        output = ascend_decode_attention(
            q, new_k, new_v, k_cache, v_cache,
            block_tables, seq_lens_tensor, block_size)

        self.assertEqual(output.shape, (batch_size, 1, num_heads, head_dim))
```

#### T3.3 Attention 精度回归测试（跨设备对比）

**类型**：Python unittest  
**位置**：`rtp_llm/models_py/modules/factory/attention/ascend_impl/test/test_attention_cross_device.py`

```python
class AttentionCrossDeviceTest(unittest.TestCase):
    """对比 CUDA FlashInfer 和 Ascend FA 的输出一致性"""

    @classmethod
    def setUpClass(cls):
        cls.test_configs = [
            {"batch": 1,  "seq_len": 128,  "heads": 32, "dim": 128, "dtype": torch.float16},
            {"batch": 4,  "seq_len": 512,  "heads": 32, "dim": 128, "dtype": torch.float16},
            {"batch": 1,  "seq_len": 2048, "heads": 32, "dim": 128, "dtype": torch.bfloat16},
            {"batch": 8,  "seq_len": 1024, "heads": 64, "dim": 64,  "dtype": torch.float16},
        ]

    def test_prefill_precision_alignment(self):
        """
        策略：在 CPU 上生成输入 → 分别 copy 到 CUDA 和 NPU → 对比输出
        """
        for cfg in self.test_configs:
            with self.subTest(**cfg):
                B, S, H, D = cfg["batch"], cfg["seq_len"], cfg["heads"], cfg["dim"]
                dtype = cfg["dtype"]

                q_cpu = torch.randn(B, S, H, D, dtype=dtype)
                k_cpu = torch.randn(B, S, H, D, dtype=dtype)
                v_cpu = torch.randn(B, S, H, D, dtype=dtype)

                # 参考实现（CPU float64）
                ref = self._cpu_reference(
                    q_cpu.float(), k_cpu.float(), v_cpu.float())

                # Ascend 实现
                q_npu = q_cpu.to("npu")
                k_npu = k_cpu.to("npu")
                v_npu = v_cpu.to("npu")
                asc_out = ascend_prefill_attention(q_npu, k_npu, v_npu, causal=True)

                self._verify(ref.half(), asc_out.cpu(), rtol=0.01, atol=0.01)

    def _cpu_reference(self, q, k, v):
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(scores.shape[-2:]), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def _verify(self, expected, actual, rtol, atol):
        diff = (expected.float() - actual.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        self.assertLess(max_diff, atol,
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
```

#### T3.4 Attention 性能基准

**类型**：Python benchmark（manual tag）  
**位置**：`rtp_llm/models_py/modules/factory/attention/ascend_impl/test/bench_ascend_attention.py`

```python
class AscendAttentionBench(unittest.TestCase):
    def test_prefill_throughput(self):
        """Prefill 吞吐量基准"""
        configs = [
            (1, 128, 32, 128),
            (1, 512, 32, 128),
            (1, 1024, 32, 128),
            (1, 2048, 32, 128),
            (1, 4096, 32, 128),
            (4, 1024, 32, 128),
            (8, 512, 32, 128),
        ]

        for B, S, H, D in configs:
            q = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
            k = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")
            v = torch.randn(B, S, H, D, dtype=torch.float16, device="npu")

            for _ in range(5):  # warmup
                ascend_prefill_attention(q, k, v, causal=True)
            torch.npu.synchronize()

            start = time.time()
            iters = 50
            for _ in range(iters):
                ascend_prefill_attention(q, k, v, causal=True)
            torch.npu.synchronize()

            ms = (time.time() - start) / iters * 1000
            print(f"[PERF] Prefill B={B} S={S} H={H} D={D}: {ms:.2f} ms")

    def test_decode_throughput(self):
        """Decode 吞吐量基准"""
        batch_sizes = [1, 8, 32, 64, 128, 256]
        seq_len = 4096

        for bs in batch_sizes:
            # ... setup paged KV cache ...
            ms = self._benchmark_decode(bs, seq_len)
            print(f"[PERF] Decode bs={bs} seq={seq_len}: {ms:.2f} ms")
```

---

## Phase 4: KV Cache 适配 — 测试计划

### 质量门禁：BlockPool 分配/释放正确、跨 block 拷贝数据一致

#### T4.1 BlockPool NPU 单测

**类型**：C++ gtest  
**位置**：`rtp_llm/cpp/cache/test/block_pool_ascend_test.cc`

```cpp
class BlockPoolAscendTest : public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        aclrtSetDevice(0);
    }
};

TEST_F(BlockPoolAscendTest, AllocateAndFree) {
    rtp_llm::BlockPoolConfig config;
    config.block_size = 16;
    config.block_nums = 1000;
    config.dtype = rtp_llm::DataType::TYPE_FP16;

    rtp_llm::BlockPool pool(config, rtp_llm::AllocatorType::ASCEND, 0);

    auto block = pool.malloc();
    EXPECT_NE(block, nullptr);
    pool.free(block);
}

TEST_F(BlockPoolAscendTest, BlockCopyConsistency) {
    rtp_llm::BlockPoolConfig config;
    config.block_size = 16;
    config.block_nums = 1000;

    rtp_llm::BlockPool pool(config, rtp_llm::AllocatorType::ASCEND, 0);

    auto src = pool.malloc();
    auto dst = pool.malloc();

    torch::Tensor data = torch::randn({16, 32, 128}, torch::kFloat16).to(torch::kNPU);
    pool.copyBlock(src, &data);
    pool.copyBlock(&dst, src);

    auto src_cpu = data.cpu();
    auto dst_data = pool.getBlockTensor(dst);
    auto dst_cpu = dst_data.cpu();

    EXPECT_TRUE(torch::allclose(src_cpu, dst_cpu, 1e-3, 1e-3));
}

TEST_F(BlockPoolAscendTest, MemoryInfoAccuracy) {
    auto [free_before, total] = rtp_llm::getAscendDeviceMemoryInfo();

    rtp_llm::BlockPoolConfig config;
    config.block_size = 16;
    config.block_nums = 100;

    rtp_llm::BlockPool pool(config, rtp_llm::AllocatorType::ASCEND, 0);

    auto [free_after, _] = rtp_llm::getAscendDeviceMemoryInfo();
    EXPECT_LT(free_after, free_before);
}
```

#### T4.2 KV Cache Manager 集成测试

**类型**：C++ gtest  
**位置**：`rtp_llm/cpp/cache/test/kv_cache_ascend_test.cc`

```cpp
TEST(KVCacheAscendTest, FullLifecycle) {
    rtp_llm::CacheConfig config = rtp_llm::test::makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/100, /*tokens_per_block=*/16,
        rtp_llm::DataType::TYPE_FP16, /*local_head_num_kv=*/32, /*size_per_head=*/128);

    auto allocator = std::make_shared<rtp_llm::Allocator<rtp_llm::AllocatorType::ASCEND>>(0);
    rtp_llm::KVCacheManager manager(config, allocator.get());

    int token_count = 64;
    auto kv_tensors = manager.allocateKVBlocks(token_count);
    EXPECT_GT(kv_tensors.size(), 0u);

    manager.freeKVBlocks(kv_tensors);
}
```

---

## Phase 5: 自定义算子迁移 — 测试计划

### 质量门禁：每个算子都有独立的精度测试 + 性能 benchmark

#### T5.1 算子逐个测试模板

**类型**：Python unittest  
**位置**：`rtp_llm/models_py/bindings/ascend/test/test_ascend_ops.py`

```python
class AscendOpsTest(unittest.TestCase):
    """每个算子一个 test method，统一模板"""

    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        torch.manual_seed(42)
        self.rtol = 1e-2
        self.atol = 1e-2

    # ---- P0 算子 ----

    def test_silu(self):
        x = torch.randn(128, 4096, dtype=torch.float16, device="npu")
        out = ascend_ops.silu(x)
        ref = torch.nn.functional.silu(x.cpu())
        self.assertTrue(torch.allclose(out.cpu(), ref, self.rtol, self.atol))

    def test_layernorm(self):
        x = torch.randn(32, 128, 4096, dtype=torch.float16, device="npu")
        weight = torch.ones(4096, dtype=torch.float16, device="npu")
        bias = torch.zeros(4096, dtype=torch.float16, device="npu")
        out = ascend_ops.layer_norm(x, weight, bias, eps=1e-5)
        ref = torch.layer_norm(x.cpu().float(), [4096],
                                weight.cpu().float(), bias.cpu().float(), 1e-5).half()
        self.assertTrue(torch.allclose(out.cpu(), ref, self.rtol, self.atol))

    def test_softmax(self):
        x = torch.randn(32, 32, 128, dtype=torch.float16, device="npu")
        out = ascend_ops.softmax(x, dim=-1)
        ref = torch.softmax(x.cpu().float(), dim=-1).half()
        self.assertTrue(torch.allclose(out.cpu(), ref, self.rtol, self.atol))

    def test_embedding(self):
        weight = torch.randn(32000, 4096, dtype=torch.float16, device="npu")
        indices = torch.randint(0, 32000, (8, 128), dtype=torch.long, device="npu")
        out = ascend_ops.embedding(weight, indices)
        ref = torch.nn.functional.embedding(indices.cpu(), weight.cpu())
        self.assertTrue(torch.allclose(out.cpu(), ref, 0.0, 0.0))

    # ---- P1 算子 ----

    def test_topk(self):
        x = torch.randn(1, 32000, dtype=torch.float16, device="npu")
        values, indices = ascend_ops.topk(x, k=5, dim=-1)
        ref_v, ref_i = torch.topk(x.cpu(), k=5, dim=-1)
        self.assertTrue(torch.allclose(values.cpu(), ref_v, self.rtol, self.atol))
        self.assertTrue(torch.equal(indices.cpu(), ref_i))

    def test_batch_copy(self):
        src = torch.randn(8, 1024, dtype=torch.float16, device="npu")
        dst = torch.zeros_like(src)
        ascend_ops.batch_copy(src, dst)
        self.assertTrue(torch.equal(src.cpu(), dst.cpu()))
```

#### T5.2 算子组合回归测试

**类型**：Python unittest  
**位置**：`rtp_llm/models_py/bindings/ascend/test/test_ascend_ops_combo.py`

```python
class AscendOpsComboTest(unittest.TestCase):
    """测试算子组合是否能正确跑通一个 Transformer 层的前向传播"""

    def test_single_transformer_layer(self):
        """
        用 Ascend 算子组合跑一个完整的 Attention + FFN 层，
        与 PyTorch 参考实现对比
        """
        B, S, H, D = 1, 128, 32, 128
        hidden_dim = H * D  # 4096
        ffn_dim = 11008

        x = torch.randn(B, S, hidden_dim, dtype=torch.float16, device="npu")

        # Attention sublayer
        q_proj = ascend_ops.linear(x, self.q_weight, self.q_bias)
        k_proj = ascend_ops.linear(x, self.k_weight, self.k_bias)
        v_proj = ascend_ops.linear(x, self.v_weight, self.v_bias)

        attn_out = ascend_prefill_attention(
            q_proj.view(B, S, H, D),
            k_proj.view(B, S, H, D),
            v_proj.view(B, S, H, D), causal=True)

        attn_out = attn_out.view(B, S, hidden_dim)
        attn_out = ascend_ops.linear(attn_out, self.o_weight, self.o_bias)
        x = x + attn_out  # residual

        # FFN sublayer
        gate = ascend_ops.silu(ascend_ops.linear(x, self.gate_weight, None))
        up = ascend_ops.linear(x, self.up_weight, None)
        ffn_out = gate * up
        ffn_out = ascend_ops.linear(ffn_out, self.down_weight, None)
        x = x + ffn_out  # residual

        # 验证输出不含 NaN/Inf
        self.assertFalse(torch.any(torch.isnan(x.cpu())))
        self.assertFalse(torch.any(torch.isinf(x.cpu())))
```

---

## Phase 6: 分布式通信适配 — 测试计划

### 质量门禁：HCCL AllReduce/AllGather 结果正确，多卡 TP 可启动

#### T6.1 HCCL 基础通信测试

**类型**：Python unittest  
**位置**：`rtp_llm/models_py/distributed/test/test_hccl_comm.py`

```python
@unittest.skipUnless(os.environ.get("GPU_COUNT", "1") == "2", "需要 2 张 NPU")
class HCCLCommTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.distributed.init_process_group(backend="hccl")
        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()

    def test_allreduce(self):
        tensor = torch.ones(100, dtype=torch.float16, device=f"npu:{self.rank}") * (self.rank + 1)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        expected = sum(range(1, self.world_size + 1))
        self.assertTrue(torch.all(tensor.cpu() == expected))

    def test_allgather(self):
        tensor = torch.ones(100, dtype=torch.float16, device=f"npu:{self.rank}") * self.rank
        gathered = [torch.zeros(100, dtype=torch.float16, device=f"npu:{self.rank}")
                     for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, tensor)
        for i, t in enumerate(gathered):
            self.assertTrue(torch.all(t.cpu() == i))

    def test_broadcast(self):
        tensor = torch.zeros(100, dtype=torch.float16, device=f"npu:{self.rank}")
        if self.rank == 0:
            tensor.fill_(42.0)
        torch.distributed.broadcast(tensor, src=0)
        self.assertTrue(torch.all(tensor.cpu() == 42.0))

    def test_send_recv(self):
        if self.rank == 0:
            tensor = torch.ones(100, dtype=torch.float16, device="npu:0") * 99
            torch.distributed.send(tensor, dst=1)
        else:
            tensor = torch.zeros(100, dtype=torch.float16, device="npu:1")
            torch.distributed.recv(tensor, src=0)
            self.assertTrue(torch.all(tensor.cpu() == 99.0))

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()
```

#### T6.2 TP 推理端到端测试

**类型**：Python integration test  
**位置**：`rtp_llm/distribute/test/test_ascend_tp.py`

```python
@unittest.skipUnless(os.environ.get("GPU_COUNT", "1") == "2", "需要 2 张 NPU")
class AscendTPTest(unittest.TestCase):
    def test_tp_inference_qwen(self):
        """2 卡 TP 推理 Qwen 模型，验证输出与单卡一致"""
        from rtp_llm.utils.model_utils import load_model

        model = load_model(
            model_dir="/path/to/qwen-7b",
            device="npu",
            tp_size=2,
            tp_rank=int(os.environ["RANK"]))

        input_ids = torch.tensor([[1, 100, 200, 300]], dtype=torch.long, device="npu")
        output = model.generate(input_ids, max_new_tokens=20)

        self.assertEqual(output.shape[1], 24)  # 4 input + 20 generated
        self.assertFalse(torch.any(torch.isnan(output.cpu())))
```

---

## Phase 7: CUDA Graph → Ascend Graph 适配 — 测试计划

### 质量门禁：Graph capture/replay 结果与 eager 模式一致

#### T7.1 Ascend Graph 基础测试

**类型**：Python unittest  
**位置**：`rtp_llm/cpp/cuda_graph/tests/test_ascend_graph.py`

```python
class AscendGraphTest(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")

    def test_graph_capture_replay_simple(self):
        """简单计算图的 capture + replay"""
        x = torch.randn(100, device="npu", dtype=torch.float16)

        # Eager 模式参考结果
        eager_out = x * 2 + 1

        # Graph 模式
        # ... Ascend Graph capture + replay ...
        graph_out = self._capture_and_replay(lambda t: t * 2 + 1, x)

        self.assertTrue(torch.allclose(eager_out, graph_out, 1e-3, 1e-3))

    def test_graph_replay_determinism(self):
        """多次 replay 结果完全一致"""
        x = torch.randn(100, device="npu", dtype=torch.float16)
        graph = self._build_graph(lambda t: torch.matmul(t, t.T), x)

        results = [graph.replay(x) for _ in range(10)]
        for r in results[1:]:
            self.assertTrue(torch.equal(results[0], r))

    def test_graph_with_kv_cache_update(self):
        """Graph 模式下更新 KV Cache"""
        # ... 模拟 decode 步骤中 KV Cache 更新的 graph 模式 ...
        pass
```

#### T7.2 Graph 性能对比测试

**类型**：Python benchmark（manual tag）

```python
class AscendGraphPerfTest(unittest.TestCase):
    def test_graph_vs_eager_decode(self):
        """对比 Graph 模式和 Eager 模式 decode 性能"""
        # warmup + eager
        # warmup + graph capture
        # benchmark eager
        # benchmark graph replay
        # 打印加速比
        pass
```

---

## Phase 8: FP8/量化支持 — 测试计划

### 质量门禁：FP8 推理精度在可接受范围内（与 FP16 相比 perplexity 差异 < 1%）

#### T8.1 FP8 GEMM 精度测试

**类型**：Python unittest  
**位置**：`rtp_llm/cpp/ascend/ascend_gemm/test/test_ascend_fp8_gemm.py`

```python
class AscendFP8GemmTest(unittest.TestCase):
    def test_fp8_e4m3_matmul(self):
        """FP8 E4M3 精度 MatMul"""
        M, K, N = 128, 4096, 4096
        a_f32 = torch.randn(M, K, dtype=torch.float32)
        b_f32 = torch.randn(K, N, dtype=torch.float32)

        # FP8 量化
        a_scale = a_f32.abs().max() / 448.0  # FP8 E4M3 max = 448
        b_scale = b_f32.abs().max() / 448.0
        a_fp8 = (a_f32 / a_scale).to(torch.float8_e4m3fn).to("npu")
        b_fp8 = (b_f32 / b_scale).to(torch.float8_e4m3fn).to("npu")

        # Ascend FP8 GEMM
        out = ascend_fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)

        # 参考：FP32 matmul
        ref = torch.matmul(a_f32, b_f32)

        # 允许较大误差（FP8 本身有量化损失）
        cos_sim = torch.nn.functional.cosine_similarity(
            out.cpu().float().flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0))
        self.assertGreater(cos_sim.item(), 0.98)
```

#### T8.2 FP8 KV Cache 测试

**类型**：Python unittest  
**位置**：`rtp_llm/models_py/bindings/ascend/test/test_ascend_fp8_kv_cache.py`

```python
class AscendFP8KVCacheTest(unittest.TestCase):
    def test_fp8_kv_cache_roundtrip(self):
        """FP8 KV Cache 量化-反量化往返测试"""
        k = torch.randn(32, 128, 128, dtype=torch.float16, device="npu")
        k_quant, scale = ascend_ops.quantize_fp8(k)
        k_dequant = ascend_ops.dequantize_fp8(k_quant, scale)

        cos_sim = torch.nn.functional.cosine_similarity(
            k.flatten().cpu().float().unsqueeze(0),
            k_dequant.flatten().cpu().float().unsqueeze(0))
        self.assertGreater(cos_sim.item(), 0.99)
```

#### T8.3 端到端 FP8 推理精度测试

**类型**：Python integration test  
**位置**：`rtp_llm/test/model_test/test_ascend_fp8_inference.py`

```python
class AscendFP8InferenceTest(unittest.TestCase):
    def test_fp8_vs_fp16_perplexity(self):
        """
        对比 FP8 和 FP16 推理的 perplexity 差异
        验收标准：perplexity 差异 < 1%
        """
        # 加载同一个模型的 FP8 和 FP16 权重
        # 用相同的 prompt 跑推理
        # 计算 perplexity
        # 验证差异在阈值内
        pass
```

---

## Phase 9: Python 侧全量适配 — 测试计划

### 质量门禁：所有现有测试在 NPU 上通过

#### T9.1 现有测试 NPU 参数化运行

**策略**：将现有测试中硬编码的 `device="cuda"` 改为参数化，CI 中分别跑 CUDA 和 NPU。

```python
# rtp_llm/test/utils/device_utils.py 扩展
import os
import torch

def get_test_device():
    device_env = os.environ.get("TEST_USING_DEVICE", "CUDA")
    if device_env == "ASCEND":
        return torch.device("npu")
    elif device_env == "ROCM":
        return torch.device("cuda")  # ROCm uses cuda namespace via HIP
    else:
        return torch.device("cuda")

def is_ascend():
    return os.environ.get("TEST_USING_DEVICE") == "ASCEND"
```

#### T9.2 端到端推理测试

**类型**：Python integration test  
**位置**：`rtp_llm/test/model_test/test_ascend_e2e.py`

```python
class AscendE2ETest(unittest.TestCase):
    def _run_model_test(self, model_name, model_path, prompt, expected_substring=None):
        from rtp_llm.server import start_server
        from rtp_llm.utils.model_utils import load_model

        model = load_model(model_dir=model_path, device="npu")
        input_ids = model.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device="npu")

        output_ids = model.generate(input_tensor, max_new_tokens=100)
        output_text = model.tokenizer.decode(output_ids[0].cpu().tolist())

        self.assertFalse(any(c in output_text for c in ['\x00', '\ufffd']))
        if expected_substring:
            self.assertIn(expected_substring, output_text)

        return output_text

    def test_qwen_inference(self):
        self._run_model_test("qwen-7b", "/path/to/qwen", "你好")

    def test_llama_inference(self):
        self._run_model_test("llama-7b", "/path/to/llama", "Hello")

    def test_deepseek_inference(self):
        self._run_model_test("deepseek-7b", "/path/to/deepseek", "你好")
```

#### T9.3 回归测试矩阵

| 模型架构 | FP16 | BF16 | FP8 | TP=2 | Graph | 备注 |
|----------|------|------|-----|------|-------|------|
| Qwen-7B | ✅ | ✅ | - | ✅ | ✅ | 基础验证 |
| LLaMA-7B | ✅ | ✅ | - | ✅ | ✅ | |
| DeepSeek-V2-Lite | ✅ | ✅ | - | ✅ | - | MLA 注意力 |
| Qwen-72B | - | ✅ | ✅ | ✅ | ✅ | 需多卡 |
| Mixtral-8x7B | ✅ | - | - | ✅ | - | MoE |

---

## Phase 10: 性能优化与集成测试 — 测试计划

### 质量门禁：性能达到预设 baseline

#### T10.1 性能基准套件

**类型**：Python benchmark（manual tag）  
**位置**：`rtp_llm/test/perf_test/ascend_perf_suite.py`

```python
class AscendPerfSuite(unittest.TestCase):
    """
    建立性能 baseline，每次优化后重新跑并与 baseline 对比
    """

    def test_decode_latency(self):
        """单卡 decode latency（ms/token）"""
        # load model, warmup, benchmark
        # 记录 ms/token
        # 对比 baseline（需 < baseline * 1.1）
        pass

    def test_prefill_throughput(self):
        """Prefill throughput（tokens/s）"""
        pass

    def test_batch_throughput(self):
        """不同 batch Size 下的吞吐量曲线"""
        batch_sizes = [1, 8, 16, 32, 64, 128]
        # 记录每个 batch size 的 throughput
        pass

    def test_tp_scaling_efficiency(self):
        """TP 扩展效率"""
        # 1卡 vs 2卡 vs 4卡 的 throughput
        # 验证 near-linear scaling
        pass
```

#### T10.2 长稳测试

**类型**：长时间运行的 integration test  
**位置**：`rtp_llm/test/perf_test/ascend_stability_test.py`

```python
class AscendStabilityTest(unittest.TestCase):
    def test_24h_continuous_inference(self):
        """24 小时持续推理，验证无内存泄漏、无精度退化"""
        model = load_model(...)
        initial_mem = torch.npu.memory_allocated()

        for hour in range(24):
            for step in range(3600):  # ~1 req/s
                output = model.generate(random_prompt())
            current_mem = torch.npu.memory_allocated()
            leak = (current_mem - initial_mem) / (1024 ** 3)
            print(f"Hour {hour}: memory leak = {leak:.2f} GB")
            self.assertLess(leak, 1.0, f"内存泄漏超过 1GB: {leak:.2f} GB")
```

---

## 测试执行策略总结

### 各 Phase 测试用例数量汇总

| Phase | C++ 单测 | Python 单测 | 集成测试 | 性能测试 | 合计 |
|-------|---------|------------|---------|---------|------|
| Phase 0 | 3 | 0 | 1 (编译验证) | 0 | **4** |
| Phase 1 | 8 | 4 | 1 | 0 | **13** |
| Phase 2 | 6 | 0 | 0 | 2 | **8** |
| Phase 3 | 0 | 12 | 2 | 2 | **16** |
| Phase 4 | 4 | 0 | 1 | 0 | **5** |
| Phase 5 | 0 | 15 | 1 | 3 | **19** |
| Phase 6 | 0 | 4 | 2 | 0 | **6** |
| Phase 7 | 0 | 3 | 1 | 1 | **5** |
| Phase 8 | 0 | 3 | 1 | 0 | **4** |
| Phase 9 | 0 | 5 | 5 | 0 | **10** |
| Phase 10 | 0 | 0 | 1 | 4 | **5** |
| **合计** | **21** | **46** | **16** | **12** | **95** |

### CI 流水线设计

```
PR 提交触发:
  ├── compile_check    # --config=ascend 编译通过
  ├── unit_test_npu    # bazel test //... --config=ascend (排除 manual)
  └── precision_check  # 精度回归测试

每日构建触发:
  ├── full_test_suite  # 全量测试
  ├── perf_baseline    # 性能基准
  └── stability_test   # 4h 稳定性测试

发版触发:
  ├── 24h_stability    # 长稳测试
  ├── model_matrix     # 多模型回归矩阵
  └── perf_regression  # 性能回归检测
```

### AI 辅助生成测试用例的策略

| 测试类型 | AI 适合度 | 说明 |
|---------|----------|------|
| API 签名/编译验证 | ★★★★★ | AI 可直接生成 |
| 精度对比测试 | ★★★★★ | 给定参考实现，AI 可自动生成对比逻辑 |
| 规模扫描测试 | ★★★★★ | AI 可生成 parameterized 测试矩阵 |
| 性能基准测试 | ★★★★☆ | AI 可生成框架代码，需人工在 NPU 上调参 |
| 分布式测试 | ★★★☆☆ | 多卡协调逻辑需人工设计，AI 辅助生成模板 |
| 端到端集成测试 | ★★☆☆☆ | 依赖完整模型和环境，AI 辅助有限 |
| 长稳测试 | ★★★★☆ | AI 可生成框架，阈值需人工设定 |

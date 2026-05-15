## * BUILD逐步验证步骤详解

### 前置准备

```bash
export ASCEND_TOOLKIT_PATH=/usr/local/Ascend/ascend-toolkit/latest
export TF_NEED_ASCEND=1
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_PATH/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/add-ons/:$LD_LIBRARY_PATH

ls $ASCEND_TOOLKIT_PATH/include/acl/acl.h
ls $ASCEND_TOOLKIT_PATH/lib64/libascendcl.so
```

### 全量编译

# Use the last release branch

# build RTP-LLM whl target
# --config=cuda12_6 build target for NVIDIA GPU with cuda12_6
# --config=rocm build target for AMD
# bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=cuda12_6 --test_output=errors --test_env="LOG_LEVEL=INFO"  --jobs=64

bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend --test_output=errors --test_env="LOG_LEVEL=INFO"

ln  -sf `pwd`/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py  `pwd`/rtp_llm/cpp/model_rpc/proto/
ln  -sf `pwd`/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py  `pwd`/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py


### Step 1：验证 Bazel 能识别 Ascend 仓库

```bash
bazel query @local_config_ascend//...
```

**预期输出**：4 个 target（`ascend_headers`, `ascend`, `aclblas`, `hccl`）

**原理**：`3rdparty/gpus/ascend_configure.bzl` 在 `TF_NEED_ASCEND=1` 时自动探测 CANN 路径，生成 `@local_config_ascend` 仓库。

**失败排查**：
- `Repository @local_config_ascend not found`：检查 `TF_NEED_ASCEND` 是否为 `1`
- `Cannot find CANN header`：检查 `$ASCEND_TOOLKIT_PATH/include/acl/acl.h` 是否存在
- 只输出了 `using_ascend` config_setting：说明进入了 `_create_dummy_repo` 分支，`TF_NEED_ASCEND` 未正确设置

### Step 2：编译 Ascend 兼容层

```bash
bazel build --config=ascend //rtp_llm/models_py/bindings/ascend:ascend_host_utils
```

**预期输出**：`Target up-to-date`

**原理**：最基础的 Ascend 兼容层 target，依赖 `@local_config_ascend//ascend:ascend_headers`。

**失败排查**：
- `fatal error: acl/acl.h: No such file`：CANN 头文件路径不对
- `nvcc not found`：`--crosstool_top` 未生效，仍在使用 CUDA 工具链

### Step 3：编译核心头文件

```bash
bazel build --config=ascend //rtp_llm/models_py/bindings/core:types_hdr
```

**预期输出**：`Target up-to-date`

**原理**：验证 `MEMORY_NPU` 枚举值和 `#if USING_ASCEND` 条件编译宏。

**失败排查**：
- `MEMORY_NPU undeclared`：`USING_ASCEND` 宏未定义
- 语法错误：检查 `Types.h` 中枚举定义

### Step 4：编译配置库

```bash
bazel build --config=ascend //:th_transformer_config # 编译报错：error: argument 2 of type 'const uint8_t *' {aka 'const unsigned char *'} declared as a pointer

bazel build --config=ascend --copt="-Wno-array-parameter" //:th_transformer_config
```



**预期输出**：`Target up-to-date`

**原理**：验证 `if_ascend` select 分支和 `using_ascend` config_setting。

**失败排查**：
- `config_setting 'using_ascend' not found`：回到 Step 1
- 链接错误：检查 `def.bzl` 中 `if_ascend` 的 deps 列表

**编译问题**
- `./rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h:3:10: fatal error: kmonitor/client/MetricsReporter.h: No such file or directory`

原因：TODO

- `error: argument 2 of type 'const uint8_t *' {aka 'const unsigned char *'} declared as a pointer`

原因：警告作为报错，bazel编译时添加参数--copt="-Wno-array-parameter"

- `rtp_llm/models_py/bindings/core/ExecOps.cc:103:16: error: 'hipDeviceSynchronize' was not declared in this scope;  103 |     ROCM_CHECK(hipDeviceSynchronize());`

原因：设备类型的宏判断，修改不彻底，添加昇腾相关的判断



### Step 5：编译计算 ops 库（关键步骤）

```bash
bazel build --config=ascend --copt="-Wno-array-parameter" //:rtp_compute_ops
```

**预期输出**：`Target up-to-date`

**原理**：最可能出问题的一步，包含大量 CUDA 代码需要条件编译适配。

**常见错误及修复**：

| 错误 | 原因 | 修复方式 |
|------|------|---------|
| `#include <cuda_runtime.h>: No such file` | Ascend 模式下无 CUDA 头文件 | 添加 `#if USING_CUDA` / `#endif` 守卫 |
| `undefined reference to cudaMalloc` | 无条件调用 CUDA API | 添加 `#if USING_CUDA` / `#elif USING_ASCEND` 条件编译 |
| `nvcc not found` | BUILD 中 `.cu` 文件仍用 nvcc 编译 | 在 BUILD 的 `select()` 中为 Ascend 分支排除 `.cu` srcs |

排查命令：
```bash
grep -rn '#include.*cuda' --include="*.cc" --include="*.h" rtp_llm/ | grep -v "#if USING_CUDA"
grep -rn '\.cu' --include="BUILD" rtp_llm/
```

**编译问题**
- `编译过程中依赖大量cuda算子实现，暂时stub掉，后续分析算子接入方案（TODO）`

原因：TODO

### Step 6：编译最终产物

```bash
bazel build --config=ascend --copt="-Wno-array-parameter" //:th_transformer # 编译报错

bazel build --config=ascend --copt="-Wno-array-parameter" --copt="-Wno-tautological-compare" //:th_transformer
```

**预期输出**：`Target up-to-date`

如果 Step 5 通过，此步通常也能通过。

---




## * http.bzl 中包的下载与使用机制

### 下载机制

`deps/http.bzl` 中的 `http_archive` 定义了两个 Ascend 外部仓库：

| 仓库 | 下载源 | 用途 |
|------|--------|------|
| `torch_cpu_ascend` | 阿里云镜像 `torch-2.9.0+cpu-cp310-aarch64.whl` | Ascend 平台的 PyTorch CPU 版 |
| `torch_npu_ascend` | PyPI `torch_npu-2.9.0-cp310-aarch64.whl` | 华为 NPU 扩展库 |

调用链：`WORKSPACE:28` → `http_deps()` → 执行所有 `http_archive` 声明。

### 下载时机

Bazel 的 `http_archive` 是**惰性下载**——只有当某个 target 实际依赖到这些仓库时才触发下载。使用 `--config=ascend` 时，`select()` 匹配到 `using_ascend` 分支，引用 `@torch_cpu_ascend` 和 `@torch_npu_ascend`，Bazel 解析依赖后触发下载。用 `--config=cuda12` 编译则不会下载。

### 潜在问题

1. **`sha256` 为空**：不会校验下载文件完整性，建议补上正确的 sha256 值
2. **URL 可能不可达**：需确认阿里云镜像和 PyPI 上对应版本的 aarch64 wheel 实际存在

### 下载后的使用流程

1. **下载并解压**：`http_archive` 将 `.whl` 作为 `type = "zip"` 解压到仓库缓存目录
2. **通过 `build_file` 映射为 Bazel Target**：
   - `torch_cpu_ascend` → `BUILD.pytorch`：映射为 `torch`、`torch_api`、`torch_libs` 三个 `cc_library`
   - `torch_npu_ascend` → `BUILD.torch_npu`：映射为 `torch_npu`、`torch_npu_api` 两个 `cc_library`
3. **通过 `select()` 注入编译依赖**：`arch_select.bzl` 中 `using_ascend` 分支引用这些 target
4. **编译期**：`-I` 包含路径自动添加头文件搜索路径
5. **链接期**：`-l` 自动链接 `.so` 动态库
6. **运行期**：`.so` 文件通过 `data` 属性打包到产出物中

### 完整依赖图

```
bazel build --config=ascend //:th_transformer
    │
    ├── arch_select.bzl: torch_deps()  [select: using_ascend]
    │       │
    │       ├── @torch_cpu_ascend//:torch_api   ← BUILD.pytorch: torch_api target
    │       │       └── torch/include/torch/csrc/api/include/*.h
    │       │
    │       ├── @torch_cpu_ascend//:torch       ← BUILD.pytorch: torch target
    │       │       ├── torch/lib/libtorch.so          (链接)
    │       │       ├── torch/lib/libtorch_cpu.so      (链接)
    │       │       ├── torch/lib/libc10.so            (链接)
    │       │       └── torch/include/**/*.h           (头文件)
    │       │
    │       ├── @torch_cpu_ascend//:torch_libs  ← BUILD.pytorch: torch_libs target
    │       │       └── torch.libs/libgomp*.so         (运行时依赖)
    │       │
    │       └── @torch_npu_ascend//:torch_npu   ← BUILD.torch_npu: torch_npu target
    │               ├── torch_npu/lib/libtorch_npu.so  (链接)
    │               ├── torch_npu/include/**/*.h       (头文件)
    │               └── @torch_cpu_ascend//:torch      (传递依赖)
    │
    └── 最终产出: th_transformer.so
            链接: -ltorch -ltorch_cpu -lc10 -ltorch_npu ...
            头文件: torch/include/ + torch_npu/include/
```

---


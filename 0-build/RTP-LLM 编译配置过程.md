# RTP-LLM 编译配置过程

> **Workspace:** c:\WorkingSpace\Code\rtp-llm

---

## User
rtp-llm编译配置过程

## Assistant
我来帮你了解 rtp-llm 的编译配置过程。首先让我查看一下项目结构和相关配置文件。

根据对 rtp-llm 项目代码的分析，我来为你详细介绍其编译配置过程。

## rtp-llm 编译配置概览

rtp-llm 使用 **Bazel** 作为构建系统，通过 `.bazelrc` 和 `WORKSPACE` 文件管理编译配置。以下是完整的编译配置流程：

---

### 1. 构建系统入口

| 文件 | 作用 |
|------|------|
| [WORKSPACE](file:///c:/WorkingSpace/Code/rtp-llm/WORKSPACE) | Bazel 工作区定义，加载外部依赖 |
| [.bazelrc](file:///c:/WorkingSpace/Code/rtp-llm/.bazelrc) | 编译选项和配置集合 |
| [BUILD](file:///c:/WorkingSpace/Code/rtp-llm/BUILD) | 根构建目标定义 |
| [def.bzl](file:///c:/WorkingSpace/Code/rtp-llm/def.bzl) | 自定义构建规则 |

---

### 2. WORKSPACE 依赖配置

[WORKSPACE](file:///c:/WorkingSpace/Code/rtp-llm/WORKSPACE) 文件负责初始化构建环境：

```starlark
workspace(name = "rtp_llm")

# 1. CUDA 配置
load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

# 2. ROCm 配置
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
rocm_configure(name = "local_config_rocm")

# 3. Python 配置
load("//3rdparty/py:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# 4. 外部依赖加载
load("//deps:http.bzl", "http_deps")
http_deps()

load("//deps:git.bzl", "git_deps")
git_deps()

# 5. Python pip 依赖
load("//deps:pip.bzl", "pip_deps")
pip_deps()
```

**配置顺序：**
1. **CUDA 自动检测** - 通过 `cuda_configure` 检测系统 CUDA 环境
2. **ROCm 自动检测** - 通过 `rocm_configure` 检测 AMD GPU 环境
3. **Python 配置** - 设置 Python 解释器路径
4. **外部依赖** - 下载第三方库（HTTP/Git）
5. **Pip 依赖** - 安装 Python 包依赖

---

### 3. .bazelrc 编译配置详解

[.bazelrc](file:///c:/WorkingSpace/Code/rtp-llm/.bazelrc) 是核心编译配置文件，定义了多种构建模式：

#### 3.1 基础编译选项
```bash
# Python 版本强制使用 conda310
build --python_top=//:python310 --incompatible_use_python_toolchains=false

# 编译策略和优化
build --spawn_strategy=local          # 本地编译，避免 nvcc 冲突
build --cxxopt="-std=c++17"           # C++17 标准
build -c opt                          # 优化模式
build --copt -O2                      # O2 优化
build --copt -g --strip=never         # 保留调试信息
build --copt -Wall                    # 显示所有警告
build --copt -Werror                  # 警告视为错误
```

#### 3.2 CUDA 编译配置 (`--config=cuda`)
```bash
build:cuda --copt="-DENABLE_BF16=1"
build:cuda --copt="-DBUILD_CUTLASS_MIXED_GEMM=ON"
build:cuda --copt="-DUSING_CUDA=1"
build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true
build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
build:cuda --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda/"
```

#### 3.3 CUDA 版本配置

| 配置 | 适用场景 | 计算能力 |
|------|----------|----------|
| `--config=cuda12` | CUDA 12.x 通用 | 7.0,7.5,8.0,8.6,8.9,9.0 |
| `--config=cuda12_6` | CUDA 12.6 | 同上 |
| `--config=cuda12_9` | CUDA 12.9 x86 | 同上 + FP4 支持 |
| `--config=cuda12_9_arm` | CUDA 12.9 ARM | 10.0 (Blackwell) |
| `--config=sm7x` | V100/T4 | 7.0,7.5 |
| `--config=sm8x` | A10/A100/L40 | 8.0,8.6,8.9 |
| `--config=sm9x` | H100/H800 | 9.0 |

#### 3.4 其他平台配置
```bash
# CPU 模式
build:cpu --define=using_cpu=true
build:cpu --action_env TF_NEED_CUDA="0"

# ROCm (AMD GPU) 模式
build:rocm --define=using_rocm=true
build:rocm --action_env TF_NEED_ROCM=1
build:rocm --action_env GPU_ARCHS=gfx942

# ARM 模式
build:arm --define=using_arm=true
```

---

### 4. CUDA 自动检测配置

CUDA 配置通过 [3rdparty/cuda_config/cuda_configure.bzl](file:///c:/WorkingSpace/Code/rtp-llm/3rdparty/cuda_config/cuda_configure.bzl) 自动检测：

**检测的环境变量：**
| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TF_NEED_CUDA` | 是否启用 CUDA | - |
| `CUDA_TOOLKIT_PATH` | CUDA 工具包路径 | `/usr/local/cuda` |
| `TF_CUDA_VERSION` | CUDA 版本 | 自动检测 |
| `TF_CUDNN_VERSION` | cuDNN 版本 | 自动检测 |
| `TF_CUDA_COMPUTE_CAPABILITIES` | GPU 计算能力 | `3.5,5.2` |
| `CUDNN_INSTALL_PATH` | cuDNN 安装路径 | `/usr/local/cuda` |
| `PYTHON_BIN_PATH` | Python 路径 | - |

---

### 5. 编译目标定义

[BUILD](file:///c:/WorkingSpace/Code/rtp-llm/BUILD) 文件定义了主要编译目标：

```starlark
# 1. 配置库 (Python C++ 扩展)
cc_binary(
    name = "th_transformer_config",
    linkshared = 1,  # 共享库
    deps = ["//rtp_llm/cpp/pybind:th_transformer_config_lib"],
)

# 2. 计算算子库
cc_binary(
    name = "rtp_compute_ops",
    linkshared = 1,
    deps = ["//rtp_llm/cpp/pybind:th_compute_lib"],
)

# 3. 主推理库
cc_binary(
    name = "th_transformer",
    linkshared = 1,
    deps = ["//rtp_llm/cpp/pybind:th_transformer_lib"],
)
```

---

### 6. 编译命令示例

#### 6.1 源码编译（推荐）
```bash
# 1. 克隆代码
git clone git@github.com:alibaba/rtp-llm.git
cd rtp-llm

# 2. 编译 whl 包（NVIDIA GPU + CUDA 12.6）
bazelisk build //rtp_llm:rtp_llm \
    --verbose_failures \
    --config=cuda12_6 \
    --jobs=64

# 3. 创建 protobuf 符号链接
ln -sf `pwd`/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py \
    `pwd`/rtp_llm/cpp/model_rpc/proto/
ln -sf `pwd`/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py \
    `pwd`/rtp_llm/cpp/model_rpc/proto/
```

#### 6.2 不同平台编译
```bash
# AMD ROCm
bazelisk build //rtp_llm:rtp_llm --config=rocm --jobs=64

# CPU 模式
bazelisk build //rtp_llm:rtp_llm --config=cpu --jobs=64

# ARM 平台
bazelisk build //rtp_llm:rtp_llm --config=arm --jobs=64

# CUDA 12.9 (支持 FP4)
bazelisk build //rtp_llm:rtp_llm --config=cuda12_9 --jobs=64
```

#### 6.3 调试编译
```bash
bazelisk build //rtp_llm:rtp_llm --config=cuda12_6 --config=debug --jobs=64
```

---

### 7. 第三方依赖管理

项目依赖通过 Bazel 的 `deps/` 目录管理：

| 依赖类型 | 文件 | 说明 |
|----------|------|------|
| HTTP 依赖 | `deps/http.bzl` | 下载预编译包 |
| Git 依赖 | `deps/git.bzl` | 克隆 Git 仓库 |
| Pip 依赖 | `deps/pip.bzl` | Python 包依赖 |

主要第三方库：
- **CUDA 相关**: CUTLASS、FlashAttention、FlashInfer、FlashMLA
- **通信库**: NCCL、NVSHMEM、Deep EP
- **推理优化**: TensorRT-LLM、xFasterTransformer
- **Python 包**: PyTorch、Transformers、Triton 等

---

### 8. 配置流程图

```
┌─────────────────┐
│   执行 bazel build │
└────────┬────────┘
         ▼
┌─────────────────┐
│  读取 WORKSPACE   │
│  加载外部依赖规则  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  cuda_configure  │
│  自动检测 CUDA   │
│  生成 toolchain │
└────────┬────────┘
         ▼
┌─────────────────┐
│  rocm_configure  │
│  自动检测 ROCm   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  python_configure│
│  配置 Python    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  读取 .bazelrc   │
│  应用编译配置    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  解析 BUILD 文件 │
│  构建目标依赖图  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  执行编译        │
│  生成 so/whl    │
└─────────────────┘
```

---

### 9. 关键配置要点

1. **Python 版本固定**：强制使用 `/opt/conda310/bin/python3`
2. **本地编译策略**：`--spawn_strategy=local` 避免 nvcc 冲突
3. **计算能力自动检测**：根据 GPU 架构选择最优代码生成
4. **多平台支持**：CUDA/ROCm/CPU/ARM 统一构建框架
5. **C++17 标准**：所有代码统一使用 C++17 编译

如需了解特定平台（如 DeepSeek V3/R1、Kimi-K2）的详细编译配置，可以参考项目文档中的 [Docker 部署方案](https://rtp-llm.ai/build/en/start/install.html)。


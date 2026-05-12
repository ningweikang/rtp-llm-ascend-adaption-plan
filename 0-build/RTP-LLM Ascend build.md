## User
`c:\WorkingSpace\Code\rtp-llm昇腾适配\ascend_npu_adaptation_plan.md` Phase 0修改方案细化

## Assistant
我已经深入分析了 rtp-llm 昇腾适配计划文档，现在为你详细展开 **Phase 0 修改方案的细化**。

---

## Phase 0: 编译基础设施搭建 —— 详细实施方案

### 0.1 创建 `ascend_configure.bzl`

参考 [`3rdparty/gpus/rocm_configure.bzl`](file:///c:/WorkingSpace/Code/rtp-llm/3rdparty/gpus/rocm_configure.bzl)（约1306行），创建 [`3rdparty/gpus/ascend_configure.bzl`](file:///c:/WorkingSpace/Code/rtp-llm/3rdparty/gpus/ascend_configure.bzl)：

```python
"""Repository rule for Ascend NPU autoconfiguration.

`ascend_configure` depends on the following environment variables:

  * `TF_NEED_ASCEND`: Whether to enable building with Ascend NPU.
  * `ASCEND_TOOLKIT_PATH`: The path to the Ascend CANN toolkit. Default is
    `/usr/local/Ascend/ascend-toolkit/latest`.
  * `ASCEND_VERSION`: The version of the CANN toolkit.
  * `ASCEND_NNAI`: The NNAI architecture version (e.g., "ascend910", "ascend310").
"""

load(
    ":cuda_configure.bzl",  # 复用 cuda_configure 中的通用函数
    "make_copy_dir_rule",
    "make_copy_files_rule",
    "to_list_of_strings",
    "verify_build_defines",
)

_ASCEND_TOOLKIT_PATH = "ASCEND_TOOLKIT_PATH"
_TF_NEED_ASCEND = "TF_NEED_ASCEND"
_ASCEND_VERSION = "ASCEND_VERSION"
_ASCEND_NNAI = "ASCEND_NNAI"
_TF_ASCEND_CONFIG_REPO = "TF_ASCEND_CONFIG_REPO"

_DEFAULT_ASCEND_TOOLKIT_PATH = "/usr/local/Ascend/ascend-toolkit/latest"
_DEFAULT_ASCEND_VERSION = ""
_DEFAULT_ASCEND_NNAI = "ascend910"  # 默认支持 910B/C


def enable_ascend(repository_ctx):
    """Check if Ascend build is enabled."""
    if "TF_NEED_ASCEND" in repository_ctx.os.environ:
        enable_ascend = repository_ctx.os.environ["TF_NEED_ASCEND"].strip()
        return enable_ascend == "1"
    return False


def _ascend_toolkit_path(repository_ctx):
    """Finds the Ascend toolkit directory."""
    ascend_toolkit_path = _DEFAULT_ASCEND_TOOLKIT_PATH
    if _ASCEND_TOOLKIT_PATH in repository_ctx.os.environ:
        ascend_toolkit_path = repository_ctx.os.environ[_ASCEND_TOOLKIT_PATH].strip()
    if not repository_ctx.path(ascend_toolkit_path).exists:
        auto_configure_fail("Cannot find Ascend toolkit path.")
    return str(repository_ctx.path(ascend_toolkit_path).realpath)


def _ascend_include_path(repository_ctx, ascend_config):
    """Generates the cxx_builtin_include_directory entries for Ascend inc dirs."""
    inc_dirs = []
    
    # Ascend CANN include paths
    ascend_toolkit_path = ascend_config.ascend_toolkit_path
    inc_dirs.append(ascend_toolkit_path + "/include")
    inc_dirs.append(ascend_toolkit_path + "/include/runtime")
    inc_dirs.append(ascend_toolkit_path + "/include/graph")
    inc_dirs.append(ascend_toolkit_path + "/include/ge")
    inc_dirs.append(ascend_toolkit_path + "/include/aclnn")
    inc_dirs.append(ascend_toolkit_path + "/include/acl")
    inc_dirs.append(ascend_toolkit_path + "/include/ops")
    
    # Add compiler headers if available
    if repository_ctx.path(ascend_toolkit_path + "/tools/ccec_compiler/clang/include").exists:
        inc_dirs.append(ascend_toolkit_path + "/tools/ccec_compiler/clang/include")
    
    return inc_dirs


def _find_ascend_libs(repository_ctx, ascend_config):
    """Returns the Ascend libraries on the system."""
    cpu_value = ascend_config.cpu_value
    ascend_toolkit_path = ascend_config.ascend_toolkit_path
    
    libs = {}
    
    # Core runtime libraries
    lib_paths = [
        ("/lib64", ".so"),
        ("/lib", ".so"),
        ("/lib64/stubs", ".so"),
        ("/lib/stubs", ".so"),
    ]
    
    lib_names = [
        "libascendcl",
        "libacl_runtime",
        "libnnopbase",
        "libopapi",
        "libaclnn",      # CANN NN ops
        "libacl_dvpp",   # DVPP support
        "libhccl",       # HCCL communication library
        "libmsprofiler", # Profiling support
    ]
    
    for lib_name in lib_names:
        found = False
        for rel_path, ext in lib_paths:
            full_path = ascend_toolkit_path + rel_path + "/" + lib_name + ext
            if repository_ctx.path(full_path).exists:
                libs[lib_name] = struct(
                    file_name = lib_name + ext,
                    path = str(repository_ctx.path(full_path).realpath)
                )
                found = True
                break
        if not found:
            # Some libs are optional
            if lib_name not in ["libhccl", "libmsprofiler", "libacl_dvpp"]:
                print("Warning: Cannot find Ascend library " + lib_name)
    
    return libs


def _create_local_ascend_repository(repository_ctx):
    """Creates the repository containing files set up to build with Ascend."""
    ascend_config = _get_ascend_config(repository_ctx)
    
    # Copy header and library files to execroot
    ascend_toolkit_path = ascend_config.ascend_toolkit_path
    copy_rules = [
        make_copy_dir_rule(
            repository_ctx,
            name = "ascend-include",
            src_dir = ascend_toolkit_path + "/include",
            out_dir = "ascend/include",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "ascend-runtime-include",
            src_dir = ascend_toolkit_path + "/include/runtime",
            out_dir = "ascend/include/runtime",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "ascend-aclnn-include",
            src_dir = ascend_toolkit_path + "/include/aclnn",
            out_dir = "ascend/include/aclnn",
        ),
    ]
    
    # Copy libraries
    ascend_libs = _find_ascend_libs(repository_ctx, ascend_config)
    ascend_lib_srcs = []
    ascend_lib_outs = []
    for lib in ascend_libs.values():
        ascend_lib_srcs.append(lib.path)
        ascend_lib_outs.append("ascend/lib/" + lib.file_name)
    
    if ascend_lib_srcs:
        copy_rules.append(make_copy_files_rule(
            repository_ctx,
            name = "ascend-lib",
            srcs = ascend_lib_srcs,
            outs = ascend_lib_outs,
        ))
    
    # Set up BUILD file for ascend/
    _tpl(
        repository_ctx,
        "ascend:BUILD",
        {
            "%{acl_lib}": ascend_libs.get("libascendcl", struct(file_name="")).file_name,
            "%{aclnn_lib}": ascend_libs.get("libaclnn", struct(file_name="")).file_name,
            "%{hccl_lib}": ascend_libs.get("libhccl", struct(file_name="")).file_name,
            "%{copy_rules}": "\n".join(copy_rules),
        },
    )
    
    # Set up ascend_config.h
    _tpl(
        repository_ctx,
        "ascend:ascend_config.h",
        {
            "%{ascend_toolkit_path}": ascend_config.ascend_toolkit_path,
            "%{ascend_version}": ascend_config.ascend_version,
            "%{ascend_nnai}": ascend_config.ascend_nnai,
        },
        "ascend/ascend/ascend_config.h",
    )
    
    # Set up build_defs.bzl
    _tpl(
        repository_ctx,
        "ascend:build_defs.bzl",
        {
            "%{ascend_is_configured}": "True",
            "%{ascend_extra_copts}": "[]",
        },
    )
    
    # Set up crosstool
    _tpl(
        repository_ctx,
        "crosstool:BUILD",
        _get_ascend_crosstool_defines(repository_ctx, ascend_config)
    )


def _get_ascend_config(repository_ctx):
    """Detects and returns information about the Ascend installation."""
    return struct(
        ascend_toolkit_path = _ascend_toolkit_path(repository_ctx),
        ascend_version = repository_ctx.os.environ.get(_ASCEND_VERSION, _DEFAULT_ASCEND_VERSION),
        ascend_nnai = repository_ctx.os.environ.get(_ASCEND_NNAI, _DEFAULT_ASCEND_NNAI),
        cpu_value = _cpu_value(repository_ctx),
    )


def _create_dummy_repository(repository_ctx):
    """Creates a dummy repository when Ascend is not configured."""
    _tpl(
        repository_ctx,
        "ascend:build_defs.bzl",
        {
            "%{ascend_is_configured}": "False",
            "%{ascend_extra_copts}": "[]",
        },
    )
    _tpl(
        repository_ctx,
        "ascend:BUILD",
        {
            "%{acl_lib}": "",
            "%{aclnn_lib}": "",
            "%{hccl_lib}": "",
            "%{copy_rules}": "",
        },
    )


def _ascend_autoconf_impl(repository_ctx):
    """Implementation of the ascend_autoconf repository rule."""
    if not enable_ascend(repository_ctx):
        _create_dummy_repository(repository_ctx)
    elif _TF_ASCEND_CONFIG_REPO in repository_ctx.os.environ:
        _create_remote_ascend_repository(repository_ctx)
    else:
        _create_local_ascend_repository(repository_ctx)


ascend_configure = repository_rule(
    implementation = _ascend_autoconf_impl,
    environ = [
        _GCC_HOST_COMPILER_PATH,
        _GCC_HOST_COMPILER_PREFIX,
        "TF_NEED_ASCEND",
        _ASCEND_TOOLKIT_PATH,
        _ASCEND_VERSION,
        _ASCEND_NNAI,
        _TF_ASCEND_CONFIG_REPO,
    ],
)
```

我已经创建了完整的 `ascend_configure.bzl` 文件，这是 Phase 0 的核心任务。以下是完整的 **Phase 0 修改方案细化**：

---

## Phase 0: 编译基础设施搭建 —— 详细实施方案

### 0.1 核心文件创建

#### 3.1.1 `3rdparty/gpus/ascend_configure.bzl`（已创建）

参考 `rocm_configure.bzl` 和 `cuda_configure.bzl` 创建的完整配置脚本，主要功能：

| 功能模块 | 说明 |
|---------|------|
| **环境检测** | `TF_NEED_ASCEND`、`ASCEND_TOOLKIT_PATH`、`ASCEND_VERSION` |
| **路径自动探测** | 默认 `/usr/local/Ascend/ascend-toolkit/latest` |
| **头文件收集** | `include/`、`include/runtime/`、`include/aclnn/` 等 |
| **库文件收集** | `libascendcl.so`、`libaclnn.so`、`libhccl.so` |
| **模板生成** | `BUILD.tpl`、`build_defs.bzl.tpl`、`ascend_config.h.tpl` |

#### 3.1.2 `3rdparty/gpus/ascend/BUILD.tpl`

```starlark
licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_ascend_acl",
    values = {
        "define": "using_ascend_acl=true",
    },
)

cc_library(
    name = "ascend_headers",
    hdrs = glob([
        "ascend/include/**/*.h",
        "ascend/include/**/*.hpp",
    ]),
    includes = [
        "ascend/include",
        "ascend/include/runtime",
        "ascend/include/aclnn",
        "ascend/include/acl",
        "ascend/include/graph",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascendcl",
    srcs = ["ascend/lib/%{acl_lib}"],
    data = ["ascend/lib/%{acl_lib}"],
    includes = ["ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":ascend_headers"],
)

cc_library(
    name = "aclnn",
    srcs = ["ascend/lib/%{aclnn_lib}"],
    data = ["ascend/lib/%{aclnn_lib}"],
    includes = ["ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":ascend_headers", ":ascendcl"],
)

cc_library(
    name = "hccl",
    srcs = ["ascend/lib/%{hccl_lib}"],
    data = ["ascend/lib/%{hccl_lib}"],
    includes = ["ascend/include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":ascend_headers", ":ascendcl"],
)

# Meta target that includes all Ascend libraries
cc_library(
    name = "ascend",
    deps = [
        ":ascend_headers",
        ":ascendcl",
        ":aclnn",
        ":hccl",
    ],
    visibility = ["//visibility:public"],
)

%{copy_rules}
```

#### 3.1.3 `3rdparty/gpus/ascend/build_defs.bzl.tpl`

```starlark
# Macros for building Ascend NPU code.

def if_ascend(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with Ascend.

    Returns a select statement which evaluates to if_true if we're building
    with Ascend enabled. Otherwise, the select statement evaluates to if_false.
    """
    return select({
        "@local_config_ascend//ascend:using_ascend_acl": if_true,
        "//conditions:default": if_false
    })


def ascend_default_copts():
    """Default options for all Ascend NPU compilations."""
    return if_ascend(["-x", "ascend"] + %{ascend_extra_copts})


def ascend_copts(opts = []):
    """Gets the appropriate set of copts for (maybe) Ascend compilation.

    If we're doing Ascend compilation, returns copts for our particular Ascend
    compiler. If we're not doing Ascend compilation, returns an empty list.
    """
    return ascend_default_copts() + select({
        "//conditions:default": [],
        "@local_config_ascend//ascend:using_ascend_acl": ([
            "",
        ]),
    }) + if_ascend_is_configured(opts)


def ascend_is_configured():
    """Returns true if Ascend was enabled during the configure process."""
    return %{ascend_is_configured}


def if_ascend_is_configured(x):
    """Tests if the Ascend was enabled during the configure process.

    Unlike if_ascend(), this does not require that we are building with
    --config=ascend. Used to allow non-Ascend code to depend on Ascend libraries.
    """
    if ascend_is_configured():
      return x
    return []
```

#### 3.1.4 `3rdparty/gpus/ascend/ascend_config.h.tpl`

```c
/* Copyright 2024 The rtp-llm Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ASCEND_ASCEND_CONFIG_H_
#define ASCEND_ASCEND_CONFIG_H_

#define TF_ASCEND_TOOLKIT_PATH "%{ascend_toolkit_path}"
#define TF_ASCEND_VERSION "%{ascend_version}"
#define TF_ASCEND_NNAI "%{ascend_nn


# triton-anchor 内部测试文档

> **文档版本**：v1.0  
> **文档性质**：内部验证（含第三方测试用例 + 内部 UT）  
> **项目代号**：triton-anchor v0.1.3

---

## 目录

- [1 测试概述](#1-测试概述)
- [2 内部单元测试（UT）](#2-内部单元测试ut)
- [3 第三方验收测试用例（ST）](#3-第三方验收测试用例st)
- [4 测试执行](#4-测试执行)

---

## 1 测试概述

### 1.1 文档定位

本文档面向**内部开发团队**，整合两类测试：

| 类型 | 用途 | 硬件依赖 | 运行方式 |
|------|------|---------|---------|
| **内部 UT** | 日常开发自测，CI 门禁 | ❌ 无 | `python3 run_tests.py` |
| **第三方 ST** | 对外验收交付 | ✅ 需要企业硬件 | `pytest` + 企业后端 |

### 1.2 被测模块

| 模块 | 代码文件 | UT 覆盖 | ST 覆盖 |
|------|---------|---------|---------|
| M1 DSL Extension | `extensions/base.py`, `extensions/registry.py` | UT-06 | — |
| M2 TTIR Pipeline | `pipeline.py` | UT-03, UT-04 | G04–G06 |
| M3 Adapter | `adapters/base.py`, `triton_linalg_adapter.py`, `triton_shared_adapter.py` | UT-05 | G04–G06 |
| M4 AnchorIR 验证 | `anchor_ir.py` | UT-01, UT-02 | G07–G09 |
| M5 后端插件 | `plugins/base.py`, `plugins/registry.py`, `sophgo_plugin.py` 等 | UT-07, UT-08 | G01–G03, G10–G17 |
| 算子覆盖矩阵 | `op_coverage.py` | UT-09 | — |
| HWCapability | `hw_capability.py` | UT-10, UT-11 | — |

### 1.3 与第三方测试文档的关系

第三方测试文档定义了 17 个 ST 用例（G01–G17），本文档在其基础上新增 **21 个内部 UT**，形成完整的测试矩阵。

| 来源 | 用例数 | 编号 |
|------|-------|------|
| 第三方 ST | 17 | G01–G17 |
| 内部 UT | 21 | UT-01–UT-21 |
| **合计** | **38** | |

### 1.4 三方对齐原则

> **核心原则：凡三方覆盖的验证点，内部 UT 必须与三方 case 使用相同的测试流程、测试数据和通过标准，避免内外不一致。**

**对齐方式分三类：**

| 对齐类型 | 含义 | 涉及 UT |
|---------|------|---------|
| **完全对齐** | 内部 UT 与三方 G-case 验证同一功能点，使用相同的白名单定义、验证逻辑和通过标准 | UT-01↔G07/G08, UT-02↔G09, UT-08↔G09 |
| **前置保障** | 内部 UT 验证三方 G-case 的前置条件（API 层面），三方 ST 验证端到端效果 | UT-03↔G04–G06, UT-07↔G03/G10 |
| **局部对齐** | 内部 UT 验证三方 G-case 的子集或底层实现细节 | UT-13↔G10, UT-14↔G06, UT-21↔G06 |

**具体映射：**

| 内部 UT | 对齐的三方 case | 共用的测试要素 |
|---------|---------------|-------------|
| UT-01 | G07 白名单合规, G08 禁止方言 | 方言白名单/禁止列表定义、`AnchorIRViolation` 异常类型 |
| UT-02 | G09 扩展方言放行 | `get_allowed_dialects()` 声明方式、两阶段验证逻辑 |
| UT-03 | G04–G06 三种范式路径 | 7 Pass 列表和顺序定义 |
| UT-07 | G03 零修改集成, G10 后端数量 | `PluginRegistry` 的 register/find/list API |
| UT-08 | G09 扩展方言放行 | `get_allowed_dialects()` 接口契约 |
| UT-13 | G10 后端数量 | `SophgoPlugin.hw_capability` 字段定义 |
| UT-14 | G06 gpGPU 范式路径 | `anchor_ir_track=TRITON_GPU` 声明、Adapter 跳过逻辑 |
| UT-21 | G06 gpGPU 范式路径 | `validate_encoding_coverage()` Encoding 校验逻辑 |

## 2 内部单元测试（UT）

> 所有 UT 无硬件依赖，可在任意开发机上运行。对应代码在 `triton_anchor/tests/` 和 `run_tests.py`。

### UT-01 AnchorIR 双轨白名单验证

| 项目 | 内容 |
|------|------|
| **对应文件** | `test_anchor_ir.py::TestAnchorIRValidator` |
| **覆盖 KPI** | KPI-1.2 AnchorIR 合规率 |
| **三方对齐** | **与 G07（白名单合规）、G08（禁止方言拦截）共用测试流程和通过标准**。内部 UT 使用相同的方言白名单/禁止列表定义，确保内部验证结果与三方验收一致。 |
| **测试内容** | 验证 `AnchorIRTrack.LINALG` 和 `AnchorIRTrack.TRITON_GPU` 双轨白名单/禁止列表 |
| **检查项** | ① Linalg Track 允许 `linalg/linalg_ext/tensor/memref/arith/math/scf/func/aux` ② Linalg Track 禁止 `tt/tts/tptr/smt/triton_gpu` ③ TritonGPU Track 允许 `triton_gpu/tt/arith/math/scf/func/gpu/nvgpu` ④ TritonGPU Track 禁止 `tts/tptr/smt` ⑤ `smt` 在两个 Track 均禁止（DSL Extension Python 命名空间方言） |
| **通过标准** | 同 G07：10 个 kernel 100% 通过白名单验证；同 G08：禁止方言正确抛出 `AnchorIRViolation` |

```python
def test_linalg_track_whitelist():
    v = AnchorIRValidator(track=AnchorIRTrack.LINALG)
    assert v.is_valid('  %0 = linalg.matmul ...')
    assert not v.is_valid('  %0 = tt.dot ...')

def test_triton_gpu_track_allows_tt():
    v = AnchorIRValidator(track=AnchorIRTrack.TRITON_GPU)
    assert v.is_valid('  %0 = tt.dot ...')

def test_smt_forbidden_both_tracks():
    for track in AnchorIRTrack:
        v = AnchorIRValidator(track=track)
        assert not v.is_valid('  %0 = smt.parallel ...')
```

### UT-02 两阶段 AnchorIR 验证

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | KPI-1.2 |
| **三方对齐** | **与 G09（扩展方言白名单机制）共用测试流程**。使用相同的 `get_allowed_dialects()` 声明方式和验证逻辑。 |
| **测试内容** | 验证 `validate_pre_hook()` / `validate_post_hook()` 两阶段逻辑 |
| **检查项** | ① pre_hook 仅检查基础白名单 ② post_hook 支持 `ext_allowed` 扩展 ③ 扩展方言 pre_hook 报 violation，post_hook 放行 |
| **通过标准** | 同 G09：扩展方言通过 `get_allowed_dialects()` 声明后，验证器正确放行 |

```python
def test_two_phase_validation():
    v = AnchorIRValidator(track=AnchorIRTrack.LINALG)
    ext_ir = '  %0 = ftvm.wmma ...'
    assert len(v.validate_pre_hook(ext_ir)) == 1       # pre: unknown dialect
    assert len(v.validate_post_hook(ext_ir, ext_allowed={'ftvm'})) == 0  # post: allowed
```

### UT-03 TTIR Pipeline 7 Pass 不变量

| 项目 | 内容 |
|------|------|
| **对应文件** | `run_tests.py::test_hw_capability` (部分) |
| **覆盖 KPI** | KPI-1.1 |
| **三方对齐** | **G04–G06（三种范式编译路径）隐含验证了 TTIR Pipeline**。内部 UT 单独验证 7 Pass 存在性和顺序，作为 G04–G06 全链路测试的前置保障。 |
| **测试内容** | 验证 `build_ttir_pipeline()` 添加的 7 个必选 Pass |
| **检查项** | inliner → combine → canonicalizer → reorder_broadcast → cse → licm → symbol_dce |

### UT-04 `_require_pass` / `_try_add_pass`

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | KPI-1.1 |
| **测试内容** | ① `_require_pass` 对不存在的 pass 抛 RuntimeError ② `_try_add_pass` 对不存在的 pass 返回 False 不报错 |

```python
def test_require_pass_raises():
    from triton_anchor.pipeline import _require_pass
    import types
    mock = types.SimpleNamespace()
    try:
        _require_pass(mock, 'nonexistent', None)
        assert False
    except RuntimeError:
        pass  # Expected

def test_try_add_pass_silent():
    from triton_anchor.pipeline import _try_add_pass
    import types
    mock = types.SimpleNamespace()
    assert _try_add_pass(mock, 'nonexistent', None) == False
```

### UT-05 Adapter 基类继承验证

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | 架构一致性 |
| **测试内容** | 验证 ABI 隔离的类型层级（v0.1.3 §5.1 Adapter 双变体设计） |
| **检查项** | ① `TritonLinalgAdapter` → `ILinalgPybindAdapter`（pybind 模式） ② `TritonSharedAdapter` → `ILinalgOptAdapter`（subprocess 模式） ③ `HybridAdapter` → `ILinalgOptAdapter`（subprocess 模式） ④ 所有 Linalg Adapter 的 `anchor_ir_track()` 返回 `AnchorIRTrack.LINALG` |

```python
def test_adapter_inheritance():
    from triton_anchor.adapters.base import ILinalgOptAdapter, ILinalgPybindAdapter
    from triton_anchor.adapters.triton_linalg_adapter import TritonLinalgAdapter
    from triton_anchor.adapters.triton_shared_adapter import TritonSharedAdapter
    from triton_anchor.adapters.hybrid_adapter import HybridAdapter
    assert issubclass(TritonLinalgAdapter, ILinalgPybindAdapter)
    assert issubclass(TritonSharedAdapter, ILinalgOptAdapter)
    assert issubclass(HybridAdapter, ILinalgOptAdapter)
```

### UT-06 DSL Extension 注册表

| 项目 | 内容 |
|------|------|
| **对应文件** | `run_tests.py::test_dsl_extensions` |
| **覆盖 KPI** | KPI-1.3 |
| **测试内容** | ① `DSLExtensionRegistry.discover()` 不报错 ② `BuiltinSpec` 创建正确 ③ `DSLExtensionPlugin` 接口完整 |

### UT-07 Plugin Registry 注册/发现

| 项目 | 内容 |
|------|------|
| **对应文件** | `test_registries.py::TestPluginRegistry` |
| **覆盖 KPI** | KPI-0.2, KPI-1.3 |
| **三方对齐** | **与 G03（零修改集成）、G10（后端数量 ≥3）共用验证逻辑**。G03 验证 `pip install` 后前端 hash 不变 + 后端自动发现；G10 验证 `≥3` 后端 + `≥2` 种范式。内部 UT 验证 Registry 的 API 正确性作为前置保障。 |
| **测试内容** | ① `register()` + `get()` ② `find_for_target()` ③ `list_plugins()` ④ `reset()` |

### UT-08 Plugin `get_allowed_dialects()`

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | KPI-1.2 |
| **三方对齐** | **与 G09（扩展方言白名单机制）共用验证逻辑**。G09 验证完整的声明→编译→放行流程；内部 UT 验证所有 Plugin 的 `get_allowed_dialects()` 接口契约正确。 |
| **测试内容** | 验证所有 plugin 的 `get_allowed_dialects()` 返回 `list` 类型 |

```python
def test_get_allowed_dialects():
    from triton_anchor.plugins.sophgo_plugin import SophgoPlugin
    from triton_anchor.plugins.spacemit_plugin import SpacemiTPlugin
    for P in [SophgoPlugin, SpacemiTPlugin]:
        ext = P().get_allowed_dialects()
        assert isinstance(ext, list)
```

### UT-09 OpCoverageMatrix

| 项目 | 内容 |
|------|------|
| **对应文件** | `test_op_coverage.py` |
| **覆盖 KPI** | KPI-2.2 |
| **测试内容** | ① 注册 plugin 后查询 op 状态 ② `find_gaps()` ③ `generate_report()` ④ `validate_kernel_ops()` ⑤ 空矩阵报告 |

### UT-10 HWCapability 创建与验证

| 项目 | 内容 |
|------|------|
| **对应文件** | `test_hw_capability.py` |
| **覆盖 KPI** | 架构一致性 |
| **测试内容** | ① Sophgo/SpacemiT/USC 三种范式创建 ② `to_gpu_target()` 兼容层 ③ 缺少 cap 抛 ValueError |

### UT-11 AnchorIRTrack 枚举与解耦

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | 架构一致性 |
| **测试内容** | ① `AnchorIRTrack.LINALG` / `TRITON_GPU` 枚举值 ② 字符串 `"linalg"` 自动解析为枚举 ③ Paradigm 与 Track 解耦（GPGPU + LINALG 允许） ④ `lowering_path` 属性兼容 |

```python
def test_track_decoupled():
    from triton_anchor.hw_capability import HWCapability, ComputeParadigm, GPGPUCapability
    from triton_anchor.anchor_ir import AnchorIRTrack
    hw = HWCapability(
        name='risc-v-gpu', arch_family='gpu',
        compute_paradigm=ComputeParadigm.GPGPU,
        anchor_ir_track=AnchorIRTrack.LINALG,  # GPGPU but Linalg!
        ptr_model='axis_info', gpgpu_cap=GPGPUCapability(),
    )
    assert hw.lowering_path == 'linalg'
```

### UT-12 Adapter Registry 选择逻辑

| 项目 | 内容 |
|------|------|
| **对应文件** | `test_registries.py::TestAdapterRegistry` |
| **测试内容** | ① `preferred_adapter` 优先选择 ② 不存在的 adapter 抛 `AdapterNotFoundError` ③ `list_adapters()` |

### UT-13 SophgoPlugin 导入路径

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | KPI-2.1 |
| **三方对齐** | **与 G10（后端集成数量）共用 HWCapability 声明检查**。G10 检查 `≥3` 后端 + `validate_environment()` 返回成功；内部 UT 单独验证 Sophgo 的 `hw_capability` 字段（`compute_paradigm=TENSOR_PROCESSOR`, `anchor_ir_track=LINALG`），确保与 G10 验证一致。 |
| **测试内容** | 验证 `SophgoPlugin` 可实例化、`hw_capability` 正确、`anchor_ir_track=LINALG` |

```python
def test_sophgo_plugin_creation():
    from triton_anchor.plugins.sophgo_plugin import SophgoPlugin
    p = SophgoPlugin()
    assert p.name == 'sophgo'
    assert p.hw_capability.anchor_ir_track.value == 'linalg'
```

### UT-14 USC TritonGPU 直通路径

| 项目 | 内容 |
|------|------|
| **对应文件** | `run_tests.py::test_usc_triton_gpu_path` |
| **三方对齐** | **与 G06（gpGPU 范式编译路径）共用 TritonGPU Track 验证逻辑**。G06 验证 gpGPU 全链路（TTIR → TritonGPU Track AnchorIR → Encoding）；内部 UT 验证 USC 的 `anchor_ir_track=TRITON_GPU` 声明和 Adapter 跳过逻辑，确保与 G06 编译路径选择一致。 |
| **测试内容** | 验证 `USCPlugin.hw_capability.lowering_path == "triton_gpu"`，编译器跳过 Adapter |

### UT-15 注释与 IR 解析

| 项目 | 内容 |
|------|------|
| **对应文件** | `test_anchor_ir.py::test_comments_ignored` |
| **测试内容** | 验证 `//` 和 `#` 注释行中的方言关键字不被误报 |

### UT-16 TritonVersionInfo Pass 探测

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | 架构一致性（v0.1.3 §4.4.4） |
| **测试内容** | ① 单例模式 `TritonVersionInfo.get()` 返回同一实例 ② `has_pass()` 对不存在的 Pass 返回 False ③ 无 triton 环境时不崩溃 |

### UT-17 HybridAdapter NotImplementedError

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | 架构一致性（v0.1.3 §5.1） |
| **测试内容** | `HybridAdapter.get_opt_tool_path()` 抛出 `NotImplementedError`，提示使用 triton-shared 或 triton-linalg |

```python
def test_hybrid_not_implemented():
    from triton_anchor.adapters.hybrid_adapter import HybridAdapter
    try:
        HybridAdapter().get_opt_tool_path()
        assert False
    except NotImplementedError:
        pass
```

### UT-18 Adapter 选择逻辑（ptr_model fallback）

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | 架构一致性（v0.1.3 §5.1 `get_adapter()`） |
| **测试内容** | ① `preferred_adapter` 优先 ② TritonGPU Track 自动选 TritonGPUAdapter ③ `ptr_model="structured"` → TritonSharedAdapter ④ `ptr_model="axis_info"` → TritonLinalgAdapter |

### UT-19 统一编译流程 6 Hook 注入点

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | 架构一致性（v0.1.3 §3.4） |
| **测试内容** | 验证 `unified_compile` 中 6 个 Hook 的执行顺序：① DSL Extension → ② on_ttir_ready → ③ Adapter.convert → ④ validate_pre_hook → on_anchor_ir_ready → validate_post_hook → ⑤ lower_anchor_ir_to_target → ⑥ create_launcher |

### UT-20 DSL Extension 后端兼容性检查

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | KPI-1.3（v0.1.3 §6.4 `validate_kernel()`） |
| **测试内容** | ① 使用 `target_backend="spacemit"` 的 DSL Extension 编译到 sophgo 后端时抛 `IncompatibleExtensionError` ② 编译到 spacemit 后端时通过 |

### UT-21 TritonGPU Track Encoding 覆盖检查

| 项目 | 内容 |
|------|------|
| **覆盖 KPI** | KPI-1.2（v0.1.3 §3.2.1 `validate_encoding_coverage()`） |
| **三方对齐** | **与 G06（gpGPU 范式编译路径）共用 Encoding 校验逻辑**。G06 验证 `triton_gpu.*` 方言正确且携带 Encoding；内部 UT 单独验证 `validate_encoding_coverage()` 对缺失 Encoding 的 tensor 正确报错，确保与 G06 的通过标准一致。 |
| **测试内容** | TritonGPU Track 的 `validate_post_hook` 额外检查：所有 tensor 类型必须携带 Encoding 属性（BlockedEncoding / MmaEncoding / SharedEncoding） |

---

## 3 第三方验收测试用例（ST）

> 以下用例来自第三方测试大纲（test20260429.docx），需企业硬件平台。

### 3.0 前置验收（G01–G03）

#### G01 开源发布可获取性验证

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | 仓库 public 可访问，含 README / LICENSE / pyproject.toml |
| **通过标准** | 仓库可访问，许可证明确 |

#### G02 第三方独立安装与运行验证

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | 干净环境 `git clone → pip install -e . → import triton_anchor → 编译 GEMM → AnchorIR 产出` |
| **通过标准** | 全部步骤无报错 |

#### G03 企业后端零修改集成验证

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | 记录 commit hash H0 → `pip install` 3 款后端 → hash 不变 → 各后端编译执行 GEMM |
| **通过标准** | 前端零修改，3 后端均成功 |

### 3.1 任务一：统一编译前端架构（G04–G09）

#### G04 Matrix AME 范式编译路径

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | AME 后端 → TTIR → TritonSharedAdapter → Linalg Track AnchorIR → `validate_anchor_ir()` |
| **通过标准** | AnchorIR 中无 `tt.*`，验证通过 |

#### G05 Tensor 范式编译路径

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | Tensor 后端 → TTIR → TritonLinalgAdapter → Linalg Track AnchorIR |
| **通过标准** | `tt.dot` 降级为 `linalg.matmul`，验证通过 |

#### G06 gpGPU 范式编译路径

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | gpGPU 后端 → TTIR → TritonGPU Track AnchorIR（携带 Encoding） |
| **通过标准** | `triton_gpu.*` 方言正确，验证通过 |

#### G07 Linalg Track 白名单合规

| 项目 | 内容 |
|------|------|
| **测试级别** | UT |
| **测试内容** | 10 个不同 kernel（GEMM/Softmax/LayerNorm/Add/Mul/ReLU/GELU/Transpose/Copy/Reduce）扫描白名单 |
| **通过标准** | 10/10 通过 |

#### G08 禁止方言拦截

| 项目 | 内容 |
|------|------|
| **测试级别** | UT |
| **测试内容** | 构造含 `tts.make_tensor_ptr` 的非法 IR → `validate_anchor_ir()` 抛异常 |
| **通过标准** | 正确抛出 `AnchorIRViolation` |

#### G09 扩展方言白名单机制

| 项目 | 内容 |
|------|------|
| **测试级别** | UT |
| **测试内容** | 后端声明 `get_allowed_dialects() → ["xsmt"]`，编译含 `xsmt.*` Op 的 IR，验证放行 |
| **通过标准** | 扩展方言通过验证 |

### 3.2 任务二：全链路编译集成（G10–G14）

#### G10 后端集成数量验证

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | 列举已安装后端 → `validate_environment()` → 统计数量与范式 |
| **通过标准** | ≥ 3 后端，≥ 2 种范式 |

#### G11 GEMM 全链路（参数化）

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **参数化** | 3 后端 × 2 dtype(FP32/FP16) × 3 shape(64²/512²/1024²) |
| **通过标准** | FP32 rtol=1e-5, FP16 rtol=1e-3 |

#### G12 Softmax 全链路

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **参数化** | 3 后端 × 2 dtype × 3 shape |
| **通过标准** | 数值在容差内 |

#### G13 LayerNorm 全链路

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **参数化** | 3 后端 × 2 dtype × hidden_size(256/768/1024) |

#### G14 Elementwise 全链路

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **参数化** | 3 后端 × 5 算子(Add/Mul/ReLU/GELU/Sigmoid) × 2 dtype |

### 3.3 任务三：一致性验证（G15–G17）

#### G15 性能 A/B 对比（GEMM）

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | A 路径(企业原生) vs B 路径(triton-anchor)，预热 10 次 + 执行 100 次取中位数 |
| **通过标准** | R = T_A / T_B ≥ 0.95 |
| **参数化** | 后端 × dtype(FP32/FP16) × shape(512²/1024²/2048²) |

#### G16 浮点容差一致性

| 项目 | 内容 |
|------|------|
| **测试级别** | ST |
| **测试内容** | CPU/NumPy 基准 vs 各后端执行 GEMM/Softmax/LayerNorm |
| **通过标准** | FP32 rtol≤1e-5 atol≤1e-6；FP16 rtol≤1e-3 atol≤1e-4 |

#### G17 整数 Bitwise 一致性

| 项目 | 内容 |
|------|------|
| **测试级别** | UT |
| **测试内容** | 纯整数运算 kernel（argmax/histogram）跨后端对比 |
| **通过标准** | 所有后端输出 bitwise identical |

---

## 4 测试执行

### 4.1 内部 UT 执行

```bash
# 方式一：无依赖运行（推荐 CI）
cd triton-anchor && python3 run_tests.py

# 方式二：pytest（需安装 pytest）
pip install pytest
pytest triton_anchor/tests/ -v
```

### 4.2 第三方 ST 执行

```bash
# 安装企业后端
pip install triton-backend-sophgo triton-backend-spacemit triton-backend-usc

# 执行全链路测试
pytest third_party_test_cases/ -v --backend=sophgo
pytest third_party_test_cases/ -v --backend=spacemit
pytest third_party_test_cases/ -v --backend=usc
```

### 4.3 测试汇总矩阵

> **三方覆盖** 列标注该内部 UT 在第三方测试大纲中对应的用例编号。标注 `✅ Gxx` 表示第三方已覆盖同等验证点；`—` 表示纯内部用例，第三方不涉及。

| 编号 | 测试项 | 级别 | 硬件 | 自动化 | 覆盖 KPI | 架构章节 | 三方覆盖 |
|------|-------|------|------|-------|---------|---------|---------|
| UT-01 | AnchorIR 双轨白名单 | UT | ❌ | ✅ | KPI-1.2 | §3.2.1 | ✅ G07, G08 |
| UT-02 | 两阶段验证 | UT | ❌ | ✅ | KPI-1.2 | §3.2.1 | ✅ G09 |
| UT-03 | TTIR 7 Pass 不变量 | UT | ❌ | ✅ | KPI-1.1 | §4.1 | ✅ G04–G06 (隐含) |
| UT-04 | _require_pass/_try_add_pass | UT | ❌ | ✅ | KPI-1.1 | §4.1.2 | — |
| UT-05 | Adapter 基类继承 | UT | ❌ | ✅ | 架构 | §5.1 | — |
| UT-06 | DSL Extension 注册 | UT | ❌ | ✅ | KPI-1.3 | §6.1–6.4 | — |
| UT-07 | Plugin Registry | UT | ❌ | ✅ | KPI-0.2 | §7.1–7.2 | ✅ G03, G10 |
| UT-08 | get_allowed_dialects() | UT | ❌ | ✅ | KPI-1.2 | §3.2.1 | ✅ G09 |
| UT-09 | OpCoverageMatrix | UT | ❌ | ✅ | KPI-2.2 | §5.4 | — |
| UT-10 | HWCapability 创建 | UT | ❌ | ✅ | 架构 | §4.2 | — |
| UT-11 | AnchorIRTrack 解耦 | UT | ❌ | ✅ | 架构 | §4.2 | — |
| UT-12 | Adapter Registry 选择 | UT | ❌ | ✅ | 架构 | §5.1 | — |
| UT-13 | SophgoPlugin 创建 | UT | ❌ | ✅ | KPI-2.1 | §7.2 | ✅ G10 (部分) |
| UT-14 | USC TritonGPU 直通 | UT | ❌ | ✅ | 架构 | §5.1 | ✅ G06 (部分) |
| UT-15 | 注释行不误报 | UT | ❌ | ✅ | KPI-1.2 | §3.2.1 | — |
| UT-16 | TritonVersionInfo Pass 探测 | UT | ❌ | ✅ | 架构 | §4.4.4 | — |
| UT-17 | HybridAdapter NotImplementedError | UT | ❌ | ✅ | 架构 | §5.1 | — |
| UT-18 | Adapter ptr_model 选择 | UT | ❌ | ✅ | 架构 | §5.1 | — |
| UT-19 | 6 Hook 注入顺序 | UT | ❌ | ✅ | 架构 | §3.4 | — |
| UT-20 | DSL Extension 后端兼容 | UT | ❌ | ✅ | KPI-1.3 | §6.4 | — |
| UT-21 | TritonGPU Encoding 覆盖 | UT | ❌ | ✅ | KPI-1.2 | §3.2.1 | ✅ G06 (部分) |
| G01 | 开源发布 | ST | ❌ | 🔲 | KPI-0.1 | §1.4 | ✅ 三方 |
| G02 | 第三方安装 | ST | ❌ | 🔲 | KPI-0.1 | §1.4 | ✅ 三方 |
| G03 | 零修改集成 | ST | ✅ | 🔲 | KPI-0.2 | §3.7 | ✅ 三方 |
| G04 | AME 范式路径 | ST | ✅ | 🔲 | KPI-1.1 | §2.1 | ✅ 三方 |
| G05 | Tensor 范式路径 | ST | ✅ | 🔲 | KPI-1.1 | §2.1 | ✅ 三方 |
| G06 | gpGPU 范式路径 | ST | ✅ | 🔲 | KPI-1.1 | §2.1 | ✅ 三方 |
| G07 | 白名单合规 | UT | ❌ | ✅ | KPI-1.2 | §3.2.1 | ✅ 三方 |
| G08 | 禁止方言拦截 | UT | ❌ | ✅ | KPI-1.2 | §3.2.1 | ✅ 三方 |
| G09 | 扩展方言放行 | UT | ❌ | ✅ | KPI-1.2 | §3.2.1 | ✅ 三方 |
| G10 | 后端数量 ≥3 | ST | ✅ | 🔲 | KPI-2.1 | §7.1 | ✅ 三方 |
| G11 | GEMM 全链路 | ST | ✅ | 🔲 | KPI-2.1 | §8.2 | ✅ 三方 |
| G12 | Softmax 全链路 | ST | ✅ | 🔲 | KPI-2.1 | §8.2 | ✅ 三方 |
| G13 | LayerNorm 全链路 | ST | ✅ | 🔲 | KPI-2.1 | §8.2 | ✅ 三方 |
| G14 | Elementwise 全链路 | ST | ✅ | 🔲 | KPI-2.1 | §8.2 | ✅ 三方 |
| G15 | 性能 A/B 对比 | ST | ✅ | 🔲 | KPI-3.1 | §8.2 | ✅ 三方 |
| G16 | 浮点容差 | ST | ✅ | 🔲 | KPI-3.2 | §8.2 | ✅ 三方 |
| G17 | 整数 Bitwise | UT | ✅ | 🔲 | KPI-3.2 | §8.2 | ✅ 三方 |

### 4.4 内部 UT 与三方覆盖统计

| 分类 | 用例数 | 说明 |
|------|-------|------|
| **三方已覆盖** | 8 个 UT | UT-01/02/03/07/08/13/14/21 在三方测试中有对应验证点 |
| **纯内部** | 13 个 UT | UT-04/05/06/09/10/11/12/15/16/17/18/19/20 为内部独有 |
| **三方 ST** | 17 个 | G01–G17 全部来自三方测试大纲 |

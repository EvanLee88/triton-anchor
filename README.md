# triton-anchor

**统一 Triton 编译前端 v0.1** — 面向多款芯片的共性编译前端，在统一的 Triton 编译前端中同时支持 **RISC-V Matrix 扩展指令集**（**AME）**、**RISC-V Tensor 扩展指令集**和 **SIMT 扩展指令集（GPGPU）** 的插件化架构。

## 架构概览

```
Layer 0  — DSL Extensions       (triton.dsl_extensions entry_points)
Layer 1  — TTIR Pipeline        (7 mandatory passes + _require_pass/_try_add_pass)
Layer 2  — Linalg Adapters      (ILinalgOptAdapter / ILinalgPybindAdapter)
Layer 2.5 — AnchorIR Spec       (dual-track whitelist + two-phase validation)
Layer 3  — Backend Plugins      (triton.backends entry_points)
```

### 核心设计特性

- **双轨 AnchorIR**：Linalg Track（AME / Tensor）与 TritonGPU Track（gpGPU）独立白名单
- **两阶段验证**：`validate_pre_hook()` → `on_anchor_ir_ready()` → `validate_post_hook()` 确保后端注入的 Op 也受契约约束
- **ABI 隔离**：`ILinalgOptAdapter`（subprocess 模式）与 `ILinalgPybindAdapter`（pybind 模式）在类型层面隔离 C++ ABI
- **Paradigm / Track 解耦**：`ComputeParadigm` 与 `AnchorIRTrack` 独立声明，后端可自由组合

## 支持的计算范式

| 范式 | 硬件 | 后端 | Adapter | AnchorIR Track | 状态 |
|------|------|------|---------|---------------|------|
| **Tensor Processor** | Sophgo BM1684X | `sophgo` | `ILinalgPybindAdapter` | Linalg | ✅ 适配中 |
| **AME Matrix** | SpacemiT X60 | `spacemit` | `ILinalgOptAdapter` | Linalg | 🔲 Stub |
| **gpGPU** | USC GPU | `usc` | (TritonGPU 直通) | TritonGPU | 🔲 Stub |
| **Reference** | CPU | `cpu` | — | Linalg | 🔲 Stub |

## 快速开始

```python
from triton_anchor import HWCapability, ComputeParadigm, AnchorIRTrack
from triton_anchor.plugins.sophgo_plugin import SophgoPlugin
from triton_anchor.adapters.triton_linalg_adapter import TritonLinalgAdapter
from triton_anchor.adapters.registry import AdapterRegistry
from triton_anchor.plugins.registry import PluginRegistry

# 注册后端
plugin = SophgoPlugin()
PluginRegistry.register(plugin)

# 注册适配器
adapter = TritonLinalgAdapter()
AdapterRegistry.register(adapter)

# 查看硬件能力
hw = plugin.hw_capability
print(f"Backend:    {plugin.name}")
print(f"Paradigm:   {hw.compute_paradigm.value}")
print(f"IR Track:   {hw.anchor_ir_track.value}")
print(f"Adapter:    {hw.preferred_adapter}")
print(f"GPUTarget:  {hw.to_gpu_target()}")

# 两阶段 AnchorIR 验证
from triton_anchor.anchor_ir import AnchorIRValidator

validator = AnchorIRValidator(track=hw.anchor_ir_track)
# Phase 1: pre-hook（基础白名单）
pre_violations = validator.validate_pre_hook(ir_text)
# ... on_anchor_ir_ready() ...
# Phase 2: post-hook（含后端扩展白名单）
post_violations = validator.validate_post_hook(ir_text, ext_allowed=set(plugin.get_allowed_dialects()))

# 查看 Op 覆盖度
from triton_anchor.op_coverage import OpCoverageMatrix
from triton_anchor.plugins.spacemit_plugin import SpacemiTPlugin

matrix = OpCoverageMatrix()
matrix.register_plugin(plugin)
matrix.register_plugin(SpacemiTPlugin())
print(matrix.generate_report())
```

## 目录结构

```
triton_anchor/
├── __init__.py              # 公共 API: HWCapability, ComputeParadigm, AnchorIRTrack
├── hw_capability.py         # HWCapability + ComputeParadigm + AnchorIRTrack 解耦
├── anchor_ir.py             # AnchorIR 双轨规范 + 两阶段验证器
├── pipeline.py              # 统一 TTIR Pipeline (7 pass + _require_pass + _try_add_pass)
├── compiler.py              # 统一编译入口 (两阶段验证 + 6 Hook 注入点)
├── op_coverage.py           # 算子覆盖矩阵
│
├── adapters/                # Layer 2: Linalg Adapters
│   ├── base.py              # ILinalgOptAdapter (subprocess) / ILinalgPybindAdapter (pybind)
│   ├── registry.py          # Adapter 注册表
│   ├── triton_linalg_adapter.py  # ✅ ILinalgPybindAdapter — triton_race 管线封装
│   ├── triton_shared_adapter.py  # 🔲 ILinalgOptAdapter  — triton-shared (SpacemiT)
│   └── hybrid_adapter.py        # 🔲 ILinalgOptAdapter  — Hybrid (Future Work)
│
├── plugins/                 # Layer 3: Backend Plugins
│   ├── base.py              # BackendPlugin 接口 (含 get_allowed_dialects())
│   ├── registry.py          # Plugin 注册表
│   ├── sophgo_plugin.py     # ✅ Sophgo TPU (wraps triton_race)
│   ├── spacemit_plugin.py   # 🔲 SpacemiT RISC-V AME
│   ├── usc_plugin.py        # 🔲 USC GPU (TritonGPU Track)
│   └── cpu_plugin.py        # 🔲 CPU Reference
│
├── extensions/              # Layer 0: DSL Extensions
│   ├── base.py              # DSLExtensionPlugin 接口
│   └── registry.py          # Extension 注册表
│
├── language/ext/            # DSL 扩展命名空间 (预留)
└── tests/                   # 单元测试
```

## 五个核心不变量

| 不变量 | 内容 | 稳定性 |
|--------|------|--------|
| **TTIR Pipeline** | 7 必选 Pass + `_require_pass` 关键路径保护 | 仅增不删 |
| **HWCapability** | 声明式硬件能力 + `AnchorIRTrack` 枚举 | 字段仅增不删 |
| **AnchorIR 双轨** | Linalg / TritonGPU 双轨白名单 + 禁止 `tt.*`/`smt.*` | 仅增不删 |
| **两阶段验证** | `pre_hook` → Hook 注入 → `post_hook` | 执行顺序不变 |
| **Plugin 接口** | `BackendPlugin` / `ILinalgOptAdapter` / `ILinalgPybindAdapter` / `DSLExtensionPlugin` | 向后兼容 |

## 编译流程

```
AST → TTIR
  │
  ├── Layer 1: build_ttir_pipeline (7 pass)
  │     ├── inliner → combine → canonicalizer → reorder_broadcast → cse → licm → symbol_dce
  │     └── [GPU] _require_pass(add_rewrite_tensor_pointer)
  │
  ├── Hook ②: on_ttir_ready
  │
  ├── Layer 2: Adapter.convert()
  │     ├── Linalg Track: TritonLinalgAdapter (pybind) / TritonSharedAdapter (subprocess)
  │     └── TritonGPU Track: 直通后端
  │
  ├── Layer 2.5: 两阶段 AnchorIR 验证
  │     ├── validate_pre_hook()  ← 基础白名单
  │     ├── Hook ④: on_anchor_ir_ready()
  │     └── validate_post_hook() ← 基础 + 扩展白名单
  │
  └── Layer 3: plugin.lower_anchor_ir_to_target() → binary
```

## 安装

```bash
# 开发模式安装
pip install -e .

# 运行测试
pip install -e ".[dev]"
pytest triton_anchor/tests/ -v

# 安装 Sophgo 后端依赖
pip install -e ".[sophgo]"
# 需要单独安装 triton_race 并配置 PPL_PROJECT_ROOT、PPLCOMPILE_PATH
```

## License

Apache 2.0

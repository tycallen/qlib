# 多GPU训练诊断工具集

本文件夹包含了用于诊断和修复Qlib多GPU训练问题的完整工具集。

## 📂 文件结构

### 🔧 诊断脚本

| 文件 | 用途 | 使用场景 |
|------|------|---------|
| **test_multi_gpu_minimal.py** | 最小化DataParallel测试 | 快速验证环境是否支持多GPU |
| **diagnose_segfault_factors.py** | Segfault因子隔离测试 | 测试Dropout、GRU、层数等因素 |
| **diagnose_pytorch_version.py** | PyTorch版本兼容性诊断 | 深度诊断PyTorch版本问题 |
| **diagnose_multi_gpu.py** | 完整多GPU诊断 | 全面的环境和配置测试 |

### 🛠️ 修复工具

| 文件 | 用途 |
|------|------|
| **fix_pytorch_env.sh** | PyTorch环境自动修复脚本 |

### 📚 文档

| 文件 | 内容 |
|------|------|
| **PYTORCH_VERSION_FIX.md** | PyTorch版本问题解决方案 |
| **PYTORCH_DEPENDENCY_FIX.md** | 依赖冲突修复指南 |
| **MULTI_GPU_DEBUG_GUIDE.md** | 完整的多GPU调试指南 |
| **GRU_CONFIG_EXAMPLES.md** | GRU模型配置示例 |

## 🚀 快速开始

### 1. 快速测试环境

```bash
cd /data10/tyc/stock/qlib_own/diagnostic_tools

# 最快速的测试
python test_multi_gpu_minimal.py
```

**预期结果**：
```
✓ 单GPU LSTM测试通过
✓ DataParallel推理通过
✓ DataParallel训练通过
```

### 2. 如果出现Segfault

```bash
# 因子隔离测试
python diagnose_segfault_factors.py
```

这会告诉您哪些配置可用（如无Dropout、GRU等）。

### 3. 如果是版本问题

```bash
# 版本诊断
python diagnose_pytorch_version.py
```

根据输出查看 `PYTORCH_VERSION_FIX.md` 获取解决方案。

### 4. 完整诊断

```bash
# 全面诊断
python diagnose_multi_gpu.py
```

## 📖 常见问题解决流程

### 问题1: Segmentation Fault

1. 运行 `test_multi_gpu_minimal.py`
2. 如果失败 → 运行 `diagnose_segfault_factors.py`
3. 查看哪个配置通过
4. 参考 `MULTI_GPU_DEBUG_GUIDE.md`

### 问题2: ImportError (ONNX)

1. 查看 `PYTORCH_DEPENDENCY_FIX.md`
2. 执行推荐的PyTorch版本降级/升级

### 问题3: 不知道用什么配置

1. 查看 `GRU_CONFIG_EXAMPLES.md`
2. 选择适合的配置模板

## 🎯 推荐配置

根据诊断结果，我们发现：

| PyTorch版本 | 多GPU | 推荐度 |
|------------|-------|--------|
| **2.0.1** | ✅ 完美 | ⭐⭐⭐⭐⭐ |
| 2.2.0 | ⚠️ ONNX bug | ⭐⭐ |
| 2.3.0 | ✅ 可用 | ⭐⭐⭐⭐ |
| 2.9.1 | ❌ Segfault | ❌ |

**推荐安装**：
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

## 📝 使用历史

这些工具是在解决以下问题时创建的：

1. **PyTorch 2.9.1 DataParallel Segfault** - 发现是版本bug
2. **ONNX ImportError (2.2.0)** - 提供降级方案
3. **依赖冲突** - 创建自动修复脚本

## 🔗 相关代码

修改的主要代码文件：
- `../qlib/contrib/model/pytorch_gats_ts.py` - GATs多GPU支持
- `../qlib/contrib/model/pytorch_gru_ts.py` - GRU多GPU支持

## 💡 贡献者提示

如果需要添加新的诊断工具：
1. 遵循现有命名规范 `diagnose_*.py`
2. 在README中更新文件列表
3. 提供清晰的输出和错误信息

## 📞 获取帮助

如果诊断工具无法解决问题：
1. 运行所有诊断脚本
2. 收集输出结果
3. 查看对应的文档
4. 考虑环境重建或使用单GPU

---

**最后更新**: 2025-11-25
**PyTorch推荐版本**: 2.0.1
**CUDA推荐版本**: 11.8

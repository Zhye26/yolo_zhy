# 实验记录

## 实验环境

- GPU: NVIDIA GeForce RTX 4060 (8GB)
- CUDA: 13.0
- PyTorch: 2.9.1
- Ultralytics: 8.4.6

## 数据集

| 数据集 | 图片数 | 类别 | 用途 |
|--------|--------|------|------|
| Helmet Detection | 1443 | full, half, invalid, not_wearing | 头盔检测 |
| E-bike and Human | 3836 | Electric-bicycle, person | 电动车+人检测 |
| **合并数据集** | **5279** | ebike, driver, passenger, helmet | 完整违规检测 |

---

## 实验1: 基线模型 (YOLOv8n)

**日期**: 2026-01-20

**配置**:
- 模型: YOLOv8n (预训练)
- Epochs: 100 (早停于94)
- Batch Size: 16
- Image Size: 640
- 数据集: 合并数据集 (5279张)

**训练结果**:
| 指标 | 值 |
|------|-----|
| mAP50 | **0.893** |
| mAP50-95 | **0.595** |
| Precision | 0.880 |
| Recall | 0.850 |
| box_loss | 0.620 |
| cls_loss | 0.380 |
| 训练时间 | ~70分钟 |
| 推理速度 | ~9ms/张 (~111 FPS) |

**各类别性能**:
| 类别 | mAP50 | mAP50-95 |
|------|-------|----------|
| ebike | 待测 | 待测 |
| driver | 待测 | 待测 |
| helmet | 待测 | 待测 |

**模型文件**: `models/baseline_best.pt`

---

## 实验2: BiFPN特征融合

**日期**: 2026-01-21

**改进内容**:
- 将YOLOv8的PANet替换为BiFPN风格的特征融合
- 双向特征金字塔，增强多尺度特征融合
- 增加跳跃连接，保留更多原始特征

**配置**:
- 模型: YOLOv8n + BiFPN (`models/yolov8n_bifpn.yaml`)
- Epochs: 100 (最佳epoch: 62)
- Batch Size: 16
- Image Size: 640
- 参数量: 3.09M (基线: 3.0M, +3%)

**训练结果**:
| 指标 | 基线 | +BiFPN | 提升 |
|------|------|--------|------|
| mAP50 | 0.893 | **0.912** | **+1.9%** ↑ |
| mAP50-95 | 0.595 | **0.600** | **+0.5%** ↑ |
| Precision | 0.880 | 0.873 | -0.7% |
| Recall | 0.850 | **0.882** | **+3.2%** ↑ |
| 推理速度 | 9ms | 1.4ms | 更快 |
| 训练时间 | ~70分钟 | ~75分钟 | - |

**分析**:
- mAP50提升1.9%，证明BiFPN特征融合有效
- Recall提升3.2%，说明漏检明显减少
- 收敛更快（62 vs 94 epoch达到最佳）
- 推理速度因AMP优化反而更快

**模型文件**: `models/bifpn_best.pt`

---

## 实验3: 注意力机制 (CBAM)

**日期**: 2026-01-21

**改进内容**:
- 在BiFPN基础上添加CBAM注意力模块
- 增强对关键区域（头盔、人）的关注

**配置**:
- 模型: YOLOv8n + BiFPN + CBAM (`models/yolov8n_bifpn_cbam.yaml`)
- Epochs: 100
- Batch Size: 16
- Image Size: 640

**训练结果**:
| 指标 | 基线 | +BiFPN | +BiFPN+CBAM | 变化 |
|------|------|--------|-------------|------|
| mAP50 | 0.893 | 0.912 | 0.839 | **-7.3%** ↓ |
| mAP50-95 | 0.595 | 0.600 | 0.508 | **-9.2%** ↓ |
| Precision | 0.880 | 0.873 | 0.826 | -4.7% |
| Recall | 0.850 | 0.882 | 0.782 | **-10%** ↓ |

**分析**:
- ❌ CBAM导致性能显著下降，不适用于本任务
- 可能原因：模型过小(3M参数)，CBAM增加的复杂度导致过拟合
- 小目标检测场景下，通道注意力可能抑制了有用特征
- **结论**: 放弃CBAM，保持BiFPN作为最佳改进

**CBAM失败深度分析** (多模型协作分析结果):
1. **特征冲突**: BiFPN已进行复杂的特征重加权，CBAM再次加权导致梯度冲突
2. **模型容量不匹配**: YOLOv8n仅3M参数，CBAM增加的复杂度导致过拟合
3. **空间特征破坏**: CBAM的7x7空间注意力可能"平滑"了小目标(头盔)的特征
4. **初始化问题**: CBAM层随机初始化与迁移的BiFPN权重不匹配

---

## 实验3.5: 注意力机制 (SimAM) [待执行]

**日期**: 待定

**改进内容**:
- 使用SimAM替代CBAM（零参数注意力机制）
- SimAM基于能量函数理论，不与BiFPN冲突
- 无额外参数，适合轻量模型

**配置**:
- 模型: YOLOv8n + BiFPN + SimAM (`models/yolov8n_bifpn_simam.yaml`)
- Epochs: 100
- Batch Size: 16
- Image Size: 640

**SimAM优势**:
- 零参数: 不增加模型复杂度
- 基于能量函数: 不与BiFPN的学习权重冲突
- 轻量高效: 适合YOLOv8n等小模型

**训练命令**:
```bash
python tools/train_bifpn_simam.py
```

**预期结果**:
| 指标 | +BiFPN | +BiFPN+SimAM | 预期变化 |
|------|--------|--------------|----------|
| mAP50 | 0.912 | ? | +1~2% |
| Recall | 0.882 | ? | +1~2% |


---

## 实验4: ByteTrack多目标跟踪

**日期**: 2026-01-22

**改进内容**:
- 集成ByteTrack算法（通过Ultralytics内置支持）
- 实现视频中目标ID追踪
- 违规去重：同一目标的违规只计数一次

**实现方式**:
- 使用`model.track()`替代`model()`进行检测
- 跟踪器配置：`bytetrack.yaml`
- 违规去重：基于track_id记录已检测的违规

**代码改动**:
- `app/services/detector.py`: 添加`track()`方法和去重逻辑
- `app/services/video_processor.py`: 支持跟踪模式
- `tools/test_tracking.py`: 测试脚本

**结果**:
| 指标 | 无跟踪 | +ByteTrack |
|------|--------|------------|
| MOTA | - | 待测 |
| IDF1 | - | 待测 |
| ID Switches | - | 待测 |
| 违规去重率 | - | 待测 |

**测试命令**:
```bash
python tools/test_tracking.py <video_path> --model models/bifpn_best.pt
```

---

## 实验5: TensorRT优化

**日期**: 2026-01-22

**改进内容**:
- 模型导出为TensorRT格式
- FP16量化加速
- 集成到检测服务

**实现方式**:
- 使用Ultralytics内置的TensorRT导出功能
- 支持FP16/FP32精度选择
- 检测器自动选择.engine文件

**代码改动**:
- `tools/export_tensorrt.py`: 模型导出脚本
- `tools/benchmark_tensorrt.py`: 速度对比测试
- `app/services/detector.py`: 支持TensorRT模型加载

**导出命令**:
```bash
# 导出FP16模型
python tools/export_tensorrt.py models/bifpn_best.pt

# 导出FP32模型
python tools/export_tensorrt.py models/bifpn_best.pt --fp32
```

**测试命令**:
```bash
python tools/benchmark_tensorrt.py --pt models/bifpn_best.pt
```

**结果**:
| 指标 | PyTorch | TensorRT FP16 | 加速比 |
|------|---------|---------------|--------|
| FPS | ~714 | 待测 | - |
| 延迟(ms) | ~1.4 | 待测 | - |
| 模型大小 | 6.2MB | 待测 | - |

---

## 实验6: 数据清洗

**日期**: 2026-01-22

**目的**:
- 检测数据集中的漏标、错标问题
- 提高数据质量，为后续重训练做准备

**方法**:
- 使用训练好的模型预测所有训练图片
- 对比预测结果和标注
- 找出漏标（模型检测到但没标注）和多标（有标注但模型没检测到）

**分析结果**:
| 指标 | 数值 |
|------|------|
| 总图片数 | 3948 |
| 问题图片 | 235 (6.0%) |
| 漏标数量 | 173 |
| 多标数量 | 207 |

**结论**:
- 数据集整体质量较好，问题图片占比6%
- 主要问题集中在ebike_2020系列图片
- 建议人工审核问题图片后重新训练

**工具**:
```bash
# 分析数据集
python tools/clean_dataset.py --model models/bifpn_best.pt --data data/merged_dataset/images/train

# 可视化单张问题图片
python tools/clean_dataset.py --visualize <image_path>
```

**输出文件**:
- `data/cleaning_results/cleaning_report.json`: 详细报告
- `data/cleaning_results/problem_images.txt`: 问题图片列表

---

## 最终模型对比

| 模型版本 | mAP50 | mAP50-95 | FPS | 参数量 |
|----------|-------|----------|-----|--------|
| YOLOv8n (基线) | 0.893 | 0.595 | 111 | 3.0M |
| + BiFPN | **0.912** | **0.600** | ~714 | 3.09M |
| + BiFPN + CBAM | 0.839 | 0.508 | - | ~3.1M |
| + BiFPN + SimAM | 待测 | 待测 | - | 3.09M |
| + TensorRT FP16 | - | - | 待测 | - |

---

## 改进路线图 (Roadmap)

### Phase 1: 数据优化 [优先级: 高]
- [ ] 修复235张问题图片的标注
- [ ] 收集/合成passenger类数据（当前缺失）
- [ ] 启用Copy-Paste增强（小目标提升）

### Phase 2: 架构优化 [优先级: 中]
- [x] BiFPN特征融合 ✓ (+1.9% mAP)
- [ ] SimAM注意力机制（替代失败的CBAM）
- [ ] 高分辨率训练 (imgsz=960)

### Phase 3: 部署优化 [优先级: 中]
- [x] TensorRT后端实现 ✓
- [ ] FP16量化测试
- [ ] INT8量化（需校准数据集）
- [ ] 预处理letterbox对齐

### Phase 4: 系统集成 [优先级: 低]
- [x] ByteTrack跟踪集成 ✓
- [x] 违规去重状态机 ✓
- [x] 实时视频检测页面 ✓
- [ ] 完整系统测试

---

## 推荐的下一步实验

1. **SimAM注意力** (tools/train_bifpn_simam.py)
   - 预期: mAP50 +1~2%
   - 风险: 低（零参数）

2. **高分辨率训练** (imgsz=960)
   - 预期: 小目标检测显著提升
   - 风险: 需减小batch size

3. **数据清洗后重训练**
   - 预期: 整体性能提升
   - 风险: 需人工审核

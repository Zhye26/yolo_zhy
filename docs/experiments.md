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
| E-bike and Human | TBD | ebike, human | 电动车+人检测 |
| 合并数据集 | TBD | ebike, driver, passenger, helmet | 完整违规检测 |

---

## 实验1: 基线模型 (YOLOv8n)

**日期**: 2026-01-20

**配置**:
- 模型: YOLOv8n
- Epochs: 50
- Batch Size: 16
- Image Size: 640

**结果** (头盔数据集):
| 指标 | 值 |
|------|-----|
| mAP50 | TBD |
| mAP50-95 | TBD |
| Precision | TBD |
| Recall | TBD |
| FPS | ~111 |

---

## 实验2: BiFPN特征融合

**日期**: TBD

**改进内容**:
- 将YOLOv8的PANet替换为BiFPN
- 双向特征金字塔，增强多尺度特征融合

**结果**:
| 指标 | 基线 | +BiFPN | 提升 |
|------|------|--------|------|
| mAP50 | - | - | - |
| mAP50-95 | - | - | - |
| FPS | - | - | - |

---

## 实验3: 注意力机制

**日期**: TBD

**改进内容**:
- 添加CBAM/SE/GLSA注意力模块
- 增强对关键区域（头盔、人）的关注

**结果**:
| 指标 | 基线 | +BiFPN | +Attention | 提升 |
|------|------|--------|------------|------|
| mAP50 | - | - | - | - |
| mAP50-95 | - | - | - | - |

---

## 实验4: ByteTrack多目标跟踪

**日期**: TBD

**改进内容**:
- 集成ByteTrack算法
- 实现视频中目标ID追踪
- 避免重复计数违规

**结果**:
| 指标 | 无跟踪 | +ByteTrack |
|------|--------|------------|
| MOTA | - | - |
| IDF1 | - | - |
| ID Switches | - | - |

---

## 实验5: TensorRT优化

**日期**: TBD

**改进内容**:
- 模型量化 (FP16)
- TensorRT推理引擎

**结果**:
| 指标 | PyTorch | TensorRT FP16 | 加速比 |
|------|---------|---------------|--------|
| FPS | - | - | - |
| 延迟(ms) | - | - | - |
| 模型大小 | - | - | - |

---

## 最终模型对比

| 模型版本 | mAP50 | mAP50-95 | FPS | 参数量 |
|----------|-------|----------|-----|--------|
| YOLOv8n (基线) | - | - | - | 3.0M |
| + BiFPN | - | - | - | - |
| + Attention | - | - | - | - |
| + TensorRT | - | - | - | - |

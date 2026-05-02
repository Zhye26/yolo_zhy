# 数据集准备指南

## 1. 数据收集

### 图片来源
- 自己拍摄电动车照片/视频
- 网络搜索（百度图片、Google Images）
- 监控视频截图
- 公开数据集

### 建议场景
- 白天/夜间
- 晴天/雨天
- 不同角度（正面、侧面、背面）
- 单人骑行、载人骑行
- 戴头盔、不戴头盔

---

## 2. 工具使用

### 2.1 视频抽帧（如果有视频）

```bash
# 单个视频
python tools/extract_frames.py --video your_video.mp4 --output data/raw_images --interval 30

# 批量处理
python tools/extract_frames.py --video-dir videos/ --output data/raw_images --interval 30
```

### 2.2 预标注

```bash
# 对图片进行预标注
python tools/pre_annotate.py --images data/raw_images --output data/raw_labels
```

预标注会自动检测：
- motorcycle/bicycle → ebike (类别0)
- person → driver (类别1)

你需要手动修正：
- 区分 driver(1) 和 passenger(2)
- 添加 helmet(3) 标注

### 2.3 使用 LabelImg 修正标注

**安装 LabelImg:**
```bash
pip install labelImg
```

**启动:**
```bash
labelImg data/raw_images data/classes.txt data/raw_labels
```

**快捷键:**
- `W` - 创建矩形框
- `D` - 下一张图片
- `A` - 上一张图片
- `Ctrl+S` - 保存
- `Del` - 删除选中的框

**标注规范:**
- 框要紧贴目标边缘
- 被遮挡超过50%的目标不标注
- 每个目标只标注一次

### 2.4 划分数据集

```bash
python tools/split_dataset.py \
    --images data/raw_images \
    --labels data/raw_labels \
    --output data \
    --train 0.7 \
    --val 0.2
```

这会生成:
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

## 3. 类别说明

| ID | 类别名 | 说明 |
|----|--------|------|
| 0 | ebike | 电动车/电瓶车整体 |
| 1 | driver | 驾驶员（骑车的人） |
| 2 | passenger | 乘客（被载的人） |
| 3 | helmet | 头盔 |

---

## 4. 标注示例

### YOLO格式
每张图片对应一个 `.txt` 文件，每行一个目标：
```
<class_id> <x_center> <y_center> <width> <height>
```

所有坐标都是归一化的（0-1之间）。

**示例 (image001.txt):**
```
0 0.5 0.6 0.3 0.4
1 0.45 0.55 0.1 0.2
3 0.45 0.48 0.05 0.05
```

---

## 5. 数据量建议

| 阶段 | 数量 | 说明 |
|------|------|------|
| 初始 | 100张 | 快速验证模型可行性 |
| 中期 | 300张 | 基本可用 |
| 最终 | 500+张 | 达到较好效果 |

每个类别至少要有50个样本。

---

## 6. 完整流程

```bash
# 1. 收集图片到 data/raw_images/

# 2. 预标注
python tools/pre_annotate.py --images data/raw_images --output data/raw_labels

# 3. 用 LabelImg 修正
labelImg data/raw_images data/classes.txt data/raw_labels

# 4. 划分数据集
python tools/split_dataset.py --images data/raw_images --labels data/raw_labels --output data

# 5. 开始训练
python train.py train --data data/dataset.yaml --epochs 100
```

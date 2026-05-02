# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## 🎯 项目目标

这是一个**毕业设计项目**，主题：**基于YOLO的电动车载人违规检测算法**

### 核心需求
1. 检测电动车上的人员（驾驶员、乘客）
2. 检测头盔佩戴情况
3. 判断违规行为：载人违规、未戴头盔
4. 实时视频流处理，FPS > 20
5. Web界面展示检测结果和统计

### 检测类别
- 0: ebike (电动车)
- 1: driver (驾驶员)
- 2: passenger (乘客)
- 3: helmet (头盔)

---

## 🔬 技术路线（重要！）

按以下顺序逐步改进，**每个改进都要做对比实验**：

```
1. 基线模型 (YOLOv8n原版) ← 当前进度
   ↓
2. 改进1: BiFPN特征融合 (替换PANet)
   ↓
3. 改进2: 注意力机制 (CBAM/SE/GLSA)
   ↓
4. 改进3: ByteTrack多目标跟踪
   ↓
5. 改进4: TensorRT部署优化
   ↓
6. 数据清洗与增强 (处理噪声标注)
```

### 实验记录
所有对比实验数据记录在：`docs/experiments.md`

---

## 📊 数据集

### 当前数据集
| 数据集 | 位置 | 图片数 | 类别 |
|--------|------|--------|------|
| 头盔检测 | `data/helmet_dataset/` | 1443 | full, half, invalid, not_wearing |
| 电动车+人 | `data/ebike_dataset/` | 3836 | Electric-bicycle, person |
| **合并数据集** | `data/merged_dataset/` | 5279 | ebike, driver, passenger, helmet |

### 数据问题
- 当前数据集存在**噪声标注**（漏标、标错）
- 后续需要数据清洗作为优化点
- passenger类别暂无数据，需要补充

---

## 🏗️ 项目结构

```
yolo_zhy/
├── app/                    # Flask Web应用
│   ├── models/             # 数据库模型
│   ├── routes/             # 路由
│   └── services/           # 检测服务
├── data/
│   ├── helmet_dataset/     # 头盔数据集
│   ├── ebike_dataset/      # 电动车数据集
│   └── merged_dataset/     # 合并后的数据集
├── models/                 # 训练好的模型权重
├── runs/                   # 训练记录
├── tools/                  # 工具脚本
│   ├── pre_annotate.py     # 预标注
│   ├── extract_frames.py   # 视频抽帧
│   ├── split_dataset.py    # 数据集划分
│   └── merge_datasets.py   # 数据集合并
├── docs/
│   ├── experiments.md      # 实验记录
│   └── dataset_guide.md    # 数据集指南
├── templates/              # 前端模板
├── static/                 # 静态文件
├── venv/                   # Python虚拟环境
├── config.py               # 配置
├── train.py                # 训练脚本
├── run.py                  # Web应用入口
├── Dockerfile              # Docker配置
├── docker-compose.yml      # Docker Compose (GPU)
└── docker-compose.cpu.yml  # Docker Compose (CPU)
```

---

## 🚀 常用命令

```bash
# 激活虚拟环境
source venv/bin/activate

# 训练模型
yolo detect train data=data/merged_dataset/data.yaml model=yolov8n.pt epochs=100

# 测试模型
yolo detect predict model=models/best.pt source=data/merged_dataset/test/images

# 验证模型
yolo detect val model=models/best.pt data=data/merged_dataset/data.yaml

# 启动Web应用
python run.py

# Docker运行
docker compose up -d
```

---

## 📝 待办事项

- [x] 项目框架搭建
- [x] Flask Web应用
- [x] 数据集整合
- [ ] 基线模型训练 (进行中)
- [ ] BiFPN改进
- [ ] 注意力机制改进
- [ ] ByteTrack跟踪集成
- [ ] TensorRT优化
- [ ] 数据清洗
- [ ] 完整系统测试

---

## ⚙️ 环境信息

- GPU: NVIDIA GeForce RTX 4060 (8GB)
- Python: 3.12
- PyTorch: 2.9.1
- Ultralytics: 8.4.6
- CUDA: 13.0

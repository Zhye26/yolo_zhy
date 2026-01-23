# 基于YOLO的电动车违规检测系统

基于 YOLOv8 的电动车载人违规和头盔检测系统，支持实时视频流处理。

## 功能特性

- **目标检测**: 检测电动车、驾驶员、乘客、头盔
- **违规识别**: 载人违规、未戴头盔违规
- **多目标跟踪**: ByteTrack 跟踪算法
- **违规去重**: 基于状态机的违规去重
- **TensorRT 加速**: 支持 TensorRT 优化部署
- **Web 界面**: Flask Web 应用展示检测结果

## 检测类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | ebike | 电动车 |
| 1 | driver | 驾驶员 |
| 2 | passenger | 乘客 |
| 3 | helmet | 头盔 |

## 项目结构

```
yolo_zhy/
├── app/                        # Flask Web 应用
│   ├── config/                 # Pydantic 配置管理
│   │   └── settings.py         # 集中配置
│   ├── core/                   # 核心模块
│   │   ├── types.py            # 数据类型定义
│   │   └── pipeline.py         # 单帧处理流水线
│   ├── inference/              # 推理后端
│   │   └── backends/
│   │       ├── base.py         # 抽象后端接口
│   │       ├── ultralytics_backend.py
│   │       └── tensorrt_backend.py
│   ├── tracking/               # 跟踪模块
│   │   └── manager.py          # ByteTrack 状态管理
│   ├── rules/                  # 规则引擎
│   │   ├── base.py             # 规则基类
│   │   ├── engine.py           # 规则引擎
│   │   ├── passenger.py        # 载人违规规则
│   │   └── helmet.py           # 头盔违规规则
│   ├── violations/             # 违规处理
│   │   └── dedup_fsm.py        # 违规去重状态机
│   ├── rendering/              # 渲染模块
│   │   └── renderer.py         # 检测结果渲染
│   ├── services/               # 业务服务
│   │   ├── detector.py         # 检测器服务
│   │   └── video_processor.py  # 视频处理服务
│   ├── routes/                 # Flask 路由
│   └── models/                 # 数据库模型
├── data/                       # 数据集
│   ├── helmet_dataset/
│   ├── ebike_dataset/
│   └── merged_dataset/
├── models/                     # 模型权重
├── runs/                       # 训练记录
├── tools/                      # 工具脚本
├── templates/                  # 前端模板
├── static/                     # 静态文件
├── config.py                   # Flask 配置
├── train.py                    # 训练脚本
└── run.py                      # Web 应用入口
```

## 架构设计

### 处理流水线

```
Frame → Detection → Tracking → Rule Evaluation → Deduplication → Result
         (YOLO)    (ByteTrack)   (Rule Engine)      (FSM)
```

### 违规去重状态机

```
IDLE → CANDIDATE → ACTIVE → COOLDOWN → IDLE
         ↑           ↓
         └───────────┘
```

- **IDLE → CANDIDATE**: 首次检测到潜在违规
- **CANDIDATE → ACTIVE**: 连续 N 帧确认后激活
- **ACTIVE → COOLDOWN**: 违规不再检测到
- **COOLDOWN → ACTIVE**: 冷却期内重新检测到
- **COOLDOWN → IDLE**: 冷却期结束

### 规则引擎

可扩展的规则系统，支持自定义违规检测规则：

```python
from app.rules import ViolationRule, RuleEngine

class CustomRule(ViolationRule):
    def evaluate(self, context: FrameContext) -> List[ViolationCandidate]:
        # 自定义违规检测逻辑
        pass

engine = RuleEngine()
engine.add_rule(CustomRule(rule_id="custom", enabled=True))
```

## 快速开始

### 环境要求

- Python 3.12+
- CUDA 13.0+ (GPU 加速)
- NVIDIA GPU (推荐 RTX 4060 或更高)

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd yolo_zhy

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# 使用合并数据集训练
yolo detect train data=data/merged_dataset/data.yaml model=yolov8n.pt epochs=100
```

### 启动 Web 应用

```bash
python run.py
```

### Docker 部署

```bash
# GPU 版本
docker compose up -d

# CPU 版本
docker compose -f docker-compose.cpu.yml up -d
```

## 配置说明

配置通过 `app/config/settings.py` 集中管理，支持环境变量覆盖：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_PATH` | `models/best.pt` | 模型路径 |
| `conf_thresh` | 0.5 | 置信度阈值 |
| `iou_thresh` | 0.45 | IoU 阈值 |
| `min_frames_to_confirm` | 3 | 违规确认帧数 |
| `cooldown_frames` | 30 | 冷却帧数 |

## 技术栈

- **检测**: YOLOv8 (Ultralytics)
- **跟踪**: ByteTrack (boxmot)
- **加速**: TensorRT
- **后端**: Flask
- **配置**: Pydantic
- **数据库**: MySQL + SQLAlchemy

## License

MIT License

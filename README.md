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
│   ├── core/                   # 核心模块 (类型定义、流水线)
│   ├── inference/              # 推理后端 (Ultralytics/TensorRT)
│   ├── tracking/               # ByteTrack 跟踪状态管理
│   ├── rules/                  # 可扩展规则引擎
│   ├── violations/             # 违规去重状态机
│   ├── rendering/              # 检测结果渲染
│   ├── services/               # 业务服务
│   ├── routes/                 # Flask 路由
│   └── models/                 # 数据库模型
├── data/                       # 数据集
├── models/                     # 模型权重
├── runs/                       # 训练记录
├── tools/                      # 工具脚本
├── templates/                  # 前端模板
├── static/                     # 静态文件
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
```

- **IDLE → CANDIDATE**: 首次检测到潜在违规
- **CANDIDATE → ACTIVE**: 连续 N 帧确认后激活
- **ACTIVE → COOLDOWN**: 违规不再检测到
- **COOLDOWN → IDLE**: 冷却期结束

## 部署方式

提供两种部署方式，选择其一即可：

### 方式一：Docker 部署（推荐）

无需手动配置环境，一键启动：

```bash
# GPU 版本（开发）
docker compose up -d

# GPU 版本（生产，Gunicorn + 严格启动检查）
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# CPU 版本
docker compose -f docker-compose.cpu.yml up -d
```

### 方式二：本地部署

需要手动配置 Python 环境：

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 启动应用（开发）
python run.py

# 启动应用（生产）
gunicorn -c gunicorn.conf.py run:app
```

## 工程落地检查

```bash
# 1) 健康检查
curl http://127.0.0.1:5000/api/healthz

# 2) 最小冒烟测试
python tools/smoke_test.py

# 3) 强制要求所有检查通过（CI/发版前）
python tools/smoke_test.py --require-healthy

# 4) 使用当前数据库配置执行冒烟（联调环境）
python tools/smoke_test.py --use-current-db --require-healthy

# 5) 发版前一键检查（含编译检查+smoke）
python tools/release_check.py
```

## 训练模型

```bash
# 激活虚拟环境
source venv/bin/activate

# 使用合并数据集训练
yolo detect train data=data/merged_dataset/data.yaml model=yolov8n.pt epochs=100
```

## 配置说明

配置通过 `app/config/settings.py` 集中管理：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_PATH` | `models/bifpn_best.pt` | 模型路径 |
| `conf_thresh` | 0.5 | 置信度阈值 |
| `iou_thresh` | 0.45 | IoU 阈值 |
| `STARTUP_STRICT` | false | 启动自检失败时是否中止进程 |
| `STARTUP_CHECK_DATABASE` | true | 启动时检查数据库连通性 |
| `STARTUP_CHECK_MODEL` | true | 启动时检查模型文件是否存在 |
| `min_frames_to_confirm` | 3 | 违规确认帧数 |
| `cooldown_frames` | 30 | 冷却帧数 |

## 环境要求

- Python 3.12+
- CUDA 13.0+ (GPU 加速，可选)
- NVIDIA GPU (推荐 RTX 4060 或更高)

## 技术栈

- **检测**: YOLOv8 (Ultralytics)
- **跟踪**: ByteTrack (boxmot)
- **加速**: TensorRT
- **后端**: Flask
- **配置**: Pydantic
- **数据库**: MySQL + SQLAlchemy

## License

MIT License

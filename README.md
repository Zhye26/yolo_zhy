# 基于 YOLO 的电动车违规检测系统

基于 YOLOv8 的电动车载人违规和头盔检测系统，支持实时视频流处理。

当前分支（`feature/merge-ljt-csv`）以 **`feature/yolo-ljt`** 的 `YOLO_TRACK_SIMPLE` 子系统为核心，提供 MOT 跟踪/对比 GUI（`run_gui_cross_ljt.py`）。

> 本 README 的"部署与启动"部分只覆盖当前分支**经过实测可用**的入口（YOLO_TRACK_SIMPLE GUI）。旧版的 Docker / Flask 本地部署等流程未在本分支验证，已不再列出。

## 功能特性

- **目标检测**: 检测电动车、驾驶员、乘客、头盔
- **违规识别**: 载人违规、未戴头盔违规
- **多目标跟踪**: ByteTrack 及多种 BoxMOT 跟踪器
- **违规去重**: 基于状态机的违规去重
- **跟踪器对比 GUI**: 同一段视频跑多种 MOT，输出帧级 / 汇总 CSV 与可标注视频

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
├── app/                          # Flask Web 应用（保留，未在本分支验证）
│   ├── config/                   # Pydantic 配置管理
│   ├── core/                     # 核心模块
│   ├── rules/                    # 规则引擎
│   ├── routes/                   # Flask 路由
│   └── ...
├── YOLO_TRACK_SIMPLE/            # MOT 跟踪/对比子系统（本分支重点）
│   ├── core/pipeline/            # 共享的 YOLO/视频/CSV 流水线
│   ├── apps/gui/                 # Tkinter GUI
│   ├── methods/                  # 跟踪 / 关联方法
│   ├── scripts/                  # 推荐入口脚本
│   └── weights/                  # wjh.pt、yolov8n.pt
├── tools/                        # 数据集 / 训练 / 部署相关脚本
├── data/                         # 数据集
├── models/                       # 模型权重
├── templates/  static/           # Flask 前端
└── run.py                        # Flask Web 应用入口
```

## 架构设计

### 主流程

```
Frame → Detection → Tracking → Rule Evaluation → Deduplication → Result
        (YOLO)     (ByteTrack)  (Rule Engine)       (FSM)
```

### 违规去重状态机

```
IDLE → CANDIDATE → ACTIVE → COOLDOWN → IDLE
```

- **IDLE → CANDIDATE**: 首次检测到潜在违规
- **CANDIDATE → ACTIVE**: 连续 N 帧确认后激活
- **ACTIVE → COOLDOWN**: 违规不再检测到
- **COOLDOWN → IDLE**: 冷却期结束

---

# 部署与启动

> 当前分支只验证了以下入口；旧的 Docker / Flask 本地部署方式不再保证可用，需要时请到对应分支查看。

## YOLO_TRACK_SIMPLE GUI（`run_gui_cross_ljt.py`）

跨模型对比 GUI：用 `wjh.pt` 和 `yolov8n.pt` 同时跑一段视频，输出帧级 / 汇总 CSV 与可标注视频。

### 1. 环境准备

```bash
# 推荐使用 conda 环境（Python 3.12 + torch 2.9.1 + ultralytics 8.4.7）
conda activate sam3   # 或你本机对应的 conda 环境名
```

依赖（已在 sam3 环境中安装）：

- `tkinter`、`opencv-python`、`Pillow`（含 `ImageTk`）、`ultralytics`、`numpy`、`torch`

如未安装可执行：

```bash
pip install ultralytics opencv-python pillow numpy
```

权重文件已随分支携带，位于 `YOLO_TRACK_SIMPLE/weights/`：

```text
wjh.pt
yolov8n.pt
```

### 2. 启动 GUI

```bash
cd YOLO_TRACK_SIMPLE
python scripts/run_gui_cross_ljt.py
```

操作步骤：

1. 点击 `Add Video` 选择视频
2. （可选）在预览图上画 ROI
3. 点击 `Start` 开始处理

### 3. 显示器要求

GUI 基于 Tkinter，需要 X 显示：

- 本地 GUI 桌面：默认 `DISPLAY=:1` 即可
- 远程 SSH：用 `ssh -X` 或 VNC

### 4. 输出

每段视频会在源目录下生成：

- `{video_stem}-yolov8n-overload-{tracker}-ljt_frames.csv`
- `{video_stem}-yolov8n-overload-{tracker}-ljt_summary.csv`
- `{video_stem}-yolov8n-overload-{tracker}-ljt.mp4`（可选）

关键 CSV 字段：

| 字段 | 含义 |
|------|------|
| `vehicle_track_id` | 车辆轨迹 / 稳定 ID（如 `M001`） |
| `matched_person_ids` | 绑定到该车的稳定 person ID（如 `P003 P007`） |
| `matched_person_count` | 匹配到的人数 |
| `match_scores` | 人车匹配分数 |
| `raw_overload` | 当前匹配人数 ≥ 2 |
| `confirmed_overload` | 连续 `confirm_frames` 帧均超载 |

---

## 环境要求

- Python 3.12+
- CUDA 13.0+ (GPU 加速，可选)
- NVIDIA GPU (推荐 RTX 4060 或更高)

## 技术栈

- **检测**: YOLOv8 (Ultralytics)
- **跟踪**: ByteTrack / OC-SORT / Deep OC-SORT / BoT-SORT / StrongSORT / HybridSORT (BoxMOT)
- **GUI**: Tkinter + PIL
- **后端**: Flask + SQLAlchemy + Pydantic（未在本分支验证）

## 分支说明

| 分支 | 说明 |
|------|------|
| `main` | 主分支 |
| `feature/yolo-ljt` | YOLO_TRACK_SIMPLE 跟踪/对比/评估子系统 |
| `feature/CSV-correction-proofreading-ypz` | 人车关系标注校正与分类器闭环 |
| `feature/yolo-sam3-cascade` | YOLO + SAM3 级联实验分支 |
| **`feature/merge-ljt-csv`** | **本分支：合并 yolo-ljt 与 CSV 校正两条特性线** |

## License

MIT License

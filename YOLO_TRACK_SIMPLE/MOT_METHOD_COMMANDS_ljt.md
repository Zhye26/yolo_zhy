# MOT 方法与运行命令 - ljt

所有命令默认在项目根目录运行：

```bash
cd /Users/lujintao/Library/CloudStorage/OneDrive-AUTUniversity/MyProject-OneDrive/yolo_zhy-feature-yolo-sam3-cascade/YOLO_TRACK_SIMPLE
conda activate xunienv
```

GUI 统一入口：

```bash
python scripts/run_gui_ljt.py
```

在 GUI 里通过 `tracker` 下拉框选择方法。

## 当前整体流程

本项目主要分两段：

1. YOLOv8n 检测：只检测 COCO 类别 `person` 和 `motorcycle`。
2. MOT/关联：不同 tracker 负责给 `motorcycle` 维持 track id；随后用同一套 person-vehicle 几何匹配逻辑统计每辆车绑定的人数。

超载判断统一为：同一辆车当前匹配人数 `>= 2` 是 `raw_overload`；连续达到 `confirm_frames` 后变成 `confirmed_overload`。

## 方法逻辑说明

| 方法 | tracker 值 | 核心逻辑 | ReID 权重 | 当前状态 |
|---|---|---|---:|---|
| Association | `association` | 参照 `yolov8n_overload_gui_ljt.py` / `yolov8n_overload_video_ljt.py`：person 和 motorcycle 分别用 ByteTrack 跟踪；raw id 映射成稳定业务 id，如 `P001`、`M001`；再用脚点位置、下半身与车辆区域重叠、水平距离、运动方向一致性做 person-vehicle 绑定；绑定达到 `association_min_hits` 后锁定，短时丢失由 `association_unbind_frames` 控制解绑。现在不包含 stationary hold / held_lost / recover 逻辑。 | 否 | 可运行 |
| IoU | `iou` | 项目内置最小基线：当前 motorcycle 检测框和历史 track 框按 IoU 贪心匹配；未匹配则新建 id；超过 `max_missed` 删除。没有 Kalman、没有 ReID、没有低分框二次关联。 | 否 | 可运行 |
| ByteTrack | `byte` | BoxMOT 的 ByteTrack：使用高分检测先关联，低分检测再参与二次关联，适合缓解短时遮挡或检测分数下降时的 id 丢失。本项目里它只跟踪 motorcycle，人与车匹配仍使用统一几何逻辑。 | 否 | 可运行 |
| OC-SORT | `ocsort` | BoxMOT 的 OC-SORT：以运动模型和 IoU 为主，但不完全依赖连续 Kalman 预测；目标重新出现后利用观测点修正轨迹，通常比基础 SORT 更抗遮挡期间的预测漂移。本项目里只作为 motorcycle tracker。 | 否 | 可运行 |
| Deep OC-SORT | `deepocsort` | 在 OC-SORT 的运动/IoU 关联基础上加入自适应 ReID 外观特征；运动不可靠或目标重叠时，外观特征帮助减少 id switch。 | 是 | 可运行 |
| BoT-SORT | `botsort` | 结合 ByteTrack 风格的高低分检测关联、Kalman 运动状态、相机运动补偿 CMC 和 ReID 外观特征；对镜头运动、外观相似、框预测偏差场景更稳。 | 是 | 可运行 |
| StrongSORT | `strongsort` | DeepSORT 系增强版本：使用 Kalman + 外观 ReID 距离 + 运动补偿，重点是保持长轨迹一致性；适合轨迹中断后用外观重新接回。 | 是 | 可运行 |
| HybridSORT | `hybridsort` | 结合 OC-SORT/ByteTrack 思路与 ReID 外观特征，使用 IoU、运动方向、检测置信度和长时 ReID 信息做混合关联。 | 是 | 可运行 |
| ORT | `ort` | 图中提到的 ORT 方法当前不在已安装 BoxMOT 中；项目里只有占位模块，等拿到实现后可接入 `methods/mot_trackers_ljt/ort_ljt.py`。 | 未知 | 暂不可运行 |
| OA-SORT | `oasort` | 图中提到的 OA-SORT 当前不在已安装 BoxMOT 中；项目里只有占位模块，等拿到实现后可接入 `methods/mot_trackers_ljt/oasort_ljt.py`。 | 未知 | 暂不可运行 |

## CLI 命令

把 `/path/to/input.mp4` 替换成你的视频路径。

### Association

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_association.mp4 \
  --tracker association
```

### IoU

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_iou.mp4 \
  --tracker iou
```

### ByteTrack

```bash
python scripts/run_video_byte_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_byte.mp4
```

等价统一入口：

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_byte.mp4 \
  --tracker byte
```

### OC-SORT

```bash
python scripts/run_video_ocsort_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_ocsort.mp4
```

等价统一入口：

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_ocsort.mp4 \
  --tracker ocsort
```

### Deep OC-SORT

```bash
python scripts/run_video_deepocsort_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_deepocsort.mp4 \
  --reid-weights weights/osnet_x0_25_msmt17.pt
```

等价统一入口：

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_deepocsort.mp4 \
  --tracker deepocsort \
  --reid-weights weights/osnet_x0_25_msmt17.pt
```

### BoT-SORT

```bash
python scripts/run_video_botsort_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_botsort.mp4 \
  --reid-weights weights/osnet_x0_25_msmt17.pt
```

等价统一入口：

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_botsort.mp4 \
  --tracker botsort \
  --reid-weights weights/osnet_x0_25_msmt17.pt
```

### StrongSORT

```bash
python scripts/run_video_strongsort_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_strongsort.mp4 \
  --reid-weights weights/osnet_x0_25_msmt17.pt
```

等价统一入口：

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_strongsort.mp4 \
  --tracker strongsort \
  --reid-weights weights/osnet_x0_25_msmt17.pt
```

### HybridSORT

```bash
python scripts/run_video_hybridsort_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_hybridsort.mp4 \
  --reid-weights weights/osnet_x0_25_msmt17.pt
```

等价统一入口：

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_hybridsort.mp4 \
  --tracker hybridsort \
  --reid-weights weights/osnet_x0_25_msmt17.pt
```

### ORT / OA-SORT

目前是占位，命令会提示当前环境未提供实现：

```bash
python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_ort.mp4 \
  --tracker ort

python scripts/run_video_association_ljt.py \
  --video /path/to/input.mp4 \
  --out outputs/output_oasort.mp4 \
  --tracker oasort
```

## ReID 权重

当前已配置：

```text
weights/osnet_x0_25_msmt17.pt
```

需要 ReID 的方法：Deep OC-SORT、BoT-SORT、StrongSORT、HybridSORT。

不需要 ReID 的方法：Association、IoU、ByteTrack、OC-SORT。

如果需要重新下载：

```bash
mkdir -p weights
python -m gdown --id 1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF -O weights/osnet_x0_25_msmt17.pt
```

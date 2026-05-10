# 稳定工作区域 Association 阶段报告

## 本轮范围

本轮只跑 `association`，用于验证更保守的 overload 输出和 per-track 生命周期统计导出。

输入数据：

```text
standardData/3
standardData/4
standardData/9
standardData/13
```

输出文件：

```text
standardData/benchmark_region_results_ljt.csv
standardData/benchmark_region_summary_ljt.csv
standardData/benchmark_region_track_stats_ljt.csv
```

`standardData/12` 当前未参与测试。

## 本轮代码变化

### 1. overload 改为三态业务输出

当前每个 track 的业务状态分为：

| 状态 | 含义 |
|---|---|
| `NORMAL` | 暂无超载证据 |
| `SUSPECTED` | 出现过 `rider_count >= 2`，但证据还不足以作为最终超载 |
| `CONFIRMED` | 达到连续性、比例和置信度条件，可作为最终超载输出 |
| `UNCERTAIN` | `U` 开头的不确定 track，不输出超载最终结果 |

这个设计符合当前业务优先级：宁可漏掉一部分真实超载，也要尽量降低“实际没超载但被判为超载”的误报。

### 2. 修复 per-track 生命周期统计丢失

`association` 现在维护 `archived_vehicle_states`。当 track 过期时，不再直接丢掉累计统计，而是归档并在导出时和活跃状态合并。

修复前可能出现：

```text
raw_overload_frames = 0
confirmed_overload_frames > 0
final_confirmed_overload = 1
```

这不是说明“完全没有超载证据却被误判”，而是说明 track 早期生命周期的累计统计在 tracker 内存里丢了，但 frame-level CSV 仍保留了当时的 confirmed 帧。修复后本轮检查结果：

```text
bad_raw0_confirmed_gt0 = 0
```

### 3. benchmark 增加 suspected 指标

明细结果现在同时包含：

```text
overload_precision / overload_recall / overload_f1
suspected_overload_precision / suspected_overload_recall / suspected_overload_f1
```

其中 `overload_*` 只看 `CONFIRMED`，更接近最终业务输出；`suspected_*` 用来观察候选池覆盖率。

## 总体结果

| Tracker | 视频数 | GT 车辆 | 预测轨迹 | Count Ratio | Overcount | GT 超载车 | 预测超载轨迹 | Overload Count Ratio | ID Switches | Avg Fragments | Mean Overload F1 | Mean Suspected F1 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `association` | 4 | 45 | 50 | 1.111 | 5 | 12 | 8 | 0.667 | 112 | 2.333 | 0.707 | 0.507 | 35.34 |

## 分视频结果

| 视频 | GT 超载车 | 预测超载轨迹 | Overload Precision | Overload Recall | Overload F1 | Suspected Precision | Suspected Recall | Count Ratio | ID Switches | Avg Fragments | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `13` | 1 | 1 | 1.000 | 0.940 | 0.969 | 0.980 | 0.960 | 1.200 | 5 | 1.800 | 34.06 |
| `3` | 4 | 2 | 0.714 | 0.556 | 0.625 | 0.054 | 0.926 | 1.000 | 36 | 2.176 | 36.53 |
| `4` | 5 | 3 | 0.486 | 0.817 | 0.609 | 0.350 | 0.939 | 0.929 | 31 | 2.357 | 35.23 |
| `9` | 2 | 2 | 0.556 | 0.714 | 0.625 | 0.288 | 1.000 | 1.556 | 40 | 3.000 | 35.53 |

## 已确认超载轨迹

| 视频 | Track | GT 超载 | Observed | Raw Frames | Confirmed Frames | Raw Ratio | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `13` | `M004` | 1 | 65 | 46 | 47 | 0.708 | `CONFIRMED` |
| `3` | `M004` | 1 | 19 | 5 | 11 | 0.263 | `CONFIRMED` |
| `3` | `M016` | 1 | 23 | 19 | 10 | 0.826 | `CONFIRMED` |
| `4` | `M001` | 1 | 146 | 55 | 133 | 0.377 | `CONFIRMED` |
| `4` | `M003` | 1 | 11 | 5 | 4 | 0.455 | `CONFIRMED` |
| `4` | `M015` | 1 | 60 | 9 | 1 | 0.150 | `CONFIRMED` |
| `9` | `M003` | 1 | 45 | 11 | 15 | 0.244 | `CONFIRMED` |
| `9` | `M009` | 1 | 19 | 9 | 12 | 0.474 | `CONFIRMED` |

本轮 `final_confirmed_overload=1` 的轨迹全部匹配到 GT 超载车辆；从车辆级 track stats 看，当前保守逻辑没有产生明显的“非超载车辆被最终判为超载”的 track-level 假阳性。

## 主要观察

1. `association` 的车辆数量统计仍有一定 count inflation：45 台 GT 车辆对应 50 条预测轨迹，`track_count_ratio=1.111`。其中 `video 9` 最明显，`track_count_ratio=1.556`，这说明它仍然有轨迹碎裂或重复计数问题。

2. `CONFIRMED` 输出明显更保守：GT 超载车 12 台，预测超载轨迹 8 条，`overload_count_ratio=0.667`。这符合“先压误报”的方向，但会牺牲召回。

3. `SUSPECTED` 召回较高，但 precision 很低，尤其 `video 3` 的 suspected precision 只有 0.054。这说明“只要出现过 2 人”可以覆盖大多数真实超载，但候选池非常脏，不能直接作为最终结果。

4. per-track 导出已经能用于调阈值。本轮没有再出现 `raw_overload_frames=0` 但 `confirmed_overload_frames>0` 的异常组合。

## overload 下一步优化位置

建议继续在 `association` 上优化，不要先扩到其他 tracker。原因：

- `association` 速度稳定，约 35 FPS，适合作为调参主线；
- 它已经具备 person 与 motorcycle 同步维护能力，能自然承载车辆级证据累计；
- 当前 confirmed track 没有明显车辆级假阳性，适合在这个基础上逐步提高召回；
- `SUSPECTED` 数据已经暴露了候选池噪声，下一步可以围绕这些统计设计更细规则。

优先优化位置：

```text
pipeline_ljt.py::update_simple_overload_stats(...)
pipeline_ljt.py::overload_status_from_stats(...)
methods/association_ljt.py::_update_vehicle_overload_states(...)
```

建议下一轮规则方向：

- `CONFIRMED` 继续保持保守，不让 `SUSPECTED` 直接输出为最终超载；
- 对 `raw_overload_frames >= 3` 且 `raw_ratio >= 0.25` 的轨迹保留为主确认通道；
- 对高置信车辆、高置信人、移动状态下的 overload 帧设置独立确认通道；
- 对 observed 很短的轨迹使用更严格规则，避免短暂遮挡或检测抖动造成误报；
- 对 `U` 开头 track 固定输出 `UNCERTAIN`，不参与最终超载车辆计数。

当前最值得人工抽查的是：

```text
standardData/benchmark_region_track_stats_ljt.csv
```

重点看 `final_overload_status=SUSPECTED` 且 `gt_overload=0` 的轨迹。这些就是候选池噪声来源，下一步规则应该先解释和过滤它们。

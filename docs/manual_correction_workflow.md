# 人工修正与关系分类器闭环

本流程用于把 MVP 中的超载误检问题拆成更可控的 `person -> vehicle` 关联问题。

## 1. 导出候选匹配样本

```bash
python tools/export_match_samples.py static/uploads/demo.mp4 \
  --model models/bifpn_best.pt \
  --output data/corrections/demo_match_samples.csv \
  --task passenger \
  --sample-every 5
```

CSV 中每一行是一组候选：

```text
某一帧的一个 person 框 + 一个 vehicle 框
```

人工主要填写：

| 字段 | 含义 |
|------|------|
| `match_label` | person 属于该 vehicle 填 `1`，否则填 `0` |
| `correct_vehicle_id` | 可选；用于记录正确车辆 ID |
| `error_type` | 可选；如 `wrong_person_match`、`missed_person`、`id_switch` |
| `notes` | 可选备注 |

## 2. 训练轻量关系分类器

如果不想直接编辑 CSV，可以打开标注界面逐条判断：

```bash
python tools/annotate_match_samples.py static/uploads/demo.mp4 \
  data/corrections/demo_match_samples.csv
```

快捷键：

| 快捷键 | 含义 |
|--------|------|
| `1` | 当前 person 属于当前 vehicle |
| `0` | 当前 person 不属于当前 vehicle |
| `s` | 跳过 |
| `← / →` | 上一条 / 下一条 |
| `Ctrl+S` | 保存 |

```bash
python tools/train_match_classifier.py data/corrections/demo_match_samples.csv \
  --output models/person_vehicle_match.joblib \
  --model-type random_forest \
  --threshold 0.55
```

脚本会跳过 `match_label` 为空的行，只使用人工标注过的样本。

## 3. 启用分类器

在运行应用前设置环境变量：

```bash
export MATCH_CLASSIFIER_ENABLED=true
export MATCH_CLASSIFIER_PATH=models/person_vehicle_match.joblib
export MATCH_CLASSIFIER_THRESHOLD=0.55
```

启用后，`PassengerRule` 会先生成候选 person-vehicle pair，再用分类器判断是否真的属于同一辆车。未启用或模型不存在时，系统保持原来的几何规则行为。

## 4. 建议标注策略

优先标注这些场景：

```text
1. 旁边行人被算到车上
2. 两辆车靠得很近
3. 边缘车辆框不完整
4. 双人同乘被合成一个大 person 框
5. ID 切换或重复车框导致的误判
```

如果统计发现主要错误是 `id_switch`，应优先优化 tracker；如果主要错误是 `wrong_person_match`，应继续扩充本 CSV 并重训关系分类器。

# 项目迭代记录

更新时间：2026-03-15  
项目目录：[yolo_zhy](/C:/Users/Zhye/Desktop/yolo/yolo_zhy)  
当前分支：`feature/yolo-sam3-cascade`

## 一、项目总体目标

本项目面向电动车违规识别场景，目标是构建一套基于 YOLO 的视频检测系统，实现对电动车骑乘行为的自动分析与可视化展示。系统初始设计目标包括以下几个方面：

1. 实现电动车、骑乘人员和头盔等关键目标的检测。
2. 实现载人/超载违规与未佩戴头盔违规的自动识别。
3. 实现视频级别的目标跟踪、违规去重和结果渲染。
4. 提供可运行的 Web 图形界面，支持视频上传、处理和结果展示。

在后续实际开发过程中，项目优化重点逐步由“通用检测能力”转向“面向固定业务场景的视频效果优化”，即以用户实际观看到的视频检测效果作为主要评价标准。

## 二、第一阶段：基础系统搭建

### 2.1 主要工作

项目初期完成了整体系统框架的搭建，建立了从视频输入到结果输出的基础处理链路。该阶段主要完成内容如下：

1. 搭建 Flask Web 应用结构。
2. 建立检测、跟踪、规则判断、违规去重和渲染的单帧处理流水线。
3. 实现视频文件读取、逐帧处理和结果视频输出。
4. 建立数据库与 Web 页面基础结构。
5. 形成完整的初始工程目录和模块划分。

### 2.2 关键文件

- [app/core/pipeline.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/core/pipeline.py)
- [app/services/video_processor.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/services/video_processor.py)
- [app/violations/dedup_fsm.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/violations/dedup_fsm.py)
- [app/rendering/renderer.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/rendering/renderer.py)
- [run.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/run.py)

### 2.3 阶段成果

该阶段结束后，项目已经具备完整的基础处理能力，可以对输入视频进行检测、简单违规识别和结果可视化输出，为后续针对业务问题进行专项优化奠定了基础。

## 三、第二阶段：检测能力扩展

### 3.1 主要工作

为提升复杂场景下的检测效果，项目逐步引入并完善了多类别检测体系，形成了 `ebike / driver / passenger / helmet` 四类目标的检测框架。同时，围绕小目标、低置信度目标和复杂遮挡场景，进一步增加了后处理增强逻辑，包括：

1. 候选区域 refine。
2. 时间维度稳定策略。
3. 同类框合并与上下文筛选。
4. 预留 YOLO + SAM3 级联处理能力。

### 3.2 关键文件

- [app/inference/backends/ultralytics_backend.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/inference/backends/ultralytics_backend.py)
- [app/services/yolo_sam3_detector.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/services/yolo_sam3_detector.py)

### 3.3 阶段成果

该阶段使系统具备了更完整的目标检测能力，并为后续头盔检测、载人检测和复杂场景优化提供了统一的检测基础。

## 四、第三阶段：跟踪模块落地

### 4.1 问题背景

原始方案设计中计划使用 ByteTrack 作为目标跟踪方案，但在实际运行环境中，`boxmot / BYTETracker` 无法正常使用，导致系统在本地条件下难以稳定完成视频跟踪任务。

### 4.2 优化措施

为解决运行依赖受限的问题，在 [app/tracking/manager.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/tracking/manager.py) 中实现了 fallback 跟踪策略，主要思路包括：

1. 基于 IoU 进行候选关联。
2. 引入中心点距离与尺度相似度辅助匹配。
3. 对跟踪框应用 EMA 平滑，减少抖动。
4. 仅保留当前帧有效目标，避免旧框拖尾和“鬼框”现象。

### 4.3 阶段成果

该阶段的改动确保了在无 ByteTrack 条件下系统仍可稳定工作，使视频连续处理能力不再依赖单一第三方跟踪组件。

## 五、第四阶段：超载检测逻辑重构

### 5.1 初始问题

在早期实现中，超载检测依赖 `driver / passenger` 的语义分类结果。然而在实际视频中，经常出现以下问题：

1. 司机和乘客分类不稳定。
2. 双人同乘时被合并为单个大人框。
3. 电动车框存在局部覆盖、不完整覆盖现象。
4. 明显超载车辆未被识别。
5. 正常车辆被误报为超载。

### 5.2 重构思路

为提高实际业务可用性，项目将超载检测逻辑由“基于角色分类判定”重构为“基于车与人的空间关联及人数判定”。新的规则思路如下：

1. 先稳定识别电动车目标。
2. 再将骑乘人员与对应电动车进行空间关联。
3. 当同一辆车关联到两个及以上骑乘人时，判定为超载。
4. 当仅检测到一个人框，但该人框与车辆框的几何关系明显符合“双人合框”特征时，使用 merged heuristic 补充判定。

### 5.3 关键文件

- [app/rules/passenger.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/rules/passenger.py)

### 5.4 主要改进内容

1. 对电动车检测框进行 canonicalize，减少重复车框。
2. 对骑乘人员进行去重，避免同一人被重复计数。
3. 将边缘车与非边缘车分开处理。
4. 引入轨迹历史信息：
   - `positive_streak`
   - `temporal_hold`
   - `weak_support_hold`
5. 将违规框统一渲染为电动车框，提升画面表达清晰度。

### 5.5 阶段成果

该阶段是项目从“能检测”向“能识别业务违规”的关键转折点。经重构后，超载检测开始具备面向实际视频做针对性优化的能力。

## 六、第五阶段：显示 ID 与可视化稳定化

### 6.1 初始问题

在视频回归过程中，用户多次指出以下显示层问题：

1. 同一个显示 ID 会跳到不同车辆上。
2. 编号存在明显跳号现象。
3. 远处小车容易抢占 ID，导致近处重点目标显示不直观。

### 6.2 优化措施

为提高可视化质量，在 [app/rendering/renderer.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/rendering/renderer.py) 中加入显示层 ID 映射机制，核心策略包括：

1. 不直接使用原始 `track_id` 作为最终显示编号。
2. 基于框重叠、中心接近度进行 display ID 桥接。
3. 仅对更近、更大、更稳定的电动车显示绿色 ID。
4. 后续继续放宽中近景大车的最小 `hits` 门槛，使其更早显示编号。

### 6.3 阶段成果

该阶段优化后，视频中的 ID 稳定性得到显著改善，用户在观察检测结果时可以更直观地追踪重点目标。

## 七、第六阶段：围绕固定视频持续回归优化

### 7.1 迭代方式变化

项目中后期的优化方式发生了明显变化。与常规算法开发不同，本项目进入了“围绕固定视频反复回归”的工作阶段，即：

1. 以指定视频作为主要回归素材。
2. 以实际视频观看效果作为主要评价标准。
3. 每轮修改后都重新生成结果视频供用户观察。

当前典型回归视频包括：

- `388eabf8d2c341daf1691b9c396097cf.mp4`
- `344c981565e9c9ffb18f0c30a804b688.mp4`
- `8f3589122c6f828d73cf466c4b258bf7.mp4`

### 7.2 典型问题与对应迭代结果

#### 7.2.1 左侧超载车可识别但不连续

优化内容：

1. 增强边缘车的 temporal hold。
2. 为边缘车增加 proxy support 判定。

结果：

左侧超载车辆在更多关键帧上保持稳定红框，连续性得到明显提升。

#### 7.2.2 中间正常车被误判为超载

优化内容：

1. 对非边缘车限制弱证据 hold 的持续时间。
2. 提高第二骑乘人的有效性筛选标准。

结果：

误报显著下降，正常车辆长时间被打红框的问题得到抑制。

#### 7.2.3 右侧明显双人黄车漏检

用户指出：右侧明显双人黄车最初仅在最后几帧才被识别为电动车并触发超载。

优化内容：

1. 增加侧向双人同乘的 merged heuristic。
2. 提升该类 merged evidence 的确认速度。
3. 在 dedup 层增加短时保活，减少红框闪烁。
4. 放宽中近景大车的显示编号条件。

结果：

该目标由“末尾少量帧出现红框”优化为“更早出现 ID，并在更长帧段内保持超载红框”，视频效果明显改善。

### 7.3 关键文件

- [app/rules/passenger.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/rules/passenger.py)
- [app/violations/dedup_fsm.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/violations/dedup_fsm.py)
- [app/rendering/renderer.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/rendering/renderer.py)

## 八、第七阶段：检测任务拆分

### 8.1 需求变化

随着项目推进，用户提出新的明确需求：头盔检测与载人检测需要分开运行，避免在某次演示或测试中同时展示两类违规。

### 8.2 主要改动

围绕该需求，项目新增了三种任务模式：

1. `helmet`
2. `passenger`
3. `all`

对应实现文件如下：

- [app/services/detector.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/services/detector.py)
- [app/services/yolo_sam3_detector.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/services/yolo_sam3_detector.py)
- [tools/test_tracking.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/tools/test_tracking.py)

### 8.3 阶段成果

该阶段完成后，命令行已经可以分别运行：

```powershell
.\.venv\Scripts\python.exe tools\test_tracking.py "<video>" --task helmet --output "<output.mp4>"
.\.venv\Scripts\python.exe tools\test_tracking.py "<video>" --task passenger --output "<output.mp4>"
.\.venv\Scripts\python.exe tools\test_tracking.py "<video>" --task all --output "<output.mp4>"
```

这意味着头盔检测和超载检测已经在工程层面实现了职责拆分。

## 九、第八阶段：头盔检测链路恢复

### 9.1 初始状态

项目中原本存在头盔检测类别和头盔违规规则，但默认处于关闭状态：

- `helmet_detection_enabled = False`
- `helmet_rule_enabled = False`

### 9.2 当前状态

经过本轮修改，系统已能够以“纯头盔模式”独立运行，并输出不包含超载信息的结果视频。

示例输出文件：

- [344c981565e9c9ffb18f0c30a804b688_helmet_only_v1.mp4](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/static/outputs/ad_hoc_20260315/344c981565e9c9ffb18f0c30a804b688_helmet_only_v1.mp4)

### 9.3 阶段意义

该阶段标志着项目已经从“单一超载优化任务”扩展为“多违规类型可切换运行”的完整系统。

## 十、当前系统状态总结

截至 2026-03-15，项目已具备如下能力：

1. 支持单视频稳定处理。
2. 支持 fallback 跟踪运行。
3. 支持较稳定的电动车显示 ID。
4. 支持超载检测的针对性视频回归优化。
5. 支持头盔检测与超载检测分开运行。
6. 已具备 Flask Web 图形界面基础。

## 十一、当前仍存在的问题

### 11.1 超载检测方面

1. 对遮挡严重、视角极端的车辆仍可能存在漏检。
2. 跟踪切换后显示层 ID 仍可能变化。
3. 不同视频上的泛化稳定性仍需继续验证。

### 11.2 头盔检测方面

1. 当前仅完成“纯头盔模式可运行”的工程打通。
2. 头盔误报与漏报尚未进行与超载同强度的专项优化。
3. 头盔检测的可视化表达仍可继续增强。

### 11.3 GUI 方面

1. Flask Web 页面已经具备基础上传和展示功能。
2. 但“仅头盔 / 仅载人 / 全部”的前端任务模式切换尚未加入页面交互。

## 十二、关键模块分布

### 12.1 检测与后处理

- [app/inference/backends/ultralytics_backend.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/inference/backends/ultralytics_backend.py)

### 12.2 跟踪模块

- [app/tracking/manager.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/tracking/manager.py)

### 12.3 违规规则模块

- [app/rules/passenger.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/rules/passenger.py)
- [app/rules/helmet.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/rules/helmet.py)

### 12.4 违规去重模块

- [app/violations/dedup_fsm.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/violations/dedup_fsm.py)

### 12.5 渲染与显示模块

- [app/rendering/renderer.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/rendering/renderer.py)

### 12.6 业务封装与任务切换

- [app/services/detector.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/services/detector.py)
- [app/services/video_processor.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/app/services/video_processor.py)
- [tools/test_tracking.py](/C:/Users/Zhye/Desktop/yolo/yolo_zhy/tools/test_tracking.py)

## 十三、后续建议

结合当前项目状态，后续建议按以下顺序推进：

1. 在 GUI 中加入任务模式选择，完成前后端统一。
2. 对头盔检测开展专项视频回归优化。
3. 在更多视频上验证超载规则的泛化稳定性。
4. 整理实验截图、结果视频和系统结构图，为毕业设计论文和答辩材料服务。

## 十四、结论

从项目启动到当前阶段，系统已完成从“基础检测工程”向“面向具体业务场景的视频违规识别系统”的演进。尤其是在跟踪替代实现、超载规则重构、显示稳定化、固定视频回归优化和任务模式拆分等方面，形成了较为完整且可运行的工程成果。该迭代过程不仅提升了系统可用性，也为后续毕业设计论文撰写、系统展示与功能扩展提供了较为清晰的技术积累。

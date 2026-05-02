# AI 交接说明（2026-03-10）

## 1. 当前仓库状态
- 仓库：`git@github.com:Zhye26/yolo_zhy.git`
- 当前分支：`feature/yolo-sam3-cascade`
- 当前最新提交（写本文档前）：`4c9066b` / `feat: refine overload detection pipeline`
- 本文档创建目的：给另一台电脑上的 AI 直接接手当前项目状态，避免重复摸索。

## 2. 当前业务目标
当前优先级不是论文、不是数据分析，而是**先把项目检测链路落地到可用**。

当前阶段的具体目标：
1. 固定用一个视频反复回归优化。
2. 暂时不管 SAM 分割效果。
3. 暂时关闭头盔违规正式链路，降低计算量。
4. 主攻“电动车超载/载人”检测。
5. 输出视频里统一用**红框标超载**，不要再细分司机/乘客文案。
6. 未来要走实时监测，所以实现尽量保持轻量、可实时化。

## 3. 固定回归视频
后续优化默认固定使用这支视频：

`/home/ubuntu/文档/xwechat_files/wxid_as7p1dyd2z7b12_16cd/msg/video/2026-03/4a547276a398ba5adcdcd59f848d617d.mp4`

用户明确说：
- 以后主要根据这支视频提意见。
- 让 AI 针对这支视频持续改。

## 4. 当前正式链路的核心策略
### 4.1 规则侧
当前已经从“driver/passenger 分类违规”切换为：
- **按车数人**判断是否超载。
- 不再强调“司机/乘客”区分。
- 规则文件仍然沿用 `PassengerRule` 类名，但逻辑已经是“超载规则”。

触发逻辑：
1. 如果同一辆 `ebike` 关联到 `>= 2` 个 rider 框，则判超载。
2. 如果只检测到 `1` 个 rider 框，但该 rider 框与车框呈现“明显双人合框”的几何特征，也判超载。

超载描述统一为：
- `电动车超载`

### 4.2 可视化侧
当前渲染收敛为：
- 普通检测框只保留 `ebike`。
- 违规统一用红框高亮。
- 用户更关心“哪辆车违规”，不想看一堆 driver/passenger 蓝橙框。

### 4.3 跟踪侧
环境里 `boxmot` 不可用，因此：
- `TrackManager` 已加了轻量 fallback tracker。
- fallback 使用 IoU + 中心距离 + 尺度相似度匹配。
- 使用 EMA 平滑框。
- 现在只输出**当前帧仍命中的 track**，不再把丢帧后的旧框继续画出来，减少闪烁和“鬼框”。

## 5. 关键代码位置
### 配置
- `app/config/settings.py`

当前重点配置：
- `helmet_detection_enabled = False`
- `helmet_rule_enabled = False`
- `enable_tile_refine = False`
- fallback tracking 相关阈值也在这里

### 规则引擎
- `app/rules/engine.py`

当前只保留 passenger 规则（实际语义是 overload 规则）作为正式链路主规则。

### 超载规则
- `app/rules/passenger.py`

这是目前最关键的文件，虽然名字还叫 passenger，但逻辑已经是：
- canonicalize ebike
- rider/ebike 关联
- merged double-rider heuristic
- 最终输出 overload 候选

### 推理后处理
- `app/inference/backends/ultralytics_backend.py`

这是第二关键文件，里面做了：
- 类别过滤
- candidate refine
- contextual prune
- temporal stabilize
- same-class consolidate
- ebike consolidate
- rider support / ebike support 几何规则

### 跟踪器
- `app/tracking/manager.py`

重点：
- `BYTETracker` 不可用时自动 fallback
- 当前 fallback 只输出 `state == tracked` 的目标

### 渲染
- `app/rendering/renderer.py`

重点：
- 只画 `ebike` 检测框
- 违规统一画红框

### 业务封装
- `app/services/detector.py`
- `tools/test_tracking.py`

当前回归视频基本都通过 `tools/test_tracking.py` 来跑。

## 6. 当前回归产物
最近基于固定视频的主要结果文件：

### 旧版本（按载人）
- `static/outputs/day_demo_20260310/4a547276_tracking_passenger_rt.mp4`
- `static/outputs/day_demo_20260310/4a547276_tracking_passenger_rt_v2.mp4`

### 新版本（按车数人 / overload）
- `static/outputs/day_demo_20260310/4a547276_overload_v3.mp4`
- `static/outputs/day_demo_20260310/4a547276_overload_v4.mp4`

当前建议用户重点看的版本：
- `static/outputs/day_demo_20260310/4a547276_overload_v4.mp4`

## 7. 当前已知问题
这是另一台 AI 接手后最应该继续攻的部分。

### 问题 1：有些电瓶车没有被检出来
表现：
- 视频里部分电瓶车根本没有绿框。
- 说明 `ebike` 本体检测/保留规则仍偏弱。

可能原因：
- 主模型对某些角度/遮挡车体置信度低。
- `ebike` 的保留仍然受 support 逻辑影响。
- 候选 refine 还不够贴合当前视频场景。

### 问题 2：用户明确点名 `ID:4` 那辆双人车没有被识别为超载
用户最新直接反馈：
- “有的电瓶车没有被检查出来”
- “然后 id4 的电瓶车是双人超载没有检测出来”

这说明：
- 当前 overload 规则虽然已经能打出一部分超载，但还没有完全覆盖用户认为最关键的那辆车。
- 后续优化应以**用户点名那辆车**为第一优先级，而不是追求整体统计好看。

### 问题 3：当前 `overload` 的 merged-rider heuristic 仍不够稳
目前 heuristic 是靠：
- rider 框和 ebike 框的重叠
- rider 尺寸是否过大
- foot point 是否落入车区域
- rider/ebike 尺度比

这能抓住一部分双人合框，但还不够稳定。

## 8. 推荐下一步优化方向
给下一台 AI 的建议顺序：

### 优先级 A：只围绕固定视频做针对性增强
不要先做泛化优化，先盯住这支视频：
- 把所有明显双人车的时间段列出来。
- 尤其把用户说的 `ID:4` 那辆车的帧段定位出来。
- 对那辆车建立更强的几何/时序规则。

### 优先级 B：从“车中心区域的人数”角度做更直接计数
用户已经给了非常关键的方向：
- “就是数个数，一个车关联了几个人就好。”

因此下一步建议：
1. 基于 `ebike` 框向上/向左右扩展一个 rider ROI。
2. 在这个 ROI 内统计：
   - 主模型 rider 框数量
   - candidate person 框数量
3. 对于 candidate person 框，不必要求一定被主模型分类成 driver/passenger，只要在车的 rider ROI 内且满足纵向/横向关系即可纳入计数。
4. 当计数 `>= 2` 时直接判超载。

换句话说：
- 不必太依赖 `driver/passenger` 语义分类。
- 更应该把它改成 `ebike + person-count` 逻辑。

### 优先级 C：对“一个大人框里包着两个人”做二次拆分
当前 candidate model 经常只给一个大 `person` 框。
下一步可尝试：
- 对 rider ROI 做局部二次检测。
- 或者直接在 rider ROI 中检查前后两个“人体上半身/头肩峰值”模式。
- 如果不想引入新模型，可以先用更轻量的几何启发式：
  - rider 框异常宽
  - rider 框异常高
  - rider 框和 ebike 框高度/宽度比异常
  - rider 框内存在两个高响应子区域（如果要继续做，可以借助通用 person detector 的 crop 再检测）

### 优先级 D：强化 `ebike` 召回
因为用户说“有的电瓶车没检出来”，后续要继续做：
- 对 candidate model 中的 `motorcycle / bicycle` 候选更积极地转成 ebike proxy
- 对近景大目标车体适当降低保留门槛
- 对遮挡车体加强 temporal memory

## 9. 已明确暂时不做的方向
- 暂时不继续做 SAM 分割效果
- 暂时不回到头盔正式链路
- 暂时不做论文/数据分析

## 10. 运行与回归命令
### 固定视频回归
```bash
./venv/bin/python tools/test_tracking.py \
'/home/ubuntu/文档/xwechat_files/wxid_as7p1dyd2z7b12_16cd/msg/video/2026-03/4a547276a398ba5adcdcd59f848d617d.mp4' \
--output static/outputs/day_demo_20260310/4a547276_overload_v4.mp4
```

### 语法检查
```bash
./venv/bin/python -m py_compile \
app/rules/passenger.py \
app/inference/backends/ultralytics_backend.py \
app/tracking/manager.py \
app/rendering/renderer.py \
app/services/detector.py
```

## 11. 哪些东西没有推到 Git
当前本地仍未推送的内容只有：
- `tmp/`：临时调试截图/中间文件
- `.ace-tool/index.json`：本地工具索引变更

这两类都不是项目源码主逻辑。

## 12. 对另一台 AI 的一句话建议
如果你是下一台接手的 AI，请不要再围绕“driver/passenger 分类精度”打转，直接把逻辑进一步改造成：

**以 ebike 为中心，统计 rider ROI 内的人数，只要一辆车上判定出 2 个骑乘人就直接红框标超载。**


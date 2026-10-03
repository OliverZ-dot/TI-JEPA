# Kill Experiment 结果（协议 B，§4.2）+ 探针梯子（协议 A，§4.1）

**状态：第一阶段 InertiaBall 预实验，Go 信号成立（有条件）。** 用的是自建的
InertiaBall 环境，不是官方 LeWM + PushT（原因见 `README.md` "范围"一节）。

## 实验设置

| 项 | 值 |
|---|---|
| 环境 | InertiaBall（自建，`ti_jepa/envs/inertia_ball.py`），64×64 RGB，无 motion blur |
| 物理 | 无摩擦（`damping=0`），撞墙反弹（`restitution=0.9`），frame-skip=1（model step = env step） |
| 数据 | 2400 条训练 episode + 400 条 held-out episode，每条 40 步，随机稀疏冲量；`impulse_scale=0.035`，典型帧间位移 ≈3–4 px（球半径 4px） |
| 数据种子 | `seed=0`（生成），`data/inertia_ball.npz` |
| 模型 | baseline(k=3)：LeWM-style，单帧 target，predictor 看 3 帧历史（对齐官方"N=3"）<br>baseline(k=1)：同上但 predictor 无历史（消融，纯粹操作化"单帧不可识别"）<br>TI-JEPA：pose(单帧,4d)+motion(帧差,4d)，predictor **无记忆**，只吃当前 `(q_t,v_t)+a_t` |
| 训练 | 8000 step，batch=256，Adam lr=1e-3，`λ_reg=1.0`(SIGReg)，`λ_v=0.1`，种子=1（超参调整原因见 `notes/collapse_debug.md`） |
| ckpt | `checkpoints/baseline.pt`（k=3）、`checkpoints/baseline_k1.pt`（k=1）、`checkpoints/tijepa.pt` |
| Kill experiment | 200 对同构型不同速度，`horizon=15`，速度大小 U(0.03, 0.08)，方向随机反向，种子=42 |
| 探针 | 全部线性 Ridge，按 episode 切 train(280)/test(120)，不按窗口随机切 |

## 协议 A：探针梯子（held-out 400 episodes）

| 输入 | baseline r | TI-JEPA r |
|---|---|---|
| 单帧 `z_t` / `q_t` | **-0.025** | **-0.038** |
| `(z_t, z_t+1)` 拼接 | +0.803 | +0.557 |
| `z_{t-k+1:t}` / `q_{t-k+1:t}` 窗口 | +0.852 | +0.585 |
| 显式 `v_t`（仅 TI-JEPA，主数字） | 无此对象 | **+0.571** |

**读法：**

- 单帧读速度：两个模型都约等于 0（负值，纯噪声），**和理论预测完全一致**——
  单帧 target 的 encoder 在信息论上不可能带速度，这是最核心、最干净的验证。
- 一旦给探针"多帧"（pair / window），两个模型都能读出中等偏高的速度信号——
  这是因为线性探针自己在做有限差分，不代表 encoder 本身识别了速度
  （§1 原文点名的混淆项）。baseline 的窗口读数字更高（0.85 > 0.59），是因为
  baseline 的 8 维 `z` 对位置的编码更"干净"（位置探针 R²=0.97 vs TI-JEPA 0.64，
  见下），给线性探针更大的自由度去拟合差分，不代表 baseline 更"懂"速度——
  下面的 rollout 测试才是真正说话的地方。
- TI-JEPA 的显式 `v_t`：r=0.57，是 LeWM 架构里**根本不存在的对象**——LeWM 没有
  任何单一向量能达到这个数字。这是 §4.1 期望表格里的"主数字"，成立。

## 协议 B：Kill Experiment（200 对同构型、不同速度，H=15）

```
o_t(+v) 和 o_t(-v) 逐像素差异： 0.0000   <- 定义性质，图见 kill_experiment.png 左两张
baseline z_t(+v) 和 z_t(-v) 的距离：      0.0000   <- 单帧 encoder，逐帧独立，恒等
TI-JEPA q_t(+v) 和 q_t(-v) 的距离：       0.0000   <- pose 头只看单帧，同样恒等
TI-JEPA v_t(+v) 和 v_t(-v) 的距离：       4.67     <- 结构化 motion 头能分开
TI-JEPA 速度符号判定准确率：              99.5%
```

**盲 rollout（a=0 滑行 15 步，不再喂真实帧）：**

| 模型 | 位置 MSE（越低越好） | branch separation / ground truth（越接近 1 越好） |
|---|---|---|
| baseline k=3（predictor 有 3 帧历史，对齐官方设定） | 0.0073 | 0.894 |
| **baseline k=1（predictor 无历史，纯粹消融）** | **0.0533** | **0.000** |
| **TI-JEPA（predictor 无历史，靠显式 v）** | **0.0228** | **0.662** |

**这是整个实验里最关键的一组数字：**

- **baseline k=1 的 branch separation 是精确的 0.000** —— 两条速度相反的分支，
  盲 rollout 之后被预测成**完全同一条轨迹**，正是 §4.2 描述的"零动作预测塌向
  平均"。这是"单帧 target 不可识别速度"最干净、没有历史混淆项污染的实证。
- 把同一个 encoder 的 predictor 换成能看 3 帧历史（baseline k=3），
  数字立刻从 0.000 冲到 0.894——**证实了 §1 自己点名的混淆项确实存在**：
  单帧 `z` 虽然不可识别速度，但如果 predictor 被允许看历史窗口，它可以在
  自己的计算图里隐式算出有限差分，从而在这一个特定任务里"表现得像"
  识别了速度。这个能力活在 predictor 的计算图里，不活在被监督的 `z` 里
  ——`z` 本身仍然不是 Markov 状态，规划时如果只能操作 `z`（比如 CEM 代价
  `‖z_H - z_g‖²`），仍然拿不到速度这个自由度。
- **TI-JEPA 和 baseline k=1 用的是完全同一条件（predictor 都无记忆、
  都不能看历史）**，但 branch separation 从 0.000 冲到 0.662——唯一的区别
  是 TI-JEPA 把 v 做成了 z 的一部分。这是本实验里对 claim 支持力度最强的
  对比：**不是"给模型历史就能猜出速度"，而是"把速度放进被监督的目标里，
  才能在没有历史的情况下也猜出速度"。**

结论图：`results/kill_experiment.png`（左：两张几乎像素级相同的当前帧；
右：baseline k=1 两条红色虚实线几乎重合在同一条轨迹上，TI-JEPA 的蓝色两条线
明显分叉，baseline k=3 的橙色两条线分得最开但那是"作弊"的对照）。

## Go / No-go 判定（按 §4.2 的标准）

**Go，但要把"predictor 有没有历史"这个变量单独控制才看得出来。**

原始判据("LeWM 单帧分不开、零动作预测塌向平均" vs "TI-JEPA 能分、MSE 明显低")
在**去掉历史混淆项**（baseline k=1 vs TI-JEPA，同样无记忆）之后清晰成立：
0.000 vs 0.662 的 branch separation，0.0533 vs 0.0228 的 MSE。

但如果直接对比"官方式"baseline（k=3，predictor 有历史）和 TI-JEPA，
baseline k=3 的绝对数字（0.894 分离度、0.0073 MSE）反而比 TI-JEPA 好——
这不是 claim 错了，而是提醒我们：**官方 LeWM 的 predictor 本来就允许看
历史窗口，所以"z 不识别速度"这件事不会在"predictor 直接 rollout"这个
任务上体现出明显劣势**；claim 真正咬人的地方是下游只能操作 `z` 本身的场景
——线性探针（协议 A）、以及规划时如果 cost 只能写成 `z` 或 `q` 的函数
（协议 C，还没做）。写论文时要把"predictor 允许看多少历史"当成一个显式
维度而不是隐藏的实现细节，本文档 §4.2 的原始描述里没有点出这一层，是
执行过程中发现的、值得补进说明书的一条经验。

## 已知局限（诚实记录，不要在论文里假装没有）

1. **环境是自建的 InertiaBall，不是 PushT / 官方 LeWM ckpt。** 按说明书自己的
   优先级（§5.2："先在这个环境上验证 claim，再迁回 PushT"），这是第一阶段
   该做的事，但 claim 最终要在 PushT / CartPole 上复核才能投稿。
2. **TI-JEPA 的 pose 头位置探针 R²=0.64，比两个 baseline（0.97 / 0.99）都差**，
   说明 pose/motion 拆分这个架构本身让单帧位置编码变得更难训好（可能是
   motion_enc 的梯度和 pose_head 的梯度有一定冲突，或者只是欠训练）。
   这不影响核心 claim（单帧仍读不出速度），但会拉低 TI-JEPA 的绝对指标，
   是后续要专门调的方向。
3. **encoder 是小 CNN，predictor 是 MLP**，不是官方的 ViT-Tiny + AdaLN
   transformer；SIGReg 超参（`λ_reg=1.0, lr=1e-3`）是在这个小规模上重新调的，
   不是照抄官方数值（原因见 `notes/collapse_debug.md`）。迁回官方架构规模
   要重新扫一遍。
4. **kill experiment 的"同构型"是手工反推的 context**（假设过去 k-1 步匀速
   滑行、无动作），不是从真实 episode 里挑出来的，因此不会经过训练分布里
   "冲量刚发生"那类瞬态；这是保守的构造方式（更干净但更"简单"），真实
   episode 里的 pair 可能更难分。
5. n_pairs=200、horizon=15、8000 训练 step 都是为了在一次会话里出第一个
   可信数字选的规模，不是收敛后的最终数字；标准误差没有单独算。

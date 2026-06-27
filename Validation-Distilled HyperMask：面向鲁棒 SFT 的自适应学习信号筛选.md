# Validation-Distilled HyperMask：面向鲁棒 SFT 的自适应学习信号筛选

## 0. 摘要

标准 SFT 通常对 answer 中所有 token 施加等权交叉熵损失，但真实 SFT 数据中不同 token 的学习价值并不相同：有些 token 承载任务意图、推理结构和答案约束；有些 token 只是模板、风格、来源痕迹、冗余解释，甚至来自错误推理或脏标注。等权训练会让模型把数据集中的伪相关和低质量信号也当成监督信号学习，从而导致过拟合、鲁棒性下降和训练效率浪费。

本项目希望让 LLM 参与自身学习过程：模型在训练时先基于完整 SFT 样本 `prompt + answer` 生成 token-level mask 或 soft weights，再用 masked loss 反向传播。第一阶段使用独立 validation 数据作为教师信号，训练模型学会识别“哪些训练 token 有助于提升目标分布”；第二阶段在新的脏数据集上不再提供 validation，模型必须仅凭自身对样本和当前状态的理解生成 mask，并用 masked SFT 继续训练。

核心目标不是简单做 token cleaning，而是让模型内化一种 valid-free 的数据信号筛选能力：在面对新的低质量数据时，模型能够识别哪些 token 是当前自己应该学习的有效信号，从而获得更高的数据效率和对偏置/噪声的鲁棒性。

---

## 1. 背景与动机

### 1.1 标准 SFT 的问题

给定 SFT 样本：

[
d=(x,y)
]

其中 (x) 是 prompt，(y=(y_1,\dots,y_T)) 是 answer。标准 SFT 优化：

[
L_{\text{SFT}}(\theta)
======================

\sum_{t=1}^{T}
\mathrm{CE}*\theta(y_t \mid x,y*{<t})
]

这隐含一个强假设：answer 中每个 token 都是等价值、等质量、等重要的监督信号。

但在真实数据中，这个假设通常不成立。answer 里可能混有：

* 核心任务意图；
* 正确推理步骤；
* 最终答案约束；
* 格式 token；
* 模板化开头和结尾；
* 冗余自然语言解释；
* 数据来源特征；
* 错误中间步骤；
* 错误最终答案；
* 与任务无关的风格偏置。

标准 SFT 会无差别学习这些信号。当数据集中存在系统性偏置时，模型可能把“驾校环境”误当成“驾驶能力本身”。

### 1.2 驾校类比

人类学习驾驶时，即便训练时间主要发生在驾校，也能迁移到真实道路。原因是人类不会只记住驾校场景，而是能识别核心因果信号：

* 方向盘操作影响车身方向；
* 油门和刹车影响速度；
* 空间位置变化反馈当前操作是否正确；
* 道路规则和车辆控制规律比驾校环境本身更重要。

对应到 LLM SFT，理想模型也应该识别：

* 哪些 token 是任务本质；
* 哪些 token 是数据集环境；
* 哪些 token 是当前模型真正需要学习的；
* 哪些 token 可能是偏置、噪声或低价值监督。

### 1.3 项目核心问题

本项目关注的问题是：

> LLM 能否通过训练期的独立 validation supervision，学会一种 valid-free 的 token-level 学习信号筛选能力，并在新的低质量 SFT 数据上自主选择有效监督信号？

更具体地说，第一阶段可以使用 validation 作为教师，但第二阶段不能依赖 validation。最终希望模型直接暴露在新的脏数据上，也能生成合理 mask 并鲁棒训练。

---

## 2. 理想 Claim

### Claim 1：对低质量数据的鲁棒性

模型在新的脏 SFT 数据上，不依赖 validation，仍能通过自身生成的 mask 过滤低质量 token，从而相比 full SFT 获得更好的泛化和更少的过拟合。

脏数据包括但不限于：

* 错误最终答案；
* 错误中间推理；
* 冗余模板话术；
* 数据源风格偏置；
* 表面 shortcut；
* verbosity bias；
* 多数据源质量混杂。

期望结果：

[
\text{Masked SFT} > \text{Full SFT}
]

尤其在数据质量下降时，性能退化更慢。

---

### Claim 2：mask 能力随模型训练增强而增强

随着 LLM 主体能力增强，它对样本意图、推理结构和自身不确定性的表示更好，因此生成 mask 的能力也应增强。

关键不是 mask head 自己变强，而是 LLM 更能把“学习相关信息”写入内部 states / memory states，使一个轻量 decoder 能读出更好的 mask。

理想证据包括：

* 后期 checkpoint 生成的 mask 更接近 teacher / oracle mask；
* 后期 checkpoint 的 mask 对 student 训练更有帮助；
* mask 与 validation-gradient utility 的相关性随训练增强；
* frozen mask decoder 下，mask 改善主要来自 LLM writer 表征变好。

---

### Claim 3：mask 反映当前模型需要学习的东西

同一条 SFT 样本，对不同能力状态的模型不应产生完全相同的 mask。

例如：

* math-weak model 应更关注公式、变量关系、推理步骤；
* format-weak model 应更关注 JSON、LaTeX、答案格式；
* reasoning-weak model 应更关注中间逻辑转折；
* 已掌握基础能力的模型应降低简单模板 token 的权重。

形式上，mask 应该是：

[
m = g_\theta(x,y)
]

而不是静态的：

[
m = g(x,y)
]

其中 (\theta) 表示当前模型状态。

---

## 3. 与 Trivial 版本的区别

本项目必须避免退化成以下 trivial 版本。

### 3.1 不是普通数据清洗

普通数据清洗通常是 sample-level 或 token-level 的静态过滤：

[
m = g(x,y)
]

而本项目希望 mask 依赖当前模型状态：

[
m = g_\theta(x,y)
]

也就是说，同一数据对不同模型、不同训练阶段应该产生不同 mask。

### 3.2 不是永远依赖 validation 的 reweighting

经典 validation-based reweighting 在每个训练 step 用 clean validation 计算权重。但本项目的最终目标是：

* 训练期可以用 validation 教模型；
* 部署期 / 第二阶段不再需要 validation；
* 模型已经把 validation-aligned filtering rule 内化为自身能力。

### 3.3 不是独立判别器

如果 mask head 很大，并且直接接收大量 token hidden states，它可能变成一个独立 token classifier，而不是 LLM 自身能力的体现。

因此架构上应限制 mask decoder 容量，让 LLM 主体负责“写出学习状态”，mask decoder 只负责读出。

### 3.4 不是简单 loss-based selection

仅根据 high loss 或 low loss 选 token 会产生冲突：

* 低质量数据场景中，高 loss token 可能是噪声；
* 当前能力不足场景中，高 loss token 也可能是重要学习信号；
* 低 loss token 可能是已掌握内容，也可能是基础但关键的格式约束。

因此 mask 应该结合：

* 当前 token loss；
* uncertainty / margin；
* LLM memory states；
* validation-distilled teacher signal；
* 可选的历史 loss dynamics。

---

## 4. 相关工作定位

### 4.1 Learning to Reweight Examples

Ren et al. 提出用 clean validation set 通过 meta-gradient 给训练样本分配权重。核心思想是：训练样本权重不应只由训练 loss 决定，而应由该样本梯度是否有助于降低 validation loss 决定。

本项目继承其 validation-aligned reweighting 思想，但区别在于：

* 从 sample-level 扩展到 SFT answer token-level；
* 从持续依赖 validation 改为训练期 distillation、部署期 valid-free；
* 权重由 LLM 自身 memory states 生成；
* 目标是让模型学会自主数据筛选能力。

参考：[Ren18]

---

### 4.2 Instruction Data Selection / Self-Guided Selection

Cherry LLM / IFD 等工作关注 instruction sample-level 数据选择，用模型自身能力估计哪些样本更适合 instruction tuning。

本项目区别：

* 选择粒度从 sample-level 到 token-level；
* 目标不是只挑“好样本”，而是在完整 answer 内选择有效学习信号；
* mask 需要依赖当前模型状态，而不是静态样本质量；
* 第二阶段目标是面对新脏数据进行 valid-free mask inference。

参考：[CherryLLM]

---

### 4.3 Gradient-based Targeted Data Selection

LESS 使用低秩梯度相似度估计哪些 instruction data 对目标能力有帮助。它强调 targeted instruction tuning，即给定目标能力样本，选择最有影响的数据。

本项目与 LESS 的关系：

相同点：

* 都关心训练数据是否有助于目标分布；
* 都可以使用 validation / target examples 作为锚点；
* 都试图超过表面形式匹配。

区别：

* LESS 主要是 example-level 数据选择；
* 本项目是 token-level mask；
* LESS selection 通常发生在训练前或离线构建数据集；
* 本项目希望训练时动态产生 mask，并在第二阶段 valid-free 部署。

参考：[LESS]

---

### 4.4 Token-level Selection / Cleaning

Rho-1 提出 Selective Language Modeling，认为不是所有 token 都值得训练，使用 reference model 给 token 打分，只对更有价值的 token 施加 loss。

Token Cleaning 关注 SFT 中 token 质量差异，从 noisy-label perspective 出发过滤无信息 token。

ssToken 提出 self-modulated and semantic-aware token selection，用 history model 与 current model 的 per-token loss difference 作为 self-modulated signal，并加入 semantic-aware token importance。

本项目与这些工作最接近。关键差异应定位为：

* 使用 independent validation 在 meta-training 阶段生成 teacher signal；
* 训练部署期 valid-free student mask；
* 借助 LLM memory states 而非单纯 loss score；
* 关注 mask 与当前模型状态的耦合；
* 进一步研究 self-induced trajectory 上的 mask drift 和 on-policy mask training。

参考：[Rho1], [TokenCleaning], [ssToken]

---

### 4.5 SHINE-style In-context Hypernetwork

SHINE 的核心思想是复用 LLM 自身参数，把 context 压缩成 memory states，再由轻量模块生成 LoRA adapter。

本项目不需要生成 LoRA，因为目标不是把 context 写入参数用于 inference，而是生成训练 loss mask。但 SHINE 的架构思想非常有启发：

[
\text{context} \rightarrow \text{LLM memory states} \rightarrow \text{small readout}
]

本项目可改造成：

[
prompt+answer \rightarrow \text{LLM learning memory states} \rightarrow mask
]

也就是说，LLM 负责理解完整样本并写出“学习状态”，小型 M2Mask decoder 只负责读取这些 states 生成 token mask。

参考：[SHINE]

---

### 4.6 DAgger / On-policy Imitation Learning

DAgger 的核心问题是：如果只在专家轨迹上训练 policy，部署时 learner 会访问由自己策略诱导出来的新状态，导致 covariate shift。因此 DAgger 让 learner 在自己的轨迹上运行，再查询 expert 进行纠偏。

本项目存在类似问题：

* 阶段二模型状态由 student mask 策略诱导；
* 如果阶段一只在 teacher-induced states 上训练，阶段二可能 OOD；
* 因此需要 student-induced on-policy rollout，让模型在训练期就见到自己 mask 后产生的状态。

参考：[DAgger]

---

## 5. 总体路线

项目可以按“两阶段训练 + 可选 on-policy 增强”设计。

### 5.1 Stage 1：Validation-distilled mask meta-training

第一阶段有多个数学数据集：

[
D_1,D_2,\dots,D_K
]

每个 step 随机采样：

[
D_i \rightarrow B_{train}, \quad D_j \rightarrow B_{valid}, \quad i \neq j
]

使用 (B_{valid}) 训练 teacher mask，使其选择能降低 independent validation loss 的 token。

同时训练 student mask，使其在不看 validation 的情况下模仿 teacher。

目标是把：

[
g_\theta(x,y,V) \rightarrow m^T
]

蒸馏为：

[
g_\theta(x,y) \rightarrow m^S
]

### 5.2 Stage 2：Valid-free dirty data deployment

第二阶段选择一个新的脏数据集 (D_{dirty})。

不再提供 validation。

每个输入：

[
(x,y) \in D_{dirty}
]

由当前 LLM 生成：

[
m^S = g_\theta(x,y)
]

然后使用 masked loss 更新模型：

[
L_{train}^{masked}
==================

\sum_t
\mathrm{stopgrad}(m^S_t)
\ell_t(\theta)
]

第二阶段是核心 inference setting：模型必须证明自己已经获得自主学习信号筛选能力。

---

## 6. 主方法：Validation-Distilled HyperMask

### 6.1 架构总览

输入：

```text
<BOS>
[Prompt x]
<ANSWER>
[Answer y]
<LEARN_MEM_1> ... <LEARN_MEM_r>
```

由于 memory tokens 放在 answer 后面，在 causal attention 下它们可以 attend 到完整 prompt+answer。

LLM 输出两类 states：

* answer token states：
  [
  H^{ans}*\theta = {h_t}*{t=1}^{T}
  ]

* learning memory states：
  [
  M_\theta = {u_j}_{j=1}^{r}
  ]

每个 answer token 还附加当前模型状态特征：

[
s_t =
[
\ell_t,\
p_\theta(y_t),\
\mathrm{entropy}_t,\
\mathrm{margin}_t,\
\mathrm{position}_t
]
]

student mask decoder：

[
m^S_t
=====

D_\psi(h_t,M_\theta,s_t)
]

其中 (D_\psi) 应尽量小，防止它变成独立判别器。

---

## 7. Mask Decoder 设计

### 7.1 V0：Token MLP Baseline

最小版本：

[
\bar{M} = \mathrm{Pool}(M_\theta)
]

[
m_t = \sigma(\mathrm{MLP}([h_t,\bar{M},s_t]))
]

用途：

* 快速验证 hidden state + scalar features 是否足够；
* 作为主方法的简单 baseline；
* 判断 SHINE-like memory 是否必要。

---

### 7.2 V1：Cross-attention M2Mask

主推版本：

[
q_t = W_q[h_t,s_t]
]

[
k_j,v_j = W_k u_j,W_v u_j
]

[
z_t = \mathrm{CrossAttn}(q_t,K_M,V_M)
]

[
m_t = \sigma(\mathrm{MLP}([h_t,z_t,s_t]))
]

解释：

* answer token query 当前 token 学习状态；
* memory states 提供完整样本级 intent / noise / current-need 信息；
* decoder 输出 token-level soft mask。

这是第一阶段最平衡的主架构。

---

### 7.3 V2：Frozen Decoder + Trainable LLM Writer

更强 claim 版本：

[
m_t = D_{\psi}^{frozen}(R_\theta(x,y))
]

其中：

* (D_\psi) 是冻结 mask decoder；
* (R_\theta(x,y)) 是 LLM 通过 memory tokens 写出的 learning representation。

这个版本的优势是：

* decoder 不能通过训练变成独立判别器；
* mask 改善主要来自 LLM writer 表征变好；
* 更支持“mask 能力随 LLM 能力增强”的 claim。

推荐策略：

1. 先用 V1 trainable decoder 跑通；
2. 再尝试冻结 decoder；
3. 如果 frozen decoder 接近 trainable decoder，可作为主文强卖点；
4. 如果掉点明显，则先将 frozen decoder 放到第二篇或 appendix。

---

## 8. Mask 输出与 Loss

### 8.1 Soft Mask

第一阶段建议使用 soft mask，而不是 hard top-k。

decoder 输出 logit：

[
a_t = D_\psi(...)
]

soft weight：

[
w_t = \sigma(a_t/\tau)
]

budget normalization：

[
\tilde{w}*t =
\frac{\rho T \cdot w_t}{\sum*{j=1}^{T} w_j+\epsilon}
]

其中 (\rho) 是平均保留比例，建议先试：

[
\rho \in {0.3,0.5,0.7}
]

训练 loss：

[
L_{train}^{masked}
==================

\frac{
\sum_t \mathrm{stopgrad}(\tilde{w}_t)\ell_t
}{
\sum_t \mathrm{stopgrad}(\tilde{w}_t)+\epsilon
}
]

### 8.2 为什么先用 Soft Mask

soft mask 更稳定：

* 不会早期 hard filtering 导致训练崩；
* 更容易蒸馏 teacher；
* 更容易分析 mask distribution；
* 后续可再转 hard top-k 做 token efficiency 实验。

---

## 9. Teacher Mask 设计

teacher mask 使用 independent validation 生成，不在第二阶段使用。

对 train batch 中每个 token 初始化 mask logit：

[
a_t = 0
]

得到初始 mask：

[
m_t = \sigma(a_t)
]

做一次虚拟更新：

[
\theta'
=
\theta
-
\alpha
\nabla_\theta
\sum_t m_t \ell_t^{train}(\theta)
]

计算 validation loss：

[
L_{valid}(\theta')
]

对 (a_t) 求 meta-gradient：

[
a_t
\leftarrow
a_t
---

\eta_m
\nabla_{a_t}L_{valid}(\theta')
]

得到 teacher mask：

[
m^T_t = \mathrm{Normalize}(\sigma(a_t))
]

teacher mask 的直觉是：如果某个 train token 的梯度方向有助于降低 independent validation loss，它应获得更高权重。

---

## 10. Student Distillation

student branch 不看 validation：

[
m^S = g^S_\theta(x,y)
]

teacher branch 看 validation：

[
m^T = g^T_\theta(x,y,V)
]

训练 student：

[
L_{distill}
===========

D(m^S,m^T)
]

可选具体形式：

### 10.1 MSE

[
L_{\text{mse}}
==============

\sum_t
(m^S_t-m^T_t)^2
]

优点：稳定、简单。

### 10.2 Ranking Loss

[
L_{rank}
========

-\sum_{i,j}
\mathbf{1}[m^T_i>m^T_j]
\log \sigma(a^S_i-a^S_j)
]

优点：更关注 top token 排序。

### 10.3 Budget Regularization

[
L_{budget}
==========

\left(
\frac{1}{T}\sum_t m^S_t - \rho
\right)^2
]

防止全 0 或全 1。

### 10.4 总 Loss

[
L_{mask}
========

\lambda_1 L_{\text{mse}}
+
\lambda_2 L_{rank}
+
\lambda_3 L_{budget}
]

---

## 11. 训练策略

### 11.1 Phase 1：Teacher Warmup

目标：

* 验证 teacher mask 有效；
* 让 student 初步模仿 teacher；
* 防止 student 初期乱 mask 影响主模型。

策略：

* 主模型主要使用 full SFT 或 teacher-masked SFT；
* student 只学习 (m^T)，不主导真实更新；
* 占总训练 steps 的 10%–20%。

---

### 11.2 Phase 2：Mixed Mask Training

真实训练 mask 混合 teacher 和 student：

[
m =
\begin{cases}
m^T, & p \
m^S, & 1-p
\end{cases}
]

其中 (p) 从较大逐步降低，例如：

[
p: 0.8 \rightarrow 0.2
]

目标：

* 逐渐减少对 validation teacher 的依赖；
* 让模型适应 student mask；
* 缩小 stage 1 与 stage 2 的 train/deploy gap。

---

### 11.3 Phase 3：Student-dominant Training

后期大部分真实训练使用 student mask：

[
p \approx 0
]

teacher 只用于蒸馏或周期性纠偏。

这是必要的，因为第二阶段完全无 validation。如果第一阶段后期仍主要依赖 teacher，第二阶段会出现 OOD。

---

## 12. Validation Dropout

Validation dropout 是第一阶段到第二阶段过渡的关键。

训练时随机 drop validation：

[
m =
\begin{cases}
g_\theta(x,y,V), & p \
g_\theta(x,y,\varnothing), & 1-p
\end{cases}
]

早期 (p) 高，后期 (p) 低。

目的：

* 让 student 提前适应无 validation setting；
* 避免模型只学会“有 valid 时筛选”；
* 提高第二阶段 valid-free mask inference 稳定性。

---

## 13. Self-referential Mask Drift 问题

### 13.1 问题定义

理想 mask 依赖当前模型状态：

[
m_t = g_{\theta_t}(x,y)
]

但当前模型状态又会被过去 mask 更新改变：

[
\theta_{t+1}
============

## \theta_t

\eta
\nabla_\theta
\sum_i
m_{t,i}
\ell_i(\theta_t)
]

因此，mask policy 不是旁观者，而是会改变未来自己所面对的状态分布。

可能出现连锁漂移：

[
\text{mask error}
\rightarrow
\text{parameter drift}
\rightarrow
\text{state OOD}
\rightarrow
\text{worse mask}
\rightarrow
\text{more drift}
]

这就是 self-referential mask drift。

---

### 13.2 为什么普通两阶段可能不够

如果第一阶段只在 teacher-induced states 上训练，而第二阶段实际使用 student mask，那么第二阶段模型访问的是：

[
\theta_{stage2}
\sim
p(\theta \mid \text{student mask rollout})
]

而第一阶段训练分布是：

[
\theta_{stage1}
\sim
p(\theta \mid \text{teacher mask rollout})
]

二者可能不一致。

---

## 14. On-policy Student-induced Rollout

为解决 mask drift，可在第一阶段加入 student-induced rollout。

### 14.1 核心原则

rollout 必须由 student mask 驱动，teacher 只负责纠偏。

也就是说，内循环更新模型时使用：

[
m^S = g^S_\theta(x,y)
]

而不是：

[
m^T = g^T_\theta(x,y,V)
]

否则模型见到的是 teacher-induced states，而不是第二阶段真正会访问的 student-induced states。

---

### 14.2 First-order Rollout 版本

完整 differentiable unroll 代价高。第一版建议使用 first-order rollout。

步骤：

1. 复制当前模型：
   [
   \bar{\theta}_0 \leftarrow \theta
   ]

2. 对 (k=0,\dots,K-1)：

   采样 train batch：

   [
   B^{train}_k
   ]

   student 无 valid 生成 mask：

   [
   m^S_k
   =====

   g^S_{\bar{\theta}_k}(B^{train}_k)
   ]

   用 student mask 更新 shadow model：

   [
   \bar{\theta}_{k+1}
   ==================

   ## \bar{\theta}_k

   \eta
   \nabla_{\bar{\theta}*k}
   L^{masked}*{train}(\bar{\theta}_k,m^S_k)
   ]

   采样 valid batch：

   [
   B^{valid}_k
   ]

   teacher 在当前 shadow state 上计算 correction：

   [
   m^T_k
   =====

   g^T_{\bar{\theta}_k}(B^{train}_k,B^{valid}_k)
   ]

   训练 student：

   [
   L_{distill}(m^S_k,m^T_k)
   ]

3. 外层更新真实模型 (\theta)。

在 first-order 版本中，可对 shadow state stop-gradient，避免反传穿过整个 rollout。

---

### 14.3 何时需要 On-policy Rollout

如果出现以下现象，应加入 on-policy rollout：

* student 在短期 stage-2 有效，但长程训练后性能下降；
* mask 分布逐渐 collapse；
* student mask 与 teacher/oracle mask 相关性随训练下降；
* 使用旧 checkpoint mask 训练当前模型效果接近当前 mask，说明 model-specific claim 不成立；
* stage 1 训练好，但 stage 2 新脏数据泛化差。

---

## 15. 第一阶段实验矩阵

### 15.1 Baselines

1. Full SFT
2. Random token mask
3. High-loss token mask
4. Low-loss token mask
5. Rho-style / excess-loss token mask
6. Sample-level filtering
7. Token Cleaning-style baseline
8. ssToken-style baseline，如果实现成本可接受

### 15.2 Our Variants

1. Teacher meta mask，有 valid，作为上界
2. Student V0：MLP readout，无 valid
3. Student V1：memory + cross-attention，无 valid
4. Student V1 + validation dropout
5. Student V1 + frozen decoder
6. Student V1 + student-induced rollout
7. Student V1 + frozen decoder + rollout

---

## 16. 数据集设计

### 16.1 Meta-train Datasets

多个数学 SFT 数据集：

[
D_1,\dots,D_K
]

每个 step 随机选择：

[
D_i \rightarrow train,\quad D_j \rightarrow valid,\quad i\neq j
]

目标是让 mask policy 学到跨数据集的有效学习信号，而不是某个数据集内部风格。

### 16.2 Meta-test Clean Datasets

完全不参与第一阶段训练，用于评估泛化数学能力。

### 16.3 Stage-2 Dirty Datasets

第二阶段核心部署测试，应构造或收集脏数据：

1. Wrong final answer
   推理大体正确，但最终答案错。

2. Wrong reasoning step
   最终答案可能正确，但中间存在错误推理。

3. Verbose noise
   加入大量模板话术、冗余解释、无关背景。

4. Source bias
   某个数据源总是某种格式或风格，但该风格与正确性无关。

5. Shortcut bias
   某些表面词和答案类型伪相关。

6. Mixed quality
   高质量、低质量、错误样本混杂。

---

## 17. 关键评价指标

### 17.1 下游性能

* GSM8K
* MATH
* SVAMP
* ASDiv
* MultiArith
* 其他未见数学 benchmark

### 17.2 鲁棒性

在不同脏数据比例下比较性能退化曲线：

[
q \in {0, 0.1, 0.2, 0.4, 0.6}
]

其中 (q) 是脏样本或脏 token 比例。

理想结果：full SFT 随 (q) 增加明显退化，HyperMask 退化更慢。

### 17.3 Token Efficiency

比较达到同等性能所需训练 token 数。

例如：

[
\text{HyperMask uses } 50% \text{ tokens but matches or exceeds full SFT}
]

### 17.4 Teacher-Student Gap

衡量：

[
\mathrm{Corr}(m^S,m^T)
]

[
\mathrm{TopKOverlap}(m^S,m^T)
]

[
D_{\mathrm{KL}}(m^T | m^S)
]

### 17.5 Model-specificity

同一数据，对不同模型生成 mask：

[
m^{A},m^{B}
]

计算 overlap：

[
\mathrm{Overlap}(m^{A},m^{B})
]

并分析差异是否对应模型短板。

### 17.6 Current-need Alignment

构造不同短板模型：

* math-weak；
* format-weak；
* reasoning-weak；
* overfitted；
* strong checkpoint。

检查 mask 是否集中在对应短板 token 类别上。

---

## 18. 决策点

### 18.1 Teacher mask 是否成立

| 观察                   | 解释                                  | 决策                                                  |
| -------------------- | ----------------------------------- | --------------------------------------------------- |
| Teacher 不优于 full SFT | valid meta signal 不稳定               | 调 teacher：batch size、mask budget、虚拟步长、normalization |
| Teacher 优于 baseline  | validation-aligned token signal 有价值 | 进入 student distillation                             |

---

### 18.2 Student 是否能逼近 Teacher

| 观察                             | 解释                    | 决策                          |
| ------------------------------ | --------------------- | --------------------------- |
| Student imitation 差            | 架构不够                  | 从 V0 升级到 V1，加 memory tokens |
| Student imitation 好但 stage-2 差 | train/deploy mismatch | 加 validation dropout        |
| Student stage-2 短期好但长程差        | self-induced drift    | 加 on-policy rollout         |

---

### 18.3 SHINE-like Memory 是否必要

| 观察               | 解释                      | 决策                      |
| ---------------- | ----------------------- | ----------------------- |
| V0 与 V1 接近       | memory compression 不是关键 | 第一篇可用简单架构               |
| V1 明显优于 V0       | LLM memory states 有价值   | 主文强调 HyperMask          |
| 多层 memory 明显优于单层 | 层间信息重要                  | 考虑更接近 SHINE 的多层 readout |

---

### 18.4 Frozen Decoder 是否可行

| 观察                     | 解释                   | 决策                            |
| ---------------------- | -------------------- | ----------------------------- |
| Frozen 接近 trainable    | 强支持 LLM writer claim | 主推 frozen decoder             |
| Frozen 明显掉点            | decoder 承担大量判别能力     | 第一篇用 trainable，frozen 留第二篇    |
| Frozen 只有加 rollout 后有效 | mask drift 是关键       | 第二篇主打 on-policy self-learning |

---

## 19. 两篇论文规划

### 19.1 Paper 1：偏效率与鲁棒数据清洗

可能标题：

* Validation-Distilled Token Masking for Efficient and Robust SFT
* Learning to Mask Noisy Supervision without Validation at Deployment
* HyperMask: Validation-Distilled Token Selection for Robust LLM Fine-tuning

核心问题：

> 能否在训练期利用 independent validation 学到 valid-free token mask，并在新脏数据上提高 SFT 效率和鲁棒性？

主要贡献：

1. 提出 validation-distilled token mask framework。
2. 提出 SHINE-like memory readout 架构，不生成 LoRA，只生成 mask。
3. 证明 deployment-time 无 validation 仍能筛选有效 token。
4. 在脏数学 SFT 数据上提升鲁棒性和 token efficiency。

不建议在第一篇中强行 claim AGI/self-improvement。

---

### 19.2 Paper 2：偏模型状态与自我学习信号识别

可能标题：

* Model-State-Conditioned Learning Signal Recognition for Self-Adaptive LLM Fine-tuning
* LLMs Can Learn What They Need to Learn
* On-policy Self-Masking for Model-Conditioned Learning Signal Selection

核心问题：

> mask 是否真的依赖当前模型状态，并反映模型当前需要学习的信号？

主要贡献：

1. 定义 self-referential mask drift 问题。
2. 提出 student-induced on-policy rollout。
3. 证明同一数据对不同能力模型产生不同 mask。
4. 证明当前模型自己的 mask 优于其他模型生成的 mask。
5. 证明 mask 与当前模型短板和 validation-gradient utility 对齐。

---

### 19.3 合并还是拆分

推荐策略：

先按一篇完整系统做实验，但按两篇布局保留接口。

如果以下成立，可以拆两篇：

* Paper 1 不依赖 on-policy rollout 也能作为 robust token cleaning 成立；
* Paper 2 有独立的 on-policy / model-specific / current-need 证据。

如果以下情况出现，应合成一篇：

* Paper 1 必须依赖 rollout 才有效；
* Paper 2 没有足够独立实验，只是 Paper 1 的 ablation；
* 三个模块高度耦合，拆开后每篇都不完整。

---

## 20. 推荐执行顺序

### Step 1：Teacher Sanity Check

只跑 teacher meta mask，确认其相对 full SFT、random mask、loss-only mask 有优势。

成功标准：

[
\Delta L_{valid}^{teacher}
<
\Delta L_{valid}^{baseline}
]

或者 teacher-masked update 在 held-out validation 上更稳定。

---

### Step 2：Student V0

实现最小 student：

[
m^S_t = \mathrm{MLP}([h_t,s_t,\bar{M}])
]

测试：

* imitation gap；
* stage-2 dirty data；
* token efficiency。

---

### Step 3：Student V1 HyperMask

加入 memory tokens + cross-attention decoder。

比较 V1 vs V0，判断 SHINE-like memory 是否有价值。

---

### Step 4：Validation Dropout

加入 teacher/student 混合训练和 annealing。

比较：

* no dropout；
* fixed 50/50；
* annealed dropout。

---

### Step 5：Frozen Decoder

先训练 decoder，再冻结，只训练 LLM writer。

比较：

* trainable decoder；
* frozen decoder；
* frozen decoder + trainable memory tokens；
* frozen decoder + trainable LLM LoRA。

---

### Step 6：On-policy Rollout

如果 stage-2 长程漂移明显，再加入 first-order student-induced rollout。

优先做：

* (K=1)
* (K=2)
* replay buffer

避免一开始做 full differentiable unroll。

---

## 21. 当前最推荐的主版本

当前建议的第一版主方法是：

> Validation-Distilled HyperMask

架构：

[
(x,y)
\rightarrow
\mathrm{LLM}*\theta
\rightarrow
(H^{ans}*\theta,M_\theta,s_\theta)
\rightarrow
D_\psi
\rightarrow
m^S
]

训练：

[
m^T = \text{validation-meta-gradient teacher}
]

[
L_{mask}=D(m^S,m^T)
]

[
L_{train}=\sum_t \mathrm{stopgrad}(m^S_t)\ell_t
]

训练策略：

* teacher warmup；
* mixed teacher/student mask；
* validation dropout；
* student-dominant 后期；
* stage-2 无 validation dirty-data inference。

on-policy rollout 先作为增强模块，不作为第一版必选模块。只有当 student 在第二阶段出现长程 drift 时，再升级为主方法。

---

## 22. 最终研究定位

本项目最强的定位不是：

> 我们提出一个新的 token cleaner。

而是：

> 我们训练 LLM 内化一种 validation-aligned learning signal selection ability，使其在没有 validation 的新脏数据上，基于自身当前状态筛选有效 token，并用这些信号继续训练自己。

更精炼的表述：

> LLM 不只是被数据训练；它还学习如何从数据中选择自己应该学习的信号。

---

## 23. 2026-06-23 teacher sanity 更新：final-answer validation 优先

当前实现已经从最初的宽 answer-window teacher，推进到更窄的 final-answer-only validation teacher。

关键设定是：

* support loss 仍然覆盖所有 completion token；
* teacher mask 仍然在 support completion token 上优化；
* target / valid outer loss 只看 final answer span；
* 不把 reasoning window、所有数字 token、operator token 自动纳入 valid loss。

对应配置为：

```yaml
target_loss:
  mode: answer_focus
  answer_window_tokens: 0
  min_tokens_per_sample: 1
  max_tokens_per_sample: 4
  include_numeric_tokens: false
  include_operator_tokens: false
  background_weight: 0.0
  focus_weight: 1.0

support_loss:
  mode: all
```

这不是把 teacher 的搜索空间改成 final answer token，而是把 teacher 的评价标准改成：

> 选出的 support token 是否能通过一次虚拟 LoRA update 改善 target final answer loss。

### 23.1 为什么改成 final-answer-only

之前的 answer-focus 设置仍然较宽：

* answer_window_tokens = 8；
* 每个样本至少保留 24 个 valid token；
* 数字 token 和 operator token 也被纳入 valid loss。

这个设置会让 validation signal 混入大量 reasoning format、换行、等号、算式和中间数字。它比 full completion loss 更接近答案，但还不是最终任务指标。

final-answer-only 的目的不是让信号变大，而是让信号更干净：

* outer loss 更接近数学任务最终评估；
* teacher 不再因为改善中间推理格式而获得过多奖励；
* top-loss baseline 的优势被削弱，因为高 loss token 不一定能改善 final answer。

### 23.2 正式 sanity 结果

正式 4GPU teacher sanity 使用：

* config: `configs/claim1_teacher_sanity_final_answer_only_qwen_v103.yaml`
* output: `outputs/claim1_teacher_sanity_final_answer_only_qwen05b_v103`
* steps: 40
* target_batch_size: 256 / rank
* world_size: 4
* effective target examples: 1024 / step
* target_micro_batch_size: 4

平均资源：

```text
step_seconds_mean   152.10
gpu_mem_max_gb_mean 22.24
```

主要结果：

```text
teacher_gain_mean                 0.2923
top_loss_gain_mean                0.1479
random_gain_mean                  0.0787
full_gain_mean                    0.0869
low_loss_gain_mean                0.0520

teacher_minus_top_loss_gain_mean  0.1444
teacher_minus_random_gain_mean    0.2136
teacher_minus_full_gain_mean      0.2054

teacher_win_top_loss_mean         0.7188
teacher_win_random_mean           0.8000
teacher_win_full_mean             0.8375
```

逐 step 看，teacher gain 只有 1/40 个 step 为负；按 step 平均 gain 差看，teacher 有 35/40 个 step 超过 top-loss。

和 v102 solver-sharp answer-window teacher 对比：

```text
v102 solver-sharp:
  teacher_gain_mean              0.3357
  teacher_minus_top_loss_mean    0.1064
  teacher_win_top_loss_mean      0.4813

v103 final-answer-only:
  teacher_gain_mean              0.2923
  teacher_minus_top_loss_mean    0.1444
  teacher_win_top_loss_mean      0.7188
```

结论：

> v103 final-answer-only 的绝对 teacher gain 略低，但相对 top-loss 更稳、更明显赢。

因此，后续 teacher-student Stage 1 应优先使用 final-answer-only target validation，而不是继续沿用宽 answer-window 版本。

### 23.3 mask 解释上的限制

final-answer-only 并不意味着 teacher mask 只落在 final answer token 上。

v103 teacher mask 平均 mass：

```text
answer    15.5%
punct     37.6%
number    24.2%
operator   9.5%
word       8.8%
format     4.4%
```

这说明 teacher 仍然经常给换行、标点、数字、算式 token 较高权重。正确解释是：

> 这些 support token 通过一次虚拟 update 能改善 target final answer loss。

而不是：

> teacher 已经学会只选择答案 token。

这个现象有两种可能解释：

1. 数学解答中的标点、换行、算式结构确实参与形成有用梯度；
2. teacher 仍有 surrogate shortcut，利用格式 token 改善 final answer proxy。

因此下一步不应马上声称 mask 语义已经可解释，而应先验证 student 是否能稳定蒸馏这个 teacher。

### 23.4 下一步实验优先级

当前推荐顺序：

1. 使用 v103 final-answer-only teacher 跑短 Stage 1 teacher-student。
2. 和 v102 solver-sharp answer-window Stage 1 对照 student_teacher_corr、student mask_std、future_gain。
3. 如果 student 仍学不到 teacher，再考虑限制 teacher support 搜索空间，例如 answer-span-biased support mask 或 token-type regularization。
4. 暂时不要把 support loss 也改成 final-answer-only；那会把训练信号压得过窄，可能让 student 更难学。

当前最稳的 claim 是：

> Validation-distilled teacher 在 final-answer validation 上能稳定优于 random、full、top-loss support-token baselines。

还不能 claim：

> teacher mask 已经等价于 final-answer token selector。

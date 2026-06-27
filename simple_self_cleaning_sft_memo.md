# Single-Pass Self-Cleaning SFT：以清洗为核心的简化路线

## 0. 一句话概括

本路线把“数据清洗”看作一个普通 downstream 任务：先用已有外部清洗方法产生 token-level 或 span-level 清洗标签，再把这种清洗能力蒸馏到 LLM 的一个轻量 mask head 中。之后在新的低质量 SFT 数据上，不再运行外部清洗器、reference model、validation-based selector 或额外打分流程，而是在正常 SFT 的同一次 forward pass 中直接生成 mask，并用该 mask 调制 token-level loss。

核心 claim 是 **效率**：外部清洗只用于训练 mask head；部署到新脏数据时，模型通过训练 forward 本身完成清洗，避免外部清洗成本。

---

## 1. 研究背景

SFT 数据质量对大语言模型的下游表现影响很大。真实 SFT 数据常包含以下问题：

- answer 中夹杂模板化话术、冗余解释或格式噪声；
- 推理步骤中存在局部错误；
- 长答案中部分 span 与问题无关；
- 数据源带有固定风格、固定结构或伪相关；
- 多源混合数据中，不同来源质量不一致；
- 自动生成或弱标注数据中存在 hallucination、错误事实、错误公式或错误 reasoning。

传统 SFT 通常对 answer 的所有 token 施加同等 CE loss：

\[
L_{SFT}=\sum_{t=1}^{T} CE_\theta(y_t|x,y_{<t})
\]

这意味着低质量 token 和高质量 token 对梯度的贡献是一样的。若低质量 token 比例较高，模型会被迫拟合错误、冗余或偏置模式。

已有 token cleaning / data selection 方法可以缓解这个问题，但常常需要额外流程，例如：

- 离线跑 reference model；
- 计算 influence 或 gradient similarity；
- 使用外部 LLM judge；
- 多轮 scoring/filtering；
- 每个新数据集都重新清洗；
- 在训练前额外生成 cleaned dataset。

这些方法有效，但部署成本较高，尤其当研究者希望持续吸收新的低质量数据时，每次都重新运行外部清洗器会带来明显计算开销。

本路线试图回答一个简单问题：

> 能否把已有外部清洗方法的能力蒸馏进 LLM，使模型在后续训练新脏数据时，通过一次正常 forward pass 直接完成 token-level 清洗？

---

## 2. 核心思想

普通 SFT 的 forward pass 只用于得到 logits 和 CE loss。实际上，forward pass 还会产生丰富的 hidden states、token-level uncertainty、loss pattern 等信息。这些信息可以被用于判断 answer 中哪些 token 更像有效学习信号，哪些 token 更像噪声或无效内容。

因此我们给 LLM 增加一个轻量 mask head：

\[
q_t = g_\phi(h_t)
\]

其中：

- \(h_t\) 是 LLM 在 token \(t\) 处的 hidden state；
- \(g_\phi\) 是轻量 mask head；
- \(q_t\in[0,1]\) 表示 token \(t\) 应该被保留用于反向传播的概率或权重。

训练阶段，mask head 通过外部清洗器产生的伪标签学习：

\[
L_{mask}=BCE(q_t, q_t^{teacher})
\]

部署到新的低质量 SFT 数据时，不再使用外部 teacher。模型在同一次 SFT forward 中生成 mask：

\[
L_{masked\text{-}SFT}=\sum_{t=1}^{T} \mathrm{stopgrad}(q_t)\cdot CE_\theta(y_t|x,y_{<t})
\]

其中 \(q_t\) 只作为 loss 权重，不让 masked SFT loss 直接更新 mask decision，避免模型通过改变 mask 逃避难 token。

---

## 3. 研究目标

本路线的目标不是提出一个新的清洗准则，而是提出一种 **清洗能力的摊销机制**：

> 昂贵外部清洗器只在训练 mask head 时使用一次；之后模型在面对未见脏数据时，可以用自身 forward pass 直接生成 token-level mask。

因此主贡献应聚焦于：

1. **效率**：避免在每个新 SFT 数据集上运行外部清洗器；
2. **低质量数据鲁棒性**：masked loss 使模型减少对噪声 token 的拟合；
3. **single-pass deployment**：mask 生成复用 SFT forward，不需要额外 reference forward 或 validation scoring；
4. **可迁移清洗能力**：mask head 在 teacher 标注数据上训练后，能泛化到未见低质量数据。

---

## 4. 方法流程

### 4.1 Stage 1：外部清洗器生成 teacher mask

准备一批用于训练清洗能力的数据：

\[
D_{cleaning-train}=\{(x_i,y_i)\}
\]

对每个样本运行已有外部清洗方法，得到 token-level soft mask：

\[
q_{i,t}^{teacher}\in[0,1]
\]

teacher mask 可以来自多种方法：

- token cleaning 方法；
- reference-model token scoring；
- LLM judge 对 answer span 的 clean/noisy 判断；
- influence-based selection 的 token 或 span 近似；
- 人工规则或合成噪声标签作为补充。

推荐使用 soft label，而不是 hard 0/1 label，因为不同清洗器对“低质量”的定义不完全一致。

---

### 4.2 Stage 2：训练 mask head，把清洗作为 downstream task

在 LLM 上接一个轻量 mask head：

\[
q_t=g_\phi(h_t)
\]

最简单实现是：

\[
g_\phi(h_t)=\sigma(W_2\,\mathrm{GELU}(W_1h_t))
\]

训练目标：

\[
L=L_{SFT}+\lambda L_{mask}
\]

其中：

\[
L_{mask}=\sum_t BCE(q_t,q_t^{teacher})
\]

可以有两种简单训练方式：

**方式 A：冻结 LLM，只训练 mask head。**

优点是稳定、简单，清洗任务不会影响主模型能力。缺点是 mask head 能力受限于已有 hidden states。

**方式 B：联合训练 LLM 和 mask head。**

优点是 LLM 可以学习暴露更适合清洗任务的表示。缺点是训练更复杂，需要防止清洗任务损害原有 SFT 能力。

在简化路线中，建议先采用方式 A 作为 baseline，再尝试方式 B。

---

### 4.3 Stage 3：在新脏数据上 single-pass masked SFT

给定一个新的低质量 SFT 数据集：

\[
D_{dirty-new}=\{(x_i,y_i)\}
\]

不再运行外部清洗器。每个训练 batch 只做正常 SFT forward：

1. 输入 prompt + answer；
2. LLM 产生 logits 和 hidden states；
3. mask head 根据 hidden states 输出 \(q_t\)；
4. 用 \(q_t\) 调制 token-level CE loss；
5. 反向传播 masked loss 更新 LLM。

训练损失为：

\[
L_{deploy}=\sum_{t=1}^{T}\mathrm{stopgrad}(q_t)\cdot CE_\theta(y_t|x,y_{<t})
\]

此时没有外部 cleaner、没有 validation set、没有 reference model、没有二次 forward。

这就是本路线的核心效率 claim。

---

## 5. 为什么说它是“single-pass”

普通 SFT 需要一次 forward 来计算 logits 和 loss。

本方法在同一次 forward 中额外读取 hidden states 并经过一个轻量 MLP 得到 mask。额外计算主要是：

\[
O(Td r)
\]

其中 \(T\) 是序列长度，\(d\) 是 hidden size，\(r\) 是 mask head 的中间维度。相较于 LLM forward/backward 的成本，这个开销很小。

因此部署阶段的边际成本接近于一个轻量 head，而不是一个完整外部模型 forward 或 influence 计算。

需要注意措辞：

- 不应说“完全没有额外算力”；
- 更准确说法是“没有外部清洗器成本，部署阶段边际开销很小，并复用已有 SFT forward”。

---

## 6. 与已有工作的区别

### 6.1 与离线数据清洗的区别

离线清洗流程通常是：

\[
D_{dirty}\rightarrow Cleaner\rightarrow D_{cleaned}\rightarrow SFT
\]

本方法是：

\[
D_{dirty}\rightarrow LLM\ forward\rightarrow Mask\rightarrow Masked\ SFT
\]

区别在于：

- 清洗不再是训练前的离线预处理；
- 每个 batch 的 mask 在训练 forward 中直接产生；
- 新脏数据上不需要重新运行外部 cleaner。

---

### 6.2 与 Token Cleaning 的区别

Token Cleaning 类方法关注如何评估 token quality，并基于 token quality 过滤无信息 token。本路线不主张提出更好的 token quality criterion，而是将已有 token cleaning 的输出蒸馏进模型内部，使其在新数据上无需外部 scoring 即可产生 mask。

因此区别是：

- Token Cleaning 是 teacher / cleaning criterion；
- 本方法是 student / amortized deployment mechanism。

---

### 6.3 与 Rho-1 / Selective Language Modeling 的区别

Selective Language Modeling 强调不是所有 token 都应训练，并使用 reference model 等方式选择高价值 token。本路线同样采用 selective loss，但关注点不同：

- Rho-1 更偏 pretraining 场景；
- 本路线面向 SFT 低质量数据；
- 本路线的重点不是 reference scoring，而是把 scoring 能力蒸馏进训练模型，使部署阶段不需要 reference model。

---

### 6.4 与 influence-based data selection 的区别

Influence-based 方法通常选择样本或估计样本对目标分布的影响，成本可能较高。本路线可以把 influence-based selection 的结果作为 teacher，但最终部署时不再计算 influence。

区别是：

- influence 方法负责提供昂贵监督；
- mask head 学会近似该监督；
- 新脏数据上只用 LLM 自身一次 forward 生成 mask。

---

## 7. 核心 claim

### Claim 1：低质量数据鲁棒性

在含噪 SFT 数据上，普通 SFT 会拟合噪声 token；本方法通过 mask 降低噪声 token 的 loss 权重，从而提升对低质量数据的鲁棒性。

实验应证明：

\[
Masked\ SFT > Noisy\ SFT
\]

并尽量接近：

\[
Teacher\text{-}cleaned\ SFT
\]

---

### Claim 2：部署阶段效率

外部清洗器只用于训练 mask head。对新脏数据进行 SFT 时，不需要：

- 外部 cleaner；
- reference model forward；
- validation gradient；
- influence estimation；
- LLM judge 离线打分；
- 预先生成 cleaned dataset。

每个 batch 的 mask 在正常 SFT forward 中产生。

---

### Claim 3：清洗能力可迁移

在一批 teacher-labeled 数据上训练 mask head 后，模型应能在未见数据源、未见噪声类型或未见任务上产生有效 mask。

实验应证明：

\[
Teacher\ labels\ on\ source\ data
\rightarrow
Useful\ masks\ on\ unseen\ dirty\ data
\]

---

## 8. 不应过度声称的内容

这条简化路线不应声称：

- 模型真正理解了“自己当前最需要学什么”；
- mask 会随着主模型训练自动变得更强；
- 它能识别任意未知噪声；
- 它优于所有外部清洗器；
- 它完全没有额外计算开销。

更稳妥的表述是：

> 我们把外部清洗器的 token-level 清洗能力蒸馏成一个轻量 downstream head，使模型在部署到新低质量 SFT 数据时，可以复用同一次 training forward 生成 mask，并用 masked loss 提升鲁棒性，从而避免在新数据上重复运行昂贵清洗流程。

---

## 9. 实验设计

### 9.1 数据设置

需要至少三类数据：

1. **Cleaning-train 数据**：用于运行外部 cleaner，训练 mask head；
2. **Dirty-SFT 数据**：部署阶段的新低质量训练数据，不运行外部 cleaner；
3. **Evaluation benchmarks**：评估下游能力，例如数学、代码、指令遵循、事实问答等。

最好设置 seen / unseen split：

- seen corruption：mask head 训练时见过的噪声类型；
- unseen corruption：mask head 未见过的噪声类型；
- seen source：来自相似数据源；
- unseen source：来自不同数据源。

---

### 9.2 对比方法

必须包含以下 baseline：

1. **Noisy SFT**：直接在低质量数据上训练；
2. **Teacher-cleaned SFT**：在目标数据上运行外部 cleaner，上界；
3. **Student single-pass mask SFT**：本方法；
4. **Static student mask**：先用 student 给目标数据生成一次 mask，然后固定训练；
5. **Random mask**：控制 token 数量；
6. **Loss-only mask**：按 CE loss 或 entropy 选择；
7. **Clean data SFT**：若有 clean 版本，可作为上界；
8. **Sample-level filtering**：只过滤整条样本，而不是 token-level mask。

其中最重要的比较是：

\[
Student\ single\text{-}pass\ mask\ SFT
\quad vs. \quad
Teacher\text{-}cleaned\ SFT
\]

和：

\[
Student\ single\text{-}pass\ mask\ SFT
\quad vs. \quad
Noisy\ SFT
\]

---

### 9.3 效率评估

必须报告完整成本，而不仅是最终性能。

建议报告：

- 外部 teacher cleaner 在目标数据上运行一次的 GPU hours；
- 训练 mask head 的成本；
- 部署阶段每 step throughput；
- mask head 带来的额外显存；
- mask head 带来的额外时间；
- 当部署到多个新脏数据集时的 amortized cost；
- 达到相同下游分数所需总训练成本。

目标是证明：

> 当需要处理多个新低质量数据集时，预训练一次 mask head 的成本可以被摊销；部署阶段避免重复运行外部 cleaner。

---

### 9.4 Mask 质量评估

如果有 teacher label，可以直接评估：

- token-level AUROC；
- token-level F1；
- rank correlation；
- precision@selected budget；
- mask density；
- 不同 token 类型上的保留比例。

但最终更重要的是下游性能：

- masked SFT 后 benchmark 分数；
- 对噪声比例的鲁棒曲线；
- 对 unseen corruption 的泛化；
- 与 teacher-cleaned SFT 的性能差距。

---

## 10. 关键消融

### 10.1 mask head 是否真的有用

比较：

- no mask；
- random mask；
- CE loss threshold；
- entropy threshold；
- mask head prediction。

如果 mask head 只和 CE loss threshold 差不多，说明贡献较弱。

---

### 10.2 是否需要 LLM hidden state

比较：

- token embedding + MLP；
- hidden state + MLP；
- hidden state + loss / entropy features；
- external classifier。

如果 hidden state 版本明显更好，可以说明清洗能力确实复用了 LLM 表示。

---

### 10.3 dynamic mask vs static mask

比较：

- 训练前对目标数据一次性生成 mask，之后固定；
- 每个 batch forward 时实时生成 mask。

若二者性能接近，则方法更像 offline student cleaner。若实时生成更好，则说明 single-pass online mask 有额外价值。

---

### 10.4 单 teacher vs 多 teacher

比较：

- 单一 cleaner teacher；
- 多 cleaner ensemble soft label；
- hard label；
- soft label。

多 teacher 若更稳定，说明 mask head 学到的是更通用的 token quality，而不是某个 cleaner 的偏见。

---

## 11. 潜在风险

### 风险 1：创新性被认为只是 teacher distillation

应对：强调部署阶段避免外部清洗成本，以及在新脏数据上的 single-pass masked SFT。

---

### 风险 2：teacher 已经跑过目标数据，效率 claim 不成立

应对：必须采用 meta-training / deployment split。teacher 只在 cleaning-train 数据上运行；目标 dirty-SFT 数据上不运行 teacher。

---

### 风险 3：mask head 只学到 teacher 的表面偏差

应对：使用多源数据、多种 teacher、多种 unseen corruption，并评估跨数据源泛化。

---

### 风险 4：mask head 随 SFT 过程中 representation drift 失效

应对：简化版本中可以先冻结 base LLM 或采用低学习率/LoRA SFT；也可以定期评估 mask quality。该路线不把“mask 随模型能力增强”作为核心 claim，因此 drift 是工程风险，不是主理论 claim。

---

### 风险 5：性能提升来自减少 token 数，而不是清洗

应对：加入 random mask 和 loss-only mask，并控制相同 mask density。

---

## 12. 推荐论文叙事

### 标题候选

- Single-Pass Self-Cleaning for Robust Supervised Fine-Tuning
- Amortized Token Cleaning for Efficient Robust SFT
- Distilling Token Cleaning into LLM Training Forward Passes
- Self-Cleaning SFT: Learning to Mask Noisy Tokens without External Cleaners

### 摘要主线

1. 低质量 SFT 数据会损害模型性能；
2. token-level cleaning 有效，但外部清洗器成本高；
3. 我们把外部清洗结果蒸馏成一个轻量 mask head；
4. 部署到新脏数据时，模型在正常 SFT forward 中直接生成 mask；
5. masked loss 提升低质量数据鲁棒性，同时避免目标数据上的外部清洗成本。

### 最稳妥的贡献表述

- 提出一个把 token-level cleaning 摊销进 LLM forward 的简单框架；
- 训练一个 downstream mask head，使 LLM 在 SFT 时直接预测 token quality；
- 在未见低质量数据上实现无外部 cleaner 的 masked SFT；
- 系统评估性能、效率和清洗质量。

---

## 13. 最小可行实验版本

如果要快速验证，建议做最小版本：

1. 选择一个数学 SFT 数据集作为 cleaning-train；
2. 用外部 cleaner 或 LLM judge 给 answer token/span 打 clean/noisy 标签；
3. 冻结 LLM，只训练 hidden-state mask head；
4. 在另一个未见数学 dirty dataset 上，不运行 cleaner，直接用 mask head 做 masked SFT；
5. 比较 noisy SFT、teacher-cleaned SFT、student-mask SFT、random mask、loss-only mask；
6. 报告性能和计算成本。

如果这个版本能接近 teacher-cleaned SFT，并显著超过 noisy SFT，就说明路线成立。

---

## 14. 当前路线的边界

这是一条以效率为核心的实用路线。它的创新点不是“发明最好的清洗标准”，而是：

> 将已有清洗能力蒸馏为模型内部的 downstream mask prediction 能力，使清洗从昂贵外部预处理变成 SFT forward 中的轻量附加输出。

因此主 claim 应该保持克制：

- 可以 claim 高效；
- 可以 claim 对低质量数据更鲁棒；
- 可以 claim 避免新脏数据上的外部清洗成本；
- 可以 claim token-level cleaning 能力可迁移；
- 不应 claim 完全自主理解学习意图；
- 不应 claim 替代所有外部清洗器；
- 不应 claim 零额外计算。

---

## 15. 2026-06-23 实现更新：从外部 cleaner 标签转向 validation-distilled teacher

当前仓库实现已经验证了一条比“外部 cleaner 先标答案 token”更贴近主项目的 teacher 路线：

> teacher 不是静态判断 token 干净与否，而是选择能通过一次虚拟 LoRA update 改善 held-out final answer loss 的 support token。

也就是说，teacher signal 来自 validation improvement，而不是来自人工规则或 LLM judge 标签。

### 当前最有效的 teacher 设置

现在推荐的 teacher sanity 设置是 final-answer-only validation：

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

注意这里有一个重要区别：

* target / validation loss 只看 final answer；
* support side 仍允许 teacher 在所有 completion token 中选择信号。

因此它不是“只训练答案 token”，而是“用 final answer 作为 teacher 的验证目标”。

### 正式 sanity 结果

4GPU 正式 run：

* config: `configs/claim1_teacher_sanity_final_answer_only_qwen_v103.yaml`
* output: `outputs/claim1_teacher_sanity_final_answer_only_qwen05b_v103`
* steps: 40
* target_batch_size: 256 / rank
* effective target examples: 1024 / step

结果：

```text
teacher_gain_mean                 0.2923
top_loss_gain_mean                0.1479
random_gain_mean                  0.0787
full_gain_mean                    0.0869

teacher_minus_top_loss_gain_mean  0.1444
teacher_win_top_loss_mean         0.7188
teacher_win_random_mean           0.8000
```

和上一版 solver-sharp answer-window teacher 对比：

```text
v102 answer-window:
  teacher_gain_mean              0.3357
  teacher_minus_top_loss_mean    0.1064
  teacher_win_top_loss_mean      0.4813

v103 final-answer-only:
  teacher_gain_mean              0.2923
  teacher_minus_top_loss_mean    0.1444
  teacher_win_top_loss_mean      0.7188
```

结论是：final-answer-only 的 teacher gain 绝对值略小，但相对 top-loss 更稳定，作为蒸馏目标更有希望。

### 对 self-cleaning 叙事的影响

原 memo 中的外部 cleaner / LLM judge 可以保留为 baseline 或替代 teacher source，但当前主线应改成：

1. 在 meta-training 阶段，用 validation-distilled teacher 产生 token mask；
2. 训练 mask head 预测该 teacher mask；
3. deployment 到新 dirty SFT 数据时，不再使用 validation teacher，只用 student mask head；
4. 对比 random mask、top-loss mask、full SFT 和 teacher mask 上限。

这样叙事更强，因为 teacher 不是简单模仿外部清洗器，而是直接由 held-out final-answer improvement 定义。

### 当前限制

final-answer-only validation 不等于 teacher mask 只选择 final answer token。v103 中 teacher mask mass 约为：

```text
answer    15.5%
punct     37.6%
number    24.2%
operator   9.5%
word       8.8%
```

所以当前应避免把方法描述为“答案 token 清洗器”。更准确的说法是：

> 以 final-answer validation 为目标，蒸馏一种 support-token learning signal selector。

下一步应先跑 final-answer-only teacher 的短 Stage 1 teacher-student，验证 student 是否能学到这个更稳定的 teacher。

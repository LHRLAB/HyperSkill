# HyperSkill: Training-Free GRPO with Self-Evolving Omni-Modal Weighted Hypergraph Skill Library

---

## Abstract

大语言模型（LLM）的强化学习微调（如GRPO、PPO）通过梯度更新改变策略参数以适应新任务，但面临计算代价高、灾难性遗忘和不可解释等问题。我们提出 **HyperSkill**，一种无需训练的GRPO框架，将策略进化从参数空间转移到显式的结构化知识空间——**自进化全模态带权超图技能库**。HyperSkill冻结模型参数，通过检索增强（retrieve→augment）实现in-context策略改进。其核心创新包括三个方面：（1）**对比步骤级技能蒸馏**——将好/坏推理轨迹拆解为步骤序列并逐步对齐，定位精确的分歧点，从分歧处提取原子技能与技能组合配方；（2）**反事实验证的技能进化**——在入库前通过反事实推理验证候选技能的因果效用（"如果agent知道该技能，失败轨迹能否成功？"），并通过LLM驱动的增删改查（CRUD）持续精炼技能库；（3）**带权超图结构化技能表示**——原子技能为节点，经验证的技能组合为超边（Skill Recipe），支持物理启发的优势动态更新与优势加权的互反排序融合（RRF）检索。在涵盖文本、视觉-语言和全模态的11个数据集上的实验表明，HyperSkill在不更新任何模型参数的情况下，通过技能库的自进化实现了持续的策略改进。

---

## 1 Introduction

大语言模型（LLM）在推理、搜索和多模态理解等任务上展现出强大能力。强化学习（RL）微调——特别是组相对策略优化（GRPO）[1]——已成为进一步提升LLM推理能力的主流范式。然而，基于梯度的策略更新存在固有局限：

- **计算代价高昂**：每次迭代需要反向传播，对显存和算力要求极高；
- **灾难性遗忘**：新任务的梯度更新可能覆盖已学到的能力；
- **不可解释**：能力提升隐藏在数十亿参数的微小变化中，难以追踪和调试；
- **模态耦合**：多模态能力难以在统一的参数空间中解耦演化。

近期工作开始探索无需梯度更新的替代路线。TrainingFreeGRPO [2] 提出用经验池（experience pool）替代参数更新，通过对比好/坏轨迹提取可复用经验并注入后续推理。HyperEvo [3] 进一步引入带权超图存储经验记忆，通过物理启发的优势更新机制实现知识的动态演化。然而，现有方法仍存在以下不足：

1. **提取粒度粗糙**：TrainingFreeGRPO对整条轨迹进行对比，无法定位到具体推理步骤的分歧点，导致提取的经验过于笼统；
2. **缺乏质量验证**：提取的技能直接入库，没有验证其是否真正具有因果效用，可能引入噪声技能；
3. **结构表达力不足**：经验以扁平文本或简单实体关系存储，无法表达多个技能必须**协同使用**才有效的组合关系。

为解决上述问题，我们提出 **HyperSkill**，一种基于自进化全模态带权超图技能库的无训练GRPO框架。HyperSkill的核心思想是：**策略模型不必更新参数就能持续进步——只需要更好的结构化推理技能**。具体而言，HyperSkill冻结LLM参数，将策略的"进化"体现为超图中技能节点权重的更新、新技能的插入、过时技能的删除，以及技能组合配方（Skill Recipe）的发现。

本文的主要贡献如下：

- **对比步骤级技能蒸馏（Contrastive Step-Level Skill Distillation）**：将推理轨迹分解为步骤序列，对齐好/坏轨迹的对应步骤，在精确的分歧点处提取原子技能与技能组合配方——相比整条轨迹对比，提取的技能更精准、更可归因。
- **反事实验证的技能进化（Counterfactual Grounded Skill Evolution）**：在候选技能入库前，通过反事实推理验证其因果效用；结合LLM驱动的CRUD操作（增/删/改/查），实现技能库的持续精炼——而非仅靠数值阈值剪枝。
- **带权超图结构化技能表示（Weighted Hypergraph-Structured Skill Representation）**：原子技能为节点，经验证的技能组合为超边，通过物理启发的优势动态与优势加权RRF检索，实现对技能组合关系的显式建模与高效利用。

---

## 2 Related Work

### 2.1 强化学习与LLM推理

GRPO [1] 通过组内相对优势归一化消除了value model的需求，成为LLM推理增强的主流方法。DeepSeek-R1 [4] 和 Qwen3 [5] 等模型通过大规模GRPO训练显著提升了推理能力。然而，这些方法依赖梯度更新，面临计算代价和灾难性遗忘问题。Search-R1 [6] 将搜索工具引入GRPO训练，使模型学会何时以及如何搜索。

### 2.2 无训练策略优化

TrainingFreeGRPO [2] 首次提出无需梯度更新的GRPO框架，通过对比好/坏轨迹提取经验，以LLM驱动的CRUD操作维护经验池，将经验注入后续推理实现策略改进。该方法证明了经验库可以替代参数更新实现策略进化，但其提取粒度为整条轨迹级，且经验以扁平字典存储，缺乏结构化表示。EvoSkill [7] 通过agent执行trace分析提出技能创建/编辑操作，采用frontier管理维护最优技能集合，但同样缺乏对技能组合关系的建模。

### 2.3 超图与知识表示

HyperGraphRAG [8] 提出用超图替代传统知识图谱进行检索增强生成，通过n-ary关系建模捕获多实体间的复杂语义关联，证明了超图在知识表示中的优越性。HyperEvo [3] 将带权超图引入强化学习经验存储，通过物理启发的优势更新机制实现知识节点权重的动态演化，但其节点为原始实体而非抽象技能，且不具备技能组合的显式表达。KnowledgeClaw [9] 实现了全模态知识摄入管道，支持文本、图像、PDF、音频、视频的统一向量化，为全模态技能库提供了基础设施支撑。

### 2.4 技能提取与组合

已有多项工作探索从agent轨迹中提取可复用技能。Voyager [10] 在Minecraft中通过LLM自动生成和验证技能代码。JARVIS-1 [11] 维护多模态记忆用于技能规划。然而，这些方法主要面向特定环境（游戏、代码），且技能之间的组合关系缺乏显式建模。HyperSkill通过超边结构自然表达"技能配方"——多个原子技能必须协同使用才有效的组合模式。

---

## 3 Preliminaries

### 3.1 组相对策略优化（GRPO）

给定问题 $q$ 和参考答案 $a^*$，GRPO从当前策略 $\pi_\theta$ 采样 $n$ 条轨迹 $\{o_i\}_{i=1}^n$，计算组内相对优势：

$$\hat{A}_i = \text{tanh}\!\left(\frac{r_i - \bar{r}}{\sigma_r + \epsilon}\right), \quad \bar{r} = \frac{1}{n}\sum_{j=1}^n r_j, \quad \sigma_r = \sqrt{\frac{1}{n}\sum_{j=1}^n (r_j - \bar{r})^2}$$

其中 $r_i$ 为轨迹 $o_i$ 的奖励。策略通过最大化以下目标更新参数：

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n \hat{A}_i \cdot \log \pi_\theta(o_i \mid q)\right]$$

### 3.2 超图

超图 $\mathcal{H} = (\mathcal{V}, \mathcal{E}, w)$ 是图的推广，其中超边 $e \in \mathcal{E}$ 可以连接**任意数量**的节点，而非仅限于两个。形式化地，每条超边 $e_j \subseteq \mathcal{V}$，$|e_j| \geq 2$。带权超图为每个节点和超边赋予权重 $w: \mathcal{V} \cup \mathcal{E} \to \mathbb{R}$。

超图相比普通图的优势在于能够自然表达**多元关系**（n-ary relations）。在本文场景中，一条超边表示"多个技能必须协同使用才能解决某类问题"——这种组合关系无法用普通图的二元边表达。

### 3.3 问题定义

给定一个冻结的LLM策略 $\pi_\theta$（参数不更新）和训练数据集 $\mathcal{D} = \{(q_k, a_k^*)\}$，我们的目标是构建一个自进化的技能超图 $\mathcal{H}$，使得技能增强后的策略性能持续提升：

$$\pi_\theta(o \mid q) \;\longrightarrow\; \pi_\theta\!\left(o \;\middle|\; q,\, \mathcal{R}(q, \mathcal{H})\right)$$

其中 $\mathcal{R}(q, \mathcal{H})$ 是从超图 $\mathcal{H}$ 中检索到的与问题 $q$ 最相关的技能集合。策略的"进化"体现为超图的更新 $\Delta\mathcal{H}$——新技能的插入、旧技能的删除/精炼、技能组合配方（超边）的发现——而非参数的梯度步 $\Delta\theta$。

---

## 4 Method: HyperSkill

HyperSkill的方法框架由三个核心模块组成：（1）对比步骤级技能蒸馏用于从推理轨迹中精准提取技能；（2）反事实验证用于保障入库技能的因果效用；（3）检索反馈驱动的带权超图进化用于技能的存储、利用和自适应淘汰。图1展示了完整框架。

### 4.1 对比步骤级技能蒸馏

现有方法（如TrainingFreeGRPO）将整条轨迹作为对比单元，提取的经验粒度粗糙。我们提出**步骤级对比蒸馏**，将提取粒度下沉到推理步骤级别。

**步骤分解**。给定一条推理轨迹 $o = (s_1, s_2, \ldots, s_T)$，我们首先将其分解为有序的推理步骤序列。分解通过规则匹配实现（零LLM开销）：识别编号步骤（"Step 1:"）、项目符号、段落分割等结构标记。

**对比对齐与分歧检测**。对于同一问题的 $n$ 条轨迹，按奖励划分为好组 $\mathcal{O}^+$（$r > \text{median}$）和坏组 $\mathcal{O}^-$（$r \leq \text{median}$）。将两组轨迹的**分解后步骤序列**输入LLM进行对齐，要求模型：

1. 找到好/坏轨迹中**解决相同子问题**的对应步骤；
2. 标注**分歧点**（⚡）——对应步骤中策略或正确性产生差异的位置；
3. 从每个分歧点提取**原子技能**——好轨迹使用但坏轨迹缺失的可泛化推理模式。

**技能组合配方发现**。当多个原子技能在成功轨迹中**共同出现**但在失败轨迹中**共同缺失**时，它们被识别为一个**技能配方（Skill Recipe）**——即必须协同使用才有效的技能组合，将作为超图中的超边。

### 4.2 反事实验证

直接将候选技能入库可能引入噪声——与成功相关但非因果的技能。我们通过**反事实推理**在入库前验证每个候选技能的因果效用。

对每个候选技能 $v$，从当前batch的失败轨迹中采样 $K$ 条（默认 $K=2$），对每条失败轨迹 $o^-$ 提问：

> "如果agent在推理时知道技能 $v$，失败轨迹 $o^-$ 是否能够成功？"

LLM给出判定 $\text{VERDICT} \in \{\text{YES}, \text{NO}\}$。仅当超过半数判定为YES时，技能通过验证并入库：

$$\text{pass}(v) = \mathbb{1}\left[\frac{1}{K}\sum_{k=1}^K \mathbb{1}[\text{verdict}_k = \text{YES}] \geq 0.5\right]$$

此步骤将技能筛选从**相关性**提升到**因果性**——只保留确实能改变失败结果的技能。通过验证的技能以初始优势 $q=0$ 插入超图，其后续权重完全由检索反馈驱动。

### 4.3 检索反馈驱动的带权超图进化

#### 4.3.1 超图结构

技能库被建模为带权超图 $\mathcal{H} = (\mathcal{V}, \mathcal{E}, w)$：

- **技能节点** $\mathcal{V}$：每个节点 $v_i$ 表示一个原子技能，包含名称、描述和累积优势权重 $q_i \in [-1, 1]$。
- **超边** $\mathcal{E}$：两种来源——（a）对比蒸馏阶段发现的**技能配方**，表示必须协同使用的技能组合；（b）检索反馈阶段自动发现的**共现模式**，表示在高奖励rollout中被同时检索的技能组合。

每个节点和超边在三个后端同步维护：键值存储（元数据）、向量存储（嵌入检索）、图存储（拓扑关系）。

#### 4.3.2 检索反馈的EMA优势更新

技能的优势权重不依赖额外的LLM调用，而是通过**检索反馈**自然演化。每个batch结束后，对被检索并注入prompt的技能，根据对应rollout的奖励更新其优势：

$$q_i \leftarrow (1 - \eta) \cdot q_i + \eta \cdot \hat{A}_i$$

其中 $\hat{A}_i$ 为该问题rollout的GRPO归一化优势，$\eta$ 为EMA学习率（默认0.1）。结果裁剪到 $[-1, 1]$。

直觉：如果一个技能被检索后对应的rollout获得高奖励，它的advantage上升（被认为有帮助）；反之下降。经过多轮训练，**真正有用的技能advantage趋向正值，无用的趋向负值**，形成自然的质量分层。

**共现超边构建**：当 $\geq 2$ 个技能在同一高奖励（$r \geq 0.5$）问题中被同时检索时，自动创建或强化一条共现超边。这使得超图的连通结构随训练自然生长，无需显式的LLM配方提取。

**定期剪枝**：每隔固定间隔，移除 $q_i < -0.5$ 的节点及其关联边，防止低质量技能积累。

#### 4.3.3 优势加权RRF检索

推理阶段，给定查询 $q$：

1. **编码**：将 $q$ 嵌入为向量 $\mathbf{v}_q$。若包含图像输入，取文本与图像嵌入的均值后L2归一化。
2. **双通道检索**：分别在技能VDB和超边VDB中检索候选，超边扩展出其连接的技能节点。
3. **优势加权RRF融合**：

$$\text{score}(v_i) = \frac{1}{k_{\text{rrf}} + \text{rank}(v_i)} \cdot \left(1 + \frac{q_i + 1}{2}\right)$$

高优势技能在排名相同时得分更高，实现检索阶段对技能质量的偏好。
4. **技能注入**：检索到的top-$N$技能格式化为自然语言描述，拼接在系统提示中。

### 4.4 Training-Free GRPO训练循环

综合上述三个模块，HyperSkill的完整训练循环如下：

对训练集的每个批次 $\mathcal{B} = \{(q_b, a_b^*)\}_{b=1}^B$：

1. **技能检索**：对每个问题 $q_b$，从超图 $\mathcal{H}$ 中检索top-$K$技能；
2. **技能增强Rollout**：将检索到的技能注入prompt，对每个问题生成 $n$ 条推理轨迹；
3. **奖励与GRPO优势**：计算组合奖励，归一化为GRPO优势；
4. **对比蒸馏 + 反事实验证**（替代梯度更新）：
   - 步骤分解 → 对比对齐 → 分歧点提取原子技能与配方
   - 反事实验证 → 过滤非因果技能 → 入库
5. **检索反馈进化**：
   - EMA更新被检索技能的优势权重
   - 高奖励共现 → 自动创建超边
   - 定期剪枝低优势节点

与标准GRPO的关键区别：**第4-5步用超图更新 $\Delta\mathcal{H}$ 替代参数更新 $\Delta\theta$**。技能从对比分歧中精准提取、经反事实验证后入库、通过检索反馈自然演化——整个过程无需梯度计算。

---

## References

[1] Shao, Z., et al. "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models." 2024.

[2] Wang, R., et al. "Training-Free GRPO: Group Relative Policy Optimization without Training." 2025.

[3] Luo, H., et al. "HyperEvo: Multimodal Reasoning with Dynamic Hypergraph Memory." 2025.

[4] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025.

[5] Qwen Team. "Qwen3 Technical Report." 2025.

[6] Jin, B., et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." 2025.

[7] Zhang, Y., et al. "EvoSkill: Self-Improving Agents through Skill Evolution." 2025.

[8] Luo, H., et al. "HyperGraphRAG: Hypergraph-Driven Retrieval-Augmented Generation." 2025.

[9] Luo, H., et al. "KnowledgeClaw: Omni-Modal Knowledge Graph Construction." 2025.

[10] Wang, G., et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models." 2023.

[11] Wang, Z., et al. "JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models." 2023.

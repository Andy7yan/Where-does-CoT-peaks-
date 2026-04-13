# Stage 1 实验规格文档（v3）

> **变更说明（v2 → v3）**：本版本基于代码审计结果修订，将已实现但未文档化的设计决策正式写入 spec。主要变更：corruption 改为两层 fallback（§3.7）、per-group temperature 梯度（§5.1）、shard 目录布局（§7.2）、实现细节备案（§5.6）、配置参数表对齐实际代码（§8.1）。

---

## §1 研究问题与假说

### 1.1 研究问题

在同一模型、同一数据集上同时测量 accuracy-optimal CoT length（L*）和 NLDD reasoning horizon（k*）时，二者的数值关系是什么？

背景：Optimal-length 论文从行为层面报告 accuracy 对 CoT 长度存在倒 U 型关系，即存在使 accuracy 最大化的长度 L*。NLDD 论文从机制层面通过 step-level counterfactual 测量因果贡献，报告存在推理地平线 k*，超过该位置后 step 对 final answer 的因果作用趋近于零。两篇论文独立发表，未讨论对方。本实验在同一批数据上同时测量两者。

### 1.2 三个互斥假说

**H1：L* ≈ k*（一致）**  
行为拐点与机制地平线重合。超过 L* 的 step 既不改善 accuracy，也不再有因果作用。行为层面的倒 U 型完全由机制层面的 faithfulness decay 解释。

**H2：L* < k*（行为拐点早于机制地平线）**  
accuracy 已经下降，但 NLDD 显示后续 step 仍有因果贡献。含义：模型忠实地执行了额外推理，但该推理有害（overthinking / 过度纠正）。

**H3：L* > k*（行为拐点晚于机制地平线）**  
NLDD 显示 step 已无因果贡献，但 accuracy 仍在上升。含义：后段 step 通过 NLDD 测不到的间接机制起作用（格式锚定、注意力稳定等），或 NLDD 的 single-step corruption 灵敏度不足。

三个假说均有独立研究价值，无论哪个被支持都构成可报告的发现。

### 1.3 Interpretation Caveat

NLDD 测量的是 step 文本内容对 final answer logit 的因果影响，不是模型在该 step 的 token 生成过程中是否"正在思考"。高 NLDD 意味着"模型的后续预测依赖了这段文本"，不直接等价于"模型在这个 step 做了计算"。这一区分在对照 Think Dot by Dot 等研究时尤其重要——CoT token 的存在与实际内部计算不一定同步。

### 1.4 Per-Group Temperature Caveat

本实验的 ICL 组使用不同的采样温度（§5.1），这意味着不同长度桶的 trace 来源于不同的温度设置。这不影响 L* 和 k* 的测量：L* 来自 accuracy 按 actual_num_steps 分桶后的均值曲线，temperature 仅增加桶内方差而非引入系统偏差；k* 的 NLDD forward pass 是 deterministic 的（greedy logit 读取），不受 generation temperature 影响。温度差异最多通过"低温 trace 在同一长度下质量更高"这一间接通道影响桶内 accuracy 的方差，但不会系统性移动 L* 位置。若后续分析需要控制此变量，可按 prompt_id 分层做 robustness check。

---

## §2 范围与非目标

### 2.1 实验定位

Stage 1 = 单模型 + 单数据集上的最小可行闭环。目标是验证"在同一批数据上同时测量 L* 和 k* 并比较"这条技术路线是否可行，产出可解读的初步结论。

### 2.2 范围

- 模型：meta-llama/Llama-3.1-8B-Instruct
- 数据集：GSM8K test split（完整 1319 题）
- 精度：float16
- 产出：accuracy(L) 曲线及 L* 估计、NLDD profile 及 k* 估计、L* vs k* 的直接比较、7 张图

### 2.3 非目标

- 不做多模型对比。Stage 2 才引入 Qwen2.5-7B-Instruct 或更强模型。
- 不做 GSM8K 以外的数据集。MATH（all levels）推迟到 Stage 2。
- 不使用 incorrect trace 做 NLDD 分析。incorrect trace 只贡献 accuracy 统计。
- 不做 multi-step corruption。每次只改一步。
- 不做 RSA / probing 等解释性分析。
- 不做 Compression Theory 层次 B（H1/H2/H3 机制解释）和层次 C（Φ 拟合）。这些在 Stage 2 数据上展开。
- 不做硬控长度指令（如 "Use exactly N steps"）。长度变异完全通过 ICL exemplar 风格差异和 per-group temperature 梯度实现。
- 不做 perplexity filtering（OQ-3 决定暂不实施）。

### 2.4 Stage 2 扩展方向

- 模型维度：保留 Llama-3.1-8B，追加 Qwen2.5-7B-Instruct 或更强模型。
- 数据集维度：MATH（all levels），需新的 step parser（句号切分）和 answer 抽取（\boxed{} 格式）。
- 分析维度：Compression Theory 层次 B/C、题目难度的预设分层、multi-step corruption。
- 分析维度：Step Complexity Proxy（Compression Theory 层次 A）：规则 + LLM judge 混合标注每个 step 的结构特征（引用前序结果数量、操作类型、是否引入新中间量）。

---

## §3 统一定义

以下定义在整个 Stage 1 中保持一致，不因测量类型（accuracy / NLDD）而改变。

### 3.1 Step 定义

一个 step = CoT 文本中被换行符分割的一个非空文本段。

切分规则，按顺序执行：

1. 以 `\n` 分割原始 completion text。
2. strip 每段首尾空白。
3. 丢弃空字符串。
4. 丢弃纯标点或纯空白段。
5. 最终答案行（包含 `####` 或 `The answer is` 的行）不计入 step，单独记录为 `final_answer_line`。

`actual_num_steps` = 上述过程产出的 step 列表长度。

此定义同时用于：accuracy 曲线的横轴分桶、NLDD corruption 的 step index、自然长度分布的统计。

### 3.2 答案抽取

GSM8K 的 gold answer 是一个数值。抽取规则按优先级：

1. 在 completion 中查找 `####` 后的内容，取第一个匹配到的数值。
2. 若无 `####`，查找 `The answer is` 后的数值。
3. 若仍未匹配，标记 `extraction_failed = True`。

数值标准化：去除逗号、美元符号、百分号，转为 float。判等条件：`abs(extracted - gold) < 1e-3`。

### 3.3 Global Calibration Constant S

为实现跨模型可比性，NLDD 使用一个全局归一化常数 S，在 clean reasoning traces 上标定：

$$
S = \frac{1}{M} \sum_{m=1}^{M} \sigma(z_m)
$$

其中，$z_m$ 是第 $m$ 条 clean trace 的 final-token logit 向量，$\sigma(\cdot)$ 计算该向量在整个 vocabulary 上的标准差。$M$ = 参与标定的 clean trace 总数。

S 反映的是模型固有的 output variability，而非绝对 logit 量级。Stage 1 中 S 在全量 clean traces 上计算一次，所有 NLDD 测量共享同一个 S。

### 3.4 Logit Difference（LD）

对一个给定 prompt，置信度定义为标准化 margin：

$$
LD = \frac{\max_{y \in Y_{correct}} \ell(y) - \max_{y' \in Y \setminus Y_{correct}} \ell(y')}{S}
$$

其中，$Y_{correct}$ 包含正确答案的所有合法 token ID（需考虑 tokenization 变体，如 leading space）。对 GSM8K 的多 token 答案，使用 first-token margin 作为稳定代理。

此定义在 clean 和 corrupt 条件下保持一致。

### 3.5 NLDD 计算

对一条长度为 $L$ 的 clean trace，corruption position $k \in \{1, \ldots, L\}$：

$$
NLDD(k) = \frac{LD_{clean} - LD_{corrupt,k}}{|LD_{clean}|} \times 100
$$

- `LD_clean`：完整 clean trace 喂入模型后的 LD 值。
- `LD_corrupt_k`：将第 $k$ 步替换为 corrupt 版本、截断第 $k+1$ 步及之后所有内容后的 LD 值。
- 排除条件：`|LD_clean| < ε`（`ε = 1e-6`）的样本不参与 NLDD 计算，避免近零基线置信度造成的噪声放大。

`NLDD > 0` 表示该 step 有因果作用（corruption 降低了答案置信度）。`NLDD ≈ 0` 表示弱耦合。`NLDD < 0` 表示 confidence reversal（corruption 反而提升了 margin）。

### 3.6 k* 定义

$k^*(L)$ = 在长度为 $L$ 的 trace 的 NLDD profile 中，NLDD 值达到峰值的 step index。

默认使用 peak-based 定义。如需切换到 steepest-decline 或 threshold-based 定义，应作为配置参数声明，不改代码逻辑。

### 3.7 Corruption 方式

Stage 1 使用两层 corruption fallback：Tier 1 arithmetic perturbation 和 Tier 2 operator swap。

#### Tier 1：Arithmetic Perturbation

1. 用正则定位目标 step 中的所有数值表达式。
2. **优先选择等号右侧的数值**（即计算结果），其次选择 operand。在同一优先级内随机选择。
3. 判断该数值是否为整数（整数判定：`value.is_integer()` 且原始文本中不含小数点；形如 `"10.0"` 按非整数处理）：
   - 整数：替换为 `original + delta`，其中 `delta ∈ {-2, -1, +1, +2}`，随机选择。若 `original = 0`，`delta ∈ {+1, +2}`。
   - 非整数：在 `[FLOAT_RANGE_LOW_MIN, FLOAT_RANGE_LOW_MAX]` 和 `[FLOAT_RANGE_HIGH_MIN, FLOAT_RANGE_HIGH_MAX]` 两个区间上各 50% 概率采样 multiplier，替换为 `original × multiplier`。默认区间 `[0.5, 0.9]` 和 `[1.1, 1.5]`。
4. 若该 step 无可替换数值，进入 Tier 2。
5. 质量过滤：corrupt step 与 clean step 的 token count 差异 ≤ `CORRUPTION_TOKEN_DELTA_MAX`。不满足的重新采样。每个候选数值最多重试 `CORRUPTION_RETRY_LIMIT` 次；遍历所有候选均失败后进入 Tier 2。

#### Tier 2：Operator Swap

若 Tier 1 失败，尝试将 step 中的算术运算符互换（如 `+` ↔ `-`，`×`/`*` ↔ `÷`/`/`），同样受 token count delta 约束。

#### Tier 3：Semantic Flip（默认关闭）

代码中保留反义词替换逻辑（如 `"more"` ↔ `"less"`），但默认不启用。可通过配置项 `ENABLE_TIER3_SEMANTIC_FLIP` 开启。Stage 1 不使用 Tier 3。

#### 失败处理

若 Tier 1 和 Tier 2 均失败，标记 `corruption_failed = True`，跳过该位点。

#### 输出记录

每条 corruption 记录包含 `corruption_tier`（1 或 2）和 `corruption_type`（`numeric_result` / `numeric_operand` / `operator_swap` / `uncorruptible`），用于下游分析时按 tier 分层。

排除条件：几何类题目不做 corruption（GSM8K 几乎不含几何，此条为 Stage 2 MATH 预留）。

### 3.8 难度定义

Post-hoc 定义，不预设难度分层：

$$
difficulty(q) = 1 - accuracy(q)
$$

其中，$accuracy(q)$ = 题目 $q$ 在全部采样中的 correct rate。

难度值在 Data Phase 全部 trace 产出后计算，附加到每题 metadata。

---

## §4 Pilot Run

Pilot Run 在正式 Data Phase 和 Analysis Phase 之前执行，目的是用低成本验证端到端流程的可行性，暴露设计假设与现实的偏差，为正式 run 的参数选择提供依据。

### 4.1 范围

从 GSM8K test split 中随机抽取 50–100 题（与正式 run 的题目可重叠，Pilot 数据不计入正式分析）。使用与正式 run 相同的 ICL exemplar 组、相同的采样参数。每题采样数可缩减，以节省成本。

Pilot 使用与正式 run 同一份配置文件，通过 `pilot.*` override 块覆盖 generation 参数（如 `pilot.samples_per_group`）。不需要单独的 config 文件。

### 4.2 必须验证的项目

#### A. 长度引导效果

ICL exemplar 是否确实产生了 `actual_num_steps` 的分布差异。具体检查：

- 各组的 `actual_num_steps` 中位数是否有可辨别的梯度。
- 合并后的 `actual_num_steps` 全距是否足够宽，以支撑 `accuracy(L)` 曲线拟合。
- 若分布高度重叠，说明 ICL 引导无效，需重新设计 exemplar 或考虑追加辅助手段。

#### B. 每个长度桶的样本量

按 `actual_num_steps` 分桶后，检查每桶的 trace 数量和 correct trace 数量：

- 若某桶总 trace 数过低，该桶的 accuracy 估计不可靠，正式 run 可能需要增大每组采样数。
- 若某桶 correct trace 数为 0，该桶无法做 NLDD 分析，正式 run 中需在该长度区间缩小 NLDD 的覆盖范围。

#### C. Step 切分与答案抽取

- `extraction_failed` 比例是否低于阈值。
- Step 切分后 `actual_num_steps = 0` 的 trace 比例是否可忽略。
- 抽查 trace 的 step 切分结果，人工确认切分边界是否合理。

#### D. Corruption 可行性

对 Pilot traces 中的 correct traces 试跑 corruption 流程：

- `corruption_failed` 比例是否低于阈值。
- 抽查 corrupt step，确认替换后的文本仍然语法通顺、token count delta ≤ 2。
- 按 corruption_tier 统计分布，确认 Tier 1 覆盖率。

#### E. NLDD 端到端烟雾测试

挑选少量 correct trace，完整跑一遍 NLDD 流程（含 S 标定、LD 计算、NLDD 计算），确认：

- `S > 0` 且量级合理。
- `LD_clean` 大部分 > 0。
- NLDD profile 形状可解读，不是全零或全噪声。

### 4.3 决策规则

Pilot 结果直接影响正式 run 的参数。

| Pilot 发现 | 决策 |
|---|---|
| ICL 引导有效 | 正式 run 维持当前 ICL 组数与采样设置 |
| ICL 引导偏弱 | 增大每组采样数，或追加新的 exemplar 组 |
| ICL 引导无效 | 停下来重设计 exemplar，不进入正式 run |
| 某些长度桶 correct trace = 0 | 正式 run 的 NLDD 跳过这些桶 |
| `extraction_failed` 过高 | 检查 answer format 指令，修正后重跑 Pilot |
| `corruption_failed` 过高 | 放宽 corruption 规则，或对该类 step 标记跳过 |

### 4.4 Pilot 产物

- `pilot_traces.jsonl`：schema 与正式 traces 相同。
- `pilot_report.md`：记录各检查项结果和决策。

Pilot 数据不纳入正式分析，但保留用于调试。

---

## §5 Data Phase

Data Phase 的目标是生成 length-varied reasoning traces，并收集足够的 correct traces 供 Analysis Phase 使用。

### 5.1 采样策略

- 使用 `NUM_ICL_GROUPS` 组不同复杂度的 ICL exemplar。每组采 `SAMPLES_PER_GROUP` 次。每题共 `NUM_ICL_GROUPS × SAMPLES_PER_GROUP` 个样本。
- 不使用 "Use exactly N steps" 等硬控长度指令。长度变异通过两个手段实现：（a）ICL exemplar 的风格差异（简短 vs 详细解法）；（b）per-group temperature 梯度（低温倾向短链、高温倾向长链）。
- 每组 ICL exemplar 绑定一个独立的 temperature，通过 `ICL_GROUP_TEMPERATURES` 配置。
- 所有下游分析按 `actual_num_steps` 分桶，不按 ICL 组别分桶。ICL 组别和 temperature 只是产生长度变异的手段，不是分析维度。

### 5.2 ICL Exemplar 要求

每组 exemplar 必须满足：

- 解法内容正确（答案与 gold answer 一致）。
- 解法风格在 step 数上有可辨别的差异（由 Pilot Run 验证）。
- 使用 `####` 格式给出最终答案，与 §3.2 的答案抽取规则一致。
- exemplar 的题目不与 GSM8K test split 重叠。

Exemplar 的具体内容和数量在 `prompts/` 目录中定义，spec 不规定文本。

### 5.3 数据结构

#### Run-level Metadata

单次运行一份 JSON（`run_meta.json`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `run_id` | str | 本次运行唯一标识 |
| `model_name` | str | 模型标识 |
| `dataset` | str | 数据集名称及 split |
| `temperature` | float \| null | 全局 temperature（若使用 per-group 梯度则为 null） |
| `icl_group_temperatures` | dict[str, float] | 每组 ICL 的实际 temperature |
| `max_new_tokens` | int | 生成上限 |
| `num_icl_groups` | int | ICL 组数 |
| `samples_per_group` | int | 每组采样数 |
| `seed` | int | 全局随机种子 |
| `prompt_ids` | list[str] | ICL 组别 ID 列表 |
| `schema_version` | str | trace schema 版本标识 |
| `timestamp` | str | 运行启动时间 |

#### Trace 表

JSONL，每行一条 trace，只含 per-sample 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `trace_id` | str | 唯一标识 |
| `question_id` | str | 题目唯一 ID |
| `question_text` | str | 原始题目文本 |
| `gold_answer` | float | 标准答案 |
| `prompt_id` | str | ICL 组别版本 ID |
| `raw_completion` | str | 原始模型输出 |
| `steps` | list[str] | 按 §3.1 切分后的 step 列表 |
| `actual_num_steps` | int | `len(steps)` |
| `final_answer_line` | str | 答案行原文 |
| `extracted_answer` | float \| null | 按 §3.2 抽取的数值答案 |
| `is_correct` | bool | 是否匹配 |
| `extraction_failed` | bool | 抽取是否失败 |
| `token_count` | int | completion token 数 |
| `timestamp` | str | 生成时间 |

### 5.4 每题 Metadata

Data Phase 全部 trace 产出后，对每题计算：

| 字段 | 类型 | 说明 |
|---|---|---|
| `question_id` | str | 题目唯一 ID |
| `difficulty` | float | `1 − accuracy(q)`，见 §3.8 |
| `accuracy` | float | 该题全部采样的 correct rate |
| `total_samples` | int | 该题总采样数 |
| `correct_count` | int | correct trace 数 |
| `optimal_length` | int \| null | accuracy 峰值对应的 `actual_num_steps`（多峰时取最小） |
| `natural_length_distribution` | dict | `{actual_num_steps: count}` 分布 |

### 5.5 Data Phase 完成条件

- 每题采样数 = `NUM_ICL_GROUPS × SAMPLES_PER_GROUP`，无遗漏。
- `extraction_failed` 全局比例 < `MAX_EXTRACTION_FAIL_RATE`（超标时打印警告，不 hard fail）。
- `actual_num_steps` 的全局分布覆盖了 Pilot Run 确认的有效范围。
- 每题 metadata 已计算并持久化。
- Data Phase 结束后即可独立计算 `accuracy(L)` 曲线，不需要等 Analysis Phase。

### 5.6 Implementation Details（备案）

以下实现细节不属于实验设计，但对可复现性有影响，故记录于此：

- **Prompt 注入方式**：ICL exemplar 以 chat 格式注入——system message + 每个 few-shot 转为 (user question, assistant answer) 消息对 + 最终 user 题目，通过 `apply_chat_template` 传入。
- **Tokenizer 配置**：若模型无 pad_token，使用 eos_token 作为 pad_token；padding_side = "left"。此为 Llama-3.1-Instruct 的标准做法，换模型时可能需要调整。
- **Batched generation**：同一 prompt 的多 sample 打成 batch 调用 `model.generate`，batch size 通过 `GENERATION_BATCH_SIZE` 配置（默认 4）。
- **Shard 并行**：Stage B 可通过 `--start-idx` / `--end-idx` 分片并行运行，产出 shard 子目录（`<run_dir>/shards/<shard_id>/traces.jsonl`）。Stage E 负责合并 shard 为根级 `traces.jsonl`。

---

## §6 Analysis Phase

Analysis Phase 消费 Data Phase 的产物（traces + 每题 metadata），产出 L* 估计、NLDD profile、TAS profile、k* 估计，以及可视化结果。

### 6.1 阶段一：accuracy(L) → L*

从全量 traces 按 `actual_num_steps` 分桶，计算每桶的 accuracy mean 和 standard error。

分桶规则：

- 桶宽 = 1 step（每个不同的 `actual_num_steps` 值为一桶）。
- 若某桶样本量 < `MIN_BIN_SIZE`，与相邻桶合并（优先向样本量更大的邻桶合并）。合并后的桶标签取所含 step 数的中位数。
- 合并仅用于展示和 L* 定位，原始 per-trace 数据不受影响。
- L* = accuracy 峰值对应的桶标签。若存在多个并列峰值，取 step 数最小的。

此阶段只需 Data Phase 产物，不需要模型 forward pass。

### 6.2 阶段二：NLDD 与 TAS Measurement

#### 6.2.1 题目选择

题目选择由人工完成。以下策略仅供参考，不作为硬性规定：

- 按 `difficulty`（§3.8）排序后等距抽取，使难度分布尽量均匀覆盖。
- 优先选择 correct trace 数量充足、覆盖多个 `actual_num_steps` 值的题目。
- 避免选择 correct trace 集中在单一长度的题目。

选出的题目分为两组：

- **全量分析组**：对其所有 correct traces 做逐 step 完整 NLDD 和 TAS 测量。
- **抽样分析组**：每条 correct trace 做 `NUM_SPOT_CHECKS` 个随机位点的 NLDD 测量；TAS 不做 corrupt 侧抽样，仅保留 clean 侧数组。

#### 6.2.2 S 标定

在所有 NLDD/TAS 测量之前，先在本次运行的全量 correct traces 上计算全局归一化常数 S（§3.3）。S 计算一次，所有后续 LD 计算共享。

#### 6.2.3 Clean Trace 选择

对每道入选题目的每个 `actual_num_steps` 值：

- 筛选 `is_correct = True` 的 traces。
- 若无 correct trace，记录 `no_clean_trace`，跳过。
- 若有多条，选 `token_count` 最接近该长度桶中位数的一条。

#### 6.2.4 NLDD 测量流程

对选定的 clean trace（长度 `L = actual_num_steps`）：

1. 计算 `LD_clean`（§3.4）。
2. 若 `|LD_clean| < ε`，排除。
3. 对每个需测量的 corruption position `k`：构造 corrupt step（§3.7）→ 拼接截断 prefix → 计算 `LD_corrupt_k` → 计算 `NLDD(k)`。
   - 全量分析组：`k` 遍历 `{1, ..., L}`。
   - 抽样分析组：`k` 从 `{1, ..., L}` 中均匀采样 `NUM_SPOT_CHECKS` 个。

**共用 forward pass 原则：** 每次 forward pass 同时通过 forward hook 捕获中间层 hidden states，并读取 final-token logits。一次 pass 同时产出 LD 和 TAS 所需原始数据，不做两次。

**NLDD prompt 格式：** clean/corrupt prefix 使用 `"Step {i}: {text}"` 格式拼接各 step，以 `"#### "` 作为 answer suffix 结尾。此格式影响模型对 prefix 的解读，换模型或换数据集时需审查。

#### 6.2.5 TAS 测量流程

轨迹提取：对每条 trace（clean 或 corrupt），从 middle transformer layer（`⌊num_layers/2⌋`）提取每个 reasoning step 末尾 token 处的 hidden state，得到序列 `{h_0, h_1, ..., h_L}`，其中 `h_0` 对应 question 末尾 token，`h_k` 对应第 `k` 个 reasoning step 的末尾 token。

逐 step 记录两个量：

- `cumulative_displacement(k) = ‖h_k − h_0‖`
- `step_length(k) = ‖h_k − h_{k-1}‖`

从这两个数组派生：

- `running_TAS(k) = cumulative_displacement(k) / Σ_{i=1}^{k} step_length(i)`
- `displacement_increment(k) = cumulative_displacement(k) − cumulative_displacement(k−1)`
- `straightness(k) = displacement_increment(k) / step_length(k)`

其中，`straightness(k) ∈ [0, 1]`。当接近 1 时，新增 step 的位移几乎完全贡献给总位移；当接近 0 时，新增 step 在绕弯。

TAS 拐点定义：`k_TAS` = 最小的 `k`，使得对所有 `j ≥ k`，都有

$$
|straightness(j) - straightness(k)| < TAS\_PLATEAU\_THRESHOLD
$$

即 straightness 曲线进入稳定平台的起始位置。

TAS 最终标量保留为：

$$
TAS = running\_TAS(L)
$$

#### 6.2.6 数据记录 schema

##### 全量分析记录

JSONL（`nldd_full.jsonl`），每行一个 corruption 测量点：

| 字段 | 类型 | 说明 |
|---|---|---|
| `nldd_id` | str | 唯一标识 |
| `question_id` | str | 题目 ID |
| `clean_trace_id` | str | 引用 trace 表 |
| `actual_clean_length` | int | clean trace 的 `actual_num_steps` |
| `corruption_step_index` | int | `k`（从 1 开始） |
| `corruption_tier` | int | 使用的 corruption 层级（1 或 2） |
| `corruption_type` | str | `numeric_result` / `numeric_operand` / `operator_swap` / `uncorruptible` |
| `corruption_failed` | bool | 是否失败 |
| `ld_clean` | float | clean LD |
| `ld_corrupt` | float | corrupt LD |
| `nldd_value` | float | NLDD 值 |
| `clean_cumulative_disp` | list[float] | clean trace 的逐 step `cumulative_displacement`，长度 L |
| `clean_step_lengths` | list[float] | clean trace 的逐 step `step_length`，长度 L |
| `corrupt_cumulative_disp` | list[float] | corrupt trace 的逐 step `cumulative_displacement`，长度 k |
| `corrupt_step_lengths` | list[float] | corrupt trace 的逐 step `step_length`，长度 k |
| `timestamp` | str | 测量时间 |

> 注：同一条 clean trace 的 clean 侧数组在不同 `k` 行中重复。实现时可单独存储并通过 `clean_trace_id` 引用；spec 只要求数据可回溯。

##### 抽样分析记录

JSONL（`nldd_spot.jsonl`），独立文件：

| 字段 | 类型 | 说明 |
|---|---|---|
| `spot_id` | str | 唯一标识 |
| `question_id` | str | 题目 ID |
| `clean_trace_id` | str | 引用 trace 表 |
| `actual_clean_length` | int | `actual_num_steps` |
| `corruption_step_index` | int | `k` |
| `corruption_tier` | int | 使用的 corruption 层级（1 或 2） |
| `corruption_type` | str | 同全量分析 |
| `corruption_failed` | bool | 是否失败 |
| `ld_clean` | float | clean LD |
| `ld_corrupt` | float | corrupt LD |
| `nldd_value` | float | NLDD 值 |
| `clean_cumulative_disp` | list[float] | clean trace 的逐 step `cumulative_displacement`，长度 L |
| `clean_step_lengths` | list[float] | clean trace 的逐 step `step_length`，长度 L |
| `timestamp` | str | 测量时间 |

#### 6.2.7 k* 提取

对全量分析组的每道题，按 `actual_clean_length` 分组，在每组的 NLDD profile 中定位 k*（§3.6，peak-based，排除 `k = 1`）。

### 6.3 可视化

共 7 张图。每张图的输入数据保存为独立 CSV 快照。

#### 图 1：Accuracy vs CoT Length

- 横轴：`actual_num_steps`
- 纵轴：accuracy（mean ± SE）
- 标注：L* 位置（竖线 + 标签）
- 数据来源：全量 traces

#### 图 2：NLDD Surface Heatmap

- 横轴：corruption position `k`
- 纵轴：`actual_clean_length = L`
- 颜色：mean NLDD
- 叠加：`k*(L)` 折线
- 数据来源：全量分析记录

#### 图 3：k*(L) vs L

- 横轴：`L`（`actual_clean_length`）
- 纵轴：`k*`
- 叠加：`L*` 竖线、`y = x` 对角线参考线
- 数据来源：全量分析记录聚合

#### 图 4：Mean NLDD vs Relative Position

- 横轴：`k / L`
- 纵轴：mean NLDD
- 分色：不同 `L` 值各一条线
- 数据来源：全量分析记录

#### 图 5：TAS vs Corruption Position

- 横轴：corruption position `k`
- 纵轴：TAS
- 分面 / 分色：按 `actual_clean_length` 分组
- 数据来源：全量分析记录

#### 图 6：Clean TAS vs CoT Length

- 横轴：`actual_num_steps`
- 纵轴：`TAS_clean`（mean ± SE）
- 标注：`L*` 位置（竖线 + 标签）
- 数据来源：全量分析组与抽样分析组的 clean traces

#### 图 7：TAS Inflection Distribution

- 横轴：`k_TAS / L` 相对位置
- 纵轴：频次直方图
- 叠加：`L* / L` 与 `k* / L` 竖线
- 数据来源：全量分析组

---

## §7 数据流与依赖

本章定义各阶段的执行顺序、输入输出依赖，以及阶段间的交接条件。不规定具体实现方式或运行环境。

### 7.1 阶段拓扑

| Stage | 名称 | 输入 | 输出 | 前置依赖 |
|---|---|---|---|---|
| A | 评测子集准备 | GSM8K test split | `eval_subset.jsonl`, `eval_subset_meta.json` | 无 |
| B | Trace 生成 | eval subset, ICL exemplar, 模型 | shard 目录（含 `traces.jsonl`）, `run_meta.json` | A |
| C | Accuracy 聚合 | `traces.jsonl`（合并后） | `accuracy_by_length.csv`, L* 估计, 每题 metadata | B |
| D | NLDD & TAS 测量 | `traces.jsonl`, S, 人工题目选择结果 | `nldd_full.jsonl`, `nldd_spot.jsonl` | B, C |
| E | 聚合 | `nldd_full.jsonl`, `nldd_spot.jsonl`, `accuracy_by_length.csv` | `nldd_surface.csv`, `nldd_by_relative_position.csv`, `horizon_summary.csv`, `tas_inflection_summary.csv` | C, D |
| F | 绘图 | 所有聚合表 | 7 张图 + 每张图的输入数据快照 | E |

### 7.2 各 Stage 细节

#### Stage A：评测子集准备

从 GSM8K test split 中固定评测子集。固定策略：对 `question_text` 做 `sha256("{hash_seed}:{question_text}")` 排序后取前 `SUBSET_SIZE` 题。hash seed 写入 metadata，保证可复现。`question_id` 在排序后由循环索引生成（`gsm8k_{global_index:04d}`），不来源于原始数据。

产物：

- `eval_subset.jsonl`：每行含 `question_id`, `question_text`, `gold_answer`
- `eval_subset_meta.json`：`SUBSET_SIZE`（字段名 `n`）、hash seed、数据集 split
- `gsm8k_test.jsonl`：完整 test split 的导出（辅助产物，非必须）

#### Stage B：Trace 生成

对评测题集中的每题执行 §5 Data Phase。

Stage B 支持分片并行：每个 shard 写入 `<run_dir>/shards/<shard_id>/traces.jsonl` + `shard_meta.json`。分片由 `--start-idx` / `--end-idx` 控制。

产物（per shard）：

- `shards/<shard_id>/traces.jsonl`
- `shards/<shard_id>/shard_meta.json`

Stage E 启动时负责将 shard 合并为根级 `traces.jsonl` + `run_meta.json`。

Stage B 是主要计算成本。完成后即可进入 Stage C。

#### Stage C：Accuracy 聚合

从 `traces.jsonl` 按 `actual_num_steps` 分桶（§6.1），产出：

- `accuracy_by_length.csv`
- L* 估计值
- 每题 metadata（`question_metadata.jsonl`）：包含 `difficulty`、`accuracy`、`optimal_length`、`natural_length_distribution`

#### Stage D：NLDD & TAS 测量

Stage D 分为两个子阶段：

**D1：Corruption 准备**（不需要 GPU forward pass）
- 读取 traces，按全量/抽样分组筛选 correct traces
- 对每条选定 trace 的每个需测量的 step 生成 corrupt text
- 产出 `all_steps.jsonl`（全量分析组，每条 trace 所有 step）和 `sampled_steps.jsonl`（抽样组，每条 trace `NUM_SPOT_CHECKS` 个随机位点）
- 同时产出 `corruption_summary.json`（per-tier/type 的成功/失败计数）

**D2：Forward Pass + 测量**（需要 GPU）
- S 标定：在全量 correct traces 上计算 S（§3.3），持久化
- 对 D1 产出的每条记录执行 forward pass，计算 LD_clean、LD_corrupt、NLDD
- 同时通过 forward hook 提取 hidden states，计算 TAS 指标
- 产出 `nldd_full.jsonl`、`nldd_spot.jsonl`（schema 见 §6.2.6）

#### Stage E：聚合

从 Stage C 和 Stage D 的产物计算：

- `nldd_surface.csv`：按 `(L, k)` 聚合 mean NLDD（全量分析组）
- `nldd_by_relative_position.csv`：按 `k / L` 归一化后聚合
- `horizon_summary.csv`：每个 `L` 对应的 `k*`、`L*`、accuracy
- `tas_inflection_summary.csv`：每条全量分析 trace 的 `k_TAS`、`k_TAS / L`、对应的 `actual_clean_length`

Stage E 启动时还负责从 shard 合并 traces（若尚未合并）。

#### Stage F：绘图

从聚合表生成 §6.3 的 7 张图。每张图连同其输入 CSV 快照一起保存。绘图在本地运行，不需要 GPU。

### 7.3 阶段间交接条件

| 交接点 | 检查内容 |
|---|---|
| A → B | `eval_subset.jsonl` 存在，行数 = `SUBSET_SIZE` |
| B → C | shard 合并后 `traces.jsonl` 存在，行数 = `SUBSET_SIZE × NUM_ICL_GROUPS × SAMPLES_PER_GROUP` |
| B → D | 同上，且 S 已标定并持久化 |
| C → D | 每题 metadata 已产出 |
| C, D → E | `accuracy_by_length.csv`、`nldd_full.jsonl`、`nldd_spot.jsonl` 均存在 |
| E → F | 所有聚合 CSV 存在且非空 |

交接检查失败时应报错并终止，不应静默跳过。

### 7.4 Pilot Run 在拓扑中的位置

Pilot Run（§4）在 Stage A 之后、正式 Stage B 之前执行。Pilot 使用独立的评测题集，产物独立存放，不与正式 run 混合。

Pilot 的输出决定正式 run 的配置参数，参数确认后才启动正式 Stage B。

---

## §8 配置参数与验证清单

### 8.1 配置参数总表

所有实验参数集中在一个配置文件中。Pilot 与正式 Run 共用同一份 config，Pilot 通过 `pilot.*` override 块覆盖 generation 参数。

#### 实验标识

| 参数 | 说明 | 当前值 |
|---|---|---|
| `run_id` | 本次运行唯一标识 | `"peak-cot-stage1-gsm8k-llama31"` |
| `seed` | 全局随机种子 | 42 |

#### 数据集

| 参数 | 说明 | 当前值 |
|---|---|---|
| `dataset.name` | 数据集标识 | `"gsm8k"` |
| `dataset.split` | split | `"test"` |
| `dataset.subset_size` | 评测子集大小 | 1319（完整 test split） |
| `dataset.hash_seed` | hash 排序种子 | 42 |

#### 模型

| 参数 | 说明 | 当前值 |
|---|---|---|
| `model.name` | 模型标识 | `"meta-llama/Llama-3.1-8B-Instruct"` |
| `model.dtype` | 推理精度 | `"float16"` |
| `model.hf_cache` | HuggingFace 缓存路径 | 由运行环境决定 |

#### 生成

| 参数 | 说明 | 当前值 |
|---|---|---|
| `generation.num_icl_groups` | ICL exemplar 组数 | 5 |
| `generation.samples_per_group` | 每组采样数 | 3 |
| `generation.max_new_tokens` | 生成 token 上限 | 512 |
| `generation.batch_size` | generation batch size | 4 |
| `generation.icl_groups` | 各 ICL 组配置（含 temperature） | 见下表 |

##### ICL 组配置

| prompt_id | temperature | 设计意图 |
|---|---|---|
| `icl_minimal` | 0.0 | greedy，最短链 |
| `icl_short` | 0.3 | 低温短链 |
| `icl_medium` | 0.5 | 中等 |
| `icl_detailed` | 0.7 | 高温长链 |
| `icl_verbose` | 0.7 | 高温详细链 |

#### Step 切分与答案抽取

| 参数 | 说明 | 当前值 |
|---|---|---|
| `step_segmentation.method` | 切分方式 | `"newline"` |
| `answer_extraction.markers` | 答案行标记列表 | `["####", "The answer is"]` |
| `answer_extraction.numeric_tolerance` | 答案判等容差 | `1e-3` |

#### NLDD

| 参数 | 说明 | 当前值 |
|---|---|---|
| `nldd.corruption_type` | corruption 类型 | `"tiered_fallback"` |
| `nldd.enable_tier3_semantic_flip` | 是否启用 Tier 3 | `false` |
| `nldd.integer_perturbation_deltas` | 整数 corruption 候选 delta | `[-2, -1, 1, 2]` |
| `nldd.float_perturbation_range` | 非整数 multiplier 边界 | `[0.5, 0.9, 1.1, 1.5]` |
| `nldd.corruption_token_delta_max` | corrupt 与 clean 的 token count 最大差异 | 2 |
| `nldd.corruption_retry_limit` | 每个候选数值的最大重试次数 | 3 |
| `nldd.ld_epsilon` | `LD_clean` 排除阈值 | `1e-6` |
| `nldd.horizon_definition` | k* 定位方法 | `"peak"` |

#### TAS

| 参数 | 说明 | 当前值 |
|---|---|---|
| `tas.layer` | hidden state 提取层 | `"middle"` (⌊num_layers/2⌋) |
| `tas.plateau_threshold` | `straightness` 平台判定阈值 | 0.05 |

#### 分析

| 参数 | 说明 | 当前值 |
|---|---|---|
| `analysis.min_bin_size` | accuracy 分桶最小样本量 | 5 |
| `analysis.num_full_analysis_questions` | 全量分析组题目数 | 人工决定 |
| `analysis.num_spot_checks` | 抽样分析每条 trace 的随机位点数 | 3 |
| `analysis.max_extraction_fail_rate` | `extraction_failed` 容忍上限 | 0.05 |

#### Pilot Override

| 参数 | 说明 | 当前值 |
|---|---|---|
| `pilot.num_icl_groups` | Pilot ICL 组数 | 同 `generation.num_icl_groups` |
| `pilot.samples_per_group` | Pilot 每组采样数 | 可缩减 |
| `pilot.max_extraction_fail_rate` | Pilot extraction 阈值 | 同 `analysis.max_extraction_fail_rate` |

### 8.2 验证清单

#### Stage A 完成后

- `eval_subset.jsonl` 行数 = `SUBSET_SIZE`
- 每行含 `question_id`, `question_text`, `gold_answer`，无空值
- `eval_subset_meta.json` 中记录了 `SUBSET_SIZE`、`hash_seed`、数据集 split

#### Stage B 完成后

- shard 合并后 `traces.jsonl` 行数 = `SUBSET_SIZE × NUM_ICL_GROUPS × SAMPLES_PER_GROUP`
- 每个 `(question_id, prompt_id)` 组合恰好有 `SAMPLES_PER_GROUP` 条
- `extraction_failed` 全局比例 < `MAX_EXTRACTION_FAIL_RATE`（超标时警告）
- `actual_num_steps` 的全局分布覆盖了 Pilot Run 确认的有效范围
- `actual_num_steps = 0` 的 trace 比例可忽略
- `run_meta.json` 与当前 config 一致

#### Stage C 完成后

- `accuracy_by_length.csv` 非空，每行含桶标签、样本量、mean、SE
- L* 可定位（存在 accuracy 峰值）
- 每题 metadata 已产出，题目数 = `SUBSET_SIZE`
- difficulty 分布合理（不全为 0 或 1）

#### Stage D 完成后

- `S > 0` 且已持久化
- `nldd_full.jsonl` 覆盖了全量分析组的全部入选题目
- `nldd_spot.jsonl` 覆盖了抽样分析组的全部入选题目
- 全量分析中每个有 clean trace 的 `(question_id, actual_clean_length)` 组合，`k` 从 1 到 L 无遗漏（除 `corruption_failed` 外）
- `corruption_failed` 全局比例 < 15%
- `nldd_value` 分布合理（大部分 > 0，少量 ≤ 0）
- `ld_clean` 大部分 > 0
- 全量分析记录中 `clean_cumulative_disp` 和 `clean_step_lengths` 长度 = `actual_clean_length`
- 全量分析记录中 `corrupt_cumulative_disp` 和 `corrupt_step_lengths` 长度 = `corruption_step_index`

#### Stage E 完成后

- `nldd_surface.csv` 的 `(L, k)` 网格中，空缺处标记原因
- `horizon_summary.csv` 中 `k*` 值在合理范围内（`1 < k* ≤ L`）
- `tas_inflection_summary.csv` 中 `k_TAS` 值在合理范围内，`k_TAS / L ∈ (0, 1]`
- 所有聚合 CSV 可被绘图脚本正常读取

#### Stage F 完成后

- 7 张图均可渲染
- 轴标签、标注（L*、k* 竖线 / 折线）与聚合数据一致
- 每张图的输入 CSV 快照已保存

### 8.3 Open Questions

| 编号 | 问题 | 标签 | 当前默认 |
|---|---|---|---|
| OQ-1 | k* 定义是否需要从 peak-based 切换到 steepest-decline | assumption | 保持 peak-based，Stage 1 结束后评估 |
| OQ-2 | `TAS_PLATEAU_THRESHOLD` 的合理值 | TODO | 暂定 0.05，由 Pilot TAS 烟雾测试验证 |
| OQ-3 | 是否需要对 NLDD 做 perplexity filtering | decided | 暂不做 |
| OQ-4 | 抽样分析组是否需要覆盖全部剩余题目，还是只抽取一部分 | decision pending | 人工决定 |

### 8.4 Pilot 确认参数记录

以下参数由 Pilot Run 结果确认。记录 Pilot Run 的 ID 和确认依据，以便溯源。

| 参数 | 确认值 | Pilot Run ID | 确认依据 |
|---|---|---|---|
| `generation.num_icl_groups` | 5 | — | 5 组 ICL exemplar 提供了足够的长度梯度 |
| `generation.samples_per_group` | 3 | — | 每题 15 samples 已足够覆盖各长度桶 |
| `generation.icl_group_temperatures` | 见 §8.1 | — | per-group 变温配合 ICL 风格产生宽分布 |
| `generation.max_new_tokens` | 512 | — | 覆盖最长 CoT |
| `analysis.min_bin_size` | 5 | — | 保证每桶 SE 可计算 |
| `analysis.num_spot_checks` | 3 | — | 平衡成本与覆盖 |
| `analysis.max_extraction_fail_rate` | 0.05 | — | Pilot 实测 extraction 成功率满足 |
| `tas.plateau_threshold` | 0.05 | — | 待 Pilot TAS 烟雾测试验证 |

> 注：Pilot Run ID 和详细确认依据在实际 Pilot 完成后由人工回填。

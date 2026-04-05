# Stage 1 实验规格文档

## 0. 范围

Stage 1 = 1 个模型 + 1 个数据集上的完整数据闭环。

- 模型：Llama-3.1-8B-Instruct
- 数据集：GSM8K (test split)
- 基础设施：UNSW Katana (PBS, 单节点单 GPU)

本文档只规定 Stage 1 的实验设计、数据流和产出物。Stage 2（多模型、多数据集扩展）的范围在末尾简述，但不做技术规定。

---

## 1. 研究问题与假说

### 1.1 研究问题（RQ）

> 在同一个模型、同一个数据集上同时测量 accuracy-optimal CoT length（L\*）和 NLDD reasoning horizon（k\*）时，二者的数值关系是什么？这种关系如何随控制长度和题目难度变化？

背景：Optimal-length 论文报告了 accuracy 对 CoT 长度的倒 U 型关系，从行为层面说明存在 L\*。NLDD 论文通过 step-level counterfactual 测量因果贡献，从机制层面说明存在 k\*。两篇论文独立发表，未讨论对方。本实验在同一批数据上同时测量两者。

### 1.2 三个互斥假说

**H1：L\* ≈ k\*（一致）**
最优长度与推理地平线重合。超过 L\* 的 step 既不改善 accuracy，也不再有因果作用。行为层面的倒 U 型完全由机制层面的 faithfulness decay 解释。

**H2：L\* < k\*（行为拐点早于机制地平线）**
accuracy 已经下降，但 NLDD 说后续 step 仍有因果贡献。含义：模型在忠实地执行额外推理，但额外推理有害（overthinking / 过度纠正）。

**H3：L\* > k\*（行为拐点晚于机制地平线）**
NLDD 说 step 已无因果贡献，但 accuracy 仍在上升。含义：后段 step 通过 NLDD 测不到的间接机制起作用（格式锚定、注意力稳定等），或 NLDD 的 single-step corruption 方法灵敏度不足。

三个假说均有独立的研究价值，无论哪个被支持都构成可报告的发现。

---

## 2. 核心设计：Length-Controlled NLDD Surface

### 2.1 设计思想

长度不是被动观测的混杂变量，而是主动操控的实验条件。对每道题在多个控制长度下生成 trace，同时收集 accuracy 和 NLDD，得到一个二维数据结构：

- 横轴：step position k（1, 2, ..., L）
- 纵轴：controlled length L
- 值：mean NLDD(k | L)
- 附加一列：accuracy(L)

从这个 surface 上直接读出：

- accuracy(L) 曲线 → 定位 L\*
- 固定一行 NLDD profile → 定位 k\*(L)
- k\*(L) vs L 的关系 → 检验 H1/H2/H3

### 2.2 长度网格

默认网格：L ∈ {3, 5, 7, 9, 11, 13} （6 档）。

选择依据：GSM8K 在 Llama-3.1-8B-Instruct 上的自然 CoT 长度多数落在 4–10 step 范围内。网格覆盖了从偏短到偏长的区间，步幅 2 足以捕捉趋势。

如果初步结果表明 L\* 落在网格边缘（如 L\* ≥ 13），追加更长的档位。

### 2.3 长度控制手段

采用 prompt 指令引导：在 system prompt 或 few-shot exemplar 中显式要求"请用恰好 N 个推理步骤解答"。

长度服从性不需要完美。按实际产出的 step 数分桶即可，prompt 指令的作用是把自然分布从窄峰拉到大致均匀覆盖目标范围。

如果模型在某个目标长度下的服从率低于 30%（即超过 70% 的 trace 偏离目标 ±1 step），则追加 few-shot exemplar 长度匹配或 max\_new\_tokens 截断。

---

## 3. 统一定义

以下定义在整个实验中保持一致，不因测量类型（accuracy / NLDD）而改变。

### 3.1 Step 定义

一个 step = CoT 文本中被换行符分割的一个非空文本段。

切分规则：

1. 以 `\n` 分割原始 completion text
2. strip 每段首尾空白
3. 丢弃空字符串
4. 丢弃纯标点或纯空白段
5. 最终答案行（包含 `####` 或 `The answer is` 的行）不计入 step，单独记录为 `final_answer_line`

`num_steps` = 上述过程产出的 step 列表长度。

这个定义同时用于：accuracy curve 的横轴、NLDD corruption 的 step index、长度控制的目标 step 数。

### 3.2 答案抽取

GSM8K 的 gold answer 是一个数值。抽取规则：

1. 在 completion 中查找 `####` 后的内容，取第一个匹配到的数值
2. 若无 `####`，查找 `The answer is` 后的数值
3. 若仍未匹配，标记 `extraction_failed = True`

数值标准化：去除逗号、美元符号、百分号，转为 float，比较时用 `abs(extracted - gold) < 1e-3`。

### 3.3 答案 token 置信度

NLDD 需要测量模型对正确答案的 logit margin。定义：

- 令 gold answer 的第一个 token 为 `t_gold`（由 tokenizer 编码 gold answer string 后取第一个 token）
- 置信度 = logit(t\_gold) − max\_{t ≠ t\_gold} logit(t)
- 即 gold token 的 logit 减去所有非 gold token 中最大的 logit

此定义在 clean 和 corrupt 条件下保持一致。

### 3.4 NLDD 计算

对一条长度为 L 的 clean trace，corruption position k ∈ {1, ..., L}：

```
NLDD(k) = (margin_clean − margin_corrupt_k) / |margin_clean|
```

- `margin_clean`：完整 clean trace 喂入模型后的答案 token 置信度
- `margin_corrupt_k`：将第 k 步替换为 corrupt 版本、截断第 k+1 步及之后内容后的答案 token 置信度
- 分母 `|margin_clean|` 为归一化常数

NLDD > 0 表示该 step 被 corrupt 后模型对正确答案更不确信（该 step 有正向因果作用）。

### 3.5 k\* 定义

k\*(L) = 在长度为 L 的 trace 的 NLDD profile 中，NLDD 值达到峰值的 step index。

默认使用 peak-based 定义。如需切换到 steepest-decline 或 threshold-based 定义，作为 config 参数声明，不改代码。

### 3.6 Corruption 方式

Stage 1 使用单一 corruption 类型：**arithmetic perturbation**。

具体操作：将目标 step 中的一个数值运算结果替换为一个偏离 ±(10%–50%) 的错误值。

选择理由：GSM8K 的每一步几乎都涉及算术，arithmetic perturbation 是最自然的 single-step corruption，不需要语言模型辅助构造，可用正则 + 随机替换实现。

构造规则：
1. 用正则定位该 step 中的数值表达式
2. 随机选择一个数值
3. 将其替换为 original × uniform(0.5, 0.9) 或 original × uniform(1.1, 1.5)，取整
4. 若该 step 无可替换数值，标记 `corruption_failed = True`，跳过

---

## 4. 单题完整流程

以下是对单道题 (q, a\_gold) 的完整处理流程。所有题共享同一流程。

### 4.1 阶段一：Length-Controlled Generation

```
输入：question q, gold answer a_gold, 长度网格 [L1, ..., Ln]
输出：traces 表（每行一条 trace，含 is_correct 和 actual_num_steps）

for L in 长度网格:
    构造 prompt_L = length_guided_prompt(q, target_steps=L)
    for i in 1..M:              # M = 每档采样条数，默认 M=5
        trace_i = autoregressive_generate(prompt_L, temperature=0.7)
        steps_i = segment_steps(trace_i)       # §3.1 的规则
        answer_i = extract_answer(trace_i)     # §3.2 的规则
        correct_i = judge(answer_i, a_gold)    # §3.2 的匹配规则
        保存 (q_id, L, i, trace_i, steps_i, answer_i, correct_i, actual_num_steps)
```

此阶段是主要 GPU 成本。每题 6 × 5 = 30 次自回归生成。

阶段一结束后即可聚合 accuracy(L)——不需要等阶段二。

### 4.2 阶段二：NLDD Measurement

```
输入：阶段一产出的 traces 表
输出：nldd 表（每行一个 (q_id, L, k, nldd_value)）

for L in 长度网格:
    candidates = traces 表中 (q_id, L, is_correct=True) 的行
    if candidates 为空:
        记录 (q_id, L, "no_clean_trace")
        continue
    clean_trace = candidates 中 actual_num_steps 最接近 L 的一条
    clean_steps = clean_trace 的 step 列表
    L_actual = len(clean_steps)

    # 1 次 clean forward pass
    clean_prompt = assemble_prompt(q, clean_steps, answer_prompt_suffix)
    margin_clean = forward_logit_margin(clean_prompt, a_gold)  # §3.3

    # L_actual 次 corrupt forward pass
    for k in 1..L_actual:
        corrupt_step_k = corrupt_arithmetic(clean_steps[k])  # §3.6
        if corruption_failed:
            记录 (q_id, L, k, "corruption_failed")
            continue
        corrupt_prefix = clean_steps[1..k-1] + [corrupt_step_k]
        corrupt_prompt = assemble_prompt(q, corrupt_prefix, answer_prompt_suffix)
        margin_corrupt_k = forward_logit_margin(corrupt_prompt, a_gold)
        nldd_k = (margin_clean - margin_corrupt_k) / abs(margin_clean)
        保存 (q_id, L, k, nldd_k, margin_clean, margin_corrupt_k)
```

此阶段是 forward pass only，无自回归。每条 clean trace 需要 (1 + L\_actual) 次 forward pass。

### 4.3 成本估算（单题）

| 阶段 | 操作类型 | 次数 | 相对成本 |
|------|---------|------|---------|
| 阶段一 | 自回归生成 | 30 | ~300 forward-pass 等价 |
| 阶段二 | 单次 forward pass | ~54 | ~54 forward-pass 等价 |
| **合计** | | | **~354 FP 等价** |

NLDD 测量成本 ≈ 生成成本的 18%。不存在组合爆炸。

### 4.4 进一步优化（可选）

- **Prefix caching**：同一条 clean trace 的不同 corruption 位置共享 [question + steps 1..k-1] 的 KV cache。按 k 从小到大处理时，每次只需增量计算一个 step 的 KV。
- **选择性 NLDD**：先跑完所有题的阶段一，确定全局 L\* 的大致范围，然后只对 L\* ± 2 档做 NLDD，将阶段二的成本再砍 50%。
- **粗扫 + 细扫**：对长 trace（L > 10），先隔步测 NLDD（k = 1, 3, 5, ...），找到变化最快的区间后再补测偶数步。

---

## 5. 数据流与执行顺序

### Stage A：准备评测子集

操作：从 GSM8K test split 中固定一个评测子集。

固定策略：按 question\_id 的 hash 排序后取前 N 题（默认 N=200）。Hash 排序保证子集与原始顺序无关且可复现。

产物：
- `eval_subset.jsonl`：每行含 question\_id, question\_text, gold\_answer
- `eval_subset_meta.json`：N, hash seed, 数据集版本

### Stage B：Length-Controlled Generation（阶段一）

操作：对 eval\_subset 中的每题执行 §4.1。

PBS 作业设计：每题独立或按 batch 分组提交。单 GPU 作业。

产物：
- `traces.jsonl`：每行一条 trace，schema 见 §6.1

此阶段结束后立即可以算 accuracy(L)。

### Stage C：构建 Accuracy-by-Length 表

操作：从 traces.jsonl 按 actual\_num\_steps 分桶，计算每桶的 accuracy mean 和 SE。

产物：
- `accuracy_by_length.csv`
- L\* 的估计值（accuracy 峰值对应的 step 数）

### Stage D：NLDD Measurement（阶段二）

操作：对 eval\_subset 中的每题执行 §4.2。

依赖：Stage B 的 traces.jsonl（需要从中挑选 clean trace）。

PBS 作业设计：和 Stage B 分开提交。NLDD 作业只做 forward pass，walltime 更短。

产物：
- `nldd_raw.jsonl`：每行一个 (q\_id, L, k, nldd\_value)，schema 见 §6.2

### Stage E：聚合

操作：

1. 从 nldd\_raw.jsonl 按 (L, k) 聚合 mean NLDD → `nldd_surface.csv`
2. 对每个 L，找 NLDD 峰值位置 → k\*(L)
3. 输出 `horizon_summary.csv`：每行含 L, k\_star, accuracy\_at\_L, L\_star

产物：
- `nldd_surface.csv`
- `nldd_by_relative_position.csv`（k/L 归一化后聚合）
- `horizon_summary.csv`

### Stage F：绘图

操作：从聚合表生成固定格式的图。绘图脚本不接触模型。

必出图：

1. **Accuracy vs CoT Length**：横轴 actual\_num\_steps，纵轴 accuracy，标注 L\*
2. **NLDD Surface Heatmap**：横轴 k，纵轴 L，颜色 mean NLDD，叠加 k\*(L) 折线
3. **k\*(L) vs L**：横轴 L，纵轴 k\*，叠加 L\* 的竖线，直观显示 H1/H2/H3
4. **Mean NLDD vs Relative Position**：横轴 k/L，纵轴 mean NLDD，不同 L 叠加

产物：
- 4 张 PNG/PDF
- 每张图的输入表快照

---

## 6. 数据 Schema

### 6.1 Trace 表 (traces.jsonl)

每行一条 trace：

| 字段 | 类型 | 说明 |
|------|------|------|
| trace\_id | str | `{q_id}_{L_target}_{sample_idx}` |
| question\_id | str | 题目唯一 ID |
| question\_text | str | 原始题目文本 |
| gold\_answer | float | 标准答案（数值） |
| model\_name | str | 模型标识 |
| target\_length | int | prompt 指示的目标 step 数 |
| prompt\_id | str | 所用 prompt 模板的版本 ID |
| temperature | float | 生成温度 |
| raw\_completion | str | 原始模型输出 |
| steps | list[str] | §3.1 切分后的 step 列表 |
| actual\_num\_steps | int | len(steps) |
| final\_answer\_line | str | 答案行原文 |
| extracted\_answer | float \| null | 抽取的数值答案 |
| is\_correct | bool | 是否与 gold\_answer 匹配 |
| extraction\_failed | bool | 答案抽取是否失败 |
| token\_count | int | completion 的 token 数 |
| timestamp | str | ISO 格式生成时间 |

### 6.2 NLDD 表 (nldd_raw.jsonl)

每行一个 corruption 测量点：

| 字段 | 类型 | 说明 |
|------|------|------|
| nldd\_id | str | `{q_id}_{L}_{k}` |
| question\_id | str | 题目 ID |
| clean\_trace\_id | str | 引用 traces.jsonl 中的 trace\_id |
| controlled\_length | int | 该行对应的控制长度 L |
| actual\_clean\_length | int | clean trace 的实际 step 数 |
| corruption\_step\_index | int | k（从 1 开始） |
| corruption\_type | str | `"arithmetic_perturbation"` |
| original\_step\_text | str | 被替换前的 step 原文 |
| corrupt\_step\_text | str | 替换后的 step 文本 |
| corruption\_failed | bool | 该 step 是否无法 corrupt |
| margin\_clean | float | clean 条件下的 logit margin |
| margin\_corrupt | float | corrupt 条件下的 logit margin |
| nldd\_value | float | (margin\_clean − margin\_corrupt) / \|margin\_clean\| |
| timestamp | str | ISO 格式测量时间 |

---

## 7. Prompt 规范

### 7.1 Length-Guided Generation Prompt

用于阶段一。结构：

```
System: You are a math tutor. Solve the following problem step by step.
        Use exactly {N} reasoning steps. Number each step.
        After all steps, write your final answer as "#### <number>".

[可选：1-2 个匹配目标长度的 few-shot exemplar]

User: {question_text}
```

每个目标长度 L 对应一个 prompt\_id = `len_guided_v1_L{L}`。

Exemplar 的 step 数必须与目标 L 匹配或接近（±1）。

### 7.2 NLDD Evaluation Prompt

用于阶段二的 forward pass。结构：

Clean 条件：
```
{question_text}

Step 1: {clean_step_1}
Step 2: {clean_step_2}
...
Step L: {clean_step_L}

#### 
```

Corrupt at k 条件：
```
{question_text}

Step 1: {clean_step_1}
...
Step k-1: {clean_step_{k-1}}
Step k: {corrupt_step_k}

#### 
```

注意：corrupt 条件下，step k 之后的所有步骤被截断。模型只需要在 `####` 之后的位置产出 answer token 的 logit，不需要生成。

prompt\_id = `nldd_eval_v1`。

### 7.3 Prompt 版本化

所有 prompt 模板以 YAML 文件形式存储在 `prompts/` 目录下。每个模板有唯一 prompt\_id。trace 表中的 prompt\_id 字段回溯到此文件。任何 prompt 文本变更必须更新版本号。

---

## 8. 模型与资源

### 8.1 模型规格

| 项目 | 值 |
|------|------|
| 模型 | meta-llama/Llama-3.1-8B-Instruct |
| 精度 | float16 |
| 显存需求 | ~16 GB |
| 来源 | Hugging Face Hub |
| 权限 | 需要接受 Meta license + HF\_TOKEN |

### 8.2 PBS 资源估算

阶段一（生成）：
- 200 题 × 30 次生成 = 6000 次自回归生成
- 估计每次生成 ~2s（含 tokenization + generation + IO）
- 总计 ~3.3h → 单个 4h PBS 作业可完成，或按长度档分 6 个作业

阶段二（NLDD forward pass）：
- 200 题 × ~54 次 forward pass = ~10800 次
- 每次 forward pass ~0.3s（无生成，只跑到最后 token）
- 总计 ~0.9h → 单个 2h PBS 作业

### 8.3 Scratch 存储估算

- traces.jsonl：~6000 行 × ~2KB/行 ≈ 12 MB
- nldd\_raw.jsonl：~10800 行 × ~1KB/行 ≈ 11 MB
- 聚合表 + 图：< 5 MB
- 模型权重缓存：~16 GB（已有则复用）

总增量：< 30 MB（不含模型缓存）。

---

## 9. 配置规范

所有实验参数集中在一个 YAML 配置文件中。单个 config 必须唯一决定一次完整实验。

```yaml
experiment:
  name: "stage1-gsm8k-llama31"
  seed: 42

dataset:
  name: "gsm8k"
  split: "test"
  subset_size: 200
  subset_hash_seed: 42

model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  dtype: "float16"
  hf_cache: "${SCRATCH}/hf-home/hub"

generation:
  length_grid: [3, 5, 7, 9, 11, 13]
  samples_per_length: 5
  temperature: 0.7
  max_new_tokens: 512          # 安全上限，防止无限生成

step_segmentation:
  method: "newline"             # §3.1 的规则
  strip_whitespace: true
  drop_empty: true
  answer_markers: ["####", "The answer is"]

answer_extraction:
  method: "marker_then_regex"   # §3.2 的规则
  numeric_tolerance: 1e-3

nldd:
  corruption_type: "arithmetic_perturbation"
  perturbation_range: [0.5, 1.5]  # multiplier 的范围，排除 [0.9, 1.1]
  confidence_metric: "logit_margin"  # §3.3
  normalization: "abs_clean_margin"  # §3.4
  horizon_definition: "peak"         # §3.5

output:
  base_dir: "${SCRATCH}/runs/${RUN_NAME}"
```

---

## 10. 验证检查清单

### 10.1 阶段一完成后

- [ ] traces.jsonl 行数 = subset\_size × len(length\_grid) × samples\_per\_length
- [ ] 每个 (question\_id, target\_length) 组合恰好有 samples\_per\_length 条
- [ ] extraction\_failed 比例 < 10%
- [ ] actual\_num\_steps 的分布覆盖了 length\_grid 的范围（长度引导有效）
- [ ] accuracy\_by\_length.csv 可生成且 L\* 可定位

### 10.2 阶段二完成后

- [ ] nldd\_raw.jsonl 中每个有 clean trace 的 (question\_id, L) 组合都有 NLDD 记录
- [ ] corruption\_failed 比例 < 15%
- [ ] nldd\_value 分布合理（大部分 > 0，少量 ≤ 0）
- [ ] margin\_clean 大部分 > 0（clean trace 原本就是 correct 的）
- [ ] 每条 nldd 记录可通过 clean\_trace\_id 回溯到 traces.jsonl

### 10.3 聚合与绘图后

- [ ] nldd\_surface.csv 的 (L, k) 网格完整，空缺处标记原因
- [ ] horizon\_summary.csv 中 k\* 值在合理范围内（1 ≤ k\* ≤ L）
- [ ] 4 张图均可渲染，轴标签和标注正确

---

## 11. 不做的事（Stage 1 显式排除）

- 不使用 incorrect trace 做分析。incorrect trace 只贡献 accuracy 统计。
- 不做多模型对比。
- 不做 GSM8K 以外的数据集。
- 不做 RSA / TAS / probing 等解释性分析。
- 不做自然长度分布的观测实验（长度是控制变量，不是观测变量）。
- 不做 multi-step corruption（每次只改一步）。
- 不做题目难度分层（留到 Stage 1 数据出来后再决定是否追加）。

---

## 12. Stage 2 扩展方向（仅记录，不规定）

Stage 1 完成后，以下维度可逐一引入：

**模型维度**（参考 NLDD 论文的模型选择逻辑）：
- Post-hoc rationalizer 型模型（如 Llama-3.1-70B-Instruct）
- Faithful reasoner 型模型（如 DeepSeek-R1 系列）
- 小模型基线（如 Gemma-2-2B）
- 目的：检验 H1/H2/H3 的结论是否依赖于模型的 faithfulness 特性

**数据集维度**（参考 optimal-length 论文的数据集选择）：
- 形式推理：Dyck-n, PrOntoQA
- 数学推理：MATH（比 GSM8K 更难）
- 常识推理：StrategyQA
- 目的：检验 L\* 与 k\* 的关系是否依赖于任务类型

**分析维度**：
- 题目难度分层：按 gold answer 的数值大小或所需运算步数分层
- Multi-step corruption：同时改多步，测交互效应
- Token-level NLDD：不以 step 为单位，而以 token 为单位做 corruption

每个扩展维度独立于 Stage 1 的代码框架，只需替换 config 中的 model/dataset 字段。
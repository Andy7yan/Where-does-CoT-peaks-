# Stage1 正式结果分析报告

本报告将“前 200 题阶段性观察”和“全量正式 run”合并整理到同一份结论里，面向当前 `results/stage1-trace-0412` 目录的最终产物。

## 1. 数据范围与目录修正

- 当前正式 run 共覆盖 `1319` 道 GSM8K 测试题。
- 每题固定 `15` 条 trace: `5` 个 prompt 组 x 每组 `3` 次采样。
- 全量 trace 总数为 `19,785`。
- 本次整理时发现根目录下旧的 `traces.jsonl`、`question_metadata.jsonl`、`accuracy_by_length.csv`、`failed_corruptions.jsonl` 仍停留在前 `200` 题版本，现已基于 `shards/` 重新汇总为全量版本。
- 正式 shard 的 `run_meta` 显示，正式 run 实际配置为每个 prompt `3` 次采样、`max_new_tokens=512`；这与当前 [configs/stage1.yaml](/abs/path/e:/VS%20ProjVault/peak-CoT/configs/stage1.yaml:1) 中仍保留的 `subset_size: 200`、`max_new_tokens: 544` 不一致，后续若要复现实验或写方法部分，需要人工确认以正式 run meta 为准。

## 2. 核心结论

### 2.1 总体表现

- 全量 trace 准确率: `78.00%` (`15432 / 19785`)
- 全量 extraction fail rate: `0.88%` (`174 / 19785`)
- 发生过 extraction fail 的题目数: `98 / 1319`
- 题目级准确率均值: `78.00%`
- 题目级准确率中位数: `86.67%`
- 满分题: `514`
- 全错题: `45`

这说明：

- 这次正式 run 已经足够稳定，整体结论不会因为少数 trace 抖动而翻转。
- 但尾部仍然明显存在两类问题: 一类是“稳定算错”的题，另一类是“本来不一定不会做，但输出跑偏/截断，导致抽取失败”的题。

### 2.2 前 200 题是否具有代表性

| 子集 | 题目数 | trace 数 | 准确率 | extraction fail rate | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 前 200 题 | 200 | 3000 | 78.40% | 1.17% | 5.48 |
| 其余 1119 题 | 1119 | 16785 | 77.93% | 0.83% | 5.41 |

结论：

- 前 200 题对全量结果是“基本有代表性”的，整体准确率只高了 `0.47` 个百分点。
- 但它对 `icl_verbose` 略偏乐观，对 `icl_short` 略偏保守，所以如果要写 prompt 间差异，仍应以全量结果为准。

### 2.3 各 prompt 组表现

| Prompt | Trace 数 | 准确率 | Extraction Fail Rate | 平均步数 | 平均 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| `icl_minimal` | 3957 | 58.38% | 0.76% | 1.66 | 49.94 |
| `icl_short` | 3957 | 80.41% | 0.68% | 3.17 | 83.68 |
| `icl_medium` | 3957 | 83.52% | 0.83% | 4.52 | 113.44 |
| `icl_detailed` | 3957 | 84.13% | 1.01% | 6.76 | 139.04 |
| `icl_verbose` | 3957 | 83.57% | 1.11% | 10.99 | 200.78 |

结论：

- `icl_minimal` 明显掉队，不适合作为后续正式分析主配置。
- 高性能组是 `icl_medium / icl_detailed / icl_verbose`，但三者准确率差距其实很小。
- 如果同时考虑 token 成本，`icl_detailed` 是“精度最高”，但 `icl_medium` 的性价比更高；`icl_verbose` 没有体现出足够的额外收益。

## 3. 长度与准确率

### 3.1 全局长度趋势

- 全量平均步数: `5.42`
- 步数中位数: `4`
- `p90 = 11`，`p95 = 13`
- 最长 trace: `66` 步
- 按 `accuracy_by_length.csv`，全局最优桶 `L* = 4`

按长度桶观察：

- `3-4` 步是最稳区间，其中 `4` 步桶准确率最高，为 `84.62%`
- `7-11` 步仍然可用，但没有明显优于 `3-4` 步
- `12` 步以后整体开始下降
- 极长链明显不稳，例如 `25` 步桶仅 `21.05%`，`30` 步桶仅 `15.15%`

### 3.2 题目级最优长度

- `588 / 1319` 题的题目级最优长度是 `1`
- `1137 / 1319 = 86.20%` 的题目级最优长度不超过 `4`
- `1263 / 1319 = 95.75%` 的题目级最优长度不超过 `8`

结论：

- 正式 run 的主要信号非常一致: “更长的链”并不自动带来更高正确率。
- 真正高频、稳定的最佳区间还是短到中等长度，尤其是 `2-4` 步。

## 4. 失败模式

### 4.1 稳定算错

全量共有 `45` 道题在所有 `15` 条 trace 上全部答错。典型例子：

- `gsm8k_0206`: 2.25 小时按“每小时或不足一小时都按 1 小时计费”应收 `3` 小时，但模型普遍按 `2.25 * 35` 直接计算。
- `gsm8k_0287`: 把“Adrien 的工资四年后涨 40%”错误传播到 Lylah，导致总额系统性偏差。
- `gsm8k_0493`: 多种 prompt 都混淆了 sticker / button 的兑换方向，错误不是格式问题，而是建模错误。
- `gsm8k_0891`: 多个 prompt 在列方程时就走偏，部分 trace 还出现无法收敛到合法整数年龄的情况。

这类题说明目前模型的主要硬伤仍然是“结构化算式建模错误”，不是单纯输出噪声。

### 4.2 Extraction fail / 输出跑偏

- 全量 extraction fail 共 `174` 条 trace，覆盖 `98` 道题。
- extraction fail 最多的 prompt 是 `icl_verbose` (`44` 条) 和 `icl_detailed` (`40` 条)。
- 说明输出越长，越容易在后段发生重复、自我修正、截断或没有干净 `####` 结尾。

典型例子：

- `gsm8k_0080`: `icl_minimal` 连续生成 `3x = 5` 并截断在 `3x =`
- `gsm8k_0421`: 多个 prompt 在错误假设下反复自我纠缠，最后没有可抽取答案
- `gsm8k_0891`: 若干 trace 明明已经接近解，但最终落入“年龄不能是分数”之类的循环解释，导致没有合法终答案

### 4.3 Corruption feasibility

基于全量正确 trace 的步骤级重算，当前 `failed_corruptions.jsonl` 共 `20546` 条失败记录。

- 正确 trace 的总 step attempts: `83837`
- `no_numeric + other` 失败率: `24.51%`
- `token_delta_exceeded` 失败率: `0.00%`（本次全量重算中未观测到）

这和 pilot 的结论是一致的：

- arithmetic corruption 不是完全不可做
- 但仍有相当多步骤本身不含可替换数字，正式 NLDD 仍然需要人工决定是否继续推进，或是否应先改 corruption 规则

## 5. 结论与建议

### 5.1 可以直接写进正式结论的内容

- 正式 run 的总体表现稳定在 `78%` 左右，前 200 题结论基本成立。
- `icl_minimal` 明显不适合作为后续主 prompt。
- `icl_medium / icl_detailed / icl_verbose` 三者差距很小，其中 `icl_detailed` 精度最好，但 `icl_medium` 的 token 成本更优。
- 长链并不天然更好，绝大多数题的最优长度集中在 `1-4` 步。
- 真正需要重点解释的尾部问题，不是均值下降，而是少量“稳定算错题”和一批“输出失控导致抽取失败”的题。

### 5.2 当前最值得人工看的地方

- 是否把 `icl_detailed` 还是 `icl_medium` 作为后续主分析 prompt
- 是否继续推进 NLDD corruption，还是先调整 corruption 设计
- 哪 `25` 题作为正式 full analysis 样本
- 哪 `3` 题作为 spot check
- 是否同步修正配置文件与文稿中的正式 run 参数

详细人工复核清单见 [manual_review_items.md](/abs/path/e:/VS%20ProjVault/peak-CoT/results/stage1-trace-0412/manual_review_items.md:1)。

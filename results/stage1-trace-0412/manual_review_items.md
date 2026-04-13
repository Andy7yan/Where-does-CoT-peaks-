# Stage1 人工复核清单

本文件只列“目前仍然需要人工判断”的部分，按优先级从高到低排列。

## 1. 必须人工决定的事项

### 1.1 后续主 prompt 选型

当前最强的三组是：

- `icl_detailed`: `84.13%`
- `icl_verbose`: `83.57%`
- `icl_medium`: `83.52%`

这里机器不能替你直接拍板，因为这是“精度 vs 成本 vs 输出稳定性”的取舍：

- 要最高准确率，选 `icl_detailed`
- 要更省 token 且表现几乎不掉，选 `icl_medium`
- `icl_verbose` 没显示出足够额外收益，通常不建议继续作为主配置

### 1.2 是否继续推进 NLDD corruption

全量步骤级统计显示：

- `failed_corruptions.jsonl` 总失败 `20546`
- 正确 trace 的 step attempts 为 `83837`
- `no_numeric + other` 失败率 `24.51%`

这仍然落在 pilot 已经提示过的 `WARN` 区间附近。是否接受这个噪声水平、继续做正式 NLDD，需要人工判断。

### 1.3 正式 run 配置口径

当前目录中的正式 shard `run_meta` 与 [configs/stage1.yaml](/abs/path/e:/VS%20ProjVault/peak-CoT/configs/stage1.yaml:1) 不一致：

- 正式 run 实际是全量 `1319` 题，不是 `subset_size: 200`
- 正式 run 的 `max_new_tokens` 是 `512`，不是配置里的 `544`

后续报告、论文或复现实验说明，应该以哪个口径为准，需要人工确认。

## 2. 建议作为正式 full analysis 的 25 题

这 25 题是按三类信号混合挑出来的：

- 稳定全错题
- extraction fail 代表题
- prompt 分歧显著题

推荐清单：

`gsm8k_0206, gsm8k_0287, gsm8k_0493, gsm8k_0607, gsm8k_0891, gsm8k_0022, gsm8k_0211, gsm8k_0426, gsm8k_0601, gsm8k_0080, gsm8k_0421, gsm8k_0042, gsm8k_0018, gsm8k_0016, gsm8k_0791, gsm8k_0261, gsm8k_0003, gsm8k_0004, gsm8k_0019, gsm8k_0036, gsm8k_1249, gsm8k_1197, gsm8k_1094, gsm8k_1012, gsm8k_0860`

## 3. 建议作为 spot check 的 3 题

- `gsm8k_0080`
  代表“短 prompt 失控 + 重复生成 + 抽取失败”
- `gsm8k_0206`
  代表“所有 prompt 都稳定算错，属于真正的 reasoning error”
- `gsm8k_1249`
  代表“不同 prompt 对同一题会落到完全不同解法路径，适合人工比较 prompt 风格差异”

## 4. 全错题清单

以下 `45` 题在全部 `15` 条 trace 上都答错：

`gsm8k_0022, gsm8k_0057, gsm8k_0152, gsm8k_0185, gsm8k_0206, gsm8k_0211, gsm8k_0219, gsm8k_0228, gsm8k_0262, gsm8k_0287, gsm8k_0317, gsm8k_0337, gsm8k_0371, gsm8k_0426, gsm8k_0428, gsm8k_0432, gsm8k_0448, gsm8k_0493, gsm8k_0601, gsm8k_0607, gsm8k_0616, gsm8k_0642, gsm8k_0662, gsm8k_0674, gsm8k_0745, gsm8k_0748, gsm8k_0805, gsm8k_0832, gsm8k_0875, gsm8k_0891, gsm8k_0925, gsm8k_0943, gsm8k_0976, gsm8k_1048, gsm8k_1052, gsm8k_1073, gsm8k_1141, gsm8k_1150, gsm8k_1169, gsm8k_1184, gsm8k_1223, gsm8k_1233, gsm8k_1264, gsm8k_1279, gsm8k_1292`

这批题建议至少人工看一遍其中的代表样本，因为它们最可能定义后续 error taxonomy。

## 5. Extraction fail 题目清单

以下 `98` 题至少出现过一次 extraction fail：

`gsm8k_0003, gsm8k_0004, gsm8k_0016, gsm8k_0018, gsm8k_0042, gsm8k_0057, gsm8k_0065, gsm8k_0075, gsm8k_0080, gsm8k_0100, gsm8k_0149, gsm8k_0150, gsm8k_0152, gsm8k_0165, gsm8k_0167, gsm8k_0169, gsm8k_0188, gsm8k_0201, gsm8k_0211, gsm8k_0220, gsm8k_0238, gsm8k_0245, gsm8k_0261, gsm8k_0268, gsm8k_0277, gsm8k_0297, gsm8k_0302, gsm8k_0303, gsm8k_0308, gsm8k_0319, gsm8k_0337, gsm8k_0338, gsm8k_0348, gsm8k_0359, gsm8k_0388, gsm8k_0390, gsm8k_0416, gsm8k_0421, gsm8k_0428, gsm8k_0458, gsm8k_0476, gsm8k_0489, gsm8k_0503, gsm8k_0520, gsm8k_0552, gsm8k_0605, gsm8k_0616, gsm8k_0617, gsm8k_0642, gsm8k_0676, gsm8k_0684, gsm8k_0686, gsm8k_0726, gsm8k_0739, gsm8k_0743, gsm8k_0745, gsm8k_0748, gsm8k_0763, gsm8k_0773, gsm8k_0786, gsm8k_0791, gsm8k_0801, gsm8k_0802, gsm8k_0821, gsm8k_0831, gsm8k_0832, gsm8k_0839, gsm8k_0850, gsm8k_0868, gsm8k_0882, gsm8k_0888, gsm8k_0891, gsm8k_0898, gsm8k_0906, gsm8k_0930, gsm8k_0957, gsm8k_0958, gsm8k_0976, gsm8k_0996, gsm8k_1006, gsm8k_1043, gsm8k_1067, gsm8k_1080, gsm8k_1083, gsm8k_1087, gsm8k_1119, gsm8k_1124, gsm8k_1131, gsm8k_1155, gsm8k_1167, gsm8k_1169, gsm8k_1183, gsm8k_1187, gsm8k_1191, gsm8k_1206, gsm8k_1253, gsm8k_1268, gsm8k_1269`

建议优先人工看这些代表题：

- `gsm8k_0080`
- `gsm8k_0421`
- `gsm8k_0891`
- `gsm8k_0042`
- `gsm8k_0261`
- `gsm8k_0791`

## 6. 需要人工解释但不必全量逐题看的内容

### 6.1 Prompt 分歧

共有 `466` 题出现了 prompt 间最大准确率差值为 `1.0` 的情况，也就是某些 prompt 全对、某些 prompt 全错。

这不意味着 `466` 题都要逐题人工看，但至少要人工挑若干代表题来解释：

- 为什么某些题对 prompt 风格极敏感
- 为什么 `icl_minimal` 会在少数题上反而是唯一最佳
- 为什么 `icl_verbose` 在前 200 题看起来更强、到全量后优势变弱

优先建议从以下题里挑：

`gsm8k_0003, gsm8k_0004, gsm8k_0019, gsm8k_0036, gsm8k_1249, gsm8k_1197, gsm8k_1094, gsm8k_1012, gsm8k_0860`

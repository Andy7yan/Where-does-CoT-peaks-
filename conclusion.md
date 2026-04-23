# Stage 1 结果收束笔记

## 0. 使用原则

- 只根据当前数据与当前实验设置写结论。
- 不回扣原始假设。
- 区分：
  - 可直接写入论文的结论
  - 需要特别说明的边界
  - 建议保留的图片

## 1. 最终结论

### 1.1 强结论

- `k*` 更像随 trace length `L` 缩放的相对 horizon，不是固定绝对步数。
- `k*` 不是 reasoning cutoff，而是边际收益转折点。
- `k*` 之后的大多数 step 仍然是正向的，但贡献变弱、几何效率变低。
- 后段 reasoning 的主特征是继续收敛，但持续减速。
- TAS 更像全局几何减速背景；NLDD 更像题目相关的局部因果结构。

### 1.2 弱结论

- `k*/L` 与 accuracy 没有明显关系。
- difficulty 主要改变 accuracy 水平，不明显改变 horizon 的相对结构。
- trace 越长，post-`k*` 的 diminishing returns 越强。

### 1.3 当前不能下的结论

- 不能强说 `accuracy(L)` 呈统一、明显的 inverted-U。
- 不能强说已经完成 `L*` 与 `k*` 的直接配对解释。
- 不能把当前结果写成对两篇目标论文的直接复现。

## 2. 需要特别说明

### 2.1 与目标论文条件不一致

- 当前不是 Optimal CoT Length paper 的直接复现。
- 当前不是 NLDD paper 的直接复现。
- 当前只做单模型、单数据集、Stage 1 子问题。
- 当前未做 RSA、linear probing、多模型对比、多数据集对比。
- 当前 corruption 未启用 NLDD paper 的质量过滤器。

### 2.2 当前设计会影响解释

- 长度变化来自 ICL 风格诱导，不是硬控长度。
- `L` 不只代表长短，也部分混入了解题风格与粒度变化。
- NLDD/TAS 只在正确 trace 上测量。
- PQ 主分析只覆盖 medium / hard，且排除了 degenerate questions。
- insufficient bins 会留空，不做插值。
- `difficulty_score` 是本 run 内生代理量，不是外部 gold 难度。
- overall 与 PQ 是两条独立流水线，且各自独立标定 `S`。

### 2.3 结果解释边界

- `k*/L = 1` 中有相当一部分是短链分辨率 artefact。
- 当前最稳的结论是 horizon 结构与 post-horizon 行为。
- 当前最不稳的部分是 behavioural optimal length 的强版本叙事。
- 当前尚未完成 `L*(q)` 与 `k*(q, L*(q))` 的直接配对结论。
- 论文表述范围应限定在当前模型、当前数据集、当前实验设置内。

## 3. 建议保留的图片

### 3.1 主文保留

1. **T1-A**
   - 用途：全局 overview
   - 承担信息：accuracy、TAS、`k*` 随 `L` 的整体结构

2. **pooled `k*/L` vs `L`**
   - 用途：支持 relative horizon
   - 承担信息：`k*` 随 `L` 缩放，`k*/L` 大致稳定但非严格常数

3. **T1-B overall heatmap**
   - 用途：核心机制图
   - 承担信息：NLDD 的中后段对角带结构；TAS 的规则性衰减

4. **Deep-dive Figure G**
   - 用途：最强 post-horizon 总结图
   - 承担信息：post-`k*` 的主导状态是“NLDD 非负 + TAS 仍在移动”

5. **Deep-dive Figure J**
   - 用途：几何支持证据
   - 承担信息：post-`k*` slope 系统性比 pre-`k*` 更平，说明进入减速收敛相

### 3.2 附录保留

- T1-D exemplar heatmaps
- Deep-dive Figure E
- Deep-dive Figure F
- Deep-dive Figure H
- Deep-dive Figure I

## 4. 不建议放入主文的图片

- T1-A-norm
- delta
- rho
- scatter
- targeted Figure A
- targeted Figure B
- targeted Figure C
- targeted Figure D

## 5. 最终论文结论的压缩版

- 当前数据最稳地支持：`k*` 是一个随 `L` 缩放的相对 horizon。
- 它标记的不是 reasoning 的终止，而是从高因果收益阶段进入低收益、低几何效率收敛阶段的转折点。
- 后段 reasoning 通常仍然正向，但越来越低效。
- 与此同时，behavioural optimal length 的证据在当前 Stage 1 设置下仍然偏弱且异质。
- 因此，当前工作更适合把重点放在 horizon 结构与 post-horizon 行为，而不是强推 `L*`–`k*` 已对齐。
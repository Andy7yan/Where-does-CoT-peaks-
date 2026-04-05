# Ref: 公开代码依赖与可复用路径

## 目的

本文件只整理**可以直接参考或直接借用的开源代码路径**。

关注点不是“论文讲了什么”，而是：

- 哪个 repo 值得看
- 具体看哪个路径
- 这段代码在本项目里可以拿来干什么

默认服务于当前第一阶段：

- 复现 optimal-length 风格的数据生成
- 在同一批显式 CoT 上追加 NLDD 风格的数据采集
- 产出长度曲线、NLDD 曲线、基础中间表

---

## A. 第一优先级：直接相关官方代码

## A1. `PKU-ML/CoT-Length`

### 作用定位

这是 **optimal-length** 论文的官方代码来源。

### 优先参考路径

#### 1. `PKU-ML/CoT-Length/real/run_math500.py`
**用途批注：**
- 参考“真实任务上批量生成多条 CoT 样本”的执行方式
- 可借用为你自己的 `run_gsm8k.py` 模板
- 重点看：输入数据读取、循环生成、样本保存、重试逻辑

#### 2. `PKU-ML/CoT-Length/real/run_winogrande.py`
**用途批注：**
- 和上一个脚本一起看，用来分离“任务无关的采样框架”和“任务相关的 prompt / 判分差异”
- 可用于提炼通用批量推理 runner

#### 3. `PKU-ML/CoT-Length/real/eval_math500_cot_length.py`
**用途批注：**
- 这是当前项目最值得直接参考的路径之一
- 可用于实现：
  - step 长度统计
  - 长度分桶
  - accuracy vs length 聚合
  - optimal length 候选值提取

#### 4. `PKU-ML/CoT-Length/real/eval_winogrande_cot_length.py`
**用途批注：**
- 用于对照上一个脚本，抽出“长度曲线分析”的通用骨架
- 适合拿来做数据后处理模块的第二个 sanity reference

#### 5. `PKU-ML/CoT-Length/real/eval_math500_task_difficulty.py`
**用途批注：**
- 当前阶段不一定直接照搬
- 但可参考它如何组织“额外维度上的聚合分析”
- 后续你如果要做 `difficulty × length × k*` 的扩展，会有帮助

#### 6. `PKU-ML/CoT-Length/real/eval_winogrande_task_difficulty.py`
**用途批注：**
- 同上
- 主要价值是学习其聚合分析脚本的结构，而不是具体任务细节

### 次级参考路径

#### 7. `PKU-ML/CoT-Length/synthetic/dataset/`
**用途批注：**
- 用来理解论文在 synthetic setting 里如何显式控制 step 数
- 对当前 GSM8K 主线不是直接依赖
- 但对你后面设计“定长 CoT 验证实验”很有参考价值

#### 8. `PKU-ML/CoT-Length/synthetic/model/`
**用途批注：**
- 主要用于理解 synthetic paper setting，不建议直接搬到当前主线
- 可帮助你理解论文中“最优长度”概念的受控实验来源

#### 9. `PKU-ML/CoT-Length/synthetic/eval.py`
**用途批注：**
- 可参考其“固定长度评测”的思路
- 适合后续做你提出的“先找 L*，再围绕少数定长做 targeted 检测”那类补充实验

---

## A2. `donald-ye/NLDD`

### 作用定位

这是 **NLDD** 论文的官方代码来源。

### 核心参考路径

#### 1. `donald-ye/NLDD/pipeline.py`
**用途批注：**
- 当前 repo 的核心实现几乎全集中在这里
- 这是你迁移 NLDD 方法时必须看的主文件
- 可直接参考的内容包括：
  - 实验配置对象
  - 统一 sample 数据结构
  - clean / corrupt pair 数据结构
  - counterfactual 有效性检查
  - RSA 计算接口
  - TAS 结果结构
  - probing 任务路由
  - reasoning horizon 相关分析骨架

### 从这个文件里优先抽出的概念

#### 2. `ExperimentConfig`
**用途批注：**
- 可作为你自己项目的 config schema 参考母版
- 尤其适合参考：样本数、dataset split、task 开关、分析项开关如何统一管理

#### 3. `TaskSample`
**用途批注：**
- 很适合作为你自己的 trace schema 参考
- 关键字段：
  - input
  - steps
  - answer
  - length
  - dataset
  - metadata
  - is_clean_correct

#### 4. `CounterfactualResult`
**用途批注：**
- 这是 clean/corrupt 配对数据结构的直接参考
- 非常适合迁移到你的 `clean_pairs.jsonl` / `corrupt_pairs.jsonl` schema
- 重点参考：
  - corruption_type
  - step_index
  - original_step
  - corrupted_step
  - token_count_delta
  - perplexity_ratio
  - validate()

#### 5. `CounterfactualAccuracyResult`
**用途批注：**
- 很适合拿来设计你自己的 per-pair result table
- 重点是把 clean / corrupt 的 correctness 与 probability delta 分开记录

#### 6. `compute_rsa(...)`
**用途批注：**
- 如果后面你要只加一个辅助分析，这里是最直接的 RSA 入口参考
- 当前阶段可先不完全照搬，但建议先理解其输入输出接口

#### 7. `TASResult`
**用途批注：**
- 可用于提前规划你自己的结果表字段
- 当前阶段哪怕不实现 TAS，也值得参考其结果容器如何组织

### 使用建议

- 这个 repo 更像“一体化单文件原型”
- 适合你：
  - 借数据结构
  - 借 counterfactual 组织方式
  - 借 metric 入口设计
- 不适合你：
  - 原样作为长期工程骨架直接复用
- 正确用法：
  - 把它当 **方法实现参考源**，不是最终项目结构模板

---

## B. 第二优先级：工程骨架与配置组织模板

## B1. `J1mL1/causal-latent-cot`

### 作用定位

这个 repo 不是当前 explicit CoT 任务的官方实现，但它的**工程结构很值得借**。

它最有价值的不是某个单独指标，而是：

- 路径组织
- config 组织
- experiment entry point 组织
- shell wrapper 组织
- plotting 组织
- external repo 对接组织

### 优先参考路径

#### 1. `J1mL1/causal-latent-cot/common/experiment_utils.py`
**用途批注：**
- 适合借作实验公共函数层的结构模板
- 你的项目里可对应：
  - run id
  - 输出目录生成
  - 实验日志初始化
  - 通用保存/加载

#### 2. `J1mL1/causal-latent-cot/common/path_utils.py`
**用途批注：**
- 非常适合借来管理项目路径与环境变量展开
- 当前项目里很适合用于：
  - `${PROJECT_ROOT}`
  - `${DATA_DIR}`
  - `${OUTPUT_DIR}`
  这类占位符路径系统

#### 3. `J1mL1/causal-latent-cot/common/model_interface.py`
**用途批注：**
- 适合作为模型抽象层模板
- 你的项目里可借用其思路，把不同 HF 模型统一成同一推理接口

#### 4. `J1mL1/causal-latent-cot/common/model_registry.py`
**用途批注：**
- 适合作为模型名到加载逻辑的注册表模板
- 当前项目后续加入多模型时非常实用

#### 5. `J1mL1/causal-latent-cot/common/metrics/grad_sensitivity.py`
**用途批注：**
- 不直接等于 NLDD
- 但它属于“指标模块独立化”的好例子
- 可参考如何把 metric 实现从 experiment 主逻辑中拆出去

#### 6. `J1mL1/causal-latent-cot/configs/rq2/explicit/*.yaml`
**用途批注：**
- 这是当前阶段最值得借的配置模板之一
- 因为它本身就是 explicit CoT 方向的配置组织方式
- 可直接启发你设计：
  - `gsm8k-llama3.1-8b.yaml`
  - `gsm8k-gemma2-9b.yaml`
  - `gsm8k-deepseek-6.7b.yaml`

#### 7. `J1mL1/causal-latent-cot/experiments/rq2/run_explicit_causal_graph.py`
**用途批注：**
- 虽然任务不同，但这是 explicit reasoning 主实验入口的直接样板
- 可参考如何做：
  - 读取 config
  - 调模型
  - 调 metric
  - 保存结果

#### 8. `J1mL1/causal-latent-cot/scripts/inference/infer.sh`
**用途批注：**
- 适合作为批处理入口脚本风格参考
- 可改造成你的：
  - `run_generate_traces.sh`
  - `run_build_clean_pairs.sh`
  - `run_compute_nldd.sh`

#### 9. `J1mL1/causal-latent-cot/scripts/inference/test_codi_infer.py`
**用途批注：**
- 虽然是 latent 方向，但它展示了“单模型 inference smoke test”应该如何独立出来
- 你当前项目也应该有类似：
  - `test_llama31_infer.py`
  - `test_gemma2_infer.py`

#### 10. `J1mL1/causal-latent-cot/scripts/plot/python/`
**用途批注：**
- 很适合作为画图模块独立化的组织参考
- 你的项目里对应：
  - plot_accuracy_length.py
  - plot_nldd_by_k.py
  - plot_nldd_by_relative_position.py

#### 11. `J1mL1/causal-latent-cot/scripts/plot/r/`
**用途批注：**
- 如果后面要做论文图风格化，可以参考
- 当前阶段主要借它的“plotting 也是独立子系统”这个组织思想

### 这个 repo 最值得借的不是算法，而是工程习惯

你当前项目最适合直接继承它的：

- `common/`
- `configs/`
- `experiments/`
- `scripts/`
- `outputs/`

这种分层思路。

---

## C. 来自 `causal-latent-cot` README 的上游外部依赖线索

该 repo README 里明确列出了其外部依赖仓库组织方式，可作为你将来扩展时的参考。

### 1. `external/coconut`
**用途批注：**
- latent CoT 扩展用
- 与当前 explicit CoT 第一阶段关系较弱

### 2. `external/codi`
**用途批注：**
- 如果后面你要把 explicit CoT / NLDD 框架迁移到 latent CoT 研究线，可作为上游模型代码入口

### 3. `external/prontoqa`
**用途批注：**
- 如果后面你要补 PrOntoQA 任务，可参考其数据组织

### 4. `external/sim-cot`
**用途批注：**
- latent CoT 扩展用

---

## D. 当前项目的实际依赖优先级

## D1. 必看

1. `PKU-ML/CoT-Length/real/eval_math500_cot_length.py`
2. `PKU-ML/CoT-Length/real/run_math500.py`
3. `donald-ye/NLDD/pipeline.py`
4. `J1mL1/causal-latent-cot/configs/rq2/explicit/*.yaml`
5. `J1mL1/causal-latent-cot/common/model_interface.py`
6. `J1mL1/causal-latent-cot/common/path_utils.py`

## D2. 第二批再看

1. `PKU-ML/CoT-Length/real/eval_winogrande_cot_length.py`
2. `J1mL1/causal-latent-cot/experiments/rq2/run_explicit_causal_graph.py`
3. `J1mL1/causal-latent-cot/scripts/inference/infer.sh`
4. `J1mL1/causal-latent-cot/scripts/plot/python/`
5. `J1mL1/causal-latent-cot/common/experiment_utils.py`

## D3. 后续扩展再看

1. `PKU-ML/CoT-Length/synthetic/*`
2. `J1mL1/causal-latent-cot/common/metrics/grad_sensitivity.py`
3. `J1mL1/causal-latent-cot/scripts/inference/test_codi_infer.py`
4. `external/codi`
5. `external/prontoqa`

---

## E. 直接落到你项目中的映射建议

### 1. trace 生成层
优先参考：
- `PKU-ML/CoT-Length/real/run_math500.py`
- `PKU-ML/CoT-Length/real/run_winogrande.py`
- `J1mL1/causal-latent-cot/common/model_interface.py`

### 2. optimal-length 聚合层
优先参考：
- `PKU-ML/CoT-Length/real/eval_math500_cot_length.py`
- `PKU-ML/CoT-Length/real/eval_winogrande_cot_length.py`

### 3. clean/corrupt 数据结构层
优先参考：
- `donald-ye/NLDD/pipeline.py` 里的 `TaskSample`
- `donald-ye/NLDD/pipeline.py` 里的 `CounterfactualResult`
- `donald-ye/NLDD/pipeline.py` 里的 `CounterfactualAccuracyResult`

### 4. config 层
优先参考：
- `J1mL1/causal-latent-cot/configs/rq2/explicit/*.yaml`
- `donald-ye/NLDD/pipeline.py` 里的 `ExperimentConfig`

### 5. plotting 层
优先参考：
- `J1mL1/causal-latent-cot/scripts/plot/python/`
- `J1mL1/causal-latent-cot/scripts/plot/r/`

### 6. 运行入口层
优先参考：
- `J1mL1/causal-latent-cot/experiments/rq2/run_explicit_causal_graph.py`
- `J1mL1/causal-latent-cot/scripts/inference/infer.sh`

---

## F. 当前最合理的借法

不是把某一个 repo 整体 fork 过来直接改。

而是分层借：

- 从 `CoT-Length` 借 **真实任务上的长度生成与长度聚合**
- 从 `NLDD` 借 **clean/corrupt 数据模型与指标思路**
- 从 `causal-latent-cot` 借 **工程目录、config 组织、脚本入口、plot 组织**

这三者拼起来，最适合当前项目第一阶段。
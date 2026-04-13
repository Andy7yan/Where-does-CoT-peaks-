# Stage 1: Minimum Viable Pipeline (MVP) & PoC

> **Important Note:** This `README.md` serves as an operational summary of the Stage 1 experiment. For exact implementation details, mathematical definitions (like Logit Difference and TAS), data schemas, and configuration parameters, please refer to the primary specification document: **`stage1-spec.md`**. If there are any discrepancies, `stage1-spec.md` strictly overrides this file.

## 🎯 Objective
Stage 1 is the Minimum Viable Pipeline (MVP) utilizing a single model (`meta-llama/Llama-3.1-8B-Instruct`) and a single dataset (GSM8K). The goal is to validate the technical feasibility of measuring the accuracy-optimal CoT length ($L^*$) and the NLDD reasoning horizon ($k^*$) simultaneously on the same data, yielding interpretable preliminary conclusions.

## 🧠 Research Question & Hypotheses
**RQ:** When measuring the accuracy-optimal CoT length ($L^*$) and the NLDD reasoning horizon ($k^*$) simultaneously on the same model and dataset, what is the numerical relationship between the two?

* **H1 ($L^* \approx k^*$): Consistent.** The behavioral inflection point coincides with the mechanistic horizon. Steps beyond $L^*$ neither improve accuracy nor possess causal influence.
* **H2 ($L^* < k^*$): Overthinking.** Accuracy has dropped, but later steps still show causal contribution. The model is faithfully executing harmful extra reasoning.
* **H3 ($L^* > k^*$): Indirect Mechanisms.** NLDD shows steps have lost causal contribution, but accuracy continues to rise. Later steps likely act through mechanisms invisible to NLDD (e.g., format anchoring, attention stabilization).

---

## 🔬 Pipeline Phases

The experiment is broken down into four distinct chronological phases. 

### 1. Pilot Phase
Before running the full computation, a low-cost pilot run (50–100 questions) is executed to validate the end-to-end pipeline. 
* **Goal:** Verify that our In-Context Learning (ICL) exemplars successfully induce a wide variance in reasoning length without using hard constraints (like "use exactly 5 steps"). 
* **Checks:** Ensures step parsing works, answer extraction doesn't fail, and that the text corruption methods produce grammatically coherent and valid alterations before burning GPU hours on the full dataset.

### 2. Data Phase (Trace Generation)
This phase generates the raw reasoning traces and establishes the natural length distributions.
* **Sampling Strategy:** We use 5 distinct groups of ICL exemplars, ranging from minimal to highly verbose solutions. Each group is paired with a specific temperature gradient (e.g., $0.0$ for minimal, up to $0.7$ for verbose). This naturally coaxes the model into generating a wide spectrum of Chain-of-Thought lengths.
* **Processing:** Model outputs are strictly segmented by **newlines (`\n`)** to define individual "steps." The final answer is extracted by parsing for `####` or `The answer is`.
* **Metadata:** Traces are evaluated for correctness, and the question's overall "difficulty" is calculated post-hoc based on its failure rate across all samples.

### 3. Analysis Phase
This is the core of the experiment, where we compute our two primary variables: $L^*$ and $k^*$.

* **Accuracy Aggregation ($L^*$):** We bin all generated traces entirely by their actual step count (`actual_num_steps`). By calculating the mean accuracy and standard error for each length bin, we can locate $L^*$—the specific step count that maximizes accuracy for the model.
* **NLDD Measurement ($k^*$):**
  For a representative subset of *correct* traces, we measure the causal contribution of each step using the Normalized Logit Difference Drop (NLDD).
  * **Corruption Fallback:** To test a step's causal weight, we perturb it using a two-tier system. Tier 1 attempts to mathematically alter a numeric result or operand (e.g., adding a delta or applying a multiplier). If no numbers are present, Tier 2 attempts to swap arithmetic operators (e.g., changing `+` to `-`).
  * **Horizon Location:** By comparing the model's confidence (Logit Difference) on the clean trace versus the corrupted trace, we find the NLDD peak. The step index where this peak occurs defines $k^*$.
* **TAS Measurement (Secondary Metric):**
  During the NLDD forward passes, we also hook into the middle transformer layer to extract hidden states at the end of each step. This allows us to calculate Trajectory Analysis Shifts (TAS) to see if the internal representation "plateaus" (indicating the model has made up its mind) at a similar point to $k^*$.

### 4. Plotting & Aggregation Phase
The final phase aggregates the CSV outputs from the Data and Analysis phases to generate 7 core visualizations. This includes plotting Accuracy vs. Length, NLDD surface heatmaps, and plotting $k^*(L)$ directly against $L$ to visually test our three hypotheses.

---

## 🚀 Future Extensions (Stage 2)
Stage 1 is intentionally scoped down. Stage 2 will introduce:
* Stronger models (e.g., `Qwen2.5-7B-Instruct`).
* More complex datasets (MATH all levels).
* Step Complexity Proxies (annotating the structural features of each step via rules and LLM-judges).
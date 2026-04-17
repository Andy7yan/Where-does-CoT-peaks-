# peak-CoT

This project studies a simple question about **Chain-of-Thought (CoT)** reasoning in language models:

> when a model writes out more reasoning steps, does that extra reasoning really help, or does it eventually become unnecessary or even harmful?

The project is motivated by two nearby ideas.

- One line of work asks about **behavioural optimal length**: for a given task, is there a CoT length at which accuracy is highest?
- Another line of work asks about **mechanistic faithfulness**: which reasoning steps are still causally doing work for the final answer?

In this repository, those ideas are connected through a shared pool of model-generated reasoning traces.

## What the key terms mean

### Chain-of-Thought (CoT)
A Chain-of-Thought is the model's written reasoning process before its final answer. For a maths word problem, this might look like a short sequence of intermediate calculations.

### Trace
A **trace** is one complete model output for one question, including:

- the reasoning text,
- the final answer line,
- the extracted numeric answer,
- whether that answer is correct,
- and basic metadata such as which prompt style produced it.

This project generates many traces per question rather than relying on a single sample.

### Step
A **step** is one non-empty line in the reasoning text. The final answer line is stored separately and does not count as a reasoning step. The number of reasoning lines is stored as `actual_num_steps`. fileciteturn5file15

### ICL prompts
The project uses **in-context learning (ICL)** prompts: few-shot exemplars that encourage the model to reason in different styles. In practice, this is how the project gets shorter and longer traces without directly forcing a fixed number of steps. The current formal run uses four prompt groups: `icl_short`, `icl_medium`, `icl_detailed`, and `icl_verbose`. fileciteturn5file15

### L*
`L*` means the **behavioural optimum**: the CoT length at which performance is best. The "optimal length" paper argues that longer CoT is not always better; accuracy often follows an inverted-U pattern, so there can be a best intermediate length rather than "the longer the better". fileciteturn4file0turn5file5

### NLDD and k*
**NLDD** is a step-level faithfulness metric from the "Mechanistic Evidence for Faithfulness Decay" paper. It works by corrupting one reasoning step, truncating what comes after it, and measuring how much the model's confidence in the correct answer drops. A larger drop means that step mattered more. The paper uses this to define a **reasoning horizon** `k*`: the step position where causal contribution is strongest before later steps start contributing less. fileciteturn4file0turn4file2

## Why this project exists

CoT is often treated as both:

- a way to improve reasoning performance, and
- a window into how the model reached its answer.

But those are not the same thing. A longer reasoning chain might improve performance, hurt performance, or simply add text that sounds plausible after the answer is already effectively determined. Related work also argues that CoT should not be judged only by whether it explicitly says every influential factor; some apparent "unfaithfulness" may instead be incomplete verbalisation of a more distributed internal process. fileciteturn5file14

So the broader aim of peak-CoT is to compare two questions on the **same traces**:

- **Behavioural question:** how long should a trace be for best accuracy?
- **Faithfulness question:** up to which step is the written reasoning still causally useful?

## What Stage 1 currently is

The most important status update is simple:

**Stage 1 is currently frozen as a full trace-generation run, not as a full analysis pipeline.** The v5 spec explicitly says the current formal run is for generating the complete trace corpus that later data-phase and analysis-phase work will consume. It does **not** yet freeze difficulty grouping, coarse analysis, or analysis metrics as part of the current executable run. fileciteturn5file15

So, at the moment, this repository's formal job is:

1. load the full GSM8K-Platinum test split,
2. run the model with four ICL prompt styles,
3. collect multiple traces for every question,
4. parse them into a consistent trace format,
5. save them as the canonical corpus for later analysis. fileciteturn5file15

## Current Stage 1 setup

The current formal run is deliberately narrow.

- **Dataset:** `madrylab/gsm8k-platinum`, `main`, `test` split, full coverage. fileciteturn5file15
- **Prompt groups:** `icl_short`, `icl_medium`, `icl_detailed`, `icl_verbose`. fileciteturn5file15
- **Samples per group:** `5`. fileciteturn5file15
- **Temperature:** `0.6`, shared globally rather than varying by prompt group. fileciteturn5file15
- **Max new tokens:** `1024`. fileciteturn5file15
- **Total traces per question:** `20`. fileciteturn5file15

Each saved trace records at least the question ID, question text, gold answer, prompt ID, raw completion, parsed steps, `actual_num_steps`, final answer line, extracted answer, correctness, extraction failure flag, token count, and timestamp. fileciteturn5file15

## What comes after this

Later stages are intended to use the generated corpus to study relationships between CoT length and faithfulness. In earlier planning, the README framed this as a direct comparison between `L*` and `k*` on shared traces. fileciteturn4file0

The v5 spec, however, intentionally stops short of freezing those later steps. It only reserves the main analysis panel that will eventually include metrics such as `accuracy(L)`, `L*`, `NLDD`, `k*(L)`, and `TAS`. fileciteturn5file15

So the clean way to read the repository today is:

- **conceptually**, it is a project about CoT length, overthinking, and faithfulness;
- **operationally**, Stage 1 is currently the data-collection layer that produces the trace corpus needed for those questions.

## One-sentence summary

This repository builds a clean corpus of short-to-long reasoning traces on GSM8K-Platinum so that later analysis can ask not just **how much** a model thinks, but **which parts of that thinking actually matter**.

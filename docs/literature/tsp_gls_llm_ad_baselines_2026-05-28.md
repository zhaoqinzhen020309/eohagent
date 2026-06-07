# TSP-GLS LLM4AD Baselines

Date: 2026-05-28

This note records public LLM-based automatic heuristic design results that are relevant to our TSP-GLS experiments. The main purpose is to separate directly comparable baselines from papers that are only useful as method inspiration.

## Directly Relevant Results

### EoH

Paper: Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Models

Source: https://openreview.net/pdf?id=BwAkaxqiLB

Task: TSP Guided Local Search, synthetic TSP100/200/500.

Reported gaps:

| Method | TSP100 | TSP200 | TSP500 |
|---|---:|---:|---:|
| EoH | 0.367 | 0.549 | 0.922 |

Use in our paper: baseline for the original single-function TSP-GLS setting.

### MCTS-AHD

Paper: MCTS-AHD: Monte Carlo Tree Search-Based Automated Heuristic Design

Source: https://openreview.net/pdf?id=Do1OdZzYHr

Task: TSP Guided Local Search, synthetic TSP100/200/500/1000.

Reported gaps:

| Method | TSP100 | TSP200 | TSP500 | TSP1000 |
|---|---:|---:|---:|---:|
| EoH | 0.332 | 0.549 | 0.958 | 1.393 |
| ReEvo | 0.319 | 0.506 | 1.039 | 1.715 |
| MCTS-AHD | 0.238 | 0.478 | 0.885 | 1.224 |

Use in our paper: direct synthetic TSP-GLS comparison if we reproduce the same generated-instance protocol.

### HiFo-Prompt

Paper: Enhancing LLM-based Automatic Heuristic Design with Hindsight and Foresight

Source: https://openreview.net/pdf?id=imSLzfZ6av

Task: TSP Guided Local Search, synthetic TSP100/200/500.

Reported gaps:

| Method | TSP100 | TSP200 | TSP500 |
|---|---:|---:|---:|
| EoH | 0.367 | 0.549 | 0.922 |
| ReEvo | 0.257 | 0.326 | 0.818 |
| HiFo | 0.180 | 0.302 | 0.741 |

Use in our paper: strong direct synthetic TSP-GLS comparison.

### MoH

Paper: Generalizable Heuristic Generation Through Large Language Models with Meta-Optimization

Source: https://openreview.net/pdf?id=tIQZ7pVN6S

Task 1: TSPLib small TSP-GLS instances.

Reported average gaps:

| Method | Average gap |
|---|---:|
| EoH-GLS | 0.21 |
| HSEvo-GLS | 0.22 |
| MoH-GLS | 0.19 |

Task 2: generated TSP50/100/200/500 KGLS.

Reported gaps:

| Method | TSP50 | TSP100 | TSP200 | TSP500 | Average |
|---|---:|---:|---:|---:|---:|
| EoH-GLS | 1.046 | 0.367 | 0.549 | 0.922 | 0.721 |
| ReEvo-KGLS | 0.462 | 0.257 | 0.326 | 0.818 | 0.466 |
| HSEvo-KGLS | 0.531 | 0.260 | 0.326 | 0.784 | 0.475 |
| MCTS-AHD-KGLS | 0.309 | 0.254 | 0.302 | 0.772 | 0.409 |
| MoH-KGLS | 0.231 | 0.225 | 0.294 | 0.812 | 0.391 |

Use in our paper: currently the most important direct comparison source. If our evaluation protocol matches their TSPLib small setting, then a serious target is average gap below 0.19. If our protocol differs, we need to rerun their best public code or report protocol differences clearly.

## Method Inspiration, Not Direct Numeric Baseline Yet

### A2DEPT

Paper: Automated Algorithm Design for Efficient Tree Search with Evolutionary Program Trees

Source: https://arxiv.org/abs/2604.24043v1

Relevant point: it argues that fixed-template LLM algorithm design is limited because the model can tune components but cannot redesign solver structure. This supports our shift from single `update_edge_distance` evolution to constrained GLS architecture design.

Direct TSP-GLS comparability: not directly comparable unless we implement and evaluate the same TSP-GLS task.

### Clade-AHD

Paper: Clade-AHD: Evolutionary Generative Framework for Automated Heuristic Design

Source: https://arxiv.org/abs/2602.00549

Relevant point: emphasizes structured evolutionary generation and validation for heuristic design.

Direct TSP-GLS comparability: needs further check; do not use as a numeric TSP-GLS baseline until a matching table is confirmed.

### PathWise

Paper: PathWise: Tree-Search-Guided Reasoning Path Optimization for LLM-based Automatic Heuristic Design

Source: https://arxiv.org/abs/2601.20539

Relevant point: treats heuristic design as a sequential search process instead of independent one-shot code generation. This supports our architecture tree and evidence-tracking design.

Direct TSP-GLS comparability: needs further check; do not use as a numeric TSP-GLS baseline until a matching table is confirmed.

### AHD Agent

Paper: Automated Design of Agentic Systems for Heuristic Optimization

Source: https://arxiv.org/abs/2605.08756

Relevant point: supports active environment interaction and state-dependent evidence gathering instead of fixed prompt context.

Direct TSP-GLS comparability: method inspiration only unless matching benchmark numbers are found.

## Current Takeaways For Our Target

1. Our old DeepSeek single-function result around 0.232 TSPLib average gap is close to public EoH-level results, but it does not clearly beat MoH's reported 0.19 average on the small TSPLib table.
2. The architecture-level version should not claim success merely by beating our weaker Qwen v3/v5 runs. It should target the public 0.19 to 0.22 TSPLib range first.
3. The safest publishable comparison plan is:
   - reproduce our original single-function EoH-style setup;
   - run constrained GLS architecture design under the same evaluator;
   - add ablations for each opened architecture layer;
   - compare against public EoH, ReEvo, HSEvo, MCTS-AHD, HiFo, and MoH numbers only when the benchmark protocol matches.
4. The most defensible innovation is not "more skills"; it is opening high-value GLS architecture layers while keeping execution contracts strict and auditable.

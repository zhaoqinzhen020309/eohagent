# Repository Guidelines

## Project Structure & Module Organization

This is a Python codebase for Evolution of Heuristics experiments. The primary package lives in `eoh/src/eoh/`, with core modules split into `llm/`, `methods/`, `problems/`, `utils/`, and `llm_local_server/`. Variant implementations are in `eoh_OPRO/`, `eoh_RAG/`, `eoh_compress/`, and `eoh_ablation/ablation*/`; synchronize changes only when behavior should match. `examples/` contains runnable experiments, datasets, populations, and evaluations. `baseline/funsearch/` is an independent FunSearch baseline. `docs/` is for documentation.

## Build, Test, and Development Commands

- `cd eoh; python -m pip install -e .` installs the main package in editable mode. It requires Python 3.10+.
- `python eoh/src/eoh/test/run.py` runs the bundled EoH smoke script, but edit runtime parameters first and supply your own LLM endpoint/key.
- `python examples/<example_name>/runEoH.py` runs an example experiment when that directory provides `runEoH.py`.
- `python -m pip install -r baseline/funsearch/requiremetns.txt` installs FunSearch baseline dependencies. The filename is misspelled in the repository; use it as-is.

There is no central build system or root `pytest` configuration.

## Coding Style & Naming Conventions

Use standard Python style: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and uppercase constants. When adding problems, place logic under `src/eoh/problems/...`, prompts near the implementation, and loaders in `get_instance.py`. Avoid committing `__pycache__/` and large generated outputs unless they are required fixtures or published results.

## Testing Guidelines

No formal unit-test suite is present. For code changes, add focused tests or reproducible smoke scripts near the affected package or example. Prefer deterministic, small datasets. Before submitting, run the smallest relevant command, such as `python eoh/src/eoh/problems/optimization/bp_online/run.py` or the affected `examples/.../runEoH.py`, and record model, API, or dataset assumptions.

## Experiment Integrity

Do not add fallback mechanisms to the `eohagent/` research workflow. If an LLM decision, generation planner, skill call, JSON parse, parent lookup, candidate generation, repair, or generation reflection fails, fail closed and record the failure. Do not replace a failed step with a generic plan, default skill, inferred skill name, parent-code repair, synthetic notebook summary, or placeholder delta. Failed offspring may be rejected, and a failed generation may stop without producing offspring; this is preferable to silently running a different workflow and contaminating the experiment.

## Agentic AHD Research Direction

For `eohagent/`, the preferred next research direction is **evidence-grounded closed-loop experimentation for automated heuristic design**, not a PoH-style MCTS continuation and not component stacking. The goal is to make the LLM act as a research agent that forms hypotheses, designs controlled experiments, reads mechanism-level evidence, revises future plans, and stops or changes unproductive routes.

Treat LangGraph, LangMem, QD archives, and harness modules as infrastructure only. Do not call them effective because they are installed, invoked, or logged. They are research-relevant only if they change a real decision point: what evidence the planner sees, what experiment is proposed, what gets rejected before evaluation, what enters the next population/archive, what is written to memory, or when a route is stopped.

When redesigning this workflow, address the observed failures directly:

- Late-stage search collapses into micro-parameter edits. The planner must justify whether an experiment is parameter tuning, structural pipeline change, conditional branch insertion, failure diagnosis, or validation, using concrete evidence.
- Training mean objective saturates and does not reliably predict final TSPLIB evaluation. Do not only add another scalar penalty; expose richer training-run diagnostics so the agent can reason about failure mechanisms.
- Memory and reflection are not useful unless they are structured, evidence-linked, and retrieved into future planning decisions. Avoid long free-form summaries.
- Harness review must verify evidence, intent, controlled change, duplicate status, and semantic effect. A non-empty `evidence_refs` field is not enough; the referenced evidence must actually exist and support the claimed intent.
- Typed solver trees are useful because they provide a structured action space, but a fixed tree schema still limits pipeline discovery. Any expansion of the tree space must be justified by solver-control evidence and remain auditable.

Use the following papers as design signals, not as methods to copy blindly:

- LLM-AutoSciLab: closed-loop discovery couples hypothesis generation, experiment selection, and mechanism refinement.
- Agents Explore but Agents Ignore: agents may discover relevant evidence but fail to use it; require explicit evidence-use checks.
- The Observability Gap: output-level feedback is insufficient when failures arise from hidden execution state; add intermediate mechanism observability.
- SWE-agent and CODESTRUCT: agent performance depends on task-specific interfaces and structured action spaces.
- Reflexion, Voyager, AgentDebug, and AgentRx: memory and feedback are useful only when grounded in concrete trial evidence, execution errors, failure localization, and actionable corrections.

Before implementing a new `eohagent/` iteration, produce a short design that maps each proposed mechanism to:

- the experimental fact it addresses;
- the paper-derived transferable mechanism;
- the exact code decision point it changes;
- the logs that will prove it is semantically active;
- the early-stop condition showing it is failing.

Do not claim a new agent, harness, memory, QD, or pipeline mechanism is implemented correctly unless semantic logs show that it changes the main workflow rather than only recording metadata.

## Commit & Pull Request Guidelines

This checkout has no Git history, so no repository-specific convention can be inferred. Use concise imperative commits, for example `Fix bp_online evaluator timeout handling`. Pull requests should describe the changed algorithm or experiment path, list commands run, note dataset/model requirements, and include result summaries when they clarify behavior.

## Security & Configuration Tips

Do not commit real API keys, model tokens, or private endpoints. Replace hardcoded credentials with environment variables or ignored local config files. Large archives, model weights, and generated outputs should be linked or documented unless intentionally versioned.

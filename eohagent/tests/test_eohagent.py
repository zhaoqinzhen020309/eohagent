from copy import deepcopy
import json
import requests
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eohagent.core.config import load_config
from eohagent.core.candidate import Candidate
from eohagent.core.engine import ExpertEoHEngine
from eohagent.core.experiment_guard import build_smoke_config
from eohagent.core.evaluator_bridge import build_evaluator
from eohagent.core.mechanism import canonicalize_family
from eohagent.core.population import PopulationManager
from eohagent.core.qd_archive import QDArchive
from eohagent.core.route_manager import (
    EMITTER_TYPES,
    EmitterScheduler,
    RouteManager,
    behavior_descriptor_for_candidate,
)
from eohagent.core.result_store import ResultStore
from eohagent.core.selection import CellRoundRobinSelector, RankSelector
from eohagent.core.langgraph_workflow import LangGraphWorkflowRunner
from eohagent.bootstrap.baseline_loader import BaselineLoader
from eohagent.harness.agent_state import AgentState
from eohagent.harness.commit_gate import CommitGate
from eohagent.harness.context_runtime import ContextOverflowError, ContextRuntime
from eohagent.harness.experiment_review import ExperimentReviewer
from eohagent.harness.generation_planner import GenerationPlanner
from eohagent.harness.skill_registry import SkillRegistry
from eohagent.harness.tool_router import ToolRouter
from eohagent.llm.client import MockLLMClient, OpenAICompatibleClient
from eohagent.llm.json_call import JSONCall
from eohagent.llm.prompt_compression import PromptCompressionError, PromptCompressionManager
from eohagent.memory.budget import bounded_notebook
from eohagent.memory.expert_notebook import ExpertNotebook
from eohagent.memory.memory_manager import MemoryManager
from eohagent.memory.provider import MemoryProviderError
from eohagent.problems.tsp_gls_arch_spec.spec_schema import apply_layer_change, build_arch_spec_code, canonical_arch_spec


def base_arch_spec() -> dict:
    return {
        "distance_update_policy": {
            "type": "linear_usage",
            "strength": 0.1,
            "usage_power": 1.0,
            "length_weight": 0.0,
            "cap_multiplier": 20.0,
        },
        "penalty_policy": {
            "edge_selection": "top_gap",
            "top_k": 8,
            "increment": 1.0,
            "decay": 0.0,
        },
        "escape_policy": {
            "trigger": "constant",
            "moves": 1,
            "period": 50,
            "stagnation_threshold": 20,
            "size_threshold": 150,
            "max_moves": 1,
        },
        "restart_policy": {
            "action": "reset_to_best",
            "trigger": "periodic",
            "period": 50,
            "stagnation_threshold": 30,
            "penalty_decay": 0.0,
        },
        "schedule_policy": {
            "strength_schedule": "constant",
            "early_iterations": 100,
            "late_multiplier": 1.0,
            "stagnation_multiplier": 1.0,
            "size_threshold": 150,
            "size_multiplier": 1.0,
        },
    }


def arch_spec_child(layer: str, layer_value: dict) -> dict:
    spec = deepcopy(base_arch_spec())
    spec[layer] = deepcopy(layer_value)
    return canonical_arch_spec(spec)


def base_solver_tree() -> dict:
    return {
        "type": "repeat",
        "budget": "main_loop",
        "body": {
            "type": "sequence",
            "steps": [
                {
                    "type": "local_search",
                    "neighborhood": "two_opt_relocate",
                    "first_improvement": False,
                },
                {
                    "type": "penalty_step",
                    "repeat": 1,
                    "distance_update": {
                        "type": "linear_usage",
                        "strength": 0.1,
                        "usage_power": 1.0,
                        "length_weight": 0.0,
                        "cap_multiplier": 20.0,
                    },
                    "penalty": {
                        "edge_selection": "top_gap",
                        "top_k": 8,
                        "increment": 1.0,
                        "decay": 0.0,
                    },
                    "schedule": {
                        "strength_schedule": "constant",
                        "early_iterations": 100,
                        "late_multiplier": 1.0,
                        "stagnation_multiplier": 1.0,
                        "size_threshold": 150,
                        "size_multiplier": 1.0,
                    },
                },
                {
                    "type": "local_search",
                    "neighborhood": "two_opt_relocate",
                    "first_improvement": False,
                },
                {
                    "type": "if_stagnated",
                    "threshold": 30,
                    "then": {
                        "type": "restart",
                        "action": "reset_to_best",
                    },
                },
            ],
        },
    }


class CaptureClient:
    def __init__(self):
        self.payloads = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.payloads.append({"raw_prompt": prompt, "system": system})
        return "{captured}\n```python\ndef score(item, bins):\n    return item - bins\n```"


class SequencedJSONClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        if len(self.prompts) <= len(self.responses):
            response = self.responses[len(self.prompts) - 1]
            if isinstance(response, BaseException):
                raise response
            return response
        return self.responses[-1]


class SequencedSelector:
    def __init__(self, parents):
        self.parents = list(parents)
        self.index = 0

    def select_parent(self, population):
        parent = self.parents[self.index % len(self.parents)]
        self.index += 1
        return parent


class ProseCodeClient:
    def complete(self, prompt: str, system: str = "") -> str:
        return """
Here is a small variation.
```python
import numpy as np

def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):
    distances = np.array([distance_matrix[current_node][i] for i in univisited_nodes])
    return univisited_nodes[int(np.argmin(distances))]
```
"""


class GLSProseCodeClient:
    def complete(self, prompt: str, system: str = "") -> str:
        return """
{Usage-weighted GLS penalty update.}
```python
import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    for i in range(n):
        a = int(local_opt_tour[i])
        b = int(local_opt_tour[(i + 1) % n])
        penalty = 1.0 + 0.1 * (1.0 + edge_n_used[a, b])
        updated_edge_distance[a, b] *= penalty
        updated_edge_distance[b, a] *= penalty
    return updated_edge_distance
```
"""


class ContextCompactionClient:
    def __init__(self):
        self.prompts = []
        self.systems = []

    def complete(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        if "context_compaction skill" in system:
            return (
                '{"resume_summary": "Continue from compacted TSP-GLS state.", '
                '"keep_rules": ["preserve signature"], '
                '"avoid_rules": ["avoid constructive selectors"], '
                '"promising_mechanisms": ["usage weighted penalties"], '
                '"failed_mechanisms": [], '
                '"active_hypotheses": ["normalize penalties"], '
                '"candidate_refs": ["p"], '
                '"next_search_hint": "try a compact penalty variation"}'
            )
        return """
{Compact usage penalty variation.}
```python
import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    for i in range(len(local_opt_tour)):
        a = int(local_opt_tour[i])
        b = int(local_opt_tour[(i + 1) % len(local_opt_tour)])
        updated_edge_distance[a, b] *= 1.0 + 0.05 * (1.0 + edge_n_used[a, b])
        updated_edge_distance[b, a] *= 1.0 + 0.05 * (1.0 + edge_n_used[a, b])
    return updated_edge_distance
```
"""


class OverflowCompactionClient:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, system: str = "") -> str:
        self.calls += 1
        return '{"resume_summary": "still compacted", "keep_rules": [], "avoid_rules": [], "promising_mechanisms": [], "failed_mechanisms": [], "active_hypotheses": [], "candidate_refs": [], "next_search_hint": ""}'


class ParentCodeRepairClient:
    def __init__(self, parent_code):
        self.parent_code = parent_code

    def complete(self, prompt: str, system: str = "") -> str:
        return "{revert to parent}\n```python\n" + self.parent_code + "\n```"


class FamilyChangingRepairClient:
    def complete(self, prompt: str, system: str = "") -> str:
        return """
{Hypothesis: Switch to a length weighted repair.
Mechanism family: length_weighted_penalty
Parent delta: replace usage-only penalty with length-aware penalty.
Expected effect: repair syntax while changing mechanism.
Preserve: function signature.
Change: mechanism family.}
```python
import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = edge_distance.copy()
    for i in range(len(local_opt_tour)):
        a = int(local_opt_tour[i])
        b = int(local_opt_tour[(i + 1) % len(local_opt_tour)])
        updated_edge_distance[a, b] *= 1.0 + 0.01 * edge_distance[a, b]
        updated_edge_distance[b, a] *= 1.0 + 0.01 * edge_distance[b, a]
    return updated_edge_distance
```
"""


class PreparedPromptClient:
    def __init__(self):
        self.prepare_calls = 0
        self.completed_prompts = []
        self.last_usage = {"prompt_cache_hit_tokens": 3, "prompt_cache_miss_tokens": 7}
        self.last_prompt_compression = {}

    def prepare_prompt(self, prompt: str, purpose: str = "llm_call") -> str:
        self.prepare_calls += 1
        self.last_prompt_compression = {"purpose": purpose, "compressed": True}
        return prompt.replace("RAW_PROMPT", "PREPARED_PROMPT")

    def complete(self, prompt: str, system: str = "") -> str:
        self.completed_prompts.append({"prompt": prompt, "system": system})
        return "done"


class FakeOpenAIResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "usage": {
                "prompt_cache_hit_tokens": 11,
                "prompt_cache_miss_tokens": 13,
            },
            "choices": [{"message": {"content": "{}"}}],
        }


class EohAgentTests(unittest.TestCase):
    def test_config_loads_defaults(self):
        config = load_config("eohagent/config/bp_online.yaml")
        self.assertEqual(config["project"]["problem"], "bp_online")
        self.assertEqual(config["llm"]["provider"], "mock")

    def test_engine_rejects_dirty_output_dir_without_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/bp_online.yaml")
            output_dir = Path(tmp) / "run"
            output_dir.mkdir()
            (output_dir / "candidate_store.json").write_text("{}", encoding="utf-8")
            config["project"]["output_dir"] = str(output_dir)
            with self.assertRaisesRegex(RuntimeError, "existing experiment state"):
                ExpertEoHEngine(config)

    def test_smoke_config_uses_isolated_output_dir(self):
        config = load_config("eohagent/config/bp_online.yaml")
        config["project"]["output_dir"] = "eohagent/results/formal_run"
        smoke_config = build_smoke_config(config)
        self.assertIn("_smoke", smoke_config["project"]["output_dir"])
        self.assertEqual(smoke_config["project"]["formal_output_dir"].replace("\\", "/").endswith("formal_run"), True)
        self.assertEqual(smoke_config["run"]["mode"], "smoke")

    def test_json_call_extracts_relaxed_qwen_style_object(self):
        data = JSONCall.extract_json("{next_skill: 'controlled_variation', reason: 'ok', arguments: {}}")
        self.assertEqual(data["next_skill"], "controlled_variation")

    def test_json_call_extracts_escaped_json_string(self):
        data = JSONCall.extract_json('\\"{\\\\\\"next_skill\\\\\\": \\\\\\"controlled_variation\\\\\\", \\\\\\"reason\\\\\\": \\\\\\"ok\\\\\\", \\\\\\"arguments\\\\\\": {}}\\"')
        self.assertEqual(data["next_skill"], "controlled_variation")

    def test_json_call_prepares_text_prompt_once_before_complete(self):
        client = PreparedPromptClient()
        json_call = JSONCall(client)
        response = json_call.complete_text_dynamic(
            lambda: "RAW_PROMPT",
            purpose="controlled_variation",
            system="STATIC_SYSTEM",
        )
        self.assertEqual(response, "done")
        self.assertEqual(client.prepare_calls, 1)
        self.assertEqual(client.completed_prompts[0]["prompt"], "PREPARED_PROMPT")
        self.assertEqual(client.completed_prompts[0]["system"], "STATIC_SYSTEM")
        self.assertEqual(json_call.last_usage["prompt_cache_hit_tokens"], 3)

    def test_prompt_compression_preserves_protected_code_fields(self):
        manager = object.__new__(PromptCompressionManager)
        manager.min_segment_chars = 1
        manager._compress_segment = lambda text: "COMPRESSED:" + text[:12]
        payload = {
            "parent_code": "def score(item, bins):\n    return item - bins",
            "expert_notebook": {"lesson": "long mechanism lesson that can be compressed"},
            "child_spec": base_arch_spec(),
        }
        compressed = manager._compress_json_like(payload, protected=False)
        self.assertEqual(compressed["parent_code"], payload["parent_code"])
        self.assertEqual(compressed["child_spec"], payload["child_spec"])
        self.assertTrue(compressed["expert_notebook"]["lesson"].startswith("COMPRESSED:"))

    def test_json_prompt_places_stable_prefix_before_dynamic_payload(self):
        text = PromptCompressionManager.render_json_user(
            {"purpose": "generation_experiment_plan", "generation": 3},
            ["generation_goal", "offspring_plans"],
        )
        self.assertLess(text.index("STABLE_GENERATION_PLAN_RULES"), text.index("DYNAMIC_INPUT_JSON:"))
        self.assertIn('"generation":3', text)

    def test_openai_client_fails_when_enabled_compressor_unavailable(self):
        config = {
            "llm": {
                "api_base": "https://api.deepseek.com",
                "api_key": "test-key",
                "model": "deepseek-v4-flash",
            },
            "prompt_compression": {"enabled": True},
        }
        with patch.dict("sys.modules", {"llmlingua": None}):
            with self.assertRaises(PromptCompressionError):
                OpenAICompatibleClient(config)

    def test_openai_client_sends_deepseek_thinking_and_records_cache_usage(self):
        config = {
            "llm": {
                "api_base": "https://api.deepseek.com",
                "api_key": "test-key",
                "model": "deepseek-v4-flash",
                "thinking": "disabled",
            },
            "prompt_compression": {"enabled": False},
        }
        with patch("eohagent.llm.client.requests.post", return_value=FakeOpenAIResponse()) as post:
            client = OpenAICompatibleClient(config)
            self.assertEqual(client.complete("hello", system="static"), "{}")
        body = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(client.last_usage["prompt_cache_hit_tokens"], 11)

    def test_openai_client_retries_transport_errors_three_times_before_success(self):
        config = {
            "llm": {
                "api_base": "https://api.deepseek.com",
                "api_key": "test-key",
                "model": "deepseek-v4-flash",
            },
            "prompt_compression": {"enabled": False},
        }
        side_effects = [
            requests.exceptions.SSLError("eof"),
            requests.exceptions.ConnectionError("reset"),
            requests.exceptions.Timeout("timeout"),
            FakeOpenAIResponse(),
        ]
        with patch("time.sleep") as sleep:
            with patch("eohagent.llm.client.requests.post", side_effect=side_effects) as post:
                client = OpenAICompatibleClient(config)
                self.assertEqual(client.complete("hello"), "{}")
        self.assertEqual(post.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    def test_openai_client_raises_after_three_transport_retries_fail(self):
        config = {
            "llm": {
                "api_base": "https://api.deepseek.com",
                "api_key": "test-key",
                "model": "deepseek-v4-flash",
            },
            "prompt_compression": {"enabled": False},
        }
        with patch("time.sleep"):
            with patch("eohagent.llm.client.requests.post", side_effect=requests.exceptions.SSLError("eof")) as post:
                client = OpenAICompatibleClient(config)
                with self.assertRaises(requests.exceptions.SSLError):
                    client.complete("hello")
        self.assertEqual(post.call_count, 4)

    def test_deepseek_config_uses_environment_api_key_placeholder(self):
        text = Path("eohagent/config/tsp_gls_arch_spec_deepseek_10x20.yaml").read_text(encoding="utf-8")
        self.assertIn("${AGENT_EOHS_API_KEY}", text)
        self.assertNotIn("sk-", text)

    def test_generation_planner_retries_protocol_mismatch_without_replacement_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["max_llm_retries"] = 2
            parents = [
                Candidate(id="p1", parents=[], algorithm="", code="p1", objective=1.0),
                Candidate(id="p2", parents=[], algorithm="", code="p2", objective=1.1),
            ]
            client = SequencedJSONClient(
                [
                    '{"generation_goal":"bad","offspring_plans":[{"offspring_index":1,"parent_id":"p1","controlled_component":"usage_signal","delta":"change one knob"}]}',
                    '{"generation_goal":"ok","avoid_repeating":[],"offspring_plans":['
                    '{"offspring_index":1,"parent_id":"p1","group_id":"g1","controlled_component":"usage_signal","delta":"change one knob"},'
                    '{"offspring_index":2,"parent_id":"p2","group_id":"g1","controlled_component":"usage_signal","delta":"change another knob"}]}',
                ]
            )
            planner = GenerationPlanner(
                config,
                MemoryManager(config),
                JSONCall(client, max_retries=1),
                SequencedSelector(parents),
            )
            plan = planner.plan(1, parents, 2, parents)
            self.assertEqual(len(plan["offspring_plans"]), 2)
            self.assertEqual([item["parent_id"] for item in plan["offspring_plans"]], ["p1", "p2"])
            self.assertEqual(len(client.prompts), 2)
            self.assertIn("previous_protocol_errors", client.prompts[1])
            self.assertIn("returned 1 plans for 2 offspring", client.prompts[1])

    def test_generation_reflection_retries_transient_llm_failure_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["max_llm_retries"] = 2
            population = [
                Candidate(id="p1", parents=[], algorithm="parent", code="p1", objective=1.0),
            ]
            offspring = [
                Candidate(
                    id="h1",
                    parents=["p1"],
                    algorithm="child",
                    code="h1",
                    objective=0.9,
                    metadata={"generation_plan": {"group_id": "g1", "controlled_component": "penalty"}},
                ),
            ]
            client = SequencedJSONClient(
                [
                    requests.exceptions.SSLError("eof"),
                    json.dumps(
                        {
                            "validated_principles": ["retry used the real LLM response"],
                            "failed_hypotheses": [],
                            "open_questions": ["next controlled delta"],
                        }
                    ),
                ]
            )
            memory = MemoryManager(config)
            planner = GenerationPlanner(
                config,
                memory,
                JSONCall(client, max_retries=1),
                SequencedSelector(population),
            )
            result = planner.reflect(1, population, offspring)
            self.assertEqual(result["validated_principles"], ["retry used the real LLM response"])
            self.assertEqual(len(client.prompts), 2)
            events = memory.session_log.recent(10)
            self.assertTrue(any(event.get("event") == "generation_reflection_retry" for event in events))
            self.assertTrue(any(event.get("event") == "generation_reflection" for event in events))

    def test_generation_reflection_fails_closed_after_llm_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["max_llm_retries"] = 2
            population = [
                Candidate(id="p1", parents=[], algorithm="parent", code="p1", objective=1.0),
            ]
            offspring = [
                Candidate(id="h1", parents=["p1"], algorithm="child", code="h1", objective=0.9),
            ]
            client = SequencedJSONClient(
                [
                    requests.exceptions.SSLError("first eof"),
                    requests.exceptions.SSLError("second eof"),
                ]
            )
            memory = MemoryManager(config)
            planner = GenerationPlanner(
                config,
                memory,
                JSONCall(client, max_retries=1),
                SequencedSelector(population),
            )
            with self.assertRaises(requests.exceptions.SSLError):
                planner.reflect(1, population, offspring)
            self.assertEqual(len(client.prompts), 2)
            events = memory.session_log.recent(10)
            self.assertTrue(any(event.get("event") == "generation_reflection_retry" for event in events))
            self.assertTrue(any(event.get("event") == "generation_reflection_failed" for event in events))
            self.assertFalse(any(event.get("event") == "generation_reflection" for event in events))

    def test_agent_loop_rejects_bad_llm_skill_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_construct.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["generations"] = 1
            config["evolution"]["offspring_per_generation"] = 1
            config["llm"]["provider"] = "mock"
            engine = ExpertEoHEngine(config)
            with self.assertRaises(ValueError):
                engine.harness._normalize_skill_name("json_format_retry")

    def test_parent_pool_excludes_archive_candidates_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/bp_online.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            engine = ExpertEoHEngine(config)
            parent = Candidate(id="p", parents=[], algorithm="", code="def score(item, bins):\n    return item - bins", objective=1.0)
            archived = Candidate(id="h_old", parents=[], algorithm="", code="def score(item, bins):\n    return item + bins", objective=0.5)
            engine.memory.candidate_store.save(archived)
            pool = engine._build_parent_pool([parent])
            self.assertEqual([candidate.id for candidate in pool], ["p"])

    def test_tool_router_extracts_prose_python_code(self):
        config = load_config("eohagent/config/tsp_construct.yaml")
        router = ToolRouter(config, JSONCall(ProseCodeClient()))
        parent = Candidate(
            id="p",
            parents=[],
            algorithm="parent",
            code="import numpy as np\n\ndef select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):\n    return univisited_nodes[0]",
        )
        state = AgentState(parent=parent, population_summary=[], compact_context={}, notebook={})
        result = router.execute("controlled_variation", "SKILL BODY", {"arguments": {}}, state)
        self.assertIn("def select_next_node", result["candidate"].code)

    def test_tool_router_extracts_tsp_gls_code(self):
        config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
        router = ToolRouter(config, JSONCall(GLSProseCodeClient()))
        parent = Candidate(
            id="p",
            parents=[],
            algorithm="parent",
            code="import numpy as np\n\ndef update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n    return edge_distance.copy()",
        )
        state = AgentState(parent=parent, population_summary=[], compact_context={}, notebook={})
        result = router.execute("controlled_variation", "SKILL BODY", {"arguments": {}}, state)
        self.assertIn("def update_edge_distance", result["candidate"].code)
        self.assertIsInstance(result["candidate"].mechanism_hypothesis, dict)
        self.assertIn("Usage-weighted", result["candidate"].mechanism_hypothesis["text"])

    def test_tsp_gls_baseline_loader_provides_multiple_mechanism_seeds(self):
        config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
        seeds = BaselineLoader(config).load()
        families = {(seed.mechanism_hypothesis or {}).get("mechanism_family") for seed in seeds}
        self.assertGreaterEqual(len(seeds), 10)
        self.assertGreaterEqual(len(families), 8)
        self.assertIn("sqrt_usage_penalty", families)
        self.assertIn("utility_ratio_penalty", families)

    def test_tsp_gls_arch_baseline_loader_provides_complete_architecture_seeds(self):
        config = load_config("eohagent/config/tsp_gls_arch_qwen35_122b_10x20.yaml")
        seeds = BaselineLoader(config).load()
        router = ToolRouter(config, JSONCall(MockLLMClient("tsp_gls_arch")))
        self.assertEqual(len(seeds), 10)
        for seed in seeds:
            errors, warnings = router._check_code(seed.code)
            self.assertEqual(errors, [], seed.id)

    def test_tsp_gls_arch_spec_baseline_loader_provides_auditable_spec_seeds(self):
        config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
        seeds = BaselineLoader(config).load()
        router = ToolRouter(config, JSONCall(MockLLMClient("tsp_gls_arch_spec")))
        self.assertEqual(len(seeds), 10)
        for seed in seeds:
            self.assertIn("arch_spec", seed.metadata)
            self.assertEqual(seed.code, build_arch_spec_code(seed.metadata["arch_spec"]))
            errors, warnings = router._check_code(seed.code)
            self.assertEqual(errors, [], seed.id)

    def test_tsp_gls_arch_spec_contract_rejects_handwritten_wrapper(self):
        config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
        router = ToolRouter(config, JSONCall(MockLLMClient("tsp_gls_arch_spec")))
        code = build_arch_spec_code(base_arch_spec()) + "\n# hand-written extension is not allowed\n"
        errors, warnings = router._check_code(code)
        self.assertTrue(any("deterministic code generated from ARCH_SPEC" in error for error in errors))

    def test_arch_spec_semantic_noop_rejected_before_evaluation(self):
        parent_spec = canonical_arch_spec(base_arch_spec())
        child_spec = apply_layer_change(
            parent_spec,
            "escape_policy",
            {**parent_spec["escape_policy"], "trigger": "periodic"},
        )
        parent = Candidate(
            id="p1",
            parents=[],
            algorithm="parent",
            code=build_arch_spec_code(parent_spec),
            objective=1.0,
            metadata={"arch_spec": parent_spec},
        )
        engine = ExpertEoHEngine.__new__(ExpertEoHEngine)
        engine.config = {"project": {"problem": "tsp_gls_arch_spec"}}
        engine.disabled_patterns = []

        class AcceptingRouter:
            def _check_code(self, code):
                return [], []

        engine.harness = type("Harness", (), {"tool_router": AcceptingRouter()})()
        engine._evaluate_individual = lambda *args: (_ for _ in ()).throw(AssertionError("evaluator should not run"))
        plan = {
            "parent_id": "p1",
            "controlled_component": "escape_policy",
            "controlled_change": "trigger",
            "delta": "change trigger while moves already equals max_moves",
            "child_spec": child_spec,
        }
        result = engine._generate_one_arch_spec_offspring([parent], 1, 1, 1, parent, plan)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["entered_evaluator"])
        self.assertIn("semantic_noop", result["error"])
        self.assertEqual(engine.disabled_patterns[0]["pattern_id"], "escape_policy.trigger_at_max_moves")
        self.assertTrue(
            any(event.get("event") == "pre_eval_review" and event.get("passed") is False for event in result["events"])
        )

    def test_arch_spec_review_rejects_disabled_pattern(self):
        parent_spec = canonical_arch_spec(base_arch_spec())
        child_spec = apply_layer_change(
            parent_spec,
            "escape_policy",
            {**parent_spec["escape_policy"], "trigger": "periodic"},
        )
        reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_arch_spec"}})
        result = reviewer.review_plan(
            parent_spec=parent_spec,
            plan={
                "parent_id": "p1",
                "controlled_component": "escape_policy",
                "controlled_change": "trigger",
                "child_spec": child_spec,
            },
            disabled_patterns=[
                {
                    "problem": "tsp_gls_arch_spec",
                    "pattern_id": "escape_policy.trigger_at_max_moves",
                }
            ],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["rejection_reason"], "disabled_pattern")
        self.assertEqual(result["disabled_pattern_hit"], "escape_policy.trigger_at_max_moves")

    def test_arch_spec_review_rejects_missing_controlled_change(self):
        parent_spec = canonical_arch_spec(base_arch_spec())
        child_spec = apply_layer_change(
            parent_spec,
            "penalty_policy",
            {**parent_spec["penalty_policy"], "top_k": 10},
        )
        reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_arch_spec"}})
        result = reviewer.review_plan(
            parent_spec=parent_spec,
            plan={
                "parent_id": "p1",
                "controlled_component": "penalty_policy",
                "child_spec": child_spec,
            },
            disabled_patterns=[],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["rejection_reason"], "missing_controlled_change")

    def test_arch_spec_review_accepts_natural_language_controlled_change(self):
        parent_spec = canonical_arch_spec(base_arch_spec())
        child_spec = apply_layer_change(
            parent_spec,
            "penalty_policy",
            {**parent_spec["penalty_policy"], "increment": 2.0},
        )
        reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_arch_spec"}})
        result = reviewer.review_plan(
            parent_spec=parent_spec,
            plan={
                "parent_id": "p1",
                "controlled_component": "penalty_policy",
                "controlled_change": "increment from 1.0 to 2.0",
                "delta": "Set penalty increment to 2.0",
                "child_spec": child_spec,
            },
            disabled_patterns=[],
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["changed_fields"], ["penalty_policy.increment"])

    def test_arch_spec_review_rejects_undeclared_changed_field(self):
        parent_spec = canonical_arch_spec(base_arch_spec())
        child_spec = apply_layer_change(
            parent_spec,
            "penalty_policy",
            {**parent_spec["penalty_policy"], "top_k": 10, "decay": 0.1},
        )
        reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_arch_spec"}})
        result = reviewer.review_plan(
            parent_spec=parent_spec,
            plan={
                "parent_id": "p1",
                "controlled_component": "penalty_policy",
                "controlled_change": "top_k from 8 to 10",
                "delta": "Set top_k to 10",
                "child_spec": child_spec,
            },
            disabled_patterns=[],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["rejection_reason"], "controlled_change_mismatch")

    def test_experiment_reviewer_rejects_unsupported_problem(self):
        reviewer = ExperimentReviewer({"project": {"problem": "bp_online"}})
        result = reviewer.review_plan(
            parent_spec={},
            plan={"controlled_component": "score_policy", "controlled_change": "weight"},
            disabled_patterns=[],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["rejection_reason"], "unsupported_problem: bp_online")

    def test_arch_spec_review_records_unknown_semantic_when_no_rule_matches(self):
        parent_spec = canonical_arch_spec(base_arch_spec())
        child_spec = apply_layer_change(
            parent_spec,
            "penalty_policy",
            {**parent_spec["penalty_policy"], "top_k": 10},
        )
        reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_arch_spec"}})
        result = reviewer.review_plan(
            parent_spec=parent_spec,
            plan={
                "parent_id": "p1",
                "controlled_component": "penalty_policy",
                "controlled_change": "top_k",
                "child_spec": child_spec,
            },
            disabled_patterns=[],
        )
        self.assertTrue(result["passed"])
        self.assertIsNone(result["rejection_reason"])
        self.assertEqual(result["changed_layers"], ["penalty_policy"])
        self.assertEqual(result["changed_fields"], ["penalty_policy.top_k"])
        self.assertEqual(result["semantic_change"], "unknown")

    def test_tsp_gls_arch_contract_rejects_missing_layer(self):
        config = load_config("eohagent/config/tsp_gls_arch_qwen35_122b_10x20.yaml")
        router = ToolRouter(config, JSONCall(MockLLMClient("tsp_gls_arch")))
        bad_code = (
            "class CandidateGLSArchitecture:\n"
            "    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):\n"
            "        return edge_distance.copy()\n"
        )
        errors, warnings = router._check_code(bad_code)
        self.assertTrue(any("select_penalty_edges" in error for error in errors))

    def test_tsp_gls_arch_evaluator_reports_load_error(self):
        config = load_config("eohagent/config/tsp_gls_arch_qwen35_122b_10x20.yaml")
        evaluator = build_evaluator(config)
        bad_code = "class CandidateGLSArchitecture(:\n    pass\n"
        with self.assertRaisesRegex(RuntimeError, "tsp_gls_arch evaluator failed while loading candidate"):
            evaluator.evaluate(bad_code)
        self.assertIn("SyntaxError", evaluator.last_evidence.get("error", ""))

    def test_tsp_gls_arch_evaluator_does_not_import_prob(self):
        original_import = __import__

        def fail_on_prob(name, *args, **kwargs):
            if name == "prob":
                raise AssertionError("tsp_gls_arch evaluator must not import prob.py")
            return original_import(name, *args, **kwargs)

        config = load_config("eohagent/config/tsp_gls_arch_qwen35_122b_10x20.yaml")
        with patch("builtins.__import__", side_effect=fail_on_prob):
            evaluator = build_evaluator(config)
        self.assertEqual(evaluator.problem, "tsp_gls_arch")
        self.assertGreaterEqual(len(evaluator._evaluator.instances), 1)

    def test_generation_planner_rejects_invalid_architecture_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_qwen35_122b_10x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["max_llm_retries"] = 1
            parent = Candidate(id="p1", parents=[], algorithm="", code="class CandidateGLSArchitecture: pass", objective=1.0)
            client = SequencedJSONClient(
                [
                    '{"generation_goal":"bad","avoid_repeating":[],"offspring_plans":['
                    '{"offspring_index":1,"parent_id":"p1","group_id":"g1","controlled_component":"usage_signal","delta":"change one knob"}]}'
                ]
            )
            planner = GenerationPlanner(
                config,
                MemoryManager(config),
                JSONCall(client, max_retries=1),
                SequencedSelector([parent]),
            )
            with self.assertRaises(RuntimeError):
                planner.plan(1, [parent], 1, [parent])

    def test_generation_planner_requires_arch_spec_single_layer_child_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["max_llm_retries"] = 1
            parent_spec = canonical_arch_spec(base_arch_spec())
            child_spec = arch_spec_child(
                "penalty_policy",
                {
                    "edge_selection": "top_gap",
                    "top_k": 10,
                    "increment": 1.0,
                    "decay": 0.0,
                },
            )
            parent = Candidate(
                id="p1",
                parents=[],
                algorithm="parent",
                code=build_arch_spec_code(parent_spec),
                objective=1.0,
                metadata={"arch_spec": parent_spec},
            )
            client = SequencedJSONClient(
                [
                    json.dumps(
                        {
                            "generation_goal": "ok",
                            "avoid_repeating": [],
                            "offspring_plans": [
                                {
                                    "offspring_index": 1,
                                    "parent_id": "p1",
                                    "group_id": "g1",
                                    "controlled_component": "penalty_policy",
                                    "delta": "increase top_k from 8 to 10",
                                    "preserve": "all other ARCH_SPEC layers",
                                    "expected_effect": "test broader penalty coverage",
                                    "child_spec": child_spec,
                                }
                            ],
                        }
                    )
                ]
            )
            planner = GenerationPlanner(
                config,
                MemoryManager(config),
                JSONCall(client, max_retries=1),
                SequencedSelector([parent]),
            )
            plan = planner.plan(1, [parent], 1, [parent])
            self.assertEqual(plan["offspring_plans"][0]["child_spec"], child_spec)

    def test_generation_planner_payload_includes_disabled_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            parent_spec = canonical_arch_spec(base_arch_spec())
            parent = Candidate(
                id="p1",
                parents=[],
                algorithm="parent",
                code=build_arch_spec_code(parent_spec),
                objective=1.0,
                metadata={"arch_spec": parent_spec},
            )
            memory = MemoryManager(config)
            memory.disabled_patterns = [
                {
                    "problem": "tsp_gls_arch_spec",
                    "pattern_id": "escape_policy.trigger_at_max_moves",
                    "changed_fields": ["escape_policy.trigger"],
                }
            ]
            planner = GenerationPlanner(
                config,
                memory,
                JSONCall(SequencedJSONClient([]), max_retries=1),
                SequencedSelector([parent]),
            )
            state = planner._planner_state([parent], [parent])
            payload = planner._plan_payload(1, [parent], [parent], [parent], state)
            self.assertEqual(payload["disabled_patterns"], memory.disabled_patterns)
            self.assertIn("disabled_patterns", payload["instruction"])

    def test_generation_planner_requires_intent_and_evidence_when_harness_gate_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["harness"] = {"require_experiment_intent": True}
            parent_spec = canonical_arch_spec(base_arch_spec())
            child_spec = apply_layer_change(
                parent_spec,
                "penalty_policy",
                {**parent_spec["penalty_policy"], "edge_selection": "tour_utility"},
            )
            parent = Candidate(
                id="p1",
                parents=[],
                algorithm="parent",
                code=build_arch_spec_code(parent_spec),
                objective=1.0,
                metadata={"arch_spec": parent_spec},
            )
            planner = GenerationPlanner(
                config,
                MemoryManager(config),
                JSONCall(SequencedJSONClient([]), max_retries=1),
                SequencedSelector([parent]),
            )

            with self.assertRaisesRegex(ValueError, "experiment_intent"):
                planner._normalize_plans(
                    {
                        "offspring_plans": [
                            {
                                "offspring_index": 1,
                                "parent_id": "p1",
                                "group_id": "g1",
                                "controlled_component": "penalty_policy",
                                "controlled_change": "edge_selection",
                                "delta": "change edge_selection to tour_utility",
                                "child_spec": child_spec,
                            }
                        ]
                    },
                    [parent],
                )

    def test_generation_planner_keeps_valid_intent_and_evidence_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["harness"] = {"require_experiment_intent": True}
            parent_spec = canonical_arch_spec(base_arch_spec())
            child_spec = apply_layer_change(
                parent_spec,
                "penalty_policy",
                {**parent_spec["penalty_policy"], "edge_selection": "tour_utility"},
            )
            parent = Candidate(
                id="p1",
                parents=[],
                algorithm="parent",
                code=build_arch_spec_code(parent_spec),
                objective=1.0,
                metadata={"arch_spec": parent_spec},
            )
            planner = GenerationPlanner(
                config,
                MemoryManager(config),
                JSONCall(SequencedJSONClient([]), max_retries=1),
                SequencedSelector([parent]),
            )

            plans = planner._normalize_plans(
                {
                    "offspring_plans": [
                        {
                            "offspring_index": 1,
                            "parent_id": "p1",
                            "experiment_intent": "expand",
                            "evidence_refs": ["parent:p1", "qd_cell:penalty_policy"],
                            "group_id": "g1",
                            "controlled_component": "penalty_policy",
                            "controlled_change": "edge_selection",
                            "delta": "change edge_selection to tour_utility",
                            "child_spec": child_spec,
                        }
                    ]
                },
                [parent],
            )

            self.assertEqual(plans[0]["experiment_intent"], "expand")
            self.assertEqual(plans[0]["evidence_refs"], ["parent:p1", "qd_cell:penalty_policy"])

    def test_generation_planner_rejects_arch_spec_multi_layer_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["max_llm_retries"] = 1
            parent_spec = canonical_arch_spec(base_arch_spec())
            child_spec = arch_spec_child(
                "penalty_policy",
                {
                    "edge_selection": "top_gap",
                    "top_k": 10,
                    "increment": 1.0,
                    "decay": 0.0,
                },
            )
            child_spec["escape_policy"]["max_moves"] = 2
            parent = Candidate(
                id="p1",
                parents=[],
                algorithm="parent",
                code=build_arch_spec_code(parent_spec),
                objective=1.0,
                metadata={"arch_spec": parent_spec},
            )
            client = SequencedJSONClient(
                [
                    json.dumps(
                        {
                            "generation_goal": "bad",
                            "avoid_repeating": [],
                            "offspring_plans": [
                                {
                                    "offspring_index": 1,
                                    "parent_id": "p1",
                                    "group_id": "g1",
                                    "controlled_component": "penalty_policy",
                                    "delta": "change penalty top_k",
                                    "child_spec": child_spec,
                                }
                            ],
                        }
                    )
                ]
            )
            planner = GenerationPlanner(
                config,
                MemoryManager(config),
                JSONCall(client, max_retries=1),
                SequencedSelector([parent]),
            )
            with self.assertRaises(RuntimeError):
                planner.plan(1, [parent], 1, [parent])

    def test_solver_tree_schema_builds_deterministic_wrapper_and_descriptor(self):
        from eohagent.problems.tsp_gls_solver_tree.tree_schema import (
            build_solver_tree_code,
            canonical_solver_tree,
            solver_tree_descriptor,
            solver_tree_hash,
        )

        tree = canonical_solver_tree(base_solver_tree())
        code = build_solver_tree_code(tree)

        self.assertIn("SOLVER_TREE", code)
        self.assertIn("SolverTreeCandidateGLSArchitecture", code)
        self.assertEqual(solver_tree_hash(tree), solver_tree_hash(canonical_solver_tree(tree)))
        descriptor = solver_tree_descriptor(tree)
        self.assertEqual(descriptor[0], "solver_tree")
        self.assertIn("penalty_after_local_search", descriptor)

    def test_solver_tree_schema_rejects_unknown_node_and_nested_repeat(self):
        from eohagent.problems.tsp_gls_solver_tree.tree_schema import canonical_solver_tree

        bad = base_solver_tree()
        bad["body"]["steps"][1] = {"type": "invent_algorithm"}
        with self.assertRaisesRegex(ValueError, "unsupported solver tree node"):
            canonical_solver_tree(bad)

        nested = base_solver_tree()
        nested["body"]["steps"].append({"type": "repeat", "budget": "main_loop", "body": {"type": "sequence", "steps": []}})
        with self.assertRaisesRegex(ValueError, "repeat node is only allowed at the root"):
            canonical_solver_tree(nested)

    def test_solver_tree_trace_changes_when_node_order_changes(self):
        from eohagent.problems.tsp_gls_solver_tree.tree_runtime import trace_solver_tree

        parent = base_solver_tree()
        reordered = base_solver_tree()
        steps = reordered["body"]["steps"]
        steps[0], steps[1] = steps[1], steps[0]

        self.assertNotEqual(trace_solver_tree(parent), trace_solver_tree(reordered))
        self.assertEqual(trace_solver_tree(parent)[:2], ["local_search:two_opt_relocate", "penalty_step:top_gap"])
        self.assertEqual(trace_solver_tree(reordered)[:2], ["penalty_step:top_gap", "local_search:two_opt_relocate"])

    def test_tool_router_accepts_only_deterministic_solver_tree_wrapper(self):
        from eohagent.problems.tsp_gls_solver_tree.tree_schema import build_solver_tree_code, canonical_solver_tree

        config = {"project": {"problem": "tsp_gls_solver_tree"}, "evaluator": {"function_name": "CandidateGLSArchitecture"}}
        router = ToolRouter(config, JSONCall(MockLLMClient("tsp_gls_solver_tree")))
        code = build_solver_tree_code(canonical_solver_tree(base_solver_tree()))

        errors, warnings = router._check_code(code)
        self.assertEqual(errors, [])

        errors, warnings = router._check_code(code + "\n# handwritten extension\n")
        self.assertTrue(any("deterministic code generated from SOLVER_TREE" in error for error in errors))

    def test_solver_tree_review_rejects_declared_micro_mutation_that_reorders_nodes(self):
        parent = base_solver_tree()
        child = base_solver_tree()
        child["body"]["steps"][0], child["body"]["steps"][1] = child["body"]["steps"][1], child["body"]["steps"][0]
        reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_solver_tree"}})

        result = reviewer.review_plan(
            parent_spec=parent,
            plan={
                "parent_id": "p1",
                "experiment_intent": "expand",
                "evidence_refs": ["parent:p1:objective", "parent:p1:solver_tree"],
                "mutation_type": "micro_param",
                "controlled_node_path": "body.steps[1].penalty.top_k",
                "controlled_change": "top_k from 8 to 10",
                "expected_trace_change": "same trace, wider penalty coverage",
                "child_solver_tree": child,
            },
            disabled_patterns=[],
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["rejection_reason"], "mutation_type_mismatch")

    def test_solver_tree_review_accepts_json_pointer_controlled_node_path(self):
        parent = base_solver_tree()
        child = base_solver_tree()
        child["body"]["steps"][1]["penalty"]["increment"] = 0.5
        reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_solver_tree"}})

        result = reviewer.review_plan(
            parent_spec=parent,
            plan={
                "parent_id": "p1",
                "experiment_intent": "exploit",
                "evidence_refs": ["parent:p1:objective", "qd_cell_elite:p1"],
                "mutation_type": "micro_param",
                "controlled_node_path": "/body/steps/1/penalty/increment",
                "controlled_change": "penalty.increment from 1.0 to 0.5",
                "expected_trace_change": "same control flow, softer penalty increment",
                "child_solver_tree": child,
            },
            disabled_patterns=[],
        )

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["changed_fields"], ["body.steps[1].penalty.increment"])

    def test_solver_tree_review_treats_edge_selection_change_as_micro_param(self):
        parent = base_solver_tree()
        child = base_solver_tree()
        child["body"]["steps"][1]["penalty"]["edge_selection"] = "tour_utility"
        reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_solver_tree"}})

        result = reviewer.review_plan(
            parent_spec=parent,
            plan={
                "parent_id": "p1",
                "experiment_intent": "expand",
                "evidence_refs": ["parent:p1:objective", "qd_cell_elite:p1"],
                "mutation_type": "micro_param",
                "controlled_node_path": "body.steps[1].penalty.edge_selection",
                "controlled_change": "penalty.edge_selection from top_gap to tour_utility",
                "expected_trace_change": "same control flow, different penalty edge selector",
                "child_solver_tree": child,
            },
            disabled_patterns=[],
        )

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["semantic_change"], "micro_param")

    def test_qd_archive_solver_tree_population_keeps_different_cells(self):
        from eohagent.problems.tsp_gls_solver_tree.tree_schema import canonical_solver_tree

        tree_a = canonical_solver_tree(base_solver_tree())
        tree_b = canonical_solver_tree(base_solver_tree())
        tree_b["body"]["steps"][0], tree_b["body"]["steps"][1] = tree_b["body"]["steps"][1], tree_b["body"]["steps"][0]
        same_as_a = canonical_solver_tree(base_solver_tree())
        candidates = [
            Candidate(id="a_best", parents=[], algorithm="", code="a", objective=0.10, metadata={"solver_tree": tree_a}),
            Candidate(id="a_second", parents=[], algorithm="", code="a2", objective=0.11, metadata={"solver_tree": same_as_a}),
            Candidate(id="b_cell", parents=[], algorithm="", code="b", objective=0.40, metadata={"solver_tree": tree_b}),
        ]
        archive = QDArchive("minimize", descriptor_mode="full")
        archive.update(candidates)

        selected = archive.select_population(2)

        self.assertEqual([candidate.id for candidate in selected], ["a_best", "b_cell"])

    def test_generation_planner_requires_solver_tree_protocol_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
            config["project"]["problem"] = "tsp_gls_solver_tree"
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["harness"] = {"require_experiment_intent": True}
            parent = Candidate(
                id="p1",
                parents=[],
                algorithm="parent",
                code="",
                objective=1.0,
                metadata={"solver_tree": base_solver_tree()},
            )
            planner = GenerationPlanner(
                config,
                MemoryManager(config),
                JSONCall(SequencedJSONClient([]), max_retries=1),
                SequencedSelector([parent]),
            )

            with self.assertRaisesRegex(ValueError, "child_solver_tree"):
                planner._normalize_plans(
                    {
                        "offspring_plans": [
                            {
                                "offspring_index": 1,
                                "parent_id": "p1",
                                "experiment_intent": "expand",
                                "evidence_refs": ["parent:p1:solver_tree"],
                                "mutation_type": "insert_node",
                                "controlled_node_path": "body.steps[1]",
                                "controlled_change": "insert penalty step at body.steps[1]",
                                "expected_trace_change": "insert penalty step",
                                "delta": "insert a penalty step before local search",
                            }
                        ]
                    },
                    [parent],
                )

    def test_langgraph_workflow_exposes_real_agent_nodes(self):
        class FakeEngine:
            def smoke(self):
                return ["seed"]

            def run_langgraph_workflow(self, smoke: bool = False):
                return ["smoke_population"] if smoke else ["run_population"]

        runner = LangGraphWorkflowRunner(FakeEngine())

        self.assertGreaterEqual(
            set(runner.node_names),
            {
                "bootstrap_or_resume",
                "select_cells",
                "plan_experiments",
                "review_generate_evaluate",
                "update_archive",
                "reflect_memory",
            },
        )
        self.assertEqual(runner.run(smoke=True), ["smoke_population"])

    def test_memory_manager_records_typed_experiment_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["memory"] = {"provider": "local"}
            manager = MemoryManager(config)

            record = manager.record_experiment_memory(
                kind="solver_tree_delta",
                candidate_id="h_1",
                generation=3,
                tree_descriptor=("solver_tree", "gls_loop"),
                verdict="supported",
                evidence_source={"objective": 0.1, "parent_id": "p1"},
                content="Penalty after local search improved this cell.",
            )

            self.assertEqual(record["candidate_id"], "h_1")
            self.assertEqual(record["tree_descriptor"], ["solver_tree", "gls_loop"])
            memories = manager.retrieve_memories("Penalty local search", limit=3)
            self.assertTrue(any(item.get("metadata", {}).get("candidate_id") == "h_1" for item in memories))

    def test_langmem_provider_keeps_disk_mirror_across_manager_recreation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["memory"] = {"provider": "langmem", "namespace": ["eohagent", "test_memory"]}
            manager = MemoryManager(config)
            manager.record_experiment_memory(
                kind="solver_tree_delta",
                candidate_id="h_persist",
                generation=4,
                tree_descriptor=("solver_tree", "persisted"),
                verdict="supported",
                evidence_source={"objective": 0.2},
                content="Persisted LangMem mirror record for solver tree.",
            )

            recreated = MemoryManager(config)
            memories = recreated.retrieve_memories("Persisted solver tree", limit=3)

            self.assertTrue(any(item.get("metadata", {}).get("candidate_id") == "h_persist" for item in memories))

    def test_tsp_gls_probe_executes_invariants(self):
        config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
        router = ToolRouter(config, JSONCall(GLSProseCodeClient()))
        parent = Candidate(
            id="p",
            parents=[],
            algorithm="parent",
            code="import numpy as np\n\ndef update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n    return edge_distance.copy()",
        )
        state = AgentState(parent=parent, population_summary=[], compact_context={}, notebook={})
        generated = router.execute("controlled_variation", "SKILL BODY", {"arguments": {}}, state)
        state.update(generated)
        result = router.execute("probe_synthesis", "SKILL BODY", {"arguments": {}}, state)
        self.assertEqual(result["event"], "probe_synthesis")
        self.assertIn("probe_penalty", result)
        self.assertGreater(len(result["probe_results"]), 0)

    def test_skill_registry_does_not_advertise_repair(self):
        skills = SkillRegistry(Path.cwd()).metadata()
        self.assertNotIn("repair", {skill["name"] for skill in skills})

    def test_repair_skill_request_fails_closed(self):
        config = load_config("eohagent/config/tsp_construct.yaml")
        parent_code = "def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):\n    return univisited_nodes[0]"
        router = ToolRouter(config, JSONCall(ParentCodeRepairClient(parent_code)))
        parent = Candidate(id="p", parents=[], algorithm="parent", code=parent_code)
        state = AgentState(parent=parent, population_summary=[], compact_context={}, notebook={})
        state.current_candidate = Candidate(
            id="c",
            parents=["p"],
            algorithm="broken",
            code="def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):\n    return missing_name",
        )
        result = router.execute("repair", "SKILL BODY", {"arguments": {}}, state)
        self.assertEqual(result["event"], "repair_failed")
        self.assertIn("disabled", result["errors"][0])

    def test_bounded_notebook_does_not_surface_successful_repairs(self):
        summary = bounded_notebook(
            {
                "validated_principles": ["keep"],
                "successful_repairs": ["old repair"],
            }
        )
        self.assertIn("validated_principles", summary)
        self.assertNotIn("successful_repairs", summary)

    def test_import_evaluator_bp_online_seed(self):
        config = load_config("eohagent/config/bp_online.yaml")
        evaluator = build_evaluator(config)
        code = "def score(item, bins):\n    import numpy as np\n    return item - bins"
        objective = evaluator.evaluate(code)
        self.assertIsInstance(objective, float)

    def test_contract_check_rejects_wrong_function(self):
        config = load_config("eohagent/config/bp_online.yaml")
        router = ToolRouter(config, JSONCall(MockLLMClient("bp_online")))
        errors, warnings = router._check_code("def heuristic(item, bins):\n    return bins")
        self.assertTrue(any("missing function score" in error for error in errors))

    def test_contract_check_accepts_correct_and_rejects_nan(self):
        config = load_config("eohagent/config/bp_online.yaml")
        router = ToolRouter(config, JSONCall(MockLLMClient("bp_online")))
        errors, warnings = router._check_code("def score(item, bins):\n    import numpy as np\n    return item - bins")
        self.assertEqual(errors, [])
        errors, warnings = router._check_code("def score(item, bins):\n    import numpy as np\n    return np.array([np.nan, 1.0, 2.0])")
        self.assertTrue(any("nan_inf_error" in error for error in errors))

    def test_contract_check_accepts_tsp_gls_signature(self):
        config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
        router = ToolRouter(config, JSONCall(MockLLMClient("tsp_gls")))
        code = (
            "import numpy as np\n\n"
            "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
            "    updated_edge_distance = edge_distance.copy()\n"
            "    for i in range(len(local_opt_tour)):\n"
            "        a = int(local_opt_tour[i])\n"
            "        b = int(local_opt_tour[(i + 1) % len(local_opt_tour)])\n"
            "        updated_edge_distance[a, b] *= 1.0 + edge_n_used[a, b]\n"
            "        updated_edge_distance[b, a] *= 1.0 + edge_n_used[a, b]\n"
            "    return updated_edge_distance"
        )
        errors, warnings = router._check_code(code)
        self.assertEqual(errors, [])
        bad_code = "def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):\n    return univisited_nodes[0]"
        errors, warnings = router._check_code(bad_code)
        self.assertTrue(any("missing function update_edge_distance" in error for error in errors))

    def test_skill_doc_is_sent_to_llm_skill_payload(self):
        config = load_config("eohagent/config/bp_online.yaml")
        client = CaptureClient()
        router = ToolRouter(config, JSONCall(client))
        parent = Candidate(id="p", parents=[], algorithm="parent", code="def score(item, bins):\n    return item - bins")
        state = AgentState(parent=parent, population_summary=[], compact_context={}, notebook={})
        router.execute("controlled_variation", "SKILL BODY", {"arguments": {}}, state)
        self.assertIn("SKILL BODY", client.payloads[-1]["system"])

    def test_context_runtime_compacts_before_code_llm_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["context_management"]["context_window"] = 5000
            config["context_management"]["compact_threshold"] = 0.70
            config["context_management"]["chars_per_token"] = 1.0
            memory = MemoryManager(config)
            memory.notebook.write_default()
            memory.compact_context.write({"problem_card": {"expected_function_name": "update_edge_distance"}, "huge_trace": "x" * 5000})
            parent = Candidate(
                id="p",
                parents=[],
                algorithm="parent",
                code="import numpy as np\n\ndef update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n    return edge_distance.copy()",
            )
            state = AgentState(
                parent=parent,
                population_summary=[{"id": "p", "objective": 1.0, "mechanism_summary": "baseline"}],
                compact_context=memory.compact_context.read(),
                notebook=memory.notebook.read(),
            )
            client = ContextCompactionClient()
            runtime = ContextRuntime(config, memory, client)
            json_call = JSONCall(client, context_runtime=runtime)
            router = ToolRouter(config, json_call)
            result = router.execute("controlled_variation", "SKILL BODY", {"arguments": {}}, state)
            self.assertEqual(result["event"], "candidate_generated")
            self.assertEqual(len(client.prompts), 1)
            self.assertIn("working_memory", state.compact_context)
            self.assertIn("context_management", state.compact_context)
            self.assertIn(state.compact_context["context_management"]["strategy"], {"bounded", "emergency"})

    def test_context_runtime_blocks_call_when_compaction_cannot_reduce_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/bp_online.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["context_management"]["context_window"] = 100
            config["context_management"]["compact_threshold"] = 0.70
            config["context_management"]["chars_per_token"] = 1.0
            config["context_management"]["max_compaction_retries"] = 1
            memory = MemoryManager(config)
            memory.notebook.write_default()
            memory.compact_context.write({"problem_card": {"expected_function_name": "score"}})
            parent = Candidate(id="p", parents=[], algorithm="parent", code="def score(item, bins):\n    return item - bins")
            state = AgentState(parent=parent, population_summary=[], compact_context=memory.compact_context.read(), notebook=memory.notebook.read())
            client = OverflowCompactionClient()
            runtime = ContextRuntime(config, memory, client)
            json_call = JSONCall(client, context_runtime=runtime)
            with self.assertRaises(ContextOverflowError):
                json_call.complete_text_dynamic(lambda: "x" * 1000, purpose="controlled_variation", context_state=state)
            self.assertEqual(client.calls, 0)

    def test_langmem_provider_records_and_retrieves_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "_repo_root": ".",
                "project": {"output_dir": str(Path(tmp) / "run")},
                "memory": {"provider": "langmem"},
            }
            memory = MemoryManager(config)

            memory.record_memory(
                "semantic",
                "sqrt_usage with tour_utility opened the strongest v5 mechanism cell",
                {"component": "penalty_policy"},
            )
            hits = memory.retrieve_memories("tour_utility", limit=3)

            self.assertTrue(any("tour_utility" in hit.get("content", "") for hit in hits))

    def test_langmem_provider_fails_closed_when_package_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "_repo_root": ".",
                "project": {"output_dir": str(Path(tmp) / "run")},
                "memory": {"provider": "langmem"},
            }
            with patch("eohagent.memory.provider.importlib.util.find_spec", return_value=None):
                with self.assertRaises(MemoryProviderError):
                    MemoryManager(config)

    def test_commit_gate_commits_after_contract_passes(self):
        config = load_config("eohagent/config/bp_online.yaml")
        gate = CommitGate(config)
        parent = Candidate(id="p", parents=[], algorithm="parent", code="def score(item, bins):\n    return item - bins")
        state = AgentState(parent=parent, population_summary=[], compact_context={}, notebook={})
        state.current_candidate = Candidate(id="c", parents=["p"], algorithm="c", code=parent.code)
        state.contract_checked = True
        state.contract_passed = True
        self.assertEqual(gate.check(state), "commit")

    def test_arch_spec_commit_gate_rejects_failed_contract_without_repair(self):
        config = load_config("eohagent/config/tsp_gls_arch_spec_qwen35_122b_10x20.yaml")
        config["agent"]["max_repairs_per_candidate"] = 2
        gate = CommitGate(config)
        parent = Candidate(id="p", parents=[], algorithm="parent", code=build_arch_spec_code(base_arch_spec()))
        state = AgentState(parent=parent, population_summary=[], compact_context={}, notebook={})
        state.current_candidate = Candidate(id="c", parents=["p"], algorithm="c", code=parent.code)
        state.contract_checked = True
        state.contract_passed = False
        self.assertEqual(gate.check(state), "reject")

    def test_commit_gate_rejects_failed_contract_without_repair(self):
        config = load_config("eohagent/config/bp_online.yaml")
        config["agent"]["max_repairs_per_candidate"] = 2
        gate = CommitGate(config)
        parent = Candidate(id="p", parents=[], algorithm="parent", code="def score(item, bins):\n    return item")
        state = AgentState(parent=parent, population_summary=[], compact_context={}, notebook={})
        state.current_candidate = Candidate(id="c", parents=["p"], algorithm="c", code=parent.code)
        state.contract_checked = True
        state.contract_passed = False
        self.assertEqual(gate.check(state), "reject")

    def test_notebook_merge_update_writes_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            notebook = ExpertNotebook(Path(tmp))
            notebook.write_default()
            notebook.merge_update({"lessons": ["Prefer normalized score terms."], "anti_patterns": ["Huge constants"]})
            data = notebook.read()
            self.assertIn("Prefer normalized score terms.", data["validated_principles"])
            self.assertIn("Huge constants", data["anti_patterns"])

    def test_rank_selector_respects_direction(self):
        low = Candidate(id="low", parents=[], algorithm="", code="", objective=1.0, selection_score=1.0)
        high = Candidate(id="high", parents=[], algorithm="", code="", objective=9.0, selection_score=9.0)
        with patch("random.choices", side_effect=lambda seq, weights, k: [seq[0]]):
            self.assertEqual(RankSelector("minimize").select_parent([high, low]).id, "low")
            self.assertEqual(RankSelector("maximize").select_parent([low, high]).id, "high")

    def test_cell_round_robin_selector_covers_cells_before_quality_repeat(self):
        a = Candidate(id="a", parents=[], algorithm="", code="", objective=0.3)
        b = Candidate(id="b", parents=[], algorithm="", code="", objective=0.1)
        c = Candidate(id="c", parents=[], algorithm="", code="", objective=0.2)
        selector = CellRoundRobinSelector("minimize")

        selected = [selector.select_parent([a, b, c]).id for _ in range(5)]

        self.assertEqual(selected, ["a", "b", "c", "b", "c"])

    def test_population_manager_deduplicates_by_objective_like_eoh(self):
        seed = Candidate(id="seed", parents=[], algorithm="", code="a", objective=1.234564)
        clone = Candidate(id="clone", parents=[], algorithm="", code="different", objective=1.234563)
        better = Candidate(id="better", parents=[], algorithm="", code="b", objective=1.111111)
        population = PopulationManager(pop_size=10, objective_direction="minimize").update([seed], [clone, better])
        self.assertEqual([candidate.id for candidate in population], ["better", "seed"])
        self.assertEqual(population[1].objective, 1.23456)

    def test_population_manager_uses_selection_score_and_diversity(self):
        a1 = Candidate(
            id="a1",
            parents=[],
            algorithm="",
            code="a1",
            objective=1.0,
            selection_score=1.0,
            mechanism_hypothesis={"mechanism_family": "linear_usage_penalty"},
        )
        a2 = Candidate(
            id="a2",
            parents=[],
            algorithm="",
            code="a2",
            objective=1.01,
            selection_score=1.01,
            mechanism_hypothesis={"mechanism_family": "linear_usage_penalty"},
        )
        b = Candidate(
            id="b",
            parents=[],
            algorithm="",
            code="b",
            objective=1.02,
            selection_score=1.02,
            mechanism_hypothesis={"mechanism_family": "length_weighted_penalty"},
        )
        population = PopulationManager(pop_size=2, objective_direction="minimize", diversity_weight=0.05).update([], [a1, a2, b])
        self.assertEqual([candidate.id for candidate in population], ["a1", "b"])

    def test_population_manager_family_champion_survival_keeps_distinct_families(self):
        a1 = Candidate(
            id="a1",
            parents=[],
            algorithm="",
            code="a1",
            objective=1.0,
            mechanism_hypothesis={"mechanism_family": "linear_usage_penalty"},
        )
        a2 = Candidate(
            id="a2",
            parents=[],
            algorithm="",
            code="a2",
            objective=1.01,
            mechanism_hypothesis={"mechanism_family": "linear_usage_penalty"},
        )
        b = Candidate(
            id="b",
            parents=[],
            algorithm="",
            code="b",
            objective=1.3,
            mechanism_hypothesis={"mechanism_family": "length_weighted_penalty"},
        )
        c = Candidate(
            id="c",
            parents=[],
            algorithm="",
            code="c",
            objective=1.4,
            mechanism_hypothesis={"mechanism_family": "utility_ratio_penalty"},
        )
        population = PopulationManager(
            pop_size=3,
            objective_direction="minimize",
            survival_selection="family_champion",
        ).update([], [a1, a2, b, c])
        self.assertEqual([candidate.id for candidate in population], ["a1", "b", "c"])

    def test_population_manager_quality_diversity_keeps_pareto_distinct_arch_spec_niches(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = arch_spec_child(
            "penalty_policy",
            {
                "edge_selection": "top_gap",
                "top_k": 10,
                "increment": 1.0,
                "decay": 0.0,
            },
        )
        mean_best = Candidate(
            id="mean_best",
            parents=[],
            algorithm="",
            code="a",
            objective=1.0,
            metadata={"arch_spec": spec_a},
            evidence={
                "formal_evaluator": {"max_gap": 1.5, "std_gap": 0.5},
                "training_diagnosis": {"tail_gap": 1.4},
            },
        )
        same_niche_worse = Candidate(
            id="same_niche_worse",
            parents=[],
            algorithm="",
            code="b",
            objective=1.01,
            metadata={"arch_spec": spec_a},
            evidence={
                "formal_evaluator": {"max_gap": 1.6, "std_gap": 0.6},
                "training_diagnosis": {"tail_gap": 1.5},
            },
        )
        tail_best = Candidate(
            id="tail_best",
            parents=[],
            algorithm="",
            code="c",
            objective=1.2,
            metadata={"arch_spec": spec_b},
            evidence={
                "formal_evaluator": {"max_gap": 0.4, "std_gap": 0.1},
                "training_diagnosis": {"tail_gap": 0.3},
            },
        )
        population = PopulationManager(
            pop_size=2,
            objective_direction="minimize",
            survival_selection="quality_diversity",
            deduplicate_objective=False,
        ).update([], [same_niche_worse, mean_best, tail_best])
        self.assertEqual([candidate.id for candidate in population], ["mean_best", "tail_best"])

    def test_qd_archive_replaces_elite_only_with_better_same_cell(self):
        spec = canonical_arch_spec(base_arch_spec())
        worse = Candidate(
            id="worse",
            parents=[],
            algorithm="",
            code="worse",
            objective=0.20,
            metadata={"arch_spec": spec},
        )
        better = Candidate(
            id="better",
            parents=[],
            algorithm="",
            code="better",
            objective=0.10,
            metadata={"arch_spec": spec},
        )
        archive = QDArchive(objective_direction="minimize")

        first = archive.add(worse)
        second = archive.add(better)

        self.assertTrue(first.inserted)
        self.assertTrue(second.replaced)
        self.assertEqual([candidate.id for candidate in archive.elites()], ["better"])

    def test_qd_archive_keeps_distinct_arch_spec_cells(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = apply_layer_change(
            spec_a,
            "distance_update_policy",
            {**spec_a["distance_update_policy"], "type": "utility_ratio"},
        )
        a = Candidate(id="a", parents=[], algorithm="", code="a", objective=0.20, metadata={"arch_spec": spec_a})
        b = Candidate(id="b", parents=[], algorithm="", code="b", objective=0.30, metadata={"arch_spec": spec_b})
        archive = QDArchive(objective_direction="minimize")

        archive.update([a, b])

        self.assertEqual({candidate.id for candidate in archive.elites()}, {"a", "b"})
        self.assertEqual(archive.coverage, 2)

    def test_qd_archive_mechanism_cells_ignore_escape_and_restart_tuning(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = apply_layer_change(
            spec_a,
            "escape_policy",
            {**spec_a["escape_policy"], "moves": 4, "max_moves": 4},
        )
        spec_c = apply_layer_change(
            spec_a,
            "restart_policy",
            {**spec_a["restart_policy"], "action": "reset_to_initial"},
        )
        best = Candidate(id="best", parents=[], algorithm="", code="best", objective=0.10, metadata={"arch_spec": spec_a})
        escape_tuned = Candidate(id="escape", parents=[], algorithm="", code="escape", objective=0.20, metadata={"arch_spec": spec_b})
        restart_tuned = Candidate(id="restart", parents=[], algorithm="", code="restart", objective=0.30, metadata={"arch_spec": spec_c})
        archive = QDArchive(objective_direction="minimize", descriptor_mode="mechanism")

        archive.update([best, escape_tuned, restart_tuned])

        self.assertEqual(archive.coverage, 1)
        self.assertEqual([candidate.id for candidate in archive.elites()], ["best"])

    def test_qd_archive_mechanism_cells_keep_distance_penalty_topk_and_schedule_distinct(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = apply_layer_change(
            spec_a,
            "penalty_policy",
            {**spec_a["penalty_policy"], "edge_selection": "tour_utility"},
        )
        spec_c = apply_layer_change(
            spec_a,
            "schedule_policy",
            {**spec_a["schedule_policy"], "strength_schedule": "decay_after_early"},
        )
        a = Candidate(id="a", parents=[], algorithm="", code="a", objective=0.30, metadata={"arch_spec": spec_a})
        b = Candidate(id="b", parents=[], algorithm="", code="b", objective=0.20, metadata={"arch_spec": spec_b})
        c = Candidate(id="c", parents=[], algorithm="", code="c", objective=0.10, metadata={"arch_spec": spec_c})
        archive = QDArchive(objective_direction="minimize", descriptor_mode="mechanism")

        archive.update([a, b, c])

        self.assertEqual(archive.coverage, 3)
        self.assertEqual([candidate.id for candidate in archive.elites()], ["c", "b", "a"])

    def test_population_manager_qd_archive_returns_cell_elites_then_best_fill(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = apply_layer_change(
            spec_a,
            "distance_update_policy",
            {**spec_a["distance_update_policy"], "type": "utility_ratio"},
        )
        a1 = Candidate(id="a1", parents=[], algorithm="", code="a1", objective=0.20, metadata={"arch_spec": spec_a})
        a2 = Candidate(id="a2", parents=[], algorithm="", code="a2", objective=0.10, metadata={"arch_spec": spec_a})
        b = Candidate(id="b", parents=[], algorithm="", code="b", objective=0.30, metadata={"arch_spec": spec_b})

        population = PopulationManager(
            pop_size=3,
            objective_direction="minimize",
            survival_selection="qd_archive",
            deduplicate_objective=False,
        ).update([a1], [a2, b])

        self.assertEqual([candidate.id for candidate in population], ["a2", "b", "a1"])

    def test_result_store_saves_qd_archive_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "_repo_root": ".",
                "project": {"output_dir": str(Path(tmp) / "run")},
            }
            spec = canonical_arch_spec(base_arch_spec())
            candidate = Candidate(
                id="elite",
                parents=[],
                algorithm="",
                code="elite",
                objective=0.1,
                metadata={"arch_spec": spec},
            )
            archive = QDArchive(objective_direction="minimize")
            archive.add(candidate)
            store = ResultStore(config)

            store.save_qd_archive(3, archive)

            path = Path(tmp) / "run" / "results" / "qd_archive" / "qd_archive_generation_3.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["coverage"], 1)
            self.assertEqual(data["cells"][0]["candidate"]["id"], "elite")

    def test_engine_qd_archive_parent_pool_uses_archive_cells(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = apply_layer_change(
            spec_a,
            "distance_update_policy",
            {**spec_a["distance_update_policy"], "type": "utility_ratio"},
        )
        a = Candidate(id="a", parents=[], algorithm="", code="a", objective=0.10, metadata={"arch_spec": spec_a})
        b = Candidate(id="b", parents=[], algorithm="", code="b", objective=0.30, metadata={"arch_spec": spec_b})
        engine = ExpertEoHEngine.__new__(ExpertEoHEngine)
        engine.config = {"evolution": {"survival_selection": "qd_archive", "allow_archive_parents": False}}
        engine.qd_archive = QDArchive(objective_direction="minimize")
        engine.qd_archive.update([a, b])

        pool = engine._build_parent_pool([a])

        self.assertEqual([candidate.id for candidate in pool], ["a", "b"])

    def test_engine_qd_cell_parent_plan_uses_archive_cells_without_emitters(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = apply_layer_change(
            spec_a,
            "penalty_policy",
            {**spec_a["penalty_policy"], "edge_selection": "tour_utility"},
        )
        a = Candidate(id="a", parents=[], algorithm="", code="a", objective=0.10, metadata={"arch_spec": spec_a})
        b = Candidate(id="b", parents=[], algorithm="", code="b", objective=0.30, metadata={"arch_spec": spec_b})
        engine = ExpertEoHEngine.__new__(ExpertEoHEngine)
        engine.config = {"evolution": {"offspring_per_generation": 3, "survival_selection": "qd_cells"}}
        engine.qd_archive = QDArchive(objective_direction="minimize", descriptor_mode="mechanism")
        engine.qd_archive.update([a, b])

        parent_pool, slot_metadata = engine._build_qd_cell_parent_plan(3)

        self.assertEqual([candidate.id for candidate in parent_pool], ["a", "b", "a"])
        self.assertEqual(len(slot_metadata), 3)
        self.assertTrue(all("qd_cell_descriptor" in slot for slot in slot_metadata))
        self.assertTrue(all("emitter" not in slot for slot in slot_metadata))

    def test_engine_saves_qd_archive_when_enabled(self):
        class FakeResultStore:
            def __init__(self):
                self.saved = []

            def save_qd_archive(self, generation, archive):
                self.saved.append((generation, archive.coverage))

        spec = canonical_arch_spec(base_arch_spec())
        candidate = Candidate(id="elite", parents=[], algorithm="", code="elite", objective=0.1, metadata={"arch_spec": spec})
        engine = ExpertEoHEngine.__new__(ExpertEoHEngine)
        engine.qd_archive = QDArchive(objective_direction="minimize")
        engine.qd_archive.add(candidate)
        engine.result_store = FakeResultStore()

        engine._save_qd_archive(7)

        self.assertEqual(engine.result_store.saved, [(7, 1)])

    def test_behavior_descriptor_uses_structure_and_training_behavior(self):
        spec = canonical_arch_spec(base_arch_spec())
        candidate = Candidate(
            id="c",
            parents=[],
            algorithm="",
            code="c",
            objective=0.2,
            metadata={"arch_spec": spec},
            evidence={
                "training_diagnosis": {
                    "failure_signature": "single_extreme_training_failure",
                    "worst_training_instances": [{"index": 60}, {"index": 51}],
                }
            },
        )

        descriptor = behavior_descriptor_for_candidate(candidate)

        self.assertIn(("distance", "linear_usage"), descriptor)
        self.assertIn(("penalty", "top_gap"), descriptor)
        self.assertIn(("failure", "single_extreme_training_failure"), descriptor)
        self.assertIn(("worst_instances", (60, 51)), descriptor)

    def test_route_manager_keeps_behavior_distinct_for_same_architecture(self):
        spec = canonical_arch_spec(base_arch_spec())
        a = Candidate(
            id="a",
            parents=[],
            algorithm="",
            code="a",
            objective=0.2,
            metadata={"arch_spec": spec},
            evidence={
                "training_diagnosis": {
                    "failure_signature": "single_extreme_training_failure",
                    "worst_training_instances": [{"index": 60}],
                }
            },
        )
        b = Candidate(
            id="b",
            parents=[],
            algorithm="",
            code="b",
            objective=0.3,
            metadata={"arch_spec": spec},
            evidence={
                "training_diagnosis": {
                    "failure_signature": "tail_heavy_training_failures",
                    "worst_training_instances": [{"index": 12}],
                }
            },
        )

        manager = RouteManager("minimize")
        manager.update([a, b])

        self.assertEqual(manager.coverage, 2)
        self.assertEqual({route.best_candidate_id for route in manager.routes()}, {"a", "b"})

    def test_route_manager_updates_best_and_credit_from_route_improvement(self):
        spec = canonical_arch_spec(base_arch_spec())
        worse = Candidate(
            id="worse",
            parents=[],
            algorithm="",
            code="worse",
            objective=0.3,
            metadata={"arch_spec": spec},
            evidence={
                "training_diagnosis": {
                    "failure_signature": "single_extreme_training_failure",
                    "worst_training_instances": [{"index": 60}],
                }
            },
        )
        better = Candidate(
            id="better",
            parents=["worse"],
            algorithm="",
            code="better",
            objective=0.2,
            metadata={"arch_spec": spec},
            evidence={
                "training_diagnosis": {
                    "failure_signature": "single_extreme_training_failure",
                    "worst_training_instances": [{"index": 60}],
                    "mean_delta_vs_parent": -0.1,
                    "tail_delta_vs_parent": -0.2,
                    "improved_instances_vs_parent": [{"index": 60, "delta_gap": -0.4}],
                }
            },
        )

        manager = RouteManager("minimize")
        manager.update([worse])
        updates = manager.update([better])
        route = manager.routes()[0]

        self.assertEqual(updates[-1].candidate_id, "better")
        self.assertTrue(updates[-1].route_best_improved)
        self.assertTrue(updates[-1].failure_repaired)
        self.assertEqual(route.best_candidate_id, "better")
        self.assertEqual(route.trials, 2)
        self.assertEqual(route.successes, 2)
        self.assertGreater(route.credit, 0.0)

    def test_route_manager_marks_route_cooldown_after_five_stale_generations(self):
        spec = canonical_arch_spec(base_arch_spec())
        candidate = Candidate(
            id="route_seed",
            parents=[],
            algorithm="",
            code="route_seed",
            objective=0.1,
            generation=0,
            metadata={"arch_spec": spec},
            evidence={"training_diagnosis": {"failure_signature": "broadly_stable_training_behavior"}},
        )
        manager = RouteManager("minimize")

        manager.update([candidate], generation=0)
        manager.refresh_statuses(generation=4)
        self.assertNotEqual(manager.routes()[0].status, "cooldown")

        manager.refresh_statuses(generation=5)
        route = manager.routes()[0]
        self.assertEqual(route.status, "cooldown")
        self.assertEqual(route.stale_generations, 5)

    def test_route_parent_plan_skips_cooldown_route_for_exploitation(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = apply_layer_change(
            spec_a,
            "distance_update_policy",
            {**spec_a["distance_update_policy"], "type": "sqrt_usage"},
        )
        stale_best = Candidate(
            id="stale_best",
            parents=[],
            algorithm="",
            code="stale_best",
            objective=0.1,
            generation=0,
            metadata={"arch_spec": spec_a},
            evidence={"training_diagnosis": {"failure_signature": "broadly_stable_training_behavior"}},
        )
        active_route = Candidate(
            id="active_route",
            parents=[],
            algorithm="",
            code="active_route",
            objective=0.2,
            generation=5,
            metadata={"arch_spec": spec_b},
            evidence={"training_diagnosis": {"failure_signature": "broadly_stable_training_behavior"}},
        )
        manager = RouteManager("minimize")

        manager.update([stale_best], generation=0)
        manager.update([active_route], generation=5)
        manager.refresh_statuses(generation=5)
        parent, route = manager.parent_for_emitter("exploit_best_route", [stale_best, active_route])

        self.assertEqual(parent.id, "active_route")
        self.assertNotEqual(route.status, "cooldown")

    def test_emitter_scheduler_initializes_all_emitters_before_repeating(self):
        scheduler = EmitterScheduler()

        emitters = scheduler.select_emitters(len(EMITTER_TYPES))

        self.assertEqual(emitters, list(EMITTER_TYPES))

    def test_emitter_scheduler_uses_ucb_credit_after_initialization(self):
        scheduler = EmitterScheduler()
        for emitter in EMITTER_TYPES:
            scheduler.record(emitter, success=(emitter == "exploit_best_route"))
        scheduler.record("exploit_best_route", success=True)
        scheduler.record("exploit_best_route", success=True)

        emitters = scheduler.select_emitters(1)

        self.assertEqual(emitters, ["exploit_best_route"])

    def test_result_store_saves_route_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "_repo_root": ".",
                "project": {"output_dir": str(Path(tmp) / "run")},
            }
            spec = canonical_arch_spec(base_arch_spec())
            candidate = Candidate(
                id="elite",
                parents=[],
                algorithm="",
                code="elite",
                objective=0.1,
                metadata={"arch_spec": spec},
                evidence={"training_diagnosis": {"failure_signature": "broadly_stable_training_behavior"}},
            )
            manager = RouteManager("minimize")
            manager.update([candidate])
            store = ResultStore(config)

            store.save_route_state(2, manager)

            path = Path(tmp) / "run" / "results" / "routes" / "route_state_generation_2.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["coverage"], 1)
            self.assertEqual(data["routes"][0]["best_candidate_id"], "elite")

    def test_engine_route_parent_plan_uses_behavior_routes_not_only_population_top(self):
        spec_a = canonical_arch_spec(base_arch_spec())
        spec_b = apply_layer_change(
            spec_a,
            "distance_update_policy",
            {**spec_a["distance_update_policy"], "type": "sqrt_usage"},
        )
        best = Candidate(
            id="best",
            parents=[],
            algorithm="",
            code="best",
            objective=0.1,
            metadata={"arch_spec": spec_a},
            evidence={"training_diagnosis": {"failure_signature": "broadly_stable_training_behavior"}},
        )
        archive_route = Candidate(
            id="archive_route",
            parents=[],
            algorithm="",
            code="archive_route",
            objective=0.4,
            metadata={"arch_spec": spec_b},
            evidence={"training_diagnosis": {"failure_signature": "tail_heavy_training_failures"}},
        )
        engine = ExpertEoHEngine.__new__(ExpertEoHEngine)
        engine.route_manager = RouteManager("minimize")
        engine.route_manager.update([best, archive_route])
        engine.emitter_scheduler = EmitterScheduler()
        engine.config = {"evolution": {"offspring_per_generation": 2}}

        parent_pool, slot_metadata = engine._build_route_parent_plan([best], 2)

        self.assertEqual(len(parent_pool), 2)
        self.assertEqual(len(slot_metadata), 2)
        self.assertIn("archive_route", [candidate.id for candidate in parent_pool])
        self.assertTrue(all("emitter" in slot for slot in slot_metadata))

    def test_langgraph_workflow_runner_invokes_engine_run_and_smoke(self):
        class FakeEngine:
            def __init__(self):
                self.calls = []

            def run(self):
                self.calls.append("run")
                return ["run_population"]

            def smoke(self):
                self.calls.append("smoke")
                return ["smoke_population"]

        engine = FakeEngine()
        runner = LangGraphWorkflowRunner(engine)

        self.assertEqual(runner.run(smoke=False), ["run_population"])
        self.assertEqual(runner.run(smoke=True), ["smoke_population"])
        self.assertEqual(engine.calls, ["run", "smoke"])

    def test_arch_spec_invalid_niche_metadata_fails_closed(self):
        candidate = Candidate(
            id="bad",
            parents=[],
            algorithm="",
            code="",
            objective=1.0,
            metadata={"arch_spec": {"distance_update_policy": {}}},
        )
        with self.assertRaises(ValueError):
            PopulationManager._niche(candidate)

    def test_selection_score_uses_training_tail_risk_when_objective_ties(self):
        engine = ExpertEoHEngine.__new__(ExpertEoHEngine)
        robust = Candidate(
            id="robust",
            parents=[],
            algorithm="",
            code="",
            objective=1.0,
            evidence={"training_diagnosis": {"tail_gap": 1.05, "max_gap": 1.1, "std_gap": 0.05}},
        )
        brittle = Candidate(
            id="brittle",
            parents=[],
            algorithm="",
            code="",
            objective=1.0,
            evidence={"training_diagnosis": {"tail_gap": 1.8, "max_gap": 2.4, "std_gap": 0.6}},
        )
        self.assertLess(engine._selection_score(robust), engine._selection_score(brittle))

    def test_pareto_metrics_prefer_training_diagnosis_over_formal_summary(self):
        candidate = Candidate(
            id="c",
            parents=[],
            algorithm="",
            code="",
            objective=1.0,
            evidence={
                "formal_evaluator": {"max_gap": 9.0, "std_gap": 8.0},
                "training_diagnosis": {"tail_gap": 1.2, "max_gap": 1.4, "std_gap": 0.2},
            },
        )
        self.assertEqual(PopulationManager._pareto_metrics(candidate), (1.0, 1.2, 1.4, 0.2))

    def test_arch_spec_plan_alignment_checks_child_spec_and_single_layer_diff(self):
        parent_spec = canonical_arch_spec(base_arch_spec())
        child_spec = arch_spec_child(
            "penalty_policy",
            {
                "edge_selection": "top_gap",
                "top_k": 10,
                "increment": 1.0,
                "decay": 0.0,
            },
        )
        plan = {
            "parent_id": "p1",
            "controlled_component": "penalty_policy",
            "delta": "increase top_k",
            "child_spec": child_spec,
        }
        candidate = Candidate(
            id="c1",
            parents=["p1"],
            algorithm="",
            code=build_arch_spec_code(child_spec),
            metadata={
                "planned_experiment": plan,
                "parent_arch_spec": parent_spec,
                "arch_spec": child_spec,
                "spec_diff": {"changed_layers": ["penalty_policy"]},
            },
        )
        self.assertTrue(ExpertEoHEngine._plan_alignment(candidate, plan)["aligned"])
        bad_plan = dict(plan)
        bad_plan["child_spec"] = parent_spec
        self.assertFalse(ExpertEoHEngine._plan_alignment(candidate, bad_plan)["aligned"])

    def test_mechanism_family_canonicalization_cleans_noisy_labels(self):
        self.assertEqual(canonicalize_family("usage_penalty.", ""), "linear_usage_penalty")
        self.assertEqual(canonicalize_family("usage penalty with cap", ""), "capped_usage_penalty")
        self.assertEqual(canonicalize_family("", "smooth sigmoid saturation over usage counts"), "smooth_saturation_penalty")

    def test_tsp_gls_requires_generation_planning_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_gls_qwen_20x20.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["agent"]["enable_generation_planning"] = False
            with self.assertRaises(RuntimeError):
                ExpertEoHEngine(config)

    def test_mock_engine_smoke_and_one_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/bp_online.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["generations"] = 1
            config["evolution"]["offspring_per_generation"] = 1
            engine = ExpertEoHEngine(config)
            population = engine.run()
            self.assertGreaterEqual(len(population), 1)
            self.assertTrue((Path(tmp) / "run" / "candidate_store.json").exists())
            self.assertTrue((Path(tmp) / "run" / "compact_context.json").exists())
            self.assertTrue((Path(tmp) / "run" / "results" / "pops" / "population_generation_1.json").exists())
            compact_context = (Path(tmp) / "run" / "compact_context.json").read_text(encoding="utf-8")
            self.assertIn("recent_successes", compact_context)
            self.assertIn("important_recent_events", compact_context)
            self.assertIn("mechanism_archive", compact_context)
            notebook = (Path(tmp) / "run" / "expert_notebook.json").read_text(encoding="utf-8")
            self.assertIn("hypothesis=", notebook)
            self.assertTrue((Path(tmp) / "run" / "mechanism_archive.json").exists())
            self.assertTrue((Path(tmp) / "run" / "results" / "workflow_audit" / "workflow_generation_1.json").exists())

    def test_mock_engine_tsp_one_generation_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config("eohagent/config/tsp_construct.yaml")
            config["project"]["output_dir"] = str(Path(tmp) / "run")
            config["evolution"]["generations"] = 1
            config["evolution"]["offspring_per_generation"] = 1
            engine = ExpertEoHEngine(config)
            population = engine.run()
            self.assertGreaterEqual(len(population), 1)
            self.assertTrue((Path(tmp) / "run" / "results" / "pops" / "population_generation_1.json").exists())
            self.assertTrue((Path(tmp) / "run" / "results" / "workflow_audit" / "workflow_generation_1.json").exists())


if __name__ == "__main__":
    unittest.main()

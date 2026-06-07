from __future__ import annotations

import json
import os
import time

import requests

from eohagent.llm.prompt_compression import PromptCompressionManager


class MockLLMClient:
    def __init__(self, problem: str):
        self.problem = problem
        self.variation_count = 0
        self.last_usage = {}

    def complete(self, prompt: str, system: str = "") -> str:
        try:
            payload = self._payload_from_prompt(prompt)
        except json.JSONDecodeError:
            return self._complete_code_prompt(prompt)
        purpose = payload.get("purpose")
        state = payload.get("state", {})
        parent = state.get("parent", {})
        parent_code = parent.get("code", "")

        if purpose == "decide_next_skill":
            step = state.get("step_count", 0)
            if step == 0:
                skill = "mechanism_decomposition"
            elif step == 1:
                skill = "controlled_variation"
            elif step == 2:
                skill = "contract_check"
            else:
                skill = "notebook_update"
            return json.dumps({"next_skill": skill, "reason": "mock deterministic progression", "arguments": {}})

        if purpose == "mechanism_decomposition":
            return json.dumps(
                {
                    "dominant_mechanism": "baseline-preserving greedy score",
                    "components": [{"name": "base score", "code_region": "function body", "intended_effect": "rank feasible options", "possible_risk": "simple baseline may be weak"}],
                    "strengths": ["simple", "stable"],
                    "weaknesses": ["limited adaptivity"],
                    "safe_to_modify": ["add a small normalized correction"],
                    "should_preserve": ["function signature", "feasibility assumptions"],
                }
            )

        if purpose == "generation_experiment_plan":
            parent_ids = payload.get("selected_parent_ids") or []
            if not parent_ids:
                raise ValueError("mock generation planner requires selected_parent_ids")
            return json.dumps(
                {
                    "generation_goal": "Test one small controlled variation for each selected parent.",
                    "avoid_repeating": ["avoid unchanged parent code"],
                    "offspring_plans": [
                        {
                            "offspring_index": index + 1,
                            "parent_id": parent_id,
                            "controlled_component": "update_strength",
                            "delta": "Change one small mechanism knob while preserving the parent core.",
                            "preserve": "signature and parent mechanism",
                            "avoid": "unrelated rewrite",
                            "expected_effect": "objective should improve or expose a clear failed direction",
                        }
                        for index, parent_id in enumerate(parent_ids)
                    ],
                }
            )

        if purpose == "generation_end_reflection":
            return json.dumps(
                {
                    "validated_principles": ["Mock generation completed with objective-only selection."],
                    "failed_hypotheses": [],
                    "open_questions": ["Which controlled delta should be tested next?"],
                }
            )

        if purpose == "controlled_variation":
            self.variation_count += 1
            if self.problem == "bp_online":
                if self.variation_count % 3 == 1:
                    code = (
                        "def score(item, bins):\n"
                        "    import numpy as np\n"
                        "    remaining_after = bins - item\n"
                        "    tight_fit = -remaining_after\n"
                        "    normalized_slack = remaining_after / (np.max(bins) + 1e-9)\n"
                        "    scores = tight_fit - 0.05 * normalized_slack\n"
                        "    return scores"
                    )
                elif self.variation_count % 3 == 2:
                    code = (
                        "def score(item, bins):\n"
                        "    import numpy as np\n"
                        "    remaining_after = bins - item\n"
                        "    scores = -remaining_after + 10000000.0\n"
                        "    return scores"
                    )
                else:
                    code = (
                        "def score(item, bins):\n"
                        "    import numpy as np\n"
                        "    pressure = item / (bins + 1e-9)\n"
                        "    return item - bins + 0.01 * pressure"
                    )
            elif self.problem == "tsp_construct":
                if self.variation_count % 2 == 1:
                    code = (
                        "import numpy as np\n\n"
                        "def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):\n"
                        "    distances = np.array([distance_matrix[current_node][i] for i in univisited_nodes])\n"
                        "    return univisited_nodes[int(np.argmin(distances))]"
                    )
                else:
                    code = (
                        "import numpy as np\n\n"
                        "def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):\n"
                        "    distances = np.array([distance_matrix[current_node][i] for i in univisited_nodes])\n"
                        "    destination_bias = np.array([distance_matrix[destination_node][i] for i in univisited_nodes])\n"
                        "    return univisited_nodes[int(np.argmin(distances + 0.01 * destination_bias))]"
                    )
            elif self.problem == "tsp_gls":
                if self.variation_count % 2 == 1:
                    code = (
                        "import numpy as np\n\n"
                        "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
                        "    updated_edge_distance = edge_distance.copy()\n"
                        "    n = len(local_opt_tour)\n"
                        "    for i in range(n):\n"
                        "        a = int(local_opt_tour[i])\n"
                        "        b = int(local_opt_tour[(i + 1) % n])\n"
                        "        usage = edge_n_used[a, b]\n"
                        "        penalty = 1.0 + 0.1 * (1.0 + usage)\n"
                        "        updated_edge_distance[a, b] *= penalty\n"
                        "        updated_edge_distance[b, a] *= penalty\n"
                        "    return updated_edge_distance"
                    )
                else:
                    code = (
                        "import numpy as np\n\n"
                        "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
                        "    updated_edge_distance = edge_distance.copy()\n"
                        "    n = len(local_opt_tour)\n"
                        "    mean_usage = np.mean(edge_n_used) + 1e-9\n"
                        "    for i in range(n):\n"
                        "        a = int(local_opt_tour[i])\n"
                        "        b = int(local_opt_tour[(i + 1) % n])\n"
                        "        usage = edge_n_used[a, b]\n"
                        "        normalized_usage = usage / mean_usage\n"
                        "        penalty = 1.0 + 0.08 * np.log1p(usage) + 0.02 * normalized_usage\n"
                        "        updated_edge_distance[a, b] *= penalty\n"
                        "        updated_edge_distance[b, a] *= penalty\n"
                        "    return updated_edge_distance"
                    )
            else:
                code = parent_code
            return json.dumps(
                {
                    "hypothesis": "a small normalized correction can improve robustness while preserving the baseline",
                    "preserve": ["signature", "baseline core"],
                    "modify": ["score calibration"],
                    "remove": [],
                    "expected_effect": "similar feasibility with modest ranking adjustment",
                    "possible_side_effect": "may be equivalent to parent on simple instances",
                    "candidate_algorithm": "Baseline-preserving controlled variation with a small normalized correction.",
                    "candidate_code": code,
                }
            )

        if purpose == "repair":
            return json.dumps(
                {
                    "repair_reason": "unsafe parent-code repair response used by mock client",
                    "preserved_mechanism": "parent mechanism",
                    "changed_parts": ["reverted candidate code"],
                    "repaired_algorithm": parent.get("algorithm", "parent repair"),
                    "repaired_code": parent_code,
                    "expected_fix": "restore executable contract",
                }
            )

        if purpose == "notebook_update":
            return json.dumps(
                {
                    "lessons": ["Mock run completed one candidate cycle with skill instructions available."],
                    "anti_patterns": [],
                    "open_questions": ["Does the latest controlled variation improve after formal evaluation?"],
                }
            )

        return "{}"

    def _complete_code_prompt(self, prompt: str) -> str:
        self.variation_count += 1
        if self.problem == "tsp_construct":
            if self.variation_count % 2 == 1:
                code = (
                    "import numpy as np\n\n"
                    "def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):\n"
                    "    distances = np.array([distance_matrix[current_node][i] for i in univisited_nodes])\n"
                    "    return univisited_nodes[int(np.argmin(distances))]"
                )
            else:
                code = (
                    "import numpy as np\n\n"
                    "def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):\n"
                    "    distances = np.array([distance_matrix[current_node][i] for i in univisited_nodes])\n"
                    "    destination_bias = np.array([distance_matrix[destination_node][i] for i in univisited_nodes])\n"
                    "    return univisited_nodes[int(np.argmin(distances + 0.01 * destination_bias))]"
                )
        elif self.problem == "tsp_gls":
            code = (
                "import numpy as np\n\n"
                "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
                "    updated_edge_distance = edge_distance.copy()\n"
                "    n = len(local_opt_tour)\n"
                "    for i in range(n):\n"
                "        a = int(local_opt_tour[i])\n"
                "        b = int(local_opt_tour[(i + 1) % n])\n"
                "        usage = edge_n_used[a, b]\n"
                "        penalty = 1.0 + 0.1 * (1.0 + usage)\n"
                "        updated_edge_distance[a, b] *= penalty\n"
                "        updated_edge_distance[b, a] *= penalty\n"
                "    return updated_edge_distance"
            )
        elif self.problem == "bp_online":
            code = (
                "def score(item, bins):\n"
                "    import numpy as np\n"
                "    pressure = item / (bins + 1e-9)\n"
                "    return item - bins + 0.01 * pressure"
            )
        else:
            raise ValueError(f"mock client does not support code prompts for problem: {self.problem}")
        return "{Mock baseline-preserving code variation.}\n```python\n" + code + "\n```"

    @staticmethod
    def _payload_from_prompt(prompt: str) -> dict:
        try:
            return json.loads(prompt)
        except json.JSONDecodeError:
            for marker in ("DYNAMIC_INPUT_JSON:", "INPUT_JSON:"):
                if marker in prompt:
                    return json.loads(prompt.split(marker, 1)[1].strip())
            raise


class OpenAICompatibleClient:
    def __init__(self, config: dict):
        llm_config = config["llm"]
        self.api_base = (
            os.getenv("AGENT_EOHS_BASE_URL")
            or llm_config.get("api_base")
            or llm_config.get("api_endpoint")
        )
        self.api_key = os.getenv("AGENT_EOHS_API_KEY") or llm_config.get("api_key")
        self.model = os.getenv("AGENT_EOHS_MODEL") or llm_config.get("model")
        self.timeout = int(llm_config.get("timeout", 120))
        self.temperature = float(llm_config.get("temperature", 0.7))
        self.transport_retries = int(llm_config.get("transport_retries", 3))
        self.last_usage = {}
        if not self.api_base or not self.api_key or str(self.api_key).startswith("${"):
            raise ValueError("openai_compatible provider requires api_base and api_key")
        self.thinking = llm_config.get("thinking")
        self.prompt_compression = PromptCompressionManager(config)
        self.last_prompt_compression = {}

    def compress_prompt(self, text: str) -> str:
        prepared = self.prompt_compression.prepare_text(text)
        self.last_prompt_compression = dict(self.prompt_compression.last_stats)
        return prepared

    def prepare_prompt(self, text: str, purpose: str = "llm_call") -> str:
        prepared = self.prompt_compression.prepare_text(text, purpose=purpose)
        self.last_prompt_compression = dict(self.prompt_compression.last_stats)
        return prepared

    def prepare_json_prompt(self, payload: dict, required: list[str]) -> str:
        prepared = self.prompt_compression.prepare_json_user(payload, required)
        self.last_prompt_compression = dict(self.prompt_compression.last_stats)
        return prepared

    def complete(self, prompt: str, system: str = "") -> str:
        base = self.api_base.rstrip("/")
        url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.thinking:
            request_body["thinking"] = self._thinking_payload(self.thinking)
        response = None
        for attempt in range(self.transport_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    data=json.dumps(request_body),
                    timeout=self.timeout,
                )
                break
            except requests.exceptions.RequestException:
                if attempt >= self.transport_retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        response.raise_for_status()
        data = response.json()
        self.last_usage = data.get("usage", {}) or {}
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _thinking_payload(value):
        if isinstance(value, dict):
            return value
        value = str(value).strip().lower()
        if value in {"disabled", "disable", "off", "false", "0"}:
            return {"type": "disabled"}
        if value in {"enabled", "enable", "on", "true", "1"}:
            return {"type": "enabled"}
        raise ValueError(f"unsupported llm.thinking value: {value!r}")


class LocalHTTPClient:
    def __init__(self, config: dict):
        llm_config = config["llm"]
        self.url = llm_config.get("api_base")
        self.timeout = int(llm_config.get("timeout", 120))
        self.last_usage = {}
        if not self.url:
            raise ValueError("local_http provider requires api_base")

    def complete(self, prompt: str, system: str = "") -> str:
        response = requests.post(self.url, json={"prompt": prompt}, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        self.last_usage = data.get("usage", {}) or {
            key: data.get(key)
            for key in ["prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration"]
            if key in data
        }
        return data.get("response") or data.get("content") or json.dumps(data)


def build_llm_client(config: dict):
    provider = config["llm"].get("provider", "mock")
    if provider == "mock":
        return MockLLMClient(config["project"]["problem"])
    if provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    if provider == "local_http":
        return LocalHTTPClient(config)
    raise ValueError(f"Unsupported LLM provider: {provider}")

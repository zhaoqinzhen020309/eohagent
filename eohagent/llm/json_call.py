from __future__ import annotations

import ast
import json
import re

from eohagent.llm.prompt_compression import PromptCompressionManager

JSON_CALL_SYSTEM = (
    "You are a JSON-only API inside an automated algorithm-design system.\n"
    "Return exactly one valid JSON object and nothing else.\n"
    "Do not include markdown fences, explanations, comments, or <think> text.\n"
    "Use double quotes for all keys and string values."
)


class JSONCall:
    def __init__(self, client, max_retries: int = 3, context_runtime=None):
        self.client = client
        self.max_retries = max_retries
        self.last_response = ""
        self.last_usage = {}
        self.last_prompt_compression = {}
        self.context_runtime = context_runtime

    def set_context_runtime(self, context_runtime) -> None:
        self.context_runtime = context_runtime

    def call(self, payload: dict, required: list[str] | None = None, context_state=None) -> dict:
        return self.call_dynamic(lambda: payload, required=required, context_state=context_state)

    def call_dynamic(self, payload_factory, required: list[str] | None = None, context_state=None) -> dict:
        required = required or []

        initial_payload = payload_factory()
        system = JSON_CALL_SYSTEM
        user = self._prepare_json_prompt(
            purpose=initial_payload.get("purpose") if isinstance(initial_payload, dict) else "json_call",
            payload_factory=payload_factory,
            required=required,
            context_state=context_state,
        )
        last_error = None
        for _ in range(self.max_retries):
            response = self.client.complete(user, system=system)
            self.last_response = response
            self.last_usage = dict(getattr(self.client, "last_usage", {}) or {})
            try:
                data = self.extract_json(response)
                missing = [key for key in required if key not in data]
                if missing:
                    raise ValueError(f"Missing required keys: {missing}")
                return data
            except Exception as exc:
                last_error = exc
                repair_payload = {
                    "purpose": "json_format_retry",
                    "error": str(exc),
                    "previous_response": response[:4000],
                    "required": required,
                    "instruction": "Rewrite the previous response as one strict JSON object only.",
                }
                user = self._prepare_json_prompt(
                    purpose="json_format_retry",
                    payload_factory=lambda: repair_payload,
                    required=required,
                    context_state=context_state,
                )
        raise RuntimeError(f"LLM JSON call failed: {last_error}")

    def complete_text_dynamic(self, prompt_factory, purpose: str, context_state=None, system: str = "") -> str:
        prompt = self._prepare_prompt(
            purpose=purpose,
            prompt=prompt_factory(),
            rebuild_prompt=prompt_factory,
            context_state=context_state,
        )
        response = self.client.complete(prompt, system=system)
        self.last_response = response
        self.last_usage = dict(getattr(self.client, "last_usage", {}) or {})
        return response

    def _prepare_prompt(self, purpose: str, prompt: str, rebuild_prompt, context_state=None) -> str:
        if hasattr(self.client, "prepare_prompt"):
            prompt = self.client.prepare_prompt(prompt, purpose=purpose)
            _rebuild = rebuild_prompt
            rebuild_prompt = lambda: self.client.prepare_prompt(_rebuild(), purpose=purpose)
            self.last_prompt_compression = dict(getattr(self.client, "last_prompt_compression", {}) or {})
        if self.context_runtime is None:
            return prompt
        return self.context_runtime.ensure_prompt(
            purpose=purpose or "llm_call",
            prompt=prompt,
            rebuild_prompt=rebuild_prompt,
            state=context_state,
        )

    def _prepare_json_prompt(self, purpose: str, payload_factory, required: list[str], context_state=None) -> str:
        def build_user() -> str:
            payload = payload_factory()
            if hasattr(self.client, "prepare_json_prompt"):
                user = self.client.prepare_json_prompt(payload, required)
                self.last_prompt_compression = dict(getattr(self.client, "last_prompt_compression", {}) or {})
                return user
            return self._build_user(payload, required)

        user = build_user()
        if self.context_runtime is None:
            return user
        return self.context_runtime.ensure_prompt(
            purpose=purpose or "json_call",
            prompt=user,
            rebuild_prompt=build_user,
            state=context_state,
        )

    @staticmethod
    def _build_user(payload: dict, required: list[str]) -> str:
        return PromptCompressionManager.render_json_user(payload, required)

    @staticmethod
    def extract_json(response: str) -> dict:
        text = JSONCall._clean_response(response)
        candidates = [text]
        for unescaped in JSONCall._unescaped_candidates(text):
            if unescaped not in candidates:
                candidates.append(unescaped)
        balanced = JSONCall._first_balanced_object(text)
        if balanced and balanced != text:
            candidates.append(balanced)
        for unescaped in list(candidates):
            balanced = JSONCall._first_balanced_object(unescaped)
            if balanced and balanced not in candidates:
                candidates.append(balanced)

        last_error = None
        for candidate in candidates:
            for variant in JSONCall._json_variants(candidate):
                try:
                    data = json.loads(variant)
                    if isinstance(data, dict):
                        return data
                    if isinstance(data, str):
                        nested = JSONCall.extract_json(data)
                        if isinstance(nested, dict):
                            return nested
                except Exception as exc:
                    last_error = exc
                try:
                    data = ast.literal_eval(variant)
                    if isinstance(data, dict):
                        return data
                    if isinstance(data, str):
                        nested = JSONCall.extract_json(data)
                        if isinstance(nested, dict):
                            return nested
                except Exception as exc:
                    last_error = exc
        raise last_error or ValueError("No JSON object found")

    @staticmethod
    def _clean_response(response: str) -> str:
        text = response.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        return text

    @staticmethod
    def _unescaped_candidates(text: str) -> list[str]:
        candidates = []
        if any(token in text for token in ['\\"', "\\'", "\\n", "\\{", "\\}"]):
            manual = (
                text.replace('\\"', '"')
                .replace("\\'", "'")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace("\\{", "{")
                .replace("\\}", "}")
                .replace("\\[", "[")
                .replace("\\]", "]")
            )
            candidates.append(manual)
            try:
                candidates.append(bytes(text, "utf-8").decode("unicode_escape"))
            except Exception:
                pass
        return [candidate.strip() for candidate in candidates if candidate.strip()]

    @staticmethod
    def _first_balanced_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        quote = ""
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                continue
            if ch in {'"', "'"}:
                in_string = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return text[start:]

    @staticmethod
    def _json_variants(text: str) -> list[str]:
        variants = [text.strip()]
        relaxed = text.strip()
        relaxed = re.sub(r",\s*([}\]])", r"\1", relaxed)
        relaxed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', relaxed)
        variants.append(relaxed)
        variants.append(relaxed.replace("'", '"'))
        return list(dict.fromkeys(variants))

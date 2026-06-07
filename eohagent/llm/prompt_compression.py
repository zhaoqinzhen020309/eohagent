from __future__ import annotations

import copy
import json
from typing import Any


class PromptCompressionError(RuntimeError):
    """Raised when configured prompt compression cannot be applied."""


class PromptCompressionManager:
    PROTECTED_KEYS = {
        "instruction",
        "required_output",
        "arch_spec_schema",
        "required_offspring_slots",
        "parent_arch_spec",
        "child_spec",
        "arch_spec",
        "architecture_open_layers",
        "function_name",
        "architecture_class_name",
        "selected_parent_ids",
        "allowed_parent_ids",
        "current_population_parent_ids",
        "parent_id",
        "offspring_index",
        "controlled_component",
        "code",
        "parent_code",
        "candidate_code",
        "repaired_code",
    }

    FORCE_TOKENS = ["\n", ":", "{", "}", "[", "]", '"', "_"]

    def __init__(self, config: dict):
        pc = config.get("prompt_compression", {})
        self.enabled = bool(pc.get("enabled", False))
        self.threshold_chars = int(pc.get("threshold_chars", 8000))
        self.rate = float(pc.get("rate", 0.55))
        self.min_segment_chars = int(pc.get("min_segment_chars", 240))
        self.last_stats = {}
        self._compressor = None
        if not self.enabled:
            return
        try:
            from llmlingua import PromptCompressor
        except Exception as exc:
            raise PromptCompressionError(
                "prompt_compression.enabled is true, but llmlingua is unavailable"
            ) from exc
        try:
            self._compressor = PromptCompressor(
                model_name=pc.get(
                    "model_name",
                    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                ),
                use_llmlingua2=bool(pc.get("use_llmlingua2", True)),
                device_map=pc.get("device_map", "cpu"),
            )
        except Exception as exc:
            raise PromptCompressionError("failed to initialize LLMLingua prompt compressor") from exc

    def prepare_text(self, text: str, purpose: str = "llm_call") -> str:
        if not self.enabled:
            self.last_stats = self._stats(purpose, len(text), len(text), compressed=False, mode="disabled")
            return text
        if len(text) <= self.threshold_chars:
            self.last_stats = self._stats(purpose, len(text), len(text), compressed=False, mode="below_threshold")
            return text
        return self._compress_text(text, purpose=purpose, mode="text")

    def prepare_json_user(self, payload: dict, required: list[str]) -> str:
        raw = self.render_json_user(payload, required)
        purpose = str(payload.get("purpose") or "json_call") if isinstance(payload, dict) else "json_call"
        if not self.enabled:
            self.last_stats = self._stats(purpose, len(raw), len(raw), compressed=False, mode="disabled")
            return raw
        if len(raw) <= self.threshold_chars:
            self.last_stats = self._stats(purpose, len(raw), len(raw), compressed=False, mode="below_threshold")
            return raw
        compact_payload = self._compress_json_like(payload, protected=False)
        compressed = self.render_json_user(compact_payload, required)
        self.last_stats = self._stats(purpose, len(raw), len(compressed), compressed=True, mode="json_leaf_values")
        return compressed

    @staticmethod
    def render_json_user(payload: dict, required: list[str]) -> str:
        purpose = str(payload.get("purpose") or "json_call") if isinstance(payload, dict) else "json_call"
        return (
            PromptCompressionManager._stable_prefix(purpose)
            +
            f"Required top-level keys: {required}.\n"
            "DYNAMIC_INPUT_JSON:\n"
            + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        )

    def _compress_json_like(self, value: Any, protected: bool) -> Any:
        if protected:
            return copy.deepcopy(value)
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                key_text = str(key)
                out[key] = self._compress_json_like(item, key_text in self.PROTECTED_KEYS)
            return out
        if isinstance(value, list):
            return [self._compress_json_like(item, protected=False) for item in value]
        if isinstance(value, str) and len(value) >= self.min_segment_chars:
            return self._compress_segment(value)
        return copy.deepcopy(value)

    def _compress_segment(self, text: str) -> str:
        result = self._compressor.compress_prompt(
            [text],
            rate=self.rate,
            force_tokens=self.FORCE_TOKENS,
            strict_preserve_uncompressed=True,
        )
        compressed = result.get("compressed_prompt", text)
        if not isinstance(compressed, str) or not compressed:
            raise PromptCompressionError("LLMLingua returned an empty compressed segment")
        return compressed

    def _compress_text(self, text: str, purpose: str, mode: str) -> str:
        raw_len = len(text)
        compressed = self._compress_segment(text)
        self.last_stats = self._stats(purpose, raw_len, len(compressed), compressed=True, mode=mode)
        return compressed

    @staticmethod
    def _stats(purpose: str, raw_chars: int, prepared_chars: int, compressed: bool, mode: str) -> dict:
        return {
            "purpose": purpose,
            "enabled": mode != "disabled",
            "compressed": compressed,
            "mode": mode,
            "raw_chars": raw_chars,
            "prepared_chars": prepared_chars,
            "saved_chars": raw_chars - prepared_chars,
            "saved_ratio": 0.0 if raw_chars <= 0 else round((raw_chars - prepared_chars) / raw_chars, 6),
        }

    @staticmethod
    def _stable_prefix(purpose: str) -> str:
        common = (
            "STABLE_WORKFLOW_CONTRACT:\n"
            "- This is an auditable automated algorithm-design workflow.\n"
            "- No fallback plans, synthetic defaults, inferred parents, risk scores, or weak probe penalties are allowed.\n"
            "- If the requested JSON cannot satisfy the contract, return the closest valid JSON for the given dynamic input only.\n"
        )
        if purpose == "generation_experiment_plan":
            return (
                common
                + "STABLE_GENERATION_PLAN_RULES:\n"
                + "- Plan exactly one offspring item per required slot.\n"
                + "- Copy offspring_index and parent_id from required slots exactly.\n"
                + "- Use controlled experiment groups and change exactly one controlled component per offspring.\n"
                + "- For ARCH_SPEC workflows, child_spec must be complete and only the controlled layer may differ.\n"
            )
        if purpose == "generation_end_reflection":
            return (
                common
                + "STABLE_REFLECTION_RULES:\n"
                + "- Summarize mechanism evidence by controlled experiment groups.\n"
                + "- Attribute deltas using mean, tail, max, and failure-signature evidence.\n"
                + "- Record reusable principles and refuted directions; do not list every raw event.\n"
            )
        if purpose == "json_format_retry":
            return common + "STABLE_JSON_RETRY_RULES:\n- Return one strict JSON object and no prose.\n"
        return common

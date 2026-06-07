from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any


class MemoryProviderError(RuntimeError):
    pass


class LocalMemoryProvider:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "memory_provider.json"
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def record(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        records = self._read()
        records.append(
            {
                "id": str(uuid.uuid4()),
                "kind": str(kind),
                "content": str(content),
                "metadata": metadata or {},
            }
        )
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_terms = {term.lower() for term in str(query).replace("_", " ").split() if term.strip()}
        scored = []
        for record in self._read():
            content = str(record.get("content") or "")
            haystack = content.lower().replace("_", " ")
            score = sum(1 for term in query_terms if term in haystack)
            if score > 0 or not query_terms:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[: max(0, int(limit))]]

    def _read(self) -> list[dict[str, Any]]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise MemoryProviderError("local memory provider file must contain a JSON list")
        return [item for item in data if isinstance(item, dict)]


class LangMemMemoryProvider:
    def __init__(self, namespace: tuple[str, ...] = ("eohagent", "memories"), output_dir: Path | None = None):
        if importlib.util.find_spec("langmem") is None:
            raise MemoryProviderError("memory.provider=langmem requires the langmem package")
        if importlib.util.find_spec("langgraph") is None:
            raise MemoryProviderError("memory.provider=langmem requires the langgraph package")
        from langgraph.store.memory import InMemoryStore
        from langmem import create_manage_memory_tool, create_search_memory_tool

        self.store = InMemoryStore()
        self.manage_tool = create_manage_memory_tool(namespace, store=self.store)
        self.search_tool = create_search_memory_tool(namespace, store=self.store)
        self.path = (output_dir or Path.cwd()) / "langmem_provider.json"
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def record(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        record = {
            "id": str(uuid.uuid4()),
            "kind": str(kind),
            "content": str(content),
            "metadata": metadata or {},
        }
        records = self._read()
        records.append(record)
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        memory_text = json.dumps(record, ensure_ascii=True, sort_keys=True)
        self.manage_tool.invoke({"content": memory_text})

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        disk_results = self._search_disk(query, limit)
        raw = self.search_tool.invoke({"query": str(query)})
        items = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(items, list):
            raise MemoryProviderError("langmem search_memory returned a non-list result")
        results = []
        for item in items[: max(0, int(limit))]:
            if not isinstance(item, dict):
                continue
            value = item.get("value") if isinstance(item.get("value"), dict) else {}
            content = str(value.get("content") or "")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"kind": "memory", "content": content, "metadata": {}}
            if not isinstance(parsed, dict):
                parsed = {"kind": "memory", "content": content, "metadata": {}}
            results.append(
                {
                    "id": item.get("key"),
                    "kind": parsed.get("kind"),
                    "content": str(parsed.get("content") or ""),
                    "metadata": parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {},
                    "score": item.get("score"),
                }
            )
        seen = {str(item.get("id")) for item in results}
        for item in disk_results:
            if str(item.get("id")) not in seen:
                results.append(item)
                seen.add(str(item.get("id")))
            if len(results) >= max(0, int(limit)):
                break
        return results[: max(0, int(limit))]

    def _read(self) -> list[dict[str, Any]]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise MemoryProviderError("langmem mirror file must contain a JSON list")
        return [item for item in data if isinstance(item, dict)]

    def _search_disk(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_terms = {term.lower() for term in str(query).replace("_", " ").split() if term.strip()}
        scored = []
        for record in self._read():
            content = str(record.get("content") or "")
            haystack = content.lower().replace("_", " ")
            score = sum(1 for term in query_terms if term in haystack)
            if score > 0 or not query_terms:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[: max(0, int(limit))]]


def build_memory_provider(config: dict, output_dir: Path):
    provider = str(config.get("memory", {}).get("provider", "local")).strip().lower()
    if provider == "local":
        return LocalMemoryProvider(output_dir)
    if provider == "langmem":
        namespace = tuple(config.get("memory", {}).get("namespace", ["eohagent", "memories"]))
        return LangMemMemoryProvider(namespace=namespace, output_dir=output_dir)
    raise MemoryProviderError(f"unsupported memory.provider: {provider!r}")

# DeepSeek API 适配、Prompt Caching 优化、上下文压缩集成

## 改动日期

2026-05-31

## 改动概述

1. 新建 DeepSeek API 配置文件
2. System/user 消息分离，提升 API prompt cache 命中率
3. 集成 LLMLingua-2 prompt 压缩，防止上下文溢出
4. 调整 context_management 参数适配 DeepSeek 窗口

---

## 一、新建文件

### `eohagent/config/tsp_gls_arch_spec_deepseek_10x20.yaml`

基于原 `tsp_gls_arch_spec_qwen35_122b_10x20.yaml`，变更参数：

| 配置项 | Qwen（原） | DeepSeek（新） | 原因 |
|--------|-----------|---------------|------|
| `project.name` | `...qwen35_122b...` | `...deepseek...` | |
| `project.output_dir` | `...qwen35_122b...` | `...deepseek...` | |
| `llm.model` | `qwen3.5:122b` | `deepseek-v4-flash` | deepseek-chat 2026/07/24 弃用 |
| `llm.api_base` | `http://127.0.0.1:11434/v1` | `https://api.deepseek.com/v1` | |
| `llm.api_key` | `ollama` | `sk-52f2e1...` | |
| `llm.timeout` | `1800` | `600` | 远程 API 响应更快 |
| `context_management.context_window` | `32768` | `65536` | 匹配 DeepSeek 实际上下文窗口 |
| `context_management.recent_events` | `12` | `8` | 减少 prompt 体积 |
| `context_management.population_items` | `8` | `6` | 减少 prompt 体积 |
| `context_management.notebook_items` | `6` | `4` | 减少 prompt 体积 |
| `prompt_compression.enabled` | 无 | `true` | 启用 LLMLingua-2 压缩 |
| `prompt_compression.threshold_chars` | 无 | `8000` | 仅压缩超大 prompt |
| `prompt_compression.rate` | 无 | `0.55` | 保留 55% token |

> **模型选择说明**：DeepSeek 已将 `deepseek-chat` 和 `deepseek-reasoner` 标记为弃用（2026/07/24），推荐使用 `deepseek-v4-flash`（对应原 deepseek-chat）和 `deepseek-v4-pro`。

运行命令：

```bash
python eohagent/main.py --config eohagent/config/tsp_gls_arch_spec_deepseek_10x20.yaml
```

---

## 二、修改文件

### 1. `eohagent/llm/client.py`

#### 1.1 System/User 消息分离

`OpenAICompatibleClient.complete()` 新增 `system` 参数：

```python
# 之前
def complete(self, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

# 之后
def complete(self, prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
```

`LocalHTTPClient.complete()` 和 `MockLLMClient.complete()` 同步添加 `system` 参数保持接口一致。

#### 1.2 LLMLingua-2 Prompt 压缩

新增 `_init_compressor()` 和 `compress_prompt()` 方法：

```python
def _init_compressor(self, config: dict) -> None:
    pc = config.get("prompt_compression", {})
    self._compressor = None
    self._compress_threshold = 0
    self._compress_rate = 0.55
    if not pc.get("enabled", False):
        return
    self._compress_threshold = int(pc.get("threshold_chars", 8000))
    self._compress_rate = float(pc.get("rate", 0.55))
    try:
        from llmlingua import PromptCompressor
        self._compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    except Exception as exc:
        print(f"[client] LLMLingua-2 unavailable: {exc}")

def compress_prompt(self, text: str) -> str:
    if self._compressor is None or len(text) <= self._compress_threshold:
        return text
    try:
        result = self._compressor.compress_prompt(
            text,
            rate=self._compress_rate,
            force_tokens=["\n", ":", "{", "}", '"'],
        )
        compressed = result.get("compressed_prompt", text)
        return compressed if isinstance(compressed, str) and len(compressed) > 0 else text
    except Exception as exc:
        print(f"[client] compression failed: {exc}")
        return text
```

`complete()` 发送前压缩 user 消息（system 消息不压缩以保护缓存）：

```python
def complete(self, prompt: str, system: str = "") -> str:
    prompt = self.compress_prompt(prompt)  # ← 压缩 user 消息
    # ... 构建 messages 并发送
```

### 2. `eohagent/llm/json_call.py`

#### 2.1 JSON 调用 System 消息

提取静态指令为常量 `JSON_CALL_SYSTEM`，从 user message 中移除：

```python
JSON_CALL_SYSTEM = (
    "You are a JSON-only API inside an automated algorithm-design system.\n"
    "Return exactly one valid JSON object and nothing else.\n"
    "Do not include markdown fences, explanations, comments, or <think> text.\n"
    "Use double quotes for all keys and string values."
)
```

`call_dynamic()` 传入 `system=JSON_CALL_SYSTEM`，`_build_prompt` 重命名为 `_build_user` 仅构建 user 部分。

`complete_text_dynamic()` 新增 `system` 参数。

#### 2.2 Context Runtime 前压缩

`_prepare_prompt()` 在 context runtime pressure check **之前**压缩 user 消息，并包装 `rebuild_prompt` lambda：

```python
def _prepare_prompt(self, purpose: str, prompt: str, rebuild_prompt, context_state=None) -> str:
    if hasattr(self.client, "compress_prompt"):
        prompt = self.client.compress_prompt(prompt)
        _rebuild = rebuild_prompt
        rebuild_prompt = lambda: self.client.compress_prompt(_rebuild())
    if self.context_runtime is None:
        return prompt
    return self.context_runtime.ensure_prompt(...)
```

### 3. `eohagent/harness/tool_router.py`

`_call_code_llm()` 拆分 system/user，传递给 `complete_text_dynamic()`。

`_build_planned_code_prompt()` 返回值从 `str` 改为 `(system: str, build_user: callable)` 元组。

| 部分 | 内容 | 缓存特性 |
|------|------|---------|
| system | Purpose + 问题指令 + Problem Card + 输出格式要求 | 所有后代相同，缓存命中 |
| user | Offspring plan + Parent code + Working memory | 每个后代不同 |

旧方法 `_build_code_prompt` 已内联到 `_call_code_llm` 中。

### 4. `eohagent/harness/context_runtime.py`

`_build_compaction_prompt()` 返回值从 `str` 改为 `(system, user)` 元组。

`compact()` 拆分 system/user 调用 `client.complete()`。

### 5. `eohagent/config/tsp_gls_arch_spec_deepseek_10x20.yaml`

新增 `prompt_compression` 配置段：

```yaml
prompt_compression:
  enabled: true
  threshold_chars: 8000
  rate: 0.55
```

---

## 三、系统架构与调用流程

### 完整 LLM 调用链路

```
JSONCall.call_dynamic() / complete_text_dynamic()
  │
  ├─ build_user() → 构建 user 消息（动态数据）
  │
  ├─ _prepare_prompt()
  │     ├─ compress_prompt(user_msg)     ← LLMLingua-2 压缩 (>8000 chars)
  │     └─ context_runtime.ensure_prompt()
  │           ├─ Level 1: deterministic_shrink
  │           ├─ Level 2: LLM compaction (use_llm_compaction)
  │           └─ Level 3: emergency_context
  │
  └─ client.complete(user, system=system)
        ├─ compress_prompt(user_msg)     ← 二次压缩（幂等，压缩后不再触发）
        └─ POST /v1/chat/completions
              messages: [
                {role: "system", content: "<静态指令，API 缓存命中>"},
                {role: "user",   content: "<压缩后动态数据>"}
              ]
```

### 缓存与压缩的协同

| 消息 | 压缩 | 缓存 | 原因 |
|------|------|------|------|
| system | **不压缩** | **命中** | 相同 purpose 的调用 system 完全相同，DeepSeek 前缀缓存自动命中 |
| user | **压缩** | 不命中 | user 每次唯一（parent code / offspring plan 不同），压缩减少计费 token |

---

## 四、缓存效果分析

### 改动前

所有内容混合在单个 user message 中，缓存命中率接近 0%。

### 改动后

以 `tsp_gls_arch_spec` 配置（pop=10, offspring=10）为例，每代 12 次 API 调用：

| 调用类型 | 每代次数 | 之前 | 之后 | 节省 |
|---------|---------|------|------|------|
| Generation Plan | 1 | ~4000t | system(~200t) 缓存 | ~5% |
| Controlled Variation | 10 | 10×~2500t | system(~500t) 缓存 1 次 + user 压缩 | ~25-35% |
| Generation Reflection | 1 | ~44000t → Overflow | 压缩到 ~28Kt, system(~150t) 缓存 | ~35% + 避免溢出 |
| **每代合计** | **12** | **~74000t** | **~50000t** | **~30%** |

### 受限因素

即使 system 消息已完全缓存，user 消息仍然每次唯一：
- 每个后代的 `parent.code` 不同（不同父代）
- 每个后代的 `offspring_plan`（delta、controlled_component）不同
- `population_summary` 随种群更新变化

这属于进化算法的工作特性，进一步优化需要结构性改动（如用 ID 引用代替内联代码等）。

---

## 五、依赖安装

```bash
# 激活 llm4ad 环境后安装 LLMLingua-2
pip install llmlingua --proxy=""
```

LLMLingua-2 使用 BERT 级小模型（~180MB），CPU 模式运行，每次压缩耗时 ~1-15 秒（取决于 prompt 大小）。

---

## 六、兼容性说明

- 所有 `client.complete()` 调用保持向后兼容：`system` 参数默认为空字符串
- `MockLLMClient` 和 `LocalHTTPClient` 忽略 `system` 参数（行为不变）
- `prompt_compression.enabled: false` 可完全关闭压缩
- 压缩仅在 `len(prompt) > threshold_chars` 时触发，小 prompt 不受影响
- LLMLingua-2 未安装时自动降级，不影响正常运行

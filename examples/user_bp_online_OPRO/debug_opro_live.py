# debug_opro_live.py
import sys
import os
import numpy as np
import time

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh_OPRO', 'src'))

from eoh.llm.interface_LLM import InterfaceLLM


def test_meta_llm():
    """测试Meta-LLM生成策略"""
    print("\n" + "=" * 80)
    print("STEP 1: Testing Meta-LLM Strategy Generation")
    print("=" * 80)

    # 初始化LLM接口
    llm = InterfaceLLM(
        api_endpoint=None,
        api_key=None,
        model_LLM="qwen2.5-coder:32b",
        llm_use_local=True,
        llm_local_url="http://localhost:11434/api/generate",
        debug_mode=False
    )

    # 构建meta-prompt（使用你当前的版本）
    meta_prompt = """You are optimizing a SCORE FUNCTION for online bin packing at generation 2.

PROBLEM: score(item, bins) returns scores for bins. Highest score bin gets the item.
Goal: Minimize total bins used.

HISTORICAL PERFORMANCE:
Generation 1: improvement=0.808, success_rate=0.60, best_fitness=0.039840

ANALYSIS: Best fitness 0.8080, Success rate 0.60, Trend: insufficient_data

TASK: Suggest improvements to the score function for generation 2.

FOCUS ON:
1. Core bin packing principles: tight fit, avoid fragmentation, capacity utilization
2. Mathematical expressions using: item, bins, np.exp, np.log, np.sqrt, ratios
3. Combine 2-4 scoring terms with weighted sum or product

CRITICAL CONSTRAINTS TO PREVENT OVERFITTING:
- Keep formula SIMPLE (under 10 lines of code)
- Use GENERAL principles, not distribution-specific tricks
- Avoid complex conditionals or item-size-specific thresholds
- The algorithm must work well across different item size distributions (Weibull, Gaussian, Uniform, etc.)
- Focus on ROBUST features: remaining capacity ratios, relative fill levels, not absolute values

Generate strategy:"""

    print("\n[Meta-Prompt]")
    print(meta_prompt)
    print("\n[Calling Meta-LLM...]")

    start = time.time()
    strategy = llm.get_response(meta_prompt)
    elapsed = time.time() - start

    print(f"\n[Meta-LLM Response] (took {elapsed:.2f}s)")
    print("-" * 80)
    print(strategy)
    print("-" * 80)

    return strategy


def test_worker_llm(strategy):
    """测试Worker-LLM生成代码"""
    print("\n" + "=" * 80)
    print("STEP 2: Testing Worker-LLM Code Generation")
    print("=" * 80)

    # 初始化LLM接口
    llm = InterfaceLLM(
        api_endpoint=None,
        api_key=None,
        model_LLM="qwen2.5-coder:32b",
        llm_use_local=True,
        llm_local_url="http://localhost:11434/api/generate",
        debug_mode=False
    )

    # 构建Worker-LLM的prompt（模拟实际使用的prompt）
    worker_prompt = f"""Based on the following evolution strategy and parent algorithms, generate a new heuristic:

Strategy:
{strategy}

Parent algorithms:
Parent 1 algorithm: The new algorithm calculates scores for each bin based on a weighted combination of the remaining capacity, proximity to maximum capacity, and a penalty for larger bins, aiming to minimize the number of used bins.
Code:
import numpy as np

def score(item, bins):
    max_capacity = 100  # Assuming a fixed maximum capacity for bins
    scores = (bins - item) / max_capacity * np.exp(-np.abs(bins - max_capacity) / max_capacity) - np.log2(bins + 1)
    return scores

The bin packing problem is to pack a set of items of different sizes into a minimum number of bins with a fixed capacity. You need to design a heuristic function named score(item, bins) to decide into which bin the item will be packed.

Please design a new heuristic algorithm.
First, describe your algorithm and main steps in one sentence. The description must be inside a brace.
Next, implement it in Python as a function named score.
This function should accept 2 input(s): 'item', 'bins'.
The function should return 1 output(s): 'scores'.
The function should return an array with the same length as bins.

Do not give additional explanations."""

    print("\n[Worker-Prompt]")
    print(worker_prompt[:500] + "\n...\n" + worker_prompt[-200:])
    print("\n[Calling Worker-LLM...]")

    start = time.time()
    response = llm.get_response(worker_prompt)
    elapsed = time.time() - start

    print(f"\n[Worker-LLM Response] (took {elapsed:.2f}s)")
    print("-" * 80)
    print(response)
    print("-" * 80)

    return response


def extract_and_test_code(llm_response):
    """提取代码并测试执行"""
    print("\n" + "=" * 80)
    print("STEP 3: Testing Code Execution")
    print("=" * 80)

    import re

    # 清理response
    response = re.sub(r'<think>.*?</think>', '', llm_response, flags=re.DOTALL)
    response = re.sub(r'```python\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    # 提取代码
    code = re.findall(r'(import.*?return[^}]*)', response, re.DOTALL)
    if len(code) == 0:
        code = re.findall(r'(def.*?return[^}]*)', response, re.DOTALL)

    if len(code) == 0:
        print("[ERROR] Could not extract code from response")
        return

    code_str = code[0].strip()
    print("\n[Extracted Code]")
    print("-" * 80)
    print(code_str)
    print("-" * 80)

    # 补全return语句
    if not code_str.strip().endswith('return'):
        code_str = code_str.rstrip() + "\n    return scores"

    print("\n[Complete Code with Return]")
    print("-" * 80)
    print(code_str)
    print("-" * 80)

    # 测试执行
    test_cases = [
        (10, np.array([50, 40, 30, 90, 100])),
        (5, np.array([100, 100, 100, 100, 100])),
        (50, np.array([60, 70, 80, 90, 100]))
    ]

    for i, (item, bins) in enumerate(test_cases):
        print(f"\n[Test Case {i + 1}] item={item}, bins={bins}")

        namespace = {'np': np}

        try:
            # 设置超时（简单版本：记录时间）
            start = time.time()
            exec(code_str, namespace)

            if 'score' not in namespace:
                print("  [ERROR] No score function defined")
                continue

            result = namespace['score'](item, bins)
            elapsed = time.time() - start

            print(f"  [SUCCESS] Result: {result}")
            print(f"  [TIME] {elapsed:.4f}s")

            if elapsed > 5.0:
                print(f"  [WARNING] Execution took {elapsed:.2f}s (>5s threshold)")

        except Exception as e:
            elapsed = time.time() - start
            print(f"  [ERROR] Execution failed after {elapsed:.4f}s")
            print(f"  [ERROR TYPE] {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# OPRO Framework Live Debug")
    print("#" * 80)

    try:
        # Step 1: 测试Meta-LLM
        strategy = test_meta_llm()

        # Step 2: 测试Worker-LLM
        worker_response = test_worker_llm(strategy)

        # Step 3: 测试代码执行
        extract_and_test_code(worker_response)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Debug stopped by user")
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\n" + "#" * 80)
    print("# Debug Complete")
    print("#" * 80)
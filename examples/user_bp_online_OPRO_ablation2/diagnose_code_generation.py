"""
代码生成失败诊断脚本

目标: 诊断为什么 Worker LLM 生成的代码全部评估失败

诊断流程:
1. 模拟实际的 P_final 提示词
2. 调用 Worker LLM (或使用模拟代码)
3. 检查语法错误
4. 尝试评估代码
5. 报告具体失败原因
"""

import sys
import os
import ast
import numpy as np
import types
import warnings
import concurrent.futures
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
eoh_src_path = os.path.join(current_dir, '..', '..', 'eoh_OPRO', 'src')
sys.path.insert(0, eoh_src_path)

from eoh.llm.interface_LLM import InterfaceLLM
from prob import BPONLINE

# ============================================================================
# 配置
# ============================================================================

LLM_CONFIG = {
    'use_local': True,
    'local_url': 'http://localhost:11434/api/generate',
    'api_endpoint': None,
    'api_key': None,
    'model': 'qwen2.5-coder:32b'
}

TIMEOUT = 30  # 评估超时时间（秒）

# ============================================================================
# 模拟典型的 P_final（来自实际运行）
# ============================================================================

SAMPLE_P_FINAL = """Design an improved bin scoring function that refines the mathematical patterns from top-performing solutions.

**Your Task:**
Create a Python function named `score` that takes two parameters:
- `item` (float): The size of the current item (0 < item <= 1)
- `bins` (numpy array): Remaining capacities of feasible bins

The function must return a numpy array of scores where HIGHER scores indicate better bins.

**Requirements:**
1. Function must be named `score`
2. Must accept parameters `item` and `bins`
3. Must return numpy array of same length as `bins`
4. Use only Python standard library and NumPy (`import numpy as np`)
5. Must be deterministic (no random operations)
6. Higher scores = better bin choices

**Strategy:**
- Use a combination of remaining capacity and fullness metrics
- Apply normalization to ensure scores are comparable
- Include penalty terms for edge cases
- Consider the ratio of item size to bin capacity

**Example Structure:**
```python
import numpy as np

def score(item, bins):
    # Calculate base score using remaining capacity
    base_score = bins
    
    # Apply mathematical transformations
    # (your creative logic here)
    
    # Normalize and return
    return scores
```

Provide your complete implementation."""

# ============================================================================
# 诊断函数
# ============================================================================

def check_syntax(code):
    """检查代码语法"""
    print("\n" + "="*80)
    print("步骤 1: 语法检查")
    print("="*80)
    
    if not code or not isinstance(code, str):
        print("✗ FAIL: Code is empty or not a string")
        return False
    
    if "def" not in code:
        print("✗ FAIL: No function definition found")
        return False
    
    try:
        ast.parse(code)
        print("✓ PASS: Code has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ FAIL: Syntax Error")
        print(f"  Line {e.lineno}: {e.msg}")
        print(f"  Text: {e.text}")
        return False
    except Exception as e:
        print(f"✗ FAIL: AST parsing error: {e}")
        return False


def check_function_signature(code):
    """检查函数签名"""
    print("\n" + "="*80)
    print("步骤 2: 函数签名检查")
    print("="*80)
    
    # 检查函数名
    if "def score(" not in code:
        print("✗ FAIL: Function not named 'score'")
        # 尝试找到实际的函数名
        import re
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if func_match:
            print(f"  Found function name: '{func_match.group(1)}'")
        return False
    
    print("✓ PASS: Function named 'score'")
    
    # 检查参数
    import re
    param_match = re.search(r'def score\((.*?)\)', code)
    if param_match:
        params = param_match.group(1)
        print(f"  Parameters: {params}")
        if 'item' in params and 'bins' in params:
            print("✓ PASS: Has required parameters 'item' and 'bins'")
            return True
        else:
            print("✗ FAIL: Missing required parameters")
            return False
    
    print("✗ FAIL: Could not parse parameters")
    return False


def check_imports(code):
    """检查导入"""
    print("\n" + "="*80)
    print("步骤 3: 导入检查")
    print("="*80)
    
    if "import numpy" in code or "import np" in code:
        print("✓ PASS: NumPy imported")
    else:
        print("⚠ WARNING: No NumPy import found (might still work if np is available)")
    
    # 检查是否有非法导入
    illegal_imports = ['random', 'torch', 'tensorflow', 'sklearn']
    found_illegal = []
    for lib in illegal_imports:
        if f"import {lib}" in code:
            found_illegal.append(lib)
    
    if found_illegal:
        print(f"✗ FAIL: Illegal imports found: {found_illegal}")
        return False
    else:
        print("✓ PASS: No illegal imports")
        return True


def test_execution(code):
    """测试代码执行"""
    print("\n" + "="*80)
    print("步骤 4: 代码执行测试")
    print("="*80)
    
    try:
        # 创建模块并执行代码
        heuristic_module = types.ModuleType("heuristic_module")
        exec(code, heuristic_module.__dict__)
        
        # 检查函数是否存在
        if not hasattr(heuristic_module, 'score'):
            print("✗ FAIL: 'score' function not found in module")
            print(f"  Available: {[x for x in dir(heuristic_module) if not x.startswith('_')]}")
            return False, None
        
        print("✓ PASS: Module created and function exists")
        return True, heuristic_module
        
    except Exception as e:
        print(f"✗ FAIL: Execution error during module creation")
        print(f"  Error: {type(e).__name__}: {e}")
        return False, None


def test_function_call(heuristic_module):
    """测试函数调用"""
    print("\n" + "="*80)
    print("步骤 5: 函数调用测试（使用真实数据范围）")
    print("="*80)
    
    # 测试用例 - 使用真实的整数值范围（基于 get_instance.py）
    test_cases = [
        {"name": "Typical case", "item": 48, "bins": np.array([52, 31, 69, 48])},
        {"name": "Large item", "item": 72, "bins": np.array([80, 75, 90])},
        {"name": "Small item", "item": 15, "bins": np.array([20, 35, 18, 22])},
        {"name": "Single bin", "item": 30, "bins": np.array([50])},
        {"name": "Many bins", "item": 40, "bins": np.array([50, 60, 45, 55, 70, 42])},
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {test['name']}")
        print(f"    item={test['item']}, bins={test['bins']}")
        
        try:
            result = heuristic_module.score(test['item'], test['bins'])
            
            # 检查返回值
            if not isinstance(result, np.ndarray):
                print(f"    ✗ FAIL: Result not numpy array (got {type(result)})")
                return False
            
            if len(result) != len(test['bins']):
                print(f"    ✗ FAIL: Result length {len(result)} != bins length {len(test['bins'])}")
                return False
            
            if not np.all(np.isfinite(result)):
                print(f"    ✗ FAIL: Result contains non-finite values: {result}")
                return False
            
            print(f"    ✓ PASS: scores={result}")
            
        except Exception as e:
            print(f"    ✗ FAIL: Runtime error")
            print(f"      {type(e).__name__}: {e}")
            return False
    
    print("\n✓ PASS: All function call tests passed")
    return True


def test_evaluation_with_problem(code):
    """使用实际问题实例进行完整评估"""
    print("\n" + "="*80)
    print("步骤 6: 完整问题评估测试")
    print("="*80)
    
    try:
        # 创建问题实例
        problem = BPONLINE()
        
        # 评估代码（带超时）
        print(f"  Running evaluation (timeout={TIMEOUT}s)...")
        
        def run_evaluation():
            return problem.evaluate(code)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_evaluation)
            try:
                start_time = time.time()
                fitness = future.result(timeout=TIMEOUT)
                elapsed = time.time() - start_time
                
                if fitness is None:
                    print(f"  ✗ FAIL: Evaluation returned None")
                    return False
                
                if not np.isfinite(fitness):
                    print(f"  ✗ FAIL: Fitness is not finite: {fitness}")
                    return False
                
                print(f"  ✓ PASS: Evaluation successful")
                print(f"    Fitness: {fitness:.6f}")
                print(f"    Time: {elapsed:.2f}s")
                return True
                
            except concurrent.futures.TimeoutError:
                print(f"  ✗ FAIL: Evaluation timeout ({TIMEOUT}s)")
                future.cancel()
                return False
            except Exception as e:
                print(f"  ✗ FAIL: Evaluation error")
                print(f"    {type(e).__name__}: {e}")
                return False
    
    except Exception as e:
        print(f"✗ FAIL: Could not create problem instance")
        print(f"  {type(e).__name__}: {e}")
        return False


def generate_code_from_llm(p_final):
    """使用 LLM 生成代码"""
    print("\n" + "="*80)
    print("步骤 0: 使用 LLM 生成代码")
    print("="*80)
    
    try:
        interface_llm = InterfaceLLM(
            LLM_CONFIG['api_endpoint'],
            LLM_CONFIG['api_key'],
            LLM_CONFIG['model'],
            LLM_CONFIG['use_local'],
            LLM_CONFIG['local_url'],
            debug_mode=False
        )
        print(f"✓ LLM Interface initialized: {LLM_CONFIG['model']}")
        
        print("\n  Calling LLM (temperature=0.8)...")
        response = interface_llm.get_response(p_final, temperature=0.8)
        
        if not response:
            print("✗ FAIL: LLM returned empty response")
            return None, None
        
        print(f"  ✓ Got response ({len(response)} chars)")
        
        # 提取代码
        import re
        
        # 提取思想
        thought_match = re.search(r'\{(.*?)\}', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "No thought found"
        
        # 提取代码
        code = ""
        code_match_md = re.search(r'```python\s*([\s\S]*?)```', response, re.DOTALL)
        if code_match_md:
            code = code_match_md.group(1).strip()
        else:
            code_match = re.search(r'(import.*?)?def.*?return', response, re.DOTALL)
            if code_match:
                code = code_match.group(0).strip()
        
        if code:
            print(f"  ✓ Code extracted ({len(code)} chars)")
            print(f"\n  Thought: {thought[:100]}...")
            return code, thought
        else:
            print("  ✗ FAIL: Could not extract code from response")
            print(f"\n  Response preview:\n{response[:500]}")
            return None, None
        
    except Exception as e:
        print(f"✗ FAIL: Error calling LLM: {e}")
        return None, None


# ============================================================================
# 主诊断流程
# ============================================================================

def diagnose(use_llm=True, custom_code=None, custom_p_final=None):
    """
    运行完整诊断
    
    Args:
        use_llm: 是否使用 LLM 生成代码（否则需要提供 custom_code）
        custom_code: 自定义代码字符串（用于测试特定代码）
        custom_p_final: 自定义 P_final（如果不使用默认）
    """
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "代码生成失败诊断脚本" + " "*37 + "║")
    print("╚" + "═"*78 + "╝")
    
    # 获取代码
    if use_llm:
        p_final = custom_p_final or SAMPLE_P_FINAL
        print(f"\nUsing P_final ({len(p_final)} chars):")
        print("-" * 80)
        print(p_final[:300] + "..." if len(p_final) > 300 else p_final)
        print("-" * 80)
        
        code, thought = generate_code_from_llm(p_final)
        if not code:
            print("\n" + "="*80)
            print("❌ 诊断终止: LLM 未能生成代码")
            print("="*80)
            return
    else:
        if not custom_code:
            print("❌ 错误: 必须提供 custom_code 或启用 use_llm")
            return
        code = custom_code
        thought = "Custom code provided"
    
    # 显示生成的代码
    print("\n" + "="*80)
    print("生成的代码:")
    print("="*80)
    print(code)
    print("="*80)
    
    # 运行诊断步骤
    results = {}
    
    # 步骤 1: 语法检查
    results['syntax'] = check_syntax(code)
    if not results['syntax']:
        print("\n" + "="*80)
        print("❌ 诊断结束: 语法错误")
        print("="*80)
        return
    
    # 步骤 2: 函数签名
    results['signature'] = check_function_signature(code)
    
    # 步骤 3: 导入检查
    results['imports'] = check_imports(code)
    
    # 步骤 4: 执行测试
    results['execution'], module = test_execution(code)
    if not results['execution']:
        print("\n" + "="*80)
        print("❌ 诊断结束: 模块创建失败")
        print("="*80)
        return
    
    # 步骤 5: 函数调用测试
    results['function_call'] = test_function_call(module)
    
    # 步骤 6: 完整评估
    results['evaluation'] = test_evaluation_with_problem(code)
    
    # 总结
    print("\n\n" + "="*80)
    print("诊断总结")
    print("="*80)
    
    steps = [
        ('语法检查', 'syntax'),
        ('函数签名', 'signature'),
        ('导入检查', 'imports'),
        ('代码执行', 'execution'),
        ('函数调用', 'function_call'),
        ('完整评估', 'evaluation'),
    ]
    
    for name, key in steps:
        status = "✓ PASS" if results.get(key) else "✗ FAIL"
        print(f"  {name}: {status}")
    
    if all(results.get(key) for _, key in steps):
        print("\n🎉 所有测试通过！代码应该可以正常使用。")
    else:
        print("\n❌ 发现问题！请根据上面的详细输出定位失败原因。")
    
    print("="*80)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='诊断代码生成失败问题')
    parser.add_argument('--no-llm', action='store_true', help='不使用 LLM（需要提供代码）')
    parser.add_argument('--code-file', type=str, help='从文件读取代码进行测试')
    
    args = parser.parse_args()
    
    if args.code_file:
        with open(args.code_file, 'r') as f:
            custom_code = f.read()
        diagnose(use_llm=False, custom_code=custom_code)
    else:
        diagnose(use_llm=not args.no_llm)


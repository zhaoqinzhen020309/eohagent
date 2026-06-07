# eoh_evolution.py - SEC V6 Evolution Class
import re
import time
import json
import os
import ast
import numpy as np
from ...llm.interface_LLM import InterfaceLLM


class Evolution():

    def __init__(self,
                 # LLM 1 Configs
                 api_endpoint_1, api_key_1, model_LLM_1, llm_use_local_1, llm_local_url_1,
                 # LLM 2 Configs
                 api_endpoint_2, api_key_2, model_LLM_2, llm_use_local_2, llm_local_url_2,
                 # Other
                 debug_mode, prompts, problem_instance=None,
                 **kwargs):

        # --- 存储问题实例（用于问题类型识别）---
        self.problem_instance = problem_instance
        
        # --- 存储 Prompt 模板信息 (来自 problem 的 prompts 对象) ---
        try:
            self.prompt_task = prompts.get_task()
            self.prompt_func_name = prompts.get_func_name()
            self.prompt_func_inputs = prompts.get_func_inputs()  # 列表
            self.prompt_func_outputs = prompts.get_func_outputs()  # 列表
            self.prompt_inout_inf = prompts.get_inout_inf()
            self.prompt_other_inf = prompts.get_other_inf()

            # 格式化输入输出 (用于 i1 Prompt)
            if len(self.prompt_func_inputs) > 1:
                self.joined_inputs = ", ".join(f"'{s}'" for s in self.prompt_func_inputs)
            else:
                self.joined_inputs = f"'{self.prompt_func_inputs[0]}'"

            if len(self.prompt_func_outputs) > 1:
                self.joined_outputs = ", ".join(f"'{s}'" for s in self.prompt_func_outputs)
            else:
                self.joined_outputs = f"'{self.prompt_func_outputs[0]}'"

            # 构建函数签名字符串 (用于 Prompt)
            input_args = ", ".join(self.prompt_func_inputs)
            self.func_signature_str = f"def {self.prompt_func_name}({input_args}):"

        except AttributeError as e:
            print(f"[ERROR] Failed to get prompt details from 'prompts' object: {e}.")
            raise ValueError("Problem's 'prompts' object is missing required methods.") from e

        self.debug_mode = debug_mode
        self.llm_call_count = 0  # 统一计数器
        
        # === LLM交互记录 ===
        self.llm_interaction_history = []  # 记录所有LLM交互
        self.current_generation = 0  # 当前代数

        # === 实例化两个独立的 InterfaceLLM ===
        print("Initializing LLM Interfaces...")
        try:
            self.interface_llm1 = InterfaceLLM(
                api_endpoint_1, api_key_1, model_LLM_1, llm_use_local_1, llm_local_url_1, debug_mode
            )
            print(f"  LLM 1 (Engineer/Worker) interface ready: Model='{model_LLM_1}', Local={llm_use_local_1}")

            self.interface_llm2 = InterfaceLLM(
                api_endpoint_2, api_key_2, model_LLM_2, llm_use_local_2, llm_local_url_2, debug_mode
            )
            print(f"  LLM 2 (Strategist) interface ready: Model='{model_LLM_2}', Local={llm_use_local_2}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize LLM interfaces: {e}")
            raise

    def _record_llm_interaction(self, llm_type, call_type, prompt_input, response_output, parsed_result=None, success=True, error_msg=None):
        """
        记录LLM交互的详细信息
        
        Args:
            llm_type: "LLM_1" 或 "LLM_2"
            call_type: 调用类型，如 "i1", "strategist", "engineer", "worker"
            prompt_input: 发送给LLM的完整prompt
            response_output: LLM返回的原始响应
            parsed_result: 解析后的结果（dict或list）
            success: 是否成功
            error_msg: 如果失败，错误信息
        """
        interaction_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generation": self.current_generation,
            "call_id": self.llm_call_count,
            "llm_type": llm_type,
            "call_type": call_type,
            "prompt_length": len(prompt_input),
            "prompt_input": prompt_input,
            "response_length": len(response_output) if response_output else 0,
            "response_output": response_output if response_output else "",
            "parsed_result": parsed_result,
            "success": success,
            "error_msg": error_msg
        }
        self.llm_interaction_history.append(interaction_record)
    
    def save_llm_interactions(self, output_dir, generation=None):
        """
        保存LLM交互历史到JSON文件
        
        Args:
            output_dir: 输出目录路径
            generation: 当前代数（用于文件名），如果为None则保存所有历史
        """
        try:
            # 创建LLM交互记录目录
            llm_log_dir = os.path.join(output_dir, "results", "llm_interactions")
            os.makedirs(llm_log_dir, exist_ok=True)
            
            if generation is not None:
                # 保存单代的交互记录
                gen_interactions = [rec for rec in self.llm_interaction_history if rec['generation'] == generation]
                if gen_interactions:
                    filename = os.path.join(llm_log_dir, f"llm_interactions_gen_{generation}.json")
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(gen_interactions, f, indent=2, ensure_ascii=False)
                    print(f"    [LLM Log] Saved {len(gen_interactions)} LLM interactions to {filename}")
            else:
                # 保存所有历史
                filename = os.path.join(llm_log_dir, "llm_interactions_complete.json")
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.llm_interaction_history, f, indent=2, ensure_ascii=False)
                print(f"[LLM Log] Saved {len(self.llm_interaction_history)} total LLM interactions to {filename}")
                
        except Exception as e:
            print(f"[WARNING] Failed to save LLM interaction history: {e}")

    def _robust_json_parser(self, response_text):
        """超强鲁棒 JSON 解析器 - 使用正则表达式直接提取字段（Strategy 5: 100% 成功率）"""
        import string
        import unicodedata
        
        def clean_control_chars(s):
            """移除字符串中的非法控制字符"""
            # 移除所有控制字符（除了换行、制表符等）
            cleaned = ''.join(char for char in s if unicodedata.category(char)[0] != 'C' or char in ('\n', '\t', '\r'))
            # 移除零宽字符和BOM
            cleaned = cleaned.replace('\u200b', '').replace('\ufeff', '')
            return cleaned
        
        # 1. 清理控制字符
        cleaned_response = clean_control_chars(response_text)
        
        # 2. 首先尝试标准 JSON 解析（快速路径）
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            pass
        
        # 3. 尝试提取 markdown 块后解析
        # 注意：P_final 中可能包含 ```python，所以要匹配 ``` 后面是换行或结束
        json_match = re.search(r"```json\s*([\s\S]*?)```\s*$", cleaned_response, re.DOTALL | re.MULTILINE)
        if not json_match:
            # 如果没找到，尝试更宽松的模式（匹配到文本末尾的 ```）
            json_match = re.search(r"```json\s*([\s\S]*?)```", cleaned_response, re.DOTALL)
        json_str = json_match.group(1).strip() if json_match else cleaned_response
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            if self.debug_mode:
                print("  [JSON Parser] Standard parsing failed, using regex extraction fallback...")
        
        # 4. 终极回退：使用正则表达式直接提取字段（绕过JSON解析）
        # 改进策略：从后向前提取 P_final，处理复杂内容
        try:
            result = {}
            
            # 提取 critique_explanation
            # 使用 Improved V1 宽松模式（测试验证 100% 成功率）
            critique_match = re.search(
                r'"critique_explanation"\s*:\s*"([\s\S]*?)"(?=\s*,?\s*"|\s*\})', 
                json_str
            )
            if critique_match:
                # 清理提取的内容：移除多余换行，保留空格
                result['critique_explanation'] = critique_match.group(1).replace('\n', ' ').strip()
            
            # 提取 rewrite_explanation  
            # 使用 Improved V1 宽松模式（测试验证 100% 成功率）
            rewrite_match = re.search(
                r'"rewrite_explanation"\s*:\s*"([\s\S]*?)"(?=\s*,?\s*"|\s*\})', 
                json_str
            )
            if rewrite_match:
                result['rewrite_explanation'] = rewrite_match.group(1).replace('\n', ' ').strip()
            
            # 提取 P_final - 终极策略：P_final 是最后一个字段，从 "P_final": " 读到结尾
            # 由于 P_final 可能包含任意字符（引号、反引号、大括号等），我们采用简单策略：
            # 找到 "P_final": " 的开始位置，然后取到最后，再清理末尾的 "}
            pfinal_start = json_str.find('"P_final"')
            if pfinal_start != -1:
                # 找到 "P_final": 后的第一个引号（值的开始）
                colon_pos = json_str.find(':', pfinal_start)
                if colon_pos != -1:
                    value_start = json_str.find('"', colon_pos) + 1
                    
                    # 取从 value_start 到字符串末尾的所有内容
                    p_final_with_tail = json_str[value_start:]
                    
                    # 清理末尾：移除最后的 "}（可能有空白）
                    # 从后向前移除 }、"、空白等字符，直到遇到内容
                    p_final_raw = p_final_with_tail.rstrip()
                    
                    # 移除末尾的 }（如果有）
                    if p_final_raw.endswith('}'):
                        p_final_raw = p_final_raw[:-1].rstrip()
                    
                    # 移除末尾的 "（P_final 值的结束引号）
                    if p_final_raw.endswith('"'):
                        p_final_raw = p_final_raw[:-1]
                    
                    # 清理连续换行
                    p_final_cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', p_final_raw)
                    result['P_final'] = p_final_cleaned.strip()
            
            # 验证是否提取到所有必需字段
            required_fields = ['critique_explanation', 'rewrite_explanation', 'P_final']
            missing_fields = [f for f in required_fields if f not in result or not result[f]]
            
            if missing_fields:
                # 输出更多调试信息
                if self.debug_mode:
                    print(f"  [JSON Parser] Extracted fields: {list(result.keys())}")
                    print(f"  [JSON Parser] Missing fields: {missing_fields}")
                    if 'P_final' in missing_fields:
                        print(f"  [JSON Parser] JSON string length: {len(json_str)}")
                        print(f"  [JSON Parser] Searching for P_final in: {json_str[-200:]}")
                raise ValueError(f"Regex extraction failed: missing or empty fields: {missing_fields}")
            
            if self.debug_mode:
                print(f"  [JSON Parser] ✓ Successfully extracted all fields using regex fallback")
                print(f"  [JSON Parser] Extracted field lengths:")
                print(f"    - critique_explanation: {len(result.get('critique_explanation', ''))} chars")
                print(f"    - rewrite_explanation: {len(result.get('rewrite_explanation', ''))} chars")
                print(f"    - P_final: {len(result.get('P_final', ''))} chars")
                if 'P_final' in result:
                    print(f"  [JSON Parser] P_final content:")
                    print(f"    First 200 chars: {result['P_final'][:200]}...")
                    print(f"    Last 200 chars: ...{result['P_final'][-200:]}")
            
            return result
            
        except Exception as e:
            # 如果连正则提取都失败了，输出详细错误信息
            print(f"[ERROR] All JSON parsing methods failed, including regex extraction: {e}")
            print(f"Response preview (first 500 chars):\n{cleaned_response[:500]}")
            print(f"Response end (last 200 chars):\n{cleaned_response[-200:]}")
            raise ValueError(f"Could not parse LLM response after all attempts: {e}") from e

    def _clean_extracted_code(self, code):
        """
        清理提取的代码：只保留 import 和主函数，移除测试代码
        
        Args:
            code: 原始提取的代码字符串
            
        Returns:
            清理后的代码字符串
        """
        lines = code.split('\n')
        cleaned_lines = []
        in_target_function = False
        in_other_function = False
        target_func_indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            # 保留 import 语句
            if stripped.startswith('import ') or stripped.startswith('from '):
                cleaned_lines.append(line)
                continue
            
            # 检测目标函数（score 或其他 prompt_func_name）
            if stripped.startswith(f'def {self.prompt_func_name}('):
                in_target_function = True
                in_other_function = False
                target_func_indent = len(line) - len(line.lstrip())
                cleaned_lines.append(line)
                continue
            
            # 检测其他函数定义（如 check_solution）
            if stripped.startswith('def ') and not in_target_function:
                in_other_function = True
                continue
            
            # 如果在目标函数中
            if in_target_function:
                # 检查是否还在函数内部（通过缩进判断）
                if line and not line[0].isspace():
                    # 遇到非缩进行，函数结束
                    in_target_function = False
                else:
                    cleaned_lines.append(line)
                    continue
            
            # 跳过其他函数的内容
            if in_other_function:
                if line and not line[0].isspace():
                    # 其他函数结束
                    in_other_function = False
                else:
                    continue
            
            # 跳过函数调用（如 check_solution()）
            if stripped and not stripped.startswith('#'):
                # 检测是否是函数调用（包含括号且不是 def）
                if '(' in stripped and ')' in stripped and not stripped.startswith('def '):
                    continue
        
        # 重新组合代码
        cleaned_code = '\n'.join(cleaned_lines)
        
        # 移除末尾多余的空行
        cleaned_code = cleaned_code.rstrip() + '\n'
        
        return cleaned_code
    
    def _get_thought_and_code(self, prompt_content, llm_interface, temperature=None, call_type="unknown"):
        """
        从指定的 LLM 接口获取响应，并解析思想和代码。
        返回 [code, algorithm] 或 ["", ""].
        """
        max_retries = 3
        last_exception = None
        final_response = ""
        final_code = ""
        final_algorithm = ""
        
        for attempt in range(max_retries):
            try:
                if self.debug_mode:
                    print(f"\n[LLM Call attempt {attempt+1}/{max_retries}] To: {llm_interface.model_LLM}")
                    if temperature is not None:
                        print(f"  Temperature: {temperature}")

                # === 调用 LLM ===
                response = llm_interface.get_response(prompt_content, temperature=temperature)
                self.llm_call_count += 1
                final_response = response  # 保存最后一次响应用于记录

                if not response:
                    print(f"[ERROR attempt {attempt+1}] Empty LLM response.")
                    if attempt < max_retries - 1:
                        time.sleep(1 + attempt)
                    continue

                if self.debug_mode:
                    print(f"  LLM Response (length: {len(response)} chars):")
                    print("  " + "-" * 76)
                    print(response)
                    print("  " + "-" * 76)

                # --- 解析 ---
                response_text = response.strip()

                # 1. 提取思想: { ... }
                algorithm_match = re.search(r"\{(.*?)\}", response_text, re.DOTALL)
                algorithm = algorithm_match.group(1).strip() if algorithm_match else "No thought found."

                # 2. 提取代码块
                code = ""
                # 优先匹配 markdown 块
                code_match_md = re.search(r"```python\s*([\s\S]*?)```", response_text, re.DOTALL)
                if code_match_md:
                    code = code_match_md.group(1).strip()
                    if self.debug_mode:
                        print("  Code extracted using markdown pattern.")
                else:
                    # 回退到寻找 import ... def 或 def ... return
                    code_match = re.search(r"(import.*?)?def.*?return", response_text, re.DOTALL)
                    if code_match:
                        code = code_match.group(0).strip()
                        if self.debug_mode:
                            print("  Code extracted using def pattern.")
                
                # 3. 清理代码：只保留主函数，移除额外的测试代码
                if code:
                    code = self._clean_extracted_code(code)

                # 4. 基本验证
                if code and "def" in code and self.prompt_func_name in code:
                    # 确保代码包含完整的 return 语句
                    if not code.strip().endswith("return " + self.prompt_func_outputs[0]):
                        # 尝试添加返回语句
                        code_lines = code.split('\n')
                        # 检查最后一行是否是 return
                        if 'return' not in code_lines[-1]:
                            # 添加返回语句
                            code = code + "\n    return " + self.prompt_func_outputs[0]
                    
                    final_code = code
                    final_algorithm = algorithm
                    
                    # 记录成功的交互
                    llm_type = "LLM_1_Worker" if llm_interface == self.interface_llm1 else "LLM_2"
                    self._record_llm_interaction(
                        llm_type=llm_type,
                        call_type=call_type,
                        prompt_input=prompt_content,
                        response_output=response,
                        parsed_result={"algorithm": algorithm, "code": code},
                        success=True
                    )
                    
                    return [code, algorithm]
                else:
                    print(f"[ERROR attempt {attempt+1}] Could not extract valid code containing 'def {self.prompt_func_name}'.")
                    if self.debug_mode:
                        print(f"  Full response was:\n{response_text[:500]}...")

                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)

            except Exception as e:
                last_exception = e
                print(f"[ERROR attempt {attempt+1}] LLM call or parsing failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt)

        # 所有重试都失败，记录失败的交互
        llm_type = "LLM_1_Worker" if llm_interface == self.interface_llm1 else "LLM_2"
        self._record_llm_interaction(
            llm_type=llm_type,
            call_type=call_type,
            prompt_input=prompt_content,
            response_output=final_response,
            success=False,
            error_msg=f"Max retries reached. Last error: {last_exception}"
        )
        
        print(f"[ERROR] Max retries ({max_retries}) reached for LLM call/parsing.")
        if last_exception:
            print(f"  Last error: {last_exception}")
        return ["", ""]

    def i1(self):
        """
        初始化方法 (使用 LLM 1 Worker)。
        返回 [code, algorithm]。
        """
        # 构建i1 prompt - 与原始EoH风格一致
        prompt_content = f"""I need help designing a new heuristic that scores a set of bins to assign an item. In each step, the item will be assigned to the bin with the maximum score. If the rest capacity of a bin equals the maximum capacity, it will not be used. The final goal is to minimize the number of used bins.

Please design a new heuristic.

Firstly, describe your new heuristic and main steps in one sentence.

Next, implement it in Python as a function named '{self.prompt_func_name}'. This function should accept {len(self.prompt_func_inputs)} inputs: {', '.join(self.prompt_func_inputs)}. The function should return {len(self.prompt_func_outputs)} output(s): {', '.join(self.prompt_func_outputs)}. {self.prompt_inout_inf}

Note that 'item' is of type int, 'bins' is a Numpy array that includes integer values, and 'scores' should be a Numpy array. {self.prompt_other_inf} Your implementation must be a computationally efficient, vectorized NumPy expression. Your logic can only depend on 'item' and 'bins'. Do not give additional explanations.
"""

        if self.debug_mode:
            print(f"\n >>> check prompt for creating algorithm using [ i1 ] (length: {len(prompt_content)} chars):")
            print("=" * 80)
            print(prompt_content)
            print("=" * 80)

        # 调用 LLM 1 Worker
        code_all, algorithm = self._get_thought_and_code(prompt_content, self.interface_llm1, temperature=0, call_type="i1_initialization")

        if self.debug_mode and code_all:
            print("\n >>> check designed algorithm [i1]: \n", algorithm)
            print("\n >>> check designed code [i1]: \n", code_all)
        elif not code_all:
            print("[ERROR] i1 failed to generate valid code.")

        return [code_all, algorithm]

    # === SEC V6 核心方法 ===

    def call_strategist(self, meta_prompt):
        """调用 LLM 2 (策略师) 并解析其 portfolio JSON 输出。"""
        try:
            # 策略生成通常需要更高的创造性/温度
            response = self.interface_llm2.get_response(meta_prompt, temperature=1.0)
            self.llm_call_count += 1
            if self.debug_mode:
                print(f"\n[Strategist Response (raw, length: {len(response)} chars)]:")
                print("=" * 80)
                print(response)  # 完整输出
                print("=" * 80)
                print()

            # 使用鲁棒的 JSON 解析器
            portfolio_data = self._robust_json_parser(response)

            # 验证返回结构
            if 'portfolio' not in portfolio_data or not isinstance(portfolio_data['portfolio'], list):
                raise ValueError("Missing or invalid 'portfolio' key in Strategist response.")

            # 记录交互
            self._record_llm_interaction(
                llm_type="LLM_2_Strategist",
                call_type="strategist_portfolio_generation",
                prompt_input=meta_prompt,
                response_output=response,
                parsed_result=portfolio_data,
                success=True
            )

            return portfolio_data

        except Exception as e:
            print(f"[ERROR] Call to Strategist (LLM 2) or JSON parsing failed: {e}")
            # 记录失败的交互
            self._record_llm_interaction(
                llm_type="LLM_2_Strategist",
                call_type="strategist_portfolio_generation",
                prompt_input=meta_prompt,
                response_output="",
                success=False,
                error_msg=str(e)
            )
            return {"analysis": f"Error: {e}", "portfolio": []}


    def execute_worker_prompt(self, p_draft_from_strategist, selected_parents):
        """
        (重构后) 构建并执行 Worker LLM Prompt（基于 EoH 风格）。
        此函数接收来自 Strategist 的 P_draft，将其注入到包含父代信息和问题约束的 Worker Prompt 中。
        根据问题类型自动选择对应的Worker Prompt模板。
        
        Args:
            p_draft_from_strategist: 来自LLM2 Strategist的操作算子指令
            selected_parents: 父代样本列表
            
        Returns:
            [code, algorithm]: 生成的代码和算法描述
        """
        
        # 1. 构建父代文本
        num_parents = len(selected_parents)
        parent_text = ""
        
        if selected_parents and num_parents > 0:
            parent_text = f"I have {num_parents} existing heuristics with their codes as follows:\n"
            for i, parent in enumerate(selected_parents):
                parent_text += f"No.{i+1} Heuristic description: {parent.get('algorithm', 'N/A')}\n"
                parent_text += f"Code:\n```python\n{parent.get('code', '# N/A')}\n```\n"
        else:
            parent_text = "I have 0 existing heuristics.\n"

        # 2. 清理 P_draft (移除可能的"Firstly,"前缀以避免重复)
        p_draft_content = p_draft_from_strategist.strip()
        if p_draft_content.lower().startswith('firstly,'):
            p_draft_content = p_draft_content[len('firstly,'):].strip()
        
        # 3. 识别问题类型并构建对应的Worker Prompt
        problem_type = self._identify_problem_type()
        
        if problem_type == "TSP":
            complete_prompt = self._build_tsp_worker_prompt(parent_text, p_draft_content)
        elif problem_type == "FSSP":
            complete_prompt = self._build_fssp_worker_prompt(parent_text, p_draft_content)
        else:  # Default: Bin Packing
            complete_prompt = self._build_binpacking_worker_prompt(parent_text, p_draft_content)
        
        # 4. 调用 Worker LLM
        temperature = 0
        return self._get_thought_and_code(
            complete_prompt, 
            self.interface_llm1, 
            temperature=temperature, 
            call_type="worker_execute_operator"
        )
    
    def _identify_problem_type(self):
        """识别问题类型（通过问题类名，与eoh.py保持一致）"""
        if self.problem_instance is None:
            # 回退到函数名识别
            func_name = self.prompt_func_name
            if "update_edge_distance" in func_name:
                return "TSP"
            elif "get_matrix_and_jobs" in func_name:
                return "FSSP"
            else:
                return "BinPacking"
        
        # 使用类名识别（与eoh.py中的方法一致）
        problem_class_name = self.problem_instance.__class__.__name__
        if "TSP" in problem_class_name.upper():
            return "TSP"
        elif "JSSP" in problem_class_name.upper() or "FSSP" in problem_class_name.upper():
            return "FSSP"
        else:
            return "BinPacking"
    
    def _build_binpacking_worker_prompt(self, parent_text, p_draft_content):
        """构建Bin Packing问题的Worker Prompt"""
        inout_inf = self.prompt_inout_inf
        other_inf = self.prompt_other_inf
        
        return f"""I need help designing a new heuristic that scores a set of bins to assign an item. In each step, the item will be assigned to the bin with the maximum score. If the rest capacity of a bin equals the maximum capacity, it will not be used. The final goal is to minimize the number of used bins.

{parent_text}

Firstly, {p_draft_content}

Secondly, analyze ALL parent heuristics you see above. Identify interesting techniques, structural patterns (such as valid_bins filtering, conditional logic), mathematical operations, or innovative approaches from ANY parent, regardless of their fitness. Then, based on the P_draft guidance and your analysis, describe your new heuristic in one sentence. The description must be inside a brace.

Thirdly, implement it in Python as a function named 'score'. This function should accept two inputs: 'item' and 'bins'. The function should return one output: 'scores'. {inout_inf}

Note that 'item' is of type int, 'bins' is a Numpy array that includes integer values, and 'scores' should be a Numpy array. {other_inf} Your implementation must be a computationally efficient, vectorized NumPy expression. Your logic can only depend on 'item' and 'bins'. Do not give additional explanations.
"""
    
    def _build_tsp_worker_prompt(self, parent_text, p_draft_content):
        """构建TSP问题的Worker Prompt"""
        inout_inf = self.prompt_inout_inf
        other_inf = self.prompt_other_inf
        
        return f"""Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. You should create a heuristic for me to update the edge distance matrix.

{parent_text}

Firstly, {p_draft_content}

Secondly, analyze ALL parent heuristics you see above. Identify interesting techniques, structural patterns (such as penalty mechanisms, usage-based scaling, mathematical transformations), or innovative approaches from ANY parent, regardless of their fitness. Then, based on the P_draft guidance and your analysis, describe your new heuristic in one sentence. The description must be inside a brace.

Thirdly, implement it in Python as a function named 'update edge distance'. This function should accept three inputs: 'edge distance', 'local opt tour', and 'edge n used'. The function should return one output: 'updated edge distance'. 'local opt tour' includes the local optimal tour of IDs, 'edge distance' and 'edge n used' are matrixes, 'edge n used' includes the number of each edge used during perturbation. All inputs and outputs are Numpy arrays. Do not give additional explanations.
"""
    
    def _build_fssp_worker_prompt(self, parent_text, p_draft_content):
        """构建FSSP问题的Worker Prompt"""
        inout_inf = self.prompt_inout_inf
        other_inf = self.prompt_other_inf

        return f"""I have n jobs and m machines. Help me create a new heuristic to update the execution time matrix and select the top jobs to perturb to avoid being trapped in the local optimum scheduling with the final goal of finding scheduling with minimized makespan.

{parent_text}

Firstly, {p_draft_content}

Secondly, analyze ALL parent heuristics you see above. Identify interesting techniques, structural patterns (such as criticality-based selection, time matrix scaling, job priority mechanisms), mathematical operations, or innovative approaches from ANY parent, regardless of their fitness. Then, based on the P_draft guidance and your analysis, describe your new heuristic in one sentence. The description must be inside a brace.

Thirdly, implement it in Python as a function named 'get_matrix_and_jobs'. This function should accept four inputs: 'current_sequence', 'time_matrix', 'm', 'n'. The function must end with exactly one return statement: 'return new_matrix, perturb_jobs'.

'current_sequence' represents the current sequence of jobs (list of job indices). 'm' and 'n' denote the number of machines and number of jobs, respectively. 'time_matrix' is a matrix of size n*m that contains the execution time of each job on each machine. 'new_matrix' is the updated time matrix and must be float type (use .astype(np.float64) if needed). 'perturb_jobs' is a numpy array containing job indices (integers from 0 to n-1) to be perturbed.

The matrix and job list are Numpy arrays. Your implementation must be a computationally efficient, vectorized NumPy expression. Do not give additional explanations.
"""

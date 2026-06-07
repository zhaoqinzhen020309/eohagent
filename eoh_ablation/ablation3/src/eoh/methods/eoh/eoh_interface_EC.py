# eoh_interface_EC.py - SEC V6 Interface for EC Operators
import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
import re
import concurrent.futures
import ast


class InterfaceEC():
    def __init__(self, pop_size,
                 # LLM 1 Configs
                 api_endpoint_1, api_key_1, llm_model_1, llm_use_local_1, llm_local_url_1,
                 # LLM 2 Configs
                 api_endpoint_2, api_key_2, llm_model_2, llm_use_local_2, llm_local_url_2,
                 # Other params
                 debug_mode, interface_prob, select, n_p, timeout,
                 use_numba=False,  # 保留参数但不使用
                 **kwargs):

        self.pop_size = pop_size
        self.interface_eval = interface_prob  # 评估器实例
        prompts = interface_prob.prompts  # 获取问题相关的 Prompt 模板信息

        # === 初始化修改后的 Evolution 类，传入所有 LLM 配置 ===
        self.evol = Evolution(
            # LLM 1
            api_endpoint_1, api_key_1, llm_model_1, llm_use_local_1, llm_local_url_1,
            # LLM 2
            api_endpoint_2, api_key_2, llm_model_2, llm_use_local_2, llm_local_url_2,
            # Other
            debug_mode, prompts, problem_instance=interface_prob, **kwargs
        )

        self.debug = debug_mode
        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select  # 父代选择器实例
        self.n_p = max(1, n_p)  # 并行评估进程/线程数 (至少为1)
        self.timeout = timeout  # 评估超时时间

    def check_duplicate(self, population, code):
        """检查代码是否在种群中已存在"""
        if not code:
            return True  # 无效代码视为重复
        for ind in population:
            ind_code = ind.get('code')
            if ind_code and ind_code == code:
                return True
        return False

    def evaluate_individual(self, code):
        """
        评估单个代码，包含基础语法验证和超时处理。
        返回 fitness (float or np.number) 或 None。
        """
        if not isinstance(code, str) or "def" not in code:
            if self.debug:
                print("[Evaluate] Invalid code received (not a string or no 'def').")
            return None

        # 1. 基础语法检查
        try:
            ast.parse(code)
        except SyntaxError as e:
            if self.debug:
                print(f"[Evaluate] Syntax Error in generated code: {e}")
            return None
        except Exception as e:
            if self.debug:
                print(f"[Evaluate] AST Parsing Error: {e}")
            return None

        # 2. 昂贵评估 (带超时)
        # 警告：Windows 上 ThreadPoolExecutor 无法强制终止线程
        # 如果代码有无限循环，线程会一直运行（但不会阻塞主线程）
        fitness = None
        print(f"    [Evaluate] Starting evaluation (timeout: {self.timeout}s)...")
        eval_start_time = time.time()

        executor = None
        try:
            # 创建 executor（不使用 with，以便超时后立即放弃）
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self.interface_eval.evaluate, code)

            try:
                fitness = future.result(timeout=self.timeout)
                elapsed = time.time() - eval_start_time
                print(f"    [Evaluate] ✓ Completed in {elapsed:.1f}s")
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - eval_start_time
                print(f"    [TIMEOUT] ✗ Evaluation exceeded {self.timeout}s (elapsed: {elapsed:.1f}s)")
                print(f"    [TIMEOUT] Abandoning stuck thread and continuing...")
                print(f"    [TIMEOUT] Note: Thread may still run in background (Windows limitation)")
                if self.debug:
                    print(f"    [DEBUG] Code that timed out (first 400 chars):")
                    print(f"    {code[:400]}")
                    print(f"    ...")

                # 立即放弃 executor（不等待线程终止）
                # 在 Windows 上，线程会继续在后台运行，但不会阻塞
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Python < 3.9 不支持 cancel_futures 参数
                    executor.shutdown(wait=False)
                executor = None
                return None
            except Exception as eval_e:
                elapsed = time.time() - eval_start_time
                print(f"    [Evaluate] ✗ Runtime error after {elapsed:.1f}s: {str(eval_e)[:100]}")
                if self.debug:
                    print(f"    [DEBUG] Full error: {eval_e}")
                return None
        except Exception as e:
            if self.debug:
                print(f"[Evaluate] Unexpected Error during evaluation process: {e}")
            return None
        finally:
            # 确保清理 executor
            if executor is not None:
                try:
                    # 尝试使用新参数（Python 3.9+）
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # 回退到旧参数（Python 3.8）
                    try:
                        executor.shutdown(wait=False)
                    except:
                        pass
                except:
                    pass

        # 验证评估结果类型（在 try-except-finally 之外）
        if fitness is None:
            if self.debug:
                print("[Evaluate] Evaluation function returned None.")
            return None

        if isinstance(fitness, np.ndarray):
            fitness = np.mean(fitness)
            if not np.isfinite(fitness):
                print("[Evaluate] Warning: Fitness array resulted in non-finite value.")
                return None

        if not np.isfinite(fitness):
            print("[Evaluate] Warning: Non-finite fitness value received.")
            return None

        return np.round(float(fitness), 5)

    def get_one_i1_individual(self):
        """
        生成并评估一个使用 i1 算子的初始个体。
        返回 offspring 字典或 None。
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if self.debug:
                    print(f"  i1 Attempt {attempt + 1}/{max_retries}")

                # 调用 Evolution 的 i1 方法
                code, algorithm = self.evol.i1()
                if not code:
                    if self.debug:
                        print("    i1 LLM failed to generate code.")
                    time.sleep(1)
                    continue

                # 评估
                objective = self.evaluate_individual(code)
                if objective is None:
                    if self.debug:
                        print("    i1 generated code failed evaluation (timeout/error/syntax).")
                        print("    Retrying...")
                    time.sleep(0.5)  # 短暂延迟后重试
                    continue

                # 成功
                offspring = {
                    'algorithm': algorithm,
                    'code': code,
                    'objective': objective,
                    'other_inf': {'source': 'i1'}
                }
                if self.debug:
                    print(f"    ✓ Evaluation successful: {objective}")
                return offspring

            except Exception as e:
                if self.debug:
                    print(f"[InterfaceEC] Error in get_one_i1_individual attempt {attempt + 1}: {e}")
                time.sleep(1)

        print("[ERROR] Max retries reached for i1 generation.")
        return None

    def generate_offspring_batch(self, population, selected_parents, p_draft, num_offspring):
        """
        (重构后) 使用指定的 P_draft 和父代样本并行生成和评估 num_offspring 个后代。
        返回一个列表，包含成功生成的后代字典。
        """
        if num_offspring <= 0:
            return []

        results = []
        try:
            # 使用 joblib 并行调用 _generate_one_offspring
            parallel_jobs = min(self.n_p, num_offspring)
            print(f"    Starting parallel generation for {num_offspring} offspring using {parallel_jobs} workers...")
            results = Parallel(n_jobs=parallel_jobs)(
                delayed(self._generate_one_offspring)(population, selected_parents, p_draft)
                for _ in range(num_offspring)
            )
            print(f"    Parallel generation finished.")
        except Exception as e:
            print(f"[InterfaceEC] Parallel generation failed: {e}")

        # 过滤掉失败的结果 (None)
        successful_offspring = [off for off in results if off is not None]

        return successful_offspring

    def _generate_one_offspring(self, population, selected_parents, p_draft):
        """
        (重构后) 生成、检查重复、评估单个后代。
        返回 offspring 字典或 None。
        """
        max_llm_retries = 3
        max_eval_retries = 1
        attempt = 0
        last_failure_reason = "Unknown"

        while attempt < max_llm_retries:
            attempt += 1
            code, algorithm = "", ""
            try:
                # 1. 调用 Evolution 执行 P_draft (传递父代样本)
                code, algorithm = self.evol.execute_worker_prompt(p_draft, selected_parents)
                if not code:
                    last_failure_reason = "Worker LLM failed to generate code"
                    if self.debug:
                        print(f"    [Worker] Retry {attempt}/{max_llm_retries}: {last_failure_reason}.")
                    time.sleep(1)
                    continue

                # 2. 检查代码重复
                if self.check_duplicate(population, code):
                    last_failure_reason = "Duplicate code generated"
                    if self.debug:
                        print(f"    [Worker] Retry {attempt}/{max_llm_retries}: {last_failure_reason}.")
                    time.sleep(0.5)
                    continue

                # 3. 评估
                eval_attempt = 0
                objective = None
                while eval_attempt < max_eval_retries:
                    objective = self.evaluate_individual(code)
                    if objective is not None:
                        break
                    eval_attempt += 1
                    last_failure_reason = "Code evaluation failed (syntax/runtime error or timeout)"
                    if self.debug and eval_attempt < max_eval_retries:
                        print(f"    [Worker] Retry {eval_attempt}/{max_eval_retries}: {last_failure_reason}.")
                        time.sleep(0.5)

                if objective is None:
                    if self.debug:
                        print(f"    [Worker] Retry {attempt}/{max_llm_retries}: {last_failure_reason}.")
                        print(f"    [Worker] Generated code (first 200 chars):\n{code[:200]}...")
                    continue

                # 成功
                offspring = {
                    'algorithm': algorithm,
                    'code': code,
                    'objective': objective,
                    'other_inf': {'source': 'dynamic_op'}
                }
                return offspring

            except Exception as e:
                last_failure_reason = f"Exception: {str(e)}"
                if self.debug:
                    print(f"[InterfaceEC] Error in _generate_one_offspring attempt {attempt}: {e}")
                time.sleep(1)

        # 最后失败时输出更详细的诊断信息
        print(f"    [Worker] FAILED after {max_llm_retries} retries. Last reason: {last_failure_reason}")
        return None

    # === 保留用于种子初始化的方法 ===
    def population_generation_seed(self, seeds, n_p):
        """种子初始化（保持原始EoH不变）"""
        population = []

        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initialization finished! Get " + str(len(seeds)) + " seed algorithms")
        return population

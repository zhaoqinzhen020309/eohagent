# runEoH.py - SEC V6 Framework Configuration
import sys
import os

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# === ABLATION 3: 使用 eoh_ablation/ablation3 代码 ===
eoh_src_path = os.path.join(current_dir, '..', '..', 'eoh_ablation', 'ablation3', 'src')
sys.path.insert(0, eoh_src_path)
print(f"[ABLATION 3] Loading code from: {eoh_src_path}")

# --- 导入必要的类 ---
from eoh import eoh
from eoh.utils.getParas import Paras
from prob import BPONLINE

# 验证导入路径
print(f"EOH module loaded from: {eoh.__file__}")

# --- 参数初始化 ---
paras = Paras()
problem_local = BPONLINE()

# --- 设置参数 (SEC V6 配置) ---
paras.set_paras(
    method="eoh",
    problem=problem_local,



    # === LLM 1 (Engineer / Worker) Configuration ===
    llm_use_local_1=True,
    llm_local_url_1="http://localhost:11434/api/generate",
    llm_model_1="qwen2.5-coder:32b",  # 用于执行和重写的模型

    # === LLM 2 (Strategist) Configuration ===
    llm_use_local_2=False,  # 使用远程 API
    llm_api_endpoint_2="api.deepseek.com",
    llm_api_key_2="YOUR_API_KEY",
    llm_model_2="deepseek-chat",  # 使用 DeepSeek 作为策略生成模型

    # === EC & Experiment Parameters ===
    ec_pop_size=20,  # 种群大小
    ec_n_pop=20,  # 进化代数
    ec_selection_num=5,  # 父代样本数 (p=5)，用于 LLM 2 输入
    ec_portfolio_size=2,  # 算子组合大小 (r=2)

    exp_n_proc=4,  # 并行评估进程数
    exp_debug_mode=True,  # 调试模式开关
    eva_timeout=60,  # 单个评估的超时时间 (秒)

    # --- 继续运行参数 ---
    exp_use_seed=False,
    exp_seed_path="",
    # exp_use_continue=False,
    # exp_continue_path="",
    # exp_continue_id=0,
    exp_use_continue=True,  # 启用加载功能
    exp_continue_path="../user_bp_online/results/pops/population_generation_0.json",  # 原始EoH种群路径
    exp_continue_id=0,  # 从第0代开始（相当于用原始种群作为初始种群）
    exp_output_path="./output"  # 结果输出路径
)

# --- 初始化 EVOL 类 ---
try:
    evolution = eoh.EVOL(paras)
except AttributeError as e:
    print(f"Error initializing EVOL class. Does 'paras' object have all required attributes? Details: {e}")
    print("Please ensure Paras class in eoh/utils/getParas.py is updated.")
    sys.exit(1)
except ImportError as e:
    print(f"Error importing required modules. Check paths and dependencies. Details: {e}")
    sys.exit(1)

# --- 运行进化 ---
try:
    evolution.run()
except Exception as e:
    print(f"An error occurred during evolution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Evolution finished.")
# runEoH.py - OPRO Framework Configuration for TSP-GLS
import sys
import os

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
eoh_src_path = os.path.join(current_dir, '..', '..', 'eoh_OPRO', 'src')
sys.path.insert(0, eoh_src_path)
print(f"Loading OPRO Framework from: {eoh_src_path}")

# --- 导入必要的类 ---
from eoh import eoh
from eoh.utils.getParas import Paras
from prob import TSPGLS

print(f"EOH module loaded from: {eoh.__file__}")

# --- 参数初始化 ---
paras = Paras()
problem_local = TSPGLS()

# --- 设置参数 (OPRO 双LLM配置) ---
paras.set_paras(
    method="eoh",
    problem=problem_local,

    # === LLM 1 (Engineer / Worker) Configuration ===
    llm_use_local_1=True,  # 使用本地 LLM
    llm_local_url_1="http://localhost:11434/api/generate",
    llm_model_1="qwen2.5-coder:32b",  # 本地模型用于代码生成

    # === LLM 2 (Strategist) Configuration ===
    llm_use_local_2=False,  # 使用远程 API
    llm_api_endpoint_2="api.deepseek.com",
    llm_api_key_2="YOUR_API_KEY",
    llm_model_2="deepseek-chat",  # DeepSeek 用于策略生成

    # === EC & Experiment Parameters ===
    ec_pop_size=10,  # 种群大小
    ec_n_pop=20,  # 进化代数
    ec_selection_num=5,  # 父代样本数 (p=5)
    ec_portfolio_size=2,  # 算子组合大小 (r=2)

    exp_n_proc=5,  # 并行评估进程数
    exp_debug_mode=True,  # 调试模式
    eva_timeout=500,  # 评估超时 (秒)

    exp_use_seed=False,
    exp_seed_path="",
    # exp_use_continue=False,
    # exp_continue_path="",
    # exp_continue_id=0,
    exp_use_continue=True,  # 启用加载功能
    exp_continue_path="./results/pops/population_generation_7.json",  # 从第7代种群继续
    exp_continue_id=7,  # 从第7代开始（接下来生成第8代到第20代）
    exp_output_path="./output"  # 结果输出路径
)

# --- 打印配置信息 ---
print("\n" + "="*70)
print("OPRO TSP-GLS 实验配置")
print("="*70)
print(f"Framework: OPRO (Dual-LLM Adaptive Strategy Learning)")
print(f"Problem: TSP with Guided Local Search")
print(f"")
print(f"LLM 1 (Worker):     {paras.llm_model_1} (Local)")
print(f"LLM 2 (Strategist): {paras.llm_model_2} (Remote API)")
print(f"")
print(f"Population size:    {paras.ec_pop_size}")
print(f"Generations:        {paras.ec_n_pop}")
print(f"Portfolio size (r): {paras.ec_portfolio_size}")
print(f"Strategist samples (p): {paras.ec_selection_num}")
print(f"")
print(f"Training instances: {problem_local.n_inst_eva}")
print(f"Time limit/instance: {problem_local.time_limit}s")
print(f"Max GLS iterations: {problem_local.ite_max}")
print(f"Parallel processes: {paras.exp_n_proc}")
print(f"")
if paras.exp_use_continue:
    print(f"Resume mode:        ENABLED")
    print(f"Loaded population:  {paras.exp_continue_path}")
    print(f"Continue from:      Generation {paras.exp_continue_id + 2} (Generation {paras.exp_continue_id + 1} already completed)")
    print(f"Will generate:      Generation {paras.exp_continue_id + 2} to {paras.ec_n_pop}")
else:
    print(f"Starting from:      Generation 1 (initial population)")
print(f"")
print(f"Baseline (Original EoH): Gap=0.0958%, Time=903.8min")
print(f"Expected runtime: 10-15 hours")
print("="*70)
print("Starting OPRO TSP-GLS Evolution...")
print("="*70 + "\n")

# --- 初始化 EVOL 类 ---
try:
    evolution = eoh.EVOL(paras)
except AttributeError as e:
    print(f"Error initializing EVOL: {e}")
    print("Please ensure Paras class has all required dual-LLM attributes.")
    sys.exit(1)
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- 运行进化 ---
try:
    evolution.run()
except Exception as e:
    print(f"Error during evolution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("OPRO TSP-GLS Evolution Finished!")
print("="*70)
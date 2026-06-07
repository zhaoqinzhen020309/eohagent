import sys
import os

# ===== 修正：eoh_RAG在EoH根目录下 =====
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前目录向上适当级数到 EoH，然后进入 eoh_RAG/src
eoh_rag_path = os.path.join(current_dir, '..', '..', 'eoh_RAG', 'src')
sys.path.insert(0, eoh_rag_path)

print(f"Loading RAG Enhanced version from: {eoh_rag_path}")

# 导入RAG版本
from eoh import eoh
from eoh.utils.getParas import Paras
from prob import BPONLINE

# 验证导入路径
print(f"EOH RAG module loaded from: {eoh.__file__}")

# Parameter initialization #
paras = Paras()

# Set your local problem
problem_local = BPONLINE()

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = problem_local, # Set local problem, else use default problems
                llm_api_endpoint = "api.deepseek.com", # set your LLM endpoint
                llm_api_key = "YOUR_API_KEY",   # set your key
                llm_model = "deepseek-chat",
                ec_pop_size = 5, # number of samples in each population
                ec_n_pop = 5,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False,
                eva_numba_decorator = True)

# initialization
evolution = eoh.EVOL(paras)

# run RAG enhanced evolution
print("=" * 60)
print("Starting RAG Enhanced Evolution of Heuristics...")
print("Phase 1: Algorithm Accumulation (LLM will decide when to build RAG)")
print("Phase 2: RAG System Construction (LLM-driven)")
print("Phase 3: RAG Enhanced Algorithm Generation")
print("=" * 60)
evolution.run()
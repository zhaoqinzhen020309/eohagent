import sys
import os

# 设置本地EoH路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从 examples/user_bp_online 向上两级到 EoH，然后进入 eoh/src
eoh_path = os.path.join(current_dir, '..', '..', 'eoh', 'src')
sys.path.insert(0, eoh_path)
print(f"Loading original EoH from: {eoh_path}")

# 导入原始EoH
from eoh import eoh
from eoh.utils.getParas import Paras
from prob import BPONLINE

# 验证导入路径
print(f"EOH module loaded from: {eoh.__file__}")

# Parameter initialization
paras = Paras()

# Set your local problem
problem_local = BPONLINE()

# 设置参数
paras.set_paras(
    method="eoh",
    problem=problem_local,

    # Ollama本地LLM配置
    llm_use_local=True,
    llm_local_url="http://localhost:11434/api/generate",
    llm_model="qwen2.5-coder:32b",

    # 其他参数
    ec_pop_size=20,
    ec_n_pop=20,
    ec_selection_num=5,
    exp_n_proc=4,
    exp_debug_mode=False,
    eva_numba_decorator=False,
    eva_timeout=300
)

# initialization
evolution = eoh.EVOL(paras)

# run
evolution.run()
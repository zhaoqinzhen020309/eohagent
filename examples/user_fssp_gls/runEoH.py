from eoh import eoh
from eoh.utils.getParas import Paras
from prob import JSSPGLS

# Parameter initilization #
paras = Paras()

# Set your local problem
problem_local = JSSPGLS()

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = problem_local, # Set local problem, else use default problems
                llm_api_endpoint = "api.deepseek.com", # set your LLM endpoint
                llm_api_key = "YOUR_API_KEY",   # set your key
                llm_model = "deepseek-chat",
                ec_pop_size = 10, # number of samples in each population
                ec_n_pop = 20,  # number of populations
                ec_selection_num=5,
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False,
                eva_numba_decorator = False,
                eva_timeout = 4000  # 约67分钟，用于64个实例评估（符合论文要求）
                # 64 instances × 60 sec/instance ≈ 3840 sec + buffer
                )

# initilization
evolution = eoh.EVOL(paras)

# run
evolution.run()
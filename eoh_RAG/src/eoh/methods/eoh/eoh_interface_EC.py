import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures


class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, llm_use_local, llm_local_url, debug_mode,
                 interface_prob, select, n_p, timeout, use_numba, **kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, llm_use_local, llm_local_url, debug_mode, prompts,
                              **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p

        self.timeout = timeout
        self.use_numba = use_numba

    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
            # Write the code to the file
            file.write(code)
        return

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate(self, population, code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation(self):

        n_create = 2

        population = []

        for i in range(n_create):
            _, pop = self.get_algorithm([], 'i1')
            for p in pop:
                if p:  # Check if p is valid
                    population.append(p)

        return population

    def population_generation_seed(self, seeds, n_p):

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

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

        return population

    def _get_alg(self, pop, operator, rag_guidance=None):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }

        try:
            if operator == "i1":
                parents = None
                [offspring['code'], offspring['algorithm']] = self.evol.i1(rag_guidance)
            elif operator == "e1":
                parents = self.select.parent_selection(pop, self.m)
                [offspring['code'], offspring['algorithm']] = self.evol.e1(parents, rag_guidance)
            elif operator == "e2":
                parents = self.select.parent_selection(pop, self.m)
                [offspring['code'], offspring['algorithm']] = self.evol.e2(parents, rag_guidance)
            elif operator == "m1":
                parents = self.select.parent_selection(pop, 1)
                [offspring['code'], offspring['algorithm']] = self.evol.m1(parents[0], rag_guidance)
            elif operator == "m2":
                parents = self.select.parent_selection(pop, 1)
                [offspring['code'], offspring['algorithm']] = self.evol.m2(parents[0], rag_guidance)
            elif operator == "m3":
                parents = self.select.parent_selection(pop, 1)
                [offspring['code'], offspring['algorithm']] = self.evol.m3(parents[0], rag_guidance)
            else:
                print(f"Evolution operator [{operator}] has not been implemented ! \n")
                parents = None

            return parents, offspring

        except Exception as e:
            print(f"Error in _get_alg: {e}")
            return None, offspring

    def get_offspring(self, pop, operator, rag_guidance=None):

        try:
            p, offspring = self._get_alg(pop, operator, rag_guidance)

            if not offspring['code'] or not offspring['algorithm']:
                return p, {
                    'algorithm': None,
                    'code': None,
                    'objective': None,
                    'other_inf': None
                }

            if self.use_numba:
                # Regular expression pattern to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"
                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])
                if match:
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']
            else:
                code = offspring['code']

            n_retry = 1
            while self.check_duplicate(pop, offspring['code']):
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")

                p, offspring = self._get_alg(pop, operator, rag_guidance)

                if not offspring['code']:
                    break

                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, offspring['code'])
                    if match:
                        function_name = match.group(1)
                        code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                    else:
                        code = offspring['code']
                else:
                    code = offspring['code']

                if n_retry > 1:
                    break

            # Evaluate the algorithm
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code)
                fitness = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5)
                future.cancel()

        except Exception as e:
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        return p, offspring

    def get_algorithm(self, pop, operator, rag_guidance=None):
        results = []
        try:
            results = Parallel(n_jobs=self.n_p, timeout=self.timeout + 15)(
                delayed(self.get_offspring)(pop, operator, rag_guidance) for _ in range(self.pop_size)
            )
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")

        time.sleep(2)

        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off
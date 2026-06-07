import numpy as np
import json
import random
import time

from .eoh_interface_EC import InterfaceEC
from .unified_rag_system import UnifiedRAGSystem
from .rag_config import RAGConfig


# main class for eoh with unified RAG enhancement
class EOH:

    # initialization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage

        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # Experimental settings
        self.pop_size = paras.ec_pop_size  # population size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc

        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        print("- EoH Unified RAG parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            if off:  # Check if offspring is valid
                for ind in population:
                    if ind['objective'] == off['objective']:
                        if (self.debug_mode):
                            print("duplicated result, retrying ... ")
                population.append(off)

    # run eoh with unified RAG enhancement
    def run(self):

        print("- Unified RAG Enhanced Evolution Start -")

        time_start = time.time()

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.use_local_llm, self.llm_local_url,
                                   self.debug_mode, interface_prob, select=self.select, n_p=self.exp_n_proc,
                                   timeout=self.timeout, use_numba=self.use_numba)

        # Initialize Unified RAG system with only 2 parameters
        print("- Initializing Unified RAG System -")
        rag_config = RAGConfig()
        # rag_config.MIN_ALGORITHMS = 20  # First generation (~80 algorithms) then activate
        # rag_config.MAX_RETRIEVED = 3  # Retrieve 5 global best experiences

        unified_rag = UnifiedRAGSystem(
            interface_llm=interface_ec.evol.interface_llm,
            config=rag_config,
            results_path=self.output_path + "/results"
        )

        # initialization
        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data, self.exp_n_proc)

            # Collect initial generation for RAG
            unified_rag.collect_and_save_generation(0, population)

            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)

                # Collect loaded generation for RAG
                unified_rag.collect_and_save_generation(0, population)

                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)

                # Collect initial generation for RAG
                unified_rag.collect_and_save_generation(0, population)

                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                n_start = 0

        # main loop with unified RAG integration
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):
            current_generation_algorithms = []

            print(f"\n--- Generation {pop + 1} ---")
            print(f"RAG Status: {'Active' if unified_rag.rag_active else 'Collecting'} "
                  f"({len(unified_rag.unified_knowledge_base)} algorithms in knowledge base)")

            # Execute operators with unified RAG enhancement
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                op_w = self.operator_weights[i]

                if (np.random.rand() < op_w):
                    # Get unified RAG guidance (cross-operator learning)
                    rag_guidance = unified_rag.get_rag_guidance(op, population)

                    if rag_guidance and unified_rag.rag_active:
                        print(" [RAG]", end="|")

                    # Execute operator with RAG guidance
                    parents, offsprings = interface_ec.get_algorithm(population, op, rag_guidance)

                    # Mark operator source for cross-operator learning
                    for off in offsprings:
                        if off and off.get('objective') is not None:
                            off['operator'] = op  # Record which operator generated this
                            current_generation_algorithms.append(off)

                else:
                    offsprings = []

                self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                for off in offsprings:
                    if off and off.get('objective') is not None:
                        print(" Obj: ", off['objective'], end="|")

                # population management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()

            # Feed generation results to unified RAG system
            unified_rag.collect_and_save_generation(pop + 1, current_generation_algorithms)

            print(
                f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time() - time_start) / 60):.1f} m")
            print(f"RAG Status: {'Active' if unified_rag.rag_active else 'Collecting'} "
                  f"- Knowledge Base: {len(unified_rag.unified_knowledge_base)} algorithms")
            print("Pop Objs: ", end=" ")
            for i in range(min(len(population), 10)):  # Show first 10
                print(str(population[i]['objective']) + " ", end="")
            print()

        print("- Unified RAG Enhanced EoH Evolution Completed -")
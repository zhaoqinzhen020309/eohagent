def build_problem_card(config: dict, seed) -> dict:
    problem = config["project"]["problem"]
    if problem == "bp_online":
        return {
            "problem_name": "Online Bin Packing",
            "heuristic_signature": "def score(item, bins): return scores",
            "objective": "minimize bin usage or excess against lower bound",
            "objective_direction": config["evaluator"].get("objective_direction", "minimize"),
            "visible_information": ["current item size", "remaining capacities of feasible bins"],
            "forbidden_information": ["future items", "offline sorting of all items"],
            "baseline_principle": seed.algorithm,
            "expected_function_name": config["evaluator"].get("function_name", "score"),
        }
    if problem == "tsp_construct":
        return {
            "problem_name": "TSP Constructive Heuristic",
            "heuristic_signature": "def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix): return next_node",
            "objective": "minimize constructed tour length",
            "objective_direction": config["evaluator"].get("objective_direction", "minimize"),
            "visible_information": ["current node", "destination node", "unvisited candidate nodes", "distance matrix"],
            "forbidden_information": ["future evaluation instances", "oracle tours"],
            "baseline_principle": seed.algorithm,
            "expected_function_name": config["evaluator"].get("function_name", "select_next_node"),
        }
    if problem == "tsp_gls":
        return {
            "problem_name": "TSP Guided Local Search",
            "heuristic_signature": "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used): return updated_edge_distance",
            "objective": "minimize average GLS percentage gap on the TSPAEL64 training instances",
            "objective_direction": config["evaluator"].get("objective_direction", "minimize"),
            "visible_information": [
                "current edge distance matrix",
                "current local optimal tour",
                "edge usage count matrix",
            ],
            "forbidden_information": [
                "TSPLib test-set results during training",
                "oracle optimal tours",
                "future GLS search trajectory",
            ],
            "baseline_principle": seed.algorithm,
            "expected_function_name": config["evaluator"].get("function_name", "update_edge_distance"),
            "training_data": "examples/user_tsp_gls_OPRO/TrainingData/TSPAEL64.pkl",
            "final_test_protocol": "examples/user_tsp_gls_OPRO/eval_tsp.py over TestingData/TSPLib",
        }
    if problem == "tsp_gls_arch":
        return {
            "problem_name": "TSP Guided Local Search Architecture Design",
            "heuristic_signature": "class CandidateGLSArchitecture with update_edge_distance, select_penalty_edges, perturbation_schedule, and restart_policy",
            "objective": "minimize average GLS percentage gap on the TSPAEL64 training instances",
            "objective_direction": config["evaluator"].get("objective_direction", "minimize"),
            "visible_information": [
                "current edge distance matrix",
                "current local optimal tour",
                "current best tour",
                "edge penalty count matrix",
                "GLS iteration number",
                "current and best costs for restart decisions",
            ],
            "forbidden_information": [
                "TSPLib test-set results during training",
                "oracle optimal tours",
                "future GLS search trajectory",
                "rewriting the TSP solver or local search operators",
            ],
            "baseline_principle": seed.algorithm,
            "expected_architecture_class": config["evaluator"].get(
                "architecture_class_name",
                "CandidateGLSArchitecture",
            ),
            "architecture_layers": [
                "distance_update",
                "penalty_edge_selection",
                "perturbation_schedule",
                "restart_policy",
            ],
            "training_data": "examples/user_tsp_gls_OPRO/TrainingData/TSPAEL64.pkl",
            "final_test_protocol": "examples/user_tsp_gls_OPRO/eval_tsp.py over TestingData/TSPLib",
        }
    return {
        "problem_name": problem,
        "heuristic_signature": "custom",
        "objective_direction": config["evaluator"].get("objective_direction", "minimize"),
        "baseline_principle": seed.algorithm,
        "expected_function_name": config["evaluator"].get("function_name", "heuristic"),
    }

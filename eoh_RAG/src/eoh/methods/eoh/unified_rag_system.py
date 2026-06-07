import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from .rag_config import RAGConfig


class UnifiedRAGSystem:
    """Enhanced unified RAG system with consolidated logging for easy analysis"""

    def __init__(self, interface_llm, config: RAGConfig = None, results_path: str = None):
        self.interface_llm = interface_llm
        self.config = config or RAGConfig()

        # CORE PARAMETERS
        self.MIN_ALGORITHMS = self.config.MIN_ALGORITHMS  # 20
        self.MAX_RETRIEVED = self.config.MAX_RETRIEVED  # 3

        # Path management
        self.base_path = results_path or self.config.RESULTS_BASE_PATH
        self.pops_path = os.path.join(self.base_path, "pops")
        self.rag_path = os.path.join(self.base_path, "rag_knowledge")
        self.best_path = os.path.join(self.base_path, "pops_best")

        # NEW: Consolidated logging
        self.consolidated_logs_path = os.path.join(self.base_path, "consolidated_logs")
        self._ensure_directories()

        # System state
        self.unified_knowledge_base = []
        self.rag_active = False
        self.current_generation = 0

        # NEW: Consolidated log structures
        self.experiment_log = {
            "experiment_info": {
                "start_time": datetime.now().isoformat(),
                "rag_config": {
                    "min_algorithms": self.MIN_ALGORITHMS,
                    "max_retrieved": self.MAX_RETRIEVED
                }
            },
            "generations": {},
            "rag_sessions": {},
            "system_events": []
        }

        self._load_existing_data()

    def _ensure_directories(self):
        """Ensure all directories exist"""
        os.makedirs(self.pops_path, exist_ok=True)
        os.makedirs(self.rag_path, exist_ok=True)
        os.makedirs(self.best_path, exist_ok=True)
        os.makedirs(self.consolidated_logs_path, exist_ok=True)

    def _debug_print(self, message):
        """Debug output"""
        if self.config.ENABLE_DEBUG:
            print(f"[UnifiedRAG] {message}")

    def _load_existing_data(self):
        """Load existing RAG data if available"""
        knowledge_file = os.path.join(self.rag_path, "unified_knowledge_base.json")
        if os.path.exists(knowledge_file):
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.unified_knowledge_base = data.get("algorithms", [])
                    if len(self.unified_knowledge_base) >= self.MIN_ALGORITHMS:
                        self.rag_active = True
                    self._debug_print(f"Loaded existing knowledge base: {len(self.unified_knowledge_base)} algorithms")
            except Exception as e:
                self._debug_print(f"Failed to load knowledge base: {e}")

    def collect_and_save_generation(self, generation_num: int, algorithms: List[Dict]):
        """Collect generation algorithms and save to unified structure"""

        self.current_generation = generation_num

        # Log generation start
        self._log_system_event("GENERATION_START", {
            "generation": generation_num,
            "algorithm_count": len(algorithms),
            "rag_active": self.rag_active
        })

        # 1. Save to pops directory (enhanced format)
        generation_data = {
            "generation": generation_num,
            "timestamp": datetime.now().isoformat(),
            "rag_active": self.rag_active,
            "total_algorithms": len(algorithms),
            "algorithms": algorithms
        }

        pops_file = os.path.join(self.pops_path, f"population_generation_{generation_num}.json")
        with open(pops_file, 'w', encoding='utf-8') as f:
            json.dump(generation_data, f, indent=2, ensure_ascii=False)

        # 2. Update unified knowledge base and log to consolidated structure
        generation_algorithms = []
        principle_extractions = {}

        for i, alg in enumerate(algorithms):
            if alg and alg.get('objective') is not None:
                alg_id = f"gen{generation_num}_alg{i}"

                knowledge_entry = {
                    "id": alg_id,
                    "algorithm": alg.get('algorithm', ''),
                    "code": alg.get('code', ''),
                    "objective": alg.get('objective'),
                    "operator": alg.get('operator', 'unknown'),
                    "generation": generation_num,
                    "timestamp": time.time(),
                    "principle_analysis": None
                }

                self.unified_knowledge_base.append(knowledge_entry)
                generation_algorithms.append(knowledge_entry)

        # 3. Check RAG activation condition
        if not self.rag_active and len(self.unified_knowledge_base) >= self.MIN_ALGORITHMS:
            self._activate_rag_system()

        # 4. Generate principle analysis for new algorithms (if RAG is active)
        elif self.rag_active:
            for entry in generation_algorithms:
                if entry["principle_analysis"] is None:
                    principle_analysis = self._generate_principle_analysis(entry)
                    entry["principle_analysis"] = principle_analysis

                    # Update in main knowledge base
                    for kb_entry in self.unified_knowledge_base:
                        if kb_entry["id"] == entry["id"]:
                            kb_entry["principle_analysis"] = principle_analysis
                            break

                    # Log principle extraction
                    principle_extractions[entry["id"]] = {
                        "timestamp": datetime.now().isoformat(),
                        "principle": principle_analysis,
                        "objective": entry["objective"],
                        "operator": entry["operator"]
                    }

        # 5. Add to consolidated log
        self.experiment_log["generations"][str(generation_num)] = {
            "timestamp": datetime.now().isoformat(),
            "algorithms": generation_algorithms,
            "principle_extractions": principle_extractions,
            "performance_summary": self._calculate_generation_performance(generation_algorithms),
            "rag_active": self.rag_active
        }

        # 6. Save all data
        self._save_rag_knowledge()
        self._save_consolidated_logs()

        # 7. Save best individual
        if algorithms:
            valid_algorithms = [alg for alg in algorithms if alg.get('objective') is not None]
            if valid_algorithms:
                best_alg = min(valid_algorithms, key=lambda x: x['objective'])
                best_file = os.path.join(self.best_path, f"population_generation_{generation_num}.json")
                with open(best_file, 'w', encoding='utf-8') as f:
                    json.dump(best_alg, f, indent=2, ensure_ascii=False)

        self._debug_print(f"Generation {generation_num} saved: {len(algorithms)} algorithms")

    def _activate_rag_system(self):
        """Activate RAG system and build knowledge base with principle extraction"""
        self._debug_print(f"Activating RAG system - analyzing {len(self.unified_knowledge_base)} algorithms")

        self._log_system_event("RAG_ACTIVATION_START", {
            "knowledge_base_size": len(self.unified_knowledge_base)
        })

        for i, alg in enumerate(self.unified_knowledge_base):
            if alg["principle_analysis"] is None:
                self._debug_print(f"Extracting principles {i + 1}/{len(self.unified_knowledge_base)}")
                alg["principle_analysis"] = self._generate_principle_analysis(alg)

        self.rag_active = True
        self._debug_print("RAG system activation completed")

        self._log_system_event("RAG_ACTIVATION_COMPLETE", {
            "total_analyzed": len(self.unified_knowledge_base)
        })

    def _generate_principle_analysis(self, alg_data):
        """Generate principle extraction with logging"""
        try:
            prompt = self.config.get_principle_extraction_prompt(alg_data)
            analysis = self.interface_llm.get_response(prompt)
            return analysis
        except Exception as e:
            self._log_system_event("PRINCIPLE_EXTRACTION_ERROR", {
                "algorithm_id": alg_data.get("id", "unknown"),
                "error": str(e)
            })
            return f"Error extracting principles: {str(e)}"

    def get_rag_guidance(self, current_operator: str, current_population: List[Dict]) -> Optional[str]:
        """Get RAG guidance with comprehensive logging"""

        session_id = f"gen{self.current_generation}_{current_operator}"

        if not self.rag_active:
            self._log_rag_session(session_id, {
                "operator": current_operator,
                "result": "SKIPPED",
                "reason": "RAG not active",
                "guidance_content": None
            })
            return None

        session_start = datetime.now()

        try:
            # Layer 1: Extract universal principles from best algorithms
            universal_principles, retrieval_details = self._extract_universal_principles_with_logging()

            if not universal_principles:
                self._log_rag_session(session_id, {
                    "operator": current_operator,
                    "result": "FAILED",
                    "reason": "No universal principles found",
                    "retrieval_details": retrieval_details,
                    "guidance_content": None
                })
                return None

            # Layer 2: Generate operator-specific guidance
            operator_guidance = self._generate_operator_specific_guidance(current_operator, universal_principles)

            if not operator_guidance:
                self._log_rag_session(session_id, {
                    "operator": current_operator,
                    "result": "FAILED",
                    "reason": "Guidance generation failed",
                    "retrieval_details": retrieval_details,
                    "guidance_content": None
                })
                return None

            # Success - log complete session
            session_duration = (datetime.now() - session_start).total_seconds()

            self._log_rag_session(session_id, {
                "operator": current_operator,
                "result": "SUCCESS",
                "session_duration_seconds": session_duration,
                "retrieval_details": retrieval_details,
                "guidance_content": operator_guidance,
                "guidance_length": len(operator_guidance)
            })

            self._debug_print(f"Generated specialized guidance for {current_operator}: {len(operator_guidance)} chars")

            return operator_guidance

        except Exception as e:
            session_duration = (datetime.now() - session_start).total_seconds()

            self._log_rag_session(session_id, {
                "operator": current_operator,
                "result": "ERROR",
                "session_duration_seconds": session_duration,
                "error": str(e),
                "guidance_content": None
            })

            self._debug_print(f"Error in RAG guidance generation: {e}")
            return None

    def _extract_universal_principles_with_logging(self) -> tuple[str, dict]:
        """Extract universal design principles with detailed logging"""

        best_algorithms = self._retrieve_global_best_experiences()

        retrieval_details = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len([alg for alg in self.unified_knowledge_base
                                     if alg.get("principle_analysis") and alg.get("objective") is not None]),
            "retrieved_count": len(best_algorithms),
            "retrieved_algorithms": []
        }

        for alg in best_algorithms:
            retrieval_details["retrieved_algorithms"].append({
                "id": alg.get("id", "unknown"),
                "objective": alg.get("objective"),
                "operator": alg.get("operator"),
                "generation": alg.get("generation"),
                "algorithm_preview": (alg.get("algorithm", "")[:100] + "...") if len(
                    alg.get("algorithm", "")) > 100 else alg.get("algorithm", "")
            })

        if not best_algorithms:
            return "", retrieval_details

        # Collect principles from best algorithms
        principles = []
        for i, alg in enumerate(best_algorithms):
            if alg.get("principle_analysis"):
                principle_text = f"Principle {i + 1}: {alg['principle_analysis']}"
                principles.append(principle_text)

        combined_principles = "\n".join(principles)
        retrieval_details["combined_principles_length"] = len(combined_principles)

        return combined_principles, retrieval_details

    def _generate_operator_specific_guidance(self, current_operator: str, universal_principles: str) -> str:
        """Generate guidance specific to one operator"""
        try:
            operator_directions = self.config.get_operator_directions()
            operator_direction = operator_directions.get(current_operator,
                                                         'Implement principles using your specialized approach')

            guidance = self.config.get_operator_guidance_prompt(
                current_operator=current_operator,
                universal_principles=universal_principles,
                operator_direction=operator_direction
            )

            return guidance
        except Exception as e:
            self._log_system_event("GUIDANCE_GENERATION_ERROR", {
                "operator": current_operator,
                "error": str(e)
            })
            return ""

    def _retrieve_global_best_experiences(self) -> List[Dict]:
        """Retrieve globally best algorithms with principle analysis"""
        analyzed_algorithms = [alg for alg in self.unified_knowledge_base
                               if alg.get("principle_analysis") and alg.get("objective") is not None]

        if not analyzed_algorithms:
            return []

        analyzed_algorithms.sort(key=lambda x: x["objective"])
        return analyzed_algorithms[:self.MAX_RETRIEVED]

    def _log_rag_session(self, session_id: str, session_data: Dict):
        """Log RAG session to consolidated structure"""
        self.experiment_log["rag_sessions"][session_id] = {
            "timestamp": datetime.now().isoformat(),
            "generation": self.current_generation,
            **session_data
        }

        # Immediately save after each RAG session for debugging
        self._save_consolidated_logs()

    def _log_system_event(self, event_type: str, event_data: Dict):
        """Log system events"""
        self.experiment_log["system_events"].append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "generation": self.current_generation,
            **event_data
        })

    def _calculate_generation_performance(self, algorithms: List[Dict]) -> Dict:
        """Calculate performance metrics for a generation"""
        if not algorithms:
            return {}

        objectives = [alg["objective"] for alg in algorithms if alg.get("objective") is not None]
        if not objectives:
            return {}

        by_operator = {}
        for alg in algorithms:
            op = alg.get("operator", "unknown")
            obj = alg.get("objective")
            if obj is not None:
                if op not in by_operator:
                    by_operator[op] = []
                by_operator[op].append(obj)

        operator_stats = {}
        for op, scores in by_operator.items():
            operator_stats[op] = {
                "count": len(scores),
                "best": min(scores),
                "worst": max(scores),
                "average": sum(scores) / len(scores)
            }

        return {
            "best_objective": min(objectives),
            "worst_objective": max(objectives),
            "average_objective": sum(objectives) / len(objectives),
            "total_algorithms": len(objectives),
            "operator_performance": operator_stats
        }

    def _save_consolidated_logs(self):
        """Save consolidated logs to single files"""
        # 1. Complete experiment trace
        trace_file = os.path.join(self.consolidated_logs_path, "complete_experiment_trace.json")
        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)

        # 2. Generate analysis report
        if self.current_generation > 0:  # Only generate after first generation
            self._generate_analysis_report()

    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "experiment_summary": self._analyze_experiment_summary(),
            "performance_evolution": self._analyze_performance_evolution(),
            "rag_effectiveness": self._analyze_rag_effectiveness(),
            "system_issues": self._identify_system_issues(),
            "recommendations": self._generate_recommendations()
        }

        report_file = os.path.join(self.consolidated_logs_path, "analysis_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Also generate human-readable summary
        self._generate_human_readable_summary(report)

    def _analyze_experiment_summary(self) -> Dict:
        """Analyze overall experiment"""
        total_algorithms = sum(len(gen["algorithms"]) for gen in self.experiment_log["generations"].values())
        total_rag_sessions = len(self.experiment_log["rag_sessions"])

        return {
            "total_generations": len(self.experiment_log["generations"]),
            "total_algorithms": total_algorithms,
            "total_rag_sessions": total_rag_sessions,
            "rag_active": self.rag_active,
            "knowledge_base_size": len(self.unified_knowledge_base)
        }

    def _analyze_performance_evolution(self) -> Dict:
        """Analyze performance trends"""
        evolution = []
        for gen_num, gen_data in self.experiment_log["generations"].items():
            perf = gen_data.get("performance_summary", {})
            if perf.get("best_objective") is not None:
                evolution.append({
                    "generation": int(gen_num),
                    "best_objective": perf["best_objective"],
                    "average_objective": perf.get("average_objective"),
                    "total_algorithms": perf.get("total_algorithms", 0)
                })

        # Calculate improvement trend
        trend = "STABLE"
        if len(evolution) >= 2:
            first_best = evolution[0]["best_objective"]
            last_best = evolution[-1]["best_objective"]
            if last_best < first_best * 0.9:
                trend = "IMPROVING"
            elif last_best > first_best * 1.1:
                trend = "DEGRADING"

        return {
            "evolution_data": evolution,
            "performance_trend": trend,
            "breakthrough_generations": [e for e in evolution if e["best_objective"] < 0.03]
        }

    def _analyze_rag_effectiveness(self) -> Dict:
        """Analyze RAG system effectiveness"""
        sessions = self.experiment_log["rag_sessions"]
        if not sessions:
            return {"message": "No RAG sessions found"}

        success_count = sum(1 for s in sessions.values() if s.get("result") == "SUCCESS")
        total_sessions = len(sessions)

        failed_sessions = [sid for sid, s in sessions.items() if s.get("result") != "SUCCESS"]

        # Group failures by generation
        failures_by_gen = {}
        for session_id in failed_sessions:
            gen = sessions[session_id].get("generation", -1)
            if gen not in failures_by_gen:
                failures_by_gen[gen] = []
            failures_by_gen[gen].append(session_id)

        return {
            "success_rate": success_count / total_sessions if total_sessions > 0 else 0,
            "total_sessions": total_sessions,
            "failed_sessions": failed_sessions,
            "failures_by_generation": failures_by_gen
        }

    def _identify_system_issues(self) -> List[Dict]:
        """Identify system issues"""
        issues = []

        # Check for missing RAG sessions
        for gen_num, gen_data in self.experiment_log["generations"].items():
            if gen_data.get("rag_active", False):
                expected_operators = ["e1", "e2", "m1", "m2"]
                existing_sessions = [sid for sid in self.experiment_log["rag_sessions"].keys()
                                     if sid.startswith(f"gen{gen_num}_")]

                missing_ops = []
                for op in expected_operators:
                    session_id = f"gen{gen_num}_{op}"
                    if session_id not in self.experiment_log["rag_sessions"]:
                        missing_ops.append(op)

                if missing_ops:
                    issues.append({
                        "type": "MISSING_RAG_SESSIONS",
                        "generation": int(gen_num),
                        "missing_operators": missing_ops,
                        "severity": "HIGH" if len(missing_ops) > 2 else "MEDIUM"
                    })

        # Check for performance degradation
        perf_evolution = self._analyze_performance_evolution()
        if perf_evolution.get("performance_trend") == "DEGRADING":
            issues.append({
                "type": "PERFORMANCE_DEGRADATION",
                "severity": "MEDIUM",
                "details": "Overall performance trend is degrading"
            })

        return issues

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        issues = self._identify_system_issues()
        rag_effectiveness = self._analyze_rag_effectiveness()

        if rag_effectiveness.get("success_rate", 0) < 0.8:
            recommendations.append(
                "RAG success rate is low. Consider increasing timeout or reducing knowledge base size.")

        missing_sessions = [issue for issue in issues if issue["type"] == "MISSING_RAG_SESSIONS"]
        if missing_sessions:
            recommendations.append(
                "Some RAG sessions are missing. This indicates system instability in later generations.")

        if len(self.unified_knowledge_base) > 80:
            recommendations.append("Knowledge base is large. Consider implementing size limits or periodic pruning.")

        return recommendations

    def _generate_human_readable_summary(self, report: Dict):
        """Generate human-readable summary"""
        summary_lines = []
        summary_lines.append("=== EOH RAG SYSTEM ANALYSIS SUMMARY ===\n")

        # Experiment overview
        exp_summary = report["experiment_summary"]
        summary_lines.append(f"Total Generations: {exp_summary['total_generations']}")
        summary_lines.append(f"Total Algorithms: {exp_summary['total_algorithms']}")
        summary_lines.append(f"RAG Sessions: {exp_summary['total_rag_sessions']}")
        summary_lines.append(f"Knowledge Base Size: {exp_summary['knowledge_base_size']}\n")

        # Performance
        perf = report["performance_evolution"]
        summary_lines.append(f"Performance Trend: {perf['performance_trend']}")
        if perf["evolution_data"]:
            best_gen = min(perf["evolution_data"], key=lambda x: x["best_objective"])
            summary_lines.append(f"Best Result: {best_gen['best_objective']:.5f} (Generation {best_gen['generation']})")

        # RAG effectiveness
        rag_eff = report["rag_effectiveness"]
        if "success_rate" in rag_eff:
            summary_lines.append(f"RAG Success Rate: {rag_eff['success_rate']:.2%}")

        # Issues
        issues = report["system_issues"]
        if issues:
            summary_lines.append(f"\nISSUES DETECTED ({len(issues)}):")
            for issue in issues:
                summary_lines.append(f"- {issue['type']}: {issue.get('details', 'See detailed report')}")

        # Recommendations
        recommendations = report["recommendations"]
        if recommendations:
            summary_lines.append(f"\nRECOMMENDATIONS ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                summary_lines.append(f"{i}. {rec}")

        summary_text = "\n".join(summary_lines)

        summary_file = os.path.join(self.consolidated_logs_path, "analysis_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)

    def _save_rag_knowledge(self):
        """Save RAG knowledge to dedicated directory"""
        knowledge_file = os.path.join(self.rag_path, "unified_knowledge_base.json")
        knowledge_data = {
            "total_algorithms": len(self.unified_knowledge_base),
            "rag_active": self.rag_active,
            "last_updated": datetime.now().isoformat(),
            "algorithms": self.unified_knowledge_base
        }

        with open(knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = {
            "rag_active": self.rag_active,
            "total_algorithms": len(self.unified_knowledge_base),
            "analyzed_count": sum(1 for alg in self.unified_knowledge_base if alg.get("principle_analysis")),
            "last_updated": datetime.now().isoformat(),
            "config": {
                "min_algorithms": self.MIN_ALGORITHMS,
                "max_retrieved": self.MAX_RETRIEVED
            },
            "statistics": self._calculate_statistics()
        }

        metadata_file = os.path.join(self.rag_path, "rag_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _calculate_statistics(self):
        """Calculate simple statistics"""
        if not self.unified_knowledge_base:
            return {}

        by_operator = {}
        by_generation = {}
        performances = []

        for alg in self.unified_knowledge_base:
            op = alg.get("operator", "unknown")
            by_operator[op] = by_operator.get(op, 0) + 1

            gen = alg.get("generation", -1)
            by_generation[str(gen)] = by_generation.get(str(gen), 0) + 1

            if alg.get("objective") is not None and alg["objective"] != float('inf'):
                performances.append(alg["objective"])

        stats = {
            "by_operator": by_operator,
            "by_generation": by_generation
        }

        if performances:
            stats["performance"] = {
                "best": min(performances),
                "worst": max(performances),
                "average": sum(performances) / len(performances),
                "count": len(performances)
            }

        return stats
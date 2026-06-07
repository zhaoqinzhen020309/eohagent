import json
import os


class RAGConfig:
    """Enhanced RAG configuration with three-layer architecture"""

    def __init__(self):
        # === CORE PARAMETERS ===
        self.MIN_ALGORITHMS = 20  # RAG activation threshold
        self.MAX_RETRIEVED = 3  # Number of experiences to retrieve

        # === Fixed paths and settings ===
        self.RESULTS_BASE_PATH = "./results"
        self.ENABLE_DEBUG = True

        # === THREE-LAYER ARCHITECTURE TEMPLATES ===

        # Layer 1: Principle Extraction Template (English) - STRICT length control
        self.PRINCIPLE_EXTRACTION_TEMPLATE = """Extract design principles from this successful algorithm (STRICT 300 character limit):

Algorithm: {algorithm}
Performance: {objective:.4f}
Operator: {operator}

## Core Success Principles (max 150 chars)
Key design concepts that made this successful? (abstract principles only, no formulas)

## Common Pitfalls Avoided (max 150 chars)
What typical problems did this approach avoid?

CRITICAL: Response must be ≤300 characters total. Focus on WHY it works, not HOW."""

        # Layer 2: Operator Specialization Template (English) - STRICT length control
        self.OPERATOR_SPECIALIZATION_TEMPLATE = """=== {current_operator} Operator Specialized Guidance (STRICT 800 char limit) ===

Universal Design Principles (from best algorithms):
{universal_principles}

Your Professional Implementation Direction:
{operator_direction}

Create a novel algorithm using your specialized approach to implement these principles.
Do NOT copy existing formulas - innovate your own implementation methods.

CRITICAL: Keep this guidance ≤800 characters. Generate new algorithm now."""

    def get_principle_extraction_prompt(self, algorithm_data):
        """Generate principle extraction prompt with length control"""
        return self.PRINCIPLE_EXTRACTION_TEMPLATE.format(**algorithm_data)

    def get_operator_guidance_prompt(self, current_operator, universal_principles, operator_direction):
        """Generate operator-specific guidance prompt"""
        return self.OPERATOR_SPECIALIZATION_TEMPLATE.format(
            current_operator=current_operator,
            universal_principles=universal_principles,
            operator_direction=operator_direction
        )

    def get_operator_directions(self):
        """Define specialized directions for each operator"""
        return {
            'e1': 'Implement principles through geometric relationship analysis and spatial configuration optimization',
            'e2': 'Implement principles through statistical prediction methods and probabilistic estimation techniques',
            'm1': 'Implement principles through heuristic rule design and logical decision frameworks',
            'm2': 'Implement principles through parameter tuning and weight balancing optimization'
        }
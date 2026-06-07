from typing import List, Dict, Tuple


class PromptCompressor:
    """Prompt压缩器，智能压缩prompt同时保留关键信息"""

    @staticmethod
    def compress_function_code(code: str, level: str = 'medium') -> str:
        """压缩函数代码"""
        if level == 'full':
            return code

        if level == 'medium':
            lines = code.split('\n')
            compressed = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if len(line) > 80:
                        line = line[:77] + '...'
                    compressed.append(line)
            return '\n'.join(compressed)

        elif level == 'high':
            lines = code.split('\n')
            result = []
            for line in lines:
                if 'def ' in line:
                    result.append(line)
                elif 'return' in line:
                    result.append('    ...')
                    result.append(line)
                    break
            return '\n'.join(result)

        return code

    @staticmethod
    def compress_algorithm_desc(algorithm: str, max_length: int = 100) -> str:
        """压缩算法描述"""
        if len(algorithm) <= max_length:
            return algorithm

        sentences = algorithm.split('.')
        if sentences:
            keywords = ['optimal', 'minimize', 'maximize', 'greedy', 'dynamic', 'edge', 'distance', 'matrix']
            important = [sentences[0]]
            for sent in sentences[1:]:
                if any(kw in sent.lower() for kw in keywords):
                    important.append(sent)
                    if len('.'.join(important)) > max_length:
                        break

            result = '.'.join(important)
            if len(result) > max_length:
                result = result[:max_length - 3] + '...'
            return result

        return algorithm[:max_length - 3] + '...'

    @staticmethod
    def hierarchical_compress_individuals(individuals: List[Dict]) -> str:
        """分层压缩多个个体"""
        if not individuals:
            return ""

        compressed_parts = []
        sorted_indivs = sorted(individuals, key=lambda x: x.get('objective', float('inf')))

        for i, indiv in enumerate(sorted_indivs):
            if i < 2:  # Top 2: 完整保留
                level = 'full'
                desc_len = 150
            elif i < 5:  # Top 3-5: 中等压缩
                level = 'medium'
                desc_len = 100
            else:  # 其余: 高度压缩
                level = 'high'
                desc_len = 50

            compressed_desc = PromptCompressor.compress_algorithm_desc(
                indiv.get('algorithm', ''), max_length=desc_len
            )
            compressed_code = PromptCompressor.compress_function_code(
                indiv.get('code', ''), level
            )

            compressed_parts.append(
                f"No. {i + 1} algorithm (score={indiv.get('objective', 'N/A'):.3f}):\n{compressed_desc}\n{compressed_code}"
            )

        return '\n'.join(compressed_parts)

    @staticmethod
    def compress_single_individual(individual: Dict, level: str = 'medium') -> Tuple[str, str]:
        """压缩单个个体"""
        desc_len = 100 if level == 'medium' else 150
        compressed_desc = PromptCompressor.compress_algorithm_desc(
            individual.get('algorithm', ''), max_length=desc_len
        )
        compressed_code = PromptCompressor.compress_function_code(
            individual.get('code', ''), level
        )
        return compressed_desc, compressed_code
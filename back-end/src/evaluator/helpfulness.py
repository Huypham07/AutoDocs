from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Optional

import adalflow as adal
from api.models.docs_gen import Section
from api.models.docs_gen import Structure
from shared.logging import get_logger
from shared.settings.advanced_configs import get_generator_model_config

from .base import BaseEvaluator

logger = get_logger(__name__)


@dataclass
class HelpfulnessScore(adal.DataClass):
    """Data class to hold helpfulness evaluation results"""
    goal_clarity: float = field(default=0.0)
    conciseness: float = field(default=0.0)
    technical_depth: float = field(default=0.0)
    diagram_completeness: float = field(default=0.0)
    repo_type_alignment: float = field(default=0.0)
    reasoning: str = field(default='')

    __output_fields__ = [
        'goal_clarity',
        'conciseness',
        'technical_depth',
        'diagram_completeness',
        'repo_type_alignment',
        'reasoning',
    ]


EVALUATION_TEMPLATE = """
You are an expert documentation evaluator. Your task is to assess the helpfulness of technical documentation for a software repository.

<DOCUMENTATION_STRUCTURE>
Repository: {{repo_name}}
Title: {{title}}
Description: {{description}}

Sections and Pages:
{{sections_content}}
</DOCUMENTATION_STRUCTURE>

<EVALUATION_CRITERIA>
Evaluate the documentation helpfulness across these 5 dimensions (each scored 0-10):

1. **Goal Clarity (0-10)**: Does the documentation clearly explain the repository's purpose, main features, and value proposition?
   - 9-10: Crystal clear purpose with compelling value proposition
   - 7-8: Clear purpose with good explanation of features
   - 5-6: Somewhat clear but missing key details
   - 3-4: Vague or confusing purpose
   - 0-2: No clear indication of repository purpose

2. **Conciseness (0-10)**: Is information presented concisely while remaining comprehensive?
   - 9-10: Perfect balance of brevity and completeness
   - 7-8: Generally concise with good information density
   - 5-6: Some redundancy or verbosity but acceptable
   - 3-4: Significant redundancy or poor information organization
   - 0-2: Extremely verbose or poorly structured

3. **Technical Depth (0-10)**: Does it provide both high-level overview and detailed technical information?
   - 9-10: Excellent progression from overview to deep technical details
   - 7-8: Good balance of overview and technical content
   - 5-6: Either too shallow or missing overview/details
   - 3-4: Poor technical depth or organization
   - 0-2: Lacks meaningful technical information

4. **Diagram Completeness (0-10)**: Are there sufficient illustrative diagrams, examples, and visual aids?
   - 9-10: Rich visual content that enhances understanding
   - 7-8: Good use of diagrams and examples
   - 5-6: Some visual aids but could be improved
   - 3-4: Minimal visual content
   - 0-2: No meaningful diagrams or examples

5. **Repo Type Alignment (0-10)**: Is the documentation style appropriate for the repository type?
   - For frameworks: Installation, API reference, examples, extensibility
   - For applications: Setup, configuration, usage, deployment
   - For libraries: API documentation, examples, integration guides
   - 9-10: Perfect alignment with repository type expectations
   - 7-8: Good alignment with minor gaps
   - 5-6: Adequate but missing some type-specific elements
   - 3-4: Poor alignment with repository type
   - 0-2: Completely inappropriate for repository type

</EVALUATION_CRITERIA>

<OUTPUT_FORMAT>
Provide your evaluation in this exact JSON format:

{
  "goal_clarity": <score_0_to_10>,
  "conciseness": <score_0_to_10>,
  "technical_depth": <score_0_to_10>,
  "diagram_completeness": <score_0_to_10>,
  "repo_type_alignment": <score_0_to_10>,
  "reasoning": "<detailed_explanation_of_scores>"
}
</OUTPUT_FORMAT>

Evaluate the documentation thoroughly and provide specific reasoning for each score.
"""


class HelpfulnessEvaluator(BaseEvaluator):
    """
    Evaluates the helpfulness of documentation using LLM-as-Judge approach.

    This evaluator assesses documentation quality across multiple dimensions:
    - Goal clarity: Does the documentation clearly state the repository's purpose?
    - Conciseness: Is the information presented concisely while being comprehensive?
    - Technical depth: Does it provide both overview and detailed technical information?
    - Diagram completeness: Are there sufficient illustrative diagrams?
    - Repo type alignment: Is the documentation appropriate for the repository type?
    """

    def __init__(self, provider: str, model: Optional[str] = None):
        super().__init__(
            name='Helpfulness Evaluator',
            description='Evaluates the helpfulness of the documentation using LLM-as-Judge approach.',
        )

        generator_config = get_generator_model_config(provider, model)

        data_parser = adal.DataClassParser(data_class=HelpfulnessScore, return_data_class=True)

        # Initialize the generator with evaluation prompt
        self.judge_generator = adal.Generator(
            model_client=generator_config['model_client'](),
            model_kwargs=generator_config['model_kwargs'],
            template=EVALUATION_TEMPLATE,
            output_processors=data_parser,
        )

    def _extract_sections_content(self, structure: Structure) -> str:
        """Extract and format sections content for evaluation"""
        content_parts = []

        def process_section(section: Section, level: int = 0) -> None:
            indent = '  ' * level
            content_parts.append(f"{indent}Section: {section.section_title}")

            # Add pages content
            for page in section.pages:
                content_parts.append(f"{indent}  Page: {page.page_title}")
                if page.content and page.content.strip():
                    # Truncate very long content
                    content = page.content.strip()
                    if len(content) > 500:
                        content = content[:500] + '...'
                    content_parts.append(f"{indent}    Content: {content}")

                if page.file_paths:
                    content_parts.append(f"{indent}    Files: {', '.join(page.file_paths)}")

            # Process subsections recursively
            for subsection in section.subsections:
                process_section(subsection, level + 1)

        for root_section in structure.root_sections:
            process_section(root_section)

        return '\n'.join(content_parts)

    def evaluate(self, structure: Structure) -> float:
        """
        Evaluate the helpfulness of the documentation structure using LLM-as-Judge.

        Args:
            structure (Structure): The documentation structure to evaluate.

        Returns:
            float: The helpfulness score between 0 and 1.
        """
        try:
            # Prepare input data for the LLM judge
            sections_content = self._extract_sections_content(structure)

            # Generate evaluation using LLM-as-Judge
            evaluation_result = self.judge_generator.call(
                prompt_kwargs={
                    'repo_name': structure.repo_name or 'Unknown Repository',
                    'title': structure.title,
                    'description': structure.description,
                    'sections_content': sections_content,
                },
            ).data

            if not isinstance(evaluation_result, HelpfulnessScore):
                raise ValueError('Evaluation result is not in the expected format.')

            # Store detailed results for debugging/analysis
            self._detailed_results = evaluation_result
            print(type(evaluation_result))

            goal_clarity = evaluation_result.goal_clarity
            conciseness = evaluation_result.conciseness
            technical_depth = evaluation_result.technical_depth
            diagram_completeness = evaluation_result.diagram_completeness
            repo_type_alignment = evaluation_result.repo_type_alignment

            overall_score = sum([goal_clarity, conciseness, technical_depth, diagram_completeness, repo_type_alignment]) / 5 / 10.0
            # Update the score
            self.score = overall_score

            logger.info(f"Helpfulness evaluation completed. Score: {self.score:.3f}")
            logger.debug(f"Detailed scores: "
                         f"Goal clarity: {evaluation_result.goal_clarity:.3f}, "
                         f"Conciseness: {evaluation_result.conciseness:.3f}, "
                         f"Technical depth: {evaluation_result.technical_depth:.3f}, "
                         f"Diagram completeness: {evaluation_result.diagram_completeness:.3f}, "
                         f"Repo type alignment: {evaluation_result.repo_type_alignment:.3f}")

            return self.score

        except Exception as e:
            logger.error(f"Error during helpfulness evaluation: {e}")
            self.score = 0.0
            return self.score

    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Get detailed evaluation results for analysis.

        Returns:
            Dict containing detailed scores and reasoning.
        """
        if hasattr(self, '_detailed_results'):
            return {
                'overall_score': self.score,
                'goal_clarity': self._detailed_results.goal_clarity,
                'conciseness': self._detailed_results.conciseness,
                'technical_depth': self._detailed_results.technical_depth,
                'diagram_completeness': self._detailed_results.diagram_completeness,
                'repo_type_alignment': self._detailed_results.repo_type_alignment,
                'reasoning': self._detailed_results.reasoning,
            }
        return {'error': 'No evaluation results available'}

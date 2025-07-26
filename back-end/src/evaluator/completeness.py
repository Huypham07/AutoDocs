from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Tuple

from api.models.docs_gen import get_pages_from_structure
from api.models.docs_gen import Section
from api.models.docs_gen import Structure

from .base import BaseEvaluator


class CompletenessEvaluator(BaseEvaluator):
    """
    Evaluates the completeness of documentation structure.

    Checks multiple aspects:
    - Presence of essential metadata (title, description)
    - Section structure completeness
    - Page content availability
    - File path coverage
    - Hierarchical structure integrity
    """

    def __init__(self):
        super().__init__(
            name='Completeness Evaluator',
            description='Evaluates the completeness of the documentation structure including metadata, sections, pages, and content coverage.',
        )

    def evaluate(self, structure: Structure) -> float:
        """
        Evaluate the completeness of the documentation structure.

        Args:
            structure (Structure): The documentation structure to evaluate.

        Returns:
            float: The completeness score between 0 and 1.
        """
        total_score = 0.0
        max_possible_score = 0.0

        self.structure = structure

        # 1. Basic metadata completeness (weight: 15%)
        self.metadata_score, self.metadata_max = self._evaluate_metadata_completeness(structure)
        total_score += self.metadata_score * 0.15
        max_possible_score += self.metadata_max * 0.15

        # 2. Section structure completeness (weight: 25%)
        self.section_score, self.section_max = self._evaluate_section_completeness(structure)
        total_score += self.section_score * 0.25
        max_possible_score += self.section_max * 0.25

        # 3. Page completeness (weight: 35%)
        self.page_score, self.page_max = self._evaluate_page_completeness(structure)
        total_score += self.page_score * 0.35
        max_possible_score += self.page_max * 0.35

        # 4. Content coverage (weight: 25%)
        self.coverage_score, self.coverage_max = self._evaluate_content_coverage(structure)
        total_score += self.coverage_score * 0.25
        max_possible_score += self.coverage_max * 0.25

        # Calculate final normalized score
        final_score = total_score / max_possible_score if max_possible_score > 0 else 0.0

        # Update internal score and return
        self.score = final_score
        return final_score

    def _evaluate_metadata_completeness(self, structure: Structure) -> Tuple[float, float]:
        """
        Evaluate basic metadata completeness.

        Returns:
            Tuple[float, float]: (actual_score, max_possible_score)
        """
        score = 0.0
        max_score = 4.0

        # Check title (weight: 1.0)
        if structure.title and structure.title.strip():
            score += 1.0

        # Check description (weight: 1.0)
        if structure.description and structure.description.strip():
            score += 1.0

        # Check repo_url (weight: 1.0)
        if structure.repo_url and structure.repo_url.strip():
            score += 1.0

        # Check repo_name (weight: 1.0)
        if structure.repo_name and structure.repo_name.strip():
            score += 1.0

        return score, max_score

    def _evaluate_section_completeness(self, structure: Structure) -> Tuple[float, float]:
        """
        Evaluate section structure completeness.

        Returns:
            Tuple[float, float]: (actual_score, max_possible_score)
        """
        if not structure.root_sections:
            return 0.0, 1.0

        total_score = 0.0
        total_sections = 0

        def evaluate_section(section: Section) -> float:
            nonlocal total_sections
            total_sections += 1

            section_score = 0.0
            max_section_score = 3.0

            # Section has title (weight: 1.0)
            if section.section_title and section.section_title.strip():
                section_score += 1.0

            # Section has valid ID (weight: 1.0)
            if section.section_id and section.section_id.strip():
                section_score += 1.0

            # Section has content (pages or subsections) (weight: 1.0)
            if section.pages or section.subsections:
                section_score += 1.0

            # Recursively evaluate subsections
            for subsection in section.subsections:
                evaluate_section(subsection)

            return section_score / max_section_score

        # Evaluate all sections
        for root_section in structure.root_sections:
            total_score += evaluate_section(root_section)

        return total_score, float(total_sections) if total_sections > 0 else 1.0

    def _evaluate_page_completeness(self, structure: Structure) -> Tuple[float, float]:
        """
        Evaluate page completeness.

        Returns:
            Tuple[float, float]: (actual_score, max_possible_score)
        """
        all_pages = get_pages_from_structure(structure)

        if not all_pages:
            return 0.0, 1.0

        total_score = 0.0
        max_score_per_page = 4.0

        for page in all_pages:
            page_score = 0.0

            # Page has title (weight: 1.0)
            if page.page_title and page.page_title.strip():
                page_score += 1.0

            # Page has valid ID (weight: 1.0)
            if page.page_id and page.page_id.strip():
                page_score += 1.0

            # Page has content (weight: 1.0)
            if page.content and page.content.strip():
                page_score += 1.0

            # Page has file paths (weight: 1.0)
            if page.file_paths:
                page_score += 1.0

            total_score += page_score

        max_possible = len(all_pages) * max_score_per_page
        return total_score, max_possible

    def _evaluate_content_coverage(self, structure: Structure) -> Tuple[float, float]:
        """
        Evaluate overall content coverage and structure integrity.

        Returns:
            Tuple[float, float]: (actual_score, max_possible_score)
        """
        score = 0.0
        max_score = 5.0

        # Check if structure has root sections (weight: 1.0)
        if structure.root_sections:
            score += 1.0

        # Check depth of structure (weight: 1.0)
        max_depth = self._calculate_max_depth(structure)
        if max_depth >= 2:  # At least 2 levels (root + subsections)
            score += 1.0

        # Check page distribution (weight: 1.0)
        all_pages = get_pages_from_structure(structure)
        if len(all_pages) >= 3:  # At least 3 pages for meaningful documentation
            score += 1.0

        # Check file path coverage (weight: 1.0)
        total_file_paths = sum(len(page.file_paths) for page in all_pages)
        if total_file_paths >= 1:  # At least some file references
            score += 1.0

        # Check content availability ratio (weight: 1.0)
        pages_with_content = sum(1 for page in all_pages if page.content and page.content.strip())
        if all_pages and (pages_with_content / len(all_pages)) >= 0.5:  # At least 50% pages have content
            score += 1.0

        return score, max_score

    def _calculate_max_depth(self, structure: Structure) -> int:
        """
        Calculate the maximum depth of the section hierarchy.

        Returns:
            int: Maximum depth of the structure
        """
        def get_section_depth(section: Section, current_depth: int = 1) -> int:
            if not section.subsections:
                return current_depth

            max_child_depth = current_depth
            for subsection in section.subsections:
                child_depth = get_section_depth(subsection, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth

        if not structure.root_sections:
            return 0

        max_depth = 0
        for root_section in structure.root_sections:
            depth = get_section_depth(root_section)
            max_depth = max(max_depth, depth)

        return max_depth

    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Get a detailed report with breakdown of completeness metrics.
        """

        all_pages = get_pages_from_structure(self.structure)

        return {
            'overall_score': self.score,
            'metadata_completeness': {
                'score': self.metadata_score / self.metadata_max if self.metadata_max > 0 else 0,
                'details': {
                    'has_title': bool(self.structure.title and self.structure.title.strip()),
                    'has_description': bool(self.structure.description and self.structure.description.strip()),
                    'has_repo_url': bool(self.structure.repo_url and self.structure.repo_url.strip()),
                    'has_repo_name': bool(self.structure.repo_name and self.structure.repo_name.strip()),
                },
            },
            'section_completeness': {
                'score': self.section_score / self.section_max if self.section_max > 0 else 0,
                'total_sections': int(self.section_max),
                'max_depth': self._calculate_max_depth(self.structure),
            },
            'page_completeness': {
                'score': self.page_score / self.page_max if self.page_max > 0 else 0,
                'total_pages': len(all_pages),
                'pages_with_content': sum(1 for page in all_pages if page.content and page.content.strip()),
                'pages_with_file_paths': sum(1 for page in all_pages if page.file_paths),
            },
            'content_coverage': {
                'score': self.coverage_score / self.coverage_max if self.coverage_max > 0 else 0,
                'total_file_references': sum(len(page.file_paths) for page in all_pages),
                'structure_depth': self._calculate_max_depth(self.structure),
            },
        }

from __future__ import annotations

import asyncio

from api.models.docs_gen import Structure
from evaluator import CompletenessEvaluator
from evaluator import HelpfulnessEvaluator
from infra.mongo.core import connect_to_mongo
from infra.mongo.documentation_repository import DocumentationRepository
from shared.logging import get_logger

logger = get_logger(__name__)


async def evaluate_generated_structure():
    # prepare needed app
    await connect_to_mongo()
    documentation_repository = DocumentationRepository()

    # fetch 5 structures from the database
    eval_structures_from_db = await documentation_repository.get_top_newest_documentations(limit=5)
    eval_structures = [Structure(**doc) for doc in eval_structures_from_db]
    logger.info(f"Fetched {len(eval_structures)} structures for evaluation.")

    completeness_evaluator = CompletenessEvaluator()
    helpfulness_evaluator = HelpfulnessEvaluator(provider='google', model='gemini-2.5-flash-lite-preview-06-17')
    # evaluate
    for structure in eval_structures:
        completeness_evaluator.evaluate(structure)
        helpfulness_evaluator.evaluate(structure)

        # get reports
        completeness_report = completeness_evaluator.get_detailed_report()
        helpfulness_report = helpfulness_evaluator.get_detailed_report()

        # print or return reports
        logger.info('-' * 40)
        logger.info(f"Completeness Report: {completeness_report}")
        logger.info(f"Helpfulness Report: {helpfulness_report}")

        total_score = (completeness_evaluator.score + helpfulness_evaluator.score) / 2
        logger.info(f"Final Total Score: {total_score}")

if __name__ == '__main__':
    asyncio.run(evaluate_generated_structure())

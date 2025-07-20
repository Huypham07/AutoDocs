from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

import nltk
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def compute_bleu(references: List[str], candidates: List[str]) -> float:
    """
    Compute average BLEU score for a list of candidate and reference answers.
    """
    scores = []
    for ref, cand in zip(references, candidates):
        ref_tokens = [nltk.word_tokenize(ref)]
        cand_tokens = nltk.word_tokenize(cand)
        # Use BLEU-4 by default
        bleu = nltk.translate.bleu_score.sentence_bleu(
            ref_tokens, cand_tokens,
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1,
        )
        scores.append(bleu)
    return sum(scores) / len(scores) if scores else 0.0


def compute_rouge(references: List[str], candidates: List[str]) -> Dict[str, float]:
    """
    Compute average ROUGE-1 and ROUGE-L F1 scores for a list of candidate and reference answers.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores, rougeL_scores = [], []
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }


def compute_bertscore(references: List[str], candidates: List[str], lang: str = 'en') -> Dict[str, float]:
    """
    Compute average BERTScore precision, recall, and F1 for a list of candidate and reference answers.
    """
    P, R, F1 = bert_score(candidates, references, lang=lang, rescale_with_baseline=True)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item(),
    }


def evaluate_all(references: List[str], candidates: List[str], lang: str = 'en') -> Dict[str, Any]:
    """
    Compute BLEU, ROUGE, and BERTScore metrics for lists of reference and candidate answers.
    """
    return {
        'bleu': compute_bleu(references, candidates),
        'rouge': compute_rouge(references, candidates),
        'bertscore': compute_bertscore(references, candidates, lang=lang),
    }


if __name__ == '__main__':
    # Example usage
    references = ['This is the correct answer.', 'Another reference answer.']
    candidates = ['This is the generated answer.', 'Another generated answer.']
    results = evaluate_all(references, candidates)
    print('Evaluation Results:')
    print(results)

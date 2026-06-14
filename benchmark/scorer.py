"""
Scoring functions for HotpotQA.
Uses exact match and token-level F1 (standard HotpotQA metrics).
"""
import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize(prediction) == normalize(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


from typing import Dict

def score(prediction: str, ground_truth: str) -> Dict:
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "f1": token_f1(prediction, ground_truth),
    }




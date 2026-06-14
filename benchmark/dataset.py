"""
dataset.py — HotpotQA loader with TF-IDF embeddings for ASR semantic scoring.
"""
import math
import re
import string
from collections import Counter
from typing import List, Dict, Any


# ------------------------------------------------------------------ #
#  Minimal TF-IDF embedder (no external deps)                        #
# ------------------------------------------------------------------ #

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _tfidf_embeddings(docs: List[str]) -> List[List[float]]:
    """Compute TF-IDF vectors for a list of documents. Returns L2-normalised vectors."""
    tokenized = [_tokenize(d) for d in docs]
    N = len(docs)

    # Build vocabulary
    vocab = sorted(set(t for doc in tokenized for t in doc))
    if not vocab:
        return [[0.0] for _ in docs]
    v2i = {v: i for i, v in enumerate(vocab)}

    # Document frequency
    df = Counter()
    for doc in tokenized:
        for t in set(doc):
            df[t] += 1

    embeddings = []
    for doc_tokens in tokenized:
        tf = Counter(doc_tokens)
        n = len(doc_tokens) or 1
        vec = [0.0] * len(vocab)
        for t, cnt in tf.items():
            if t in v2i:
                tfidf = (cnt / n) * math.log((N + 1) / (df[t] + 1))
                vec[v2i[t]] = tfidf
        # L2 normalise
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        embeddings.append([x / norm for x in vec])

    return embeddings


# ------------------------------------------------------------------ #
#  HotpotQA loader                                                   #
# ------------------------------------------------------------------ #

def download_hotpotqa(max_samples: int = 200) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("Loading HotpotQA from Hugging Face (first time may take a minute)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    return [ds[i] for i in range(min(max_samples, len(ds)))]


def sample_to_pages(sample: Dict[str, Any]):
    from contextos.simulator.page import Page

    supporting_titles = set(sample["supporting_facts"]["title"])
    titles    = sample["context"]["title"]
    sentences = sample["context"]["sentences"]

    contents = []
    for title, sents in zip(titles, sentences):
        contents.append(f"[{title}]\n" + " ".join(sents))

    # Compute TF-IDF embeddings for all pages + the question
    question = sample["question"]
    all_texts = contents + [question]
    embeddings = _tfidf_embeddings(all_texts)
    page_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    pages = []
    for i, (title, content) in enumerate(zip(titles, contents)):
        page = Page(page_id=title, content=content, embedding=page_embeddings[i])
        pages.append(page)

    # Set dependencies between supporting pages
    supporting_ids = [p.page_id for p in pages if p.page_id in supporting_titles]
    for p in pages:
        if p.page_id in supporting_titles:
            p.dependencies = [sid for sid in supporting_ids if sid != p.page_id]

    return pages, question, sample["answer"], query_embedding
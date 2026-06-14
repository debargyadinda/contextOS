# ContextOS

### Operating-System Inspired Memory Management for Large Language Models

ContextOS is a research framework that explores a simple question:

> What if LLM context windows were managed like operating systems manage memory?

Large Language Models operate under fixed context-window constraints. As conversations and documents grow, information must be discarded to remain within a limited token budget.

ContextOS treats this challenge as a memory-management problem rather than purely a retrieval problem.

By mapping classical operating-system concepts to LLM context handling, ContextOS investigates how page replacement policies influence memory efficiency under constrained context budgets.

---

## Core Idea

ContextOS models LLM context as a virtual memory system.

| Operating Systems | ContextOS                |
| ----------------- | ------------------------ |
| Physical RAM      | Active Context Window    |
| Virtual Memory    | External Knowledge Store |
| Memory Page       | Document Chunk           |
| Page Fault        | Retrieval Event          |
| Page Replacement  | Context Eviction         |
| Working Set       | Active Context           |

When the context window becomes full, a replacement policy decides which chunks remain in memory and which are evicted.

---

## Implemented Policies

### LRU (Least Recently Used)

Evicts the page that has not been accessed recently.

### LFU (Least Frequently Used)

Evicts the page with the lowest access frequency.

### CLOCK

Approximates LRU while reducing bookkeeping overhead.

### Random

Randomly selects a page for eviction.

### Working Set

Maintains pages that have been accessed within a recent usage window.

### ASR (Adaptive Semantic Replacement)

ASR is the primary contribution of ContextOS.

Unlike classical replacement policies that only consider recency or frequency, ASR incorporates semantic relevance into eviction decisions.

Eviction score:

```text
score =
0.40 × semantic distance
+ 0.25 × recency score
+ 0.20 × access frequency
+ 0.15 × dependency penalty
```

Pages that are semantically irrelevant, old, infrequently accessed, and weakly connected receive higher eviction scores and are removed first.

ASR attempts to preserve information that remains useful for ongoing reasoning rather than simply preserving what was used recently.

---

## Experimental Setup

Dataset:

* HotpotQA

Configuration:

* 30 sampled multi-hop reasoning questions
* Dynamic page loading and eviction
* Multi-round memory pressure simulation

Token Budgets:

* 60
* 100
* 150
* 200
* 300

Metrics:

* Fault Rate
* Hit Rate
* Evictions

> Note: These results represent a preliminary proof-of-concept evaluation using 30 HotpotQA samples. Larger-scale experiments are planned for future work.

---

## Results

### Best Fault Rate by Token Budget

| Budget | Best Policy           |
| ------ | --------------------- |
| 60     | Working Set (97.67%)  |
| 100    | LRU (97.00%)          |
| 150    | LFU (97.00%)          |
| 200    | Working Set (96.22%)  |
| 300    | Random / ASR (94.45%) |

### 300 Token Budget

| Policy      | Fault Rate | Hit Rate | Evictions |
| ----------- | ---------- | -------- | --------- |
| Random      | 94.45%     | 5.55%    | 844       |
| ASR         | 94.45%     | 5.55%    | **836**   |
| LFU         | 95.22%     | 4.78%    | 851       |
| LRU         | 95.67%     | 4.33%    | 848       |
| CLOCK       | 96.11%     | 3.89%    | 849       |
| Working Set | 96.78%     | 3.22%    | 862       |

### Key Findings

* Memory pressure remains extremely high under constrained token budgets.
* Classical operating-system replacement policies remain surprisingly competitive.
* Semantic-aware replacement can reduce unnecessary evictions.
* ASR achieved the lowest eviction count among all evaluated policies.
* At the largest tested budget, ASR matched the best observed fault rate while requiring fewer evictions than every competing policy.

---

## Repository Structure

```text
contextOS/
│
├── benchmark/
│   ├── dataset.py
│   ├── runner.py
│   └── scorer.py
│
├── simulator/
│   ├── policies/
│   ├── context.py
│   ├── frame.py
│   ├── page.py
│   └── page_table.py
│
├── results/
│   └── plots.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Running the Benchmark

```bash
git clone https://github.com/debargyadinda/contextOS.git

cd contextOS

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python main.py
```

---

## Why This Matters

Most modern retrieval systems focus on finding relevant information.

ContextOS focuses on a different question:

> Once memory is full, what information should stay?

This perspective allows decades of operating-systems research to be applied directly to long-context reasoning systems.

---

## Future Work

* Large-scale HotpotQA evaluation (1000+ samples)
* LongBench benchmarking
* Learned replacement policies
* Reinforcement-learning-based eviction
* Hierarchical memory architectures
* Multi-agent shared memory systems
* Integration with production LLM pipelines

---


---

### "What if LLM context windows were managed like operating systems manage memory?"

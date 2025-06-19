# Information Retrieval
Information retrieval with Encoder models, this IR systems combine both of these approaches by using a Bi-encoder to quickly retrieve a set of candidates, then using a Cross-encoder to re-rank candidate more accurately.

## System Overview
A typical high-performance search system operates in two stages to balance speed and accuracy:

Retrieval (Candidate Generation): A fast model (the Bi-Encoder) scans a massive corpus of documents (e.g., million documents) and quickly retrieves a smaller, relevant subset of candidates (e.g., top 10).

Re-ranking: A slower but more powerful model (the Cross-Encoder) carefully examines this small subset of candidates and re-orders them to produce the final, highly accurate ranking.

## Dataset Usage
In this repo, i use dataset: Zalo-AI-2021 Legal Text Retrieval.

### Example Dataset
Hard negatives dataset are also created using BM25 and get top 3 negative documents relative to positive documents. After preprocessing, the training set has format as follow:
```json
[
    {
        "question": "...",
        "relevant_articles": [
            {
                "law_id": "...",
                "article_id": "...",
                "title": "...",
                "text": "..."
            }
        ],
        "num_articles": 1,
        "hard_negatives": [
            {
                "law_id": "...",
                "article_id": "...",
                "title": "...",
                "text": "..."
            },
            {
                "law_id": "...",
                "article_id": "...",
                "title": "...",
                "text": "..."
            },
            {
                "law_id": "...",
                "article_id": "...",
                "title": "...",
                "text": "..."
            }
        ],
    },
]
```



## Installation
```sh
pip install -r requirements.txt
```

## Usage
Training and evaluating for Bi-encoder model:
```sh
bash train_be.sh
```

Training and evaluating for Cross-encoder model:
```sh
bash train_ce.sh
```

## Future work
- Applying Rerank for finetuned models
- Build Cross-encoder model for Rerank task  (Done)